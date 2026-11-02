"""Build the field dictionary and game-level metadata table for a frozen batch."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DICTIONARY_COLUMNS = (
    "dataset",
    "field",
    "data_type",
    "nullable",
    "unit",
    "source",
    "transformation",
    "allowed_values",
    "description",
    "analysis_role",
    "caveat",
)


def field(
    dataset,
    name,
    data_type,
    source,
    description,
    *,
    nullable=False,
    unit="none",
    transformation="none",
    allowed_values="",
    analysis_role="identifier/context",
    caveat="",
):
    return {
        "dataset": dataset,
        "field": name,
        "data_type": data_type,
        "nullable": nullable,
        "unit": unit,
        "source": source,
        "transformation": transformation,
        "allowed_values": allowed_values,
        "description": description,
        "analysis_role": analysis_role,
        "caveat": caveat,
    }


def review_field_specs():
    dataset = "review_datasets"
    raw = "Steam review endpoint, frozen language reviews.jsonl"
    clean = "scripts/clean_reviews.py"
    freeze = "freeze_manifest selected_runs"
    specs = [
        field(dataset, "game", "string", freeze, "Configured game name."),
        field(dataset, "appid", "integer", raw, "Steam application identifier."),
        field(dataset, "language", "category", raw, "Requested Steam review language.", allowed_values="english|schinese", analysis_role="grouping variable", caveat="API language filter, not an independently detected language."),
        field(dataset, "recommendationid", "string", raw, "Steam review/recommendation identifier.", analysis_role="primary key", caveat="Kept as text; globally unique in clean_master."),
        field(dataset, "steamid", "string", raw, "Steam author identifier.", analysis_role="author key", caveat="Identifier, not a numeric measure; do not convert to floating point."),
        field(dataset, "review", "string", raw, "Original review text returned by Steam.", nullable=True, analysis_role="raw text"),
        field(dataset, "review_clean", "string", clean, "Whitespace-normalized review text with original case retained.", nullable=True, transformation="Collapse all whitespace runs to one space and trim.", analysis_role="model input"),
        field(dataset, "voted_up", "boolean", raw, "Steam recommendation label.", allowed_values="true|false", analysis_role="outcome", caveat="A recommendation indicator, not a continuous sentiment score."),
        field(dataset, "timestamp_created", "integer", raw, "Review creation time as Unix timestamp.", unit="seconds since 1970-01-01 00:00:00 UTC", analysis_role="time"),
        field(dataset, "timestamp_updated", "integer", raw, "Review last-update time as Unix timestamp.", unit="seconds since 1970-01-01 00:00:00 UTC", analysis_role="time"),
        field(dataset, "created_utc", "datetime", "scripts/steam_data.py", "ISO-8601 UTC creation timestamp produced during collection.", unit="UTC", transformation="Convert timestamp_created from Unix seconds to ISO-8601.", analysis_role="time"),
        field(dataset, "created_utc_clean", "datetime", clean, "Validated ISO-8601 creation timestamp.", nullable=True, unit="UTC", transformation="Reparse timestamp_created and serialize with +00:00 offset.", analysis_role="time"),
        field(dataset, "created_date_utc", "date", clean, "UTC calendar date of review creation.", nullable=True, unit="UTC calendar date", transformation="Date part of created_utc_clean.", analysis_role="time/grouping"),
        field(dataset, "timestamp_dev_responded", "integer", raw, "Developer response time when provided by Steam.", nullable=True, unit="seconds since 1970-01-01 00:00:00 UTC", analysis_role="time"),
        field(dataset, "developer_response", "string", raw, "Developer response text when present.", nullable=True, analysis_role="context text"),
        field(dataset, "playtime_at_review", "number", raw, "Author playtime recorded when the review was submitted.", nullable=True, unit="minutes", analysis_role="exposure/covariate", caveat="This is explicitly minutes; convert before interpreting as hours."),
        field(dataset, "playtime_forever", "number", raw, "Author cumulative playtime at collection time.", nullable=True, unit="minutes", analysis_role="context/covariate", caveat="May exceed playtime_at_review because play continued after posting."),
        field(dataset, "playtime_at_review_hours", "number", clean, "Review-time playtime converted to hours.", nullable=True, unit="hours", transformation="Valid non-negative playtime_at_review / 60.", analysis_role="covariate"),
        field(dataset, "playtime_forever_hours", "number", clean, "Cumulative playtime converted to hours.", nullable=True, unit="hours", transformation="Valid non-negative playtime_forever / 60.", analysis_role="context/covariate"),
        field(dataset, "log1p_playtime_hours", "number", clean, "Log-transformed review-time playtime.", nullable=True, unit="log(1 + hours)", transformation="ln(1 + playtime_at_review / 60).", analysis_role="primary playtime covariate"),
        field(dataset, "playtime_p99_group_hours", "number", clean, "99th percentile within appid-language group.", nullable=True, unit="hours", transformation="Linear-interpolated P99 of valid non-negative playtime_at_review hours.", analysis_role="outlier threshold"),
        field(dataset, "playtime_hours_winsorized_p99", "number", clean, "Review-time hours capped at group P99.", nullable=True, unit="hours", transformation="min(playtime_at_review_hours, playtime_p99_group_hours).", analysis_role="sensitivity covariate"),
        field(dataset, "votes_up", "integer", raw, "Helpful votes received by the review.", unit="votes", analysis_role="engagement context"),
        field(dataset, "votes_funny", "integer", raw, "Funny votes received by the review.", unit="votes", analysis_role="engagement context"),
        field(dataset, "weighted_vote_score", "number", raw, "Steam-provided weighted helpfulness score.", unit="Steam score", analysis_role="engagement context", caveat="Use as returned; formula is controlled by Steam."),
        field(dataset, "comment_count", "integer", raw, "Number of review comments.", unit="comments", analysis_role="engagement context"),
        field(dataset, "source", "category", "scripts/steam_data.py", "Review collection backend.", allowed_values="official|legacy", analysis_role="provenance"),
        field(dataset, "game_era", "category", "scripts/steam_data.py", "CS2/CS:GO date-based era label; other games use other.", allowed_values="cs2|csgo|other", transformation="For appid 730, compare creation time with 2023-09-27T00:00:00Z.", analysis_role="provenance/context", caveat="appid 730 cumulative playtime may span both CS:GO and CS2."),
        field(dataset, "hardware", "string", raw, "Steam hardware field when returned.", nullable=True, analysis_role="context"),
        field(dataset, "source_relative_path", "string", clean, "Project-relative source JSONL path.", analysis_role="provenance"),
        field(dataset, "source_row_number", "integer", clean, "One-based line number in source JSONL.", unit="row number", analysis_role="provenance"),
        field(dataset, "text_key", "string", clean, "Key used for exact normalized-text duplicate detection.", nullable=True, transformation="Collapse whitespace, trim, then Unicode casefold.", analysis_role="deduplication key"),
        field(dataset, "text_length", "integer", clean, "Length of text_key.", unit="Unicode code points", analysis_role="quality feature"),
        field(dataset, "english_word_count", "integer", clean, "Count of English alphabetic word tokens.", unit="tokens", transformation="Regex count of alphabetic words including internal apostrophes.", analysis_role="quality feature"),
        field(dataset, "chinese_character_count", "integer", clean, "Count of CJK unified ideograph characters.", unit="characters", transformation="Regex count over CJK ideograph ranges.", analysis_role="quality feature"),
        field(dataset, "text_duplicate_group_size", "integer", clean, "Number of rows sharing appid, language and text_key.", unit="reviews", analysis_role="duplicate control"),
        field(dataset, "text_duplicate_rank", "integer", clean, "Deterministic position within a normalized-text duplicate group.", nullable=True, unit="rank", transformation="Sort by timestamp_created then recommendationid; first row is canonical.", analysis_role="duplicate control"),
        field(dataset, "duplicate_inverse_weight", "number", clean, "Inverse normalized-text group frequency.", nullable=True, unit="weight", transformation="1 / text_duplicate_group_size for non-empty text.", analysis_role="sensitivity weight"),
        field(dataset, "author_review_group_size", "integer", clean, "Reviews by the same author within appid-language.", unit="reviews", analysis_role="duplicate control"),
        field(dataset, "common_window_start_utc", "date", clean, "Start of the overlapping English/Chinese date window for the game.", nullable=True, unit="UTC calendar date", transformation="Maximum of the two language-specific earliest valid dates.", analysis_role="language comparison"),
        field(dataset, "common_window_end_utc", "date", clean, "End of the overlapping English/Chinese date window for the game.", nullable=True, unit="UTC calendar date", transformation="Minimum of the two language-specific latest valid dates.", analysis_role="language comparison"),
    ]

    raw_booleans = {
        "steam_purchase": "Steam marks the review as associated with a Steam purchase.",
        "received_for_free": "Steam marks the product as received for free.",
        "written_during_early_access": "Review was written during early access.",
        "primarily_steam_deck": "Steam marks the review as primarily played on Steam Deck.",
        "refunded": "Steam refund indicator when returned.",
    }
    specs.extend(
        field(dataset, name, "boolean", raw, description, allowed_values="true|false", analysis_role="review context")
        for name, description in raw_booleans.items()
    )

    flag_details = {
        "is_empty_text": ("Review text is empty after whitespace normalization.", "text_key == ''."),
        "is_short_text": ("Non-empty normalized text is shorter than the configured threshold.", "0 < len(text_key) < 10 in the current build."),
        "is_low_information": ("Language-specific low-information text flag.", "English: fewer than 3 alphabetic words; Simplified Chinese: fewer than 4 CJK characters and fewer than 3 English words."),
        "is_duplicate_text": ("Text belongs to a normalized duplicate group with more than one row.", "text_duplicate_group_size > 1."),
        "is_text_canonical": ("Row selected to represent a normalized duplicate text group.", "First row by timestamp_created then recommendationid."),
        "is_duplicate_author_review": ("Author has more than one review in the appid-language group.", "author_review_group_size > 1."),
        "is_author_record_canonical": ("Row retained as the canonical review for a repeated author.", "Latest timestamp_updated, then recommendationid."),
        "is_invalid_timestamp": ("timestamp_created cannot be converted to UTC datetime.", "Timestamp parsing failed."),
        "playtime_at_review_missing": ("Review-time playtime is absent or non-numeric.", "Numeric parsing returned missing."),
        "playtime_forever_missing": ("Cumulative playtime is absent or non-numeric.", "Numeric parsing returned missing."),
        "playtime_at_review_invalid": ("Review-time playtime is negative.", "Parsed minutes < 0."),
        "playtime_forever_invalid": ("Cumulative playtime is negative.", "Parsed minutes < 0."),
        "is_playtime_outlier": ("Review-time playtime exceeds the configured hard threshold.", "playtime_at_review > 5,256,000 minutes (10 years)."),
        "is_playtime_inconsistent": ("Review-time playtime exceeds cumulative playtime.", "playtime_at_review > playtime_forever when both are valid."),
        "topic_unique_nonempty": ("Row is a canonical author record with canonical non-empty text.", "Canonical author AND non-empty AND canonical text."),
        "topic_fit_eligible": ("Row is included in the primary topic-model fitting dataset.", "topic_unique_nonempty AND not low-information."),
        "association_eligible": ("Row is included in recommendation/theme association data.", "Canonical author record AND non-empty text; duplicate text remains with a weight."),
        "playtime_analysis_eligible": ("Row has valid playtime for RQ4 analyses.", "association_eligible with non-missing, non-negative, consistent and below-hard-limit playtime."),
        "in_common_language_window": ("Review date lies in its game's English/Chinese overlap window.", "created_date_utc between common window endpoints, inclusive."),
        "language_comparison_game_eligible": ("Game meets the minimum clean text count in both languages inside the overlap window.", "At least 100 unique non-empty texts per language in the current build."),
        "language_comparison_row_eligible": ("Row is usable in the strict language comparison subset.", "association_eligible AND in common window AND eligible game."),
    }
    specs.extend(
        field(dataset, name, "boolean", clean, description, allowed_values="true|false", transformation=transform, analysis_role="cleaning/eligibility flag")
        for name, (description, transform) in flag_details.items()
    )
    return specs


def quality_field_specs():
    dataset = "quality_table"
    audit = "scripts/verify_data.py over the frozen review JSONL files"
    fields = [
        ("game", "string", "none", "Configured game name."),
        ("appid", "integer", "none", "Steam application identifier."),
        ("language", "category", "none", "Requested review language."),
        ("raw_review_count", "integer", "reviews", "Rows collected for the game-language group."),
        ("source_total_matching", "integer", "reviews", "Steam-reported matching reviews for the query."),
        ("recommendation_count", "integer", "reviews", "Rows with voted_up=true."),
        ("recommendation_rate", "number", "proportion", "recommendation_count / raw_review_count."),
        ("earliest_review_date_utc", "date", "UTC calendar date", "Earliest valid review creation date."),
        ("latest_review_date_utc", "date", "UTC calendar date", "Latest valid review creation date."),
        ("actual_coverage_days", "integer", "inclusive UTC days", "Calendar-day span from earliest through latest review."),
        ("empty_text_count", "integer", "reviews", "Reviews empty after whitespace normalization."),
        ("short_text_count", "integer", "reviews", "Non-empty normalized reviews shorter than 10 characters."),
        ("duplicate_text_count", "integer", "extra reviews", "Extra non-empty reviews after appid-language case/whitespace normalization."),
        ("playtime_missing_rate", "number", "proportion", "Missing/non-numeric playtime_at_review divided by raw_review_count."),
        ("playtime_zero_rate", "number", "proportion", "Zero playtime_at_review divided by raw_review_count."),
    ]
    for percentile_value in (25, 50, 75, 90, 95, 99):
        fields.append((f"playtime_p{percentile_value}_hours", "number", "hours", f"P{percentile_value} of valid non-negative playtime_at_review."))
    fields.extend([
        ("unique_author_count", "integer", "authors", "Distinct non-empty Steam author IDs."),
        ("duplicate_author_count", "integer", "authors", "Authors contributing more than one review in the group."),
        ("duplicate_author_review_count", "integer", "extra reviews", "Reviews beyond one per repeated author."),
        ("core_endpoint_status", "category", "none", "Combined players/news/store endpoint status."),
        ("review_endpoint_status", "category", "none", "Review endpoint status for the language."),
        ("endpoint_status", "category", "none", "Combined core and review endpoint status."),
        ("target_count", "integer", "reviews", "Requested reviews per language."),
        ("target_shortfall", "integer", "reviews", "max(target_count - raw_review_count, 0)."),
        ("target_met", "boolean", "none", "Whether raw_review_count reaches target_count."),
        ("met_200_target", "boolean", "none", "Whether raw_review_count reaches 200."),
        ("directory", "string", "none", "Source game-language directory."),
    ])
    return [
        field(dataset, name, data_type, audit, description, unit=unit, nullable=name.startswith("playtime_p"), transformation="Audit aggregation from raw review rows.", analysis_role="quality control")
        for name, data_type, unit, description in fields
    ]


GAME_META_COLUMNS = (
    "appid", "game", "selected_run", "source_directory", "sample_status",
    "partial_languages", "partial_reason", "requested_per_language",
    "english_review_count", "schinese_review_count", "total_review_count",
    "english_source_total_matching", "schinese_source_total_matching",
    "english_target_shortfall", "schinese_target_shortfall", "partial_shortfall_total",
    "english_review_endpoint_status", "schinese_review_endpoint_status",
    "sampling_method", "sampling_window_start_utc", "sampling_window_end_utc",
    "sampling_window_days", "common_window_start_utc", "common_window_end_utc",
    "english_unique_nonempty_common_window_count",
    "schinese_unique_nonempty_common_window_count", "language_comparison_eligible",
    "project_genres", "project_primary_genre", "steam_genres", "steam_genre_ids",
    "steam_categories", "steam_category_ids", "store_type", "is_free", "price_label",
    "developers", "publishers", "release_date_store_text", "release_date_iso",
    "release_date_parse_status", "coming_soon", "platform_windows", "platform_mac",
    "platform_linux", "metacritic_score", "store_recommendations_total",
    "store_fetched_at_utc", "store_endpoint", "store_endpoint_status",
    "store_response_success", "store_fallback_cc", "store_metadata_source",
    "store_metadata_note", "config_verification_status", "name_match",
)


def game_meta_field_specs():
    dataset = "game_meta"
    source_map = {
        "project": "config/games_clustering.json",
        "freeze": "freeze manifest and per-run manifest",
        "quality": "data/quality_table.csv",
        "clean": "cleaning_summary.json",
        "store": "frozen Steam Store appdetails store.json",
    }
    definitions = {
        "appid": ("integer", "none", "project", "Steam application identifier."),
        "game": ("string", "none", "freeze", "Frozen configured game name."),
        "selected_run": ("string", "none", "freeze", "Selected collection run identifier."),
        "source_directory": ("string", "none", "freeze", "Project-relative frozen run directory."),
        "sample_status": ("category", "none", "quality", "complete when both languages meet target; otherwise partial."),
        "partial_languages": ("string list", "none", "quality", "Pipe-separated languages below target_count."),
        "partial_reason": ("string", "none", "quality", "Observed counts and targets explaining partial status."),
        "requested_per_language": ("integer", "reviews", "freeze", "Requested review target for each language."),
        "english_review_count": ("integer", "reviews", "quality", "Collected English reviews."),
        "schinese_review_count": ("integer", "reviews", "quality", "Collected Simplified Chinese reviews."),
        "total_review_count": ("integer", "reviews", "freeze", "Total English plus Simplified Chinese reviews."),
        "english_source_total_matching": ("integer", "reviews", "quality", "Steam-reported English matches."),
        "schinese_source_total_matching": ("integer", "reviews", "quality", "Steam-reported Simplified Chinese matches."),
        "english_target_shortfall": ("integer", "reviews", "quality", "English target shortfall."),
        "schinese_target_shortfall": ("integer", "reviews", "quality", "Simplified Chinese target shortfall."),
        "partial_shortfall_total": ("integer", "reviews", "quality", "Sum of language target shortfalls."),
        "english_review_endpoint_status": ("category", "none", "quality", "English review endpoint audit status."),
        "schinese_review_endpoint_status": ("category", "none", "quality", "Simplified Chinese review endpoint audit status."),
        "sampling_method": ("string", "none", "freeze", "Collection sampling description."),
        "sampling_window_start_utc": ("datetime", "UTC", "freeze", "Frozen sampling-window start."),
        "sampling_window_end_utc": ("datetime", "UTC", "freeze", "Frozen sampling-window end."),
        "sampling_window_days": ("integer", "days", "freeze", "Configured sampling-window length."),
        "common_window_start_utc": ("date", "UTC calendar date", "clean", "English/Chinese overlap-window start."),
        "common_window_end_utc": ("date", "UTC calendar date", "clean", "English/Chinese overlap-window end."),
        "english_unique_nonempty_common_window_count": ("integer", "reviews", "clean", "Unique non-empty English texts in common window."),
        "schinese_unique_nonempty_common_window_count": ("integer", "reviews", "clean", "Unique non-empty Chinese texts in common window."),
        "language_comparison_eligible": ("boolean", "none", "clean", "Both languages meet the configured common-window threshold."),
        "project_genres": ("string list", "none", "project", "Project-curated genres used for selection and interpretation."),
        "project_primary_genre": ("string", "none", "project", "First project-curated genre; not an official Steam primary genre."),
        "steam_genres": ("string list", "none", "store", "Pipe-separated official Steam Store genre descriptions."),
        "steam_genre_ids": ("string list", "none", "store", "Pipe-separated Steam genre IDs."),
        "steam_categories": ("string list", "none", "store", "Pipe-separated Steam feature/category descriptions."),
        "steam_category_ids": ("string list", "none", "store", "Pipe-separated Steam category IDs."),
        "store_type": ("category", "none", "store", "Steam Store app type."),
        "is_free": ("boolean", "none", "store", "Steam Store free-to-play indicator."),
        "price_label": ("string", "none", "project", "Project configuration price label."),
        "developers": ("string list", "none", "store", "Pipe-separated Steam Store developers."),
        "publishers": ("string list", "none", "store", "Pipe-separated Steam Store publishers."),
        "release_date_store_text": ("string", "none", "store", "Release date text exactly as returned by Steam."),
        "release_date_iso": ("date", "calendar date", "store", "Parsed release date in YYYY-MM-DD when recognized."),
        "release_date_parse_status": ("category", "none", "store", "ok, missing, or unparsed release-date status."),
        "coming_soon": ("boolean", "none", "store", "Steam Store coming-soon flag."),
        "platform_windows": ("boolean", "none", "store", "Steam Store Windows support flag."),
        "platform_mac": ("boolean", "none", "store", "Steam Store macOS support flag."),
        "platform_linux": ("boolean", "none", "store", "Steam Store Linux support flag."),
        "metacritic_score": ("integer", "score", "store", "Metacritic score when supplied by Steam."),
        "store_recommendations_total": ("integer", "recommendations", "store", "Store-level recommendation total at snapshot time."),
        "store_fetched_at_utc": ("datetime", "UTC", "store", "Store snapshot retrieval time."),
        "store_endpoint": ("string", "none", "store", "Steam Store appdetails endpoint URL."),
        "store_endpoint_status": ("category", "none", "freeze", "Per-run store endpoint status."),
        "store_response_success": ("boolean", "none", "store", "Final Steam Store response success flag."),
        "store_fallback_cc": ("category", "country code", "freeze", "Fallback country code used after the initial store response failed."),
        "store_metadata_source": ("string", "none", "store", "Provenance label for metadata values."),
        "store_metadata_note": ("string", "none", "freeze", "Material endpoint or provenance note."),
        "config_verification_status": ("category", "none", "project", "Configuration-time store verification status."),
        "name_match": ("boolean", "none", "project", "Configured name matches verified Steam Store name."),
    }
    specs = []
    for name in GAME_META_COLUMNS:
        data_type, unit, source_key, description = definitions[name]
        nullable = name in {
            "partial_languages", "partial_reason", "release_date_iso",
            "metacritic_score", "store_fallback_cc", "store_metadata_note",
        }
        caveat = ""
        if name == "sample_status":
            caveat = "partial means insufficient available reviews in at least one language, not necessarily an endpoint failure."
        elif name == "project_primary_genre":
            caveat = "Project convention only; Steam does not designate the first genre as primary."
        elif name in {"store_recommendations_total", "metacritic_score"}:
            caveat = "Snapshot value; may change after collection."
        elif name == "store_fallback_cc":
            caveat = "Path of Exile used cc=us after cc=cn returned unsuccessful; final store response was successful."
        specs.append(field(dataset, name, data_type, source_map[source_key], description, nullable=nullable, unit=unit, transformation="Flatten or derive from the stated frozen source.", analysis_role="game-level metadata", caveat=caveat))
    return specs


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def latest_freeze(root):
    candidates = sorted((root / "data").glob("freeze_*/freeze_manifest.json"))
    if not candidates:
        raise FileNotFoundError("No frozen dataset manifest found")
    return candidates[-1]


def read_header(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return next(csv.reader(source))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value):
    return str(value).lower() == "true"


def join_values(values, key=None):
    result = []
    for value in values or []:
        item = value.get(key) if key and isinstance(value, dict) else value
        if item is not None and str(item) not in result:
            result.append(str(item))
    return "|".join(result)


def parse_release_date(value):
    if not value:
        return "", "missing"
    for pattern in ("%b %d, %Y", "%d %b, %Y"):
        try:
            return datetime.strptime(value, pattern).date().isoformat(), "ok"
        except ValueError:
            pass
    return "", "unparsed"


def validate_dictionary_coverage(clean_path, quality_path, dictionary_rows):
    documented = {}
    for row in dictionary_rows:
        documented.setdefault(row["dataset"], set()).add(row["field"])
    checks = {
        "review_datasets": set(read_header(clean_path)),
        "quality_table": set(read_header(quality_path)),
        "game_meta": set(GAME_META_COLUMNS),
    }
    problems = []
    for dataset, actual in checks.items():
        missing = sorted(actual - documented.get(dataset, set()))
        extra = sorted(documented.get(dataset, set()) - actual)
        if missing:
            problems.append(f"{dataset} undocumented: {', '.join(missing)}")
        if extra:
            problems.append(f"{dataset} definitions without field: {', '.join(extra)}")
    if problems:
        raise ValueError("Data dictionary coverage failed:\n" + "\n".join(problems))


def build_game_meta(root, freeze, config, quality_rows, cleaning_summary):
    quality = {(int(row["appid"]), row["language"]): row for row in quality_rows}
    cleaned = {(int(row["appid"]), row["language"]): row for row in cleaning_summary["groups"]}
    configured = {int(row["appid"]): row for row in config["games"]}
    rows = []
    for selected in freeze["selected_runs"]:
        appid = int(selected["appid"])
        game = configured[appid]
        run_path = resolve(root, selected["path"])
        manifest = load_json(run_path / "manifest.json")
        store_wrapper = load_json(run_path / "store.json")
        store_record = store_wrapper["data"].get(str(appid), {})
        store_data = store_record.get("data") or game.get("store_metadata", {})
        english = quality[(appid, "english")]
        schinese = quality[(appid, "schinese")]
        english_clean = cleaned[(appid, "english")]
        schinese_clean = cleaned[(appid, "schinese")]
        partial_languages = [
            language
            for language, quality_row in (("english", english), ("schinese", schinese))
            if not parse_bool(quality_row["target_met"])
        ]
        partial_parts = []
        for language, quality_row in (("english", english), ("schinese", schinese)):
            if language in partial_languages:
                partial_parts.append(
                    f"{language}: {quality_row['raw_review_count']}/{quality_row['target_count']} "
                    f"reviews; source_total_matching={quality_row['source_total_matching']}"
                )
        release_text = (store_data.get("release_date") or {}).get("date", "")
        release_iso, release_status = parse_release_date(release_text)
        fallback_cc = manifest.get("endpoints", {}).get("store", {}).get("fallback_cc", "")
        store_note = ""
        if fallback_cc:
            store_note = (
                "Initial Steam Store appdetails request with cc=cn returned unsuccessful; "
                f"collector retried with cc={fallback_cc} and saved the successful response. "
                "This does not indicate a review endpoint failure."
            )
        row = {
            "appid": appid,
            "game": selected["name"],
            "selected_run": selected["run"],
            "source_directory": selected["path"],
            "sample_status": "partial" if partial_languages else "complete",
            "partial_languages": "|".join(partial_languages),
            "partial_reason": "; ".join(partial_parts),
            "requested_per_language": manifest["requested_per_language"],
            "english_review_count": int(english["raw_review_count"]),
            "schinese_review_count": int(schinese["raw_review_count"]),
            "total_review_count": int(english["raw_review_count"]) + int(schinese["raw_review_count"]),
            "english_source_total_matching": int(english["source_total_matching"]),
            "schinese_source_total_matching": int(schinese["source_total_matching"]),
            "english_target_shortfall": int(english["target_shortfall"]),
            "schinese_target_shortfall": int(schinese["target_shortfall"]),
            "partial_shortfall_total": int(english["target_shortfall"]) + int(schinese["target_shortfall"]),
            "english_review_endpoint_status": english["review_endpoint_status"],
            "schinese_review_endpoint_status": schinese["review_endpoint_status"],
            "sampling_method": manifest["sampling"],
            "sampling_window_start_utc": freeze["sampling_window"]["start_utc"],
            "sampling_window_end_utc": freeze["sampling_window"]["end_utc"],
            "sampling_window_days": freeze["sampling_window"]["days"],
            "common_window_start_utc": english_clean["common_window_start_utc"],
            "common_window_end_utc": english_clean["common_window_end_utc"],
            "english_unique_nonempty_common_window_count": english_clean["unique_nonempty_common_window_count"],
            "schinese_unique_nonempty_common_window_count": schinese_clean["unique_nonempty_common_window_count"],
            "language_comparison_eligible": english_clean["language_comparison_game_eligible"],
            "project_genres": join_values(game.get("genre", [])),
            "project_primary_genre": (game.get("genre") or [""])[0],
            "steam_genres": join_values(store_data.get("genres", []), "description"),
            "steam_genre_ids": join_values(store_data.get("genres", []), "id"),
            "steam_categories": join_values(store_data.get("categories", []), "description"),
            "steam_category_ids": join_values(store_data.get("categories", []), "id"),
            "store_type": store_data.get("type", game.get("verified_type", "")),
            "is_free": store_data.get("is_free", ""),
            "price_label": game.get("price", ""),
            "developers": join_values(store_data.get("developers", [])),
            "publishers": join_values(store_data.get("publishers", [])),
            "release_date_store_text": release_text,
            "release_date_iso": release_iso,
            "release_date_parse_status": release_status,
            "coming_soon": (store_data.get("release_date") or {}).get("coming_soon", ""),
            "platform_windows": (store_data.get("platforms") or {}).get("windows", ""),
            "platform_mac": (store_data.get("platforms") or {}).get("mac", ""),
            "platform_linux": (store_data.get("platforms") or {}).get("linux", ""),
            "metacritic_score": (store_data.get("metacritic") or {}).get("score", ""),
            "store_recommendations_total": (store_data.get("recommendations") or {}).get("total", ""),
            "store_fetched_at_utc": store_wrapper.get("fetched_at", ""),
            "store_endpoint": store_wrapper.get("endpoint", ""),
            "store_endpoint_status": manifest.get("endpoints", {}).get("store", {}).get("status", ""),
            "store_response_success": bool(store_record.get("success")),
            "store_fallback_cc": fallback_cc,
            "store_metadata_source": "frozen Steam Store appdetails snapshot",
            "store_metadata_note": store_note,
            "config_verification_status": game.get("verification_status", ""),
            "name_match": game.get("name_match", ""),
        }
        rows.append(row)
    return rows


def markdown_escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_markdown(dictionary_rows, game_rows, freeze_label):
    partial_rows = [row for row in game_rows if row["sample_status"] == "partial"]
    poe = next(row for row in game_rows if int(row["appid"]) == 238960)
    lines = [
        "# Data Dictionary and Game Metadata",
        "",
        f"Data version: `{freeze_label}`. All review creation and comparison dates use UTC.",
        "",
        "## Dataset Rules",
        "",
        "- `playtime_at_review` and `playtime_forever` are stored in minutes in the raw review data. Hour fields divide by 60.",
        "- `language` is the Steam API language filter (`english` or `schinese`), not a separate language-classification result.",
        "- `partial` means at least one requested language returned fewer than 200 reviews in the frozen 183-day window. It does not by itself mean that an endpoint failed, and no rows are imputed or duplicated.",
        "- `steam_genres` and `steam_categories` come from the frozen Steam Store `appdetails` response. `project_genres` are curated labels in the project configuration and may be more specific, such as FPS or ARPG.",
        "- Use `steam_genres` for official store taxonomy. Use `project_genres` only when the analysis explicitly calls for the curated project grouping.",
        "",
        "## Partial Samples",
        "",
        "| Game | Partial languages | Counts and reason |",
        "| --- | --- | --- |",
    ]
    for row in partial_rows:
        lines.append(f"| {markdown_escape(row['game'])} | {markdown_escape(row['partial_languages'])} | {markdown_escape(row['partial_reason'])} |")
    lines.extend([
        "",
        "## Path of Exile Store Endpoint",
        "",
        f"Path of Exile has `store_endpoint_status={poe['store_endpoint_status']}` and `store_fallback_cc={poe['store_fallback_cc']}`. The initial Store `appdetails` request with `cc=cn` returned an unsuccessful application record, so the collector retried with `cc=us`. The US response succeeded and supplies the genres and other store metadata in `game_meta.csv`. This fallback is unrelated to the review endpoint; the Path of Exile sample is partial because only 22 Simplified Chinese reviews matched the frozen review window.",
        "",
        "## Field Dictionary",
        "",
    ])
    for dataset in ("review_datasets", "quality_table", "game_meta"):
        lines.extend([
            f"### {dataset}",
            "",
            "| Field | Type | Unit | Source | Transformation | Description | Caveat |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in dictionary_rows:
            if row["dataset"] != dataset:
                continue
            lines.append(
                "| " + " | ".join(markdown_escape(row[key]) for key in (
                    "field", "data_type", "unit", "source", "transformation", "description", "caveat"
                )) + " |"
            )
        lines.append("")
    return "\n".join(lines)


def run(args, root=ROOT):
    freeze_path = resolve(root, args.freeze_manifest) if args.freeze_manifest else latest_freeze(root)
    quality_path = resolve(root, args.quality_table)
    clean_path = resolve(root, args.clean_master)
    cleaning_summary_path = resolve(root, args.cleaning_summary)
    output_dir = resolve(root, args.output_dir) if args.output_dir else root / "data" / "metadata" / freeze_path.parent.name
    docs_path = resolve(root, args.docs_output)
    outputs = [output_dir / "data_dictionary.csv", output_dir / "game_meta.csv", output_dir / "metadata_manifest.json", docs_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Metadata outputs already exist; re-run with --overwrite")
    if args.overwrite:
        for path in existing:
            path.unlink()

    freeze = load_json(freeze_path)
    config_path = resolve(root, freeze["collection_config"])
    config = load_json(config_path)
    quality_rows = read_csv(quality_path)
    cleaning_summary = load_json(cleaning_summary_path)
    dictionary_rows = review_field_specs() + quality_field_specs() + game_meta_field_specs()
    validate_dictionary_coverage(clean_path, quality_path, dictionary_rows)
    game_rows = build_game_meta(root, freeze, config, quality_rows, cleaning_summary)
    if len(game_rows) != freeze["game_count"]:
        raise ValueError(f"Expected {freeze['game_count']} game rows, built {len(game_rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary_path = output_dir / "data_dictionary.csv"
    game_meta_path = output_dir / "game_meta.csv"
    write_csv(dictionary_path, dictionary_rows, DICTIONARY_COLUMNS)
    write_csv(game_meta_path, game_rows, GAME_META_COLUMNS)
    docs_path.write_text(
        build_markdown(dictionary_rows, game_rows, freeze_path.parent.name) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest": freeze_path.relative_to(root).as_posix(),
        "sources": [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
            for path in (freeze_path, config_path, quality_path, clean_path, cleaning_summary_path)
        ],
        "counts": {
            "dictionary_fields": len(dictionary_rows),
            "review_fields": sum(row["dataset"] == "review_datasets" for row in dictionary_rows),
            "quality_fields": sum(row["dataset"] == "quality_table" for row in dictionary_rows),
            "game_meta_fields": sum(row["dataset"] == "game_meta" for row in dictionary_rows),
            "games": len(game_rows),
            "partial_games": sum(row["sample_status"] == "partial" for row in game_rows),
            "store_fallback_games": sum(bool(row["store_fallback_cc"]) for row in game_rows),
        },
        "files": [
            {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (dictionary_path, game_meta_path, docs_path)
        ],
    }
    manifest_path = output_dir / "metadata_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"status": "built", "output_dir": output_dir.relative_to(root).as_posix(), **manifest["counts"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--quality-table", type=Path, default=Path("data/quality_table.csv"))
    parser.add_argument("--clean-master", type=Path, default=Path("data/cleaned/freeze_20261101/clean_master.csv"))
    parser.add_argument("--cleaning-summary", type=Path, default=Path("data/cleaned/freeze_20261101/cleaning_summary.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--docs-output", type=Path, default=Path("docs/data-dictionary.md"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    try:
        run(build_parser().parse_args())
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as error:
        print(f"build_metadata: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

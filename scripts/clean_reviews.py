"""Build reproducible analysis datasets from a frozen Steam review batch."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHORT_TEXT = 10
DEFAULT_MIN_ENGLISH_WORDS = 3
DEFAULT_MIN_CHINESE_CHARS = 4
DEFAULT_MIN_LANGUAGE_COMPARISON = 100
DEFAULT_MAX_PLAYTIME_MINUTES = 10 * 365 * 24 * 60
OUTPUT_FILES = (
    "clean_master.jsonl",
    "clean_master.csv",
    "topic_fit.jsonl",
    "topic_fit.csv",
    "association_data.jsonl",
    "association_data.csv",
    "cleaning_summary.json",
    "cleaning_summary.csv",
    "cleaning_manifest.json",
)
ENGLISH_WORD = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?")
CHINESE_CHARACTER = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(root: Path, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def portable_path(path: Path, root: Path):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def latest_freeze_manifest(root: Path):
    candidates = sorted((root / "data").glob("freeze_*/freeze_manifest.json"))
    if not candidates:
        raise FileNotFoundError("No data/freeze_*/freeze_manifest.json found")
    return candidates[-1]


def normalize_whitespace(value):
    return " ".join(str(value or "").split())


def normalize_text_key(value):
    return normalize_whitespace(value).casefold()


def text_features(
    value, language, short_text, min_english_words, min_chinese_chars
):
    clean = normalize_whitespace(value)
    key = clean.casefold()
    english_words = len(ENGLISH_WORD.findall(clean))
    chinese_chars = len(CHINESE_CHARACTER.findall(clean))
    if language == "english":
        low_information = english_words < min_english_words
    elif language == "schinese":
        low_information = (
            chinese_chars < min_chinese_chars
            and english_words < min_english_words
        )
    else:
        low_information = english_words < min_english_words and chinese_chars == 0
    return {
        "review_clean": clean,
        "text_key": key,
        "text_length": len(key),
        "english_word_count": english_words,
        "chinese_character_count": chinese_chars,
        "is_empty_text": not bool(key),
        "is_short_text": bool(key) and len(key) < short_text,
        "is_low_information": bool(key) and low_information,
    }


def parse_number(value):
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_timestamp(value):
    try:
        return datetime.fromtimestamp(int(value), timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_quality_table(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    return {(str(row["appid"]), row["language"]): row for row in rows}


def quality_counts(rows, short_text):
    keys = [normalize_text_key(row.get("review")) for row in rows]
    nonempty = [key for key in keys if key]
    return {
        "raw_review_count": len(rows),
        "empty_text_count": sum(not key for key in keys),
        "short_text_count": sum(bool(key) and len(key) < short_text for key in keys),
        "duplicate_text_count": sum(
            count - 1 for count in Counter(nonempty).values() if count > 1
        ),
    }


def reconcile_quality(rows_by_group, quality_rows, short_text):
    columns = (
        "raw_review_count",
        "empty_text_count",
        "short_text_count",
        "duplicate_text_count",
    )
    mismatches = []
    for key, rows in rows_by_group.items():
        expected = quality_rows.get(key)
        if expected is None:
            mismatches.append(f"missing quality row for appid={key[0]} language={key[1]}")
            continue
        actual = quality_counts(rows, short_text)
        for column in columns:
            if actual[column] != int(expected[column]):
                mismatches.append(
                    f"appid={key[0]} language={key[1]} {column}: "
                    f"quality={expected[column]} computed={actual[column]}"
                )
    extra = sorted(set(quality_rows) - set(rows_by_group))
    mismatches.extend(
        f"quality row has no source group: appid={appid} language={language}"
        for appid, language in extra
    )
    if mismatches:
        raise ValueError("Quality table reconciliation failed:\n" + "\n".join(mismatches))


def author_id(row):
    return str(row.get("steamid") or row.get("author", {}).get("steamid") or "")


def record_time_key(row):
    updated = row.get("timestamp_updated")
    created = row.get("timestamp_created")
    try:
        stamp = int(updated if updated not in (None, "") else created)
    except (TypeError, ValueError):
        stamp = -1
    return stamp, str(row.get("recommendationid", ""))


def canonical_text_key(row):
    created = row.get("timestamp_created")
    try:
        stamp = int(created)
    except (TypeError, ValueError):
        stamp = 2**63 - 1
    return stamp, str(row.get("recommendationid", ""))


def add_cleaning_fields(records, args):
    recommendation_ids = {}
    for record in records:
        recommendation_id = str(record.get("recommendationid", ""))
        if not recommendation_id:
            raise ValueError(
                f"Missing recommendationid for appid={record['_appid']} "
                f"language={record['_language']}"
            )
        if recommendation_id in recommendation_ids:
            other = recommendation_ids[recommendation_id]
            raise ValueError(
                f"Duplicate recommendationid {recommendation_id}: "
                f"{other['_appid']}/{other['_language']} and "
                f"{record['_appid']}/{record['_language']}"
            )
        recommendation_ids[recommendation_id] = record

    authors = defaultdict(list)
    texts = defaultdict(list)
    playtime_groups = defaultdict(list)
    for record in records:
        language = record["_language"]
        features = text_features(
            record.get("review"),
            language,
            args.short_text,
            args.min_english_words,
            args.min_chinese_chars,
        )
        features["is_short_text"] = (
            bool(features["text_key"])
            and len(features["text_key"]) < args.short_text
        )
        record.update(features)

        steamid = author_id(record)
        record["steamid"] = steamid
        if steamid:
            authors[(record["_appid"], language, steamid)].append(record)

        if record["text_key"]:
            texts[(record["_appid"], language, record["text_key"])].append(record)

        created = parse_timestamp(record.get("timestamp_created"))
        record["created_utc_clean"] = created.isoformat() if created else ""
        record["created_date_utc"] = created.date().isoformat() if created else ""
        record["is_invalid_timestamp"] = created is None

        at_review = parse_number(record.get("playtime_at_review"))
        forever = parse_number(record.get("playtime_forever"))
        record["playtime_at_review_missing"] = at_review is None
        record["playtime_forever_missing"] = forever is None
        record["playtime_at_review_invalid"] = at_review is not None and at_review < 0
        record["playtime_forever_invalid"] = forever is not None and forever < 0
        valid_at_review = at_review if at_review is not None and at_review >= 0 else None
        valid_forever = forever if forever is not None and forever >= 0 else None
        record["playtime_at_review_hours"] = (
            round(valid_at_review / 60, 6) if valid_at_review is not None else ""
        )
        record["playtime_forever_hours"] = (
            round(valid_forever / 60, 6) if valid_forever is not None else ""
        )
        record["is_playtime_outlier"] = (
            valid_at_review is not None and valid_at_review > args.max_playtime_minutes
        )
        record["is_playtime_inconsistent"] = (
            valid_at_review is not None
            and valid_forever is not None
            and valid_at_review > valid_forever
        )
        record["log1p_playtime_hours"] = (
            round(math.log1p(valid_at_review / 60), 9)
            if valid_at_review is not None
            else ""
        )
        if valid_at_review is not None:
            playtime_groups[(record["_appid"], language)].append(valid_at_review / 60)

    for group in authors.values():
        canonical = max(group, key=record_time_key)
        for record in group:
            record["author_review_group_size"] = len(group)
            record["is_duplicate_author_review"] = len(group) > 1
            record["is_author_record_canonical"] = record is canonical
    for record in records:
        if "is_author_record_canonical" not in record:
            record["author_review_group_size"] = 1
            record["is_duplicate_author_review"] = False
            record["is_author_record_canonical"] = True

    for group in texts.values():
        ordered = sorted(group, key=canonical_text_key)
        size = len(ordered)
        for rank, record in enumerate(ordered, 1):
            record["text_duplicate_group_size"] = size
            record["text_duplicate_rank"] = rank
            record["is_duplicate_text"] = size > 1
            record["is_text_canonical"] = rank == 1
            record["duplicate_inverse_weight"] = round(1 / size, 9)
    for record in records:
        if "is_text_canonical" not in record:
            record["text_duplicate_group_size"] = 0
            record["text_duplicate_rank"] = ""
            record["is_duplicate_text"] = False
            record["is_text_canonical"] = False
            record["duplicate_inverse_weight"] = ""

    p99_by_group = {
        key: percentile(values, 0.99) for key, values in playtime_groups.items()
    }
    for record in records:
        hours = record["playtime_at_review_hours"]
        p99 = p99_by_group.get((record["_appid"], record["_language"]))
        record["playtime_p99_group_hours"] = round(p99, 6) if p99 is not None else ""
        record["playtime_hours_winsorized_p99"] = (
            round(min(float(hours), p99), 6)
            if hours != "" and p99 is not None
            else ""
        )
        record["topic_unique_nonempty"] = (
            record["is_author_record_canonical"]
            and not record["is_empty_text"]
            and record["is_text_canonical"]
        )
        record["topic_fit_eligible"] = (
            record["topic_unique_nonempty"] and not record["is_low_information"]
        )
        record["association_eligible"] = (
            record["is_author_record_canonical"] and not record["is_empty_text"]
        )
        record["playtime_analysis_eligible"] = (
            record["association_eligible"]
            and not record["playtime_at_review_missing"]
            and not record["playtime_at_review_invalid"]
            and not record["is_playtime_outlier"]
            and not record["is_playtime_inconsistent"]
        )


def add_common_window_fields(records, languages, minimum):
    dates = defaultdict(list)
    for record in records:
        if record["created_date_utc"]:
            dates[(record["_appid"], record["_language"])].append(
                record["created_date_utc"]
            )

    windows = {}
    appids = {record["_appid"] for record in records}
    for appid in appids:
        language_ranges = []
        for language in languages:
            values = dates.get((appid, language), [])
            if not values:
                language_ranges = []
                break
            language_ranges.append((min(values), max(values)))
        if language_ranges:
            start = max(value[0] for value in language_ranges)
            end = min(value[1] for value in language_ranges)
            windows[appid] = (start, end) if start <= end else ("", "")
        else:
            windows[appid] = ("", "")

    unique_common_counts = Counter()
    for record in records:
        start, end = windows[record["_appid"]]
        in_window = bool(
            start
            and record["created_date_utc"]
            and start <= record["created_date_utc"] <= end
        )
        record["common_window_start_utc"] = start
        record["common_window_end_utc"] = end
        record["in_common_language_window"] = in_window
        if in_window and record["topic_unique_nonempty"]:
            unique_common_counts[(record["_appid"], record["_language"])] += 1

    eligible_appids = {
        appid
        for appid in appids
        if all(unique_common_counts[(appid, language)] >= minimum for language in languages)
    }
    for record in records:
        record["language_comparison_game_eligible"] = record["_appid"] in eligible_appids
        record["language_comparison_row_eligible"] = (
            record["association_eligible"]
            and record["in_common_language_window"]
            and record["_appid"] in eligible_appids
        )
    return windows, unique_common_counts, eligible_appids


def public_record(record):
    return {key: value for key, value in record.items() if not key.startswith("_")}


def csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def ordered_fields(rows):
    preferred = [
        "game",
        "appid",
        "language",
        "recommendationid",
        "steamid",
        "review",
        "review_clean",
        "voted_up",
        "timestamp_created",
        "created_utc_clean",
        "created_date_utc",
        "playtime_at_review",
        "playtime_at_review_hours",
        "log1p_playtime_hours",
        "playtime_hours_winsorized_p99",
        "is_empty_text",
        "is_short_text",
        "is_low_information",
        "is_duplicate_text",
        "is_text_canonical",
        "text_duplicate_group_size",
        "duplicate_inverse_weight",
        "is_duplicate_author_review",
        "is_author_record_canonical",
        "topic_unique_nonempty",
        "topic_fit_eligible",
        "association_eligible",
        "playtime_analysis_eligible",
        "in_common_language_window",
        "language_comparison_game_eligible",
        "language_comparison_row_eligible",
    ]
    keys = set()
    for row in rows:
        keys.update(row)
    fields = [field for field in preferred if field in keys]
    fields.extend(sorted(keys - set(fields) - {"author"}))
    return fields


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows):
    fields = ordered_fields(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def group_summary(records, rows_by_group, quality_rows, unique_common_counts, eligible_appids):
    summaries = []
    grouped_records = defaultdict(list)
    for record in records:
        grouped_records[(str(record["_appid"]), record["_language"])].append(record)
    for key in rows_by_group:
        group = grouped_records[key]
        quality = quality_rows[key]
        summaries.append({
            "game": group[0]["game"],
            "appid": int(key[0]),
            "language": key[1],
            "raw_review_count": len(group),
            "unique_recommendation_count": len({row["recommendationid"] for row in group}),
            "empty_text_count": sum(row["is_empty_text"] for row in group),
            "short_text_count": sum(row["is_short_text"] for row in group),
            "low_information_count": sum(row["is_low_information"] for row in group),
            "duplicate_text_extra_count": sum(
                row["is_duplicate_text"] and not row["is_text_canonical"] for row in group
            ),
            "duplicate_author_extra_count": sum(
                row["is_duplicate_author_review"]
                and not row["is_author_record_canonical"]
                for row in group
            ),
            "unique_nonempty_text_count": sum(row["topic_unique_nonempty"] for row in group),
            "topic_fit_count": sum(row["topic_fit_eligible"] for row in group),
            "association_count": sum(row["association_eligible"] for row in group),
            "playtime_analysis_count": sum(row["playtime_analysis_eligible"] for row in group),
            "common_window_start_utc": group[0]["common_window_start_utc"],
            "common_window_end_utc": group[0]["common_window_end_utc"],
            "unique_nonempty_common_window_count": unique_common_counts[
                (int(key[0]), key[1])
            ],
            "language_comparison_game_eligible": int(key[0]) in eligible_appids,
            "target_count": int(quality["target_count"]),
            "target_shortfall": int(quality["target_shortfall"]),
            "target_met": quality["target_met"].lower() == "true",
            "endpoint_status": quality["endpoint_status"],
        })
    return summaries


def prepare_output_directory(path: Path, overwrite: bool):
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / name for name in OUTPUT_FILES if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist in {path}; re-run with --overwrite"
        )
    if overwrite:
        for target in existing:
            target.unlink()


def run(args, root=ROOT):
    freeze_path = resolve_path(root, args.freeze_manifest) if args.freeze_manifest else latest_freeze_manifest(root)
    quality_path = resolve_path(root, args.quality_table)
    freeze = load_json(freeze_path)
    config_path = resolve_path(root, freeze["collection_config"])
    config = load_json(config_path)
    languages = list(config["collection_config"]["languages"])
    quality_rows = load_quality_table(quality_path)

    rows_by_group = {}
    records = []
    quarantined = []
    for selected in freeze["selected_runs"]:
        run_path = resolve_path(root, selected["path"])
        manifest = load_json(run_path / "manifest.json")
        endpoint_ok = all(
            item.get("status") == "ok" for item in manifest.get("endpoints", {}).values()
        )
        for language in languages:
            review_status = manifest.get("reviews", {}).get(language, {}).get("status")
            key = (str(selected["appid"]), language)
            if not endpoint_ok or review_status != "ok":
                quarantined.append({
                    "appid": selected["appid"],
                    "game": selected["name"],
                    "language": language,
                    "endpoint_ok": endpoint_ok,
                    "review_status": review_status,
                })
                continue
            source_path = run_path / language / "reviews.jsonl"
            group_rows = read_jsonl(source_path)
            rows_by_group[key] = group_rows
            for source_index, row in enumerate(group_rows):
                record = dict(row)
                record["game"] = selected["name"]
                record["appid"] = int(selected["appid"])
                record["language"] = language
                record["source_relative_path"] = source_path.relative_to(root).as_posix()
                record["source_row_number"] = source_index + 1
                record["_appid"] = int(selected["appid"])
                record["_language"] = language
                records.append(record)

    if quarantined:
        raise ValueError(
            "Endpoint or review group is not clean: "
            + json.dumps(quarantined, ensure_ascii=False)
        )
    reconcile_quality(rows_by_group, quality_rows, args.short_text)
    add_cleaning_fields(records, args)
    windows, unique_common_counts, eligible_appids = add_common_window_fields(
        records, languages, args.min_language_comparison
    )

    public_rows = [public_record(record) for record in records]
    topic_rows = [
        public_record(record) for record in records if record["topic_fit_eligible"]
    ]
    association_rows = [
        public_record(record) for record in records if record["association_eligible"]
    ]
    summary_rows = group_summary(
        records, rows_by_group, quality_rows, unique_common_counts, eligible_appids
    )
    summary = {
        "generated_at": utc_now(),
        "freeze_manifest": freeze_path.relative_to(root).as_posix(),
        "quality_table": quality_path.relative_to(root).as_posix(),
        "parameters": {
            "short_text_characters": args.short_text,
            "min_english_words": args.min_english_words,
            "min_chinese_characters": args.min_chinese_chars,
            "min_language_comparison_unique_nonempty_per_language": args.min_language_comparison,
            "max_playtime_minutes": args.max_playtime_minutes,
        },
        "counts": {
            "clean_master": len(public_rows),
            "topic_fit": len(topic_rows),
            "association_data": len(association_rows),
            "empty_text": sum(record["is_empty_text"] for record in records),
            "short_text": sum(record["is_short_text"] for record in records),
            "low_information": sum(record["is_low_information"] for record in records),
            "duplicate_text_extra": sum(
                record["is_duplicate_text"] and not record["is_text_canonical"]
                for record in records
            ),
            "duplicate_author_extra": sum(
                record["is_duplicate_author_review"]
                and not record["is_author_record_canonical"]
                for record in records
            ),
            "language_comparison_eligible_games": len(eligible_appids),
            "games": len({record["_appid"] for record in records}),
            "groups": len(rows_by_group),
        },
        "definitions": {
            "clean_master": "All globally unique recommendation IDs with cleaning flags; source rows are not overwritten.",
            "topic_fit": "Canonical author records with non-empty, canonical normalized text and sufficient language-specific information.",
            "association_data": "Canonical author records with non-empty text; duplicate text is retained with inverse-frequency weights.",
            "language_comparison": "Both configured languages have the minimum unique non-empty texts inside the same-game common UTC date window.",
        },
        "eligible_appids": sorted(eligible_appids),
        "common_windows": {
            str(appid): {"start_utc": start, "end_utc": end}
            for appid, (start, end) in sorted(windows.items())
        },
        "groups": summary_rows,
    }

    if args.output_dir:
        output_dir = resolve_path(root, args.output_dir)
    else:
        output_dir = root / "data" / "cleaned" / freeze_path.parent.name
    prepare_output_directory(output_dir, args.overwrite)

    write_jsonl(output_dir / "clean_master.jsonl", public_rows)
    write_csv(output_dir / "clean_master.csv", public_rows)
    write_jsonl(output_dir / "topic_fit.jsonl", topic_rows)
    write_csv(output_dir / "topic_fit.csv", topic_rows)
    write_jsonl(output_dir / "association_data.jsonl", association_rows)
    write_csv(output_dir / "association_data.csv", association_rows)
    (output_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "cleaning_summary.csv", summary_rows)

    generated = []
    for name in OUTPUT_FILES:
        path = output_dir / name
        if name == "cleaning_manifest.json":
            continue
        generated.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    cleaning_manifest = {
        "generated_at": summary["generated_at"],
        "source_freeze_manifest": freeze_path.relative_to(root).as_posix(),
        "source_freeze_manifest_sha256": sha256(freeze_path),
        "source_quality_table": quality_path.relative_to(root).as_posix(),
        "source_quality_table_sha256": sha256(quality_path),
        "cleaning_script": portable_path(Path(__file__), root),
        "cleaning_script_sha256": sha256(Path(__file__)),
        "parameters": summary["parameters"],
        "counts": summary["counts"],
        "files": generated,
    }
    (output_dir / "cleaning_manifest.json").write_text(
        json.dumps(cleaning_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "cleaned",
        "output_dir": output_dir.relative_to(root).as_posix(),
        **summary["counts"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        help="Frozen batch manifest; defaults to the latest data/freeze_*/freeze_manifest.json",
    )
    parser.add_argument(
        "--quality-table", type=Path, default=Path("data/quality_table.csv")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--short-text", type=int, default=DEFAULT_SHORT_TEXT)
    parser.add_argument(
        "--min-english-words", type=int, default=DEFAULT_MIN_ENGLISH_WORDS
    )
    parser.add_argument(
        "--min-chinese-chars", type=int, default=DEFAULT_MIN_CHINESE_CHARS
    )
    parser.add_argument(
        "--min-language-comparison",
        type=int,
        default=DEFAULT_MIN_LANGUAGE_COMPARISON,
    )
    parser.add_argument(
        "--max-playtime-minutes",
        type=int,
        default=DEFAULT_MAX_PLAYTIME_MINUTES,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    try:
        run(build_parser().parse_args())
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"clean_reviews: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

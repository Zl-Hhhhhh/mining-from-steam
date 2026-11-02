"""Run a reproducible, offline quality audit of a frozen Steam review batch."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHORT_TEXT = 10
DEFAULT_MAX_PLAYTIME_MINUTES = 10 * 365 * 24 * 60
CSV_FIELDS = ["severity", "code", "appid", "game", "language", "count", "path", "details"]
QUALITY_CSV_FIELDS = [
    "game", "appid", "language", "raw_review_count", "source_total_matching",
    "recommendation_count", "recommendation_rate", "earliest_review_date_utc",
    "latest_review_date_utc", "actual_coverage_days", "empty_text_count",
    "short_text_count", "duplicate_text_count", "playtime_missing_rate",
    "playtime_zero_rate", "playtime_p25_hours", "playtime_p50_hours",
    "playtime_p75_hours", "playtime_p90_hours", "playtime_p95_hours",
    "playtime_p99_hours", "unique_author_count", "duplicate_author_count",
    "duplicate_author_review_count", "core_endpoint_status", "review_endpoint_status",
    "endpoint_status", "target_count", "target_shortfall", "target_met",
    "met_200_target", "directory",
]
PLAYTIME_PERCENTILES = ((25, 0.25), (50, 0.50), (75, 0.75), (90, 0.90), (95, 0.95), (99, 0.99))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def normalize_text(value):
    return " ".join(str(value or "").split()).casefold()


def rate(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class Audit:
    def __init__(self, root, short_text, max_playtime):
        self.root = root
        self.short_text = short_text
        self.max_playtime = max_playtime
        self.issues = []
        self.games = []
        self.quality_rows = []
        self.author_games = defaultdict(set)
        self.global_recommendations = defaultdict(list)

    def issue(self, severity, code, *, appid="", game="", language="", count=1, path="", details=""):
        self.issues.append({
            "severity": severity,
            "code": code,
            "appid": appid,
            "game": game,
            "language": language,
            "count": count,
            "path": str(path),
            "details": details,
        })

    def error(self, code, **kwargs):
        self.issue("error", code, **kwargs)

    def warning(self, code, **kwargs):
        self.issue("warning", code, **kwargs)

    def info(self, code, **kwargs):
        self.issue("info", code, **kwargs)

    def load_json(self, path, context):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            self.error("invalid_json", path=path, details=f"{context}: {error}")
            return None

    def load_jsonl(self, path, *, appid, game, language):
        rows = []
        try:
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as error:
                        self.error("invalid_jsonl", appid=appid, game=game, language=language,
                                   path=path, details=f"line {line_number}: {error}")
                        continue
                    if not isinstance(value, dict):
                        self.error("invalid_review_record", appid=appid, game=game, language=language,
                                   path=path, details=f"line {line_number} is not an object")
                        continue
                    rows.append(value)
        except OSError as error:
            self.error("missing_or_unreadable_file", appid=appid, game=game, language=language,
                       path=path, details=str(error))
        return rows

    def load_csv(self, path, *, appid, game, language):
        try:
            with path.open(encoding="utf-8-sig", newline="") as source:
                return list(csv.DictReader(source))
        except (OSError, csv.Error) as error:
            self.error("missing_or_unreadable_file", appid=appid, game=game, language=language,
                       path=path, details=str(error))
            return []

    def audit_dataset(self, directory, *, appid, game, language, expected_count, window, quality=True):
        jsonl_path = directory / "reviews.jsonl"
        csv_path = directory / "reviews.csv"
        rows = self.load_jsonl(jsonl_path, appid=appid, game=game, language=language)
        csv_rows = self.load_csv(csv_path, appid=appid, game=game, language=language)
        label = language or "combined"

        if len(rows) != len(csv_rows) or len(rows) != expected_count:
            self.error(
                "row_count_mismatch", appid=appid, game=game, language=language,
                count=max(len(rows), len(csv_rows), expected_count), path=directory,
                details=f"{label}: jsonl={len(rows)}, csv={len(csv_rows)}, manifest={expected_count}",
            )

        json_ids = [str(row.get("recommendationid", "")) for row in rows]
        csv_ids = [str(row.get("recommendationid", "")) for row in csv_rows]
        if json_ids != csv_ids:
            self.error("csv_jsonl_id_mismatch", appid=appid, game=game, language=language,
                       path=directory, details=f"{label}: recommendation ID order/content differs")

        missing_ids = sum(not value for value in json_ids)
        duplicate_ids = len(json_ids) - len(set(json_ids))
        if missing_ids:
            self.error("missing_recommendation_id", appid=appid, game=game, language=language,
                       count=missing_ids, path=jsonl_path, details=f"{label}: empty recommendation IDs")
        if duplicate_ids:
            self.error("duplicate_recommendation_id", appid=appid, game=game, language=language,
                       count=duplicate_ids, path=jsonl_path, details=f"{label}: duplicate IDs")

        invalid_appids = sum(str(row.get("appid", "")) != str(appid) for row in rows)
        invalid_languages = sum(bool(language) and row.get("language") != language for row in rows)
        invalid_csv_appids = sum(str(row.get("appid", "")) != str(appid) for row in csv_rows)
        invalid_csv_languages = sum(bool(language) and row.get("language") != language for row in csv_rows)
        invalid_csv_labels = sum(str(row.get("voted_up", "")).casefold() not in ("true", "false")
                                 for row in csv_rows)
        if invalid_appids:
            self.error("appid_mismatch", appid=appid, game=game, language=language,
                       count=invalid_appids, path=jsonl_path, details=f"expected appid {appid}")
        if invalid_languages:
            self.error("language_mismatch", appid=appid, game=game, language=language,
                       count=invalid_languages, path=jsonl_path, details=f"expected language {language}")
        if invalid_csv_appids:
            self.error("csv_appid_mismatch", appid=appid, game=game, language=language,
                       count=invalid_csv_appids, path=csv_path, details=f"expected appid {appid}")
        if invalid_csv_languages:
            self.error("csv_language_mismatch", appid=appid, game=game, language=language,
                       count=invalid_csv_languages, path=csv_path, details=f"expected language {language}")
        if invalid_csv_labels:
            self.error("invalid_csv_recommendation_label", appid=appid, game=game, language=language,
                       count=invalid_csv_labels, path=csv_path, details="voted_up must be true or false")

        invalid_timestamps = 0
        outside_window = 0
        invalid_labels = 0
        missing_playtime = 0
        negative_playtime = 0
        abnormal_playtime = 0
        empty_text = 0
        short_text = 0
        texts = []
        authors = []
        created_dates = []
        recommendation_count = 0
        playtime_at_review_missing = 0
        playtime_at_review_zero = 0
        playtime_at_review_values = []
        start, end = window
        for row in rows:
            timestamp = row.get("timestamp_created")
            try:
                created = datetime.fromtimestamp(int(timestamp), timezone.utc)
            except (TypeError, ValueError, OSError, OverflowError):
                invalid_timestamps += 1
            else:
                outside_window += created < start or created > end
                created_dates.append(created)

            if not isinstance(row.get("voted_up"), bool):
                invalid_labels += 1
            elif row["voted_up"]:
                recommendation_count += 1

            for field in ("playtime_forever", "playtime_at_review"):
                value = row.get(field)
                if value is None or value == "":
                    missing_playtime += 1
                    continue
                try:
                    minutes = float(value)
                except (TypeError, ValueError):
                    missing_playtime += 1
                    continue
                negative_playtime += minutes < 0
                abnormal_playtime += minutes > self.max_playtime

            review_playtime = row.get("playtime_at_review")
            if review_playtime is None or review_playtime == "":
                playtime_at_review_missing += 1
            else:
                try:
                    review_minutes = float(review_playtime)
                except (TypeError, ValueError):
                    playtime_at_review_missing += 1
                else:
                    if review_minutes >= 0:
                        playtime_at_review_values.append(review_minutes)
                        playtime_at_review_zero += review_minutes == 0

            text = normalize_text(row.get("review"))
            empty_text += not text
            short_text += bool(text) and len(text) < self.short_text
            if text:
                texts.append(text)

            author = str(row.get("steamid") or row.get("author", {}).get("steamid") or "")
            if author:
                authors.append(author)

            recommendation_id = str(row.get("recommendationid", ""))
            if recommendation_id and not language:
                self.global_recommendations[recommendation_id].append((appid, jsonl_path))
            if author and not language:
                self.author_games[author].add(appid)

        checks = [
            (invalid_timestamps, "invalid_timestamp", "timestamp_created is missing or invalid"),
            (outside_window, "review_outside_freeze_window", f"expected {start.isoformat()} through {end.isoformat()}"),
            (invalid_labels, "invalid_recommendation_label", "voted_up must be a JSON boolean"),
            (missing_playtime, "missing_playtime", "playtime_forever/playtime_at_review missing or non-numeric"),
            (negative_playtime, "negative_playtime", "playtime must be non-negative"),
            (abnormal_playtime, "abnormal_playtime", f"playtime exceeds {self.max_playtime} minutes"),
            (empty_text, "empty_review_text", "review text is empty after whitespace normalization"),
            (short_text, "short_review_text", f"normalized review text is shorter than {self.short_text} characters"),
        ]
        for count, code, details in (checks if quality else []):
            if count:
                self.error(code, appid=appid, game=game, language=language, count=count,
                           path=jsonl_path, details=details)

        duplicate_texts = sum(count - 1 for count in Counter(texts).values() if count > 1)
        if quality and duplicate_texts:
            self.error("duplicate_review_text", appid=appid, game=game, language=language,
                       count=duplicate_texts, path=jsonl_path,
                       details="case/whitespace-normalized duplicate review text")

        duplicate_authors = sum(count - 1 for count in Counter(authors).values() if count > 1)
        if not language and duplicate_authors:
            self.error("duplicate_author_within_game", appid=appid, game=game, language=language,
                       count=duplicate_authors, path=jsonl_path,
                       details="additional reviews by an author already present in this dataset")

        author_counts = Counter(authors)
        earliest = min(created_dates, default=None)
        latest = max(created_dates, default=None)
        coverage_days = (latest.date() - earliest.date()).days + 1 if earliest and latest else None
        playtime_percentiles = {
            f"playtime_p{label}_hours": (
                round(value / 60, 2) if (value := percentile(playtime_at_review_values, quantile)) is not None else None
            )
            for label, quantile in PLAYTIME_PERCENTILES
        }
        stats = {
            "raw_review_count": len(rows),
            "recommendation_count": recommendation_count,
            "recommendation_rate": rate(recommendation_count, len(rows)),
            "earliest_review_date_utc": earliest.date().isoformat() if earliest else None,
            "latest_review_date_utc": latest.date().isoformat() if latest else None,
            "actual_coverage_days": coverage_days,
            "empty_text_count": empty_text,
            "short_text_count": short_text,
            "duplicate_text_count": duplicate_texts,
            "playtime_missing_rate": rate(playtime_at_review_missing, len(rows)),
            "playtime_zero_rate": rate(playtime_at_review_zero, len(rows)),
            "unique_author_count": len(author_counts),
            "duplicate_author_count": sum(count > 1 for count in author_counts.values()),
            "duplicate_author_review_count": sum(count - 1 for count in author_counts.values() if count > 1),
            **playtime_percentiles,
        }
        return rows, stats

    def audit_game(self, selected, games_by_appid, freeze_window, target_per_language, expected_languages):
        appid = selected.get("appid")
        game = selected.get("name", "")
        relative_path = Path(str(selected.get("path", "")))
        directory = self.root / relative_path
        expected_prefix = Path("data") / "steam" / str(appid)
        if len(relative_path.parts) != 4 or relative_path.parts[:3] != expected_prefix.parts:
            self.error("official_directory_mismatch", appid=appid, game=game, path=relative_path,
                       details=f"expected data/steam/{appid}/<run>")
        if str(selected.get("run", "")) != relative_path.name:
            self.error("run_directory_mismatch", appid=appid, game=game, path=relative_path,
                       details=f"run={selected.get('run')!r}, directory={relative_path.name!r}")

        config_game = games_by_appid.get(str(appid))
        if config_game is None:
            self.error("appid_not_in_config", appid=appid, game=game, path=relative_path,
                       details="selected appid is absent from collection config")
        elif game != config_game.get("name"):
            self.error("game_name_mismatch", appid=appid, game=game, path=relative_path,
                       details=f"freeze={game!r}, config={config_game.get('name')!r}")

        manifest = self.load_json(directory / "manifest.json", f"manifest for appid {appid}")
        if not isinstance(manifest, dict):
            return
        if str(manifest.get("appid")) != str(appid):
            self.error("manifest_appid_mismatch", appid=appid, game=game, path=directory / "manifest.json",
                       details=f"manifest appid={manifest.get('appid')!r}")
        manifest_directory = Path(str(manifest.get("directory", "")))
        if manifest_directory.parts != relative_path.parts:
            self.error("manifest_directory_mismatch", appid=appid, game=game, path=directory / "manifest.json",
                       details=f"manifest={manifest_directory}, freeze={relative_path}")

        manifest_window = manifest.get("sampling_window", {})
        try:
            run_window = (parse_utc(manifest_window.get("start_utc")), parse_utc(manifest_window.get("end_utc")))
        except (TypeError, ValueError):
            run_window = None
        if run_window != freeze_window:
            self.error("sampling_window_mismatch", appid=appid, game=game, path=directory / "manifest.json",
                       details="run sampling window differs from freeze manifest")

        endpoints = manifest.get("endpoints", {})
        required_endpoints = {"players", "news", "store"}
        failed_endpoints = sorted(
            name for name in required_endpoints | set(endpoints)
            if not isinstance(endpoints.get(name), dict) or endpoints[name].get("status") != "ok"
        )
        if failed_endpoints:
            self.error("endpoint_failure", appid=appid, game=game, count=len(failed_endpoints),
                       path=directory / "manifest.json", details=", ".join(failed_endpoints))

        store = self.load_json(directory / "store.json", f"store data for appid {appid}")
        store_name = None
        if isinstance(store, dict):
            store_entry = store.get("data", {}).get(str(appid), {})
            store_name = store_entry.get("data", {}).get("name") if store_entry.get("success") else None
        if not store_name:
            self.error("missing_store_name", appid=appid, game=game, path=directory / "store.json",
                       details="successful store name not found")
        expected_store_name = (config_game or {}).get("verified_store_name", game)
        if store_name and store_name != expected_store_name:
            self.error("game_name_mismatch", appid=appid, game=game, path=directory / "store.json",
                       details=f"expected store={expected_store_name!r}, actual store={store_name!r}")

        reviews = manifest.get("reviews", {})
        manifest_languages = manifest.get("languages", [])
        if set(manifest_languages) != set(reviews):
            self.error("manifest_language_mismatch", appid=appid, game=game, path=directory / "manifest.json",
                       details=f"languages={manifest_languages!r}, reviews={list(reviews)!r}")
        if set(manifest_languages) != set(expected_languages):
            self.error("configured_language_mismatch", appid=appid, game=game,
                       path=directory / "manifest.json",
                       details=f"configured={expected_languages!r}, manifest={manifest_languages!r}")

        language_total = 0
        language_stats = {}
        languages_to_audit = list(dict.fromkeys([*expected_languages, *manifest_languages, *reviews]))
        for language in languages_to_audit:
            raw_state = reviews.get(language, {})
            state = raw_state if isinstance(raw_state, dict) else {}
            raw_count = state.get("count", 0)
            count = raw_count if isinstance(raw_count, int) and raw_count >= 0 else 0
            if raw_count != count:
                self.error("invalid_manifest_count", appid=appid, game=game, language=language,
                           path=directory / "manifest.json", details=f"count={raw_count!r}")
            language_total += count
            if state.get("status") != "ok":
                self.error("review_endpoint_failure", appid=appid, game=game, language=language,
                           path=directory / "manifest.json", details=str(state.get("error", state.get("status"))))
            raw_target = state.get("requested", manifest.get("requested_per_language", target_per_language))
            target = raw_target if isinstance(raw_target, int) and raw_target >= 0 else target_per_language
            if count < target:
                self.error("language_target_shortfall", appid=appid, game=game, language=language,
                           count=target - count, path=directory / "manifest.json",
                           details=f"collected={count}, target={target}")
            language_rows, quality_stats = self.audit_dataset(
                directory / language, appid=appid, game=game, language=language,
                expected_count=count, window=freeze_window,
            )
            language_stats[language] = {"count": len(language_rows), "target": target}
            review_endpoint_status = str(state.get("status", "missing"))
            core_endpoint_status = "ok" if not failed_endpoints else f"failed:{','.join(failed_endpoints)}"
            endpoint_status = "ok" if core_endpoint_status == review_endpoint_status == "ok" else (
                f"core={core_endpoint_status};review={review_endpoint_status}"
            )
            self.quality_rows.append({
                "game": game,
                "appid": appid,
                "language": language,
                **quality_stats,
                "source_total_matching": state.get("total_matching"),
                "core_endpoint_status": core_endpoint_status,
                "review_endpoint_status": review_endpoint_status,
                "endpoint_status": endpoint_status,
                "target_count": target,
                "target_shortfall": max(0, target - len(language_rows)),
                "target_met": len(language_rows) >= target,
                "met_200_target": len(language_rows) >= 200,
                "directory": str(relative_path / language),
            })

        manifest_count = manifest.get("review_count", 0)
        freeze_count = selected.get("review_count", 0)
        if manifest_count != language_total or manifest_count != freeze_count:
            self.error("manifest_count_mismatch", appid=appid, game=game, path=directory / "manifest.json",
                       details=f"freeze={freeze_count}, manifest={manifest_count}, language_sum={language_total}")
        combined_rows, _ = self.audit_dataset(
            directory, appid=appid, game=game, language="",
            expected_count=manifest_count, window=freeze_window, quality=False,
        )
        self.games.append({
            "appid": appid,
            "name": game,
            "directory": str(relative_path),
            "review_count": len(combined_rows),
            "languages": language_stats,
        })

    def finish_global_checks(self):
        duplicate_ids = {key: values for key, values in self.global_recommendations.items() if len(values) > 1}
        if duplicate_ids:
            self.error("duplicate_recommendation_id_across_games", count=sum(len(v) - 1 for v in duplicate_ids.values()),
                       details=f"{len(duplicate_ids)} recommendation IDs occur in multiple combined datasets")

        cross_game = {author: appids for author, appids in self.author_games.items() if len(appids) > 1}
        self.info("cross_game_author_count", count=len(cross_game),
                  details="unique authors appearing in reviews for more than one game")
        return cross_game


def latest_freeze_manifest(root):
    candidates = sorted((root / "data").glob("freeze_*/freeze_manifest.json"), reverse=True)
    if not candidates:
        raise FileNotFoundError("no data/freeze_*/freeze_manifest.json found")
    return candidates[0]


def write_csv(path, issues):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(issues)
    temporary.replace(path)


def write_quality_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=QUALITY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, help="freeze manifest to audit; defaults to latest")
    parser.add_argument("--output-json", type=Path, default=ROOT / "data/verification.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "data/verification.csv")
    parser.add_argument("--quality-json", type=Path, default=ROOT / "data/quality_table.json")
    parser.add_argument("--quality-csv", type=Path, default=ROOT / "data/quality_table.csv")
    parser.add_argument("--short-text", type=int, default=DEFAULT_SHORT_TEXT,
                        help=f"minimum normalized review length (default: {DEFAULT_SHORT_TEXT})")
    parser.add_argument("--max-playtime-minutes", type=int, default=DEFAULT_MAX_PLAYTIME_MINUTES,
                        help="values above this threshold are abnormal")
    return parser


def run(args, root=ROOT):
    manifest_path = args.freeze_manifest or latest_freeze_manifest(root)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    audit = Audit(root, args.short_text, args.max_playtime_minutes)
    freeze = audit.load_json(manifest_path, "freeze manifest")
    if not isinstance(freeze, dict):
        freeze = {}

    collection_path = root / freeze.get("collection_config", "config/games_clustering.json")
    collection = audit.load_json(collection_path, "collection config") or {}
    games_by_appid = {str(game.get("appid")): game for game in collection.get("games", [])}
    target_per_language = collection.get("collection_config", {}).get("reviews_per_language", 0)
    expected_languages = collection.get("collection_config", {}).get("languages", [])

    try:
        window = (
            parse_utc(freeze.get("sampling_window", {}).get("start_utc")),
            parse_utc(freeze.get("sampling_window", {}).get("end_utc")),
        )
    except (TypeError, ValueError):
        audit.error("invalid_freeze_window", path=manifest_path, details="missing or invalid sampling_window")
        window = (datetime.min.replace(tzinfo=timezone.utc), datetime.max.replace(tzinfo=timezone.utc))

    selected_runs = freeze.get("selected_runs", [])
    if freeze.get("game_count") != len(selected_runs):
        audit.error("freeze_game_count_mismatch", path=manifest_path,
                    details=f"game_count={freeze.get('game_count')}, selected_runs={len(selected_runs)}")
    for selected in selected_runs:
        audit.audit_game(selected, games_by_appid, window, target_per_language, expected_languages)

    cross_game = audit.finish_global_checks()
    audited_reviews = sum(game["review_count"] for game in audit.games)
    if freeze.get("total_review_count") != audited_reviews:
        audit.error("freeze_review_count_mismatch", path=manifest_path,
                    details=f"freeze={freeze.get('total_review_count')}, audited={audited_reviews}")
    severity_counts = Counter(issue["severity"] for issue in audit.issues)

    try:
        freeze_label = str(manifest_path.relative_to(root))
    except ValueError:
        freeze_label = str(manifest_path)
    audited_at = utc_now()
    report = {
        "audited_at": audited_at,
        "status": "failed" if severity_counts["error"] else "ok",
        "freeze_manifest": freeze_label,
        "parameters": {
            "short_text": args.short_text,
            "max_playtime_minutes": args.max_playtime_minutes,
        },
        "summary": {
            "games": len(audit.games),
            "reviews": audited_reviews,
            "unique_authors": len(audit.author_games),
            "cross_game_authors": len(cross_game),
            "errors": severity_counts["error"],
            "warnings": severity_counts["warning"],
            "info": severity_counts["info"],
        },
        "games": audit.games,
        "issues": audit.issues,
    }
    quality_report = {
        "generated_at": audited_at,
        "freeze_manifest": freeze_label,
        "definitions": {
            "row_granularity": "one row per game and language",
            "actual_coverage_days": "inclusive count of UTC calendar dates from earliest to latest valid review",
            "duplicate_text_count": "extra non-empty reviews after case/whitespace normalization",
            "duplicate_author_count": "unique authors with more than one review in the row",
            "duplicate_author_review_count": "extra reviews beyond one per author",
            "playtime_field": "playtime_at_review",
            "playtime_rates_denominator": "raw_review_count",
            "playtime_percentiles": "valid non-negative values, converted from minutes to hours with linear interpolation",
        },
        "rows": audit.quality_rows,
    }
    save_json(args.output_json, report)
    write_csv(args.output_csv, audit.issues)
    save_json(args.quality_json, quality_report)
    write_quality_csv(args.quality_csv, audit.quality_rows)
    # ASCII escapes keep stdout portable on Windows consoles using legacy code pages.
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 1 if report["status"] == "failed" else 0


def main():
    args = build_parser().parse_args()
    try:
        return run(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"verify_data: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

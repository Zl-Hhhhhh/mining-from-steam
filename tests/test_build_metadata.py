import argparse
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "build_metadata", Path(__file__).parents[1] / "scripts/build_metadata.py"
)
build_metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_metadata)


class BuildMetadataTests(unittest.TestCase):
    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_csv(self, path, rows, fields):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def make_fixture(self, root):
        review_fields = [row["field"] for row in build_metadata.review_field_specs()]
        quality_fields = [row["field"] for row in build_metadata.quality_field_specs()]
        clean = root / "data/cleaned/freeze_test/clean_master.csv"
        self.write_csv(clean, [{field: "" for field in review_fields}], review_fields)
        quality = root / "data/quality_table.csv"
        rows = []
        for appid, game, english_count, chinese_count in (
            (10, "Complete Game", 200, 200),
            (238960, "Path of Exile", 200, 22),
        ):
            for language, count in (("english", english_count), ("schinese", chinese_count)):
                row = {field: "" for field in quality_fields}
                row.update({
                    "game": game, "appid": appid, "language": language,
                    "raw_review_count": count, "source_total_matching": count,
                    "recommendation_count": count, "recommendation_rate": 1,
                    "earliest_review_date_utc": "2026-01-01",
                    "latest_review_date_utc": "2026-01-31", "actual_coverage_days": 31,
                    "empty_text_count": 0, "short_text_count": 0, "duplicate_text_count": 0,
                    "playtime_missing_rate": 0, "playtime_zero_rate": 0,
                    "unique_author_count": count, "duplicate_author_count": 0,
                    "duplicate_author_review_count": 0, "core_endpoint_status": "ok",
                    "review_endpoint_status": "ok", "endpoint_status": "ok",
                    "target_count": 200, "target_shortfall": 200 - count,
                    "target_met": count >= 200, "met_200_target": count >= 200,
                    "directory": f"data/steam/{appid}/run/{language}",
                })
                rows.append(row)
        self.write_csv(quality, rows, quality_fields)

        games = []
        selected = []
        summary_groups = []
        for appid, game in ((10, "Complete Game"), (238960, "Path of Exile")):
            games.append({
                "appid": appid, "name": game, "genre": ["RPG", "ARPG"],
                "price": "Free", "verification_status": "ok", "name_match": True,
            })
            selected.append({"appid": appid, "name": game, "run": "run", "path": f"data/steam/{appid}/run"})
            manifest = {
                "requested_per_language": 200,
                "sampling": "fixed test window; not a random sample",
                "endpoints": {"store": {"status": "ok"}},
            }
            if appid == 238960:
                manifest["endpoints"]["store"]["fallback_cc"] = "us"
            self.write_json(root / f"data/steam/{appid}/run/manifest.json", manifest)
            store_data = {
                "type": "game", "name": game, "is_free": True,
                "genres": [{"id": "3", "description": "RPG"}],
                "categories": [{"id": 1, "description": "Multi-player"}],
                "developers": ["Developer"], "publishers": ["Publisher"],
                "release_date": {"coming_soon": False, "date": "Oct 23, 2013"},
                "platforms": {"windows": True, "mac": False, "linux": False},
            }
            self.write_json(root / f"data/steam/{appid}/run/store.json", {
                "fetched_at": "2026-01-31T00:00:00+00:00",
                "endpoint": "https://store.steampowered.com/api/appdetails",
                "data": {str(appid): {"success": True, "data": store_data}},
            })
            for language in ("english", "schinese"):
                count = 200 if appid == 10 or language == "english" else 22
                summary_groups.append({
                    "appid": appid, "game": game, "language": language,
                    "common_window_start_utc": "2026-01-01",
                    "common_window_end_utc": "2026-01-31",
                    "unique_nonempty_common_window_count": count,
                    "language_comparison_game_eligible": appid == 10,
                })
        config = root / "config/games_clustering.json"
        self.write_json(config, {"collection_config": {"languages": ["english", "schinese"]}, "games": games})
        freeze = root / "data/freeze_test/freeze_manifest.json"
        self.write_json(freeze, {
            "collection_config": "config/games_clustering.json", "game_count": 2,
            "sampling_window": {"start_utc": "2026-01-01T00:00:00+00:00", "end_utc": "2026-01-31T23:59:59+00:00", "days": 31},
            "selected_runs": selected,
        })
        summary = root / "data/cleaned/freeze_test/cleaning_summary.json"
        self.write_json(summary, {"groups": summary_groups})
        return freeze, quality, clean, summary

    def test_builds_complete_dictionary_and_documents_partial_and_store_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze, quality, clean, summary = self.make_fixture(root)
            args = argparse.Namespace(
                freeze_manifest=freeze, quality_table=quality, clean_master=clean,
                cleaning_summary=summary, output_dir=root / "data/metadata/freeze_test",
                docs_output=root / "docs/data-dictionary.md", overwrite=False,
            )
            result = build_metadata.run(args, root=root)
            self.assertEqual(result["games"], 2)
            self.assertEqual(result["partial_games"], 1)
            self.assertEqual(result["store_fallback_games"], 1)

            dictionary = build_metadata.read_csv(args.output_dir / "data_dictionary.csv")
            playtime = next(row for row in dictionary if row["dataset"] == "review_datasets" and row["field"] == "playtime_at_review")
            self.assertEqual(playtime["unit"], "minutes")
            game_meta = build_metadata.read_csv(args.output_dir / "game_meta.csv")
            poe = next(row for row in game_meta if row["game"] == "Path of Exile")
            self.assertEqual(poe["sample_status"], "partial")
            self.assertEqual(poe["partial_languages"], "schinese")
            self.assertEqual(poe["store_fallback_cc"], "us")
            self.assertIn("cc=cn", poe["store_metadata_note"])
            docs = args.docs_output.read_text(encoding="utf-8")
            self.assertIn("Path of Exile Store Endpoint", docs)
            self.assertIn("only 22 Simplified Chinese reviews", docs)


if __name__ == "__main__":
    unittest.main()

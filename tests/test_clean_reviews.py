import argparse
import csv
import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "clean_reviews", Path(__file__).parents[1] / "scripts/clean_reviews.py"
)
clean_reviews = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clean_reviews)


class CleanReviewsTests(unittest.TestCase):
    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def make_fixture(self, root):
        created = int(datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp())

        def row(recommendationid, steamid, language, review, offset=0):
            return {
                "appid": 10,
                "recommendationid": recommendationid,
                "steamid": steamid,
                "language": language,
                "review": review,
                "voted_up": True,
                "timestamp_created": created + offset,
                "timestamp_updated": created + offset,
                "playtime_forever": 180,
                "playtime_at_review": 120,
            }

        english = [
            row("e1", "s1", "english", "Great tactical combat", 0),
            row("e2", "s2", "english", "  GREAT  tactical combat ", 1),
            row("e3", "s3", "english", "ok", 2),
            row("e4", "s4", "english", "", 3),
        ]
        schinese = [
            row("z1", "s5", "schinese", "画面非常优秀", 0),
            row("z2", "s6", "schinese", "好玩", 1),
            row("z3", "s7", "schinese", "画面非常优秀", 2),
        ]
        run = root / "data/steam/10/run1"
        self.write_jsonl(run / "english/reviews.jsonl", english)
        self.write_jsonl(run / "schinese/reviews.jsonl", schinese)
        self.write_json(run / "manifest.json", {
            "endpoints": {"players": {"status": "ok"}, "store": {"status": "ok"}},
            "reviews": {
                "english": {"status": "ok"},
                "schinese": {"status": "ok"},
            },
        })
        self.write_json(root / "config/games_clustering.json", {
            "collection_config": {
                "languages": ["english", "schinese"],
                "reviews_per_language": 4,
            },
            "games": [{"appid": 10, "name": "Test Game"}],
        })
        freeze = root / "data/freeze_20260131/freeze_manifest.json"
        self.write_json(freeze, {
            "collection_config": "config/games_clustering.json",
            "selected_runs": [{
                "appid": 10,
                "name": "Test Game",
                "path": "data/steam/10/run1",
            }],
        })
        quality = root / "data/quality_table.csv"
        quality.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "appid", "language", "raw_review_count", "empty_text_count",
            "short_text_count", "duplicate_text_count", "target_count",
            "target_shortfall", "target_met", "endpoint_status",
        ]
        with quality.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "appid": 10, "language": "english", "raw_review_count": 4,
                "empty_text_count": 1, "short_text_count": 1,
                "duplicate_text_count": 1, "target_count": 4,
                "target_shortfall": 0, "target_met": True, "endpoint_status": "ok",
            })
            writer.writerow({
                "appid": 10, "language": "schinese", "raw_review_count": 3,
                "empty_text_count": 0, "short_text_count": 3,
                "duplicate_text_count": 1, "target_count": 4,
                "target_shortfall": 1, "target_met": False, "endpoint_status": "ok",
            })
        return freeze, quality

    def make_args(self, freeze, quality, output):
        return argparse.Namespace(
            freeze_manifest=freeze,
            quality_table=quality,
            output_dir=output,
            short_text=10,
            min_english_words=3,
            min_chinese_chars=4,
            min_language_comparison=1,
            max_playtime_minutes=clean_reviews.DEFAULT_MAX_PLAYTIME_MINUTES,
            overwrite=False,
        )

    def test_builds_three_analysis_datasets_and_reconciles_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze, quality = self.make_fixture(root)
            output = root / "data/cleaned/test"
            result = clean_reviews.run(self.make_args(freeze, quality, output), root=root)

            self.assertEqual(result["clean_master"], 7)
            self.assertEqual(result["association_data"], 6)
            self.assertEqual(result["topic_fit"], 2)
            self.assertEqual(result["empty_text"], 1)
            self.assertEqual(result["duplicate_text_extra"], 2)
            self.assertEqual(result["language_comparison_eligible_games"], 1)

            summary = json.loads(
                (output / "cleaning_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["counts"]["low_information"], 2)
            topic_rows = [
                json.loads(line)
                for line in (output / "topic_fit.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual({row["recommendationid"] for row in topic_rows}, {"e1", "z1"})
            self.assertTrue((output / "cleaning_manifest.json").exists())

    def test_quality_mismatch_stops_cleaning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze, quality = self.make_fixture(root)
            with quality.open(encoding="utf-8-sig") as source:
                rows = list(csv.DictReader(source))
            rows[0]["empty_text_count"] = "0"
            with quality.open("w", encoding="utf-8-sig", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "reconciliation failed"):
                clean_reviews.run(
                    self.make_args(freeze, quality, root / "data/cleaned/test"), root=root
                )


if __name__ == "__main__":
    unittest.main()

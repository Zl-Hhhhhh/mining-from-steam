import argparse
import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "verify_data", Path(__file__).parents[1] / "scripts/verify_data.py"
)
verify_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_data)


class VerifyDataTests(unittest.TestCase):
    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_dataset(self, path, row):
        path.mkdir(parents=True, exist_ok=True)
        (path / "reviews.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        with (path / "reviews.csv").open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)

    def make_fixture(self, root):
        started = "2026-01-01T00:00:00+00:00"
        finished = "2026-01-31T23:59:59+00:00"
        created = int(datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp())
        row = {
            "appid": 10,
            "recommendationid": "r1",
            "steamid": "s1",
            "language": "english",
            "review": "A sufficiently long review.",
            "voted_up": True,
            "timestamp_created": created,
            "playtime_forever": 120,
            "playtime_at_review": 100,
        }
        run = root / "data/steam/10/run1"
        self.write_dataset(run, row)
        self.write_dataset(run / "english", row)
        self.write_json(run / "store.json", {
            "data": {"10": {"success": True, "data": {"name": "Test Game"}}}
        })
        self.write_json(run / "manifest.json", {
            "appid": 10,
            "directory": "data/steam/10/run1",
            "languages": ["english"],
            "requested_per_language": 1,
            "sampling_window": {"start_utc": started, "end_utc": finished},
            "endpoints": {"players": {"status": "ok"}, "news": {"status": "ok"},
                          "store": {"status": "ok"}},
            "reviews": {"english": {"status": "ok", "requested": 1, "count": 1}},
            "review_count": 1,
        })
        self.write_json(root / "config/games_clustering.json", {
            "collection_config": {"languages": ["english"], "reviews_per_language": 1},
            "games": [{"appid": 10, "name": "Test Game", "verified_store_name": "Test Game"}],
        })
        freeze = root / "data/freeze_20260131/freeze_manifest.json"
        self.write_json(freeze, {
            "sampling_window": {"start_utc": started, "end_utc": finished},
            "collection_config": "config/games_clustering.json",
            "game_count": 1,
            "total_review_count": 1,
            "selected_runs": [{"appid": 10, "name": "Test Game", "run": "run1",
                               "path": "data/steam/10/run1", "review_count": 1}],
        })
        return freeze, run, row

    def make_args(self, freeze, root):
        return argparse.Namespace(
            freeze_manifest=freeze,
            output_json=root / "audit.json",
            output_csv=root / "audit.csv",
            short_text=10,
            max_playtime_minutes=verify_data.DEFAULT_MAX_PLAYTIME_MINUTES,
        )

    def test_clean_batch_passes_and_quality_error_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze, run, row = self.make_fixture(root)
            args = self.make_args(freeze, root)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(verify_data.run(args, root=root), 0)
            report = json.loads(args.output_json.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertTrue(args.output_csv.exists())

            row["review"] = ""
            self.write_dataset(run, row)
            self.write_dataset(run / "english", row)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(verify_data.run(args, root=root), 1)
            report = json.loads(args.output_json.read_text(encoding="utf-8"))
            self.assertIn("empty_review_text", {issue["code"] for issue in report["issues"]})


if __name__ == "__main__":
    unittest.main()

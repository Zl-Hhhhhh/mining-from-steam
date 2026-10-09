import gzip
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

SPEC = importlib.util.spec_from_file_location("steam_data", Path(__file__).parents[1] / "scripts/steam_data.py")
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def test_shared_appid_does_not_relabel_csgo_review_as_cs2(self):
        old = {"recommendationid": "123", "timestamp_created": collector.CS2_START - 1,
               "timestamp_updated": collector.CS2_START + 100, "author": {"steamid": "76561198000000001"}}
        self.assertEqual(collector.normalize(old, 730, "official")["game_era"], "csgo")
        old["timestamp_created"] = collector.CS2_START
        self.assertEqual(collector.normalize(old, 730, "official")["game_era"], "cs2")

    def test_dataset_accepts_both_formats_without_executing_input(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "sample.gz"
            with gzip.open(filename, "wt", encoding="utf-8") as output:
                output.write('{"recommend": true}\n')
                output.write("{'recommend': False, 'review': '中文'}\n")
            self.assertEqual(list(collector.dataset_records(filename)),
                             [{"recommend": True}, {"recommend": False, "review": "中文"}])
            with gzip.open(filename, "wt", encoding="utf-8") as output:
                output.write("__import__('os').system('echo unexpected')\n")
            with self.assertRaises((ValueError, SyntaxError)):
                list(collector.dataset_records(filename))

    def test_rate_limit_retries_and_honors_retry_after(self):
        error = HTTPError("https://example.com", 429, "rate limit", {"Retry-After": "3"}, None)
        with patch.object(collector, "urlopen", side_effect=[error, "response"]) as request, patch.object(
            collector.time, "sleep"
        ) as sleep:
            self.assertEqual(collector.open_url("https://example.com"), "response")
            self.assertEqual(request.call_count, 2)
            sleep.assert_called_once_with(3)

    def test_permission_error_is_not_retried(self):
        error = HTTPError("https://example.com", 403, "forbidden", {}, None)
        with patch.object(collector, "urlopen", side_effect=error) as request:
            with self.assertRaises(HTTPError):
                collector.open_url("https://example.com")
            self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()

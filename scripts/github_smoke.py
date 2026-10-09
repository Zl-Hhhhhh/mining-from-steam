"""Run one bounded page through the installed upstream steamreviews downloader."""
import argparse
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from unittest.mock import patch

import requests
import steamreviews.download_reviews as upstream

from steam_data import ROOT, CONFIG, CS2_START, export_reviews, normalize, save_json


class TimeoutSession(requests.Session):
    def get(self, url, **kwargs):
        kwargs.setdefault("timeout", 45)
        return super().get(url, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appid", type=int, default=730)
    parser.add_argument("--language", default="english")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / "data/github" / str(args.appid) / stamp
    report = {"fetched_at": datetime.now(timezone.utc).isoformat(), "appid": args.appid,
              "language": args.language, "version": version("steamreviews"),
              "repository": CONFIG["github"]["repository"], "commit": CONFIG["github"]["commit"],
              "endpoint": upstream.get_steam_api_url(), "max_queries": 1,
              "directory": str(directory.relative_to(ROOT)), "status": "error"}
    limits = {**upstream.get_steam_api_rate_limits(), "max_num_queries": 1}
    try:
        with TimeoutSession() as session, patch.object(upstream, "requests", session), patch.object(
            upstream, "get_steam_api_rate_limits", return_value=limits
        ):
            success, reviews, summary, queries, cursor = upstream.download_reviews_for_app_id_with_offset(
                args.appid, 0, chosen_request_params={"language": args.language, "filter": "recent",
                                                    "purchase_type": "all", "filter_offtopic_activity": 0}
            )
        if not success:
            raise ValueError("Upstream downloader returned success=False")
        save_json(directory / "raw.json", {"reviews": reviews, "query_summary": summary, "cursor": cursor})
        rows = [normalize(r, args.appid, "github/steamreviews") for r in reviews
                if args.appid != 730 or r.get("timestamp_created", 0) >= CS2_START]
        export_reviews(directory, rows)
        report.update(status="ok", downloaded=len(reviews), retained=len(rows), queries=queries)
    except Exception as error:
        report["error"] = str(error)
    save_json(directory / "manifest.json", report)
    save_json(ROOT / "data/github/latest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

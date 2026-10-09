"""Small, reproducible Steam collectors; Python 3.11+, no core dependencies."""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/sources.json").read_text(encoding="utf-8"))
CS2_START = int(datetime.fromisoformat(CONFIG["cs2_start_utc"].replace("Z", "+00:00")).timestamp())
API = "https://api.steampowered.com"


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def open_url(url, retries=3):
    for attempt in range(retries):
        try:
            return urlopen(Request(url, headers={"User-Agent": "mining-from-steam/course-research"}), timeout=45)
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            retry_after = error.headers.get("Retry-After", "")
            time.sleep(min(60, int(retry_after)) if retry_after.isdigit() else 5 * (2 ** attempt))
        except (URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def get_json(endpoint, params):
    with open_url(endpoint + "?" + urlencode(params)) as response:
        return json.load(response)


def fetch_page(appid, language, cursor, backend, end_time):
    if backend == "official":
        params = {"appid": appid, "languages": [language], "filter": 1,
                  "review_type": 0, "purchase_type": 1, "num_per_page": 100,
                  "filter_offtopic_activity": False, "cursor": cursor}
        if appid == 730:
            params.update(date_range_start=CS2_START, date_range_end=end_time)
        result = get_json(API + "/IUserReviewsService/GetAppReviews/v1/",
                          {"input_json": json.dumps(params, separators=(",", ":"))})
        result = result.get("response", result)
    else:
        result = get_json(f"https://store.steampowered.com/appreviews/{appid}",
                          {"json": 1, "language": language, "filter": "recent",
                           "review_type": "all", "purchase_type": "all",
                           "num_per_page": 100, "filter_offtopic_activity": 0, "cursor": cursor})
        if result.get("success") != 1:
            raise ValueError("Legacy review endpoint returned unsuccessful response")
    if not isinstance(result.get("reviews"), list):
        raise ValueError("Review response has no reviews list")
    return result


def export_reviews(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "reviews.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = ["appid", "recommendationid", "steamid", "language", "review", "voted_up",
              "timestamp_created", "timestamp_updated", "created_utc", "votes_up",
              "votes_funny", "playtime_forever", "playtime_at_review", "game_era", "source"]
    with (directory / "reviews.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize(review, appid, source):
    author = review.get("author", {})
    created = review.get("timestamp_created")
    return {**review, "appid": appid, "source": source,
            "recommendationid": str(review.get("recommendationid", "")),
            "steamid": str(author.get("steamid", "")),
            "playtime_forever": author.get("playtime_forever"),
            "playtime_at_review": author.get("playtime_at_review"),
            "created_utc": datetime.fromtimestamp(created, timezone.utc).isoformat() if created else None,
            "game_era": ("cs2" if created and created >= CS2_START else "csgo") if appid == 730 else "other"}


def collect(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / "data" / "steam" / str(args.appid) / stamp
    report = {"started_at": now(), "appid": args.appid, "directory": str(directory.relative_to(ROOT)),
              "requested_per_language": args.limit, "languages": args.languages,
              "sampling": "latest reviews, chronological; not a random sample",
              "cs2_boundary_utc": CONFIG["cs2_start_utc"], "endpoints": {}, "reviews": {}}
    endpoints = {
        "players": (API + "/ISteamUserStats/GetNumberOfCurrentPlayers/v1/", {"appid": args.appid}),
        "news": (API + "/ISteamNews/GetNewsForApp/v2/", {"appid": args.appid, "count": 20, "maxlength": 0}),
        "store": ("https://store.steampowered.com/api/appdetails", {"appids": args.appid, "cc": "cn", "l": "english"}),
    }
    for name, (url, params) in endpoints.items():
        try:
            result = get_json(url, params)
            if name == "store" and not result.get(str(args.appid), {}).get("success"):
                raise ValueError("Store appdetails returned unsuccessful response")
            if name == "players" and result.get("response", {}).get("result") != 1:
                raise ValueError("Player count endpoint returned unsuccessful response")
            save_json(directory / f"{name}.json", {"fetched_at": now(), "endpoint": url, "data": result})
            report["endpoints"][name] = {"status": "ok"}
        except Exception as error:
            report["endpoints"][name] = {"status": "error", "error": str(error)}
    all_rows = []
    end_time = int(time.time())
    for language in args.languages:
        backend = args.backend
        cursor, seen, rows = "*", set(), []
        state = {"status": "ok", "backend": backend, "pages": 0, "requested": args.limit}
        try:
            for page_number in range(args.max_pages):
                try:
                    result = fetch_page(args.appid, language, cursor, backend, end_time)
                except HTTPError as error:
                    if backend != "official" or error.code not in (404, 410):
                        raise
                    backend = "legacy"
                    state.update(backend=backend, fallback_reason=f"official HTTP {error.code}")
                    result = fetch_page(args.appid, language, cursor, backend, end_time)
                save_json(directory / language / "pages" / f"{page_number:04d}.json", result)
                state["pages"] += 1
                if page_number == 0:
                    state["query_summary"] = result.get("query_summary", {})
                    state["total_matching"] = result.get("total_matching")
                page_rows = result["reviews"]
                crossed_boundary = False
                for review in page_rows:
                    row = normalize(review, args.appid, backend)
                    if args.appid == 730 and row["game_era"] != "cs2":
                        crossed_boundary = True
                        continue
                    if row["recommendationid"] in seen:
                        continue
                    seen.add(row["recommendationid"])
                    rows.append(row)
                    if len(rows) >= args.limit:
                        break
                next_cursor = result.get("cursor")
                if len(rows) >= args.limit or not page_rows or crossed_boundary or not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor
                time.sleep(args.delay)
        except Exception as error:
            state.update(status="partial" if rows else "error", error=str(error))
        state.update(count=len(rows), positive=sum(bool(row.get("voted_up")) for row in rows))
        export_reviews(directory / language, rows)
        report["reviews"][language] = state
        all_rows.extend(rows)
        time.sleep(args.delay)
    export_reviews(directory, all_rows)
    report.update(finished_at=now(), review_count=len(all_rows))
    save_json(directory / "manifest.json", report)
    save_json(ROOT / "data/steam/latest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if any(s["status"] != "ok" for s in report["reviews"].values()) or any(
        s["status"] != "ok" for s in report["endpoints"].values()) else 0


def dataset_records(filename):
    with gzip.open(filename, "rt", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)  # These files also contain Python dict literals.


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    with open_url(url) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    # Validate gzip fully before accepting a download as complete.
    with gzip.open(temporary, "rb") as source:
        while source.read(1024 * 1024):
            pass
    temporary.replace(target)


def research(args):
    base = ROOT / "data/research"
    report = {"started_at": now(), "appid": args.appid, "catalog": CONFIG["research"]["catalog"],
              "era": "historical appid 730 / CS:GO; not contemporary CS2" if args.appid == 730 else "historical",
              "sources": {}, "citations": CONFIG["research"]["citations"]}
    for kind in args.files:
        filename = CONFIG["research"]["files"][kind]
        target = base / "raw" / filename
        url = CONFIG["research"]["base_url"] + filename
        state = {"url": url, "status": "ok"}
        extracted = []
        print(f"Processing research archive: {filename}", flush=True)
        try:
            if not target.exists() or args.redownload:
                download(url, target)
            digest = hashlib.sha256()
            with target.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            state.update(bytes=target.stat().st_size, sha256=digest.hexdigest())
            scanned = 0
            for record in dataset_records(target):
                scanned += 1
                if kind == "reviews":
                    for review in record.get("reviews", []):
                        if str(review.get("item_id")) == str(args.appid):
                            extracted.append({"appid": args.appid, "user_id": record.get("user_id"),
                                              "source": filename, "game_era": "historical_csgo" if args.appid == 730 else "historical", **review})
                elif kind == "items":
                    for item in record.get("items", []):
                        if str(item.get("item_id")) == str(args.appid):
                            extracted.append({"appid": args.appid, "user_id": record.get("user_id"),
                                              "steamid": str(record.get("steam_id", "")), "source": filename, **item})
                elif str(record.get("id")) == str(args.appid):
                    extracted.append(record)
            state.update(scanned_records=scanned, extracted_records=len(extracted))
            out = base / str(args.appid) / f"{kind}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as output:
                for row in extracted:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as error:
            state.update(status="error", error=str(error))
        report["sources"][kind] = state
        save_json(base / str(args.appid) / "manifest.json", report)
        print(f"{kind}: {state.get('status')}; matches={state.get('extracted_records', 0)}", flush=True)
    report["finished_at"] = now()
    save_json(base / str(args.appid) / "manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(any(s["status"] != "ok" for s in report["sources"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("collect", help="Collect live app snapshots and a bounded review sample")
    live.add_argument("--appid", type=int, default=730)
    live.add_argument("--languages", nargs="+", default=["english", "schinese"])
    live.add_argument("--limit", type=int, default=300, help="Review limit per language")
    live.add_argument("--max-pages", type=int, default=50)
    live.add_argument("--delay", type=float, default=3)
    live.add_argument("--backend", choices=["official", "legacy"], default="official")
    live.set_defaults(action=collect)
    historical = commands.add_parser("research", help="Download UCSD archives and extract one app")
    historical.add_argument("--appid", type=int, default=730)
    historical.add_argument("--files", nargs="+", choices=list(CONFIG["research"]["files"]), default=["reviews", "items", "metadata"])
    historical.add_argument("--redownload", action="store_true")
    historical.set_defaults(action=research)
    args = parser.parse_args()
    if getattr(args, "limit", 1) <= 0 or getattr(args, "max_pages", 1) <= 0 or getattr(args, "delay", 0) < 0:
        parser.error("limit/max-pages must be positive; delay must be nonnegative")
    return args.action(args)


if __name__ == "__main__":
    sys.exit(main())

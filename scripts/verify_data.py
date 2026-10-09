"""Validate saved CS2 samples and UCSD archives without network access."""
import csv
import hashlib
import json
import re
from collections import Counter

from steam_data import CONFIG, CS2_START, ROOT, now, save_json


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_reviews(directory, expected):
    rows = lines(directory / "reviews.jsonl")
    ids = [row["recommendationid"] for row in rows]
    check(len(rows) == expected, f"{directory}: manifest count mismatch")
    check(len(ids) == len(set(ids)), f"{directory}: duplicate recommendation IDs")
    check(all(row["appid"] == 730 and row["timestamp_created"] >= CS2_START
              and row["game_era"] == "cs2" for row in rows), f"{directory}: invalid CS2 boundary/appid")
    check(all(isinstance(row["steamid"], str) for row in rows), "SteamID must be a string")
    with (directory / "reviews.csv").open(encoding="utf-8-sig", newline="") as source:
        csv_rows = list(csv.DictReader(source))
    check([row["recommendationid"] for row in csv_rows] == ids, "CSV/JSONL review IDs differ")
    return rows


def main():
    live = load(ROOT / "data/steam/latest.json")
    directory = ROOT / live["directory"]
    rows = validate_reviews(directory, live["review_count"])
    check(all(s["status"] == "ok" for s in live["endpoints"].values()), "Live endpoint failure")
    result = {"verified_at": now(), "status": "ok", "official_reviews": len(rows), "languages": {}}
    for language, state in live["reviews"].items():
        check(state["status"] == "ok", f"{language}: collection failure")
        subset = validate_reviews(directory / language, state["count"])
        check(all(r["language"] == language for r in subset), "Unexpected review language")
        check(sum(bool(r["voted_up"]) for r in subset) == state["positive"], "Sentiment count mismatch")
        check(len(list((directory / language / "pages").glob("*.json"))) == state["pages"], "Raw page count mismatch")
        result["languages"][language] = {"count": len(subset), "positive": state["positive"],
                                          "positive_percent": round(state["positive"] / len(subset) * 100, 2) if subset else None,
                                          "earliest_utc": min((r["created_utc"] for r in subset), default=None),
                                          "latest_utc": max((r["created_utc"] for r in subset), default=None)}
    players = load(directory / "players.json")
    news = load(directory / "news.json")
    store = load(directory / "store.json")
    result.update(player_count=players["data"]["response"]["player_count"], player_count_fetched_at=players["fetched_at"],
                  news_count=len(news["data"]["appnews"]["newsitems"]),
                  game_name=store["data"]["730"]["data"]["name"])
    github = load(ROOT / "data/github/latest.json")
    check(github["status"] == "ok", "GitHub smoke check failed")
    github_rows = validate_reviews(ROOT / github["directory"], github["retained"])
    overlap = set(r["recommendationid"] for r in rows) & set(r["recommendationid"] for r in github_rows)
    result.update(github_reviews=len(github_rows), overlap_with_official=len(overlap),
                  unique_live_reviews=len(rows) + len(github_rows) - len(overlap))
    historical = load(ROOT / "data/research/730/manifest.json")
    result["research"] = {}
    for kind, state in historical["sources"].items():
        check(state["status"] == "ok", f"Research source failed: {kind}")
        raw = ROOT / "data/research/raw" / CONFIG["research"]["files"][kind]
        digest = hashlib.sha256()
        with raw.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        check(digest.hexdigest() == state["sha256"], f"Archive SHA-256 mismatch: {kind}")
        subset = lines(ROOT / "data/research/730" / f"{kind}.jsonl")
        check(len(subset) == state["extracted_records"], f"Research extraction count mismatch: {kind}")
        check(all(str(r.get("item_id", r.get("id"))) == "730" for r in subset), f"Wrong research appid: {kind}")
        result["research"][kind] = {"records": len(subset)}
        if kind in ("reviews", "items"):
            result["research"][kind]["unique_users"] = len(set(r["user_id"] for r in subset))
        if kind == "reviews":
            check(all(r["game_era"] == "historical_csgo" for r in subset), "Historical reviews mislabeled as CS2")
            years = [re.search(r"\b(19\d{2}|20\d{2})\b", str(r.get("posted", ""))) for r in subset]
            result["research"][kind]["posted_years"] = dict(Counter(
                match.group(1) if match else "year_missing" for match in years
            ))
    save_json(ROOT / "data/verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

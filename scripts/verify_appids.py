"""Verify appids from games_clustering.json by querying Steam store API."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
GAMES_CONFIG = ROOT / "config/games_clustering.json"


def get_game_info(appid):
    """Fetch game info from Steam store API."""
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=english"
    try:
        request = Request(url, headers={"User-Agent": "mining-from-steam/appid-verification"})
        with urlopen(request, timeout=10) as response:
            data = json.load(response)
            if str(appid) in data and data[str(appid)].get("success"):
                return data[str(appid)]["data"]
            return None
    except (HTTPError, URLError, TimeoutError) as e:
        print(f"  Error fetching {appid}: {e}")
        return None


def verify_games():
    """Verify all appids in games_clustering.json."""
    config = json.loads(GAMES_CONFIG.read_text(encoding="utf-8"))
    games = config["games"]
    
    print(f"Verifying {len(games)} games...\n")
    
    invalid = []
    valid = []
    
    for i, game in enumerate(games, 1):
        appid = game["appid"]
        expected_name = game["name"]
        
        print(f"[{i}/{len(games)}] Checking {appid} - {expected_name}...")
        
        info = get_game_info(appid)
        
        if info is None:
            print(f"  ❌ INVALID: Could not fetch data for appid {appid}")
            invalid.append({"appid": appid, "expected": expected_name, "reason": "API returned no data"})
        else:
            actual_name = info.get("name", "")
            if actual_name.lower() == expected_name.lower() or expected_name in actual_name or actual_name in expected_name:
                print(f"  ✅ VALID: {actual_name}")
                valid.append({"appid": appid, "name": actual_name})
            else:
                print(f"  ⚠️  NAME MISMATCH:")
                print(f"     Expected: {expected_name}")
                print(f"     Actual:   {actual_name}")
                invalid.append({"appid": appid, "expected": expected_name, "actual": actual_name, "reason": "Name mismatch"})
        
        # Rate limiting
        time.sleep(1.5)
    
    print(f"\n{'='*60}")
    print(f"Verification complete:")
    print(f"  ✅ Valid: {len(valid)}")
    print(f"  ❌ Invalid: {len(invalid)}")
    
    if invalid:
        print(f"\n{'='*60}")
        print("Invalid entries:")
        for item in invalid:
            print(f"  appid {item['appid']}: {item['expected']}")
            print(f"    Reason: {item['reason']}")
            if "actual" in item:
                print(f"    Actual name: {item['actual']}")
    
    # Save results
    output = ROOT / "data/appid_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"valid": valid, "invalid": invalid}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {output}")
    
    return len(invalid) == 0


if __name__ == "__main__":
    sys.exit(0 if verify_games() else 1)

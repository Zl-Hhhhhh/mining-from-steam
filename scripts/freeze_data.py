"""Freeze one completed Steam collection window and remove older snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEAM_ROOT = ROOT / "data" / "steam"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def target_end(value: str) -> str:
    day = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return day.replace(hour=23, minute=59, second=59).isoformat()


def select_runs(end_utc: str):
    config = load_json(ROOT / "config" / "games_clustering.json")
    selected = []
    missing = []
    for game in config["games"]:
        appid = str(game["appid"])
        app_root = STEAM_ROOT / appid
        candidates = []
        if app_root.exists():
            for run in app_root.iterdir():
                manifest_path = run / "manifest.json"
                if not run.is_dir() or not manifest_path.exists():
                    continue
                try:
                    manifest = load_json(manifest_path)
                    endpoint_ok = all(
                        item.get("status") == "ok"
                        for item in manifest.get("endpoints", {}).values()
                    )
                    review_ok = all(
                        item.get("status") == "ok"
                        for item in manifest.get("reviews", {}).values()
                    )
                    if manifest.get("sampling_window", {}).get("end_utc") == end_utc and endpoint_ok and review_ok:
                        candidates.append((run.name, run, manifest))
                except (OSError, json.JSONDecodeError, TypeError):
                    continue
        if not candidates:
            missing.append({"appid": int(appid), "name": game["name"]})
            continue
        run_name, run_path, manifest = sorted(candidates, reverse=True)[0]
        selected.append({
            "appid": int(appid),
            "name": game["name"],
            "run": run_name,
            "path": run_path,
            "manifest": manifest,
        })
    return config, selected, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-date", required=True, help="Target UTC date in YYYY-MM-DD format")
    parser.add_argument("--apply", action="store_true", help="Delete non-selected snapshots")
    args = parser.parse_args()

    end_utc = target_end(args.end_date)
    config, selected, missing = select_runs(end_utc)
    selected_paths = {item["path"].resolve() for item in selected}
    configured_appids = {str(game["appid"]) for game in config["games"]}

    removal_paths = []
    if STEAM_ROOT.exists():
        for app_root in STEAM_ROOT.iterdir():
            if app_root.is_dir():
                if app_root.name not in configured_appids:
                    removal_paths.append(app_root)
                    continue
                for run in app_root.iterdir():
                    if run.is_dir() and run.resolve() not in selected_paths:
                        removal_paths.append(run)
            elif app_root.is_file() and app_root.name == "latest.json":
                removal_paths.append(app_root)

    if missing:
        print(json.dumps({"status": "blocked", "missing": missing}, ensure_ascii=False, indent=2))
        return 2
    if len(selected) != len(config["games"]):
        raise RuntimeError("Selected run count does not match configured game count")

    freeze_root = ROOT / "data" / f"freeze_{args.end_date.replace('-', '')}"
    manifest_path = freeze_root / "freeze_manifest.json"
    sums_path = freeze_root / "SHA256SUMS.txt"
    artifact_hashes_path = freeze_root / "freeze_artifact_hashes.json"

    print(json.dumps({
        "end_utc": end_utc,
        "selected_runs": len(selected),
        "removal_paths": len(removal_paths),
        "apply": args.apply,
        "freeze_root": str(freeze_root.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    if not args.apply:
        print("Dry run only. Re-run with --apply to delete non-selected snapshots.")
        return 0

    freeze_root.mkdir(parents=True, exist_ok=True)
    for path in removal_paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    retained_files = sorted(
        path for path in STEAM_ROOT.rglob("*") if path.is_file()
    )
    file_records = []
    sums_lines = []
    for path in retained_files:
        relative = path.relative_to(ROOT).as_posix()
        digest = sha256(path)
        file_records.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
        sums_lines.append(f"{digest}  {relative}")

    frozen_at = datetime.now(timezone.utc).isoformat()
    provenance_paths = [
        ROOT / "config" / "games_clustering.json",
        ROOT / "config" / "sources.json",
        ROOT / "scripts" / "collect_batch.py",
        ROOT / "scripts" / "steam_data.py",
        ROOT / "scripts" / "freeze_data.py",
    ]
    freeze_manifest = {
        "freeze_id": args.end_date,
        "frozen_at_utc": frozen_at,
        "sampling_window": {
            "start_utc": selected[0]["manifest"]["sampling_window"]["start_utc"],
            "end_utc": end_utc,
            "days": 183,
        },
        "collection_config": "config/games_clustering.json",
        "provenance": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in provenance_paths
        ],
        "game_count": len(selected),
        "total_review_count": sum(item["manifest"].get("review_count", 0) for item in selected),
        "selected_runs": [
            {
                "appid": item["appid"],
                "name": item["name"],
                "run": item["run"],
                "path": item["path"].relative_to(ROOT).as_posix(),
                "review_count": item["manifest"].get("review_count", 0),
            }
            for item in selected
        ],
        "removed_path_count": len(removal_paths),
        "removed_paths": [path.relative_to(ROOT).as_posix() for path in removal_paths],
        "files": file_records,
    }
    manifest_path.write_text(json.dumps(freeze_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    sums_path.write_text("\n".join(sums_lines) + "\n", encoding="utf-8")
    artifact_hashes = {
        "freeze_id": args.end_date,
        "freeze_manifest_sha256": sha256(manifest_path),
        "sha256s_file_sha256": sha256(sums_path),
    }
    artifact_hashes_path.write_text(json.dumps(artifact_hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for stale in (ROOT / "data" / "batch_progress.json", ROOT / "data" / "batch_summary.json"):
        if stale.exists():
            stale.unlink()
    print(json.dumps({
        "status": "frozen",
        "manifest": str(manifest_path.relative_to(ROOT)),
        "sha256s": str(sums_path.relative_to(ROOT)),
        "artifact_hashes": str(artifact_hashes_path.relative_to(ROOT)),
        "retained_files": len(file_records),
        "removed_paths": len(removal_paths),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

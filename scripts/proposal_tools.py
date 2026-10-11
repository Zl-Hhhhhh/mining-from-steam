"""Prepare the NeurIPS proposal, audit its game list, compile and render it."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "proposal"
OUTPUT = ROOT / "output/pdf"
TEMP = ROOT / "tmp/pdfs"
STYLE_URL = "https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip"


def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "mining-from-steam-proposal"}), timeout=45) as response:
        return response.read()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def names_match(expected, actual):
    expected = "".join(c for c in expected.casefold() if c.isalnum())
    actual = "".join(c for c in actual.casefold() if c.isalnum())
    return expected == actual or expected in actual or actual in expected


def setup():
    PROPOSAL.mkdir(exist_ok=True)
    kit = fetch(STYLE_URL)
    with zipfile.ZipFile(io.BytesIO(kit)) as archive:
        style_name = next(n for n in archive.namelist() if n.endswith("neurips_2026.sty"))
        style = archive.read(style_name)
        (PROPOSAL / "neurips_2026.sty").write_bytes(style)
    write_json(PROPOSAL / "template-source.json", {"source": STYLE_URL, "year": 2026,
                                                   "style_sha256": hashlib.sha256(style).hexdigest(),
                                                   "downloaded_at": datetime.now(timezone.utc).isoformat()})
    print("Official NeurIPS 2026 style downloaded", flush=True)


def audit():
    original = ROOT / "config/games_clustering.json"
    config = json.loads(original.read_text(encoding="utf-8"))
    games = [dict(g) for g in config["games"] if g["appid"] != 431960]
    corrections = []
    for game in games:
        if game["name"] == "Dead Cells" and game["appid"] == 381210:
            game["appid"] = 588650
            corrections.append("Dead Cells: appid corrected from 381210 to 588650")
        elif game["name"] == "Dead by Daylight" and game["appid"] == 588650:
            game["appid"] = 381210
            corrections.append("Dead by Daylight: appid corrected from 588650 to 381210")
        elif game["name"] == "NARAKA: BLADEPOINT" and game["appid"] == 1088850:
            game["appid"] = 1203220
            corrections.append("NARAKA: BLADEPOINT: appid corrected from 1088850 to 1203220")
    assert len(games) == len(set(g["appid"] for g in games)) == 50
    for index, game in enumerate(games, 1):
        try:
            result = json.loads(fetch(f"https://store.steampowered.com/api/appdetails?appids={game['appid']}&l=english&cc=us"))
            entry = result[str(game["appid"])]
            if not entry.get("success"):
                raise ValueError("Store endpoint returned success=False")
            info = entry["data"]
            game.update(verified_store_name=info["name"], verified_type=info.get("type"), verification_status="ok")
            game["store_metadata"] = {key: info.get(key) for key in
                                      ("is_free", "genres", "categories", "release_date", "price_overview", "developers", "publishers")}
            game["name_match"] = names_match(game["name"], info["name"])
        except Exception as error:
            game.update(verification_status="error", verification_error=str(error))
        print(f"[{index:02d}/50] {game['appid']}: {game['name']} -> {game.get('verified_store_name', game.get('verification_error'))}", flush=True)
        time.sleep(0.5)
    write_json(PROPOSAL / "games_reviewed.json", {
        "reviewed_at": datetime.now(timezone.utc).isoformat(), "original_config": "config/games_clustering.json",
        "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(), "original_entries": len(config["games"]),
        "excluded": [{"appid": 431960, "name": "Wallpaper Engine", "reason": "Software utility; excluded from the 50-game study"}],
        "corrections": corrections, "games_count": len(games), "collection_config": config["collection_config"], "games": games})
    print("Saved reviewed 50-game manifest; original config unchanged", flush=True)
    print("VERIFICATION_ERRORS", sum(g["verification_status"] != "ok" for g in games))
    print("NAME_MISMATCHES", [(g["name"], g.get("verified_store_name")) for g in games if not g.get("name_match", False)])


def refresh():
    """Refresh the one newly discovered wrong appid without re-querying all games."""
    report = json.loads((PROPOSAL / "games_reviewed.json").read_text(encoding="utf-8"))
    for game in report["games"]:
        if game["name"] == "NARAKA: BLADEPOINT" and game["appid"] == 1088850:
            game["appid"] = 1203220
            entry = json.loads(fetch("https://store.steampowered.com/api/appdetails?appids=1203220&l=english&cc=us"))["1203220"]
            assert entry["success"], "NARAKA appid verification failed"
            info = entry["data"]
            game.update(verified_store_name=info["name"], verified_type=info.get("type"), verification_status="ok")
            game["store_metadata"] = {key: info.get(key) for key in
                                      ("is_free", "genres", "categories", "release_date", "price_overview", "developers", "publishers")}
            report["corrections"].append("NARAKA: BLADEPOINT: appid corrected from 1088850 to 1203220")
        game["name_match"] = names_match(game["name"], game.get("verified_store_name", ""))
    assert len(report["games"]) == len(set(g["appid"] for g in report["games"])) == 50
    assert all(g["verification_status"] == "ok" and g["verified_type"] == "game" and g["name_match"]
               for g in report["games"]), "Remaining store verification mismatch"
    report["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(PROPOSAL / "games_reviewed.json", report)
    print("All 50 game appids, normalized names, and game types verified")
    print("Corrections:", report["corrections"])


def compiler():
    found = shutil.which("pdflatex")
    if found:
        return found
    candidates = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/MiKTeX/miktex/bin/x64/pdflatex.exe",
                  Path("C:/Program Files/MiKTeX/miktex/bin/x64/pdflatex.exe")]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("pdflatex is not installed or cannot be located")


def build():
    TEMP.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    executable = compiler()
    source = source_file()
    command = [executable, "--enable-installer", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape",
               "-output-directory=" + str(TEMP), source.name]
    for _ in range(2):
        subprocess.run(command, cwd=PROPOSAL, check=True)
    destination = OUTPUT / source.with_suffix(".pdf").name
    shutil.copyfile(TEMP / source.with_suffix(".pdf").name, destination)
    print("PDF:", destination, flush=True)


def inspect():
    import pymupdf as fitz
    filename = OUTPUT / source_file().with_suffix(".pdf").name
    document = fitz.open(filename)
    print("PAGE_COUNT", len(document))
    assert 1 <= len(document) <= 2, "Proposal must fit in 1-2 pages, including references"
    text = []
    for index, page in enumerate(document):
        destination = TEMP / f"proposal-page-{index + 1}.png"
        page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).save(destination)
        text.append(page.get_text())
        print("RENDER", destination)
        print("PAGE", index + 1, page.get_text())
    (TEMP / "proposal-extracted.txt").write_text("\n".join(text), encoding="utf-8")
    for term in ("Background", "Methodology", "Expected Outcomes", "Timeline", "Student ID"):
        assert term in "\n".join(text), f"Missing required proposal element: {term}"


def source_file():
    if args.tex:
        source = (PROPOSAL / args.tex).resolve()
        if source.parent != PROPOSAL.resolve() or not source.is_file():
            raise ValueError("--tex must name a LaTeX file directly inside proposal/")
        return source
    sources = list(PROPOSAL.glob("Group*_Proposal.tex"))
    if len(sources) != 1:
        raise ValueError("Expected one Group*_Proposal.tex; use --tex to select the main file")
    return sources[0]


def lint():
    script = Path.home() / ".codex/skills/write/scripts/check-punctuation.sh"
    python = shlex.quote(sys.executable.replace("\\", "/"))
    shim_dir = TEMP / "punctuation-bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "python3").write_text(
        '#!/usr/bin/env bash\nscript_path="$(cygpath -w "$1")"\nshift\n'
        + 'exec ' + python + ' "$script_path" "$@"\n', encoding="utf-8")
    command = 'export PATH="$(cygpath -u ' + shlex.quote(shim_dir.as_posix()) + '):$PATH"; bash '
    command += shlex.quote(script.as_posix()) + " --lang en "
    command += shlex.quote((TEMP / "proposal-extracted.txt").as_posix())
    subprocess.run(["C:/Program Files/Git/bin/bash.exe", "-c", command], cwd=ROOT, check=True)


def bundle():
    source = source_file()
    directory = ROOT / "output/latex"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (source.stem + "_Source.zip")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in (source.name, "neurips_2026.sty", "README.md", "games_reviewed.json", "template-source.json"):
            archive.write(PROPOSAL / filename, arcname=filename)
    print("LaTeX source bundle:", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "audit", "refresh", "build", "inspect", "lint", "bundle"])
    parser.add_argument("--tex", help="Main filename inside proposal/; otherwise find the sole Group*_Proposal.tex")
    args = parser.parse_args()
    globals()[args.action]()

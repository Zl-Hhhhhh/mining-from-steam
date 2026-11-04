"""Shared input validation and provenance for review analyses."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def hash_status(path, expected):
    """Accept an original hash reconstructed by *only* changing line endings."""
    data = Path(path).read_bytes()
    lf = data.replace(b"\r\n", b"\n")
    for kind, candidate in (("exact", data), ("LF-equivalent", lf),
                            ("CRLF-equivalent", lf.replace(b"\n", b"\r\n"))):
        if hashlib.sha256(candidate).hexdigest() == expected:
            return kind
    raise ValueError(f"Content hash mismatch beyond uniform line endings: {path}")


def load_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    for column in ("recommendationid", "steamid"):
        frame[column] = frame[column].astype(str)
    if frame.recommendationid.duplicated().any():
        raise ValueError(f"Duplicate review IDs: {path}")
    if not frame.voted_up.map(lambda v: isinstance(v, bool)).all():
        raise ValueError("Recommendation values must be booleans")
    return frame


def runtime_versions():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in
        ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "statsmodels", "jieba", "joblib")}}


def load_inputs(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    freeze_path = ROOT / manifest["source_freeze_manifest"]
    provenance = [{"path": relative(manifest_path), "actual_sha256": sha256(manifest_path)}]
    for path, expected in ((freeze_path, manifest["source_freeze_manifest_sha256"]),
                           (ROOT / manifest["source_quality_table"], manifest["source_quality_table_sha256"])):
        provenance.append({"path": relative(path), "actual_sha256": sha256(path),
                           "recorded_sha256": expected, "match": hash_status(path, expected)})
    freeze = read_json(freeze_path)
    frozen_json = Counter()
    for record in freeze["files"]:
        path = ROOT / record["path"]
        if path.suffix in (".json", ".jsonl"):
            frozen_json[hash_status(path, record["sha256"])] += 1
    frames = {}
    for name in ("clean_master", "topic_fit", "association_data"):
        record = next(f for f in manifest["files"] if f["path"].endswith(f"/{name}.jsonl"))
        path = ROOT / record["path"]
        provenance.append({"path": relative(path), "actual_sha256": sha256(path),
                           "recorded_sha256": record["sha256"], "match": hash_status(path, record["sha256"])})
        frame = load_jsonl(path)
        if len(frame) != manifest["counts"][name]:
            raise ValueError(f"Row count mismatch: {name}")
        frames[name] = frame
    master, fit, association = (frames[k] for k in ("clean_master", "topic_fit", "association_data"))
    if not set(fit.recommendationid) <= set(association.recommendationid) <= set(master.recommendationid):
        raise ValueError("Clean datasets do not have the documented subset relation")
    if not fit.topic_fit_eligible.all() or not association.association_eligible.all():
        raise ValueError("Eligibility flags disagree with the input dataset")
    start = pd.Timestamp(freeze["sampling_window"]["start_utc"]).timestamp()
    end = pd.Timestamp(freeze["sampling_window"]["end_utc"]).timestamp()
    if not master.timestamp_created.between(start, end).all():
        raise ValueError("Review outside frozen date window")
    if master.appid.nunique() != freeze["game_count"]:
        raise ValueError("Game count mismatch")
    quality = read_json(ROOT / "data/verification.json")
    known_quality = {"empty_review_text", "short_review_text", "duplicate_review_text", "language_target_shortfall"}
    unknown_errors = [x for x in quality["issues"] if x["severity"] == "error" and x["code"] not in known_quality]
    if unknown_errors:
        raise ValueError(f"Unresolved structural audit failures: {unknown_errors[:3]}")
    return frames, {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sampling_window": freeze["sampling_window"], "counts": manifest["counts"],
        "sources": provenance, "frozen_json_hash_matches": dict(frozen_json),
        "prior_audit_issue_counts": dict(Counter(x["code"] for x in quality["issues"])),
        "csv_policy": "Model inputs are hash-verified JSONL. CSV bytes can differ after Git conversion of embedded newlines; CSVs are not used for model fitting.",
        "runtime": runtime_versions(),
    }


def normalize_weights(weights, components):
    """Remove arbitrary NMF component scale before interpreting row proportions."""
    scaled = weights * np.linalg.norm(components, axis=1)
    totals = scaled.sum(axis=1, keepdims=True)
    return np.divide(scaled, totals, out=np.zeros_like(scaled), where=totals > 0)

"""Prepare the 160-review blind audit for the relaxed Task12 topic input."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from clean_reviews import normalize_text_key
from embedding_bge_sentences import read_json, sha_file, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output/topic_review/task12-relaxed-blind-20261125"
TRIAL = ROOT / ".cache-topic/lexical-relaxed-20261125/topic-matching-probe"
TASK12 = ROOT / "output/topic_selection/task12-20261114"
PACKAGE_ID = "task12-relaxed-blind-20261125"
POLICY = "score_065_gap_006"
LANGUAGES = ("english", "schinese")
STRATA = ("accepted", "blocked_by_cluster", "other_candidates", "no_candidates")
SAMPLE_SEED = 261125
LONG_LENGTH = 250
MIN_LONG = 3
PER_CELL = 20


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def cluster_blocked(row: dict) -> bool:
    return (
        not row["global_top_allowed_in_cluster"]
        and row["global_top_score"] >= 0.65
        and row["global_gap"] >= 0.06
    )


def sample(destination: Path):
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")

    draft = TASK12 / "topic-list-draft.json"
    old_units = TASK12 / "handoff/sentence_units.jsonl"
    new_handoff = TASK12 / "handoff-lexical-relaxed-20261125"
    seed_path = ROOT / "output/topic_selection/task12-20261118-sentence-clustering-v1/sentence_k10_s42/topic-seed-units.json"
    baseline_path = ROOT / "output/topic_selection/topics-v1-exploratory-20261118/review_topic_weights.jsonl"
    master_path = ROOT / "data/cleaned/freeze_20261101/clean_master.jsonl"
    prior_path = ROOT / "docs/review/review_sample.json"
    template_path = Path(__file__).with_suffix(".html")

    master = {row["recommendationid"]: row for row in jsonl(master_path)}
    review_map = read_json(new_handoff / "review_to_units.json")
    if master.keys() != review_map.keys():
        raise ValueError("Review mapping and clean master cover different review IDs")
    matched = {row["unit_id"]: row for row in jsonl(TRIAL / "new_unit_matches.jsonl")}
    metrics = read_json(TRIAL / "summary.json")["score_and_gap"][POLICY]
    accepted = {key for key, row in matched.items() if POLICY in row["passed_policies"]}
    blocked = {key for key, row in matched.items() if cluster_blocked(row)}
    if (len(accepted), len(blocked)) != (metrics["assigned_units"], metrics["would_pass_except_cluster_route"]):
        raise ValueError("Unit decisions disagree with the topic matching trial")
    with (TRIAL / "audit_reviewed.csv").open(encoding="utf-8-sig", newline="") as handle:
        prior_audit = {row["unit_id"] for row in csv.DictReader(handle)}
    excluded_reviews = {rid for key in prior_audit for rid in matched[key]["review_ids"]}
    seed_ids = {row["unit_id"] for row in read_json(seed_path)["seed_units"]}
    excluded_reviews.update(rid for row in jsonl(old_units) if row["unit_id"] in seed_ids for rid in row["review_ids"])
    historical = read_json(prior_path)
    historical_cases = historical["cases"] + [example for group in historical["groups"] for example in group["examples"]]
    excluded_texts = {master[rid]["text_key"] for rid in excluded_reviews}
    excluded_texts.update(normalize_text_key(row["text"]) for row in historical_cases if row.get("text"))
    excluded_texts.discard("")

    pools = defaultdict(list)
    available = defaultdict(list)
    for rid, keys in review_map.items():
        row = master[rid]
        if accepted.intersection(keys):
            stratum = "accepted"
        elif blocked.intersection(keys):
            stratum = "blocked_by_cluster"
        else:
            stratum = "other_candidates" if keys else "no_candidates"
        cell = (row["language"], stratum)
        pools[cell].append(rid)
        if rid not in excluded_reviews and row["text_key"] not in excluded_texts:
            available[cell].append(rid)

    rng = random.Random(SAMPLE_SEED)
    selected = []
    selected_texts = set()
    sampling_counts = {}
    for language in LANGUAGES:
        sampling_counts[language] = {}
        for stratum in STRATA:
            cell = (language, stratum)
            eligible = sorted(available[cell])
            rng.shuffle(eligible)
            long = [rid for rid in eligible if len(master[rid]["review"].strip()) > LONG_LENGTH]
            cell_sample = []
            for candidates, target in ((long, MIN_LONG), (eligible, PER_CELL)):
                for rid in candidates:
                    if len(cell_sample) == target:
                        break
                    text_key = master[rid]["text_key"]
                    if text_key not in selected_texts:
                        selected_texts.add(text_key)
                        cell_sample.append(rid)
            if len(cell_sample) != PER_CELL:
                raise ValueError(f"Insufficient unique eligible reviews for {cell}")
            selected.extend((rid, stratum) for rid in cell_sample)
            sampling_counts[language][stratum] = {
                "population": len(pools[cell]), "eligible_after_exclusions": len(eligible), "sample": PER_CELL,
                "long_in_sample": sum(len(master[rid]["review"].strip()) > LONG_LENGTH for rid in cell_sample),
            }
    rng.shuffle(selected)

    baseline = {row["review_id"]: row["topic_presence"] for row in jsonl(baseline_path)}
    generic = {rid for row in jsonl(new_handoff / "auxiliary_units.jsonl") for rid in row["review_ids"]}
    cases = []
    key = {}
    for index, (rid, stratum) in enumerate(selected, 1):
        sid = f"R{index:03}"
        row = master[rid]
        cases.append({"id": sid, "text": row["review"]})
        key[sid] = {
            "review_id": rid, "appid": row["appid"], "source_language": row["language"], "stratum": stratum,
            "raw_character_count": len(row["review"].strip()),
            "raw_text_sha256": hashlib.sha256(row["review"].encode("utf-8")).hexdigest(),
            "baseline_topic_ids": baseline[rid], "has_a01_fragment": rid in generic,
            "candidate_unit_count": len(review_map[rid]),
            "new_units": [
                {
                    "unit_id": unit["unit_id"], "text": unit["text"], "v1_topic_ids": unit["topic_ids"],
                    "nearest_cluster": unit["cluster_id"], "top_topic": unit["global_top_topic"],
                    "score": unit["global_top_score"], "gap": unit["global_gap"],
                    "allowed_in_cluster": unit["global_top_allowed_in_cluster"],
                    "passes_policy": unit_id in accepted,
                }
                for unit_id in review_map[rid] if (unit := matched.get(unit_id)) is not None
            ],
        }

    sample_hash = hashlib.sha256(json.dumps(cases, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    topics = [{"id": row["topic_id"], "label": row["name_zh"], "definition": row["definition"]}
              for row in read_json(draft)["primary_topics"]]
    topics.append({"id": "A04", "label": "其他具体方面", "definition": "八个主题以外、能从原文指出的具体关注点。"})
    template = template_path.read_text(encoding="utf-8")
    for reviewer in ("A", "B"):
        order = list(cases)
        random.Random(f"{SAMPLE_SEED}:{reviewer}").shuffle(order)
        payload = {
            "package_id": PACKAGE_ID, "assignment": reviewer, "sample_sha256": sample_hash,
            "cases": order, "topics": topics,
        }
        embedded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
        path = destination / f"reviewer_{reviewer}" / f"blind_review_{reviewer}.html"
        path.parent.mkdir(parents=True)
        path.write_text(template.replace("__REVIEW_PAYLOAD__", embedded), encoding="utf-8")

    inputs = [
        master_path, prior_path, old_units, seed_path, baseline_path, draft, template_path, Path(__file__),
        TRIAL / "audit_reviewed.csv", TRIAL / "new_unit_matches.jsonl", TRIAL / "summary.json",
        new_handoff / "review_to_units.json", new_handoff / "auxiliary_units.jsonl",
    ]
    write_json(destination / "coordinator/sampling_key.json", {
        "package_id": PACKAGE_ID, "sample_sha256": sample_hash, "seed": SAMPLE_SEED,
        "matching_policy": {"minimum_score": 0.65, "minimum_gap": 0.06, "requires_cluster_agreement": True},
        "stratum_priority": list(STRATA), "long_review_threshold_exclusive": LONG_LENGTH,
        "minimum_long_per_cell_where_available": MIN_LONG,
        "sampling_counts": sampling_counts, "reviews_excluded_by_prior_audit_or_seed_id": len(excluded_reviews),
        "inputs": {path.relative_to(ROOT).as_posix(): sha_file(path) for path in inputs}, "cases": key,
    })
    print(json.dumps(sampling_counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sample(parser.parse_args().output)

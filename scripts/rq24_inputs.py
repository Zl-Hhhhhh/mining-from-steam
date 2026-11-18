"""Verified inputs and topic variants for the exploratory RQ2/RQ4 analysis."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re

import numpy as np
import pandas as pd

import analysis_common as common
import assign_topics_v1_cluster_derived as assignment


ROOT = common.ROOT
TOPIC_VERSION = "topics-v1-exploratory-20261118"
TOPICS = tuple(f"T{i:02d}" for i in range(1, 9))
TOPIC_ROOT = ROOT / "output/topic_selection" / TOPIC_VERSION
CLUSTER_ROOT = ROOT / "output/topic_selection/task12-20261118-sentence-clustering-v1"
CLEAN = ROOT / "data/cleaned/freeze_20261101"
META = ROOT / "data/metadata/freeze_20261101"
TIME_EXPRESSION = re.compile(r"\d[\d,.]*\s*(?:hours?\b|hrs?\b|h\b|(?:个)?小时|小時|分钟|分鐘)", re.I)


def json_lines(path):
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            yield json.loads(line)


def weight_matrix(reviews, scope):
    key = f"topic_weights_{scope}_units"
    result = pd.DataFrame(
        [{topic: row[key].get(topic, 0.0) for topic in TOPICS} for row in reviews],
        index=pd.Index([row["review_id"] for row in reviews], name="review_id"),
    )
    values = result.to_numpy()
    totals = values.sum(axis=1)
    if not result.index.is_unique or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid {scope} topic weights")
    if not (np.isclose(totals, 1, atol=5e-6, rtol=0) | (totals == 0)).all():
        raise ValueError(f"{scope} topic weights are not normalized")
    return result.div(result.sum(axis=1).replace(0, 1), axis="index")


def k8_cluster_policy(primary, alternative, config):
    """Project outcome-blind cluster allowances using overlap on shared fit units."""
    overlaps = defaultdict(Counter)
    for k10, k8 in zip(primary, alternative, strict=True):
        if k10["assignment_scope"] != k8["assignment_scope"]:
            raise ValueError("Different fit partitions across candidates")
        if k8["assignment_scope"] == "fit_candidate":
            overlaps[k8["cluster"]][k10["cluster"]] += 1
    policy, audit = {}, {}
    for cluster, counts in sorted(overlaps.items()):
        total = sum(counts.values())
        selected = [parent for parent in sorted(counts) if counts[parent] / total >= .20]
        if not selected:
            selected = [max(sorted(counts), key=counts.get)]
        if len(selected) == 1:
            policy[cluster] = config["cluster_topic_map"][selected[0]]
        else:
            topics = {topic for parent in selected for topic in config["cluster_topic_map"][parent]["topic_ids"]}
            policy[cluster] = {"topic_ids": sorted(topics), "confidence": "low", "route": "multi_candidate"}
        audit[cluster] = {
            "fit_units": total, "k10_cluster_counts": dict(sorted(counts.items())),
            "k10_clusters_used": selected, "policy": policy[cluster],
        }
    return policy, audit


def annotation_sample(unit_labels, seed_package):
    excluded = {row["unit_id"] for row in seed_package["seed_units"]}
    rng = np.random.default_rng(20261118)
    sample = {}

    def pick(items, count):
        items = [row for row in items if row["unit_id"] not in excluded and row["unit_id"] not in sample]
        for index in rng.choice(len(items), min(count, len(items)), replace=False):
            row = items[index]
            sample[row["unit_id"]] = row

    for topic in TOPICS:
        for language in ("english", "schinese"):
            group = [row for row in unit_labels if row["primary_topic_id"] == topic
                     and row["source_languages"] == [language]]
            pick([row for row in group if row["fit_candidate"]], 5)
            pick([row for row in group if not row["fit_candidate"]], 1)
    for route in ("A01", "A02"):
        pick([row for row in unit_labels if route in row["auxiliary_routes"]], 8)
    ordered = sorted(sample.values(), key=lambda row: row["unit_id"])
    return [{
        "unit_id": row["unit_id"],
        "text": "'" + row["text"] if row["text"].lstrip().startswith(("=", "+", "-", "@")) else row["text"],
        "human_topic_ids": "",
        "human_auxiliary_route": "", "evidence_or_note": "",
    } for row in ordered]


def load():
    provenance = {}

    def record(path, expected=None):
        provenance[common.relative(path)] = {
            "sha256": common.sha256(path),
            "recorded_sha256": expected,
            "match": common.hash_status(path, expected) if expected else "snapshot",
        }
        return path

    def from_manifest(directory, manifest_name, data_name):
        manifest = common.read_json(record(directory / manifest_name))
        entry = next(row for row in manifest["files"] if row["path"].endswith("/" + data_name))
        return record(ROOT / entry["path"], entry["sha256"])

    clean_path = from_manifest(CLEAN, "cleaning_manifest.json", "clean_master.jsonl")
    meta_path = from_manifest(META, "metadata_manifest.json", "game_meta.csv")
    fields = (
        "recommendationid", "steamid", "appid", "language", "voted_up", "timestamp_created",
        "association_eligible", "playtime_analysis_eligible", "playtime_at_review",
        "playtime_at_review_missing", "playtime_at_review_invalid", "is_playtime_inconsistent",
        "is_playtime_outlier", "in_common_language_window", "is_text_canonical",
        "is_short_text", "is_low_information",
    )
    master = pd.DataFrame(({k: row[k] for k in fields} for row in json_lines(clean_path)))
    master = master.rename(columns={"recommendationid": "review_id", "voted_up": "recommend"}).set_index(
        "review_id", verify_integrity=True,
    )
    if not master.recommend.map(lambda value: isinstance(value, bool)).all():
        raise ValueError("Unexpected recommendation field")
    master["recommend"] = master.recommend.astype(int)
    meta = pd.read_csv(meta_path, usecols=["appid", "sample_status"]).set_index("appid", verify_integrity=True)
    if not set(master.appid) <= set(meta.index) or not set(master.language) <= {"english", "schinese"}:
        raise ValueError("Unexpected games or languages in clean data")
    master["complete_game"] = master.appid.map(meta.sample_status).eq("complete")
    master["hours"] = master.playtime_at_review / 60
    valid_hours = master.loc[master.playtime_analysis_eligible, "hours"].to_numpy()
    if not np.isfinite(valid_hours).all() or (valid_hours < 0).any():
        raise ValueError("Unexpected playtime after eligibility filter")

    manifest = common.read_json(record(TOPIC_ROOT / "topic_v1.json"))
    contract = common.read_json(record(TOPIC_ROOT / "analysis_input_contract.json"))
    if manifest["topic_version"] != TOPIC_VERSION or contract["topic_version"] != TOPIC_VERSION:
        raise ValueError("Unexpected topic version")
    if tuple(row["topic_id"] for row in manifest["topics"]) != TOPICS:
        raise ValueError("Unexpected primary topic IDs")
    if any(contract[key] for key in ("recommendation_label_used_for_topic_generation",
                                     "playtime_used_for_topic_generation")):
        raise ValueError("Topic contract uses an analysis outcome")

    source = manifest["source_files"]
    config = common.read_json(record(ROOT / "config/topics_v1_exploratory.json", source["config"]))
    units, lookup = assignment.load_units(record(ROOT / config["units"], source["units"]))
    primary = assignment.load_assignments(record(ROOT / config["assignments"], source["assignments"]), units)
    record(ROOT / config["topic_draft"], source["topic_draft"])
    review_to_units = common.read_json(record(ROOT / config["review_to_units"], source["review_to_units"]))
    seed_package = common.read_json(record(ROOT / config["topic_seeds"], source["topic_seeds"]))
    cache = ROOT / config["vectors_dir"]
    vectors_manifest = common.read_json(record(cache / "sentence_embedding_manifest.json"))
    ids = common.read_json(record(cache / "unit_ids.json", vectors_manifest["unit_ids_file_sha256"]))
    vectors = np.load(record(cache / "sentence_vectors.npy", source["vectors"]), mmap_mode="r", allow_pickle=False)
    if ids != list(lookup) or list(vectors.shape) != vectors_manifest["shape"]:
        raise ValueError("Vector IDs or shape are incompatible with the handoff")
    if set(review_to_units) != set(master.index):
        raise ValueError("Topic handoff and clean data have different review populations")
    for unit in units:
        if not unit["review_ids"] or any(unit["unit_id"] not in review_to_units[r] for r in unit["review_ids"]):
            raise ValueError(f"Invalid unit-to-review mapping for {unit['unit_id']}")

    prototypes = assignment.build_prototypes(seed_package, units, lookup, vectors)
    published_units = list(json_lines(record(TOPIC_ROOT / "unit_topic_assignments.jsonl")))
    rebuilt = assignment.assign_units(config, units, primary, vectors, prototypes)
    keys = ("unit_id", "cluster_id", "assignment_scope", "topic_ids", "auxiliary_routes", "confidence", "review_ids")
    if any(any(a[k] != b[k] for k in keys) for a, b in zip(published_units, rebuilt, strict=True)):
        raise ValueError("Published unit labels disagree with rederived K=10 labels")
    published_reviews = list(json_lines(record(TOPIC_ROOT / "review_topic_weights.jsonl")))
    reviews = assignment.review_records(config, published_units, units, review_to_units)
    if published_reviews != reviews or set(row["review_id"] for row in reviews) != set(master.index):
        raise ValueError("Published review assignments disagree with rederived review weights")
    master["confidence"] = pd.Series({row["review_id"]: row["confidence"] for row in reviews})
    master["fit_units"] = pd.Series({row["review_id"]: row["fit_unit_count"] for row in reviews})
    master["all_units"] = pd.Series({row["review_id"]: row["unit_count"] for row in reviews})

    alternative = assignment.load_assignments(record(CLUSTER_ROOT / "sentence_k8_s17/assignments.csv"), units)
    policy, k8_audit = k8_cluster_policy(primary, alternative, config)
    k8 = assignment.assign_units({**config, "cluster_topic_map": policy}, units, alternative, vectors, prototypes)

    filters = {
        "k8_s17": k8,
        "no_a01_a02": [row for row in published_units if not {"A01", "A02"}.intersection(row["auxiliary_routes"])],
        "no_time_mentions": [row for row in published_units if not TIME_EXPRESSION.search(row["text"])],
    }
    weights = {
        "primary": weight_matrix(reviews, "fit"),
        "all_units": weight_matrix(reviews, "all"),
        **{key: weight_matrix(assignment.review_records(config, rows, units, review_to_units), "fit")
           for key, rows in filters.items()},
    }
    diagnostics = {
        "topic_source": TOPIC_VERSION, "review_and_unit_assignments_regenerated": True,
        "alternative_cluster_mapping": k8_audit,
        "fit_unit_counts_after_topic_filters": {
            key: sum(row["fit_candidate"] for row in rows) for key, rows in filters.items()
        },
        "seed_units_excluded_from_annotation_sample": len(seed_package["seed_units"]),
        "ordered_review_ids_sha256": hashlib.sha256(("\n".join(master.index) + "\n").encode()).hexdigest(),
    }
    return master, weights, provenance, diagnostics, annotation_sample(published_units, seed_package)

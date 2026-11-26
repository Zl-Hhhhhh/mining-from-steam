"""Verify the structured delivery and rebuild its fragment-level sensitivity weights."""
from collections import Counter
import json

import numpy as np
import pandas as pd

import analysis_common as common
import assign_topics_v1_cluster_derived as assignment


TOPIC_VERSION = "topics-v1-structured-exploratory-20261125"
CONFIG = common.ROOT / "config/rq3_structured_exploratory.json"
TOPICS = tuple(f"T{i:02d}" for i in range(1, 9))


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle]


def load(paths, review_ids, cluster_policy):
    topic = common.read_json(paths["topic_manifest"])
    delivery = common.read_json(paths["contract"])
    if any(package["topic_version"] != TOPIC_VERSION for package in (topic, delivery)):
        raise ValueError("Unexpected structured topic version")
    if tuple(row["topic_id"] for row in topic["topics"]) != TOPICS:
        raise ValueError("Unexpected primary topic IDs")

    for key in ("topic_manifest", "reviews", "review_csv", "labels"):
        path = paths[key]
        common.hash_status(path, delivery["files"][path.name]["sha256"])
    reviews = read_jsonl(paths["reviews"])
    labels = read_jsonl(paths["labels"])
    units, lookup = assignment.load_units(paths["units"])
    review_map = common.read_json(paths["review_map"])
    if set(review_map) != set(review_ids) or any(len(ids) != len(set(ids)) for ids in review_map.values()):
        raise ValueError("Missing, extra or duplicated review/fragment associations")
    for unit, label in zip(units, labels, strict=True):
        if any(unit[key] != label[key] for key in ("unit_id", "review_ids", "fit_candidate", "text")):
            raise ValueError(f"Published label differs from the structured handoff: {unit['unit_id']}")
        if set(label["topic_ids"]) - set(TOPICS):
            raise ValueError(f"Unexpected topic in {unit['unit_id']}")
    if {(rid, uid) for rid, ids in review_map.items() for uid in ids} != {
        (rid, unit["unit_id"]) for unit in units for rid in unit["review_ids"]
    }:
        raise ValueError("Invalid inverse review/fragment mapping")

    config = common.read_json(paths["topic_config"])
    rebuilt = assignment.review_records(config, labels, units, review_map)
    csv = pd.read_csv(paths["review_csv"], dtype={"review_id": str}).set_index("review_id", verify_integrity=True)
    if set(csv.index) != set(review_map) or len(reviews) != len(review_map):
        raise ValueError("Published weight tables do not cover the structured review population")
    for published, expected in zip(reviews, rebuilt, strict=True):
        # The structured export also counts standalone auxiliary fragments in the JSONL routes.
        if any(published[key] != value for key, value in expected.items() if key != "auxiliary_routes"):
            raise ValueError(f"Published structured weights do not match fragments: {expected['review_id']}")
    for scope in ("fit", "all"):
        expected = np.array([[row[f"topic_weights_{scope}_units"].get(t, 0) for t in TOPICS] for row in reviews])
        observed = csv.loc[[row["review_id"] for row in reviews], [f"{scope}_{t}" for t in TOPICS]]
        if not np.array_equal(expected, observed.to_numpy()):
            raise ValueError(f"CSV and JSONL disagree on structured {scope} weights")
        if np.count_nonzero(expected.sum(axis=1)) != delivery["counts"][f"{scope}_evidence_reviews"]:
            raise ValueError(f"Unexpected structured {scope} coverage")

    seeds = common.read_json(paths["seeds"])
    legacy_units, legacy_lookup = assignment.load_units(paths["legacy_units"])
    vectors = np.load(paths["vectors"], mmap_mode="r", allow_pickle=False)
    if common.read_json(paths["vector_ids"]) != list(legacy_lookup) or list(vectors.shape) != common.read_json(paths["vector_manifest"])["shape"]:
        raise ValueError("Legacy vector IDs or shape differ from the frozen archive")
    primary = assignment.load_assignments(paths["clusters"], legacy_units)
    alternative = assignment.load_assignments(paths["k8_clusters"], legacy_units)
    prototypes = assignment.build_prototypes(seeds, legacy_units, legacy_lookup, vectors)
    k8, audit = cluster_policy(primary, alternative, config)
    unrestricted = {key: {**value, "topic_ids": list(TOPICS), "route": "multi_candidate"}
                    for key, value in config["cluster_topic_map"].items()}

    def reaggregate(rows):
        return assignment.review_records(config, rows, units, review_map)

    def reassign(mapping, clusters):
        replacements = {row["unit_id"]: row for row in assignment.assign_units(
            {**config, "cluster_topic_map": mapping}, legacy_units, clusters, vectors, prototypes,
        ) if row["unit_id"] in lookup}
        return reaggregate([replacements.get(row["unit_id"], row) for row in labels])

    variants = {
        "no_auxiliary": reaggregate([row for row in labels if not {"A01", "A02"}.intersection(row["auxiliary_routes"])]),
        "k8_overlap_proxy": reassign(k8, alternative),
        "unrestricted_seeds": reassign(unrestricted, primary),
    }
    diagnostics = {
        "review_weights_reaggregated": True,
        "csv_jsonl_weights_match": True,
        "master_reviews": len(review_map),
        "fit_evidence_reviews": delivery["counts"]["fit_evidence_reviews"],
        "all_evidence_reviews": delivery["counts"]["all_evidence_reviews"],
        "unit_confidence_counts": dict(Counter(row["confidence"] for row in labels)),
        "high_confidence_topics": dict(Counter(t for row in labels if row["confidence"] == "high" for t in row["topic_ids"])),
        "k8_proxy_mapping": audit,
        "alternative_mapping_scope": "Retained legacy fragments only; structured links and added fragment labels stay fixed.",
        "legacy_fragments_retained": len(legacy_lookup.keys() & lookup.keys()),
        "structured_fragments": len(labels),
        "alternative_mapping_uses_outcomes": False,
    }
    return reviews, labels, variants, seeds, reaggregate, diagnostics

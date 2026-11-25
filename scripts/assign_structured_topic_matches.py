"""Assemble the structured sentence cache and run the frozen topic comparison."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

import assign_topics_v1_cluster_derived as assignment
import embedding_bge_sentences as encoder
from reject_generic_topic_matches import MODEL_FIELDS, is_pure_generic


ROOT = encoder.ROOT
BASE = ROOT / "output/topic_selection/task12-20261114"
WORK = ROOT / ".cache-topic/structured-20261125"
DEFAULT_OUTPUT = ROOT / "output/topic_selection/topics-v1-structured-exploratory-20261125"
RELAXED = ROOT / ".cache-topic/lexical-relaxed-20261125"
MINIMUM_SCORE = 0.65
MINIMUM_GAP = 0.06


def jsonl(path):
    with path.open(encoding="utf-8") as handle:
        yield from map(json.loads, handle)


def load_cache(config_path):
    encoder.verify(config_path)
    config, _, units, _ = encoder.load_inputs(config_path)
    directory = ROOT / config["output"]
    manifest = encoder.read_json(directory / "sentence_embedding_manifest.json")
    vectors = np.load(directory / manifest["vector_file"], mmap_mode="r", allow_pickle=False)
    return config, units, manifest, vectors


def score_application_units(units, vectors, centers, prototypes, cluster_map):
    topics = sorted(prototypes)
    matrix = vectors @ np.stack([prototypes[topic] for topic in topics]).T
    nearest = np.argmax(2 * (vectors @ centers.T) - np.sum(centers * centers, axis=1), axis=1)
    rankings = np.argsort(-matrix, axis=1, kind="stable")
    rows = []
    for unit, cluster, scores, order in zip(units, nearest, matrix, rankings):
        top = topics[order[0]]
        score = float(scores[order[0]])
        gap = score - float(scores[order[1]])
        cluster_policy = cluster_map[str(cluster)]
        allowed = top in cluster_policy["topic_ids"]
        if is_pure_generic(unit["text"]):
            route = "A01"
            reason = "whole_fragment_generic"
        elif score < MINIMUM_SCORE or gap < MINIMUM_GAP:
            route = "A03"
            reason = "score_or_margin"
        elif not allowed:
            route = "A03"
            reason = "cluster_disagreement"
        else:
            route = None
            reason = "assigned"
        confidence = assignment.lower_confidence(cluster_policy["confidence"] if route is None else "low")
        if cluster_policy["route"] == "multi_candidate":
            confidence = assignment.lower_confidence(confidence)
        rows.append({
            "unit_id": unit["unit_id"], "assignment_scope": "application_only", "fit_candidate": False,
            "cluster_id": int(cluster), "topic_ids": [] if route else [top],
            "primary_topic_id": None if route else top, "auxiliary_routes": [route] if route else [],
            "decision": reason, "confidence": confidence, "global_top_topic": top,
            "global_top_score": round(score, 6), "global_runner_up": topics[order[1]],
            "global_gap": round(gap, 6), "global_top_allowed_in_cluster": allowed,
            "topic_scores": {topic: round(float(value), 6) for topic, value in zip(topics, scores)},
        })
    return rows


def assemble_cache(handoff, units, caches):
    request_path = handoff / "encoding_request.json"
    unit_path = handoff / "sentence_units.jsonl"
    request = encoder.read_json(request_path)
    ids = [u["unit_id"] for u in units]
    if encoder.ordered_hash(ids) != request["ordered_unit_ids_sha256"] or len(set(ids)) != len(ids):
        raise ValueError("Structured input ID order does not match its encoding request")
    if encoder.sha_file(unit_path) != request["input_sha256"] or [len(ids), 1024] != request["required_shape"]:
        raise ValueError("Structured input bytes or shape differ from the encoding request")
    location = {}
    for source, (_, cached_units, manifest, _) in caches.items():
        for row, unit in enumerate(cached_units):
            previous = location.setdefault(unit["unit_id"], (source, row, unit["text_sha256"]))
            if previous != (source, row, unit["text_sha256"]):
                raise ValueError("The source vector caches contain overlapping unit IDs")
        if any(manifest[field] != caches["original"][2][field] for field in MODEL_FIELDS):
            raise ValueError(f"Model settings differ in vector source {source}")
    source_rows = {source: ([], []) for source in caches}
    for destination_row, unit in enumerate(units):
        source, source_row, text_hash = location[unit["unit_id"]]
        if text_hash != unit["text_sha256"]:
            raise ValueError(f"Text hash differs for {unit['unit_id']}")
        source_rows[source][0].append(destination_row)
        source_rows[source][1].append(source_row)
    if len(source_rows["structured_delta"][0]) != len(caches["structured_delta"][1]):
        raise ValueError("The structural delta contains rows outside the structured handoff")

    directory = WORK / "merged_vectors"
    config_path = WORK / "merged_encoding_config.json"
    manifest_path = directory / "sentence_embedding_manifest.json"
    counts = {key: len(indices[0]) for key, indices in source_rows.items()}
    sources = {source: {"used_rows": counts[source], "cpu_threads": cache[2]["cpu_threads"],
                        "vectors_sha256": cache[2]["vectors_sha256"]} for source, cache in caches.items()}
    if manifest_path.is_file():
        existing = encoder.read_json(manifest_path)
        if existing["input_sha256"] != request["input_sha256"] or existing["vector_sources"] != sources:
            raise ValueError("The published merged cache uses different vector sources")
        encoder.verify(config_path)
        return counts
    directory.mkdir(exist_ok=True)
    vector_path = directory / "sentence_vectors.npy"
    staging = directory / "sentence_vectors.assembling.npy"
    array = np.lib.format.open_memmap(staging, mode="w+", dtype=np.float32, shape=(len(ids), 1024))
    for source, (destinations, origins) in source_rows.items():
        for start in range(0, len(origins), 2048):
            end = start + 2048
            array[destinations[start:end]] = caches[source][3][origins[start:end]]
    array.flush()
    del array
    staging.replace(vector_path)
    id_path = directory / "unit_ids.json"
    encoder.write_json(id_path, ids)
    model = {key: caches["original"][2][key] for key in MODEL_FIELDS}
    encoder.write_json(config_path, {
        **caches["original"][0], "input": unit_path.relative_to(ROOT).as_posix(),
        "input_sha256": request["input_sha256"], "encoding_request": request_path.relative_to(ROOT).as_posix(),
        "encoding_request_sha256": encoder.sha_file(request_path), "output": directory.relative_to(ROOT).as_posix(),
    })
    encoder.write_json(manifest_path, {
        **model,
        "schema_version": "1.0", "assembly_only": True,
        "input_path": unit_path.relative_to(ROOT).as_posix(), "input_sha256": request["input_sha256"],
        "shape": request["required_shape"], "ordered_unit_ids_sha256": request["ordered_unit_ids_sha256"],
        "vector_file": vector_path.name, "vectors_sha256": encoder.sha_file(vector_path),
        "unit_ids_file_sha256": encoder.sha_file(id_path), "vector_sources": sources,
        "fit_candidate_units": sum(bool(unit["fit_candidate"]) for unit in units),
        "reviews_with_units": len({rid for unit in units for rid in unit["review_ids"]}),
    })
    encoder.verify(config_path)
    return counts


def add_auxiliary_routes(reviews, handoff):
    a01_counts = Counter(rid for row in jsonl(handoff / "auxiliary_units.jsonl") for rid in row["review_ids"])
    for row in reviews:
        if count := a01_counts[row["review_id"]]:
            row["auxiliary_routes"]["A01"] = row["auxiliary_routes"].get("A01", 0) + count


def validate_review_map(units, review_map):
    assigned_reviews = defaultdict(set)
    for rid, ids in review_map.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate unit association for review {rid}")
        for unit_id in ids:
            assigned_reviews[unit_id].add(rid)
    if assigned_reviews != {unit["unit_id"]: set(unit["review_ids"]) for unit in units}:
        raise ValueError("Review-to-unit associations differ from the unit metadata")


def collect_topic_presence(rows_by_id, review_map):
    return {rid: {topic for unit_id in unit_ids for topic in rows_by_id[unit_id]["topic_ids"]}
            for rid, unit_ids in review_map.items()}


def compare_review_maps(before, after, metadata, review_ids):
    ids = set(review_ids)
    old = {(rid, topic) for rid in ids for topic in before[rid]}
    new = {(rid, topic) for rid in ids for topic in after[rid]}
    gained = new - old
    lost = old - new
    return {
        "reviews": len(ids), "before_detected": sum(bool(before[rid]) for rid in ids),
        "after_detected": sum(bool(after[rid]) for rid in ids),
        "first_detected": sum(not before[rid] and bool(after[rid]) for rid in ids),
        "lost_detection": sum(bool(before[rid]) and not after[rid] for rid in ids),
        "gained_topic_pairs": len(gained), "lost_topic_pairs": len(lost),
        "review_topics_changed": sum(before[rid] != after[rid] for rid in ids),
        "by_source_language": {
            language: [sum(bool(before[rid]) for rid in ids if metadata[rid] == language),
                       sum(bool(after[rid]) for rid in ids if metadata[rid] == language)]
            for language in sorted(set(metadata.values()))
        },
    }


def compare_audit(before, after):
    root = ROOT / "output/topic_review/task12-relaxed-blind-20261125"
    key = encoder.read_json(root / "coordinator/sampling_key.json")
    answer = encoder.read_json(root / "reviewer_A/task12-relaxed-blind-20261125_A_complete.json")
    if (answer["complete"] is not True or answer["assignment"] != "A"
            or answer["sample_sha256"] != key["sample_sha256"] or answer["answers"].keys() != key["cases"].keys()):
        raise ValueError("Reviewer A answers do not match the sampling key")
    available = {case_id: case for case_id, case in key["cases"].items()
                 if answer["answers"][case_id]["judgment"] != "A03"}
    report = {}
    for name, cases in (("all", available),
                        ("long", {key: r for key, r in available.items() if r["raw_character_count"] > 250})):
        gold = {(r["review_id"], topic) for key, r in cases.items()
                for topic in answer["answers"][key]["evidence"] if topic.startswith("T")}
        rids = {r["review_id"] for r in cases.values()}
        old = {(rid, topic) for rid in rids for topic in before[rid]}
        new = {(rid, topic) for rid in rids for topic in after[rid]}
        report[name] = {
            "reviews": len(cases), "reference_pairs": len(gold),
            "before_true": len(old & gold), "before_unsupported": len(old - gold), "before_missing": len(gold - old),
            "after_true": len(new & gold), "after_unsupported": len(new - gold), "after_missing": len(gold - new),
            "gained_supported": len((new - old) & gold), "lost_supported": len((old - new) & gold),
            "gained_unsupported": len((new - old) - gold), "lost_unsupported": len((old - new) - gold),
        }
    report["answer_sha256"] = encoder.sha_file(root / "reviewer_A/task12-relaxed-blind-20261125_A_complete.json")
    report["sample_sha256"] = key["sample_sha256"]
    return report, key["cases"]["R097"]["review_id"]


def run(output):
    if output.exists():
        raise FileExistsError(f"Choose a new topic output directory: {output}")
    config_path = ROOT / "config/topics_v1_exploratory.json"
    config = encoder.read_json(config_path)
    handoff = BASE / "handoff-structured-20261125"
    previous_handoff = BASE / "handoff-lexical-relaxed-20261125"
    units, _ = assignment.load_units(handoff / "sentence_units.jsonl")
    caches = {
        "original": load_cache(None),
        "relaxed_delta": load_cache(RELAXED / "encoding_config.json"),
        "structured_delta": load_cache(WORK / "encoding_config.json"),
    }
    _, original_units, _, original_vectors = caches["original"]
    original_lookup = {u["unit_id"]: i for i, u in enumerate(original_units)}
    seed_path = ROOT / config["topic_seeds"]
    seeds = encoder.read_json(seed_path)
    prototypes = assignment.build_prototypes(seeds, original_units, original_lookup, original_vectors)
    centers_path = (ROOT / config["assignments"]).with_name("centers.npz")
    with np.load(centers_path, allow_pickle=False) as center_file:
        centers = center_file["centers"]
    vectors_by_source = assemble_cache(handoff, units, caches)

    frozen = list(jsonl(ROOT / config["output"] / "unit_topic_assignments.jsonl"))
    if [r["unit_id"] for r in frozen] != list(original_lookup):
        raise ValueError("Frozen labels do not match the original vector IDs")
    _, relaxed_units, _, relaxed_vectors = caches["relaxed_delta"]
    _, structural_units, _, structural_vectors = caches["structured_delta"]
    relaxed = score_application_units(relaxed_units, relaxed_vectors, centers, prototypes, config["cluster_topic_map"])
    novel = score_application_units(structural_units, structural_vectors, centers, prototypes, config["cluster_topic_map"])
    reference = RELAXED / "generic-rejection-probe/unit_decisions.jsonl"
    if reference.is_file():
        expected = {r["unit_id"]: r["topics_by_variant"]["rule_only"] for r in jsonl(reference)}
        if {r["unit_id"]: r["primary_topic_id"] for r in relaxed} != expected:
            raise ValueError("Replayed relaxed labels differ from the previous generic-rejection trial")
    all_by_id = {r["unit_id"]: r for r in frozen + relaxed + novel}
    new_assignments = []
    for unit in units:
        row = all_by_id[unit["unit_id"]]
        if row["fit_candidate"] != bool(unit["fit_candidate"]):
            raise ValueError(f"Fitting scope changed for retained unit {unit['unit_id']}")
        new_assignments.append({
            **row, **{key: unit[key] for key in ("text", "review_ids", "appids", "source_languages")},
        })

    old_map = encoder.read_json(previous_handoff / "review_to_units.json")
    new_map = encoder.read_json(handoff / "review_to_units.json")
    meta = {str(r["recommendationid"]): r["language"] for r in jsonl(
        ROOT / "data/derived/embedding-bge/freeze_20261101-v1/master_index.jsonl"
    )}
    if old_map.keys() != new_map.keys() or new_map.keys() != meta.keys():
        raise ValueError("Structured and relaxed handoffs cover different review IDs")
    validate_review_map(original_units + relaxed_units, old_map)
    validate_review_map(units, new_map)
    before = assignment.review_records(config, frozen + relaxed, original_units + relaxed_units, old_map)
    after = assignment.review_records(config, new_assignments, units, new_map)
    add_auxiliary_routes(after, handoff)
    old_topics = {r["review_id"]: set(r["topic_presence"]) for r in before}
    new_topics = {r["review_id"]: set(r["topic_presence"]) for r in after}
    changed = {rid for rid in new_map if old_map[rid] != new_map[rid]}
    audit, example_rid = compare_audit(old_topics, new_topics)
    strict_original = score_application_units(
        original_units, original_vectors, centers, prototypes, config["cluster_topic_map"]
    )
    strict_by_id = {r["unit_id"]: r for r in strict_original + relaxed + novel}
    same_rule_before = collect_topic_presence(strict_by_id, old_map)
    same_rule_after = collect_topic_presence(strict_by_id, new_map)
    same_rule_audit, _ = compare_audit(same_rule_before, same_rule_after)
    new_by_id = {row["unit_id"]: row for row in new_assignments}
    audit["R097"] = {
        "before": sorted(old_topics[example_rid]), "after": sorted(new_topics[example_rid]),
        "units": [
            {key: new_by_id[unit_id].get(key) for key in
             ("text", "topic_ids", "global_top_topic", "global_top_score", "global_gap", "decision")}
            for unit_id in new_map[example_rid]
        ],
    }
    output.mkdir(parents=True)
    assignment.write_jsonl(output / "unit_topic_assignments.jsonl", new_assignments)
    assignment.write_jsonl(output / "review_topic_weights.jsonl", after)
    assignment.write_review_csv(output / "review_topic_weights.csv", after, sorted(prototypes))
    summary = {
        "status": "exploratory_structured_comparison",
        "policy": {"minimum_score": MINIMUM_SCORE, "minimum_gap": MINIMUM_GAP,
                   "nearest_cluster_agreement_required": True, "generic_rejection": "whole_fragment_only",
                   "original_units_use_frozen_v1_labels": True},
        "source_checksums": {"topic_config": encoder.sha_file(config_path), "seeds": encoder.sha_file(seed_path),
                             "centers": encoder.sha_file(centers_path),
                             "matcher": encoder.sha_file(Path(__file__)),
                             "generic_rule": encoder.sha_file(Path(is_pure_generic.__code__.co_filename)),
                             "encoding_request": encoder.sha_file(handoff / "encoding_request.json")},
        "frozen_seed_count": len(seeds["seed_units"]), "vectors": vectors_by_source,
        "new_structural_units_matched": sum(bool(row["topic_ids"]) for row in novel),
        "structural_unit_decisions": Counter(row["decision"] for row in novel),
        "matched_structural_unit_topics": Counter(topic for row in novel for topic in row["topic_ids"]),
        "all_reviews": compare_review_maps(old_topics, new_topics, meta, meta),
        "reviews_with_changed_candidates": compare_review_maps(old_topics, new_topics, meta, changed),
        "reviewer_A": audit,
        "same_rule_sensitivity": {
            "scope": "Apply the same score, gap, cluster and generic rules to every unit on both sides.",
            "all_reviews": compare_review_maps(same_rule_before, same_rule_after, meta, meta),
            "reviewer_A": same_rule_audit,
        },
    }
    encoder.write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)

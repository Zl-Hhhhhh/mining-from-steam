"""Generate a transparent cluster-derived topics-v1 assignment package."""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

import analysis_common as common

ROOT = common.ROOT
DEFAULT_CONFIG = ROOT / "config/topics_v1_cluster_derived.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_units(path: Path) -> tuple[list[dict], dict[str, int]]:
    units = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lookup = {row["unit_id"]: index for index, row in enumerate(units)}
    if len(lookup) != len(units):
        raise ValueError("Duplicate sentence unit IDs")
    return units, lookup


def load_assignments(path: Path, units: list[dict]) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["unit_id"] for row in rows] != [unit["unit_id"] for unit in units]:
        raise ValueError("Cluster assignment order does not match sentence units")
    return rows


def confidence_rank(value: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(value, 1)


def lower_confidence(value: str) -> str:
    return {"high": "medium", "medium": "low", "low": "low"}[value]


def generic_text(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if len(normalized) > 80:
        return False
    patterns = (
        r"^(this is|it's|it is|great|good|nice|fun|awesome|amazing|perfect|worth it|best game)",
        r"^(好玩|很好玩|不错|很棒|推荐|值得|上头|神作|垃圾游戏)$",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def build_prototypes(seed_package: dict, units: list[dict], lookup: dict[str, int], vectors: np.ndarray) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for seed in seed_package["seed_units"]:
        if seed["unit_id"] not in lookup:
            raise ValueError(f"Seed unit missing from handoff: {seed['unit_id']}")
        grouped[seed["topic_id"]].append(lookup[seed["unit_id"]])
    prototypes = {}
    for topic_id, indices in sorted(grouped.items()):
        prototypes[topic_id] = normalize(np.asarray(vectors[indices]).mean(axis=0, keepdims=True))[0]
    return prototypes


def assign_units(config: dict, units: list[dict], assignments: list[dict], vectors: np.ndarray,
                 prototypes: dict[str, np.ndarray]) -> list[dict]:
    topic_ids = sorted(prototypes)
    prototype_matrix = np.vstack([prototypes[topic_id] for topic_id in topic_ids])
    results = []
    for index, (unit, cluster_row) in enumerate(zip(units, assignments)):
        cluster = int(cluster_row["cluster"])
        cluster_policy = config["cluster_topic_map"][str(cluster)]
        allowed = cluster_policy["topic_ids"]
        scores = vectors[index] @ prototype_matrix.T
        ranked = sorted(((topic_ids[i], float(scores[i])) for i in range(len(topic_ids)) if topic_ids[i] in allowed),
                        key=lambda item: (-item[1], item[0]))
        selected = []
        if ranked and ranked[0][1] >= config["min_topic_score"]:
            selected.append(ranked[0])
            if len(ranked) > 1 and ranked[1][1] >= config["secondary_min_score"]:
                if ranked[0][1] - ranked[1][1] <= config["secondary_margin"]:
                    selected.append(ranked[1])
        auxiliary = []
        if not selected:
            auxiliary.append("A01" if generic_text(unit["text"]) else "A03")
        elif len(selected) > 1:
            auxiliary.append("A02")
        elif cluster_policy["route"] == "multi_candidate" and generic_text(unit["text"]):
            auxiliary.append("A01")
        base_confidence = cluster_policy["confidence"]
        if len(selected) > 1 or cluster_policy["route"] == "multi_candidate":
            base_confidence = lower_confidence(base_confidence)
        if cluster_row["assignment_scope"] == "application_only":
            base_confidence = lower_confidence(base_confidence)
        results.append({
            "unit_id": unit["unit_id"],
            "cluster_id": cluster,
            "assignment_scope": cluster_row["assignment_scope"],
            "fit_candidate": bool(unit["fit_candidate"]),
            "topic_ids": [topic_id for topic_id, _ in selected],
            "primary_topic_id": selected[0][0] if selected else None,
            "auxiliary_routes": auxiliary,
            "topic_scores": {topic_id: round(score, 6) for topic_id, score in ranked},
            "confidence": base_confidence,
            "review_ids": unit["review_ids"],
            "text": unit["text"],
            "source_languages": unit.get("source_languages", []),
            "appids": unit.get("appids", []),
        })
    return results


def review_records(config: dict, assignments: list[dict], units: list[dict], review_to_units: dict[str, list[str]]) -> list[dict]:
    by_unit = {row["unit_id"]: row for row in assignments}
    results = []
    for review_id, unit_ids in sorted(review_to_units.items(), key=lambda item: int(item[0])):
        unit_rows = [by_unit[unit_id] for unit_id in unit_ids if unit_id in by_unit]
        all_scores = Counter()
        fit_scores = Counter()
        all_counts = Counter()
        fit_counts = Counter()
        routes = Counter()
        low_confidence = False
        for row in unit_rows:
            topics = row["topic_ids"]
            for route in row["auxiliary_routes"]:
                routes[route] += 1
            if row["confidence"] == "low":
                low_confidence = True
            if topics:
                contribution = 1 / len(topics)
                for topic_id in topics:
                    all_scores[topic_id] += contribution
                    all_counts[topic_id] += 1
                    if row["assignment_scope"] == "fit_candidate":
                        fit_scores[topic_id] += contribution
                        fit_counts[topic_id] += 1
        def normalized(counter: Counter) -> dict[str, float]:
            total = sum(counter.values())
            return {topic_id: round(value / total, 6) for topic_id, value in sorted(counter.items())} if total else {}
        if not all_scores:
            review_confidence = "low"
        elif low_confidence:
            review_confidence = "low"
        elif any(row["confidence"] == "medium" for row in unit_rows):
            review_confidence = "medium"
        else:
            review_confidence = "high"
        results.append({
            "review_id": review_id,
            "unit_count": len(unit_rows),
            "fit_unit_count": sum(row["assignment_scope"] == "fit_candidate" for row in unit_rows),
            "application_only_unit_count": sum(row["assignment_scope"] == "application_only" for row in unit_rows),
            "no_topic_evidence": not bool(all_scores),
            "topic_presence": sorted(all_scores),
            "topic_evidence_counts": dict(sorted(all_counts.items())),
            "topic_weights_all_units": normalized(all_scores),
            "topic_weights_fit_units": normalized(fit_scores),
            "auxiliary_routes": dict(sorted(routes.items())),
            "confidence": review_confidence,
            "unit_ids": unit_ids,
        })
    return results


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_review_csv(path: Path, rows: list[dict], topic_ids: list[str]) -> None:
    fields = ["review_id", "unit_count", "fit_unit_count", "application_only_unit_count", "no_topic_evidence", "confidence", "auxiliary_routes"]
    fields += [f"all_{topic_id}" for topic_id in topic_ids]
    fields += [f"fit_{topic_id}" for topic_id in topic_ids]
    output = []
    for row in rows:
        item = {field: row.get(field, "") for field in fields}
        item["auxiliary_routes"] = json.dumps(row["auxiliary_routes"], ensure_ascii=False)
        for topic_id in topic_ids:
            item[f"all_{topic_id}"] = row["topic_weights_all_units"].get(topic_id, 0.0)
            item[f"fit_{topic_id}"] = row["topic_weights_fit_units"].get(topic_id, 0.0)
        output.append(item)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def run(config_path: Path) -> dict:
    config = read_json(config_path)
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError(f"Output already exists. Choose a new version: {output}")
    output.mkdir(parents=True)
    units, lookup = load_units(ROOT / config["units"])
    assignments = load_assignments(ROOT / config["assignments"], units)
    manifest = read_json(ROOT / config["vectors_dir"] / "sentence_embedding_manifest.json")
    vectors_path = ROOT / config["vectors_dir"] / manifest["vector_file"]
    if common.sha256(vectors_path) != manifest["vectors_sha256"]:
        raise ValueError("Sentence vector hash mismatch")
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    seed_package = read_json(ROOT / config["topic_seeds"])
    prototypes = build_prototypes(seed_package, units, lookup, vectors)
    unit_assignments = assign_units(config, units, assignments, vectors, prototypes)
    review_to_units = read_json(ROOT / config["review_to_units"])
    reviews = review_records(config, unit_assignments, units, review_to_units)
    topic_ids = sorted(prototypes)
    source_files = {
        "config": common.sha256(config_path),
        "units": common.sha256(ROOT / config["units"]),
        "assignments": common.sha256(ROOT / config["assignments"]),
        "vectors": manifest["vectors_sha256"],
        "review_to_units": common.sha256(ROOT / config["review_to_units"]),
        "topic_draft": common.sha256(ROOT / config["topic_draft"]),
        "topic_seeds": common.sha256(ROOT / config["topic_seeds"]),
    }
    topic_draft = read_json(ROOT / config["topic_draft"])
    topic_file = {
        "schema_version": "1.0-topics-v1-cluster-derived",
        "topic_version": config["version"],
        "status": "cluster_derived_provisional",
        "derivation": "sentence_k10_s42 cluster mapping plus topic seed prototypes",
        "topics": topic_draft["primary_topics"],
        "auxiliary_routing": topic_draft["auxiliary_routing"],
        "cluster_topic_map": config["cluster_topic_map"],
        "assignment_policy": {
            "unit_of_evidence": "sentence_or_clause",
            "multi_label_allowed": True,
            "generic_route": "A01",
            "mixed_route": "A02",
            "unknown_route": "A03",
            "recommendation_is_topic": False,
            "selection_uses_recommendation_labels": False
        },
        "counts": {
            "units": len(unit_assignments),
            "fit_candidate_units": sum(row["assignment_scope"] == "fit_candidate" for row in unit_assignments),
            "application_only_units": sum(row["assignment_scope"] == "application_only" for row in unit_assignments),
            "reviews": len(reviews),
            "reviews_with_topic_evidence": sum(not row["no_topic_evidence"] for row in reviews),
            "reviews_without_topic_evidence": sum(row["no_topic_evidence"] for row in reviews),
        },
        "source_files": source_files,
        "semantic_review_status": "cluster-level review completed, unit-level manual signoff pending",
        "frozen_for_cluster_derived_analysis": True,
        "frozen_for_final_semantic_analysis": False,
        "limitations": [
            "Cluster-derived assignments inherit mixed semantics from clusters 0, 2, 3, 5, 7, 8 and 9.",
            "Topic prototypes use the current unreviewed seed proposal and should be revised if seed decisions change.",
            "Application-only units were assigned by nearest cluster centroid and receive lower confidence.",
            "Review weights are normalized equal contributions across assigned topics and are not model probabilities.",
            "This version supports exploratory RQ1 summaries. Final causal or confirmatory claims require semantic review."
        ]
    }
    write_json(output / "topic_v1.json", topic_file)
    write_json(output / "cluster_topic_map.json", {"candidate": config["candidate"], "mapping": config["cluster_topic_map"], "source": common.relative(config_path)})
    write_jsonl(output / "unit_topic_assignments.jsonl", unit_assignments)
    write_jsonl(output / "review_topic_weights.jsonl", reviews)
    write_review_csv(output / "review_topic_weights.csv", reviews, topic_ids)
    write_json(output / "assignment_summary.json", {
        "topic_unit_counts": dict(sorted(Counter(topic_id for row in unit_assignments for topic_id in row["topic_ids"]).items())),
        "auxiliary_route_counts": dict(sorted(Counter(route for row in unit_assignments for route in row["auxiliary_routes"]).items())),
        "confidence_counts": dict(sorted(Counter(row["confidence"] for row in unit_assignments).items())),
        "cluster_counts": dict(sorted(Counter(str(row["cluster_id"]) for row in unit_assignments).items())),
    })
    (output / "README.md").write_text(
        "# topics-v1 cluster-derived\n\n"
        "This package freezes a transparent cluster-derived topic version from `sentence_k10_s42`. "
        "All sentence units and all review IDs receive an output record. Mixed clusters retain multiple candidate topics or auxiliary routes. "
        "The package is frozen for exploratory cluster-derived analysis and remains provisional for final semantic analysis. "
        "Review weights are normalized equal contributions across assigned topics, not probabilities.\n",
        encoding="utf-8",
    )
    return topic_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run(args.config)
    print(json.dumps({"status": result["status"], "topic_version": result["topic_version"],
                      "counts": result["counts"], "seconds": round(time.perf_counter() - started, 1)}, ensure_ascii=False), flush=True)

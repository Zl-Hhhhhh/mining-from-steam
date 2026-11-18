"""Cluster the delivered sentence-level BGE cache and measure resampling stability."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

import analysis_common as common
import check_topic_stability as stability

ROOT = common.ROOT
DEFAULT_CONFIG = ROOT / "config/topic_sentence_clustering.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_inputs(config: dict) -> tuple[list[dict], np.ndarray, list[str], np.ndarray, dict]:
    units_path = ROOT / config["input_units"]
    embedding_dir = ROOT / config["embedding_dir"]
    manifest_path = embedding_dir / "sentence_embedding_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    units = [json.loads(line) for line in units_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = json.loads((embedding_dir / manifest["unit_ids_file"]).read_text(encoding="utf-8"))
    vectors_path = embedding_dir / manifest["vector_file"]
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    actual_input_hash = common.sha256(units_path)
    actual_vectors_hash = common.sha256(vectors_path)
    actual_ids_hash = common.sha256(embedding_dir / manifest["unit_ids_file"])
    if actual_input_hash != manifest["input_sha256"]:
        raise ValueError("Sentence input hash mismatch")
    if actual_vectors_hash != manifest["vectors_sha256"]:
        raise ValueError("Sentence vector hash mismatch")
    if actual_ids_hash != manifest["unit_ids_file_sha256"]:
        raise ValueError("Sentence ID file hash mismatch")
    if len(units) != len(ids) or [row["unit_id"] for row in units] != ids:
        raise ValueError("Sentence ID order mismatch")
    if tuple(vectors.shape) != tuple(manifest["shape"]):
        raise ValueError("Sentence vector shape mismatch")
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("Sentence vectors are not finite float32 values")
    fit_indices = np.array([i for i, row in enumerate(units) if bool(row["fit_candidate"])], dtype=int)
    if len(fit_indices) != int(manifest["fit_candidate_units"]):
        raise ValueError("Fit candidate count mismatch")
    provenance = {
        "input_units": common.sha256(units_path),
        "embedding_manifest": common.sha256(manifest_path),
        "vectors": manifest["vectors_sha256"],
        "unit_ids": manifest["unit_ids_file_sha256"],
        "topic_draft": common.sha256(ROOT / config["topic_draft"]),
        "fit_candidate_units": len(fit_indices),
        "all_units": len(units),
        "reviews_with_units": len({review_id for row in units for review_id in row["review_ids"]}),
    }
    return units, vectors, ids, fit_indices, provenance


def strata_for(units: list[dict], indices: np.ndarray) -> list[str]:
    result = []
    for i in indices:
        row = units[int(i)]
        languages = ",".join(sorted(str(x) for x in row.get("source_languages", []))) or "unknown"
        appids = ",".join(sorted(str(x) for x in row.get("appids", []))) or "unknown"
        result.append(f"{languages}:{appids}")
    return result


def candidate_representatives(units: list[dict], ids: list[str], vectors: np.ndarray,
                              indices: np.ndarray, model: KMeans, count: int) -> list[dict]:
    centers = normalize(model.cluster_centers_)
    fit_vectors = np.asarray(vectors[indices])
    similarities = fit_vectors @ centers.T
    representatives = []
    for cluster in range(model.n_clusters):
        members = np.flatnonzero(model.labels_ == cluster)
        ranked = members[np.argsort(-similarities[members, cluster])[:count]]
        for rank, local_index in enumerate(ranked, 1):
            row = units[int(indices[local_index])]
            representatives.append({
                "cluster": cluster,
                "rank": rank,
                "unit_id": ids[int(indices[local_index])],
                "text": row["text"],
                "review_ids": row["review_ids"],
                "source_languages": row.get("source_languages", []),
                "lexical_categories_diagnostic": row.get("lexical_categories_diagnostic", []),
                "cosine_to_center": float(similarities[local_index, cluster]),
            })
    return representatives


def profile_rows(units: list[dict], indices: np.ndarray, labels: np.ndarray) -> list[dict]:
    rows = []
    for cluster in sorted(set(int(x) for x in labels)):
        member_indices = indices[labels == cluster]
        categories = Counter()
        languages = Counter()
        reviews = set()
        for index in member_indices:
            row = units[int(index)]
            categories.update(row.get("lexical_categories_diagnostic", []))
            languages.update(row.get("source_languages", []))
            reviews.update(row.get("review_ids", []))
        rows.append({
            "cluster": cluster,
            "unit_count": int(len(member_indices)),
            "unit_share": float(len(member_indices) / len(indices)),
            "review_count": len(reviews),
            "source_languages": dict(languages),
            "lexical_categories_diagnostic": dict(categories),
        })
    return rows


def component_cosine(left: np.ndarray, right: np.ndarray) -> float:
    similarity = normalize(left) @ normalize(right).T
    row, column = linear_sum_assignment(-similarity)
    return float(similarity[row, column].mean())


def save_assignments(path: Path, units: list[dict], ids: list[str], labels: np.ndarray,
                     fit_indices: np.ndarray) -> None:
    fit_set = set(int(x) for x in fit_indices)
    rows = []
    for index, label in enumerate(labels):
        row = units[index]
        rows.append({
            "unit_id": ids[index],
            "cluster": int(label),
            "assignment_scope": "fit_candidate" if index in fit_set else "application_only",
            "fit_candidate": bool(row["fit_candidate"]),
            "review_ids": json.dumps(row["review_ids"], ensure_ascii=False),
            "source_languages": json.dumps(row.get("source_languages", []), ensure_ascii=False),
            "appids": json.dumps(row.get("appids", []), ensure_ascii=False),
            "text": row["text"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path, output: Path | None = None) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    destination = ROOT / (output or config["output"])
    if destination.exists():
        raise FileExistsError(f"Output already exists. Choose a new output directory: {destination}")
    destination.mkdir(parents=True)
    units, vectors, ids, fit_indices, provenance = read_inputs(config)
    fit_vectors = np.asarray(vectors[fit_indices])
    strata = strata_for(units, fit_indices)
    config_hash = common.sha256(config_path)
    started = time.perf_counter()
    candidates = []
    resampling_records = []
    all_candidate_models = {}

    for k in config["cluster_counts"]:
        for seed in config["seeds"]:
            name = f"sentence_k{k}_s{seed}"
            candidate_dir = destination / name
            candidate_dir.mkdir()
            with threadpool_limits(limits=config["cpu_threads"]):
                model = KMeans(n_clusters=k, random_state=seed, n_init=config["n_init"],
                               max_iter=config["max_iter"], algorithm="lloyd").fit(fit_vectors)
            all_candidate_models[name] = model
            all_labels = model.predict(np.asarray(vectors))
            np.savez_compressed(candidate_dir / "centers.npz", centers=model.cluster_centers_)
            save_assignments(candidate_dir / "assignments.csv", units, ids, all_labels, fit_indices)
            profiles = profile_rows(units, fit_indices, model.labels_)
            representatives = candidate_representatives(units, ids, vectors, fit_indices, model,
                                                         config["representatives_per_cluster"])
            write_json(candidate_dir / "cluster_profiles.json", profiles)
            write_json(candidate_dir / "representatives.json", representatives)
            candidate = {
                "candidate": name,
                "k": k,
                "seed": seed,
                "fit_units": len(fit_indices),
                "all_units_assigned": len(all_labels),
                "smallest_cluster": min(row["unit_count"] for row in profiles),
                "largest_cluster_share": max(row["unit_share"] for row in profiles),
                "inertia": float(model.inertia_),
                "assignment_file": common.relative(candidate_dir / "assignments.csv"),
                "profile_file": common.relative(candidate_dir / "cluster_profiles.json"),
                "representative_file": common.relative(candidate_dir / "representatives.json"),
            }
            candidates.append(candidate)
            for resample_seed in config["resample_seeds"]:
                selected = stability.stratified_subsample(strata, config["sample_fraction"], resample_seed)
                with threadpool_limits(limits=config["cpu_threads"]):
                    refit = KMeans(n_clusters=k, random_state=seed, n_init=config["n_init"],
                                   max_iter=config["max_iter"], algorithm="lloyd").fit(fit_vectors[selected])
                    proposed = refit.predict(fit_vectors)
                reference = model.labels_
                oob = np.ones(len(fit_indices), dtype=bool)
                oob[selected] = False
                metrics = stability.partition_metrics(reference, proposed, oob)
                resampling_records.append({
                    "candidate": name,
                    "scope": "fit_candidate_units",
                    "resample_seed": resample_seed,
                    "selected_units": int(len(selected)),
                    "population_units": int(len(fit_indices)),
                    "sample_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
                    "component_cosine": component_cosine(model.cluster_centers_, refit.cluster_centers_),
                    **metrics,
                })

    summary_by_candidate = []
    grouped = defaultdict(list)
    for record in resampling_records:
        grouped[record["candidate"]].append(record)
    for name, records in sorted(grouped.items()):
        summary_by_candidate.append({
            "candidate": name,
            "resamples": len(records),
            "ari_mean": float(np.mean([x["ari"] for x in records])),
            "ari_min": float(min(x["ari"] for x in records)),
            "ari_max": float(max(x["ari"] for x in records)),
            "out_of_bag_ari_mean": float(np.mean([x["out_of_bag_ari"] for x in records if x["out_of_bag_ari"] is not None])),
            "minimum_cluster_jaccard_min": float(min(x["minimum_cluster_jaccard"] for x in records)),
            "component_cosine_mean": float(np.mean([x["component_cosine"] for x in records])),
        })

    write_json(destination / "resampling_results.json", {
        "status": "complete",
        "config": config,
        "records": resampling_records,
        "summary": summary_by_candidate,
        "provenance": provenance,
    })
    result = {
        "status": "cluster_candidates_ready_for_semantic_review",
        "config": config,
        "config_sha256": config_hash,
        "provenance": provenance,
        "candidates": candidates,
        "resampling_summary": summary_by_candidate,
        "selection_uses_recommendation_labels": False,
        "topic_assignment_performed": False,
        "task12_topics_frozen": False,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Cluster numbers are candidate partitions and are not final topic IDs.",
            "Non-fit units receive application-only nearest-centroid assignments and do not define centroids.",
            "Diagnostic lexical categories are input metadata and do not assign final topics.",
            "Semantic review and topic-list-draft confirmation remain required before topics-v1.",
        ],
    }
    write_json(destination / "cluster_results.json", result)
    write_json(destination / "run_manifest.json", {
        "status": "complete",
        "config_sha256": config_hash,
        "provenance": provenance,
        "candidate_count": len(candidates),
        "resampling_record_count": len(resampling_records),
        "topic_assignment_performed": False,
        "task12_topics_frozen": False,
    })
    (destination / "README.md").write_text(
        "# Task12 sentence-level clustering\n\n"
        "This directory contains KMeans candidates on the delivered sentence-level BGE-M3 vectors. "
        "The fit population is the 7,269-unit `fit_candidate=true` subset. The remaining 1,710 units "
        "are assigned to nearest centroids for application-only inspection. Cluster IDs are not final topic IDs. "
        "Five 80% stratified resamples per candidate are reported in `resampling_results.json`. "
        "Semantic review against `topic-list-draft.json` is still required before freezing `topics-v1`.\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({"status": result["status"], "candidates": len(result["candidates"]),
                      "resampling_records": sum(x["resamples"] for x in result["resampling_summary"]),
                      "elapsed_seconds": round(result["elapsed_seconds"], 1)}, ensure_ascii=False), flush=True)

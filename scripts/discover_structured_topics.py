"""Discover candidate clusters in the structured Task 12 sentence corpus."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from threadpoolctl import threadpool_limits

import analysis_common as common
from assign_structured_topic_matches import jsonl, load_cache
import cluster_bge
from cluster_bge_sentences import fit, matched_clusters
from cluster_sentence_bge import write_json
from reject_generic_topic_matches import is_pure_generic


ROOT = common.ROOT
RELAXED = "output/topic_selection/task12-20261114/handoff-lexical-relaxed-20261125"
CACHE_CONFIG = ROOT / ".cache-topic/structured-20261125/merged_encoding_config.json"
DEFAULT_OUTPUT = ROOT / "output/topic_selection/topics-structured-candidate-20261125"
SETTINGS = {"cluster_counts": [8, 10, 12], "seeds": [17, 42], "n_init": 5, "max_iter": 300, "cpu_threads": 6}


def profile(model, vectors, units, scripts, structural):
    distances = model.transform(vectors)
    ranks = np.sort(distances, axis=1)[:, :2]
    margin = ranks[:, 1] - ranks[:, 0]
    rng = np.random.default_rng(20261125)
    profiles = []
    for cluster in range(model.n_clusters):
        members = np.flatnonzero(model.labels_ == cluster)
        closest = members[np.argsort(distances[members, cluster], kind="stable")]
        selections = [("random", rng.choice(members, min(12, len(members)), replace=False))]
        for script in ("Han-dominant", "Latin-script"):
            in_script = closest[scripts[closest] == script]
            selections.extend([
                (f"center_{script}", in_script[:5]),
                (f"new_structure_{script}", in_script[structural[in_script]][:5]),
            ])
        samples = {}
        for group, indices in selections:
            for i in indices:
                sample = samples.setdefault(units[i]["unit_id"], {
                    "unit_id": units[i]["unit_id"], "text": units[i]["text"],
                    "script_hint": scripts[i], "distance": float(distances[i, cluster]),
                    "margin": float(margin[i]), "sample_groups": [],
                })
                sample["sample_groups"].append(group)
        profiles.append({
            "cluster": cluster, "units": len(members),
            "reviews": len({rid for i in members for rid in units[i]["review_ids"]}),
            "script_hints": dict(Counter(scripts[members])),
            "source_languages": dict(Counter(lang for i in members for lang in units[i]["source_languages"])),
            "new_structural_units": int(structural[members].sum()),
            "no_lexical_hit": sum(not units[i]["lexical_categories_diagnostic"] for i in members),
            "lexical_categories_diagnostic": dict(Counter(tag for i in members for tag in units[i]["lexical_categories_diagnostic"])),
            "samples": list(samples.values()),
        })
    return profiles


def run(output):
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    config, all_units, manifest, cached = load_cache(CACHE_CONFIG)
    relaxed_path = ROOT / RELAXED / "sentence_units.jsonl"
    relaxed = {row["unit_id"] for row in jsonl(relaxed_path)}
    selected = np.flatnonzero([not is_pure_generic(row["text"]) for row in all_units])
    vectors = np.asarray(cached[selected])
    units = [all_units[i] for i in selected]
    scripts = np.array([cluster_bge.script_hint(row["text"]) for row in units])
    structural = np.array([row["unit_id"] not in relaxed for row in units])
    output.mkdir(parents=True)
    fits, stability = [], []
    with threadpool_limits(limits=SETTINGS["cpu_threads"]):
        for k in SETTINGS["cluster_counts"]:
            labels = {}
            for seed in SETTINGS["seeds"]:
                model = fit(vectors, k, seed, SETTINGS)
                labels[seed] = model.labels_
                name = f"k{k}_s{seed}"
                profiles = profile(model, vectors, units, scripts, structural)
                write_json(output / f"{name}_profiles.json", profiles)
                np.savez_compressed(output / f"{name}_model.npz", centers=model.cluster_centers_, labels=model.labels_)
                fits.append({"candidate": name, "k": k, "seed": seed, "iterations": model.n_iter_,
                             "script_ami": float(adjusted_mutual_info_score(scripts, model.labels_)),
                             "cluster_sizes": [p["units"] for p in profiles]})
                print(f"{name} complete", flush=True)
            s1, s2 = SETTINGS["seeds"]
            stability.append({"k": k, "ari": float(adjusted_rand_score(labels[s1], labels[s2])),
                              "matched_clusters": matched_clusters(labels[s1], labels[s2])})
    write_json(output / "fit_unit_ids.json", [unit["unit_id"] for unit in units])
    write_json(output / "discovery.json", {
        "status": "exploratory_cluster_candidates", "settings": SETTINGS,
        "provenance": {"input_units": config["input"], "input_sha256": manifest["input_sha256"],
                       "vectors_sha256": manifest["vectors_sha256"], "embedding_model": manifest["model_name"],
                       "embedding_revision": manifest["model_revision"]},
        "relaxed_input_sha256": common.sha256(relaxed_path), "script_sha256": common.sha256(Path(__file__)),
        "policy": "Lloyd KMeans on normalized cached BGE-M3 vectors. Exclude whole-fragment generic text. No lexical gate, old topic score, old allowlist or review labels used.",
        "units_fit": len(units), "units_excluded_by_generic_rule": len(all_units) - len(units),
        "reviews_with_fit_units": len({rid for row in units for rid in row["review_ids"]}),
        "fit_units_from_new_structure": int(structural.sum()),
        "fit_units_without_lexical_hit": sum(not row["lexical_categories_diagnostic"] for row in units),
        "candidates": fits, "seed_stability": stability,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)

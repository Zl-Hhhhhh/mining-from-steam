"""Compare KMeans partitions of the published Task 12 sentence embeddings."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import warnings

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

import numpy as np
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from threadpoolctl import threadpool_limits

import check_topic_stability as stability
import cluster_bge
import embedding_bge_sentences as embedding


ROOT = embedding.ROOT
DEFAULT_OUTPUT = ROOT / "output/topic_selection/task12-20261118-sentence-v1"
SETTINGS = ROOT / "config/topic_rework.json"
DRAFT = ROOT / "output/topic_selection/task12-20261114/topic-list-draft.json"
SCOPE = "sentence_fit_candidates"


def category(values):
    return str(values[0]) if len(values) == 1 else "multiple_or_unknown"


def fit(vectors, k, seed, settings):
    model = KMeans(
        n_clusters=k, random_state=seed, n_init=settings["n_init"],
        max_iter=settings["max_iter"], algorithm="lloyd",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(vectors)
    if len(np.unique(model.labels_)) != k:
        raise ValueError(f"K={k}, seed={seed} did not produce {k} clusters")
    for warning in caught:
        warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    return model


def matched_clusters(reference, proposed):
    left, right = np.unique(reference), np.unique(proposed)
    counts = contingency_matrix(reference, proposed)
    rows, cols = linear_sum_assignment(-counts)
    totals_a, totals_b = counts.sum(axis=1), counts.sum(axis=0)
    return {
        int(left[i]): {
            "cluster": int(right[j]),
            "jaccard": float(counts[i, j] / (totals_a[i] + totals_b[j] - counts[i, j])),
        }
        for i, j in zip(rows, cols)
    }


def describe_clusters(model, vectors, units, languages, games, scripts, sample_seed):
    distances = model.transform(vectors)
    nearest_two = np.sort(distances, axis=1)[:, :2]
    margins = nearest_two[:, 1] - nearest_two[:, 0]
    rng = np.random.default_rng(sample_seed)
    profiles = []
    for cluster in range(model.n_clusters):
        members = np.flatnonzero(model.labels_ == cluster)
        ordered = members[np.argsort(distances[members, cluster], kind="stable")]
        flags = Counter(
            aspect for i in members for aspect in units[i]["lexical_categories_diagnostic"]
        )
        samples = []
        for script in ("Han-dominant", "Latin-script"):
            samples.extend((f"center_{script}", i) for i in ordered[scripts[ordered] == script][:2])
        samples.extend(("random", i) for i in rng.choice(members, min(6, len(members)), replace=False))
        boundary = members[np.argsort(margins[members], kind="stable")[:2]]
        samples.extend(("boundary", i) for i in boundary)
        profiles.append({
            "cluster": cluster,
            "size": len(members),
            "source_languages": dict(Counter(languages[members])),
            "script_hints": dict(Counter(scripts[members])),
            "top_games": Counter(games[members]).most_common(3),
            "diagnostic_lexical_counts": dict(flags.most_common()),
            "samples": [
                {"kind": kind, "unit_id": units[i]["unit_id"], "text": units[i]["text"],
                 "appids": units[i]["appids"], "script_hint": scripts[i],
                 "center_distance": float(distances[i, cluster]), "assignment_margin": float(margins[i])}
                for kind, i in samples
            ],
        })
    return profiles


def run(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Use a new output directory: {destination}")
    started = time.perf_counter()
    embedding.verify()
    config, _, units, source = embedding.load_inputs()
    settings = embedding.read_json(SETTINGS)
    cache = ROOT / config["output"]
    fit_rows = np.flatnonzero([unit["fit_candidate"] for unit in units])
    units = [units[i] for i in fit_rows]
    vectors = np.load(cache / "sentence_vectors.npy", mmap_mode="r", allow_pickle=False)[fit_rows]
    languages = np.array([category(unit["source_languages"]) for unit in units])
    games = np.array([category(unit["appids"]) for unit in units])
    scripts = np.array([cluster_bge.script_hint(unit["text"]) for unit in units])
    strata = list(zip(languages, games))
    subsamples = {
        seed: stability.stratified_subsample(strata, settings["sample_fraction"], seed)
        for seed in settings["resample_seeds"]
    }
    candidates, records, seed_pairs, assignments = [], [], [], {}
    with threadpool_limits(limits=settings["cpu_threads"]):
        for k in settings["cluster_counts"]:
            references = {}
            for seed in settings["seeds"]:
                name = f"k{k}_s{seed}"
                full = fit(vectors, k, seed, settings)
                labels = full.labels_
                references[seed] = labels
                assignments[name] = labels.tolist()
                profiles = describe_clusters(full, vectors, units, languages, games, scripts, 20261118)
                for sample_seed, selected in subsamples.items():
                    sampled = fit(vectors[selected], k, seed, settings)
                    predicted = sampled.predict(vectors)
                    held_out = np.ones(len(vectors), dtype=bool)
                    held_out[selected] = False
                    records.append({
                        "candidate": name, "k": k, "model_seed": seed, "resample_seed": sample_seed,
                        "scope": SCOPE, "sample_size": len(selected), "n_iter": sampled.n_iter_,
                        "converged": sampled.n_iter_ < settings["max_iter"],
                        **stability.partition_metrics(labels, predicted, held_out),
                        "matched_clusters": matched_clusters(labels, predicted),
                    })
                candidates.append({
                    "candidate": name, "k": k, "seed": seed, "inertia": float(full.inertia_),
                    "n_iter": full.n_iter_, "converged": full.n_iter_ < settings["max_iter"],
                    "source_language_ami": float(adjusted_mutual_info_score(languages, labels)),
                    "script_hint_ami": float(adjusted_mutual_info_score(scripts, labels)),
                    "game_ami": float(adjusted_mutual_info_score(games, labels)),
                    "clusters": profiles,
                })
                print(f"{name}: 1 full fit + {len(subsamples)} resamples complete", flush=True)
            first, second = settings["seeds"]
            seed_pairs.append({
                "k": k, "reference_seed": first, "compared_seed": second,
                **stability.partition_metrics(references[first], references[second]),
                "matched_clusters": matched_clusters(references[first], references[second]),
            })
    sources = [
        Path(__file__), ROOT / "scripts/check_topic_stability.py", ROOT / "scripts/cluster_bge.py",
        ROOT / "scripts/embedding_bge_sentences.py", ROOT / embedding.DEFAULT_CONFIG,
        ROOT / config["input"], cache / "sentence_embedding_manifest.json",
        cache / "sentence_vectors.npy", cache / "unit_ids.json", SETTINGS, DRAFT,
    ]
    destination.mkdir(parents=True)
    stability.write(destination / "cluster_profiles.json", candidates)
    (destination / "fit_assignments.json").write_text(
        json.dumps({"unit_ids": [unit["unit_id"] for unit in units], "clusters": assignments},
                   ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8", newline="\n",
    )
    stability.write(destination / "stability.json", {
        "status": "exploratory_not_frozen",
        "algorithm": "Lloyd KMeans on unmodified L2-normalized dense embeddings, Euclidean distance",
        "parameters": {key: settings[key] for key in (
            "cluster_counts", "seeds", "resample_seeds", "sample_fraction", "cpu_threads", "n_init", "max_iter",
        )},
        "strata": "source language and appid, with a separate category for ambiguous multi-parent fields",
        "held_out_scope": "deduplicated units left out of refits; parent reviews may overlap",
        "source": source,
        "source_language_counts": dict(Counter(languages)),
        "script_hint_counts": dict(Counter(scripts)),
        "strata_count": len(set(strata)),
        "inputs_sha256": {path.resolve().relative_to(ROOT).as_posix(): embedding.sha_file(path) for path in sources},
        "label_inputs_used": False,
        "runtime": {"numpy": np.__version__, "scikit-learn": sklearn.__version__,
                    "seconds": round(time.perf_counter() - started, 2)},
        "seed_comparison": seed_pairs,
        "resample_summary": stability.summarize(records),
        "resample_records": records,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)

"""Filter explicit-content texts and recluster cached vectors, without label training."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

import analysis_common as common
import cluster_bge
import check_topic_stability as stability

ROOT = common.ROOT


def lexical_flags(text, config):
    matches = {key: sorted(set(m.group(0).lower() for m in re.finditer(pattern, text, re.I)))
               for key, pattern in config["patterns"].items()}
    matches = {key: value for key, value in matches.items() if value}
    template = bool(re.search(r"[☐☑☒□✓✔]|\[\s?[xX ]\s?\]", text))
    return {"matches": matches, "template": template,
            "long_or_many_aspects": len(text) > config["maximum_core_characters"] or len(matches) > config["maximum_core_lexical_categories"]}


def run(config, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Use a new unsupervised rework directory")
    destination.mkdir(parents=True)
    watched = {}
    def watch(path):
        path = Path(path); watched[path.relative_to(ROOT).as_posix()] = common.sha256(path); return path
    for name in ("config/topic_rework.json", "scripts/rework_unsupervised_topics.py", "scripts/check_topic_stability.py"):
        watch(ROOT / name)
    manifest = common.read_json(watch(ROOT / config["embedding_manifest"]))
    input_path = watch(ROOT / config["inputs"])
    if common.sha256(input_path) != manifest["source_input_sha256"]:
        raise ValueError("Cached embedding texts changed")
    source = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    source = [row for row in source if row["text_clean"].strip()]
    vectors_path = watch(ROOT / manifest["vector_file"])
    if common.sha256(vectors_path) != manifest["vectors_sha256"]:
        raise ValueError("Embedding vectors changed")
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    ids = common.read_json(watch(ROOT / manifest["cache_path"] / "review_ids.json"))
    cluster_bge.validate_alignment(source, ids, vectors)
    flags = [lexical_flags(row["text_clean"], config) for row in source]
    canonical, seen = [], set()
    for i in sorted(range(len(source)), key=lambda j: ids[j]):
        text = re.sub(r"\s+", " ", source[i]["text_clean"]).strip()
        f = flags[i]
        if f["matches"] and not f["template"] and not f["long_or_many_aspects"] and not source[i]["nonlexical"] and text not in seen:
            canonical.append(i); seen.add(text)
    core = np.array(sorted(canonical), dtype=int)
    X = np.asarray(vectors[core])
    strata = [source[i]["language"] + ":" + str(source[i]["appid"]) for i in core]
    stability.write(destination / "config.json", config)
    records, candidates, models = [], [], {}
    profiles, representatives = [], []
    started = time.perf_counter()
    with threadpool_limits(limits=config["cpu_threads"]):
        for k in config["cluster_counts"]:
            for seed in config["seeds"]:
                t = time.perf_counter()
                name = f"filtered_kmeans_k{k}_s{seed}"
                model = KMeans(n_clusters=k, random_state=seed, n_init=config["n_init"], max_iter=config["max_iter"], algorithm="lloyd").fit(X)
                models[k, seed] = model
                unit = normalize(model.cluster_centers_)
                similarity = X @ unit.T
                counts = Counter(model.labels_.tolist())
                candidates.append({"candidate": name, "k": k, "seed": seed, "fit_rows": len(core),
                    "smallest_cluster": min(counts.values()), "largest_cluster_share": max(counts.values()) / len(core),
                    "source_language_ami": float(adjusted_mutual_info_score([source[i]["language"] for i in core], model.labels_)),
                    "seconds": time.perf_counter() - t})
                folder = destination / name; folder.mkdir()
                np.savez_compressed(folder / "centers.npz", raw_centers=model.cluster_centers_, unit_centers=unit)
                stability.write(folder / "fit_assignments.json", {ids[i]: int(label) for i, label in zip(core, model.labels_)})
                rng = np.random.default_rng(20261114)
                for c in range(k):
                    members = np.flatnonzero(model.labels_ == c)
                    langs = Counter(source[core[j]]["language"] for j in members)
                    categories = Counter(key for j in members for key in flags[core[j]]["matches"])
                    profiles.append({"candidate": name, "cluster": c, "count": len(members), "source_languages": dict(langs),
                                     "lexical_category_counts_diagnostic": dict(categories)})
                    selected = []
                    for language in ("english", "schinese"):
                        ordered = sorted((j for j in members if source[core[j]]["language"] == language), key=lambda j: -similarity[j, c])
                        selected += [("center_" + language, j) for j in ordered[:config["representatives_per_language"]]]
                    selected += [("random", int(j)) for j in rng.choice(members, min(len(members), config["random_representatives"]), replace=False)]
                    for kind, j in selected:
                        i = core[j]
                        representatives.append({"candidate": name, "cluster": c, "kind": kind, "recommendationid": ids[i],
                            "source_language": source[i]["language"], "text": source[i]["text_clean"],
                            "cosine": float(similarity[j, c]), "lexical_categories": list(flags[i]["matches"])})
                print(json.dumps(candidates[-1]), flush=True)
            model = models[k, config["seeds"][0]]
            for seed in config["resample_seeds"]:
                t = time.perf_counter()
                selected = stability.stratified_subsample(strata, config["sample_fraction"], seed)
                refit = KMeans(n_clusters=k, random_state=config["seeds"][0], n_init=config["n_init"], max_iter=config["max_iter"], algorithm="lloyd").fit(X[selected])
                labels = refit.predict(X)
                oob = np.ones(len(core), dtype=bool); oob[selected] = False
                sim = normalize(model.cluster_centers_) @ normalize(refit.cluster_centers_).T
                a, b = linear_sum_assignment(-sim)
                record = {"candidate": f"filtered_kmeans_k{k}_s{config['seeds'][0]}", "scope": "filtered_core",
                    "resample_seed": seed, "component_cosine": float(sim[a, b].mean()),
                    **stability.partition_metrics(model.labels_, labels, oob), "seconds": time.perf_counter() - t}
                records.append(record); print(json.dumps({"candidate": record["candidate"], "resample": seed, "ari": record["ari"]}), flush=True)
                stability.write(destination / "resampling_results.json", {"status": "running", "records": records})
    result = {"status": "candidates_ready_for_semantic_review", "model_type": "unsupervised_kmeans_on_cached_bge_vectors",
        "config": config, "core_rows": len(core), "vector_rows": len(source),
        "source_language_counts": dict(Counter(source[i]["language"] for i in core)),
        "input_exclusions": {"no_lexical_evidence": sum(not f["matches"] for f in flags),
            "templates": sum(f["template"] for f in flags), "long_or_many_aspects": sum(f["long_or_many_aspects"] for f in flags)},
        "candidates": candidates, "profiles": profiles, "representatives": representatives,
        "seed_stability": [{"k": k, "ari": float(adjusted_rand_score(models[k,17].labels_, models[k,42].labels_))} for k in config["cluster_counts"]],
        "resampling_summary": stability.summarize(records), "sources": watched,
        "human_labels_used_for_training": False, "recommendation_labels_used": False,
        "encoder_loaded": False, "new_embeddings": 0, "task12_topics_frozen": False,
        "limitations": ["Lexical screening defines a restricted fitting scope; it can miss implicit aspects and cannot establish their absence.",
            "Long/multi-aspect/template reviews are withheld from centroid fitting and require separate application checks.",
            "Source-language fields are not new actual-language annotations. Cluster numbers are not shared aspect labels."]}
    if any(common.sha256(ROOT / path) != digest for path, digest in watched.items()):
        raise RuntimeError("An input changed")
    stability.write(destination / "rework_results.json", result)
    stability.write(destination / "resampling_results.json", {"status": "complete", "records": records, "summary": stability.summarize(records)})
    print(json.dumps({"complete": True, "core_rows": len(core), "seconds": time.perf_counter() - started}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/topic_rework.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = common.read_json(args.config)
    run(config, args.output or ROOT / config["output"])

"""Task12: bounded, sequential resampling of archived unsupervised candidates."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import csv
import hashlib
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

import analysis_common as common
import cluster_bge
import nmf_baseline

ROOT = common.ROOT


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def stratified_subsample(strata, fraction, seed):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, key in enumerate(strata):
        groups[str(key)].append(i)
    chosen = []
    for key in sorted(groups):
        group = groups[key]
        n = min(len(group), max(1, int(np.floor(len(group) * fraction))))
        chosen.extend(rng.choice(group, n, replace=False).tolist())
    return np.array(sorted(chosen), dtype=int)


def partition_metrics(reference, proposed, out_of_bag=None):
    reference, proposed = np.asarray(reference), np.asarray(proposed)
    valid = (reference >= 0) & (proposed >= 0)
    a, b = reference[valid], proposed[valid]
    if not len(a):
        return {"n": 0, "excluded_zero_or_outside": int(len(reference)), "ari": None,
                "matched_accuracy": None, "macro_cluster_jaccard": None, "minimum_cluster_jaccard": None,
                "out_of_bag_n": 0, "out_of_bag_ari": None}
    left, right = np.unique(a), np.unique(b)
    counts = np.array([[np.sum((a == x) & (b == y)) for y in right] for x in left])
    li, ri = linear_sum_assignment(-counts)
    jaccards = []
    for x in range(len(left)):
        found = np.flatnonzero(li == x)
        if not len(found):
            jaccards.append(0.)
        else:
            y = ri[found[0]]
            intersection = counts[x, y]
            jaccards.append(float(intersection / (np.sum(a == left[x]) + np.sum(b == right[y]) - intersection)))
    oob = np.zeros(len(reference), dtype=bool) if out_of_bag is None else np.asarray(out_of_bag, dtype=bool)
    ov = valid & oob
    return {"n": int(valid.sum()), "excluded_zero_or_outside": int((~valid).sum()),
            "ari": float(adjusted_rand_score(a, b)), "matched_accuracy": float(counts[li, ri].sum() / len(a)),
            "macro_cluster_jaccard": float(np.mean(jaccards)), "minimum_cluster_jaccard": float(min(jaccards)),
            "out_of_bag_n": int(ov.sum()),
            "out_of_bag_ari": float(adjusted_rand_score(reference[ov], proposed[ov])) if ov.sum() >= 2 else None}


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["candidate"]].append(row)
    result = []
    for key, group in sorted(groups.items()):
        item = {"candidate": key, "resamples": len(group), "scope": group[0]["scope"]}
        for field in ("ari", "out_of_bag_ari", "macro_cluster_jaccard", "minimum_cluster_jaccard", "component_cosine"):
            values = [r[field] for r in group if r.get(field) is not None]
            item[field] = {"mean": float(np.mean(values)), "minimum": float(min(values)), "maximum": float(max(values))} if values else None
        item["nonconverged_runs"] = sum(not r.get("converged", True) for r in group)
        result.append(item)
    return result


def run(config, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Use a new stability output directory")
    destination.mkdir(parents=True)
    watched = {}

    def watch(path):
        path = Path(path)
        watched[path.relative_to(ROOT).as_posix()] = common.sha256(path)
        return path

    watch(Path(__file__))
    for source in ("scripts/nmf_baseline.py", "scripts/analysis_common.py", "scripts/cluster_bge.py", "config/topic_stability.json"):
        watch(ROOT / source)
    nmf_config = common.read_json(watch(ROOT / config["nmf_config"]))
    bge_config = common.read_json(watch(ROOT / config["bge_config"]))
    started = time.perf_counter()
    records = []

    def save(record):
        records.append(record)
        write(destination / "resampling_results.json", {"status": "running", "config": config, "records": records, "summary": summarize(records)})
        print(json.dumps({"candidate": record["candidate"], "resample": record["resample_seed"],
                          "ari": round(record["ari"], 4), "seconds": round(record["seconds"], 2)}), flush=True)

    with threadpool_limits(limits=config["cpu_threads"]):
        frames, validation = common.load_inputs(ROOT / nmf_config["cleaning_manifest"])
        for source in validation["sources"]:
            watch(ROOT / source["path"])
        watch(ROOT / nmf_config["metadata"])
        meta = pd.read_csv(ROOT / nmf_config["metadata"], keep_default_na=False)
        base = ROOT / config["nmf_root"]
        for language in nmf_config["languages"]:
            frame = frames["topic_fit"].query("language == @language").sort_values("recommendationid").reset_index(drop=True)
            order = rows(watch(base / language / "fit_review_order.csv"))
            if [r["recommendationid"] for r in order] != frame.recommendationid.tolist():
                raise ValueError("NMF fit row order changed")
            tokenizer = nmf_baseline.ReviewTokenizer(language, nmf_config, meta.game.tolist())
            documents = np.array([" ".join(tokenizer(text)) for text in frame.review_clean], dtype=object)
            vectorizer = joblib.load(watch(base / language / "vectorizer.joblib"))
            words = vectorizer.get_feature_names_out()
            strata = frame.appid.astype(str).tolist()
            for k in config["topic_counts"]:
                name = f"{language}_k{k}_s{config['anchor_seed']}"
                baseline = joblib.load(watch(base / name / "nmf.joblib"))
                with np.load(watch(base / name / "weights.npz"), allow_pickle=False) as weights:
                    reference = weights["fit_weights"].argmax(axis=1)
                    reference[weights["fit_weights"].sum(axis=1) == 0] = -1
                for seed in config["resample_seeds"]:
                    t = time.perf_counter()
                    selected = stratified_subsample(strata, config["sample_fraction"], seed)
                    vf = clone(vectorizer)
                    xm = vf.fit_transform(documents[selected])
                    all_x = vf.transform(documents)
                    model = NMF(n_components=k, init="random", random_state=config["anchor_seed"],
                                max_iter=nmf_config["max_iter"], tol=nmf_config["tol"], solver="cd")
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(xm)
                        w = model.transform(all_x)
                    labels = common.normalize_weights(w, model.components_).argmax(axis=1)
                    labels[(all_x.getnnz(axis=1) == 0) | (w.sum(axis=1) == 0)] = -1
                    _, a, b = np.intersect1d(words, vf.get_feature_names_out(), return_indices=True)
                    similarity = normalize(baseline.components_[:, a]) @ normalize(model.components_[:, b]).T
                    li, ri = linear_sum_assignment(-similarity)
                    oob = np.ones(len(frame), dtype=bool); oob[selected] = False
                    save({"candidate": "nmf_aspect_v2/" + name, "scope": language + "_all_fit_rows",
                          "resample_seed": seed, "selected_rows": len(selected), "population_rows": len(frame),
                          "sample_indices_sha256": hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),
                          "vocabulary_features": xm.shape[1], "shared_features": len(a), "refit_vocabulary_idf": True,
                          "component_cosine": float(similarity[li, ri].mean()),
                          "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                          **partition_metrics(reference, labels, oob), "seconds": time.perf_counter() - t})
        del frames, frame, documents, xm, all_x

        manifest = common.read_json(watch(ROOT / bge_config["embedding_summary"]))
        vectors_path = watch(ROOT / manifest["vector_file"])
        if common.sha256(vectors_path) != bge_config["vectors_sha256"]:
            raise ValueError("BGE cache checksum mismatch")
        input_path = watch(ROOT / bge_config["inputs_directory"] / "model_reviews.jsonl")
        if common.sha256(input_path) != manifest["source_input_sha256"]:
            raise ValueError("BGE input checksum mismatch")
        source = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
        source = [row for row in source if row["text_clean"].strip()]
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
        ids = common.read_json(watch(ROOT / manifest["cache_path"] / "review_ids.json"))
        cluster_bge.validate_alignment(source, ids, vectors)
        full = cluster_bge.fit_indices(source)
        bge_base = ROOT / config["bge_root"]
        common_ids = common.read_json(watch(bge_base / "common_review_order.json"))
        lookup = {row["recommendationid"]: i for i, row in enumerate(source)}
        common_idx = np.array([lookup[i] for i in common_ids])
        for scope, indices, algorithms in (("full_fit", full, ("kmeans",)),
                                           ("common5000", common_idx, ("kmeans", "ward", "average"))):
            X = np.asarray(vectors[indices])
            strata = [source[i]["language"] + ":" + str(source[i]["appid"]) for i in indices]
            reference = {}
            centers = {}
            for algorithm in algorithms:
                for k in config["topic_counts"]:
                    name = (f"full_kmeans_k{k}_s{config['anchor_seed']}" if scope == "full_fit" else
                            f"common_kmeans_k{k}_s{config['anchor_seed']}" if algorithm == "kmeans" else f"{algorithm}_k{k}")
                    assign = {r["recommendationid"]: int(r["raw_cluster"]) for r in rows(watch(bge_base / name / "assignments.csv"))}
                    reference[algorithm, k] = np.array([assign[ids[i]] for i in indices])
                    if algorithm == "kmeans":
                        with np.load(watch(bge_base / name / "centers.npz"), allow_pickle=False) as z:
                            centers[k] = z["raw_centers"]
            for seed in config["resample_seeds"]:
                selected = stratified_subsample(strata, config["sample_fraction"], seed)
                oob = np.ones(len(indices), dtype=bool); oob[selected] = False
                for algorithm in algorithms:
                    t = time.perf_counter()
                    cuts = None
                    if algorithm != "kmeans":
                        tree = linkage(pdist(X[selected], metric="euclidean" if algorithm == "ward" else "cosine"),
                                       method=algorithm, optimal_ordering=False)
                        cuts = cut_tree(tree, n_clusters=config["topic_counts"])
                    for column, k in enumerate(config["topic_counts"]):
                        kt = time.perf_counter()
                        if algorithm == "kmeans":
                            model = KMeans(n_clusters=k, random_state=config["anchor_seed"], n_init=bge_config["n_init"],
                                           max_iter=bge_config["max_iter"], algorithm="lloyd").fit(X[selected])
                            labels = model.predict(X)
                            similarity = normalize(centers[k]) @ normalize(model.cluster_centers_).T
                            li, ri = linear_sum_assignment(-similarity)
                            score = partition_metrics(reference[algorithm, k], labels, oob)
                            cosine = float(similarity[li, ri].mean())
                            evaluation = "fixed_original_fit_population_with_out_of_bag_subset"
                        else:
                            score = partition_metrics(reference[algorithm, k][selected], cuts[:, column])
                            cosine = None
                            evaluation = "overlap_only_no_out_of_sample_prediction"
                        name = (f"full_kmeans_k{k}_s{config['anchor_seed']}" if scope == "full_fit" else
                                f"common_kmeans_k{k}_s{config['anchor_seed']}" if algorithm == "kmeans" else f"{algorithm}_k{k}")
                        save({"candidate": "bge/" + name, "scope": scope, "evaluation": evaluation,
                              "resample_seed": seed, "selected_rows": len(selected), "population_rows": len(indices),
                              "sample_indices_sha256": hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),
                              "component_cosine": cosine, **score,
                              "seconds": time.perf_counter() - (kt if algorithm == "kmeans" else t)})
    if any(common.sha256(ROOT / path) != digest for path, digest in watched.items()):
        raise RuntimeError("An archived input changed")
    result = {"status": "complete", "config": config, "records": records, "summary": summarize(records),
              "seconds": time.perf_counter() - started, "sources": watched,
              "original_inputs_unchanged": True, "encoder_loaded": False, "new_embeddings": 0,
              "recommendation_labels_used_for_selection": False, "task12_topics_frozen": False,
              "limitations": ["Resampling stability does not establish semantic aspect coherence.",
                  "NMF refits vocabulary, IDF and components; BGE refits clusters on fixed cached representations.",
                  "KMeans/NMF compare a fixed original fit population; hierarchical methods compare only sampled overlap.",
                  "Five 80% subsamples are descriptive perturbations, not inferential confidence intervals."]}
    write(destination / "resampling_results.json", result)
    print(json.dumps({"complete": True, "records": len(records), "seconds": round(result['seconds'], 2)}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/topic_stability.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = common.read_json(args.config)
    run(cfg, args.output or ROOT / cfg["output"])

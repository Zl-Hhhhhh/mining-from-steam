"""Unsupervised NMF comparison on explicit-content, short, non-template texts."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import argparse
import json
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

import analysis_common as common
import check_topic_stability as stability
import nmf_baseline
from rework_unsupervised_topics import lexical_flags

ROOT = common.ROOT


def labels_for(weights, components, matrix):
    proportions = common.normalize_weights(weights, components)
    labels = proportions.argmax(axis=1)
    labels[(matrix.getnnz(axis=1) == 0) | (weights.sum(axis=1) == 0)] = -1
    return labels, proportions


def run(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new NMF rework directory")
    output.mkdir(parents=True)
    source_paths = [ROOT / p for p in ("config/topic_rework_nmf.json", config["gate_config"], config["tokenizer_config"],
        config["inputs"], config["metadata"], "scripts/rework_nmf_topics.py", "scripts/rework_unsupervised_topics.py", "scripts/nmf_baseline.py", "scripts/analysis_common.py", "scripts/check_topic_stability.py")]
    hashes = {p.relative_to(ROOT).as_posix(): common.sha256(p) for p in source_paths}
    gate = common.read_json(ROOT / config["gate_config"])
    token_config = common.read_json(ROOT / config["tokenizer_config"])
    source = [json.loads(line) for line in (ROOT / config["inputs"]).read_text(encoding="utf-8").splitlines()]
    selected, seen = [], set()
    for row in sorted(source, key=lambda x: x["recommendationid"]):
        flags = lexical_flags(row["text_clean"], gate)
        text = re.sub(r"\s+", " ", row["text_clean"]).strip()
        if flags["matches"] and not flags["template"] and not flags["long_or_many_aspects"] and not row["nonlexical"] and text not in seen:
            selected.append(row); seen.add(text)
    names = pd.read_csv(ROOT / config["metadata"], keep_default_na=False).game.tolist()
    profiles, representatives, records, seeds = [], [], [], []
    started = time.perf_counter()
    with threadpool_limits(limits=config["cpu_threads"]):
        for language in config["languages"]:
            data = [r for r in selected if r["language"] == language]
            tokenizer = nmf_baseline.ReviewTokenizer(language, token_config, names)
            documents = np.array([" ".join(tokenizer(r["text_clean"])) for r in data], dtype=object)
            vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, preprocessor=None, lowercase=False,
                min_df=config["min_df"], max_df=config["max_df"], max_features=config["max_features"], ngram_range=(1,2), sublinear_tf=True)
            X = vectorizer.fit_transform(documents)
            words = vectorizer.get_feature_names_out()
            lang_dir = output / language; lang_dir.mkdir()
            joblib.dump(vectorizer, lang_dir / "vectorizer.joblib")
            strata = [str(r["appid"]) for r in data]
            for k in config["topic_counts"]:
                fitted = {}
                for seed in config["seeds"]:
                    t = time.perf_counter()
                    model = NMF(n_components=k, init="random", random_state=seed, max_iter=config["max_iter"], tol=config["tol"])
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        w = model.fit_transform(X)
                    labels, proportions = labels_for(w, model.components_, X)
                    fitted[seed] = (model, labels)
                    name = f"content_nmf_{language}_k{k}_s{seed}"
                    folder = output / name; folder.mkdir()
                    joblib.dump(model, folder / "nmf.joblib")
                    np.savez_compressed(folder / "fit_weights.npz", proportions=proportions, ids=np.array([r["recommendationid"] for r in data]))
                    cosine = np.asarray(X @ normalize(model.components_).T)
                    rng = np.random.default_rng(20261114)
                    for c in range(k):
                        members = np.flatnonzero(labels == c)
                        terms = words[np.argsort(model.components_[c])[::-1][:12]].tolist()
                        profiles.append({"candidate": name, "language": language, "topic": c, "count": len(members),
                            "top_terms": terms, "zero_vector_rows": int((labels < 0).sum()),
                            "converged": not any(issubclass(x.category, ConvergenceWarning) for x in caught)})
                        center = sorted(members, key=lambda j: -cosine[j,c])[:4]
                        random = rng.choice(members, min(len(members),3), replace=False).tolist()
                        for kind, indices in (("center", center), ("random", random)):
                            for j in indices:
                                representatives.append({"candidate": name, "topic": c, "kind": kind,
                                    "recommendationid": data[j]["recommendationid"], "text": data[j]["text_clean"]})
                    print(json.dumps({"candidate": name,"fit_rows":len(data),"features":len(words),"seconds":time.perf_counter()-t}),flush=True)
                seeds.append({"language": language, "k": k, "ari": float(adjusted_rand_score(fitted[17][1],fitted[42][1]))})
                reference_model, reference = fitted[17]
                for seed in config["resample_seeds"]:
                    t = time.perf_counter()
                    sample = stability.stratified_subsample(strata, config["sample_fraction"], seed)
                    vf=clone(vectorizer); train=vf.fit_transform(documents[sample]); test=vf.transform(documents)
                    model=NMF(n_components=k,init="random",random_state=17,max_iter=config["max_iter"],tol=config["tol"])
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always",ConvergenceWarning)
                        model.fit(train); w=model.transform(test)
                    labels,_=labels_for(w,model.components_,test)
                    _,a,b=np.intersect1d(words,vf.get_feature_names_out(),return_indices=True)
                    similarity=normalize(reference_model.components_[:,a])@normalize(model.components_[:,b]).T
                    li,ri=linear_sum_assignment(-similarity)
                    oob=np.ones(len(data),bool);oob[sample]=False
                    record={"candidate":f"content_nmf_{language}_k{k}_s17","scope":language+"_content_core",
                        "resample_seed":seed,"component_cosine":float(similarity[li,ri].mean()),
                        "converged":not any(issubclass(x.category,ConvergenceWarning) for x in caught),
                        **stability.partition_metrics(reference,labels,oob),"seconds":time.perf_counter()-t}
                    records.append(record)
    if any(common.sha256(ROOT/path)!=digest for path,digest in hashes.items()):raise RuntimeError("Source changed")
    result={"config":config,"core_rows":len(selected),"profiles":profiles,"representatives":representatives,
        "seed_stability":seeds,"resampling_records":records,"resampling_summary":stability.summarize(records),
        "sources":hashes,"human_labels_used_for_training":False,"recommendation_labels_used":False,
        "task12_topics_frozen":False,"seconds":time.perf_counter()-started}
    stability.write(output/"rework_results.json",result)
    print(json.dumps({"complete":True,"seconds":result['seconds']}),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,default=ROOT/"config/topic_rework_nmf.json")
    parser.add_argument("--output",type=Path);args=parser.parse_args();config=common.read_json(args.config)
    run(config,args.output or ROOT/config["output"])

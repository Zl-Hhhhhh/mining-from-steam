"""Member B: reproducible, language-specific TF-IDF + NMF topic pilots.

No recommendation labels, votes, playtime or author IDs enter model fitting,
candidate diagnostics or model selection. Independent human review remains required.
"""
from __future__ import annotations

import argparse
import html
import logging
import re
import shutil
import unicodedata
import warnings
from itertools import combinations
from pathlib import Path

import jieba
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import save_npz
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from analysis_common import (ROOT, load_inputs, normalize_weights, read_json,
                               relative, save_json, sha256, write_csv)

NEGATIONS = {"no", "not", "never", "none", "nothing", "neither", "nor", "without", "cannot"}
CHINESE_NEGATIONS = {"不", "没", "无", "非", "没有", "不是", "不能", "不会", "不好"}
GENERIC_SENTIMENT = {"recommend", "recommended", "amazing", "awesome", "terrible", "boring",
                     "推荐", "好玩", "喜欢", "垃圾", "神作", "好评", "差评", "不好玩"}


class ReviewTokenizer:
    """Label-free tokenizer; game-name aliases are removed *before* segmentation."""

    def __init__(self, language, config, game_names):
        self.language = language
        self.english_stop = (set(ENGLISH_STOP_WORDS) | set(config["english_extra_stopwords"])) - NEGATIONS
        self.chinese_stop = set(config["chinese_stopwords"]) - CHINESE_NEGATIONS
        self.aliases = sorted(set(config["chinese_game_aliases"]), key=len, reverse=True)
        self.game_tokens = set()
        for name in game_names:
            self.game_tokens.update(re.findall(r"[a-z]+", str(name).lower()))
        # Do not mask English negation even if it appears in a game title.
        self.game_tokens -= NEGATIONS
        jieba.setLogLevel(logging.WARNING)

    def __call__(self, text):
        text = unicodedata.normalize("NFKC", html.unescape(str(text))).lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\[/?[a-z][^\]]*\]|<[^>]*>", " ", text)
        text = re.sub(r"\bcan['’]?t\b|\bcannot\b", "can not", text)
        text = re.sub(r"\bwon['’]?t\b", "will not", text)
        text = re.sub(r"\bdont\b", "do not", text)
        text = text.replace("n't", " not").replace("n’t", " not")
        for alias in self.aliases:
            text = text.replace(alias, " ")
        if self.language == "english":
            tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
        else:
            # Segment even mixed-language texts rather than treating all CJK as one token.
            tokens = list(jieba.cut(text, HMM=False))
        result = []
        for token in tokens:
            token = token.strip()
            if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", token):
                if len(token) >= 2 and token not in self.english_stop and token not in self.game_tokens:
                    result.append(token)
            elif self.language == "schinese" and re.fullmatch(r"[\u3400-\u9fff]+", token):
                if (len(token) >= 2 or token in CHINESE_NEGATIONS) and token not in self.chinese_stop:
                    result.append(token)
        return result


def get_assignments(weights, matrix, components, input_frame, config):
    proportions = normalize_weights(weights, components)
    top = proportions.argmax(axis=1)
    share = proportions.max(axis=1)
    # TF-IDF rows are unit length; compare to unit NMF components.
    similarities = np.asarray(matrix @ normalize(components).T)
    cosine = similarities[np.arange(len(top)), top]
    retained = matrix.getnnz(axis=1)
    # Zero vectors have no evidence for *any* topic; argmax(zeros) is not topic 0.
    top[retained == 0] = -1
    clear = ((share >= config["minimum_top_share"]) &
             (cosine >= config["minimum_cosine"]) &
             (retained >= config["minimum_retained_terms"]) &
             ~input_frame.is_low_information.to_numpy(dtype=bool))
    return proportions, top, share, cosine, retained, clear


def topic_diagnostics(matrix, components, proportions, labels, frame, words, count):
    terms, stats = [], []
    binary = (matrix > 0).astype(np.float64).tocsc()
    n_docs = matrix.shape[0]
    doc_freq = np.asarray(binary.sum(axis=0)).ravel()
    for topic in range(components.shape[0]):
        indexes = np.argsort(components[topic])[::-1][:count]
        top_words = words[indexes]
        for rank, index in enumerate(indexes, start=1):
            terms.append({"topic_index": topic, "rank": rank, "term": words[index],
                          "weight": float(components[topic, index]), "document_frequency": int(doc_freq[index])})
        npmi = []
        for a, b in combinations(indexes[:10], 2):
            joint = float(binary[:, a].multiply(binary[:, b]).sum()) / n_docs
            if joint <= 0:
                npmi.append(-1.0)
            elif joint < 1:
                npmi.append(float(np.log(joint / ((doc_freq[a] / n_docs) * (doc_freq[b] / n_docs))) / -np.log(joint)))
        members = frame.iloc[np.flatnonzero(labels == topic)]
        game_share = float(members.appid.value_counts(normalize=True).iloc[0]) if len(members) else 0.0
        sentiment_fraction = sum(w in GENERIC_SENTIMENT for w in top_words[:10]) / min(10, len(top_words))
        flags = []
        if len(members) < max(30, n_docs * 0.01):
            flags.append("tiny_topic")
        if game_share > 0.5:
            flags.append("single_game_dominance_check")
        if sentiment_fraction >= 0.2:
            flags.append("generic_sentiment_check")
        stats.append({"topic_index": topic, "top_terms": " | ".join(top_words),
                      "hard_assignment_count": len(members), "mean_weight": float(proportions[:, topic].mean()),
                      "games": int(members.appid.nunique()), "largest_game_share": game_share,
                      "top10_npmi_train": float(np.mean(npmi)) if npmi else 0.0,
                      "generic_sentiment_fraction": sentiment_fraction,
                      "flags": ";".join(flags), "human_interpretation_status": "pending"})
    return terms, stats


def aligned_stability(left_components, right_components, left_labels, right_labels, words):
    similarities = normalize(left_components) @ normalize(right_components).T
    a, b = linear_sum_assignment(-similarities)
    jaccards = []
    for x, y in zip(a, b):
        first = set(words[np.argsort(left_components[x])[-10:]])
        second = set(words[np.argsort(right_components[y])[-10:]])
        jaccards.append(len(first & second) / len(first | second))
    return {"adjusted_rand_index": float(adjusted_rand_score(left_labels, right_labels)),
            "matched_component_cosine": float(similarities[a, b].mean()),
            "matched_top10_jaccard": float(np.mean(jaccards))}


def sample_for_review(frame, assignment, config):
    """Labels are used ONLY to balance manual evaluation, never to fit/select topics."""
    merged = frame.merge(assignment, on=["recommendationid", "appid", "language"], validate="one_to_one")
    rng = np.random.default_rng(config["review_sample_seed"])
    picked = {}

    def add(indices, reason):
        for idx in indices:
            if len(picked) >= config["review_sample_size"]:
                return
            key = merged.at[idx, "recommendationid"]
            if key not in picked:
                picked[key] = (idx, reason)

    for (_, _), group in merged[merged.clear_theme].groupby(["language", "topic_index"]):
        add(group.sort_values("cosine", ascending=False).head(3).index, "central")
        remaining = group[~group.recommendationid.isin(picked)]
        if len(remaining):
            add(rng.choice(remaining.index, min(4, len(remaining)), replace=False), "random_within_topic")
    for lang in config["languages"]:
        edge = merged[(merged.language == lang) & (~merged.clear_theme | merged.is_short_text)]
        edge = edge[~edge.recommendationid.isin(picked)]
        if len(edge):
            add(rng.choice(edge.index, min(15, len(edge)), replace=False), "short_or_unclear")
    # Fill across language and recommendation strata without showing those labels to raters.
    while len(picked) < config["review_sample_size"]:
        added = False
        for lang in config["languages"]:
            for vote in (False, True):
                pool = merged[(merged.language == lang) & (merged.voted_up == vote) & ~merged.recommendationid.isin(picked)]
                if len(pool) and len(picked) < config["review_sample_size"]:
                    add([rng.choice(pool.index)], "stratified_fill")
                    added = True
        if not added:
            break
    selected = list(picked.values())
    rng.shuffle(selected)
    blind, key = [], []
    for number, (idx, reason) in enumerate(selected, start=1):
        row = merged.loc[idx]
        review_id = f"R{number:03d}"
        blind.append({"review_id": review_id, "language": row.language, "review_text": row.review_clean,
                      "theme_label": "", "aspect_summary": "", "fit_rating_0_1_2": "",
                      "no_clear_theme": "", "notes": ""})
        key.append({"review_id": review_id, "recommendationid": row.recommendationid, "appid": int(row.appid),
                    "language": row.language, "candidate_theme": row.topic_id,
                    "clear_theme": bool(row.clear_theme), "voted_up": bool(row.voted_up),
                    "reason": reason, "cosine": float(row.cosine)})
    return pd.DataFrame(blind), pd.DataFrame(key)


def run(config_path):
    config = read_json(config_path)
    output = ROOT / config["output"]
    # Existing model outputs are never silently overwritten.
    if (output / "run_manifest.json").exists():
        raise ValueError(f"Output already contains a completed run; choose another config output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source_code"
    source_dir.mkdir(exist_ok=True)
    for source in (Path(__file__), Path(__file__).with_name("analysis_common.py"), Path(config_path)):
        shutil.copyfile(source, source_dir / source.name)
    frames, audit = load_inputs(ROOT / config["cleaning_manifest"])
    save_json(output / "input_validation.json", audit)
    meta = pd.read_csv(ROOT / config["metadata"], keep_default_na=False)
    all_comparisons, stability, all_topic_stats, pilot_assignments = [], [], [], []
    for language in config["languages"]:
        fit = frames["topic_fit"].query("language == @language").sort_values("recommendationid").reset_index(drop=True)
        association = frames["association_data"].query("language == @language").sort_values("recommendationid").reset_index(drop=True)
        tokenizer = ReviewTokenizer(language, config, meta.game.tolist())
        documents = [" ".join(tokenizer(t)) for t in fit.review_clean]
        association_documents = [" ".join(tokenizer(t)) for t in association.review_clean]
        vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, preprocessor=None,
                                     lowercase=False, min_df=config["min_df"], max_df=config["max_df"],
                                     max_features=config["max_features"], ngram_range=tuple(config["ngram_range"]),
                                     sublinear_tf=True, dtype=np.float64)
        matrix = vectorizer.fit_transform(documents)
        association_matrix = vectorizer.transform(association_documents)
        words = vectorizer.get_feature_names_out()
        lang_dir = output / language
        lang_dir.mkdir(exist_ok=True)
        joblib.dump(vectorizer, lang_dir / "vectorizer.joblib")
        save_npz(lang_dir / "tfidf_fit.npz", matrix)
        write_csv(lang_dir / "fit_review_order.csv", fit[["recommendationid", "appid", "language"]])
        lexical_audit = [{"recommendationid": r.recommendationid, "retained_tokens": len(doc.split()),
                          "retained_features": int(matrix[i].getnnz())}
                         for i, (r, doc) in enumerate(zip(fit.itertuples(), documents))]
        write_csv(lang_dir / "tokenization_audit.csv", lexical_audit)
        for k in config["topic_counts"]:
            models = []
            for seed in config["seeds"]:
                candidate_id = f"{language}_k{k}_s{seed}"
                destination = output / candidate_id
                destination.mkdir(exist_ok=True)
                model = NMF(n_components=k, init="random", random_state=seed,
                            max_iter=config["max_iter"], tol=config["tol"], solver="cd")
                with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
                    warnings.simplefilter("always", ConvergenceWarning)
                    weights = model.fit_transform(matrix)
                    test_weights = model.transform(association_matrix)
                proportions, labels, share, cosine, retained, clear = get_assignments(
                    weights, matrix, model.components_, fit, config)
                ap, al, ash, ac, ar, acl = get_assignments(
                    test_weights, association_matrix, model.components_, association, config)
                terms, stats = topic_diagnostics(matrix, model.components_, proportions, labels, fit, words, config["top_words"])
                for row in terms + stats:
                    row.update({"candidate_id": candidate_id, "language": language,
                                "topic_id": f"{language}:T{row['topic_index']:02d}"})
                write_csv(destination / "topic_terms.csv", terms)
                write_csv(destination / "topic_summary.csv", stats)
                all_topic_stats.extend(stats)
                representatives = []
                for topic in range(k):
                    indexes = np.flatnonzero(labels == topic)
                    indexes = indexes[np.argsort(cosine[indexes])[::-1]][:config["representatives_per_topic"]]
                    for rank, index in enumerate(indexes, start=1):
                        row = fit.iloc[index]
                        representatives.append({"candidate_id": candidate_id, "topic_id": f"{language}:T{topic:02d}",
                                                "rank": rank, "recommendationid": row.recommendationid,
                                                "appid": int(row.appid), "game": row.game, "language": language,
                                                "top_share": float(share[index]), "cosine": float(cosine[index]),
                                                "clear_theme": bool(clear[index]), "review_text": row.review_clean})
                write_csv(destination / "representative_reviews.csv", representatives)
                assignment = association[["recommendationid", "appid", "language"]].copy()
                assignment["topic_index"] = al
                assignment["topic_id"] = [f"{language}:T{x:02d}" if good else "no_clear_theme" for x, good in zip(al, acl)]
                assignment["top_share"] = ash
                assignment["cosine"] = ac
                assignment["retained_features"] = ar
                assignment["clear_theme"] = acl
                assignment["candidate_id"] = candidate_id
                write_csv(destination / "assignments.csv", assignment)
                np.savez_compressed(destination / "weights.npz", fit_weights=proportions, association_weights=ap,
                                    fit_ids=fit.recommendationid.to_numpy(dtype=str),
                                    association_ids=association.recommendationid.to_numpy(dtype=str))
                joblib.dump(model, destination / "nmf.joblib")
                comparison = {"candidate_id": candidate_id, "language": language, "k": k, "seed": seed,
                              "fit_rows": len(fit), "association_rows": len(association), "features": matrix.shape[1],
                              "zero_vector_fit_rows": int((matrix.getnnz(axis=1) == 0).sum()),
                              "relative_reconstruction_error": float(model.reconstruction_err_ / np.sqrt(matrix.multiply(matrix).sum())),
                              "iterations": int(model.n_iter_), "converged": not any(isinstance(x.message, ConvergenceWarning) for x in caught),
                              "min_topic_rows": min(x["hard_assignment_count"] for x in stats),
                              "flagged_topics": sum(bool(x["flags"]) for x in stats),
                              "mean_top10_npmi_train": float(np.mean([x["top10_npmi_train"] for x in stats])),
                              "association_clear_rows": int(acl.sum()),
                              "association_no_clear_rows": int((~acl).sum()),
                              "status": "candidate_pending_independent_human_review"}
                all_comparisons.append(comparison)
                models.append((seed, model.components_, labels[matrix.getnnz(axis=1) > 0]))
                if k == config["review_sample_k"] and seed == config["review_sample_seed"]:
                    pilot_assignments.append(assignment)
                print(f"DONE {candidate_id}: fit={len(fit)}, terms={matrix.shape[1]}, clear={int(acl.sum())}/{len(association)}, flags={comparison['flagged_topics']}", flush=True)
            for first, second in combinations(models, 2):
                stability.append({"language": language, "k": k, "seed_a": first[0], "seed_b": second[0],
                                  **aligned_stability(first[1], second[1], first[2], second[2], words)})
    write_csv(output / "candidate_comparison.csv", all_comparisons)
    write_csv(output / "seed_stability.csv", stability)
    write_csv(output / "topic_diagnostics.csv", all_topic_stats)
    if len(pilot_assignments) == len(config["languages"]):
        review_dir = output / "human_review"
        review_dir.mkdir(exist_ok=True)
        blind, key = sample_for_review(frames["association_data"], pd.concat(pilot_assignments), config)
        for reviewer in ("A", "B"):
            write_csv(review_dir / f"reviewer_{reviewer}_blank.csv", blind)
        write_csv(review_dir / "coordinator_key.csv", key)
        save_json(review_dir / "review_manifest.json", {
            "status": "awaiting_two_independent_human_raters", "sample_size": len(blind),
            "language_counts": blind.language.value_counts().to_dict(),
            "game_count": int(key.appid.nunique()), "sampling_reasons": key.reason.value_counts().to_dict(),
            "recommendation_counts_for_sampling_only": {str(k): int(v) for k, v in key.voted_up.value_counts().items()},
            "pilot_k": config["review_sample_k"], "pilot_seed": config["review_sample_seed"],
            "warning": "k=10/seed=17 is a predeclared sampling anchor, not a selected/frozen main model. Independent language IDs are NOT shared themes. C's embedding candidates must be reviewed separately before task 12."})
    output_files = [p for p in output.rglob("*") if p.is_file()]
    save_json(output / "run_manifest.json", {
        "status": "completed_baseline_candidates_not_frozen_topics", "config": config,
        "config_sha256": sha256(config_path), "script_sha256": sha256(Path(__file__)),
        "common_script_sha256": sha256(Path(__file__).with_name("analysis_common.py")),
        "metadata_sha256": sha256(ROOT / config["metadata"]), "input_validation": audit,
        "model_selection": "No final selection. k=10 seed=17 used only to prepare an initial blind review sample. No recommendation-label metric used for candidate selection.",
        "feature_policy": "Whitespace/markup/URL normalization; game aliases and English title tokens excluded; negation retained. Input fitting columns: review_clean, language only; game names provide a fixed exclusion lexicon.",
        "output_files": [{"path": relative(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(output_files)],
    })
    print(f"Outputs: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/nmf_baseline.json")
    run(parser.parse_args().config)

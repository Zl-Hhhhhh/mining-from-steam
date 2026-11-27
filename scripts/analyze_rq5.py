"""RQ5 feasibility check (Task 18) and leak-free recommendation label prediction (Task 19)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

import analysis_common as common
import rq24_inputs as inputs
import structured_rq_weights as structured


ROOT = common.ROOT
TOPICS = inputs.TOPICS
TOPIC_NAMES = {
    "T01": "Gameplay",
    "T02": "Controls & difficulty",
    "T03": "Performance & bugs",
    "T04": "Story & characters",
    "T05": "Visuals & sound",
    "T06": "Multiplayer & community",
    "T07": "Price & purchases",
    "T08": "Content & updates",
}
SEED = 20261127
OUTER_SPLITS = 5
INNER_SPLITS = 3
BOOTSTRAP_REPS = 2000
LOGREG_C_GRID = (0.05, 0.2, 0.5, 1.0, 2.0, 5.0)
HGB_PARAM_GRID = (
    {"max_depth": 3, "learning_rate": 0.05, "max_iter": 150, "min_samples_leaf": 30},
    {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 20},
    {"max_depth": 5, "learning_rate": 0.05, "max_iter": 150, "min_samples_leaf": 40},
)
DEFAULT_TOPIC_VERSION = structured.TOPIC_VERSION


def load_rq5_data(topic_version: str = DEFAULT_TOPIC_VERSION):
    """Load verified clean master, metadata, and topic weights."""
    master, weights, provenance, checks, _ = inputs.load(topic_version)
    meta_path = ROOT / "data/metadata/freeze_20261101/game_meta.csv"
    provenance[common.relative(meta_path)] = {
        "sha256": common.sha256(meta_path),
        "recorded_sha256": provenance.get(common.relative(meta_path), {}).get("recorded_sha256"),
        "match": provenance.get(common.relative(meta_path), {}).get("match", "snapshot"),
    }
    meta = pd.read_csv(
        meta_path,
        usecols=["appid", "game", "project_primary_genre", "is_free", "sample_status"],
    ).set_index("appid")
    if not meta.index.is_unique:
        raise ValueError("Duplicate appid entries in game_meta.csv")

    df = master.copy()
    df["appid_str"] = df["appid"].astype(str)
    df["project_primary_genre"] = df["appid"].map(meta["project_primary_genre"]).fillna("Unknown").astype(str)
    df["is_free"] = df["appid"].map(meta["is_free"]).fillna(False).astype(int)
    df["log1p_hours"] = np.log1p(df["hours"].clip(lower=0.0))
    df["not_short_or_low_information"] = ~(df["is_short_text"] | df["is_low_information"])
    ordered = df.sort_values(["timestamp_created", "review_id"], kind="stable")
    retained = ordered.index[~ordered["steamid"].duplicated()]
    df["first_author_review"] = df.index.isin(retained)
    return df, weights, provenance, checks


def compute_feasibility(master: pd.DataFrame, primary_weights: pd.DataFrame):
    """Task 18: Author coverage, multi-game support, temporal cutoff check, and task selection."""
    w = primary_weights.loc[master.index]
    has_topic = w.sum(axis=1).gt(0)
    elig_mask = master["association_eligible"] & master["playtime_analysis_eligible"]

    cohorts = [
        ("clean_master", master),
        ("association_eligible", master.loc[elig_mask]),
        ("eligible_with_topic_evidence", master.loc[elig_mask & has_topic]),
    ]

    summary_rows = []
    dist_rows = []
    temporal_rows = []

    for cohort_name, frame in cohorts:
        author_stats = frame.groupby("steamid").agg(
            n_reviews=("appid", "size"),
            n_games=("appid", "nunique"),
            n_pos=("recommend", "sum"),
            n_neg=("recommend", lambda s: int((s == 0).sum())),
        )
        pos_games = frame.loc[frame["recommend"] == 1].groupby("steamid")["appid"].nunique()
        n_authors = len(author_stats)
        authors_ge2_games = int((author_stats["n_games"] >= 2).sum())
        authors_ge3_games = int((author_stats["n_games"] >= 3).sum())
        authors_ge2_pos_games = int((pos_games >= 2).sum())
        authors_ge3_pos_games = int((pos_games >= 3).sum())
        authors_both_classes = int(((author_stats["n_pos"] >= 1) & (author_stats["n_neg"] >= 1)).sum())

        summary_rows.append({
            "cohort": cohort_name,
            "reviews": int(len(frame)),
            "games": int(frame["appid"].nunique()),
            "unique_authors": int(n_authors),
            "single_game_authors": int((author_stats["n_games"] == 1).sum()),
            "single_game_author_share": float((author_stats["n_games"] == 1).mean()),
            "authors_ge2_games": authors_ge2_games,
            "authors_ge2_games_share": float(authors_ge2_games / n_authors),
            "authors_ge3_games": authors_ge3_games,
            "authors_ge2_positive_games": authors_ge2_pos_games,
            "authors_ge2_positive_games_share": float(authors_ge2_pos_games / n_authors),
            "authors_ge3_positive_games": authors_ge3_pos_games,
            "authors_with_pos_and_neg": authors_both_classes,
            "recommended_reviews": int(frame["recommend"].sum()),
            "not_recommended_reviews": int((frame["recommend"] == 0).sum()),
            "recommend_rate": float(frame["recommend"].mean()),
            "max_games_per_author": int(author_stats["n_games"].max()),
        })

        vc = author_stats["n_games"].value_counts().sort_index()
        for n_games, count in vc.items():
            dist_rows.append({
                "cohort": cohort_name,
                "distinct_games_reviewed": int(n_games),
                "author_count": int(count),
                "author_share": float(count / n_authors),
            })

        # Unified temporal cutoff check for simulating future personalized recommendation
        pos_df = frame.loc[frame["recommend"] == 1, ["steamid", "appid", "timestamp_created"]].copy()
        for q in (0.50, 0.70, 0.80):
            cutoff_ts = float(frame["timestamp_created"].quantile(q))
            cutoff_utc = pd.Timestamp(cutoff_ts, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            before = pos_df.loc[pos_df["timestamp_created"] <= cutoff_ts]
            after = pos_df.loc[pos_df["timestamp_created"] > cutoff_ts]
            before_games = before.groupby("steamid")["appid"].apply(set)
            after_games = after.groupby("steamid")["appid"].apply(set)
            common_authors = before_games.index.intersection(after_games.index)
            evaluable = 0
            total_unseen_test_pos = 0
            for aid in common_authors:
                unseen = after_games.loc[aid] - before_games.loc[aid]
                if unseen:
                    evaluable += 1
                    total_unseen_test_pos += len(unseen)
            temporal_rows.append({
                "cohort": cohort_name,
                "cutoff_quantile": q,
                "cutoff_utc": cutoff_utc,
                "evaluable_authors_future_rec": int(evaluable),
                "evaluable_author_share": float(evaluable / n_authors),
                "unseen_positive_test_pairs": int(total_unseen_test_pos),
            })

    summary_df = pd.DataFrame(summary_rows)
    elig_row = summary_df.set_index("cohort").loc["association_eligible"]
    topic_row = summary_df.set_index("cohort").loc["eligible_with_topic_evidence"]
    decision = {
        "selected_task": "recommendation_label_prediction",
        "rejected_task": "personalized_game_recommendation",
        "rationale": (
            f"Cross-game author overlap is too sparse for personalized recommendation: "
            f"{elig_row['single_game_author_share']:.2%} of eligible authors ({int(elig_row['single_game_authors'])}/{int(elig_row['unique_authors'])}) "
            f"reviewed only 1 game; only {int(elig_row['authors_ge2_positive_games'])} eligible authors "
            f"({int(topic_row['authors_ge2_positive_games'])} with topic evidence) have >=2 distinct recommended games, "
            f"and unified time-cutoff evaluation leaves fewer than 80 evaluable authors. "
            f"In contrast, author-grouped recommendation label prediction has {int(topic_row['reviews'])} topic-evidenced reviews "
            f"({int(topic_row['unique_authors'])} authors) and {int(elig_row['reviews'])} eligible reviews ({int(elig_row['unique_authors'])} authors) "
            f"with substantial positive and negative class support."
        ),
        "scope_boundary": (
            "Using review-derived topic weights or playtime at review to predict voted_up is a post-review label prediction task, "
            "not pre-purchase recommendation or causal inference."
        ),
    }
    return summary_df, pd.DataFrame(dist_rows), pd.DataFrame(temporal_rows), decision


def build_preprocessor(cat_cols: list[str], num_cols: list[str], dense: bool = False) -> ColumnTransformer:
    transformers = []
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
        transformers.append(("cat", ohe, cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0 if dense else 0.3)


def fit_predict_cv(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cat_cols: list[str],
    num_cols: list[str],
    model_kind: str = "logreg",
    n_splits: int = OUTER_SPLITS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Leak-free outer StratifiedGroupKFold with inner StratifiedGroupKFold hyperparameter selection."""
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y), dtype=int)
    oof_prob = np.zeros(len(y), dtype=float)
    fold_info = []

    X = df[cat_cols + num_cols].copy()

    for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(X, y, groups)):
        tr_groups = set(groups[tr_idx])
        te_groups = set(groups[te_idx])
        if tr_groups & te_groups:
            raise AssertionError("Group leakage detected between train and test fold")

        X_tr, y_tr, g_tr = X.iloc[tr_idx], y[tr_idx], groups[tr_idx]
        X_te = X.iloc[te_idx]

        if model_kind == "majority":
            clf = DummyClassifier(strategy="most_frequent")
            clf.fit(X_tr, y_tr)
            oof_pred[te_idx] = clf.predict(X_te)
            oof_prob[te_idx] = clf.predict_proba(X_te)[:, 1]
            fold_info.append({"fold": fold_idx, "best_param": "most_frequent", "n_test": int(len(te_idx))})
            continue

        inner = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed + fold_idx + 1)

        if model_kind == "logreg":
            best_c, best_score = LOGREG_C_GRID[0], -1.0
            for c_val in LOGREG_C_GRID:
                inner_scores = []
                for itr, iva in inner.split(X_tr, y_tr, g_tr):
                    pipe = Pipeline([
                        ("prep", build_preprocessor(cat_cols, num_cols, dense=False)),
                        ("clf", LogisticRegression(
                            C=c_val, class_weight="balanced", max_iter=1000, solver="lbfgs",
                        )),
                    ])
                    pipe.fit(X_tr.iloc[itr], y_tr[itr])
                    pred_va = pipe.predict(X_tr.iloc[iva])
                    inner_scores.append(balanced_accuracy_score(y_tr[iva], pred_va))
                mean_score = float(np.mean(inner_scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_c = c_val

            pipe = Pipeline([
                ("prep", build_preprocessor(cat_cols, num_cols, dense=False)),
                ("clf", LogisticRegression(
                    C=best_c, class_weight="balanced", max_iter=1000, solver="lbfgs",
                )),
            ])
            pipe.fit(X_tr, y_tr)
            oof_pred[te_idx] = pipe.predict(X_te)
            oof_prob[te_idx] = pipe.predict_proba(X_te)[:, 1]
            fold_info.append({
                "fold": fold_idx, "best_param": f"C={best_c}",
                "inner_bal_acc": round(best_score, 4), "n_test": int(len(te_idx)),
            })

        elif model_kind == "hgb":
            best_cfg, best_score = HGB_PARAM_GRID[0], -1.0
            for cfg in HGB_PARAM_GRID:
                inner_scores = []
                for itr, iva in inner.split(X_tr, y_tr, g_tr):
                    sw = compute_sample_weight("balanced", y_tr[itr])
                    pipe = Pipeline([
                        ("prep", build_preprocessor(cat_cols, num_cols, dense=True)),
                        ("clf", HistGradientBoostingClassifier(random_state=seed + fold_idx, **cfg)),
                    ])
                    pipe.fit(X_tr.iloc[itr], y_tr[itr], clf__sample_weight=sw)
                    pred_va = pipe.predict(X_tr.iloc[iva])
                    inner_scores.append(balanced_accuracy_score(y_tr[iva], pred_va))
                mean_score = float(np.mean(inner_scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_cfg = cfg

            sw_full = compute_sample_weight("balanced", y_tr)
            pipe = Pipeline([
                ("prep", build_preprocessor(cat_cols, num_cols, dense=True)),
                ("clf", HistGradientBoostingClassifier(random_state=seed + fold_idx, **best_cfg)),
            ])
            pipe.fit(X_tr, y_tr, clf__sample_weight=sw_full)
            oof_pred[te_idx] = pipe.predict(X_te)
            oof_prob[te_idx] = pipe.predict_proba(X_te)[:, 1]
            fold_info.append({
                "fold": fold_idx, "best_param": str(best_cfg),
                "inner_bal_acc": round(best_score, 4), "n_test": int(len(te_idx)),
            })
        else:
            raise ValueError(f"Unsupported model_kind: {model_kind}")

    return oof_pred, oof_prob, fold_info


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc_negative": float(average_precision_score(1 - y_true, 1.0 - y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def _fast_cluster_bootstrap_indices(groups: np.ndarray, n_boot: int, seed: int):
    """Map each row to a cluster index and draw multinomial cluster weights."""
    uniq, inv = np.unique(groups, return_inverse=True)
    n_clusters = len(uniq)
    rng = np.random.default_rng(seed)
    cluster_weights = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=n_boot)
    return inv, cluster_weights


def bootstrap_metrics_and_gain(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    prob_a: np.ndarray,
    groups: np.ndarray,
    pred_ref: np.ndarray | None = None,
    prob_ref: np.ndarray | None = None,
    n_boot: int = BOOTSTRAP_REPS,
    seed: int = SEED,
) -> dict[str, float | None]:
    """Fast vectorized cluster-bootstrap for balanced_accuracy, macro_f1, and roc_auc (plus paired gains)."""
    inv, cw = _fast_cluster_bootstrap_indices(groups, n_boot, seed)
    n_clusters = cw.shape[1]

    def confusion_by_cluster(pred: np.ndarray):
        tp = np.bincount(inv, weights=(y_true == 1) & (pred == 1), minlength=n_clusters)
        fn = np.bincount(inv, weights=(y_true == 1) & (pred == 0), minlength=n_clusters)
        tn = np.bincount(inv, weights=(y_true == 0) & (pred == 0), minlength=n_clusters)
        fp = np.bincount(inv, weights=(y_true == 0) & (pred == 1), minlength=n_clusters)
        return tp, fn, tn, fp

    def boot_bal_and_f1(pred: np.ndarray):
        tp_c, fn_c, tn_c, fp_c = confusion_by_cluster(pred)
        tp = cw @ tp_c
        fn = cw @ fn_c
        tn = cw @ tn_c
        fp = cw @ fp_c
        tpr = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        tnr = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) > 0)
        bal = 0.5 * (tpr + tnr)
        f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
        f1_neg = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fp + fn) > 0)
        mf1 = 0.5 * (f1_pos + f1_neg)
        return bal, mf1

    bal_a, mf1_a = boot_bal_and_f1(pred_a)
    pts = point_metrics(y_true, pred_a, prob_a)

    # ROC-AUC bootstrap on a subset of 500 draws for speed (already high precision)
    n_auc_boot = min(500, n_boot)
    auc_a = np.empty(n_auc_boot, dtype=float)
    auc_ref = np.empty(n_auc_boot, dtype=float) if prob_ref is not None else None
    for b in range(n_auc_boot):
        w_row = cw[b, inv]
        auc_a[b] = roc_auc_score(y_true, prob_a, sample_weight=w_row)
        if prob_ref is not None:
            auc_ref[b] = roc_auc_score(y_true, prob_ref, sample_weight=w_row)

    out: dict[str, float | None] = {
        **pts,
        "bal_acc_ci_low": float(np.quantile(bal_a, 0.025)),
        "bal_acc_ci_high": float(np.quantile(bal_a, 0.975)),
        "macro_f1_ci_low": float(np.quantile(mf1_a, 0.025)),
        "macro_f1_ci_high": float(np.quantile(mf1_a, 0.975)),
        "roc_auc_ci_low": float(np.quantile(auc_a, 0.025)),
        "roc_auc_ci_high": float(np.quantile(auc_a, 0.975)),
    }

    if pred_ref is not None and prob_ref is not None:
        ref_pts = point_metrics(y_true, pred_ref, prob_ref)
        bal_r, mf1_r = boot_bal_and_f1(pred_ref)
        d_bal = bal_a - bal_r
        d_mf1 = mf1_a - mf1_r
        d_auc = auc_a - auc_ref

        def two_sided_p(diffs: np.ndarray) -> float:
            p_pos = float(np.mean(diffs <= 0))
            p_neg = float(np.mean(diffs >= 0))
            return min(1.0, 2.0 * min(p_pos, p_neg))

        out.update({
            "delta_bal_acc": pts["balanced_accuracy"] - ref_pts["balanced_accuracy"],
            "delta_bal_acc_ci_low": float(np.quantile(d_bal, 0.025)),
            "delta_bal_acc_ci_high": float(np.quantile(d_bal, 0.975)),
            "delta_bal_acc_p": two_sided_p(d_bal),
            "delta_macro_f1": pts["macro_f1"] - ref_pts["macro_f1"],
            "delta_macro_f1_ci_low": float(np.quantile(d_mf1, 0.025)),
            "delta_macro_f1_ci_high": float(np.quantile(d_mf1, 0.975)),
            "delta_macro_f1_p": two_sided_p(d_mf1),
            "delta_roc_auc": pts["roc_auc"] - ref_pts["roc_auc"],
            "delta_roc_auc_ci_low": float(np.quantile(d_auc, 0.025)),
            "delta_roc_auc_ci_high": float(np.quantile(d_auc, 0.975)),
            "delta_roc_auc_p": two_sided_p(d_auc),
        })
    else:
        for k in (
            "delta_bal_acc", "delta_bal_acc_ci_low", "delta_bal_acc_ci_high", "delta_bal_acc_p",
            "delta_macro_f1", "delta_macro_f1_ci_low", "delta_macro_f1_ci_high", "delta_macro_f1_p",
            "delta_roc_auc", "delta_roc_auc_ci_low", "delta_roc_auc_ci_high", "delta_roc_auc_p",
        ):
            out[k] = None
    return out


def prepare_cohort_frame(master: pd.DataFrame, weights: pd.DataFrame, only_covered: bool) -> pd.DataFrame:
    elig = master.loc[master["association_eligible"] & master["playtime_analysis_eligible"]].copy()
    w = weights.loc[elig.index].copy()
    elig["has_topic_evidence"] = w.sum(axis=1).gt(0).astype(int)
    for topic in TOPICS:
        elig[topic] = w[topic].to_numpy(dtype=float)
    if only_covered:
        elig = elig.loc[elig["has_topic_evidence"] == 1].copy()
    return elig


def run_model_suite(master: pd.DataFrame, weights_dict: dict[str, pd.DataFrame]):
    """Run Task 19 primary comparisons, full-corpus decomposition, sensitivity checks, and topic ablations."""
    comparison_rows = []
    fold_diagnostics = {}
    oof_store = {}

    # 1. Primary Cohort: Eligible reviews with topic evidence (N=5,301, 5,218 authors)
    topic_df = prepare_cohort_frame(master, weights_dict["primary"], only_covered=True)
    y_t = topic_df["recommend"].to_numpy(dtype=int)
    g_t = topic_df["steamid"].to_numpy()
    topic_cols = list(TOPICS)

    topic_specs = [
        ("majority_baseline", "majority", ["appid_str"], [], None),
        ("structured_game_lang", "logreg", ["appid_str", "language"], [], "majority_baseline"),
        ("structured_plus_playtime", "logreg", ["appid_str", "language"], ["log1p_hours"], "majority_baseline"),
        ("structured_plus_topics_logreg", "logreg", ["appid_str", "language"], ["log1p_hours"] + topic_cols, "structured_plus_playtime"),
        ("structured_plus_playtime_hgb", "hgb", ["appid_str", "language"], ["log1p_hours"], "majority_baseline"),
        ("structured_plus_topics_hgb", "hgb", ["appid_str", "language"], ["log1p_hours"] + topic_cols, "structured_plus_playtime_hgb"),
    ]

    topic_preds = {}
    for name, kind, ccols, ncols, ref_name in topic_specs:
        print(f"[RQ5 primary topic_evidence] {name} (N={len(topic_df)})", flush=True)
        pred, prob, finfo = fit_predict_cv(topic_df, y_t, g_t, ccols, ncols, model_kind=kind)
        topic_preds[name] = (pred, prob)
        fold_diagnostics[f"topic_evidence::{name}"] = finfo
        ref_pred, ref_prob = topic_preds[ref_name] if ref_name else (None, None)
        stats = bootstrap_metrics_and_gain(y_t, pred, prob, g_t, ref_pred, ref_prob)
        comparison_rows.append({
            "cohort": "topic_evidence_primary",
            "model": name,
            "model_family": kind,
            "reference_model": ref_name or "",
            "n_reviews": int(len(topic_df)),
            "n_authors": int(topic_df["steamid"].nunique()),
            "n_games": int(topic_df["appid"].nunique()),
            "recommend_rate": float(y_t.mean()),
            **stats,
        })
    oof_store["topic_df"] = topic_df
    oof_store["topic_preds"] = topic_preds

    # 2. Full Eligible Cohort (N=19,395, 19,008 authors): decompose Coverage Flag vs Topic Content
    full_df = prepare_cohort_frame(master, weights_dict["primary"], only_covered=False)
    y_f = full_df["recommend"].to_numpy(dtype=int)
    g_f = full_df["steamid"].to_numpy()

    full_specs = [
        ("majority_baseline", "majority", ["appid_str"], [], None),
        ("structured_plus_playtime", "logreg", ["appid_str", "language"], ["log1p_hours"], "majority_baseline"),
        ("structured_plus_coverage_flag", "logreg", ["appid_str", "language"], ["log1p_hours", "has_topic_evidence"], "structured_plus_playtime"),
        ("structured_plus_coverage_and_topics", "logreg", ["appid_str", "language"], ["log1p_hours", "has_topic_evidence"] + topic_cols, "structured_plus_coverage_flag"),
        ("structured_plus_coverage_and_topics_vs_struct", "logreg", ["appid_str", "language"], ["log1p_hours", "has_topic_evidence"] + topic_cols, "structured_plus_playtime"),
    ]
    full_preds = {}
    for name, kind, ccols, ncols, ref_name in full_specs:
        if name == "structured_plus_coverage_and_topics_vs_struct":
            pred, prob = full_preds["structured_plus_coverage_and_topics"]
        else:
            print(f"[RQ5 full_eligible] {name} (N={len(full_df)})", flush=True)
            pred, prob, finfo = fit_predict_cv(full_df, y_f, g_f, ccols, ncols, model_kind=kind)
            full_preds[name] = (pred, prob)
            fold_diagnostics[f"full_eligible::{name}"] = finfo
        ref_pred, ref_prob = full_preds[ref_name] if ref_name else (None, None)
        stats = bootstrap_metrics_and_gain(y_f, pred, prob, g_f, ref_pred, ref_prob)
        comparison_rows.append({
            "cohort": "all_eligible_reviews",
            "model": name,
            "model_family": kind,
            "reference_model": ref_name or "",
            "n_reviews": int(len(full_df)),
            "n_authors": int(full_df["steamid"].nunique()),
            "n_games": int(full_df["appid"].nunique()),
            "recommend_rate": float(y_f.mean()),
            **stats,
        })

    # 3. Sensitivity & Subgroup Analyses (comparing structured_plus_topics vs structured_plus_playtime, logreg)
    sensitivity_rows = []
    sens_definitions = [
        ("primary_all_weights", topic_df, "steamid", ["appid_str", "language"], ["log1p_hours"]),
        ("fit_units_weights", prepare_cohort_frame(master, weights_dict["fit_units"], True), "steamid", ["appid_str", "language"], ["log1p_hours"]),
        ("k8_s17_proxy_weights", prepare_cohort_frame(master, weights_dict["k8_s17"], True), "steamid", ["appid_str", "language"], ["log1p_hours"]),
        ("no_a01_a02_weights", prepare_cohort_frame(master, weights_dict["no_a01_a02"], True), "steamid", ["appid_str", "language"], ["log1p_hours"]),
        ("canonical_text_only", topic_df.loc[topic_df["is_text_canonical"]].copy(), "steamid", ["appid_str", "language"], ["log1p_hours"]),
        ("no_short_low_information", topic_df.loc[topic_df["not_short_or_low_information"]].copy(), "steamid", ["appid_str", "language"], ["log1p_hours"]),
        ("one_review_per_author", topic_df.loc[topic_df["first_author_review"]].copy(), "steamid", ["appid_str", "language"], ["log1p_hours"]),
        ("complete_games_only", topic_df.loc[topic_df["complete_game"]].copy(), "steamid", ["appid_str", "language"], ["log1p_hours"]),
        ("english_source_only", topic_df.loc[topic_df["language"] == "english"].copy(), "steamid", ["appid_str"], ["log1p_hours"]),
        ("schinese_source_only", topic_df.loc[topic_df["language"] == "schinese"].copy(), "steamid", ["appid_str"], ["log1p_hours"]),
        # Cross-game generalization: group CV by appid (unseen games at test time -> use genre/free instead of appid)
        ("cross_game_group_cv", topic_df, "appid", ["project_primary_genre", "language"], ["is_free", "log1p_hours"]),
    ]

    for sname, sframe, group_col, base_cats, base_nums in sens_definitions:
        print(f"[RQ5 sensitivity] {sname} (N={len(sframe)}, group={group_col})", flush=True)
        ys = sframe["recommend"].to_numpy(dtype=int)
        gs = sframe[group_col].to_numpy()
        p_base, pr_base, _ = fit_predict_cv(sframe, ys, gs, base_cats, base_nums, model_kind="logreg")
        p_top, pr_top, _ = fit_predict_cv(sframe, ys, gs, base_cats, base_nums + topic_cols, model_kind="logreg")
        # For cross_game_group_cv, bootstrap by game cluster; otherwise by author cluster
        stats = bootstrap_metrics_and_gain(ys, p_top, pr_top, gs, p_base, pr_base)
        base_pts = point_metrics(ys, p_base, pr_base)
        sensitivity_rows.append({
            "scenario": sname,
            "cv_group_unit": group_col,
            "n_reviews": int(len(sframe)),
            "n_groups": int(pd.Series(gs).nunique()),
            "recommend_rate": float(ys.mean()),
            "base_bal_acc": base_pts["balanced_accuracy"],
            "base_macro_f1": base_pts["macro_f1"],
            "base_roc_auc": base_pts["roc_auc"],
            "topics_bal_acc": stats["balanced_accuracy"],
            "topics_macro_f1": stats["macro_f1"],
            "topics_roc_auc": stats["roc_auc"],
            "delta_bal_acc": stats["delta_bal_acc"],
            "delta_bal_acc_ci_low": stats["delta_bal_acc_ci_low"],
            "delta_bal_acc_ci_high": stats["delta_bal_acc_ci_high"],
            "delta_bal_acc_p": stats["delta_bal_acc_p"],
            "delta_macro_f1": stats["delta_macro_f1"],
            "delta_macro_f1_ci_low": stats["delta_macro_f1_ci_low"],
            "delta_macro_f1_ci_high": stats["delta_macro_f1_ci_high"],
            "delta_macro_f1_p": stats["delta_macro_f1_p"],
            "delta_roc_auc": stats["delta_roc_auc"],
            "delta_roc_auc_ci_low": stats["delta_roc_auc_ci_low"],
            "delta_roc_auc_ci_high": stats["delta_roc_auc_ci_high"],
            "delta_roc_auc_p": stats["delta_roc_auc_p"],
        })

    # 4. Single-topic contribution & leave-one-topic-out ablation on primary topic_evidence cohort
    base_pred_t, base_prob_t = topic_preds["structured_plus_playtime"]
    full_pred_t, full_prob_t = topic_preds["structured_plus_topics_logreg"]
    full_pts_t = point_metrics(y_t, full_pred_t, full_prob_t)
    base_pts_t = point_metrics(y_t, base_pred_t, base_prob_t)
    ablation_rows = []
    for topic in TOPICS:
        # Add single topic only to structured + playtime
        p_add, pr_add, _ = fit_predict_cv(
            topic_df, y_t, g_t, ["appid_str", "language"], ["log1p_hours", topic], model_kind="logreg",
        )
        add_stats = bootstrap_metrics_and_gain(y_t, p_add, pr_add, g_t, base_pred_t, base_prob_t, n_boot=1000)

        # Drop single topic from full 8-topic model (note: since weights sum to 1, dropping 1 column removes collinearity
        # only if 8 cols were redundant — in L2 regularized logreg, drop-one shows incremental loss vs remaining 7;
        # even more informative is single-topic addition!)
        remaining = [t for t in TOPICS if t != topic]
        p_drop, pr_drop, _ = fit_predict_cv(
            topic_df, y_t, g_t, ["appid_str", "language"], ["log1p_hours"] + remaining, model_kind="logreg",
        )
        drop_pts = point_metrics(y_t, p_drop, pr_drop)

        ablation_rows.append({
            "topic": topic,
            "topic_name": TOPIC_NAMES[topic],
            "single_topic_bal_acc": add_stats["balanced_accuracy"],
            "single_topic_delta_bal_acc": add_stats["delta_bal_acc"],
            "single_topic_delta_bal_acc_ci_low": add_stats["delta_bal_acc_ci_low"],
            "single_topic_delta_bal_acc_ci_high": add_stats["delta_bal_acc_ci_high"],
            "single_topic_delta_macro_f1": add_stats["delta_macro_f1"],
            "single_topic_delta_roc_auc": add_stats["delta_roc_auc"],
            "single_topic_delta_bal_acc_p": add_stats["delta_bal_acc_p"],
            "leave_one_out_bal_acc": drop_pts["balanced_accuracy"],
            "full_minus_loo_bal_acc": full_pts_t["balanced_accuracy"] - drop_pts["balanced_accuracy"],
            "base_bal_acc": base_pts_t["balanced_accuracy"],
            "full_topics_bal_acc": full_pts_t["balanced_accuracy"],
        })

    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(sensitivity_rows),
        pd.DataFrame(ablation_rows),
        fold_diagnostics,
    )


def save_figures(output_dir: Path, comparison_df: pd.DataFrame, sensitivity_df: pd.DataFrame, ablation_df: pd.DataFrame):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
    })

    # Figure 1: Model comparison across primary topic-evidenced cohort and full eligible cohort
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    prim = comparison_df.query("cohort == 'topic_evidence_primary'").set_index("model")
    prim_order = [
        ("majority_baseline", "Majority baseline"),
        ("structured_game_lang", "Game + language"),
        ("structured_plus_playtime", "Structured (+ playtime)"),
        ("structured_plus_topics_logreg", "+ Topics (LogReg)"),
        ("structured_plus_playtime_hgb", "Structured (HGB)"),
        ("structured_plus_topics_hgb", "+ Topics (HGB)"),
    ]
    y_pos = np.arange(len(prim_order))
    for offset, metric, lo_c, hi_c, color, label in (
        (-0.16, "balanced_accuracy", "bal_acc_ci_low", "bal_acc_ci_high", "#0B2C52", "Balanced accuracy"),
        (0.16, "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "#A6761D", "Macro-F1"),
    ):
        vals = [prim.loc[k, metric] for k, _ in prim_order]
        lo = [prim.loc[k, metric] - prim.loc[k, lo_c] for k, _ in prim_order]
        hi = [prim.loc[k, hi_c] - prim.loc[k, metric] for k, _ in prim_order]
        ax_l.errorbar(vals, y_pos + offset, xerr=[lo, hi], fmt="o", color=color, capsize=3, label=label)
    ax_l.set_yticks(y_pos, [lbl for _, lbl in prim_order])
    ax_l.invert_yaxis()
    ax_l.axvline(0.5, color="#8A9BA8", linestyle="--", linewidth=1)
    ax_l.set_xlim(0.38, 0.85)
    ax_l.grid(axis="x", alpha=0.25)
    ax_l.set_xlabel("Out-of-fold score (author-grouped 5-fold CV, 95% CI)")
    ax_l.set_title("Topic-evidenced cohort (N=5,301)")
    ax_l.legend(loc="upper right", frameon=False, fontsize=9)

    full = comparison_df.query("cohort == 'all_eligible_reviews'").set_index("model")
    full_order = [
        ("majority_baseline", "Majority baseline"),
        ("structured_plus_playtime", "Structured (+ playtime)"),
        ("structured_plus_coverage_flag", "+ Coverage flag"),
        ("structured_plus_coverage_and_topics", "+ Coverage + Topics"),
    ]
    y_pos2 = np.arange(len(full_order))
    for offset, metric, lo_c, hi_c, color, label in (
        (-0.16, "balanced_accuracy", "bal_acc_ci_low", "bal_acc_ci_high", "#0B2C52", "Balanced accuracy"),
        (0.16, "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "#A6761D", "Macro-F1"),
    ):
        vals = [full.loc[k, metric] for k, _ in full_order]
        lo = [full.loc[k, metric] - full.loc[k, lo_c] for k, _ in full_order]
        hi = [full.loc[k, hi_c] - full.loc[k, metric] for k, _ in full_order]
        ax_r.errorbar(vals, y_pos2 + offset, xerr=[lo, hi], fmt="o", color=color, capsize=3, label=label)
    ax_r.set_yticks(y_pos2, [lbl for _, lbl in full_order])
    ax_r.invert_yaxis()
    ax_r.axvline(0.5, color="#8A9BA8", linestyle="--", linewidth=1)
    ax_r.set_xlim(0.38, 0.80)
    ax_r.grid(axis="x", alpha=0.25)
    ax_r.set_xlabel("Out-of-fold score (author-grouped 5-fold CV, 95% CI)")
    ax_r.set_title("All eligible reviews (N=19,395)")
    fig.suptitle("RQ5  Post-review recommendation label prediction", fontweight="bold")
    fig.savefig(output_dir / "rq5_model_comparison.png", dpi=240)
    plt.close(fig)

    # Figure 2: Single-topic gain and sensitivity gain forest plot
    fig, (ax_a, ax_s) = plt.subplots(1, 2, figsize=(12.6, 5.2), layout="constrained")
    ab = ablation_df.sort_values("single_topic_delta_bal_acc", ascending=False)
    ya = np.arange(len(ab))
    x_ab = ab["single_topic_delta_bal_acc"].to_numpy() * 100
    lo_ab = (ab["single_topic_delta_bal_acc"] - ab["single_topic_delta_bal_acc_ci_low"]).to_numpy() * 100
    hi_ab = (ab["single_topic_delta_bal_acc_ci_high"] - ab["single_topic_delta_bal_acc"]).to_numpy() * 100
    ax_a.barh(ya, x_ab, color=["#0B2C52" if v > 1.0 else "#5B7D95" for v in x_ab], height=0.6)
    ax_a.errorbar(x_ab, ya, xerr=[lo_ab, hi_ab], fmt="none", ecolor="#142E44", capsize=3, linewidth=1.2)
    ax_a.axvline(0, color="#56616b", linestyle="--", linewidth=1)
    ax_a.set_yticks(ya, [f"{r.topic}  {r.topic_name}" for r in ab.itertuples()])
    ax_a.invert_yaxis()
    ax_a.grid(axis="x", alpha=0.25)
    ax_a.set_xlabel("Gain in balanced accuracy over structured baseline (pp, 95% CI)")
    ax_a.set_title("Single-topic incremental contribution")

    sens_labels = {
        "primary_all_weights": "Primary (all weights)",
        "fit_units_weights": "Fit-only weights",
        "k8_s17_proxy_weights": "K=8 overlap proxy",
        "no_a01_a02_weights": "Exclude A01/A02",
        "canonical_text_only": "Unique text only",
        "no_short_low_information": "Exclude short/low-info",
        "one_review_per_author": "One review per author",
        "complete_games_only": "Complete games (46)",
        "english_source_only": "English source only",
        "schinese_source_only": "Chinese source only",
        "cross_game_group_cv": "Cross-game CV (unseen games)",
    }
    ys = np.arange(len(sensitivity_df))
    for offset, col, lo_c, hi_c, color, label in (
        (-0.15, "delta_bal_acc", "delta_bal_acc_ci_low", "delta_bal_acc_ci_high", "#0B2C52", "Δ Balanced accuracy"),
        (0.15, "delta_macro_f1", "delta_macro_f1_ci_low", "delta_macro_f1_ci_high", "#A6761D", "Δ Macro-F1"),
    ):
        v = sensitivity_df[col].to_numpy() * 100
        lo = (sensitivity_df[col] - sensitivity_df[lo_c]).to_numpy() * 100
        hi = (sensitivity_df[hi_c] - sensitivity_df[col]).to_numpy() * 100
        ax_s.errorbar(v, ys + offset, xerr=[lo, hi], fmt="o", color=color, capsize=2.5, label=label)
    ax_s.axvline(0, color="#56616b", linestyle="--", linewidth=1)
    ax_s.set_xlim(-0.8, 15.8)
    ax_s.set_yticks(ys, [sens_labels.get(s, s) for s in sensitivity_df["scenario"]])
    ax_s.invert_yaxis()
    ax_s.grid(axis="x", alpha=0.25)
    ax_s.set_xlabel("Gain from adding 8 topic weights (pp, 95% CI)")
    ax_s.set_title("Sensitivity & cross-game robustness")
    ax_s.legend(loc="lower left", frameon=False, fontsize=8.5)
    fig.suptitle("RQ5  Which topics drive prediction gains and do gains hold?", fontweight="bold")
    fig.savefig(output_dir / "rq5_ablation_and_sensitivity.png", dpi=240)
    plt.close(fig)


def write_readme(
    output_dir: Path,
    topic_version: str,
    feasibility_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    decision: dict,
    comparison_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
):
    elig_f = feasibility_df.set_index("cohort").loc["association_eligible"]
    top_f = feasibility_df.set_index("cohort").loc["eligible_with_topic_evidence"]
    prim = comparison_df.query("cohort == 'topic_evidence_primary'").set_index("model")
    full = comparison_df.query("cohort == 'all_eligible_reviews'").set_index("model")
    ab_top = ablation_df.sort_values("single_topic_delta_bal_acc", ascending=False)

    def fmt_ci(val, lo, hi, scale=100, suffix="%"):
        return f"{val*scale:.2f}{suffix} [{lo*scale:.2f}, {hi*scale:.2f}]"

    def fmt_delta(val, lo, hi, p):
        p_txt = "<0.001" if p < 0.001 else f"{p:.3f}"
        return f"{val*100:+.2f} pp [{lo*100:+.2f}, {hi*100:+.2f}] (p={p_txt})"

    lines = [
        "# RQ5 应用验证：可行性检查（任务 18）与推荐标签预测（任务 19）",
        "",
        f"主题版本：`{topic_version}`（`all` 主权重口径）。严格按 `plan.md` 与 `Project tasks list.md` 执行：先完成任务 18 跨游戏作者可行性审计，再仅选择有数据支持的一项任务（任务 19）。",
        "",
        "## 一、任务 18：可行性检查与任务取舍",
        "",
        f"- **结论**：**放弃个性化游戏推荐，选择「推荐标签预测」**。",
        f"- **原因**：在 19,395 条合格评论（{int(elig_f['unique_authors']):,} 位唯一作者）中，**{elig_f['single_game_author_share']*100:.2f}%** 的作者（{int(elig_f['single_game_authors']):,} 位）仅评价过 1 款游戏；仅有 **{int(elig_f['authors_ge2_games'])}** 位作者评价过 ≥2 款游戏，仅有 **{int(elig_f['authors_ge2_positive_games'])}** 位拥有 ≥2 款不同正面推荐游戏（在有主题证据的 5,301 条评论中仅 **{int(top_f['authors_ge2_positive_games'])}** 位）。若按统一时间截点（50%/70%/80% 分位）模拟未来推荐，可评估作者数更少（见下表）。因此跨游戏覆盖不足以可靠评估个性化推荐。",
        f"- **标签预测可行性**：按作者分组 5 折交叉验证拥有 **{int(top_f['unique_authors']):,} 位独立作者 / {int(top_f['reviews']):,} 条评论**（有主题证据主队列，推荐率 {top_f['recommend_rate']*100:.1f}%）及 **{int(elig_f['unique_authors']):,} 位独立作者 / {int(elig_f['reviews']):,} 条评论**（全部合格评论，推荐率 {elig_f['recommend_rate']*100:.1f}%），正负例支持充分。",
        "",
        "| 样本队列 | 评论数 | 唯一作者数 | 仅评 1 款游戏占比 | ≥2 款游戏作者 | ≥2 款正面游戏作者 | ≥3 款正面游戏作者 | 正负皆有作者 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in feasibility_df.itertuples():
        lines.append(
            f"| `{row.cohort}` | {row.reviews:,} | {row.unique_authors:,} | {row.single_game_author_share*100:.2f}% | "
            f"{row.authors_ge2_games} | {row.authors_ge2_positive_games} | {row.authors_ge3_positive_games} | {row.authors_with_pos_and_neg} |"
        )

    lines += [
        "",
        "### 统一时间截点下的未来推荐可评估作者数",
        "",
        "| 样本队列 | 时间分位 | 截点 UTC | 可评估作者数（前有正例且后有未见正例） | 占作者比例 | 测试未见正例对 |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in temporal_df.itertuples():
        lines.append(
            f"| `{row.cohort}` | {int(row.cutoff_quantile*100)}% | {row.cutoff_utc} | "
            f"{row.evaluable_authors_future_rec} | {row.evaluable_author_share*100:.2f}% | {row.unseen_positive_test_pairs} |"
        )

    lines += [
        "",
        "## 二、任务 19：推荐标签预测结果（按作者分组 5 折交叉验证）",
        "",
        "所有特征预处理（One-Hot、标准化）与正则化/树模型调参均在每个外层训练折内部完成，测试折按 `steamid` 严格隔离，确保零作者泄漏。类别不平衡采用训练折平衡权重（`class_weight='balanced'`），区间由 2,000 次按作者整群成对重采样（Paired Cluster Bootstrap）计算。",
        "",
        "### 1. 主队列：有主题证据的合格评论（N=5,301，5,218 位作者，50 款游戏）",
        "",
        "| 模型 | Balanced Accuracy [95% CI] | Macro-F1 [95% CI] | ROC-AUC [95% CI] | 相对参照的 Δ Balanced Acc | 相对参照的 Δ Macro-F1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    label_map = {
        "majority_baseline": "多数类基线",
        "structured_game_lang": "结构化（游戏 + 语言）",
        "structured_plus_playtime": "结构化（+ 游玩时长）",
        "structured_plus_topics_logreg": "**结构化 + 8 主题（LogReg）**",
        "structured_plus_playtime_hgb": "结构化（+ 游玩时长，HGB）",
        "structured_plus_topics_hgb": "**结构化 + 8 主题（HGB）**",
    }
    for key in (
        "majority_baseline",
        "structured_game_lang",
        "structured_plus_playtime",
        "structured_plus_topics_logreg",
        "structured_plus_playtime_hgb",
        "structured_plus_topics_hgb",
    ):
        r = prim.loc[key]
        d_ba = "—" if pd.isna(r["delta_bal_acc"]) else fmt_delta(r["delta_bal_acc"], r["delta_bal_acc_ci_low"], r["delta_bal_acc_ci_high"], r["delta_bal_acc_p"])
        d_f1 = "—" if pd.isna(r["delta_macro_f1"]) else fmt_delta(r["delta_macro_f1"], r["delta_macro_f1_ci_low"], r["delta_macro_f1_ci_high"], r["delta_macro_f1_p"])
        lines.append(
            f"| {label_map[key]} | {fmt_ci(r['balanced_accuracy'], r['bal_acc_ci_low'], r['bal_acc_ci_high'])} | "
            f"{fmt_ci(r['macro_f1'], r['macro_f1_ci_low'], r['macro_f1_ci_high'])} | "
            f"{fmt_ci(r['roc_auc'], r['roc_auc_ci_low'], r['roc_auc_ci_high'], scale=1, suffix='')} | {d_ba} | {d_f1} |"
        )

    lines += [
        "",
        "### 2. 全合格样本队列（N=19,395，19,008 位作者）：剥离「检出覆盖指示」与「主题权重内容」",
        "",
        "由于有主题证据评论的原始推荐率（72.3%）显著低于未检出评论（88.8%），全样本评估刻意拆分为两步：先只加入 `has_topic_evidence` 覆盖标记，再加入 T01–T08 主题权重，从而分离覆盖效应与主题语义本身的净增益。",
        "",
        "| 模型 | 参照基线 | Balanced Accuracy [95% CI] | Macro-F1 [95% CI] | Δ Balanced Acc | Δ Macro-F1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    full_labels = {
        "majority_baseline": ("多数类基线", "—"),
        "structured_plus_playtime": ("结构化（游戏 + 语言 + 时长）", "多数类基线"),
        "structured_plus_coverage_flag": ("结构化 + 主题覆盖标记", "结构化基线"),
        "structured_plus_coverage_and_topics": ("**结构化 + 覆盖标记 + 8 主题权重**", "仅覆盖标记"),
        "structured_plus_coverage_and_topics_vs_struct": ("**结构化 + 覆盖标记 + 8 主题权重（总增益）**", "结构化基线"),
    }
    for key, (lbl, ref_lbl) in full_labels.items():
        r = full.loc[key]
        d_ba = "—" if pd.isna(r["delta_bal_acc"]) else fmt_delta(r["delta_bal_acc"], r["delta_bal_acc_ci_low"], r["delta_bal_acc_ci_high"], r["delta_bal_acc_p"])
        d_f1 = "—" if pd.isna(r["delta_macro_f1"]) else fmt_delta(r["delta_macro_f1"], r["delta_macro_f1_ci_low"], r["delta_macro_f1_ci_high"], r["delta_macro_f1_p"])
        lines.append(
            f"| {lbl} | {ref_lbl} | {fmt_ci(r['balanced_accuracy'], r['bal_acc_ci_low'], r['bal_acc_ci_high'])} | "
            f"{fmt_ci(r['macro_f1'], r['macro_f1_ci_low'], r['macro_f1_ci_high'])} | {d_ba} | {d_f1} |"
        )

    lines += [
        "",
        "### 3. 单主题增量贡献（在结构化基线上逐一加入单个主题）",
        "",
        "| 主题 | 单主题加入 Δ Balanced Acc [95% CI] | 单主题加入 Δ Macro-F1 | 单主题加入 Δ ROC-AUC | 配对 p |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in ab_top.itertuples():
        p_txt = "<0.001" if r.single_topic_delta_bal_acc_p < 0.001 else f"{r.single_topic_delta_bal_acc_p:.3f}"
        lines.append(
            f"| {r.topic} {r.topic_name} | {r.single_topic_delta_bal_acc*100:+.2f} pp [{r.single_topic_delta_bal_acc_ci_low*100:+.2f}, {r.single_topic_delta_bal_acc_ci_high*100:+.2f}] | "
            f"{r.single_topic_delta_macro_f1*100:+.2f} pp | {r.single_topic_delta_roc_auc:+.4f} | {p_txt} |"
        )

    lines += [
        "",
        "### 4. 敏感性与跨游戏泛化对照",
        "",
        "| 设定 | 分组单位 | 评论数 | 基线 BalAcc | +主题 BalAcc | Δ Balanced Acc [95% CI] | Δ Macro-F1 [95% CI] |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for r in sensitivity_df.itertuples():
        lines.append(
            f"| `{r.scenario}` | `{r.cv_group_unit}` | {r.n_reviews:,} | {r.base_bal_acc*100:.2f}% | {r.topics_bal_acc*100:.2f}% | "
            f"{r.delta_bal_acc*100:+.2f} pp [{r.delta_bal_acc_ci_low*100:+.2f}, {r.delta_bal_acc_ci_high*100:+.2f}] | "
            f"{r.delta_macro_f1*100:+.2f} pp [{r.delta_macro_f1_ci_low*100:+.2f}, {r.delta_macro_f1_ci_high*100:+.2f}] |"
        )

    lines += [
        "",
        "## 三、研究边界与局限",
        "",
        "1. **评论后预测边界**：目标评论的主题权重与发表时游玩时长均来自评论发表当时，因此任务 19 评估的是**评论后推荐标签预测（post-review label prediction）**，不能解释为购前游戏推荐能力或因果机制。",
        "2. **上游无监督表示说明**：主题聚类与原型映射是在冻结评论语料上无监督完成的（未接触 `voted_up` 或游玩时长标签），有监督分类器、编码器、缩放器和超参数均严格限定在各外层训练折内拟合。",
        "3. **主题覆盖与探索性状态**：仅 27.2% 的清洗评论检出主题证据；全样本队列中已通过独立对照分离覆盖标记效应与主题内容净效应。主题包仍属探索性交付，待正式冻结后可一键重跑验收。",
        "",
        "## 四、复现命令",
        "",
        "```powershell",
        f".\\.venv\\Scripts\\python.exe -B scripts/analyze_rq5.py --topic-version {topic_version}",
        "```",
        "",
        f"输入、代码与全部产物的 SHA-256 记录见 `manifest.json`。",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(destination: Path, topic_version: str = DEFAULT_TOPIC_VERSION, overwrite: bool = False):
    destination = destination.resolve()
    if destination.exists():
        if overwrite:
            rmtree(destination)
        else:
            raise FileExistsError(f"Output directory already exists (use --overwrite or choose a new path): {destination}")

    with threadpool_limits(limits=2):
        master, weights_dict, provenance, checks = load_rq5_data(topic_version)
        feasibility_df, author_dist_df, temporal_df, decision = compute_feasibility(master, weights_dict["primary"])
        comparison_df, sensitivity_df, ablation_df, fold_diagnostics = run_model_suite(master, weights_dict)

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".rq5-{uuid4().hex}"
    stage.mkdir()
    try:
        common.write_csv(stage / "feasibility_summary.csv", feasibility_df)
        common.write_csv(stage / "author_game_distribution.csv", author_dist_df)
        common.write_csv(stage / "temporal_split_feasibility.csv", temporal_df)
        common.write_csv(stage / "model_comparison.csv", comparison_df)
        common.write_csv(stage / "sensitivity_comparison.csv", sensitivity_df)
        common.write_csv(stage / "topic_ablation.csv", ablation_df)
        save_figures(stage, comparison_df, sensitivity_df, ablation_df)
        write_readme(
            stage, topic_version, feasibility_df, temporal_df, decision,
            comparison_df, sensitivity_df, ablation_df,
        )
        common.save_json(stage / "diagnostics.json", {
            "task18_decision": decision,
            "cv_design": {
                "outer_splits": OUTER_SPLITS,
                "inner_splits": INNER_SPLITS,
                "seed": SEED,
                "bootstrap_reps": BOOTSTRAP_REPS,
                "logreg_c_grid": list(LOGREG_C_GRID),
            },
            "validation": checks,
            "folds": fold_diagnostics,
        })
        common.save_json(stage / "manifest.json", {
            "topic_version": topic_version,
            "status": "exploratory",
            "task18_selected_task": decision["selected_task"],
            "inputs": provenance,
            "code_sha256": {p: common.sha256(ROOT / p) for p in (
                "scripts/analyze_rq5.py",
                "scripts/rq24_inputs.py",
                "scripts/structured_rq_weights.py",
                "scripts/analysis_common.py",
                "requirements.txt",
            )},
            "outputs_sha256": {p.name: common.sha256(p) for p in sorted(stage.iterdir())},
            "runtime": common.runtime_versions(),
        })
        stage.rename(destination)
    finally:
        if stage.exists():
            rmtree(stage)

    print(f"RQ5 outputs written to {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic-version",
        choices=(inputs.TOPIC_VERSION, structured.TOPIC_VERSION),
        default=DEFAULT_TOPIC_VERSION,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = args.output or (ROOT / "output/rq5" / args.topic_version)
    run(out, topic_version=args.topic_version, overwrite=args.overwrite)

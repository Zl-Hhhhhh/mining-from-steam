"""Game fixed effects and game-clustered inference for RQ2/RQ4."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, linprog
from scipy.stats import t
from statsmodels.api import GLM, OLS, families
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from rq24_inputs import TOPICS


def separation_status(x, y):
    # A nonzero margin with no adverse margins detects quasi or complete separation.
    z = (2 * y[:, None] - 1) * x
    options = {"threads": 1}
    bounds = [(-1, 1)] * x.shape[1]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected", category=OptimizeWarning)
        any_strict = linprog(-z.mean(axis=0), A_ub=-z, b_ub=np.zeros(len(z)),
                             bounds=bounds, method="highs", options=options)
        if not any_strict.success:
            return f"solver_status_{any_strict.status}"
        if -any_strict.fun <= 1e-8:
            return "none"
        all_strict = linprog(np.zeros(x.shape[1]), A_ub=-z, b_ub=-np.ones(len(z)),
                             bounds=[(None, None)] * x.shape[1], method="highs", options=options)
    if all_strict.success:
        return "complete"
    return "quasi" if all_strict.status == 2 else f"solver_status_{all_strict.status}"


def fixed_effects(frame):
    game = pd.get_dummies(frame.appid, prefix="game", drop_first=True, dtype=float)
    control = pd.DataFrame({"const": 1.0, "lang_schinese": frame.language.eq("schinese").astype(float)})
    if control.lang_schinese.nunique() < 2:
        control = control.drop(columns="lang_schinese")
    return pd.concat([control, game], axis=1)


def fit(frame, predictors, outcome="recommend", *, check_separation=False):
    logistic = outcome == "recommend"
    y = frame[outcome]
    game = frame.appid
    keep = game.map(frame.groupby("appid")[outcome].nunique()).gt(1) if logistic else y.notna()
    excluded_game_ids = sorted(int(g) for g in game.loc[~keep].unique()) if logistic else []
    frame, y, game = frame.loc[keep], y.loc[keep], game.loc[keep]
    predictors = predictors.loc[keep]
    diagnostics = {
        "n_input": len(keep), "n_used": len(y), "games_used": int(game.nunique()),
        "excluded_single_outcome_games": excluded_game_ids,
        "excluded_rows": int((~keep).sum()),
    }
    # Homogeneous-outcome games cannot identify slopes in a game fixed-effect logit.
    if logistic:
        diagnostics.update(recommended=int(y.sum()), not_recommended=int(len(y) - y.sum()))
    terms = pd.concat([fixed_effects(frame), predictors], axis=1)
    x = terms.to_numpy(dtype=float)
    diagnostics["parameters"] = len(terms.columns)
    diagnostics["cluster_df"] = int(game.nunique() - 1)
    if len(y) <= x.shape[1] or y.nunique() < 2 or game.nunique() < 2:
        return None, diagnostics | {"status": "insufficient_variation"}
    rank = int(np.linalg.matrix_rank(x))
    diagnostics["rank"] = rank
    if rank != x.shape[1]:
        return None, diagnostics | {"status": "rank_deficient"}
    if check_separation:
        separation = separation_status(x, y.to_numpy())
        diagnostics["separation"] = separation
        if separation in {"complete", "quasi"}:
            return None, diagnostics | {"status": "separated"}
        if separation != "none":
            return None, diagnostics | {"status": "separation_check_failed"}
    model = GLM(y, terms, family=families.Binomial()) if logistic else OLS(y, terms)
    covariance = {"groups": game.to_numpy(), "use_correction": True, "df_correction": True}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(cov_type="cluster", cov_kwds=covariance, use_t=True)
    warning_types = sorted({warning.category.__name__ for warning in caught})
    diagnostics.update(warnings=warning_types, converged=bool(getattr(result, "converged", True)))
    if any(issubclass(warning.category, PerfectSeparationWarning) for warning in caught):
        return None, diagnostics | {"status": "separated"}
    if not diagnostics["converged"] or not np.isfinite(result.params).all() or not np.isfinite(result.bse).all():
        return None, diagnostics | {"status": "invalid_fit"}
    diagnostics.update(log_likelihood=float(result.llf), aic=float(result.aic))
    if logistic:
        predictions = np.asarray(result.fittedvalues)
        diagnostics.update(
            iterations=int(result.fit_history["iteration"]),
            max_absolute_coefficient=float(abs(result.params).max()),
            fitted_probability_min=float(predictions.min()),
            fitted_probability_max=float(predictions.max()),
            brier_in_sample=float(np.mean((y.to_numpy() - predictions) ** 2)),
        )
    return result, diagnostics | {"status": "ok"}


def effect(result, contrast, measure, model, scenario, term, diagnostics):
    row = {"model": model, "scenario": scenario, "term": term, "measure": measure,
           "n": diagnostics["n_used"], "games": diagnostics["games_used"], "status": diagnostics["status"]}
    if result is None or not set(contrast) <= set(result.params.index):
        if result is not None:
            row["status"] = "unsupported_topic"
        return row
    params = result.params.loc[list(contrast)]
    cov = result.cov_params().loc[params.index, params.index].to_numpy()
    scale = np.asarray(list(contrast.values()), dtype=float)
    beta = float(scale @ params.to_numpy())
    variance = float(scale @ cov @ scale)
    if variance <= 0 or not np.isfinite(variance):
        return row | {"status": "nonpositive_contrast_variance"}
    se = np.sqrt(variance)
    df = diagnostics["cluster_df"]
    delta = t.ppf(.975, df) * se
    transform = np.exp if measure == "odds_ratio" else float
    values = [float(transform(value)) for value in (beta, beta - delta, beta + delta)]
    return row | {"coefficient": beta, "cluster_se": float(se), "df": df,
                  "estimate": values[0], "ci_low": values[1], "ci_high": values[2],
                  "p": float(2 * t.sf(abs(beta / se), df))}


def holm(rows):
    valid = [row for row in rows if row.get("p") is not None]
    if valid:
        for row, adjusted in zip(valid, multipletests([row["p"] for row in valid], method="holm")[1], strict=True):
            row["p_holm"] = float(adjusted)


def dominant_topic(weights):
    maximum = weights.max(axis=1).to_numpy()[:, None]
    leaders = np.isclose(weights, maximum, atol=1e-8, rtol=0)
    return weights.idxmax(axis=1).where(leaders.sum(axis=1) == 1, "mixed_tie")


def topic_predictors(weights, representation):
    variables = pd.get_dummies(dominant_topic(weights), dtype=float) if representation == "dominant" else weights * 10
    present = [key for key in (*TOPICS, "mixed_tie") if key in variables and variables[key].ne(0).any()]
    if not present:
        raise ValueError("No topic weights in the requested scenario")
    reference = present[0]
    return variables.loc[:, present[1:]], reference, present


def rq2(frame, weights, scenario, *, timed, representation="weight", check_separation=False):
    predictors, reference, present = topic_predictors(weights, representation)
    if timed:
        predictors = predictors.assign(time_log2p1=frame.time_log2p1)
    model = f"rq2_{representation}_{'time' if timed else 'base'}"
    result, diag = fit(frame, predictors, check_separation=check_separation)
    diag.update(model=model, scenario=scenario, reference=reference, topics_present=present)
    if representation == "dominant":
        included = frame.loc[~frame.appid.isin(diag["excluded_single_outcome_games"])]
        groups = included.recommend.groupby(dominant_topic(weights).loc[included.index]).agg(["size", "sum"])
        diag["dominant_topic_support"] = {
            label: {"n": int(row["size"]), "recommended": int(row["sum"]),
                    "not_recommended": int(row["size"] - row["sum"])}
            for label, row in groups.iterrows()
        }
    rows = []
    for topic in (*TOPICS, "mixed_tie"):
        if topic == reference or (topic == "mixed_tie" and representation != "dominant"):
            continue
        row = effect(result, {topic: 1}, "odds_ratio", model, scenario, f"{topic}_vs_{reference}", diag)
        rows.append(row)
    holm(rows)
    if {"T02", "T06"} <= set(present):
        contrast = {key: sign for key, sign in (("T06", 1), ("T02", -1)) if key != reference}
        rows.append(effect(result, contrast, "odds_ratio", model, scenario, "T06_vs_T02_common", diag))
    if timed:
        rows.append(effect(result, {"time_log2p1": 1}, "odds_ratio", model, scenario, "time_log2p1", diag))
    return rows, diag


def rq4(frame, *, topic_weights=None, group=False, controls=None, scenario, check_separation=False):
    predictors = (pd.DataFrame({"mid_vs_low": frame.tertile.eq("mid"), "high_vs_low": frame.tertile.eq("high")})
                  .astype(float) if group else frame[["time_log2p1"]])
    time_terms = list(predictors.columns)
    measure = "percentage_points" if topic_weights is not None else "odds_ratio"
    target = "weights" if topic_weights is not None else (
        "recommend_topic_adjusted" if controls is not None else "recommend")
    model = f"rq4_{target}_{'tertile' if group else 'time'}"
    if controls is not None:
        predictors = predictors.join(controls)
    outcomes = TOPICS if topic_weights is not None else ("recommend",)
    rows, diagnostics = [], []
    for outcome in outcomes:
        data = frame.assign(**{outcome: topic_weights[outcome] * 100}) if topic_weights is not None else frame
        result, diag = fit(data, predictors, outcome, check_separation=check_separation)
        diag.update(model=model, scenario=scenario, outcome=outcome)
        diagnostics.append(diag)
        for term in time_terms:
            row = effect(result, {term: 1}, measure, model, scenario, term, diag)
            rows.append(row | {"outcome": outcome})
    holm(rows)
    return rows, diagnostics

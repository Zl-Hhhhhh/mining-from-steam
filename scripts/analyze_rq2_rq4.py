"""Reproduce B-Lin's RQ2 and RQ4 analyses from the published exploratory topics."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from threadpoolctl import threadpool_limits

import analysis_common as common
import rq24_inputs as inputs
import rq24_models as models


ROOT = common.ROOT
DEFAULT_OUTPUT = ROOT / "output/rq2_rq4" / inputs.TOPIC_VERSION
TOPICS = inputs.TOPICS
TERTILES = ("low", "mid", "high")
NAMES = ("Gameplay", "Controls & difficulty", "Technical issues", "Story & characters",
         "Audiovisual", "Multiplayer & community", "Price & payment", "Content & support")
COLORS = ("#307885", "#c18a2e", "#a94c43")
SAMPLE_FILTERS = {
    "common_dates": "in_common_language_window",
    "canonical_text": "is_text_canonical",
    "complete_games": "complete_game",
}


def prepare_playtime(master):
    data = master.loc[master.association_eligible & master.playtime_analysis_eligible].copy()
    by_game = data.groupby("appid").hours
    lower, upper = by_game.quantile(1 / 3), by_game.quantile(2 / 3)
    data["tertile"] = np.select(
        [data.hours.le(data.appid.map(lower)), data.hours.le(data.appid.map(upper))],
        ["low", "mid"], default="high",
    )
    data["time_log2p1"] = np.log1p(data.hours) / np.log(2)
    data["p99_hours"] = data.groupby(["appid", "language"]).hours.transform("quantile", q=.99)
    data["not_short_or_low_information"] = ~(data.is_short_text | data.is_low_information)
    first_author_review = data.sort_values(["timestamp_created", "review_id"], kind="stable")
    retained = first_author_review.index[~first_author_review.steamid.duplicated()]
    data["first_author_review"] = data.index.isin(retained)
    cutoffs = pd.DataFrame({"p33_hours": lower, "p67_hours": upper})
    sizes = data.groupby(["appid", "tertile"]).size().unstack(fill_value=0)
    cutoffs = cutoffs.join(sizes.reindex(columns=TERTILES, fill_value=0))
    cutoffs["at_p33"] = data.hours.eq(data.appid.map(lower)).groupby(data.appid).sum()
    cutoffs["at_p67"] = data.hours.eq(data.appid.map(upper)).groupby(data.appid).sum()
    counts = {
        "reviews_master": len(master), "association_eligible": int(master.association_eligible.sum()),
        "valid_playtime": len(data), "missing_playtime_in_association": int(
            (master.association_eligible & master.playtime_at_review_missing).sum()),
        "invalid_playtime_in_association": int((master.association_eligible & master.playtime_at_review_invalid).sum()),
        "inconsistent_playtime_in_association": int((master.association_eligible & master.is_playtime_inconsistent).sum()),
        "hard_outliers_in_association": int((master.association_eligible & master.is_playtime_outlier).sum()),
        "zero_hours": int(data.hours.eq(0).sum()),
        "repeated_game_time_rows_beyond_first": int(data.duplicated(["appid", "hours"]).sum()),
        "games_with_equal_tertile_cutoffs": int(lower.eq(upper).sum()),
        "rows_above_group_p99": int(data.hours.gt(data.p99_hours).sum()),
        "hours_quantiles": {str(q): float(data.hours.quantile(q)) for q in (0, .01, .25, .5, .75, .95, .99, 1)},
    }
    return data, cutoffs.reset_index(), counts


def topic_population(data, weights):
    weights = weights.loc[data.index]
    covered = weights.sum(axis=1).gt(0)
    return data.loc[covered], weights.loc[covered]


def topic_scenarios(data, weights):
    main_data, main_weights = topic_population(data, weights["primary"])
    yield "primary", main_data, main_weights
    for key in ("all_units", "k8_s17", "no_a01_a02", "no_time_mentions"):
        cohort, selected = topic_population(data, weights[key])
        yield key, cohort, selected
        if key == "no_a01_a02":
            yield "a01_a02_same_sample_baseline", main_data.loc[cohort.index], main_weights.loc[cohort.index]
    yield "all_units_same_sample", main_data, weights["all_units"].loc[main_data.index]
    yield "high_confidence", *topic_population(data.loc[data.confidence.eq("high")], weights["primary"])
    for name, column in {**SAMPLE_FILTERS, "no_short_low_information": "not_short_or_low_information",
                         "one_review_per_author": "first_author_review"}.items():
        yield name, *topic_population(data.loc[data[column]], weights["primary"])
    capped = np.minimum(main_data.hours, main_data.p99_hours)
    yield "cap_p99", main_data.assign(time_log2p1=np.log1p(capped) / np.log(2)), main_weights
    yield "trim_p99", *topic_population(data.loc[data.hours.le(data.p99_hours)], weights["primary"])


def recommendation_scenarios(data, topic_data):
    yield "primary", data
    yield "fit_topic_sample", topic_data
    for name, column in {**SAMPLE_FILTERS, "no_short_low_information": "not_short_or_low_information",
                         "one_review_per_author": "first_author_review"}.items():
        yield name, data.loc[data[column]]
    capped = np.minimum(data.hours, data.p99_hours)
    yield "cap_p99", data.assign(time_log2p1=np.log1p(capped) / np.log(2))
    yield "trim_p99", data.loc[data.hours.le(data.p99_hours)]


def coverage(master, weights):
    data = master.loc[master.association_eligible]
    summaries = []
    for scope in ("primary", "all_units"):
        available = weights[scope].loc[data.index].sum(axis=1).gt(0)
        scoped = data.assign(topic_evidence=available)
        for (language, present), rows in scoped.groupby(["language", "topic_evidence"]):
            summaries.append({"scope": scope, "language": language, "topic_evidence": bool(present),
                              "n": len(rows), "games": rows.appid.nunique(),
                              "recommend_rate": rows.recommend.mean(), "median_hours": rows.hours.median()})
    return summaries


def topic_support(data, weights, scenario):
    rows = []
    for topic in TOPICS:
        support = data.loc[weights[topic].gt(0)]
        rows.append({"scenario": scenario, "topic": topic, "positive_weight_reviews": len(support),
                     "recommended": int(support.recommend.sum()),
                     "not_recommended": int(len(support) - support.recommend.sum()),
                     "games": support.appid.nunique(), "total_weight": weights[topic].sum()})
    return rows


def grouped_summary(data, weights, scope):
    rows = []
    games = np.sort(data.appid.unique())
    rng = np.random.default_rng(20261118)
    boot = rng.multinomial(len(games), np.repeat(1 / len(games), len(games)), size=2000)
    for group in TERTILES:
        block = data.loc[data.tertile.eq(group)].join(weights)
        covered = block.loc[block[list(TOPICS)].sum(axis=1).gt(0)]
        for cohort, selected in (("all_reviews", block), ("with_topic_evidence", covered)):
            stats = selected.groupby("appid").recommend.agg(["size", "sum"]).reindex(games, fill_value=0)
            n_boot = boot @ stats["size"].to_numpy()
            estimates = (boot @ stats["sum"].to_numpy())[n_boot > 0] / n_boot[n_boot > 0]
            lo, hi = np.quantile(estimates, [.025, .975])
            values = {
                "scope": scope, "tertile": group, "cohort": cohort, "n": len(selected),
                "games": selected.appid.nunique(), "topic_coverage": len(covered) / len(block),
                "median_hours": selected.hours.median(), "recommend_rate": selected.recommend.mean(),
                "recommend_bootstrap_low": lo, "recommend_bootstrap_high": hi,
            }
            if cohort == "with_topic_evidence":
                values.update({topic: selected[topic].mean() for topic in TOPICS})
            rows.append(values)
    return rows


def run_models(data, weights, association):
    rq2_rows, rq4_rows, support, diagnostics = [], [], [], []
    main_data, main_weights = topic_population(data, weights["primary"])
    for scenario, frame, selected in topic_scenarios(data, weights):
        print(f"Topic scenario: {scenario} ({len(frame)} reviews)", flush=True)
        support.extend(topic_support(frame, selected, scenario))
        specs = [("weight", True)]
        if scenario == "primary":
            specs += [("weight", False), ("dominant", False), ("dominant", True)]
        for representation, timed in specs:
            model_frame, model_weights = (frame, selected) if timed else topic_population(association, weights["primary"])
            rows, diag = models.rq2(model_frame, model_weights, scenario, timed=timed, representation=representation,
                                    check_separation=True)
            rq2_rows.extend(rows)
            diagnostics.append(diag)
        for group in ((False,) if scenario == "cap_p99" else (False, True)):
            rows, diags = models.rq4(frame, topic_weights=selected, group=group, scenario=scenario)
            rq4_rows.extend(rows)
            diagnostics.extend(diags)
        # Same sample as the corresponding topic model, holding the exploratory composition fixed.
        controls, _, _ = models.topic_predictors(selected, "weight")
        if len(controls.columns):
            rows, diags = models.rq4(frame, controls=controls, scenario=scenario, check_separation=True)
            rq4_rows.extend(rows)
            diagnostics.extend(diags)
    for scenario, frame in recommendation_scenarios(data, main_data):
        print(f"Playtime recommendation scenario: {scenario} ({len(frame)} reviews)", flush=True)
        for group in ((False,) if scenario == "cap_p99" else (False, True)):
            rows, diags = models.rq4(frame, group=group, scenario=scenario, check_separation=True)
            rq4_rows.extend(rows)
            diagnostics.extend(diags)
    controls, _, _ = models.topic_predictors(main_weights, "weight")
    rows, diags = models.rq4(main_data, controls=controls, group=True, scenario="primary", check_separation=True)
    rq4_rows.extend(rows)
    diagnostics.extend(diags)
    return rq2_rows, rq4_rows, support, diagnostics


def save_figures(output, rq2, groups):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True})
    effects = pd.DataFrame(rq2)
    fig, ax = plt.subplots(figsize=(8.3, 4.2), layout="constrained")
    for model, offset, color, name in (
        ("rq2_weight_base", -.13, COLORS[0], "Game and language"),
        ("rq2_weight_time", .13, COLORS[2], "Also adjusted for playtime"),
    ):
        selected = effects.query("scenario == 'primary' and model == @model").set_index("term")
        selected = selected.loc[[f"{topic}_vs_T01" for topic in TOPICS[1:]]]
        x = selected.estimate
        ax.errorbar(x, np.arange(len(x)) + offset, xerr=[x - selected.ci_low, selected.ci_high - x],
                    fmt="o", color=color, capsize=3, label=name)
    ax.set_yticks(np.arange(7), [f"{topic}  {name}" for topic, name in zip(TOPICS[1:], NAMES[1:], strict=True)])
    ax.set_xscale("log")
    ticks = (.7, .8, .9, 1., 1.1, 1.2, 1.3, 1.4)
    ax.set_xticks(ticks, [f"{value:g}" for value in ticks])
    ax.minorticks_off()
    ax.axvline(1, color="#56616b", linestyle="--")
    ax.grid(axis="x", alpha=.25)
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)
    ax.set(xlabel="Recommendation odds ratio per 10 pp transferred from T01 (95% CI)",
           title="RQ2  Exploratory themes and recommendation")
    fig.savefig(output / "rq2_topic_odds.png", dpi=240)
    plt.close(fig)

    group_frame = pd.DataFrame(groups).query("scope == 'primary'")
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5.1), layout="constrained",
                                    gridspec_kw={"width_ratios": [1.55, 1]})
    content = group_frame.query("cohort == 'with_topic_evidence'").set_index("tertile").loc[list(TERTILES)]
    y = np.arange(len(TOPICS))
    for i, (name, row) in enumerate(content.iterrows()):
        left.barh(y + (i - 1) * .24, row[list(TOPICS)].astype(float) * 100, height=.21, color=COLORS[i],
                  label=f"{name.title()} (n={int(row.n):,})")
    left.set_yticks(y, [f"{topic}  {name}" for topic, name in zip(TOPICS, NAMES, strict=True)])
    left.invert_yaxis()
    left.grid(axis="x", alpha=.25)
    left.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16), ncols=3, fontsize=8.5)
    left.set(xlabel="Mean theme weight (%) among reviews with evidence", title="Theme composition")
    for cohort, color, name in (("all_reviews", COLORS[0], "All eligible reviews"),
                                ("with_topic_evidence", COLORS[2], "Reviews with topic evidence")):
        selected = group_frame.query("cohort == @cohort").set_index("tertile").loc[list(TERTILES)]
        value = selected.recommend_rate * 100
        right.errorbar(range(3), value, yerr=[value - selected.recommend_bootstrap_low * 100,
                                              selected.recommend_bootstrap_high * 100 - value],
                       marker="o", capsize=3, color=color, label=name)
    right.set_xticks(range(3), [g.title() for g in TERTILES])
    right.set(ylabel="Recommended (%)", xlabel="Playtime tertile within each game", title="Recommendation rate")
    right.grid(axis="y", alpha=.25)
    right.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16), fontsize=8.5)
    fig.suptitle("RQ4  Playtime at review and exploratory outcomes", fontweight="bold")
    fig.savefig(output / "rq4_playtime_groups.png", dpi=240)
    plt.close(fig)


def run(destination):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    with threadpool_limits(limits=2):
        master, weights, provenance, checks, annotation = inputs.load()
        data, cutoffs, playtime = prepare_playtime(master)
        rq2, rq4, support, diagnostics = run_models(data, weights, master.loc[master.association_eligible])
        groups = grouped_summary(data, weights["primary"], "primary")
        groups += grouped_summary(data, weights["all_units"], "all_units")
    for name in ("rq2_weight_base", "rq2_weight_time", "rq2_dominant_base", "rq2_dominant_time",
                 "rq4_recommend_time", "rq4_recommend_tertile"):
        if not any(row["model"] == name and row["scenario"] == "primary" and row["status"] == "ok"
                   for row in rq2 + rq4):
            raise ValueError(f"Primary model failed: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".rq24-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "results"
        stage.mkdir()
        for filename, rows in {
            "rq2_models.csv": rq2, "rq2_topic_support.csv": support, "rq4_models.csv": rq4,
            "rq4_groups.csv": groups, "rq4_tertile_cutoffs.csv": cutoffs,
            "review_coverage.csv": coverage(master, weights), "annotation_sample.csv": annotation,
        }.items():
            common.write_csv(stage / filename, rows)
        save_figures(stage, rq2, groups)
        common.save_json(stage / "diagnostics.json", {
            "validation": checks, "playtime": playtime, "models": diagnostics,
            "annotation_sample_size": len(annotation),
            "model_covariance": "game-clustered with finite G and N corrections; t(G-1)",
        })
        common.save_json(stage / "manifest.json", {
            "topic_version": inputs.TOPIC_VERSION,
            "status": "exploratory",
            "inputs": provenance,
            "code_sha256": {p: common.sha256(ROOT / p) for p in (
                "scripts/analyze_rq2_rq4.py", "scripts/rq24_inputs.py", "scripts/rq24_models.py",
                "scripts/analysis_common.py", "scripts/assign_topics_v1_cluster_derived.py",
                "requirements-analysis.txt",
            )},
            "outputs_sha256": {p.name: common.sha256(p) for p in sorted(stage.iterdir())},
            "runtime": common.runtime_versions(),
        })
        stage.rename(destination)
    print(f"Reproduced exploratory results in {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)

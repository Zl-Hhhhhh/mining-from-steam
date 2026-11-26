"""Generate reproducible exploratory RQ1 topic-distribution results."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis_common as common


ROOT = common.ROOT
DEFAULT_TOPIC_PACKAGE = ROOT / "output/topic_selection/topics-v1-exploratory-20261118"
DEFAULT_CLEAN_MASTER = ROOT / "data/cleaned/freeze_20261101/clean_master.jsonl"
DEFAULT_GAME_META = ROOT / "data/metadata/freeze_20261101/game_meta.csv"
DEFAULT_OUTPUT = ROOT / "output/rq1/topics-v1-exploratory-20261118"
TOPIC_IDS = [f"T{index:02d}" for index in range(1, 9)]
LANGUAGE_LABELS = {"english": "English", "schinese": "Simplified Chinese"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def topic_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{topic_id}" for topic_id in TOPIC_IDS]


def confidence_rank(value: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(value, 0)


def load_inputs(topic_package: Path, clean_master: Path, game_meta: Path) -> tuple[pd.DataFrame, dict, dict]:
    topic_file = topic_package / "topic_v1.json"
    weights_file = topic_package / "review_topic_weights.csv"
    assignments_file = topic_package / "unit_topic_assignments.jsonl"
    for path in (topic_file, weights_file, assignments_file, clean_master, game_meta):
        if not path.exists():
            raise FileNotFoundError(path)

    topic_definition = common.read_json(topic_file)
    definitions = {row["topic_id"]: row for row in topic_definition["topics"]}
    if set(definitions) != set(TOPIC_IDS):
        raise ValueError(f"Expected topic IDs {TOPIC_IDS}, got {sorted(definitions)}")

    master = common.load_jsonl(clean_master)
    master["recommendationid"] = master["recommendationid"].astype(str)
    if master.recommendationid.duplicated().any():
        raise ValueError("Duplicate recommendation IDs in clean master")

    weights = pd.read_csv(weights_file, encoding="utf-8-sig", keep_default_na=False)
    weights["review_id"] = weights["review_id"].astype(str)
    if weights.review_id.duplicated().any():
        raise ValueError("Duplicate review IDs in topic weights")
    for column in topic_columns("fit") + topic_columns("all"):
        if column not in weights.columns:
            raise ValueError(f"Missing topic weight column: {column}")
        weights[column] = pd.to_numeric(weights[column], errors="raise")
        if (weights[column] < 0).any() or not np.isfinite(weights[column]).all():
            raise ValueError(f"Invalid topic weights in {column}")

    metadata = pd.read_csv(game_meta, encoding="utf-8-sig", keep_default_na=False)
    metadata["appid"] = pd.to_numeric(metadata["appid"], errors="raise").astype(int)
    if metadata.appid.duplicated().any():
        raise ValueError("Duplicate appids in game metadata")
    required_metadata = {"appid", "game", "sample_status", "partial_languages", "project_primary_genre", "project_genres"}
    missing = required_metadata - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    joined = master.merge(weights, left_on="recommendationid", right_on="review_id", how="left", validate="one_to_one")
    if joined.review_id.isna().any():
        raise ValueError("Some clean-master reviews have no topic-weight record")
    joined = joined.merge(metadata, on="appid", how="left", suffixes=("", "_meta"), validate="many_to_one")
    if joined.game_meta.isna().any():
        raise ValueError("Some reviews have no game metadata")
    joined["language_label"] = joined["language"].map(LANGUAGE_LABELS).fillna(joined["language"])
    for prefix in ("fit", "all"):
        columns = topic_columns(prefix)
        joined[f"{prefix}_topic_total"] = joined[columns].sum(axis=1)
        joined[f"{prefix}_has_topic"] = joined[f"{prefix}_topic_total"] > 1e-9
    return joined, topic_definition, {"assignments": read_jsonl(assignments_file), "definitions": definitions}


def _group_metrics(frame: pd.DataFrame, weight_prefix: str) -> dict:
    total_column = f"{weight_prefix}_topic_total"
    has_column = f"{weight_prefix}_has_topic"
    total_reviews = len(frame)
    topic_reviews = int(frame[has_column].sum())
    return {
        "total_reviews": total_reviews,
        "topic_evidence_reviews": topic_reviews,
        "no_topic_evidence_reviews": total_reviews - topic_reviews,
        "topic_coverage": topic_reviews / total_reviews if total_reviews else 0.0,
        "topic_weight_denominator": float(frame[total_column].sum()),
    }


def aggregate_topics(frame: pd.DataFrame, group_columns: list[str], weight_prefix: str, topic_names: dict) -> pd.DataFrame:
    weight_columns = topic_columns(weight_prefix)
    records: list[dict] = []
    grouped = [((), frame)] if not group_columns else frame.groupby(group_columns, dropna=False, sort=True, observed=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        metrics = _group_metrics(group, weight_prefix)
        denominator = metrics["topic_weight_denominator"]
        base = dict(zip(group_columns, key))
        for topic_id, weight_column in zip(TOPIC_IDS, weight_columns):
            weight_sum = float(group[weight_column].sum())
            record = {
                **base,
                "topic_id": topic_id,
                "topic_name_zh": topic_names[topic_id]["name_zh"],
                "topic_name_en": topic_names[topic_id]["name_en"],
                **metrics,
                "topic_weight_sum": weight_sum,
                "topic_share": weight_sum / denominator if denominator else 0.0,
            }
            records.append(record)
    return pd.DataFrame(records)


def build_coverage_table(frame: pd.DataFrame, primary_scope: str = "fit") -> pd.DataFrame:
    rows = []
    grouping = ["appid", "game", "language", "sample_status", "project_primary_genre"]
    for key, group in frame.groupby(grouping, sort=True, observed=False):
        row = dict(zip(grouping, key))
        row.update(_group_metrics(group, primary_scope))
        for scope in ("fit", "all"):
            row[f"{scope}_topic_evidence_reviews"] = int(group[f"{scope}_has_topic"].sum())
            row[f"{scope}_topic_coverage"] = float(group[f"{scope}_has_topic"].mean()) if len(group) else 0.0
        row["high_confidence_reviews"] = int((group["confidence"] == "high").sum())
        row["medium_confidence_reviews"] = int((group["confidence"] == "medium").sum())
        row["low_confidence_reviews"] = int((group["confidence"] == "low").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_sensitivity(frame: pd.DataFrame, topic_names: dict) -> pd.DataFrame:
    records = []
    variants = {
        "fit": (frame, "fit"),
        "all": (frame, "all"),
        "all_same_fit_sample": (frame[frame["fit_has_topic"]], "all"),
        "fit_high_confidence": (frame[frame["confidence"] == "high"], "fit"),
        "all_high_confidence": (frame[frame["confidence"] == "high"], "all"),
    }
    for variant, (subset, prefix) in variants.items():
        table = aggregate_topics(subset, [], prefix, topic_names)
        table.insert(0, "analysis_variant", variant)
        records.append(table)
    return pd.concat(records, ignore_index=True)


def representative_comments(frame: pd.DataFrame, assignments: list[dict], topic_names: dict,
                            weight_prefix: str = "fit") -> pd.DataFrame:
    review_lookup = frame.set_index("recommendationid")
    candidates = []
    for unit in assignments:
        if weight_prefix == "fit" and unit.get("assignment_scope") != "fit_candidate":
            continue
        topic_scores = unit.get("topic_scores", {})
        for topic_id in unit.get("topic_ids", []):
            if topic_id not in TOPIC_IDS:
                continue
            score = float(topic_scores.get(topic_id, 0.0))
            for review_id in unit.get("review_ids", []):
                review_id = str(review_id)
                if review_id not in review_lookup.index:
                    continue
                review = review_lookup.loc[review_id]
                candidates.append({
                    "topic_id": topic_id,
                    "topic_name_zh": topic_names[topic_id]["name_zh"],
                    "topic_name_en": topic_names[topic_id]["name_en"],
                    "review_id": review_id,
                    "appid": int(review.appid),
                    "game": review.game,
                    "language": review.language,
                    "language_label": review.language_label,
                    "confidence": unit.get("confidence", "low"),
                    "confidence_rank": confidence_rank(unit.get("confidence", "low")),
                    "sentence_score": score,
                    "review_topic_weight": float(review.get(f"{weight_prefix}_{topic_id}", 0.0)),
                    "text": unit.get("text", ""),
                })
    if not candidates:
        return pd.DataFrame(columns=["topic_id", "topic_name_zh", "topic_name_en", "language", "review_id", "game", "text"])
    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["topic_id", "language", "confidence_rank", "sentence_score", "review_topic_weight", "review_id"],
        ascending=[True, True, False, False, False, True],
    )
    selected = candidate_frame.drop_duplicates(["topic_id", "language", "review_id"])
    selected = selected.groupby(["topic_id", "language"], sort=True, group_keys=False).head(3).copy()
    selected["rank"] = selected.groupby(["topic_id", "language"], sort=True).cumcount() + 1
    return selected.drop(columns=["confidence_rank"]).sort_values(["topic_id", "language", "rank"])


def plot_results(overall: pd.DataFrame, game: pd.DataFrame, genre: pd.DataFrame, language: pd.DataFrame,
                 topic_names: dict, output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    labels = [topic_names[topic]["name_en"] for topic in TOPIC_IDS]

    ordered_overall = overall.set_index("topic_id").loc[TOPIC_IDS].reset_index()
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.bar(labels, ordered_overall["topic_share"], color="#24788b")
    ax.set_ylabel("Share among reviews with topic evidence")
    ax.set_title("RQ1 overall topic distribution")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "overall_topic_share.png", dpi=180)
    plt.close(fig)

    game_pivot = game.pivot(index="game", columns="topic_id", values="topic_share").reindex(columns=TOPIC_IDS).fillna(0)
    game_support = game.drop_duplicates("game").set_index("game")["topic_evidence_reviews"]
    game_pivot = game_pivot.loc[game_support.sort_values(ascending=False).index]
    fig_height = max(8, 0.25 * len(game_pivot) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
    image = ax.imshow(game_pivot.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.01, float(game_pivot.to_numpy().max())))
    ax.set_xticks(range(len(TOPIC_IDS)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(game_pivot)), game_pivot.index)
    ax.set_title("RQ1 topic distribution by game")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Game, sorted by topic-evidence count")
    fig.colorbar(image, ax=ax, label="Topic share")
    fig.savefig(output / "game_topic_heatmap.png", dpi=180)
    plt.close(fig)

    genre_pivot = genre.pivot(index="project_primary_genre", columns="topic_id", values="topic_share").reindex(columns=TOPIC_IDS).fillna(0)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(genre_pivot) + 2)), constrained_layout=True)
    image = ax.imshow(genre_pivot.to_numpy(), aspect="auto", cmap="YlOrBr", vmin=0, vmax=max(0.01, float(genre_pivot.to_numpy().max())))
    ax.set_xticks(range(len(TOPIC_IDS)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(genre_pivot)), genre_pivot.index)
    ax.set_title("RQ1 topic distribution by project primary genre")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Project primary genre")
    fig.colorbar(image, ax=ax, label="Topic share")
    fig.savefig(output / "genre_topic_heatmap.png", dpi=180)
    plt.close(fig)

    language_pivot = language.pivot(index="language_label", columns="topic_id", values="topic_share").reindex(columns=TOPIC_IDS).fillna(0)
    x = np.arange(len(TOPIC_IDS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for offset, language_name, color in [(-width / 2, "English", "#24788b"), (width / 2, "Simplified Chinese", "#d87936")]:
        if language_name in language_pivot.index:
            ax.bar(x + offset, language_pivot.loc[language_name].to_numpy(), width, label=language_name, color=color)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Share among reviews with topic evidence")
    ax.set_title("RQ1 topic distribution by language")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "language_topic_share.png", dpi=180)
    plt.close(fig)


def write_readme(output: Path, frame: pd.DataFrame, topic_definition: dict, primary_scope: str) -> None:
    fit_count = int(frame["fit_has_topic"].sum())
    all_count = int(frame["all_has_topic"].sum())
    lines = [
        "# RQ1 主题发现与分布",
        "",
        f"本目录记录 `{topic_definition['topic_version']}` 的探索性结果。",
        "",
        "## 输入与口径",
        "",
        f"- 评论总数：{len(frame):,}。游戏数：{frame.appid.nunique()}。语言：{', '.join(sorted(frame.language.unique()))}。",
        f"- 主结果使用 `{primary_scope}_T01`–`{primary_scope}_T08`。",
        f"- `fit` 有主题证据评论 {fit_count:,} 条（{fit_count / len(frame):.1%}），`all` 有 {all_count:,} 条（{all_count / len(frame):.1%}）。",
        "- 主题比例在有主题证据的评论内归一化。无主题评论保留在覆盖率诊断中。",
        "- 游戏类型主口径为项目配置中的 `project_primary_genre`。",
        "- 主题生成没有使用推荐标签或游玩时长。本目录不支持因果解释。",
        "",
        "## 结果文件",
        "",
        "- `overall_topic_distribution.csv` 总体主题比例。",
        "- `game_topic_distribution.csv` 游戏—主题长表。",
        "- `genre_topic_distribution.csv` 项目类型—主题长表。",
        "- `language_topic_distribution.csv` 语言—主题长表。",
        "- `coverage_diagnostics.csv` 游戏和语言的覆盖率、置信度及样本状态。",
        "- `representative_comments.csv` 每个主题和语言的代表句子。",
        "- `sensitivity_overall.csv` fit、all、固定 fit 样本的 all 权重和高置信度子集的总体比较。",
        "",
        "## 解释限制",
        "",
        "- 覆盖率表示当前主题规则能识别到证据的评论比例。",
        "- 主题权重是句子证据的归一化贡献，不是概率或人工确认概率。",
        "- 当前主题包中的低置信度记录需要在正式主题冻结后重新检查。",
        "- 主题版本冻结后，应使用相同脚本重新生成正式 RQ1 结果。",
        "",
        f"主题版本：`{topic_definition['topic_version']}`。运行参数记录在 `rq1_manifest.json`。",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite to replace: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frame, topic_definition, auxiliary = load_inputs(args.topic_package.resolve(), args.clean_master.resolve(), args.game_meta.resolve())
    topic_names = auxiliary["definitions"]
    scope = args.primary_scope

    overall = aggregate_topics(frame, [], scope, topic_names)
    game = aggregate_topics(frame, ["appid", "game", "sample_status", "partial_languages", "project_primary_genre"], scope, topic_names)
    language = aggregate_topics(frame, ["language", "language_label"], scope, topic_names)
    genre = aggregate_topics(frame, ["project_primary_genre"], scope, topic_names)
    coverage = build_coverage_table(frame, scope)
    representatives = representative_comments(frame, auxiliary["assignments"], topic_names, scope)
    sensitivity = build_sensitivity(frame, topic_names)

    common.write_csv(output / "overall_topic_distribution.csv", overall)
    common.write_csv(output / "game_topic_distribution.csv", game)
    common.write_csv(output / "language_topic_distribution.csv", language)
    common.write_csv(output / "genre_topic_distribution.csv", genre)
    common.write_csv(output / "coverage_diagnostics.csv", coverage)
    common.write_csv(output / "representative_comments.csv", representatives)
    common.write_csv(output / "sensitivity_overall.csv", sensitivity)
    common.write_csv(output / "topic_definitions.csv", pd.DataFrame(topic_definition["topics"]))
    plot_results(overall, game, genre, language, topic_names, output)
    write_readme(output, frame, topic_definition, scope)

    source_paths = [args.topic_package / "topic_v1.json", args.topic_package / "review_topic_weights.csv",
                    args.topic_package / "unit_topic_assignments.jsonl", args.clean_master, args.game_meta]
    manifest = {
        "schema_version": "rq1-results-1",
        "analysis": "RQ1 topic discovery and distribution",
        "topic_version": topic_definition["topic_version"],
        "topic_status": topic_definition["status"],
        "primary_weight_prefix": scope,
        "sensitivity_weight_prefix": "fit" if scope == "all" else "all",
        "counts": {
            "reviews": len(frame),
            "games": int(frame.appid.nunique()),
            "fit_topic_evidence_reviews": int(frame["fit_has_topic"].sum()),
            "all_topic_evidence_reviews": int(frame["all_has_topic"].sum()),
            "representative_comments": len(representatives),
        },
        "parameters": {
            "topic_ids": TOPIC_IDS,
            "genre_field": "project_primary_genre",
            "topic_share_denominator": "sum of topic weights among reviews with topic evidence",
            "language_labels": LANGUAGE_LABELS,
        },
        "sources": [{"path": common.relative(path), "sha256": common.sha256(path)} for path in source_paths],
        "outputs": [{"path": common.relative(path), "sha256": common.sha256(path)}
                    for path in sorted(output.iterdir()) if path.is_file() and path.name != "rq1_manifest.json"],
        "runtime": common.runtime_versions(),
    }
    common.save_json(output / "rq1_manifest.json", manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic-package", type=Path, default=DEFAULT_TOPIC_PACKAGE)
    parser.add_argument("--clean-master", type=Path, default=DEFAULT_CLEAN_MASTER)
    parser.add_argument("--game-meta", type=Path, default=DEFAULT_GAME_META)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--primary-scope", choices=("fit", "all"), default="fit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()

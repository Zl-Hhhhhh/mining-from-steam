"""Build a compact diagnostic report from existing NMF outputs, without refitting."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import ROOT, read_json, relative, save_json, sha256, write_csv


def report(run_directory):
    root = Path(run_directory)
    manifest = read_json(root / "run_manifest.json")
    for item in manifest["output_files"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Changed model output: {item['path']}")
    candidates = pd.read_csv(root / "candidate_comparison.csv")
    stability = pd.read_csv(root / "seed_stability.csv")
    review = read_json(root / "human_review/review_manifest.json")
    notes = []
    diagnostic_topics = pd.read_csv(root / "topic_diagnostics.csv", keep_default_na=False)
    grammatical_terms = {"not", "no", "never", "did", "does", "不", "没", "没有"}
    for row in diagnostic_topics.itertuples():
        reasons = [x for x in row.flags.split(";") if x]
        first_term = row.top_terms.split(" | ")[0]
        if first_term in grammatical_terms:
            reasons.append("negation_or_function_word_dominance")
        fit_n = int(candidates.loc[candidates.candidate_id == row.candidate_id, "fit_rows"].iloc[0])
        if row.hard_assignment_count / fit_n > 0.4:
            reasons.append("large_mixed_topic_check")
        if reasons:
            notes.append({"candidate_id": row.candidate_id, "topic_id": row.topic_id,
                          "check_reasons": ";".join(reasons), "top_terms": row.top_terms,
                          "status": "automatic_or_text_inspection_flag_pending_human_review"})
    write_csv(root / "candidate_review_notes.csv", notes)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for language, color in (("english", "#24788b"), ("schinese", "#d87936")):
        rows = candidates[candidates.language == language]
        grouped = rows.groupby("k")
        avg = grouped.relative_reconstruction_error.mean()
        axes[0, 0].plot(avg.index, avg.values, "o-", color=color, label=language)
        fraction = grouped.apply(lambda frame: (frame.association_clear_rows / frame.association_rows).mean(), include_groups=False)
        axes[0, 1].plot(fraction.index, fraction.values, "o-", color=color, label=language)
        matches = stability[stability.language == language]
        axes[1, 0].plot(matches.k, matches.adjusted_rand_index, "o-", color=color, label=language)
        axes[1, 1].plot(matches.k, matches.matched_top10_jaccard, "o-", color=color, label=language)
    for ax in axes.flat:
        ax.set_xlabel("Number of topics (k)")
        ax.set_xticks(sorted(candidates.k.unique()))
        ax.grid(alpha=0.2)
    axes[0, 0].set(title="Reconstruction error (lower is better)*", ylabel="Relative error")
    axes[0, 1].set(title="Reviews passing provisional theme thresholds", ylabel="Fraction of association rows", ylim=(0, 1))
    axes[1, 0].set(title="Assignment stability across seeds", ylabel="Adjusted Rand index", ylim=(0, 1))
    axes[1, 1].set(title="Aligned keyword stability across seeds", ylabel="Top-10 word Jaccard overlap", ylim=(0, 1))
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Language-specific NMF baseline diagnostics, pending human review\n*Training error decreases with k; it does not select an interpretable topic model.", fontsize=12)
    for suffix in ("png", "svg"):
        fig.savefig(root / f"candidate_overview.{suffix}", dpi=160)
    plt.close(fig)
    lines = ["# NMF 基线运行结果", "", "状态：候选模型已生成，尚未人工核验或冻结为正式主题。", "",
             "## 输入与实验", "",
             f"- 正式窗口：{manifest['input_validation']['sampling_window']['start_utc']} 至 {manifest['input_validation']['sampling_window']['end_utc']}。",
             "- 使用 A 的 topic_fit 训练，association_data 只用于变换和分配候选主题；原始文件未修改。",
             f"- 共 {len(candidates)} 个候选：中英文独立训练，主题数为 6、10、14，各用随机种子 17、42。",
             f"- 收敛模型：{int(candidates.converged.sum())}/{len(candidates)}。推荐标签、游玩时长和投票数不进入主题训练或候选选择。",
             "- 零向量不归入任何主题；保留 no_clear_theme。主题权重经过成分尺度校正，但不是概率或校准后的置信度。", "",
             "## 候选稳定性", "", "同一语言、同一主题数下比较两个随机种子；ARI(评论分组一致程度)与关键词重合率越高越稳定。", "",
             "| 语言 | 主题数 | ARI | 对齐后关键词重合率 |", "| --- | --- | --- | --- |"]
    for row in stability.itertuples():
        lines.append(f"| {row.language} | {row.k} | {row.adjusted_rand_index:.3f} | {row.matched_top10_jaccard:.3f} |")
    lines += ["", "## 人工核验抽样锚点", "",
              "k=10、seed=17 仅用于准备核验样本，不代表已选为最佳或主模型。两个语言的 T00 等编号没有跨语言对应关系。", "",
              "| 语言 | 训练评论 | 变换评论 | 有候选主题 | 无明确主题 | 训练零向量 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in candidates[(candidates.k == 10) & (candidates.seed == 17)].itertuples():
        lines.append(f"| {row.language} | {row.fit_rows} | {row.association_rows} | {row.association_clear_rows} | {row.association_no_clear_rows} | {row.zero_vector_fit_rows} |")
    lines += ["", f"已生成 {review['sample_size']} 条盲核验样本，中英文各 {review['language_counts']['english']} 条，覆盖 {review['game_count']} 款游戏。两份 reviewer 空表内容相同，人工标签均未填写。",
              "", "## 已知限制与失败检查", "",
              "- 初始 v1 中多个主题主要由泛泛评价和措辞组成；aspect_v2 是仅根据文本观察增加停用词的对照，保留否定词。两者均未用推荐标签挑选模型。",
              "- 过滤泛泛词汇后，部分评论失去可建模词汇；低覆盖率本身需要报告，不能为了得到更多主题而强制分配。",
              "- 训练重构误差和训练词共现分数不是人工可解释性的证据；关键词稳定也不等于评论分组稳定。",
              "- 大而泛的主题、语法或否定词主导的主题、极小主题仍需人工检查，诊断 flags 不能代替人工命名。",
              "- candidate_review_notes.csv 逐项记录需复核的候选编号与原因；这些是自动或文本检查标记，不是两名成员的人工核验结论。",
              "- 原审计将空文本、短文本、重复文本和不足 200 条列为 error；本次验证确认没有其他结构性错误，并使用清洗后的资格标记。",
              "- JSON/JSONL 输入已校验；清洗 JSONL 与 A 的哈希仅相差换行。CSV 不作为本次建模输入，原 CSV 哈希恢复问题不在本任务中修改。",
              "- C 的共享 embedding 结果、两名成员的独立核验、全员冻结主题均未完成；不输出正式 RQ2/RQ4 主题关联结论。", "",
              "## 文件入口", "", "- candidate_comparison.csv：候选指标及收敛情况。",
              "- topic_diagnostics.csv：各主题关键词、规模及自动检查标记。",
              "- 各候选目录：topic_terms.csv、representative_reviews.csv、assignments.csv、weights.npz、nmf.joblib。",
              "- human_review：两个独立核验空表、协调人对应表及抽样说明。",
              "- source_code 和 run_manifest.json：本次源码快照、配置、版本和输入输出哈希。", ""]
    (root / "baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(root / "report_manifest.json", {
        "source_run_manifest_sha256": sha256(root / "run_manifest.json"),
        "report_script_sha256": sha256(Path(__file__)),
        "files": [{"path": relative(p), "sha256": sha256(p)} for p in
                  (root / "baseline_summary.md", root / "candidate_review_notes.csv", root / "candidate_overview.png", root / "candidate_overview.svg")]})
    print(root / "baseline_summary.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "output/baseline_nmf/freeze_20261101_aspect_v2")
    report(parser.parse_args().run)

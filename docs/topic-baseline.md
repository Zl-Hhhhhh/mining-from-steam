# B 的主题基线与交接

本次完成任务 9，并为任务 11 准备第一轮核验材料。独立人工核验、任务 12 的全员主题冻结和正式 RQ2、RQ4 分析尚未执行。

## 数据与实验范围

使用 A 的 `data/cleaned/freeze_20261101/`，其中 `topic_fit.jsonl` 有 13,149 条，英文 7,145 条、简中 6,004 条；`association_data.jsonl` 有 19,395 条。实际窗口为 2026-05-02 至 2026-11-01，以冻结清单为准。

`analysis_common.py` 校验冻结 JSON/JSONL、清洗数据哈希、记录资格、数量和日期边界。Git 改变换行的清洗 JSONL 允许还原换行后核验，并同时记录本机字节哈希。CSV 不用于训练，原清单未被改写。

中英文各自训练 TF-IDF(有代表性的词和词组) + NMF(归纳共同主题)。尝试 6、10、14 个主题，每组用随机种子 17、42；两种语言的主题编号不能直接对应。

- v1：`config/nmf_baseline.json`，结果在 `output/baseline_nmf/freeze_20261101/`，保留初始实验记录。
- aspect_v2：`config/nmf_baseline_aspect.json`，结果在 `output/baseline_nmf/freeze_20261101_aspect_v2/`。根据首轮文本检查，增加泛泛评价和口头词的过滤，保留否定词；这是对照方案，不是已批准的主模型。
- 标点、链接和 Steam/HTML 标记在建模输入中标准化；英文用词切分，中文用 jieba，支持混合文字。过滤配置中已记录的游戏名称及别名。
- 推荐标签、投票、时长和作者编号不作为主题特征，也不作为候选模型选择依据。
- 主题权重先消除 NMF 成分的任意尺度，再按行归一化；这些权重不是校准后的概率。
- 低信息、零向量、主题权重过于分散或与主题相似度太低的评论保留为 `no_clear_theme`(无明确主题)，不强制归类。

主要方法参考 [scikit-learn 的 TF-IDF + NMF 主题示例](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html)。训练误差和自动词共现分数只作诊断，不能代替人工判断。

## 运行命令

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe -X utf8 scripts\nmf_baseline.py --config config\nmf_baseline_aspect.json
.\.venv\Scripts\python.exe -X utf8 scripts\report_nmf.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

训练脚本不会覆盖已完成的运行。重新实验时复制配置并指定新的 `output` 目录；生成报告不重新训练。每次运行保存源码、配置、模型、词表、评论顺序、输入输出哈希和实际依赖版本。

## 结果入口

- [本次结果摘要](../output/baseline_nmf/freeze_20261101_aspect_v2/baseline_summary.md)。
- [候选比较表](../output/baseline_nmf/freeze_20261101_aspect_v2/candidate_comparison.csv)。
- [随机种子稳定性](../output/baseline_nmf/freeze_20261101_aspect_v2/seed_stability.csv)。
- [候选诊断图](../output/baseline_nmf/freeze_20261101_aspect_v2/candidate_overview.png)。
- 每个候选目录内保存关键词、代表评论、候选主题标签及全部主题权重。公开表格不含作者 SteamID。

## 任务 11 的人工核验

`human_review/` 内有 `reviewer_A_blank.csv`、`reviewer_B_blank.csv` 两份相同空表，以及仅供协调人使用的 `coordinator_key.csv`。A/B 指两个独立核验席位；按任务分配由成员 A 和 B 各自填写，不能把同一人或 AI 的两次判断当成两名成员。

样本共 250 条，中英文各 125 条，包含主题中心、主题内随机、短文本及低置信度评论，并补足不同推荐标签。标签只在核验抽样时使用，核验表不显示原推荐标签、机器候选标签或作者编号。

1. 核验前约定候选标签定义及 `no_clear_theme`；英文和中文用不同命名空间，如 `english:T00` 与 `schinese:T00`。k=10、seed=17 只是预先指定的抽样锚点，不代表主模型已选定。
2. 两人独立填写 `theme_label`(本人判断的主题)、`aspect_summary`(关注点摘要)、`fit_rating_0_1_2`(该主题能否解释评论：0 不能，1 部分，2 明确)、`no_clear_theme`(是否无明确主题)和 `notes`(疑问)。不提前讨论逐条答案。
3. 协调人收到两份已完成表后，计算一致率、Cohen's kappa(扣除偶然一致后的指标)，整理分歧及需合并、拆分或重训的主题。每种语言单独计算，未填完的表会被脚本拒绝。
4. C 的 embedding 候选到位后，还需对应的共同核验与模型比较；当前 NMF 核验包不能代替这一步。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\review_topics.py --reviewer-a path\to\A_completed.csv --reviewer-b path\to\B_completed.csv --output output\topic_review\round1
```

上述命令仅生成一致率与分歧记录，不自动冻结主题。最终主题定义、模型选择及 `topics-v1` 版本须在任务 12 中由全员确认。后续 RQ2、RQ4 使用该共同版本，不把独立 NMF 的中英文同号主题当作相同主题。

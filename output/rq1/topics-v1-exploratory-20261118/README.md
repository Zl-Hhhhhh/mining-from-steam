# RQ1 主题发现与分布

本目录是基于 `topics-v1-exploratory-20261118` 的探索性结果。主题包仍未完成最终语义冻结，因此结果用于 RQ1 试运行和图表准备。

## 输入与口径

- 评论总数：19,486。游戏数：50。语言：english, schinese。
- 主结果使用 `fit_T01`–`fit_T08`。有主题证据评论：4,218，覆盖率：21.6%。
- `all_T01`–`all_T08` 敏感性结果有主题证据评论：4,879，覆盖率：25.0%。
- 主题比例在有主题证据的评论内归一化。无主题评论保留在覆盖率诊断中。
- 游戏类型主口径为项目配置中的 `project_primary_genre`。
- 主题生成没有使用推荐标签或游玩时长。本目录不支持因果解释。

## 结果文件

- `overall_topic_distribution.csv` 总体主题比例。
- `game_topic_distribution.csv` 游戏—主题长表。
- `genre_topic_distribution.csv` 项目类型—主题长表。
- `language_topic_distribution.csv` 语言—主题长表。
- `coverage_diagnostics.csv` 游戏和语言的覆盖率、置信度及样本状态。
- `representative_comments.csv` 每个主题和语言的代表句子。
- `sensitivity_overall.csv` fit、all 和高置信度子集的总体比较。

## 解释限制

- 当前 `fit` 覆盖率较低，不能把无主题证据评论解释为不涉及八个主题。
- 主题权重是句子证据的归一化贡献，不是概率或人工确认概率。
- 当前主题包中的低置信度记录需要在正式主题冻结后重新检查。
- 主题版本冻结后，应使用相同脚本重新生成正式 RQ1 结果。

主题版本：`topics-v1-exploratory-20261118`。运行参数记录在 `rq1_manifest.json`。

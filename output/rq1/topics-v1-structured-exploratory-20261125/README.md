# RQ1 主题发现与分布

本目录记录 `topics-v1-structured-exploratory-20261125` 的探索性结果。

## 输入与口径

- 评论总数：19,486。游戏数：50。语言：english, schinese。
- 主结果使用 `all_T01`–`all_T08`。
- `fit` 有主题证据评论 3,996 条（20.5%），`all` 有 5,301 条（27.2%）。
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
- `sensitivity_overall.csv` fit、all、固定 fit 样本的 all 权重和高置信度子集的总体比较。

## 解释限制

- 覆盖率表示当前主题规则能识别到证据的评论比例。
- 主题权重是句子证据的归一化贡献，不是概率或人工确认概率。
- 当前主题包中的低置信度记录需要在正式主题冻结后重新检查。
- 主题版本冻结后，应使用相同脚本重新生成正式 RQ1 结果。

主题版本：`topics-v1-structured-exploratory-20261125`。运行参数记录在 `rq1_manifest.json`。

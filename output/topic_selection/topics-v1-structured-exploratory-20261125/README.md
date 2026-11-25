# Steam 评论主题交付包

`topics-v1-structured-exploratory-20261125` 是供项目组复核和探索分析的版本。统一使用 T01–T08 作为一级主题编号。主题定义附有 4 个候选子主题及本地化与语言支持观察项，当前权重覆盖 8 个一级主题。

## 文件与读取方式

| 文件 | 用途 |
| --- | --- |
| [topic_v1.json](topic_v1.json) | 一级主题定义、候选子主题、例句和标注边界 |
| [review_topic_weights.csv](review_topic_weights.csv) | 评论级主题权重，以 `review_id` 连接主评论表 |
| [review_topic_weights.jsonl](review_topic_weights.jsonl) | 评论级权重、片段编号、证据计数及辅助路由 |
| [unit_topic_assignments.jsonl](unit_topic_assignments.jsonl) | 片段级主题、评分、置信度及原评论编号 |
| [summary.json](summary.json) | 新旧入口对照、统一门槛敏感性对照以及 Reviewer A 抽样结果 |
| [delivery_manifest.json](delivery_manifest.json) | 文件校验值、版本口径和已通过的检查 |

评论表覆盖全部 19,486 条评论，片段表共 34,491 条。`all_T01` 至 `all_T08` 汇总全部已匹配片段，5,301 条评论有主题证据。`fit_T01` 至 `fit_T08` 只汇总历史拟合口径下保留的片段，3,996 条评论有主题证据。权重表示同一条评论内各主题的相对证据贡献。空主题表示当前规则没有检出证据。

## 匹配口径与复核状态

沿用旧版主题原型和聚类限制。保留下来的历史片段使用旧标签，新增片段用 `.65` 的相似度门槛、`.06` 的前两名分差、簇限制和整段泛评拒判。`summary.json` 另列对所有片段统一使用严格门槛的比较，供判断入口修复的影响。

已逐条核对片段与评论的关联，重新聚合评论权重并校验 CSV 与 JSONL 一致。RQ1 已通过输入与总体汇总验证，RQ2–RQ4 的现有脚本需要接入该版本后再验证。

当前为探索性交付。部分模板项有明确主题线索，但分差不足，仍会被自动拒判。4 个候选子主题待独立语义复核，正式冻结前需确认它们的边界、名称及是否单独产出权重。

# Task10交付状态与目录整合

2026-11-13仅完成已交付内容的目录整合与引用更新；模型版本、输入、缓存、候选归属、诊断指标及历史报告仍对应原freeze_20261101-v1。

| Task10要求 | 当前入口与状态 |
| --- | --- |
| 多语言模型选择 | config/embedding_bge.json；BGE-M3固定修订，1024维CLS/L2/float32、8192 token。 |
| 批量编码、缓存与ID | scripts/embedding_bge.py；output/embedding-bge/freeze_20261101-v1/cache/。19392条向量，主表19486条，94条空文本保留索引。 |
| 候选算法比较 | scripts/cluster_bge.py；output/clustering-bge/freeze_20261101-v1/。15组KMeans/Ward/average候选，算法比较使用共同5000条；层次标签不外推。 |
| 规模、语言与代表评论 | candidate_metrics.csv、cluster_profiles.csv、representative_reviews.csv、report.html；另60条来源语言分层中心评论，待实际语言/语义核验。 |
| 干扰检查 | 源语言、书写系统、游戏、推荐标签AMI作拟合后诊断。泛评、语言和多方面长评混杂仍存在，未宣称干扰消除。 |
| 低置信度/无明确主题 | output/clustering-bge/freeze_20261101-v1/provisional_assignment_status.csv及规则；逐条暂定拒绝/边界阅读状态，人工标签/概率为空。 |
| 质量与复现 | output/task10-validation/freeze_20261101-v1/integration-20261113/记录迁移后缓存验证、测试、15组复现和Git检出检查。 |

模型选择、数据范围、运行命令及结果解释见[完整说明](embedding-bge.md)。在仓库根执行python -B scripts/embedding_bge.py verify即可验证缓存；交互报告位于output/clustering-bge/freeze_20261101-v1/report.html，下载后用浏览器打开。

向量、输入文本、候选逐条归属、规模统计及原HTML/图表按原字节保留。原始manifest和源代码/运行快照留作历史证据；当前活动配置和缓存manifest更新到仓库相对路径，新迁移清单另存。新文件哈希不能冒充旧运行脚本的哈希。

共同5000条、K=10：KMeans最大簇741、Ward791、average4987；average的较高silhouette不能抵消大簇退化。K=10只是核验锚点，不是最终主题数。gap<0.03及严格泛评规则未经校准，不能替代人工主题或概率。

原拟合资格排除部分有信息短评，长评可能有多个方面，模板未勾选项也可能干扰向量。Task11答卷已本地回收，后续仍需一致性统计、分歧裁决、合并/拆分/重做决定、最终主题语义及重采样稳定性验证。原CSV字节恢复差异仍未解决；此次使用校验过的冻结JSONL，不等于Task6全部质量验收完成。

Task11后续更新（2026-11-14）：四份原始答卷及双方分工复核草稿已收齐，讨论前一致性统计保持不变；180条评论和30组已有[依据复核意见的AI裁决](../output/topic_review/steam-review-20261107-v1/adjudicated-20261114/裁决说明.md)，B-Lin已认可。待候选模型映射、稳定性检查及全员冻结；Task10模型与候选未改动。

# Task10：共享多语言Embedding与聚类交付

完整交付目录：[task10/](../task10/)。先读[操作与目录说明](../task10/README.md)和[逐项交付状态](../task10/HANDOFF_STATUS.md)。

| 交付物 | 位置 |
| --- | --- |
| 固定模型、生成与缓存复用脚本 | task10/config/embedding.json、task10/code/task10.py |
| 实际向量与评论ID | task10/artifacts/（19392×1024，float32） |
| 聚类候选 | task10/results/clustering/（15组KMeans/Ward/average候选） |
| 中英文来源分层代表评论 | task10/bilingual_representatives.csv（60条，待人工核对） |
| 聚类质量和规模统计 | task10/results/clustering/candidate_metrics.csv、cluster_profiles.csv |
| 交互结果展示 | task10/results/clustering/report.html（下载后用浏览器打开） |
| 低置信度/无明确主题暂定状态 | task10/provisional_assignment_status.csv、provisional_status_rules.json |
| 运行与复现检查 | task10/validation/、results/中的原始证据 |

BGE-M3固定修订5617a9f61b028005a4858fdac845db406aefb181；数据来自8381107冻结JSONL副本。主表19486条，94条空文本无向量但保留索引。无需重算embedding即可重跑聚类。

这是Task10的候选交付。K=10只是人工核验锚点；泛评/语言/长评多方面等问题仍存在。Task11材料另行分发，两名成员独立标注与命名完成前，不报告人工一致率、最终主题体系或主题准确率。

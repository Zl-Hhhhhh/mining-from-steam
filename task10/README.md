# Task10目录已整合

Task10已归入仓库统一结构，请使用[完整目录与运行说明](../docs/embedding-bge.md)、[逐项交付状态](../docs/task10-handoff-status.md)或[简要索引](../docs/task10-bge-m3.md)。

| 旧入口 | 当前入口 |
| --- | --- |
| code/task10.py | scripts/embedding_bge.py |
| code/cluster_bge_portable.py | scripts/cluster_bge.py |
| code/export_review_status.py | scripts/export_bge_review_status.py |
| config/ | config/embedding_bge.json、config/clustering_bge.json |
| inputs/ | data/derived/embedding-bge/freeze_20261101-v1/ |
| artifacts/ | output/embedding-bge/freeze_20261101-v1/cache/ |
| results/encoding/ | output/embedding-bge/freeze_20261101-v1/ |
| results/clustering/ | output/clustering-bge/freeze_20261101-v1/ |
| validation/ | tests/test_embedding_bge_pipeline.py及output/task10-validation/freeze_20261101-v1/ |

交互报告位于[新报告入口](../output/clustering-bge/freeze_20261101-v1/report.html)，下载HTML后用浏览器打开。现有向量和历史结果保持原字节；这里不再保留第二套活动脚本或缓存。

所有新命令从仓库根执行：

```powershell
python -B scripts/embedding_bge.py verify
python -B scripts/embedding_bge.py cluster --output output/clustering-bge/my-reproduction
```

需要回看原独立目录时，访问[原交付提交0f22b97](https://github.com/Zl-Hhhhhh/mining-from-steam/tree/0f22b97a7c0d1b1766ea449a84e1802017e2837b/task10)。迁移是组织结构调整，不代表已完成最终主题或人工裁决。

# BGE-M3 正式评论向量：Step 3

模型决定见 `docs/embedding-model-selection-20261105.md`；固定设置见 `config/embedding_bge.json`。本轮只准备向量，不执行聚类或生成正式主题标签。

## 输入与表示

固定来源提交 `8381107e811675f34bab3727b1b13a2c6d07a735`，数据冻结 `freeze_20261101`。使用上一轮独立核对的JSONL副本，先重新核对准备清单内的所有本地文件和规范化文本哈希。

共同规范化处理URL、HTML/Steam标记和空白，保留大小写、中文、否定、emoji、咒骂与游戏词汇；与此前Pilot保持一致，不套用NMF专属停用词。

- 全体主表19,486条保留索引。
- 其中91条源文本为空，不生成向量。
- 非空源评论19,395条中，另3条清除格式后为空，同样保留索引但不生成向量。
- 实际为19,392条评论保存1024维向量。每条ID都有自己的行号，空文本不使用零向量占位。
- 精确相同的规范化文本只编码一次，再映射回各评论ID；不做忽略大小写或语义去重，不删除重复评论。

本轮为**整条评论的完整dense向量**，没有将长评静默截断，也没有进行分块平均。原生8192上限内，实际最大3993 tokens；139条超过512 tokens的长评通过缩小批量完整处理。单个整评向量仍可能混合多个方面，后续主题实验需考虑这个局限。

## 缓存与恢复

`.cache-topic/embeddings-bge-formal/<cache_key>/` 保存：

- `review_vectors.npy`：19,392 × 1024 的float32单位向量，按来源顺序。
- `review_ids.json`：与矩阵行严格对应的字符串ID。
- `unique_vectors.npy`：精确文本复用的中间矩阵。
- `completed.npy`：中间矩阵的逐行完成标记。
- `manifest.json`：模型/文本/版本身份、输入顺序、校验值、进度和耗时。

缓存身份包含固定模型修订、全部有序ID/文本、精度、前缀、池化、归一化、上限和依赖版本。每批写入向量后才标记完成；中断后只续算未完成文本。完整缓存复用前检查哈希、ID、形状和单位范数。

此前BGE Pilot中的600条真实评论向量，经修订、库版本、精度、前缀、原文、ID和长度核对后直接复用。构造句不混入正式数据。Pilot上限512、本轮8192，但复用文本均未在任一上限处截断，因此内容一致。

## 结果入口

`output/embedding-bge/freeze_20261101-v1/`：

- `summary.md`：本轮结果与限制。
- `run_summary.json`：实际计数、时间、内存、版本、缓存位置及哈希。
- `review_index.csv`：主表全部ID到向量行号的映射，保留空文本状态、源拟合资格与表达诊断标记；这些字段不是编码特征。
- `encoding_plan.json`：长度分布、精确文本数、复用量及批量计划。
- `validation.json`：独立实现/数据检查，不是主题准确率。
- `source_code/`：本轮源文件与依赖快照。

向量缓存在忽略目录，不随普通代码提交复制大权重或向量。交接电脑时需单独共享缓存及其哈希，或按相同冻结文本重建。原CSV字节恢复问题仍未在此修复；本流程使用已核对的JSONL。

## 运行与验证

在仓库根目录使用已有环境：

```powershell
.\.venv-topic\Scripts\python.exe -X utf8 scripts/embedding_bge.py
# 相同配置重复执行会核对并复用完整缓存，不重新编码。
```

选型与编码配置改变时，使用新的输出目录保留前后版本：

```powershell
.\.venv-topic\Scripts\python.exe -X utf8 scripts/embedding_bge.py --config config/embedding_bge.json --output output/embedding-bge/new-version
.\.venv-topic\Scripts\python.exe -m unittest discover -s tests -p test_embedding_bge.py -v
```

`run_summary.json`标记Step 3完成后，才进入Step 4的聚类。语言归属、主题边界、多方面处理以及最终人工核验仍是后续任务；完整编码不代表这些研究问题已解决。

# Task12：句子／片段级 BGE-M3 编码

本轮完成编码入口适配、小批量检查和全量句子向量缓存，供 B 继续进行聚类与稳定性比较。输入来自提交 `1ee364d5cf41d58b829d56f2ef44dbf7961d9abf` 中的 Task12 handoff；保留历史整评向量与主题草案，不在编码阶段分配主题。

## 队友直接使用缓存

更新 `main` 后，从[向量缓存目录](../output/embedding-bge-sentences/task12-20261117-v1/README.md)读取三个正式文件即可；输入单元、评论映射和主题草案也都在同一仓库，不需要单独传文件。读取缓存和聚类不需要 BGE 模型权重、Torch 或 GPU。

先在项目原有、含 NumPy 的环境中校验：

```powershell
.\.venv\Scripts\python.exe -B scripts/embedding_bge_sentences.py verify
```

再用 `unit_ids.json` 对应向量行，关联原始 `sentence_units.jsonl` 的元数据；用 `fit_candidate=true` 的 7,269 个单元拟合，保留其余 1,710 个向量供后续应用。不要自行按 ID 排序或把去重单元数当作评论数。数据内容的注意点见[近邻阅读记录](../output/embedding-bge-sentences/task12-20261117-v1/pilot/reading_notes.md)。

## 文件与对应关系

输入固定为 `output/topic_selection/task12-20261114/handoff/sentence_units.jsonl`：8,979 个去重单元，其中 7,269 个 `fit_candidate=true`，其余 1,710 个仍需编码。它们覆盖 4,885 条原评论；`review_to_units.json` 保留全部 19,486 条评论的映射。没有抽取出单元的评论不能直接记为“无主题”。

输出目录：`output/embedding-bge-sentences/task12-20261117-v1/`。

| 文件 | 用途 |
| --- | --- |
| `sentence_vectors.npy` | 8,979 × 1,024，float32，L2 归一化的密集向量 |
| `unit_ids.json` | 第 i 个 ID 对应向量第 i 行，严格沿用输入 JSONL 的顺序 |
| `sentence_embedding_manifest.json` | 模型、输入、ID 顺序、向量校验值、运行环境和编码统计 |
| `pilot/report.html` | 48 个真实单元的跨书写系统近邻及 6 对构造对照句；用浏览器打开 |
| `pilot/pilot_summary.json` | 完整小批量示例及检查记录 |
| `pilot/reading_notes.md` | 对真实近邻的定性阅读记录与后续核验注意点 |
| `validation/` | 测试、缓存重复使用及全量交付校验记录 |

向量的每一行代表唯一句子／片段，不能直接视为一条独立评论或用于最终主题比例。相同单元关联的所有原评论、拟合标记和其他元数据以原始 handoff 为准。

## 固定模型与设置

- 模型：`BAAI/bge-m3`，修订 `5617a9f61b028005a4858fdac845db406aefb181`。
- CLS pooling、L2 归一化、float32，无检索提示词；使用普通 Transformer / Pooling / Normalize 模块，不加载远程代码。
- 最长 8,192 tokens；先检查全部输入的实际 token 数，不允许静默截断。本批最长 1,543 tokens，只有 2 个单元超过 512 tokens。
- CPU 2 线程，单批最多 8 个单元；按长度排序，通常每批 padding 后不超过 4,096 tokens，单个较长单元可单独编码。
- 编码前检查配置中列出的 10 个本地模型文件的 SHA-256。配置见 `config/embedding_bge_sentences.json`，依赖快照沿用 `requirements-embedding-lock.txt`。

8 个候选主题及辅助类别只供后续语义核验参考。此脚本不读取主题目录、人工作答或推荐标签，不强制聚成 8 类，也不复用旧整评向量。

## 运行与缓存

在仓库根目录运行。实际编码使用现有 Python 3.12 编码环境；下例 `.venv-topic` 代表已安装锁定依赖的环境。`verify` 只需要 NumPy，无需模型权重、Torch 或 GPU。

```powershell
.\.venv-topic\Scripts\python.exe -B scripts/embedding_bge_sentences.py preflight
.\.venv-topic\Scripts\python.exe -B scripts/embedding_bge_sentences.py pilot --model-snapshot '本地快照目录/5617a9f61b028005a4858fdac845db406aefb181'
.\.venv-topic\Scripts\python.exe -B scripts/embedding_bge_sentences.py encode --model-snapshot '本地快照目录/5617a9f61b028005a4858fdac845db406aefb181'
.\.venv-topic\Scripts\python.exe -B scripts/embedding_bge_sentences.py verify
```

模型须提前缓存到本地；脚本离线运行，不自动下载。未提供 `--model-snapshot` 时，只在配置的本地模型缓存中查找固定修订。权重不随交付包分发。

中间结果保存在 `.cache-topic/encoding-bge-sentences/<fingerprint>/`，每批完成即刷新向量和完成标记。指纹涵盖输入、编码配置、运行库版本、模型文件及生成脚本；同一指纹可断点续跑，小批量的 48 条向量直接供全量复用。并发运行同一任务会被文件锁阻止。中间缓存不提交 Git。

已有完整输出时，`encode` 会校验缓存后返回，不重新加载编码器。输入、ID 顺序或输出校验值不符时立即停止；不覆盖损坏或来源不明的输出。确需新实验时，另建输出版本，保留历史文件。

## 小批量检查的解释

48 个真实单元中，粗略规则将 21 个记为 `Han-dominant`、27 个记为 `Latin-script`，覆盖短文本、长片段以及拟合／非拟合单元。后者只要求存在若干拉丁字母且没有汉字，不能证明其实际语言或主要书写系统；这些提示不用于训练或过滤。抽样中可见俄语和西班牙语内容，不能把来源语言字段直接当成实际文本语言。

6 对人为构造的中英文对照句均将对应译句排在 6 个中文候选中的首位，余弦相似度约 0.755–0.963。这是运行与语义对齐的基本检查，不是独立测试集上的准确率。构造句单独保存，不进入 8,979 行正式向量。

人工阅读真实近邻发现：上手难度等语义能对齐；反讽可能接近普通好评，泛评及多方面片段仍会混杂。48 条内的最近邻还可能只是“相对最近”，不代表相似主题。后续聚类应检查这些问题，不能据此冻结主题或设定统一的相似度阈值。

## 交给下一阶段

1. 将三个正式输出文件与原 handoff 配套交给 B，用 `unit_ids.json` 连接输入元数据，禁止按字典排序重排向量。
2. 拟合候选由 `fit_candidate` 选择；保留全部单元供后续应用与多主题汇总。
3. 按既定计划比较 K=6/8/10、seed=17/42 及 80% 重采样稳定性，并检查语言、游戏名称、情感和簇规模；本轮未执行这些实验。
4. 用句子级候选定向确认 `topic-list-draft`，记录必要的边界调整与合并拆分，再由全员确认 `topics-v1`。本轮没有冻结主题、生成最终标签或主题比例。

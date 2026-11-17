# Task12 句子级 BGE 缓存 · 2026-11-17

状态：全量编码及校验通过。使用原 Task12 handoff 的 8,979 个去重单元，保留输入顺序、文本、拟合标记和评论映射。

| 文件 | 内容 |
| --- | --- |
| [sentence_vectors.npy](sentence_vectors.npy) | 8,979 × 1,024，float32，L2 归一化 |
| [unit_ids.json](unit_ids.json) | 第 i 个 ID 对应向量第 i 行 |
| [sentence_embedding_manifest.json](sentence_embedding_manifest.json) | 固定模型、输入、向量与 ID 校验值及运行统计 |
| [pilot/report.html](pilot/report.html) | 48 个真实单元及 6 对构造对照句，浏览器打开 |
| [pilot/reading_notes.md](pilot/reading_notes.md) | 近邻定性阅读记录，包含反讽、混杂和语言诊断的局限 |
| [validation/checks.json](validation/checks.json) | 独立校验记录 |

固定 `BAAI/bge-m3` 修订 `5617a9f61b028005a4858fdac845db406aefb181`，CLS pooling、归一化、float32、CPU 2 线程。最长实际输入 1,543 tokens，没有截断。

48 条小批量向量逐位复用于全量，另外编码 8,931 条；全量阶段约 20.0 分钟，编码进程报告的峰值工作集约 2.18 GiB。全部向量有限且归一化通过，范数范围约 0.99999982–1.00000012。已有成品再次运行编码命令约 0.24 秒完成校验，未加载编码器或改写输出。

输入覆盖 4,885 / 19,486 条原评论，包含 7,269 个拟合候选和 1,710 个其他单元；后续应依原 handoff 的 `fit_candidate` 筛选拟合行，用 `review_to_units.json` 汇总多主题。没有抽取出单元的原评论不能直接判为无主题。

本轮没有执行句子聚类、冻结主题或生成最终主题比例。构造对照句不混入正式向量；小样本最近邻不代表主题准确率。

完整运行方法与下一阶段说明见[句子级编码文档](../../../docs/embedding-bge-sentences.md)。

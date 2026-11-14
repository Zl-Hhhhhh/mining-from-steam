# G01–G30 来源重建

仓库没有新版原始协调人对照表；现已用例子原文、逐条归属及 BGE 中心记录重建全部30组的候选来源。

[逐例证据与来源哈希](group_source_mapping.json)。这份结果属于归档证据重建，K=10、seed=17 仍是核验锚点。

| G编号 | 模型 | 数据语言范围 | K | seed | 原主题／簇编号 |
| --- | --- | --- | ---: | ---: | --- |
| G01 | TF-IDF + NMF | 中文 | 10 | 17 | schinese:T00 |
| G02 | TF-IDF + NMF | 中文 | 10 | 17 | schinese:T03 |
| G03 | TF-IDF + NMF | 中文 | 10 | 17 | schinese:T07 |
| G04 | TF-IDF + NMF | 中文 | 10 | 17 | schinese:T02 |
| G05 | BGE-M3 + KMeans | 中英共享 | 10 | 17 | 4 |
| G06 | BGE-M3 + KMeans | 中英共享 | 10 | 17 | 2 |
| G07 | BGE-M3 + KMeans | 中英共享 | 10 | 17 | 9 |
| G08 | TF-IDF + NMF | 英文 | 10 | 17 | english:T07 |
| G09 | BGE-M3 + KMeans | 中英共享 | 10 | 17 | 1 |
| G10 | TF-IDF + NMF | 英文 | 10 | 17 | english:T06 |
| G11 | TF-IDF + NMF | 中文 | 10 | 17 | schinese:T09 |
| G12 | TF-IDF + NMF | 英文 | 10 | 17 | english:T03 |
| G13 | TF-IDF + NMF | 英文 | 10 | 17 | english:T02 |
| G14 | BGE-M3 + KMeans | 中英共享 | 10 | 17 | 7 |
| G15 | BGE-M3 + KMeans | 中英共享 | 10 | 17 | 3 |
| G16 | BGE-M3 + KMeans | 中英共享 | 10 | 17 | 6 |
| G17 | BGE-M3 + KMeans | 中英共享 | 10 | 17 | 8 |
| G18 | TF-IDF + NMF | 英文 | 10 | 17 | english:T05 |
| G19 | TF-IDF + NMF | 英文 | 10 | 17 | english:T01 |
| G20 | TF-IDF + NMF | 中文 | 10 | 17 | schinese:T05 |
| G21 | TF-IDF + NMF | 英文 | 10 | 17 | english:T09 |
| G22 | TF-IDF + NMF | 英文 | 10 | 17 | english:T00 |
| G23 | TF-IDF + NMF | 中文 | 10 | 17 | schinese:T01 |
| G24 | TF-IDF + NMF | 中文 | 10 | 17 | schinese:T04 |
| G25 | TF-IDF + NMF | 英文 | 10 | 17 | english:T08 |
| G26 | BGE-M3 + KMeans | 中英共享 | 10 | 17 | 5 |
| G27 | TF-IDF + NMF | 中文 | 10 | 17 | schinese:T06 |
| G28 | TF-IDF + NMF | 中文 | 10 | 17 | schinese:T08 |
| G29 | TF-IDF + NMF | 英文 | 10 | 17 | english:T04 |
| G30 | BGE-M3 + KMeans | 中英共享 | 10 | 17 | 0 |

主题与簇编号从0开始。NMF来自 `output/baseline_nmf/freeze_20261101_aspect_v2/`，BGE来自 `output/clustering-bge/freeze_20261101-v1/full_kmeans_k10_s17/`。

## 核对口径

- 180个例子对应179条不同原文，均可在清洗主表中按原文匹配；只统一空白字符，没有模糊搜索或语义猜测。
- 比较仓库两个NMF版本和全部BGE候选。每个模型／语言家族同时满足六例同主题、各组主题不同的条件，BGE还需匹配发布的中心例子，最终各剩一个锚点。
- `no_clear_theme` 是NMF暂定拒绝结果，回溯使用其原始 `topic_index`；BGE负编号视为不适用，不当作簇。
- 同文多条评论保留所有候选ID；这不影响已核实的组级主题归属，但没有恢复不存在的原始抽样ID清单。
- 源语言标签不代替实际语言核验；此表不选择主模型，不冻结最终主题。

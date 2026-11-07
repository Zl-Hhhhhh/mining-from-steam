# Task 10：共享中英文Embedding与聚类

本目录交付固定版本BGE-M3共享向量、缓存生成脚本、聚类候选、代表评论和质量/规模统计。**当前是候选方案，尚未冻结最终主题。** 人工核验（Task11）的两份空白标注包另行分发，不在此次提交中。

## 先查看结果

- [交互聚类报告](results/clustering/report.html)：下载HTML到本地，用Edge或Chrome打开。可以切换算法、K、簇、语言，查看中心/随机评论及统计；不需要Python或模型。
- [逐项交付状态与限制](HANDOFF_STATUS.md)：对照Task10要求说明已完成部分与待核验事项；也提供可本地打开的[HTML版](HANDOFF_STATUS.html)。
- [候选质量指标](results/clustering/candidate_metrics.csv)、[簇规模与构成](results/clustering/cluster_profiles.csv)、[中心/随机评论](results/clustering/representative_reviews.csv)。
- [双语来源分层代表表](bilingual_representatives.csv)：10个BGE候选簇，每簇按API来源语言选中英文各3条中心评论，共60条。来源语言还需人工核对，不等于已经确认的主题。

## 目录

```text
task10/
  README.md                        操作入口
  HANDOFF_STATUS.md / .html         逐项交付说明及结果边界
  code/
    task10.py                      验证、下载固定权重、批量编码、聚类入口
    cluster_bge_portable.py         KMeans / Ward / average候选实验
    export_review_status.py        逐条待核验状态再生成
  config/                          固定模型和聚类参数
  artifacts/
    review_vectors.npy             实际19392×1024 float32缓存，79,429,760字节
    review_ids.json                每一行向量对应的评论ID
    embedding_manifest.json        模型修订、相对路径及缓存校验值
  inputs/
    model_reviews.jsonl            校验过的规范化编码输入
    master_index.jsonl             全19486条ID及拟合后诊断用精简字段
  results/
    encoding/                      原始编码运行证据
    clustering/                    15组候选、逐条归属、统计、图表和交互报告
  bilingual_representatives.csv     60条来源语言分层中心评论
  provisional_assignment_status.csv 全19486条暂定状态；人工字段为空
  provisional_status_rules.json     未校准的泛评/边界读取规则
  validation/                      缓存、复现和运行逻辑检查
  source_lineage.json               输入来源与哈希
  requirements-cluster.txt          复用缓存进行聚类的依赖
  requirements-lock.txt             原编码环境完整依赖快照
  FILE_MANIFEST.json               本目录逐文件大小与SHA256
```

## 模型、数据与实验范围

模型为`BAAI/bge-m3`，修订`5617a9f61b028005a4858fdac845db406aefb181`，1024维、CLS pooling、L2归一、float32、无附加指令、最大8192 token。选择依据是此前中英文真实近邻对照及人工阅读后的决定；不声称对所有语境最优。

数据固定于提交`8381107e811675f34bab3727b1b13a2c6d07a735`的JSONL副本。主表19486条，源空文本91条、标记清理后另空3条，共94条没有向量但保留状态；实际编码19392条，精确文本去重后15335种文本。输入和ID映射均可核验。

15组候选：全拟合集KMeans（K=6/10/14，seed=17/42）6组；共同5000条上KMeans、Ward层次聚类和average层次聚类（各K=6/10/14）9组。拟合集为沿用source fit_eligible且有词汇内容的13145条；KMeans应用到全部19392条已编码评论。共同样本中英文来源各2500条、50款游戏；silhouette共用1500条。层次聚类只给共同5000条标签，不外推其他评论。K=10、seed17只是浏览/核验锚点。

## 直接复用缓存

以下命令从本目录`task10`执行。只验证缓存需要Python与numpy；重跑聚类安装轻量依赖即可，无需GPU、模型权重或原Pilot缓存。原环境使用Python3.12。

```powershell
cd task10
python -m pip install -r requirements-cluster.txt
python -B code/task10.py verify
python -B code/task10.py cluster --output results/my-reproduction
```

输出目录须为新目录，避免覆盖证据。验证包含输入哈希、模型修订、向量哈希、ID顺序、float32及单位范数。相对路径可随整个目录移动。`.gitattributes`保留原始字节，避免Windows换行转换破坏输入校验值。

重新生成暂定状态（仅Python标准库，默认基于全量KMeans K=10 seed17）：

```powershell
python -B code/export_review_status.py --out results/status-reproduced.csv
```

该脚本不接受将层次聚类未覆盖评论当作已分组来外推。状态不是人工主题或置信度概率。

## 确实需要重新编码时

实际缓存已经随目录交付，通常不用重算。若另行重建，先在独立环境按`requirements-lock.txt`安装编码依赖（其中CPU版torch需要对应CPU包源）；原环境记录不保证其他平台逐版本可安装。随后运行：

```powershell
python -B code/task10.py download
python -B code/task10.py encode --offline
```

已存在缓存时`encode`只验证并复用。**需要重建时先保留原缓存和manifest副本，再将原review_vectors.npy移出artifacts；不要直接覆盖原证据。** 编码固定模型修订，按长度组批（最多8条、4096 padded token预算，单条保留8192上限），精确文本去重、检查点恢复，再按ID顺序映射。不会静默截断超长文本。模型权重和`.encoding-work/`不随提交提供。

重建数值可能受运行环境影响，结束后使用新生成的配置并另存结果：

```powershell
python -B code/task10.py cluster --cluster-config config/clustering-reproduced.json --output results/new-cache-run
```

原真实CPU编码使用4线程，耗时约37分钟、进程峰值约2.9GiB；这是原环境记录。复用向量的聚类另有实际运行耗时记录，不能据此保证其他电脑速度。

## 怎样解释这些结果

共同5000条、K=10时：KMeans最大簇741条，Ward最大簇791条；average最大簇4987条，其较高silhouette对应退化的大簇，不能据此判为最好。全拟合集KMeans K=10两个seed的ARI约0.958；种子稳定不等于语义有效或重采样稳定。

语言、书写系统、游戏和Steam推荐标签AMI都是拟合后诊断；推荐标签未参与拟合/选K，也不是情绪金标准。已经观察到泛赞、语言和多方面长评混杂；部分有信息短评被原拟合资格排除、模板未勾选项也可能影响向量，仍需人工核验。

暂定状态按空文本→纯符号→严格泛评规则→中心余弦gap<0.03边界→其他候选记录。gap和泛评规则未经人工校准，不是概率；`human_*`与`confidence_probability`列为空。混杂簇不能整体判为无主题。最终主题数、名称、多标签方案和合并/拆分/重训决定需等Task11。

## 验证与历史证据

```powershell
python -B -m unittest discover -s validation -p 'test_task10_portable.py' -v
```

缓存已逐字节核对；独立路径实际重跑15组，簇标签、拟合资格与边界标记逐条一致，分数差小于1e-6。编码恢复/去重/ID映射另用合成编码器测试，不能当作真实模型质量测试。本次不重复计算全量真实embedding。

`results/`保留原运行摘要、代码快照与绝对机器路径作追溯，阶段3摘要中的“clustering_pending”是当时的历史状态；当前聚类已见`results/clustering/`。实际可迁移入口是`code/`和相对配置。历史报告的中心例子做过阅读去重，重跑入口默认保留原始排序，簇归属不受影响。

原CSV恢复/字节差异未由此次提交解决；本次使用的是校验过的冻结JSONL，不能据此宣称Task6全部验收完成。没有真实人工标注、一致率、最终主题冻结或后续RQ统计结论。

# BGE-M3共享Embedding与聚类（Task10）

Task10已整合到仓库标准目录，与NMF路线并列。所有命令从仓库根目录执行。固定模型、向量、候选归属和历史证据保持原版本；本次目录迁移不冻结最终主题。

## 查看交付

- [交互聚类报告](../output/clustering-bge/freeze_20261101-v1/report.html)：下载到本地后用Edge/Chrome打开；可按算法、K、簇和语言查看中心/随机评论，无需安装Python。
- [逐项状态与限制](task10-handoff-status.md)（另有[本地HTML版](task10-handoff-status.html)）。
- [候选质量统计](../output/clustering-bge/freeze_20261101-v1/candidate_metrics.csv)、[规模和语言构成](../output/clustering-bge/freeze_20261101-v1/cluster_profiles.csv)、[中心与随机评论](../output/clustering-bge/freeze_20261101-v1/representative_reviews.csv)。
- [60条来源语言分层中心评论](../output/clustering-bge/freeze_20261101-v1/bilingual_representatives.csv)、[全量暂定状态](../output/clustering-bge/freeze_20261101-v1/provisional_assignment_status.csv)。来源语言及语义仍需人工核验。
- [迁移与复现验证](../output/task10-validation/freeze_20261101-v1/integration-20261113/README.md)、[文件迁移映射](../output/task10-validation/freeze_20261101-v1/migration_manifest_20261113.json)。

## 当前目录

```text
scripts/embedding_bge.py                  验证、下载固定权重、编码和聚类总入口
scripts/cluster_bge.py                    KMeans / Ward / average实验
scripts/export_bge_review_status.py       待核验状态生成
config/embedding_bge.json                 模型、输入、缓存和批次配置
config/clustering_bge.json                候选算法、输出与诊断配置
data/derived/embedding-bge/freeze_20261101-v1/
    model_reviews.jsonl                  规范化编码输入
    master_index.jsonl                   全19486条精简主表
output/embedding-bge/freeze_20261101-v1/
    cache/review_vectors.npy             实际19392×1024 float32向量
    cache/review_ids.json                 向量行对应的评论ID
    cache/embedding_manifest.json         当前相对路径和校验值
    ...                                  原编码记录与代码快照
output/clustering-bge/freeze_20261101-v1/
    report.html                          交互报告
    candidate_metrics.csv                15组候选指标
    cluster_profiles.csv                 规模、语言/推荐构成和游戏集中度
    */assignments.csv                    每组逐条归属和边界标记
    bilingual_representatives.csv         来源语言分层中心阅读表
    provisional_assignment_status.csv    全量暂定状态；人工字段为空
output/task10-validation/freeze_20261101-v1/                              迁移前及迁移后的验证证据
tests/test_embedding_bge_pipeline.py      缓存复用、恢复/去重/ID和长度检查
requirements.txt                         日常分析和缓存聚类依赖
requirements-embedding-lock.txt          原CPU编码环境依赖快照
```

旧目录的路径对应表见[仓库 README](../README.md#旧目录迁移与复现)。原独立交付完整保存在[提交 `0f22b97`](https://github.com/Zl-Hhhhhh/mining-from-steam/tree/0f22b97a7c0d1b1766ea449a84e1802017e2837b/task10)。

## 直接复用缓存

验证仅需 NumPy。重跑聚类使用日常分析环境，无需 GPU 或模型权重。

```powershell
python -m pip install -r requirements.txt
python -B scripts/embedding_bge.py verify
python -B scripts/embedding_bge.py cluster --output output/clustering-bge/my-reproduction
```

输出目录必须为新目录。相对配置以仓库根解析；显式`--output`和`--cluster-config`按调用时工作目录解析。重跑不会覆盖freeze_20261101-v1中的历史结果。验证检查输入哈希、模型修订、ID顺序、向量哈希、float32、有限值和单位范数。Git字节保护规则覆盖输入、缓存及证据文件，Windows换行转换不会破坏原校验值。

单独运行引擎或重新导出暂定状态：

```powershell
python -B scripts/cluster_bge.py --config config/clustering_bge.json --output output/clustering-bge/another-run
python -B scripts/export_bge_review_status.py --out output/clustering-bge/status-reproduced.csv
python -B -m unittest discover -s tests -p test_embedding_bge_pipeline.py -v
```

`export_bge_review_status.py`默认基于已交付的全量KMeans K=10、seed17；不把层次聚类之外的评论当作有标签。`requirements.txt` 也涵盖 NMF 依赖。全量测试命令见[环境验证说明](local-environment.md#验证)。

## 模型、数据与实验范围

模型BAAI/bge-m3，固定修订5617a9f61b028005a4858fdac845db406aefb181；1024维、CLS pooling、L2归一、float32、无附加指令、最大8192 token。选择源于此前Jina/BGE真实近邻对照和人工阅读后的决定。

输入源自8381107冻结JSONL副本。主表19486条，源空文本91条、标记清理后另空3条，共94条无向量但保留状态。实际编码19392条，精确去重后15335种文本；向量缓存79,429,760字节，SHA256为693c02eedb156efbaf50b3b13ba171f40caa50b15f7ee763b3a307ee26788d9b。

拟合范围沿用source fit_eligible且有词汇内容的13145条。全拟合集KMeans（K=6/10/14，seed17/42）6组；共同5000条上的KMeans、Ward、average（各K=6/10/14）9组。共同样本来源中英文各2500条、50款游戏，silhouette共用1500条。KMeans应用到全部19392条已编码评论，层次聚类只对共同5000条给标签。K=10/seed17是浏览与核验锚点，未定为最终主题数。

共同5000条、K=10：KMeans最大簇741条，Ward最大簇791条；average最大簇4987条，较高silhouette对应退化大簇，不能据此判为最佳。全拟合集KMeans K=10两种seed的ARI约0.958；种子稳定不是语义有效性或重采样稳定性。

## 确需重新编码时

原编码环境使用 Windows x64 和 Python 3.12。在独立环境中先从 PyTorch CPU 软件源安装固定版 `torch`，再安装完整编码依赖。

```powershell
py -3.12 -m venv .venv-topic
.\.venv-topic\Scripts\python.exe -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
.\.venv-topic\Scripts\python.exe -m pip install -r requirements-embedding-lock.txt
```

先保留原缓存和 manifest，再将原 `review_vectors.npy` 移出活动缓存目录，随后运行以下命令。

```powershell
.\.venv-topic\Scripts\python.exe -B scripts/embedding_bge.py download
.\.venv-topic\Scripts\python.exe -B scripts/embedding_bge.py encode --offline
.\.venv-topic\Scripts\python.exe -B scripts/embedding_bge.py cluster --cluster-config config/clustering_bge_reproduced.json --output output/clustering-bge/new-cache-run
```

已有向量时encode只校验并复用。权重放在被忽略的.cache-topic/models-bge/，检查点放在.cache-topic/encoding-bge/。编码固定模型修订、精确文本去重、按长度组批（最多8条，4096 padded token预算；单条可到8192），支持恢复并映射全部ID，不静默截断超长输入。新环境数值哈希可能不同，使用新生成的clustering_bge_reproduced.json并另存结果。

原4线程CPU编码约37分钟，进程峰值约2.9GiB；这仅是原环境记录。目录迁移只做了缓存读取与聚类复现，没有重新编码全部真实评论。合成编码器测试验证恢复和映射控制流，不代表模型语义效果。

## 结果边界与历史记录

语言、书写系统、游戏和Steam推荐标签的AMI是拟合后诊断；推荐标签未参与拟合或选K，也不是情绪金标准。仍存在泛赞/语言混杂、多方面长评、模板未勾选项干扰，以及有信息短评被原拟合资格排除等问题。

暂定状态按空文本→纯符号→严格泛评规则→gap<0.03边界→其他候选记录。gap和泛评规则未经人工校准，不是概率；human_*与confidence_probability列为空。混杂簇不能整体判为无主题。

历史输出、原脚本快照、旧路径/哈希和旧阶段状态原样留存。当前运行入口是根scripts/与config/，可用文件索引见迁移manifest；原FILE_MANIFEST在output/task10-validation/freeze_20261101-v1/pre_migration_FILE_MANIFEST.json。历史报告的中心例子做过阅读去重，重跑入口仍输出原始中心排序；原交付报告和归属保持原字节。

Task11四份答卷已于2026-11-13本地归档，一致性统计、分歧裁决与最终主题确认尚待开展；答卷及协调人材料未随目录整合上传。原CSV恢复/字节差异也未由本次整合解决，本次实际使用的是已校验的冻结JSONL。

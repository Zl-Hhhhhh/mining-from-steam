# BGE Step 4：聚类实验与结果范围

2026-11-05。复用Step 3固定的BGE-M3向量；不加载编码器、不生成新向量。结果目录：`output/clustering-bge/freeze_20261101-v1/`。

先读 `findings.md`，再用Edge/Chrome打开 `report.html`。网页支持15个候选、簇、来源语言、关键词、低信息与emoji筛选，以及中心/随机评论浏览。HTML为离线文件，可拖进浏览器；原始文本中的内容仅作为数据展示。

## 比较设计

- 来源：固定提交8381107；19,392条有向量评论，与Step 3 ID顺序和矩阵哈希一致。主表19,486条在每个候选的分配索引中全部保留。
- 完整拟合集：源 `fit_eligible` 且规范化后非空、有词汇内容，共13,145条。源拟合资格13,149条中，3条格式清理后为空、1条纯符号不拟合。
- 完整拟合集KMeans：K=6、10、14，seed=17、42，n_init=10，最多300次迭代，拟合后分配全部19,392条有向量评论；94条空文本仍保留无向量/无分组状态。纯符号仍保留raw_cluster诊断，不解释为具体主题。
- 公平算法比较：从共同拟合集按来源语言等额、游戏轮转抽5,000条，中英文各2,500条，覆盖50款游戏。共同样本上比较KMeans、Ward/欧氏距离、平均连接/余弦距离，各切K=6、10、14，共9组。
- 所有候选的余弦轮廓系数在共同样本中同一批1,500条上计算。5,000条是开发对照样本，1,500条是内部几何诊断样本，都不是独立语义准确率测试集。
- Ward使用欧氏距离；平均连接使用余弦距离。分别构建一次树再切不同K，不用余弦距离调用Ward。
- 层次聚类只报告这5,000条的直接成员关系；其他评论标为不在该比较样本中，不以最近中心分配冒充层次预测。

## 指标与阅读边界

簇规模、过小簇数、共同样本轮廓系数、与来源语言/写作系统/游戏/推荐标签的AMI均是诊断。Steam推荐标签只在拟合后关联分析，不输入特征、调参或选赢家，也不是情绪人工金标准。汉字/拉丁字母写作系统规则不等于真实语言识别，源API标签仍需审阅。

K=10、seed=17预先作为浏览锚点，没有据最高指标选出“10个正式主题”。不同候选的C0等编号没有语义对应关系。KMeans两种子的ARI只说明对随机初值的一致性，不证明主题正确或重采样稳定。

CSV中保留到本簇中心的余弦值与中心差距，用于辅助阅读。0.03的边界提示不是校准置信度；层次树成员不一定最接近该簇的均值中心，不能用负差距直接判算法错误。本轮未冻结正式 `no_clear_theme` 规则，也不把源低信息或字数短直接等同无主题。

## 代表评论与可复现性

每组导出中心5条、随机5条，以及各来源语言中心最多3条。中心展示按完全相同的规范化文本去重，随机ID保留原抽样；未对拟合数据或簇规模去重。不同selection之间可出现同一ID，代表表不应当作额外独立样本。

初次运行单独保存了共同样本成员，但未另存层次树行顺序。`complete_scope_order.py`使用同一固定输入、种子和未改动的抽样函数补全顺序记录；独立检查将树重新切分，与6组已保存的层次成员关系逐一核对。没有重新拟合模型。该补充记录及来源说明保留于结果目录。

`refine_representatives.py`从已保存成员和中心相似度改善展示例子的多样性，保留原随机ID并核对全部分组文件哈希；不改变成员、指标或模型。主运行源代码和这些后处理脚本均在 `source_code/` 归档。

## 主要文件

- `report.html`、`findings.md`、两张诊断PNG：中文浏览与解读。
- `candidate_metrics.csv`、`seed_stability.json`：15个候选及种子一致性。
- 各候选 `assignments.csv`：全部主表ID、raw/candidate分组、实际拟合资格与范围状态。
- 各候选 `cluster_profiles.csv`、`representative_reviews.csv`、`centers.npz`。
- `ward_linkage.npy`、`average_linkage.npy`、`common_review_order.json`：两棵树及其行ID顺序。
- `scope_index.csv`：19,392条有向量评论的完整拟合/共同样本/指标样本标记。
- `validation.json`、`representative_validation.json`、`browser_validation.json`：实现与交付检查，不是独立人工语义核验。
- `output_checksums.json`、`source_code/`：交付校验与执行证据。

在仓库根目录可运行：

```powershell
.\.venv-topic\Scripts\python.exe -X utf8 scripts/cluster_bge.py --config config/clustering_bge.json --output output/clustering-bge/new-version
.\.venv-topic\Scripts\python.exe -m unittest discover -s tests -p test_cluster_bge.py -v
```

主脚本拒绝覆盖已有目录。结果目录里归档的后处理与核验脚本以本次v1路径配置，重做其他版本时需相应修改它们的OUT设置。向量缓存按Step 3原位置读取，未复制或重新编码。实验由AI辅助实现和初读；正式主题解释仍需成员核验。

算法定义参考：[SciPy linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)、[scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)。

# Task11 双人答卷与人工复核

2026-11-14：双方分工复核草稿已收齐，180条评论和30组均已填写；已按意见及规则生成本轮 AI 裁决，B-Lin 已在项目对话中表示认可。当前入口为[裁决说明](../../output/topic_review/steam-review-20261107-v1/adjudicated-20261114/裁决说明.md)和[裁决检查页](../../output/topic_review/steam-review-20261107-v1/adjudicated-20261114/裁决检查.html)。认可记录绑定结果哈希，见[human_confirmation.json](../../output/topic_review/steam-review-20261107-v1/adjudicated-20261114/human_confirmation.json)。Task12 尚未冻结。

本轮可复现命令（输出须为新目录）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B scripts\adjudicate_review_round1.py --output output\topic_review\another-adjudication
```

`adjudication.json` 保存180条裁决及30组处理；`resolved_sample.json` 汇集300条结果，其中120条沿用原双方标签共识；`decision_audit.json` 记录原复核意见、改动与来源哈希。检查页使用独立存储键，避免旧的空白整轮缓存覆盖本轮结果。

四份 `*_complete.json` 是 A/B 两人的原始答卷，保留原文件。`review_sample.json` 从原始离线核验页提取 300 条评论和 30 组共 180 个例子，不含 AI 预填答案；`CODEBOOK.md` 是原版 v1 规则的副本。`group_review_guidance.json` 是本次 AI 辅助分析提示，不是人工裁决。

讨论前的一致率、κ 和原始分歧见[统计摘要](../../output/topic_review/steam-review-20261107-v1/SUMMARY.md)，机器可读记录为同目录的 `review_results.json`。

`assignments/` 保留两份实际返回的 JSON 草稿及 `assignment_manifest.json`，用于校验两人各90条评论、15组的覆盖与来源。原草稿的 pending、空姓名及未总确认状态保留；对话认可另存，未改写为两人的完成签署。

旧整轮空白页和两份分工空白 HTML 已清理，当前只保留载入裁决的检查页。修改后从页面导出 JSON；页面自动保存只在当前浏览器有效。若修改了已认可的结果，应为新结果另记确认。

## 复现与导入裁决

在仓库根目录执行，无需下载模型或重跑聚类：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B scripts\review_topics_json.py --output output\topic_review\another-comparison
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -p test_review_topics_json.py -v
```

输出目录必须是新目录，脚本拒绝覆盖已有结果。输出包含 `review_results.json`（统计、双方原答、复核理由及文件哈希）、`SUMMARY.md`、`人工复核.html` 和裁决 JSON 模板。

导出裁决后可验证并生成新的对照目录，示例路径需换成实际保存位置：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B scripts\review_topics_json.py --decisions path\to\steam-review-20261107-v1_adjudication_complete.json --output output\topic_review\adjudicated-round1
```

导入裁决不会重算讨论前一致性，也不会自动冻结主题。原 `scripts/review_topics.py` 继续用于旧版 250 条单标签 CSV；本轮 300 条多标签 JSON 使用新脚本。

需要新一轮分工页时，可执行 `scripts/split_review_adjudication.py split --output <新目录>`。该脚本的 `merge` 仅接受两份实际完成的分任务文件；本轮按用户授权使用已填写草稿裁决，由 `adjudicate_review_round1.py` 复现，输入哈希固定。

## 方法口径

- 语言、有无具体方面、表达状态按原始类别比较；有无方面的“不确定”保留为独立类别。
- 每个方面分别计算二元一致率、Cohen κ、双方阳性和阴性一致率。任一方“方面不确定”的记录不进入该项二元比较，当前共同分母为 288。
- 特殊表达标记同样排除任一方“表达不确定”的记录。原始计数和分母都保存，不把不确定当未选。
- 分语言统计只纳入双方实际语言判断相同的记录；12 条语言分歧另列。组名按语义核对，不用文字相同率代替主题一致性。
- B 的文件保留 AI 预填后人工审查来源；A 没有 AI 声明字段，仅记录“未声明”。这组指标描述本轮两份答卷，不称为两名完全无辅助标注者的信度。
- 样本中的 `sample_sha256` 是原包声明的版本标识。本次提取样本另算文件 SHA256，不声称重建了未提供的协调人包哈希。

## 仍需协调人与全员完成

当前目录没有匿名 G 编号到 NMF／BGE 模型、K、簇的协调人映射。先保存匿名组的裁决，补齐映射后再对应到各候选模型；不根据组名或例子猜来源。Task12 仍需模型稳定性检查、主题边界确认及全员冻结，之后才能开展正式 RQ2／RQ4 分析。

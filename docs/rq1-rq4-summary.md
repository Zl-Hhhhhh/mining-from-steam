# RQ1–RQ4｜新版主题权重结果

本轮使用 [2026-11-25 结构化主题包](../output/topic_selection/topics-v1-structured-exploratory-20261125/README.md) 的八个一级主题。主分析采用包含新增片段证据的 `all` 权重。在 50 款游戏的 19,486 条清洗评论中，5,301 条检出主题，占 27.2%。上一版同口径为 4,879 条（25.0%）。新版 `fit` 有 3,996 条（20.5%）。下文同时使用新版 `fit` 和固定 `fit` 评论后换用 `all` 权重的对照。

## RQ1 玩家评论谈什么

| 主题 | 新版 all | 新版 fit | 上一版 all |
| --- | ---: | ---: | ---: |
| T08 内容与维护 | 17.55% | 19.98% | 20.49% |
| T01 玩法与机制 | 15.44% | 16.43% | 18.40% |
| T04 剧情与角色 | 15.28% | 16.11% | 17.14% |
| T03 性能与故障 | 14.21% | 11.89% | 12.00% |
| T05 画面与声音 | 13.60% | 10.77% | 10.31% |
| T06 联机与社区 | 11.54% | 11.86% | 10.32% |
| T02 操作与难度 | 7.27% | 8.04% | 7.11% |
| T07 价格与付费 | 5.10% | 4.92% | 4.24% |

前三项顺序保持一致，玩法与剧情仅差 0.16 个百分点。固定新版 `fit` 评论并换用 `all` 权重后，性能与故障、画面与声音的权重分别为 12.20% 和 11.28%。包含全部有证据评论时，这两项都进一步上升。高置信度子集有 232 条评论，覆盖操作与难度、联机与社区两个主题。

[分布表、代表片段与图表索引](../output/rq1/topics-v1-structured-exploratory-20261125/README.md) · [敏感性表](../output/rq1/topics-v1-structured-exploratory-20261125/sensitivity_overall.csv)

## RQ2 评论主题与推荐

主模型使用 50 款游戏中有主题证据的 5,301 条合格评论，控制游戏、来源语言和发表时的游玩时长。OR 对应将玩法权重的 10 个百分点转给相应主题。区间为逐项 95% 游戏聚类稳健区间，p 值采用 Holm 校正。

| 主题 | 推荐 OR | 95% 区间 | 校正 p | 通过校正的可估计设定 |
| --- | ---: | ---: | ---: | ---: |
| T03 性能与故障 | 0.75 | 0.73–0.78 | <0.001 | 14/14 |
| T04 剧情与角色 | 1.12 | 1.06–1.18 | <0.001 | 14/14 |
| T05 画面与声音 | 1.31 | 1.23–1.40 | <0.001 | 14/14 |

三项在新版 `fit` 的 49 款游戏、3,939 条实际估计评论中也通过校正。剧情在上轮排除 A01／A02 片段后失去校正显著性，本轮对应设定仍通过校正。新版主分析中，有证据评论的原始推荐率为 72.3%，其余合格评论为 88.8%，主题模型的结果限于有证据样本。

[模型表](../output/rq2_rq4/topics-v1-structured-exploratory-20261125/rq2_models.csv) · [主题支持](../output/rq2_rq4/topics-v1-structured-exploratory-20261125/rq2_topic_support.csv) · [森林图](../output/rq2_rq4/topics-v1-structured-exploratory-20261125/rq2_topic_odds.png)

## RQ3 中英文来源评论的差异

按同一游戏配对、每款游戏等权。主题构成用 47 款游戏的 5,061 条有证据评论，其中英文来源 2,824 条，简体中文来源 2,237 条。

| 主题 | 新版 all 差异（中文减英文，百分点） | 新版 fit 差异 | 上一版 all 差异 |
| --- | ---: | ---: | ---: |
| T03 性能与故障 | +15.12 | +12.03 | +11.18 |
| T08 内容与维护 | +8.51 | +6.85 | +8.28 |
| T05 画面与声音 | −13.95 | −8.05 | −8.80 |
| T06 联机与社区 | −9.02 | −8.65 | −8.07 |

前三项在 11 个范围和样本设定及 3 个映射设定中均保持方向并通过校正。联机的差异在核心设定中保持方向，开放全部候选主题后反向。48 款游戏的配对比较中，中文来源的主题检出率低 6.11 个百分点，推荐率低 3.23 个百分点。

交互模型中，联机相对玩法的推荐 OR 在中文来源是英文来源的 1.24 倍。共同日期窗口和新版 `fit` 下仍通过校正，排除 A01／A02 后信号减弱。视听与价格的交互在主模型通过校正，两个对照的显著性随设定变化。来源语言是采集标签，主题构成比较仅覆盖有证据评论。

[浏览器报告与图表](../output/rq3/topics-v1-structured-exploratory-20261125/report.html) · [配对估计](../output/rq3/topics-v1-structured-exploratory-20261125/paired_estimates.csv) · [稳健性表](../output/rq3/topics-v1-structured-exploratory-20261125/robustness_summary.csv) · [交互模型](../output/rq3/topics-v1-structured-exploratory-20261125/interaction_effects.csv)

## RQ4 游玩时长、推荐与主题

19,395 条合格评论按游戏内时长三分位分组。低、中、高组推荐率分别为 **82.2%、86.0%、84.8%**。控制游戏和来源语言后，中组相对低组 OR 为 **1.39**（95% CI 1.13–1.70，Holm p=0.005）。高组相对低组 OR 为 1.23（0.90–1.69）。连续模型中，「1＋游玩小时数」每翻倍的 OR 为 1.04（0.98–1.11）。

在 5,301 条有主题证据的评论中，两项主题变动在连续和分组主模型中均通过校正。

| 主题 | 高时长组减低时长组 | 「1＋游玩小时数」每翻倍 |
| --- | ---: | ---: |
| T08 内容与维护 | +3.69 个百分点，Holm p=0.003 | +0.72 个百分点，p=0.002 |
| T07 价格与付费 | −1.83 个百分点，Holm p=0.029 | −0.31 个百分点，p=0.024 |

两项在可估计的对照中方向一致。T08 高组对低组有 10/13 个设定通过校正，T07 有 6/13 个。换用新版 `fit` 或固定 `fit` 评论后换用 `all` 权重，两项均未通过校正。主题证据覆盖率由低时长组的 24.3% 升到高组的 30.4%，主题权重的关联应结合样本覆盖解读。

[时长组统计](../output/rq2_rq4/topics-v1-structured-exploratory-20261125/rq4_groups.csv) · [模型表](../output/rq2_rq4/topics-v1-structured-exploratory-20261125/rq4_models.csv) · [时长图](../output/rq2_rq4/topics-v1-structured-exploratory-20261125/rq4_playtime_groups.png)

## 复现与口径

K8 与开放全部候选主题的对照只重判新版中保留的历史片段，沿用新版的评论关联和新增片段标签。主题包仍待独立语义复核，权重表示各主题在单条评论中的相对证据贡献。各分析报告的是探索性关联。

在仓库根目录使用已安装 `requirements.txt` 的 Python 环境运行。

```powershell
$version = 'topics-v1-structured-exploratory-20261125'
python -B scripts/analyze_rq1.py --topic-package "output/topic_selection/$version" --primary-scope all --output "output/rq1/$version"
python -B scripts/analyze_rq2_rq4.py --topic-version $version
python -B scripts/analyze_rq3.py --config config/rq3_structured_exploratory.json
python -B scripts/analyze_rq3.py --config config/rq3_structured_exploratory.json --verify-output
```

重新运行时可加 `--output` 指定新的目录。各版输入和产物记录在 [RQ1 manifest](../output/rq1/topics-v1-structured-exploratory-20261125/rq1_manifest.json)、[RQ2／RQ4 manifest](../output/rq2_rq4/topics-v1-structured-exploratory-20261125/manifest.json) 和 [RQ3 manifest](../output/rq3/topics-v1-structured-exploratory-20261125/manifest.json)。

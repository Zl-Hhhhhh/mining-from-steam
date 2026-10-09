# Setup 和实测结果

2026-10-09 在本工作区完成。所有采集结果保留在本地 `data/`，不自动加入 Git。

## 环境

- Python 3.13.1，项目虚拟环境 `.venv/`。
- Git 2.53.0.windows.1。
- GitHub 工具：[`woctezuma/download-steam-reviews`](https://github.com/woctezuma/download-steam-reviews)，包版本 0.9.6.1，MIT。
- 源码 commit：`27adb12737b2169117dfddd9092e3dc6f5934307`，下载至 `external/download-steam-reviews/`。
- 所需依赖已安装，版本在 `requirements-lock.txt`。官方采集器自身仅使用 Python 标准库。

## Steam 官方接口：CS2 / 730

采集时间：2026-10-09 09:53:12–09:53:45 UTC（北京时间 17:53:12–17:53:45）。

| 项目 | 实际结果 |
| --- | --- |
| 在线人数快照 | 658,302，采集于 09:53:12 UTC |
| 游戏新闻 | 20 条 |
| 商店元数据 | 成功，游戏名 Counter-Strike 2 |
| 英文评论 | 300 条，238 条正面，样本正面比例 79.33% |
| 简体中文评论 | 300 条，249 条正面，样本正面比例 83.00% |
| 合并评论 | 600 条，不重复 |
| 评论接口 | 新版 `IUserReviewsService/GetAppReviews/v1/`，未使用回退 |

英文样本创建时间为 2026-10-08 20:20:50 至 2026-10-09 09:38:59 UTC；中文样本为 2026-10-07 14:00:21 至 2026-10-09 09:40:15 UTC。两种语言的 300 条并不覆盖相同时间长度；正面比例不能当作整体评价率或用来直接比较人群。

文件目录：`data/steam/730/20261009T095312052379Z/`。

- [合并 CSV](../data/steam/730/20261009T095312052379Z/reviews.csv)
- [合并 JSONL](../data/steam/730/20261009T095312052379Z/reviews.jsonl)
- [采集 manifest](../data/steam/730/20261009T095312052379Z/manifest.json)

官方来源：[评论接口](https://partner.steamgames.com/doc/webapi/IUserReviewsService)、[在线人数接口](https://partner.steamgames.com/doc/webapi/ISteamUserStats#GetNumberOfCurrentPlayers)、[新闻接口](https://partner.steamgames.com/doc/webapi/ISteamNews)。在线人数响应可能缓存，因此这是采集时获取的快照。

## GitHub 工具

于 09:55:44 UTC 使用原版单页下载函数成功获取 100 条 CS2 英文评论。请求端点为 `https://store.steampowered.com/ajaxappreviews/730`，共一次请求，未启动全量下载。

100 条全部与官方的 600 条评论重叠；两条采集路径合并后的独立评论数仍然是 **600**。

- [工具导出的 CSV](../data/github/730/20261009T095544155409Z/reviews.csv)
- [版本和运行 manifest](../data/github/730/20261009T095544155409Z/manifest.json)

`scripts/github_smoke.py` 在单页验证中为原版函数增加请求超时和请求次数限制。GitHub 源码保持原样，固定提交以便重现。

## 现成研究数据集

已从 [UCSD McAuley Lab 官方目录](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data)下载三个完整压缩文件并提取 appid 730：

| 文件 | 压缩字节数 | 扫描记录数 | 730 匹配记录数 |
| --- | ---: | ---: | ---: |
| australian_user_reviews.json.gz | 6,935,922 | 25,799 | 3,759 |
| australian_users_items.json.gz | 74,035,391 | 88,310 | 43,776 |
| steam_games.json.gz | 2,664,464 | 32,135 | 1 |

这里扫描记录数是顶层文件行数；评论和用户游戏记录嵌套于每个用户对象中，不能将行数当作整个文件的评论数或交互数。

评论对应 3,708 个不同用户，其中有 51 条完全重复记录；用户游戏记录对应 43,331 个不同用户，其中有 440 条完全重复记录。导出保留原始记录，建模前需要按目标主键去重。742 条评论日期缺少年份，未推断或补写年份。

这些数据是 **CS:GO 历史资料**。appid 相同不意味着其内容属于 CS2。Steam 商店说明 2023-09-27 之前的评论来自 CS:GO；当前采集使用这一天 UTC 零点作近似边界。[Steam 商店说明](https://store.steampowered.com/app/730/CounterStrike_2/)

- [历史评论](../data/research/730/reviews.jsonl)
- [历史用户游戏记录](../data/research/730/items.jsonl)
- [历史游戏元数据](../data/research/730/metadata.jsonl)
- [来源、SHA-256 和引用信息](../data/research/730/manifest.json)

完整版 Version 2 评论集约 1.3 GB，本次未下载；已下载的三份资料可用于课程项目的历史基线与用户行为分析。

## 验证

- 4 个单元测试通过：时代边界、JSON/字典字面量安全解析、429 重试、403 停止重试。
- 真数据验证通过：CSV 与 JSONL ID 一致；当前评论无重复，全部符合 CS2 日期边界；语言、推荐计数、原始分页数与 manifest 一致。
- 研究文件 SHA-256、导出行数和 appid 校验通过。
- [完整校验结果](../data/verification.json)。

扩大当前样本：

```powershell
./.venv/Scripts/python.exe scripts/steam_data.py collect --limit 1000
```

# mining-from-steam

AIAA3111 课程项目：通过 Steam 官方接口、GitHub 开源工具和研究数据集获取游戏数据。默认目标为 **CS2 / appid 730**。

## 已配置的数据来源

| 来源 | 内容 | 本项目入口 |
| --- | --- | --- |
| Steam 官方 Web API | 近期评论、当前在线人数、游戏新闻 | `scripts/steam_data.py collect` |
| Steam 商店端点 | 游戏名称、类型、平台等元数据 | 同上，`store.json` |
| GitHub `woctezuma/download-steam-reviews` | 原版 `steamreviews` 下载器，单页验证 | `scripts/github_smoke.py` |
| UCSD McAuley Lab | Australian reviews、users/items、Steam 游戏元数据 | `scripts/steam_data.py research` |

评论默认使用新版 [IUserReviewsService/GetAppReviews](https://partner.steamgames.com/doc/webapi/IUserReviewsService)，无需 API key。只有新版返回 HTTP 404/410 时才回退旧 `/appreviews`；429 会限速重试。也可通过 `--backend legacy` 手动使用旧端点。GitHub 原版工具使用 `/ajaxappreviews/`，已单独实测。

其他官方接口：[当前在线人数](https://partner.steamgames.com/doc/webapi/ISteamUserStats#GetNumberOfCurrentPlayers)、[游戏新闻](https://partner.steamgames.com/doc/webapi/ISteamNews)。商店 `appdetails` 是公开端点，稳定性不等同于已文档化的 Web API。

## 环境安装（Windows PowerShell）

需要 Python 3.11+ 和 Git。本机已使用 Python 3.13.1 创建 `.venv`。

```powershell
./scripts/setup.ps1
# 若使用其他 Python 版本：
./scripts/setup.ps1 -PythonVersion 3.11
```

脚本在项目内创建虚拟环境，将 [GitHub 工具](https://github.com/woctezuma/download-steam-reviews) 下载到 `external/download-steam-reviews` 并安装。源码固定至 `config/sources.json` 中的 commit；依赖快照见 `requirements-lock.txt`。不会自动更新到最新源码。

只使用官方采集器无需安装第三方包：`./scripts/setup.ps1 -SkipGitHub`。也可使用 `pip install -r requirements-lock.txt` 从 PyPI 安装相同包版本，但这不代表使用了相同 GitHub 源码提交。

本项目已实现的接口均无需 key。`.env.example` 仅作未来用户统计接口的配置提示，目前脚本不读取 `.env`。如果需要读取指定玩家的统计、库存或游戏库，仍需分别核实接口权限、key 和玩家隐私设置。

## 获取 CS2 当前数据

```powershell
./.venv/Scripts/python.exe scripts/steam_data.py collect --limit 300
# 扩大样本：每种语言 1000 条，最多请求 50 页/语言
./.venv/Scripts/python.exe scripts/steam_data.py collect --limit 1000
# 指定语言或其他游戏
./.venv/Scripts/python.exe scripts/steam_data.py collect --appid 730 --languages english schinese --limit 300
```

默认按创建时间从新到旧采集中英文评论，保留所有购买来源、正负评价和 off-topic 评论，使用 cursor 分页，每页最多 100 条，并按 recommendationid 去重。遇到异常会保存已获取的样本及错误原因，进程返回非零状态。

每次执行创建独立的 `data/steam/730/<UTC时间>/`，包含：

- `reviews.jsonl`、`reviews.csv`：合并评论，保留文本、语言、推荐标志、创建时间、作者游玩分钟数等。
- `english/`、`schinese/`：各语言导出及 `pages/` 原始分页响应。
- `players.json`、`news.json`、`store.json`：带采集时间的原始快照。
- `manifest.json`：参数、成功/失败状态、样本数及接口汇总。

最近一次采集的位置保存在 `data/steam/latest.json`。CSV 使用 UTF-8 BOM；SteamID 等长整数请优先从 JSONL 读取，以免 Excel 自动转为浮点数。

## 验证 GitHub 采集工具

```powershell
./.venv/Scripts/python.exe scripts/github_smoke.py
./.venv/Scripts/python.exe scripts/github_smoke.py --language schinese
```

该脚本直接调用已安装包的单页下载函数，限制为一次请求并添加 45 秒超时。它不会启动原版工具默认的全量下载。结果在 `data/github/730/<UTC时间>/`，最近位置见 `data/github/latest.json`。原始响应、标准化评论和源码版本均保留；与官方样本有重叠，不能直接相加当作独立评论总数。

## 下载现成研究数据集

```powershell
./.venv/Scripts/python.exe scripts/steam_data.py research
# 只下载较小的评论集
./.venv/Scripts/python.exe scripts/steam_data.py research --files reviews
```

下载 [UCSD 官方目录](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) 的三个压缩文件：`australian_user_reviews.json.gz`、`australian_users_items.json.gz`、`steam_games.json.gz`。总压缩大小约 80 MB；用户游戏记录的解析需要数分钟。默认复用已完成下载，`--redownload` 可重新获取。较大的 Version 2 评论集约 1.3 GB，当前未自动下载。

原始压缩文件位于 `data/research/raw/`；appid 730 的评论、用户游戏记录和元数据位于 `data/research/730/`。manifest 记录下载 URL、SHA-256、扫描行数、提取行数和论文引用。旧数据存在 Python 字典字面量，解析使用 `ast.literal_eval`，不使用 `eval`。

研究中请按目录要求引用：Kang & McAuley（ICDM 2018）、Wan & McAuley（RecSys 2018）、Pathak et al.（SIGIR 2017）。引用信息也在 `config/sources.json`。目录未明确提供统一授权条款，不将其默认视为 MIT；GitHub 工具的 MIT 许可证保存在源码目录中。

## 区分 CS:GO 和 CS2

[Steam 商店](https://store.steampowered.com/app/730/CounterStrike_2/)明确说明，2023-09-27 之前的评论属于 CS:GO。两者共享 appid 730：

- 官方和 GitHub 当前评论导出仅保留创建时间不早于 `2023-09-27T00:00:00Z` 的记录。按日期近似划分，并非精确的上线时刻。
- 研究数据中的 appid 730 标记为历史 CS:GO，不与 CS2 当前评论混用。
- 更新过的旧评论仍按创建时间判断，不能因为更新时间较新就归入 CS2。
- 累计游玩时间可能同时包含 CS:GO 和 CS2，不能解释为纯 CS2 游玩时长。
- 样本是近期评论，不是随机样本；正面比例仅描述本次样本。API 汇总的范围取决于语言、日期和其他过滤参数，不等同于整个游戏的全部历史评论。

这些数据适合评论情感/主题分析、推荐标签分析和用户游戏行为研究。在线人数是一次实时快照，不是历史曲线；当前入口不提供比赛 demo、回合事件或逐场战绩。

## 验证与结果

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -v
./.venv/Scripts/python.exe scripts/verify_data.py
```

测试覆盖 CS:GO/CS2 创建时间边界、旧研究文件安全解析、429 重试及 403 停止重试。数据校验会检查评论去重、数量、时代边界以及研究文件哈希。

本次 setup 和实际采集结果见 [docs/setup-results.md](docs/setup-results.md)。`data/`、`external/`、`.venv/` 已加入 `.gitignore`；数据保留在本地，Git 提交只包含代码、配置与结果说明。

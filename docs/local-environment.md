# 项目独立环境（Windows）

2026-11-11 已在本机建立 `mining-from-steam/.venv`：Python 3.13.17、pip 26.2.2、Windows x64。`include-system-site-packages = false`，用户级 site-packages 未启用，项目依赖安装在 `.venv` 内。

## 直接使用

在 `mining-from-steam` 根目录执行：

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -X utf8 scripts\steam_data.py --help
.\.venv\Scripts\python.exe -X utf8 scripts\nmf_baseline.py --help
.\.venv\Scripts\python.exe -X utf8 scripts\embedding_bge.py verify
```

无需激活，也不依赖全局 `python` / `pip` 命令。若希望激活，在允许执行脚本的 PowerShell 会话中运行 `.\.venv\Scripts\Activate.ps1`，结束后运行 `deactivate`。

用 VS Code 单独打开 `mining-from-steam` 文件夹时，`.vscode/settings.json` 默认选择此解释器，并为新终端设置 UTF-8 和项目内 Matplotlib 缓存。若编辑器已记住其他解释器，手动选择 `.venv\Scripts\python.exe`。在其他受限终端绘图前，可以设置：

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache\matplotlib'
```

## 在新电脑恢复

安装 Python 3.13 后，在仓库根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-local.ps1
# Python 安装在其他位置时：
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-local.ps1 -PythonExecutable 'C:\path\to\python.exe'
```

执行策略参数只作用于这次 PowerShell 进程。脚本创建或复用 `.venv`，安装 `requirements.txt`，检查隔离状态和依赖一致性。`.venv` 绑定本机解释器与路径，移动项目或换电脑后应在新位置重新创建，不复制虚拟环境。

`requirements.txt` 沿用本机环境的 31 个固定版本，覆盖采集、分析及缓存聚类。已有虚拟环境也可直接运行 `python -m pip install -r requirements.txt`。历史报告使用的 `requirements-analysis.txt` 转到这份清单。原始依赖文件可从 Git 提交 `c287992` 查看，例如 `git show c287992:requirements-analysis.txt`。`.venv/` 和 `.cache/` 已被 Git 忽略。

## 当前覆盖范围

- 官方采集、审计、清洗、metadata 脚本；`steamreviews` 使用原固定版本的 PyPI 包。
- TF-IDF / NMF、中文分词、图表、统计建模、旧版 CSV 一致性比较及本轮 Task11 多标签 JSON 对照。
- Task10 已交付向量缓存的验证和聚类。

需要固定 GitHub 源码提交的采集器时，使用 README 中原 `scripts/setup.ps1` 入口；PyPI 同版本不代表相同 Git 提交。

Task10 和 Task12 重新编码使用单独的 `requirements-embedding-lock.txt`。CPU PyTorch 安装命令和模型修订见 [BGE 编码说明](embedding-bge.md#确需重新编码时)。现有向量可直接用日常环境验证和聚类。

`reviewer_B` 的 HTML 人工标注本身只需浏览器。新版 300 条多标签 JSON / 30 组命名与 `scripts/review_topics.py` 的旧版 250 条单标签 CSV 输入不同，环境安装不会改变这个格式差异。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_embedding_bge_pipeline.py -v
```

全量测试包含句子编码测试，需要 BGE 编码环境。配置好 `.venv-topic` 后，补装日常分析依赖并运行全量测试。

```powershell
.\.venv-topic\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-topic\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

2026-11-11 初次配置时：主项目 19 项测试、Task10 3 项测试全部通过；采集、分词、数值计算、绘图及 `statsmodels.api` 均可导入。2026-11-13 更新仓库后，依赖检查通过，已整合到 `tests/` 的 22 项测试全部通过。Task10 测试验证真实缓存和合成编码控制流程，不代表真实模型主题质量已通过人工核验。

# 批量采集脚本使用指南

## 📋 脚本功能

`collect_batch.py` 自动采集 `config/games_clustering.json` 中配置的所有游戏的评论数据。

### 特性
- ✅ 自动读取游戏配置（50 个游戏）
- ✅ 每个游戏采集 200 英文 + 200 简中评论
- ✅ 断点续传（中途中断可以继续）
- ✅ 实时进度显示和 ETA 估算
- ✅ 错误处理和日志记录
- ✅ 自动生成汇总报告

---

## 🚀 使用方法

### 1️⃣ 完整采集（所有 50 个游戏）

```bash
# Windows PowerShell
python scripts/collect_batch.py

# 或使用虚拟环境（如果已创建）
.venv/Scripts/python.exe scripts/collect_batch.py
```

**预计时间：** 约 3.5 小时  
**输出目录：** `data/{appid}/collection_history/{timestamp}/`

---

### 2️⃣ 测试模式（先采集 3 个游戏）

```bash
# 只采集前 3 个游戏，测试脚本是否正常工作
python scripts/collect_batch.py --limit 3
```

**推荐：** 第一次运行时先用 `--limit 3` 测试，确认无误后再运行完整采集。

---

### 3️⃣ 断点续传

如果采集过程中被中断（Ctrl+C、网络断开、电脑关机等）：

```bash
# 从上次中断的地方继续
python scripts/collect_batch.py --resume
```

脚本会：
- ✅ 跳过已完成的游戏
- ✅ 从下一个未完成的游戏继续
- ✅ 保留之前的进度记录

---

### 4️⃣ 重新开始（清空进度）

如果想从头开始，忽略之前的进度：

```bash
python scripts/collect_batch.py --force
```

---

### 5️⃣ 调整游戏之间的延迟

```bash
# 增加游戏间延迟到 5 秒（默认 3 秒）
python scripts/collect_batch.py --game-delay 5
```

---

## 📊 输出说明

### 采集过程中的输出

```
======================================================================
Batch Collection Started
======================================================================
Total games to collect: 50
Reviews per game: 200 × 2 languages = 400
Languages: english, schinese
Estimated time: ~200 minutes (3.3 hours)
======================================================================

[1/50] Collecting: Counter-Strike 2 (appid: 730)
  Genre: Action, FPS
  Progress: 0 successful, 0 failed
  ✅ SUCCESS (245.3s)
  Waiting 3s before next game...
  Elapsed: 4.1min | ETA: 196.5min

[2/50] Collecting: Dota 2 (appid: 570)
  Genre: Strategy, MOBA
  Progress: 1 successful, 0 failed
  ✅ SUCCESS (198.7s)
  ...
```

### 最终汇总

```
======================================================================
Batch Collection Complete
======================================================================
Successful: 48/50
Failed: 2/50
Total time: 185.3 minutes (3.09 hours)
Average per game: 222.4 seconds
======================================================================

Summary saved to: data/batch_summary.json
```

---

## 📁 数据存储结构

```
data/
├── batch_progress.json          # 进度检查点（用于断点续传）
├── batch_summary.json           # 最终汇总报告
│
├── 730/                         # Counter-Strike 2
│   └── collection_history/
│       └── 20261012T143256789Z/
│           ├── manifest.json    # 采集元数据
│           ├── reviews.jsonl    # 合并的所有评论
│           ├── reviews.csv      # 合并的所有评论（CSV）
│           ├── players.json     # 在线人数快照
│           ├── news.json        # 游戏新闻
│           ├── store.json       # 商店元数据
│           ├── english/
│           │   ├── reviews.jsonl
│           │   ├── reviews.csv
│           │   └── pages/       # 原始 API 响应
│           │       ├── 0000.json
│           │       ├── 0001.json
│           │       └── ...
│           └── schinese/
│               ├── reviews.jsonl
│               ├── reviews.csv
│               └── pages/
│
├── 570/                         # Dota 2
│   └── collection_history/
│       └── ...
│
└── ...（其他 48 个游戏）
```

---

## 📋 重要文件说明

### `data/batch_progress.json`（进度检查点）
```json
{
  "completed": [
    {
      "appid": 730,
      "name": "Counter-Strike 2",
      "collected_at": "2026-10-12T14:35:42Z",
      "elapsed_seconds": 245.3
    }
  ],
  "failed": [
    {
      "appid": 12345,
      "name": "Some Game",
      "failed_at": "2026-10-12T15:20:10Z",
      "error": "HTTPError 404"
    }
  ],
  "last_updated": "2026-10-12T15:20:10Z"
}
```

### `data/batch_summary.json`（最终报告）
```json
{
  "collection_type": "batch",
  "started_at": "2026-10-12T14:30:00Z",
  "finished_at": "2026-10-12T17:35:22Z",
  "total_games": 50,
  "successful": 48,
  "failed": 2,
  "elapsed_seconds": 11122,
  "config_file": "config/games_clustering.json",
  "reviews_per_language": 200,
  "languages": ["english", "schinese"],
  "completed_games": [...],
  "failed_games": [...]
}
```

---

## ⚠️ 常见问题

### Q1: 脚本运行很慢，正常吗？
**A:** 正常。每个游戏需要：
- 采集 200 英文评论（~2 分钟）
- 采集 200 简中评论（~2 分钟）
- 获取元数据（~10 秒）
- 游戏间延迟（3 秒）

平均每个游戏 **4-5 分钟**，50 个游戏约 **3.5 小时**。

### Q2: 某个游戏失败了怎么办？
**A:** 脚本会：
1. 记录失败信息到 `batch_progress.json`
2. 继续采集下一个游戏
3. 最后在汇总报告中列出失败的游戏

你可以稍后单独重新采集失败的游戏：
```bash
python scripts/steam_data.py collect --appid 12345 --limit 200 --languages english schinese
```

### Q3: 我想中途停止，数据会丢失吗？
**A:** 不会。按 `Ctrl+C` 停止后：
- ✅ 已完成的游戏数据已保存
- ✅ 进度记录已更新
- ✅ 下次用 `--resume` 继续

### Q4: 如何验证采集的数据？
**A:** 运行验证脚本：
```bash
python scripts/verify_data.py
```

---

## 🎯 推荐的采集流程

### 第一步：测试（5-10 分钟）
```bash
# 先采集 2-3 个游戏测试
python scripts/collect_batch.py --limit 3
```

检查 `data/` 目录，确认：
- ✅ 每个游戏有 400 条评论（200 英文 + 200 简中）
- ✅ `manifest.json` 显示状态为 "ok"
- ✅ CSV 和 JSONL 文件正常

### 第二步：完整采集（3-4 小时）
```bash
# 确认无误后，运行完整采集
python scripts/collect_batch.py
```

**建议：**
- 🌙 在电脑空闲时运行（晚上或周末）
- 🔌 确保电脑不会休眠/断电
- 🌐 保持网络连接稳定

### 第三步：验证数据
```bash
# 采集完成后验证
python scripts/verify_data.py
```

---

## 🔧 高级选项

### 只采集特定游戏

如果只想采集某几个游戏，可以：

1. **编辑 `games_clustering.json`**，只保留需要的游戏
2. **或者使用 `--limit` 配合手动编辑游戏顺序**

### 自定义每游戏评论数

编辑 `config/games_clustering.json`：
```json
{
  "collection_config": {
    "reviews_per_game": 400,
    "reviews_per_language": 200,  // 改为 300 则每游戏采集 600 条
    "languages": ["english", "schinese"]
  }
}
```

### 只采集英文评论

编辑 `config/games_clustering.json`：
```json
{
  "collection_config": {
    "languages": ["english"]  // 移除 "schinese"
  }
}
```

---

## 📞 遇到问题？

1. **检查 `data/batch_progress.json`** 查看详细错误信息
2. **查看单个游戏的 `manifest.json`** 了解采集状态
3. **如果某个游戏的 appid 错误**，运行：
   ```bash
   python scripts/verify_appids.py
   ```

---

## ✅ 成功标志

采集完成后，你应该看到：

```
data/
├── batch_summary.json           # 总结报告
├── 730/, 570/, ..., 250900/     # 50 个游戏目录
└── 每个游戏目录包含：
    - 400 条评论（200 英文 + 200 简中）
    - manifest.json（状态显示 "ok"）
    - players/news/store 元数据
```

**总数据量预估：**
- 50 游戏 × 400 评论 = **20,000 条评论**
- 磁盘空间：约 **200-500 MB**（包含原始 JSON）

---

**准备好开始了吗？运行：**
```bash
python scripts/collect_batch.py --limit 3  # 先测试 3 个游戏
```

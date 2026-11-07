# Task10验证证据

data_validation.json保留实际向量、ID、15组候选复现对照、全量状态及双语来源分层代表检查。portable_reproduction_summary.json为独立路径重跑摘要；每组簇归属、拟合资格和边界标记一致，余弦分数差小于1e-6。

test_task10_portable.py只验证运行逻辑，合成编码器不代表真实语义效果。读取真实19392条缓存的测试阻止导入编码器；另测批次中断恢复、精确文本去重、ID映射和超长拒绝。运行：

```powershell
python -B -m unittest discover -s validation -p 'test_task10_portable.py' -v
```

上面命令在task10目录执行。原始Step3/Step4证据另保留在results/encoding及results/clustering。没有真人标签、人工一致率或最终主题验证。Task11材料不在本次提交。

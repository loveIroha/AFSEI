# 续算到原定第二次刺激

该实验从两组已完成的 1011.8 ms 状态继续，使用原有 `run_afsei_job.py` 和 `run_afsei_cycle.py`，没有新增流固或 EP 求解方法。

每组分两段：

1. 1011.8 → 1299.9 ms，保存刺激前检查点。
2. 1299.9 → 1300.0 ms，只推进原定刺激边界的一步，取得第二次收缩前 EDV，并核对阶段变为第二周期等容收缩。

两个入口波形分别保持原比例 1.0 和提前平台比例 0.1。仍在 GPU1 上顺序运行。原有 120 ms 结果被读取和复用，不覆盖或重算。

标准 JSON 配置：`configs/gao2017/filling_to1300_20261022.json`。

启动：`python scripts/run_afsei_filling_followup.py configs/gao2017/filling_to1300_20261022.json`。

输出：`results/gao_filling_to1300_20261022`。进度在 `status.json`；每个子阶段沿用原有 `progress.json`、`pv.csv`、`report.json`、`checkpoint.npz`。运行结束自动合并新旧充盈记录，并生成 `comparison.json`、`filling_to1300.png/pdf` 和 `analysis_zh.md`。

延后刺激的分支应从 `baseline/pre_stimulus/checkpoint.npz` 或 `early/pre_stimulus/checkpoint.npz` 开始。1300 ms 的检查点已触发原定刺激，不能作为未刺激状态直接继续延时。本次没有修改刺激安排、原充盈时间锚点或正式两周期配置。

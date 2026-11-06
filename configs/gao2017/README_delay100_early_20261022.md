# 第二次刺激从 1300 延后到 1400 ms

从提前供压组已有的 1299.9 ms 刺激前检查点继续，不重算此前的收缩或充盈。

第二次刺激改为 1400 ms。充盈波形原来的终点保留在 1300 ms，因此 1300–1400 ms 入口保持 8 mmHg、辅助壁载荷保持 0。人工张力卸载、NS、MAC/IB、EP PDE、GPB/Land、力学本构和求解方法保持原设置。

原控制器要求两次刺激间隔与周期元数据一致，本诊断把周期元数据改为 900 ms，以对应 500→1400 ms；控制器据此得到的后续终点是 2300 ms。本试验明确只算到 1400 ms，没有启动到 2300 ms 的完整两周期，也没有决定后续周期的最终设置。

迁移只修改刺激/阶段控制元数据，逐一验证全部物理数组（位移、流速、压力、EP/ODE 状态、力和历史量等）保持逐位相同。原检查点保持原样，修改后的检查点另存到 migration 目录，支持标准重启。

运行分段：

1. 1299.9→1300 ms，确认原刺激时刻仍处于充盈。
2. 1300→1399.9 ms，保存新刺激前检查点。
3. 1399.9→1400 ms，记录延后的 EDV，并触发第二次刺激边界/等容收缩阶段。刺激电流从随后 EP 子步开始，本次不继续计算该收缩过程。

标准配置：`configs/gao2017/delay100_early_20261022.json`。

运行：`python scripts/run_afsei_delayed_pacing_probe.py configs/gao2017/delay100_early_20261022.json`。

输出：`results/gao_delay100_early_20261022`；日志：`results/logs/gao_delay100_early_20261022.log`。

完成后自动生成 comparison.json、extra_filling.csv、delay100_comparison.png/pdf 及 analysis_zh.md。它们只评价新增 100 ms 的充盈收益，不宣称随后收缩、射血分数或周期稳态已得到改善。

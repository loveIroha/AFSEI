# 相同检查点的新旧循环连接对照

目的：检查物理截断边界接入后，收缩中期的核心腔压、出口压力、实际流量和左心室容积下降量如何改变。不是全周期验证，也不是同硬件耗时基准。

两组共同读取 `afsei_peak665_av060_20261019` 的 664.4 ms 检查点，保留其 EP/GPB/Land、主动张力、材料、初始几何、循环状态和数值容差。以原有 0.1 ms 步长推进 20 步，至 666.4 ms。继续使用该检查点的原始 EP/ODE 设置，不替换为其他已完成两周期实验的设置。固定在射血阶段，运行区间远离阶段切换；不人为改变主动张力，不推进全周期控制器。

- `legacy.json`：原基底虚拟压力面连接，无流场迁移。
- `physical_cut.json`：固定流道末端的物理 MAC 截面流量驱动 RCR，返回压力施加于同一截面，基底两端口的压力跳跃均为零。

新组的流道几何固定于同一次运行的预加载检查点。新增固定分隔面需要掩码及散度投影初始化速度，心肌位置、电生理状态和历史变量均不改。该初始化的动能变化写入报告。因此本实验测量“新增固定流道＋边界连接＋相应压力传递”整体干预，不能拆解成单独某个 RCR 方程项的因果影响，也不能把初始瞬态当作周期稳态。

两种压力位置的物理含义不同：旧边界压力是内部功共轭压力面乘子，新边界压力是物理截断面压力。核心腔压采用共同统计方式。出口流量也需要区别：新组满足

    dV_LV/dt + dV_tract/dt + Q_cut = 0.

因此同时比较左心室真实几何容积变化、截面流量及流道储量，不能仅比较两个出口流量就认定射血增减。

每个已接受步保存完整诊断；保留原质量守恒、RCR 平衡及力功配对检查，不放松容差，不重试失败步。`progress.json` 为运行进度，`report.json` 为完成或失败报告，`comparison.json` 和图像由 `scripts/analyze_gao_connection_pair.py` 生成。

运行命令（项目目录下，使用 afsei 环境）：

```bash
python scripts/check_afsei_cut_boundary.py configs/gao2017/connection_pair_20261026/legacy.json
python scripts/check_afsei_cut_boundary.py configs/gao2017/connection_pair_20261026/physical_cut.json
python scripts/analyze_gao_connection_pair.py results/gao_connection_pair_20261026
```

配置输出目录必须不存在，以免覆盖已有结果。现有实验已由 `launch_gao_connection_pair.py` 在 GPU0/GPU1 启动，不要重复执行上述命令。生产全周期入口和新连接重启仍未迁移。

# 二维与三维封接圆管：守恒诊断

带 `hydraulic_domain` 的二维、三维驱动自动调用 `simulation.tube_budget.TubeBudget`。未封接的旧算例保持原路径。诊断不参与力、压力、约束或接受判据；NS、EP、GPB/Land、本构、FE/IB 和时间推进不变。

每个接受步在 GPU 上累计端口 Qin、Qout，时间积分采用与 RC 一致的向后欧拉。正号表示从低轴向端进入、从高轴向端流出。二维体积与流量均乘配置中的挤出厚度。

同时计算两种有明确边界的几何体积：

- `budget_moving_lumen_volume_ml`：真实 FE 内壁，加随端节点运动的诊断封盖。
- `budget_fixed_end_volume_ml`：真实 FE 内壁，加连接到参考端环的虚拟积分面，再由固定隔板平面上的参考封盖闭合。

虚拟连接面可能被流体穿过，也可能经过 IB 虚域；不代表新增的不可渗透管壁。固定隔板参考封盖覆盖整个初始管口，其中孔外法向 MAC 速度为零；实际端口流量直接使用现有 MAC 孔面求和。

`budget_fixed_control_imbalance_ml` 是固定端控制体体积变化减逐步净入流积分。它包含虚拟连接面的相对通量、壁面运动学差异等，不能直接命名为“壁面漏流”。随动封盖扫掠量另存为 `budget_moving_cap_sweep_ml`。

P2 内壁曲线使用精确阶数的 Gauss 面积/扫掠积分；三维使用实际 P1 内表面三角形和端环扇形封盖。累计扫掠量还用于两个独立的几何恒等式残差，检验方向和连接关系。

仅在日志采样时，额外用 MAC 面通量的 RT0 重构计算旧几何上的壁面和连接面通量，与两套空间积分细分结果比较。该重构的单元散度等于 MAC 散度，但其切向分量跨单元不连续，未经单元切割的表面积分存在误差。`budget_surface_quadrature_difference_ml_s` 反映细分敏感性，并非严格误差上界。RT0 也不是原 IB 的正则化 delta 插值，故 `budget_wall_motion_minus_rt0_ml_s` 不能当作原离散方程的求解残差。

所有 `budget_*_ml` 为 mL，`*_ml_s` 为 mL/s。时间累计只在每个接受步更新；GPU→CPU 汇总只在日志、检查点或最终输出时执行。新的最终状态/检查点中保存累计量与预算步数。当前演示驱动没有新增续算功能；若单独复用保存状态检查一段窗口，必须以该窗口的旧坐标作为 `initial_x`，不能把一个末步的积分当成整段累计值。

已有结果可以这样分析，不必重新模拟：

```bash
python scripts/analyze_tube_budget.py \
  --case-2d results/tube_2d_budget_20261023 \
  --reference-2d results/tube_2d_sealed_vented_rc_check_20261023 \
  --saved-3d results/tube_3d_sealed_rc_check_20261023 \
  --output results/tube_budget_analysis_20261023
```

其中二维结果必须已包含本诊断字段；三维可直接读取旧 `final_state.npz` 的 `previous_x` 与 `x`，只检查最后一个接受步。分析只读源结果，生成 JSON 和图，不推进任何方程。

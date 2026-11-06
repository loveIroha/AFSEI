# 验证记录（2026-10-16）

代码已添加到服务器 `/mnt/large2/gjh/AFSEI`，并注册 `active-tube-3d` JSON 入口。

已执行：

1. 新增 Python 文件的语法编译检查，通过。
2. `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_circular_tube.py`：
   **6 passed**（方法来源核对后更新）。仅在 CPU 上检查网格、体积、静息力、主动收缩虚功、客观性、
   力与切线一致性，以及配置校验。测试没有调用流固时间步推进或 GPB/Land CUDA 内核。
3. `CUDA_VISIBLE_DEVICES='' python demo/run.py demo/active_tube_3d/configs/tube_gpu.json --check-config`：
   `configuration-valid`，`simulation_started=false`，默认网格 6336 节点/24576 个固体四面体，
   MAC 64×64×80；`porous_media=false`、`valves=false`。
   明确输出 `spatial_coupling=griffith-luo-2017-unified`、`time_scheme=be-be`，
   `complete_griffith_luo_2017_time_scheme=false`。
4. 服务器 `git diff --check`，通过。

尚未执行：GPU 三场时间推进、PVD 实际帧生成/ParaView 查验、收缩曲线与步长/网格收敛分析、
运行耗时和性能对比。当前没有声称默认 500 ms 运行已通过，也没有用 CPU 检查替代 CUDA 正确性验证。

已存在的二维 demo 性能工具及其未提交改动未纳入本 demo 的验证结论。

方法来源核对增加一项 CPU 检查：构造实际 `BEIBStepper` 与 `AdaptiveP1Transfer`，
确认 FSI 使用带非零非对角项的一致质量矩阵，对生成圆管在同一 IB 模板上双向传递随机
力和速度，并核对带 MAC 面控制体积权重的流体功率与固体节点功率一致。
没有调用时间步推进。Griffith–Luo（2017）空间 IB 的 unified 分支与附录 A 的
CN–AB2 时间格式分别标注；按用户确认保留 BE–BE。
当前没有圆管同条件 CN–AB2/BE–BE 效率对比，`cnab_vs_bebe_performance_measured=false`，
保留 BE–BE 不构成已经证明其性能优于 CN–AB2 的结论。

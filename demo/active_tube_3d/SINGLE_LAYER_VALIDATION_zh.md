# 三维单层圆管：电传播与电驱动收缩验证

本算例替代本轮双层圆管工作，历史双层配置和结果保留。它是明确参数的数值验证问题，不声称复现 Sharifi、真实血管或真实心室。求解器直接调用 AFSEI 已有实现；没有新增流固外迭代、规定壁面位移或体积约束。

## 物理和离散定义

| 项目 | 定义 |
|---|---|
| 参考管壁 | 轴向 z，长 2 cm，内半径 0.3 cm，单层厚 0.1 cm |
| FE 网格 | 轴向 40、周向 32、厚度 2 层逻辑单元，3936 节点、15360 P1 四面体，共形周期接缝 |
| 流体 | 管内、管外都是同一不可压缩流体，初始静止；ρ=1 g/cm³，μ=0.04 dyn·s/cm² |
| MAC 网格 | 32×32×64，箱体 1.4×1.4×2.8 cm，三个方向间距均 0.04375 cm |
| 流体外边界 | 原求解器的零压力/速度 Neumann 开放箱体边界；管口处于计算域内部 |
| 管壁 | 原 Neo-Hookean，本构 E=3×10⁵ dyn/cm²=30 kPa，ν=0.3；不是近不可压缩材料标定 |
| 固体端部 | 原有限轴向表面弹簧 k=10⁶ dyn/cm³，径向自由；不是严格固支，不新增固定接头 |
| 机械肌纤维 | 参考环向 DG0，λ=‖F f₀‖，实际变形与畸变历史反馈 Land；不使用二维等长简化 |
| 主动加载 | 原 Land PK1 应力，扣除静息参考后取正，tension_scale=1；两端 0.15 cm 平滑衰减 |
| 电传播 | 参考构形单域方程，轴向主电导；所有心肌表面绝缘，无额外电刺激边界通量 |
| 电参数 | 原项目 Cm=1 μF/cm²、χ=0.1 cm⁻¹、σ=(0.117,0.039,0.039) μS/cm；保留既有单位换算 |
| 刺激 | z 起点附近 0.1 cm 区域，5–7 ms，35.7142857 μA/cm² |
| 时间 | FSI/EP=0.05 ms，ODE=0.002 ms；EP nodal P1+CN+Jacobi-PCG；流固原 BE–BE |
| 空间耦合 | Griffith–Luo unified 积分点 FE/IB；原 MAC 压力多重网格与非线性求解 |

这些是公开、可重复的测试参数。几何、弹性、支撑和波形没有全部匹配某篇论文，因此不能给出“文献复现误差”。保留三维环向长度反馈会自然限制肌肉短缩，强行追求完全闭合不是验收目标。默认半径保护比仍为 0.2。

不设置瓣膜、多孔区、RC 或隔离储液区。本例旨在检查传播和收缩，不用于验证单向净泵血、出口循环阻抗或真实管口压力。管外流体、有限端弹簧和壁内虚拟流体都是当前 IB 问题定义的一部分。

## 运行

在服务器 AFSEI 环境，从项目根目录执行。每次使用新的输出目录。

```bash
# 固定几何的电传播与等长张力参考，不推进流体或固体
python demo/active_tube_3d/validate_single.py --config demo/active_tube_3d/configs/tube_single_validation_gpu.json --mode ep --output results/single_ep

# 120 ms 三场短程验证
python demo/active_tube_3d/validate_single.py --config demo/active_tube_3d/configs/tube_single_validation_gpu.json --mode coupled --output results/single_coupled

# 无刺激和零主动应力对照，仍调用同一三场驱动
python demo/active_tube_3d/validate_single.py --config demo/active_tube_3d/configs/tube_single_validation_gpu.json --mode no-stimulus --end-ms 10 --output results/single_no_stim
python demo/active_tube_3d/validate_single.py --config demo/active_tube_3d/configs/tube_single_validation_gpu.json --mode passive --end-ms 10 --output results/single_passive
```

`tube_single_smoke_gpu.json` 为 2 ms 接入检查；`tube_single_cycle_gpu.json` 为 500 ms 单次刺激观察窗口。500 ms 文件存在不表示完整恢复已验证，也不代表周期重复刺激。可用 `demo/run.py CONFIG.json` 调用原标准驱动；额外验证图和统计由 `validate_single.py` 生成。

## 输出和解释

- `config.resolved.json`：本次实际参数。
- `report.json`、`history.json`：状态、采样张力/形变/流量/散度/迭代计数与耗时。
- `final_state.npz`：参考与当前几何、电位和首次激活时间等。
- `validation.json`、`validation.png`：沿轴向首次越过 −20 mV 的时间、传播速度拟合、实际施加的张力与收缩。
- 三场 `vtk/solid.pvd`、`fluid.pvd`、`ep.pvd`；EP-only 仅 `vtk/ep.pvd`。

传播速度只拟合管长 20%–80% 内、刺激区之外、所有周向和径向节点均激活的截面。未激活截面的时间为缺失值，不能当作 0 ms。固定壁面的张力是等长参考，三场张力受到真实短缩和速率反馈，二者不应强求相等。

本轮验证要分别记录：①局部刺激后的顺序传播；②无刺激/无主动应力时的静息响应；③非零主动应力下的环向收缩；④原流固收敛与有限性；⑤后续空间/时间收敛。运行完成不自动等于这五项全部通过。

当前三维开放管口的采样流量是截面插值诊断；运动管口的容积守恒需要相对通量和运动控制面处理，不能直接用 Q_in−Q_out−dV/dt 宣称漏流率。此配置没有沿用封接 RC 算例的固定控制体预算。

不以跨模型/跨硬件耗时宣称加速。短程计时包括输出，应单独报告初始化。后续性能基准必须固定物理问题、离散误差和求解容差。

## 2026-10-24 实际验证

结果位于 `results/single_tube_validation_20261024/`，本轮六个任务均已结束。

- 原圆管与新统计测试：8 项通过。实际 CUDA 三场另行完成，不能用上述单元测试替代。
- 无刺激 10 ms：无激活、无位移；有刺激但主动应力置零 60 ms：全管激活、无位移。
- 正式刺激从 5 ms 开始，粗网格首次激活 6.065 ms，全部激活 39.857 ms。没有预激活或探针加载过渡。
- 同一三场驱动到 60 ms：1200 步，最大内向位移 0.18861 mm，最大截面面积缩小 12.030%，最小等效半径缩小 6.208%，管腔体积缩小 4.137%。末时刻实际最大张力 5.062 kPa；张力比例为 1。
- 采样最大散度 7.277×10⁻¹⁰ s⁻¹，采样最大非线性残差/阈值 0.948。它们是离散求解诊断，不等于移动管腔体积误差。
- 主循环 225.461 s（含输出），初始化 14.286 s，平均 0.18788 s/步。本次是短程成本，不是完整周期或跨方法性能结论。
- 轴向 EP 网格 0.5/0.25/0.125 mm 的拟合速度分别为 55.677/64.818/68.830 cm/s；相邻两档以较细结果为分母的差异为 14.10% 和 5.83%。只检查了轴向分辨率，没有证明完整空间或时间收敛。
- 流固验证使用 0.5 mm 轴向固体网格；两档额外 EP 检查在固定几何上执行，没有改动流固验证的网格。
- 三场固体、流体、EP 的 PVD 各有 7 帧，引用文件全部存在。

汇总图：`analysis/single_tube_validation.png`；汇总数据：`analysis/summary.json`。
可用 `python demo/active_tube_3d/summarize_single.py --root results/single_tube_validation_20261024` 重画，不需重算模拟。

当前通过的是短程功能与因果对照。完整收缩—舒张、移动管腔预算、时间和流固空间收敛、独立参考方法对照仍未通过验证。500 ms 配置仅已准备，未启动。

# 三维电激活圆管收缩

**当前单层验证入口（2026-10-24）**：参见 [参数、边界与验证记录](SINGLE_LAYER_VALIDATION_zh.md)。
使用 `configs/tube_single_validation_gpu.json` 和 `validate_single.py` 检查局部刺激、轴向传播、环向收缩以及无刺激/无主动应力对照。
本轮不继续双层 Sharifi 复现；历史配置保留。单层 60 ms 三场短程已完成，完整恢复和收敛验证尚未完成。
下文 `tube_gpu.json` 为原始单层配置，其尺寸、材料和黏度与新增验证配置不同。

本算例验证 EP → GPB/Ca → Land → 环向主动应力 → 管壁与流体运动的完整连接。
采用完整圆管的三维计算，不使用轴对称降维。没有多孔介质、阀门、接触和循环模型。
模型由电激活产生收缩，不预设管壁位移，也不预设主动应力的时间波形。

## 配置与启动

在 AFSEI 项目根目录激活现有环境后，先检查配置（不生成网格，不编译 CUDA，不启动模拟）：

```bash
python demo/run.py demo/active_tube_3d/configs/tube_gpu.json --check-config
```

需要实际计算时：

```bash
python demo/run.py demo/active_tube_3d/configs/tube_gpu.json --output results/active_tube_3d
```

短时调试可增加 `--end-ms 2`；被动对照增加 `--passive`；无刺激对照增加 `--no-stimulus`。
这两个对照仍计算全部三场，分别将主动应力系数或刺激电流置零。
运行目录必须不存在或为空，避免覆盖已有结果。
设备使用激活脚本中的 `AFSEI_GPU` 选择；`--device cuda:0` 指可见设备中的第一张卡。
本次添加仅进行配置、网格及有限元力/切线检查，没有推进三场时间步或测量 GPU 性能。

## 几何、材料与边界

全部求解量使用 cm–g–s 单位制。默认 JSON 是该算例的唯一参数入口。

| 项目 | 默认值与含义 |
|---|---|
| 圆管 | 沿 z 轴，长 4 cm，内半径 1 cm，壁厚 0.2 cm |
| 放置 | 中心轴 x=y=2 cm，z 从 0.5 到 4.5 cm |
| 固体/EP 网格 | 32 个轴向区间、64 个周向区间、2 个壁厚区间；6336 个节点、24576 个 P1 四面体；周向共享接缝节点 |
| MAC 网格 | 4×4×5 cm 的规则盒，64×64×80 个单元，间距均为 0.0625 cm |
| 流体 | 不可压 Newtonian，密度 1 g/cm³，黏度 **10 dyn·s/cm² = 1 Pa·s** |
| 被动管壁 | 可压 Neo-Hookean，E=10⁶ dyn/cm²=100 kPa，ν=0，故 μ=5×10⁵ dyn/cm²、λ=0 |
| 管端 | 两个环形端面施加轴向参考面弹簧，β=10⁷ dyn/cm³，径向自由；是有限刚度支承，不是精确固定 z 位移 |
| 流体外边界 | 沿用原 3D BE MAC：六个盒面 p=0，速度齐次法向导数；不施加入流速度或端口阻力 |
| EP 边界 | 管壁所有表面绝缘，n·D∇V=0 |
| 刺激 | z≤0.65 cm 的整圈管壁，10–12 ms，35.7142857 μA/cm² |
| 时间 | FSI/EP 均 0.05 ms，ODE 子步 0.002 ms；默认算至 500 ms，共 10000 步 |

被动能量采用项目已有函数：

`W = μ/2 (I₁−3) − μ ln J + λ/2 (ln J)²`。

端部弹簧牵引仅为 `t_z = −β (z−Z)`，按参考三角面的一致质量矩阵积分。
外部背景流体和管腔流体共用原来的 MAC 盒；管壁是浸入式实体。
管道开口位于盒内，外部背景区与管腔通过开口相通，盒面不等同于精确管口边界。
原投影边界是压力与速度导数条件，不能解释为严格的 Cauchy 零牵引条件。

## 电激活、主动应力与离散

MAC/IB **空间耦合**采用 Griffith–Luo（2017）的 unified weak form：
规则交错 MAC 网格上的速度/压力、有限元实体管壁、随变形选择的单元内部交互积分点、
Peskin 四点核，以及双向一致质量矩阵投影。令 b 为 FE 积分节点力，则
`M F=b; f=Kᵀ W B F/h³; M U=Bᵀ W K u`，两向使用同一构形及积分点，
在质量矩阵容差内满足离散功率伴随。FSI 的 M 是一致质量矩阵，不随 EP 的 nodal 集中质量改变。
这里选用论文的 unified 分支，没有额外单独散布表面力。
参考：[Griffith & Luo, 2017, DOI 10.1002/cnm.2888](https://arxiv.org/html/1612.05916)。

**时间推进**按用户确认保留现有 BE–BE，不等同于该论文附录 A 的 CN–AB2/半步构形格式。
JSON 分别用 `fsi.spatial_coupling` 和 `fsi.time_scheme` 固定这两项含义，不能仅修改标签切换算法。
当前没有圆管算例的 CN–AB2 与 BE–BE 同条件性能数据，不能将保留 BE–BE 解释为已证实其更快。
论文附录格式通常减少显式弹性更新的单步求解工作，但有弹性/IB 步长限制；BE–BE 有联合
非线性求解开销。总耗时取决于每步耗时、可接受步长、误差目标及失败重算，
对比必须使用相同圆管模型、边界及网格；已有闭盒 CN–AB2 后端不能直接替代这里的开放盒边界。

EP 调用现有 `Monodomain`：P1 nodal FEM 顶点积分/集中质量，默认 CN 扩散格式，
轻量 Jacobi-PCG。`ep.numerics.backend` 可设为 `amgx_pcg_amg` 使用已安装的可选 AmgX；
不要求另建 PDE 求解器。参考坐标中的电导主方向为 z，横向方向为径向和环向。
电参数继承已有 GPB/Land 演示配置，而非声称来自 FEBio；数值电导单位和源代码
换算约定见 `ep/monodomain.py`，本算例没有额外乘 1000。

GPB 和 Land 直接调用现有 CUDA ODE 内核。管壁的机械纤维是参考构形中的环向 DG0
单位向量 `fθ=(-sinθ, cosθ, 0)`，与轴向电导方向分别定义。
采用已有 Land PK1：

`P_active = ΔTa [1+4.9(max(|F fθ|,1)−1)] (F fθ) ⊗ fθ`。

`ΔTa=max(Ta−Ta_rest,0)`；`Ta_rest` 由初始静息钙与 λ=1 的 Land 平衡态确定。
这沿用二维 demo 的静息张力平衡约定，防止初始化即发生收缩。
`tension_scale=1` 不对张力进行额外放大。端部 0.2 cm 使用平滑空间渐消，
不改变中段张力。没有人为快速卸载过程，张力消退由 GPB/Land 状态决定。

被动与主动 PK1 在现有 P1 有限元力中一起积分，Land 的 P1 节点状态在现有
15 点力学积分规则上插值。机械切线通过原 `P1Tangent` 对相同 PK1 及弹簧求导，
包含张力对当前环向伸长的依赖。环向伸长向 ODE 节点的反馈使用项目已有的一致 L² 投影。

每步先用已接受的伸长与畸变状态推进 GPB/Land、EP，再调用**原 `BEIBStepper`**
隐式求解新时刻管壁与流体；IB 积分、扩散、压力多重网格及 Newton–GMRES 均调用已有实现。
FSI 内迭代不推进 GPB/Land，不改变其时间历史；畸变状态在成功接受 FSI 步后才更新。
EP/ODE 与 FSI 之间是一阶顺序分裂；扩散为 CN 不意味着整个三场系统二阶。
默认不使用新增的额外耦合外迭代，也未修改既有二维/左心室求解入口。

## 结果

- `config.resolved.json`、`fsi.resolved.json`：实际参数及转换后的原 FSI 配置。
- `history.json`：管腔体积、截面积、半径比、电位、钙、施加张力、截面流量和求解迭代数。
- `vtk/solid.pvd`：已变形管壁、位移、FE 节点力和 det F。
- `vtk/fluid.pvd`：完整规则网格中的流速、压力和散度。
- `vtk/ep.pvd`：已变形管壁上的电位、钙、环向伸长、首次越过 −20 mV 的激活时刻；未激活标记为 −1。
- `checkpoint.npz`（周期输出）、`final_state.npz`：已接受时刻的三场状态快照。
  当前 demo 尚未提供恢复运行入口；这些快照不能当作已经验证的 restart 功能。
- `report.json`：完成或失败状态；失败时保留已接受状态及已写入的 PVD 帧。

管腔体积由变形内表面与两个**仅用于积分**的端盖计算，端盖不封闭流体，也不施加载荷。
截面流量为沿 +z 的带符号量，负入口流量表示从左端向外排液。
输出通过 MAC 面速度和移动截面的二维投影多边形估算流量，不作为质量守恒约束。
`minimum_radius_ratio` 是截面积等效半径比。低于配置中的 0.2 时停止，防止进入无接触模型
不适用的几何范围；此条件不产生接触力，也不追求闭塞。
等压、无阀圆管收缩可向两端排液，舒张可从两端吸液；本算例不验证单向泵送。

## 文献对应及验证范围

几何、被动材料 E/ν、流体密度和 1 Pa·s 黏度参考 FEBio 官方
[Quarter-Symmetry Contracting Tube](https://febio.org/knowledgebase/case-studies/fluid-structure-interactions/flow-regulation-using-biphasic-fsi-domains/)
（其方法参考 Shim et al., 2021, DOI: [10.1115/1.4050646](https://doi.org/10.1115/1.4050646)）。
该官方案例有两端可变渗透率的双相区，并使用给定余弦波主动应力；本算例按用户要求省去
多孔区，改由 GPB/Land 生成环向应力，保留 AFSEI 不可压 MAC/IB 离散。
此外采用完整圆管 P1 四面体、有限轴向弹簧、空间渐消及外部背景流体。
因此只作为电激活收缩的简化验证案例，**不是官方 FEBio 结果的定量复现或效率对比**。
后续性能对比必须保持物理配置、网格、时间步和误差标准一致，并在空闲 GPU 上计时。

CPU 检查覆盖接缝/拓扑、解析多边形体积、静息力平衡、主动收缩虚功、刚体客观性、
主动/被动/端部弹簧切线与有限差分，以及配置中不支持物理项的拒绝。
这些检查不构成 GPU 三场演化、收缩幅度、时间步稳定性或实际效率验证。


## Sharifi RC boundary-validation stage (2026-10-23)

A separate `configs/tube_sharifi_rc_gpu.json` connects the implicit two-element RC outlet from Sharifi (2021), Eq. (2), with no porous domain. See [Chinese scope, units and commands](SHARIFI_RC_zh.md). This loads the outer MAC box face; the immersed tube mouths, exterior fluid and original single-layer geometry have not yet been aligned with the paper. This stage does not establish quantitative reproduction.


## Isolated hydraulic path update

`configs/tube_sharifi_rc_gpu.json` now selects sealed end reservoirs and fixed normal-flux septa/collars. Use [current scope and numerical boundary details](SEALED_PORTS_zh.md). The former unpartitioned box-RC case is retained as `configs/tube_box_rc_legacy_gpu.json`. Flexible-wall material/EP and FE/IB transfer are unchanged. The MAC pressure operator now enforces the fixed partitions, using Galerkin-MG-preconditioned PCG. The geometry is still centimeter-scale and single-layer; it is not a Sharifi reproduction.

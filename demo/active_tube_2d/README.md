# 二维电激活直管：AFSEI demo

本算例验证电传播 → GPB 钙动力学 → Land 张力 → 管壁变形 → 流体运动。
采用用户确认的上下壁配对主动牵引，代表二维纵截面省略的环向肌肉。
壁面位移由求解器计算，不指定收缩位移曲线；不添加瓣膜或 Windkessel。
它是受文献启发的功能验证算例，**不是完整人体血管、胚胎心脏或某篇论文的定量复现**。

## 运行与结果

在服务器 `/mnt/large2/gjh/AFSEI` 中运行：

```bash
source scripts/activate_afsei.sh
python demo/run.py demo/active_tube_2d/configs/tube_gpu.json --output results/my_active_tube
```

后台运行：

```bash
mkdir -p results/logs
nohup python demo/run.py demo/active_tube_2d/configs/tube_gpu.json \
  --output results/my_active_tube > results/logs/my_active_tube.log 2>&1 &
```

必须先激活环境。输出目录须为空，重复运行使用新目录；本 demo 暂不支持断点续算。
默认模拟 100 ms，2000 个流固步。缩短检查可附加 `--end-ms 20`。
同一 JSON 可以运行两个独立对照：

```bash
python demo/run.py demo/active_tube_2d/configs/tube_gpu.json --output results/my_tube_passive --passive
python demo/run.py demo/active_tube_2d/configs/tube_gpu.json --output results/my_tube_unstimulated --no-stimulus
python demo/active_tube_2d/plot.py results/my_active_tube \
  --passive results/my_tube_passive --no-stimulus results/my_tube_unstimulated --gif
```

`--passive` 关闭施加到管壁的主动力，仍计算 EP/GPB/Land；`--no-stimulus`
关闭电流刺激，保留 Land 静息平衡初始化及其载荷。后者用于识别初始化瞬态。
绘图只读结果，不重复运行求解器。

- `config.resolved.json`：实际参数，包括命令行覆盖。
- `report.json`、`history.json`：状态、耗时、间距、流量和电位等。
- `snapshots.npz`：真实二维坐标、完整 MAC 场及各时刻节点电位。
- `vtk/solid.pvd`、`vtk/fluid.pvd`、`vtk/ep.pvd`：ParaView 时间序列。
- `tube_overview.png/.pdf`、`tube_activation.gif`：离线绘图生成；形变比例为 1。

## 几何、材料与参数来源

统一采用 cm–g–s；电位内部为 V，电流为 μA/cm²。
二维流量为 cm²/s，即每单位面外厚度的体积流量，不直接标为 mL/s。

| 参数 | 默认值 | 依据或选择理由 |
|---|---:|---|
| 可变形管长 | 2.4 cm | 为现有成人 GPB/Land 时间尺度选取的厘米级验证尺寸，非胚胎尺寸 |
| 初始内腔间距 | 0.6 cm | 演示设计值，便于分辨电传播和缩窄 |
| 单侧壁厚 | 0.1 cm | 演示设计值 |
| 背景流体域 | 3.2 × 1.6 cm | 管壁端部及外侧保留 IB 核支撑空间 |
| MAC 网格 | 64 × 32，间距 0.05 cm | 基础演示分辨率，未宣称网格收敛 |
| 每侧壁三角网格 | 轴向 48、厚向 2 个矩形，各分两三角形 | 力学 P2；EP 使用同一网格顶点的 P1 |
| 流体/浸入式惯性密度 | 1.025 g/cm³ | Battista–Miller 表 I 的 1025 kg/m³ |
| 动力黏度 | 0.04 dyn·s/cm² = 4 mPa·s | 沿用 AFSEI 牛顿血液的演示值；不是该论文的唯一黏度，该文扫描黏度 |
| Neo-Hookean 剪切模量 μs | 10⁵ dyn/cm² = 10 kPa | 参考 Sharifi 等心管研究的约 10 kPa 材料常数量级；不将不同能量形式的 C10 直接等同于 μs |
| Lamé λs | 2×10⁵ dyn/cm² = 20 kPa | 本算例选择，等效小应变泊松比 1/3；不是严格不可压缩固体 |
| 端面弹簧系数 | 10⁶ dyn/cm³ | 数值支承参数，未拟合生理数据；不是严格固定端位移 |
| 左、右端压力 | 200、0 dyn/cm² | 20 Pa 压差，演示设计值；维持向右的基础流动 |
| 电导主方向 | (1,0)，沿管长 | 与收缩方向分开定义 |
| Cm、χ | 1 μF/cm²、0.1 cm⁻¹ | 保留现有 AFSEI 源模型的数值系数与单位约定 |
| 纵/横电导 | 0.117/0.039 μS/cm | 同现有 AFSEI；D∥=1.17 cm²/s，D⊥=0.39 cm²/s |
| 刺激区 | 两壁左端 0.1 cm，含整个壁厚 | 数值演示设计值 |
| 刺激时段、电流 | 10–12 ms，35.7142857 μA/cm² | 沿用已验证的 AFSEI 电刺激量级 |
| FSI、EP、ODE 步长 | 0.05 ms、0.05 ms、0.002 ms | 每 EP 步含 25 个 ODE 子步 |
| 主动力倍率 | 1 | 不额外放大或削弱 Land 张力 |
| 张力参考 | 静息平衡张力 T0 | 施加正的张力增量；见下文，避免未刺激的零初值建立张力瞬态 |

三维参数接口仍保留第三个电导分量；二维计算明确不使用面外分量。
若改细网格或增大模量，原二维显式 FSI 的稳定步长可能需要减小；程序检查
黏性数、对流稳定条件、单步位移、正 Jacobian 和最小管隙，并在不满足时停止。
这些检查不等价于完整的稳定性证明。

参考文献：

1. Battista NA, Miller LA. *Bifurcations in valveless pumping techniques from a
   coupled fluid-structure-electrophysiology model in heart development* (2017).
   [论文与全文](https://arxiv.org/abs/1709.05264)，尤其 §II.C、式 (14) 与图 6。
   该文用二维 IB、沿心管传播的电信号和连接上下壁的肌肉力。原文采用
   FitzHugh–Nagumo、弹簧/梁壁和封闭环路；本 demo 明确改用 GPB/Land、
   Neo-Hookean 有厚度壁与两端压力开口，未复制其微米级几何。
2. Sharifi A, Gendernalik A, Garrity D, Bark D. *Valveless pumping behavior of the
   simulated embryonic heart tube as a function of contractile patterns and
   myocardial stiffness*. Biomech Model Mechanobiol 20, 2001–2012 (2021).
   [DOI](https://doi.org/10.1007/s10237-021-01489-7)，
   [作者所在机构的摘要与书目信息](https://research.monash.edu/en/publications/valveless-pumping-behavior-of-the-simulated-embryonic-heart-tube-/)。
   该研究支持使用 Neo-Hookean 心管并考察约 10 kPa 的材料常数量级，
   不能据此声称本算例全部参数已由实验标定。

## 方程和边界条件

### 电生理与主动载荷

管壁参考域上求解单域方程：

\[
 \chi C_m\partial_t V-\nabla_X\cdot(D\nabla_X V)
   =\chi(I_{stim}-I_{ion}),\qquad n\cdot D\nabla_X V=0.
\]

两侧壁是两个不连接的电学域；在其左端同时施加刺激。
**nodal FEM** 指三角形 P1 基函数配顶点积分，质量矩阵为对角矩阵。
扩散采用 CN，默认轻量 Jacobi-PCG；可在配置中选择现有 AmgX 后端。
离子电流仍使用现有 GPB/Land 的一阶分裂与 CUDA ODE 子步。
没有改成将电压超过阈值后直接指定一个收缩波形。

力学反馈的环向简化伸长为上下内壁配对位置距离与参考距离之比。
该伸长进入 Land 长度依赖与畸变历史；电导张量固定在参考域。
原 Land 的张力参数及畸变衰减率复用现有 `ep.land`。

本 demo 的 Land 初值由初始 GPB 胞质钙、λ=1、Zs=Zw=0 下的 ODE
平衡方程确定，已用现有 Land 更新函数验证为不动点。记录参考张力 T0，
加载 `[Ta−T0]+`；这是本算例明确的静息参考约定，**不是原三维 LV 驱动的
初值或载荷定义被全局改变**。若沿用 Land 全零初值直接加载绝对张力，
静息钙也会在没有电刺激时缓慢建立张力，使很柔顺的直管先自行收缩。
因此零初值试跑只保留为调试记录，不作为最终 demo 的电刺激验证。
`tension_max_dyn_cm2` 是原始 Land 张力；`applied_tension_increment_max_dyn_cm2`
才是扣除静息参考并乘倍率后的最大施加载荷张力。

对上下内壁对应的 P2 线积分点定义

\[
 \lambda_\theta=\frac{\lVert x_+-x_-\rVert}{H_0},\quad
 p_a=\alpha\frac{2h_0}{H_0}[T_a-T_0]_+\,w(X),\quad
 t_- =p_a\frac{x_+-x_-}{\lVert x_+-x_-\rVert},\quad t_+=-t_- .
\]

这里是以参考半径 H0/2 为尺度的**简化环向力平衡**，按参考线长度积分，
并非有限厚度圆管的精确 Laplace 定律或三维主动 PK1。
`w` 在距端部 0.1 cm 内光滑减至零，其余为 1，表示被支承的被动端部。
采用相同积分点及 P2 形函数施加配对力，保持净力和净力矩为零。
此模型表示被纵截面省略的环向肌肉，**没有在流体中放置实体横向肌肉杆**。

### 管壁与流体

管壁采用平面应变 Neo-Hookean 材料，F33=1：

\[
 W=\frac{\mu_s}{2}(F:F-2)-\mu_s\ln J+
   \frac{\lambda_s}{2}(\ln J)^2,\qquad
 P=\mu_s(F-F^{-T})+\lambda_s\ln J F^{-T}.
\]

两端壁面使用弹簧支承，壁的其余部分可运动。
使用 AFSI 现有浸入式等密度惯性表述，没有另加独立的固体密度修正。

流体仍为规则 MAC 网格上的不可压缩 Navier–Stokes 方程。
**两端不规定流速**，沿用 AFSEI 开口算子的约定：端面压力给定、速度法向导数为零；
上下外框为不滑移边界。对应向量 Laplacian 写法的自然伪牵引
`(μ∇u−pI)n=−pb n`；没有声称实现完整对称 Cauchy 应力的离散牵引。
因此速度采用 Neumann 开口时，**压力泊松方程的两端是 Dirichlet 条件**。
不能把两端压力泊松边界也机械地改成齐次 Neumann，否则不能施加这个压差。

管壁浸入背景流体域，管壁外侧和固体覆盖区域仍存在 AFSI 背景场。
这不同于只在随壁运动的管腔内建立贴体网格。左右外框位于管壁端部外 0.4 cm。
输出同时列出整个背景域端部流量和中部管腔流量；两者不可混用。
中部管腔流量使用移动壁位置与 MAC 面元重叠长度求积，属于有限分辨率的诊断。

### 保留的 AFSI 路径及实现边界

- 复用原 `mac2d.ValveStepper`：流体 → 壁面位移 → 更新力的显式滞后顺序。
  这是仓库原有二维路径，**不是**三维左心室的隐式 Newton 路径，也没有新增强耦合外迭代。
- 复用 `TriangleTransfer` 的 P2 积分传递、质量求解及伴随 gather/spread。
- 复用二维几何多重网格 V-cycle 和 CUDA Graph 工作区；只扩展左端压力边界
  的算子、对角线和延拓奇延拓。原速度入口模式默认行为不变。
- EP 扩展到真实二维三角形，不使用薄层四面体冒充二维。
- 目前是一次刺激的 100 ms 功能演示；未验证完整舒张周期、接触闭合、
  大幅压塌或网格/时间步收敛。间距低于参考值的 20% 时停止，不静默穿透。

## 文件分工

`demo/active_tube_2d/configs/tube_gpu.json` 只定义算例；`run.py` 只解析入口；
`src/afsi_torch/simulation/active_tube.py` 组织时间推进与输出；
`solids/planar_tube.py` 定义网格/Neo-Hookean 壁；`ep/planar_tube.py` 定义
明确命名的二维载荷简化；`mac2d/pressure_channel.py` 提供可复用的双压力开口。
主 EP、ODE、传递、多重网格及二维 FSI 求解器均直接调用库模块。

实际 GPU 检查和结果另见 `VALIDATION.md`。

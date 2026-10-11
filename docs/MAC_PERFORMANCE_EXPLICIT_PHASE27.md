# 第二十七阶段：MAC 性能优化与显式流体评估

## 本次交付

主线继续采用原 128³ MAC、13 cm 外箱、BE–BE 隐式流固方程和原压力边界。
本次改变执行方式，不改变源 PK1、EP/GPB/Land、80 ms 卸载、四阶段规则、RCR 或接受容差。
另提供显式黏性加压力投影的独立性能探针；它没有接入正式周期驱动，不能作为完整显式三场稳定性证明。
本阶段只做短程验证，不运行完整周期。

1. 基底/端口流量与压力反力改用紧凑 CSR 和材料面片矩阵。沿用原积分点、权重、面片划分，流量与反力保持转置配对。
2. 同一步内按阀门活动分支复用基底预条件器 LU；行缩放改为批量矩阵运算。
3. 压力模式功计算融合为 GPU 归约。紧支撑保证外箱法向边界面的目标力为零，因此面权重可精确化简为单元体积。
   刚体力矩检查通过一维坐标矩求和，省去重复构造三维坐标和向量场。
4. 质量矩阵 CUDA Graph 每批检查间隔由 4 调整为 16，减少 CPU/GPU 往返。
   保留原 PCG 相对/绝对容差、40 步真残差重算、GPU 收敛活动标志及最终真残差验收。
   新检查点记录 `compact_cap_operators` 与 `mass_graph_check_every`；旧检查点缺省采用新执行配置。

预条件器缓存仅限当前物理步，且分别缓存开放/闭合分支；不跨越几何和步长变化盲目复用。
原积分算子仍可用于对照，紧凑版本的类型、形状和有限压力检查保留。

## 实际 GPU 计时

硬件：RTX 4090，物理 GPU 1；float64；原真实 LV 参考网格 26,889 节点、135,430 单元。
共同输入为已接受的 560 ms 原生 RCR 检查点：
`results/phase25_mac128_balanced_rcr_native_60ms_v5/checkpoint.npz`。
优化前代码为 `11a93ba675ce74226e1afd4fd127bdcc27e37ac1`。

每组独立加载同一输入，先走 1 步初始化，再计时 5 个物理步；计时前后同步 CUDA。
包含 EP、ODE、固体、流体、接口和接受诊断，排除检查点/文件输出和首次编译。
结果属于射血初段的短窗口，不是全周期均值。两种步长覆盖的物理窗口长度不同。

| 执行版本 | dt / s | 暖步平均 / s | 暖步范围 / s | 回退 |
|---|---:|---:|---:|---:|
| 优化前 | 1e-4 | 2.548145 | 2.438–2.734 | 0 |
| 紧凑基底算子与预条件器复用 | 1e-4 | 2.369386 | 2.297–2.413 | 0 |
| 最终版本 | 1e-4 | 2.339311 | 2.304–2.364 | 0 |
| 最终版本，限制为一个 EP tick | 5e-5 | 1.635850 | 1.598–1.706 | 0 |

最终版本同 dt 平均耗时减少 8.20%，速度为原来的 1.089 倍。
首次初始化步约 12–13 s，包含图捕获、编译及预条件器建立。
按两个 800 ms 周期的 1.6 s 物理时长外推：dt=1e-4 的 16,000 步约 10.4 h；
dt=5e-5 的 32,000 步约 14.5 h。首次启动、文件输出、阶段差异和可能回退另计，不能视为完周期保证。
小步长虽然单步更快，单位物理时间仍更慢。

最终版本 dt=1e-4 的暖步分段平均：

| 工作 | 秒/步 |
|---|---:|
| GPB/Land | 0.05751 |
| 单域 PDE | 0.01737 |
| 隐式 FSI，含接口准备与诊断 | 2.24378 |
| 载荷准备和应变反馈 | 0.01996 |

EP PDE 和 ODE 合计约 3.2%，隐式 FSI 约 95.9%。
该窗口每步约 4 次 Newton 修正、58–61 次流体响应，主要来自 Krylov 雅可比作用。
这是原联合隐式残差内部的响应求解，不是额外增加外层完整流固迭代。
CPU 剖析还显示重复传递/质量矩阵求解和 GPU 同步较多；异步 GPU 的 CPU 累积时间不能直接相加作为独立耗时比例。

## 数值验证范围

最终版本与优化前在相同 560.6 ms 端点比较全部检查点数组及物理时钟、阶段、载荷历史与 RCR 状态：

| 量 | 最大绝对差 |
|---|---:|
| 心肌位置 | 7.99e-15 cm |
| 流体速度 | 2.18e-9 cm/s |
| 流体压力 | 1.75e-6 dyn/cm²，约 1.32e-9 mmHg |
| 固体力 | 1.80e-7 dyn |
| 电压 | 1.43e-16 V |
| RCR 电容压力 | 0 mmHg |

dt=1e-4 六步最大质量误差 4.93e-10 ml/s；dt=5e-5 六步为 1.15e-9 ml/s，均无回退。
紧凑算子与原积分实现直接比较闭合、MV、AV 三种分组的流量/反力及功配对。
还检查完整动态雅可比、预加载、四阶段事件、恢复重放、失败回退，以及每批 4/16 次的质量矩阵收敛与输出所有权。
相关回归共 178 项通过，用时 133.63 s；日志为 `/tmp/afsei_phase27_regression.log`。
完整周期与局部逐点无穿透仍需分别验证；积分质量守恒不等于局部滑移已经消失。

结果目录：`results/phase27_perf_baseline_v1`、`phase27_perf_opt1_v1`、`phase27_perf_opt2_v1`、
`phase27_perf_dt50_v1`；完整字段比较存于 `phase27_perf_opt2_v1/comparison.json`。
这些检查点和测试产物不提交 Git。

## 显式流体：能省多少，哪些限制仍存在

现有对流项已经按旧速度做一次半拉格朗日推进；耗时主要不来自隐式求解对流项。
最小的算法变更是将黏性改为显式，保留压力投影与心肌/接口约束的隐式求解。
将心肌力也完全滞后会进一步改变耦合方法，其稳定性不能由流体 CFL 单独判断。

当前 h=13/128=0.1015625 cm，rho=1 g/cm³，mu=0.04 g/(cm·s)，nu=0.04 cm²/s。
对三维中心差分黏性算子使用前向 Euler，一个充分的扩散稳定步长上限是：

```text
dt_nu <= 1 / [2 nu (1/hx² + 1/hy² + 1/hz²)]
      = h² / (6 nu) = 0.0429789225 s
```

因此 1e-4 和 5e-5 s 均远低于纯黏性限制。
真实 560 ms 速度场的 `dt * sum(max|u_i|/h_i)` 分别为 0.18730 和 0.09365。
这只是该状态的速度尺度，不是全周期最大 CFL，也不是任意显式对流格式的稳定性证明。
当前半拉格朗日格式的性质不能直接等同于前向 Euler 中心差分对流。

独立探针冻结同一个对流出发场，不加心肌/端口体力，只比较：

```text
原流体块：       u* = (I - dt nu L)^(-1) u_departure; u = P u*
显式黏性探针：   u* = u_departure + dt nu L u_departure; u = P u*
```

P 为相同原 MAC 零压力外箱的压力投影，两个路径均真实求解同一压力 Poisson 算子。
每个路径预热 3 次，再计时 30 次；不包含共同的对流计算、EP、固体、接口或初始化。

| dt / s | 原 BE 流体块 / ms | 显式黏性＋投影 / ms | 流体块加速 | 与 BE 速度相对差 |
|---|---:|---:|---:|---:|
| 1e-4 | 9.351 | 3.032 | 3.084 倍 | 1.001e-6 |
| 5e-5 | 8.694 | 2.901 | 2.997 倍 | 2.513e-7 |

显式探针最大散度分别约 1.71e-13、1.43e-13 s^-1，压力真残差通过原阈值。
结果在 `results/phase27_explicit_fluid_probe_v1/report.json`。
这些小差值仅衡量单次无体力流体块，不能推断有主动心肌力的时间累积误差或完整三场稳定性。

按约 60 次流体响应粗略计算，省去每次约 6 ms 对应每步约 0.36 s 的潜在节省，
即当前总步耗时的约 15%；这只是成本估计。真正接入后迭代次数、强迫项和缓存会改变，应重新实测。
不能将流体块的 3 倍加速直接套用到全程序。

## 文献依据及本项目取舍

- [Newren, Fogelson, Guy, Kirby (2007), Unconditionally Stable Discretizations of the Immersed Boundary Equations](https://www.math.ucdavis.edu/~guy/papers/newrenfogelsonguykirbyJCP2007.pdf)，JCP 222:702–719，DOI 10.1016/j.jcp.2006.08.004。
  显式 IB 弹性耦合可受比对流/扩散更严格的步长限制。其稳定性证明依赖特定力学算子、流体模型与离散伴随条件；
  不能直接替非线性主动心肌、当前开边界和 RCR 给出无条件稳定结论。
- [Newren et al. (2008), A comparison of implicit solvers for the immersed boundary equations](https://www.math.ucdavis.edu/~guy/papers/newren_fogelson_guy_kirby_CMAME.pdf)，CMAME 197:2290–2304。
  比较了半隐式 IB 线性系统的求解成本；采用滞后但配对的传递算子可避免一部分几何非线性。
  线性本构下的结论不能直接把当前非线性心肌改成一步显式力更新。
- [Griffith & Luo (2017), Hybrid finite difference/finite element immersed boundary method](https://onlinelibrary.wiley.com/doi/full/10.1002/cnm.2888)，IJNMBE 33:e2888。
  使用规则流体网格与 FE 结构，可用于心脏力学；附录 A 的流体推进仍需求解 CN 型非定常 Stokes 系统，采用 FGMRES。
  不能将其视为无需压力或黏性求解的完全显式范例。
- [Zhang, Guy, Philip (2014), A projection preconditioner for solving the implicit immersed boundary equations](https://www.math.ucdavis.edu/~guy/papers/zhang_guy_philip_NM_2014_projectionIB.pdf)，NMTMA 7:473–498。
  研究隐式 IB 的投影预条件器，效果与雷诺数和结构刚度有关；支持继续减少联合系统的迭代成本。
- [Brown, Cortez, Minion (2001), Accurate Projection Methods for the Incompressible Navier–Stokes Equations](https://www.math.tulane.edu/~cortez/Prints/accproj.html)，JCP 168:464–499。
  压力边界与投影处理影响精度。动量项显式后仍须满足不可压缩约束，不能简单删除压力求解。

建议继续保留 MAC，优先验证“显式黏性＋压力投影＋隐式刚性心肌/接口”的 IMEX 选项，目标先取 1e-4 s。
下一次算法修改还必须同步修改压力模式的已知响应：当前 `phi=(I-dt nu Lp)q` 配合 BE 逆算子，
去掉黏性逆算子后不能照搬；须重新推导与实际投影算子一致的梯度目标、伴随功和解析雅可比。
应在等容收缩、射血、快速卸载、充盈及闭瓣切换的短窗口，比较 1e-4/5e-5 s 与原 BE 的质量、功、压力、体积和主动张力。
通过这些验证之前，正式周期继续使用本次优化后的隐式默认路径。

## 复现命令

```bash
source scripts/activate_afsei.sh
python scripts/benchmark_afsei_balanced.py \
  --input results/phase25_mac128_balanced_rcr_native_60ms_v5/checkpoint.npz \
  --output results/my_perf_dt100 --profile
python scripts/benchmark_afsei_balanced.py \
  --input results/phase25_mac128_balanced_rcr_native_60ms_v5/checkpoint.npz \
  --step-ticks 1 --output results/my_perf_dt50
python scripts/check_afsei_explicit_fluid.py \
  --input results/phase25_mac128_balanced_rcr_native_60ms_v5/checkpoint.npz \
  --output results/my_explicit_fluid_probe
python -m pytest -q --show-capture=no \
  tests/test_afsei_*.py tests/test_ep_*.py tests/test_mac_spectral_dirichlet.py \
  tests/test_mac_solid_mass.py::test_graph_mass_recurrence_checks_and_owned_outputs
```

输出目录要求尚不存在，防止覆盖已有结果。`--step-ticks` 只限制这次基准的接受步长，
不更改原检查点配置中的最大允许步长。正式后台周期命令继续使用第二十六阶段文档中的入口。

# TorchCor 原生 PDE 与 AFSEI nodal FEM＋AMG 的比较

本实验只比较两个完整 PDE 步骤：TorchCor 原生 P1-Galerkin／Jacobi-CG，以及 AFSEI 的 PyTorch nodal P1／AmgX PCG-AMG。两者均使用 CN。没有把 nodal FEM 接入 TorchCor 作为本实验的比较对象。

在 RTX 4090 上，采用明确配置的两层 AMG 时，AFSEI 在 0.5、0.25 mm 网格上分别达到约 2.28、1.72 倍加速；在 0.125 mm 上耗时反而约为 TorchCor 的 1.60 倍。因此结果支持前两档的效率提升，不能据此声称所有网格尺度上都更快。实验仅包含这三档网格。

## 相同的外部条件与计时范围

- 固定长方体组织域：2 × 0.7 × 0.3 cm；同一四面体剖分，纤维沿 x、片方向沿 y，所有表面电学边界为零法向电流。膜电容为 1 μF/cm²、表面积体积比为 0.1 cm⁻¹，传导系数为 (0.1171, 0.0390333333, 0.0390333333) μS/cm，对应纵向扩散系数 1.171 cm²/s，横向为其三分之一。这是此前受控比较协议，不声称所有参数均为 Santiago 原文值。
- 同一 CN 扩散步长 0.05 ms、FP64、GPU 和物理单位下的真残差条件：`||b-Au|| <= max(1e-13, 1e-10 ||b||)`。
- 使用先前已完成 GPB 实验的 0、2、10、20、40 ms 电位快照作为预先给定的输入。离子电流在本次 PDE 回放中设为零，两方法施加相同的规定刺激场；刺激为原点附近 0.15 cm 立方区域内的 35.7142857 μA/cm²，持续 2 ms。
- 同一输入电位作为线性求解初值；关闭跨步骤线性外推，因为回放的五个快照不是连续的新时间轨迹。
- 每个方法先预热 10 次，然后测量三个计时块，每块 25 次，共 75 次 PDE 推进。两方法的计时顺序交替；每个块首尾同步 GPU，以墙钟时间统计右端构造与求解。
- 排除 GPB／Land ODE、网格组装、AMG 建层、结果输出及绘图；这些一次性准备耗时另行记录。图中误差线为三个计时块的最小至最大范围，不是置信区间。

这是一项固定输入的 PDE 效率实验，不是重新计算的生理时间序列，也没有证明两种空间离散的整段电位／激活时间完全相同。外部条件相同，但一致质量与集中质量正是被比较的方法差异。

## 原生实现与精度核对

TorchCor 固定在提交 `b02daee87d4283888981cbfc3a6b00f57789a820`，来源为 [TorchCor](https://github.com/sagebei/torchcor/tree/b02daee87d4283888981cbfc3a6b00f57789a820)。直接调用其原生 `Matrices3D`、Jacobi 预条件和 TorchScript CG。为了只计 PDE，校验源码 SHA256 后，从原生 `Monodomain.step` 提取未经改写的 `b=u.clone()` 和电学步骤，省略 ODE、NVML 与输出。适配代码在 `validation/torchcor_native_pde.py`，来源哈希写入实验报告。

TorchCor 使用 mV/ms，AFSEI 使用 V/s；导电矩阵、源项和残差按照同一物理方程换算。原生 CG 以预条件残差停机，因此在计时外为每个输入校准其容限，并在计时外检查未预条件的真残差；没有给原生每次推进附加 AFSEI 的输入／残差检查。AFSEI 直接调用实际 `Monodomain.step`，保留生产代码中的检查。

三档网格的刚度矩阵相对差异不超过 1.1×10⁻¹⁵，原生一致质量的行和与 nodal 集中质量的相对差异不超过 5.3×10⁻¹⁵。两层 AMG 实验中，两方法所有五个验证输入的真残差均通过共同条件，最大残差／容限比约 0.246。这里检查的是求解精度与组装等价关系，不把质量矩阵不同导致的解差异当成舍入误差。

## AMG 配置与测量结果

先在固定 0.25 mm 网格上比较声明好的 SIZE_2、SIZE_4、SIZE_8 聚合配置，选择有效配置中平均耗时最低的 SIZE_8，随后固定该配置完成三档对比。它使用最多 20 层、D2 插值、V 循环、前后各一次 Jacobi 平滑和粗层直接解。

随后增加显式可选的 `size8_two_level` 配置：最多两层，SIZE_8 聚合，粗层采用固定四次、松弛系数 0.5 的 Jacobi 近似求解。它保留聚合、限制／延拓和粗层校正，是两层 AMG 预条件；PCG 仍需满足同一真残差条件。三档均使用同一配置，没有按测试网格分别挑选最优结果。生产默认配置仍为 `size2`，本实验没有自动更改现有周期配置。

两层配置的实测结果：

| 间距 | 节点／单元 | TorchCor 原生 PDE，ms/步 | AFSEI nodal＋AMG，ms/步 | 原生耗时／AFSEI 耗时 | 平均 CG／PCG 次数 |
|---|---:|---:|---:|---:|---:|
| 0.5 mm | 4,305／20,160 | 2.9604 | 1.2997 | 2.2777 | 17.8／6.0 |
| 0.25 mm | 30,537／161,280 | 2.5207 | 1.4697 | 1.7151 | 14.2／7.4 |
| 0.125 mm | 229,425／1,290,240 | 2.4488 | 3.9184 | 0.6249 | 12.2／10.0 |

完整多层 SIZE_8 配置的对照结果也保留：

| 间距 | TorchCor 原生 PDE，ms/步 | AFSEI nodal＋完整 AMG，ms/步 | 原生耗时／AFSEI 耗时 |
|---|---:|---:|---:|
| 0.5 mm | 4.7630 | 2.6204 | 1.8177 |
| 0.25 mm | 2.6630 | 2.5247 | 1.0548 |
| 0.125 mm | 2.5057 | 5.5261 | 0.4534 |

这两批实验各自是交替顺序的配对比较；基线耗时在批次之间有变化，不能把跨批次差值全部解释为某一配置的优化收益。特别是 0.5 mm 基线批次变化较大，应以每批内部的配对结果解释。

两层配置减少了 AMG 的层级工作，但最细一档仍较慢；其 PCG 次数虽然少于原生 CG，每次 AMG 迭代的成本却更高。CN 此步长下也不能假定原生 CG 迭代数随网格细化必然增长。本次没有计入迭代内部逐核计时，不能把差额精确归因于某个 CUDA 内核。

## 实现与验证

AMG 配置作为 `MonodomainNumerics.amg_profile` 选项接入实际求解器，支持 `size2`、`size4`、`size8`、`size8_two_level`。矩阵与层级按步长缓存复用。

组装准备阶段也减少了暂存：仿射四面体体积／常梯度使用一个几何积分点，在全局 COO 合并前释放局部张量，并在 AmgX 初始化前归还 PyTorch 不再使用的分配缓存。nodal PDE 的质量与传导张量仍在四个顶点积分，未修改 nodal 积分规则；这些操作不计入本次暖态 PDE 耗时。

实际 CUDA 验证：四种配置与独立稠密 CN 解核对；nodal 顶点规则、变量方向／张量、缓存复用检查通过。几何准备与配置检查共 11 项通过；暂存释放后额外的独立顶点积分 CPU/CUDA 检查 2 项通过，没有重跑已完成的整段 ODE 实验。

## 配置、运行与结果位置

在服务器项目根目录激活现有环境后，使用标准 demo 入口：

```bash
python demo/run.py demo/ep_slab/configs/native_pair_two_level.json --output results/my_native_pair_two_level
```

完整多层配置及预声明的聚合参数筛选：

```bash
python demo/run.py demo/ep_slab/configs/native_pair_benchmark.json --output results/my_native_pair_full_amg
```

配置依赖已保存的 `results/ep_slab_cell_models_20261013` 三档快照和固定版本的 TorchCor 源码；新的检出需先按既有 slab 实验准备这些输入。已完成案例可以使用 `--resume-experiment` 复用，但配置必须完全相同。重新绘图只需 `--plot-only`，不会求解 PDE。

服务器已有结果：

- 两层 AMG：`/mnt/large2/gjh/AFSEI/results/ep_native_pair_two_level_20261014`。
- 完整多层 AMG：`/mnt/large2/gjh/AFSEI/results/ep_native_pair_20261014`。

每个目录包含 `report.json`、每档 `_pair.json`、`pilot.json`、`config_resolved.json` 和 `native_pair_pde.png/pdf`。原始报告与图像保存在结果目录，不提交仿真输出至 Git 仓库。

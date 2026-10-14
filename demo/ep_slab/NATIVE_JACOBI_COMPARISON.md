# 三档 nodal FEM＋轻量 Jacobi-PCG 与 TorchCor 原生方法

本次在 0.5、0.25、0.125 mm 三档网格上，测量实际 AFSEI `Monodomain.step` 的轻量 Jacobi-PCG 后端，与固定版本 TorchCor 的原生 P1-Galerkin／Jacobi-CG 电学步骤配对比较。只测 PDE，不推进 GPB、Land、流体或固体。

## 实际使用的两个实现

- AFSEI：四面体 nodal P1 顶点积分／集中质量，CN；`ep/jacobi.py` 独立实现 Jacobi-PCG，用 PyTorch TorchScript 编译迭代循环。它直接作用于缓存的 CSR 系统，没有通用 Dirichlet 提升与掩码，也没有 AmgX、跨库向量复制。节点输入检查保留；每次迭代检查收敛和正有限曲率，收敛候选及定期残差替换时重新计算真残差，必要时重启 PCG。
- TorchCor：版本 `b02daee87d4283888981cbfc3a6b00f57789a820` 的原生 P1 一致质量组装、原生 Jacobi 预条件／TorchScript CG 与未经改写的电学步骤。通过已有哈希校验和 AST 提取适配排除 ODE/NVML/输出，没有把 AFSEI 的逐步检查附加到基线上。

这次的 AFSEI 轻量循环没有调用或复制 TorchCor CG；先前四组消融中的 `nodal_torchcor_jacobi` 则使用 TorchCor 内核。因此之前 0.125 mm 的 2.378 ms 不能直接当作当前 AFSEI 自身后端的性能。
控制内核及 PDE 包装相同的三档同图见 [相同求解器的空间离散比较](SAME_SOLVER_PDE_COMPARISON.md)，它复用先前完整测量，三档均更快。

轻量后端以 `ep_numerics.backend="torch_pcg_jacobi_light"` 显式选择，旧通用 `torch_pcg_jacobi` 和 AMG 后端仍可使用。生产默认后端没有更换。轻量后端每次迭代检查停机；本次 JSON 明确设 `check_every=1`。Jacobi 对角、系统矩阵和求解器对象按 EP 步长缓存，输入电位不被就地覆盖。

## 共同条件与协议

RTX 4090（物理 GPU 1）、FP64、CN 步长 0.05 ms。网格、固定组织区域、方向、传导系数、刺激场、单位换算、保存快照均继承 [原生配对实验](NATIVE_PDE_COMPARISON.md)。区域为 2 × 0.7 × 0.3 cm，所有表面为零法向电流。每个方法使用同样的 0/2/10/20/40 ms 保存电位作为初值，零离子电流，规定刺激；关闭跨步骤线性外推。

共同真残差条件为 `||b-Au|| <= max(1e-13,1e-10 ||b||)`。TorchCor 的预条件残差容限在计时外换算，并在计时外验证真残差。AFSEI 在实际推进中检查真残差。相同外部条件不意味着一致质量与集中质量离散的电位解完全相同。

每方法先预热 10 次，再测 3 个计时块，每块 25 次，共 75 次。两方法交替计时；计时块首尾同步 GPU。计时包含实际右端构造、检查和求解，排除装配／初始化、ODE、I/O、绘图。图中误差线表示三个计时块的最小至最大值，不是置信区间。

## 同批结果

| 网格间距 | 节点／单元 | TorchCor 原生，ms/步 | AFSEI nodal＋轻量 Jacobi-PCG，ms/步 | 原生／AFSEI 耗时 | 平均原生 CG／AFSEI PCG 次数 |
|---|---:|---:|---:|---:|---:|
| 0.5 mm | 4,305／20,160 | 2.9600 | 1.2433 | 2.3807 | 17.8／4.6 |
| 0.25 mm | 30,537／161,280 | 2.5621 | 1.6570 | 1.5462 | 14.2／6.4 |
| 0.125 mm | 229,425／1,290,240 | 2.4608 | 2.6664 | 0.9229 | 12.2／10.4 |

前两档 AFSEI 更快；最细一档 AFSEI 比原生 TorchCor 耗时多约 8.4%。迭代次数减少没有确保总耗时降低，当前结果不支持三档全部加速，也不能直接外推真实 LV 或整个三场周期。没有对 CUDA 内核与检查开销分别计时，因此不把差额精确归因到某个操作。

在准备本轮轻量后端时，也测量了已有通用 Python Jacobi-PCG 的原生配对，0.125 mm 为 7.568 ms/步，对应原生基线 2.472 ms/步。该版本包含通用边界、掩码及检查逻辑，结果保存在独立目录 `results/ep_native_jacobi_pair_20261014`。轻量后端减少了这类通用工作；两个实现的测量是不同批次，不能把跨批次差值全部作为某个单项优化的收益。

## 验证与保存位置

CPU/CUDA 共 13 项检查通过：独立稠密 SPD 解、残差替换／重启、零初始残差、迭代上限失败、输入初值保持、实际 nodal CN 方程的独立稠密解、缓存复用，以及既有 nodal／AmgX 检查。本批所有代表性右端通过共同真残差验收；AFSEI 最大残差／容限比约 0.851，TorchCor 约 0.214。组装核对的刚度相对差异小于 1.1×10⁻¹⁵，集中质量与原生一致质量行和差异小于 5.3×10⁻¹⁵。

实际服务器结果目录：

`/mnt/large2/gjh/AFSEI/results/ep_native_jacobi_light_pair_20261014`

其中 `native_pair_pde.png/pdf` 为三档同图，`report.json` 为原始计时和来源报告，每档 `_pair.json` 与 `config_resolved.json` 可供复核。日志为 `results/logs/ep_native_jacobi_light_pair_20261014.log`，验证日志为 `results/logs/ep_jacobi_light_validation_20261014.log`。

```bash
python demo/run.py demo/ep_slab/configs/native_pair_jacobi_light.json \
  --output results/my_native_jacobi_light_pair
```

仅重新绘制本次已完成结果：

```bash
python demo/run.py demo/ep_slab/configs/native_pair_jacobi_light.json \
  --output results/ep_native_jacobi_light_pair_20261014 --plot-only
```

配置依赖服务器已有的三档保存快照及固定版本 TorchCor 源码。没有 0.0625 mm 档位。

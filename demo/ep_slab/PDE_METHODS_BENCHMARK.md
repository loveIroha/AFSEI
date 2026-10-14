# PDE 耗时：TorchCor 标准 P1-Galerkin 与 nodal FEM + AMG

本次只比较 PDE，使用此前已经保存的三维 slab 电位快照。没有重新求解细胞
ODE，也没有运行流固或完整 EP 传播。主性能指标为矩阵/预条件准备完成并预热后的
单步 PDE 墙钟耗时，包含输入检查、源项/右端项计算和线性求解，排除文件输出和绘图。
不是仅计 CUDA kernel 的时间，也不是 EP 或三场周期加速比。

## 条件与实际调用

- GPU：NVIDIA RTX 4090（服务器物理 GPU 1）；PyTorch 2.14.0+cu130，FP64。
- 几何：固定 `2 × 0.7 × 0.3 cm` 三维组织，沿用保存的相同节点和四面体。
- 网格 h：0.5 / 0.25 / 0.125 mm，节点数 4,305 / 30,537 / 229,425。
- D：`(1.171,1.171/3,1.171/3) cm²/s`；相同恒定纤维/片层方向，绝缘外边界。
- 时间推进统一为后向欧拉，dt=0.05 ms；矩阵为 `A=K+Sv*Cm*M/dt`。
- TorchCor 固定版本：`b02daee87d4283888981cbfc3a6b00f57789a820`。
  直接执行原始 `Matrices3D`、`Preconditioner`、TorchScript `ConjugateGradient`。
  未复制、重写或修改上游 PDE 核心源码；启动时核对 3 个源码文件的 SHA256。
- TorchCor 原 simulator 默认 `theta=0.5`（Crank–Nicolson）；本次为统一时间
  方程采用 theta=1 的后向欧拉。因此这是同条件 PDE 方法对照，不是 TorchCor
  默认全 EP 流程的原样性能。也不将 AFSEI 自己的 Python PCG 当作 TorchCor。
- 所有方法使用相同给定 Vm 作为初始猜测，关闭 TorchCor 的线性外推，避免
  在不连续快照回放中让不同方法得到不同初始猜测。
- 回放 0、2、10、20、40 ms 保存的 GPB Vm；膜面离子电流设置为零，只有 0 ms
  快照施加相同角部脉冲。它是代表性非均匀电位场的线性 PDE 单步测试，不是
  再现这些时刻的真实 GPB 电流或继续原生理轨迹。
- rtol=1e-10，atol=1e-13，最多 2000 次迭代。验收标准统一为
  `||b-Ax|| <= max(atol,rtol*||b||)`。
- TorchCor 原 CG 按预条件残差停止，与 AFSEI 的真实残差定义不同。适配器用
  `||r|| <= max(diag(A))*||diag(A)^-1 r||` 保守换算其相对阈值，并保留 0.25
  裕量；AmgX 接口也使用 0.25 裕量。不是给两者直接传同一个名义 rtol 就认为
  精度相同。标定在计时外完成，实际 RHS 和 CG 内部范数计算仍计时。
- 每方法预热 10 次；3 个计时块，每块 25 次求解，共 75 次。不同计时块轮换
  方法顺序，块前后同步 CUDA。图中误差线为三个块均值的最小至最大范围，
  不代表统计置信区间。没有在每个计时步额外同步或执行逐步误差比较。

TorchCor `assemble_matrices` 将 Python 列表转换回稀疏张量时依赖默认浮点
类型。本次显式设置 `torch.set_default_dtype(torch.float64)`，防止其在声明
FP64 的实验中先按 FP32 构造矩阵。两者使用相同精度，没有安装 TorchCor
整套可选 ECG/可视化依赖或修改 AFSEI 环境。

## 实测稳态 PDE 单步时间

单位 ms/步，取三个计时块的平均。

| h / mm | TorchCor 标准 P1＋Jacobi-PCG | 标准 P1＋AmgX | nodal P1＋TorchCor Jacobi-PCG | AFSEI nodal P1＋AmgX | 主比较比值：TorchCor / AFSEI |
|---|---:|---:|---:|---:|---:|
| 0.5 | 2.9388 | 2.8958 | 1.2541 | 2.1729 | 1.3525 |
| 0.25 | 3.4723 | 4.8392 | 2.6778 | 4.4236 | 0.7849 |
| 0.125 | 3.2184 | 6.8471 | 3.1835 | 6.4563 | 0.4985 |

在 0.5 mm 上，当前 nodal FEM + AmgX 比 TorchCor 标准 P1 约快 1.35 倍；
在 0.25 mm 上耗时约为其 1.27 倍，在 0.125 mm 上约为其 2.01 倍。
**这些结果不支持“nodal FEM + AMG 在此类小 dt slab PDE 上始终更快”。**

同一离散下加入 AMG 的效果也显示在中间两组对照中：0.25 mm 时
nodal + 原始 TorchCor Jacobi-PCG 为 2.68 ms，而 nodal + 当前 AmgX 为 4.42 ms。
当前计时范围内，节点积分/集中质量与 AMG 的收益不能混为一谈。
这四组里 nodal + TorchCor Jacobi-PCG 的平均值最低；最细网格与标准 P1 的
差距很小、计时范围重叠，不能称为显著优势。

代表性 RHS 的平均迭代数：

| h / mm | 标准 P1＋TorchCor PCG | 标准 P1＋AmgX | nodal＋TorchCor PCG | nodal＋AmgX |
|---|---:|---:|---:|---:|
| 0.5 | 16.2 | 10.4 | 6.0 | 7.2 |
| 0.25 | 12.6 | 10.4 | 9.0 | 8.0 |
| 0.125 | 15.0 | 9.6 | 15.2 | 9.2 |

AMG 减少部分迭代数，但该小时间步系统的 Jacobi 迭代已经不多。AMG 每次
迭代需要多层 V-cycle，当前 AmgX 接口还包含设备缓冲交换、跨库同步和真实
残差验收；这些实现成本会影响实际墙钟时间。这里未逐项剖分，不能将全部差额
都归因于 CPU/GPU 同步，也不能据此宣称 AMG 在其他 dt、网格、各向异性或硬件
条件下必然较慢。后续若评估新配置，应该继续以真实 PDE 耗时而非迭代数选择后端。

历史 `scripts/benchmark_afsei_ep.py` 的基线是 AFSEI 的 Python Jacobi-PCG，
本次基线是 TorchCor 原始 TorchScript CG。两种基线实现不同，不能直接沿用
历史测试的加速比。当前真实 LV 的 PDE/纤维场也不等于本次恒定方向 slab。
本轮没有改变 AFSEI 默认 PDE 求解器。

## 验收与准备成本

所有代表性输入的四组解通过真实残差验收，最大残差/允许残差小于 0.225。
TorchCor 一致质量的行和与 AFSEI 顶点积分质量相符，相对差异不超过 5.3e-15；
恒定方向下刚度矩阵相对差异不超过 1.2e-15。
同一质量离散的两个求解器电位差不超过 1.1e-7 mV，证明对照没有通过放松
精度制造加速。标准 P1 与 nodal 之间的电位差属于不同质量离散的差异，
这些单步测试不证明传播精度等价或空间收敛。

矩阵装配和预条件准备单独记录，不放入主要单步图及加速比：

| h / mm | TorchCor 原装配 / s | AFSEI nodal 装配 / s | TorchCor PCG 准备 / s | AFSEI nodal AMG 准备 / s |
|---|---:|---:|---:|---:|
| 0.5 | 0.4470 | 0.1010 | 0.00824 | 0.09085 |
| 0.25 | 1.3390 | 0.2124 | 0.00044 | 0.00813 |
| 0.125 | 11.4935 | 1.7172 | 0.00062 | 0.04130 |

这是每网格一次准备的实际记录，首组还包含库初始化差异，不能作为严格的
重复启动性能结论。TorchCor 上游装配使用 `.tolist()` 和 Python 列表构造全局
稀疏矩阵；当前 AFSEI 使用张量装配。该准备差异不等于周期内每步都支付此成本。

## 配置、运行与结果

```bash
cd /mnt/large2/gjh/AFSEI
source scripts/activate_afsei.sh

# 下载固定版本到 .deps；已存在且散列一致时直接复用，不安装依赖。
python scripts/fetch_torchcor_pde_reference.py

# 只需已有 slab 的 mesh.npz、电位快照和解析配置，不再运行 ODE。
python demo/run.py demo/ep_slab/configs/pde_methods_benchmark.json \
  --output results/my_pde_methods

# 仅重新绘图，不重复计时或求解。
python demo/ep_slab/benchmark_pde_methods.py \
  --output results/ep_pde_methods_20261014 --plot-only
```

标准 JSON 声明参考数据目录、固定源码版本、时间快照和采样规则；具体物理
参数、dt 与几何从每个保存算例的 `config_resolved.json` 读取。
主 PDE 仍直接调用 `Monodomain.step`；两个交叉对照由隔离的性能适配器组织。
源码和方法适配器位于 `validation/torchcor_pde.py`，没有更改上游库文件。

本轮结果：`/mnt/large2/gjh/AFSEI/results/ep_pde_methods_20261014`。
包含 `report.json`、`config_resolved.json`、主比较 `pde_main_comparison.png/.pdf`
以及四组对照 `pde_methods_comparison.png/.pdf`；日志为
`/mnt/large2/gjh/AFSEI/results/logs/ep_pde_methods_20261014.log`。
计时结果和下载的第三方源码不提交仓库。

上游依据：[装配](https://github.com/sagebei/torchcor/blob/b02daee87d4283888981cbfc3a6b00f57789a820/torchcor/core/assemble.py)、
[原始 CG](https://github.com/sagebei/torchcor/blob/b02daee87d4283888981cbfc3a6b00f57789a820/torchcor/core/solver.py)、
[默认 Monodomain](https://github.com/sagebei/torchcor/blob/b02daee87d4283888981cbfc3a6b00f57789a820/torchcor/simulator/monodomain.py)。

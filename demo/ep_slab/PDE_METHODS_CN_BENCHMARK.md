# 统一 CN 的 PDE 耗时比较

AFSEI 新运行的单域扩散默认使用 Crank–Nicolson。TorchCor 对照也使用
theta=0.5，与其原始 Monodomain 的 CN 电扩散公式一致。仍直接执行固定版本
`b02daee87d4283888981cbfc3a6b00f57789a820` 的上游 P1 一致质量装配和
TorchScript Jacobi-PCG，没有更换为 AFSEI 的 Python PCG。

## 方程与改动

令 `alpha=Sv*Cm/dt`，统一 PDE 方程为

```
(alpha*M + 0.5*K) V_new
    = alpha*M*V_old - 0.5*K*V_old + Sv*M*(stimulus-ionic).
```

标准 P1 使用一致质量 M，nodal FEM 使用顶点积分集中质量 M；两个
预条件后端在各自同一离散下执行完全相同的线性方程。原来的 BE 保留为
`time_scheme=backward_euler`，新主程序默认 `crank_nicolson`。
矩阵/AMG 按固定 dt 缓存，Jacobi 对角相应使用 `alpha*diag(M)+0.5*diag(K)`。
CN 需要额外计算旧电位的 `K*V_old`，这项实际开销计入 PDE 时间。

ODE 先行、冻结 Vm 的反应更新和子步平均离子电流没有更换。CN 的二阶性质
仅针对扩散步骤，不能宣称整个非线性 EP/ODE 分裂已变成二阶。
MAC BE–BE 流固求解、RCR、Land 投影及主动张力卸载保持原有实现。

旧检查点没有时间格式字段时恢复为 BE，包含以前的 nodal/AmgX 检查点。
以前的 slab 传播实验 JSON 已显式填写 BE，保持已有对照可复用；本次使用
独立 `pde_methods_cn_benchmark.json`，不会覆盖上一轮 BE 结果。

## 同条件 PDE 实测

与上一轮保持相同的 GPU、网格和保存电位输入：RTX 4090、FP64，
dt=0.05 ms，rtol=1e-10、atol=1e-13。使用 0/2/10/20/40 ms 快照，
膜面离子电流设零，只在 0 ms 快照施加公共角部刺激。
没有重算 ODE、原 GPB 传播或完整周期。这是线性 PDE 单步回放，不是 CN
产生的新生理轨迹。共 3 个网格、4 种方法，每方法预热 10 次、计时 75 次。

计时为同步 CUDA 后的块平均墙钟时间，包含输入检查、源项/右端项和求解器，
排除装配/AMG setup、文件输出与绘图。图中误差线为 3 个块均值的最小至最大范围。
仍按统一真实残差验收；TorchCor 的预条件残差阈值使用保守界换算，
相同初始猜测且关闭线性外推。方法接口细节见 [BE 对照说明](PDE_METHODS_BENCHMARK.md)。

单位：ms/步。

| h / mm | 节点 | TorchCor 标准 P1＋Jacobi-PCG | 标准 P1＋AmgX | nodal＋TorchCor Jacobi-PCG | AFSEI nodal＋AmgX | 主比值：TorchCor / AFSEI |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 | 4,305 | 3.1805 | 2.9520 | 1.1475 | 2.0637 | 1.5411 |
| 0.25 | 30,537 | 2.7677 | 3.3605 | 1.5364 | 2.6541 | 1.0428 |
| 0.125 | 229,425 | 2.5754 | 6.5874 | 2.3780 | 5.9038 | 0.4362 |

粗网格当前 AFSEI 约快 1.54 倍，中网格两者接近（本次均值约快 4.3%），
细网格 AFSEI 耗时约为 TorchCor 的 2.29 倍。因此改成 CN 后，仍不能称
当前 nodal FEM + AMG 在所有网格上优于 TorchCor 标准 P1。
本次四组中 nodal + TorchCor Jacobi-PCG 的均值最低，但未据此更换主程序后端。

| h / mm | 标准 P1＋TorchCor 迭代 | 标准 P1＋AmgX 迭代 | nodal＋TorchCor 迭代 | nodal＋AmgX 迭代 |
|---|---:|---:|---:|---:|
| 0.5 | 17.8 | 11.0 | 5.0 | 6.8 |
| 0.25 | 14.2 | 10.0 | 7.0 | 7.4 |
| 0.125 | 12.2 | 9.4 | 11.4 | 8.4 |

减少迭代不直接意味着减少时间。此处 AMG V-cycle、当前跨库设备缓冲交换
与同步/验收仍有成本，Jacobi 在该小 dt 系统中只需要较少迭代；没有逐项
剖分时，不能把全部时间差都归因于某一项。结论只限本次 slab 和配置，
不能直接替代真实 LV 变方向场的基准或外推整个 EP/三场周期。
BE 表格保留为上一轮独立测量；未重新运行 BE，跨轮耗时变化也不能全部
归因于时间格式，因为运行时缓存和设备状态未固定。

## 验证

CPU/CUDA 合计 28 项单域、CN 与 AmgX 检查完成通过，包括独立稠密方程、
BE 回归、闭域质量守恒、均匀电流单位、半离散扩散特征模的约四倍误差收敛比、
实际 CUDA AmgX 同方程解、层次缓存及多实例关闭安全性。
恒流精度测试使用比产品默认更严格的线性容差，保证其断言精度与求解精度一致。
另有 4 项配置/观测检查通过，验证 BE 历史 JSON 与 CN 默认值、旧配置
签名和新时间格式不会混用。已通过的检查未重复运行整套。

全部代表性 CN PDE 右端项通过真实残差验收，最大残差/容差小于 0.216。
同一一致质量方程的两个求解器电位差不超过 1.95e-7 mV，同一 nodal 方程
差异不超过 5.0e-8 mV。刚度与集中质量同上一轮吻合。标准 P1 与 nodal
的单步差异不代表线性求解误差，也不证明传播精度等价。
本轮没有重新验证完整 EP 生理传播或两周期三场运行。

## 运行与保存位置

```bash
cd /mnt/large2/gjh/AFSEI
source scripts/activate_afsei.sh
python demo/run.py demo/ep_slab/configs/pde_methods_cn_benchmark.json \
  --output results/my_pde_cn_comparison

# 仅重新绘图；不会运行计时、ODE 或 PDE。
python demo/ep_slab/benchmark_pde_methods.py \
  --output results/ep_pde_methods_cn_20261014 --plot-only
```

实际结果目录：`/mnt/large2/gjh/AFSEI/results/ep_pde_methods_cn_20261014`。
主比较 `pde_main_comparison.png/.pdf`，四组对照 `pde_methods_comparison.png/.pdf`，
原始统计 `report.json`，计时协议 `config_resolved.json`。
日志：`results/logs/ep_pde_methods_cn_20261014.log`。
验证日志：`ep_cn_validation_20261014.log`（26 项先通过）、
`ep_cn_conservation_precision_20261014.log`（2 项精度检查修正后通过）、
`ep_cn_config_compatibility_20261014.log`（4 项配置/观测检查）。

源码依据：[TorchCor 原始 CN 公式](https://github.com/sagebei/torchcor/blob/b02daee87d4283888981cbfc3a6b00f57789a820/torchcor/simulator/monodomain.py)。

# AFSEI 优化 nodal 路径与 openCARP 的三维对照

本目录是独立 benchmark 适配，不修改左心室或生产 GPB/Land 的默认模型。

## 对齐的物理条件

- 计算域为 20×7×3 mm 长方体，纤维 X、sheet Y、normal Z；全表面绝缘零通量。
- 复用已运行 openCARP 的原始 0.0625 mm 网格：1,777,377 节点、8,601,600 四面体，每个笛卡尔小格五个四面体。没有重新划分或重排网格。
- 角点 [0,1.5]³ mm 内 15,625 节点施加 35.71 µA/cm² 刺激，持续 0–2 ms。
- Cm=1 µF/cm²，β=1400 cm⁻¹。原始细胞内/外电导率沿纤维为 0.17/0.62 S/m，横向为 0.019/0.24 S/m；等效单域电导率取调和组合，换算为 (1334.1772151898733,176.06177606177604,176.06177606177604) µS/cm。
- 原生 openCARP v20 Grandi 的全部 GPB 物理参数和完整默认初态；无刺激预平衡，初始电位统一 −80.9763 mV。
- EP 时间步 0.02 ms，模拟 0–50 ms；每 1 ms 保存全部节点电位。这是激活与早期平台期 benchmark，不涵盖完整复极。

## 保留的 AFSEI 数值方法

PDE 直接调用 `afsi_torch.ep.monodomain.Monodomain.step`：P1 顶点积分、对角质量、CN 扩散；`BufferedRHS` 融合右端，轻量 Jacobi-PCG 使用 `GraphJacobiPCG` 和真实残差验收；不使用 AMG/AmgX，不调用 TorchCor 求解核。

细胞 ODE 使用 AFSEI 原有的两阶段 SDIRK 算法：γ=0.7886751345948129，每阶段最多三次不动点迭代；ODE 步长 0.002 ms，节点电压在 EP 步内冻结，取子步末电流的平均。PDE 采用该平均电流作为 CN 方程右端源项，未改成 openCARP 原生的反应电压更新后纯扩散分裂。

为了完整对齐 GPB 物理模型，从已安装的官方 Grandi.cc/.h 机械提取 RHS，保留官方物理常数、默认参数、版权及源文件 SHA256。原生辅助缓存字段不作为物理状态，54 个物理状态用 FP64；Cai 外部 mM 与原生 µM 之间使用精确单位转换。`reference_grandi_rl_check.h` 中的原生 RL 更新仅用于独立导数检查，模拟轨迹未调用该更新。

`validation.json` 包含 CPU/GPU RHS、电流、SDIRK 一致性、原生 RL 微小步导数检查，以及独立 RK4 的步长收敛检查。短程模拟通过后才启动完整模拟。

## 配置与运行

```bash
cd /mnt/large2/gjh/AFSEI/demo/ep_afsei_nodal_opencarp_matched
CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/mnt/large2/gjh/conda-envs/afsei \
TORCH_CUDA_ARCH_LIST=8.9 \
/mnt/large2/gjh/conda-envs/afsei/bin/python run_afsei_matched.py --config afsei_matched_full.json
```

驱动拒绝覆盖已有结果。更换输出目录后才可另行运行；本次已有运行无需重复启动。

结果目录：`/mnt/large2/gjh/AFSEI/results/ep_afsei_nodal_opencarp_matched_20261030`。

完成后使用 `analyze_afsei_matched.py` 比较已有 openCARP 数据并画激活/电位/性能图；`export_afsei_matched_vti.py` 生成完整 51 帧三维 PVD。两者只做后处理，不启动求解。

## 对比的适用范围

本次直接检验“实际 AFSEI 优化实现，在 openCARP 同物理条件下的结果和耗时”。openCARP 仍使用自己的标准一致质量 P1、Rush–Larsen、分裂格式、线性预条件和门变量精度；其 16 CPU MPI 进程与 AFSEI 的单张 RTX4090 也不同。不能将跨软件计时比值全部归因于 nodal 离散或 GPU 优化。不能把以前 0.125 mm、不同 GPB 参数的 TorchCor 结果当作本次同条件标准 CG 对照。

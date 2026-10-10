# 第十五阶段：原 MAC 压力求解加速与边界验证

本阶段优化现有等容联立求解，并降低既有基底密封诊断的压力求解成本。
生产流体网格仍为 128³ 规则 MAC，保留原 BE–BE 隐式方程、旧几何转移模板、
源 EP/GPB/Land、PK1 主动力及 80 ms 卸载实现。
没有增加外层完整 FSI 迭代，也没有接入四面体血液网格。
实际流量驱动 RCR、正式局部密封和四阶段切换仍需后续完成。

## 同一压力矩阵的直接求解

原 MAC 压力算子为 cell-centred `-DG`，物理外箱面压力为零。
沿每个方向作奇延拓与 DST-II，其特征值为

\[
\lambda_{k_x,k_y,k_z}=\sum_{a=x,y,z}\frac{4}{h_a^2}
\sin^2\!\left(\frac{\pi k_a}{2N_a}\right),\qquad k_a=1,\ldots,N_a.
\]

先变换 RHS、除以特征值，再作逆变换，得到原离散方程的解。
使用 PyTorch float64 CUDA FFT；这不是周期边界近似，亦不改变压力方程或压力梯度。
每次仍检查原 `negative_laplacian` 的真实残差，使用检查点原有 rtol/atol。
严格阈值下可进行两次舍入误差修正，仍不满足则明确失败。
仅适用于当前均匀 Cartesian 外箱、零面压力条件；未来其他边界需重新建立相应算子。

等容模式默认使用 `spectral`，可选原 `multigrid`。
普通规定压力 MAC 模式仍使用原求解器与配置。
原 `paper_coupling.py`、`backward_euler.py`、`multigrid.py`、
`assembled_transfer.py`、`adaptive_transfer.py` 与 `paper_lv.py` 均与固定 AFSI 基线字节一致。

## 消除重复响应并保持续算

等容问题的初始机械残差已计算，原 Newton 入口又求同一点的残差。
当前对完全相同的未知量、同一旧状态复用该结果；修改未知量会使缓存失效。
缓存只在一个冻结 EP 主动载荷的机械试探内有效，不跨物理步复用力、流速或几何。

压力单位缩放仅用于联合未知量的条件数估计，在同一步长求解器内可复用，
免去每步的一次单位压力流体响应。
每次 Newton 的实际压力列仍根据当前心肌几何和当步 IB 模板重新计算。
不同步长保留各自缩放。缩放与后端一起写入检查点，避免重启时改变该数值设置。
失败试探不会推进物理状态、腔压、体积目标或 RCR 电容压力。

计数明确区分 `pressure_cycles` 与 `pressure_direct_solves`。
直接求解时压力多重网格循环为 0，压力方程仍在每次流体响应中求解。

## 真实网格对照

GPU：RTX 4090；LV：26889 节点、135430 单元；MAC：128³，13 cm 外箱。
起点均为 `results/phase14_mac128_isovolumic_10ms_v1/checkpoint.npz`，即 510 ms。
EP 步长 0.05 ms、ODE 子步 0.002 ms、机械步长 0.1 ms。
每组先执行一个不发布的冷启动试探，再从相同输入状态推进 10 步到 511 ms，
同步计时，排除冷启动、检查点读取和输出写入。
物理配置完全一致，10 步最终联合残差阈值均为 1e-9 cm，无回退。

|执行方式|平均 s/机械步|
|---|---:|
|第十四阶段原多重网格与响应方式|1.776429|
|仅换为 DST-II|1.129707|
|DST-II + 重复残差/缩放复用|1.090390|

最终版本速度为参考的 1.629 倍，耗时下降约 38.6%。
优化组末步 1.08314 s：流固 0.96994 s、GPB/Land 0.06986 s、单域 PDE 0.02265 s、
加载 0.00535 s、应变反馈 0.01468 s。
末步 3 次 Newton、14 次 GMRES；流体响应从参考的 26 次降为 24 次，
24 次直接压力解替换多重网格迭代。剩余成本仍主要在隐式流固与转移运算。

10 步后完整状态对照：

- 最大坐标差 1.013e-13 cm。
- 最大三分量流速差 1.023e-9 cm/s。
- MAC 压力场相对 L2 差 2.607e-9，最大差 3.561e-6 dyn/cm²。
- 力相对 L2 差 9.229e-10，最大差 3.228e-7 dyn。
- 电位最大差 4.006e-15 V。
- 腔压差约 2.67e-9 mmHg，优化组最终腔压 10.67034705 mmHg。
- 固定体积偏差约 1.393e-12 ml，RCR 状态与目标体积一致。

输入检查点 SHA256 未改变：`5a2de4f6e132fce250f3280b55dae848767d9761289011797e9f59a0e08c6186`。
性能审计保存在服务器 `results/phase15_performance_audit.json`。
以上是早期等容窗口的实测，不能据此声称完成两周期或完整循环边界下的速度。

实际驱动继续从 510 推进到 515 ms（新增 50 步），无回退。
去除首步后均值 1.099812 s/步，首步 6.866746 s；末步 1.102103 s。
末最大主动张力已增至 33627.63 dyn/cm²，腔压 13.69856473 mmHg，
体积偏差 1.478e-12 ml，最小 detF=0.89157。
末步流固 0.989042 s、GPB/Land 0.069988 s、单域 PDE 0.022529 s。
结果保存在 `results/phase15_mac128_isovolumic_15ms_v1`。

## 下一步边界工作的成本验证

基底投影使用同一套 213 个局部弱约束、同一 520 ms 规定压力检查点和同一 PCG 阈值，
仅替换内部压力求解器，预热后各计时 5 次：

|压力后端|平均 s/基底投影|
|---|---:|
|原多重网格|0.617620|
|DST-II|0.073838|

降低约 88%，速度为 8.36 倍。校正后的净泄漏约 2.213e-9 ml/s，
但积分点绝对泄漏仍为 29.640 ml/s，最大点法向滑移约 20.548 cm/s。
因此本投影仍是独立诊断，不能作为已完成局部不可穿透的证据。
材料反力、压力功和能量检查保留；此成本不包含在上述主线机械步时间中。
后续边界工作需解决空间约束质量，并把反力纳入原隐式残差，才能接入实际端口流量/RCR。

## 使用与复现

```bash
python scripts/run_afsei_isovolumic.py \
  --resume results/phase14_mac128_isovolumic_10ms_v1/checkpoint.npz \
  --duration-ms 5 --profile --output results/my_fast_isovolumic_15ms

# 对已保存的参考后端检查点，显式切换数值后端；不覆盖输入文件
python scripts/run_afsei_isovolumic.py --resume INPUT.npz \
  --pressure-solver spectral --duration-ms 1 --output results/my_spectral

# 两组从同一输入开始。输出必须为全新目录。
python scripts/benchmark_afsei_isovolumic.py --checkpoint INPUT.npz \
  --backend multigrid --reference-responses --steps 10 --output results/my_reference
python scripts/benchmark_afsei_isovolumic.py --checkpoint INPUT.npz \
  --backend spectral --steps 10 --output results/my_optimized

python scripts/check_afsei_mac_seal.py --checkpoint INPUT.npz \
  --pressure-solver spectral --repeats 5 --output results/my_fast_cap_diagnostic
```

旧等容检查点未记录数值后端时，使用本阶段的加速默认值。
新检查点记载具体后端、响应复用开关和压力缩放，普通 `run_afsei.py --resume` 会恢复这些设置。
`--reference-responses` 禁用复用，便于执行对照；改变它不修改物理模型。

GPU/CPU 检查覆盖独立稠密离散矩阵、最低/最高正弦模态、非立方与奇数网格、
原 BE 投影对照、联合 Jacobian、修改未知量时缓存失效、控制状态与完整物理张量重启，
以及两种压力后端的基底压力功、能量和散度。
完整力/压力字段对照使用能分辨测试 LV 的 32³/5 cm 网格和更严格的验证阈值，
生产网格与收敛配置没有相应改动。

本次 53 项 GPU/CPU 检查全部通过（54.46 s）。

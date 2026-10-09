# 第二阶段：EP–GPB–Land 与隐式流固统一推进

第三阶段补充审计：[源力学对齐](SOURCE_MECHANICS_PHASE3.md)。原力学直接绑定单元表达式，
DG0 与其相符；以下第二阶段的“力学 CG1 待接入”推断已纠正。
本页的数值验证记录仍保留当时继承的 AFSI 材料与积分规则。

## 已实现的范围

单域 PDE 继续使用 PyTorch 稀疏矩阵与 PCG；GPB/Land 继续使用 CUDA 内核。
新增 AFSEI 驱动将其接入原 `BEIBStepper`，继承原 MAC 流体、P1 H–O 固体、
浸入边界传递、Newton 固体切线和流固耦合求解。
这是分区 EP→隐式 FSI→变形反馈，不是把三个场装入同一个 Newton 系统。
原 AFSI 时间推进文件未修改。

默认一个流固步 `0.1 ms` 对应两个 EP 步 `0.05 ms`；每个 EP 步包含 25 个
`0.002 ms` 的 GPB/Land 子步。接受状态由整数 EP tick 定义唯一时间，
避免累加浮点时间导致刺激、松弛或重启偏移。

一次试算依次执行：

1. 用上个接受状态的节点伸长和 Z 历史推进 EP/GPB/Land。
2. 将新 Land 状态和旧变形历史插值到固体积分点，在整个隐式流固求解中固定这些历史。
3. 求解流固终点变形、流速、压力与主动/被动力，并验证真正的非线性残差。
4. 将接受变形的单元纤维伸长以一致质量矩阵投影到节点，更新节点 Z 历史。
5. 最后发布所有物理状态。

求解失败时，Vm、GPB、Ca、Land、Z、变形、流速、压力和周期状态均保持旧值，
恢复主动力缓冲区并清除传递求解器暖启动，再减小流固步长重算。
不同步长各自拥有流体求解器，避免仅修改 dt 却继续使用旧 Helmholtz 系数或 CUDA 图。
最小流固步为一个 EP tick；EP 步长和 ODE 子步不随重试变化。

## 编译载荷与重启

`BufferedLandLoad` 的张量对象保持固定，只更新内容。
编译后的固体力和单元切线可以读取新 Land 状态、dt、模式和松弛因子，
避免每次试算替换载荷对象造成旧张量被编译图捕获。

自包含 NPZ 检查点保存网格、全部三场状态、ODE 状态、周期状态与时间配置，
并对元数据、数组名称、dtype、shape 和内容进行 SHA-256 校验。
保存和替换是原子的，失败运行保存最后一个已接受状态。
恢复时还重新检查变形、浓度、网格形状、整数时钟及端点力。
AFSI-only 检查点不能用于恢复 EP。

**端点力历史必须单独保存。** 求当前端点力时使用的是本步前的变形/Z 和本步新 Land，
随后才发布变形反馈或关闭事件后的 Land 重置。
如果重启仅保存最新节点 Z/Land，就无法复现检查点中的端点力。

## 原项目串行 CG1 插值核对

原项目把由单元属性计算的纤维方向、片层方向和刺激标志插值到 CG1。
共享节点取最后访问单元的表达式值；不能替换为单元平均值，也不能直接把全部
内膜表面节点当作刺激区域。这与 DOLFIN 的逐单元覆盖实现一致：
[FunctionSpace.cpp](https://github.com/live-clones/dolfin/blob/2019.1.0/dolfin/function/FunctionSpace.cpp#L165)。

新增 PyTorch 映射采用最后关联单元。`validation/ep_mapping` 使用固定源提交的
`Active_land.h` 和服务器已有 DOLFIN 2019.1 实际执行原类插值，再按坐标匹配顶点。
真实网格核对结果：

| 项目 | 结果 |
|---|---:|
| 节点 / 四面体 | 26,889 / 135,430 |
| 纤维 / 片层最大误差 | 0 / 0 |
| 单元归属 / 刺激标志不匹配 | 0 / 0 |
| 源串行插值刺激节点 | 2,502 |
| 内膜表面节点 | 3,267 |

默认 `direction_mapping='source-serial-cg1'` 作用于 EP PDE 与刺激。
`afsi-dg0` 可显式恢复第一阶段的方向与内膜节点设置。
未验证原项目多 MPI 分区下的覆盖顺序，不宣称与并行原运行逐点一致。
第三阶段确认：`main.cpp` 将单元表达式 `f0/s0` 直接绑定到力学残差、切线和伸长投影。
全局 `fiber_func/sheet_func` 只用于 EP。因此力学 DG0 已与源绑定相符；
力与切线的直接组装验证见第三阶段，不能将 EP 的 CG1 覆盖规则推广到力学。

FEniCS oracle 需要独立环境。服务器可按以下方式生成 CSV：

```bash
cd /mnt/large2/gjh/AFSEI
mkdir -p .cache/ep-reference-5bdde73
git -C /mnt/large2/gjh/test_ele_soild_heart-test show \
  5bdde73c6be10cefadb005a0fcec5fdb6521f336:Active_land.h \
  > .cache/ep-reference-5bdde73/Active_land.h
(
  source /home/fenics/spack/share/spack/setup-env.sh
  spack load /nvb7ww7
  cmake -S validation/ep_mapping -B .cache/ep-mapping-build \
    -DDOLFIN_DIR=/home/fenics/spack/opt/spack/linux-ubuntu24.04-icelake/gcc-11.4.0/fenics-2019.1.0.post0-nvb7ww7rb4ee65zs7d7bf5b54n6zzvbi/share/dolfin/cmake \
    -DAFSEI_REFERENCE_INCLUDE="$PWD/.cache/ep-reference-5bdde73" \
    -DBoost_NO_SYSTEM_PATHS=ON -DBoost_NO_BOOST_CMAKE=ON
  cmake --build .cache/ep-mapping-build -j 2
  .cache/ep-mapping-build/afsei_ep_mapping \
    /mnt/large2/gjh/realistic_left_ventricle .cache/ep-mapping-dolfin.csv
)
source scripts/activate_afsei.sh
python scripts/check_afsei_mapping.py --oracle-csv .cache/ep-mapping-dolfin.csv
```

括号使 Spack Python 3.10 的路径不会污染 AFSEI Python 3.12 环境。
缓存中的源头文件由固定提交提取，不读取或覆盖原项目的未提交修改。

## 关闭、松弛和恢复接口

已提供明确的 `attempt(..., closure=True, filling=True)` 事件入口。
关闭事件指本步接受终点；`filling=True` 指整个本步处于充盈。
它们必须由后续物理瓣膜/循环模型给出，当前运行脚本不自动制造这些事件。

接受关闭时，在发布 Z 反馈之前冻结空间 Land 张力作为积分点锚点，
随后将 Land 重置到首次刺激前的快照，Z 清零；GPB、Vm 和 Ca 连续保留。
锚点张力在 80 ms 内线性松弛，主动力仍使用当前变形和原长度因子。
关闭 200 ms 后，只有已确认充盈且 Vm/Ca 恢复和漂移检查连续通过 20 ms，
才能冻结 EP；下一次刺激重新开放推进，保留已有 GPB 状态。
机械步自动裁剪到刺激起止、松弛终点和恢复检查边界。

## 验证

2026-10-09，RTX 4090，CUDA 13.0，Python 3.12.14。
相关回归 **72 项通过**：第一阶段与原 Paper LV 检查 65 项，第二阶段 7 项。
新增检查包括编译缓冲更新与切线、真实 CUDA 统一推进、完整试算后注入失败、
步长减半重算、端点力历史校验、检查点篡改拒绝、关闭锚点连续性、
Land 重置保持 GPB、运行脚本失败保存、连续运行与重启一致性。
测试中的小 Land 种子和人工关闭事件只验证算法，不代表生理测量。

真实 LV、32³ MAC 网格，从首次刺激 500 ms 推进至 502 ms：

| 项目 | 结果 |
|---|---:|
| 流固接受步 / EP 步 / ODE 子步 | 20 / 40 / 1,000 |
| 拒绝次数 | 0 |
| 电位范围 | −87.9317 至 61.6168 mV |
| Ca 范围 | 0.1070116 至 0.1075180 μM |
| 最大 Land 张力 | 0.212208 dyn/cm² |
| 最大坐标分量位移 | 3.10×10⁻⁸ cm |
| 最大流速分量 | 5.62×10⁻⁵ cm/s |
| 终点非线性残差 / 容差 | 8.14×10⁻¹⁵ / 10⁻⁹ |
| 终点散度 L2 / 功率误差 | 6.06×10⁻¹⁶ / 1.29×10⁻¹³ |
| 含首次编译与初始化时间 | 95.4 s |

2 ms 处于刺激初期，张力和运动很小，此结果证明接口推进及守恒诊断可运行，
不能据此判断整周期收缩强度、射血分数或 GPU 性能。
验证结果 JSON 位于 `docs/EP_FSI_VALIDATION.json`。
真实网格检查点也已恢复并继续至 502.2 ms，总计 22 个流固步、44 个 EP 步、
1,100 个 ODE 子步，无重试。小网格测试另外对照连续运行与重启结果。

## 剩余模型工作

当前初态没有 8 mmHg 预充盈，MAC 外箱为零压力开放边界。
基底封口、真实 AV/MV 通道、物理端口压力和 RCR 尚未接入，
本阶段尚未对齐的材料和基底参数已在第三阶段处理；力学 DG0 的源绑定也已核对。
因此报告保持 `three_field_integration_implemented=false`、
`windkessel_ports_connected=false`，同时单列已完成的
`ep_fsi_time_integration_implemented=true`。

后续需要建立可验证的腔室封闭与物理端口。
RCR 必须由实际端口通量驱动：

\[
Q_{ao}=\int_{\Gamma_{ao}}(u-w_{port})\cdot n\,dA,\qquad
C\frac{p_c^{n+1}-p_c^n}{\Delta t}
=Q_{ao}^{n+1}-\frac{p_c^{n+1}-p_v}{R_p},\qquad
p_{ao}^{n+1}=p_c^{n+1}+R_cQ_{ao}^{n+1}.
\]

固定端口 `w_port=0`；运动端口必须考虑相对流量与控制体一致性。
端口压力通过流体边界进入，固体内膜由流体传递受力，不能再把相同腔内压力
作为额外全壁载荷叠加，也不能在端口和 Windkessel 两处重复计入 `R_c Q`。
瓣膜状态来自压差/通量及其阻力模型；检查
`dV_LV/dt≈Q_MV−Q_AV`、端口功率、泄漏和 RCR 更新后，才启动完整刺激周期。

# EP–GPB–Land 核心与主动载荷接口

本阶段完成独立组件和固体载荷适配器。完整真实左心室三场心动周期尚未实现。
源版本为电–力项目提交 `5bdde73c6be10cefadb005a0fcec5fdb6521f336`，
原始文件与 SHA256 保存在 `validation/ep_reference/`，不依赖服务器源项目的未提交文件。

## 已实现

| 组件 | 实现及验证范围 |
| --- | --- |
| 单域 PDE | PyTorch 的 P1 一致质量矩阵、各向异性刚度和 GPU PCG；参考构形固定，绝缘边界 |
| GPB | CUDA 中保留原 57 状态方程；膜电位由 PDE 提供；保留源代码的正常条件分支 `heartfailureConditions=2` |
| 时间离散 | EP 0.05 ms；GPB/Land 0.002 ms，每 EP 步 25 子步；GPB 保留原两级 SDIRK 的实际不动点迭代、最多 3 次和 1e-6 阈值 |
| 离子电流 | 每个 ODE 子步使用更新后状态的电流，再求子步平均；不改成 SDIRK stage 平均 |
| Land ODE | 原 XS/XW/CaTrpn/TmB 欧拉更新、夹紧和长度/应变反馈；可暂停 Land，同时继续 GPB |
| 主动应力 | XS/XW/Z 历史作为 P1 场在积分点插值，Ta 对当前几何可微；CGS 第一 Piola 应力 |
| 固体接口 | `EPDrivenPaperSolid` 通过现有 BE–IB API 替换自定义张力；保留原被动材料；不额外向整个内膜施加流体压力 |
| 切线 | 主动 Ta 的几何依赖进入节点 JVP 和稀疏固体切线，含 Land 畸变增量项 |
| Windkessel | 真实端口流量驱动的 RCR backward Euler 关系、解析端口阻抗和移动三角端口相对流量积分；尚未接到 MAC 物理边界 |

所有 EP/ODE 推进返回新状态，输入保持不变。非线性重试从同一已接受状态重新开始。
节点 ODE 内核使用 PyTorch 当前 CUDA stream；低层 ODE 支持提前编译后的 CUDA Graph。
PDE PCG 与高层异常检查包含主机标量判断，未声明整个 EP/FSI 步可进行 CUDA Graph 捕获。

## 方程与单位

单域离散为

\[
(S_v C_m M/\Delta t+K)V^{n+1}
=M[S_v C_m V^n/\Delta t+S_v(I_{stim}-I_{ion})].
\]

网格长度 cm，电位 V，时间 s，电流 μA/cm²，电容 μF/cm²。
默认扩散数字为 `(0.117,0.039,0.039)`，保留 `Active_land.ufl` 的数值行为。
在上述单位组合中应解释为 μS/cm；源文件的 mS/cm 注释与 V/s/μA 不一致。
没有执行会改变传播速度的额外 ×1000。将来若调整物理单位，必须作为显式模型变更重新标定。

GPB 使用 mV/ms，入口电位乘 1000，ODE 入口时间乘 1000。
GPB `state[:,37]` 的 Cai 为 mM，Land 输入乘 1000 得到 μM。
`Tref=1.2e6 dyn/cm²=120 kPa`，主动系数 Ta 是 nominal coefficient，不是归一化的 Cauchy 张力。

\[
z^{trial}=z^{old}e^{-c\Delta t}
+10(\lambda^{trial}-\lambda^{old})\frac{1-e^{-c\Delta t}}{c\Delta t},
\quad c_s=40.14,\ c_w=405.86\ \mathrm{s^{-1}}.
\]

力学积分点的旧伸长从上一步接受的几何梯度求得。ODE 的节点伸长可通过一致 L2 投影反馈；
两者不能混用。主动几何乘子为 `1+4.9*(max(lambda,1)-1)`，与电–力 UFL 对齐。
闭瓣后可输入接受状态的冻结 Ta 积分点场及衰减因子；不会把空间场替换成全局平均。
80 ms 衰减的事件触发、完整周期恢复门控仍由后续驱动程序接入。

## 两项明确记录的源模型问题

1. GPB 声明 58 完整状态，初值只提供 57 项，原程序读取 `full[57]` 会越界。
   AFSEI 显式将缺失的未使用状态设为 0；其导数在源方程中恒为 0。其余初值和参数保持原值。
2. 原 GPB 公式在精确 `Vm=0,-5,-47.13 mV` 有可去奇点，原程序可能用归零保护掩盖非有限值。
   本阶段保留原算术；高层接口对保护触发抛出异常，供耦合步回退，避免静默接受异常状态。
   后续如补充解析极限，应另做明确的方程数值实现修正与对照验证。

## Windkessel 从固体到 FSI

\[
p_c^{trial}=\frac{C p_c^n+\Delta t(Q_\Gamma^{trial}+P_v/R_p)}{C+\Delta t/R_p},
\quad p_\Gamma^{trial}=p_c^{trial}+R_c Q_\Gamma^{trial},
\quad \frac{\partial p_\Gamma}{\partial Q_\Gamma}=R_c+\frac{\Delta t}{C+\Delta t/R_p}.
\]

内部 mmHg/mL/s，输出压力转换为 dyn/cm²。Rc 只计入一次。
流量必须来自实际出口 `∫(u-wΓ)·n dA`，外法向指向流体区域外侧。
闭瓣时只把 0D 流量设零并不能封闭 3D 流体；现有虚拟基底封口只是体积测量面。
若 3D 已解析瓣膜，不重复添加 0D 瓣阻；若端口位于瓣膜上游，需要单独加入瓣压降并封闭关闭端口。
动脉端口压力、平均腔压和壁面流体应力属于不同位置，不能直接等同。

## 运行验证

```bash
cd /mnt/large2/gjh/AFSEI
source scripts/activate_afsei.sh
python -m pytest -q tests/test_ep_*.py
python scripts/check_afsei_ep.py --steps 40 --output .cache/ep-check/real-lv-2ms.json
```

第一次构建新的 CUDA ODE 扩展会使用本项目独立缓存。
组件检查在实际 LV 参考网格施加 2 ms 内膜节点刺激，计算 PDE→GPB Ca→Land→力和切线。
它不推进固体运动或流体，不是三场心动周期结果，也不是与源项目完整周期一致性的证明。
该检查显式使用 AFSI DG0 方向；生产 PDE 已支持源项目的 P1 节点方向，但尚需确认与源 DOLFIN 插值的节点映射。

## 下一阶段

1. 对齐源 EP 的 CG1 方向、刺激节点选择与两项目网格映射；核对实际生成力学代码的材料参数及基底约束。
2. 建立 AFSEI 独立时间驱动：每 0.1 ms 力学步包含两个 EP 步；接受/回退覆盖 EP、Land、几何、闭瓣锚点和循环状态。
3. 接入刺激→Land→闭瓣 80 ms 冻结场衰减、源项目 Land 重置与 EP 恢复门控，并补充完整 checkpoint。
4. 建立 AV/MV 物理端口及基底封闭，处理实际压力/流量边界与 Windkessel；监测质量守恒和 IB 泄漏。
5. 验证完整心动周期的电位、Ca、Ta、PV 曲线、端口流量、动脉压与源模型差异。

现有 `demo/real_lv_fsi/run_mac.py` 仍是 AFSI 对照入口，没有被冒充成三场入口。
完整驱动与额外物理状态未就绪前，不用旧 AFSI checkpoint 恢复新的 EP/循环状态。

## 本阶段验收记录

新增组件与原流固回归共 **65 passed**，其中包括真实 CPU/GPU BE–IB 端点验证。
真实 LV 网格 26,889 节点、135,430 单元的 40 个 EP 步检查通过，
并验证 wheel 包含全部新增 CUDA 源文件。详细记录见 [EP_LAND_VALIDATION.json](EP_LAND_VALIDATION.json)。

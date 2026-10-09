# 移动血液腔体、几何守恒与 EP–FSI 几何单向传递

本阶段建立随真实心肌几何移动的血液域，验证移动壁面、流体端口、独立集总阀门及
接受步事件。原 `run_afsei.py` 路径保留；新增驱动是明确标记的**单向几何接入**。
血液反力会计算并导出，但没有反馈到固体。不能据此声称完整三场力学、预充盈或生理周期已完成。

## 网格与方程

源网格仅有 LV，没有解剖瓣口。保留原内膜及其 P1 表面几何，用移动基底三角扇封口。
对基底长边作共形二分，向相邻内膜传播边分裂，避免挂点；圆形端口按基底子三角形
质心标记，所以离散边缘仍是近似。端口区域随后随基底节点移动，不每步重新打标签。

用内部核点向封闭表面各三角形构造四面体。血液节点到原固体节点的稀疏线性映射
同时给出位置、位移和壁面平均速度。参考 LV 的所有锥单元均正定，体积与原闭合内膜
一致。此网格细长、内部拓扑简单，仅用于几何/接口验证，不能替代质量血液网格或收敛研究。
检查整个线性移动路径上的四面体体积三次多项式极值；即使两端体积都正，也拒绝中途反转。
出现非正单元时拒绝并要求改进网格，不删除单元、不修补心肌运动。

流体采用质量集中的 CR 面平均速度与 P0 单元压力、BE 黏性预测和压力投影。
黏性使用分量梯度形式；开放端口为均匀压力自然载荷。默认启用基于旧场的显式 ALE 对流，
超出 CFL 检查界限则拒绝步骤。当前的阶数、锥网格、近似对流积分和投影分裂均未完成收敛验证。
没有输出完整移动流体/固体总能量平衡的通过结论；本阶段能量相关检查是界面传递的虚功一致性。

## 逐单元几何守恒与流量

线性节点运动时三角形面积向量随路径参数是二次函数，使用 Simpson 精确平均：

\[
\bar{\mathbf A}_f=(\mathbf A_f^n+4\mathbf A_f^{n+1/2}+\mathbf A_f^{n+1})/6,
\quad g_f=\bar{\mathbf A}_f\cdot\mathbf w_f.
\]

每个单元检查
\( (V^{n+1}-V^n)/\Delta t=\sum_f s_f g_f \)。位移先映射，再除以 dt；
不先计算两个大绝对坐标稀疏乘积再相减，避免零运动出现伪壁面速度。

流体不可压约束为 \(\sum_f s_f\bar{\mathbf A}_f\cdot\mathbf u_f=0\)，
端口采用相对流量 \(q_j=\sum_{f\in\Gamma_j}\bar{\mathbf A}_f\cdot(\mathbf u_f-\mathbf w_f)\)。
壁面平均速度设为映射得到的实际几何速度。因此全域满足

\[
\frac{V^{n+1}-V^n}{\Delta t}+q_{\rm MV}+q_{\rm AV}=0,
\quad Q_{\rm in}=-q_{\rm MV},\quad Q_{\rm out}=q_{\rm AV}.
\]

流量来自解出的流体面速度；没有用体积差直接赋值端口流量，没有截断流量或修正质量残差。
几何体积变化与实际流量仅用于接受条件的独立核对。

压力算子采用稳定的面压力差作用，避免近常量压力下两个大项相消；Jacobi 加常量模态
的一维 SPD 粗空间逆，仍检查真实残差。公共 PCG 增加可选 SPD 逆预处理回调，默认路径保持 Jacobi。
默认相对容差 1e-10；接受流量残差单独检查，不以递推残差代替物理质量检查。

## 独立阀门、RCR 与关闭事件

枚举 MV/AV 四种开闭状态。开放阀满足单向相对流量及集总阻力关系；关闭阀壁面速度
确实设为网格速度，并检查其法向反力对应的压差是否要求重新开放。
RCR 由实际 AV 相对流量后向 Euler 推进，Rc 只计一次。

规定运动模式默认初始 Pc=85 mmHg、Pat=8.5 mmHg，供带后负荷的接口检查；
这不是已完成的源项目 8 mmHg 心肌预充盈。EP 单向短程可显式使用 Pc=8 mmHg 做端口检查。
两个端口都关闭时，规定壁面不得改变腔体体积；不兼容运动直接拒绝，压力固定行不能充当隐藏出口。
闭合腔的压力常量仍缺少固体体积/压力约束，本阶段只选满足阀门压力范围的参考值，不能解释成真实 LV 压力。

只有先有接受的 MV 已关闭、AV 正向排出状态，再接受 AV 关闭，才记录关闭事件。
两个端口同时开放的被动穿流不算收缩期射血，不能据此提前触发 Land 卸载。
EP 单向桥接把有效事件连接到已有加载控制：保留关闭步实际使用的 `force_load` 历史，
重建空间张力锚点，再重置 Land、清零 Z；GPB/Vm/Ca 连续保留。不能用已发布的新 Z 再计算锚点。
下一步开始按关闭时间在 80 ms 内线性卸载；新刺激端点重新开启周期并清除旧射血标志。
接受事件逻辑及 40/80 ms 因子与已有源关闭接口核对；真实短程 EP 检查尚未经历完整射血/关闭/卸载。

## 界面反力与单向桥接

从离散流体动量反力取得流体对封闭壁面的力，以壁面平均速度映射的转置传回固体节点，核对
\(\mathbf F_s\cdot\dot{\mathbf x}_s=-\mathbf R_f\cdot\mathbf w_f\)。
虚功一致性证明传递算子配对，**不证明固体已经受到该反力**。

EP–FSI 原驱动目前仍使用原开放外箱 IB 流体决定心肌运动。新血液域只接收其接受步候选几何。
只有两边均成功才发布 EP/GPB/Land/原流固状态及新血液/RCR 状态；失败全部保留旧接受状态并缩步重试。
新反力只导出至 `interface_reaction.npz`，`feedback_applied=false`，不叠加到原 IB 流体之上。
完整双向接入需要替换原机械/流体块并处理封闭期的固体体积约束，否则会重复计算血液作用。

## 检查与输出

```bash
cd /mnt/large2/gjh/AFSEI
source scripts/activate_afsei.sh
python scripts/run_afsei_moving_cavity.py --duration-ms 2 --write-vtk --output results/my_moving_cavity_2ms
python scripts/run_afsei_moving_cavity.py --ep-checkpoint results/my_source_mechanics_2ms/checkpoint.npz --initial-pc-mmhg 8 --duration-ms 0.2 --output results/my_one_way_ep_02ms
python scripts/run_afsei_moving_cavity.py --resume results/my_one_way_ep_02ms/checkpoint.npz --duration-ms 0.2 --output results/my_one_way_ep_04ms
```

规定运动是 2 ms 周期的三角形缩放，幅度 1e-5，仅用来检验收缩/扩张端口关系。
`--ep` 可新建源力学 EP–FSI 单向运行；`--ep-checkpoint` 保留已有检查点的材料和状态。
首次接入旧 EP 检查点时，新血液域从零速度初始化；这是新子问题的启动，不能解释为
恢复了旧程序尚未具备的一致三场血流历史。后续新格式 `--resume` 才会恢复两套状态。
`--resume` 恢复模式、运动、物理参数，`--duration-ms` 为追加时间。输出目录须新建或为空。

输出包含 `history.csv`、`summary.json`、纯数值检查点、反力及可选移动域 VTK。
单向运行用一个原子 SHA-256 检查点保存 EP 和新血液两套完整状态，包含关闭锚点和移动网格上一几何。
producer=`afsei-one-way-ep-ale`；独立移动流体格式为 `afsei-moving-cavity-ale`；旧 EP/FSI 和固定腔体格式不混用。
失败会保存最后接受状态与失败摘要，不把失败试算的几何、流量或关闭事件写作已完成结果。

验证记录见 [MOVING_CAVITY_VALIDATION.json](MOVING_CAVITY_VALIDATION.json)。
`moving_cavity_implemented=true`、`independent_lumped_valves_implemented=true`；
单向模式另有 `ep_geometry_drives_ale_blood=true`、`accepted_valve_events_connected_to_ep=true`。
`two_way_interface_feedback_implemented=false`、`preload_performed=false`、`three_field_integration_implemented=false`。

剩余工作：质量血液网格和收敛检查、用拟合血液替换原外箱流体的双向力学求解、
等容期的固体体积/压力约束、8 mmHg 预充盈，以及真实完整周期内的关闭卸载验证。

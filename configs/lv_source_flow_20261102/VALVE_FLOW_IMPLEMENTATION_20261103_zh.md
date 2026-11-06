# 纯固体瓣口机制 + 流体流量驱动 RCR

本记录更新 STATUS_zh.md / REFERENCE_AUDIT_20261103_zh.md 中“有限阻力瓣口未实现”的状态；内部基底几何仍未实现。

## 方程

原始三元素 Windkessel 本身已经含 Q。加入三维流体不需要额外叠加一份流体压力或惯性项，而是改变 Q 的确定方式和压力返回位置。

实际通量 Q = integral(u dot n dA)，固定端口向外为正。
移动截面采用 (u-w) dot n，需在其几何离散实现后使用。

```
C*(pc_next-pc_old)/dt = Q - (pc_next-pv)/Rp
pAO = pc_next + Rc*Q
Q = Phi(p_up-pAO)
Q = Q_free + G*p_up
```

最后一式是现有 MAC 求解器在物理出口压力上的流量响应（G<0）。
这四式联立，Q 是最终三维速度积分值；没有将几何体积差分替换实际流量。
p_up 是未解析瓣口的上游截面压力；pAO 是下游动脉压力；两者不能都作为同一截面的压力。
Rc 只出现一次，遗留 Raortic 不使用。压力不再额外叠加到心内膜。
血液惯性和黏性在三维流体方程中保留；不凭空再向 RCR 加惯性项。

Phi 完全采用纯固体 windkessel.h 的严格单向、平滑有限阻力律。
关闭分支抛出 ValveClosureRequired，调用者必须回退到原接受态，以闭口条件重算同一区间；不能用压力值令净流量为零冒充逐点封闭。

## 实现与接口

- ep/aortic_valve.py：原瓣口律、解析导数、实际流量对应压力、流体响应与瓣口/RCR 联立标量求根。
- mac/valved_cut_plane.py：在已有 MAC 物理截面算子上加入瓣口压降；response 是最近 advance 状态的非线性切线。
- SinglePortBoundary(..., valve=AorticValveParameters(...)) 启用该机制。现有默认 None 保持旧组件兼容。
- LVCirculation 默认使用源瓣口参数；射血 commit 必须提供 upstream_pressure_mmhg，并验证实际流量的瓣口残差。
- 各次求根冻结 pc_old，只有接受完整 FSI 步后提交。

标量求根使用已缓存的流体压力响应，不在每次标量迭代重解三维流场。
非线性 FSI 调用方应在每次更新基态后刷新切线，不能复用旧的全程线性阻抗。

## 已验证

CPU/GPU MAC 边界实际流量、散度、RCR 平衡、瓣口耗散非负、非线性切线有限差分；源循环阶段和卸载。
直接编译服务器原版 windkessel.h，dt=1e-4 与 5e-5 s，压力 80/85/85.01/86/100/160 mmHg。
10个正流量样本的最大流量差 2.5011e-11 mL/s，最大上游压力差 1.2648e-12 mmHg。
源码对比文件：results/lv_valve_alignment_20261103/source_comparison.json。

## 适用范围

这是循环方程和物理 MAC 截面边界组件验证，不是完整真实 LV 模拟。
盒内自然基底开口仍未有可验证的开闭/压降边界。不得把盒外压力截面直接映射成内部开口，也不得恢复弃用的 FE 连接壁。

Gao et al. 2017 §2.4（https://eprints.gla.ac.uk/143010/1/143010.pdf）支持三维出口实际流量输入 RCR、返回出口压力的连接方式。
本版本保留纯固体有限阻力瓣口和 BE 隐式步，故不是 Gao 理想开闭阀和显式时序的逐项复制。

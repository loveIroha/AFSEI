# 参考范围核对（2026-11-03）

用户明确：Windkessel 相关机制参考纯固体 LV EP–力学，不以 Gao 理想阀替换。

## 原始几何参考

服务器 `/mnt/large2/gjh/npuheart-LV-systole`，HEAD ceb74d5c661f014555bb349e90d23acf48a27e85。

- test/FSI/IBFE/3D/fsi_real_LV_systole.cpp：原 LV 浸入 15 cm 立方流体盒；StaticFlow；无新增流道或被动 FE 连接壁。
- PhysicsSolver/StokesFlow3D/StaticFlow.h：六个面速度类型均为 1，压力类型均为 2。
- PhysicsSolver/multigrid/BoundaryTypes.h：1=Dirichlet，2=Neumann。
- 静止算例不填入非零边界值，盒外边界为零速度、压力齐次 Neumann；不是 p=0 开放储库。
- RealLeftVentricleSystole/ConstitutiveLawLV.ufl：心内膜 pressure 载荷；基底环向和纵向罚约束，径向自由。
- 基底开口位于流体盒内部，自然连通腔内外；没有专门的基底流体截面边界。
- 该路径没有 Windkessel、分阶段基底闭口和严格等容条件。代码中的“等容收缩期”是预设压力函数分段名称，不能证明体积不变。

## 循环机制参考

服务器 `/mnt/large2/gjh/test_ele_soild_heart-test`，HEAD 5bdde73c6be10cefadb005a0fcec5fdb6521f336。

windkessel.h 的现用 pressure-driven 路径：

```
dp = pLV - pAO
Q = 0                                      if dp <= 0
R(dp) = (Rmax-Rmin)*exp(-2dp/eps)/(1+exp(-2dp/eps)) + Rmin
Q = dp/R(dp)                               if dp > 0
pAO = pc + Rc*Q
C*(pc_next-pc_old)/dt = Q - (pc_next-Pvenous)/Rp
```

默认 Rmin=.001、Rmax=1e4、eps=.1，单位 mmHg、mL、s。参数从实际算例配置读取；不擅自重标定。
Raortic=.01 为现用 pressure-driven 路径未使用的遗留量，不可再叠加到 Rc。
lv_cycle.h 使用力学体积与 Q 的守恒残量联立求腔压；无出入流时冻结阶段体积；Pc 只初始化一次并持续演化。
主动张力在出口关闭时冻结，80 ms 卸载；保留用户明确的首次 200+300 ms 充盈时序。

## 当前代码差异及下一步约束

ep.lv_circulation 已完成的阶段组件采用理想开关，尚未实现上述平滑有限瓣阻力；不能称作循环机制完全对齐。
SinglePortBoundary 只有物理盒面端口，尚未支持原 LV 的内部基底开口。

迁移时实际流体 Q 必须同时满足瓣口关系和 RCR，不能从零维模型另算一个 Q 覆盖实际流量。
需要明确内部基底处的压力差/流量条件及闭口运动学条件；不新增实体 FE 壁，不把 Windkessel 压力重复加到整个心内膜。
纯固体统一腔压在 FSI 中对应哪个可功率配对的端口压力，需在边界离散中定义；不能任意用核心平均压力代替并宣称完全等价。
自然开口可直接参照此项目；等容闭口和循环压降仍是新增边界任务，原参考程序没有给出其现成实现。

本次只读核对参考源码并记录差异，未更改运行方程或启动计算。

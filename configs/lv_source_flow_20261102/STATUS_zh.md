# 原始左心室：阶段控制与流量驱动 Windkessel

2026-11-02。本记录对应可复用组件修改，不是完整左心室算例通过报告。

## 已实现

- `ep.lv_circulation.LVCirculation`：不可变阶段计划 + 接受后提交。
- 首次心内膜加载 200 ms 升至 8 mmHg，保持 300 ms，500 ms 刺激。
- 首次刺激初始化 Pc=85 mmHg；后续周期继承 Pc，不重置。
- 等容期双分支关闭，Q=0；动脉侧继续按 RCR 远端泄放更新。
- 等容体积目标取阶段进入时的接受体积；检查逐步质量余额与累积等容漂移。
- 射血时实际有向流量进入 RCR；Pc 与出口压力分开，不返回心内膜载荷。
- 负射血流量拒绝，重算同一区间关闭状态；关闭时通知调用者捕获已接受主动张力。
- 闭口后 80 ms 线性卸载，跨越卸载终点的步自动截断；之后仍保持等容，直至满足释放条件。
- 再充盈从结束等容舒张开始计时，200 ms 升压 + 300 ms 保持后再次刺激。
- `SinglePortBoundary.commit` 防止逆流和重复提交；求解期间 Pc 不变。
- 新增被动 FE 连接壁的 `lv_dual_port_cycle` 入口已停用，旧命令在创建结果前报错。

本轮未修改 NS、MAC、IB 传递、EP 离散、GPB/Land 方程或心肌本构。
循环控制器不装配任何额外固体、封盖力、体积修正或压力补偿。
先前 housing 网格及结果只属于弃用诊断，不作为当前 LV 模型。

## 调用约定

1. 对接受态调用 `plan(dt_s, lv_pressure_mmhg=...)`。
2. 按 `boundary_mode` 接储库/RCR/闭口；wall_pressure 仅为独立心内膜外加载荷。
3. stimulus、closure snapshot 和组织状态先暂存；不可提前发布。
4. 在 plan.dt_s 上求解 FSI，计算实际端口通量及 LV 体积。
5. `commit(plan, outward_flow_ml_s=..., volume_ml=...)` 验证并提交循环状态；随后发布已暂存的 FSI/EP 状态。
6. 反向出流以原接受态重新 `plan(..., close_outlet=True)` 并重算；禁止截零 Q 或修改已接受体积。

流量单位 mL/s，向腔外为正；时间 s；循环压力 mmHg。
RCR 参数显式由调用者提供，不改变原实验标定值。默认类参数不代表已选定的实验配置。
目前阀事件由已接受腔压触发，分辨率为调用步长；不声称已实现连续时间精确寻根。
plan 的卸载因子作用于关闭时冻结张力；调用者仍须使用 BufferedLandLoad 的 closure_snapshot。

## 验证及范围

服务器执行：

```
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/mnt/large2/gjh/conda-envs/afsei/bin/python -m pytest -q \
tests/test_lv_circulation.py tests/test_single_port_boundary.py
```

结果：10 passed，2.56 s。包含真实 CPU/GPU MAC 盒面端口求解；循环测试使用给定体积/流量数据，不是完整心室模拟。

## 尚未完成：原基底开口的流体边界

SinglePortBoundary 只支持物理 MAC 盒面；真实基底位于盒内，不能直接调用它代表原心室基底。
尚需为原开口实现明确的流体域截断/边界离散，以及开闭切换时与径向移动基底的连接。
不能通过改盒面数组索引、在内部面直接赋压、或恢复旧封盖乘子声称完成此事。
原心肌完整保留，基底仍允许径向运动，心尖自由。不得以新增被动 FE 壁替代该边界任务。

本版本尚不能启动符合上述几何约束的完整两周期；不提供伪装成已验证的运行命令。

参考：纯固体源码 lv_cycle.h / windkessel.h / isovolumic_newton.h；Gao et al. 2017 §2.4，https://doi.org/10.1016/j.medengphy.2017.06.042 。
保留用户的充盈时序及 BE–BE，而非逐项复现 Gao 的几何、瓣叶和时间格式。

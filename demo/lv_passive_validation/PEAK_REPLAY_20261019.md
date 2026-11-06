# 第一周期峰值诊断实验（2026-10-19）

本实验从已有验证预加载重新开始，计算 500–665 ms，以核查核心压力与人工出口 RCR 乘子的差异。保留原两周期起搏配置，但主动截取第一周期的早期区间，不作为完整周期验证。

## 设置和保护范围

- EP：当前 nodal P1、CN、AmgX，与已完成的 `afsei_two_cycles_wall_fill` 一致。
- 流固：原 legacy-balanced、BE–BE、MAC128、spectral 压力、BE projection、centroid 盖面。
- RCR：Rc=0.0345、Rp=0.7245 mmHg·s/mL，C=1.3 mL/mmHg，初始 Pc=85 mmHg。
- 采用 `circulation_source_rcr_wall_closed_reference.json`；盖面参考补齐仅在充盈阶段启用，665 ms 前尚不进入充盈。
- NS、IB 传递、EP/GPB/Land、心肌本构和人工张力卸载公式未改。服务器 `provenance.json` 记录受保护模块哈希及与上一轮配置的比较。

## 新增观测功能

`scripts/run_afsei_cycle.py` 可选 `--diagnostics-config`。不指定时原运行路径继续适用。

本实验在 500.1、550、557.1、580、620、650、664.3、664.4、665 ms 保存接受状态，并保存该步真实的上一接受速度、按实际 dt 重算的原对流输运场。指定时刻都位于 EP 整数时钟上。诊断只在这些时刻复制/压缩数据，不在每步额外求解流体。

每个存档位于 `diagnostics/tick_XXXXXXXX/`：

- `checkpoint.npz`：标准当前接受态。
- `history.npz`：上一速度、对流输运场和实际时间步记录；不是生产重启文件。
- `pressure_diagnostics.json`：离线分析成功后生成。

`scripts/analyze_afsei_snapshots.py <实验目录>` 对已有存档做离线后处理。已通过重构检查且晚于输入文件的分析会复用；显式 `--force` 才重新计算。输出包括当前/冻结端口面积、相对流量、两侧 ±1h…±4h 压力及法向黏性应力，和原 BE 线性响应分解。

压力分解为上一速度提升、对流输运增量、被动/基底力、主动张力、规定壁载荷、材料盖面反力及流体盖面牵引；每项继续使用现有线性流体求解器，已知压力势按当前离散代数加入。所有项相加需重构保存的压力与速度，误差相对范数阈值为 1e-7。这些分量不是独立的生理压力或耗散项。有限距离的两侧应力差也不是界面零距离极限残差。

## 定向核验

6 项历史存档/时钟检查和 2 项制造解测试通过。500.1 ms 实际快照的分解重构相对误差：压力 7.81e-13，速度 2.43e-9；速度归一化分母至少为1。与上一轮首步比较，容积、核心压力、主动张力相同，出口乘子差 2.66e-15 mmHg，流量差 1.37e-13 mL/s。这里只证明首步存档及分解可用，峰值结论需等实验结束。

## 本次服务器任务

结果：`/mnt/large2/gjh/AFSEI/results/afsei_peak665_closed_reference_20261019`

日志：`/mnt/large2/gjh/AFSEI/results/logs/afsei_peak665_closed_reference_20261019.log`

启动和任务记录：`results/lv_peak_experiment_20261019_setup/job.json`、`run_peak665.sh`。第一次启动在环境激活时因严格 unset 检查退出，未进入计算；原日志保留为 `activation_failed_attempt.log`，启动器已修正。

一次性后处理任务等待本次模拟结束，再依次分解存档、比较上轮相同时刻的数据并绘图。状态见 `results/lv_peak_experiment_20261019_setup/postprocess_status.json`，日志见同目录 `postprocess.log`。成功后生成：

- `diagnostics/snapshot_analysis.json`
- `diagnostics/experiment_summary.json`
- `diagnostics/peak_diagnostics.png`、`peak_diagnostics.pdf`

此前日志中同区间的接受步耗时约3606 s；本次另有启动、稀疏存档和离线诊断开销。任务启动不代表峰值或完整周期已通过。

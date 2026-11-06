# 更新后的粗 EP 网格两周期测试

从原预加载状态重新计算两个完整周期，检验结果与性能。不是从短时延时诊断状态继续，也不加密 EP、固体或流体网格。

## 时间安排与压力含义

- 第一次刺激：500 ms；第二次刺激：1400 ms。
- 每周期 900 ms；完整动态计算为 500–2300 ms。
- 人工主动张力卸载仍为出口关闭后的 80 ms。
- 舒张期入口在压力波形前 10% 内由 0 升至 8 mmHg，然后保持。
- 额外腔壁载荷仍为 0→12→0 mmHg。原波形在下一刺激/周期终点前 100 ms 结束，之后入口保持 8 mmHg、额外壁载荷保持 0。
- 第一周期的原压力波形终点为 1300 ms，之后保持至 1400 ms，复现已验证的 100 ms 延时；第二周期波形终点为 2200 ms，之后保持至 2300 ms。
- 充盈入口仍由实际卸载完成与压力门控决定，射血结束、充盈开始等事件不被人为固定。

`period_ms=900`、`filling_hold_ms=100` 是本次标准 JSON 的显式配置。旧配置默认仍为 800 ms 周期、0 ms 保持。检查点会保存两项时间设置。

## 离散与模型

EP 加密因子明确为 1，原网格 26889 节点/135430 四面体。EP 为 nodal P1、节点扩散积分、CN、AmgX PCG-AMG（size2）。dt_EP=0.05 ms、dt_ODE=0.001 ms，GPB/Land 和原人工卸载保留。

固体原网格与本构、MAC 128³ 规则网格、既有 legacy-balanced MAC/IB、谱压力、BE–BE/BE 投影、端口几何和求解容差沿用此前设置。本次没有把 legacy 分支描述为完整 Griffith–Luo 空间实现，也没有更换耦合算法。

RCR：Rc=0.0345、Rp=0.7245 mmHg·s/mL，C=1.3 mL/mmHg，静脉参考 8 mmHg，初始电容压力 85 mmHg。端口仍为理想化端口，无解剖瓣膜。

## 运行与产物

标准作业：`configs/gao2017/two_cycles_coarse_20261023.json`。

带外部性能记录的启动方式：

```bash
python scripts/run_afsei_profiled_job.py configs/gao2017/two_cycles_coarse_20261023.json
```

普通启动方式：

```bash
python scripts/run_afsei_job.py configs/gao2017/two_cycles_coarse_20261023.json
```

不要同时启动两条命令；它们对应同一输出目录。

输出：`results/afsei_two_cycles_coarse_900ms_20261023`。

标准产物为 pv.csv、steps.jsonl、report.json、checkpoint.npz、configuration.json、circulation.json。1399.9 和 2299.9 ms 保存独立检查点及诊断历史。阶段、周期、空间配置均记录在输出中。

带性能记录的入口每 60 秒从独立父进程读取一次 GPU 占用、功耗/温度/时钟和进程列表，不同步求解器张量。运行开始时 GPU1 空闲；后续是否有其他任务共享，以 .gpu.jsonl 实际记录为准。

结束后自动生成 performance_results.json 与 pv_and_histories.png/pdf，包含各阶段累计耗时、每步耗时分布、EP/GPB/Land/FSI 分项、两周期 EDV/ESV/EF 和记录得到的 PV 曲线，不人为闭合曲线。

900 ms 周期比此前 800 ms 周期多计算 12.5% 物理时间，工况也改变。比较性能时应同时查看每步/单位模拟时间成本、分项及 GPU 共享情况，不直接把总墙钟差当作算法加速比。阶段验收通过不代表已达到生理目标或周期稳态。

# 第二十六阶段：正式周期驱动与后台续算

本阶段提供能够持续推进周期的入口，不把短程检查等同于完整周期已经通过。
按用户要求，不启动两个完整周期；完整周期由用户在服务器后台运行。
原 MAC 128³ / BE–BE 隐式流固框架、源材料、基底、纤维、EP/GPB/Land 参数与生产求解容差保留。
没有增加外层完整 FSI 迭代或贴体四面体血液网格。

## 源规则与流固扩展

参考固定源提交 `5bdde73c6be10cefadb005a0fcec5fdb6521f336` 的 `lv_cycle.h`、
`active_tension_cycle.h` 和 `main.cpp`；notebook 的等容约束是补充参考。

1. 预加载读取第二十四阶段已经验收的 P8 压力配对状态及重新计算的 100×5 ms Zs/Zw；
   不导入旧的仅内膜规定压力位移。500 ms 首次刺激，间隔 800 ms，脉冲宽度仍为 2 ms。
2. 等容收缩：理想化 MV 关闭，AV 使用实际相对通量的单向 RCR 互补条件。
   出口未开放时保持进入该阶段的固定 EDV。
3. 射血：由已接受的 AV 条件决定开放；RCR 使用实际测得的相对 Qout，
   每步独立核对有限步体积、总血量与功配对。绝不把 `-DeltaV/dt` 当作端口测量。
4. 出口由开放转为关闭后，记录该接受状态实际施加的空间 Ta。
   Land 恢复为 500 ms 前刺激快照，Zs/Zw 归零，保留 Vm/GPB/Ca；
   以该空间快照线性卸载 80 ms，MV/AV 均关闭，体积固定为接受的 ESV。
   不按任意 800/1600 ms 截止时刻制造闭瓣事件。
5. 80 ms 卸载完成且已求得的 MV 盖面平均压力降至不超过 8 mmHg 后，
   保留该压力作为充盈起点，采用源 smoothstep：
   `Pfill=Pentry+(8-Pentry)*s²*(3-2s)`，终点为下一次名义刺激。
   原纯固体将此压力施加到内膜；流固模型将其作为理想化 MV 的压力储库边界，
   心室压力由流体方程计算。允许有符号净流入，不引入源程序没有的 Rin 或单向 MV 阻力。
   流量 `Qin=-(B_MV*u-C_MV*dx/dt)` 由实际流场测量；AV 在此阶段关闭。
   核心流体平均压力与端口压力分别输出，不能把两者在非稳态时假定完全相同。
6. 下一次刺激的结束端点保留已达到的 EDV、动脉压力和 EP 状态；从下一步重新推进 Land。
   EP 冻结仅在充盈期满足既有恢复门控后触发，刺激时恢复。

每个试算采用同一旧电容压力，只有接受后发布 Pc。Pc 在所有阶段连续衰减/推进，不按周期重置。
每步未知量仍只有位移与盖面压力。闭合行施加平均相对通量条件，开放压力行在同一 Newton 步求解。
旧盖面几何、压力分区及主动载荷历史单独保存，阶段切换不会重新拟合已经接受的反力。

固定质量目标为每步 1e-5 ml/s、累计体积 1e-3 ml，基底弱闭合通量 1e-5 ml/s，
压力律 1e-7 mmHg，伴随功 1e-7。全周期通过还要求每个刺激周期实际经历四阶段并结束于充盈。
若下一刺激之前未完成卸载/进入充盈，明确报错并保留最后接受检查点，不伪造完成的心搏。
完整周期在 500–2100 ms 运行，只有 500/1300 ms 两次刺激；2100 ms 不额外制造第三次刺激。

## 后台运行

代码直接同步到 `/mnt/large2/gjh/AFSEI`，服务器无需再 `git pull`。
从新 P8 预加载开始；参考检查点只供原网格/生产配置。输出目录必须尚不存在。

```bash
cd /mnt/large2/gjh/AFSEI
source scripts/activate_afsei.sh
mkdir -p results/logs
nohup python -u scripts/run_afsei_cycle.py \
  --preload results/phase24_mac128_balanced_preload_final_v1/preload_result.npz \
  --reference-checkpoint results/phase10_mac128_source_preload_v1/checkpoint.npz \
  --cycles 2 --output results/my_two_cycles \
  > results/logs/my_two_cycles.log 2>&1 < /dev/null &
echo $!
```

激活脚本使用现有 afsei 环境和物理 GPU 1；不要同时运行另一个 GPU 验收进程。

```bash
tail -f results/logs/my_two_cycles.log
cat results/my_two_cycles/progress.json
```

续算恢复完整相同的配置和总周期数，使用新输出目录；先确认原进程已退出。
例如中断后继续剩余时间，不能再指定 `--cycles` 更改既有刺激计划：

```bash
nohup python -u scripts/run_afsei_cycle.py \
  --resume results/my_two_cycles/checkpoint.npz \
  --output results/my_two_cycles_resume \
  > results/logs/my_two_cycles_resume.log 2>&1 < /dev/null &
echo $!
```

## 输出与短程验收范围

默认每 100 步和每次阶段事件原子保存 `checkpoint.npz`，启动前及异常退出也保存最后接受状态。
异常或主动中断会生成 `status=incomplete`，保存退出原因。异常不会被转换为通过。
检查点生产者 `afsei-mac-four-phase-cycle` 与旧区间/普通入口不同，必须用本入口续算。

- `pv.csv`：时间、阶段、心搏、体积、核心压力、Pc、实际 Qin/Qout、Ta、质量误差与完整步耗时。
- `steps.jsonl`：每步非线性迭代、EP/ODE、耗时分项、阀/阶段事件、RCR 与功检查。
- `progress.json`：约每 1 ms 更新并原子发布，可在运行时读取。
- `report.json`：退出状态、检查点、阶段覆盖、最大质量误差。

GPU 测试包括实际入口通流及切线、闭瓣快照、80 ms 释放门控、连续充盈、下一刺激保留 Pc/EP、
独立检查点重放与失败回退；事件测试不等于完整物理周期轨迹。
`--duration-ms 0.2` 用于真实 128³ 两步启动检查，仍配置两个周期，但只推进 0.2 ms。
短程检查结束时 `full_cycle_validation_passed=false` 是正确的范围标记，可从其检查点继续到总周期终点。

已经完成的验收（不再重复运行）：全套 GPU 回归 163 项通过；补充最终 RCR 能量接受检查与
闭瓣状态切换重启后，相关 9 项通过。真实 128³ 正式入口从 500 ms 启动两步至 500.2 ms，
再通过正式 `--resume` 命令续算两步至 500.4 ms，两次均 `status=completed`，
最大每步质量误差 9.64e-13 ml/s，累计体积误差 1.71e-16 ml。
原生实际出口开放检查点另作两步分段重放：18 个物理字段一致性检查通过，
最大位移差 1.07e-14 cm，压力差约 1.25e-9 mmHg，Pc 差为零。
压力按 1e-8 mmHg 绝对目标核对，允许缓存重新建立后的浮点舍入差。
结果保存在服务器 `results/phase26_mac128_cycle_smoke_v1`、
`results/phase26_mac128_cycle_resume_smoke_v1` 与 `results/phase25_mac128_balanced_replay_v2`。

当前原生收缩/通流窗口实测完整暖步约 2.26–2.88 s/步，dt=0.1 ms，
EP/GPB/Land 约 0.083–0.092 s，其余主要是隐式 FSI。两个周期需要约 16000 步；
按该窗口速度外推约 10–13 小时，实际卸载、大变形、充盈和文件输出可能改变耗时。
完整周期、PV 环合理性、空间/时间收敛、动态角动量与点位无穿透仍需要进一步验证。

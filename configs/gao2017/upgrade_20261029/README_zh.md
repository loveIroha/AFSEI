# 真实左心室：连接、完整充盈、周期状态和主动力对照

> 已被用户2026-10-29明确的200 ms升压＋300 ms保持时序取代。此目录800 ms配置仅保留用于历史追溯；当前运行入口为`configs/gao2017/source_clock_20261029/`。

这些配置使用原粗EP/固体网格、128³ MAC、原NS及基础IB离散、BE–BE、源被动材料/基底罚项和Land主动应力。闭口后冻结接受的空间Ta并于80 ms内人为线性归零。

## 本轮的四项修改

1. `cut_geometry.shared_collar_cells=3`：固定双流道下部共享短连接区，上部仍为两个独立截面。内部隔板移到运动开口及其两格IB支撑范围之外；外壁、顶部截面积和总储量积分域不变。不再要求下部共享区域内的材料开口永远留在单独矩形里。步初与候选接受末态都做覆盖检查。存在横向交换时，流道有限体积预算改用两条流道的联合控制体积。此项是边界几何修改，不是放松几何容差，也没有加入真实瓣膜或新的FE材料区。
2. `filling_clock`：从实际进入充盈起计时，入口0→8 mmHg持续800 ms，额外心内膜载荷0→12→0 mmHg峰值在400 ms、800 ms撤去，再保持入口8 mmHg至少100 ms。显式物理时钟优先于继承的归一化`gao_filling`比例字段。必要时整体推迟尚未执行的刺激和最终终点；绝不压缩波形，也不提前刺激尚未完成卸载的心肌。新RR将随实际阶段变化，不能与旧900 ms结果当作单因素对照。
3. `periodic_criteria`：指定初始Pc仍为85 mmHg并连续演化。可选完整流量模板的0D周期初值仅作为新运行的种子，不会在重启或每个周期重置Pc。只有同相位EDV、Pc、RR、电位、钙、Land、伸长、位移、完整MAC速度和压力场满足条件并连续通过，才发布周期状态。全场参考只在刺激边界复制，检查点精确保存；不进入Newton残差或Jv调用。早期仅保存EP参考的开发检查点会保留物理状态、清除旧稳态计数并重新采样。
4. `active_diagnostics_every`与`compare_active_source_load.py`：输出体积加权Ta、主动Cauchy应力、伸长和收缩功；可在完全相同构形、内部状态及压力下独立比较源UFL公式与GPU有限元力。保持Ta尺度不变。这是本构与装配验证，不宣称已证明两个完整泵血轨迹相同。

## 入口选择

- `two_cycles.json`：从兼容预加载重新计算两个完整周期。刺激起点最初为500/1400 ms，第二次实际时刻按完整充盈需求延后。输出是否达到周期状态；两周期完成不自动等于稳态。
- `periodic_then_two_cycles.json`：最多8个刺激周期作为上限；先等待完整三场周期判据连续通过，再记录两个周期。`report.json.reporting_window`标明应分析的正式时段。上限内未完成则返回`incomplete`，保留检查点，不把瞬态结果标成稳态。
- `short.json`：两步真实左室短验证，不能用其推算全周期耗时或EF。
- `active_comparison.json`：保存收缩构形上的独立同负荷装配对照，不推进任何物理时间。

正式运行使用`scripts/run_afsei_cut_cycle.py`加JSON路径。目录已有结果时程序拒绝覆盖。

## 服务器运行

```bash
cd /mnt/large2/gjh/AFSEI
mkdir -p results/logs
export CUDA_HOME=/mnt/large2/gjh/conda-envs/afsei
export CONDA_PREFIX=/mnt/large2/gjh/conda-envs/afsei
export TORCH_CUDA_ARCH_LIST=8.9
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/afsei-mpl
CUDA_VISIBLE_DEVICES=0 nohup /mnt/large2/gjh/conda-envs/afsei/bin/python scripts/run_afsei_cut_cycle.py configs/gao2017/upgrade_20261029/two_cycles.json > results/logs/afsei_full_filling_two_cycles_20261029.log 2>&1 &
```

若需要“先稳定后取两个周期”，将配置改为`periodic_then_two_cycles.json`，同时换一个日志名。该运行可能明显长于两个瞬态周期，配置不会自动放宽收敛标准来提前结束。

## 重启与初值

新模型使用新的正式运行目录和原100点兼容静态加载历史。旧停止点的单步连接试算明确标记为`model_transition_probe`，仅用于检验连接修复，不能混入正式新周期。

重启配置只指定`resume`、新的`output`及执行/诊断选项。几何、时钟、刺激日程、周期参考场和循环参数都从检查点恢复；禁止在普通重启配置中替换模型设置或重置Pc。

`rcr_initialization.mode="frozen_flow"`需要CSV路径`template`及完整`period_start_ms`、`period_end_ms`，列名为`time_ms`、`q_out_ml_s`，记录间隔不超过1 ms且覆盖完整名义周期。它只计算给定固定流量的BE周期种子，不证明耦合解已经周期稳定；正式判据仍必须通过。默认配置采用`specified`，使用循环配置中的85 mmHg。

本轮没有以增大Ta、减小RCR电阻、强制腔压等于出口压力或指定ESV来拟合PV。更改完整充盈时标后的EDV/ESV及运行性能，需要实际完整周期结果确认。

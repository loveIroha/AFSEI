# 腔壁充盈压力：IBFE 参考核对与接入

## 实际参考实现

用户指定的服务器入口是 `/mnt/large2/gjh/npuheart-LV-systole/test/FSI/IBFE/3D/fsi_real_LV_diastole.cpp`，参考项目 HEAD 为 `ceb74d5c661f014555bb349e90d23acf48a27e85`。本轮读取了实际工作区文件，并将五个直接相关文件保存于 `results/wall_filling_20261018/ibfe_diastole_reference.tar.gz`。该项目另有电生理相关未提交修改，本轮未更改参考项目。

调用链：

1. 入口创建 `RealLeftVentricleSolver` 和 `StaticFlow`，调用 `fsi_lid_simulation_explicit`。
2. `RealLeftVentricleSolver.h` 中 `WallPressure` 返回 `10665.789 * min(1,t)` dyn/cm²，即约 1 s 内线性升到 8 mmHg；每次固体力装配前把 `pressure->t` 更新为物理时间。
3. `ConstitutiveLawLV.ufl` 将压力加入心内膜 `ds(2)` 的表面弱式，经 `L=rhs(H)` 进入固体力右端。
4. `solveOneStep` 求质量矩阵下的有限元力密度；IB 驱动将它散布到 Euler 网格，求解流体，再插值速度更新固体。
5. 参考 `StaticFlow` 的背景盒为零速度 Dirichlet、压力 Neumann，没有本例的 MV/AV 端口、Windkessel 或心动周期阶段控制。

因此，这个参考的压力是规定的心内膜表面外力，并不是入口储库压力，也没有通过控制流体腔内平均压力来实现。直接腔壁压力不能一概判成重复加载。是否重复取决于同一物理外载是否被再次计入另一条加载路径。

参考源码法向项逐字写为 `det(FF)*inv(FF)*N`。AFSEI 沿用现有定向三角面元的当前面积法向，等价于标准随形压力的 `J F^{-T} N`，没有为了逐字复刻参考而更改现有几何法向约定。参考程序的线性 1 s 升压、显式固体更新和背景盒边界也未照搬；本轮对齐的是心内膜表面加载位置与有限元外力传递路径。

## AFSEI 的具体选择

按用户要求保留：

- EP/GPB/Land 主动张力及其反馈。
- 接受闭瓣时刻保存主动张力，随后 **80 ms 人工线性卸载至零**；充盈阶段施加主动张力为零。
- 电力项目的周期平滑升压曲线和下一次起搏时刻。
- 当前 NS、MAC、IB 传递、EP PDE、心肌本构及 BE–BE 时间格式。
- 原 RCR 微分方程、接受步提交及所有阶段的电容演化。

新增配置 `configs/circulation_source_rcr_wall.json` 明确选择：

```json
{
  "filling_pressure_location": "endocardium",
  "filling_reservoir_mmhg": 0.0
}
```

在充盈区间端点，以现有周期时钟计算 `p_wall(t)`，通过既有 `MACEndocardialPressure` 装配

\[
\mathbf F_{wall}(\mathbf x,t)=-p_{wall}(t)\int_{\Gamma_{endo}(t)}N_a\mathbf n_s\,dA,
\]

加入原固体力，再走原来的 S/J 和流固残差。这里只增加载荷及其一致方向导数，没有替换本构、IB 算子或增加外层流固迭代。

充盈供液端口仍开放，入口为独立的 **0 mmHg 表压参考**，不再把同一条 0→8 mmHg 曲线再次赋给入口。流体继续求解，流量由实际相对端口流量计算；AV 关闭，电容按 Q_AV=0 继续推进。这个零参考入口是将参考算例的表面外力加载适配到当前有端口模型的明确边界选择，并非参考代码中存在同一个入口。

壁压为规定外力，不强制流体核心区压力等于该数值，也没有强制恢复首次预加载 EDV。输出分别保存 `wall_pressure_mmhg`、`boundary_pressure_mmhg`、原始 `pressure_core_mmhg` 和表面外力功率 `wall_load_power_erg_s`。核心压力是原始流体表压，没有人为平移或改名为总腔压。

在下一次起搏端点，最后一个充盈区间仍使用其端点壁压；随后区间撤去规定的充盈壁压，进入原闭瓣/RCR 条件。接受状态单独保存上一物理步实际使用的壁压，因此阶段标签已经改变时，重启也能精确重建上一物理步的力。这个离散交接已做短步检查，但不据此宣称完整周期压力交接或生理结果已经验证。

RCR 参数选择与纯固体实际轨迹相同：Rc=0.0345、Rp=0.7245、C=1.3、Pv=8、初始 Pc=85，单位仍为 mmHg/mL/s。AFSEI 原理想单向阀保持不变，尚未改为源固体模型的平滑瓣阻。

## 兼容与验证

旧 `circulation_source_rcr.json` 仍只调整 RCR，保留入口升压控制，便于单项对照。未配置新选项时仍使用旧入口方案；旧 checkpoint 缺失新字段时恢复为入口方案和零规定壁压。改变加载位置须从预加载开始新运行，不能把旧周期历史标为新模型结果。

四项新增 GPU 检查已分别通过：

1. 原压力平衡路径：完整流固雅可比与独立有限差分一致；载荷只装配一次，有限元表面功率一致。
2. 标准 IB 路径：同样的雅可比、单次加载与表面功率检查通过；S/J 算子未改。
3. 实际充盈步、保存和恢复、继续推进一致；失败回滚不改变接受的壁压和力；人工阶段发布保持力历史可重建。
4. 实际推进到下一次起搏并重启，再推进首个闭瓣步：最后充盈步保留 8 mmHg 壁压，随后步壁压为零，原有质量和残差验收通过。

早期尝试中的失败来自编译力重构的 1e-9 dyn 量级舍入比较过严，以及没有先激活 CUDA 13 环境；已修正检查精度并使用项目激活脚本，只继续未完成项，没有重做已通过的检查。测试记录保留在 `results/wall_filling_20261018/{tests,event_tests,pacing_tests}.xml`。

本轮未跑完整周期，未证明 EF 或充盈不足已修复。表面载荷、端口空间离散、流动时间尺度都仍可能影响结果。

另完成原真实左室 MAC128 网格三个新的被动诊断步（`results/lv_wall_filling_short_20261018`）：保持历史的 legacy-balanced / CSR / DST / 分步 BE，关闭 EP/主动张力，从历史第二周期末状态开始，施加恒定 8 mmHg 壁压和 0 mmHg 入口参考。三个 dt=0.1 ms 步均接受，体积 103.329638956 → 103.354073785 mL，累计预算差 6.33e-14 mL，末步质量缺陷 4.37e-10 mL/s，功率相对误差 1.18e-12。三步只有 0.3 ms，不是充盈平衡或周期验证。

该诊断末步原始流体核心表压为 -4.58073 mmHg，而入口参考为零；壁面外力向外拉动，流体在该参考下被吸入。这个值不是规定壁压 8 mmHg 的测量失败，二者在新模型中是不同物理量，也不能将新方案流体表压直接接在旧方案的 PV 曲线上。完整周期比较须明确采用同一种压力定义，不能用壁压替换流体腔压来制造目标曲线。

冷启动首步 52.14 s，随后两步 1.265/1.207 s；这些仅是该被动三步的观测，不是带 EP 的完整周期性能预测。最终检查汇总保存在 `results/wall_filling_20261018/validation_summary.json`，保留各次测试记录。

## 新的两周期命令

从服务器项目根目录运行：

```bash
source scripts/activate_afsei.sh
CUDA_VISIBLE_DEVICES=0 python scripts/run_afsei_cycle.py \
  --preload results/phase24_mac128_balanced_preload_final_v1/preload_result.npz \
  --reference-checkpoint results/phase10_mac128_source_preload_v1/checkpoint.npz \
  --cycles 2 \
  --circulation-config demo/lv_passive_validation/configs/circulation_source_rcr_wall.json \
  --output results/afsei_two_cycles_wall_fill
```

命令保留当前周期入口的原 MAC/IB 选项，不通过本次修改自动切换到近期联立 Stokes 或标准 IB 候选。EP 使用该入口当前默认的 nodal P1/CN/AmgX。输出目录必须尚不存在；续算继续使用原来的 `--resume`，不再指定 `--circulation-config`。正式周期由用户后台运行。

# 真实左心室：被动充盈与 MAC/IB 空间传递验证

2026-10-19：直接壁面升压已补齐配对盖面的压力参考力，并通过静态保持、阶段重启和真实网格短程验证，见 [压力基准闭合](CAP_REFERENCE_20261019.md)。旧配置保留；新配置显式设置 `filling_cap_reference: true`，尚未运行新完整周期。

已完成实验与未通过的验收项见 [2026-10-18 结果报告](RESULTS_20261018.md)。标准路径目前仍为候选，不能将短程完成等同于完整周期已验证。

后续仅增加一步的预算分解、冻结预加载力审计和只读表面积分见 [差额来源诊断](BUDGET_DIAGNOSIS_20261018.md)。

盖面高斯积分已实现并完成三个新增耦合步验证，见 [积分修改与验证](CAP_QUADRATURE_20261018.md)。
新候选 `configs/production_standard_gauss12_candidate.json` 保留分步 BE，显式选用十二点盖面积分；总预算与静态交接仍未达标。

本目录用于排查 `afsei_two_cycles_fast_20261012_232652` 的周期回填不足。
按 IBFE 舒张参考接入的直接腔壁升压、独立供液边界与四项新增检查见 [腔壁加载说明](WALL_FILLING_20261018.md)。
保持原网格、材料、基底弹簧、血液参数和 BE–BE；被动验证不推进 EP/ODE，机械主动张力明确设为零。
`initial_state=final` 保留第二拍末的实际速度、压力和几何；`preload` 使用已验证的预加载几何和静止流体。
这些是单独实验，不将历史检查点伪装成新方法已接受的历史。

## 方法范围

| 环节 | 本次选择 |
|---|---|
| 流体空间 | 原 Cartesian MAC，面速度、单元压力、二阶梯度/散度/黏性 |
| 流体时间 | 保留 BE 黏性与半拉格朗日对流 |
| 流体线性解 | 候选使用 BE 联立 Stokes/FGMRES，多重网格投影预条件；旧 Chorin 分步保留作对照 |
| 心肌时间 | 保留 BE 端点力、旧构型固定交互 stencil、Newton–GMRES |
| 标准 IB | 上游 AFSI_GPU 的一致参考质量、自适应 Gaussian 积分点、Peskin 四点核、功伴随传播/插值 |
| 压力线性解 | 配置明确指定多重网格；spectral 仅供控制对照 |
| 基底端口 | AFSEI 的移动盖面相对流量及成对反力扩展，不声称来自论文 |
| EP | 本验证关闭。旧周期是 consistent P1/BE/Jacobi；项目新 EP 可选 nodal P1/CN/AmgX 或 Jacobi |

`griffith-luo-unified` 不包含 AFSEI 全局压力模态修正；`legacy-balanced` 保留该修正作为对照。
标准路径仍含 AFSEI 的端口约束，不能被称为没有端口扩展的原版论文算例。
按用户指示保留 BE–BE，因此不宣称完整采用论文附录 A 的 CN–AB2/半步时间格式。
`fluid_solver=be-stokes-fgmres` 移植上游 `stokes_fgmres.py` 的缩放鞍点系统和预条件结构，
改用 BE 系数 alpha=dt*mu/rho。适配原 AFSEI 开放盒的压力 Dirichlet / 速度 Neumann 算子，
不复制封闭盒的零法向速度或压力去均值操作。独立检查真实动量残差与散度。
旧对照的 `be-projection` 才采用 Chorin 分步；不会把分步投影改个名字当作联立 Stokes。
对流仍按 BE 路线保留半拉格朗日，而非论文 CN–AB2/xsPPM7；此差异与时间格式一并明确保留。

## 来源与迁移

源项目：服务器 `/mnt/large2/gjh/AFSI_GPU`，HEAD `cb2d30b`；空间传递最新更新 `9c39e32`。
直接复制 `mac/adaptive_transfer.py`、`p2_cuda_transfer.py`、`cuda_ib.py` 与两个 P2 CUDA 源文件。
兼容别名 `AdaptiveP1Transfer` 保留；本真实 LV 继续用四节点 P1 网格。
没有把 P2 固体或者理想 LV 的本构强行换入真实 LV。

核对时 `grid.py`、`compact_transfer.py`、`paper_coupling.py`、`shared_stencil.py`、`adaptive_cell.py`、`open_boundary.py`
与上游完全同 SHA256，因此不重复修改。AFSEI 的 BE 工作区和质量矩阵 CUDA Graph 优化继续保留，避免回退性能。
迁移前服务器备份：`results/afsei_before_gl2017_20261017.tar.gz`。

论文：[Griffith–Luo 2017](https://doi.org/10.1002/cnm.2888)，第 3 节统一弱形式与离散传递。

## 独立运行

仓库根目录激活环境后：

```bash
python demo/run.py demo/lv_passive_validation/configs/standard_preload.json
python demo/run.py demo/lv_passive_validation/configs/standard_hold.json
python demo/run.py demo/lv_passive_validation/configs/legacy_hold.json
python demo/run.py demo/lv_passive_validation/configs/standard_stokes_hold.json
```

每次生成新输出目录。`--steps` 可缩短检查；`--check-config` 只解析配置。
`legacy_hold_long.json` 将入口固定在 8 mmHg，最多继续 400 ms。
该保持实验改变了历史周期的边界时间历程，目的是区分输运时间与静态平衡，不是伪装成相同生理周期的性能比较。

输出 `history.jsonl`、`report.json`、`methods.json`、有效配置、`diagnostic_state.npz`。
后者仅为诊断快照，不是正式周期可直接重启的检查点。
非线性残差、盖面约束、压力边界和功伴随必须通过；独立腔容积预算误差逐步记录，超过 0.1 mL 则停止。
这个 0.1 mL 是诊断停止线，不是生产质量验收线；正式周期原有 1e-5 mL/s 质量验收仍保留，没有放宽。
`equilibrium_reached` 另检查腔压与入口压差 <0.05 mmHg、入口流量绝对值 <1 mL/s、
最大速度 <0.1 cm/s；程序跑完不等于已经达到平衡。
闭合预加载使用更严格的静态保持标准：最大速度 <1e-6 cm/s、最大位移 <1e-9 cm、
容积漂移 <1e-3 mL。它是在检查已加载的固体预加载能否直接交接，不是为所有 IB 网格宣布通用精度标准。

## 正式周期的显式选择

`scripts/run_afsei_cycle.py --spatial-config demo/lv_passive_validation/configs/production_spatial_candidate.json ...`
可为新启动周期选择标准 IB/多重网格/BE 联立 Stokes。该配置目前是候选，必须以验证结果决定能否用于完整周期。
旧检查点续算自动保留原空间传递及压力后端，禁止用新开关覆盖其物理历史。
正式路径仍执行原有严格质量检查；新路径若不满足，失败应被修复或报告，不能用全局修正偷偷重新包装成“标准 IB”。

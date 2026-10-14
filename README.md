# AFSEI

AFSEI：电–流固三场耦合（electrophysiology–fluid–solid interaction）。

独立的固定组织 EP 传播实验见 [Santiago 图 6 / GPB demo](demo/ep_slab/README.md)。
统一外部条件的 [GPB / 原始 ORd 传播对照](demo/ep_slab/CELL_MODEL_COMPARISON.md)
提供三维角部刺激、时间/空间细化和平面波控制；使用已有 nodal FEM + AmgX。
区域、组织参数、刺激、边界、初值和观测在标准 JSON 中声明，直接调用现有
nodal FEM/AmgX 与 GPB 求解器：
`python demo/run.py demo/ep_slab/configs/santiago_gpb.json`。

同一基准上的 [TorchCor 原生 PDE 与 nodal FEM＋AMG 耗时比较](demo/ep_slab/NATIVE_PDE_COMPARISON.md)
使用 CN、共同真残差条件和三档网格，仅测量 PDE。实测同时保留加速与未加速的结果。

当前主线采用原 AFSI_GPU 的规则 MAC 网格与 BE–BE 隐式流固求解器，入口为 `scripts/run_afsei.py`。
EP 新运行默认使用四面体 P1 顶点积分、集中质量和 CN 扩散时间推进，
PyTorch 装配、AmgX PCG-AMG GPU 求解；旧检查点保留原 BE 时间格式。
GPB/Land 使用 CUDA，主动张力取代原自定义波形。配置、安装及实测见
[EP 顶点积分与 AmgX](docs/EP_NODAL_AMGX.md)。旧一致质量/Jacobi 后端可显式选择。
默认配置恢复为原真实 LV 的 128³、13 cm 外箱及已有求解器优化，保留源电–力材料、基底和方向映射。
直接新建运行从未预加载的参考几何开始，也可通过下述 MAC 预加载入口初始化。
第二十五阶段完成压力配对预加载后的主动收缩与原生实际出口流量/RCR 短程验证。
第二十六阶段提供正式四阶段周期入口 `scripts/run_afsei_cycle.py`，支持原子检查点与后台续算；
主动收缩、闭瓣后固定体积 80 ms 卸载、平滑充盈至 8 mmHg 和再次刺激按源规则组织。
该分支使用独立检查点。旧入口仍保留；两个完整周期与局部无穿透仍需实际验证。
后台运行、续算与输出说明见 [四阶段周期驱动](docs/MAC_CARDIAC_CYCLE_PHASE26.md)。
第二十七阶段优化基底算子、预条件器与 GPU 批量求解，保持原隐式方程和容差；同一射血短窗口
从 2.548 降至 2.339 s/步。两个目标步长的实测及显式黏性文献评估见
[MAC 性能与显式流体](docs/MAC_PERFORMANCE_EXPLICIT_PHASE27.md)。显式探针尚未接入正式周期。
第二十八阶段优化切线分批、GPU 检查调度及步内机械切线复用，同一短窗口降至约 2.172 s/步。
原非线性与物理验收阈值保持不变，可用 `--no-reuse-step-tangent` 关闭复用。
已有对照及适用范围见 [本轮性能优化](docs/MAC_PERFORMANCE_PHASE28.md)。
第二十九阶段合并重复 GPU 检查、直接写入流体工作区、就地累加基底载荷，并缓存静态几何规则。
保持现有计算方法，同一短窗口降至约 1.909 s/步，比第二十八阶段耗时减少 12.12%。
具体修改、数值对照及测量范围见 [同步与内存优化](docs/MAC_PERFORMANCE_PHASE29.md)。
第三十阶段补齐质量求解初始化/真实残差 CUDA Graph、FFT/压力 RHS 工作区和 GMRES
旋转融合及 Krylov 存储复用，保持原方程与收敛判据。同一短窗口观测约 1.888 s/步，
较上轮低约 1.1%，样本不足以认定持续加速。验证与适用范围见
[质量图、流体工作区与 GMRES](docs/MAC_PERFORMANCE_PHASE30.md)。
贴体四面体路线保留为已有实验记录；其检查点与 MAC 不兼容。
运行方法与实测性能见 [MAC 主线](docs/MAC_MAINLINE_PHASE9.md)。

第十阶段增加源内膜规定压力的 MAC 预加载：保留 100×5 ms 的 Land 应变历史，
门控保持冻结，用原 MAC 被动试算检查初始漂移。该规定压力预加载与实际流量/RCR 接口分开记录，
详情见 [MAC 预加载与性能](docs/MAC_PRELOAD_PHASE10.md)。

第十一阶段将源 Land 主动力的自动微分切线改为解析切线并编译为 GPU 张量内核。
保留空间积分、伸长反馈、额外主动应力乘子和冻结卸载导数，沿用原 MAC 隐式残差与收敛条件。
实测对照和循环端口的后续工作见 [主动切线优化](docs/MAC_ACTIVE_TANGENT_PHASE11.md)。

第十二阶段建立随基底边缘运动的封面、理想化 MV/AV 区域与原 MAC 上的相对流量/压力功算子。
真实 LV 的三个状态已检查质量缺口和非端口泄流；密封、单向阀与 RCR 尚未施加到主线。
检查入口、VTK 输出与耗时见 [MAC 端口算子](docs/MAC_PORT_OPERATORS_PHASE12.md)。

第十三阶段新增原 MAC 上的局部基底约束投影验证，联合检查散度、相对流量、材料反力和能量。
真实检查点上净泄漏降至约 1e-9 ml/s，但积分点残余泄漏仍需处理；
本组件尚未接入完整隐式流固主线，单向阀与 RCR 仍待完成。
实际额外压力成本及运行方式见 [局部基底密封验证](docs/MAC_LOCAL_SEAL_PHASE13.md)。

补充参考 `LV_hg_coarse_WK.ipynb` 的等容压力求解与 Windkessel 体积关系；
主动张力和 80 ms 卸载仍按电–力源项目。采用范围与 RC/RCR 差异见
[Windkessel 与等容参考核对](docs/WINDKESSEL_NOTEBOOK_REFERENCE.md)。

第十四阶段在原 MAC 流体响应中联立位移与一个腔压未知量，增加可续算的固定体积验证入口。
固定目标按进入该模式时的封闭腔体积保存；源内膜压力载荷由求解腔压替换，保留主动力与卸载实现。
RCR 电容压力仅按假定闭合出口 Q=0 更新，本阶段尚未接入局部密封、实际端口流量或四阶段自动切换。
数学关系、检查点和实测结果见 [MAC 等容验证](docs/MAC_ISOVOLUMIC_PHASE14.md)。

第十五阶段为等容入口增加 GPU DST-II 直接压力求解，保持原 MAC 的零压力外箱离散矩阵，
并复用重复残差与压力单位缩放。同一真实 128³ 检查点的对照从 1.7764 降至 1.0904 s/步，
最终收敛阈值与物理参数一致。基底投影验证从 0.6176 降至 0.07384 s/次，仍作为独立诊断。
新等容运行默认启用加速；检查点保存后端和缩放，普通规定压力运行保留原多重网格。
范围、误差及复现命令见 [MAC 压力加速](docs/MAC_PRESSURE_PERFORMANCE_PHASE15.md)。

第十六阶段把基底弱约束及其反力接入同一个原 MAC 隐式 Newton 步，
联立心肌位移、源内膜等容控制压力和基底面片压差。闭合阶段直接使用完整盖面，
不要求存在 MV/AV 几何；未增加外层完整流固强耦合循环，也未更换规则流体网格。
检查点保存面片压力及其冻结的旧几何，续算时恢复反力；主动张力和 80 ms 卸载不变。
真实 128³ 网格的短程检查中，复用预条件器后约 1.26 s/步，首次初始化约 11.3 s。
面片平均通量与固定体积已验证，但积分点仍有残余滑移，实际端口/RCR 与完整周期继续标记待完成。
数学关系、性能范围及运行命令见 [联合基底反力验证](docs/MAC_COUPLED_CAP_PHASE16.md)。

第十七阶段增加完整内膜与运动基底盖面的只读质量预算，区分原 IB 的体积 L2 速度投影、
表面采样滑移、MAC 散度和运动几何的时间积分误差；不会增加每步默认计算或改变接受阈值。
同一 516 ms 状态，加密基底约束面片从 3 格到 2 格使盖面绝对相对通量从约 0.976 降至 0.683 ml/s，
但内膜表面采样仍约 4.109 ml/s，尚不能据此认定局部无穿透。绝对通量不是净血液损失。
新增少量压力方向的缓存更新，使 2 格方案同窗口排除首步后的耗时从 2.296 降至 1.335 s/步。
原规则 MAC 网格、隐式流固方程、源张力/卸载和默认 3 格面片宽度保持原配置。
实测、判据含义及 VTK 输出见 [完整表面质量预算与面片加密](docs/MAC_MASS_BUDGET_PHASE17.md)。

第十八阶段增加与原 IB 质量矩阵投影速度一致的离散腔体输运，并用精确时间积分核对体积变化和压力功配对。
四个真实检查点的输运/体积变化率差小于 6e-7 ml/s；这证明运动方程和体积测量一致，
不表示直接表面采样滑移已消失，也不把基底材料运动的输运当成 AV/MV 流量。
原等容方程的体积梯度改为数学等价的解析 GPU 表达式，源内膜压力载荷仍只作用于内膜。
真实 128³、0.1 ms 步的短程实测约 1.32 s/步；主要成本仍为隐式 FSI。
核对结果、源压力与虚拟盖面的区别及后续范围见 [IB 输运一致性](docs/MAC_TRANSPORT_PHASE18.md)。

第十九阶段在原 MAC 的冻结几何边界响应中连接实际相对出口流量与三元素 RCR，
联立非出口基底弱闭合条件、出口单向条件及其配对材料反力。
沿用已有理想化端口区域，未添加解剖瓣膜或替换主线网格、Land 张力及卸载。
两个真实 128³ 几何的接口检查通过，暖边界响应约 0.046–0.047 s/步；此时间不含 EP/固体求解。
该组件尚未接入 `run_afsei.py`，开放状态仍有体积/出口质量缺口，不能作为完整三场或生理周期结果。
数学关系、性能范围及接入条件见 [MAC 出口与 RCR 边界响应](docs/MAC_RCR_BOUNDARY_PHASE19.md)。

第二十阶段把实际 AV 相对流量、有限步体积质量关系、单向出口条件及 RCR 放进原 MAC 同一个 Newton 步，
EP/GPB/Land 与运动心肌一同推进，不调用第十九阶段的冻结响应凝聚，也不增加外层完整 FSI 迭代。
检查点原子保存循环状态与盖面反力历史，普通 `run_afsei.py --resume` 正确恢复 RCR 模式。
真实 128³ 网格的关闭与开放验证均零回退，排除首步后的完整计算约 1.32–1.33 s/步。
开放验证使用明确标记的非生理动脉初值探针；源腔压控制量尚不能直接认定为实际血压。
主动张力/80 ms 卸载实现保持源项目，实际阶段事件和局部无穿透仍待验证。
方程、性能、压力对应的限制及续算命令见 [同一步 MAC/EP/RCR 验证](docs/MAC_RCR_COUPLED_PHASE20.md)。

第二十一阶段增加独立的源内膜控制压力/MAC 腔内压力、载荷响应和完整内膜局部通量核对。
真实预加载状态的控制压力约 8 mmHg 而 MAC 压力接近零；517 ms 原生状态分别约 11.98/2.04 mmHg，
当前端口反力也未证明是生理腔压，必须先统一压力与受力再启用完整周期事件。
诊断正确排除开放 AV 通流；该入口不推进检查点，也不增加正式步耗时。
结果、压力交接的后续要求与运行方法见 [压力和界面核对](docs/MAC_PRESSURE_INTERFACE_PHASE21.md)。

第二十二阶段实现去除规定内膜压力及其切线、保留功配对基底反力的原 MAC 隐式交接验证分支。
真实 128³ 预加载试算的位移/流速及均压未通过静态交接，候选状态不发布为正式检查点。
源预加载几何和 Land 历史保持原数据；下一修复需建立与 MAC 离散一致的压力支承预加载平衡。
验证入口、拒绝发布条件和实测结果见 [压力交接验证](docs/MAC_PRESSURE_HANDOFF_PHASE22.md)。

第二十三阶段增加静态压力载荷的离散兼容性检查，区分原 MAC 压力求解器、
源内膜节点力的体积传递、基底闭合以及边界核的积分误差。
真实 128³ 网格的制造静压对照达到 8 mmHg、最大速度约 2.15e-9 cm/s；
现有源材料力仍产生约 2.20 cm/s 的速度，不能只靠压力显示值确认平衡。
局部复合 B 样条仅作为独立边界载荷候选，积分加密后仍有约 0.00362 cm/s 的残余速度，
尚未接入正式 FE/基底传递或源力学；新压力支承预加载继续标记待完成。
检查入口不推进源检查点，原 MAC 网格、隐式框架和源张力/卸载保持原配置。
结果及下一步的联合验证要求见 [静态压力载荷兼容性](docs/MAC_PRESSURE_COMPATIBILITY_PHASE23.md)。

第二十四阶段实现隔离的功配对压力模态传递与闭合基底预加载验证。
源内膜载荷保留独立定义，基底压力材料反力另行组装；从参考几何重算 100×5 ms Land 应变历史。
去除规定内膜压力后，在原 MAC BE 方程中联立位移与基底压力，验证实际腔压和静止性。
修正仅作用于一个全局压力模态，尚未启用正式周期驱动，不代表点位无穿透或完整三场周期。
加载状态先单独保存，静态验收通过才另写带校验和的 `preload_result.npz`；两者均不是普通续算检查点。
数学定义、真实验证结果及性能范围见 [压力配对与重新预加载](docs/MAC_BALANCED_PRELOAD_PHASE24.md)。

第二十五阶段开启新预加载的 EP/GPB/Land，并在同一个原 MAC Newton 步连接实际相对 AV 流量和 RCR。
删除这条新路径的旧规定内膜压力控制项；主动 PK1、伸长反馈、应变历史和 80 ms 卸载实现沿用源项目。
压力传递模态采用沿候选位移精确积分的 P1 体积梯度，修复张力升高后的有限步质量缺口，
配对传递和完整解析切线同时更新；不增加额外腔压控制未知量或外层完整流固求解。
完整盖面关闭与实际出口开放使用独立验证检查点，自动四阶段及两周期验收继续待完成。
数学关系、真实验证结果、性能及续算命令见 [主动收缩与实际流量 RCR](docs/MAC_BALANCED_DYNAMIC_RCR_PHASE25.md)。

```bash
python scripts/run_afsei_mac_balanced.py --preload results/phase24_mac128_balanced_preload_final_v1/preload_result.npz --reference-checkpoint results/phase10_mac128_source_preload_v1/checkpoint.npz --mode rcr --duration-ms 40 --output results/my_balanced_rcr_40ms
```

```bash
python scripts/initialize_afsei_mac_balanced.py --checkpoint results/phase10_mac128_source_preload_v1/checkpoint.npz --trials 3 --output results/my_mac_balanced_preload
```

```bash
python scripts/check_afsei_mac_pressure_compatibility.py --checkpoint results/phase10_mac128_source_preload_v1/checkpoint.npz --levels 3 4 --output results/my_mac_pressure_compatibility
```

```bash
python scripts/check_afsei_mac_pressure_handoff.py --checkpoint results/phase10_mac128_source_preload_v1/checkpoint.npz --steps 3 --output results/my_pressure_handoff_check
```

```bash
python scripts/check_afsei_mac_pressure.py --checkpoint results/phase20_mac128_rcr_coupled_native_v2/checkpoint.npz --output results/my_mac_pressure_audit
```

```bash
python scripts/run_afsei_mac_rcr.py --checkpoint results/phase18_mac128_cap2_analytic_gradient_10steps_v1/checkpoint.npz --steps 10 --output results/my_mac_rcr_coupled
python scripts/run_afsei.py --resume results/my_mac_rcr_coupled/checkpoint.npz --duration-ms 1 --profile --output results/my_mac_rcr_more_1ms
```

```bash
python scripts/check_afsei_mac_rcr_boundary.py --checkpoint results/my_mac_closed_cap_10steps/checkpoint.npz --steps 20 --write-vtk --output results/my_mac_rcr_boundary
```

```bash
python scripts/check_afsei_mac_transport.py --checkpoint results/my_mac_closed_cap_10steps/checkpoint.npz --benchmark-gradients 30 --output results/my_mac_transport
```

```bash
python scripts/check_afsei_mac_mass_budget.py --checkpoint results/my_mac_closed_cap_10steps/checkpoint.npz --levels 2 3 4 --write-vtk --output results/my_mac_mass_budget
```

```bash
python scripts/run_afsei_closed_cap.py --checkpoint results/my_mac_isovolumic_2ms/checkpoint.npz --steps 10 --output results/my_mac_closed_cap_10steps
# 128³ 默认机械步长为 0.1 ms：10 步为 1 ms，不是 10 ms。
# 普通续算入口保留已保存的联合闭合盖面模式：
python scripts/run_afsei.py --resume results/my_mac_closed_cap_10steps/checkpoint.npz --duration-ms 1 --profile --output results/my_mac_closed_cap_more_1ms
```

```bash
python scripts/run_afsei_isovolumic.py --resume results/my_mac_preload_8mmhg/checkpoint.npz --duration-ms 2 --profile --output results/my_mac_isovolumic_2ms
# 普通续算入口也会保留检查点中的等容模式、固定目标、腔压和 RCR 状态
python scripts/run_afsei.py --resume results/my_mac_isovolumic_2ms/checkpoint.npz --duration-ms 1 --output results/my_mac_isovolumic_3ms
```

```bash
python scripts/prepare_afsei_mac_preload.py --output results/my_mac_preload_8mmhg
python scripts/run_afsei.py --resume results/my_mac_preload_8mmhg/checkpoint.npz --duration-ms 10 --profile --output results/my_mac_preloaded_10ms
```

```bash
cd /mnt/large2/gjh/AFSEI
source scripts/activate_afsei.sh
python scripts/run_afsei.py --duration-ms 1 --profile --output results/my_mac_ep_1ms
# 32³ 仅用于快速接入检查
python scripts/run_afsei.py --fluid-shape 32 --duration-ms 0.2 --output results/my_mac_ep_coarse
```

本仓库从 AFSI_GPU 已提交版本初始化，保留 GPU 流固求解器和验证用例。
已实现 PyTorch 单域 PDE、CUDA GPB/Land、积分点主动载荷，并接入隐式流固时间推进。
支持完整物理状态的失败回退、减小步长重试和断点续算。
真实左心室的三场双向短程推进已验证；新增被动预加载与相容初始化，完整心动周期和高质量血液网格仍待验证。
新增固定真实左心室腔体的理想化基底端口/RCR 子问题，验证实际流量、质量与离散能量守恒。
已进一步对齐源项目的材料、基底参数及积分规则，并用重新生成的原 UFL 组装力与切线核对。
详细记录见 [第一阶段](docs/EP_LAND_PHASE1.md)、[第二阶段](docs/EP_FSI_PHASE2.md)
和 [源力学对齐](docs/SOURCE_MECHANICS_PHASE3.md)。
端口几何、数学关系和当前边界限制见 [第四阶段](docs/CAVITY_PORTS_PHASE4.md)。
第五阶段增加拟合移动血液域、几何守恒、独立集总阀门和 EP–FSI 几何单向传递。
该阶段只传递几何；详见 [移动腔体阶段](docs/MOVING_CAVITY_PHASE5.md)。
第六阶段用拟合血液反力替换原外箱流体，联立心肌运动与双阀关闭时的体积/压力约束。
新三场驱动连接实际端口流量、RCR 和首次真实 AV 关闭后的 80 ms 张力卸载。
数学模型与短算限制见 [双向三场阶段](docs/TWO_WAY_PHASE6.md)。
第七阶段增加闭合界面 8 mmHg 被动预加载；平衡条件、初始循环状态及限制见 [预加载阶段](docs/PRELOAD_PHASE7.md)。
第八阶段独立核对原纯固体预加载方程，保留源程序 100 个 5 ms 加载步产生的 Land 应变历史。
压力载荷的差别、核对结果和下一步网格工作见 [预加载源对齐](docs/PRELOAD_SOURCE_PHASE8.md)。

## 来源

- AFSI_GPU 基线：`f4f88204c5a494265611237ba6d3ce317bc179bd`
- 电–力参考：[loveIroha/test_ele_soild_heart](https://github.com/loveIroha/test_ele_soild_heart/tree/codex/wk-closed-volume-fill8-20260924)
- 电–力参考提交：`5bdde73c6be10cefadb005a0fcec5fdb6521f336`
- 原 AFSI 说明：[固定版本说明](https://github.com/loveIroha/AFSI_GPU/blob/f4f88204c5a494265611237ba6d3ce317bc179bd/README.md)，副本保存在 `docs/AFSI_GPU_BASELINE_README.md`。

Python 分发包名称为 `afsei`，保留 `afsi_torch` 导入路径。

## 服务器环境

项目：`/mnt/large2/gjh/AFSEI`。
独立 Conda 环境：`/mnt/large2/gjh/conda-envs/afsei`，由原 `afsi-torch` 环境克隆。
原 AFSI_GPU 目录和环境均保留；AFSEI 可编辑安装指向本仓库。

```bash
cd /mnt/large2/gjh/AFSEI
source scripts/activate_afsei.sh
python scripts/check_afsei_environment.py
```

默认使用物理 GPU 1，进程内设备为 `cuda:0`。激活前可设置 `AFSEI_GPU=0`。
激活脚本配置独立编译缓存及 CUDA 工具链，不修改全局 shell 启动文件。
默认编译 RTX 4090 的 `8.9` 架构，可用 `AFSEI_CUDA_ARCH_LIST`、`AFSEI_BUILD_JOBS` 覆盖。
环境记录见 [environment/README.md](environment/README.md)。

## 已保留的四面体实验与早期检查

以下四面体入口用于复现既有阶段结果；当前主线使用上面的 MAC 入口。
四面体实验可从 8 mmHg 被动平衡开始三场短算：

```bash
python scripts/prepare_afsei_preload.py --write-vtk --output results/my_preload_8mmhg
python scripts/run_afsei_two_way.py --resume results/my_preload_8mmhg/checkpoint.npz --duration-ms 0.2 --write-vtk --output results/my_preloaded_two_way_02ms
```

预加载保留原参考几何，并将静态腔内压力和初始心房压力都设为 8 mmHg，固体和血液初始速度为零。
动脉电容初始 85 mmHg；它会在物理时间推进后向外周释放流量，因此这不是整个循环的稳态。
仅完成并核对平衡的三场检查点可续算；加载期间冻结电位、GPB/Land 门控和循环状态。
默认预加载按源程序在 0–500 ms 的线性压力斜坡采样，每 5 ms 更新伸长和 Land 的 `Zs/Zw`。
电位、GPB 和 Land 门控状态在此期间冻结，首次刺激时保留加载得到的 `Zs/Zw`。
`--history-mode static-zero-z` 可复现第七阶段的静态零历史对照。
旧检查点保留原历史；要使用新的预加载历史，需要新建预加载，不能通过续算旧文件补回。

第六阶段的双向三场入口如下。真实网格首个 `0.2 ms` 约需两分钟；先做短算。
默认从未预充盈的参考几何开始，这不是生理心动周期。

```bash
python scripts/run_afsei_two_way.py --duration-ms 0.2 --write-vtk --output results/my_two_way_02ms
python scripts/run_afsei_two_way.py --resume results/my_two_way_02ms/checkpoint.npz --duration-ms 0.2 --output results/my_two_way_04ms
```

新驱动不创建外箱流体；使用 `26889` 个心肌节点和 `8584` 个血液单元的验证网格。
固体采用 P1 集总惯性、后向 Euler，默认密度 `1 g/cm³`；源项目材料和主动 PK1 载荷保留。
只有新驱动自己的三场检查点可以续算；早期外箱/单向检查点不能直接用于它。
结果包含 `summary.json`、`history.csv`、三场 `checkpoint.npz` 和界面反力文件。
`--write-vtk` 输出心肌电位/运动及血液压力/速度。

下面保留各早期阶段的运行入口，供分模块检查。

```bash
# 第一阶段组件检查，不推进流固运动
python scripts/check_afsei_ep.py --steps 40

# 第二阶段：EP + GPB/Land + 隐式流固推进，首次刺激后推进 2 ms
python scripts/run_afsei.py --duration-ms 2 --output results/my_ep_fsi_2ms

# 从已接受状态继续推进 0.2 ms
python scripts/run_afsei.py --resume results/my_ep_fsi_2ms/checkpoint.npz --duration-ms 0.2

# 相关回归检查
python -m pytest -q tests/test_source_mechanics.py tests/test_afsei_driver.py tests/test_ep_*.py tests/test_paper_lv.py
```

新建运行需要未使用的输出目录；重启的 `--duration-ms` 是追加时间。
当前 MAC 主入口默认流体网格为 `128³`，可用 `--fluid-shape 32` 做粗网格接入检查；EP 步长 `0.05 ms`，ODE 子步 `0.002 ms`，
流固步长 `0.1 ms`。失败可退至一个 EP 步长；所有物理状态只在完整步骤成功后发布。
结果包括 `history.csv`、`checkpoint.npz`、`report.json`。

EP 默认采用经实际串行 DOLFIN 核对的全局 CG1 方向与刺激映射，
因此电位结果会与第一阶段 DG0/内膜节点检查不同。
原项目的力学形式直接绑定单元纤维表达式，DG0 与它相符；第三阶段通过实际
原 UFL 组装核对了被动/主动载荷、完整切线及伸长投影。
此前第二阶段文档提出的“力学应改为全局 CG1”推断已纠正。

新建 CLI 运行默认 `--mechanics source`，使用源项目材料与基底配置；
`--mechanics inherited` 可显式保留 AFSI 配置。
Python API 使用 `AFSEIConfig.source(fsi=...)` 选择源配置。
旧检查点自动保留其原始参数和积分规则，续算不会改用新材料。
因此对照新配置时请新建输出目录，不要从旧材料的检查点开始。

服务器上已生成力学参考数据，可直接检查：

```bash
python scripts/check_afsei_mechanics.py
```

新检出仓库需要先运行 `bash scripts/prepare_afsei_mechanics_reference.sh`，
此步骤使用现有独立 FEniCS 环境，不向 AFSEI 环境安装 DOLFIN。

`run_afsei.py` 直接新建运行时从参考几何的首次刺激时刻开始；
第十阶段 `prepare_afsei_mac_preload.py` 可提供源内膜规定压力的 MAC 预加载检查点。
MAC 外箱仍为零压力开放边界，AV/MV/Windkessel 物理端口待接入。
报告将 `ep_fsi_time_integration_implemented` 与完整三场周期状态分别列出。

## 固定腔体端口与 Windkessel 检查

现有网格只有左心室，没有 AV/MV 几何。第四阶段在基底设置可配置的理想化端口，
封闭其他边界，并在血液腔体内推进非定常 Stokes/RCR；不包含电生理、运动心肌或预充盈。
原 `run_afsei.py` 的运动流固驱动仍未连接这些端口。两个检查点格式相互独立。

```bash
python scripts/run_afsei_cavity_ports.py --steps 20 --write-vtk --output results/my_cavity_ports_20ms
python scripts/run_afsei_cavity_ports.py --resume results/my_cavity_ports_20ms/checkpoint.npz --steps 5 --output results/my_cavity_ports_25ms
python -m pytest -q tests/test_cavity_ports.py
```

默认时间步为 1 ms，间距 0.18 cm，心房压力 8.5 mmHg，初始 RCR 电容压力 8 mmHg。
这些压力用于端口测试，**不是已经完成的 8 mmHg 心肌预充盈**。
端口中心及半径由 `--mv-center-cm`、`--av-center-cm`、`--mv-radius-cm`、`--av-radius-cm` 调整，
坐标采用以基底平均位置为原点的局部平面。默认参数为验证配置，不代表患者瓣口。
结果包含 `summary.json`、`history.csv`、数值检查点及可选 VTK/端口图。
`fluid_ports_connected=true` 只表示固定流体子问题；完整三场标记仍为 `false`。

## 移动腔体与 EP 几何单向接入

```bash
# 真实 LV 上的小幅规定运动：收缩排出、扩张流入；没有 EP
python scripts/run_afsei_moving_cavity.py --duration-ms 2 --write-vtk --output results/my_moving_cavity_2ms

# 已有 EP–FSI 检查点的几何驱动移动血液域；8 mmHg 仅为端口检查配置
python scripts/run_afsei_moving_cavity.py --ep-checkpoint results/my_source_mechanics_2ms/checkpoint.npz --initial-pc-mmhg 8 --duration-ms 0.2 --output results/my_one_way_ep_02ms

# 单个原子检查点同时恢复 EP/GPB/Land/原流固状态和移动血液/RCR 状态
python scripts/run_afsei_moving_cavity.py --resume results/my_one_way_ep_02ms/checkpoint.npz --duration-ms 0.2 --output results/my_one_way_ep_04ms
```

规定运动模式默认初始外周电容压力为 85 mmHg，时间步 0.1 ms；运动振幅仅 1e-5，
用于接口检查，不代表真实心肌收缩或射血分数。`--ep` 可从源力学配置新建单向运行。
移动端口采用相对流量，逐步检查 `dV/dt + qMV + qAV = 0`，不再强制进出口相等。
接受的射血后 AV 关闭事件可连接到已有 80 ms 张力卸载，但短程 EP 检查未覆盖实际闭瓣。
新血液域采用保留原内膜的锥形验证网格，尚未完成质量网格/空间收敛验证。
原 EP–FSI 外箱流体仍参与固体运动，新计算的血液反力只导出；
`two_way_interface_feedback_implemented=false`，完整三场状态仍为 `false`。

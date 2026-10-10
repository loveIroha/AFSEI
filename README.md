# AFSEI

AFSEI：电–流固三场耦合（electrophysiology–fluid–solid interaction）。

当前主线采用原 AFSI_GPU 的规则 MAC 网格与 BE–BE 隐式流固求解器，入口为 `scripts/run_afsei.py`。
EP 单域 PDE 使用 PyTorch GPU，GPB/Land 使用 CUDA；主动张力取代原自定义波形。
默认配置恢复为原真实 LV 的 128³、13 cm 外箱及已有求解器优化，保留源电–力材料、基底和方向映射。
直接新建运行从未预加载的参考几何开始，也可通过下述 MAC 预加载入口初始化。
MAC 的物理瓣口与流量驱动 RCR 待接入。
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

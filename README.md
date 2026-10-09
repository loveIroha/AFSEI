# AFSEI

AFSEI：电–流固三场耦合（electrophysiology–fluid–solid interaction）。

本仓库从 AFSI_GPU 已提交版本初始化，保留 GPU 流固求解器和验证用例。
已实现 PyTorch 单域 PDE、CUDA GPB/Land、积分点主动载荷，并接入隐式流固时间推进。
支持完整物理状态的失败回退、减小步长重试和断点续算。
真实左心室的短程推进已验证；完整心动周期、预充盈和运动心肌的 Windkessel 端口连接仍待实现。
新增固定真实左心室腔体的理想化基底端口/RCR 子问题，验证实际流量、质量与离散能量守恒。
已进一步对齐源项目的材料、基底参数及积分规则，并用重新生成的原 UFL 组装力与切线核对。
详细记录见 [第一阶段](docs/EP_LAND_PHASE1.md)、[第二阶段](docs/EP_FSI_PHASE2.md)
和 [源力学对齐](docs/SOURCE_MECHANICS_PHASE3.md)。
端口几何、数学关系和当前边界限制见 [第四阶段](docs/CAVITY_PORTS_PHASE4.md)。
第五阶段增加拟合移动血液域、几何守恒、独立集总阀门和 EP–FSI 几何单向传递。
界面反力已计算并核对虚功，尚未反馈到固体；详见 [移动腔体阶段](docs/MOVING_CAVITY_PHASE5.md)。

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

## 检查和短程推进

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
默认流体网格为 `32³`，用于接入检查；EP 步长 `0.05 ms`，ODE 子步 `0.002 ms`，
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

当前运行从参考几何的首次刺激时刻开始，**没有完成 8 mmHg 预充盈，也没有连接
AV/MV/Windkessel 物理端口**。外箱仍为零压力开放边界。
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

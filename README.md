# AFSEI

AFSEI：电–流固三场耦合（electrophysiology–fluid–solid interaction）。

本仓库从 AFSI_GPU 已提交版本初始化，保留 GPU 流固求解器和验证用例。
已实现 PyTorch 单域 PDE、CUDA GPB/Land、积分点主动载荷，并接入隐式流固时间推进。
支持完整物理状态的失败回退、减小步长重试和断点续算。
真实左心室的短程推进已验证；完整心动周期、预充盈和 Windkessel 物理端口仍待实现。
详细记录见 [第一阶段](docs/EP_LAND_PHASE1.md) 和 [第二阶段](docs/EP_FSI_PHASE2.md)。

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
python -m pytest -q tests/test_afsei_driver.py tests/test_ep_*.py tests/test_paper_lv.py
```

新建运行需要未使用的输出目录；重启的 `--duration-ms` 是追加时间。
默认流体网格为 `32³`，用于接入检查；EP 步长 `0.05 ms`，ODE 子步 `0.002 ms`，
流固步长 `0.1 ms`。失败可退至一个 EP 步长；所有物理状态只在完整步骤成功后发布。
结果包括 `history.csv`、`checkpoint.npz`、`report.json`。

第二阶段默认采用经过实际串行 DOLFIN 核对的 EP CG1 方向与刺激映射，
因此电位结果会与第一阶段 DG0/内膜节点检查不同。主动应力公式保持源模型的
Land 与长度依赖形式；力学纤维离散仍为 AFSI DG0，尚未完成源 CG1 主动应力积分对齐。

当前运行从参考几何的首次刺激时刻开始，**没有完成 8 mmHg 预充盈，也没有连接
AV/MV/Windkessel 物理端口**。外箱仍为零压力开放边界。
报告将 `ep_fsi_time_integration_implemented` 与完整三场周期状态分别列出。

# AFSEI

AFSEI：电–流固三场耦合（electrophysiology–fluid–solid interaction）。

本仓库从 AFSI_GPU 的已提交版本初始化，保留其历史、GPU 流固求解器与验证用例。
当前阶段完成项目和环境准备；EP–GPB–Land 与 Windkessel–FSI 接入尚未实现。

## 来源

- AFSI_GPU 基线：`f4f88204c5a494265611237ba6d3ce317bc179bd`
- 电–力参考：`loveIroha/test_ele_soild_heart`，分支 `codex/wk-closed-volume-fill8-20260924`
- 电–力参考提交：`5bdde73c6be10cefadb005a0fcec5fdb6521f336`
- 原 AFSI 说明：[固定版本说明](https://github.com/loveIroha/AFSI_GPU/blob/f4f88204c5a494265611237ba6d3ce317bc179bd/README.md)；另保存未改动的说明副本 `docs/AFSI_GPU_BASELINE_README.md`。

Python 分发包名称为 `afsei`；现阶段保留 `afsi_torch` 导入路径，避免在模型接入前改动已有求解器。

## 服务器环境

项目：`/mnt/large2/gjh/AFSEI`

独立 Conda 环境：`/mnt/large2/gjh/conda-envs/afsei`，由现有 `afsi-torch` 环境克隆。
原 AFSI_GPU 目录和原环境均保留。AFSEI 的可编辑安装指向本仓库。

```bash
cd /mnt/large2/gjh/AFSEI
source scripts/activate_afsei.sh
python scripts/check_afsei_environment.py
```

默认使用物理 GPU 1，可在激活前设置 `AFSEI_GPU=0`。进程内设备编号为 `cuda:0`。
激活脚本设置 CUDA 工具链和独立编译缓存，不修改全局 shell 启动文件。
默认仅编译 RTX 4090 的 `8.9` 架构；可用 `AFSEI_CUDA_ARCH_LIST` 和 `AFSEI_BUILD_JOBS` 覆盖编译设置。
环境版本清单和验收记录位于 [environment/README.md](environment/README.md)。

## 后续接入约束

单域 PDE 使用 PyTorch；GPB/Land ODE 可采用 CUDA 内核。
先验证电生理、空间主动应力和一致切线，再建立瓣膜通道及 Windkessel–流体边界耦合。
新增代码不得把尚未实现或未验证的三场耦合描述为已完成。

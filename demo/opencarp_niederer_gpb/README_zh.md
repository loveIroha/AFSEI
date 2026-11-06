# openCARP 三维 GPB benchmark

采用服务器原有的官方 `03E_study_resolution`（Niederer N-version benchmark）脚本和参数文件，建立独立运行副本。原文件一并保留。该实验是替换为 GPB 的 benchmark 变体，不能直接作为原文 ten Tusscher 模型的数值复现。

## 配置

| 项目 | 设置 |
|---|---|
| 几何 | 20 × 7 × 3 mm；坐标起点 (0,0,0) |
| 名义网格间距 | 0.0625 mm = 62.5 μm；原生四面体网格 |
| 纤维 | X 方向 |
| 边界 | 全部组织边界为齐次零通量 |
| 细胞 | 内置 Grandi：GPB 2010 心室模型；默认 EPI 类型 |
| 初始状态 | Grandi 自带初值；不读取原 ten Tusscher 的 singlecell.sv；未额外预平衡 |
| 刺激区域 | [0,1.5] × [0,1.5] × [0,1.5] mm |
| 刺激 | 35.71 μA/cm²，0–2 ms |
| 表面积体积比 | 0.14 μm⁻¹，标准算例原值 |
| 内/外电导率 | 纵向 0.17/0.62 S/m；横向与法向 0.019/0.24 S/m |
| PDE | 单域；theta=0.5；一致质量矩阵 |
| 时间步、时长 | 20 μs、50 ms；标准算例默认值 |
| 输出 | 原生 Vm IGB（1 ms 间隔）、0 mV 阈值 LAT、对角线激活图、完整参数及日志 |
| 执行 | openCARP v20.0，FP64，16 个 CPU MPI 进程，每进程一个线程 |

仅改动目标网格间距和细胞模型，并取消与 GPB 不兼容的 ten Tusscher 初始状态文件。求解器选项来自原 carputils。该运行不是与 AFSEI 的最终同条件性能对照，后续比较需统一时间步、初态和实际网格连接。

## 环境与目录

- Python 环境：`/mnt/large2/gjh/venvs/opencarp`
- 官方便携运行包：`/mnt/large2/gjh/opencarp-runtime/v20.0`
- 算例：`/mnt/large2/gjh/AFSEI/demo/opencarp_niederer_gpb`
- 结果：`/mnt/large2/gjh/AFSEI/results/opencarp_gpb_h0p0625_20261030`

激活环境：

```bash
source /mnt/large2/gjh/opencarp-runtime/v20.0/activate.sh
```

重新运行时采用新的输出目录，避免覆盖已有结果：

```bash
cd /mnt/large2/gjh/AFSEI/demo/opencarp_niederer_gpb
python run.py --dx 62.5 --np 16 --ID /mnt/large2/gjh/AFSEI/results/opencarp_gpb_h0p0625_repeat
```

后台任务状态位于结果目录的 `run_status.json`，进展见 `run.log`；实际求解文件保存在 `simulation_v2/`，`simulation/` 是首次环境兼容性失败的空目录。`mesh_validation.json` 记录实际节点数、单元数、坐标范围和间距验证。carputils 精确锁定为 v20.0 对应的提交 `49bc2a4eabc5efc019aab174142ddf89fda18812`。

官方模型下载及在线编译入口：https://opencarp.org/download/ionic-model-compilation

官方 benchmark：https://opencarp.org/documentation/examples/02_ep_tissue/03g_study_resolution

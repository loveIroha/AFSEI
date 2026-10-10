# MAC 主线恢复与 EP 性能分解

主入口为 `scripts/run_afsei.py`，继续使用原 `BEIBStepper`、MAC 外箱和原 BE–BE 残差。
每个时间步先推进 EP/GPB/Land，再求解隐式流固，并在接受后更新 Land 伸长历史。
流固部分保留原有隐式反馈；电生理与力学采用分裂推进。

## 配置

`configs/afsei_mac.json` 的求解器参数来自固定 AFSI_GPU 基线中的
`demo/real_lv_fsi/configs/active_cuda.json`：128³、13 cm 外箱、Anderson–Newton、
Graph Helmholtz/压力/质量求解、cached-hash/CUDA CSR IB 响应、原容差和积分设置。
输出频率调整用于短程检查；移除激活波形配置，附加规定压力设为零。
源电–力材料和基底配置仍通过 `AFSEIConfig.source` 设置。
`EPDrivenPaperSolid` 只使用 Land 主动力，流体反馈由原 MAC/IB 框架提供。

```bash
cd /mnt/large2/gjh/AFSEI
source scripts/activate_afsei.sh
python scripts/run_afsei.py --duration-ms 1 --profile --output results/my_mac_ep_1ms
python scripts/run_afsei.py --resume results/my_mac_ep_1ms/checkpoint.npz --duration-ms 1 --profile
```

可用 `--fsi-config` 加载其他原 PaperLV JSON，用 `--fluid-shape`、
`--mechanical-dt-ms`、`--nonlinear-solver` 覆盖新建运行。
默认 EP 0.05 ms、ODE 0.002 ms、流固 0.1 ms，未更改数值步长和接受条件。
旧 MAC 检查点保留全部已保存设置，续算不会自动改成新预设。
四面体三场预加载检查点不能在此入口续算；不能只复制位移后宣称获得相容 MAC 初态。

## 性能记录

`--profile` 用 GPU 同步测量 GPB/Land、单域 PDE、载荷 staging、隐式流固、应变反馈。
计时覆盖失败后重试；流固计时包含主动切线和原求解器内部工作。
Python 调度仍存在，GPU 并行不代表所有过程都能同时执行。
profiling 会增加同步开销，正式运行可去掉该选项。
报告区分设置/初始化、第一步及后续步；第一步可能包含懒编译、图捕获和缓存建立。
去掉第一步也不保证已经充分热身。每步时间不包含输出诊断及检查点写盘。
未启用 profiling 时的每步时间为主机调用耗时，不额外强制 GPU 同步。

`history.csv` 记录每步流体求解次数、累计压力多重网格循环、Helmholtz sweeps、
质量求解、切线组装和 Jacobian 操作。这里的代数计数覆盖最终接受试算；
若时间步回退，失败试算的阶段耗时包含在计时中，代数计数不包含失败试算。
`report.json` 的 performance 只统计本次调用，物理进度仍保留续算累计值。

## 当前物理限制及后续工作

本阶段验证原 MAC 框架中的 EP 主动载荷与求解器兼容性。
初态仍是首次刺激时刻的未预加载参考几何，外箱边界为原开放零压力。
实际 AV/MV 端口、闭瓣密封、流量驱动 Windkessel 和 MAC 相容的 8 mmHg 预加载尚待接入。
80 ms 张力卸载逻辑已存在，但必须由真实流动闭瓣事件触发；此入口尚无该事件。
源预加载应变历史数学及独立验证结果可复用，已有四面体平衡状态不能直接当作 MAC 平衡状态。
短程初始激活耗时不能直接代表完整射血阶段或两个心动周期的运行时间。

后续在原 MAC 框架内设计端口流量和压力接口，并验证闭腔阶段的体积、质量及功率。
在接口完成前，保留 `three_field_integration_implemented=false` 和
`windkessel_ports_connected=false`，避免把组件推进误称为完整生理循环。

## 实际 GPU 验证（2026-10-10）

RTX 4090、真实 LV 26889 节点/135430 心肌四面体、MAC128³。
运行目录为 `results/phase9_mac128_ep_1ms_v1` 及
`results/phase9_mac128_ep_2ms_v1`（第二段从第一段检查点续算）。
两段合计推进 2 ms，20 个流固步、40 个 EP 步、1000 个 ODE 子步，无步长回退。
所有步的真实非线性残差均满足原容差，detF 保持为正；最大 divergence L2
为 1.44e-14，最大离散界面功率差为 2.26e-14。
每步 3 次非线性迭代、4 次流体求解，通常累计 42 个压力多重网格循环。

| 计时 | 第一段 | 续算段 |
| --- | ---: | ---: |
| 初始化/加载检查点 | 10.42 s | 4.57 s |
| 首个时间步 | 22.76 s | 4.89 s |
| 其后 9 步平均 | 0.447 s/步 | 0.454 s/步 |

续算段去掉首步后的平均分解：

| 阶段 | 每 0.1 ms 物理步的耗时 |
| --- | ---: |
| CUDA GPB/Land | 0.0699 s |
| PyTorch 单域 PDE（两步） | 0.0256 s |
| 主动载荷 staging | 0.0054 s |
| 原 MAC 隐式流固 | 0.3417 s |
| 应变历史反馈 | 0.0107 s |

计时启用 GPU 同步；步骤总耗时包含部分未单列的调度与检查。
激活仍很早，最终 Ta 最大仅 0.21221 dyn/cm²，且没有预加载和物理循环接口。
本次未触发 Newton 切线组装；射血阶段可能需要更多迭代及主动切线计算。
因此不能将 0.45 s/步外推为两个完整心动周期的可靠耗时，
也不能将本次结果与已预加载的四面体算例作为相同物理问题的速度比较。

原 MAC 耦合、后向 Euler、多重网格、IB 组装/传递和 PaperLV 配置模块
与固定上游提交逐字节一致。31 项相关回归用例全部通过，
覆盖 GPU ODE、单域 PDE、源力学、时间推进、回退、检查点和 profiling 数值一致性。
生成的输出、验证 JSON 和 GPU 缓存保留在服务器，不提交到仓库。

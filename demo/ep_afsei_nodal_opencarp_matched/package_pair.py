from pathlib import Path
import json,shutil,tarfile,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path('/mnt/large2/gjh/AFSEI');pair=ROOT/'results/ep_euler_fem_pair_20261031';dest=pair/'comparison'
assert json.loads((pair/'postprocessing_completed.json').read_text())['status']=='completed'
r=json.loads((dest/'report.json').read_text());cases=r['runs'];c=cases['Classic'];n=cases['Nodal'];oc=r['openCARP']
m=r['comparison'];p=r['performance'];control=r['common_PDE_control'];audits={}
for name,folder in [('Classic','torchcor_cg'),('Nodal','nodal_afsei')]:
    path=pair/folder;entries=list(ET.parse(path/'visualization/ep.pvd').iter('DataSet'))
    assert len(entries)==51 and all((path/'visualization'/e.attrib['file']).stat().st_size>7100000 for e in entries)
    assert float(entries[0].attrib['timestep'])==0 and float(entries[-1].attrib['timestep'])==.05
    state=np.load(path/'final_gpb_state.npy',mmap_mode='r');assert state.shape==(54,1777377) and state.dtype==np.float64
    audits[name]=dict(frames=51,first_time_s=0,last_time_s=.05,final_cell_state_shape=list(state.shape),files_present=True)
    for file in ['report.json','configuration.json','resolved_configuration.json']:
        shutil.copy2(path/file,dest/(folder+'_'+file))
for file in ['preflight.json','common_pde_control.json']:shutil.copy2(pair/file,dest/file)
diff_c=m['Classic_vs_openCARP'];diff_n=m['Nodal_vs_openCARP'];diff_pair=m['Nodal_vs_Classic']
speedtext=(f"nodal 的完整轨迹 PDE 较快，经典/nodal 时间比 {p['classic_over_nodal_PDE_time_ratio']:.4f}" if p['classic_over_nodal_PDE_time_ratio']>1
    else f"经典 FEM 的完整轨迹 PDE 较快，nodal/经典时间比 {1/p['classic_over_nodal_PDE_time_ratio']:.4f}")
text=f'''# 同一 Euler ODE：AFSEI nodal 与 TorchCor 经典 CG FEM 对比

两组新的完整 0–50 ms 三维实验已在同一 RTX4090 GPU0 顺序完成。GPU1 上的既有模拟未被中断。已有 openCARP 为参考，未重新运行。

## 对齐条件和方法

20×7×3 mm，同一张 0.0625 mm 原生 openCARP 网格：1,777,377 节点、8,601,600 个四面体；相同节点顺序和连接，纤维 X，全部边界零通量。角部 [0,1.5]³ mm 施加 35.71 µA/cm²、0–2 ms 刺激。完整 Grandi GPB 参数与初态相同，Vm 初始 −80.9763 mV，无预平衡。

两组共用同一个 FP64 CUDA Euler 内核：1 µs ODE 步长，每个 20 µs EP 步有 20 子步，冻结 Vm、子步末电流平均。扩散均为 CN。细胞模型在两种 FEM 中都逐节点计算，未给经典 FEM 换用其他 ODE、细胞模型或初值。

经典组：真正调用固定 TorchCor 提交 `{r['torchcor']['commit']}` 的未修改 CN 右端语句及原生 Jacobi-CG；空间方法是连续 P1-Galerkin，一致质量矩阵。全局矩阵复用同网格解析 P1 缓存，抽样 1024 单元与原生 Matrices3D 核对，质量/扩散相对差分别为 {r['preflight']['relative_mass_gap']:.3e}/{r['preflight']['relative_stiffness_gap']:.3e}；未把装配时间计入 PDE 性能。

nodal 组：AFSEI 顶点积分、集中质量、BufferedRHS、GraphJacobiPCG，无 AMG。两组满足共同的物理真实残差阈值 max(1e-13,1e-10||b||)。TorchCor 内部停止范数不同，采用严格范数上界换算，并逐步核对真实残差，没有额外 0.25 安全系数人为增加其精度要求。经典组早期 report 的计时范围文字曾沿用 nodal 描述，后处理仅更正此标签并保留原文供审计，未改动数值或耗时。

## 完整结果

| 指标 | openCARP 参考 | TorchCor 经典 FEM＋Euler | AFSEI nodal＋Euler |
|---|---:|---:|---:|
| 最远角激活时间 ms | {oc['far_corner_activation_ms']:.6f} | {c['far_corner_activation_ms']:.6f} | {n['far_corner_activation_ms']:.6f} |
| 对 openCARP 最远角相对差 | 0 | {diff_c['far_corner_relative_delta_percent']:+.5f}% | {diff_n['far_corner_relative_delta_percent']:+.5f}% |
| 对 openCARP 全节点激活 MAE ms | 0 | {diff_c['activation_MAE_ms']:.6f} | {diff_n['activation_MAE_ms']:.6f} |
| 电模块/PDE 平均 ms/EP步 | {oc['electrics_compute_ms_per_ep_step']:.3f} | {c['pde_mean_ms']:.3f} | {n['pde_mean_ms']:.3f} |
| PDE 总 s | {oc['physics_compute_s']['Electrics']:.3f} | {c['pde_s']:.3f} | {n['pde_s']:.3f} |
| ODE 总 s | {oc['physics_compute_s']['Myocard Ionics']:.3f} | {c['ode_s']:.3f} | {n['ode_s']:.3f} |
| PDE＋ODE 总 s | {sum(oc['physics_compute_s'].values()):.3f} | {c['pde_s']+c['ode_s']:.3f} | {n['pde_s']+n['ode_s']:.3f} |
| 实际循环墙钟 s | {oc['whole_job_wall_s']:.3f} | {c['wall_s']:.3f} | {n['wall_s']:.3f} |
| 平均线性迭代数 | 原生设置 | {c['mean_iterations']:.3f} | {n['mean_iterations']:.3f} |

{speedtext}。两种 GPU FEM 总体计算耗时比（经典/nodal）为 {p['classic_over_nodal_compute_time_ratio']:.5f}；ODE 占比很高，整体差异应与 PDE 差异分别理解。

经典与 nodal 的全节点激活 MAE {diff_pair['activation_MAE_ms']:.6f} ms，最大差异 {diff_pair['activation_max_abs_delta_ms']:.6f} ms；nodal 最远角比经典晚 {diff_pair['far_corner_delta_ms']:.6f} ms。对 openCARP 的 5% 最远角范围：经典{'满足' if diff_c['far_corner_within5percent'] else '不满足'}，nodal{'满足' if diff_n['far_corner_within5percent'] else '不满足'}。

两组物理参数、细胞 ODE、EP 时间步和分裂源项一致，经典/顶点积分的区别主要体现为质量矩阵及源项积分；恒定方向和电导下的扩散 K 相同。线性误差远小于传播差异，不能把该传播差距归因于是否逐节点计算 GPB 或 SDIRK。与 openCARP 仍存在原生 RL/分裂格式、门变量精度和内部线性标准的差别，因此它是交叉验证参考，不是解析精确解。

## 固定同输入 PDE 控制：离散与实现分开

此小测试复用同一个 15 ms 电位快照，给三者相同的 I=0 与规定角部源项，只测线性 PDE，不再求解 EP 轨迹。这是代数输入控制，不代表 15 ms 的真实刺激和离子电流。

| 方法 | 同输入 PDE 中位 ms | 迭代数 |
|---|---:|---|
| 经典 FEM＋TorchCor 原生 CG | {control['timings']['consistent_torchcor']['median_ms']:.4f} | {control['timings']['consistent_torchcor']['iterations']} |
| 经典 FEM＋AFSEI 优化 CG | {control['timings']['consistent_afsei']['median_ms']:.4f} | {control['timings']['consistent_afsei']['iterations']} |
| nodal FEM＋同一 AFSEI 优化 CG | {control['timings']['nodal_afsei']['median_ms']:.4f} | {control['timings']['nodal_afsei']['iterations']} |

共享 AFSEI 核时，经典/nodal 时间比为 {control['shared_AFSEI_backend_standard_over_nodal_time_ratio']:.5f}；同一致质量方程下，TorchCor/AFSEI 执行时间比为 {control['same_consistent_system_native_TorchCor_over_AFSEI_time_ratio']:.5f}。一致质量两种实现的解最大差异 {control['consistent_solutions_max_difference_mv']:.3e} mV。不能把原生 TorchCor 与 AFSEI nodal 的全部时间差直接解释为 nodal 离散收益。

## 验证与文件

两组均 100% 激活，细胞异常计数为零；最大真实线性残差/容差：经典 {c['max_true_residual_ratio']:.6f}，nodal {n['max_true_residual_ratio']:.6f}。初始帧逐节点与 openCARP 完全一致。每组保存完整 51 帧三维电位、全节点激活、每 EP 步探针、Cai/电流、计时、残差和最终全部 FP64 细胞状态。

结果根目录：`{pair}`。

- `torchcor_cg/visualization/ep.pvd`：经典 FEM，0–50 ms 每 1 ms。
- `nodal_afsei/visualization/ep.pvd`：nodal FEM，0–50 ms 每 1 ms。
- 各自 `final_gpb_state.npy` 和 `state_names.json`：完整最终 GPB 状态。
- `comparison/`：激活图、体对角线、电位波形、钙、完整三维表面、性能、CSV 和报告。
- `common_pde_control.json`、`preflight.json`、`torchcor_provenance.json`：方法、源码与容差核对记录。

PVD 时间单位 s、几何 mm；VTI 是原节点场的无重采样可视化，求解仍使用原始四面体。激活差值图各自标出差值范围，不能只看颜色比较误差大小。

两份 JSON 位于 `/mnt/large2/gjh/AFSEI/demo/ep_afsei_nodal_opencarp_matched/torchcor_cg.json` 和 `nodal_afsei.json`。调用同一 `run_afsei_matched.py --config ...`，驱动拒绝覆盖已有结果。

openCARP 用 16 CPU MPI 进程，AFSEI/TorchCor 用 GPU0；跨软件绝对计时包含硬件和计时范围差异。50 ms 覆盖激活和早期平台期，未验证完整复极、Land、流体和心肌运动。这组实验不应被表述为所有模型、网格和格式下的普遍优劣结论。
'''
(dest/'ANALYSIS_zh.md').write_text(text)
(dest/'output_audit.json').write_text(json.dumps(audits,indent=2))
with tarfile.open(pair/'comparison_bundle.tar.gz','w:gz') as tar:
    for file in sorted(dest.iterdir()):
        if file.suffix in ['.png','.pdf','.json','.csv','.md']:tar.add(file,arcname=file.name)
print(json.dumps(dict(comparison=m,performance=p,output_audit=audits),indent=2));print(pair/'comparison_bundle.tar.gz')

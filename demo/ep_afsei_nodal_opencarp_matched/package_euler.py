from pathlib import Path
import json,shutil,tarfile,xml.etree.ElementTree as ET
ROOT=Path('/mnt/large2/gjh/AFSEI');ours=ROOT/'results/ep_afsei_nodal_euler1us_20261031';dest=ours/'comparison'
assert json.loads((ours/'postprocessing_completed.json').read_text())['status']=='completed'
r=json.loads((dest/'report.json').read_text());a=r['Euler'];s=r['SDIRK'];c=r['openCARP'];p=r['performance']
m=r['comparison'];ca=json.loads((dest/'calcium_reference_report.json').read_text())
entries=list(ET.parse(ours/'visualization/ep.pvd').iter('DataSet'))
assert len(entries)==51 and all((ours/'visualization'/e.attrib['file']).stat().st_size>7100000 for e in entries)
assert float(entries[0].attrib['timestep'])==0 and float(entries[-1].attrib['timestep'])==.05
doc=f'''# 显式 ODE 的完整三维 EP 对照

0–50 ms 完整模拟完成。EP 指单域 PDE＋GPB 细胞 ODE；本算例未推进流体、固体或 Land。PDE 仍是 nodal P1 顶点积分、CN、轻量 Jacobi-PCG，未改为显式扩散。ODE 改为前向 Euler，步长 1 µs，每个 20 µs EP 步含 20 子步；电压在子步内冻结，子步末电流平均后进入原 CN 右端。

物理条件与已有 SDIRK/openCARP 对齐：20×7×3 mm，同一张 0.0625 mm 网格（1,777,377 节点、8,601,600 四面体）；纤维 X、全边界零通量；角部 1.5 mm 立方体刺激 35.71 µA/cm²、持续 2 ms；完整原生 Grandi 参数与初态 −80.9763 mV，无预平衡。未重新运行两份参考 PDE。

## 完整结果

| 指标 | openCARP | AFSEI SDIRK 2 µs | AFSEI Euler 1 µs |
|---|---:|---:|---:|
| 最远角激活 ms | {c['far_corner_activation_ms']:.6f} | {s['far_corner_activation_ms']:.6f} | {a['far_corner_activation_ms']:.6f} |
| 电模块/PDE 平均 ms/步 | {c['electrics_compute_ms_per_ep_step']:.3f} | {s['pde_mean_ms']:.3f} | {a['pde_mean_ms']:.3f} |
| ODE 总 s | {c['physics_compute_s']['Myocard Ionics']:.3f} | {s['ode_s']:.3f} | {a['ode_s']:.3f} |
| PDE＋ODE 总 s | {sum(c['physics_compute_s'].values()):.3f} | {s['ode_s']+s['pde_s']:.3f} | {a['ode_s']+a['pde_s']:.3f} |
| 运行墙钟 s | {c['whole_job_wall_s']:.3f} | {s['wall_s']:.3f} | {a['wall_s']:.3f} |

Euler 相对完整旧 SDIRK 基线的 ODE 计时速度比 {p['ODE_speedup_vs_completed_SDIRK']:.3f}，PDE＋ODE 速度比 {p['compute_speedup_vs_completed_SDIRK']:.3f}。旧完整 SDIRK 早于最近执行优化，因此该速度比同时包含执行优化与 ODE 格式/步长调整；复用已有优化 SDIRK 的前 2 ms 数据时，对应 ODE 比值为 {p['first2ms_ODE_speedup_vs_optimized_SDIRK']:.3f}。openCARP 为 16 CPU 进程，AFSEI 为 GPU1 RTX4090，电模块和 PDE 包装计时范围不同，不把跨硬件比值当作单纯 nodal 离散收益。

## 精度与适用范围

对 SDIRK：最远角差异 {m['SDIRK']['far_corner_difference_percent']:+.6f}%（{m['SDIRK']['far_corner_difference_ms']:+.6f} ms），全节点激活 MAE {m['SDIRK']['activation_MAE_ms']:.6f} ms，P95 绝对差异 {m['SDIRK']['activation_p95_absolute_difference_ms']:.6f} ms；共同 51 帧电位 RMSE {m['SDIRK']['common_51frame_voltage_RMSE_mv']:.6f} mV。

对 openCARP：最远角差异 {m['openCARP']['far_corner_difference_percent']:+.6f}%（{m['openCARP']['far_corner_difference_ms']:+.6f} ms），全节点激活 MAE {m['openCARP']['activation_MAE_ms']:.6f} ms。{'满足' if abs(m['openCARP']['far_corner_difference_percent'])<=5 else '尚不满足'}此前讨论的最远角 5% 范围。显式与 SDIRK 的接近程度应与 nodal 对标准 FEM 的空间/算法差异分开判断。

四个探针的实际 Euler 电压轨迹另行驱动 CPU SDIRK 0.5 µs 参考，只回放细胞 ODE、不重跑 PDE：钙相对 L2 差异 {ca['calcium_relative_L2_difference']*100:.6f}%，钙最大差异 {ca['calcium_max_absolute_difference_um']:.8g} µM，电流相对 L2 差异 {ca['current_relative_L2_difference']*100:.6f}%。该检查使用时间加密参考，不是解析解，也不是 Land 或运动中的三场反馈验证。

全部节点激活比例 {a['activated_fraction']:.6%}，细胞异常计数 {a['max_cell_guard']}，最大真实线性残差/容差 {a['max_true_residual_ratio']:.6g}；最终 Cai 范围 {a['final_calcium_range_um']} µM。初始电位逐节点与两份参考完全一致。独立 CPU Euler 递推与 GPU 的状态/电流差异记录在 euler_validation_20261031.json。

## 文件和运行

```bash
cd /mnt/large2/gjh/AFSEI/demo/ep_afsei_nodal_opencarp_matched
CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/mnt/large2/gjh/conda-envs/afsei TORCH_CUDA_ARCH_LIST=8.9 \\
/mnt/large2/gjh/conda-envs/afsei/bin/python run_afsei_matched.py --config euler1us_full.json
```

此运行已完成，驱动拒绝覆盖现有结果；若另行测试需更改 JSON 的 output。

结果目录：`{ours}`。

- `visualization/ep.pvd`：完整三维 51 帧，0–50 ms，每 1 ms；PVD 时间单位 s，几何 mm。VTI 是原始节点场的无重采样可视化，求解单元仍为四面体。
- `vm.igb`、`activation_ms.npy`：全部节点电位与激活时刻。
- `traces.npz`：每 EP 步探针电位、ODE/PDE 计时与残差。
- `cell_traces.npz`：探针 Cai 和区间平均电流。
- `final_gpb_state.npy`、`state_names.json`：完整最终 FP64 细胞状态，支持后续精度审查。
- `comparison/`：三方激活分布、三维表面、电位、性能、钙图及报告。差值图各自标明数值范围，不能只凭颜色强度比较两个差异的大小。

50 ms 覆盖本 benchmark 的激活和早期平台期，未覆盖完整动作电位复极。
'''
(dest/'ANALYSIS_zh.md').write_text(doc)
base=ROOT/'demo/ep_afsei_nodal_opencarp_matched'
for name in ['euler1us_full.json','euler_validation_20261031.json']:shutil.copy2(base/name,dest/name)
for name in ['report.json','configuration.json','resolved_configuration.json']:shutil.copy2(ours/name,dest/('euler_'+name))
audit=dict(frames=51,first_time_s=0,last_time_s=.05,all_frame_files_present=True,full_3d_nodes=a['nodes'],
    final_cell_state_shape=[54,a['nodes']],all_reference_runs_reused=True)
(dest/'output_audit.json').write_text(json.dumps(audit,indent=2))
with tarfile.open(ours/'comparison_bundle.tar.gz','w:gz') as tar:
    for f in sorted(dest.iterdir()):
        if f.suffix in ['.json','.md','.csv','.png','.pdf','.npz']:tar.add(f,arcname=f.name)
print(json.dumps(audit,indent=2));print(ours/'comparison_bundle.tar.gz')

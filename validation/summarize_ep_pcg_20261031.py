from pathlib import Path
import json,hashlib
r=Path('/mnt/large2/gjh/AFSEI');p=r/'results/ep_pcg_optimization_20261031_final'
reg=json.loads((p/'regression.json').read_text());fixed=json.loads((p/'fixed_input.json').read_text())
before=reg['before'];after=reg['after']
rows=[x for x in fixed if 'before' in x]
lines=['# EP GPU执行优化验证，2026-10-31','',
'本轮修改可复用的 AFSEI EP 模块。物理参数、顶点/一致质量离散、CN、Euler ODE、最终真实残差容差均保持原设置。未运行新的完整50 ms benchmark；未重跑openCARP。','',
'实现：CN nodal 右端融合生成初始残差，复用 K·V；Graph-PCG仅在递推残差收敛、周期校正、最终步骤时计算真实残差；融合 x/r更新、Jacobi除法与残差/预条件内积局部归约；融合方向更新；默认采用实测更快的 torch.mv(CSR,out=...)。原Triton CSR可用 backend="triton" 选择。','',
'真实残差不合格时重启PCG，迭代上限耗尽时报错。返回电位仍独立拥有内存。诊断次数明确是执行槽数，包括Graph段内收敛后的槽位；未将其冒充有效迭代数。','',
'GPU0，RTX4090，FP64，原0.0625 mm网格，1,777,377节点。固定输入取既有0/5/15/30/50 ms电位，I=0，规定相同角部刺激；这是代数控制输入，不是该时间实际生理源项。每项2次预热、5次同步墙钟采样，中位数。原实现从备份独立加载。','',
'| 离散 | 快照ms | 原PDE ms | 优化PDE ms | 加速倍数 | 最大解差mV |',
'|---|---:|---:|---:|---:|---:|']
for x in rows:
    lines.append(f"| {'nodal' if x['nodal'] else '一致质量+AFSEI求解器'} | {x['frame_ms']} | {x['before']['median_ms']:.4f} | {x['after']['median_ms']:.4f} | {x['speedup']:.3f} | {x['max_difference_mv']:.3e} |")
lines+=['',f"2 ms短程EP+GPB（100步）：PDE {before['pde_mean_ms']:.4f}→{after['pde_mean_ms']:.4f} ms/步，耗时减少 {(1-after['pde_s']/before['pde_s'])*100:.2f}%；ODE {before['ode_s']:.4f}→{after['ode_s']:.4f} s；完整循环 {before['wall_s']:.4f}→{after['wall_s']:.4f} s。",
    f"探针逐步电位最大差 {reg['trace_gaps']['voltage_mv']:.3e} mV；三维存储帧差 {reg['stored_vm_max_difference_mv']:.3e} mV；最终全部54项细胞状态最大绝对差 {reg['cell_state_max_abs_difference']:.3e}（各状态原单位）。",
    f"最大真实残差/容差 {after['max_true_residual_ratio']:.9f}<1，细胞异常计数 {after['max_cell_guard']}。FP32/FP64、零初始残差、非零初始猜测、RCM重排、周期重启、尾段以及返回值内存独立性检查通过。",'',
    '稀疏乘法首轮直接对照：旧Triton约0.614 ms，torch.mv约0.416 ms。最终fixed_input.json中SpMV记录的custom字段沿用初版脚本标签，实际指最终默认CSRAction，即torch.mv；library也是torch.mv，不能把它解读成第二次Triton对照。脚本后续已将字段更名default_action。','',
    '本轮一致质量控制使用AFSEI求解器；没有把其结果冒充新的原生TorchCor性能。完整传播时序和0.125/0.0625 mm跨方法性能仍待后续完整实验确认。优化不针对修正既有nodal传播延迟。','',
    '源码备份：backups/ep_pcg_20261031/。验证脚本：validation/validate_ep_pcg_20261031.py（--output指定新的目录，拒绝覆盖）。']
(p/'ANALYSIS_zh.md').write_text('\n'.join(lines)+'\n')
hashes={name:hashlib.sha256((r/'src/afsi_torch/ep'/name).read_bytes()).hexdigest() for name in ('graph_pcg.py','gpu_kernels.py','jacobi.py','monodomain.py')}
(p/'source_hashes.json').write_text(json.dumps(hashes,indent=2));print('\n'.join(lines))

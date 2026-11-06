"""Read-only comparison of completed runs. No solver invocation or rerun."""
from pathlib import Path
import json,csv
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/mnt/large2/gjh/AFSEI/results')
ours=root/'ep_afsei_nodal_opencarp_matched_20261030'
ref=root/'opencarp_gpb_h0p0625_20261030'
out=ours/'comparison';out.mkdir(exist_ok=True)
r=json.loads((ours/'report.json').read_text());nr=json.loads((ref/'analysis/report.json').read_text())
assert r['status']=='completed' and r['duration_ms']==nr['duration_ms']==50
X=np.load(root/'ep_opencarp_same_mesh_20261030/coordinates_cm.npy')*10
N=len(X);A=np.load(ours/'activation_ms.npy');B=np.loadtxt(ref/'simulation_v2/init_acts_vm_act-thresh.dat')
V=np.memmap(ours/'vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,N))
W=np.memmap(ref/'simulation_v2/vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,N))
tr=np.load(ours/'traces.npz');ids=tr['probe_ids'];tt=tr['time_ms'];trace=tr['voltage_mv']
assert ids.tolist()==[p['node'] for p in nr['probes']]
ijk=np.rint(X/.0625).astype(np.int64)
assert np.array_equal(ijk[:,0]+321*ijk[:,1]+321*113*ijk[:,2],np.arange(N))
both=np.isfinite(A)&np.isfinite(B)&(B>=0);delta=A[both]-B[both]
stim=(X<=1.5+1e-10).all(1);outside=both&~stim
sq=0.;count=0;maximum=0.;initial_min=float(V[0].min());initial_max=float(V[0].max())
snap=[]
for k in range(51):
    d=V[k].astype(np.float64)-W[k].astype(np.float64)
    sq+=np.dot(d,d);count+=len(d);maximum=max(maximum,float(abs(d).max()))
    snap.append([k,float(np.sqrt(np.mean(d*d))),float(abs(d).max())])
probe_metrics=[]
for j,p in enumerate(ids):
    aligned=np.interp(np.arange(51),tt,trace[:,j]);d=aligned-W[:,p]
    probe_metrics.append(dict(node=int(p),position_mm=X[p].tolist(),afsei_activation_ms=float(A[p]),
        opencarp_activation_ms=float(B[p]),activation_difference_ms=float(A[p]-B[p]),
        voltage_rmse_common_1ms_mv=float(np.sqrt(np.mean(d*d))),
        afsei_min_dense_mv=float(trace[:,j].min()),opencarp_min_1ms_mv=float(W[:,p].min()),
        afsei_peak_dense_mv=float(trace[:,j].max()),opencarp_peak_1ms_mv=float(W[:,p].max())))
metrics=dict(activation_MAE_ms=float(abs(delta).mean()),activation_RMSE_ms=float(np.sqrt(np.mean(delta*delta))),
    activation_mean_signed_difference_ms=float(delta.mean()),activation_p95_absolute_difference_ms=float(np.percentile(abs(delta),95)),
    activation_max_absolute_difference_ms=float(abs(delta).max()),activation_compared_nodes=int(both.sum()),
    outside_stimulus_activation_MAE_ms=float(abs(A[outside]-B[outside]).mean()),
    far_corner_difference_ms=float(A[-1]-B[-1]),far_corner_difference_percent=float((A[-1]/B[-1]-1)*100),
    common_51frame_voltage_RMSE_mv=float(np.sqrt(sq/count)),common_51frame_voltage_max_difference_mv=maximum,
    afsei_voltage_min_mv=float(V.min()),afsei_voltage_max_mv=float(V.max()),
    afsei_initial_range_mv=[initial_min,initial_max],probes=probe_metrics)
result=dict(scope='Same 3D geometry/connectivity/physical parameters/GPB RHS and initial state/EP dt/stimulus; original AFSEI numerical method retained',
    opencarp_reused=True,AFSEI=r,openCARP=nr,comparison=metrics,
    performance=dict(opencarp_Electrics_s=nr['physics_compute_s']['Electrics'],afsei_PDE_s=r['pde_s'],
        opencarp_ODE_s=nr['physics_compute_s']['Myocard Ionics'],afsei_ODE_s=r['ode_s'],
        afsei_PDE_fraction=(r['pde_s']/(r['pde_s']+r['ode_s'])),
        raw_Electrics_to_PDE_time_ratio=nr['physics_compute_s']['Electrics']/r['pde_s'],
        raw_compute_time_ratio=sum(nr['physics_compute_s'].values())/(r['pde_s']+r['ode_s']),
        raw_wall_time_ratio=nr['whole_job_wall_s']/r['wall_s']),
    remaining_differences=['AFSEI diagonal nodal mass versus openCARP consistent P1 mass.',
        'AFSEI SDIRK with 2 us ionic substeps/frozen voltage/averaged source; openCARP native Rush-Larsen and reaction-then-diffusion split.',
        'AFSEI FP64 including gates; native openCARP gate states float32.',
        'AFSEI true residual rtol 1e-10, atol 1e-13 with Jacobi; openCARP native PETSc/block-Jacobi+ILU criterion differs.',
        'One RTX 4090 versus 16 CPU MPI ranks; GPU0 had another existing job. No isolated discretization-only speedup is inferred.',
        'openCARP Electrics is a physics-module timer; AFSEI PDE is synchronized wrapper time including checks/RHS/CG. Wall/setup scopes differ.'],
    source_model_validation=json.loads((root.parent/'demo/ep_afsei_nodal_opencarp_matched/validation.json').read_text()))
late=both&(B>=5)
result['comparison']['activation_p95_absolute_relative_difference_percent_where_reference_ge_5ms']=float(np.percentile(abs(A[late]/B[late]-1)*100,95))
result['comparison']['far_corner_within_5_percent']=bool(abs(metrics['far_corner_difference_percent'])<=5)
source=root.parent/'src/afsi_torch/ep'
result['production_PDE_source_sha256']={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
    for name in ['monodomain.py','jacobi.py','graph_pcg.py','gpu_kernels.py']}
result['physical_parameters_and_mesh']=json.loads((root/'ep_opencarp_same_mesh_20261030/common_configuration.json').read_text())
result['physical_parameters_and_mesh'].pop('operator_splitting',None) # cache metadata describes the old driver, not this run
result['ODE_execution_notes']=dict(substeps_per_EP=10,rhs_evaluations_per_SDIRK_substep_min=6,
    rhs_evaluations_per_SDIRK_substep_max=8,advance_kernel_registers_per_thread=255,
    advance_kernel_stack_bytes_per_thread=2688,
    evidence='demo/ep_afsei_nodal_opencarp_matched/kernel_resources.txt; cuobjdump of the actual compiled benchmark cell adapter',
    interpretation='Original SDIRK entails multiple RHS evaluations. High register use and per-thread stack are further implementation costs; resource counts alone do not quantify their runtime share.')
(out/'report.json').write_text(json.dumps(result,indent=2))
plt.rcParams.update({'font.size':10,'axes.titleweight':'bold'})
colors=['#364553','#e27f28'];names=['openCARP: standard P1','AFSEI: nodal P1']
nx,ny,nz=321,113,49;x=np.linspace(0,20,nx);y=np.linspace(0,7,ny)
def plane(values):return values[:nx*ny].reshape(ny,nx)
fig,axes=plt.subplots(2,2,figsize=(14,8.5),layout='constrained')
vmax=max(float(np.nanmax(A)),float(np.nanmax(B)))
for ax,name,a in zip(axes[0],names,[B,A]):
    im=ax.pcolormesh(x,y,plane(a),shading='auto',vmin=0,vmax=vmax,cmap='turbo',rasterized=True)
    ax.contour(x,y,plane(a),levels=np.arange(5,vmax,5),colors='white',linewidths=.6)
    ax.add_patch(plt.Rectangle((0,0),1.5,1.5,fill=False,color='black',ls='--'))
    ax.set(title=name+' | z = 0 mm',xlabel='x [mm]',ylabel='y [mm]');ax.set_aspect('equal')
    fig.colorbar(im,ax=ax,shrink=.75,label='Activation [ms]')
ax=axes[1,0];d=A-B;lim=max(float(np.nanmax(abs(d))),1e-8)
im=ax.pcolormesh(x,y,plane(d),shading='auto',vmin=-lim,vmax=lim,cmap='coolwarm',rasterized=True)
ax.set(title='AFSEI minus openCARP',xlabel='x [mm]',ylabel='y [mm]');ax.set_aspect('equal')
fig.colorbar(im,ax=ax,shrink=.75,label='Activation difference [ms]')
q=np.arange(17)[:,None]/16*np.array([320,112,48]);q=np.rint(q).astype(int)
di=q[:,0]+nx*q[:,1]+nx*ny*q[:,2];dist=np.linalg.norm(X[di],axis=1)
ax=axes[1,1]
for a,c,n in zip([B,A],colors,names):ax.plot(dist,a[di],'-o',color=c,label=n,ms=3)
ax.set(title=f'Body diagonal | far-corner difference {metrics["far_corner_difference_percent"]:+.2f}%',
    xlabel='Distance [mm]',ylabel='Activation [ms]');ax.legend();ax.grid(alpha=.2)
fig.suptitle('Same native 3D mesh and physical parameters | h = 0.0625 mm | GPB | dt = 0.02 ms\nAFSEI keeps nodal CN + SDIRK + Graph Jacobi-PCG; no AMG',fontsize=13)
for ext in ['png','pdf']:fig.savefig(out/('activation_comparison.'+ext),dpi=165)
fig=plt.figure(figsize=(14,5.3),layout='constrained')
boundary=np.any(np.isclose(X,0)|np.isclose(X,[20,7,3]),axis=1)
surface=np.flatnonzero(boundary)[::3]
for j,(a,n) in enumerate(zip([B,A],names)):
    ax=fig.add_subplot(1,2,j+1,projection='3d')
    sc=ax.scatter(*X[surface].T,c=a[surface],cmap='turbo',vmin=0,vmax=vmax,s=2,depthshade=False,rasterized=True)
    ax.set(xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title=n)
    ax.set_box_aspect([20,7,3]);ax.view_init(23,-58)
    fig.colorbar(sc,ax=ax,shrink=.6,pad=.02,label='Activation [ms]')
fig.suptitle('3D activation on the six outer faces | identical mesh and color range',fontsize=13)
for ext in ['png','pdf']:fig.savefig(out/('activation_3d_comparison.'+ext),dpi=165)
fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
for j,ax in enumerate(axes.flat):
    p=int(ids[j]);ax.plot(np.arange(51),W[:,p],'-o',color=colors[0],ms=2,lw=1.3,label=names[0]+' (1 ms output)')
    ax.plot(tt,trace[:,j],color=colors[1],lw=1.4,label=names[1]+' (0.02 ms trace)')
    ax.set(title='('+', '.join(f'{n:g}' for n in X[p])+') mm',xlabel='Time [ms]',ylabel='Vm [mV]',xlim=(0,50))
    ax.axhline(0,color='gray',ls=':',lw=.7);ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Voltage waveforms | identical native GPB physical parameters and initial state',fontsize=13)
for ext in ['png','pdf']:fig.savefig(out/('voltage_comparison.'+ext),dpi=165)
fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
labels=['openCARP\n16 CPU ranks','AFSEI\nRTX 4090']
panels=[([nr['electrics_compute_ms_per_ep_step'],r['pde_mean_ms']],'Electrics module / PDE wrapper','Time per step [ms]'),
    ([nr['ionics_compute_ms_per_ep_step'],r['ode_mean_ms']],'GPB ionics: native RL / AFSEI SDIRK','Time per step [ms]'),
    ([sum(nr['physics_compute_s'].values()),r['pde_s']+r['ode_s']],'PDE + ODE compute for 50 ms','Total compute [s]')]
for ax,(vals,title,ylab) in zip(axes,panels):
    bars=ax.bar(labels,vals,color=colors);ax.bar_label(bars,fmt='%.2f',padding=4)
    ax.set(title=title,ylabel=ylab,ylim=(0,max(vals)*1.2));ax.grid(axis='y',alpha=.15)
fig.suptitle('Measured runtime | same mesh / EP dt / 50 ms duration; native time integrators retained',fontsize=13)
fig.supxlabel('Different hardware and linear/ionic solvers; these are measured implementation timings, not isolated nodal-discretization gains.',fontsize=9)
for ext in ['png','pdf']:fig.savefig(out/('performance_comparison.'+ext),dpi=165)
with (out/'snapshot_errors.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['time_ms','Vm_RMSE_mV','Vm_max_difference_mV']);w.writerows(snap)
with (out/'diagonal_activation.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['distance_mm','openCARP_ms','AFSEI_ms']);w.writerows(zip(dist,B[di],A[di]))
with (out/'probe_waveforms.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['time_ms']+[f'AFSEI_node{p}_mV' for p in ids]+[f'openCARP_node{p}_mV' for p in ids])
    w.writerows(np.column_stack([np.arange(51),*[np.interp(np.arange(51),tt,trace[:,j]) for j in range(4)],W[:,ids]]))
analysis=f'''# AFSEI nodal 与已有 openCARP 三维算例对照

计算完成，未重新运行 openCARP。采用同一张 20×7×3 mm、0.0625 mm 网格（{N:,} 节点，8,601,600 四面体），纤维沿 X；全部边界零通量，角部 1.5 mm 立方体内刺激 35.71 µA/cm²、持续 2 ms。

物理 GPB 方程/参数/完整初态、电导率、膜电容、表面积体积比、EP 时间步 0.02 ms 已对齐。为了遵守“数值方法不变”，AFSEI 保留原 SDIRK（ODE 步长 0.002 ms、冻结电压、子步电流平均）及 nodal CN；没有切换成 openCARP 的 Rush–Larsen / 反应后扩散格式。使用的是 AFSEI 可复用的 BufferedRHS / GraphJacobiPCG，不调用 TorchCor 或 AMG。

| 指标 | openCARP | AFSEI nodal |
|---|---:|---:|
| 最远角激活时间 ms | {B[-1]:.6f} | {A[-1]:.6f} |
| PDE/电模块平均 ms/步 | {nr['electrics_compute_ms_per_ep_step']:.3f} | {r['pde_mean_ms']:.3f} |
| ODE 平均 ms/步 | {nr['ionics_compute_ms_per_ep_step']:.3f} | {r['ode_mean_ms']:.3f} |
| PDE＋ODE 累计 s | {sum(nr['physics_compute_s'].values()):.3f} | {r['pde_s']+r['ode_s']:.3f} |

最远角差异 {metrics['far_corner_difference_percent']:+.3f}%；全节点激活时间 MAE {metrics['activation_MAE_ms']:.6f} ms、P95 {metrics['activation_p95_absolute_difference_ms']:.6f} ms。共同 51 帧电位 RMSE {metrics['common_51frame_voltage_RMSE_mv']:.6f} mV。激活终点差异不能替代波前与波形检查，图中同时显示了激活分布、体对角线和四处电位。

按此前讨论的最远角 5% 判断：{'满足该指标' if metrics['far_corner_within_5_percent'] else '尚不满足，不能宣称已达到同等精度'}。原始电模块/PDE 计时比值为 {result['performance']['raw_Electrics_to_PDE_time_ratio']:.3f}，整体计算耗时比值（openCARP/AFSEI）为 {result['performance']['raw_compute_time_ratio']:.3f}；该比值小于 1 表示本次 AFSEI 整体更慢。

AFSEI 的 PDE 占计算时间 {100*result['performance']['afsei_PDE_fraction']:.2f}%。本次是实际实现的跨硬件对照：openCARP 用 16 CPU MPI 进程，AFSEI 用 GPU1 的 RTX4090；openCARP 的 Electrics 模块计时与 AFSEI PDE 包装计时范围也不同，不能将其比值全部归因于 nodal 离散。ODE 格式/容差/门变量精度保留各自数值方法，因此这不是单独改变质量矩阵的实验。

每个 EP 步中 AFSEI 有 10 个 SDIRK 子步，每子步需要 6–8 次 RHS 评估，与原生 RL 单步的工作量不同。实际编译的本 benchmark 细胞内核还使用每线程 255 个寄存器、2688 B 栈空间，提示较高的寄存器/私有存储压力；这项资源计数不等于已完成性能归因的 profiler 结果。内核信息保存在 kernel_resources.txt。该信息描述本次参数适配器，不能无条件推广到所有生产 GPB/Land 路径。

初态采用本次要求的 openCARP 原始值 −80.9763 mV，无预平衡；仅影响本 benchmark，未修改生产 GPB 默认初态。全部节点激活比例 {r['activated_fraction']:.6%}，细胞异常计数 {r['max_cell_guard']}，最大真实线性残差/容差 {r['max_true_residual_ratio']:.6g}。

文件：vm.igb 为 0–50 ms 每 1 ms 的完整三维节点电位；activation_ms.npy 为全部节点激活时间；traces.npz 包含每 EP 步的探针、PDE/ODE 时间和残差；comparison/ 下为报告、PNG/PDF 与 CSV。
'''
(out/'ANALYSIS_zh.md').write_text(analysis)
print(json.dumps(dict(comparison=metrics,performance=result['performance']),indent=2))

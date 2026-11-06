"""Compare quadrature only within AFSEI; immutable fine openCARP reference."""
from pathlib import Path
import json,csv,hashlib,struct,xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/mnt/large2/gjh/AFSEI');OUT=ROOT/'results/ep_shared_fem_20261031';DEST=OUT/'comparison';DEST.mkdir(exist_ok=True)
OC=ROOT/'results/opencarp_gpb_h0p0625_20261030';FINE=ROOT/'results/ep_opencarp_same_mesh_20261030'
assert json.loads((OUT/'progress.json').read_text())['status']=='simulations_completed'
for name,digest in json.loads((OUT/'source_hashes.json').read_text()).items():
    assert hashlib.sha256((ROOT/'src/afsi_torch/ep'/name).read_bytes()).hexdigest()==digest,'Solver source changed during comparison'
refact=np.loadtxt(OC/'simulation_v2/init_acts_vm_act-thresh.dat');refvm=np.memmap(OC/'simulation_v2/vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,len(refact)))
colors={'consistent_afsei':'#2878b5','nodal_afsei':'#ed8b23','openCARP':'#333f4d'}
names={'consistent_afsei':'AFSEI consistent P1','nodal_afsei':'AFSEI nodal P1','openCARP':'openCARP reference'}
groups={};metrics={};reports={};configs={};wave_arrays={}
for h,key,shape in [(.125,'h0p125',(161,57,25)),(.0625,'h0p0625',(321,113,49))]:
    cache=OUT/'operators_h0p125' if h==.125 else FINE
    ids=np.load(cache/'openCARP_reference_node_ids.npy') if h==.125 else np.arange(len(refact))
    X=np.load(cache/'coordinates_cm.npy')*10;N=len(X);act={};vm={};traces={}
    ref=refact[ids]
    assert np.allclose(np.load(FINE/'coordinates_cm.npy')[ids]*10,X,atol=1e-12)
    for method in ('consistent_afsei','nodal_afsei'):
        label=key+'_'+method;folder=OUT/label;r=json.loads((folder/'report.json').read_text());cfg=json.loads((folder/'configuration.json').read_text())
        assert r['status']=='completed' and r['duration_ms']==50 and r['max_cell_guard']==0 and r['max_true_residual_ratio']<=1 and r['activated_fraction']==1
        assert r['physical_gpu']=='0' and not r['numerical_methods']['torchcor_solver']
        assert r['numerical_methods']['backend']=='afsi_torch.ep.graph_pcg.GraphJacobiPCG' and r['numerical_methods']['rhs']=='BufferedRHS'
        assert r['ode_method']=='explicit_euler' and r['ode_dt_ms']==.001 and r['ode_substeps_per_ep']==20
        reports[label]=r;configs[label]=cfg;act[method]=np.load(folder/'activation_ms.npy')
        vm[method]=np.memmap(folder/'vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,N));traces[method]=np.load(folder/'traces.npz')
        delta=act[method]-ref;late=ref>=5
        squares=0.;maxvoltage=0.;frame_errors=[]
        assert np.array_equal(vm[method][0],refvm[0,ids])
        for k in range(51):
            d=vm[method][k].astype(np.float64)-refvm[k,ids];sq=float(np.dot(d,d));squares+=sq;maxvoltage=max(maxvoltage,float(abs(d).max()))
            frame_errors.append([k,np.sqrt(sq/N),float(abs(d).max())])
        metrics[label]=dict(h_mm=h,far_corner_activation_ms=float(act[method][-1]),far_corner_delta_ms=float(delta[-1]),
            far_corner_relative_delta_percent=float(delta[-1]/ref[-1]*100),far_corner_within_5percent=bool(abs(delta[-1]/ref[-1])<=.05),
            activation_MAE_ms=float(abs(delta).mean()),activation_RMSE_ms=float(np.sqrt(np.mean(delta*delta))),
            activation_p95_abs_ms=float(np.percentile(abs(delta),95)),activation_max_abs_ms=float(abs(delta).max()),
            activation_p95_relative_percent_ref_ge5ms=float(np.percentile(abs(delta[late]/ref[late])*100,95)),
            voltage_RMSE_mv=float(np.sqrt(squares/(51*N))),voltage_max_abs_mv=maxvoltage)
        with (DEST/(label+'_frame_errors.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['time_ms','voltage_RMSE_mv','voltage_max_abs_mv']);w.writerows(frame_errors)
    for field in ('operator_cache','dt_s','ode_dt_ms','solver','ode_method','ode_execution','mesh_mm','physical_parameters','stimulus_ua_cm2','stimulus_ms'):
        assert configs[key+'_consistent_afsei'][field]==configs[key+'_nodal_afsei'][field],field
    groups[key]=dict(h=h,shape=shape,X=X,ids=ids,reference=ref,act=act,vm=vm,traces=traces)
    # Same-node classic-nodal trajectory difference, independent of reference.
    d=act['nodal_afsei']-act['consistent_afsei'];metrics[key+'_nodal_vs_consistent']=dict(activation_MAE_ms=float(abs(d).mean()),
        far_corner_delta_ms=float(d[-1]),PDE_consistent_over_nodal=reports[key+'_consistent_afsei']['pde_s']/reports[key+'_nodal_afsei']['pde_s'])

plt.rcParams.update({'font.size':10})
def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(DEST/(name+'.'+ext),dpi=170)
    plt.close(fig)
# Performance: identical solver/backend and timing boundaries in all four cases.
fig,axs=plt.subplots(1,3,figsize=(16,4.8),layout='constrained');keys=list(groups);xs=np.arange(2)
for method,shift in [('consistent_afsei',-.18),('nodal_afsei',.18)]:
    rr=[reports[k+'_'+method] for k in keys]
    for ax,field,title,unit in zip(axs,['pde_mean_ms','ode_mean_ms','wall_s'],['PDE per EP step','GPB Euler per EP step','Whole EP loop'],['ms','ms','s']):
        values=[r[field] for r in rr];bars=ax.bar(xs+shift,values,.34,color=colors[method],label=names[method]);ax.bar_label(bars,fmt='%.2f',padding=3)
        ax.set(title=title,ylabel=unit,xticks=xs,xticklabels=['0.125 mm','0.0625 mm']);ax.set_ylim(0,max([reports[k+'_'+m][field] for k in keys for m in ('consistent_afsei','nodal_afsei')])*1.18);ax.grid(axis='y',alpha=.2)
axs[0].legend(fontsize=8)
fig.suptitle('Shared optimized AFSEI | CN 20 us, identical GPB Euler 1 us | RTX 4090 GPU0',fontsize=13);save(fig,'performance_comparison')
# Reference accuracy, all compare to existing h=.0625 openCARP data.
fig,axs=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
for method,shift in [('consistent_afsei',-.18),('nodal_afsei',.18)]:
    for ax,field,title in zip(axs,['far_corner_relative_delta_percent','activation_MAE_ms','voltage_RMSE_mv'],['Far-corner LAT difference [%]','LAT mean absolute difference [ms]','51-frame Vm RMSE [mV]']):
        values=[metrics[k+'_'+method][field] for k in keys];bars=ax.bar(xs+shift,values,.34,color=colors[method],label=names[method]);ax.bar_label(bars,fmt='%.3f',padding=3);ax.set(title=title,xticks=xs,xticklabels=['0.125 mm','0.0625 mm']);ax.grid(axis='y',alpha=.2);ax.margins(y=.2)
axs[0].axhline(5,color='gray',ls='--',lw=1,label='+5%');axs[0].axhline(-5,color='gray',ls='--',lw=1);axs[0].legend(fontsize=8)
fig.suptitle('Differences from reused openCARP h=0.0625 mm reference | coarse comparison at coincident nodes');save(fig,'accuracy_comparison')
# Activation maps at bottom face; reference evaluated at same nodes.
maxact=max(float(np.max(g['act'][m])) for g in groups.values() for m in g['act'])
fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
for row,(key,g) in enumerate(groups.items()):
    nx,ny,nz=g['shape'];x=np.linspace(0,20,nx);y=np.linspace(0,7,ny)
    for ax,method in zip(axes[row],('openCARP','consistent_afsei','nodal_afsei')):
        a=g['reference'] if method=='openCARP' else g['act'][method];a=a[:nx*ny].reshape(ny,nx)
        im=ax.pcolormesh(x,y,a,cmap='turbo',vmin=0,vmax=maxact,shading='auto',rasterized=True);ax.contour(x,y,a,levels=np.arange(5,maxact,5),colors='white',linewidths=.5)
        ax.set(title=f"{names[method]} | comparison h={g['h']:g} mm",xlabel='x [mm]',ylabel='y [mm]');ax.set_aspect('equal');fig.colorbar(im,ax=ax,shrink=.65,label='LAT [ms]')
fig.suptitle('Activation distribution | z=0 mm | identical scale, threshold Vm=0 mV');save(fig,'activation_comparison')
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
limit=max(metrics[k+'_'+m]['activation_max_abs_ms'] for k in keys for m in ('consistent_afsei','nodal_afsei'))
for row,(key,g) in enumerate(groups.items()):
    nx,ny,nz=g['shape'];x=np.linspace(0,20,nx);y=np.linspace(0,7,ny)
    for ax,method in zip(axes[row],('consistent_afsei','nodal_afsei')):
        d=g['act'][method]-g['reference'];im=ax.pcolormesh(x,y,d[:nx*ny].reshape(ny,nx),cmap='coolwarm',vmin=-limit,vmax=limit,shading='auto',rasterized=True)
        ax.set(title=f"{names[method]} minus openCARP | h={g['h']:g}",xlabel='x [mm]',ylabel='y [mm]');ax.set_aspect('equal');fig.colorbar(im,ax=ax,shrink=.7,label='LAT difference [ms]')
fig.suptitle('Activation differences | common scale for all four panels');save(fig,'activation_difference')
fig,axes=plt.subplots(2,4,figsize=(19,8),layout='constrained')
for row,(key,g) in enumerate(groups.items()):
    probe=g['traces']['nodal_afsei']['probe_ids'];coords=g['X'][probe]
    for j,ax in enumerate(axes[row]):
        for method in ('consistent_afsei','nodal_afsei'):
            tr=g['traces'][method];ax.plot(tr['time_ms'],tr['voltage_mv'][:,j],label=names[method],color=colors[method],lw=1.4)
        ax.plot(np.arange(51),refvm[:,g['ids'][probe[j]]],color=colors['openCARP'],ls='--',marker='o',ms=2,label=names['openCARP'])
        ax.set(title=f"h={g['h']:g} | ({','.join(f'{v:g}' for v in coords[j])}) mm",xlabel='Time [ms]',ylabel='Vm [mV]',xlim=(0,50));ax.grid(alpha=.2)
        if row==0 and j==0:ax.legend(fontsize=8)
fig.suptitle('Voltage waveforms at identical physical probes | both AFSEI methods: same GPB Euler');save(fig,'voltage_comparison')
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for ax,(key,g) in zip(axes,groups.items()):
    shape=np.asarray(g['shape']);q=np.rint(np.linspace(0,1,33)[:,None]*(shape-1)).astype(int);idx=q[:,0]+shape[0]*q[:,1]+shape[0]*shape[1]*q[:,2];distance=np.linalg.norm(g['X'][idx],axis=1)
    for method in ('consistent_afsei','nodal_afsei'):ax.plot(distance,g['act'][method][idx],color=colors[method],label=names[method])
    ax.plot(distance,g['reference'][idx],color=colors['openCARP'],ls='--',label=names['openCARP']);ax.set(title=f"h={g['h']:g} mm",xlabel='Body-diagonal distance [mm]',ylabel='LAT [ms]');ax.grid(alpha=.2);ax.legend(fontsize=8)
save(fig,'diagonal_activation')
# Lossless PVD/VTI at each mesh's original nodes (display topology only).
for key,g in groups.items():
    nx,ny,nz=g['shape'];extent=f'0 {nx-1} 0 {ny-1} 0 {nz-1}';h=g['h']
    def vti(path,array,name):
        data=np.asarray(array,dtype='<f4').tobytes();header=f'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64"><ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="{h} {h} {h}"><Piece Extent="{extent}"><PointData Scalars="{name}"><DataArray type="Float32" Name="{name}" format="appended" offset="0"/></PointData><CellData/></Piece></ImageData><AppendedData encoding="raw">_'
        with path.open('wb') as f:f.write(header.encode());f.write(struct.pack('<Q',len(data)));f.write(data);f.write(b'</AppendedData></VTKFile>')
    for method in ('consistent_afsei','nodal_afsei'):
        folder=OUT/(key+'_'+method);vis=folder/'visualization';vis.mkdir(exist_ok=True);root=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian');col=ET.SubElement(root,'Collection')
        for k in range(51):
            filename=f'ep_{k:03d}.vti';vti(vis/filename,g['vm'][method][k],'voltage_mV');ET.SubElement(col,'DataSet',timestep=str(k*.001),group='',part='0',file=filename)
        ET.indent(root);ET.ElementTree(root).write(vis/'ep.pvd',encoding='utf-8',xml_declaration=True);vti(vis/'activation.vti',g['act'][method],'activation_time_ms')
        (vis/'README_zh.md').write_text('ep.pvd：0–50ms，51帧；时间s，坐标mm。电位固定色标建议[-90,80]mV。VTI逐节点无重采样，仅用规则显示拓扑；求解使用四面体。\n')

result=dict(scope='Shared optimized AFSEI framework, identical FP64 GPB Euler1us and CN20us; standard consistent P1 versus vertex-quadrature nodal P1. Four fresh 50ms runs sequential GPU0.',
    reports=reports,metrics=metrics,openCARP_reference=json.loads((OC/'analysis/report.json').read_text()),
    reference_reused=True,coarse_reference_comparison='Exact coincident-node restriction of existing h0.0625 openCARP result, no spatial interpolation and no new coarse openCARP simulation.',
    remaining_reference_differences='openCARP native RL, splitting, gate precision and solver differ; reference is cross-validation, not analytic truth. 50ms covers activation/early plateau only.',
    source_hashes=json.loads((OUT/'source_hashes.json').read_text()))
(DEST/'report.json').write_text(json.dumps(result,indent=2))
lines=['# 共享AFSEI框架：nodal FEM与经典一致质量CG FEM','',
'四组全新0–50ms计算在GPU0 RTX4090顺序完成。两组均使用AFSEI优化的BufferedRHS、GraphJacobiPCG、Jacobi预条件和相同GPB显式Euler内核，没有调用TorchCor。差异仅为顶点/一致质量积分及相应源项积分。','',
'共同条件：20×7×3mm；纤维沿X；各面零通量；角部1.5mm立方体0–2ms施加35.71µA/cm²；GPB原始初态Vm=-80.9763mV、无预平衡；原生openCARP等效电导率；CN20µs；Euler1µs；FP64；真实残差max(1e-13,1e-10||b||)，无AMG。','',
'0.0625mm复用原生openCARP四面体网格（1,777,377节点，8,601,600单元）；0.125mm采用同一交替五四面体模板（229,425节点，1,075,200单元）。每档两方法的几何、节点与单元完全相同。','',
'| h mm | 方法 | PDE ms/步 | ODE ms/步 | 循环s | 远端LAT ms | 对openCARP差% | LAT MAE ms | Vm RMSE mV |',
'|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for key in keys:
    for method in ('consistent_afsei','nodal_afsei'):
        label=key+'_'+method;r=reports[label];m=metrics[label]
        lines.append(f"| {m['h_mm']} | {'一致质量CG' if method=='consistent_afsei' else 'nodal'} | {r['pde_mean_ms']:.4f} | {r['ode_mean_ms']:.4f} | {r['wall_s']:.3f} | {m['far_corner_activation_ms']:.5f} | {m['far_corner_relative_delta_percent']:+.3f} | {m['activation_MAE_ms']:.5f} | {m['voltage_RMSE_mv']:.4f} |")
lines+=['',f'已有openCARP远端LAT={refact[-1]:.6f}ms。没有重跑openCARP。0.125mm在重合节点上直接取细网格参考值，不做插值；这不是与openCARP同粗网格误差的比较。',
'两种AFSEI方法均逐节点计算GPB，集中质量不取消CN隐式求解。恒定电导和线性四面体下两组K相同，主要差别在质量/电流源项积分。性能比较同时包括这种离散差异引起的迭代数差异。',
'openCARP的ODE/分裂格式等不同，因此与其差值是交叉验证差异，不能直接当解析误差。图中传播波形相位差会放大逐点Vm误差，应结合LAT、波形与空间分布。',
'每组visualization/ep.pvd保存51帧完整三维节点场，activation.vti为激活时间；时间s、坐标mm。',
'performance_comparison.png为两种AFSEI离散的性能；accuracy_comparison.png为对细网格openCARP差值；activation_comparison.png及activation_difference.png为共同色标激活图；voltage_comparison.png为同坐标波形；diagonal_activation.png为体对角线传播。']
for key in keys:
    ratio=metrics[key+'_nodal_vs_consistent']['PDE_consistent_over_nodal'];lines.append(f"{groups[key]['h']:g}mm：一致质量/nodal PDE时间比={ratio:.4f}；nodal耗时相对减少{(1-1/ratio)*100:.2f}%。")
(DEST/'ANALYSIS_zh.md').write_text('\n'.join(lines)+'\n')
(OUT/'analysis_completed.json').write_text(json.dumps(dict(status='completed'),indent=2));print('\n'.join(lines),flush=True)

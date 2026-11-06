from pathlib import Path
import json,csv,tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
R=Path('/mnt/large2/gjh/AFSEI');p=R/'results/tube_cg_euler_wave_20261031/full';out=p/'comparison'
r=json.loads((p/'report.json').read_text());cfg=json.loads((p/'config.resolved.json').read_text());history=json.loads((p/'history.json').read_text())
assert r['status']=='completed' and abs(r['time_s']-.6)<1e-12
d=np.load(p/'wave_sections.npz');t=d['time_s']*1000;x=d['x_cm']*10;xe=d['edge_x_cm']*10
closure=(1-d['gap_cm']/.6)*100;traction=d['traction_dyn_cm2'];v=d['voltage_mv']
def crossing(values,levels):
    levels=np.broadcast_to(levels,(values.shape[1],));a=np.full(values.shape[1],np.nan)
    for j in range(values.shape[1]):
        hits=np.flatnonzero(values[:,j]>=levels[j])
        if len(hits):
            k=int(hits[0]);a[j]=t[k] if k==0 else t[k-1]+(t[k]-t[k-1])*(levels[j]-values[k-1,j])/(values[k,j]-values[k-1,j])
    return a
def fit(xx,yy,mask):
    valid=mask&np.isfinite(yy);coef=np.polyfit(xx[valid],yy[valid],1);pred=np.polyval(coef,xx[valid]);ss=float(np.sum((yy[valid]-yy[valid].mean())**2))
    return dict(slope_ms_per_mm=float(coef[0]),r_squared=1-float(np.sum((pred-yy[valid])**2))/ss,nodes=int(valid.sum()))
electrical=crossing(v,0);force10=crossing(traction,.1*traction.max(0));force50=crossing(traction,.5*traction.max(0));mechanical=crossing(closure,5.)
inactive=traction.max(0)<=1e-12;force10[inactive]=np.nan;force50[inactive]=np.nan
active=(x>=3)&(x<=21);edgeactive=(xe>=3)&(xe<=21)
peak=np.unravel_index(int(closure.argmax()),closure.shape)
mins=np.array([row['minimum_gap_cm'] for row in history]);power=np.array([abs(row.get('transfer_power_abs_error_erg_s',0)) for row in history]);rel=np.array([row.get('transfer_power_relative_error',0) for row in history]);power_scale=np.array([max(abs(row.get('transfer_solid_power_erg_s',0)),abs(row.get('transfer_fluid_power_erg_s',0))) for row in history]);relevant=power_scale>1
volume0=history[0]['budget_initial_moving_volume_ml'];volume=np.array([row['budget_moving_lumen_volume_ml'] for row in history]);imbalance=np.array([row['budget_fixed_control_imbalance_ml'] for row in history])
stats=dict(status='completed',duration_ms=float(t[-1]),frames=len(t),fsi_steps=r['accepted_steps'],ep_steps=r['ep_steps'],
    elapsed_s=r['elapsed_seconds'],elapsed_minutes=r['elapsed_seconds']/60,seconds_per_step=r['seconds_per_step_excluding_first_10'],shared_gpu=True,
    electrical_first_last_ms=[float(electrical[0]),float(electrical[-1])],
    electrical_fit=fit(x,electrical,active),force10_fit=fit(xe,force10,edgeactive),force50_fit=fit(xe,force50,edgeactive),
    max_gap_reduction_percent=float(closure.max()),minimum_sampled_gap_mm=float(d['gap_cm'].min()*10),
    peak_contraction_time_ms=float(t[peak[0]]),peak_contraction_reference_x_mm=float(x[peak[1]]),
    max_expansion_percent=float(-closure.min()),minimum_lumen_volume_ml=float(volume.min()),initial_lumen_volume_ml=float(volume0),
    max_volume_reduction_percent=float((1-volume.min()/volume0)*100),final_minimum_gap_mm=float(r['final']['minimum_gap_cm']*10),
    final_mean_gap_mm=float(r['final']['mean_gap_cm']*10),final_vm_range_mv=[r['final']['voltage_min_mv'],r['final']['voltage_max_mv']],
    maximum_sampled_divergence=r['max_sampled_divergence_s_inv'],max_transfer_abs_power_error_erg_s=float(power.max()),
    raw_max_transfer_relative_error=float(rel.max()),max_transfer_relative_error_when_power_gt1erg_s=float(rel[relevant].max()),
    max_integrated_volume_budget_imbalance_ml=float(abs(imbalance).max()),max_budget_imbalance_percent_initial_volume=float(abs(imbalance).max()/volume0*100),
    final_volume_budget_imbalance_ml=float(imbalance[-1]),final_volume_budget_imbalance_percent_initial_volume=float(abs(imbalance[-1])/volume0*100),
    max_rc_pressure_mmhg=float(max(row['rc_pressure_mmhg'] for row in history)),min_rc_pressure_mmhg=float(min(row['rc_pressure_mmhg'] for row in history)),
    net_rc_outlet_ml=r['final']['rc_integrated_outlet_ml'],
    time_sampling_note='Activation/force/mechanical crossing times interpolated from2ms output; precision is not20us of EP solve.')
probes=[]
for desired in [3,5,7,10,12,15,18,21]:
    j=int(abs(x-desired).argmin());k=int(abs(xe-desired).argmin())
    probes.append(dict(x_mm=float(x[j]),ep_ms=float(electrical[j]),force10_ms=float(np.interp(x[j],xe,force10)),force50_ms=float(np.interp(x[j],xe,force50)),
        gap5_ms=float(mechanical[j]),peak_gap_reduction_percent=float(closure[:,j].max()),peak_gap_time_ms=float(t[closure[:,j].argmax()])))
stats['probes']=probes
# A readable early propagation window, retaining the full-duration existing plot.
fig,axes=plt.subplots(3,1,figsize=(11,9),layout='constrained',sharex=True)
keep=t<=250
for ax,xx,val,label,cmap in zip(axes,[x,xe,x],[v,traction,closure],['Vm [mV]','Actual inward traction [dyn/cm2]','Gap reduction [%]'],['turbo','magma','viridis']):
    im=ax.pcolormesh(xx,t[keep],val[keep],shading='auto',cmap=cmap);fig.colorbar(im,ax=ax,label=label);ax.set_ylabel('Time [ms]')
axes[0].plot(x,electrical,'w--',lw=1,label='Vm=0 crossing');axes[0].legend(loc='upper left',fontsize=8)
axes[1].plot(xe,force10,'w--',lw=1,label='10% local peak traction');axes[1].legend(loc='upper left',fontsize=8)
axes[2].plot(x,mechanical,'w--',lw=1,label='5% gap reduction');axes[2].legend(loc='upper left',fontsize=8)
axes[2].set_xlabel('Reference axial position [mm]');fig.suptitle('Completed 600ms EP-FSI simulation | early 0-250ms response\nCG P1 + CN / GPB-Land Euler / Neo-Hookean P2 / MAC-IB')
for ext in ('png','pdf'):fig.savefig(out/('early_electromechanical_wave.'+ext),dpi=170)
plt.close(fig)
# Actual P2 node positions, shown with a four-triangle subdivision only for plotting.
s=np.load(p/'snapshots.npz');cells=s['cells'];reference=s['reference'];n=int(s['vertex_count']);times=s['times']*1000
sub=cells[:,[[0,3,4],[3,1,5],[4,5,2],[3,5,4]]].reshape(-1,3)
fig,axes=plt.subplots(2,4,figsize=(16,7),layout='constrained')
for ax,target in zip(axes.flat,[0,40,80,120,180,250,400,600]):
    k=int(abs(times-target).argmin());pos=s['x'][k]*10;pos=pos-np.array([4.,8.]);voltage=np.empty(len(reference));voltage[:n]=s['voltage'][k]
    for a,b,mid in [(0,1,3),(0,2,4),(1,2,5)]:voltage[cells[:,mid]]=.5*(s['voltage'][k,cells[:,a]]+s['voltage'][k,cells[:,b]])
    im=ax.tripcolor(Triangulation(pos[:,0],pos[:,1],sub),voltage,shading='gouraud',cmap='turbo',vmin=-90,vmax=60)
    ax.axhline(3,color='gray',ls='--',lw=.6);ax.axhline(-3,color='gray',ls='--',lw=.6)
    ax.set(title=f'{times[k]:g} ms',xlabel='x [mm]',ylabel='y [mm]',xlim=(0,24),ylim=(-4.7,4.7));ax.set_aspect('equal')
fig.colorbar(im,ax=axes,shrink=.8,label='Vm [mV]');fig.suptitle('Actual moving wall geometry (no displacement exaggeration)\nDashed lines: initial lumen walls | color: interpolated P1 voltage')
for ext in ('png','pdf'):fig.savefig(out/('wall_deformation_sequence.'+ext),dpi=170)
plt.close(fig)
stats['max_sampled_solid_vertical_displacement_mm']=float(abs(s['x'][:,:,1]-reference[None,:,1]).max()*10)
(out/'completed_analysis.json').write_text(json.dumps(stats,indent=2))
with (out/'arrival_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(probes[0]));w.writeheader();w.writerows(probes)
lines=['# 600ms二维EP-FSI完成结果','',f"完整计算{r['accepted_steps']}个FSI步，{r['ep_steps']}个EP步，耗时{stats['elapsed_minutes']:.2f}min，平均{r['seconds_per_step_excluding_first_10']:.5f}s/FSI步。共享GPU1，不作为独占性能比较。",'',
    f"电激活（Vm=0，2ms采样插值）：首端{electrical[0]:.3f}ms，末端{electrical[-1]:.3f}ms。最大采样截面间隙缩小{closure.max():.3f}%，最小间隙{stats['minimum_sampled_gap_mm']:.4f}mm，发生在{stats['peak_contraction_time_ms']:.1f}ms、参考x={stats['peak_contraction_reference_x_mm']:.1f}mm。",'',
    f"最大单侧固体垂向位移{stats['max_sampled_solid_vertical_displacement_mm']:.4f}mm。600ms最小/平均间隙分别{stats['final_minimum_gap_mm']:.4f}/{stats['final_mean_gap_mm']:.4f}mm；电位范围{stats['final_vm_range_mv']}mV。",'',
    '位移不是电位波前的逐点平移。固定端、流体压力和弹性非局部传递会使局部先收缩或膨胀；应分别读电激活、实际牵引和间隙阈值曲线。',
    '本例保持二维等长环向配对牵引简化，不是完整三维环向肌肉，也没有瓣膜，不能据此宣称有效单向泵送或文献定量复现。',
    f"最大采样散度{stats['maximum_sampled_divergence']:.3e}s^-1。原报告相对功率误差最大1来自近零功率时刻；有效功率>1erg/s时最大比值{stats['max_transfer_relative_error_when_power_gt1erg_s']:.3e}，绝对误差峰值{stats['max_transfer_abs_power_error_erg_s']:.3e}erg/s。",
    f"体积预算积分偏差峰值为初始体积的{stats['max_budget_imbalance_percent_initial_volume']:.4f}%，终值{stats['final_volume_budget_imbalance_percent_initial_volume']:.4f}%。该偏差不是已证明的壁面泄漏，但也不应称为严格离散体积守恒。",'',
    '位置时序（2ms输出插值）：','| x mm | EP ms | 牵引10% ms | 间隙缩小5% ms | 局部峰值缩小% |','|---:|---:|---:|---:|---:|']
for q in probes:lines.append(f"| {q['x_mm']:.1f} | {q['ep_ms']:.2f} | {q['force10_ms']:.2f} | {q['gap5_ms']:.2f} | {q['peak_gap_reduction_percent']:.2f} |")
(out/'COMPLETED_ANALYSIS_zh.md').write_text('\n'.join(lines)+'\n')
with tarfile.open(p/'completed_review.tar.gz','w:gz') as tar:
    for name in ['electromechanical_wave.png','wave_probes.png','early_electromechanical_wave.png','wall_deformation_sequence.png','completed_analysis.json','COMPLETED_ANALYSIS_zh.md','arrival_summary.csv']:
        tar.add(out/name,arcname=name)
print(json.dumps(stats,indent=2),flush=True)

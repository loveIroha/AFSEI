"""Actual solved voltage, traction and gap; never prescribe a travelling force."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);a=ap.parse_args();p=a.source
data=np.load(p/'wave_sections.npz');t=data['time_s']*1000;x=data['x_cm']*10;xe=data['edge_x_cm']*10
cfg=json.loads((p/'config.resolved.json').read_text());ref=cfg['geometry']['lumen_height_cm'];closure=(1-data['gap_cm']/ref)*100
out=p/'comparison';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(3,1,figsize=(11,10),layout='constrained',sharex=True)
for ax,xx,v,label,cmap in zip(axes,[x,xe,x],[data['voltage_mv'],data['traction_dyn_cm2'],closure],
    ['Vm [mV]','Actual inward traction [dyn/cm2]','Lumen-gap reduction [%]'],['turbo','magma','viridis']):
    im=ax.pcolormesh(xx,t,v,shading='auto',cmap=cmap);fig.colorbar(im,ax=ax,label=label);ax.set_ylabel('Time [ms]')
axes[-1].set_xlabel('Reference axial position [mm]');fig.suptitle('Solved EP activation -> Land traction -> Neo-Hookean wall / MAC-IB response')
for ext in ('png','pdf'):fig.savefig(out/('electromechanical_wave.'+ext),dpi=170)
plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');probe=np.rint(np.array([.15,.35,.55,.75,.9])*(len(x)-1)).astype(int)
for j in probe:
    axes[0].plot(t,data['voltage_mv'][:,j],label=f'x={x[j]:.1f}mm')
    axes[1].plot(t,data['calcium_um'][:,j]);axes[2].plot(t,closure[:,j])
for ax,title,ylabel in zip(axes,['Electrical propagation','Calcium response','Wall contraction'],['Vm [mV]','Cai [uM]','Gap reduction [%]']):ax.set(title=title,xlabel='Time [ms]',ylabel=ylabel);ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
for ext in ('png','pdf'):fig.savefig(out/('wave_probes.'+ext),dpi=170)
plt.close(fig)
def first_cross(values,level):
    arr=np.full(values.shape[1],np.nan)
    for j in range(values.shape[1]):
        above=np.flatnonzero(values[:,j]>=level)
        if len(above):
            k=above[0]
            arr[j]=t[k] if k==0 else t[k-1]+(t[k]-t[k-1])*(level-values[k-1,j])/(values[k,j]-values[k-1,j])
    return arr
ep=first_cross(data['voltage_mv'],0);mechanics=first_cross(closure,5)
np.savetxt(out/'propagation_times.csv',np.column_stack((x,ep,mechanics,mechanics-ep)),delimiter=',',header='reference_x_mm,Vm0_crossing_ms,gap5percent_crossing_ms,delay_ms',comments='')
report=dict(time_ms=float(t[-1]),frames=len(t),max_gap_reduction_percent=float(closure.max()),
    EP_section_activated_fraction=float(np.isfinite(ep).mean()),gap5percent_section_fraction=float(np.isfinite(mechanics).mean()),
    interpretation='Traction follows GPB/Land; displacement is the coupled FSI response. Elastic/fluid communication can precede local EP activation; a mechanical wave is not guaranteed merely by the load direction.')
(out/'wave_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

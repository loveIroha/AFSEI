"""Same-mesh full-trajectory time control after fixed-dt spatial acceptance."""
from pathlib import Path
import json
import numpy as np,meshio
from analyze_accuracy_campaign import collect,read_vti,setup,save
from analyze_available_volume_results import measure
root=Path('/mnt/large2/gjh/AFSEI/results/accuracy_performance_20261025')
baseline=root/'solid3_mac128';fine=root/'time_control/solid3_mac128_dt20us'

def main():
    if json.loads((fine/'status.json').read_text()).get('stage')!='completed':raise RuntimeError('time control not completed')
    a,b=setup(baseline,'baseline'),setup(fine,'time-refined')
    if not np.array_equal(a['geo']['X'],b['geo']['X']) or not np.array_equal(a['geo']['cells'],b['geo']['cells']):raise ValueError('time control changed mesh')
    sa,sb=collect(baseline,'solid'),collect(fine,'solid');fa,fb=collect(baseline,'fluid'),collect(fine,'fluid')
    common=sorted(set(sa)&set(sb)&set(fa)&set(fb))
    if not common or abs(common[-1]-.3)>1e-8:raise ValueError('full common 300ms fields required')
    cells=a['geo']['cells'];V=a['geo']['vol'];rows=[]
    def norm(value):
        q=value[cells];return float(np.sqrt(np.sum(V/20*(np.sum(q.sum(1)**2,axis=1)+np.sum(q*q,axis=(1,2))))))
    def lumen(q,geo):
        X=q-q[geo['rings']].mean((0,1));tri=X[geo['inner_faces']]
        volume=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()
        for ring,sign in ((geo['rings'][0],-1.),(geo['rings'][-1],1.)):
            p=X[ring];center=p.mean(0);volume+=sign*np.sum(np.einsum('ij,ij->i',np.broadcast_to(center,p.shape),np.cross(p,np.roll(p,-1,axis=0))))
        return volume/6
    for t in common:
        ma,mb=meshio.read(sa[t]),meshio.read(sb[t]);ua=ma.points-a['geo']['X'];ub=mb.points-b['geo']['X']
        qa=measure(ma.points,ma.cell_data_dict['J_min']['tetra'],a['geo'],3)
        qb=measure(mb.points,mb.cell_data_dict['J_min']['tetra'],b['geo'],3)
        row=dict(time_ms=t*1000,displacement_error=norm(ua-ub),displacement_reference=norm(ub),
            diameter_relative_error=abs(qa['sampled_min_normal_diameter_mm']/qb['sampled_min_normal_diameter_mm']-1),
            lumen_relative_error=abs(lumen(ma.points,a['geo'])/lumen(mb.points,b['geo'])-1))
        da,_,h=read_vti(fa[t]);db,_,_=read_vti(fb[t])
        for key,label in (('velocity_cm_per_s','velocity'),('pressure_dyn_per_cm2','pressure')):
            row[label+'_error']=float(np.sqrt(np.sum((da[key]-db[key])**2)*np.prod(h)))
            row[label+'_reference']=float(np.sqrt(np.sum(db[key]**2)*np.prod(h)))
        rows.append(row)
    metrics={label:max(r[label+'_error'] for r in rows)/max(max(r[label+'_reference'] for r in rows),1e-30) for label in ('displacement','velocity','pressure')}
    metrics.update(diameter=max(r['diameter_relative_error'] for r in rows),lumen=max(r['lumen_relative_error'] for r in rows))
    passed=all(metrics[k]<=.05 for k in ('displacement','velocity','pressure')) and metrics['diameter']<=.02 and metrics['lumen']<=.02
    save(root/'time_control/comparison.json',dict(completed=True,time_control_passed=passed,metrics=metrics,curve=rows,
        dt_coarse_s=5e-5,dt_fine_s=2e-5,from_rest=True,full_space_time_discrete_controls_passed=passed,
        limitations=['Finite reference is not continuum truth.','Conditional on the fixed coarse EP mesh and stated model/window.','No independent software speed claim follows from this report.']))
    print(json.dumps(dict(time_control_passed=passed,metrics=metrics),indent=2))

if __name__=='__main__':main()

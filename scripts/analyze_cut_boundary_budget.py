"""Offline exact clipped-cap storage budget; no PDE/ODE or time advance."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from afsi_torch.p1 import cavity_rim


def clip(vertices,axis,value,keep_high):
    result=[]
    if not vertices:return result
    for p,q in zip(vertices,vertices[1:]+vertices[:1]):
        pin=p[axis]>=value if keep_high else p[axis]<=value
        qin=q[axis]>=value if keep_high else q[axis]<=value
        if pin:result.append(p)
        if pin!=qin:
            t=(value-p[axis])/(q[axis]-p[axis])
            result.append(p+t*(q-p))
    return result


def tract_volume(x,rim,bounds,cut,axis):
    other=[a for a in range(3) if a!=axis]
    center=x[rim[:,0]].mean(0)
    area=0.; volume=0.
    for a,b in rim:
        poly=[x[a],x[b],center]
        for k,(lo,hi) in zip(other,bounds):
            poly=clip(poly,k,lo,True)
            poly=clip(poly,k,hi,False)
        for j in range(1,len(poly)-1):
            tri=np.array([poly[0],poly[j],poly[j+1]])
            e=tri[1:,other]-tri[0,other]
            ar=abs(e[0,0]*e[1,1]-e[0,1]*e[1,0])/2
            area+=ar
            volume+=ar*(cut-tri[:,axis].mean())
    return volume,area


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    report=json.loads((args.output/'report.json').read_text())
    with np.load(report['source'],allow_pickle=False) as a:
        faces=torch.as_tensor(a['faces'][a['facet_tags']==1])
        rim=cavity_rim(faces).numpy()
        meta=json.loads(str(a['metadata']))
        dt=meta['config']['fsi']['time']['dt']
    with np.load(args.output/'discarded_trial.npz',allow_pickle=False) as a:
        old=a['old_x'];new=a['trial_x']
    geom=report['tract_geometry'];result=[]
    for bounds in geom['tract_bounds_cm']:
        before,area0=tract_volume(old,rim,bounds,geom['cut_coordinate_cm'],geom['axis'])
        after,area1=tract_volume(new,rim,bounds,geom['cut_coordinate_cm'],geom['axis'])
        exact=(bounds[0][1]-bounds[0][0])*(bounds[1][1]-bounds[1][0])
        if max(abs(area0-exact),abs(area1-exact))>1e-9:
            raise RuntimeError('cap does not cover the fixed tract cross-section')
        result.append(dict(volume_old_ml=before,volume_trial_ml=after,
            volume_rate_ml_s=(after-before)/dt,covered_area_cm2=area1))
    active=0 if report['mode']=='filling' else 1
    f=report['failed_trial'];b=f['tract_budget'];storage=result[active]['volume_rate_ml_s']
    total=(f['finite_step_mass_error_ml_s']-f['basal_flux_ml_s']+storage+b['cut_outward_ml_s'])
    evidence=dict(dt_s=dt,tracts=result,active_tract_storage_ml_s=storage,
        core_plus_active_tract_mass_error_ml_s=total,
        fixed_tract_fv_mass_error_ml_s=b['fixed_tract_mass_error_ml_s'],
        direct_cut_minus_relative_basal_ml_s=b['cut_outward_ml_s']-f['basal_flux_ml_s'],
        note='moving-cap storage included; remaining error is not relabeled physical pressure loss')
    (args.output/'storage_budget.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()

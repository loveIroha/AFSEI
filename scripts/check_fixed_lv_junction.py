"""Audit a fixed junction matching the loaded LV rim; no new FSI run.

Uses accepted coordinates from a prior, different circulation model only as
motion samples. Rim mismatch is geometry, not a measured fluid leak.
"""
import argparse,json
from pathlib import Path
import numpy as np


def ordered(edges):
    adjacent={}
    for a,b in edges:
        adjacent.setdefault(int(a),[]).append(int(b));adjacent.setdefault(int(b),[]).append(int(a))
    if any(len(v)!=2 for v in adjacent.values()):raise ValueError('non-manifold rim')
    ids=[min(adjacent)];previous=None
    while True:
        nxt=next(v for v in adjacent[ids[-1]] if v!=previous)
        if nxt==ids[0]:break
        if nxt in ids:raise ValueError('multiple rim loops')
        previous=ids[-1];ids.append(nxt)
    if len(ids)!=len(adjacent):raise ValueError('disconnected rim')
    return np.asarray(ids)


def radial_samples(poly,center,count=3600):
    a=poly[:,:2]-center;e=np.roll(a,-1,axis=0)-a
    angles=(np.arange(count)+.37)*2*np.pi/count
    d=np.stack((np.cos(angles),np.sin(angles)),-1)
    cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
    denom=cross(d[:,None],e[None]);safe=np.where(abs(denom)>1e-14,denom,np.nan)
    t=cross(a,e)[None]/safe;s=cross(a[None],d[:,None])/safe
    valid=(t>0)&(s>=0)&(s<1)
    if not np.all(valid.sum(1)==1):raise ValueError('rim not star-shaped about chosen center')
    return np.min(np.where(valid,t,np.inf),axis=1)


def distance_to_rim(points,poly):
    a=poly;e=np.roll(a,-1,axis=0)-a
    v=points[:,None]-a[None]
    t=np.clip((v*e).sum(-1)/(e*e).sum(-1),0,1)
    return np.linalg.norm(v-t[...,None]*e,axis=-1).min(1)


def run(source,history,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    d=np.load(source);X=d['X'];ids=ordered(d['inner']);base=np.unique(d['base'])
    center=X[ids,:2].mean(0);c=X[base].mean(0)
    _,_,vh=np.linalg.svd(X[base]-c,full_matrices=False);normal=vh[-1]
    if normal[2]<0:normal=-normal
    paths=sorted((history/'phase_checkpoints').glob('*.npz'))
    if not paths:raise ValueError('missing accepted motion samples')
    anchor=np.load(paths[0])['x'];fixed=anchor[ids];r0=radial_samples(fixed,center)
    # Independent circular scaling check for the polar area diagnostic.
    theta=np.arange(720)*2*np.pi/720
    circle=np.stack((np.cos(theta),np.sin(theta),0*theta),1)
    ra=radial_samples(circle,np.zeros(2));rb=radial_samples(circle*1.2,np.zeros(2))
    observed=np.pi*np.mean(abs(rb*rb-ra*ra))
    assert abs(observed-np.pi*(1.2**2-1))<2e-5
    assert distance_to_rim(circle,circle).max()<1e-12
    radial=np.column_stack((X[base,:2]-np.array([7.30042735,7.6448494]),np.zeros(len(base))))
    radial/=np.linalg.norm(radial,axis=1)[:,None]
    rows=[];fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(fixed[:,0]*10,fixed[:,1]*10,'k--',label='Fixed mouth: first loaded sample')
    for path in paths:
        q=np.load(path);x=q['x']
        if x.shape!=X.shape:raise ValueError('motion sample mesh differs')
        if 'X' in q and not np.array_equal(q['X'],X):
            if not np.allclose(q['X'],X,rtol=0,atol=1e-12):raise ValueError('reference coordinates differ')
        r=radial_samples(x[ids],center);dist=distance_to_rim(x[ids],fixed)
        meta=json.loads(str(q['metadata']));time=meta.get('time_s',meta.get('time',None))
        name=path.stem
        rows.append(dict(sample=name,time_s=time,max_rim_mismatch_mm=float(dist.max()*10),
            projected_symmetric_difference_mm2=float(np.pi*np.mean(abs(r*r-r0*r0))*100),
            max_baseline_plane_departure_mm=float(abs((x[base]-anchor[base])@normal).max()*10),
            max_basal_z_displacement_mm=float(abs((x-X)[base,2]).max()*10)))
        axes[0].plot(x[ids,0]*10,x[ids,1]*10,label=name.split('_')[0])
    axes[0].set(xlabel='x (mm)',ylabel='y (mm)',title='Actual inner rim vs fixed matching mouth')
    axes[0].axis('equal');axes[0].legend(fontsize=8)
    axes[1].bar(np.arange(len(rows)),[v['max_rim_mismatch_mm'] for v in rows])
    axes[1].set(xlabel='Accepted motion sample',ylabel='Maximum rim mismatch (mm)',title='Geometric mismatch; not measured leakage')
    axes[1].set_xticks(np.arange(len(rows)),[str(i) for i in range(len(rows))])
    fig.tight_layout();fig.savefig(out/'fixed_junction_audit.png',dpi=180);plt.close(fig)
    report=dict(scope='fixed rim-matching junction geometry audit; no connected fluid solve',
        source_reference=str(source),motion_history=str(history),anchor=str(paths[0]),
        source_nodes=len(X),source_cells=len(d['cells']),base_normal=normal.tolist(),
        basal_radial_direction_max_normal_component=float(abs(radial@normal).max()),
        horizontal_cut_geometric_range_mm=float(np.ptp(X[base,2])*10),
        max_reference_plane_error_mm=float(abs((X[base]-c)@normal).max()*10),
        samples=rows,geometry_tests_passed=True,
        fixed_rim_matching_connection_passed=all(v['max_rim_mismatch_mm']<1e-3 for v in rows),
        gate_tolerance_mm=1e-3,
        limitation='1 micrometre is a geometry contact diagnostic, not a derived fluid tolerance; no leakage, pressure or EF claim',
        full_cycle_started=False)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--history',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.history,a.output)

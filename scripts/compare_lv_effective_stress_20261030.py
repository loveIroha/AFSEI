"""Reconstruct solid-only effective stress from saved CG1 VTK fields, no solve."""
import json,csv,itertools
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import meshio

root=Path('/mnt/large2/gjh/AFSEI')
source=Path('/mnt/large2/gjh/test_ele_soild_heart-test/build_test/results_iso14_20261004_182408')
snapshot=root/'results/afsei_source_timing_neck_two_cycles_20261029/phase_checkpoints/02_beat1_ejection_150ms_sample_tick00013199.npz'
with np.load(snapshot) as a:X=a['X'];cells=a['cells'];fiber=a['fiber']
keys={tuple(np.round(v*1e4).astype(int)):i for i,v in enumerate(X)}
cellids={tuple(sorted(c)):i for i,c in enumerate(cells)}
def indices(points):
    ids=[]
    for pt in points:
        key=tuple(np.round(pt*1e4).astype(int));idx=keys.get(key)
        if idx is None or abs(X[idx]-pt).max()>1e-6:
            matches=[keys[k] for off in itertools.product((-1,0,1),repeat=3)
                if (k:=tuple(key[j]+off[j] for j in range(3))) in keys and abs(X[keys[k]]-pt).max()<=1e-6]
            if len(matches)!=1:raise ValueError('ambiguous reference mesh mapping')
            idx=matches[0]
        ids.append(idx)
    return np.array(ids)

def files(field,t):
    ds=ET.parse(source/(field+'.pvd')).getroot().findall('.//DataSet')
    step=min(ds,key=lambda v:abs(float(v.attrib['timestep'])*1000-t))
    assert abs(float(step.attrib['timestep'])*1000-t)<1e-5
    pv=source/step.attrib['file']
    return [pv.parent/p.attrib['Source'] for p in ET.parse(pv).getroot().findall('.//Piece')]

pv=list(csv.DictReader((source/'pv.csv').open()))
stats=list(csv.DictReader((source/'statistics.csv').open()))
events=[dict(t=float(r['time_ms']),phase=r['solved_phase'],next=r['next_phase'],V=float(r['V_LV_mL']))
        for r in pv if r['solved_phase']!=r['next_phase'] and int(r['beat'])==1]
reports=[]
for t in [580,660,750]:
    seen=set();sums=np.zeros(5);Jmin=1e9;Jmax=-1e9;maxerr=0
    for tf,df in zip(files('Ta',t),files('displacement',t)):
        ta=meshio.read(tf);disp=meshio.read(df)
        assert np.array_equal(ta.points,disp.points)
        ids=indices(ta.points);maxerr=max(maxerr,float(abs(X[ids]-ta.points).max()))
        top=ta.cells_dict['tetra'];keep=[];orig=[]
        for j,tet in enumerate(ids[top]):
            k=tuple(sorted(tet));ci=cellids[k]
            if ci not in seen:keep.append(j);orig.append(ci);seen.add(ci)
        top=top[keep];orig=np.array(orig)
        if not len(top):continue
        xx=X[ids[top]];yy=xx+disp.point_data['displacement'][top]
        dx=xx[:,1:]-xx[:,:1];dy=yy[:,1:]-yy[:,:1]
        F=np.linalg.solve(dx,dy).transpose(0,2,1);J=np.linalg.det(F)
        vol=abs(np.linalg.det(dx))/6;lam=np.linalg.norm(np.einsum('eij,ej->ei',F,fiber[orig]),axis=1)
        tension=ta.point_data['Ta'][top].mean(1)
        multiplier=1+4.9*(np.maximum(lam,1)-1)
        sums+=np.array([vol.sum(),(vol*J).sum(),(vol*lam).sum(),(vol*tension).sum(),
                       (vol*tension*multiplier*lam**2).sum()])
        Jmin=min(Jmin,J.min());Jmax=max(Jmax,J.max())
    if len(seen)!=len(cells):raise ValueError(f'missing cells: {len(seen)} vs {len(cells)}')
    stat=min(stats,key=lambda r:abs(float(r['time_ms'])-t))
    reports.append(dict(time_ms=t,cells=len(seen),coordinate_mapping_error_cm=maxerr,
        reference_mean_stretch=sums[2]/sums[0],reference_mean_ta_kpa=sums[3]/sums[0]/1e4,
        current_mean_active_fiber_cauchy_kpa=sums[4]/sums[1]/1e4,minimum_J=Jmin,maximum_J=Jmax,
        source_statistics=stat))
report=dict(scope='Source Ta is a saved CG1 projection, so effective stress is a reconstruction, not exact source quadrature stress.',
            source=str(source),events=events,frames=reports)
out=root/'results/lv_basal_audit_20261030/source_stress.json'
out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

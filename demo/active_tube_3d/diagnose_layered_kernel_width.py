"""Kernel-width sensitivity on ONE unchanged saved RT0 field.

Fine arrays are exact RT0 subdivisions, not newly solved fluid states.
This diagnoses delta smoothing; it is explicitly not mesh convergence.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from afsi_torch.mac.grid import MACGrid, divergence
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube
from diagnose_layered_transfer import raw_delta_velocity


def prolong(velocity, factor):
    result = []
    for c,field in enumerate(velocity):
        v = field
        n = field.shape[c]-1
        for a in range(3):
            if a != c:
                v = v.repeat_interleave(factor,dim=a)
        q = torch.arange(n*factor+1,device=v.device,dtype=v.dtype)/factor
        left = q.floor().long().clamp(max=n-1)
        shape = [1]*3;shape[c] = len(q)
        t = (q-left).reshape(shape)
        result.append((1-t)*v.index_select(c,left)+t*v.index_select(c,left+1))
    return tuple(result)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--exact-diagnosis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('choose an empty output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    c = json.loads((a.case/'config.resolved.json').read_text())
    g,_,_,_ = resolve_config(c)
    mesh = generate_circular_tube(g,device='cuda')
    grid = MACGrid(tuple(c['fluid']['shape']),tuple(c['fluid']['lengths_cm']),tuple(c['fluid']['origin_cm']))
    with np.load(a.case/'final_state.npz') as s:
        old = torch.from_numpy(s['previous_x'].copy()).to(mesh.X)
        u = tuple(torch.from_numpy(s[k].copy()).to(mesh.X) for k in ('u','v','w'))
    report = json.loads((a.exact_diagnosis/'report.json').read_text())
    if Path(report['case']).resolve() != a.case.resolve():
        raise ValueError('exact diagnosis refers to a different state')
    exact = report['exact_rt0_wall_flux_ml_s']
    with (a.exact_diagnosis/'axial_flux.csv').open() as f:
        exact_axial = [float(r['exact_rt0_ml_s']) for r in csv.DictReader(f)]
    ids = mesh.inner_faces
    tri = old[ids]
    z = mesh.X[ids,2].mean(1)-g.axial_start_cm
    zones = (z/(g.length_cm/g.axial_elements)).floor().long().clamp(0,g.axial_elements-1)
    coarse_div = divergence(u,grid.spacing)
    all_rows = []
    for factor in (1,2,4):
        fine = MACGrid(tuple(factor*n for n in grid.shape),grid.lengths,grid.origin)
        v = u if factor == 1 else prolong(u,factor)
        expected = coarse_div
        for axis in range(3):
            expected = expected.repeat_interleave(factor,dim=axis)
        div_error = float((divergence(v,fine.spacing)-expected).abs().max())
        scale = max(float(coarse_div.abs().max()),max(float(t.abs().max()) for t in u)/min(grid.spacing))
        if div_error > 1e-11*max(scale,1e-20):
            raise AssertionError('RT0 subdivision changed divergence')
        del expected
        for level in (3,4):
            t = tri
            for _ in range(level):
                x,y,z = t.unbind(1)
                xy,yz,zx = (x+y)/2,(y+z)/2,(z+x)/2
                t = torch.stack([torch.stack(w,1) for w in
                    ((x,xy,zx),(xy,y,yz),(zx,yz,z),(xy,yz,zx))],1).reshape(-1,3,3)
            bary = old.new_tensor([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
            points = torch.einsum('qa,eai->eqi',bary,t).reshape(-1,3)
            values = torch.cat([raw_delta_velocity(fine,v,q) for q in points.split(16384)])
            values = values.reshape(len(t),3,3).mean(1)
            areas = torch.linalg.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])*.5
            face_flux = (values*areas).sum(-1).reshape(len(ids),4**level).sum(1)
            axial = old.new_zeros(g.axial_elements).index_add_(0,zones,face_flux)
            total = float(axial.sum())
            row = dict(factor=factor,surface_level=level,diagnostic_h_um=fine.spacing[0]*1e4,
                wall_flux_ml_s=total,exact_rt0_flux_ml_s=exact,gap_ml_s=total-exact,
                relative_gap=abs(total-exact)/abs(exact),rt0_divergence_preservation_max_s_inv=div_error,
                axial_flux_ml_s=axial.cpu().tolist(),exact_axial_flux_ml_s=exact_axial)
            all_rows.append(row);print(json.dumps({k:v for k,v in row.items() if 'axial' not in k}),flush=True)
        del v
    result = dict(scope='kernel-width diagnostic on identical RT0 velocity; no fine-grid PDE solve',
        full_mesh_convergence_test=False,production_solver_modified=False,rows=all_rows)
    (a.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    selected = [r for r in all_rows if r['surface_level']==4]
    axes[0].bar([str(r['factor']) for r in selected],[r['relative_gap']*100 for r in selected])
    axes[0].set(xlabel='Diagnostic subdivision factor',ylabel='Wall flux difference / exact RT0 (%)',
        title='Same saved flow; narrower Peskin kernel')
    z = (np.arange(g.axial_elements)+.5)*g.length_cm/g.axial_elements*1e4
    for r in selected:
        axes[1].plot(z,(np.array(r['axial_flux_ml_s'])-exact_axial)*1e9,label=f"h / {r['factor']}")
    axes[1].axhline(0,color='k',lw=.6);axes[1].legend()
    axes[1].set(xlabel='Axial position (micrometre)',ylabel='Flux difference (pL/s)',title='Spatial location of smoothing difference')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(a.output/'kernel_width.png',dpi=180);fig.savefig(a.output/'kernel_width.svg')


if __name__=='__main__':main()

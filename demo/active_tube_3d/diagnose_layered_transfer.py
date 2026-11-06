"""Saved-state IB flux decomposition; advances no PDE, ODE or mechanics.

RT0 surface integration clips triangles against MAC cells, then integrates
each affine cell field exactly. IB surface samples are diagnostics only;
the actual volume-quadrature/consistent-mass transfer is called unchanged.
"""
import argparse
import csv
import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.ep.layered_circular_tube import LayeredCircularTubeSolid
from afsi_torch.mac.paper_coupling import BEIBStepper
from afsi_torch.ib import peskin4
from afsi_torch.mac.sealed_ports import SealedDomain
from afsi_torch.simulation.tube_budget import TubeBudget


def clip(poly, axis, bound, lower):
    """Sutherland-Hodgman clipping with preserved polygon orientation."""
    out = []
    a = poly[-1]
    da = (a[axis]-bound)*(1 if lower else -1)
    for b in poly:
        db = (b[axis]-bound)*(1 if lower else -1)
        if (da >= 0) != (db >= 0):
            out.append(a+(b-a)*(da/(da-db)))
        if db >= 0:
            out.append(b)
        a, da = b, db
    return np.asarray(out)


def polygon_moment(poly):
    a = poly[0]
    vectors = np.cross(poly[1:-1]-a, poly[2:]-a)*.5
    areas = np.linalg.norm(vectors, axis=1)
    total = areas.sum()
    if total == 0:
        return None
    centers = (a+poly[1:-1]+poly[2:])/3
    return vectors.sum(0), (centers*areas[:, None]).sum(0)/total


def exact_rt0(triangles, velocity, grid):
    h, origin = np.asarray(grid.spacing), np.asarray(grid.origin)
    result = np.zeros(len(triangles))
    for face, tri in enumerate(triangles):
        # Clip in lattice coordinates. Snap roundoff-level grid-plane
        # coordinates, otherwise a cap can be counted in both adjacent cells.
        q = (tri-origin)/h
        q = np.where(np.abs(q-np.rint(q)) < 1e-11, np.rint(q), q)
        low = np.maximum(0, np.floor(q.min(0)).astype(int))
        high = np.minimum(np.asarray(grid.shape)-1,
                          np.floor(q.max(0)).astype(int))
        for index in product(*(range(a, b+1) for a, b in zip(low, high))):
            p = q.copy()
            for axis in range(3):
                for lower in (True, False):
                    p = clip(p, axis, index[axis]+int(not lower), lower)
                    if len(p) < 3:
                        break
                if len(p) < 3:
                    break
            if len(p) < 3:
                continue
            moment = polygon_moment(p*h)
            if moment is None:
                continue
            area, center = moment
            f = center/h-np.asarray(index)
            value = []
            for axis in range(3):
                right = list(index)
                right[axis] += 1
                value.append((1-f[axis])*velocity[axis][index]+f[axis]*velocity[axis][tuple(right)])
            result[face] += np.dot(value, area)
    return result


def verify_clipping(grid):
    # An oblique tetrahedron spanning many cells, with a global affine field.
    h = np.asarray(grid.spacing)
    x = np.asarray(grid.origin)+h*np.array([[20.2,20.4,20.1], [24.7,20.1,20.3],
                                         [20.5,24.8,20.2], [20.1,20.3,24.9]])
    faces = np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    velocity = []
    slope = np.array([.7,-.2,.4])
    for c in range(3):
        coords = [grid.face_origin(c)[a]+h[a]*np.arange(grid.face_shape(c)[a]) for a in range(3)]
        values = slope[c]*coords[c]
        shape = [1]*3
        shape[c] = len(values)
        velocity.append(np.broadcast_to(values.reshape(shape), grid.face_shape(c)))
    measured = exact_rt0(x[faces], velocity, grid).sum()
    expected = slope.sum()*abs(np.linalg.det((x[1:]-x[0]).T))/6
    relative = abs(measured-expected)/abs(expected)
    if relative > 1e-11:
        raise AssertionError(('affine divergence theorem check failed', measured, expected))
    return relative


def raw_delta_velocity(grid, velocity, points):
    """Independent oracle for the unchanged four-point Peskin gather."""
    offsets = torch.cartesian_prod(*(torch.arange(4,device=points.device) for _ in range(3)))
    result = []
    for c,u in enumerate(velocity):
        q = (points-points.new_tensor(grid.face_origin(c)))/points.new_tensor(grid.spacing)
        nodes = torch.floor(q-1).long()[:,None,:]+offsets[None]
        weights = peskin4(q[:,None,:]-nodes.to(points.dtype)).prod(-1)
        ids = (nodes[...,0]*u.shape[1]+nodes[...,1])*u.shape[2]+nodes[...,2]
        result.append((u.reshape(-1)[ids]*weights).sum(-1))
    return torch.stack(result,-1)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('choose an empty diagnostic output directory')
    a.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    c = json.loads((a.case/'config.resolved.json').read_text())
    g, fsi, _, _ = resolve_config(c)
    mesh = generate_circular_tube(g, device='cuda')
    solid = LayeredCircularTubeSolid(mesh, g,
        **{k:v for k,v in c['solid'].items() if k != 'end_condition'},
        **{k:v for k,v in c['active'].items() if k not in ('mechanism','rest_reference')},
        dt_s=fsi.time.dt)
    d = BEIBStepper(solid, fsi, 'cuda')
    with np.load(a.case/'final_state.npz') as s:
        get = lambda k: torch.from_numpy(s[k].copy()).to(mesh.X)
        old, new = get('previous_x'), get('x')
        u = tuple(get(k) for k in ('u','v','w'))
    domain = SealedDomain(d.grid, g, c['hydraulic_domain'], mesh.X, dimension=3)
    budget = TubeBudget(mesh, domain, initial_x=old)
    stencil = d.transfer.prepare(old)
    projected, info = d.transfer.interpolate(u, stencil)
    material = (new-old)/fsi.time.dt
    ids = mesh.inner_faces
    tri = old[ids]
    area = torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])*.5
    flux = lambda v: (v[ids].mean(1)*area).sum(-1)
    motion, fe_flux = flux(material), flux(projected)
    cpu = lambda v: v.detach().cpu().numpy()
    rt0 = exact_rt0(cpu(tri), tuple(cpu(v) for v in u), d.grid)
    print('Exact RT0 wall integration completed', flush=True)
    raw_fluxes = {}
    # Integrate smooth regularized-delta interpolation on the inner surface.
    for level in (1,2,3):
        t = tri
        for _ in range(level):
            x,y,z = t.unbind(1)
            xy,yz,zx = (x+y)/2,(y+z)/2,(z+x)/2
            t = torch.stack([torch.stack(v,1) for v in
                ((x,xy,zx),(xy,y,yz),(zx,yz,z),(xy,yz,zx))],1).reshape(-1,3,3)
        bary = old.new_tensor([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
        points = torch.einsum('qa,eai->eqi',bary,t).reshape(-1,3)
        values = []
        for chunk in points.split(16384):
            values.append(raw_delta_velocity(d.grid,u,chunk))
        values = torch.cat(values).reshape(len(t),3,3).mean(1)
        av = torch.linalg.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])*.5
        raw = (values*av).sum(-1).reshape(len(ids),4**level).sum(1)
        raw_fluxes[level] = cpu(raw)
        print(f'IB surface integration level={level} completed',flush=True)
    old_ext = budget.extend(old)
    connector = exact_rt0(cpu(old_ext[budget.connectors.indices]),tuple(cpu(v) for v in u),d.grid).sum()
    caps = exact_rt0(cpu(old_ext[budget.fixed_caps.indices]),tuple(cpu(v) for v in u),d.grid).sum()
    closed_flux = float(rt0.sum()+connector+caps)
    if abs(closed_flux) > 1e-8*max(abs(float(rt0.sum())),abs(float(caps)),1e-20):
        raise AssertionError(('closed RT0 surface fails saved divergence-free flux check',closed_flux))
    z = cpu(mesh.X[ids,2].mean(1))-g.axial_start_cm
    zones = np.clip(np.floor(z/(g.length_cm/g.axial_elements)).astype(int),0,g.axial_elements-1)
    rows = []
    m, fe = cpu(motion), cpu(fe_flux)
    for k in range(g.axial_elements):
        mask = zones == k
        rows.append(dict(axial_element=k,z_mid_um=(k+.5)*g.length_cm/g.axial_elements*1e4,
            material_ml_s=float(m[mask].sum()),projected_ib_ml_s=float(fe[mask].sum()),
            raw_ib_ml_s=float(raw_fluxes[3][mask].sum()),exact_rt0_ml_s=float(rt0[mask].sum()),
            nonlinear_gap_ml_s=float((m-fe)[mask].sum()),
            trace_projection_gap_ml_s=float((fe-raw_fluxes[3])[mask].sum()),
            smoothing_gap_ml_s=float((raw_fluxes[3]-rt0)[mask].sum())))
    with (a.output/'axial_flux.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    result = dict(scope='saved final accepted step; no PDE/ODE/time advance',case=str(a.case),
        equation_or_solver_modified=False,affine_clipping_relative_error=verify_clipping(d.grid),
        material_wall_flux_ml_s=float(m.sum()),projected_ib_wall_flux_ml_s=float(fe.sum()),
        raw_ib_wall_flux_ml_s={str(k):float(v.sum()) for k,v in raw_fluxes.items()},
        exact_rt0_wall_flux_ml_s=float(rt0.sum()),connector_rt0_ml_s=float(connector),
        fixed_cap_rt0_ml_s=float(caps),closed_rt0_flux_ml_s=closed_flux,
        material_minus_projected_norm_cm_s=float((material-projected).norm()),
        material_norm_cm_s=float(material.norm()),mass_solver_iterations=info.iterations,
        regions=rows,elapsed_seconds=perf_counter()-started,
        caveat='trace projection and smoothing terms can cancel; decomposition is not an independent mesh convergence test')
    (a.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(10,4.5),layout='constrained')
    for key,label in [('nonlinear_gap_ml_s','Accepted motion - FE IB'),
                      ('trace_projection_gap_ml_s','FE IB - raw delta interpolation'),
                      ('smoothing_gap_ml_s','Raw delta interpolation - exact RT0')]:
        ax.plot([r['z_mid_um'] for r in rows],[r[key]*1e9 for r in rows],marker='o',ms=3,label=label)
    ax.axhline(0,color='k',lw=.6);ax.set(xlabel='Axial position (micrometre)',
        ylabel='Flux difference (pL/s)',title='Saved-step spatial flux decomposition; no new simulation')
    ax.legend(fontsize=9);ax.grid(alpha=.25)
    fig.savefig(a.output/'axial_flux.png',dpi=180);fig.savefig(a.output/'axial_flux.svg')
    print(json.dumps({k:v for k,v in result.items() if k!='regions'},indent=2),flush=True)


if __name__ == '__main__':
    main()

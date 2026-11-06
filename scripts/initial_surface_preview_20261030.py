from pathlib import Path
import json,numpy as np,meshio
base=Path('/mnt/large2/gjh/AFSEI/results/ep_opencarp_sigma_nativeinit_20261030')
m=meshio.read(base/'3d/consistent_tc/ep_00000.vtu');X=m.points;C=m.cells_dict['tetra']
v=m.point_data['voltage_mV'];frames={}
for a in range(3):
    for side,value in enumerate([X[:,a].min(),X[:,a].max()]):
        on=np.isclose(X[:,a],value,rtol=0,atol=1e-12)
        cc=C[(on[C]).sum(1)>=3]
        ff=cc[:,np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])].reshape(-1,3)
        frames[f'{a}_{side}']=ff[np.all(on[ff],axis=1)]
faces=np.concatenate(list(frames.values()));ids,inv=np.unique(faces,return_inverse=True)
dest=base/'initial_surface_preview';dest.mkdir(exist_ok=True)
meshio.write(dest/'initial_surface.vtu',meshio.Mesh(X[ids],[('triangle',inv.reshape(-1,3))],point_data={'voltage_mV':v[ids]}))
(dest/'ep.pvd').write_text('<?xml version="1.0"?><VTKFile type="Collection" version="0.1"><Collection><DataSet timestep="0" file="initial_surface.vtu"/></Collection></VTKFile>')
audit=dict(source='3d/consistent_tc/ep_00000.vtu',nodes=len(X),all_voltages_finite=bool(np.isfinite(v).all()),
    voltage_min_mv=float(v.min()),voltage_max_mv=float(v.max()),surface_preview_nodes=len(ids),surface_triangles=len(faces),
    note='Preview extracts the actual exterior surface; the numerical audit uses every volumetric node.')
(base/'initial_frame_early_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))

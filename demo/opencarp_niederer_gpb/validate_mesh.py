"""Verify actual benchmark geometry without launching a second simulation."""
from pathlib import Path
import json,numpy as np
demo=Path('/mnt/large2/gjh/AFSEI/demo/opencarp_niederer_gpb')
out=Path('/mnt/large2/gjh/AFSEI/results/opencarp_gpb_h0p0625_20261030')
paths=list((demo/'meshes').glob('*/block.pts'))
if len(paths)!=1:raise RuntimeError(f'Expected one generated mesh, got {len(paths)}')
pts=paths[0]
p=np.loadtxt(pts,skiprows=1)
coords=[np.unique(p[:,i]) for i in range(3)]
for i,axis in enumerate(coords):
    if not np.allclose(np.diff(axis),62.5,rtol=0,atol=1e-6):raise RuntimeError('Incorrect mesh spacing')
if not np.allclose(p.min(0),[0,0,0]) or not np.allclose(p.max(0),[20000,7000,3000]):raise RuntimeError('Incorrect cuboid')
elem=pts.with_suffix('.elem')
with elem.open() as f:elements=int(f.readline())
lon=pts.with_suffix('.lon')
with lon.open() as f:
    header=f.readline().strip()
    samples=np.array([[float(v) for v in f.readline().split()] for _ in range(4096)])
if not np.allclose(samples[:,:3],[1,0,0],atol=1e-8):raise RuntimeError('Fiber sample is not X oriented')
r=dict(mesh_path=str(pts.with_suffix('')),nodes=len(p),tetrahedra=elements,coordinate_units='um',
    minimum_um=p.min(0).tolist(),maximum_um=p.max(0).tolist(),axis_node_counts=[len(a) for a in coords],
    nominal_spacing_um=62.5,spacing_verified_on_all_points=True,fiber_sample_size=len(samples),
    fiber_sample_direction=[1,0,0],lon_header=header,
    stimulus_nodes=int(np.all(p<=1500.+1e-6,axis=1).sum()),
    note='Native mesher uses 5 tetrahedra per Cartesian cell; this is not the AFSEI 6-tetrahedron mesh.')
(out/'mesh_validation.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))

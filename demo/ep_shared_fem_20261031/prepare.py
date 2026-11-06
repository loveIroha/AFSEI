"""Shared AFSEI framework; coarse 5-tet topology follows native parity."""
from pathlib import Path
import sys,json,time,hashlib,itertools
import numpy as np,torch
ROOT=Path('/mnt/large2/gjh/AFSEI');sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from afsi_torch.ep.refined_mesh import _assemble_cache
from afsi_torch.ep.monodomain import MonodomainParameters
OUT=ROOT/'results/ep_shared_fem_20261031';OUT.mkdir(exist_ok=False)
fine=ROOT/'results/ep_opencarp_same_mesh_20261030';common=json.loads((fine/'common_configuration.json').read_text())
coarse=OUT/'operators_h0p125';coarse.mkdir()
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
Xf=np.load(fine/'coordinates_cm.npy');nf=np.array([321,113,49]);nc=(nf-1)//2+1
iz,iy,ix=np.meshgrid(np.arange(nc[2]),np.arange(nc[1]),np.arange(nc[0]),indexing='ij')
ids=(2*ix+nf[0]*2*iy+nf[0]*nf[1]*2*iz).ravel();X=Xf[ids]
assert np.allclose(X.max(0),[2,.7,.3])
base=common['original_mesh_base']
# Read just enough native tetrahedra to recover both cube-parity templates.
sample=np.loadtxt(base+'.elem',skiprows=1,max_rows=20000,usecols=(1,2,3,4),dtype=np.int64)
grid=np.rint(Xf[sample]/.00625).astype(int);orig=grid.min(1)
templates=[]
for origin in (np.array([0,0,0]),np.array([1,0,0])):
    local=grid[(orig==origin).all(1)]-origin
    assert local.shape==(5,4,3),(origin,local.shape)
    assert np.all((local==0)|(local==1));templates.append(local)
# Generate only the coarse mesh, preserving alternating native cube pattern.
oz,oy,ox=np.meshgrid(np.arange(nc[2]-1),np.arange(nc[1]-1),np.arange(nc[0]-1),indexing='ij')
origin=np.column_stack((ox.ravel(),oy.ravel(),oz.ravel()))
local=np.asarray(templates)[origin.sum(1)%2]+origin[:,None,None,:]
cells=(local[:,:,:,0]+nc[0]*local[:,:,:,1]+nc[0]*nc[1]*local[:,:,:,2]).reshape(-1,4)
del local
xx=X[cells];volume=np.abs(np.linalg.det((xx[:,1:]-xx[:,:1]).transpose(0,2,1)))/6
assert (volume>0).all() and abs(volume.sum()-.42)<1e-10
assert len(X)==229425 and len(cells)==1075200
fields=np.zeros((len(X),6));fields[:,0]=1;fields[:,4]=1
t=time.perf_counter();_assemble_cache(coarse/'operators.npz',dict(X=X,cells=cells,fields=fields),MonodomainParameters(**common['parameters']))
np.save(coarse/'coordinates_cm.npy',X);np.save(coarse/'openCARP_reference_node_ids.npy',ids)
c=common|dict(nodes=len(X),cells=len(cells),mesh_mm=.125,coordinates_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
    cells_sha256=hashlib.sha256(cells.tobytes()).hexdigest(),original_mesh_base=None,
    topology='Same native alternating five-tetrahedra cube templates; h=0.125mm; all nodes coincide with fine reference',setup_time_s=time.perf_counter()-t)
c.pop('operator_splitting',None);(coarse/'common_configuration.json').write_text(json.dumps(c,indent=2))
config_base=json.loads((ROOT/'demo/ep_afsei_nodal_opencarp_matched/nodal_afsei.json').read_text())
jobs=[]
for h,cache in [(.125,coarse),(.0625,fine)]:
    for method in ('consistent_afsei','nodal_afsei'):
        name=('h0p125' if h==.125 else 'h0p0625')+'_'+method
        cfg=config_base|dict(output=str(OUT/name),operator_cache=str(cache),mesh_mm=h,pde_method=method,
            retained_numerics='Shared optimized AFSEI Graph-Jacobi-PCG and BufferedRHS; CN20us, common GPB Euler1us; only mass/source quadrature differs',
            save_final_cell_state=False)
        cfg.pop('sdirk_reference_result',None)
        path=OUT/(name+'.json');path.write_text(json.dumps(cfg,indent=2));jobs.append(dict(name=name,config=str(path)))
(OUT/'jobs.json').write_text(json.dumps(jobs,indent=2));print(json.dumps(dict(status='prepared',coarse_nodes=len(X),coarse_cells=len(cells),jobs=jobs),indent=2),flush=True)

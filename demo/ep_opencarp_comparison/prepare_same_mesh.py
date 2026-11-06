"""Chunked existing AFSEI P1 assembly on the exact native openCARP mesh."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,torch
root=Path('/mnt/large2/gjh/AFSEI');sys.path.insert(0,str(root/'src'))
from afsi_torch.ep.refined_mesh import _assemble_cache
from afsi_torch.ep.monodomain import MonodomainParameters
out=root/'results/ep_opencarp_same_mesh_20261030';out.mkdir(exist_ok=True)
info=json.loads((root/'results/opencarp_gpb_h0p0625_20261030/mesh_validation.json').read_text())
base=info['mesh_path'];start=time.perf_counter()
X=np.loadtxt(base+'.pts',skiprows=1)/10000 # micrometres to cm
# All original nodes and connectivity are used. No mesh regeneration.
cells=np.loadtxt(base+'.elem',skiprows=1,usecols=(1,2,3,4),dtype=np.int64)
fields=np.zeros((len(X),6));fields[:,0]=1;fields[:,4]=1
sigma_s_m=np.array([.17*.62/(.17+.62),.019*.24/(.019+.24),.019*.24/(.019+.24)])
conductivity=sigma_s_m*10000 # S/m -> microS/cm
parameters=MonodomainParameters(surface_to_volume_per_cm=1400,conductivity_us_per_cm=tuple(conductivity))
data=dict(X=X,cells=cells,fields=fields)
print('ASSEMBLY_START',len(X),len(cells),conductivity.tolist(),flush=True)
_assemble_cache(out/'operators.npz',data,parameters)
np.save(out/'coordinates_cm.npy',X)
manifest=dict(scope='Same native 5-tet mesh; only mass quadrature differs between AFSEI methods',
    original_mesh_base=base,nodes=len(X),cells=len(cells),coordinates_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
    cells_sha256=hashlib.sha256(cells.tobytes()).hexdigest(),parameters=parameters.__dict__,
    equivalent_monodomain_conductivity_s_per_m=sigma_s_m.tolist(),dt_s=2e-5,duration_s=.05,
    initial_state='Official Grandi defaults; no conditioning',operator_splitting='Reaction then CN diffusion, native openCARP default',
    setup_time_s=time.perf_counter()-start,
    assembler='Existing afsi_torch.ep.refined_mesh._assemble_cache; GPU local P1 integration, CPU sparse topology compression')
(out/'common_configuration.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)

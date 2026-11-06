"""Export nonzero MAC step and adaptive IB operator cases; CPU avoids GPU timing interference."""
from pathlib import Path
from dataclasses import replace
import numpy as np,torch,json
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.backward_euler import BackwardEulerFlow,BEFlowOptions
from afsi_torch.mac.multigrid import MGOptions
from afsi_torch.mac.adaptive_transfer import AdaptiveP1Transfer,InteractionQuadratureOptions
from afsi_torch.p1 import prepare_p1
from afsi_torch.fluid.solvers import SolverOptions
torch.set_num_threads(2)
root=Path('/mnt/large2/gjh/AFSEI/results')
out=root/'reliability_campaign_20261025/independent_mac_ib';out.mkdir(exist_ok=True)
old=np.load(root/'single_tube_full_20261024/active_120ms/final_state.npz')
grid=MACGrid((32,32,64),(1.4,1.4,2.8));dt=5e-5
vel=tuple(torch.from_numpy(old[k].copy()) for k in ('u','v','w'))
density=tuple(1000*torch.sin(grid.coordinates(c)[...,0]*2)*torch.cos(grid.coordinates(c)[...,2]) for c in range(3))
flow=BackwardEulerFlow(grid,dt=dt,rho=1.,mu=.04,pressure_options=MGOptions(rtol=1e-10,atol=1e-11,max_cycles=100,smooth=4,check_every=2),
                       options=BEFlowOptions(helmholtz_rtol=1e-12,helmholtz_atol=1e-13))
dep=flow.advect(vel);v,p,info=flow.advance(dep,density)
data=dict(shape=np.array(grid.shape),lengths=np.array(grid.lengths),dt=dt,mu=.04,rho=1.,pressure=p.numpy())
for c in range(3):
    data[f'old_{c}']=vel[c].numpy();data[f'density_{c}']=density[c].numpy()
    data[f'departure_{c}']=dep[c].numpy();data[f'velocity_{c}']=v[c].numpy()
np.savez_compressed(out/'mac_inputs.npz',**data)
# An explicitly documented subset tests IB implementation; this is NOT the full-tube trajectory.
ids=np.linspace(0,len(old['cells'])-1,24,dtype=int)
global_cells=old['cells'][ids];nodes,inv=np.unique(global_cells,return_inverse=True)
cells=torch.from_numpy(inv.reshape(-1,4));X=torch.from_numpy(old['reference'][nodes].copy());x=torch.from_numpy(old['x'][nodes].copy())
g=prepare_p1(X,cells,2)
transfer=AdaptiveP1Transfer(grid,g,mass_backend='pcg',warm_start=False,fused=False,
    options=SolverOptions(rtol=1e-12,atol=1e-13,max_iterations=500),
    quadrature_options=InteractionQuadratureOptions(mode='adaptive',point_density=2,max_order=12,rule_family='xiao-gimbutas'))
stencil=transfer.prepare(x)
force=torch.from_numpy(old['force'][nodes].copy())
spread,_=transfer.spread(force,stencil);interp,_=transfer.interpolate(vel,stencil)
data=dict(X=X.numpy(),x=x.numpy(),cells=cells.numpy(),force=force.numpy(),interpolated=interp.numpy(),
          shape=np.array(grid.shape),lengths=np.array(grid.lengths),group_count=len(stencil.rule.groups))
for c in range(3):data[f'velocity_{c}']=vel[c].numpy();data[f'spread_{c}']=spread[c].numpy()
for i,group in enumerate(stencil.rule.groups):
    data[f'cells_{i}']=group.cells.numpy();data[f'N_{i}']=group.values.numpy();data[f'W_{i}']=group.weights.numpy()
np.savez_compressed(out/'ib_inputs.npz',**data)
(out/'export.json').write_text(json.dumps(dict(mac_diagnostics=info,ib_subset_elements=24,ib_subset_nodes=len(X),ib_points=stencil.rule.point_count),indent=2))
print('exported MAC step and IB subset',len(X),stencil.rule.point_count)

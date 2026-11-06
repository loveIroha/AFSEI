"""Export a saved physical state and AFSEI FE outputs for an external oracle."""
from pathlib import Path
import json, numpy as np, torch
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.ep.circular_tube import CircularTubeSolid
from afsi_torch.ep.tissue import TissueState
from afsi_torch.ep.monodomain import Monodomain

torch.set_num_threads(2)
root=Path('/mnt/large2/gjh/AFSEI')
src=root/'results/single_tube_full_20261024/active_120ms'
out=root/'results/reliability_campaign_20261025/independent_fem'
out.mkdir(exist_ok=True)
c=json.loads((src/'config.resolved.json').read_text())
g,fsi,p,n=resolve_config(c);m=generate_circular_tube(g)
s=CircularTubeSolid(m,g,**{k:v for k,v in c['solid'].items() if k!='end_condition'},
    **{k:v for k,v in c['active'].items() if k not in ('mechanism','rest_reference')},
    end_support=c['solid']['end_condition'],dt_s=fsi.time.dt)
with np.load(src/'final_state.npz') as a:
    get=lambda k:torch.from_numpy(a[k].copy())
    x=get('x');v=get('voltage_v');xs=get('land')[:,0]
    resting=get('resting_tension_dyn_cm2')
    tissue=TissueState(v,get('gpb'),get('land'))
    s.set_reference(resting);s.stage(x,tissue,get('zs'),get('zw'))
    force_saved=a['force'].copy()
axial=m.X.new_tensor([0.,0.,1.]).expand_as(m.X)
sheet=torch.cat((m.X[:,:2]-.7,torch.zeros_like(m.X[:,:1])),1)
sheet=sheet/sheet.norm(dim=1,keepdim=True)
ep=Monodomain(m.X,m.cells,axial,sheet,parameters=p,numerics=n)
z=(m.X[:,2]-.4)/2
probe=torch.stack((.01*torch.sin(torch.pi*z),.007*torch.sin(2*torch.pi*z),.005*z),1)
force=s.force(x,.12)
jvp=torch.func.jvp(lambda xx:s.force(xx,.12),(x,),(probe,))[1]
# Actual recorded voltage plus a separately specified smooth PDE field; frozen
# source verifies the diffusion stage only, without claiming an independent GPB.
test_voltage=-.06+.025*torch.cos(torch.pi*z)+.01*(m.X[:,0]-.7)
ionic=.1*torch.sin(2*torch.pi*z);stimulus=torch.zeros_like(ionic)
nextv,info=ep.step(test_voltage,ionic,stimulus,5e-5)
data=dict(X=m.X.numpy(),cells=m.cells.numpy(),x=x.numpy(),probe=probe.numpy(),
    xs=xs.numpy(),resting=resting.numpy(),force=force.detach().numpy(),force_saved_gpu=force_saved,
    force_jvp=jvp.detach().numpy(),voltage=test_voltage.numpy(),ionic=ionic.numpy(),
    next_voltage=nextv.numpy(),ep_sheet=sheet.numpy())
for key,A in [('M',ep.mass),('K',ep.stiffness)]:
    data[key+'_indptr']=A.crow_indices().numpy();data[key+'_indices']=A.col_indices().numpy();data[key+'_data']=A.values().numpy()
np.savez_compressed(out/'inputs.npz',**data)
(out/'config.json').write_text(json.dumps(c,indent=2))
print(json.dumps(dict(nodes=len(m.X),cells=len(m.cells),saved_force_cpu_relative_error=float((force-torch.from_numpy(force_saved)).norm()/force.norm()))))
ep.close()

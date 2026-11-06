"""Export new stress force/JVP for an external UFL assembly check; no stepping."""
from pathlib import Path
import json,numpy as np,torch
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.ep.circular_tube import CircularTubeSolid
from afsi_torch.ep.tissue import TissueState


@torch.no_grad()
def main():
    repo=Path(__file__).resolve().parents[2];out=repo/'results/volumetric_stabilization_20261025/independent_stress'
    out.mkdir(exist_ok=False);torch.set_num_threads(2)
    wrapper=json.loads((repo/'demo/verification/configs/volume_switch_solid1_60to100ms.json').read_text())
    c=wrapper['model'];c['geometry'].update(axial_elements=8,circumferential_elements=16,thickness_elements=1)
    c.pop('solid_refinement_factor',None);g,_,_,_=resolve_config(c);mesh=generate_circular_tube(g)
    a=np.load(repo/'results/independent_coupled_20261025/initial_state_complete.npz')
    assert np.array_equal(a['X'],mesh.X.numpy()) and np.array_equal(a['cells'],mesh.cells.numpy())
    get=lambda k:torch.as_tensor(a[k].copy(),dtype=mesh.X.dtype)
    tissue=TissueState(get('voltage'),get('gpb'),get('land'));rest=get('resting_tension');zero=torch.zeros(len(mesh.X),dtype=mesh.X.dtype)
    solid=CircularTubeSolid(mesh,g,**{k:v for k,v in c['solid'].items() if k!='end_condition'},
        **{k:v for k,v in c['active'].items() if k not in ('mechanism','rest_reference')},
        dt_s=c['time']['fsi_dt_s'],end_support=c['solid']['end_condition'])
    solid.set_reference(rest);solid.stage(mesh.X,tissue,zero,zero)
    x=mesh.X.clone();center=x.new_tensor(g.center_xy_cm);x[:,:2]=center+.9*(x[:,:2]-center)
    x[:,2]=g.axial_start_cm+1.03*(x[:,2]-g.axial_start_cm)
    x[:,0]+=.015*torch.sin(torch.pi*(mesh.X[:,2]-g.axial_start_cm)/g.length_cm)
    rng=np.random.default_rng(20261025);direction=torch.as_tensor(rng.normal(size=tuple(x.shape)),dtype=x.dtype);direction/=direction.norm()
    force=solid.force(x,.06);matrix=solid.tangent_factory().assemble(x,.06)
    jvp=torch.mv(matrix,direction.flatten()).reshape_as(x)
    np.savez_compressed(out/'input.npz',X=mesh.X.numpy(),cells=mesh.cells.numpy(),x=x.numpy(),rest=rest.numpy(),
        land=tissue.land.numpy(),direction=direction.numpy(),force=force.numpy(),jvp=jvp.numpy())
    (out/'model.json').write_text(json.dumps(c,indent=2));print('Exported actual stabilized force/JVP on',len(x),'nodes')


if __name__=='__main__':main()

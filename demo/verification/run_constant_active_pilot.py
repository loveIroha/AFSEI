"""AFSEI FSI pilot with the identical fixed EP-derived amplitude as IBAMR."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,time
import numpy as np,torch
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.ep.circular_tube import CircularTubeSolid
from afsi_torch.mac.paper_coupling import BEIBStepper
from afsi_torch.mac.output import MACWriter

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    c=json.loads(args.config.read_text());c['solid_refinement_factor']=1;c['fluid']['shape']=[64]*3;c['time']['end_time_s']=.002
    c['output'].update(log_every=10,field_every=20,checkpoint_every=40)
    g,fsi,_,_=resolve_config(c);out=args.output;out.mkdir(parents=True,exist_ok=False)
    (out/'config.resolved.json').write_text(json.dumps(c,indent=2));started=time.perf_counter()
    mesh=generate_circular_tube(g,device='cuda:0');a=np.load(args.input)
    if not np.array_equal(a['reference'],mesh.X.cpu().numpy()):raise ValueError('pilot mesh mismatch')
    get=lambda key:torch.tensor(a[key],device=mesh.X.device,dtype=mesh.X.dtype)
    solid=CircularTubeSolid(mesh,g,**{k:v for k,v in c['solid'].items() if k!='end_condition'},
        **{k:v for k,v in c['active'].items() if k not in ('mechanism','rest_reference')},dt_s=fsi.time.dt,end_support=c['solid']['end_condition'])
    solid.set_reference(get('resting_tension_dyn_cm2'));solid.stage(mesh.X,SimpleNamespace(land=get('land')),get('zs'),get('zw'))
    driver=BEIBStepper(solid,fsi,mesh.X.device);state=driver.initialize(mesh.X)
    writer=MACWriter(out/'vtk',solid,driver.grid,pressure_description='original AFSEI split open boundary, zero exterior projection pressure')
    writer.write(state);torch.cuda.synchronize();setup=time.perf_counter()-started;blocks=[];tick=time.perf_counter()
    for step in range(1,41):
        if step==40:
            previous_pressure=state.pressure.clone();previous_pressure_time=state.time
        state,info=driver.step(state,diagnostics=step%10==0)
        if step%10==0:
            torch.cuda.synchronize();blocks.append(dict(end_step=step,seconds=time.perf_counter()-tick,diagnostics=info))
            if step%20==0:writer.write(state)
            tick=time.perf_counter()
    np.savez_compressed(out/'final_state.npz',reference=mesh.X.cpu().numpy(),cells=mesh.cells.cpu().numpy(),x=state.x.cpu().numpy(),
        u=state.velocity[0].cpu().numpy(),v=state.velocity[1].cpu().numpy(),w=state.velocity[2].cpu().numpy(),pressure=state.pressure.cpu().numpy(),
        pressure_time_s=state.pressure_time,previous_pressure=previous_pressure.cpu().numpy(),previous_pressure_time_s=previous_pressure_time)
    report=dict(status='completed',steps=40,time_s=.002,setup_seconds=setup,compute_seconds=sum(b['seconds'] for b in blocks),
        warmed_compute_seconds_per_step=sum(b['seconds'] for b in blocks[1:])/30,
        constant_active_amplitude=True,EP_advanced=False,ODE_advanced=False,
        initial_value_problem='reference wall, zero fluid, fixed Land state from 60ms; not a mechanical restart',
        formal_cross_software_accuracy_gate_passed=False,blocks=blocks,final=solid.diagnostics(state.x))
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()

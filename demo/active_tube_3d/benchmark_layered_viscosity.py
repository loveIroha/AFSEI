"""Matched real-grid fluid solve: original BE-Jacobi vs same-matrix BE-PCG."""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.ep.layered_circular_tube import LayeredCircularTubeSolid
from afsi_torch.ep.tissue import TissueState
from afsi_torch.mac.paper_coupling import BEIBStepper
from afsi_torch.mac.sealed_ports import SealedDomain
from afsi_torch.mac.sealed_flow import SealedBEFlow
from afsi_torch.cycle_checkpoint import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text());g,fsi,_,_=resolve_config(c)
    a.output.mkdir(parents=True,exist_ok=True)
    m=generate_circular_tube(g,device='cuda')
    s=LayeredCircularTubeSolid(m,g,**{k:v for k,v in c['solid'].items() if k!='end_condition'},
        **{k:v for k,v in c['active'].items() if k not in ('mechanism','rest_reference')},dt_s=fsi.time.dt)
    s.scale=.25
    with np.load(a.source) as old:
        x=torch.from_numpy(old['x'].copy()).to(m.X);s.set_reference(torch.from_numpy(old['resting_tension_dyn_cm2'].copy()).to(m.X))
    with np.load(a.prepared) as states:
        get=lambda k:torch.from_numpy(states[k].copy()).to(m.X)
        tissue=TissueState(get('voltage_v'),get('gpb'),get('land'));s.stage(x,tissue,get('zs'),get('zw'))
    d=BEIBStepper(s,fsi,'cuda');domain=SealedDomain(d.grid,g,c['hydraulic_domain'],m.X,dimension=3)
    force=s.force(x,0.);stencil=d.transfer.prepare(x);density,_=d.transfer.spread(force,stencil)
    departure=d.grid.zeros(device='cuda')
    flows={k:SealedBEFlow(d.flow,domain,fused=True,graphs=True,helmholtz_solver=k) for k in ('jacobi','pcg')}
    outputs={};timings={};info={};cold={}
    for k,flow in flows.items():
        torch.cuda.synchronize();started=perf_counter();u,p,diag=flow.advance(departure,density)
        torch.cuda.synchronize();cold[k]=perf_counter()-started
        times=[]
        for repeat in range(3):
            torch.cuda.synchronize();started=perf_counter();u,p,diag=flow.advance(departure,density)
            torch.cuda.synchronize();times.append(perf_counter()-started)
        outputs[k]=(tuple(t.clone() for t in u),p.clone());timings[k]=times;info[k]=diag
    norms=lambda v:sum(t.square().sum() for t in v).sqrt()
    relative=float(norms(tuple(a-b for a,b in zip(outputs['jacobi'][0],outputs['pcg'][0])))/norms(outputs['jacobi'][0]))
    report=dict(scope='same physical forcing, BE matrix, pressure projection and original residual targets',
        gpu=torch.cuda.get_device_name(),shape=d.grid.shape,cold_seconds=cold,warmed_seconds=timings,
        median_speedup=float(np.median(timings['jacobi'])/np.median(timings['pcg'])),
        velocity_relative_l2_difference=relative,diagnostics=info)
    atomic_json(a.output/'report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()

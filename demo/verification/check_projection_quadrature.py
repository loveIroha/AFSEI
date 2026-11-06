"""Higher-order check of frozen regularized-divergence integrals; no solves."""
from pathlib import Path
import argparse, json, time
import numpy as np
import torch
from afsi_torch.ep.refined_mesh import lattice_template
from afsi_torch.p1 import tetra_rule
from afsi_torch.mac.grid import MACGrid
from diagnose_projection_convergence import sample_component


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path)
    args=parser.parse_args()
    r=json.loads(args.report.read_text());cfg=r['config'];c=r['source_config']
    a=np.load(cfg['checkpoint']);torch.set_num_threads(2)
    cast=lambda z:torch.as_tensor(np.asarray(z).copy(),device=cfg['device'],dtype=torch.float64)
    grid=MACGrid(tuple(c['fluid']['shape']),tuple(c['fluid']['lengths_cm']),tuple(c['fluid']['origin_cm']))
    velocity=tuple(cast(a[k]) for k in ('u','v','w'))
    cells=np.sort(a['cells'],axis=1);x=cast(a['x'][cells])
    volume=torch.linalg.det((x[:,1:]-x[:,:1]).transpose(1,2)).abs()/6
    template,tc=lattice_template(6);template=cast(template[tc]/6)
    q,w=tetra_rule(5,x);N=torch.cat((1-q.sum(1,keepdim=True),q),1)
    bary=torch.einsum('qa,cai->cqi',N,template).reshape(-1,4)
    local_w=(w[None].expand(len(template),-1)*6/216).reshape(-1)
    sample=torch.compile(sample_component,fullgraph=True,dynamic=True,options={'triton.cudagraphs':False})
    spacing=x.new_tensor(grid.spacing);origins=x.new_tensor([grid.face_origin(i) for i in range(3)])
    total=x.new_zeros(3);begin=time.perf_counter()
    stride=max(1,65536//len(bary))
    for start in range(0,len(cells),stride):
        stop=min(start+stride,len(cells))
        points=torch.einsum('qa,eai->eqi',bary,x[start:stop]).reshape(-1,3)
        weights=(volume[start:stop,None]*local_w[None]).reshape(-1)
        div=sum(sample(points,velocity[i],origins[i],spacing,i)[1] for i in range(3))
        total+=torch.stack((weights.sum(),(weights*div).sum(),(weights*div.square()).sum()))
    m=total.cpu().numpy();rms=float(np.sqrt(m[2]/m[0]));mean=float(m[1]/m[0])
    base=r['common_quadrature'];rms_diff=abs(rms/base['delta_divergence_rms_s_inv']-1)
    mean_diff=abs(mean-base['delta_divergence_mean_s_inv'])
    report=dict(scope='postprocessing quadrature only; no EP/ODE/FSI/IB projection repeated',
        template_factor=6,low_degree=2,high_degree=5,points_high=len(cells)*len(bary),
        current_volume_ml=float(m[0]),high_rms_s_inv=rms,high_mean_s_inv=mean,
        rms_relative_change=rms_diff,mean_absolute_change_s_inv=mean_diff,
        thresholds=dict(rms_relative_change=.005,mean_absolute_change_s_inv=.01),
        passed=bool(rms_diff<.005 and mean_diff<.01),elapsed_seconds=time.perf_counter()-begin)
    (args.report.parent/'quadrature_check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if not report['passed']:raise AssertionError('Common quadrature requires further refinement')


if __name__=='__main__':main()

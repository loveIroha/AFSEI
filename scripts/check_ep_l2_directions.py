"""Read-only real-mesh direction comparison; writes ParaView VTU and metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import meshio
from afsi_torch.ep.mapping import l2_project_cell_fields, orthonormalize_directions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args(); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    with np.load(a.checkpoint,allow_pickle=False) as z:
        X=torch.as_tensor(z['X'],device=a.device)
        cells=torch.as_tensor(z['cells'],device=a.device)
        f=torch.as_tensor(z['fiber'],device=a.device)
        s=torch.as_tensor(z['sheet'],device=a.device)
    owner=cells.new_full((len(X),),-1)
    owner.scatter_reduce_(0,cells.flatten(),torch.arange(len(cells),device=cells.device).repeat_interleave(4),
                          reduce='amax',include_self=True)
    if (owner<0).any():raise ValueError('isolated node')
    raw=l2_project_cell_fields(X,cells,torch.cat((f,s),1))
    pf,ps,pn=orthonormalize_directions(raw[:,:3],raw[:,3:])
    oldf,olds=f[owner],s[owner];oldn=torch.linalg.cross(oldf,olds)
    cpu=lambda v:v.detach().cpu().numpy()
    def angle(u,v):
        c=(u*v).sum(-1)/(torch.linalg.vector_norm(u,dim=-1)*torch.linalg.vector_norm(v,dim=-1))
        return torch.rad2deg(torch.acos(c.clamp(-1,1)))
    fa,sa=angle(oldf,pf),angle(olds,ps)
    frame=torch.stack((pf,ps,pn),-1)
    orth=(frame.transpose(-1,-2)@frame-torch.eye(3,device=X.device,dtype=X.dtype)).abs().max().item()
    # Actual residual checked inside projection; integral conservation before normalization.
    vol=torch.linalg.det((X[cells[:,1:]]-X[cells[:,:1]]).transpose(-1,-2)).abs()/6
    lump=X.new_zeros(len(X)).index_add_(0,cells.flatten(),(vol[:,None]/4).expand(-1,4).flatten())
    integral_error=((lump[:,None]*raw).sum(0)-(vol[:,None]*torch.cat((f,s),1)).sum(0)).abs().max().item()
    report={'nodes':len(X),'cells':len(cells),'device':str(X.device),
            'orthonormality_max_error':orth,'raw_projection_integral_max_error':integral_error,
            'minimum_frame_determinant':torch.linalg.det(frame).min().item()}
    for name,v in [('fiber',fa),('sheet',sa)]:
        report[name+'_angle_degrees']={'mean':v.mean().item(),'p95':torch.quantile(v,.95).item(),'max':v.max().item()}
        axial=torch.minimum(v,180-v)
        report[name+'_axis_angle_degrees']={'mean':axial.mean().item(),'p95':torch.quantile(axial,.95).item(),
                                          'max':axial.max().item(),'nodes_above_30deg':int((axial>30).sum())}
    if orth>1e-12:raise RuntimeError('orthonormality failed')
    points=cpu(X); topology=[('tetra',cpu(cells))]
    original={'fiber':cpu(oldf),'sheet':cpu(olds),'normal':cpu(oldn),'owner_cell_id':cpu(owner)}
    projected={'fiber':cpu(pf),'sheet':cpu(ps),'normal':cpu(pn),
               'fiber_l2_raw':cpu(raw[:,:3]),'sheet_l2_raw':cpu(raw[:,3:]),
               'fiber_angle_from_legacy_deg':cpu(fa),'sheet_angle_from_legacy_deg':cpu(sa)}
    meshio.write(out/'original_cell_directions.vtu',meshio.Mesh(points,topology,
        cell_data={'fiber':[cpu(f)],'sheet':[cpu(s)],'normal':[cpu(torch.linalg.cross(f,s))]}))
    meshio.write(out/'legacy_max_incident_cell.vtu',meshio.Mesh(points,topology,point_data=original))
    meshio.write(out/'l2_orthonormal_directions.vtu',meshio.Mesh(points,topology,point_data=projected))
    both={**{'legacy_'+k:v for k,v in original.items()},**{'l2_'+k:v for k,v in projected.items()}}
    meshio.write(out/'direction_comparison.vtu',meshio.Mesh(points,topology,point_data=both))
    for name in ['original_cell_directions','legacy_max_incident_cell','l2_orthonormal_directions','direction_comparison']:
        saved=meshio.read(out/(name+'.vtu'))
        if len(saved.points)!=len(X) or len(saved.cells[0].data)!=len(cells):
            raise RuntimeError('VTK readback mesh mismatch')
    report['vtk_readback']='passed'
    report['status']='passed'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

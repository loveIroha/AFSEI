"""Compare actual source-DOLFIN residual/Jacobian with AFSEI on the real LV."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from afsi_torch.paper_lv import PaperLVConfig,imported_model
from afsi_torch.ep.config import AFSEIConfig
from afsi_torch.ep.buffered_load import BufferedLandLoad,ActiveLoadState
from afsi_torch.ep.solid import EPDrivenPaperSolid
from afsi_torch.ep.monodomain import Monodomain
from afsi_torch.ep.source_mechanics import SOURCE_COMMIT


def discrepancy(actual,expected,*,rtol=1e-9,atol=1e-6):
    error=actual-expected
    scale=expected.abs().max().item()
    relative=(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(expected).clamp_min(1e-30)).item()
    maximum=error.abs().max().item()
    return dict(passed=maximum <= atol+rtol*scale and relative <= rtol,
                relative_l2_error=relative,max_absolute_error=maximum,reference_max_absolute=scale,
                relative_tolerance=rtol,absolute_tolerance=atol)


@torch.no_grad()
def check(oracle_dir,source_dir,*,device='cuda',assembled_tangent=True):
    started=perf_counter(); oracle_dir=Path(oracle_dir)
    required=('Active_land.ufl','Active_land.h','passive.csv','land.csv','frozen.csv','cg1_land.csv')
    if any(not (oracle_dir/name).is_file() for name in required):
        raise FileNotFoundError('Generate the reference first: bash scripts/prepare_afsei_mechanics_reference.sh')
    config=AFSEIConfig.source(fsi=PaperLVConfig(source_dir=str(source_dir)))
    passive=imported_model(config.fsi,device); mesh=passive.mesh
    buffer=BufferedLandLoad(passive.geometry,mesh.fiber)
    model=EPDrivenPaperSolid(mesh,config.fsi,buffer,source_quadrature=True)
    ep=Monodomain(mesh.X,mesh.cells,mesh.fiber,mesh.sheet,directions='cell')
    raw=mesh.X.cpu().numpy(); order=np.lexsort(raw.T[::-1])
    tangent=model.tangent_factory() if assembled_tangent else None
    cases={}; reference_force=None; passive_force=None; passive_jvp=None
    for mode in ('passive','land','frozen'):
        csv=np.loadtxt(oracle_dir/(mode+'.csv'),delimiter=',')
        source_order=np.lexsort(csv[:,:3].T[::-1])
        if (csv.shape != (len(raw),26) or not np.allclose(csv[source_order,:3],raw[order],rtol=0,atol=1e-12)):
            raise ValueError('source and AFSEI reference vertices disagree')
        values=np.empty_like(csv); values[order]=csv[source_order]
        tensor=lambda a: torch.as_tensor(a,device=device,dtype=torch.float64)
        x,previous,direction,anchor_x=(tensor(values[:,a:a+3]) for a in (3,6,9,12))
        xs,xw,zs,zw=(tensor(values[:,a]) for a in range(15,19))
        active=ActiveLoadState(previous,xs,xw,zs,zw,.001)
        if mode=='frozen':
            buffer.stage(active); anchor=buffer.closure_snapshot(anchor_x)
            active=ActiveLoadState(previous,xs,xw,zs,zw,.001,False,anchor,.37)
        buffer.stage(active)
        force=model.force(x,.501)
        jvp=torch.func.jvp(lambda position:model.force(position,.501),(x,),(direction,))[1]
        metrics=dict(force=discrepancy(force,tensor(values[:,19:22])),
                     force_jvp=discrepancy(jvp,tensor(values[:,22:25])))
        if mode=='passive':
            passive_force=tensor(values[:,19:22]); passive_jvp=tensor(values[:,22:25])
        else:
            metrics['active_force']=discrepancy(buffer.force(x),tensor(values[:,19:22])-passive_force)
            metrics['active_force_jvp']=discrepancy(buffer.force_jvp(x,direction),tensor(values[:,22:25])-passive_jvp)
        if tangent is not None:
            J=tangent.assemble(x,.501)
            action=torch.sparse.mm(J,direction.reshape(-1,1)).reshape_as(direction)
            metrics['assembled_tangent']=discrepancy(action,tensor(values[:,22:25]))
        F=model.element_gradient(x)
        cell_stretch=torch.linalg.vector_norm(torch.einsum('eij,ej->ei',F,mesh.fiber),dim=-1)
        # Compare the weak projection RHS directly; no solver error is hidden.
        local=ep.geometry.weights.sum(1)[:,None]*cell_stretch[:,None]/4
        rhs=mesh.X.new_zeros(len(mesh.X)).index_add(0,mesh.cells.reshape(-1),local.expand(-1,4).reshape(-1))
        metrics['stretch_projection_rhs']=discrepancy(rhs,tensor(values[:,25]),rtol=1e-10,atol=1e-12)
        cases[mode]=metrics
        if mode=='land': reference_force=values[:,19:22].copy()
    # Deliberately bind globally interpolated CG1 directions as a control.
    wrong=np.loadtxt(oracle_dir/'cg1_land.csv',delimiter=',')
    wrong_order=np.lexsort(wrong[:,:3].T[::-1]); aligned=np.empty_like(wrong); aligned[order]=wrong[wrong_order]
    control_relative=np.linalg.norm(aligned[:,19:22]-reference_force)/np.linalg.norm(reference_force)
    return dict(passed=bool(all(m['passed'] for c in cases.values() for m in c.values()) and control_relative>1e-4),
        source_commit=SOURCE_COMMIT,scope='regenerated source UFL residual, tangent and stretch projection; zero pressure',
        node_count=len(mesh.X),cell_count=len(mesh.cells),device=str(device),
        gpu=torch.cuda.get_device_name(mesh.X.device) if mesh.X.is_cuda else None,
        mechanics_profile=config.mechanics_profile,cases=cases,
        global_cg1_negative_control_relative_force_difference=float(control_relative),
        source_mechanical_direction_binding='cell-dependent Expression; DG0-equivalent',
        source_ep_direction_binding='globally interpolated CG1 Function',
        regenerated_ufl_sha256=hashlib.sha256((oracle_dir/'Active_land.ufl').read_bytes()).hexdigest(),
        regenerated_header_sha256=hashlib.sha256((oracle_dir/'Active_land.h').read_bytes()).hexdigest(),
        assembled_tangent_verified=assembled_tangent,physical_ports_connected=False,
        full_cardiac_cycle_verified=False,elapsed_seconds=perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle-dir',type=Path,default=Path('.cache/source-mechanics-ref'))
    parser.add_argument('--source-dir',default='/mnt/large2/gjh/realistic_left_ventricle')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--skip-assembled-tangent',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=check(args.oracle_dir,args.source_dir,device=args.device,assembled_tangent=not args.skip_assembled_tangent)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))
    if not report['passed']: raise SystemExit('Source mechanical residual/tangent comparison failed')


if __name__=='__main__': main()

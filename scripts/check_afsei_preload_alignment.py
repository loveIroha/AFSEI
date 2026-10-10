"""Compare source ds(2) preload with the closed-interface AFSEI preload.

The source comparison solves the original passive endocardial pressure
problem, retains its 100 x 5ms loading distortion history, then verifies
equilibrium and derivatives with regenerated pinned UFL in isolated FEniCS.
It cannot create a three-field checkpoint from an endocardial-only balance.
"""
import argparse,json,subprocess,time,hashlib
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from afsi_torch.two_way import checkpoint
from afsi_torch.two_way.preload import (ClosedPressure,EndocardialPressure,PassivePreloader,PreloadConfig,SourceLoadingHistory)
from afsi_torch.moving_cavity.checkpoint import atomic
from afsi_torch.ep.buffered_load import ActiveLoadState
from afsi_torch.paper_lv import PaperHOTangent
from afsi_torch.circulation.checkpoint import digest


def metric(actual,expected,*,rtol=1e-9,atol=1e-6,relative=True):
    delta=actual-expected; norm=expected.norm().clamp_min(1e-30)
    error=(delta.norm()/norm).item(); maximum=delta.abs().max().item()
    return dict(passed=maximum<=atol+rtol*expected.abs().max().item() and (not relative or error<=rtol),
        relative_l2_error=error,max_absolute_error=maximum,relative_tolerance=rtol,absolute_tolerance=atol,
        relative_check_applied=relative)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--closed-checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--reuse-source-result',help='numeric source_solid_preload.npz from an earlier completed comparison')
    a=p.parse_args(); out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()): p.error('output directory must be new or empty')
    out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); start=time.perf_counter()
    d,closed,initial=checkpoint.load(a.closed_checkpoint)
    if closed.ep.fsi.step!=0 or not initial.get('preload_performed') or initial.get('preload_pressure_mmhg')!=8.:
        p.error('supply a completed, unadvanced 8mmHg three-field preload')
    if d.config.mechanics_profile!='source': p.error('source mechanics profile required for the pinned UFL comparison')
    reference=d.mesh.X; source=EndocardialPressure(d.mesh,d.blood_driver.mesh); whole=ClosedPressure(d.blood_driver.mesh)
    binding=hashlib.sha256()
    for value in (reference,d.mesh.cells,d.mesh.fiber,d.mesh.sheet): binding.update(value.cpu().numpy().tobytes())
    binding=binding.hexdigest()
    zero=reference.new_zeros(len(reference)); d.buffer.stage(ActiveLoadState(reference,zero,zero,zero,zero,.005,False))
    settings=PreloadConfig(nominal_steps=100,minimum_step_mmhg=.005)
    if a.reuse_source_result:
        with np.load(a.reuse_source_result,allow_pickle=False) as f:
            arrays={k:f[k] for k in f.files}; meta=json.loads(str(arrays.pop('metadata'))); sha=meta.pop('sha256')
            if (meta.get('producer')!='afsei-source-endo-preload' or meta.get('schema')!=1 or not meta.get('completed')
                or meta.get('pressure_mmhg')!=8. or meta.get('mesh_binding_sha256')!=binding or digest(meta,arrays)!=sha):
                raise ValueError('wrong or incomplete source preload result')
            if set(arrays)!={'reference_x_cm','x_cm','zs','zw'} or any(v.dtype!=np.float64 or not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError('invalid source preload arrays')
            x=torch.as_tensor(arrays['x_cm'],device=reference.device)
            torch.testing.assert_close(torch.as_tensor(arrays['reference_x_cm'],device=reference.device),reference,rtol=0,atol=0)
            zs=torch.as_tensor(arrays['zs'],device=reference.device); zw=torch.as_tensor(arrays['zw'],device=reference.device)
            if zs.shape!=reference[:,0].shape or zw.shape!=zs.shape: raise ValueError('wrong source distortion dimensions')
            source_history=meta['loading_history']
    else:
        history=SourceLoadingHistory(d)
        def progress(e):
            if e.get('physical_loading_endpoint') and round(e['physical_loading_time_s']*200)%10==0:
                print(f"Source-only preload P={e['pressure_mmhg']:.6g}mmHg V={e['volume_ml']:.9g}ml residual={e['residual_norm_dyn']:.3g}",flush=True)
        result=PassivePreloader(d,settings,pressure_load=source).solve(progress,observe_state=history.observe)
        h=history.final(); x=result.x; zs=h.zs; zw=h.zw; source_history=result.history
    d.model.validate(x)
    equilibrium=(d.model.force(x,.5)+source.force(x,8)).norm().item()
    if equilibrium>1e-3: raise ValueError('source preload fails its current true equilibrium check')
    atomic(out/'source_solid_preload.npz',dict(producer='afsei-source-endo-preload',schema=1,completed=True,
        pressure_mmhg=8.,three_field_checkpoint=False,mesh_binding_sha256=binding,loading_history=source_history),
        dict(reference_x_cm=reference.cpu().numpy(),x_cm=x.cpu().numpy(),zs=zs.cpu().numpy(),zw=zw.cpu().numpy()))
    direction=torch.sin(reference*.6)*.02+reference@reference.new_tensor([[.001,0.,0.],[0.,.002,0.],[0.,0.,-.001]])
    cases={'source':x,'closed':closed.ep.fsi.x}
    for name,positions in cases.items():
        np.savetxt(out/(name+'_input.csv'),torch.cat((reference,positions,direction),1).cpu().numpy(),delimiter=',',fmt='%.17g')
    subprocess.run(['bash','scripts/prepare_afsei_preload_reference.sh',str(d.config.fsi.source_dir),str(out)],check=True)
    tangent=PaperHOTangent(d.model); order=np.lexsort(reference.cpu().numpy().T[::-1]); verification={}
    for name,positions in cases.items():
        raw=np.loadtxt(out/(name+'_oracle.csv'),delimiter=','); indices=np.lexsort(raw[:,:3].T[::-1])
        if raw.shape!=(len(reference),19) or not np.allclose(raw[indices,:3],reference.cpu().numpy()[order],rtol=0,atol=1e-12):
            raise ValueError('source oracle mesh/column mismatch')
        aligned=np.empty_like(raw); aligned[order]=raw[indices]
        tensor=lambda lo,hi:torch.as_tensor(aligned[:,lo:hi],device=reference.device)
        passive=d.model.force(positions,.5); pressure=source.force(positions,8); J=tangent.assemble(positions,.5)
        j0=torch.sparse.mm(J,direction.reshape(-1,1)).reshape_as(direction); jp=source.action(positions,8,direction)
        F=d.model.element_gradient(positions)
        stretch=torch.linalg.vector_norm(torch.einsum('eij,ej->ei',F,d.mesh.fiber),dim=-1)
        local=d.ep.geometry.weights.sum(1)[:,None]*stretch[:,None]/4
        rhs=reference.new_zeros(len(reference)).index_add(0,d.mesh.cells.flatten(),local.expand(-1,4).flatten())
        metrics=dict(passive_force=metric(passive,tensor(6,9)),
            endocardial_pressure_force=metric(pressure,tensor(9,12)-tensor(6,9)),
            full_source_force=metric(passive+pressure,tensor(9,12),atol=1e-5,relative=name!='source'),
            passive_tangent=metric(j0,tensor(12,15)),full_source_tangent=metric(j0+jp,tensor(15,18)),
            pressure_tangent=metric(jp,tensor(15,18)-tensor(12,15),rtol=5e-8,atol=2e-5),
            stretch_projection_rhs=metric(rhs,torch.as_tensor(aligned[:,18],device=reference.device),rtol=1e-10,atol=1e-12))
        source_geometry=json.loads((out/(name+'_oracle.csv.json')).read_text())
        metrics['volume']=dict(passed=abs(source_geometry['volume_ml']-whole.volume(positions).item())<=1e-9,
            source_ml=source_geometry['volume_ml'],afsei_ml=whole.volume(positions).item())
        metrics['open_surface_volume']=dict(passed=abs(source_geometry['open_ufl_ml']-source_geometry['open_endo_ml'])<=1e-9,
            difference_ml=source_geometry['open_ufl_ml']-source_geometry['open_endo_ml'])
        verification[name]=dict(metrics=metrics,source_geometry=source_geometry)
    fcap=whole.force(closed.ep.fsi.x,8)-source.force(closed.ep.fsi.x,8)
    closed_as_source=d.model.force(closed.ep.fsi.x,.5)+source.force(closed.ep.fsi.x,8)
    source_as_closed=d.model.force(x,.5)+whole.force(x,8)
    ref=Path('.cache/source-preload-ref')
    summary=dict(passed=all(v['passed'] for c in verification.values() for v in c['metrics'].values()),
        source_commit='5bdde73c6be10cefadb005a0fcec5fdb6521f336',scope='Independent source passive preload weak equilibrium/pressure/tangent/closed volume and loading history',
        regenerated_ufl_sha256=hashlib.sha256((ref/'Active_land.ufl').read_bytes()).hexdigest(),
        regenerated_header_sha256=hashlib.sha256((ref/'Active_land.h').read_bytes()).hexdigest(),
        source_cavity_header_sha256=hashlib.sha256((ref/'cavity_volume.h').read_bytes()).hexdigest(),
        node_count=len(reference),cell_count=len(d.mesh.cells),pressure_mmhg=8.,preload_settings=asdict(settings),
        source_endocardial_volume_ml=whole.volume(x).item(),closed_interface_volume_ml=whole.volume(closed.ep.fsi.x).item(),
        volume_difference_ml=whole.volume(closed.ep.fsi.x).item()-whole.volume(x).item(),
        source_force_residual_norm_dyn=equilibrium,closed_force_residual_norm_dyn=(d.model.force(closed.ep.fsi.x,.5)+whole.force(closed.ep.fsi.x,8)).norm().item(),
        closed_geometry_in_source_residual_norm_dyn=closed_as_source.norm().item(),
        source_geometry_in_closed_residual_norm_dyn=source_as_closed.norm().item(),
        cap_pressure_force_norm_dyn=fcap.norm().item(),
        cap_pressure_resultant_dyn=fcap.sum(0).cpu().tolist(),
        source_basal_resultant_dyn=d.model.basal_force(x).sum(0).cpu().tolist(),
        closed_basal_resultant_dyn=d.model.basal_force(closed.ep.fsi.x).sum(0).cpu().tolist(),
        source_initial_zs_max_abs=zs.abs().max().item(),source_initial_zw_max_abs=zw.abs().max().item(),
        closed_initial_zs_max_abs=closed.ep.strain.zs.abs().max().item(),closed_initial_zw_max_abs=closed.ep.strain.zw.abs().max().item(),
        maximum_position_difference_cm=(x-closed.ep.fsi.x).norm(dim=1).max().item(),
        source_ep_frozen_during_preload=True,source_land_frozen_during_preload=True,
        source_distortion_evolves_during_preload=True,whole_beat_validated=False,quality_blood_mesh_implemented=False,
        verification=verification,elapsed_seconds=time.perf_counter()-start)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
    if not summary['passed']: raise SystemExit('source preload alignment failed')


if __name__=='__main__': main()

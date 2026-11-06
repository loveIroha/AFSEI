"""Frozen production IB quadrature-density comparison; no time advancement.

Exactly one reference mesh, one accepted x, native MAC velocity, integrated
nodal force, consistent mass matrix, kernel and linear-solve tolerance.
Only the interaction point_density is varied. Fixed fields are never mutated.
"""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,json,time,hashlib,statistics,xml.etree.ElementTree as ET
import numpy as np
import torch
from afsi_torch.p1 import prepare_p1
from afsi_torch.mac.grid import MACGrid,divergence
from afsi_torch.fluid.solvers import SolverOptions
from afsi_torch.mac.adaptive_transfer import AdaptiveP1Transfer,InteractionQuadratureOptions,interaction_quadrature_plan
from diagnose_projection_convergence import physical_divergence


def l2_square(value,cells,current_volume):
    """Exact current-volume P1 tetrahedral vector L2 integral."""
    local=value[cells]
    return (current_volume/20*(local.square().sum((1,2))+local.sum(1).square().sum(1))).sum()


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);a=p.parse_args()
    cfg=json.loads(a.config.read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    (out/'status.json').write_text(json.dumps({'stage':'running'}));(out/'config.json').write_text(json.dumps(cfg,indent=2))
    source=Path(cfg['checkpoint']);source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash!=cfg['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    c=json.loads(Path(cfg['source_config']).read_text());data=np.load(source)
    if cfg['densities']!=[2,3]:raise ValueError('This prescribed comparison uses densities2and3')
    torch.set_num_threads(2);start=time.perf_counter()
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA validation required')
    cast=lambda x:torch.as_tensor(np.asarray(x).copy(),device=cfg['device'],dtype=torch.float64)
    X,x=cast(data['reference']),cast(data['x']);cells=torch.as_tensor(data['cells'].copy(),device=cfg['device'])
    velocity=tuple(cast(data[k]) for k in ('u','v','w'));force=cast(data['force'])
    grid=MACGrid(tuple(c['fluid']['shape']),tuple(c['fluid']['lengths_cm']),tuple(c['fluid']['origin_cm']))
    base=InteractionQuadratureOptions(**c['fsi']['interaction_quadrature'])
    plans={str(d):interaction_quadrature_plan(x,cells,grid,replace(base,point_density=float(d))) for d in cfg['densities']}
    # Ceilings are resource checks, never clipping the chosen quadrature.
    max_order=max(base.max_order,max(p['maximum_order'] for p in plans.values()))
    max_points=max(base.max_points,max(p['point_count'] for p in plans.values()))
    common_options=replace(base,max_order=max_order,max_points=max_points)
    geometry=prepare_p1(X,cells,degree=5)  # Same exact P1 mass as production.
    transfer=AdaptiveP1Transfer(grid,geometry,options=SolverOptions(**c['fsi']['mass_solver']),
        mass_backend=c['fsi']['execution']['mass_backend'],warm_start=False,
        quadrature_options=common_options,fused=True)
    macdiv=divergence(velocity,grid.spacing)
    report=dict(scope='one frozen accepted280ms state; no EP/ODE/NS/solid time step',config=cfg,
        checkpoint_sha256=source_hash,time_s=float(data['time_s']),force_time_s=float(data['force_time_s']),
        solid_nodes=len(X),solid_cells=len(cells),fluid_shape=grid.shape,spacing_cm=grid.spacing,
        device=torch.cuda.get_device_name(),physical_gpu=__import__('os').environ.get('AFSEI_GPU'),
        reference_mass='single unchanged production degree5 consistent P1 matrix',mass_nnz=transfer.mass.values().numel(),
        cold_mass_initial_guess_for_each_solve=True,plans=plans,resource_ceilings=dict(max_order=max_order,max_points=max_points),
        original_resource_ceilings=dict(max_order=base.max_order,max_points=base.max_points),
        native_mac_divergence=dict(rms_s_inv=float(macdiv.square().mean().sqrt()),max_s_inv=float(macdiv.abs().max())),
        records=[],checks={})
    snapshots={}
    def pair(keep=False):
        stencil=transfer.prepare(x)
        nodal,ui=transfer.interpolate(velocity,stencil)
        # The mass solver owns its output buffer. Keep a snapshot only OUTSIDE
        # the timed calls; the subsequent force projection reuses that buffer.
        saved=nodal.clone() if keep else None
        spread,fi=transfer.spread(force,stencil)
        return saved,tuple(v.clone() for v in spread) if keep else None,ui,fi
    for density in cfg['densities']:
        transfer.quadrature_options=replace(common_options,point_density=float(density))
        pair()  # Compile/cache this density's actual prepare/gather/spread path.
        torch.cuda.synchronize();samples=[]
        for _ in range(cfg['timing_repeats']):
            began=time.perf_counter();pair();torch.cuda.synchronize();samples.append(time.perf_counter()-began)
        U,spread,ui,fi=pair(keep=True)
        div,J=physical_divergence(geometry,x,U)
        volume=geometry.weights.sum(1)*J;total=volume.sum()
        lagrangian_power=(force*U).sum();eulerian_power=grid.volume*sum((f*u).sum() for f,u in zip(spread,velocity))
        scale=torch.maximum((force*U).abs().sum(),grid.volume*sum((f*u).abs().sum() for f,u in zip(spread,velocity))).clamp_min(1e-30)
        power_error=float((lagrangian_power-eulerian_power).abs()/scale)
        if power_error>1e-8:raise AssertionError('Scaled adjoint power identity failed')
        if float(J.min())<=0:raise AssertionError('Checkpoint geometry inverted')
        snapshots[density]=dict(U=U,spread=spread,div=div)
        record=dict(density=density,**transfer.prepared_quadrature_statistics(),
            stencil_storage_bytes=transfer._last_stencil_bytes,
            mass_interpolate=asdict(ui),mass_spread=asdict(fi),
            wall_volume_ml=float(total),velocity_rms_cm_s=float((l2_square(U,cells,volume)/total).sqrt()),
            structure_divergence_rms_s_inv=float(((volume*div.square()).sum()/total).sqrt()),
            relative_volume_rate_s_inv=float((volume*div).sum()/total),
            volume_rate_ml_s=float((volume*div).sum()),
            lagrangian_power_erg_s=float(lagrangian_power),eulerian_power_erg_s=float(eulerian_power),
            scaled_adjoint_power_defect=power_error,
            prepare_interpolate_spread_median_s=statistics.median(samples),timing_samples_s=samples)
        report['records'].append(record);print(json.dumps(record),flush=True)
        (out/'partial.json').write_text(json.dumps(report,indent=2))
    v2,v3=snapshots[2],snapshots[3];dU=v2['U']-v3['U'];ddiv=v2['div']-v3['div']
    force_den=grid.volume*sum(f.square().sum() for f in v3['spread'])
    force_num=grid.volume*sum((a-b).square().sum() for a,b in zip(v2['spread'],v3['spread']))
    report['comparison']=dict(reference_for_differences='density3 is a finer quadrature, not an exact solution',
        velocity_relative_current_L2=float((l2_square(dU,cells,volume)/l2_square(v3['U'],cells,volume)).sqrt()),
        structure_divergence_difference_rms_s_inv=float(((volume*ddiv.square()).sum()/volume.sum()).sqrt()),
        structure_divergence_relative_L2=float(((volume*ddiv.square()).sum()/(volume*v3['div'].square()).sum()).sqrt()),
        spread_force_relative_eulerian_L2=float((force_num/force_den).sqrt()),
        relative_volume_rate_absolute_difference_s_inv=abs(report['records'][0]['relative_volume_rate_s_inv']-report['records'][1]['relative_volume_rate_s_inv']),
        volume_rate_absolute_difference_ml_s=abs(report['records'][0]['volume_rate_ml_s']-report['records'][1]['volume_rate_ml_s']),
        point_count_ratio_3_to_2=report['records'][1]['point_count']/report['records'][0]['point_count'],
        isolated_operator_time_ratio_3_to_2=report['records'][1]['prepare_interpolate_spread_median_s']/report['records'][0]['prepare_interpolate_spread_median_s'])
    report['checks']=dict(all_mass_solves_accepted=True,adjoint_power_identity_passed=True,
        unchanged_wall_volume=abs(report['records'][0]['wall_volume_ml']/report['records'][1]['wall_volume_ml']-1)<1e-13,
        source_checkpoint_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==source_hash,
        same_mass_matrix_object=True,no_time_advancement=True)
    fields={}
    for density,s in snapshots.items():
        fields[f'velocity_density{density}']=s['U'].cpu().numpy();fields[f'divergence_density{density}']=s['div'].cpu().numpy()
        for name,f in zip(('x','y','z'),s['spread']):fields[f'spread_density{density}_{name}']=f.cpu().numpy()
    np.savez_compressed(out/'operator_fields.npz',**fields)
    if cfg['write_vtk']:
        import meshio
        meshio.write(out/'frozen_transfer.vtu',meshio.Mesh(x.cpu().numpy(),[('tetra',cells.cpu().numpy())],
            point_data=dict(velocity_density2_cm_s=fields['velocity_density2'],velocity_density3_cm_s=fields['velocity_density3'],
                velocity_difference_cm_s=dU.cpu().numpy(),integrated_nodal_force_dyn=force.cpu().numpy()),
            cell_data=dict(J=[J.cpu().numpy()],divergence_density2_s_inv=[fields['divergence_density2']],
                divergence_density3_s_inv=[fields['divergence_density3']],divergence_difference_s_inv=[ddiv.cpu().numpy()])),
            file_format='vtu',binary=True,compression='zlib')
        tree=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian');collection=ET.SubElement(tree,'Collection')
        ET.SubElement(collection,'DataSet',timestep=repr(float(data['time_s'])),group='',part='0',file='frozen_transfer.vtu')
        ET.ElementTree(tree).write(out/'frozen.pvd',encoding='utf-8',xml_declaration=True)
    report['elapsed_diagnostic_seconds']=time.perf_counter()-start
    report['limitations']=['Only the accepted280ms state, not the old279.95ms transfer of the preceding BE step.',
        'Frozen-field differences do not establish new-trajectory accuracy or spatial convergence.',
        'Instantaneous volume rates must not be extrapolated over the full300ms run.',
        'Raw spread-force differences are not differences of pressure-projected fluid responses.',
        'Timing is cached prepare+interpolate+spread with cold mass guesses and no snapshot copies; not full-step timing.']
    report['status']='completed'
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'status.json').write_text(json.dumps(dict(stage='completed',checks_passed=all(report['checks'].values()),time_steps_advanced=0),indent=2))
    print(json.dumps(report['comparison'],indent=2),flush=True)


if __name__=='__main__':main()

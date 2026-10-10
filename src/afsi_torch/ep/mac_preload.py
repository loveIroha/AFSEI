"""Source endocardial-traction preload for the original MAC/IB framework.

The myocardium is equilibrated against a prescribed pressure traction.
MAC fluid starts at rest with zero gauge pressure, as in the original AFSI
box. The traction is retained after initialization. It must be replaced,
rather than added again, when a physical flow-driven cavity pressure is used.
No fitted blood mesh is constructed in this module.
"""
from dataclasses import replace
import hashlib
import json
import numpy as np
import torch
from .. import p1
from ..circulation.checkpoint import digest
from ..two_way.preload import (EndocardialPressure,PassivePreloader,PreloadConfig,
    PreloadResult,PreloadStrainHistory,SourceLoadingHistory)
from .driver import AFSEIStepper,StrainHistory
from .buffered_load import ActiveLoadState


class MACEndocardialPressure(EndocardialPressure):
    """Original source endocardium load; fan cap is used only for volume."""
    def __init__(self,model):
        self.faces=model.endo_faces
        self.rim=model.rim

    def volume(self,x):
        return p1.cavity_volume(x,self.faces,self.rim)


def mesh_binding(mesh):
    binding=hashlib.sha256()
    for value in (mesh.X,mesh.cells,mesh.fiber,mesh.sheet):
        binding.update(value.detach().cpu().numpy().tobytes())
    return binding.hexdigest()


def _check_driver(driver):
    if driver.config.mechanics_profile!='source':
        raise ValueError('source mechanics required for source MAC preload')
    if driver.config.prescribed_endocardial_pressure_mmhg!=0:
        raise ValueError('solve preload with zero existing prescribed traction')
    if driver.config.start_time_s!=driver.protocol.pulse_starts_s[0]:
        raise ValueError('MAC preload must initialize at the first source stimulus')


@torch.no_grad()
def solve(driver,settings=None,*,progress=None):
    _check_driver(driver)
    settings=settings or PreloadConfig(nominal_steps=100,minimum_step_mmhg=.005)
    if settings.nominal_steps<1:
        raise ValueError('nominal source loading steps required')
    history=SourceLoadingHistory(driver,settings.target_mmhg,driver.config.start_time_s,settings.nominal_steps)
    def check_support(old,x):
        driver._fsi_stepper(driver.maximum_ticks).check_support(x)
    loader=PassivePreloader(driver,settings,pressure_load=MACEndocardialPressure(driver.model),
                            geometry_validator=check_support)
    result=loader.solve(progress,observe_state=history.observe)
    return result,history.final()


@torch.no_grad()
def reuse_source_result(driver,path,*,target_mmhg=8.):
    """Read an independently generated SOURCE solid result, never a fluid checkpoint.

Checksum, source loading clock, mesh binding and current true equilibrium
are mandatory. Gates are initialized separately and remain unadvanced.
"""
    _check_driver(driver)
    with np.load(path,allow_pickle=False) as archive:
        arrays={name:archive[name] for name in archive.files}
    try:
        metadata=json.loads(str(arrays.pop('metadata')))
        checksum=metadata.pop('sha256')
        if (metadata.get('producer')!='afsei-source-endo-preload' or metadata.get('schema')!=1
                or metadata.get('completed') is not True or metadata.get('pressure_mmhg')!=target_mmhg
                or metadata.get('mesh_binding_sha256')!=mesh_binding(driver.mesh)
                or digest(metadata,arrays)!=checksum):
            raise ValueError('source preload checksum, mesh, pressure or completion mismatch')
        if set(arrays)!={'reference_x_cm','x_cm','zs','zw'}:
            raise ValueError('a source solid result is required; fluid checkpoints cannot be imported')
        for name,value in arrays.items():
            shape=driver.mesh.X.shape if name.endswith('x_cm') else driver.mesh.X[:,0].shape
            if value.dtype!=np.float64 or value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('invalid source preload array shape/type/values')
        if not np.array_equal(arrays['reference_x_cm'],driver.mesh.X.cpu().numpy()):
            raise ValueError('source preload reference geometry mismatch')
        history=metadata['loading_history']
        endpoints=[e for e in history if e.get('physical_loading_endpoint')]
        if len(endpoints)!=100 or driver.config.start_time_s!=.5:
            raise ValueError('source result must retain 100 x 5 ms loading endpoints')
        for index,entry in enumerate(endpoints,1):
            if (abs(entry['pressure_mmhg']-target_mmhg*index/100)>1e-10
                    or abs(entry['physical_loading_time_s']-.005*index)>1e-10):
                raise ValueError('source pressure/history endpoint mismatch')
        tensor=lambda name:torch.as_tensor(arrays[name],device=driver.mesh.X.device)
        x=tensor('x_cm'); driver.model.validate(x)
        zero=torch.zeros_like(driver.mesh.X[:,0])
        driver.buffer.stage(ActiveLoadState(driver.mesh.X,zero,zero,zero,zero,
                                           driver.config.fsi.time.dt,False))
        pressure=MACEndocardialPressure(driver.model)
        residual=(driver.model.force(x,driver.config.start_time_s)+pressure.force(x,target_mmhg)).norm().item()
        tolerance=max(1e-3,1e-8*pressure.force(x,target_mmhg).norm().item())
        if residual>tolerance: raise ValueError('source preload fails current passive equilibrium')
        F=driver.model.element_gradient(x)
        cell=torch.linalg.vector_norm(torch.einsum('eij,ej->ei',F,driver.mesh.fiber),dim=-1)
        stretch=driver.ep.project_cell_field(cell)
        strain=PreloadStrainHistory(x,stretch,tensor('zs'),tensor('zw'),target_mmhg,.5,100)
        for name in ('zs','zw'):
            observed=getattr(strain,name).abs().max().item()
            if abs(observed-endpoints[-1][name+'_max_abs'])>1e-10:
                raise ValueError('source distortion history does not match final loading endpoint')
        result=PreloadResult(x,target_mmhg,residual,tolerance,pressure.volume(x).item(),
            pressure.volume(driver.mesh.X).item(),history,[])
        return result,strain
    except (KeyError,TypeError,IndexError) as exc:
        raise ValueError('malformed source preload result') from exc


@torch.no_grad()
def initialize(driver,result,strain_history,*,validate_static=True):
    """Create a NEW MAC checkpoint with retained pressure and zero initial flow."""
    _check_driver(driver)
    h=strain_history
    if h.pressure_mmhg!=result.pressure_mmhg or not torch.equal(h.x,result.x):
        raise ValueError('source strain history belongs to another preload geometry')
    config=replace(driver.config,prescribed_endocardial_pressure_mmhg=result.pressure_mmhg)
    prepared=AFSEIStepper(driver.mesh,config)
    state=prepared.initialize(x=result.x)
    prepared.ode._validate(state.tissue,h.stretch,h.zs,h.zw,state.fsi.time)
    if not torch.allclose(state.strain.stretch,h.stretch,rtol=1e-11,atol=1e-11):
        raise ValueError('loading stretch and final geometry disagree')
    strain=StrainHistory(h.stretch.clone(),h.zs.clone(),h.zw.clone())
    load=replace(state.force_load,use_land=False,zs_old=strain.zs.clone(),zw_old=strain.zw.clone())
    prepared.buffer.stage(load)
    force=prepared.model.force(state.fsi.x,state.fsi.time)
    error=force.norm().item()
    if error>result.force_tolerance_dyn:
        raise RuntimeError('MAC force does not retain the source preload balance')
    state=replace(state,strain=strain,force_load=load,fsi=replace(state.fsi,force=force))
    static={}
    if validate_static:
        # Discard this diagnostic trial: no EP, gates, history or clock are published.
        endpoint=prepared._fsi_stepper(prepared.maximum_ticks)
        probe,info=endpoint.physical_step(state.fsi,state.fsi.time+config.fsi.time.dt,True)
        displacement=(probe.x-state.fsi.x).norm().item()
        speed=max(v.abs().max().item() for v in probe.velocity)
        if displacement>1e-9 or speed>1e-6:
            raise RuntimeError(f'MAC passive preload creates excess drift: |dx|={displacement:g}, |u|={speed:g}')
        static=dict(static_probe_displacement_l2_cm=displacement,static_probe_fluid_speed_max_cm_s=speed,
            static_probe_nonlinear_residual=info['nonlinear']['residual_norm'],
            static_probe_nonlinear_tolerance=info['nonlinear']['tolerance'],
            static_probe_divergence_l2=info['divergence_l2'],static_probe_power_error=info['power_error'])
        prepared._reset_scratch(state)
    report=dict(preload_performed=True,preload_kind='source-endocardial-traction; original MAC box',
        prescribed_endocardial_pressure_mmhg=result.pressure_mmhg,
        preload_volume_ml=result.volume_ml,reference_volume_ml=result.reference_volume_ml,
        force_residual_norm_dyn=error,force_tolerance_dyn=result.force_tolerance_dyn,
        loading_history_steps=h.steps,loading_history_duration_s=h.duration_s,
        initial_zs_max_abs=strain.zs.abs().max().item(),initial_zw_max_abs=strain.zw.abs().max().item(),
        initial_mac_gauge_pressure_mmhg=0.,initial_fluid_velocity_max_cm_s=0.,
        ep_advanced_during_preload=False,land_gates_advanced_during_preload=False,
        strain_history_advanced_during_preload=True,fluid_advanced_during_preload=False,
        reference_geometry_retained=True,physical_cavity_pressure_initialized=False,
        windkessel_ports_connected=False,three_field_integration_implemented=False,**static)
    return prepared,state,report

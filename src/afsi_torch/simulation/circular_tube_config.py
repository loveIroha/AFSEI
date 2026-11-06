"""Strict JSON adapter for the 3D active tube and the existing BEIBStepper.

This contains configuration only. It does not build a mesh, initialize CUDA,
compile kernels, or advance a simulation.
"""
from dataclasses import dataclass
from math import isfinite
from ..config import TimeConfig, FluidConfig, LVExecutionConfig, OutputConfig
from ..mac.backward_euler import BEFlowOptions
from ..mac.multigrid import MGOptions
from ..mac.adaptive_transfer import InteractionQuadratureOptions
from ..mac.midpoint_solver import AndersonOptions
from ..nonlinear import NewtonOptions, GMRESOptions
from ..fluid.solvers import SolverOptions
from ..ep.monodomain import MonodomainParameters
from ..ep.numerics import MonodomainNumerics
from ..solids.circular_tube import CircularTubeGeometry
from ..solids.volumetric_stabilization import validate_volumetric_stabilization


@dataclass(frozen=True)
class TubeFSIConfig:
    time: TimeConfig
    fluid: FluidConfig
    flow: BEFlowOptions
    pressure_solver: MGOptions
    mass_solver: SolverOptions
    execution: LVExecutionConfig
    interaction_quadrature: InteractionQuadratureOptions
    nonlinear: NewtonOptions
    output: OutputConfig
    interaction_degree: int = 2
    nonlinear_solver: str = 'newton'
    newton_preconditioner: str = 'none'
    ib_response_backend: str = 'quadrature'
    support_backend: str = 'vertices'
    anderson: AndersonOptions = AndersonOptions()
    max_courant: float = 1.


def _keys(value, names, where):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f'{where} requires exactly: {", ".join(sorted(names))}')


def _positive(value, where, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
            or value < 0 or (not zero and value == 0)):
        raise ValueError(f'{where} must be finite and {"nonnegative" if zero else "positive"}')


def resolve_config(c):
    """Validate physical meaning and instantiate existing solver option types."""
    _keys({k:v for k,v in c.items() if k not in ('circulation', 'hydraulic_domain', 'solid_refinement_factor')}, ('demo', 'description', 'geometry', 'fluid', 'solid', 'active', 'ep',
              'stimulus', 'time', 'fsi', 'output'), 'tube config')
    if c['demo'] != 'active-tube-3d' or not isinstance(c['description'], str):
        raise ValueError('demo must be active-tube-3d with a text description')
    factor = c.get('solid_refinement_factor', 1)
    if type(factor) is not int or factor not in (1, 2, 3, 4):
        raise ValueError('solid_refinement_factor must be 1, 2, 3 or 4')
    if factor != 1 and ('layers' in c['solid'] or 'hydraulic_domain' in c or 'circulation' in c
                        or c['active'].get('stretch_feedback') != 'isometric-driver'):
        raise ValueError('solid-only refinement currently requires the open single-layer isometric-driver case')
    geometry = CircularTubeGeometry(**c['geometry'])
    f = c['fluid']
    _keys(f, ('shape', 'lengths_cm', 'origin_cm', 'density_g_per_cm3',
              'viscosity_dyn_s_per_cm2', 'boundary'), 'fluid')
    rc = f['boundary'] == 'zero-reference-pressure-with-rc-box-outlet'
    if rc:
        from ..circulation.rc_outlet import parse_rc
        parse_rc(c.get('circulation'), dimension=3)
    elif f['boundary'] != 'zero-pressure-neumann-velocity-all-box-faces' or 'circulation' in c:
        raise ValueError('select the original zero-pressure box or the explicit RC boundary-validation case')
    if 'hydraulic_domain' in c:
        from ..mac.sealed_ports import validate_hydraulic
        validate_hydraulic(c['hydraulic_domain'])
        if not rc: raise ValueError('sealed reservoirs require the RC case')
    fluid = FluidConfig(shape=f['shape'], lengths=f['lengths_cm'], origin=f['origin_cm'],
                        rho=f['density_g_per_cm3'], mu=f['viscosity_dyn_s_per_cm2'])
    if len(fluid.shape) != 3:
        raise ValueError('three-dimensional MAC grid required')
    # Four-point delta support needs two grid spacings on every box face.
    radius = geometry.inner_radius_cm+geometry.wall_thickness_cm
    low = (*[v-radius for v in geometry.center_xy_cm], geometry.axial_start_cm)
    high = (*[v+radius for v in geometry.center_xy_cm], geometry.axial_start_cm+geometry.length_cm)
    if any(a-o <= 2*h or o+L-b <= 2*h for a,b,o,L,h in
           zip(low, high, fluid.origin, fluid.lengths, fluid.spacing)):
        raise ValueError('reference wall must lie more than two MAC spacings inside every box face')
    t = c['time']
    _keys(t, ('fsi_dt_s', 'ep_dt_s', 'ode_dt_ms', 'end_time_s'), 'time')
    for name, value in t.items():
        _positive(value, 'time.'+name)
    time = TimeConfig(t['fsi_dt_s'], t['end_time_s'])
    if abs(t['ep_dt_s']-t['fsi_dt_s']) > 1e-14:
        raise ValueError('this demo uses one EP update per FSI step: ep_dt_s must equal fsi_dt_s')
    ratio = t['ep_dt_s']*1000/t['ode_dt_ms']
    if round(ratio) < 1 or abs(ratio-round(ratio)) > 1e-9:
        raise ValueError('EP dt must contain an integer number of ODE substeps')
    s = c['solid']
    _positive(s['end_spring_dyn_cm3'], 'solid.end_spring', zero=True)
    if 'layers' in s:
        if 'volumetric_stabilization' in s:
            raise ValueError('volumetric stabilization currently requires the single-layer tube')
        from ..ep.layered_circular_tube import validate_layers
        _keys(s, ('layers','end_condition','end_spring_dyn_cm3'), 'layered solid')
        validate_layers(s['layers'])
        if len(geometry.radial_elements_per_layer) != 2:
            raise ValueError('two conforming radial layers required')
        if s['end_condition'] != 'full-reference-surface-spring-approximation' or s['end_spring_dyn_cm3'] <= 0:
            raise ValueError('layered case currently uses explicitly approximate full-vector end springs')
    else:
        _keys({k:v for k,v in s.items() if k not in ('volumetric_stabilization', 'collar_support')},
              ('young_modulus_dyn_cm2', 'poisson_ratio', 'end_condition','end_spring_dyn_cm3'), 'solid')
        validate_volumetric_stabilization(s.get('volumetric_stabilization'))
        if 'collar_support' in s:
            anchor = s['collar_support']
            _keys(anchor, ('width_cm', 'stiffness_dyn_cm4'), 'solid.collar_support')
            for key, value in anchor.items():
                _positive(value, 'solid.collar_support.'+key)
            if ('hydraulic_domain' not in c or
                    anchor['width_cm'] < c['hydraulic_domain']['collar_cells']*fluid.spacing[2]-1e-12 or
                    anchor['width_cm']+c['active']['end_taper_cm'] >= geometry.length_cm/2):
                raise ValueError('elastic collar must cover the hydraulic collar and leave a free active segment')
            dz = geometry.length_cm/geometry.axial_elements
            if abs(anchor['width_cm']/dz-round(anchor['width_cm']/dz)) > 1e-9:
                raise ValueError('elastic collar must align with solid axial layers')
        _positive(s['young_modulus_dyn_cm2'], 'solid.E')
        if geometry.radial_interfaces_cm:
            raise ValueError('radial layer geometry requires explicit layer materials')
        if s['end_condition'] not in ('axial-reference-surface-spring','full-reference-surface-spring-approximation') or not 0 <= s['poisson_ratio'] < .5:
            raise ValueError('use an explicitly finite end spring and 0 <= Poisson ratio < 0.5')
    _keys({k:v for k,v in c['active'].items() if k not in ('stretch_feedback','land_reference_ratio')}, ('mechanism', 'rest_reference', 'tension_scale', 'end_taper_cm',
                        'minimum_radius_ratio'), 'active')
    a = c['active']
    reference_ratio = a.get('land_reference_ratio', 1.)
    if isinstance(reference_ratio, bool) or not isfinite(reference_ratio) or not 0 < reference_ratio <= 1.2:
        raise ValueError('active.land_reference_ratio must be positive and at most 1.2')
    if a.get('stretch_feedback','circumferential') not in ('circumferential','isometric-driver'):
        raise ValueError('unsupported active.stretch_feedback')
    if 'layers' in s and a.get('stretch_feedback','circumferential') != 'circumferential':
        raise ValueError('isometric-driver is only enabled for the single-layer verification tube')
    if (a['mechanism'] not in ('circumferential-land-pk1', 'paired-transverse-land-traction') or
            a['rest_reference'] != 'land-equilibrium-increment'):
        raise ValueError('supported Land actuator with its resting equilibrium reference required')
    if a['mechanism'] == 'paired-transverse-land-traction':
        if ('layers' in s or factor != 1 or geometry.circumferential_elements % 4 or
                a.get('stretch_feedback') != 'isometric-driver' or reference_ratio != 1.):
            raise ValueError('paired actuator requires single-layer same mesh, lambda=1 isometric driver, and fourfold circumferential resolution')
    if (not isfinite(a['tension_scale']) or not 0 <= a['tension_scale'] <= 1 or
            not isfinite(a['end_taper_cm']) or not 0 < a['end_taper_cm'] < geometry.length_cm/2 or
            not isfinite(a['minimum_radius_ratio']) or not 0 < a['minimum_radius_ratio'] < 1):
        raise ValueError('invalid active scale, end taper or no-contact stop limit')
    ep_keys = ('parameters', 'numerics', 'conductivity_direction', 'boundary')
    _keys(c['ep'], ep_keys + tuple(k for k in ('ode_method','deformation_feedback') if k in c['ep']), 'ep')
    if c['ep'].get('deformation_feedback','none') not in ('none','santiago2018'):
        raise ValueError('EP deformation_feedback must be none or santiago2018')
    if c['ep'].get('deformation_feedback','none')!='none' and (factor!=1 or 'layers' in s):
        raise ValueError('3D tube EP deformation feedback currently requires the single-layer same-mesh case')
    if c['ep'].get('ode_method', 'sdirk') not in ('sdirk', 'explicit_euler'):
        raise ValueError('EP ode_method must be sdirk or explicit_euler')
    if c['ep']['conductivity_direction'] != 'axial-z' or c['ep']['boundary'] != 'insulating-all-wall-surfaces':
        raise ValueError('EP conduction must be axial with homogeneous insulating boundaries')
    parameters = MonodomainParameters(**c['ep']['parameters'])
    n = dict(c['ep']['numerics']); n['solver'] = SolverOptions(**n['solver'])
    numerics = MonodomainNumerics(**n)
    if numerics.discretization not in ('nodal_p1', 'consistent_p1'):
        raise ValueError('active-tube-3d requires P1 FEM')
    stimulus = c['stimulus']
    _keys(stimulus, ('start_s', 'duration_s', 'width_cm', 'amplitude_ua_per_cm2'), 'stimulus')
    for name, value in stimulus.items():
        _positive(value, 'stimulus.'+name, zero=name in ('start_s', 'amplitude_ua_per_cm2'))
    if stimulus['width_cm'] > geometry.length_cm:
        raise ValueError('stimulus axial width exceeds tube length')
    fs = c['fsi']
    _keys(fs, ('spatial_coupling', 'time_scheme', 'execution', 'flow', 'pressure_solver', 'mass_solver', 'interaction_quadrature',
               'nonlinear', 'nonlinear_solver', 'max_courant'), 'fsi')
    if fs['spatial_coupling'] != 'griffith-luo-2017-unified':
        raise ValueError('this demo uses the Griffith-Luo unified quadrature FE/IB spatial coupling')
    if fs['time_scheme'] != 'be-be':
        raise ValueError('this demo currently connects BEIBStepper; CN-AB2 cannot be selected by relabeling the config')
    flow_options=dict(fs['flow'])
    viscous_solver=flow_options.pop('helmholtz_solver','jacobi')
    if viscous_solver not in ('jacobi','pcg'):
        raise ValueError('helmholtz_solver must be jacobi or pcg')
    if viscous_solver!='jacobi' and 'hydraulic_domain' not in c:
        raise ValueError('optional PCG viscosity solver currently requires the sealed tube adapter')
    nonlinear = dict(fs['nonlinear']); nonlinear['linear'] = GMRESOptions(**nonlinear['linear'])
    if fs['nonlinear_solver'] not in ('newton', 'anderson-newton', 'jfnk'):
        raise ValueError('unsupported original BEIBStepper nonlinear solver')
    _positive(fs['max_courant'], 'fsi.max_courant')
    _keys(c['output'], ('log_every', 'field_every', 'checkpoint_every', 'write_vtk'), 'output')
    output = OutputConfig(log_every=c['output']['log_every'], output_every=c['output']['field_every'],
        checkpoint_every=c['output']['checkpoint_every'], write_vtk=c['output']['write_vtk'])
    result = TubeFSIConfig(time, fluid, BEFlowOptions(**flow_options), MGOptions(**fs['pressure_solver']),
        SolverOptions(**fs['mass_solver']), LVExecutionConfig(**fs['execution']),
        InteractionQuadratureOptions(**fs['interaction_quadrature']), NewtonOptions(**nonlinear), output,
        nonlinear_solver=fs['nonlinear_solver'], max_courant=fs['max_courant'])
    if result.execution.ib_backend != 'reference':
        raise ValueError('this P1 BE demo selects its shared IB implementation through interaction_quadrature')
    if fs['nonlinear_solver'] == 'jfnk' and (result.nonlinear.linear.check_policy != 'periodic'
                                           or result.nonlinear.linear_forcing != 'fixed'):
        raise ValueError('JFNK requires the original periodic linear checks and fixed forcing')
    return geometry, result, parameters, numerics


def config_summary(c):
    geometry, fsi, _, numerics = resolve_config(c)
    na, nt, nr = geometry.axial_elements, geometry.circumferential_elements, geometry.thickness_elements
    factor=c.get('solid_refinement_factor',1)
    return dict(status='configuration-valid', simulation_started=False,
        demo=c['demo'], solid_nodes=(factor*na+1)*(factor*nr+1)*factor*nt,
        solid_tetrahedra=6*na*nr*nt*factor**3,solid_refinement_factor=factor,
        original_solid_nodes=(na+1)*(nr+1)*nt,
        fluid_shape=fsi.fluid.shape, planned_steps=round(fsi.time.end_time/fsi.time.dt),
        ep_discretization=numerics.discretization, ep_backend=numerics.backend,
        active_mechanism=c['active']['mechanism'],
        fluid_solid_solver='original BEIBStepper / backward-Euler MAC / pressure multigrid',
        spatial_coupling=c['fsi']['spatial_coupling'], time_scheme=c['fsi']['time_scheme'],
        complete_griffith_luo_2017_time_scheme=False,
        cnab_vs_bebe_performance_measured=False,
        porous_media=False, valves=False, circulation='circulation' in c,
        rc_outlet_scope=c.get('circulation', {}).get('outlet_scope'),
        literature_geometry_aligned=False,
        physical_tube_mouth_boundary_aligned=False,
        hydraulic_path_isolated='hydraulic_domain' in c,
        layered_wall='layers' in c['solid'],
        myocardium_only_ep='layers' in c['solid'],
        exact_fixed_solid_ends=False,
        viscous_linear_solver=c['fsi']['flow'].get('helmholtz_solver','jacobi'),
        gpu_numerics_validated=False,
        volumetric_stabilization=c['solid'].get('volumetric_stabilization'))

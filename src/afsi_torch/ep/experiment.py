"""Typed, serializable fixed-tissue experiment settings; solver settings reused."""
from dataclasses import dataclass, field
from math import isfinite
from .initial import RESTING_VOLTAGE_V
from .monodomain import MonodomainParameters
from .numerics import MonodomainNumerics


def _vector(value, name, *, positive=False):
    value = tuple(value)
    if len(value) != 3 or any(isinstance(v, bool) or not isinstance(v, (int,float)) or
            not isfinite(v) or (positive and v <= 0) for v in value):
        raise ValueError(f'{name} requires three finite'+(' positive' if positive else '')+' numbers')
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')


def _grid_time(value, dt, name):
    if isinstance(value, bool) or not isfinite(value) or value < 0 or abs(round(value/dt)*dt-value) > 1e-12:
        raise ValueError(f'{name} must be nonnegative and on the EP time grid')


@dataclass(frozen=True)
class TissueDomainConfig:
    lengths_cm: tuple = (1., 1., 1.)
    origin_cm: tuple = (0., 0., 0.)
    spacing_cm: float = .05
    mesh: str = 'structured_freudenthal_tetrahedra'
    fiber: tuple = (1., 0., 0.)
    sheet: tuple = (0., 1., 0.)

    def __post_init__(self):
        for name in ('lengths_cm','origin_cm','fiber','sheet'):
            object.__setattr__(self, name, _vector(getattr(self,name), name, positive=name=='lengths_cm'))
        _positive(self.spacing_cm, 'mesh spacing')
        if self.mesh != 'structured_freudenthal_tetrahedra':
            raise ValueError('unsupported tissue mesh')
        if any(round(v/self.spacing_cm) < 1 or abs(round(v/self.spacing_cm)*self.spacing_cm-v)>1e-12
               for v in self.lengths_cm):
            raise ValueError('lengths must be integer multiples of spacing_cm')
        if (abs(sum(v*v for v in self.fiber)-1) > 1e-12 or
                abs(sum(v*v for v in self.sheet)-1) > 1e-12 or
                abs(sum(a*b for a,b in zip(self.fiber,self.sheet))) > 1e-12):
            raise ValueError('demo fiber and sheet must form an orthonormal frame')


@dataclass(frozen=True)
class EPExperimentTime:
    dt_ep_s: float = 5e-5
    dt_ode_ms: float = .002
    end_time_s: float = .08

    def __post_init__(self):
        for name in ('dt_ep_s','dt_ode_ms','end_time_s'):
            _positive(getattr(self,name), name)
        _grid_time(self.end_time_s, self.dt_ep_s, 'end_time_s')
        ratio = self.dt_ep_s*1000/self.dt_ode_ms
        if round(ratio) < 1 or abs(round(ratio)-ratio) > 1e-9:
            raise ValueError('EP step must contain an integer number of ODE substeps')


@dataclass(frozen=True)
class TissueStimulus:
    kind: str = 'box_current'
    minimum_cm: tuple = (0., 0., 0.)
    maximum_cm: tuple = (.15, .15, .15)
    amplitude_ua_per_cm2: float = 25.
    start_s: float = 0.
    duration_s: float = .002

    def __post_init__(self):
        object.__setattr__(self,'minimum_cm',_vector(self.minimum_cm,'stimulus minimum'))
        object.__setattr__(self,'maximum_cm',_vector(self.maximum_cm,'stimulus maximum'))
        if self.kind != 'box_current' or any(b<=a for a,b in zip(self.minimum_cm,self.maximum_cm)):
            raise ValueError('stimulus must be a nonempty box current')
        _positive(self.amplitude_ua_per_cm2, 'stimulus amplitude')
        _positive(self.duration_s, 'stimulus duration')
        if not isfinite(self.start_s) or self.start_s < 0:
            raise ValueError('nonnegative stimulus start required')


@dataclass(frozen=True)
class TissueBoundary:
    electrical: str = 'zero_normal_current_all_faces'
    geometry: str = 'fixed_reference'
    solid_displacement_cm: tuple = (0., 0., 0.)
    fluid_velocity_cm_s: tuple = (0., 0., 0.)

    def __post_init__(self):
        for name in ('solid_displacement_cm','fluid_velocity_cm_s'):
            object.__setattr__(self,name,_vector(getattr(self,name),name))
        if (self.electrical != 'zero_normal_current_all_faces' or self.geometry != 'fixed_reference' or
                any(self.solid_displacement_cm) or any(self.fluid_velocity_cm_s)):
            raise ValueError('this EP-only experiment requires fixed geometry and insulating boundaries')


@dataclass(frozen=True)
class TissueInitial:
    state: str = 'afsei_pinned_gpb_rest'
    voltage_v: float = RESTING_VOLTAGE_V
    stretch: float = 1.
    zs: float = 0.
    zw: float = 0.

    def __post_init__(self):
        if (self.state != 'afsei_pinned_gpb_rest' or self.voltage_v != RESTING_VOLTAGE_V or
                self.stretch != 1. or self.zs != 0. or self.zw != 0.):
            raise ValueError('use the pinned GPB rest state with unit stretch and zero distortion')


@dataclass(frozen=True)
class TissueObservations:
    activation_threshold_mv: float = 0.
    slice_axis: int = 2
    slice_coordinate_cm: float = 0.
    line_axis: int = 0
    line_anchor_cm: tuple = (0., 0., 0.)
    probes_cm: tuple = ((0.,0.,0.), (.5,0.,0.), (1.,0.,0.))

    def __post_init__(self):
        if (type(self.slice_axis) is not int or self.slice_axis not in (0,1,2) or
                type(self.line_axis) is not int or self.line_axis not in (0,1,2) or
                not isfinite(self.activation_threshold_mv) or not isfinite(self.slice_coordinate_cm)):
            raise ValueError('finite threshold/slice and axes 0, 1 or 2 required')
        object.__setattr__(self,'line_anchor_cm',_vector(self.line_anchor_cm,'line anchor'))
        object.__setattr__(self,'probes_cm',tuple(_vector(p,'probe') for p in self.probes_cm))
        if not self.probes_cm:
            raise ValueError('at least one observation probe required')


@dataclass(frozen=True)
class TissueOutput:
    log_every: int = 100
    trace_every: int = 10
    vtk_every: int = 200
    write_vtk: bool = True
    write_plots: bool = True
    snapshot_times_s: tuple = (0.,.002,.005,.01,.02,.04,.06,.08)

    def __post_init__(self):
        if any(type(v) is not int or v<1 for v in (self.log_every,self.trace_every,self.vtk_every)):
            raise ValueError('output intervals must be positive integers')
        if type(self.write_vtk) is not bool or type(self.write_plots) is not bool:
            raise ValueError('output switches must be bool')
        times=tuple(self.snapshot_times_s)
        if any(not isfinite(v) or v<0 for v in times) or tuple(sorted(set(times))) != times:
            raise ValueError('snapshot times must be finite, nonnegative, increasing and unique')
        object.__setattr__(self,'snapshot_times_s',times)


@dataclass(frozen=True)
class TissueReference:
    doi: str = ''
    figure: str = ''
    comparison_scope: str = 'fixed tissue propagation with AFSEI GPB'
    parameter_source: str = 'explicit experiment configuration'
    stimulus_source: str = 'explicit experiment configuration'


@dataclass(frozen=True)
class EPTissueConfig:
    schema_version: int = 1
    case_name: str = 'fixed_ep_tissue'
    cell_model: str = 'gpb'
    advance_land: bool = False
    domain: TissueDomainConfig = field(default_factory=TissueDomainConfig)
    parameters: MonodomainParameters = field(default_factory=MonodomainParameters)
    ep_numerics: MonodomainNumerics = field(default_factory=MonodomainNumerics)
    time: EPExperimentTime = field(default_factory=EPExperimentTime)
    stimulus: TissueStimulus = field(default_factory=TissueStimulus)
    boundary: TissueBoundary = field(default_factory=TissueBoundary)
    initial: TissueInitial = field(default_factory=TissueInitial)
    observations: TissueObservations = field(default_factory=TissueObservations)
    output: TissueOutput = field(default_factory=TissueOutput)
    reference: TissueReference = field(default_factory=TissueReference)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or not self.case_name:
            raise ValueError('named experiment with schema_version=1 required')
        if self.cell_model != 'gpb' or self.advance_land is not False:
            raise ValueError('EP-only experiment uses GPB and does not advance Land')
        lo=self.domain.origin_cm
        hi=tuple(o+L for o,L in zip(lo,self.domain.lengths_cm))
        for a,b,m,M in zip(lo,hi,self.stimulus.minimum_cm,self.stimulus.maximum_cm):
            if m<a-1e-12 or M>b+1e-12:
                raise ValueError('stimulus box must be inside the tissue')
        for name,value in (('stimulus start',self.stimulus.start_s),
                           ('stimulus end',self.stimulus.start_s+self.stimulus.duration_s)):
            _grid_time(value,self.time.dt_ep_s,name)
        if self.stimulus.start_s >= self.time.end_time_s:
            raise ValueError('stimulus must start inside the simulated interval')
        for value in self.output.snapshot_times_s:
            _grid_time(value,self.time.dt_ep_s,'snapshot time')
        for p in (*self.observations.probes_cm,self.observations.line_anchor_cm):
            if any(v<a-1e-12 or v>b+1e-12 for v,a,b in zip(p,lo,hi)):
                raise ValueError('observation coordinates must be inside the tissue')
        axis=self.observations.slice_axis
        if not lo[axis]-1e-12<=self.observations.slice_coordinate_cm<=hi[axis]+1e-12:
            raise ValueError('slice must be inside the tissue')

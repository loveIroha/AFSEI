"""Serializable integration settings; full circulation remains a later stage."""
from dataclasses import dataclass, field
from math import isfinite
from ..paper_lv import PaperLVConfig
from .monodomain import MonodomainParameters
from .numerics import MonodomainNumerics
from .protocol import EPProtocol,ticks
from .tissue import GPBLand, ODERecoveryConfig
from .source_mechanics import source_fsi_config,source_parameters_match


@dataclass(frozen=True)
class AFSEIConfig:
    fsi: PaperLVConfig = field(default_factory=PaperLVConfig)
    ep: MonodomainParameters = field(default_factory=MonodomainParameters)
    ep_numerics: MonodomainNumerics = field(default_factory=MonodomainNumerics)
    protocol: EPProtocol = field(default_factory=EPProtocol)
    dt_ode_ms: float = .002
    ode_recovery: ODERecoveryConfig = field(default_factory=ODERecoveryConfig)
    start_time_s: float = .5
    max_retries: int = 4
    direction_mapping: str = 'l2-cg1'
    # Missing in old checkpoints: inherited preserves their exact parameters.
    mechanics_profile: str = 'inherited'
    # Original AFSI endocardial traction; distinct from physical fluid/RCR pressure.
    prescribed_endocardial_pressure_mmhg: float = 0.
    ep_refinement_factor: int = 1
    ep_refinement_cache: str | None = None
    ep_deformation_feedback: str = 'none'

    @classmethod
    def source(cls,*,fsi=None,**kwargs):
        return cls(fsi=source_fsi_config(fsi or PaperLVConfig()),mechanics_profile='source',**kwargs)

    def __post_init__(self):
        if self.ep_deformation_feedback not in ('none','santiago2018'):
            raise ValueError('EP deformation feedback must be none or santiago2018')
        if self.ep_deformation_feedback!='none' and self.ep_refinement_factor!=1:
            raise ValueError('deformation feedback currently requires matching EP/solid nodes')
        if (not isinstance(self.fsi,PaperLVConfig) or not isinstance(self.ep,MonodomainParameters)
                or not isinstance(self.ep_numerics,MonodomainNumerics)
                or not isinstance(self.protocol,EPProtocol) or self.fsi.solid_degree != 5
                or type(self.max_retries) is not int or self.max_retries < 0
                or self.direction_mapping not in ('source-serial-cg1','l2-cg1','afsi-dg0')
                or self.mechanics_profile not in ('inherited','source')
                or not isfinite(self.start_time_s) or self.start_time_s < 0
                or not isfinite(self.prescribed_endocardial_pressure_mmhg)
                or self.prescribed_endocardial_pressure_mmhg < 0):
            raise ValueError('invalid AFSEI integration settings')
        if type(self.ep_refinement_factor) is not int or self.ep_refinement_factor not in (1,3):
            raise ValueError('supported EP edge subdivision factors are 1 and 3')
        if self.mechanics_profile == 'source' and not source_parameters_match(self.fsi):
            raise ValueError('source mechanics profile requires the pinned source material and basal coefficients')
        GPBLand(self.protocol.dt_ep_s,self.dt_ode_ms,recovery=self.ode_recovery)
        start = ticks(self.start_time_s,self.protocol.dt_ep_s)
        if start > self.protocol.pulse_ticks[0]:
            raise ValueError('a fresh run cannot skip the first stimulus; use a checkpoint to resume')
        dt_ticks = ticks(self.fsi.time.dt,self.protocol.dt_ep_s)
        if dt_ticks < 1:
            raise ValueError('mechanical step must contain at least one EP tick')
        if dt_ticks & (dt_ticks-1):
            raise ValueError('mechanical maximum step must contain a power-of-two number of EP ticks')


def decode_checkpoint_config(data):
    """Missing numerical settings identify the original consistent/Jacobi EP.

    Fresh configs default to nodal/light Jacobi/CN; a restart never silently switches
    discretization or time scheme. A fresh run from a reference checkpoint selects defaults
    explicitly in its entry point.
    """
    from dataclasses import asdict
    from ..config import _decode
    if not isinstance(data, dict):
        raise ValueError('checkpoint AFSEI config must be an object')
    numerics=dict(data.get('ep_numerics',asdict(MonodomainNumerics.legacy())))
    # Old serialized numerical objects may omit defaults. Preserve their
    # original AMG backend and serial direction assignment on archival restart.
    if 'ep_numerics' in data:
        numerics.setdefault('backend','amgx_pcg_amg')
    # All checkpoints written before time_scheme existed used backward Euler,
    # including the later nodal/AmgX checkpoints. Do not retime a restart.
    numerics.setdefault('time_scheme','backward_euler')
    payload=dict(data,ep_numerics=numerics)
    payload.setdefault('direction_mapping','source-serial-cg1')
    return _decode(AFSEIConfig,payload)

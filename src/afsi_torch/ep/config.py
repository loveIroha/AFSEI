"""Serializable integration settings; full circulation remains a later stage."""
from dataclasses import dataclass, field
from math import isfinite
from ..paper_lv import PaperLVConfig
from .monodomain import MonodomainParameters
from .numerics import MonodomainNumerics
from .protocol import EPProtocol,ticks
from .tissue import GPBLand
from .source_mechanics import source_fsi_config,source_parameters_match


@dataclass(frozen=True)
class AFSEIConfig:
    fsi: PaperLVConfig = field(default_factory=PaperLVConfig)
    ep: MonodomainParameters = field(default_factory=MonodomainParameters)
    ep_numerics: MonodomainNumerics = field(default_factory=MonodomainNumerics)
    protocol: EPProtocol = field(default_factory=EPProtocol)
    dt_ode_ms: float = .002
    start_time_s: float = .5
    max_retries: int = 4
    direction_mapping: str = 'source-serial-cg1'
    # Missing in old checkpoints: inherited preserves their exact parameters.
    mechanics_profile: str = 'inherited'
    # Original AFSI endocardial traction; distinct from physical fluid/RCR pressure.
    prescribed_endocardial_pressure_mmhg: float = 0.

    @classmethod
    def source(cls,*,fsi=None,**kwargs):
        return cls(fsi=source_fsi_config(fsi or PaperLVConfig()),mechanics_profile='source',**kwargs)

    def __post_init__(self):
        if (not isinstance(self.fsi,PaperLVConfig) or not isinstance(self.ep,MonodomainParameters)
                or not isinstance(self.ep_numerics,MonodomainNumerics)
                or not isinstance(self.protocol,EPProtocol) or self.fsi.solid_degree != 5
                or type(self.max_retries) is not int or self.max_retries < 0
                or self.direction_mapping not in ('source-serial-cg1','afsi-dg0')
                or self.mechanics_profile not in ('inherited','source')
                or not isfinite(self.start_time_s) or self.start_time_s < 0
                or not isfinite(self.prescribed_endocardial_pressure_mmhg)
                or self.prescribed_endocardial_pressure_mmhg < 0):
            raise ValueError('invalid AFSEI integration settings')
        if self.mechanics_profile == 'source' and not source_parameters_match(self.fsi):
            raise ValueError('source mechanics profile requires the pinned source material and basal coefficients')
        GPBLand(self.protocol.dt_ep_s,self.dt_ode_ms)
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

    Fresh configs default to nodal/AmgX; a restart never silently switches
    discretization. A fresh run from a reference checkpoint selects defaults
    explicitly in its entry point.
    """
    from dataclasses import asdict
    from ..config import _decode
    if not isinstance(data, dict):
        raise ValueError('checkpoint AFSEI config must be an object')
    return _decode(AFSEIConfig, dict(data, ep_numerics=data.get(
        'ep_numerics', asdict(MonodomainNumerics.legacy()))))

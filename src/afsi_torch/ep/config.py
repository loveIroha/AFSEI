"""Serializable integration settings; full circulation remains a later stage."""
from dataclasses import dataclass, field
from math import isfinite
from ..paper_lv import PaperLVConfig
from .monodomain import MonodomainParameters
from .protocol import EPProtocol,ticks
from .tissue import GPBLand


@dataclass(frozen=True)
class AFSEIConfig:
    fsi: PaperLVConfig = field(default_factory=PaperLVConfig)
    ep: MonodomainParameters = field(default_factory=MonodomainParameters)
    protocol: EPProtocol = field(default_factory=EPProtocol)
    dt_ode_ms: float = .002
    start_time_s: float = .5
    max_retries: int = 4
    direction_mapping: str = 'source-serial-cg1'

    def __post_init__(self):
        if (not isinstance(self.fsi,PaperLVConfig) or not isinstance(self.ep,MonodomainParameters)
                or not isinstance(self.protocol,EPProtocol) or self.fsi.solid_degree != 5
                or type(self.max_retries) is not int or self.max_retries < 0
                or self.direction_mapping not in ('source-serial-cg1','afsi-dg0')
                or not isfinite(self.start_time_s) or self.start_time_s < 0):
            raise ValueError('invalid AFSEI integration settings')
        GPBLand(self.protocol.dt_ep_s,self.dt_ode_ms)
        start = ticks(self.start_time_s,self.protocol.dt_ep_s)
        if start > self.protocol.pulse_ticks[0]:
            raise ValueError('a fresh run cannot skip the first stimulus; use a checkpoint to resume')
        dt_ticks = ticks(self.fsi.time.dt,self.protocol.dt_ep_s)
        if dt_ticks < 1:
            raise ValueError('mechanical step must contain at least one EP tick')
        if dt_ticks & (dt_ticks-1):
            raise ValueError('mechanical maximum step must contain a power-of-two number of EP ticks')

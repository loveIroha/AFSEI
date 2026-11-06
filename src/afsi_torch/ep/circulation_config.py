"""Fresh-run circulation parameters, independent of NS/IB/EP discretization.

These values are already persisted by the cardiac checkpoint in its accepted
state and execution settings. Never apply a fresh profile when resuming.
"""
from dataclasses import dataclass, field, replace
from math import isfinite
from .windkessel import WindkesselParameters
from .gao_filling import GaoFillingParameters


@dataclass(frozen=True)
class CardiacCirculationConfig:
    windkessel: WindkesselParameters = field(default_factory=WindkesselParameters)
    initial_pc_mmhg: float = 85.
    fill_target_mmhg: float = 8.
    valve_resistance_mmhg_s_per_ml: float = 0.
    filling_pressure_location: str = 'inlet'
    filling_reservoir_mmhg: float = 0.
    filling_cap_reference: bool = False
    filling_model: str = "source"
    gao_filling: GaoFillingParameters = field(default_factory=GaoFillingParameters)

    def __post_init__(self):
        if not isinstance(self.windkessel, WindkesselParameters):
            raise ValueError('windkessel must be WindkesselParameters')
        values=(self.initial_pc_mmhg,self.fill_target_mmhg,self.valve_resistance_mmhg_s_per_ml,
                self.windkessel.rc_mmhg_s_per_ml,self.windkessel.rp_mmhg_s_per_ml,
                self.windkessel.compliance_ml_per_mmhg,self.windkessel.venous_mmhg,
                self.filling_reservoir_mmhg)
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not isfinite(v) for v in values):
            raise ValueError('finite numeric circulation parameters required')
        if self.initial_pc_mmhg < self.fill_target_mmhg or self.valve_resistance_mmhg_s_per_ml < 0:
            raise ValueError('initial arterial pressure must cover filling pressure; valve resistance >=0')
        if self.filling_model not in ('source','gao2017') or not isinstance(self.gao_filling,GaoFillingParameters):
            raise ValueError('valid filling model and Gao parameters required')
        if self.filling_model=='gao2017':
            if self.filling_cap_reference or self.filling_pressure_location!='inlet':
                raise ValueError('Gao uses independent inlet and wall loads, without cap reference')
            self.gao_filling.pressures(0.,self.fill_target_mmhg)
        if self.filling_model=='source' and abs(self.fill_target_mmhg-self.windkessel.venous_mmhg)>1e-12:
            raise ValueError('current open-loop protocol requires fill target = venous pressure')
        if self.filling_pressure_location not in ('inlet','endocardium'):
            raise ValueError('filling pressure location must be inlet or endocardium')
        if type(self.filling_cap_reference) is not bool or (self.filling_cap_reference
                and self.filling_pressure_location!='endocardium'):
            raise ValueError('cap reference requires endocardial filling pressure')

    def apply_initial_state(self, state):
        """Attach parameters only at the accepted preload endpoint, never mid-cycle."""
        if state.fsi.step != 0 or state.accepted_port_steps != 0:
            raise ValueError('circulation configuration is for fresh preloaded runs only')
        return replace(state,windkessel=self.windkessel,arterial_pc_mmhg=self.initial_pc_mmhg)


def circulation_metadata(driver,state):
    from dataclasses import asdict
    return dict(windkessel=asdict(state.windkessel),current_pc_mmhg=state.arterial_pc_mmhg,
        fill_target_mmhg=driver.fill_target_mmhg,
        valve_resistance_mmhg_s_per_ml=driver.valve_resistance_mmhg_s_per_ml,
        valve_law='ideal-one-way-series-resistance' if driver.one_way else 'signed-series-resistance',
        filling_model=driver.filling_model,
        gao_filling=asdict(driver.gao_filling),
        filling_law=('linear-inlet-and-triangular-extra-wall' if driver.filling_model=='gao2017'
                     else 'smoothstep-until-next-pacing'),
        inlet_law='ideal-inward-only' if driver.inlet_one_way else 'signed-pressure-reservoir',
        literature_doi='10.1016/j.medengphy.2017.06.042' if driver.filling_model=='gao2017' else None,
        anatomical_valves_present=False,
        filling_duration='accepted-unloading-end-to-next-pacing',
        cycle_period_ms=driver.period_ticks*driver.protocol.dt_ep_s*1000,
        filling_hold_ms=driver.filling_hold_ticks*driver.protocol.dt_ep_s*1000,
        pressure_waveform_end='next-pacing-minus-filling-hold',
        pressure_reference='effective-immersed-cap-traction-against-background',
        filling_pressure_location=driver.filling_pressure_location,
        filling_reservoir_mmhg=driver.filling_reservoir_mmhg,
        filling_cap_reference=driver.filling_cap_reference,
        cap_multiplier_reference=('relative-to-prescribed-wall-pressure'
            if driver.filling_cap_reference else 'unshifted'),
        pressure_location=('independent-inlet-and-extra-endocardial-traction; flow-driven-RCR'
            if driver.filling_model=='gao2017' else 'endocardial-follower-traction-during-filling; flow-driven-RCR-during-ejection'
            if driver.filling_pressure_location=='endocardium' else 'moving-fluid-port'),
        unloading_s=driver.protocol.relaxation_s)

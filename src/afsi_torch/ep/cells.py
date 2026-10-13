"""Common fixed-tissue adapter; existing GPB/Land implementation is reused."""
from dataclasses import replace
import torch
from .tissue import GPBLand,initial_state


class GPBCell:
    def __init__(self,dt_ep_s,dt_ode_ms):
        self.solver=GPBLand(dt_ep_s,dt_ode_ms)
        self.dt_ep_s,self.dt_ode_ms,self.substeps=self.solver.dt_ep_s,self.solver.dt_ode_ms,self.solver.substeps
        self._feedback=None

    def initial(self,like): return initial_state(like)
    def ionic(self,state): return state.gpb
    def calcium(self,state): return state.gpb[:,37]
    def voltage(self,state,value): return replace(state,voltage_v=value)

    def _fields(self,state):
        if self._feedback is None or self._feedback[0].shape!=state.voltage_v.shape:
            z=torch.zeros_like(state.voltage_v)
            self._feedback=(z+1,z)
        return self._feedback

    def advance_ode(self,state,time_s):
        stretch,zero=self._fields(state)
        return self.solver.advance_ode(state,stretch,zero,zero,time_s,advance_land=False)

    def step(self,ep,state,time_s,stimulus):
        stretch,zero=self._fields(state)
        return self.solver.step(ep,state,stretch,zero,zero,time_s,stimulus,advance_land=False)

    def provenance(self):
        return dict(model='AFSEI source GPB',source_revision='5bdde73c6be10cefadb005a0fcec5fdb6521f336',
                    celltype='epicardial',ionic_states_excluding_voltage=57)


def prepare_cell_model(config,like):
    if config.cell_model=='gpb':
        model=GPBCell(config.time.dt_ep_s,config.time.dt_ode_ms)
    else:
        from .ord import ORdCell
        model=ORdCell(config.time.dt_ep_s,config.time.dt_ode_ms,
                       celltype=1 if config.cell_model=='ord2011_epi' else 0)
    # A uniform isolated-cell conditioning protocol is applied equally to both.
    scalar=model.initial(like.new_zeros(1))
    native_voltage=scalar.voltage_v.item()*1000
    steps=round(config.initial.conditioning_s/config.time.dt_ep_s)
    for i in range(steps):
        ode,current=model.advance_ode(scalar,i*model.dt_ep_s)
        scalar=model.voltage(ode,scalar.voltage_v-model.dt_ep_s*current/config.parameters.capacitance_uf_per_cm2)
    state=model.initial(like)
    ionic=model.ionic(scalar).expand(len(like),-1).clone()
    if config.cell_model=='gpb': state=replace(state,gpb=ionic)
    else: state=replace(state,ionic=ionic)
    state=model.voltage(state,scalar.voltage_v.expand(len(like)).clone())
    model.initialization_report=dict(policy=config.initial.state,conditioning_s=config.initial.conditioning_s,
        conditioning_ep_steps=steps,conditioning_ode_substeps=steps*model.substeps,
        native_voltage_mv=native_voltage,conditioned_voltage_mv=scalar.voltage_v.item()*1000,
        conditioned_calcium_um=model.calcium(scalar).item()*1000)
    return model,state

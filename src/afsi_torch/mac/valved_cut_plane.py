"""Source finite-resistance valve + flow-driven RCR at physical MAC cuts.

The upstream pressure includes valve drop; downstream arterial pressure does
not. response() is the Jacobian at the last advance(), not a globally affine
fluid response. No internal basal boundary geometry is created here.
"""
import torch
from .cut_plane import CutPlaneFlow
from ..ep.aortic_valve import AorticValveParameters, solve_fluid_valve


class ValvedCutPlaneFlow(CutPlaneFlow):
    def __init__(self,base,planes,*,valve=None):
        super().__init__(base,planes)
        self.valve=valve or AorticValveParameters()
        self.valve_trial=None

    def prescribe(self,pressures_mmhg):
        super().prescribe(pressures_mmhg)
        self.valve_trial=None

    def _boundary_values(self,velocity,*,derivative):
        if self.mode!='rcr': return super()._boundary_values(velocity,derivative=derivative)
        values=torch.zeros_like(self.pressures) if derivative else self.pressures.clone()
        j=self.outlet;values[j]=0.
        q0=self.flux(velocity)[j]+self.conductance[j]@values
        if derivative:
            if self.valve_trial is None:raise RuntimeError('advance before evaluating valve tangent')
            z=self.valve_trial['tangent_impedance']
            values[j]=z*q0/(1-z*self.conductance[j,j])
        else:
            self.valve_trial=None
            self.valve_trial=solve_fluid_valve(float(q0),float(self.conductance[j,j]),
                float(self.old_pc),self.dt,self.parameters,self.valve)
            values[j]=self.valve_trial['upstream_mmhg']
        return values

    @torch.no_grad()
    def advance(self,departure,density,pressure_initial=None):
        u,p,info=super().advance(departure,density,pressure_initial)
        if self.mode=='rcr':
            b=self.last_boundary;v=self.valve_trial;q=b['outward_flux_ml_s'][self.outlet]
            dp=b['pressures_mmhg'][self.outlet]-self.last_circuit.outlet_mmhg
            expected,_=self.valve.flow_and_derivative(float(dp))
            b['valve_flow_residual_ml_s']=q-expected
            b['valve_drop_mmhg']=dp
            b['arterial_pressure_mmhg']=self.last_circuit.outlet_mmhg
            b['rcr_gap_mmhg']=b['pressures_mmhg'][self.outlet]-v['valve_drop_mmhg']-self.last_circuit.outlet_mmhg
            b['valve_dissipation_erg_s']=dp*q*1333.22368421
            b['pressure_application']='upstream of unresolved source valve; RCR pressure is downstream'
        return u,p,info

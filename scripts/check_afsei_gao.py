"""Independent short passive port probes on the real LV/MAC production mesh.

Each probe starts from the same validated preload. They are NOT a trajectory,
do not advance EP/ODE, and cannot establish a full cardiac-cycle PV result.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_cycle import CardiacStepper
from afsi_torch.ep.circulation_config import CardiacCirculationConfig
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from afsi_torch.config import load_config
from afsi_torch.paper_lv import PaperLVConfig
from afsi_torch.circulation.geometry import PortGeometry


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-checkpoint',type=Path,required=True)
    p.add_argument('--preload',type=Path,required=True)
    p.add_argument('--fsi-config',type=Path,required=True)
    p.add_argument('--circulation-config',type=Path,required=True)
    p.add_argument('--spatial-config',type=Path,required=True)
    p.add_argument('--port-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fraction',type=float,action='append',help='selected independent probes only')
    p.add_argument('--supply-control-mmhg',type=float,help='independent open-inlet positive control, wall load zero')
    a=p.parse_args()
    old,_,_=checkpoint.load(a.reference_checkpoint)
    cfg=replace(old.config,prescribed_endocardial_pressure_mmhg=0.,
        fsi=load_config(a.fsi_config,PaperLVConfig),ep_refinement_factor=1)
    spatial=json.loads(a.spatial_config.read_text());ib=spatial.pop('ib_response_backend')
    cfg=replace(cfg,fsi=replace(cfg.fsi,ib_response_backend=ib))
    c=load_config(a.circulation_config,CardiacCirculationConfig)
    d=CardiacStepper(old.mesh,cfg,filling_model=c.filling_model,gao_filling=c.gao_filling,
        fill_target_mmhg=c.fill_target_mmhg,**spatial)
    d.port_geometry=load_config(a.port_config,PortGeometry)
    del old
    state=d.initialize_preload(a.preload,windkessel=c.windkessel,arterial_pc_mmhg=c.initial_pc_mmhg)
    e=d._fsi_stepper(d.maximum_ticks)
    reports=[]
    cases=[(fraction,*d.gao_filling.pressures(fraction,c.fill_target_mmhg),'profile')
           for fraction in (a.fraction if a.fraction is not None else (0.,.5,1.))]
    if a.supply_control_mmhg is not None:
        cases.append((None,a.supply_control_mmhg,0.,'positive-control'))
    for fraction,supply,wall,label in cases:
        control=replace(state,mode='filling',wall_pressure_mmhg=wall,
            cap_reference_mmhg=0.,filling_pressure_mmhg=supply)
        d._boundary_context=control
        try:
            trial,info=e.physical_step(state.fsi,state.fsi.time+e.flow.dt,True)
        finally:
            d._boundary_context=None
        r=info['balanced']
        assert abs(r['mass_error_ml_s'])<=1e-5
        assert r['inlet_flow_ml_s']>=-1e-5 and not r['outlet_open']
        assert r['boundary_work_relative_error']<=1e-7
        assert r['transfer_power_relative_error']<=1e-7
        assert abs(r['inlet_reservoir_power_erg_s']-supply*C*r['inlet_flow_ml_s'])<1e-6
        reports.append(dict(label=label,fraction=fraction,**r,nonlinear=info['nonlinear'],
            max_displacement_cm=(trial.x-state.fsi.x).abs().max().item()))
        print(json.dumps(dict(fraction=fraction,supply_mmhg=supply,wall_mmhg=wall,
            q_in_ml_s=r['inlet_flow_ml_s'],inlet_open=r['inlet_open'],
            port_mmhg=r['boundary_pressure_mmhg'],core_mmhg=r['pressure_core']['mean_mmhg'],
            mass_error_ml_s=r['mass_error_ml_s']),allow_nan=False),flush=True)
    # A sudden wall-load perturbation of an end-diastolic preload need not
    # open the inlet in its first 0.1 ms. Test the true diode law, and use an
    # independently raised supply pressure as the positive-flow control.
    passed=all((r['inlet_open'] and r['inlet_flow_ml_s']>0) if r['label']=='positive-control'
               else not r['inlet_open'] if r['fraction']==0. else True for r in reports)
    report=dict(passed=passed,scope='independent passive real LV port probes',
        structural_nodes=len(d.mesh.X),fluid_grid=list(e.grid.shape),
        spatial_coupling=d.spatial_coupling,ep_advanced=False,ode_advanced=False,
        full_cycle_validated=False,probes=reports)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not passed:raise SystemExit('port probe expectations failed; diagnostic JSON retained')


if __name__=='__main__':main()

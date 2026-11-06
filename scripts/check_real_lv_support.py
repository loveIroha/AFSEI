"""One reference-mesh BE/IB step with physical basal support; no cycle/ports."""
from pathlib import Path
from dataclasses import replace
from time import perf_counter
import json,traceback
import torch
from afsi_torch.config import _decode,TimeConfig
from afsi_torch.paper_lv import PaperLVConfig,imported_model
from afsi_torch.ep.buffered_load import BufferedLandLoad,ActiveLoadState
from afsi_torch.ep.solid import EPDrivenPaperSolid
from afsi_torch.mac.paper_coupling import BEIBStepper
from afsi_torch.mac.basal_support import enable_basal_support
from afsi_torch.nonlinear import NewtonOptions,GMRESOptions


def main():
    root=Path('results/real_lv_support_20261103');root.mkdir(exist_ok=True)
    source=json.loads(Path('results/afsei_source_timing_neck_two_cycles_20261029/configuration.json').read_text())['config']['fsi']
    cfg=_decode(PaperLVConfig,source)
    cfg=replace(cfg,beta=0.,time=TimeConfig(1e-4,1e-4),nonlinear_solver='newton',
        newton_preconditioner='none',ib_response_backend='quadrature',ib_csr_contraction_backend='sites',
        nonlinear=NewtonOptions(atol=1e-10,rtol=1e-8,max_iterations=5,linear_tolerance_fraction=.25,
            linear=GMRESOptions(atol=1e-12,rtol=1e-4,restart=50,max_iterations=150,check_policy='estimated')))
    started=perf_counter();report=dict(scope='one step, original LV + original-size MAC; no connected tracts or EP',
        beta=0.,pressure_mmhg=.1,dt_s=1e-4,fluid_shape=cfg.fluid.shape)
    try:
        passive=imported_model(cfg,device='cuda:0');mesh=passive.mesh
        load=BufferedLandLoad(passive.geometry,mesh.fiber);zero=mesh.X.new_zeros(len(mesh.X))
        load.stage(ActiveLoadState(mesh.X,zero,zero,zero,zero,cfg.time.dt,False))
        solid=EPDrivenPaperSolid(mesh,cfg,load,source_quadrature=True,prescribed_pressure_mmhg=.1)
        driver=enable_basal_support(BEIBStepper(solid,cfg,'cuda:0'))
        state=driver.initialize(mesh.X)
        print('Prepared original LV physical support; solving one BE/IB step',flush=True)
        prepared=perf_counter()
        new,info=driver.step(state)
        report.update(status='passed',node_count=len(mesh.X),cell_count=len(mesh.cells),time_s=new.time,
            max_displacement_cm=float((new.x-mesh.X).abs().max()),info=info,
            preparation_s=prepared-started,solve_s=perf_counter()-prepared)
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    report['elapsed_s']=perf_counter()-started
    (root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()

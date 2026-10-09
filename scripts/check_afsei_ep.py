"""GPU component check on the actual LV mesh; not a coupled heart simulation."""
import argparse
import json
from pathlib import Path
import time
import torch
from afsi_torch.paper_lv import PaperLVConfig, imported_model
from afsi_torch.ep import Monodomain, GPBLand, initial_state, LandQuadratureLoad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir',default='/mnt/large2/gjh/realistic_left_ventricle')
    parser.add_argument('--steps',type=int,default=40,help='EP steps only; 40 steps = 2 ms')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 400:
        parser.error('component check needs 1..400 EP steps')
    if not torch.cuda.is_available():
        raise RuntimeError('this check requires the actual CUDA device')
    t0 = time.perf_counter()
    solid = imported_model(PaperLVConfig(source_dir=args.source_dir),device='cuda')
    mesh = solid.mesh
    # This smoke check explicitly uses AFSI's DG0 directions. Reproducing the
    # source EP's CG1 interpolation requires a verified nodal direction mapping.
    ep = Monodomain(mesh.X,mesh.cells,mesh.fiber,mesh.sheet,directions='cell')
    ode = GPBLand()
    state = initial_state(mesh.X[:,0])
    stretch = state.voltage_v.new_ones(len(mesh.X))
    zero = torch.zeros_like(stretch)
    stimulus = torch.zeros_like(stretch)
    endo_nodes = torch.unique(solid.endo_faces)
    stimulus[endo_nodes] = 25.0
    max_residual_ratio = 0.0
    for step in range(args.steps):
        state,current,info = ode.step(ep,state,stretch,zero,zero,.5+step*ode.dt_ep_s,stimulus)
        max_residual_ratio = max(max_residual_ratio,info.residual_norm/info.tolerance)
    # Actual GPB calcium drives Land, which then supplies the spatial load.
    load = LandQuadratureLoad.prepare(mesh.X,mesh.cells,mesh.fiber,mesh.X,
        state.land[:,0],state.land[:,1],zero,zero,args.steps*ode.dt_ep_s)
    ta = load.tension(mesh.X)
    force = load.force(mesh.X)
    direction = (mesh.X-mesh.X.mean(0))*1e-3
    tangent_action = load.force_jvp(mesh.X,direction)
    for value in (state.voltage_v,state.gpb,state.land,current,ta,force,tangent_action):
        if not torch.isfinite(value).all(): raise FloatingPointError('nonfinite LV component result')
    torch.cuda.synchronize()
    report = dict(passed=True,scope='EP/GPB/Land components on the real LV reference mesh',
        gpu=torch.cuda.get_device_name(),node_count=len(mesh.X),cell_count=len(mesh.cells),
        ep_steps=args.steps,ep_step_ms=ode.dt_ep_s*1000,ode_step_ms=ode.dt_ode_ms,
        substeps_per_ep=ode.substeps,stimulated_endo_nodes=len(endo_nodes),
        ep_directions='AFSI DG0; source CG1 mapping pending',
        voltage_min_mv=state.voltage_v.min().item()*1000,voltage_max_mv=state.voltage_v.max().item()*1000,
        calcium_min_um=state.gpb[:,37].min().item()*1000,calcium_max_um=state.gpb[:,37].max().item()*1000,
        tension_min_cgs=ta.min().item(),tension_max_cgs=ta.max().item(),
        active_force_norm=force.norm().item(),active_tangent_action_norm=tangent_action.norm().item(),
        max_pcg_residual_to_tolerance=max_residual_ratio,elapsed_seconds=time.perf_counter()-t0,
        solid_motion_advanced=False,fluid_advanced=False,windkessel_ports_connected=False,
        three_field_integration_implemented=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()

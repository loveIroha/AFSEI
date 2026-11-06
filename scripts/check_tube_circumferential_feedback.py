"""GPU manufactured-deformation audit of the production tube/Land feedback.

This is NOT a free FSI contraction experiment or a proof of buckling stability.
Uses the actual configured mesh, L2 projection, CG/CN operator and Euler kernel.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import torch
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.ep.circular_tube import CircularTubeSolid
from afsi_torch.ep.monodomain import Monodomain
from afsi_torch.ep.tissue import initial_state, GPBLand
from afsi_torch.ep.planar_tube import land_resting_equilibrium
from afsi_torch.ep.land import land_tension, land_ode_step, distortion_step


@torch.no_grad()
def check(config, device='cuda'):
    assert torch.cuda.is_available(), 'Actual GPU validation required'
    geometry, fsi, parameters, numerics = resolve_config(config)
    assert config['active']['stretch_feedback'] == 'circumferential'
    assert config['ep']['ode_method'] == 'explicit_euler'
    assert config.get('solid_refinement_factor', 1) == 1
    assert 'layers' not in config['solid']
    mesh = generate_circular_tube(geometry, device=device)
    model = CircularTubeSolid(mesh, geometry,
        **{k:v for k,v in config['solid'].items() if k != 'end_condition'},
        **{k:v for k,v in config['active'].items() if k not in ('mechanism','rest_reference')},
        end_support=config['solid']['end_condition'], dt_s=fsi.time.dt)
    radial = mesh.X.clone(); radial[:, :2] -= mesh.X.new_tensor(geometry.center_xy_cm)
    radial[:, 2] = 0
    radial /= torch.linalg.vector_norm(radial, dim=1, keepdim=True)
    axial = mesh.X.new_tensor([0.,0.,1.]).expand_as(mesh.X)
    ep = Monodomain(mesh.X, mesh.cells, axial, radial, parameters=parameters, numerics=numerics)
    ep.prepare(config['time']['ep_dt_s'])
    state = initial_state(mesh.X[:, 0])
    one = torch.ones_like(state.voltage_v); zero = torch.zeros_like(one)
    rest = land_resting_equilibrium(state.gpb[:,37]*1000, one)
    model.set_reference(land_tension(one, one, rest[:,0], rest[:,1], zero, zero, fsi.time.dt))
    # Finite manufactured active state, not an equilibrated physiological state.
    gpb = state.gpb.clone(); gpb[:,37] = .001
    land = state.land.clone(); land[:] = land.new_tensor([.03,.05,.3,.4])
    state = replace(state, gpb=gpb, land=land)
    center = mesh.X.new_tensor([*geometry.center_xy_cm, geometry.axial_start_cm])
    angle = .37
    rotation = mesh.X.new_tensor([[1.,0.,0.],[0.,math.cos(angle),-math.sin(angle)],
                                 [0.,math.sin(angle),math.cos(angle)]])
    rows = []
    for ratio in (.85,.95,1.,1.05):
        x = (mesh.X-center)*mesh.X.new_tensor([ratio,ratio,1.13])+center
        cell = model.cell_stretch(x)
        nodal = model.activation_stretch(model.nodal_stretch(ep,x))
        torch.testing.assert_close(cell, torch.full_like(cell,ratio),rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(nodal, torch.full_like(nodal,ratio),rtol=2e-8,atol=2e-8)
        moved = x@rotation.T+mesh.X.new_tensor([.17,-.21,.08])
        torch.testing.assert_close(model.cell_stretch(moved),cell,rtol=1e-10,atol=1e-10)
        # Hold each deformation: zero shortening rate isolates length dependence.
        model.stage(x,state,zero,zero)
        ta = model.applied_tension(x)
        expected = (land_tension(nodal,nodal,land[:,0],land[:,1],zero,zero,fsi.time.dt)
                    -land_tension(one,one,rest[:,0],rest[:,1],zero,zero,fsi.time.dt)).clamp_min(0)
        torch.testing.assert_close(ta,model.scale*model.taper*expected.mean(),rtol=1e-7,atol=1e-5)
        rows.append(dict(prescribed_hoop_stretch=ratio, projected_min=float(nodal.min()),
                         projected_max=float(nodal.max()), maximum_tension_kpa=float(ta.max())/1e4))
    assert rows[0]['maximum_tension_kpa'] < rows[2]['maximum_tension_kpa'] < rows[3]['maximum_tension_kpa']
    # Rigid motion/axial elongation cannot masquerade as hoop shortening.
    assert abs(rows[2]['projected_min']-1.) < 2e-8
    n = len(one)
    stretch = torch.linspace(.85,1.1,n,device=one.device,dtype=one.dtype)
    zs = torch.linspace(-.2,.1,n,device=one.device,dtype=one.dtype)
    zw = torch.linspace(-.15,.05,n,device=one.device,dtype=one.dtype)
    # Independent readable Land RHS checks the CUDA kernel uses EACH node's
    # stretch and distortion, with its updated GPB calcium (one Euler substep).
    single = GPBLand(1e-6,.001,method='explicit_euler')
    updated,current = single.advance_ode(state,stretch,zs,zw,0.)
    expected = land_ode_step(state.land,updated.gpb[:,37]*1000,stretch,zs,zw,.001)
    torch.testing.assert_close(updated.land,expected,rtol=2e-13,atol=2e-14)
    ode = GPBLand(config['time']['ep_dt_s'],config['time']['ode_dt_ms'],method='explicit_euler')
    iso,_,_ = ode.step(ep,state,one,zero,zero,0.,zero)
    actual,_,_ = ode.step(ep,state,stretch,zs,zw,0.,zero)
    torch.testing.assert_close(actual.gpb,iso.gpb,rtol=0,atol=0)
    torch.testing.assert_close(actual.voltage_v,iso.voltage_v,rtol=0,atol=0)
    difference = float((actual.land-iso.land).abs().max())
    assert difference > 1e-6
    # Trial force/JVP evaluation must not publish strain history or Land states.
    model.stage(mesh.X,state,zs,zw)
    buffers = [model.previous_F.clone(),model.land_fields.clone()]
    compressed = (mesh.X-center)*mesh.X.new_tensor([.95,.95,1.])+center
    model.applied_tension(compressed); model.applied_tension(compressed)
    torch.testing.assert_close(model.previous_F,buffers[0],rtol=0,atol=0)
    torch.testing.assert_close(model.land_fields,buffers[1],rtol=0,atol=0)
    # Exact constant-rate distortion step: dt is seconds, decay rates are /s.
    dt=fsi.time.dt; dl=-.001
    z = distortion_step(zero,torch.full_like(zero,dl),dt,40.14)
    expected_z=10*dl*(1-math.exp(-40.14*dt))/(40.14*dt)
    torch.testing.assert_close(z,torch.full_like(z,expected_z),rtol=1e-12,atol=1e-14)
    report=dict(passed=True,scope='manufactured geometry and actual GPU EP/ODE feedback; no FSI trajectory',
        gpu=torch.cuda.get_device_name(),nodes=len(mesh.X),cells=len(mesh.cells),
        feedback=model.stretch_feedback,projection='consistent L2 DG0 scalar stretch to CG1',
        affine_checks=rows,cuda_land_reference_max_error=float((updated.land-expected).abs().max()),
        production_step_land_difference_vs_isometric=difference,
        gpb_and_voltage_unchanged_by_stretch=True,history_not_mutated=True,
        distortion_for_delta_lambda_minus_0p001=float(z[0]),full_fsi_buckling_verified=False)
    ep.close()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=check(json.loads(Path(args.config).read_text()))
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

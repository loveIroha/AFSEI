"""Geometry, physical-unit, stimulus timing and activation-observation contracts."""
import json
import runpy
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import pytest
import torch
from afsi_torch.config import load_config
from afsi_torch.geometry.tissue_slab import structured_tissue_slab
from afsi_torch.ep import Monodomain,MonodomainNumerics,MonodomainParameters
from afsi_torch.ep.experiment import EPTissueConfig,TissueDomainConfig,TissueStimulus,EPExperimentTime
from afsi_torch.simulation.ep_tissue import first_activation,pulse_active,stimulus_mask,stimulus_field


def test_conforming_six_tetra_mesh_positive_volume_and_boundary():
    mesh=structured_tissue_slab((.2,.1,.1),.1,origin_cm=(.3,-.1,1.))
    assert mesh.shape==(2,1,1) and len(mesh.cells)==12 and len(mesh.X)==12
    v=mesh.X[mesh.cells]
    volumes=torch.linalg.det((v[:,1:]-v[:,:1]).transpose(1,2))/6
    assert (volumes>0).all()
    torch.testing.assert_close(volumes.sum(),mesh.X.new_tensor(.002))
    faces=torch.cat([mesh.cells[:,ids] for ids in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))]).sort(1).values
    unique,count=torch.unique(faces,dim=0,return_counts=True)
    assert count.max()==2
    boundary=mesh.X[unique[count==1]]
    lo=mesh.X.min(0).values; hi=mesh.X.max(0).values
    assert (((boundary-lo).abs()<1e-12).all(1)|((boundary-hi).abs()<1e-12).all(1)).any(1).all()


def test_physical_conductivity_tensor_and_insulating_nodal_integrals():
    mesh=structured_tissue_slab((.1,.1,.1),.1)
    f=mesh.X.new_tensor([1.,0.,0.]).expand(len(mesh.X),-1)
    s=mesh.X.new_tensor([0.,1.,0.]).expand(len(mesh.X),-1)
    p=MonodomainParameters(1.,.1,(1.17,.39,.39))
    ep=Monodomain(mesh.X,mesh.cells,f,s,parameters=p,
                  numerics=MonodomainNumerics(backend='torch_pcg_jacobi'))
    torch.testing.assert_close(ep.mass_diagonal.sum(),mesh.X.new_tensor(.001))
    for a,g in enumerate(p.conductivity_us_per_cm):
        field=mesh.X[:,a]
        energy=field@torch.sparse.mm(ep.stiffness,field[:,None]).flatten()
        torch.testing.assert_close(energy,mesh.X.new_tensor(g*.001))
    constant=mesh.X.new_ones(len(mesh.X))
    torch.testing.assert_close(torch.sparse.mm(ep.stiffness,constant[:,None]),
                              torch.zeros_like(constant[:,None]),atol=1e-14,rtol=0)


def test_finite_stimulus_region_and_half_open_time_interval():
    stim=TissueStimulus(minimum_cm=(.1,.1,.1),maximum_cm=(.15,.15,.15),start_s=.001,duration_s=.002)
    X=torch.tensor([[.1,.1,.1],[.15,.15,.15],[.151,.15,.15]],dtype=torch.float64)
    assert stimulus_mask(X,stim).tolist()==[True,True,False]
    precise=replace(stim,amplitude_ua_per_cm2=50000/1400)
    values=stimulus_field(X,precise)
    assert values.dtype==torch.float64 and values[0].item()==precise.amplitude_ua_per_cm2
    dt=5e-5
    active=[i for i in range(100) if pulse_active(i,dt,stim)]
    assert active==list(range(20,60)) and len(active)*dt==pytest.approx(.002)


def test_first_activation_interpolates_and_never_overwrites():
    old=torch.tensor([-.08,-.01,.01,-.05],dtype=torch.float64)
    new=torch.tensor([-.07,.03,-.01,.01],dtype=torch.float64)
    lat=torch.tensor([float('nan'),float('nan'),float('nan'),.001],dtype=torch.float64)
    result=first_activation(old,new,lat,.002,.00005,0.)
    assert torch.isnan(result[0]) and torch.isnan(result[2])
    assert result[1]==pytest.approx(.0020125) and result[3]==.001


def test_declared_json_rejects_unknown_keys_and_inconsistent_units(tmp_path):
    path=tmp_path/'case.json'
    path.write_text(json.dumps(dict(demo='ep-slab',**asdict(EPTissueConfig()))))
    assert load_config(path,EPTissueConfig)==EPTissueConfig()
    path.write_text(json.dumps({'demo':'ep-slab','stimulus':{'amplitude_ua_per_cm3':50000}}))
    with pytest.raises(ValueError,match='unknown'):
        load_config(path,EPTissueConfig)
    with pytest.raises(ValueError,match='integer multiples'):
        TissueDomainConfig(spacing_cm=.03)
    with pytest.raises(ValueError,match='time grid'):
        replace(EPTissueConfig(),stimulus=TissueStimulus(duration_s=.00201))
    with pytest.raises(ValueError,match='inside'):
        replace(EPTissueConfig(),stimulus=TissueStimulus(maximum_cm=(2.,.1,.1)))
    with pytest.raises(ValueError,match='integer number'):
        EPExperimentTime(dt_ode_ms=.003)


def test_actual_cuda_demo_propagates_after_pulse_without_moving_geometry(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('actual GPB CUDA and AmgX required')
    from afsi_torch.simulation.ep_tissue import run
    from afsi_torch.ep.experiment import TissueOutput,TissueObservations
    config=EPTissueConfig(domain=TissueDomainConfig(lengths_cm=(.3,.2,.2),spacing_cm=.05),
        parameters=MonodomainParameters(1.,.1,(1.17,.39,.39)),
        stimulus=TissueStimulus(amplitude_ua_per_cm2=50000/1400),
        time=EPExperimentTime(end_time_s=.012),
        observations=TissueObservations(probes_cm=((0.,0.,0.),(.3,.2,.2))),
        output=TissueOutput(log_every=240,trace_every=10,write_vtk=False,write_plots=False))
    report=run(config,tmp_path/'result')
    assert report['status']=='completed' and report['linear_solver_max_residual_to_tolerance']<=1
    assert report['amg_setup_count']==1
    assert report['activated_nodes']>report['stimulated_nodes']
    assert report['fixed_geometry'] and not report['solid_motion_advanced'] and not report['fluid_advanced']
    assert not report['land_advanced']
    data=np.load(tmp_path/'result'/'final_state.npz',allow_pickle=False)
    assert np.isfinite(data['voltage_v']).all() and (data['gpb'][:,30:38]>0).all()
    initial=np.load(tmp_path/'result'/'mesh.npz',allow_pickle=False)
    np.testing.assert_allclose(initial['coordinates_cm'].max(0),config.domain.lengths_cm)


def test_case_defaults_match_declared_json():
    root=Path(__file__).resolve().parents[1]
    settings=runpy.run_path(str(root/'demo/ep_slab/run.py'),run_name='test_case')['CONFIG']
    assert load_config(root/'demo/ep_slab/configs/santiago_gpb.json',EPTissueConfig)==settings


def test_comparison_refuses_mixed_physics_and_reuses_existing_curves(tmp_path):
    from afsi_torch.simulation.ep_tissue_analysis import compare_runs
    folders=[]
    for i,h in enumerate((.05,.025)):
        p=tmp_path/f'case{i}'; p.mkdir(); folders.append(p)
        config=replace(EPTissueConfig(),domain=TissueDomainConfig(spacing_cm=h))
        (p/'config_resolved.json').write_text(json.dumps(asdict(config)))
        r=dict(status='completed',node_count=10,cell_count=20,activated_fraction=1.,last_activation_ms=4.,
               integration_seconds=1.,stimulus_nodal_quadrature_volume_cm3=.003)
        (p/'report.json').write_text(json.dumps(r))
        np.savetxt(p/'activation_line.csv',np.array([[0,0,0,1],[1,0,0,4+i*.1]]),delimiter=',',
                   header='x_cm,y_cm,z_cm,activation_ms',comments='')
    report=compare_runs(folders,tmp_path/'comparison',write_plot=False)
    assert report['finest_spacing_mm']==.25
    assert report['cases'][0]['max_line_difference_from_finest_ms']==pytest.approx(.1)
    cfg=json.loads((folders[1]/'config_resolved.json').read_text()); cfg['parameters']['surface_to_volume_per_cm']=1.
    (folders[1]/'config_resolved.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='same equations'):
        compare_runs(folders,tmp_path/'bad_comparison',write_plot=False)

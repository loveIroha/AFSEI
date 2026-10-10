"""MAC preset fidelity and profiling must preserve the coupled physical step."""
from dataclasses import replace
from pathlib import Path
import runpy
import pytest
import torch
from afsi_torch.config import load_config
from afsi_torch.paper_lv import PaperLVConfig,imported_model
from afsi_torch.ep.driver import AFSEIStepper
from afsi_torch.ep.config import AFSEIConfig
from test_real_lv import real_case
from test_paper_lv import config_for
from test_afsei_driver import assert_same

ROOT = Path(__file__).resolve().parents[1]
CLI = runpy.run_path(str(ROOT/'scripts/run_afsei.py'))


def test_mac_preset_keeps_upstream_solver_settings_and_replaces_active_waveform():
    upstream = load_config(ROOT/'demo/real_lv_fsi/configs/active_cuda.json',PaperLVConfig)
    config = CLI['fresh_config'](CLI['parse_args']([]))
    fsi = config.fsi
    for name in ('fluid','flow','pressure_solver','mass_solver','execution',
                 'nonlinear','anderson','interaction_quadrature','nonlinear_solver',
                 'ib_response_backend','ib_csr_assembly_backend','ib_csr_contraction_backend'):
        assert getattr(fsi,name) == getattr(upstream,name)
    assert fsi.fluid.shape == (128,128,128)
    assert config.mechanics_profile == 'source'
    assert fsi.applied_loads.at(.5) == (0.,0.)
    coarse = CLI['fresh_config'](CLI['parse_args'](['--fluid-shape','32','--mechanical-dt-ms','0.05']))
    assert coarse.fsi.fluid.shape == (32,32,32)
    assert coarse.fsi.time.dt == .00005
    with pytest.raises(SystemExit): CLI['parse_args'](['--resume','old.npz','--fluid-shape','32'])


def test_profile_preserves_source_ep_and_original_mac_step(real_case):
    if not torch.cuda.is_available(): pytest.skip('actual CUDA required')
    fsi = replace(config_for(real_case,solver='anderson-newton'),
        execution=PaperLVConfig().execution,
        interaction_quadrature=PaperLVConfig().interaction_quadrature,ib_response_backend='csr',
        ib_csr_assembly_backend='cached-hash',ib_csr_contraction_backend='cuda')
    config = AFSEIConfig.source(fsi=fsi)
    mesh = imported_model(config.fsi,'cuda').mesh
    driver = AFSEIStepper(mesh,config)
    old = driver.initialize()
    direct,plain = driver.step(old)
    driver._reset_scratch(old)
    driver.profile = True
    profiled,info = driver.step(old)
    assert_same(direct,profiled)
    assert 'timings' not in plain
    assert set(info['timings']) == {'gpb_land_seconds','monodomain_seconds',
        'load_staging_seconds','implicit_fsi_seconds','strain_feedback_seconds'}
    assert all(v>0 for v in info['timings'].values())
    assert info['nonlinear']['residual_norm'] <= info['nonlinear']['tolerance']
    assert info['ep']['steps'] == 2

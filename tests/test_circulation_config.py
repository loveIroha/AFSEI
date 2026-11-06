from dataclasses import dataclass, replace
from types import SimpleNamespace
import pytest
from afsi_torch.config import load_config
from afsi_torch.ep.circulation_config import CardiacCirculationConfig
from afsi_torch.ep.windkessel import WindkesselParameters, flow_driven_trial


def test_effective_source_profile_and_default(tmp_path):
    path=tmp_path/'circulation.json'
    path.write_text('{"windkessel":{"rc_mmhg_s_per_ml":0.0345,"rp_mmhg_s_per_ml":0.7245}}')
    cfg=load_config(path,CardiacCirculationConfig)
    assert cfg.windkessel.compliance_ml_per_mmhg == 1.3
    assert CardiacCirculationConfig().windkessel == WindkesselParameters()
    old=85.
    for q in (200.,0.,0.,90.):
        dt=1e-4; trial=flow_driven_trial(old,q,dt,cfg.windkessel)
        assert 1.3*(trial.pc_mmhg-old)/dt == pytest.approx(q-(trial.pc_mmhg-8)/.7245,abs=3e-10)
        old=trial.pc_mmhg


def test_reject_unknown_config_and_midcycle_mutation(tmp_path):
    path=tmp_path/'bad.json'; path.write_text('{"windkessel":{"Rc":0.0345}}')
    with pytest.raises(ValueError,match='unknown'): load_config(path,CardiacCirculationConfig)
    @dataclass(frozen=True)
    class State:
        fsi: object
        accepted_port_steps: int=0
        windkessel: object=None
        arterial_pc_mmhg: float=0.
    original=State(SimpleNamespace(step=0))
    cfg=CardiacCirculationConfig(windkessel=WindkesselParameters(.0345,.7245))
    accepted=cfg.apply_initial_state(original)
    assert accepted.windkessel==cfg.windkessel and accepted.arterial_pc_mmhg==85.
    assert original.arterial_pc_mmhg==0.
    with pytest.raises(ValueError,match='fresh'):
        cfg.apply_initial_state(replace(accepted,fsi=SimpleNamespace(step=1)))


@pytest.mark.parametrize('values',[
    {'initial_pc_mmhg':float('nan')},{'initial_pc_mmhg':True},
    {'valve_resistance_mmhg_s_per_ml':-1.},{'initial_pc_mmhg':7.},
    {'fill_target_mmhg':9.},
])
def test_invalid_circulation(values):
    with pytest.raises(ValueError): CardiacCirculationConfig(**values)

import pytest
from afsi_torch.mac.multigrid import MGOptions
from afsi_torch.mac.sealed_pressure import tightened_pressure_options


def test_default_options_are_unchanged_and_override_only_tightens():
    old=MGOptions()
    assert tightened_pressure_options(old,None) is old
    new=tightened_pressure_options(old,dict(rtol=1e-12,atol=1e-14))
    assert old.rtol==1e-10 and old.atol==1e-12
    assert new.rtol==1e-12 and new.atol==1e-14 and new.smooth==old.smooth


@pytest.mark.parametrize('spec',[{},dict(rtol=1e-8,atol=1e-14),dict(rtol=True,atol=1e-14),
    dict(rtol=float('nan'),atol=1e-14),dict(rtol=0.,atol=0.)])
def test_invalid_or_looser_options_are_rejected(spec):
    with pytest.raises(ValueError):tightened_pressure_options(MGOptions(),spec)

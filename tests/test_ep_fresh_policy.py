from dataclasses import asdict
from afsi_torch.ep.config import AFSEIConfig, decode_checkpoint_config


def test_fresh_ep_policy_and_old_checkpoint_defaults():
    fresh = AFSEIConfig()
    assert fresh.direction_mapping == 'l2-cg1'
    assert fresh.ep_numerics.backend == 'torch_pcg_jacobi_light'
    assert fresh.ep_numerics.time_scheme == 'crank_nicolson'
    assert decode_checkpoint_config(asdict(fresh)) == fresh
    old = asdict(fresh)
    del old['direction_mapping']
    del old['ep_numerics']['backend']
    restored = decode_checkpoint_config(old)
    assert restored.direction_mapping == 'source-serial-cg1'
    assert restored.ep_numerics.backend == 'amgx_pcg_amg'
    del old['ep_numerics']
    restored = decode_checkpoint_config(old)
    assert restored.ep_numerics.backend == 'torch_pcg_jacobi'
    assert restored.ep_numerics.discretization == 'consistent_p1'

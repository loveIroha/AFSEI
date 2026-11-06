"""A speedup must not be accepted across different problems or failed trajectories."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tube_comparison',ROOT/'demo/active_tube_2d/benchmark.py')
comparison=importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def fixture_run(path,profile):
    path.mkdir()
    config=json.loads((ROOT/'demo/active_tube_2d/configs/tube_gpu.json').read_text())
    config['execution']={'profile':profile}
    config['solver'].update(pressure_backend='reference' if profile=='reference' else 'graph')
    (path/'config.resolved.json').write_text(json.dumps(config))
    report=dict(status='completed',accepted_steps=2000,ep_steps=2000,time_s=.1,gpu='same GPU',
                warmed_compute_seconds=2. if profile=='reference' else 1.)
    (path/'report.json').write_text(json.dumps(report))
    (path/'history.json').write_text(json.dumps([dict(time_s=.1,minimum_gap_cm=.5,mid_gap_cm=.55,lumen_mid_flow_cm2_s=2.)]))
    np.savez(path/'snapshots.npz',reference=np.ones((2,2)),cells=np.array([[0,1]]),lower=[0],upper=[1],times=[.1],
             x=np.full((1,2,2),1.1),voltage=np.ones((1,2)),u=np.ones((1,2,2)),v=np.ones((1,2,2)),p=np.ones((1,2,2)))
    np.savez(path/'final_state.npz',gpb=np.ones((2,57)),land=np.ones((2,4)),zs=np.ones(2),zw=np.ones(2),force=np.ones((2,2)))


def test_rejects_parameter_changes_and_nonmatching_fields(tmp_path):
    a,b=tmp_path/'reference',tmp_path/'optimized'
    fixture_run(a,'reference'); fixture_run(b,'optimized')
    result=comparison.compare_runs(a,b)
    assert result['passed'] and result['speedup']==2.
    assert not result['external_literature_validation']
    path=b/'config.resolved.json'
    config=json.loads(path.read_text())
    config['solid']['shear_modulus']*=2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='model/mesh'):
        comparison.compare_runs(a,b)
    config['solid']['shear_modulus']/=2
    path.write_text(json.dumps(config))
    with np.load(b/'snapshots.npz') as data:
        changed={k:data[k] for k in data.files}
    changed['voltage']+=1.
    np.savez(b/'snapshots.npz',**changed)
    result=comparison.compare_runs(a,b)
    assert not result['passed'] and result['speedup'] is None


def test_rejects_incomplete_trajectory(tmp_path):
    a,b=tmp_path/'reference',tmp_path/'optimized'
    fixture_run(a,'reference'); fixture_run(b,'optimized')
    path=b/'report.json'
    report=json.loads(path.read_text()); report['status']='failed'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='complete'):
        comparison.compare_runs(a,b)

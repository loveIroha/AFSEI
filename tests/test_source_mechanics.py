"""Source profile integration and preservation of pre-profile checkpoints."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from test_afsei_driver import case,assert_same,physical_arrays
from test_real_lv import real_case
from afsi_torch.ep.config import AFSEIConfig
from afsi_torch.ep.driver import AFSEIStepper
from afsi_torch.ep.checkpoint import save,load,digest
from afsi_torch.simulation.afsei import run


def test_old_checkpoint_without_profile_preserves_inherited_physics(case,tmp_path):
    original,old=case; accepted,_=original.step(old)
    path=tmp_path/'legacy.npz'; save(path,original,accepted)
    with np.load(path,allow_pickle=False) as archive: data=dict(archive)
    metadata=json.loads(str(data.pop('metadata')))
    metadata.pop('sha256'); metadata['config'].pop('mechanics_profile')
    metadata['sha256']=digest(metadata,data)
    with path.open('wb') as stream: np.savez_compressed(stream,metadata=json.dumps(metadata),**data)
    driver,restart,_=load(path)
    assert driver.config.mechanics_profile=='inherited'
    assert driver.config.fsi.material==original.config.fsi.material
    assert driver.config.fsi.beta==original.config.fsi.beta
    assert driver.config.fsi.basal_center_cm==original.config.fsi.basal_center_cm
    assert_same(accepted,restart,exact=True)
    a,_=original.step(accepted); b,_=driver.step(restart)
    assert_same(a,b)
    source=AFSEIConfig.source(fsi=original.config.fsi)
    assert source.fsi.material!=driver.config.fsi.material
    with pytest.raises(ValueError,match='source mechanics'):
        replace(source,fsi=original.config.fsi)


def test_source_profile_rolls_back_and_restarts_consistently(case,tmp_path,monkeypatch):
    inherited,seed=case
    config=AFSEIConfig.source(fsi=inherited.config.fsi)
    driver=AFSEIStepper(inherited.mesh,config)
    old=driver.initialize(tissue=seed.tissue)
    project=driver.ep.project_cell_field; failed=[]
    snapshot=[v.clone() for v in physical_arrays(old)]
    def inject(value):
        if not failed:
            failed.append(True); raise FloatingPointError('source profile post-FSI rejection')
        return project(value)
    monkeypatch.setattr(driver.ep,'project_cell_field',inject)
    accepted,info=driver.step(old)
    assert accepted.tick==old.tick+1 and info['ep']['steps']==1 and len(info['rejected_attempts'])==1
    for a,b in zip(physical_arrays(old),snapshot): torch.testing.assert_close(a,b,rtol=0,atol=0)
    save(tmp_path/'source.npz',driver,accepted)
    restored,restart,_=load(tmp_path/'source.npz')
    assert restored.config==config and restored.config.mechanics_profile=='source'
    direct,_=driver.step(accepted); resumed,_=restored.step(restart)
    assert_same(direct,resumed)
    first=run(config=config,output=tmp_path/'run',duration_s=.0001)
    continued=run(resume=tmp_path/'run/checkpoint.npz',duration_s=.00005)
    uninterrupted=run(config=config,output=tmp_path/'direct',duration_s=.00015)
    assert first['source_mechanics_parameters_aligned'] and continued['mechanics_profile']=='source'
    assert continued['ep_steps']==uninterrupted['ep_steps']==3
    _,a,_=load(tmp_path/'run/checkpoint.npz'); _,b,_=load(tmp_path/'direct/checkpoint.npz')
    assert_same(a,b)

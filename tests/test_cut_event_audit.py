"""The independent pressure audit must reject a wrong application of Rc Q."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('cut_event_audit',Path(__file__).parents[1]/'scripts/analyze_gao_event_window.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


@pytest.mark.parametrize('extra_pressure',[0.,.25])
def test_independent_rcr_audit_catches_pressure_error(tmp_path,extra_pressure):
    pars=dict(rc_mmhg_s_per_ml=.05,rp_mmhg_s_per_ml=1.05,compliance_ml_per_mmhg=1.3,venous_mmhg=8.)
    np.savez(tmp_path/'initial_checkpoint.npz',metadata=json.dumps(dict(balanced=dict(windkessel=pars))))
    pc0=85.;dt=1e-4;q=3.;pc=(1.3*pc0+dt*(q+8./1.05))/(1.3+dt/1.05)
    pout=pc+.05*q+extra_pressure
    b=dict(mode='rcr',arterial_pc_old_mmhg=pc0,arterial_pc_mmhg=pc,physical_outlet_flow_ml_s=q,
        prescribed_cut_pressure_mmhg=pout,internal_pressure_sheets_mmhg=[0.,0.],
        pressure_core=dict(mean_mmhg=90.),volume_ml=100.,tract_volume_ml=10.,
        boundary_pressure_mmhg=pout,boundary_pressure_kind='prescribed-physical-cut',mass_error_ml_s=1e-10)
    c=dict(events=['PHYSICAL_AV_OPENING'],total_blood_volume_ml=110.,cumulative_mass_error_ml=1e-12)
    row=dict(time_ms=550.,step_seconds=3.,accepted_dt_s=dt,phase='EJECTION',balanced=b,cardiac=c,
             tension_max_cgs=50000.,rejected_attempts=[],valve_trial_discarded=False)
    (tmp_path/'steps.jsonl').write_text(json.dumps(row)+'\n')
    (tmp_path/'report.json').write_text(json.dumps(dict(status='passed',event_window=dict(completed=True))))
    result=audit.summarize(tmp_path)
    assert result['actual_ejection_steps']==1
    assert result['ejection_audit']['pressure_law_error_mmhg']==pytest.approx(extra_pressure,abs=1e-12)
    assert result['requested_validation_passed']==(extra_pressure==0.)

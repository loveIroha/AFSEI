"""Compact read-only status; no GPU transfers, solver work, or other jobs."""
from pathlib import Path
import json
root=Path('/mnt/large2/gjh/AFSEI/results/accuracy_performance_20261025')
report={}
for name in ('solid3_mac64','solid3_mac96','solid3_mac128'):
    folder=root/name;p=folder/'status.json';state=json.loads(p.read_text()) if p.exists() else dict(stage='queued')
    rows=[]
    for phase in ('prefix_1ms','continued_300ms'):
        p=folder/phase/'history.json'
        if p.exists():rows.extend(json.loads(p.read_text()))
    r=dict(stage=state.get('stage'),last_recorded_time_ms=1000*max((h['time_s'] for h in rows),default=0.))
    if state.get('error'):r['error']=state['error']
    report[name]=r
for label,name in [('spatial_analysis','analysis/report.json'),('time_control','time_control_status.json'),
                   ('original_AFSI_primary_baseline','afsi_original_pilots_status.json'),
                   ('native_release','ibamr_release_pilots_status.json'),('electrical_and_GPU_pilot','software_gpu_inputs_status.json')]:
    p=root/name
    if p.exists():
        value=json.loads(p.read_text())
        report[label]={k:v for k,v in value.items() if k in ('stage','active_case','error','fixed_dt_spatial_acceptance_passed','full_space_time_convergence_established','formal_performance_established','formal_matched_accuracy_speedup_established','case','completed','EP_history_checked')}
        if label=='original_AFSI_primary_baseline':
            report[label]['cases']={k:{key:v.get(key) for key in ('status','time_s','compute_seconds')} for k,v in value.get('cases',{}).items()}
    else:report[label]=dict(stage='pending')
scope=root/'user_scope_update_stop128.json'
if scope.exists() and report.get('solid3_mac128',{}).get('stage')=='stopped-by-user':
    snapshot=root/'latest_analysis_snapshot.json'
    report['spatial_analysis'].update(stage='ended-at-user-request-current-scalar-differences-accepted',
        full_space_time_convergence_established=False,
        latest_snapshot=json.loads(snapshot.read_text()).get('output') if snapshot.exists() else None)
print(json.dumps(report,indent=2))

"""Summarize completed controls without rerunning a fluid or cardiac solve."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    original=json.loads((a.output/'report.json').read_text())
    verified=json.loads((a.output/'work_pair_verified/report.json').read_text())
    assert original['completed'] and original['checkpoint_unchanged']
    assert verified['completed'] and verified['checkpoint_unchanged']
    assert verified['checkpoint_sha256']==original['checkpoint_sha256']
    assert all(q['passed'] for q in verified['pressure_work_tests']+verified['cap_pair_tests'])
    r=dict(original)
    r['initial_diagnostic_pressure_work_tests']=original['pressure_work_tests']
    for k in ('pressure_work_tests','cap_pair_tests','unit_target_core_mmhg'):
        r[k]=verified[k]
    r['diagnostic_correction']='The first work-only lifting expression multiplied a Python float by a Boolean aperture, constructing float32 intermediate data. The verified expression casts aperture to float64 first, exactly as the production cut-plane solver. No production operator changed; all original hydrostatic results are retained.'
    r['work_verification_seconds']=verified['elapsed_seconds']
    r['checks_passed']={
        'pressure_work_identity':True,'cap_pair_work_identity':True,
        'nodal_cap_gradient_identity':r['nodal_cap_to_gradient_relative_error']<1e-10,
        'checkpoints_unchanged':True}
    r['scope']='Frozen-geometry manufactured hydrostatics and work identities only; not a new coupled physical trajectory, EF or ESV measurement.'
    for case in r['hydrostatic_cases']:
        target=case['pressure_scale_mmhg']*r['unit_target_core_mmhg']['mean']
        case['target_core_mmhg']=target
        case['computed_core_mmhg']=target+case['pressure_core_error_mmhg']
        case['computed_core_minus_cut_mmhg']=case['computed_core_mmhg']-case['pressure_scale_mmhg']
    test80=next(q for q in r['hydrostatic_cases'] if q['name']=='current_gradient_80mmhg')
    r['physical_scale_comparison']={
        'previous_actual_ejection_outflow_ml_s':324.156234996,
        'manufactured_80mmhg_abs_outflow_over_actual':abs(test80['outlet_flow_ml_s'])/324.156234996,
        'previous_actual_core_to_cut_gap_mmhg':26.042813552,
        'manufactured_core_pressure_error_over_actual_gap':abs(test80['pressure_core_error_mmhg'])/26.042813552,
        'interpretation':'This comparison bounds the magnitude in one manufactured static experiment only; it does not bound all long-time coupled, nonhydrostatic or moving-cap errors.'}
    (a.output/'reviewed_report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')

    names=('current_gradient','BE_compatible_manufactured_control','projection_only_control')
    labels=('Current BE/projection','BE-compatible load\n(manufactured control)','Projection only\n(diagnostic control)')
    fig,ax=plt.subplots(1,2,figsize=(11.5,5.2),layout='constrained')
    colors=('#2672b2','#e18a2e')
    for j,(dt,color) in enumerate(zip((1e-4,5e-5),colors)):
        selected=[next(q for q in r['hydrostatic_cases'] if q['name']==name and q['dt_s']==dt) for name in names]
        for axis,key in zip(ax,('max_velocity_cm_s','outlet_flow_ml_s')):
            values=[abs(q[key]) for q in selected]
            bars=axis.bar(np.arange(3)+(j-.5)*.34,values,width=.32,color=color,label=f'dt = {dt*1000:g} ms')
            axis.bar_label(bars,labels=[f'{v:.2e}' for v in values],fontsize=8,padding=3)
    for axis,title,ylabel in zip(ax,('Residual static velocity','Residual outlet flux'),('Max |u| [cm/s]','|Qout| [mL/s]')):
        axis.set_yscale('log');axis.set_ylim(1e-13,2e-2)
        axis.set_xticks(np.arange(3),labels,fontsize=9)
        axis.set(title=title,ylabel=ylabel);axis.grid(axis='y',alpha=.2);axis.legend(fontsize=9)
    fig.suptitle('577.9 ms frozen LV geometry: unit-pressure manufactured controls',fontsize=13)
    fig.supxlabel('At 80 mmHg: max |u| = 0.0233 cm/s, |Qout| = 0.000923 mL/s; no physical time advanced.',fontsize=9)
    fig.savefig(a.output/'static_port_summary.png',dpi=170)
    plt.close(fig)
    print(json.dumps(dict(checks_passed=r['checks_passed'],
        unit_target_core_mmhg=r['unit_target_core_mmhg'],
        pressure_work_tests=r['pressure_work_tests'],cap_pair_tests=r['cap_pair_tests'],
        max_velocity_half_to_full_ratio=r['max_velocity_half_to_full_ratio'],
        physical_scale_comparison=r['physical_scale_comparison']),indent=2))


if __name__=='__main__':main()

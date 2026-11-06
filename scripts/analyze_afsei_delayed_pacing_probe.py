"""Analyze the extra 100 ms without inventing a delayed-contraction result."""
import csv
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/afsei_mpl_20261022')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main(out):
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((out/'experiment.json').read_text())
    reference_dir = root/cfg['comparison']
    ref = json.loads((reference_dir/'comparison.json').read_text())
    original = ref['cases']['early']
    rows = []
    elapsed = steps = retries = 0
    for stage, expected_phase in [('original_boundary','FILLING'), ('extended_filling','FILLING'),
                                   ('delayed_stimulus','ISOVOLUMIC_CONTRACTION')]:
        folder = out/stage
        report = json.loads((folder/'report.json').read_text())
        assert report['status']=='completed' and report['failure'] is None and report['phase']==expected_phase
        elapsed += report['elapsed_seconds']; steps += report['accepted_steps_this_run']
        part = read(folder/'pv.csv')
        rows.extend(part)
        with (folder/'steps.jsonl').open() as f:
            for line in f:
                retries += len(json.loads(line)['retries'])
    values = lambda key: np.array([float(row[key]) for row in rows])
    t, volume, q = values('time_ms'), values('volume_ml'), values('q_in_ml_s')
    assert np.all(np.diff(t)>0) and abs(t[0]-1300)<1e-7 and abs(t[-1]-1400)<1e-7
    assert rows[0]['phase']=='FILLING' and rows[-1]['phase']=='ISOVOLUMIC_CONTRACTION'
    assert np.max(np.abs(values('wall_pressure_mmhg'))) < 1e-10
    assert np.max(np.abs(values('supply_pressure_mmhg')-8.)) < 1e-10
    assert np.max(np.abs(values('tension_max_cgs'))) < 1e-10
    reference_edv = original['edv_at_1300_ml']+original['reference_edv_gap_ml']
    gain = float(volume[-1]-volume[0])
    dt = np.diff(t)/1000
    integrated_in = float(np.dot(q[1:],dt))
    integrated_out = float(np.dot(values('q_out_ml_s')[1:],dt))
    assert abs(gain-integrated_in+integrated_out) < 1e-7
    result = dict(status='completed', source_branch='early-inlet', original_pacing_ms=1300.,
        delayed_pacing_ms=1400., extra_filling_ms=100.,
        original_edv_at_1300_ml=original['edv_at_1300_ml'],
        delayed_branch_volume_at_1300_ml=float(volume[0]),
        restart_difference_at_1300_ml=float(volume[0]-original['edv_at_1300_ml']),
        delayed_edv_at_1400_ml=float(volume[-1]), added_filling_ml=gain,
        average_qin_extra_interval_ml_s=integrated_in/.1,
        reference_edv_ml=reference_edv, remaining_reference_edv_gap_ml=float(reference_edv-volume[-1]),
        final_qin_ml_s=float(q[-1]), final_core_pressure_mmhg=float(values('pressure_core_mmhg')[-1]),
        max_mass_error_ml_s=float(np.max(np.abs(values('mass_error_ml_s')))),
        integrated_volume_balance_error_ml=float(gain-integrated_in+integrated_out),
        retries=retries, accepted_steps=steps, elapsed_seconds=elapsed,
        second_contraction_computed=False, full_two_cycles=False,
        waveform_stretched=False, artificial_unloading_changed=False,
        pre_stimulus_checkpoint=str(out/'extended_filling/checkpoint.npz'))
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    keys = rows[0].keys()
    with (out/'extra_filling.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader();writer.writerows(rows)
    history = [r for r in read(reference_dir/'early_filling.csv') if float(r['time_ms'])<1300]+rows
    ht = np.array([float(r['time_ms']) for r in history])
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    fields = ['volume_ml','q_in_ml_s','pressure_core_mmhg','supply_pressure_mmhg']
    labels = ['Volume (mL)','Inlet flow (mL/s)','Cavity-core pressure (mmHg)','Prescribed pressure (mmHg)']
    for ax, key, label in zip(axes.flat,fields,labels):
        y=np.array([float(r[key]) for r in history])
        ax.plot(ht,y,color='tab:orange',label='Early inlet, delayed second stimulus')
        ax.axvline(1300,color='gray',linestyle='--',label='Original stimulus time')
        ax.axvspan(1300,1400,color='tab:green',alpha=.10)
        ax.set(xlabel='Time (ms)',ylabel=label);ax.grid(alpha=.25)
    axes[0,0].plot(1300,original['edv_at_1300_ml'],'o',color='tab:blue',label='Original EDV')
    axes[0,0].axhline(reference_edv,color='k',linestyle=':',label='Solid-only reference EDV')
    axes[1,1].plot(ht,[float(r['wall_pressure_mmhg']) for r in history],'k--',label='Wall load')
    for ax in axes.flat:ax.legend(fontsize=8)
    fig.suptitle(f"100 ms extra filling: {gain:.3f} mL added; delayed EDV {volume[-1]:.3f} mL\nSecond stimulus 1300 -> 1400 ms; original pressure history retained")
    for suffix in ['png','pdf']:fig.savefig(out/f'delay100_comparison.{suffix}',dpi=180)
    text=(f"# 第二次刺激延后 100 ms 的结果\n\n"
        f"原提前供压组 1300 ms EDV：{original['edv_at_1300_ml']:.5f} mL。\n\n"
        f"延后组 1400 ms EDV：{volume[-1]:.5f} mL；新增充盈：{gain:.5f} mL。\n\n"
        f"相对纯固体参考仍差：{reference_edv-volume[-1]:.5f} mL；末端流入：{q[-1]:.5f} mL/s。\n\n"
        "1300–1400 ms 的入口维持 8 mmHg，辅助壁载荷和实际主动张力均为零。\n\n"
        "本次只延长第二次刺激之前的充盈并到达新刺激边界，没有计算随后收缩或完整两周期。\n")
    (out/'analysis_zh.md').write_text(text)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))

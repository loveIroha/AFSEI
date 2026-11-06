"""Compare the saved early-filling segment and its standard-driver continuation."""
import csv
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/afsei_mpl_20261022')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(output):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((output/'experiment.json').read_text())
    traces = {}
    summary = dict(scope=config['scope'], original_second_stimulus_ms=1300.,
        pacing_changed=False, full_two_cycles=False, cases={})
    for case in config['cases']:
        name = case['name']
        first = root/case['first_segment']
        initial = json.loads((first/'report.json').read_text())
        data = read_csv(first/'pv.csv')
        continuation = []
        for stage in ('pre_stimulus', 'at_stimulus'):
            folder = output/name/stage
            report = json.loads((folder/'report.json').read_text())
            assert report['status'] == 'completed' and report['failure'] is None
            continuation.append(report)
            data.extend(read_csv(folder/'pv.csv'))
        times = np.array([float(row['time_ms']) for row in data])
        assert np.all(np.diff(times) > 0) and abs(times[-1]-1300.) < 1e-7
        values = lambda key, rows=data: np.array([float(row[key]) for row in rows])
        traces[name] = dict(time=times, values=values, data=data)
        last = data[-1]
        duration = (times[-1]-initial['initial_time_ms'])/1000
        net = float(last['volume_ml'])-initial['initial_volume_ml']
        entry = dict(initial_volume_ml=initial['initial_volume_ml'],
            filling_start_ms=initial['initial_time_ms'], available_filling_ms=duration*1000,
            edv_at_1300_ml=float(last['volume_ml']), net_filling_ml=net,
            mean_net_filling_ml_s=net/duration, qin_at_1300_ml_s=float(last['q_in_ml_s']),
            core_pressure_at_1300_mmhg=float(last['pressure_core_mmhg']),
            supply_pressure_at_1300_mmhg=float(last['supply_pressure_mmhg']),
            reference_edv_gap_ml=config['reference_edv_ml']-float(last['volume_ml']),
            max_mass_error_ml_s=float(np.max(np.abs(values('mass_error_ml_s')))),
            preserved_pre_stimulus_checkpoint=str(output/name/'pre_stimulus/checkpoint.npz'),
            continuation_wall_seconds=sum(r['elapsed_seconds'] for r in continuation),
            physiological_convergence_established=False)
        summary['cases'][name] = entry
        # Preserve a compact, continuously sampled filling history without synthetic closure.
        keys = ['time_ms','phase','volume_ml','pressure_core_mmhg','pc_mmhg',
            'wall_pressure_mmhg','supply_pressure_mmhg','boundary_pressure_mmhg',
            'inlet_open','q_in_ml_s','q_out_ml_s','mass_error_ml_s','step_seconds']
        with (output/f'{name}_filling.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows({key:row[key] for key in keys} for row in data)
    a, b = summary['cases']['baseline'], summary['cases']['early']
    summary['additional_volume_at_1300_ml'] = b['edv_at_1300_ml']-a['edv_at_1300_ml']
    summary['relative_net_filling_gain_percent'] = 100*(b['net_filling_ml']/a['net_filling_ml']-1)
    summary['delay_duration_not_yet_validated'] = True
    (output/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fields = ['volume_ml','q_in_ml_s','supply_pressure_mmhg','pressure_core_mmhg']
    labels = ['Volume (mL)','Inlet flow (mL/s)','Prescribed pressure (mmHg)','Cavity-core pressure (mmHg)']
    for name, label, color in [('baseline','Original inlet ramp','tab:blue'),('early','Early inlet plateau','tab:orange')]:
        trace = traces[name]
        for ax, key in zip(axes.flat, fields):
            ax.plot(trace['time'], trace['values'](key), label=label, color=color)
    trace = traces['baseline']
    axes[1,0].plot(trace['time'], trace['values']('wall_pressure_mmhg'), 'k--', label='Shared wall load')
    for ax, label in zip(axes.flat, labels):
        ax.set(xlabel='Time (ms)', ylabel=label)
        ax.grid(alpha=.25); ax.legend()
    fig.suptitle('Completed filling comparison to the original 1300 ms stimulus\n'
                 f"Additional volume: {summary['additional_volume_at_1300_ml']:.3f} mL; pacing unchanged")
    for suffix in ('png','pdf'):
        fig.savefig(output/f'filling_to1300.{suffix}', dpi=180)
    plt.close(fig)
    lines = ['# 完整充盈窗口续算结果', '',
        '两组复用既有 1011.8 ms 检查点，保存 1299.9 ms 刺激前状态，再推进到原定 1300 ms。', '',
        '| 指标 | 原入口升压 | 提前供压 |','|---|---:|---:|']
    for label, key in [('1300 ms EDV，mL','edv_at_1300_ml'),('净充盈量，mL','net_filling_ml'),
            ('末端 Qin，mL/s','qin_at_1300_ml_s'),('核心平均压力，mmHg','core_pressure_at_1300_mmhg'),
            ('距参考 EDV 的差，mL','reference_edv_gap_ml')]:
        lines.append(f"| {label} | {a[key]:.5f} | {b[key]:.5f} |")
    lines.extend(['', f"提前供压额外增加容积 {summary['additional_volume_at_1300_ml']:.5f} mL。", '',
        '本实验保留原定刺激时刻，未执行延后刺激或新的完整两周期。是否延后以及延后多少，需要结合完整曲线和刺激前检查点进行后续判断。', '',
        '保存的刺激前检查点位于每组 pre_stimulus/checkpoint.npz；不能用 at_stimulus/checkpoint.npz 直接冒充未刺激状态。'])
    (output/'analysis_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))

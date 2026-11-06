"""Replot PV and draw cycle 3 from saved left ventricular results."""
import csv
import h5py
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import meshio
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


root = Path(sys.argv[1])
output = root / 'plots' / 'passive08_pv_cycle3_20261106'
output.mkdir(parents=True, exist_ok=True)
run = json.loads((root / 'summary.json').read_text())
assert run['status'] == 'completed' and run['time_s'] == 2.5
with (root / 'history.csv').open() as stream:
    history = list(csv.DictReader(stream))
events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines() if line]
time = np.array([float(row['time_s']) for row in history])
volume = np.array([float(row['volume_ml']) for row in history])
pressure = np.array([float(row['endocardial_load_mmhg']) for row in history])
starts = [event for event in events if event['phase'] == 'IVC']
assert len(starts) == 2 and len(history) == run['accepted_steps']
assert np.all(np.diff(time) > 0) and time[-1] == run['time_s']

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.dpi': 210})


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{ext}')
    plt.close(fig)


def phase_event(beat, phase):
    return next(event for event in events if event['phase'] == phase and event['beat'] == beat)


initial_volume = volume[0] - float(history[0]['cumulative_in_ml']) + float(history[0]['cumulative_out_ml'])
fig, ax = plt.subplots(figsize=(10, 7.2))
fig.subplots_adjust(left=.12, right=.97, bottom=.12, top=.83)
preload = time <= starts[0]['time_s'] + 1e-12
ax.plot(np.r_[initial_volume, volume[preload]], np.r_[0., pressure[preload]], color='.55', lw=1.5,
        label='Cycle 1 | passive loading, 0–0.8 s')
ax.scatter(initial_volume, 0., color='.55', s=25, zorder=5)
cycle_stats = []
for i, start in enumerate(starts):
    beat, onset = start['beat'], start['time_s']
    end = starts[i+1]['time_s'] if i+1 < len(starts) else time[-1]
    opened, closed, filling = (phase_event(beat, phase) for phase in ('EJECTION', 'IVR', 'FILLING'))
    selection = (time >= onset - 1e-12) & (time <= end + 1e-12)
    ejection = selection & (time >= opened['time_s'] - 1e-12) & (time <= closed['time_s'] + 1e-12)
    edv = float(start['volume_ml'])
    esv = float(closed['volume_ml'])
    ef = 100 * (edv - esv) / edv
    color = ['#1f77b4', '#e07918'][i]
    draw = selection.copy()
    if i == 0:
        omitted = selection & (time > onset) & (time <= onset + .0003 + 1e-12) & (pressure < 0)
        assert np.count_nonzero(omitted) == 3
        draw &= ~omitted
    ax.plot(volume[draw], pressure[draw], color=color, lw=1.8,
            label=f'Cycle {i+2} | active beat {beat}, EF {ef:.2f}%')
    ax.scatter(volume[selection][0], pressure[selection][0], color=color, s=34, zorder=5)
    ax.scatter(volume[selection][-1], pressure[selection][-1], facecolors='white',
               edgecolors=color, s=44, zorder=6)
    trough = np.flatnonzero(selection)[np.argmin(pressure[selection])]
    cycle_stats.append({'cycle': i+2, 'stimulus_s': onset, 'open_s': opened['time_s'],
                        'close_s': closed['time_s'], 'fill_s': filling['time_s'], 'end_s': end,
                        'volume_at_stimulus_ml': edv, 'esv_at_closure_ml': esv, 'ef_pct': ef,
                        'ejected_ml': closed['cumulative_out_ml'] - start['cumulative_out_ml'],
                        'peak_ejection_pressure_mmhg': float(pressure[ejection].max()),
                        'min_pressure_mmhg': float(pressure[trough]), 'min_pressure_time_s': float(time[trough]),
                        'ending_volume_ml': float(volume[selection][-1]),
                        'ending_pressure_mmhg': float(pressure[selection][-1])})
ax.set(xlabel='LV cavity volume (mL)', ylabel='Endocardial coupling pressure (mmHg)')
ax.grid(alpha=.23)
ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), frameon=False, fontsize=10)
fig.suptitle('Left ventricular pressure–volume trajectories, 0–2.5 s', y=.98, fontsize=16)
save(fig, 'pv_all_times_onset_spike_omitted')


output = root / 'plots' / 'passive08_20261106'
layers = ('endo_side', 'middle', 'epi_side')
labels = ('Endocardial side', 'Midwall', 'Epicardial side')
colors = ('#c34e32', '#1b8d78', '#3166ac')
with (root / 'active_tension_layers.csv').open() as stream:
    tension_rows = list(csv.DictReader(stream))
tension_time = np.array([float(row['time_s']) for row in tension_rows])
tension = np.array([[float(row[f'{layer}_Ta_mean_kpa']) for layer in layers] for row in tension_rows])
assert np.allclose(tension[tension_time <= .8 + 1e-12], 0.)
with h5py.File(root / 'active_load_history.h5', 'r') as h5:
    force_time = h5['time_s'][:]
    phi = h5['transmural_phi'][:]
    layer_nodes = (phi < 1/3, (phi >= 1/3) & (phi < 2/3), phi >= 2/3)
    assert all(mask.any() for mask in layer_nodes)
    nodal_rms = np.empty((len(force_time), len(layers)))
    for i in range(len(force_time)):
        force = h5['active_force_dyn'][i]
        squared_norm = np.sum(force * force, axis=1)
        nodal_rms[i] = [np.sqrt(squared_norm[mask].mean()) * .01 for mask in layer_nodes]

fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.6), sharex=True)
fig.subplots_adjust(left=.11, right=.97, bottom=.11, top=.9, hspace=.17)
for ax, ts, data, ylabel in zip(axes, (tension_time, force_time), (tension, nodal_rms),
                               ('Mean applied Land tension (kPa)', 'RMS active nodal force (mN)')):
    for j, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(ts, data[:, j], label=label, color=color, lw=1.8)
    for cycle in cycle_stats:
        ax.axvline(cycle['stimulus_s'], color='.45', ls='--', lw=1)
        ax.axvspan(cycle['close_s'], cycle['close_s'] + .08, color='.8', alpha=.35)
    ax.set(ylabel=ylabel, xlim=(0., time[-1]))
    ax.grid(alpha=.22)
axes[0].legend(loc='upper left', frameon=False)
axes[-1].set_xlabel('Time (s)')
fig.suptitle('Active contraction in three myocardial layers', y=.97, fontsize=16)
save(fig, 'active_contraction_layers')

"""Plot the saved passive-then-active LV run without modifying its history."""
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import matplotlib
import meshio
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


root = Path(sys.argv[1])
output = root / 'plots' / 'passive08_20261106'
output.mkdir(parents=True, exist_ok=True)
run = json.loads((root / 'summary.json').read_text())
assert run['status'] == 'completed' and run['time_s'] == 2.5
with (root / 'history.csv').open() as stream:
    history = list(csv.DictReader(stream))
events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines() if line]
time = np.array([float(row['time_s']) for row in history])
volume = np.array([float(row['volume_ml']) for row in history])
pressure = np.array([float(row['endocardial_load_mmhg']) for row in history])
step_wall = np.array([float(row['wall_seconds']) for row in history])
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
fig.subplots_adjust(left=.12, right=.97, bottom=.22, top=.83)
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
    ax.plot(volume[selection], pressure[selection], color=color, lw=1.8,
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
fig.text(.12, .045, 'Solid markers show segment starts; open markers show recorded endpoints.\n'
         'All accepted pressure samples are shown, including the switching transients.\n'
         'Ejection fraction uses the volume at stimulus onset and at outlet closure.',
         fontsize=9, color='.35', linespacing=1.5)
save(fig, 'pv_all_times')

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
fig.subplots_adjust(left=.11, right=.97, bottom=.22, top=.9, hspace=.17)
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
fig.text(.11, .045, 'Layers are thirds of the reference transmural coordinate. Tension is volume-weighted.\n'
         'Dashed lines mark the stimuli; shaded bands indicate the 80 ms unloading intervals.\n'
         'The force panel shows a node-wise magnitude RMS for each layer.', fontsize=9, color='.35', linespacing=1.5)
save(fig, 'active_contraction_layers')

cycle2 = cycle_stats[0]
onset = cycle2['stimulus_s']
targets = [onset, onset+.01, onset+.03, cycle2['open_s'], .92, cycle2['close_s'],
           cycle2['close_s']+.04, 1.40, cycle2['end_s']]
records = ET.parse(root / 'wall.pvd').findall('.//DataSet')
records = [(float(rec.get('timestep')), rec.get('file')) for rec in records]
records = [(t, file) for t, file in records if onset-1e-12 <= t <= cycle2['end_s']+1e-12]
chosen = [min(records, key=lambda record: abs(record[0] - target)) for target in targets]
assert len(set(chosen)) == 9 and chosen == sorted(chosen)
frames = [(t, name, meshio.read(root / name)) for t, name in chosen]
with np.load(root / 'checkpoint.npz') as checkpoint:
    reference, faces = checkpoint['X'].copy(), checkpoint['faces'].copy()
base = frames[0][2].points
center = base.mean(axis=0)
cut = float(np.median(reference[:, 1]))

# Clip the same material triangles in every frame, using their reference positions.
selected, coefficients, cut_edges = [], [], []
eye = np.eye(3)
for face in faces:
    signed = reference[face, 1] - cut
    polygon, crossings = [], []
    for b in range(3):
        a = (b-1) % 3
        if (signed[a] >= 0) != (signed[b] >= 0):
            point = eye[a] + signed[a] / (signed[a] - signed[b]) * (eye[b] - eye[a])
            polygon.append(point)
            crossings.append(point)
        if signed[b] >= 0:
            polygon.append(eye[b])
    for j in range(1, len(polygon)-1):
        selected.append(face)
        coefficients.append([polygon[0], polygon[j], polygon[j+1]])
    if len(crossings) == 2:
        cut_edges.append(np.asarray(crossings) @ base[face])
selected, coefficients = np.asarray(selected), np.asarray(coefficients)
contour = (np.asarray(cut_edges) - center) * 10
bounds = (np.concatenate([mesh.points for _, _, mesh in frames]) - center) * 10
lo, hi = bounds.min(axis=0) - 3, bounds.max(axis=0) + 3
norm, cmap = Normalize(-85, 45), plt.get_cmap('turbo')
fig = plt.figure(figsize=(15, 13.5))
fig.subplots_adjust(left=.02, right=.91, bottom=.07, top=.90, hspace=.17, wspace=.03)
frame_metadata = []
for j, (t, name, mesh) in enumerate(frames):
    assert len(mesh.points) == len(reference)
    voltage = mesh.point_data['voltage_mV']
    points = (mesh.points - center) * 10
    triangles = np.einsum('tab,tbi->tai', coefficients, points[selected])
    values = np.einsum('tab,tb->ta', coefficients, voltage[selected]).mean(axis=1)
    ax = fig.add_subplot(3, 3, j+1, projection='3d')
    wall = Poly3DCollection(triangles, facecolors=cmap(norm(values)), edgecolors='none')
    wall.set_rasterized(True)
    ax.add_collection3d(wall)
    ax.add_collection3d(Line3DCollection(contour, colors='.5', linewidths=.5, linestyles='--', alpha=.5))
    index = int(np.argmin(abs(time-t)))
    assert abs(time[index]-t) < 1e-8
    ax.set_title(f'{t*1000:.1f} ms  |  LV volume {volume[index]:.2f} mL', fontsize=11, pad=4)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    ax.set_box_aspect(hi-lo)
    ax.set_proj_type('ortho')
    ax.view_init(elev=18, azim=-80)
    ax.tick_params(labelsize=6, pad=0)
    for axis, label in zip((ax.xaxis, ax.yaxis, ax.zaxis), ('x', 'y', 'z')):
        axis.pane.set_alpha(0)
        axis.set_label_text(f'{label} (mm)', fontsize=7)
        axis.labelpad = 0
    ax.grid(False)
    frame_metadata.append({'time_s': t, 'cavity_volume_ml': float(volume[index]),
                           'file': name, 'phase': history[index]['phase']})
bar = fig.add_axes([.935, .2, .012, .60])
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar, label='Membrane voltage (mV)')
fig.suptitle('Cycle 2 | Electrical propagation and wall deformation', y=.97, fontsize=18)
fig.text(.48, .936, 'First active beat after the 0.8 s passive stage', ha='center', fontsize=11)
fig.text(.48, .027, 'Deformed half-wall at true scale. Dashed contours show the wall at 800 ms.\n'
         'Panel labels report the recorded cavity volume at each snapshot.', ha='center', fontsize=10, linespacing=1.5)
save(fig, 'cycle2_voltage_deformation_nine')

timing = []
for i, (a, b) in enumerate(zip((0., .8, 1.65), (.8, 1.65, 2.5)), 1):
    selected_steps = step_wall[(time > a + 1e-12) & (time <= b + 1e-12)]
    timing.append({'cycle': i, 'steps': int(selected_steps.size), 'step_wall_sum_s': float(selected_steps.sum()),
                   'mean_s_per_step': float(selected_steps.mean())})

peaks = []
for cycle in cycle_stats:
    indices = np.flatnonzero((tension_time >= cycle['stimulus_s']-1e-12) & (tension_time <= cycle['end_s']+1e-12))
    peaks.append({'cycle': cycle['cycle'], 'tension_peaks_kpa': dict(zip(layers, tension[indices].max(axis=0).tolist()))})
metrics = {'runtime': {'preparation_s': run['preparation_seconds'], 'run_s': run['run_seconds'],
                       'accepted_steps': len(history), 'run_mean_s_per_step': run['run_seconds']/len(history),
                       'phase_step_wall_times': timing},
           'cycles': cycle_stats, 'layer_tension_peaks': peaks, 'frames': frame_metadata}
(output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
print(json.dumps(metrics, indent=2))

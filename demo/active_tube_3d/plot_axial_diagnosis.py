import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args()
stress=json.loads((a.root/'axial_stress.json').read_text())['results']
shape=json.loads((a.baseline/'analysis/geometry_audit.json').read_text())['saved_frame_metrics']
fig,axs=plt.subplots(1,3,figsize=(15,4.3))
axs[0].semilogy([r['time_ms'] for r in shape],np.maximum([r['maximum_transverse_centerline_shift_mm'] for r in shape],1e-12))
axs[0].set(xlim=(100,400),xlabel='Time (ms)',ylabel='Centerline transverse shift (mm)',title='Growth of lateral perturbation')
axs[1].plot([r['time_ms'] for r in stress],[r['central_mean_axial_cauchy_stress_kpa'] for r in stress],'.-')
axs[1].axhline(0,color='gray',ls=':');axs[1].set(xlabel='Time (ms)',ylabel='Central mean axial stress (kPa)',title='Positive means tension')
for j,label in [(0,'x bend, one wave'),(2,'y bend, one wave')]:
 v=[r['material_trial_mode_curvatures'][j]['normalized_curvature'] for r in stress]
 axs[2].plot([r['time_ms'] for r in stress],np.array(v)/v[0],'.-',label=label)
axs[2].axhline(0,color='gray',ls=':');axs[2].set(xlabel='Time (ms)',ylabel='Trial stiffness / initial value',title='Material-only directional tangent');axs[2].legend(fontsize=8)
for ax in axs:ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(a.root/'axial_diagnosis.png',dpi=170);plt.close(fig)

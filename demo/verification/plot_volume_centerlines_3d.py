"""Existing final centerlines only; no simulation or changed constraints."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/mnt/large2/gjh/AFSEI/results');out=root/'volumetric_stabilization_20261025/results_analysis'
fig=plt.figure(figsize=(12,5.5),layout='constrained')
for k,old in [(1,'coarse_box2'),(2,'fine_box2')]:
    ax=fig.add_subplot(1,2,k,projection='3d')
    folders=[root/f'domain_space_campaign_20261025/{old}',root/f'volumetric_stabilization_20261025/from_rest_300ms/solid{k}']
    for folder,color,label in zip(folders,['#666666','#176d9c'],['Baseline','Stabilized']):
        a=np.load(folder/'continued_300ms/final_state.npz');c=a['x'][a['inner_rings']].mean(1)
        ax.plot(c[:,2]*10,(c[:,0]-.7)*10,(c[:,1]-.7)*10,color=color,lw=2,label=label)
    ax.set(xlabel='Current axial z (mm)',ylabel='Transverse x (mm)',zlabel='Transverse y (mm)',
           title=f'Solid {k} at 300 ms',xlim=(2,26),ylim=(-9,9),zlim=(-9,9))
    ax.set_box_aspect((24,18,18));ax.view_init(elev=23,azim=-57);ax.legend(fontsize=9)
fig.suptitle('Centerlines with original finite end springs; no centering or bending constraint',fontsize=12)
for ext in ('png','pdf'):fig.savefig(out/f'centerlines_3d.{ext}',dpi=180)
plt.close(fig)

"""Read existing snapshots only; plot myocardium inside the original MAC box."""
from pathlib import Path
import json, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def hull_vertices(points):
    points=sorted(set(map(tuple,points)))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[];upper=[]
    for p in points:
        while len(lower)>1 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper)>1 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    hull=lower[:-1]+upper[:-1]
    return np.array(hull+[hull[0]])

root = Path('/mnt/large2/gjh/AFSEI')
run = root/'results/afsei_source_timing_neck_two_cycles_20261029'
out = root/'results/lv_domain_position_20261030'
out.mkdir(exist_ok=True)
cfg = json.loads((run/'configuration.json').read_text())['config']['fsi']
lo = np.array(cfg['fluid']['origin']); hi = lo + cfg['fluid']['lengths']
center = (lo+hi)/2
with np.load(run/'initial_checkpoint.npz', allow_pickle=False) as a:
    X, x, faces, tags = (a[k] for k in ('X','x','faces','facet_tags'))
    meta = json.loads(str(a['metadata']))
with np.load(run/'last_accepted_checkpoint.npz', allow_pickle=False) as a:
    last = a['x']; lastmeta = json.loads(str(a['metadata']))
basefaces = faces[tags==3]
epi = faces[tags==2]
if len(epi)==0 or len(basefaces)==0: raise RuntimeError('Missing mapped epi/base tags')
def stats(p):
    mn, mx = p.min(0), p.max(0)
    return dict(min_cm=mn.tolist(),max_cm=mx.tolist(),bbox_center_cm=((mn+mx)/2).tolist(),
        bbox_center_offset_from_box_cm=((mn+mx)/2-center).tolist(),
        lower_clearance_cm=(mn-lo).tolist(),upper_clearance_cm=(hi-mx).tolist(),
        base_mean_cm=p[np.unique(basefaces)].mean(0).tolist())
report = dict(source=str(run), fluid_box_min_cm=lo.tolist(),fluid_box_max_cm=hi.tolist(),
    fluid_box_center_cm=center.tolist(),fluid_shape=cfg['fluid']['shape'],
    spacing_cm=(np.array(cfg['fluid']['lengths'])/cfg['fluid']['shape']).tolist(),
    reference=stats(X),preloaded=stats(x),preloaded_time_s=meta['time'],
    last_accepted=stats(last),last_accepted_time_s=lastmeta['time'],
    scope='Actual original MAC box and myocardium; historical tract masks/caps not rendered; no new single-port run')
(out/'geometry_report.json').write_text(json.dumps(report,indent=2))
plt.rcParams.update({'font.size':11,'axes.titleweight':'bold'})
fig=plt.figure(figsize=(14,12),layout='constrained')
ax=fig.add_subplot(221,projection='3d')
verts=np.array(list(itertools.product(*zip(lo,hi))))
for a,b in itertools.combinations(verts,2):
    if np.count_nonzero(a!=b)==1: ax.plot(*np.array([a,b]).T,color='#536779',lw=1.1)
ax.add_collection3d(Poly3DCollection(x[epi],facecolor='#4c9fcb',edgecolor='none',alpha=.45,rasterized=True))
ax.add_collection3d(Poly3DCollection(x[basefaces],facecolor='#d84a46',edgecolor='none',alpha=.95))
sc=(x.min(0)+x.max(0))/2
ax.scatter(*center,color='black',marker='x',s=80,label='MAC box center')
ax.scatter(*sc,color='#e29a23',s=45,label='LV bounding-box center')
ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='x [cm]',ylabel='y [cm]',zlabel='z [cm]',title='Actual geometry: preloaded LV at 500 ms')
ax.set_box_aspect(hi-lo);ax.view_init(elev=20,azim=-60);ax.legend(loc='upper left',fontsize=9)
for n,(i,j,title) in enumerate([(0,2,'Front: x-z'),(1,2,'Side: y-z'),(0,1,'Top: x-y')],start=2):
    a=fig.add_subplot(2,2,n)
    a.add_patch(plt.Rectangle((lo[i],lo[j]),hi[i]-lo[i],hi[j]-lo[j],fill=False,color='#536779',lw=2))
    # All projected boundary triangles form the silhouette; reference boundary edges are shown as an outline envelope.
    a.add_collection(PolyCollection(x[epi][:,:,[i,j]],facecolors='#81b8d6',edgecolors='none',alpha=1,rasterized=True))
    # Hull is labelled as an envelope, not a geometric surface or cavity section.
    for pts,col,label in [(X,'#7d42a0','Reference LV envelope'),(x,'#267aaa','Preloaded LV envelope')]:
        envelope=hull_vertices(pts[:,[i,j]])
        a.plot(*envelope.T,color=col,lw=1.5,ls='--' if pts is X else '-',label=label)
    a.add_collection(PolyCollection(x[basefaces][:,:,[i,j]],facecolors='#d84a46',edgecolors='#d84a46',linewidths=.3,label='Basal myocardium'))
    a.scatter(center[i],center[j],marker='x',c='black',s=90,zorder=8,label='MAC box center')
    a.scatter(sc[i],sc[j],c='#e29a23',s=45,zorder=8,label='LV bounding-box center')
    a.set(xlim=(lo[i]-.4,hi[i]+.4),ylim=(lo[j]-.4,hi[j]+.4),xlabel='xyz'[i]+' [cm]',ylabel='xyz'[j]+' [cm]',title=title)
    a.set_aspect('equal');a.grid(alpha=.15)
    if n==4:a.legend(loc='lower left',fontsize=8)
fig.suptitle('LV position inside the MAC computational box | 13 x 13 x 13 cm, 128 cubed cells',fontsize=17)
fig.supxlabel('Red: basal tissue (not a closed cap). Orthographic views show projected silhouettes, not cross-sections.\nOriginal box only; retired tract / cap constructions are not shown.',fontsize=11)
fig.savefig(out/'lv_in_mac_domain.png',dpi=180)
fig.savefig(out/'lv_in_mac_domain.pdf',dpi=160)
print(json.dumps(report,indent=2))

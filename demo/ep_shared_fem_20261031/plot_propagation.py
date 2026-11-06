"""Render existing final EP fields only; no solver invocation."""
from pathlib import Path
import json,itertools,tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
R=Path('/mnt/large2/gjh/AFSEI/results');P=R/'ep_shared_fem_20261031';OUT=P/'comparison'
shape=(49,113,321);N=int(np.prod(shape));norm=Normalize(-90,50);cmap=plt.get_cmap('turbo')
folders=[R/'opencarp_gpb_h0p0625_20261030/simulation_v2',P/'h0p0625_consistent_afsei',P/'h0p0625_nodal_afsei']
names=['openCARP reference','AFSEI consistent CG FEM','AFSEI nodal FEM']
fields=[np.memmap(f/'vm.igb',mode='r',dtype='<f4',offset=1024,shape=(51,N)) for f in folders]
x=np.linspace(0,20,321);y=np.linspace(0,7,113);z=np.linspace(0,3,49)
times=[5,10,15,20,30,35]
fig,axes=plt.subplots(3,6,figsize=(19,8.5),layout='constrained')
for row,(name,vm) in enumerate(zip(names,fields)):
    for col,t in enumerate(times):
        ax=axes[row,col];v=vm[t].reshape(shape)[24]
        im=ax.pcolormesh(x,y,v,cmap=cmap,norm=norm,shading='auto',rasterized=True)
        if v.min()<0<v.max():ax.contour(x,y,v,levels=[0],colors='white',linewidths=.9)
        ax.set_aspect('equal');ax.set_title(f'{t} ms',fontsize=10)
        ax.set_xticks([0,10,20]);ax.set_yticks([0,3.5,7])
        if row==2:ax.set_xlabel('x [mm]')
        if col==0:ax.set_ylabel(name+'\ny [mm]',fontsize=9)
fig.colorbar(im,ax=axes,shrink=.8,label='Vm [mV]')
fig.suptitle('Final 3D benchmark propagation | h=0.0625 mm | midplane z=1.5 mm\nSame color scale; white line: Vm=0 mV wavefront | existing 0-50 ms outputs',fontsize=13)
for ext in ('png','pdf'):fig.savefig(OUT/('final_h0p0625_propagation_slices.'+ext),dpi=180)
plt.close(fig)
# Three visible exterior surfaces of the actual 3D cuboid, no projection of solve.
times3=[10,20,30,35];fig=plt.figure(figsize=(16,10),layout='constrained')
for row,(name,vm) in enumerate(zip(names,fields)):
    for col,t in enumerate(times3):
        ax=fig.add_subplot(3,4,row*4+col+1,projection='3d');v=vm[t].reshape(shape)
        xx,yy=np.meshgrid(x[::2],y[::2]);zz=np.full_like(xx,3.)
        ax.plot_surface(xx,yy,zz,facecolors=cmap(norm(v[-1,::2,::2])),rstride=1,cstride=1,shade=False,linewidth=0,antialiased=False)
        xx,zz=np.meshgrid(x[::2],z[::2]);yy=np.zeros_like(xx)
        ax.plot_surface(xx,yy,zz,facecolors=cmap(norm(v[::2,0,::2])),rstride=1,cstride=1,shade=False,linewidth=0,antialiased=False)
        yy,zz=np.meshgrid(y[::2],z[::2]);xx=np.full_like(yy,20.)
        ax.plot_surface(xx,yy,zz,facecolors=cmap(norm(v[::2,::2,-1])),rstride=1,cstride=1,shade=False,linewidth=0,antialiased=False)
        corners=list(itertools.product([0,20],[0,7],[0,3]))
        for a,b in itertools.combinations(corners,2):
            if sum(u!=w for u,w in zip(a,b))==1:ax.plot(*zip(a,b),color='#20252a',lw=.45,alpha=.5)
        ax.set(xlim=(0,20),ylim=(0,7),zlim=(0,3),xticks=[0,10,20],yticks=[0,7],zticks=[0,3]);ax.tick_params(labelsize=7,pad=0)
        ax.set_box_aspect((20,7,3));ax.view_init(elev=24,azim=-62)
        ax.set_title(f'{name}\n{t} ms',fontsize=10,pad=0)
fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=fig.axes,shrink=.7,pad=.02,label='Vm [mV]')
fig.suptitle('Final 3D propagation | 20 x 7 x 3 mm cuboid | h=0.0625 mm\nExterior surfaces, common Vm scale; display every second surface vertex only',fontsize=13)
for ext in ('png','pdf'):fig.savefig(OUT/('final_h0p0625_propagation_3d.'+ext),dpi=180)
plt.close(fig)
old=json.loads((R/'ep_same_solver_cn_20261014/same_solver_report.json').read_text())
current=json.loads((OUT/'report.json').read_text())
audit=dict(old_report=old,current_metrics=current['metrics'],current_runs=current['reports'],
    rendering='Existing fields only. Slice at z1.5mm uses all original 321x113 nodes; 3D exterior rendering uses every second surface vertex, only for visualization.')
(OUT/'propagation_speedup_audit.json').write_text(json.dumps(audit,indent=2))
lines=['# 历史加速比与当前对照的差异','',
'核对的历史原始报告：results/ep_same_solver_cn_20261014/same_solver_report.json。旧统一TorchCor CG核的加速比为0.5mm 2.772倍、0.25mm 1.801倍、0.125mm 1.083倍。因此此前约1.7–1.8倍不能直接外推到当前0.0625mm。','',
'旧测试为5个既有快照（0/2/10/20/40ms）的PDE回放，Iion=0；重复75次，未计ODE，不是完整EP轨迹性能。当前为每组2500个CN步的完整0–50ms实际离子源项轨迹。','',
'旧dt=0.05ms，新dt=0.02ms；旧有效扩散D=(1.171,0.390333,0.390333)cm²/s，新D=(0.952984,0.125758,0.125758)cm²/s。应比较sigma/(chi*Cm)，不能只比较带不同单位缩放的sigma数值。CN矩阵是chi*Cm*M/dt+K/2，其谱和Jacobi迭代还取决于D*dt/h²、网格拓扑及右端；不能简单推断改变D一定令某方法更快。','',
'旧四面体为每立方体6个，当前为openCARP相同的交替5个；以0.125mm为例单元数从1290240变为1075200。此项改变连接/算子结构，虽节点数相同，矩阵并非同一张。','',
'旧0.25mm同核PCG平均迭代14.2对7.0，nodal有明显迭代优势；旧0.125mm为12.2对11.4，优势已缩小。当前AFSEI Graph在0.125mm执行槽数11.704对8.728，在0.0625mm为11.518对11.848。Graph槽数包括段内收敛后的槽位，不应当作和旧实际迭代数完全相同的统计量；但细网格没有明显执行槽数优势。','',
'GPU执行也改变了：两组均应用Graph、缓冲复用、融合和较快的CSR乘法，经典FEM同样受益；小问题上的启动/同步收益不能直接代表大问题上显存带宽主导的性能。当前大部分成本是两种方法共有的稀疏K作用和CG向量操作。','',
'当前0.125mm加速1.295倍，甚至高于旧同尺寸的1.083倍；因此数据不支持“只因为对齐openCARP，nodal优势就被消除”的单一因果解释。要量化各参数的独立贡献，需要固定网格/输入/执行内核做逐因素测试，本轮没有新增求解实验。','',
'最新细网格nodal对参考远端激活仍晚7.523%，CG为-0.304%。历史加速比没有同时证明等精度优势，不应作为nodal在同等精度下更快的证据。']
(OUT/'propagation_speedup_analysis_zh.md').write_text('\n'.join(lines)+'\n')
with tarfile.open(P/'propagation_bundle.tar.gz','w:gz') as tar:
    for pattern in ('final_h0p0625_propagation*','propagation_speedup*'):
        for f in OUT.glob(pattern):tar.add(f,arcname=f.name)
print('\n'.join(lines),flush=True)

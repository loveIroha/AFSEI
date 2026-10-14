"""Plot saved tube fields and controls; never reruns any solver."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path)
    parser.add_argument('--passive',type=Path)
    parser.add_argument('--no-stimulus',type=Path)
    parser.add_argument('--gif',action='store_true')
    args=parser.parse_args(argv)
    out=args.result
    c=json.loads((out/'config.resolved.json').read_text())
    history=json.loads((out/'history.json').read_text())
    data=np.load(out/'snapshots.npz')
    n=int(data['vertex_count'])
    X=data['reference']; lower,upper=data['lower'],data['upper']
    length,height=c['fluid']['lengths_cm']
    times=data['times']*1000
    colors=['#0077b6','#00a896','#e9a23b','#cf4673','#6d28d9']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(13.4,8.2),constrained_layout=True)
    selected=int(np.argmin(abs(times-40)))
    x=data['x'][selected]
    tri=mtri.Triangulation(x[:n,0]*10,x[:n,1]*10,data['cells'][:,:3])
    artist=ax[0,0].tripcolor(tri,data['voltage'][selected],vmin=-90,vmax=40,cmap='turbo',shading='gouraud')
    for nodes in (lower,upper):
        ax[0,0].plot(X[nodes,0]*10,X[nodes,1]*10,'k--',lw=.8,alpha=.5)
    ax[0,0].annotate('Pressure-driven flow',xy=(22,8),xytext=(8,8),
                     arrowprops={'arrowstyle':'->','color':'#0077b6'},color='#0077b6',va='center')
    ax[0,0].set(title=f'EP propagation on moving walls, t = {times[selected]:g} ms',
                xlabel='x (mm)',ylabel='y (mm)',xlim=(0,length*10),ylim=(height*2.2,height*7.8),aspect='equal')
    fig.colorbar(artist,ax=ax[0,0],label='Voltage (mV)',shrink=.7)
    chosen=np.unique(np.linspace(0,len(times)-1,5).round().astype(int))
    for j,color in zip(chosen,colors):
        gap=(data['x'][j,upper,1]-data['x'][j,lower,1])*10
        ax[0,1].plot(X[lower,0]*10,gap,label=f'{times[j]:g} ms',color=color)
    ax[0,1].set(title='Lumen narrowing without prescribed wall motion',xlabel='Reference x (mm)',ylabel='Wall separation (mm)')
    ax[0,1].legend(ncol=3,fontsize=9); ax[0,1].grid(alpha=.2)
    controls=[('EP + Land',history,'#cf4673','-')]
    if args.passive:
        controls.append(('Active force off',json.loads((args.passive/'history.json').read_text()),'#0077b6','--'))
    if args.no_stimulus:
        controls.append(('No electrical stimulus',json.loads((args.no_stimulus/'history.json').read_text()),'#6b7280',':'))
    for name,h,color,style in controls:
        t=np.array([r['time_s'] for r in h])*1000
        ax[1,0].plot(t,[r['minimum_gap_cm']*10 for r in h],style,color=color,label=name,lw=2)
        ax[1,1].plot(t,[r['lumen_mid_flow_cm2_s'] for r in h],style,color=color,label=name,lw=2)
    ax[1,0].set(title='Minimum lumen gap',xlabel='Time (ms)',ylabel='Gap (mm)')
    ax[1,1].set(title='Mid-lumen flow per unit depth (rightward positive)',xlabel='Time (ms)',ylabel=r'Flow ($\mathrm{cm^2/s}$)')
    for a in ax[1]:
        a.axvspan(c['stimulus']['start_s']*1000,(c['stimulus']['start_s']+c['stimulus']['duration_s'])*1000,
                 color='#e9a23b',alpha=.22,label='Stimulus')
        a.grid(alpha=.2); a.legend(fontsize=9)
    ax[1,1].axhline(0,color='k',lw=.7)
    fig.suptitle('AFSEI 2D active tube | nodal FEM + GPB/Land | Neo-Hookean walls | MAC multigrid',fontsize=14)
    fig.savefig(out/'tube_overview.png',dpi=180)
    fig.savefig(out/'tube_overview.pdf')
    plt.close(fig)
    if args.gif:
        fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
        ax.set(xlim=(0,length*10),ylim=(height*2,height*8),xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
        tri=mtri.Triangulation(X[:n,0]*10,X[:n,1]*10,data['cells'][:,:3])
        art=ax.tripcolor(tri,data['voltage'][0],vmin=-90,vmax=40,cmap='turbo',shading='flat')
        fig.colorbar(art,ax=ax,label='Voltage (mV)',shrink=.8)
        for nodes in (lower,upper):
            ax.plot(X[nodes,0]*10,X[nodes,1]*10,'k--',lw=.8,alpha=.4)
        title=ax.set_title('')
        holder=[art]
        def update(j):
            holder[0].remove()
            x=data['x'][j]
            tri=mtri.Triangulation(x[:n,0]*10,x[:n,1]*10,data['cells'][:,:3])
            holder[0]=ax.tripcolor(tri,data['voltage'][j],vmin=-90,vmax=40,cmap='turbo',shading='flat')
            title.set_text(f'EP-driven inward wall motion | t = {times[j]:.0f} ms | displacement scale = 1')
            return holder[0],title
        FuncAnimation(fig,update,frames=len(times),interval=250,blit=False).save(out/'tube_activation.gif',writer=PillowWriter(fps=4))
        plt.close(fig)
    print(out/'tube_overview.png')


if __name__=='__main__':
    main()

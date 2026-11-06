"""Spatial activation, far-corner action potentials and numeric comparison."""
import argparse,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();root=Path(a.folder)
    report=json.loads((root/'report.json').read_text());dt=json.loads((root/'config.json').read_text())['dt_ep_s']
    fig,axes=plt.subplots(2,3,figsize=(14,7.5),layout='constrained');extra=[]
    for row,dim in enumerate([2,3]):
        data=[np.load(root/f'{dim}d'/k/'fields.npz') for k in ['torchcor_cg','afsei_nodal']]
        X=data[0]['X']; plane=np.ones(len(X),dtype=bool) if dim==2 else abs(X[:,2])<1e-12
        values=[z['activation_s'][plane]*1000 for z in data];lo=min(np.nanmin(v) for v in values);hi=max(np.nanmax(v) for v in values)
        nodal_label='AFSEI nodal explicit' if 'explicit' in report.get('method','') else 'AFSEI nodal'
        if 'Jacobi' in report.get('method',''):
            nodal_label='AFSEI nodal CN / Jacobi'
        for col,(z,name) in enumerate(zip(data,['TorchCor CG (CN)',nodal_label])):
            im=axes[row,col].tricontourf(X[plane,0],X[plane,1],values[col],levels=np.linspace(lo,hi,16),cmap='turbo')
            axes[row,col].set(xlabel='x (cm)',ylabel='y (cm)')
            axes[row,col].set_title(f'{dim}D {name}\nActivation (ms)'+(' at z=0' if dim==3 else ''),fontsize=11)
            axes[row,col].set_aspect('equal');fig.colorbar(im,ax=axes[row,col],shrink=.65)
            trace=z['traces_v'];axes[row,2].plot(np.arange(len(trace))*dt*1000,trace[:,1]*1000,label=name)
        axes[row,2].set(xlabel='Time (ms)',ylabel='Voltage (mV)',title=f'{dim}D far corner');axes[row,2].legend()
        diff=data[1]['activation_s']-data[0]['activation_s'];e={'dimension':dim}
        for k,z in zip(['torchcor','nodal'],data):
            e[k+'_far_corner_activation_ms']=float(z['activation_s'][-1]*1000)
            e[k+'_latest_activation_ms']=float(np.nanmax(z['activation_s'])*1000)
        e['activation_relative_l2_difference']=float(np.linalg.norm(diff)/np.linalg.norm(data[0]['activation_s']))
        extra.append(e)
    fig.savefig(root/'activation_and_waveforms.png',dpi=180);plt.close(fig)
    (root/'additional_metrics.json').write_text(json.dumps(extra,indent=2));print(json.dumps(extra,indent=2))


if __name__=='__main__':main()

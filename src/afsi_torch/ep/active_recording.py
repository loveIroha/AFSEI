"""Read-only accepted-load recording. Stress in dyn/cm2, nodal force in dyn.

Inner/middle/outer labels denote material thirds of a Laplace transmural
coordinate (endo=0, epi=1, natural condition on base), not point surface loads.
"""
from pathlib import Path
import csv,json
import numpy as np
import torch
import h5py
from .monodomain import _assemble
from ..fluid.solvers import pcg,SolverOptions


def transmural_coordinate(mesh,geometry):
    cells=geometry.cells;grad=geometry.gradients[:,0];volume=geometry.weights.sum(1)
    local=torch.einsum('e,eai,ebi->eab',volume,grad,grad)
    n=len(mesh.X);K=_assemble(cells,local,n)
    endo=torch.unique(mesh.surface(1));epi=torch.unique(mesh.surface(2))
    if torch.isin(endo,epi).any():raise ValueError('endo/epi nodes overlap')
    free=mesh.X.new_ones(n);free[endo]=0;free[epi]=0
    known=mesh.X.new_zeros(n);known[epi]=1
    mv=lambda v:torch.sparse.mm(K,v[:,None]).flatten()
    diagonal=mesh.X.new_zeros(n).index_add(0,cells.flatten(),local.diagonal(dim1=-2,dim2=-1).flatten())
    rhs=-free*mv(known)
    value,info=pcg(lambda v:free*mv(free*v)+(1-free)*v,rhs,
        free*diagonal+1-free,options=SolverOptions(rtol=1e-10,atol=1e-12,max_iterations=2000,check_every=8))
    if info.residual_norm>info.tolerance:raise ValueError('transmural Laplace solve failed')
    return known+free*value


class ActiveStressRecorder:
    def __init__(self,root,driver,settings):
        self.root=Path(root);self.driver=driver;self.settings=settings;self.last_fields={};self.last_field_tick=-1;self.last_stat_tick=-1
        g=driver.buffer.geometry;self.g=g
        phi=transmural_coordinate(driver.mesh,g)
        self.phi=torch.as_tensor(phi,device=driver.mesh.X.device,dtype=driver.mesh.X.dtype)
        qphi=torch.einsum('qa,ea->eq',g.values,self.phi[g.cells])
        self.masks=[qphi<1/3,(qphi>=1/3)&(qphi<2/3),qphi>=2/3]
        self.names=['endo_side','middle','epi_side']
        self.weights=g.weights
        if any(not bool(m.any()) for m in self.masks):raise ValueError('empty transmural layer')
        self.stream=(self.root/'active_tension_layers.csv').open('a',newline='')
        columns=['time_s','step','phase','beat','load_use_land','load_ramp_factor']
        columns += [f'{name}_{quantity}_kpa' for name in self.names for quantity in ('Ta_mean','Ta_max','active_cauchy_mean','active_cauchy_max')]
        self.writer=csv.DictWriter(self.stream,fieldnames=columns)
        if self.stream.tell()==0:self.writer.writeheader()
        self.h5=h5py.File(self.root/'active_load_history.h5','a')
        if 'reference_X_cm' not in self.h5:
            for name,value in dict(reference_X_cm=driver.mesh.X,cells=g.cells,fiber=driver.mesh.fiber,
                    gradients=g.gradients[:,0],quadrature_values=g.values,reference_weights=g.weights,transmural_phi=self.phi).items():
                self.h5.create_dataset(name,data=value.detach().cpu().numpy(),compression='gzip',compression_opts=1)
            self.h5.attrs['definition']='Ta at accepted force-history quadrature; includes accepted 80ms unload; no reset at plot time'
            self.h5.attrs['layer_definition']='Laplace endo=0 epi=1, thirds; reference-volume weighted statistics'
        (self.root/'active_recording.json').write_text(json.dumps(dict(settings=settings,
            units=dict(Ta='dyn/cm2',force='dyn',stress_csv='kPa'),
            layer_definition='inner/middle/outer thirds of reference Laplace transmural coordinate',
            mean_definition='reference quadrature-volume weighted; actual material-side layers',
            physical_surface_traction=False),indent=2))

    def append(self,name,value):
        value=np.asarray(value)
        if name not in self.h5:
            shape=(0,*value.shape)
            self.h5.create_dataset(name,shape=shape,maxshape=(None,*value.shape),dtype=value.dtype,
                chunks=(1,*value.shape) if value.ndim<2 else (1,min(len(value),2048),*value.shape[1:]),
                compression='gzip',compression_opts=1)
        d=self.h5[name];d.resize(len(d)+1,axis=0);d[-1]=value

    @torch.no_grad()
    def record(self,state,*,field=False,force=False):
        stat_due=force or state.fsi.step%self.settings.get('statistics_every_steps',10)==0
        field_due=field or force or state.fsi.step%self.settings.get('fields_every_steps',100)==0
        if not stat_due and not field_due:return
        if (not stat_due or state.tick==self.last_stat_tick) and (not field_due or state.tick==self.last_field_tick):return
        # No buffer.stage here: record precisely the load used for acceptance,
        # including the closure interval before the GPB/Land history reset.
        ta=self.driver.buffer.tension(state.fsi.x)
        if stat_due and state.tick!=self.last_stat_tick:
            F=torch.einsum('eai,eaJ->eiJ',state.fsi.x[self.g.cells],self.g.gradients[:,0])
            lam2=torch.einsum('eij,ej->ei',F,self.driver.mesh.fiber).square().sum(-1)
            sigma=ta*((1+4.9*(lam2.clamp_min(1).sqrt()-1))*lam2/torch.linalg.det(F))[:,None]
            row=dict(time_s=state.fsi.time,step=state.fsi.step,phase=state.phase,beat=state.beat,
                load_use_land=state.force_load.use_land,load_ramp_factor=state.force_load.ramp_factor)
            values=[]
            for mask in self.masks:
                w=self.weights[mask];a=ta[mask];s=sigma[mask]
                values.extend([(a*w).sum()/w.sum(),a.max(),(s*w).sum()/w.sum(),s.max()])
            vals=(torch.stack(values)/1e4).cpu().tolist()
            row.update(zip(self.writer.fieldnames[6:],vals));self.writer.writerow(row);self.stream.flush()
            self.last_stat_tick=state.tick
        if field_due and state.tick!=self.last_field_tick:
            active_force=self.driver.buffer.force(state.fsi.x)
            self.last_fields=dict(active_tension_mean_dyn_cm2=(ta*self.weights).sum(1)/self.weights.sum(1),
                                  active_force_dyn=active_force)
            for name,value in dict(time_s=np.asarray(state.fsi.time),step=np.asarray(state.fsi.step,dtype=np.int64),
                    Ta_dyn_cm2=ta.cpu().numpy(),x_cm=state.fsi.x.cpu().numpy(),
                    active_force_dyn=active_force.cpu().numpy(),voltage_v=state.tissue.voltage_v.cpu().numpy()).items():
                self.append(name,value)
            self.h5.flush();self.last_field_tick=state.tick

    def close(self):
        self.stream.close();self.h5.close()

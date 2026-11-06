"""Nested EP-only mesh, exact P1 field transfer, and shared nodal diffusion.

The structural mesh, its quadrature, IB and fluid grid are not refined.
Meshes are derived deterministically from checkpointed coarse geometry.
Sparse topology compression is startup CPU work; local FE integration,
time stepping and all field transfers execute with PyTorch on the GPU.
"""
from dataclasses import replace
from itertools import permutations, product
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import torch
from .monodomain import Monodomain
from .numerics import MonodomainNumerics
from ..p1 import prepare_p1


def lattice_template(factor):
    points=np.array([(x,y,z) for x in range(factor+1)
        for y in range(x+1) for z in range(y+1)],dtype=np.int64)
    index={tuple(p):i for i,p in enumerate(points)}
    cells=[]
    for origin in product(range(factor),repeat=3):
        for order in permutations(range(3)):
            v=np.array(origin);path=[v.copy()]
            for axis in order:
                v=v.copy();v[axis]+=1;path.append(v)
            if all(tuple(p) in index for p in path):
                cells.append([index[tuple(p)] for p in path])
    bary=np.column_stack((factor-points[:,0],points[:,0]-points[:,1],
        points[:,1]-points[:,2],points[:,2]))
    cells=np.asarray(cells,dtype=np.int64)
    if len(cells)!=factor**3:raise RuntimeError('invalid edgewise tetra subdivision')
    return bary,cells


def nested_mesh(X,cells,factor):
    """Global vertex ordering makes every shared-face subdivision identical."""
    bary,template=lattice_template(factor)
    sorted_cells=np.sort(cells,axis=1)
    sentinel=np.iinfo(np.int64).max
    key=sorted_cells[:,None,:]*(factor+1)+bary[None,:,:]
    key=np.where(bary[None,:,:]>0,key,sentinel)
    key.sort(axis=2)
    unique,inverse=np.unique(key.reshape(-1,4),axis=0,return_inverse=True)
    present=unique!=sentinel
    ids=np.where(present,unique//(factor+1),0)
    weights=np.where(present,unique%(factor+1),0)/factor
    original=present.sum(1)==1
    remap=np.empty(len(unique),dtype=np.int64)
    remap[original]=ids[original,0]
    remap[~original]=np.arange(len(X),len(unique))
    coarse_ids=np.empty_like(ids);prolong_weights=np.empty_like(weights)
    coarse_ids[remap]=ids;prolong_weights[remap]=weights
    newX=(X[coarse_ids]*prolong_weights[:,:,None]).sum(1)
    local=remap[inverse].reshape(len(cells),len(bary))
    newcells=local[:,template].reshape(-1,4)
    return dict(X=newX,cells=newcells,coarse_ids=coarse_ids,weights=prolong_weights,
        template_bary=bary/factor,template_cells=template,
        permutations=np.argsort(cells,axis=1))


def quadrature_map(data,values):
    """Locate original structural quadrature in a child tetra, analytically."""
    template=data['template_cells'];bary=data['template_bary']
    perms=np.array(list(permutations(range(4))),dtype=np.int64)
    local_points=bary[template,1:]
    D=(local_points[:,1:]-local_points[:,:1]).transpose(0,2,1)
    inverse=np.linalg.inv(D)
    children=[];weights=[]
    for perm in perms:
        target=values[:,perm][:,1:]
        local=np.einsum('tij,tqj->tqi',inverse,target[None,:,:]-local_points[:,:1,:])
        w=np.concatenate((1-local.sum(2,keepdims=True),local),axis=2)
        valid=(w>=-2e-12).all(2)&(w<=1+2e-12).all(2)
        if not valid.any(0).all():raise RuntimeError('structural quadrature outside nested EP mesh')
        child=valid.argmax(0)
        children.append(child);weights.append(w[child,np.arange(len(values))])
    code=lambda a:((a[:,0]*4+a[:,1])*4+a[:,2])*4+a[:,3]
    lookup=np.full(256,-1);lookup[code(perms)]=np.arange(24)
    which=lookup[code(data['permutations'])]
    child=np.asarray(children)[which]
    w=np.asarray(weights)[which]
    fine=data['cells'].reshape(len(which),len(template),4)
    ids=fine[np.arange(len(which))[:,None],child]
    return ids,w


class NestedP1Transfer:
    def __init__(self,data,geometry):
        like=geometry.weights
        self.coarse_nodes=geometry.node_count;self.fine_nodes=len(data['X'])
        self.coarse_ids=torch.as_tensor(data['coarse_ids'],device=like.device)
        self.prolong_weights=torch.as_tensor(data['weights'],device=like.device,dtype=like.dtype)
        ids,weights=quadrature_map(data,geometry.values.cpu().numpy())
        self.quadrature_ids=torch.as_tensor(ids,device=like.device)
        self.quadrature_weights=torch.as_tensor(weights,device=like.device,dtype=like.dtype)

    def prolong(self,value):
        if len(value)==self.fine_nodes:return value
        if len(value)!=self.coarse_nodes:raise ValueError('field is on neither structural nor EP nodes')
        shape=(self.fine_nodes,)+value.shape[1:]
        result=value.new_zeros(shape)
        for j in range(4):
            w=self.prolong_weights[:,j].reshape((self.fine_nodes,)+(1,)*(value.ndim-1))
            result.add_(value[self.coarse_ids[:,j]]*w)
        return result

    def quadrature(self,value):
        if len(value)!=self.fine_nodes:raise ValueError('fine EP nodal field required')
        result=value.new_zeros((*self.quadrature_ids.shape[:2],*value.shape[1:]))
        for j in range(4):
            w=self.quadrature_weights[:,:,j].reshape((*self.quadrature_ids.shape[:2],)+(1,)*(value.ndim-1))
            result.add_(value[self.quadrature_ids[:,:,j]]*w)
        return result


def _assemble_cache(path,data,parameters,chunk=25000):
    if path.exists():return
    X=data['X'];cells=data['cells'];fields=data['fields'];n=len(X);m=len(cells)
    keys=(cells[:,:,None]*n+cells[:,None,:]).reshape(-1)
    order=np.argsort(keys,kind='stable');keys=keys[order]
    starts=np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1]
    unique=keys[starts];del keys
    crow=np.r_[0,np.cumsum(np.bincount(unique//n,minlength=n))]
    col=unique%n;del unique
    raw=np.empty(m*16);volume=np.empty(m)
    xt=torch.as_tensor(X,device='cuda');ft=torch.as_tensor(fields[:,:6],device='cuda')
    gf,gs,gn=parameters.conductivity_us_per_cm
    for lo in range(0,m,chunk):
        ct=torch.as_tensor(cells[lo:lo+chunk],device='cuda')
        g=prepare_p1(xt,ct,degree=1);vol=g.weights.sum(1);grad=g.gradients[:,0]
        f=ft[ct,:3];s=ft[ct,3:6];normal=torch.linalg.cross(f,s)
        outer=lambda v:v[:,:,:,None]*v[:,:,None,:]
        D=(gf*outer(f)+gs*outer(s)+gn*outer(normal)).mean(1)
        local=vol[:,None,None]*torch.einsum('eai,eij,ebj->eab',grad,D,grad)
        raw[lo*16:(lo+len(ct))*16]=local.cpu().numpy().reshape(-1)
        volume[lo:lo+len(ct)]=vol.cpu().numpy()
    K=np.add.reduceat(raw[order],starts);del raw,ft,xt
    localM=volume[:,None,None]/20*(np.ones((4,4))+np.eye(4))
    M=np.add.reduceat(localM.reshape(-1)[order],starts)
    del localM,order,starts
    lump=np.bincount(cells.reshape(-1),weights=np.repeat(volume/4,4),minlength=n)
    rows=np.repeat(np.arange(n),np.diff(crow));kd=K[rows==col]
    temporary=path.with_suffix('.working.npz')
    np.savez(temporary,crow=crow,col=col,K=K,M=M,lump=lump,kd=kd)
    temporary.replace(path)


class RefinedMonodomain:
    """Fine EP solver with the ORIGINAL coarse consistent strain projection."""
    def __init__(self,mesh,config,fiber,sheet,stimulus,geometry):
        if config.ep_numerics.diffusion_quadrature!='nodal_p1':
            raise ValueError('refined EP currently requires explicit nodal diffusion quadrature')
        factor=config.ep_refinement_factor
        h=hashlib.sha256(f'nested-ep-edgewise-v1-{factor}'.encode())
        arrays=[a.detach().cpu().numpy() for a in (mesh.X,mesh.cells,fiber,sheet,stimulus)]
        for array in arrays:h.update(np.ascontiguousarray(array).tobytes())
        h.update(json.dumps(config.ep.__dict__,sort_keys=True).encode())
        base=Path(config.ep_refinement_cache or Path(__file__).resolve().parents[3]/'.cache/ep_refinement')
        self.cache=base/h.hexdigest()[:20];self.cache.mkdir(parents=True,exist_ok=True)
        path=self.cache/'mesh.npz'
        if not path.exists():
            data=nested_mesh(arrays[0],arrays[1],factor)
            coarse_fields=np.column_stack((arrays[2],arrays[3],arrays[4].astype(float)))
            data['fields']=(coarse_fields[data['coarse_ids']]*data['weights'][:,:,None]).sum(1)
            temporary=path.with_suffix('.working.npz');np.savez(temporary,**data);temporary.replace(path)
        with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        self.transfer=NestedP1Transfer(data,geometry)
        _assemble_cache(self.cache/'operators.npz',data,config.ep)
        p=Monodomain.__new__(Monodomain)
        p.X=torch.as_tensor(data['X'],device=mesh.X.device)
        p.cells=torch.as_tensor(data['cells'],device=mesh.X.device)
        p.parameters=config.ep;p.numerics=config.ep_numerics;p.solver_options=p.numerics.solver;p._systems={}
        with np.load(self.cache/'operators.npz',allow_pickle=False) as z:
            crow=torch.as_tensor(z['crow'],device=mesh.X.device)
            col=torch.as_tensor(z['col'],device=mesh.X.device)
            p.stiffness=torch.sparse_csr_tensor(crow,col,torch.as_tensor(z['K'],device=mesh.X.device),size=(len(p.X),)*2)
            p.stiffness_diagonal=torch.as_tensor(z['kd'],device=mesh.X.device)
            lump=torch.as_tensor(z['lump'],device=mesh.X.device)
            if p.numerics.discretization=='consistent_p1':
                p.mass=torch.sparse_csr_tensor(crow,col,torch.as_tensor(z['M'],device=mesh.X.device),size=(len(p.X),)*2)
                p.mass_diagonal=lump*.4
            else:
                p.mass_diagonal=lump
                p.mass=torch.sparse_csr_tensor(torch.arange(len(p.X)+1,device=mesh.X.device),
                    torch.arange(len(p.X),device=mesh.X.device),lump,size=(len(p.X),)*2)
        self.operator=p
        # This projector exactly retains the original L2 mechanical feedback.
        coarse_numerics=replace(config.ep_numerics,discretization='consistent_p1',backend='torch_pcg_jacobi')
        self.coarse=Monodomain(mesh.X,mesh.cells,fiber,sheet,directions='nodal',
            parameters=config.ep,numerics=coarse_numerics)
        self.stimulus=torch.as_tensor(data['fields'][:,6],device=mesh.X.device)
        self.metadata=dict(factor=factor,ep_nodes=len(p.X),ep_cells=len(p.cells),
            structural_nodes=len(mesh.X),structural_cells=len(mesh.cells),cache=str(self.cache),
            transfer='nested-P1-to-structural-quadrature',diffusion_quadrature='nodal_p1')
        (self.cache/'metadata.json').write_text(json.dumps(self.metadata,indent=2)+'\n')

    def __getattr__(self,name):return getattr(self.operator,name)
    def project_cell_field(self,value):return self.coarse.project_cell_field(value)
    def close(self):self.operator.close();self.coarse.close()

"""Conservative fixed-cavity MAC pressure/viscosity operators.

Full-cell staircase walls have zero normal velocity and odd tangential ghosts.
Open ports impose pressure; their normal velocity has a half-cell dual mass.
The scalar graph viscosity is dissipative, with homogeneous tangential ghosts
at port edges (this prototype does not impose a full viscous traction outlet).
"""
import torch
from ..mac.grid import slab, divergence
from ..fluid.solvers import pcg


def sparse_graph(shape, diagonal, pairs):
    index=torch.arange(diagonal.numel(),device=diagonal.device).reshape(shape)
    rows=[index.flatten()]; cols=[index.flatten()]; values=[diagonal.flatten()]
    for a,b,w in pairs:
        rows.extend((a,b)); cols.extend((b,a)); values.extend((-w,-w))
    return torch.sparse_coo_tensor(torch.stack((torch.cat(rows),torch.cat(cols))),
        torch.cat(values),(index.numel(),index.numel()),check_invariants=True).coalesce().to_sparse_csr()


class CavityOperators:
    def __init__(self,domain,open_ports):
        self.domain=domain; self.grid=domain.grid; self.open_ports=open_ports
        left,right=domain.adjacent(); self.left=left; self.right=right
        self.sign=tuple(l.to(torch.float64)-r.to(torch.float64) for l,r in zip(left,right))
        self.potential=tuple(torch.where(l&r,1.,torch.where(t>0,.5,0.)).to(torch.float64)
            for l,r,t in zip(left,right,domain.tags))
        self.active=tuple((l&r)|((t>0)&open_ports) for l,r,t in zip(left,right,domain.tags))
        self.mass=tuple(m*a for m,a in zip(self.potential,self.active))
        device=domain.fluid.device; shape=self.grid.shape
        index=torch.arange(domain.fluid.numel(),device=device).reshape(shape)
        diagonal=torch.zeros(shape,device=device,dtype=torch.float64); pairs=[]
        for c,h in enumerate(self.grid.spacing):
            a=index[slab(c,None,-1)]; b=index[slab(c,1,None)]
            valid=domain.fluid[slab(c,None,-1)]&domain.fluid[slab(c,1,None)]
            w=torch.full_like(a,1/h**2,dtype=torch.float64)[valid]
            pairs.append((a[valid],b[valid],w))
            diagonal.flatten().index_add_(0,a[valid],w); diagonal.flatten().index_add_(0,b[valid],w)
            if open_ports:
                port=(domain.tags[c]>0).to(torch.float64)*2/h**2
                diagonal+=port[slab(c,1,None)]*left[c][slab(c,1,None)]
                diagonal+=port[slab(c,None,-1)]*right[c][slab(c,None,-1)]
        self.fixed=~domain.fluid.clone()
        if not open_ports:
            first=domain.fluid.flatten().nonzero()[0,0]; self.fixed.flatten()[first]=True
        # Any zero diagonal here belongs to an exterior row, replaced by I.
        self.pressure_diagonal=diagonal.masked_fill(~domain.fluid,1.)
        self.pressure_matrix=sparse_graph(shape,self.pressure_diagonal,pairs)
        self.viscosity=[]; self.viscosity_diagonal=[]
        for c,active in enumerate(self.active):
            mass=self.mass[c]; index=torch.arange(active.numel(),device=device).reshape(active.shape)
            diag=torch.zeros_like(mass); pairs=[]
            for axis,h in enumerate(self.grid.spacing):
                lo=slab(axis,None,-1); hi=slab(axis,1,None)
                a=index[lo]; b=index[hi]; both=active[lo]&active[hi]
                coefficient=(torch.ones_like(mass[lo]) if axis==c else torch.minimum(mass[lo],mass[hi]))/h**2
                w=coefficient[both]; pairs.append((a[both],b[both],w))
                diag.flatten().index_add_(0,a[both],w); diag.flatten().index_add_(0,b[both],w)
                factor=1. if axis==c else 2.
                diag[lo]+=factor*mass[lo]*(active[lo]&~active[hi])/h**2
                diag[hi]+=factor*mass[hi]*(active[hi]&~active[lo])/h**2
                for boundary in (slab(axis,0,1),slab(axis,-1,None)):
                    # Normal port faces have zero outward derivative. Other
                    # exterior ghosts are no-slip; their approximation is
                    # explicitly included in the energy diagnostic.
                    free=(domain.tags[c][boundary]>0)&open_ports if axis==c else torch.zeros_like(active[boundary])
                    diag[boundary]+=2*mass[boundary]*(active[boundary]&~free)/h**2
            self.viscosity_diagonal.append(diag)
            self.viscosity.append(sparse_graph(active.shape,diag,pairs))

    @staticmethod
    def apply(matrix,x): return torch.sparse.mm(matrix,x.reshape(-1,1)).reshape(x.shape)

    def divergence(self,u): return divergence(u,self.grid.spacing).masked_fill(~self.domain.fluid,0.)

    def gradient(self,p,b):
        result=[]
        for c,h in enumerate(self.grid.spacing):
            z=p.new_zeros(tuple(1 if a==c else n for a,n in enumerate(p.shape)))
            left=torch.cat((z,p),c); right=torch.cat((p,z),c)
            g=(right-left)/h
            t=self.domain.tags[c]; port=(t>0)&self.open_ports
            boundary=b[(t-1).clamp_min(0)]
            inside=torch.where(self.left[c],left,right)
            g=torch.where(port,2*self.sign[c]*(boundary-inside)/h,g)
            result.append(g.masked_fill(~self.active[c],0.))
        return tuple(result)

    def flux(self,u):
        return torch.stack([sum((v*s*(t==label)).sum()*self.grid.volume/h
            for v,s,t,h in zip(u,self.sign,self.domain.tags,self.grid.spacing)) for label in (1,2)])

    def norm2(self,u): return sum((m*v.square()).sum()*self.grid.volume for m,v in zip(self.potential,u))

    def viscous(self,old,dt,nu,options):
        result=[]; infos=[]
        for v,m,active,K,kdiag in zip(old,self.mass,self.active,self.viscosity,self.viscosity_diagonal):
            diag=(m+dt*nu*kdiag).masked_fill(~active,1.)
            action=lambda x,m=m,K=K,active=active: (m*x+dt*nu*self.apply(K,x)).masked_fill(~active,0.)
            out,info=pcg(action,m*v,diag,fixed=~active,options=options)
            result.append(out); infos.append(info)
        return tuple(result),infos

    def project(self,u,b,dt,rho,options):
        zero=torch.zeros_like(self.domain.fluid,dtype=torch.float64)
        rhs=-rho/dt*self.divergence(u)+self.divergence(self.gradient(zero,b))
        p,info=pcg(lambda x:self.apply(self.pressure_matrix,x),rhs,self.pressure_diagonal,
            fixed=self.fixed,options=options)
        g=self.gradient(p,b)
        return tuple((v-dt/rho*w).masked_fill(~a,0.) for v,w,a in zip(u,g,self.active)),p,info

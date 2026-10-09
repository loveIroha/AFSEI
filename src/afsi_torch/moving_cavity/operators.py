"""Mass-lumped CR velocity, conservative swept-face divergence, P0 pressure."""
import torch
from ..fluid.solvers import pcg


def csr(indices,values,size):
    return torch.sparse_coo_tensor(indices,values,size,check_invariants=True).coalesce().to_sparse_csr()


class ALEOperators:
    def __init__(self,mesh,geometry,opened):
        self.mesh=mesh; self.geometry=geometry; self.opened=opened
        t=mesh.tags; self.fixed=(t==0)|((t==1)&(not opened[0]))|((t==2)&(not opened[1]))
        self.mass=geometry.volumes.new_zeros(len(mesh.faces)).index_add(0,mesh.cell_faces.flatten(),
            (geometry.volumes[:,None].expand(-1,4)/4).flatten())
        local=geometry.volumes[:,None,None]*torch.einsum('eai,ebi->eab',geometry.gradients,geometry.gradients)
        rows=mesh.cell_faces[:,:,None].expand(-1,4,4); cols=mesh.cell_faces[:,None,:].expand(-1,4,4)
        self.K=csr(torch.stack((rows.flatten(),cols.flatten())),local.flatten(),(len(t),len(t)))
        self.kdiag=self.mass.new_zeros(len(t)).index_add(0,mesh.cell_faces.flatten(),local.diagonal(dim1=1,dim2=2).flatten())
        self.B=geometry.areas[mesh.cell_faces]*mesh.signs[:,:,None]
        self.inverse_mass=torch.where(self.fixed,0.,1/self.mass)
        conductance=geometry.areas.square().sum(1)*self.inverse_mass
        self.conductance=conductance
        diag=self.mass.new_zeros(len(mesh.cells)); diag.index_add_(0,mesh.left,conductance)
        inside=mesh.right>=0; diag.index_add_(0,mesh.right[inside],conductance[inside])
        n=len(mesh.cells); index=torch.arange(n,device=t.device)
        a=mesh.left[inside]; b=mesh.right[inside]; w=conductance[inside]
        self.A=csr(torch.stack((torch.cat((index,a,b)),torch.cat((index,b,a)))),torch.cat((diag,-w,-w)),(n,n))
        self.pdiag=diag; self.pressure_fixed=torch.zeros(n,device=t.device,dtype=torch.bool)
        if not any(opened): self.pressure_fixed[0]=True
        # Small ports weakly anchor the nearly constant pressure mode. Add an
        # SPD one-dimensional coarse inverse to Jacobi, not a residual fixup.
        self.coarse=diag.new_ones(n).masked_fill(self.pressure_fixed,0.)
        self.coarse_energy=(self.coarse*self.pressure_action(self.coarse)).sum()
        if self.coarse_energy<=0: raise ValueError('pressure coarse inverse must have positive energy')

    @staticmethod
    def apply(A,x):
        shape=x.shape; return torch.sparse.mm(A,x.reshape(shape[0],-1)).reshape(shape)

    def divergence(self,u): return (self.B*u[self.mesh.cell_faces]).sum((1,2))

    def transpose(self,p):
        # Form pressure differences before multiplying by face area. Summing
        # two large p*A terms separately loses accuracy near a constant mode.
        jump=p[self.mesh.left].clone(); interior=self.mesh.right>=0
        jump[interior]-=p[self.mesh.right[interior]]
        return self.geometry.areas*jump[:,None]

    def pressure_action(self,p):
        interior=self.mesh.right>=0; jump=p[self.mesh.left].clone()
        jump[interior]-=p[self.mesh.right[interior]]
        flux=self.conductance*jump
        out=torch.zeros_like(p).index_add(0,self.mesh.left,flux)
        out.index_add_(0,self.mesh.right[interior],-flux[interior])
        return out

    def port_load(self,b):
        t=self.mesh.tags; load=self.geometry.areas*b[(t-1).clamp(0,1),None]
        active=((t==1)&self.opened[0])|((t==2)&self.opened[1])
        return load.masked_fill(~active[:,None],0.)

    def flux(self,u,relative=True):
        q=(u*self.geometry.areas).sum(1)
        if relative: q=q-self.geometry.grid_flux
        return torch.stack([q[self.mesh.tags==label].sum() for label in (1,2)])

    def convection(self,u):
        # CR mean value is the average of its four facet values.
        cells=u[self.mesh.cell_faces]; adv=(cells-self.geometry.grid_velocity[self.mesh.cell_faces]).mean(1)
        gradient=torch.einsum('eai,eaj->eij',cells,self.geometry.gradients)
        value=torch.einsum('eij,ej->ei',gradient,adv)
        local=self.geometry.volumes[:,None,None]*value[:,None,:].expand(-1,4,-1)/4
        return u.new_zeros(u.shape).index_add(0,self.mesh.cell_faces.flatten(),local.reshape(-1,3))

    def viscous(self,old,dt,rho,mu,options,convection):
        rhs=self.mass[:,None]*old
        if convection: rhs=rhs-dt*self.convection(old)
        diagonal=self.mass+dt*mu/rho*self.kdiag
        H=lambda x:self.mass[:,None]*x+dt*mu/rho*self.apply(self.K,x)
        fixed=self.fixed[:,None].expand(-1,3); values=self.geometry.grid_velocity
        u,info=pcg(H,rhs,diagonal[:,None].expand(-1,3),fixed=fixed,values=values,initial=old,options=options)
        return u,rhs,info

    def project(self,star,b,dt,rho,options):
        L=self.port_load(b)
        rhs=-rho/dt*self.divergence(star)+self.divergence(self.inverse_mass[:,None]*L)
        # Closed trial compatibility is checked BEFORE any pressure gauge
        # elimination. No pressure row is permitted to act as a hidden leak.
        if not any(self.opened) and rhs.sum().abs()>max(1e-8,1e-10*rhs.abs().sum().item()):
            raise ValueError('closed moving cavity changes volume; two-way solid volume constraint is required')
        precondition=lambda r:r/self.pdiag+self.coarse*(self.coarse*r).sum()/self.coarse_energy
        p,info=pcg(self.pressure_action,rhs,self.pdiag,fixed=self.pressure_fixed,options=options,preconditioner=precondition)
        u=star+dt/rho*self.inverse_mass[:,None]*(self.transpose(p)-L)
        return u,p,info

    def reaction(self,u,p,star,rhs,dt,rho,mu,b):
        return rho/dt*(self.mass[:,None]*u-rhs)+mu*self.apply(self.K,star)-self.transpose(p)+self.port_load(b)

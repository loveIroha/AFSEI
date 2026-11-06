"""Physical basal support reactions in the original implicit BE/IB equations.

Unknowns are displacement y and two support reactions per basal vertex.
R=y-dt J H(S(F(X+y)+B^T lambda)), B(X+y-Xref)=0.
This adds a mechanical support, not a cavity cap or volume constraint. The
original spread/gather pair and fluid solver are called without alteration.
"""
import torch
from .paper_coupling import BEProblem
from ..nonlinear import normalized_linear_action


class BasalSupport:
    def __init__(self,X,faces,center_xy):
        self.reference=X
        self.nodes=torch.unique(faces)
        r=X[self.nodes,:2]-X.new_tensor(center_xy)
        norm=r.norm(dim=-1)
        if bool((norm<=1e-12).any()):raise ValueError('undefined basal radial direction')
        r=r/norm[:,None]
        self.theta=torch.stack((-r[:,1],r[:,0],torch.zeros_like(r[:,0])),-1)
        self.axis=torch.zeros_like(self.theta);self.axis[:,2]=1.

    def apply(self,u):
        v=u[self.nodes]
        return torch.stack(((v*self.theta).sum(-1),v[:,2]),-1)

    def transpose(self,lam):
        out=torch.zeros_like(self.reference)
        return out.index_add(0,self.nodes,lam[:,:1]*self.theta+lam[:,1:]*self.axis)

    def error(self,x):return self.apply(x-self.reference)


class SupportedBEProblem(BEProblem):
    def __init__(self,driver,state,stencil,departure):
        super().__init__(driver,state,stencil,departure)
        self.support=driver.basal_support
        if driver.config.nonlinear_solver!='newton':
            raise ValueError('physical basal support currently requires assembled Newton')
        if float(driver.model.beta)!=0.:
            raise ValueError('disable the old basal penalty when enabling exact support')
        if float(self.support.error(state.x).abs().max())>driver.support_tolerance_cm:
            raise ValueError('initial state violates basal support; start a fresh preload, do not project a checkpoint')
        # Reaction variable has displacement units; this is numerical scaling,
        # not a spring stiffness. Physical reactions are returned in dyn.
        cells=driver.model.mesh.cells;X=driver.model.mesh.X
        p=X[cells];vol=torch.linalg.det(torch.stack((p[:,1]-p[:,0],p[:,2]-p[:,0],p[:,3]-p[:,0]),-1)).abs()/6
        m=X.new_zeros(len(X)).index_add(0,cells.reshape(-1),(vol[:,None]/4).expand(-1,4).reshape(-1))
        self.scale=(driver.flow.rho*m[self.support.nodes]/driver.flow.dt**2)[:,None]
        self.n=state.x.numel()

    def split(self,z):return z[:self.n].reshape_as(self.state.x),z[self.n:].reshape(-1,2)

    def initial(self,y):return torch.cat((y.reshape(-1),y.new_zeros(len(self.support.nodes)*2)))

    def validate(self,z):self.driver.solid.validate(self.state.x+self.split(z)[0])

    def residual(self,z):
        y,eta=self.split(z);x=self.state.x+y;d=self.driver
        d.solid.validate(x)
        reaction=self.support.transpose(self.scale*eta)
        force=d.solid.force(x,self.time)+reaction
        d.transfer.reset_warm_start()
        density,spread=d.transfer.spread(force,self.stencil)
        u,p,flow=d.flow.advance(self.departure,density,self.pressure_initial)
        nodal,interpolation=d.transfer.interpolate(u,self.stencil)
        r=torch.cat(((y-d.flow.dt*nodal).reshape(-1),self.support.error(x).reshape(-1)))
        if not bool(torch.isfinite(r).all()):raise FloatingPointError('nonfinite supported residual')
        self.last_z=z.clone();self.last_residual=r;self.last_x=x
        self.last_reaction=reaction
        self.last_data=force,u,p,flow,nodal,density,spread,interpolation
        self.pressure_initial=p;self.evaluations+=1
        return r

    def accepted(self,z):
        if not hasattr(self,'last_z') or not torch.equal(z,self.last_z):self.residual(z)
        error=float(self.support.error(self.last_x).abs().max())
        if error>self.driver.support_tolerance_cm:
            raise RuntimeError('physical basal support exceeds acceptance tolerance')
        self.driver.last_support_report=dict(max_constraint_cm=error,
            reaction_dyn=self.last_reaction.sum(0).detach().cpu().tolist(),
            support_power_erg_s=float((self.last_reaction*self.last_data[4]).sum()),
            kind='physical basal nodal support; no cavity volume constraint')
        return self.last_residual,self.last_x

    def linearization(self,z):
        d=self.driver;y,_=self.split(z)
        if d.assembler is None:d.assembler=d.model.tangent_factory()
        tangent=d.assembler.assemble(self.state.x+y,self.time);self.tangent_assemblies+=1
        def action(v):
            self.jacobian_actions+=1
            dy,de=self.split(v)
            df=torch.sparse.mm(tangent,dy.reshape(-1,1)).reshape_as(dy)+self.support.transpose(self.scale*de)
            d.transfer.reset_warm_start()
            density,_=d.transfer.spread(df,self.stencil)
            du=d.flow.response(density)
            nodal,_=d.transfer.interpolate(du,self.stencil)
            return torch.cat(((dy-d.flow.dt*nodal).reshape(-1),self.support.apply(dy).reshape(-1)))
        return normalized_linear_action(action)

    def preconditioner(self,z):return None


def enable_basal_support(driver,*,tolerance_cm=1e-9):
    """Explicit opt-in; original model must already have beta=0."""
    if tolerance_cm<=0:raise ValueError('positive support tolerance required')
    if float(driver.model.beta)!=0.:raise ValueError('set beta=0 before constructing the solid/driver')
    if driver.config.newton_preconditioner!='none':
        raise ValueError('augmented basal support requires its own preconditioner; select none for validation')
    driver.basal_support=BasalSupport(driver.model.mesh.X,driver.model.base.faces,driver.config.basal_center_cm)
    driver.support_tolerance_cm=tolerance_cm
    driver.problem_factory=SupportedBEProblem
    return driver

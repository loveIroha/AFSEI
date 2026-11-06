"""Conforming passive connector added to an unchanged EP-driven myocardium.

Original myocardial stress, Land quadrature and basal penalty are retained.
Only added cells use Neo-Hookean stress; EP never runs in those added cells.
"""
from types import SimpleNamespace
import torch
from ..solids.p1_model import P1Solid, BoundaryForce
from ..materials import neo_hookean_pk1
from ..paper_lv import paper_pk1
from .land import active_pk1


class ConnectedLVSolid(P1Solid):
    def __init__(self,original,housing,*,young_modulus_dyn_cm2=1e5,poisson_ratio=.45,
                 fixed_end_penalty_dyn_cm3=1e8):
        if young_modulus_dyn_cm2<=0 or not 0<=poisson_ratio<.5 or fixed_end_penalty_dyn_cm3<=0:
            raise ValueError('positive connector modulus/penalty and 0<=nu<0.5 required')
        self.original=original;self.original_nodes=len(original.mesh.X);self.original_cells=len(original.mesh.cells)
        like=original.mesh.X;tensor=lambda a:torch.as_tensor(a,device=like.device)
        X=tensor(housing['X']).to(like.dtype);newcells=tensor(housing['cells']).long()
        if not torch.equal(X[:self.original_nodes],like):raise ValueError('original myocardium nodes changed')
        extra=len(newcells);f=torch.cat((original.mesh.fiber,like.new_tensor([1.,0,0]).expand(extra,3)))
        s=torch.cat((original.mesh.sheet,like.new_tensor([0.,1,0]).expand(extra,3)))
        mesh=SimpleNamespace(X=X,cells=torch.cat((original.mesh.cells,newcells)),fiber=f,sheet=s)
        self.buffer=original.active_load;self.pressure=like.new_zeros((len(original.endo_faces),1))
        self.mu=young_modulus_dyn_cm2/(2*(1+poisson_ratio))
        self.lame=young_modulus_dyn_cm2*poisson_ratio/((1+poisson_ratio)*(1-2*poisson_ratio))
        # Material IDs are tensor fields; the original active buffers remain
        # owned by the EP model and are copied into the shared cell table.
        nq=self.buffer.anchor.shape[1];count=len(mesh.cells)
        self.oldF=like.new_zeros(count,3,3);self.oldF.copy_(torch.eye(3,device=like.device,dtype=like.dtype))
        self.land=like.new_zeros(count,nq,4);self.anchor=like.new_zeros(count,nq)
        weights=torch.cat((self.buffer.geometry.weights,
                          like.new_ones(extra,nq)),0)
        weights=weights/weights.sum(1,keepdim=True)
        myocardial=torch.cat((like.new_ones(self.original_cells),like.new_zeros(extra)))
        fields=(f,s,myocardial,self.oldF,self.land,self.anchor,weights)
        def stress(F,values,time):
            fiber,sheet,region,oldF,land,anchor,w=values
            # Do not evaluate myocardial exponentials at housing strains.
            Fm=region*F+(1-region)*torch.eye(3,device=F.device,dtype=F.dtype)
            P=paper_pk1(Fm,fiber,sheet,original.parameters,0.)
            ta=self.buffer._tension(Fm,fiber,oldF,land,anchor)
            P=P+active_pk1(Fm,fiber,(w*ta).sum())
            passive=neo_hookean_pk1(F,self.mu,self.lame)
            return region*P+(1-region)*passive
        base=original.base;N=base.values
        def basal(nodes,values,time):
            ref,radial,weights=values
            u=N@nodes-ref
            xy=(u[:,:2]*radial).sum(-1,keepdim=True)*radial-u[:,:2]
            reaction=torch.cat((xy,-u[:,2:]),-1)
            return original.beta*torch.einsum('q,qa,qi->ai',weights,N,reaction)
        boundaries=[BoundaryForce(base.faces,basal,(base.reference_positions,original.radial,base.reference_weights))]
        def pressure(nodes,values,time):
            area=.5*torch.linalg.cross(nodes[1]-nodes[0],nodes[2]-nodes[0])
            return (-values[0][0]*area/3).expand(3,3)
        boundaries.append(BoundaryForce(original.endo_faces,pressure,(self.pressure,)))
        end=torch.cat((tensor(housing['fixed_end_0']),tensor(housing['fixed_end_1']))).long()
        ref=X[end].clone();area=.5*torch.linalg.cross(ref[:,1]-ref[:,0],ref[:,2]-ref[:,0]).norm(dim=-1)
        def tether(nodes,values,time):
            ref,area=values;u=nodes-ref
            return -fixed_end_penalty_dyn_cm3*area/12*(u+u.sum(0))
        boundaries.append(BoundaryForce(end,tether,(ref,area)))
        super().__init__(mesh,stress,cell_fields=fields,boundaries=boundaries,degree=5)
        self.fixed_end_faces=end;self.endo_faces=original.endo_faces
        self.refresh()

    @torch.no_grad()
    def refresh(self,pressure_mmhg=0.):
        from .windkessel import MMHG_TO_DYN_CM2
        n=self.original_cells
        self.oldF[:n].copy_(self.buffer.previous_F);self.land[:n].copy_(self.buffer.fields);self.anchor[:n].copy_(self.buffer.anchor)
        self.pressure.fill_(pressure_mmhg*MMHG_TO_DYN_CM2)

    def diagnostics(self,x):
        report=super().diagnostics(x)
        report.update(myocardium=self.original.diagnostics(x[:self.original_nodes]),
            connector_volume_cm3=float((torch.linalg.det(self.element_gradient(x)[self.original_cells:])*self.volumes[self.original_cells:]).sum()))
        return report

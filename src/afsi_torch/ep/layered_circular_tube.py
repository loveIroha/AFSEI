"""Two-layer heart tube; existing P1 assembly, Land law, and MAC/IB unchanged.

Cardiac jelly is passive and non-excitable. EP/GPB/Land DOFs cover myocardium
only, sharing the interface mechanically. Layer constants are Neo-Hookean
shear moduli, not Young's moduli. Finite full-vector end springs approximate
fixed ends; they are explicitly not exact Dirichlet conditions.
"""
from math import isfinite
import torch
from .circular_tube import CircularTubeSolid
from .land import active_pk1
from ..materials import neo_hookean_pk1
from ..solids.p1_model import BoundaryForce


def validate_layers(layers):
    if not isinstance(layers,list) or len(layers) != 2:
        raise ValueError('exactly two radial materials: cardiac-jelly, myocardium')
    for i,layer in enumerate(layers):
        if set(layer) != {'name','shear_modulus_dyn_cm2','poisson_ratio','active'}:
            raise ValueError('layer requires name, shear_modulus_dyn_cm2, poisson_ratio, active')
        if layer['name'] != ('cardiac-jelly','myocardium')[i] or layer['active'] is not bool(i):
            raise ValueError('only the outer myocardium is active; jelly is passive')
        mu,nu=layer['shear_modulus_dyn_cm2'],layer['poisson_ratio']
        if (isinstance(mu,bool) or not isfinite(mu) or mu <= 0 or
                isinstance(nu,bool) or not isfinite(nu) or not 0 <= nu < .5):
            raise ValueError('positive shear modulus and 0 <= nu < 0.5 required per layer')


def myocardium_ep_mesh(mesh):
    cells=torch.where(mesh.cell_layer == 1)[0]
    nodes=torch.unique(mesh.cells[cells])
    inverse=torch.full((len(mesh.X),),-1,device=mesh.X.device,dtype=torch.long)
    inverse[nodes]=torch.arange(len(nodes),device=mesh.X.device)
    return nodes,cells,inverse[mesh.cells[cells]]


class LayeredCircularTubeSolid(CircularTubeSolid):
    def __init__(self,mesh,geometry,*,layers,end_spring_dyn_cm3,**active):
        validate_layers(layers)
        if len(geometry.radial_elements_per_layer) != 2:
            raise ValueError('two resolved radial layers are required')
        myo=layers[1];mu=myo['shear_modulus_dyn_cm2'];nu=myo['poisson_ratio']
        super().__init__(mesh,geometry,young_modulus_dyn_cm2=2*mu*(1+nu),poisson_ratio=nu,
                         end_spring_dyn_cm3=0.,**active)
        self.ep_nodes,self.ep_cell_ids,self.ep_cells=myocardium_ep_mesh(mesh)
        self.layers=layers
        mu=mesh.X.new_tensor([v['shear_modulus_dyn_cm2'] for v in layers])[mesh.cell_layer]
        nu=mesh.X.new_tensor([v['poisson_ratio'] for v in layers])[mesh.cell_layer]
        lam=2*mu*nu/(1-2*nu)
        self.taper.mul_((mesh.cell_layer == 1)[:,None])
        self.cell_fields=(*self.cell_fields,mu,lam)

        def stress(F,values,time):
            fiber,previous,land,reference,taper,weights,mu,lam=values
            ta=self._tension(F,fiber,previous,land,reference)*taper*self.scale
            return neo_hookean_pk1(F,mu,lam)+active_pk1(F,fiber,(weights*ta).sum())
        self.stress=stress
        # Full vector end springs reuse the original FE boundary integration.
        reference=mesh.X[mesh.end_faces].clone()
        area=.5*torch.linalg.vector_norm(torch.linalg.cross(
            reference[:,1]-reference[:,0],reference[:,2]-reference[:,0]),dim=-1)
        def spring(nodes,values,time):
            ref,face_area=values
            u=nodes-ref
            return -end_spring_dyn_cm3*face_area/12*(u+u.sum(0))
        self.boundaries=(BoundaryForce(mesh.end_faces,spring,(reference,area)),)
        self.end_nodes=torch.unique(mesh.end_faces)

    def _lift(self,local):
        return local.new_zeros((len(self.mesh.X),*local.shape[1:])).index_copy(0,self.ep_nodes,local)

    @torch.no_grad()
    def set_reference(self,nodal_tension):
        super().set_reference(self._lift(nodal_tension))

    @torch.no_grad()
    def stage(self,previous_x,tissue,zs,zw):
        self.previous_F.copy_(self.element_gradient(previous_x))
        nodal=self._lift(torch.stack((tissue.land[:,0],tissue.land[:,1],zs,zw),-1))
        self.land_fields.copy_(torch.einsum('qa,eak->eqk',self.geometry.values,nodal[self.mesh.cells]))

    def nodal_stretch(self,ep,x):
        value=ep.project_cell_field(self.cell_stretch(x)[self.ep_cell_ids])
        if not torch.isfinite(value).all() or (value <= 0).any():
            raise ValueError('nonpositive myocardium stretch')
        return value

    def diagnostics(self,x):
        r=super().diagnostics(x)
        F=self.element_gradient(x)
        J=torch.linalg.det(F)
        for i,layer in enumerate(self.layers):
            selected=self.mesh.cell_layer == i
            r[layer['name']+'_volume_cm3']=float((J[selected]*self.volumes[selected]).sum())
            r[layer['name']+'_min_detF']=float(J[selected].min())
            r[layer['name']+'_max_detF']=float(J[selected].max())
        r['maximum_end_displacement_cm']=float((x[self.end_nodes]-self.mesh.X[self.end_nodes]).norm(dim=-1).max())
        r['end_condition_is_exact_dirichlet']=False
        return r

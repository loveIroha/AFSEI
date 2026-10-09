"""Close the LV rim, align a Cartesian cavity grid, and mark idealized ports.

The solid input is read-only. No anatomical valve geometry is inferred.
The cavity discretization is a fixed, full-cell staircase approximation;
moving cut cells/GCL and coupling to the deforming myocardium remain pending.
"""
from dataclasses import dataclass,field
from collections import deque
from math import ceil,isfinite,pi
import numpy as np
import torch
from ..mac.grid import MACGrid,slab
from ..p1 import cavity_rim,cavity_volume


@dataclass(frozen=True)
class PortGeometry:
    mv_center_cm: tuple=(.65,0.)
    av_center_cm: tuple=(-.65,0.)
    mv_radius_cm: float=.55
    av_radius_cm: float=.40
    spacing_cm: float=.18

    def __post_init__(self):
        for name in ('mv_center_cm','av_center_cm'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        if (any(len(getattr(self,n))!=2 for n in ('mv_center_cm','av_center_cm'))
            or any(not isfinite(v) for n in ('mv_center_cm','av_center_cm') for v in getattr(self,n))
            or any(not isfinite(v) or v<=0 for v in (self.mv_radius_cm,self.av_radius_cm,self.spacing_cm))):
            raise ValueError('finite two-dimensional centres and positive radii/spacing required')
        distance=sum((a-b)**2 for a,b in zip(self.mv_center_cm,self.av_center_cm))**.5
        if distance<=self.mv_radius_cm+self.av_radius_cm:
            raise ValueError('idealized MV and AV discs must be disjoint')


@dataclass(frozen=True)
class CavityDomain:
    grid: MACGrid
    fluid: torch.Tensor
    tags: tuple
    frame_center: torch.Tensor
    frame_basis: torch.Tensor
    metadata: dict=field(default_factory=dict)

    def validate(self):
        if (self.fluid.shape!=self.grid.shape or self.fluid.dtype!=torch.bool
            or len(self.tags)!=3 or not self.fluid.any()
            or self.frame_center.shape!=(3,) or self.frame_basis.shape!=(3,3)
            or self.frame_center.dtype!=torch.float64 or self.frame_basis.dtype!=torch.float64
            or self.frame_center.device!=self.fluid.device or self.frame_basis.device!=self.fluid.device
            or not torch.isfinite(self.frame_center).all() or not torch.isfinite(self.frame_basis).all()
            or not torch.allclose(self.frame_basis@self.frame_basis.T,torch.eye(3,device=self.fluid.device,dtype=torch.float64),atol=1e-12,rtol=0)
            or torch.linalg.det(self.frame_basis)<=0):
            raise ValueError('invalid connected cavity geometry/frame')
        left,right=self.adjacent()
        for c,tag in enumerate(self.tags):
            if (tag.shape!=self.grid.face_shape(c) or tag.dtype!=torch.int64 or tag.device!=self.fluid.device
                or (tag<0).any() or (tag>2).any() or ((tag>0)&~(left[c]^right[c])).any()):
                raise ValueError('ports must lie on oriented fluid/exterior boundary faces')
        for label in (1,2):
            if sum(int((t==label).sum()) for t in self.tags)<4:
                raise ValueError('each idealized port must resolve at least four MAC faces; refine the grid')
        # A disconnected pressure component would need a separate gauge.
        raw=self.fluid.cpu().numpy(); first=tuple(np.argwhere(raw)[0]); visited={first}; queue=deque([first])
        while queue:
            cell=queue.popleft()
            for c in range(3):
                for sign in (-1,1):
                    neighbour=list(cell); neighbour[c]+=sign; neighbour=tuple(neighbour)
                    if 0<=neighbour[c]<raw.shape[c] and raw[neighbour] and neighbour not in visited:
                        visited.add(neighbour); queue.append(neighbour)
        if len(visited)!=int(raw.sum()): raise ValueError('voxel cavity has disconnected fluid cells; refine the grid')

    def adjacent(self):
        left=[]; right=[]
        for c in range(3):
            shape=list(self.grid.shape); shape[c]=1
            zero=torch.zeros(shape,device=self.fluid.device,dtype=torch.bool)
            left.append(torch.cat((zero,self.fluid),c)); right.append(torch.cat((self.fluid,zero),c))
        return tuple(left),tuple(right)

    def world(self,points): return points@self.frame_basis+self.frame_center


@torch.no_grad()
def vertical_inside(vertices,triangles,xy,z,*,chunk=128):
    """Ray crossings with coincident shared-edge roots grouped before parity.

    Opposite signed crossings at a tangency cancel. This also handles grid
    columns landing exactly on a triangulation diagonal, unlike naïve counts.
    All surface/column calculations remain on the tensor device.
    """
    nodes=vertices[triangles]; a=nodes[:,0]; e1=nodes[:,1]-a; e2=nodes[:,2]-a
    determinant=e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0]
    valid=determinant.abs()>1e-14
    denominator=torch.where(valid,determinant,1.)
    columns=[]
    for start in range(0,len(xy),chunk):
        delta=xy[start:start+chunk,None]-a[None,:,:2]
        u=(delta[...,0]*e2[:,1]-delta[...,1]*e2[:,0])/denominator
        v=(e1[:,0]*delta[...,1]-e1[:,1]*delta[...,0])/denominator
        hit=valid[None]&(u>=-1e-12)&(v>=-1e-12)&(u+v<=1+1e-12)
        roots=a[:,2]+u*e1[:,2]+v*e2[:,2]
        roots=torch.where(hit,roots,float('inf'))
        roots,order=roots.sort(dim=1)
        signs=determinant.sign()[None].expand_as(roots).gather(1,order)
        finite=torch.isfinite(roots)
        unique=finite.clone(); unique[:,1:] &= (roots[:,1:]-roots[:,:-1]).abs()>1e-9
        group=(unique.to(torch.int64).cumsum(1)-1).clamp_min(0)
        signed=torch.zeros_like(roots).scatter_add(1,group,torch.where(finite,signs,0.))
        unique &= signed.gather(1,group).abs()>0
        roots=torch.where(unique,roots,float('inf')).sort(dim=1).values
        below=torch.searchsorted(roots.contiguous(),z[None].expand(len(roots),-1).contiguous(),right=True)
        count=unique.sum(1)[:,None]-below
        columns.append((count.remainder(2)==1))
    return torch.cat(columns)


def _check_disc(polygon,center,radius):
    center=polygon.new_tensor(center); a=polygon; b=polygon.roll(-1,0)
    crossing=((a[:,1]>center[1])!=(b[:,1]>center[1]))
    denominator=b[:,1]-a[:,1]
    x=a[:,0]+(center[1]-a[:,1])*(b[:,0]-a[:,0])/torch.where(denominator!=0,denominator,1)
    inside=int((crossing&(x>center[0])).sum())%2==1
    edge=b-a; t=((center-a)*edge).sum(-1)/edge.square().sum(-1)
    distance=torch.linalg.vector_norm(center-a-t.clamp(0,1)[:,None]*edge,dim=-1).min().item()
    if not inside or distance<=radius:
        raise ValueError('idealized port disc must fit strictly inside the basal rim; supply different centres/radii')


@torch.no_grad()
def from_lv_mesh(mesh,ports=None):
    ports=ports or PortGeometry(); X=mesh.X; endo=mesh.surface(1); rim=cavity_rim(endo)
    rim_vertices=X[rim[:,0]]; center=rim_vertices.mean(0)
    _,_,basis=torch.linalg.svd(rim_vertices-center)
    normal=basis[-1]
    if (normal*(center-X[torch.unique(endo)].mean(0))).sum()<0: normal=-normal
    e1=X.new_tensor([1.,0.,0.]); e1=e1-(e1*normal).sum()*normal; e1/=torch.linalg.vector_norm(e1)
    frame=torch.stack((e1,torch.linalg.cross(normal,e1),normal))
    local=(X-center)@frame.T
    plane_error=(local[rim[:,0],2]).abs().max().item()
    if plane_error>1e-5: raise ValueError('this prototype requires a planar basal rim; no silent flattening')
    # The imported endocardium points OUT of solid and INTO blood; reverse it.
    cap=torch.cat((rim,rim.new_full((len(rim),1),len(X))),1)
    vertices=torch.cat((local,local.new_zeros(1,3)))
    closed=torch.cat((endo[:,[0,2,1]],cap))
    cap_nodes=vertices[cap]; cap_area=.5*torch.linalg.cross(cap_nodes[:,1]-cap_nodes[:,0],cap_nodes[:,2]-cap_nodes[:,0])
    if (cap_area[:,2]<=0).any(): raise ValueError('basal rim is not star-shaped around its mean; a different cap triangulation is needed')
    edges=torch.cat((closed[:,[0,1]],closed[:,[1,2]],closed[:,[2,0]])).cpu().numpy()
    counts={}
    for a,b in edges:
        key=(min(a,b),max(a,b)); n,s=counts.get(key,(0,0)); counts[key]=(n+1,s+(1 if a<b else -1))
    if any(n!=2 or s!=0 for n,s in counts.values()): raise ValueError('capped blood surface must be a closed oriented manifold')
    # cavity_rim validates a single cycle but returns incidence order, which
    # is not polygon order. Follow the directed cycle for disc containment.
    following=dict(rim.cpu().tolist()); current=int(rim[0,0]); ordered=[]
    for _ in range(len(rim)):
        ordered.append(current); current=following[current]
    polygon=local[torch.tensor(ordered,device=X.device),:2]
    for c,r in ((ports.mv_center_cm,ports.mv_radius_cm),(ports.av_center_cm,ports.av_radius_cm)):
        _check_disc(polygon,c,r)
    blood_vertices=local[torch.unique(endo)]; h=ports.spacing_cm
    lower=blood_vertices.min(0).values; upper=blood_vertices.max(0).values
    shape=(ceil((upper[0]-lower[0]).item()/h)+4,ceil((upper[1]-lower[1]).item()/h)+4,ceil(-lower[2].item()/h)+2)
    origin=(lower[0].item()-2*h,lower[1].item()-2*h,-shape[2]*h)
    grid=MACGrid(shape,tuple(n*h for n in shape),origin)
    points=grid.coordinates(device=X.device,dtype=X.dtype)
    inside=vertical_inside(vertices,closed,points[:,:,0,:2].reshape(-1,2),points[0,0,:,2]).reshape(shape)
    tags=tuple(torch.zeros(grid.face_shape(c),device=X.device,dtype=torch.int64) for c in range(3))
    top=grid.coordinates(2,device=X.device,dtype=X.dtype)[:,:,-1,:2]
    for label,c,r in ((1,ports.mv_center_cm,ports.mv_radius_cm),(2,ports.av_center_cm,ports.av_radius_cm)):
        disc=(top-top.new_tensor(c)).square().sum(-1)<=r*r
        if (disc&~inside[:,:,-1]).any(): raise ValueError('port intersects a voxel wall; refine or move the idealized port')
        tags[2][:,:,-1][disc]=label
    exact=cavity_volume(X,endo,rim).item()
    metadata=dict(geometry='fixed real LV; idealized basal MV/AV discs; staircase full-cell cavity',
        anatomical_valves_present=False,planar_rim_error_cm=plane_error,rim_nodes=len(rim),
        endocardial_faces=len(endo),cap_faces=len(cap),closed_surface_manifold=True,
        exact_reference_cavity_ml=exact,voxel_cavity_ml=int(inside.sum())*grid.volume,
        voxel_relative_volume_error=(int(inside.sum())*grid.volume-exact)/exact,
        ports_local_cm=dict(mv_center=list(ports.mv_center_cm),av_center=list(ports.av_center_cm),
                            mv_radius=ports.mv_radius_cm,av_radius=ports.av_radius_cm),
        nominal_port_area_cm2=dict(mv=pi*ports.mv_radius_cm**2,av=pi*ports.av_radius_cm**2))
    domain=CavityDomain(grid,inside,tags,center,frame,metadata); domain.validate(); return domain

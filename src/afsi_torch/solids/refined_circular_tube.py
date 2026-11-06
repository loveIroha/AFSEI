"""Nested solid-only P1 refinement; retain the original EP mesh and geometry."""
from dataclasses import replace
import numpy as np
import torch
from ..ep.refined_mesh import nested_mesh


class CoarseEPFieldTransfer:
    def __init__(self, data, coarse_nodes, like):
        self.coarse_nodes=coarse_nodes
        self.fine_nodes=len(data['X'])
        self.ids=torch.as_tensor(data['coarse_ids'],device=like.device)
        self.weights=torch.as_tensor(data['weights'],device=like.device,dtype=like.dtype)

    def prolong(self,value):
        if len(value)==self.fine_nodes:return value
        if len(value)!=self.coarse_nodes:
            raise ValueError('EP field is on neither the original nor the refined solid nodes')
        shape=(self.fine_nodes,)+(1,)*(value.ndim-1)
        result=value.new_zeros((self.fine_nodes,)+value.shape[1:])
        for j in range(4):result.add_(value[self.ids[:,j]]*self.weights[:,j].reshape(shape))
        return result

    def pullback_force(self,value):
        """Adjoint restriction for virtual-work tests; never used as a force filter."""
        if value.shape!=(self.fine_nodes,3):raise ValueError('fine nodal vector required')
        result=value.new_zeros((self.coarse_nodes,3))
        for j in range(4):result.index_add_(0,self.ids[:,j],self.weights[:,j,None]*value)
        return result


def refine_circular_tube(mesh,geometry,factor=2):
    """factor**3 children per tetra; preserve the original polyhedral domain.

    Original vertices keep their indices. DG0 material directions are inherited
    from the parent instead of recomputed on the finer mesh.
    """
    if type(factor) is not int or factor not in (2,3,4):
        raise ValueError('nested solid refinement factor must be 2, 3 or 4')
    X=mesh.X.detach().cpu().numpy();cells=mesh.cells.cpu().numpy()
    data=nested_mesh(X,cells,factor);Y=data['X'];newcells=data['cells'].copy()
    signed=np.linalg.det((Y[newcells[:,1:]]-Y[newcells[:,:1]]).transpose(0,2,1))
    if np.any(abs(signed)<1e-16):raise ValueError('degenerate nested solid child')
    negative=signed<0;newcells[negative,:2]=newcells[negative,1::-1]
    ids=data['coarse_ids'];weights=data['weights']
    mid={tuple(sorted(row[w>0])):i for i,(row,w) in enumerate(zip(ids,weights)) if np.count_nonzero(w)==2}
    def split_faces(faces):
        if factor != 2:
            return _split_surface_from_template(faces.cpu().numpy(),X,Y,data,factor)
        result=[]
        for a,b,c in faces.cpu().numpy():
            ab=mid[tuple(sorted((a,b)))];ac=mid[tuple(sorted((a,c)))];bc=mid[tuple(sorted((b,c)))]
            result.extend(((a,ab,ac),(ab,b,bc),(ac,bc,c),(ab,bc,ac)))
        return np.asarray(result,dtype=np.int64)
    original_inner=np.zeros(len(X),dtype=bool);original_inner[mesh.inner_rings.cpu().numpy().ravel()]=True
    on_inner=np.all(original_inner[ids] | (weights==0),axis=1)
    inner_ids=np.flatnonzero(on_inner)
    planes=np.round(Y[inner_ids,2],12)
    rings=[]
    for z in np.unique(planes):
        ring=inner_ids[planes==z]
        angle=np.mod(np.arctan2(Y[ring,1]-geometry.center_xy_cm[1],Y[ring,0]-geometry.center_xy_cm[0]),2*np.pi)
        # Roundoff at the positive x seam must sort as angle zero.
        angle[np.isclose(angle,2*np.pi,atol=1e-12)]=0
        rings.append(ring[np.argsort(angle)])
    if len(rings)!=factor*geometry.axial_elements+1 or any(len(q)!=factor*geometry.circumferential_elements for q in rings):
        raise ValueError('nested inner-wall section topology mismatch')
    flt=lambda a:torch.as_tensor(a,device=mesh.X.device,dtype=mesh.X.dtype)
    integer=lambda a:torch.as_tensor(np.asarray(a).copy(),device=mesh.X.device,dtype=torch.int64)
    refined=replace(mesh,X=flt(Y),cells=integer(newcells),
        fiber=mesh.fiber.repeat_interleave(factor**3,dim=0),sheet=mesh.sheet.repeat_interleave(factor**3,dim=0),
        inner_faces=integer(split_faces(mesh.inner_faces)),end_faces=integer(split_faces(mesh.end_faces)),
        inner_rings=integer(rings),
        cell_layer=mesh.cell_layer.repeat_interleave(factor**3) if mesh.cell_layer is not None else None)
    transfer=CoarseEPFieldTransfer(data,len(X),mesh.X)
    return refined,transfer


def _split_surface_from_template(faces,X,Y,data,factor):
    """Use the tetra subdivision's own facet rule, including global ordering."""
    from itertools import combinations
    bary=np.rint(data['template_bary']*factor).astype(np.int64)
    boundary=[]
    for child in data['template_cells']:
        for face in combinations(child,3):
            if np.all(bary[list(face),3]==0):boundary.append(face)
    boundary=np.asarray(boundary,dtype=np.int64)
    if len(boundary)!=factor**2:raise ValueError('invalid tetra surface template')
    nodes=np.unique(boundary)
    ids=data['coarse_ids'];counts=np.rint(data['weights']*factor).astype(np.int64)
    def key(row,w):return tuple(sorted((int(i),int(n)) for i,n in zip(row,w) if n))
    lookup={key(row,w):i for i,(row,w) in enumerate(zip(ids,counts))}
    result=[]
    for face in faces:
        ordered=np.sort(face)
        mapping={int(j):lookup[key(ordered,bary[j,:3])] for j in nodes}
        triangles=np.asarray([[mapping[int(j)] for j in tri] for tri in boundary],dtype=np.int64)
        normal=np.cross(X[face[1]]-X[face[0]],X[face[2]]-X[face[0]])
        child_normal=np.cross(Y[triangles[:,1]]-Y[triangles[:,0]],Y[triangles[:,2]]-Y[triangles[:,0]])
        flip=child_normal@normal<0
        triangles[flip,:2]=triangles[flip,1::-1]
        result.append(triangles)
    return np.concatenate(result,axis=0)

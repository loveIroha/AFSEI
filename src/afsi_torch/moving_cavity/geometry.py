"""Fitted, linearly moving LV cavity for geometry/interface verification.

The source endocardium and moving basal fan are retained exactly. A kernel
point cones each boundary triangle into a tetrahedron: this is a deliberately
simple verification mesh, not a quality/converged blood-flow mesh. Reject any
inverted cone instead of silently repairing the heart or dropping cells.
"""
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import torch
from ..p1 import cavity_rim
from ..circulation.geometry import PortGeometry,_check_disc

FACETS=((1,2,3),(0,3,2),(0,1,3),(0,2,1))


@dataclass(frozen=True)
class MovingCavityMesh:
    mapping: torch.Tensor
    cells: torch.Tensor
    faces: torch.Tensor
    cell_faces: torch.Tensor
    signs: torch.Tensor
    tags: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    reference_solid: torch.Tensor
    metadata: dict

    def coordinates(self,solid_x):
        if solid_x.shape!=self.reference_solid.shape or solid_x.dtype!=torch.float64 or solid_x.device!=self.reference_solid.device or not torch.isfinite(solid_x).all():
            raise ValueError('finite float64 solid coordinates on the reference device required')
        return torch.sparse.mm(self.mapping,solid_x)

    def transfer_reaction(self,face_force):
        if face_force.shape!=(len(self.faces),3): raise ValueError('face reaction shape mismatch')
        local=face_force[:,None,:].expand(-1,3,-1)/3
        blood=face_force.new_zeros((self.mapping.shape[0],3)).index_add(0,self.faces.flatten(),local.reshape(-1,3))
        return torch.sparse.mm(self.mapping.transpose(0,1),blood)


def tetra_volumes(x,cells):
    n=x[cells]; return ((n[:,1]-n[:,0])*torch.linalg.cross(n[:,2]-n[:,0],n[:,3]-n[:,0])).sum(-1)/6


def area_vectors(x,faces):
    n=x[faces]; return .5*torch.linalg.cross(n[:,1]-n[:,0],n[:,2]-n[:,0])


@dataclass(frozen=True)
class SweptGeometry:
    x: torch.Tensor
    volumes: torch.Tensor
    old_volumes: torch.Tensor
    areas: torch.Tensor
    grid_velocity: torch.Tensor
    grid_flux: torch.Tensor
    gradients: torch.Tensor
    gcl_max_ml_s: float
    minimum_height_cm: float


def swept_geometry(mesh,old_x,new_x,dt):
    if dt<=0: raise ValueError('positive moving-mesh dt required')
    old=mesh.coordinates(old_x)
    # Map the increment directly. Subtracting two large absolute sparse
    # coordinate products can create spurious wall velocities at zero motion.
    displacement=mesh.coordinates(new_x-old_x); new=old+displacement; mid=.5*(old+new)
    old_vol=tetra_volumes(old,mesh.cells); vol=tetra_volumes(new,mesh.cells)
    if (old_vol<=1e-14).any() or (vol<=1e-14).any() or not torch.isfinite(vol).all():
        raise ValueError('moving fan mesh inverted/degenerate; a quality cavity remesh is required')
    # Positive endpoints alone do not certify a linear mesh path. Volume is
    # cubic in path fraction; inspect its interior stationary points as well.
    e=old[mesh.cells][:,1:]-old[mesh.cells][:,:1]
    d=displacement[mesh.cells][:,1:]-displacement[mesh.cells][:,:1]
    triple=lambda a,b,c:(a*torch.linalg.cross(b,c)).sum(1)/6
    a1=triple(d[:,0],e[:,1],e[:,2])+triple(e[:,0],d[:,1],e[:,2])+triple(e[:,0],e[:,1],d[:,2])
    a2=triple(d[:,0],d[:,1],e[:,2])+triple(d[:,0],e[:,1],d[:,2])+triple(e[:,0],d[:,1],d[:,2])
    a3=triple(d[:,0],d[:,1],d[:,2])
    epsilon=1e-14*(old_vol.abs()+a1.abs()+a2.abs()+a3.abs())
    quadratic=a3.abs()>epsilon; denominator=torch.where(quadratic,3*a3,1.)
    root=(a2.square()-3*a3*a1).clamp_min(0).sqrt()
    linear=-a1/torch.where(a2.abs()>epsilon,2*a2,1.)
    candidates=torch.stack((torch.where(quadratic,(-a2+root)/denominator,linear),
                            torch.where(quadratic,(-a2-root)/denominator,linear),a1.new_full(a1.shape,.5)))
    candidates=candidates.clamp(0,1)
    intermediate=old_vol+candidates*(a1+candidates*(a2+candidates*a3))
    if (intermediate<=1e-14).any(): raise ValueError('linear moving-mesh path inverts between its endpoints')
    # Quadratic area under linear nodal motion: Simpson integration is exact.
    area=(area_vectors(old,mesh.faces)+4*area_vectors(mid,mesh.faces)+area_vectors(new,mesh.faces))/6
    w=(displacement/dt)[mesh.faces].mean(1); flux=(area*w).sum(-1)
    residual=(vol-old_vol)/dt-(mesh.signs*flux[mesh.cell_faces]).sum(-1)
    scale=max(((vol-old_vol)/dt).abs().max().item(),1.)
    if residual.abs().max()>2e-9*scale: raise RuntimeError('swept-face geometric conservation failed')
    nodes=mid[mesh.cells]; J=(nodes[:,1:]-nodes[:,:1]).transpose(1,2)
    inv=torch.linalg.inv(J); dL=nodes.new_tensor([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    # CR facet i is opposite tetrahedron vertex i: N_i=1-3 lambda_i.
    gradients=-3*torch.einsum('aj,ejk->eak',dL,inv)
    a=area_vectors(new,mesh.faces).norm(dim=-1)[mesh.cell_faces]
    height=(3*vol[:,None]/a).min().item()
    return SweptGeometry(new,vol,old_vol,area,w,flux,gradients,residual.abs().max().item(),height)


def from_solid(mesh,ports=None,cap_max_edge_cm=.30):
    ports=ports or PortGeometry(); X=mesh.X; endo=mesh.surface(1); rim=cavity_rim(endo)
    if not np.isfinite(cap_max_edge_cm) or cap_max_edge_cm<=0: raise ValueError('positive cap edge length required')
    source=sorted(set(endo.cpu().numpy().flatten().tolist())); lookup={s:i for i,s in enumerate(source)}
    rows=[{s:1.} for s in source]; coord=[v for v in X[source].cpu().numpy()]
    rimids=rim.cpu().tolist(); baseids=[a for a,b in rimids]
    basal=len(rows); rows.append({s:1/len(baseids) for s in baseids}); coord.append(X[baseids].mean(0).cpu().numpy())
    faces=[[lookup[int(c)],lookup[int(b)],lookup[int(a)]] for a,b,c in endo.cpu().tolist()]
    base=[False]*len(faces)
    faces.extend([[lookup[a],lookup[b],basal] for a,b in rimids]); base.extend([True]*len(rimids))
    faces=np.array(faces,dtype=np.int64); base=np.array(base,dtype=bool); coord=np.array(coord)
    # Conforming edge bisection propagates a rim split into its endocardial
    # neighbour. There are no hanging edges between refined cap and LV wall.
    for iteration in range(20):
        f=faces[base]; edges=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])); edges=np.unique(np.sort(edges,1),axis=0)
        lengths=np.linalg.norm(coord[edges[:,0]]-coord[edges[:,1]],axis=1)
        marked=edges[lengths>cap_max_edge_cm*(1+1e-12)]
        if not len(marked): break
        for a,b in marked:
            selected=((faces==a).any(1)&(faces==b).any(1)); owners=faces[selected]; flags=base[selected]
            if len(owners)!=2: raise ValueError('cap refinement needs a closed oriented surface')
            m=len(rows); weights=defaultdict(float)
            for k,v in rows[a].items(): weights[k]+=.5*v
            for k,v in rows[b].items(): weights[k]+=.5*v
            rows.append(dict(weights)); coord=np.vstack((coord,.5*(coord[a]+coord[b])))
            children=[]
            for face in owners:
                for j in range(3):
                    if set((face[j],face[(j+1)%3]))==set((a,b)):
                        u,v,z=face[j],face[(j+1)%3],face[(j+2)%3]; children.extend(((u,m,z),(m,v,z))); break
            faces=np.vstack((faces[~selected],children)); base=np.concatenate((base[~selected],np.repeat(flags,2)))
    else: raise RuntimeError('cap refinement did not terminate')
    center=X[baseids].mean(0); _,_,V=torch.linalg.svd(X[baseids]-center); normal=V[-1]
    if (normal*(center-X[source].mean(0))).sum()<0: normal=-normal
    e1=X.new_tensor([1.,0.,0.]); e1-=normal*(e1*normal).sum(); e1/=e1.norm()
    frame=torch.stack((e1,torch.linalg.cross(normal,e1),normal))
    following=dict(rimids); current=rimids[0][0]; ordered=[]
    for _ in rimids: ordered.append(current); current=following[current]
    polygon=((X[ordered]-center)@frame.T)[:,:2]
    for c,r in ((ports.mv_center_cm,ports.mv_radius_cm),(ports.av_center_cm,ports.av_radius_cm)):
        _check_disc(polygon,c,r)
    if ((X[baseids]-center)@normal).abs().max()>1e-5: raise ValueError('reference basal rim must be planar')
    local=(torch.as_tensor(coord,device=X.device,dtype=X.dtype)-center)@frame.T
    centroids=local[torch.as_tensor(faces,device=X.device)].mean(1)
    tags=np.zeros(len(faces),dtype=np.int64)
    for label,c,r in ((1,ports.mv_center_cm,ports.mv_radius_cm),(2,ports.av_center_cm,ports.av_radius_cm)):
        hit=((centroids[:,:2]-X.new_tensor(c)).square().sum(1)<=r*r).cpu().numpy()&base
        tags[hit]=label
        if hit.sum()<4: raise ValueError('idealized moving port has too few triangles; refine the cap')
    core=len(rows); rows.append({s:1/len(source) for s in source})
    row_index=[]; columns=[]; values=[]
    for i,row in enumerate(rows):
        for s,w in row.items(): row_index.append(i); columns.append(s); values.append(w)
    mapping=torch.sparse_coo_tensor(torch.tensor([row_index,columns],device=X.device),X.new_tensor(values),
        (len(rows),len(X)),check_invariants=True).coalesce().to_sparse_csr()
    surface=torch.as_tensor(faces,device=X.device)
    cells=torch.cat((surface.new_full((len(surface),1),core),surface),1)
    outward=cells[:,torch.tensor(FACETS,device=X.device)].reshape(-1,3)
    unique,inverse=torch.unique(outward.sort(1).values,dim=0,return_inverse=True)
    cellfaces=inverse.reshape(-1,4); counts=torch.bincount(inverse,minlength=len(unique))
    if not ((counts==1)|(counts==2)).all(): raise ValueError('fan tetrahedra must be conforming')
    first=torch.full((len(unique),),len(outward),device=X.device,dtype=torch.int64)
    first.scatter_reduce_(0,inverse,torch.arange(len(outward),device=X.device),reduce='amin')
    globalfaces=outward[first]
    signs=torch.where(torch.arange(len(outward),device=X.device)==first[inverse],1.,-1.).reshape(-1,4).to(torch.float64)
    left=first//4; right=torch.full_like(left,-1)
    other=torch.arange(len(outward),device=X.device)!=first[inverse]
    right[inverse[other]]=torch.arange(len(outward),device=X.device)[other]//4
    fulltags=torch.full_like(left,-1); fulltags[cellfaces[:,0]]=torch.as_tensor(tags,device=X.device)
    metadata=dict(mesh='fitted star-kernel cone verification mesh; CR facet velocity/P0 pressure',
        anatomical_valves_present=False,cap_max_edge_cm=cap_max_edge_cm,solid_nodes=len(X),blood_vertices=len(rows),
        blood_cells=len(cells),blood_faces=len(unique),wall_faces=int((fulltags==0).sum()),mv_faces=int((fulltags==1).sum()),av_faces=int((fulltags==2).sum()),
        ports=dict(mv_center_cm=list(ports.mv_center_cm),av_center_cm=list(ports.av_center_cm),mv_radius_cm=ports.mv_radius_cm,av_radius_cm=ports.av_radius_cm),
        frame_center=center.tolist(),frame_basis=frame.tolist())
    result=MovingCavityMesh(mapping,cells,globalfaces,cellfaces,signs,fulltags,left,right,X.clone(),metadata)
    geometry=swept_geometry(result,X,X,.001)
    metadata.update(reference_volume_ml=geometry.volumes.sum().item(),minimum_reference_height_cm=geometry.minimum_height_cm)
    return result

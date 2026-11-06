"""Idealized fixed LV tracts terminating at physical MAC box cut planes.

The added tracts are explicit case geometry. They are not anatomical valves.
Conservative staircase normal-impermeable partitions isolate two reservoirs
from the external fluid. Their tangential condition follows SealedBEFlow.
"""
from math import ceil, floor
import torch
from ..mac.cut_plane import CutPlane, CutPlaneFlow
from ..mac.sealed_flow import SealedBEFlow


def _footprint_rectangles(apertures,grid,other):
    """Disjoint exact rectangles of fixed MAC cells; prepared once per phase."""
    rectangles=[]
    for aperture in apertures:
        rows=aperture.detach().cpu().tolist()
        previous=None;begin=0
        for i,row in enumerate(rows+[[]]):
            runs=[];start=None
            for j,inside in enumerate(row+[False]):
                if inside and start is None:start=j
                elif not inside and start is not None:runs.append((start,j));start=None
            runs=tuple(runs)
            if runs!=previous:
                if previous:
                    for lo,hi in previous:
                        rectangles.append([
                            (grid.origin[other[0]]+begin*grid.spacing[other[0]],
                             grid.origin[other[0]]+i*grid.spacing[other[0]]),
                            (grid.origin[other[1]]+lo*grid.spacing[other[1]],
                             grid.origin[other[1]]+hi*grid.spacing[other[1]])])
                previous=runs;begin=i
    return rectangles


def cut_rows(endpoint,x):
    """Closed non-port patches; both basal holes connect freely to tracts."""
    from .mac_rcr_coupled import outlet_rows
    from .mac_seal import prepare_local_cap,LocalCapRows
    from .mac_compact_cap import compact_rows
    p=outlet_rows(endpoint,x,label=2).prepared
    closed=prepare_local_cap(p,patch_width_cells=endpoint.owner.cap_patch_width_cells,regions=(0,))
    patches=torch.empty_like(p.cap.labels)
    patches[closed.selected]=closed.patches
    area=[];diagonal=[];components=[]
    for label in (1,2):
        select=p.cap.labels==label
        patches[select]=closed.count+label-1
        area.append(p.area_vectors[select].norm(dim=-1).sum())
        diagonal.append(x.new_zeros(()))
    for local,port in zip(closed.components,p.rows.components):
        r,index,coefficient=local
        labels,pi,pc=port
        rs=[r];ids=[index];values=[coefficient]
        for label in (1,2):
            keep=labels==label
            rs.append(torch.full_like(pi[keep],closed.count+label-1))
            ids.append(pi[keep]);values.append(pc[keep])
            diagonal[label-1]+=pc[keep].square().sum()/endpoint.grid.volume
        components.append(tuple(torch.cat(v) for v in (rs,ids,values)))
    return compact_rows(endpoint,LocalCapRows(p,torch.arange(len(p.points),device=x.device),patches,
        torch.cat((closed.area,torch.stack(area))),tuple(components),
        torch.cat((closed.diagonal,torch.stack(diagonal)))))


class LVTractDomain:
    def __init__(self, grid, cap, x, *, mode, collar_cells=3, margin_cells=.4,
                 pressure_max_iterations=600, shared_collar_cells=0,neck_corner_trim_cells=0):
        if mode not in ('filling', 'rcr', 'closed'):
            raise ValueError('unknown LV cut-boundary phase')
        self.grid = grid
        self.mode = mode
        self.config = dict(pressure_max_iterations=pressure_max_iterations)
        # Galerkin coarsening permits arbitrary staircase locations.
        self.planes = ()
        normal = cap.frame[-1]
        axis = int(normal.abs().argmax())
        if float(normal[axis]) < .99:
            raise ValueError('this idealized tract requires the basal outward normal near a positive MAC axis')
        self.axis = axis
        if type(shared_collar_cells) is not int or shared_collar_cells<0:
            raise ValueError('shared_collar_cells must be a nonnegative integer')
        if (type(neck_corner_trim_cells) is not int or neck_corner_trim_cells<0
                or neck_corner_trim_cells and not shared_collar_cells):
            raise ValueError('neck corner trim requires nonnegative integer cells and a shared collar')
        other = [a for a in range(3) if a != axis]
        points, areas = cap.surface(x)
        h = grid.spacing
        coordinates = [grid.origin[a] + h[a] * (torch.arange(grid.shape[a],device=x.device,dtype=x.dtype)+.5)
                       for a in range(3)]
        self.bottom_index = floor((float(points[:,axis].min())-grid.origin[axis])/h[axis])-collar_cells
        self.shared_top_index=(ceil((float(points[:,axis].max())-grid.origin[axis])/h[axis])
                               +shared_collar_cells if shared_collar_cells else None)
        if self.shared_top_index is not None and self.shared_top_index>=grid.shape[axis]-2:
            raise ValueError('shared collar leaves insufficient fixed upper tract')
        if self.bottom_index < 2 or self.bottom_index >= grid.shape[axis]-4:
            raise ValueError('insufficient room for a fixed tract and its basal collar')
        self.bounds = []
        self.apertures = []
        self.label = torch.zeros(grid.shape,device=x.device,dtype=torch.int64)
        for label in (1,2):
            p = points[cap.labels==label]
            bounds = []
            aperture = torch.ones(tuple(grid.shape[a] for a in other),device=x.device,dtype=torch.bool)
            for j,a in enumerate(other):
                low = float(p[:,a].min())-margin_cells*h[a]
                high = float(p[:,a].max())+margin_cells*h[a]
                lo = floor((low-grid.origin[a])/h[a])
                hi = ceil((high-grid.origin[a])/h[a])
                if lo < 2 or hi > grid.shape[a]-2:
                    raise ValueError('tract touches a lateral box boundary')
                bounds.append((grid.origin[a]+lo*h[a],grid.origin[a]+hi*h[a]))
                line = (coordinates[a]>=bounds[-1][0])&(coordinates[a]<bounds[-1][1])
                shape=[1,1];shape[j]=len(line)
                aperture &= line.reshape(shape)
            if any((aperture & q).any() for q in self.apertures):
                raise ValueError('fixed inlet/outlet tracts overlap; geometry must be revised explicitly')
            self.bounds.append(bounds)
            self.apertures.append(aperture)
            shape=[1]*3
            for a,n in zip(other,aperture.shape):shape[a]=n
            region=aperture.reshape(shape).expand(grid.shape)
            z=torch.arange(grid.shape[axis],device=x.device)
            profile=[1]*3;profile[axis]=len(z)
            region=region & (z>=self.bottom_index).reshape(profile)
            self.label[region]=label
        self.neck_apertures=self.apertures
        self.storage_bounds=self.bounds
        self.storage_top_coordinate_cm=grid.origin[axis]+grid.lengths[axis]
        self.storage_fixed_volume_ml=0.
        if neck_corner_trim_cells:
            union=self.apertures[0]|self.apertures[1]
            indices=union.nonzero();lo=indices.amin(0);hi=indices.amax(0)+1
            n=neck_corner_trim_cells
            if bool((hi-lo<=2*n).any()):raise ValueError('corner trim removes the fixed neck')
            ii=torch.arange(union.shape[0],device=x.device)[:,None]
            jj=torch.arange(union.shape[1],device=x.device)[None,:]
            corners=((ii<lo[0]+n)|(ii>=hi[0]-n))&((jj<lo[1]+n)|(jj>=hi[1]-n))
            self.neck_apertures=[a&~corners for a in self.apertures]
            if any(not bool(a.any()) for a in self.neck_apertures):
                raise ValueError('corner trim removes a fixed port neck')
            z=torch.arange(grid.shape[axis],device=x.device)
            profile=[1]*3;profile[axis]=len(z)
            lower=(z<self.shared_top_index).reshape(profile)
            for label,(aperture,neck) in enumerate(zip(self.apertures,self.neck_apertures),1):
                shape=[1]*3
                for a,size in zip(other,aperture.shape):shape[a]=size
                removed=(aperture&~neck).reshape(shape).expand(grid.shape)&lower
                self.label.masked_fill_(removed,0)
            self.storage_bounds=_footprint_rectangles(self.neck_apertures,grid,other)
            self.storage_top_coordinate_cm=grid.origin[axis]+self.shared_top_index*h[axis]
            cut=grid.origin[axis]+grid.lengths[axis]
            self.storage_fixed_volume_ml=float(union.sum())*grid.volume/h[axis]*(cut-self.storage_top_coordinate_cm)
            self._neck_bounds_tensor=x.new_tensor(self.storage_bounds)
        self.masks=list(torch.ones(grid.face_shape(c),device=x.device,dtype=x.dtype) for c in range(3))
        for c in range(3):
            lo=self.label.narrow(c,0,grid.shape[c]-1)
            hi=self.label.narrow(c,1,grid.shape[c]-1)
            crossing=(lo!=hi)&((lo>0)|(hi>0))
            if c!=axis and self.shared_top_index is not None:
                # A zero-thickness internal septum starts ABOVE the moving cap
                # and its IB support. The lower collar is one connected blood
                # space. Outer walls, cut areas and geometric storage do not change.
                shape=[1]*3;shape[axis]=grid.shape[axis]
                lower=(torch.arange(grid.shape[axis],device=x.device)<self.shared_top_index).reshape(shape)
                crossing &= ~((lo>0)&(hi>0)&lower)
            # Bottoms are open connections to the ventricle; only lateral
            # tract walls are impermeable.
            if c != axis:
                self.masks[c].narrow(c,1,grid.shape[c]-1).masked_fill_(crossing,0.)
            elif neck_corner_trim_cells:
                # The upper rectangle overhangs the trimmed lower neck. Seal
                # its fixed horizontal ledges; only the basal neck is open.
                shape=[1]*3;shape[axis]=grid.shape[axis]-1
                ledge=(torch.arange(1,grid.shape[axis],device=x.device)==self.shared_top_index).reshape(shape)
                self.masks[c].narrow(c,1,grid.shape[c]-1).masked_fill_(crossing&ledge,0.)
        opened = 1 if mode=='filling' else 2 if mode=='rcr' else 0
        for label,aperture in enumerate(self.apertures,1):
            if label!=opened:
                self.masks[axis].select(axis,-1).masked_fill_(aperture,0.)
        self.masks=tuple(self.masks)
        self.cut_planes=tuple(CutPlane(axis,-1,aperture,name)
            for label,aperture,name in zip((1,2),self.apertures,('mv_cut','av_cut')) if label==opened)
        self.metadata=dict(kind='gao-fixed-tracts-to-physical-MAC-cut-plane',
            axis=axis,cut_coordinate_cm=grid.origin[axis]+grid.lengths[axis],
            bottom_coordinate_cm=grid.origin[axis]+self.bottom_index*h[axis],
            basal_normal=normal.tolist(),tract_bounds_cm=self.bounds,
            cut_areas_cm2=[float(a.sum())*grid.volume/h[axis] for a in self.apertures],
            mode=mode,anatomical_valves=False,
            shared_collar_cells=shared_collar_cells,
            neck_corner_trim_cells=neck_corner_trim_cells,
            neck_areas_cm2=[float(a.sum())*grid.volume/h[axis] for a in self.neck_apertures],
            storage_partition_rectangles_cm=self.storage_bounds,
            storage_top_coordinate_cm=self.storage_top_coordinate_cm,
            storage_fixed_volume_ml=self.storage_fixed_volume_ml,
            shared_collar_top_cm=(grid.origin[axis]+self.shared_top_index*h[axis]
                                  if self.shared_top_index is not None else None),
            fixed_partitions='normal-impermeable staircase; original sealed-flow tangential discretization',
            geometry_matches_Gao_patient=False)

    def mask(self,u):return tuple(v*m for v,m in zip(u,self.masks))

    @torch.no_grad()
    def budget(self,velocity,prepared,material_velocity):
        """Separate fixed-tract FV balance from moving-cap quadrature balance."""
        from .mac_ports import _stencil
        label=1 if self.mode=='filling' else 2
        aperture=self.apertures[label-1]
        area=self.grid.volume/self.grid.spacing[self.axis]
        qcut=(velocity[self.axis].select(self.axis,-1)*aperture).sum()*area
        qneck=(velocity[self.axis].select(self.axis,self.bottom_index)*self.neck_apertures[label-1]).sum()*area
        branch_qneck=qneck
        if self.shared_top_index is not None:
            # Internal lateral exchange cancels only for the UNION control
            # volume. A per-branch neck balance is not valid below the septum.
            aperture=self.apertures[0]|self.apertures[1]
            qcut=(velocity[self.axis].select(self.axis,-1)*aperture).sum()*area
            neck=self.neck_apertures[0]|self.neck_apertures[1]
            qneck=(velocity[self.axis].select(self.axis,self.bottom_index)*neck).sum()*area
        points=prepared.points
        take=torch.zeros(len(points),device=points.device,dtype=torch.bool)
        other=[a for a in range(3) if a!=self.axis]
        bounds_to_use=self.storage_bounds if self.shared_top_index is not None else [self.bounds[label-1]]
        for bounds in bounds_to_use:
            inside=torch.ones_like(take)
            for a,(lo,hi) in zip(other,bounds):inside &= (points[:,a]>=lo)&(points[:,a]<hi)
            take |= inside
        sampled=torch.stack([(u.reshape(-1)[ids]*w).sum(-1)
            for u,(ids,w) in zip(velocity,_stencil(self.grid,points))],-1)
        fluid=(sampled*prepared.area_vectors).sum(-1)
        motion=(prepared.cap.material_points(material_velocity)*prepared.area_vectors).sum(-1)
        opening=prepared.cap.labels==label
        return dict(cut_outward_ml_s=qcut,fixed_neck_outward_ml_s=qneck,
            branch_neck_outward_ml_s=branch_qneck,
            control_volume='shared-collar-union' if self.shared_top_index is not None else 'single-tract',
            fixed_tract_mass_error_ml_s=qcut-qneck,
            basal_opening_absolute_ml_s=fluid[opening].sum(),
            basal_opening_motion_ml_s=motion[opening].sum(),
            cap_over_tract_absolute_ml_s=fluid[take].sum(),
            cap_over_tract_motion_ml_s=motion[take].sum(),
            closed_cap_over_tract_relative_ml_s=(fluid-motion)[take&~opening].sum(),
            cut_minus_cap_absolute_ml_s=qcut-fluid[take].sum())

    def validate_wall(self,cap,x):
        points,_=cap.surface(x)
        other=[a for a in range(3) if a!=self.axis]
        if self.shared_top_index is not None:
            p=points[(cap.labels==1)|(cap.labels==2)]
            covered=torch.zeros(len(p),dtype=torch.bool,device=p.device)
            if hasattr(self,'_neck_bounds_tensor'):
                # Check the union, including internal partition seams. Four
                # tiny offsets reject contact with a true outer neck wall.
                bounds=self._neck_bounds_tensor
                eps=32*torch.finfo(p.dtype).eps*max(self.grid.lengths)
                offsets=p.new_tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])*eps
                q=p[:,None,None,other]+offsets[None,:,None,:]
                inside=((q>=bounds[None,None,:,:,0])&(q<=bounds[None,None,:,:,1])).all(-1)
                covered=inside.any(-1).all(-1)
            else:
                for bounds in self.bounds:
                    inside=torch.ones_like(covered)
                    for a,(low,high) in zip(other,bounds):inside &= (p[:,a]>low)&(p[:,a]<high)
                    covered |= inside
            # Septum must remain outside the four-point IB kernel support.
            top=self.metadata['shared_collar_top_cm']-2*self.grid.spacing[self.axis]
            if not bool(covered.all()) or float(p[:,self.axis].max())>=top:
                raise ValueError('moving virtual ports left shared collar or its IB support margin')
            if float(points[:,self.axis].min())<=self.metadata['bottom_coordinate_cm']:
                raise ValueError('moving basal cap left the fixed tract collar')
            return
        for label,bounds in zip((1,2),self.bounds):
            p=points[cap.labels==label]
            for a,(low,high) in zip(other,bounds):
                if float(p[:,a].min())<=low or float(p[:,a].max())>=high:
                    raise ValueError('moving virtual port has detached from its fixed tract')
        if float(points[:,self.axis].min()) <= self.metadata['bottom_coordinate_cm']:
            raise ValueError('moving basal cap left the fixed tract collar')


def prepare_cut_flow(endpoint,state,control):
    """Cache geometry/response per phase, preserving source NS/FE/IB parameters."""
    from .mac_rcr_coupled import outlet_rows
    if not hasattr(endpoint,'_gao_original_flow'):
        endpoint._gao_original_flow=endpoint.flow
        endpoint._gao_tract_flows={}
        cap=outlet_rows(endpoint,state.x,label=2).prepared.cap
        endpoint._gao_material_cap=cap
        anchor=getattr(endpoint.owner,'gao_tract_anchor',None)
        if anchor is None:
            endpoint.owner.gao_tract_anchor=state.x.clone()
            anchor=endpoint.owner.gao_tract_anchor
        if anchor.shape!=state.x.shape or anchor.device!=state.x.device or anchor.dtype!=state.x.dtype:
            raise ValueError('fixed tract geometry anchor must match the mechanical mesh')
        endpoint._gao_tract_anchor=anchor
    if control.mode not in endpoint._gao_tract_flows:
        domain=LVTractDomain(endpoint.grid,endpoint._gao_material_cap,endpoint._gao_tract_anchor,
            mode=control.mode,**getattr(endpoint.owner,'cut_geometry',{}))
        domain.config['pressure_compact_geometry']=getattr(endpoint.owner,'cut_pressure_compact_geometry',False)
        domain.config['pressure_solver_tolerances']=getattr(endpoint.owner,'cut_pressure_tolerances',None)
        base=SealedBEFlow(endpoint._gao_original_flow,domain,fused=True,graphs=True)
        cut=CutPlaneFlow(base,domain.cut_planes)
        from .tract_storage import TractStorage
        storage_type=TractStorage
        if getattr(endpoint.owner,'cut_tract_geometry_gpu',False):
            from .tract_storage_gpu import GPUTractStorage
            storage_type=GPUTractStorage
        domain.storage=storage_type(endpoint._gao_material_cap,domain)
        endpoint._gao_tract_flows[control.mode]=cut
    endpoint.flow=endpoint._gao_tract_flows[control.mode]
    endpoint.flow.domain.validate_wall(endpoint._gao_material_cap,state.x)
    if control.mode=='rcr':
        endpoint.flow.rcr(control.arterial_pc_mmhg,control.windkessel)
    elif control.mode=='filling':
        endpoint.flow.prescribe([control.filling_pressure_mmhg])
    else:
        endpoint.flow.prescribe([])
    endpoint.cut_boundary=True

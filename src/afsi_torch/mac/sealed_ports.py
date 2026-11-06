"""Fixed MAC septa, sealing collars and isolated end reservoirs.

Flexible wall coupling is still the original FE/IB pair. Fixed partitions
remove normal face velocities only (tangential velocity is not clamped).
The central exterior remains vented in 3D; reservoir side faces are closed.
No porous resistance, pressure cancellation or prescribed pumping is used.
Collars lie in the reference wall, joining each septum to the immersed wall;
their geometry and reservoir lengths are additional case data, not Sharifi's
published geometry. Apertures use conservative whole MAC faces.
"""
from math import cos,pi,isfinite,prod
import torch


def validate_hydraulic(c):
    names={'kind','aperture_margin_cells','collar_cells','pressure_max_iterations'}
    if not isinstance(c,dict) or set(c)!=names or c['kind']!='sealed-end-reservoirs':
        raise ValueError('hydraulic_domain requires sealed-end-reservoirs and explicit margin/collar/pressure controls')
    if not isfinite(c['aperture_margin_cells']) or not 0<c['aperture_margin_cells']<=1:
        raise ValueError('aperture margin must be in (0,1] MAC cells')
    if type(c['collar_cells']) is not int or c['collar_cells']<2:
        raise ValueError('sealing collar must extend at least two MAC cells into the wall')
    if type(c['pressure_max_iterations']) is not int or c['pressure_max_iterations']<1:
        raise ValueError('positive pressure iteration limit required')


class SealedDomain:
    def __init__(self,grid,geometry,config,like,*,dimension,depth=1.):
        validate_hydraulic(config)
        self.grid,self.geometry,self.config=grid,geometry,config
        self.dimension,self.depth=dimension,depth
        self.axis=0 if dimension==2 else 2
        axis=self.axis;h=grid.spacing;self.spacing=h
        origin=getattr(grid,'origin',(0.,)*dimension)
        start=geometry.inlet_buffer_cm if dimension==2 else geometry.axial_start_cm
        ends=(start,start+geometry.length_cm)
        self.planes=tuple(round((v-origin[axis])/h[axis]) for v in ends)
        if any(abs(origin[axis]+i*h[axis]-v)>1e-10*h[axis] for i,v in zip(self.planes,ends)):
            raise ValueError('septa must coincide with axial MAC faces')
        lo,hi=self.planes;n=grid.shape[axis];nc=config['collar_cells']
        if lo<2 or hi>n-2 or hi-lo<=2*nc:
            raise ValueError('need two-cell end reservoirs and nonoverlapping collars')
        self.masks=list(torch.ones(grid.face_shape(c),device=like.device,dtype=like.dtype) for c in range(dimension))
        positions=[origin[c]+h[c]*(torch.arange(grid.shape[c],device=like.device,dtype=like.dtype)+.5) for c in range(dimension)]
        others=[c for c in range(dimension) if c!=axis]
        query=torch.meshgrid(*(positions[c] for c in others),indexing='ij')
        margin=config['aperture_margin_cells']*max(h[c] for c in others)
        if dimension==2:
            low=geometry.center_y_cm-geometry.lumen_height_cm/2
            high=geometry.center_y_cm+geometry.lumen_height_cm/2
            aperture=(query[0]-h[1]/2>=low+margin)&(query[0]+h[1]/2<=high-margin)
            transverse=(positions[1]>low-geometry.wall_thickness_cm/2)&(positions[1]<high+geometry.wall_thickness_cm/2)
            edges=torch.where(transverse[:-1]!=transverse[1:])[0]+1
            if len(edges)!=2: raise ValueError('unresolved sealing collar')
            self.collar_bounds=tuple(float(origin[1]+i*h[1]) for i in edges)
            inside=transverse[None,:].expand(grid.shape)
        else:
            radius=geometry.inner_radius_cm*cos(pi/geometry.circumferential_elements)-margin
            aperture=torch.ones_like(query[0],dtype=torch.bool)
            for sx in (-.5,.5):
                for sy in (-.5,.5):
                    aperture &= ((query[0]+sx*h[0]-geometry.center_xy_cm[0])**2+
                        (query[1]+sy*h[1]-geometry.center_xy_cm[1])**2<=radius**2)
            inside=((positions[0][:,None,None]-geometry.center_xy_cm[0])**2+
                (positions[1][None,:,None]-geometry.center_xy_cm[1])**2<
                (geometry.inner_radius_cm+geometry.wall_thickness_cm/2)**2).expand(grid.shape)
        if int(aperture.count_nonzero())<2: raise ValueError('sealing aperture is unresolved')
        self.aperture=aperture
        for i in self.planes: self.masks[axis].select(axis,i).copy_(aperture)
        # Reservoir sides are impermeable. Exterior in the central region
        # remains pressure-open in 3D, allowing passive wall volume changes.
        indices=torch.arange(n,device=like.device)
        central=(indices>=lo)&(indices<hi)
        collars=((indices>=lo)&(indices<lo+nc))|((indices>=hi-nc)&(indices<hi))
        radial_bounds=[]
        for c in others:
            for side in (0,-1):
                target=self.masks[c].select(c,side)
                shape=[1]*(dimension-1);shape[axis-(axis>c)]=n
                target.mul_(central.reshape(shape))
            # Collars embedded inside the wall, connected to the septa. Only
            # normal velocities crossing their staircase surface are fixed.
            crossing=inside.narrow(c,0,grid.shape[c]-1)!=inside.narrow(c,1,grid.shape[c]-1)
            profile=[1]*dimension;profile[axis]=n
            crossing &= collars.reshape(profile)
            if dimension==3:
                ids=torch.nonzero(crossing).to(like.dtype)
                ids[:,c]+=1
                point=like.new_tensor(origin)+like.new_tensor(h)*(ids+.5)
                point[:,c]-=.5*h[c]
                radius=(point[:,:2]-point.new_tensor(geometry.center_xy_cm)).norm(dim=-1)
                radial_bounds.extend((float(radius.min()-.5*max(h[:2])),float(radius.max()+.5*max(h[:2]))))
            self.masks[c].narrow(c,1,grid.shape[c]-1).masked_fill_(crossing,0.)
        if dimension==3:self.collar_bounds=(min(radial_bounds),max(radial_bounds))
        self.masks=tuple(self.masks)
        self.area=prod(h[c] for c in others)*depth
        self.max_inlet_mismatch=like.new_zeros(())
        self.max_outlet_mismatch=like.new_zeros(())

    def mask(self,u): return tuple(v*m for v,m in zip(u,self.masks))

    def bind_wall(self,mesh):
        """Cache reference groups for a once-per-accepted-step sealing guard."""
        self.wall_mesh=mesh
        X=mesh.X;g=self.geometry;h=self.spacing[self.axis]
        start=g.inlet_buffer_cm if self.dimension==2 else g.axial_start_cm
        z=X[:,self.axis];end=start+g.length_cm
        self.end_ids=[torch.where((z-v).abs()<1e-10*h)[0] for v in (start,end)]
        collar=(z-start<=self.config['collar_cells']*h+1e-12)|(end-z<=self.config['collar_cells']*h+1e-12)
        if self.dimension==3:
            r=(X[:,:2]-X.new_tensor(g.center_xy_cm)).norm(dim=-1)
            self.inner=torch.where(collar & ((r-g.inner_radius_cm).abs()<1e-10))[0]
            self.outer=torch.where(collar & ((r-g.inner_radius_cm-g.wall_thickness_cm).abs()<1e-10))[0]
        else:
            low=g.center_y_cm-g.lumen_height_cm/2;high=g.center_y_cm+g.lumen_height_cm/2
            self.groups=[torch.where(collar & ((X[:,1]-v).abs()<1e-10))[0]
                for v in (low-g.wall_thickness_cm,low,high,high+g.wall_thickness_cm)]

    def validate_wall(self,x):
        g=self.geometry;h=self.spacing[self.axis]
        start=g.inlet_buffer_cm if self.dimension==2 else g.axial_start_cm
        shifts=torch.stack([(x[ids,self.axis]-v).abs().max() for ids,v in zip(self.end_ids,(start,start+g.length_cm))])
        if self.dimension==3:
            center=x.new_tensor(g.center_xy_cm)
            ri=(x[self.inner,:2]-center).norm(dim=-1).max()
            ro=(x[self.outer,:2]-center).norm(dim=-1).min()
            low,high=self.collar_bounds
            overlaps=torch.stack((low-ri,ro-high))
        else:
            low,high=self.collar_bounds
            a,b,c,d=(x[ids,1] for ids in self.groups)
            overlaps=torch.stack((low-a.max(),b.min()-low,high-c.max(),d.min()-high))
        metrics=torch.stack((shifts.max(),overlaps.min())).tolist()
        if metrics[0]>=.5*self.config['collar_cells']*h or metrics[1]<=0:
            raise ValueError('moving wall detached from its fixed sealing collar; stop rather than silently reconnect exterior fluid')

    def flux(self,u,index):
        return (u[self.axis].select(self.axis,index)*self.aperture).sum()*self.area

    def accepted(self,u):
        qi=self.flux(u,self.planes[0]);qo=self.flux(u,self.planes[1])
        bi=u[self.axis].select(self.axis,0).sum()*self.area
        bo=u[self.axis].select(self.axis,-1).sum()*self.area
        self.max_inlet_mismatch=torch.maximum(self.max_inlet_mismatch,(qi-bi).abs())
        self.max_outlet_mismatch=torch.maximum(self.max_outlet_mismatch,(qo-bo).abs())
        self.last=dict(inlet_aperture_ml_s=qi,outlet_aperture_ml_s=qo,
            inlet_reservoir_boundary_ml_s=bi,outlet_reservoir_boundary_ml_s=bo,
            inlet_flux_mismatch_ml_s=qi-bi,outlet_flux_mismatch_ml_s=qo-bo)

    def diagnostics(self):
        result={k:float(v) for k,v in getattr(self,'last',{}).items()}
        result.update(max_inlet_flux_mismatch_ml_s=float(self.max_inlet_mismatch),
            max_outlet_flux_mismatch_ml_s=float(self.max_outlet_mismatch),
            resolved_port_area_cm2=float(self.aperture.sum())*self.area,
            hydraulic_path_isolated=True)
        return result

    def summary(self):
        return dict(kind='sealed-end-reservoirs',planes=list(self.planes),
            resolved_port_area_cm2=float(self.aperture.sum())*self.area,
            collar_cells=self.config['collar_cells'],aperture_margin_cells=self.config['aperture_margin_cells'],
            stationary_partition='normal impermeability; tangential velocity not clamped',
            central_exterior='reference-pressure vented; reservoir side faces impermeable',
            paper_geometry_aligned=False)

    def write_geometry(self,directory):
        """Static fixed-face geometry for ParaView; setup-only CPU export."""
        from pathlib import Path
        import numpy as np
        import meshio
        import xml.etree.ElementTree as ET
        folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
        origin=np.array(getattr(self.grid,'origin',(0.,)*self.dimension))
        h=np.array(self.spacing);points=[];cells=[];kinds=[]
        for normal,m in enumerate(self.masks):
            indices=torch.nonzero(m==0).detach().cpu().numpy()
            tangents=[a for a in range(self.dimension) if a!=normal]
            for index in indices:
                center=origin+h*(index+.5);center[normal]=origin[normal]+h[normal]*index[normal]
                offsets=[(-.5,),(.5,)] if self.dimension==2 else [(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)]
                row=[]
                for offset in offsets:
                    p=center.copy()
                    for a,s in zip(tangents,offset):p[a]+=s*h[a]
                    row.append(len(points));points.append(np.pad(p,(0,3-self.dimension)))
                cells.append(row)
                outer=index[normal] in (0,self.grid.shape[normal])
                kinds.append(3 if outer else 1 if normal==self.axis else 2)
        kind='line' if self.dimension==2 else 'quad'
        name='hydraulic_boundaries.vtu'
        meshio.write(folder/name,meshio.Mesh(np.array(points),[(kind,np.array(cells))],
            cell_data={'boundary_kind_1_septum_2_collar_3_reservoir_side':[np.array(kinds)]}),binary=True)
        doc=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
        collection=ET.SubElement(doc,'Collection')
        ET.SubElement(collection,'DataSet',timestep='0',group='',part='0',file=name)
        ET.ElementTree(doc).write(folder/'hydraulic_boundaries.pvd',encoding='utf-8',xml_declaration=True)

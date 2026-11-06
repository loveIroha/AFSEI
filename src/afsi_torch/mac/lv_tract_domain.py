"""Fixed upper tract collars continuing the explicit FE housing to box cuts.

Whole-face staircase normal impermeability; original exterior remains vented.
This is a documented geometric approximation, not an anatomical valve model.
FE walls overlap the collars; they never meet the physical box boundary, so
the original IB kernels retain their full support.
"""
import torch
from .cut_plane import CutPlane


class LVTractDomain:
    def __init__(self,grid,housing,like,*,collar_start_cm=12.3,pressure_max_iterations=600):
        self.grid=grid;self.planes=();self.config={'pressure_max_iterations':pressure_max_iterations}
        self.masks=list(torch.ones(grid.face_shape(c),device=like.device,dtype=like.dtype) for c in range(3))
        origin=getattr(grid,'origin',(0.,0.,0.));h=grid.spacing
        xs=[origin[i]+h[i]*(torch.arange(grid.shape[i],device=like.device,dtype=like.dtype)+.5) for i in range(3)]
        self.centers=like.new_tensor(housing['port_centers']);self.radii=like.new_tensor(housing['port_radii'])
        if float(housing['top_z'])+2*h[2]>=origin[2]+grid.lengths[2]:
            raise ValueError('FE tract end must retain two full IB support cells below outer cut')
        self.apertures=[]
        for center,r in zip(self.centers,self.radii):
            inside=(xs[0][:,None]-center[0]).square()+(xs[1][None,:]-center[1]).square()<r*r
            if inside.sum()<4:raise ValueError('unresolved fixed tract aperture')
            self.apertures.append(inside)
            for axis in (0,1):
                crossing=inside.narrow(axis,0,grid.shape[axis]-1)!=inside.narrow(axis,1,grid.shape[axis]-1)
                blocked=crossing[:,:,None] & (xs[2][None,None,:]>=collar_start_cm)
                self.masks[axis].narrow(axis,1,grid.shape[axis]-1).masked_fill_(blocked,0.)
        if (self.apertures[0]&self.apertures[1]).any():raise ValueError('resolved tracts overlap')
        self.masks=tuple(self.masks)
        self.inlet=CutPlane(2,-1,self.apertures[0],'inflow')
        self.outlet=CutPlane(2,-1,self.apertures[1],'outflow')
        self.collar_start_cm=collar_start_cm

    def mask(self,u):return tuple(a*m for a,m in zip(u,self.masks))

    def summary(self):
        area=self.grid.spacing[0]*self.grid.spacing[1]
        return dict(kind='explicit FE passive housing plus fixed upper MAC collars',
            physical_cut_z_cm=self.grid.origin[2]+self.grid.lengths[2],
            collar_start_cm=self.collar_start_cm,
            resolved_port_areas_cm2=[float(a.sum())*area for a in self.apertures],
            external_reference_pressure_mmhg=0.,anatomical_valves=False,
            fixed_wall_discretization='whole-face staircase normal impermeability; overlapping tethered FE tract wall',
            virtual_cap=False,pressure_mode_correction=False)

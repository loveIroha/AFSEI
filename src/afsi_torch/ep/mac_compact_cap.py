"""Same cap quadrature and adjoints, evaluated with compact GPU matrices."""
import torch


class CompactCapRows:
    @torch.no_grad()
    def __init__(self,reference):
        self.reference=reference
        self.matrices=[]
        for row,index,value in reference.components:
            faces,compact=torch.unique(index,sorted=True,return_inverse=True)
            matrix=torch.sparse_coo_tensor(torch.stack((row,compact)),value,
                (reference.count,len(faces)),device=value.device,check_invariants=True).coalesce()
            self.matrices.append((faces,matrix.to_sparse_csr(),matrix.T.coalesce().to_sparse_csr()))
        p=reference.prepared; cap=p.cap
        self.nodes=torch.unique(cap.rim)
        edges=torch.searchsorted(self.nodes,cap.rim[cap.parent[reference.selected]])
        area=p.area_vectors[reference.selected]
        bary=cap.barycentric[reference.selected]
        matrix=area.new_zeros((reference.count,len(self.nodes),3))
        flat=matrix.reshape(-1,3)
        for endpoint in (0,1):
            key=reference.patches*len(self.nodes)+edges[:,endpoint]
            flat.index_add_(0,key,area*bary[:,endpoint,None])
        center=area.new_zeros((reference.count,3))
        center.index_add_(0,reference.patches,area*bary[:,2,None]/len(cap.rim))
        center_nodes=torch.searchsorted(self.nodes,cap.rim[:,0].contiguous())
        matrix.index_add_(1,center_nodes,center[:,None,:].expand(-1,len(center_nodes),-1))
        self.material_matrix=matrix.reshape(reference.count,-1).contiguous()

    def __getattr__(self,name): return getattr(self.reference,name)

    def flow(self,velocity):
        self.prepared.cap.grid.check_velocity(velocity)
        if any(u.dtype!=self.area.dtype or u.device!=self.area.device for u in velocity):
            raise ValueError('matching compact cap velocity required')
        return sum(torch.sparse.mm(matrix,u.reshape(-1)[faces,None]).flatten()
            for u,(faces,matrix,transpose) in zip(velocity,self.matrices))

    def lift(self,pressure):
        self._check_pressure(pressure)
        grid=self.prepared.cap.grid
        result=grid.zeros(device=pressure.device,dtype=pressure.dtype)
        for u,(faces,matrix,transpose) in zip(result,self.matrices):
            value=torch.sparse.mm(transpose,pressure[:,None]).flatten()/grid.volume
            u.reshape(-1).index_copy_(0,faces,value)
        return result

    def _check_pressure(self,pressure,*,check_finite=True):
        if (pressure.shape!=self.area.shape or pressure.dtype!=self.area.dtype
                or pressure.device!=self.area.device
                or (check_finite and not torch.isfinite(pressure).all())):
            raise ValueError('finite matching compact cap pressures required')

    def add_lift_into(self,pressure,density,*,check_finite=True):
        """Add the same lift to caller-owned density without three zero grids.

        Internal coupled callers may defer finite checks to the following mass
        or BE solve. Public calls check pressure by default, including NaNs on
        empty component supports. Metadata checks always remain enabled.
        """
        self._check_pressure(pressure,check_finite=check_finite)
        grid=self.prepared.cap.grid
        grid.check_velocity(density)
        if any(u.dtype!=pressure.dtype or u.device!=pressure.device or not u.is_contiguous()
               for u in density):
            raise ValueError('owned contiguous matching cap density required')
        for u,(faces,matrix,transpose) in zip(density,self.matrices):
            value=torch.sparse.mm(transpose,pressure[:,None]).flatten()/grid.volume
            u.reshape(-1).index_add_(0,faces,value)
        return density

    def material_flow(self,velocity):
        if (velocity.shape!=self.prepared.cap.X.shape or velocity.dtype!=self.area.dtype
                or velocity.device!=self.area.device):
            raise ValueError('matching cap material velocity required')
        return self.material_matrix@velocity[self.nodes].flatten()

    def reaction(self,pressure,*,check_finite=True):
        self._check_pressure(pressure,check_finite=check_finite)
        result=torch.zeros_like(self.prepared.cap.X)
        result.index_copy_(0,self.nodes,(self.material_matrix.T@pressure).reshape(-1,3))
        return result


def cap_reaction(rows,pressure,*,check_finite=True):
    if isinstance(rows,CompactCapRows):
        return rows.reaction(pressure,check_finite=check_finite)
    return rows.reaction(pressure)


def add_cap_lift(rows,pressure,density,*,check_finite=True):
    if isinstance(rows,CompactCapRows):
        return rows.add_lift_into(pressure,density,check_finite=check_finite)
    return tuple(a+b for a,b in zip(density,rows.lift(pressure)))


def compact_rows(endpoint,rows):
    return CompactCapRows(rows) if getattr(endpoint.owner,'compact_cap_operators',False) else rows

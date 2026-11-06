"""Exact fixed-rectangle/P1-cap clipping with GPU derivatives.

Only discrete clipping topology is cached on the host. Every evaluation
checks the cached branch decisions; intersections, volume, gradient and
Hessian use current FP64 coordinates. A changed topology is rebuilt, with
the original implementation retained for an ambiguous branch boundary.
"""
import torch
from .tract_storage import TractStorage

MAX_VERTICES=12


class _Topology:
    def __init__(self,storage):
        self.storage=storage
        shape=(4,len(storage.bounds),len(storage.rim),MAX_VERTICES)
        like=storage.nodes
        self.p=torch.zeros(shape,device=like.device,dtype=torch.int64)
        self.q=torch.zeros_like(self.p)
        self.edge=torch.zeros(shape,device=like.device,dtype=torch.bool)
        self.active=torch.zeros_like(self.edge)
        self.inside=torch.zeros_like(self.edge)
        self.valid=torch.zeros_like(self.edge)
        self.ready=False

    def build(self,r):
        r=r.detach().cpu();s=self.storage
        center=r[[a for a,b in s.rim]].mean(0)
        polys=[[[r[a],r[b],center] for a,b in s.rim] for _ in s.bounds]
        arrays=[torch.zeros_like(self.p,device='cpu') for _ in range(2)]
        pp,qq=arrays
        edge,active,inside,valid=[torch.zeros_like(self.edge,device='cpu') for _ in range(4)]
        for stage in range(4):
            axis=s.other[stage//2];lower=stage%2==0
            for label in range(len(s.bounds)):
                value=s.bounds[label][stage//2][stage%2]
                for tri,poly in enumerate(polys[label]):
                    count=len(poly)
                    if not count:continue
                    flags=[bool(v[axis]>=value) if lower else bool(v[axis]<=value) for v in poly]
                    active[stage,label,tri,:count]=True
                    inside[stage,label,tri,:count]=torch.tensor(flags)
                    out=[];descriptors=[]
                    for j,(a,b) in enumerate(zip(poly,poly[1:]+poly[:1])):
                        nxt=(j+1)%count
                        if flags[j]:out.append(a);descriptors.append((j,j,False))
                        if flags[j]!=flags[nxt]:
                            out.append(a+(value-a[axis])/(b[axis]-a[axis])*(b-a))
                            descriptors.append((j,nxt,True))
                    if len(out)>MAX_VERTICES:raise ValueError('clipped polygon exceeds padded capacity')
                    for j,(a,b,cross) in enumerate(descriptors):
                        pp[stage,label,tri,j]=a;qq[stage,label,tri,j]=b
                        edge[stage,label,tri,j]=cross;valid[stage,label,tri,j]=True
                    polys[label][tri]=out
        for target,value in zip((self.p,self.q,self.edge,self.active,self.inside,self.valid),
                                (pp,qq,edge,active,inside,valid)):
            target.copy_(value)
        self.ready=True;s.topology_builds+=1


class GPUTractStorage(TractStorage):
    def __init__(self,cap,domain):
        super().__init__(cap,domain)
        if not self.nodes.is_cuda:raise ValueError('GPU tract storage requires actual CUDA tensors')
        self.reference=TractStorage(cap,domain)
        self.rim_gpu=torch.tensor(self.rim,device=self.nodes.device,dtype=torch.int64)
        self.bounds_tensor=torch.tensor(self.bounds,device=self.nodes.device,dtype=torch.float64)
        self.expected=(self.bounds_tensor[:,0,1]-self.bounds_tensor[:,0,0])*(self.bounds_tensor[:,1,1]-self.bounds_tensor[:,1,0])
        self.topology_builds=0;self.fallback_count=0
        self.contexts=[_Topology(self) for _ in range(4)]
        self.gradient=torch.func.grad(self._scalar,argnums=0)
        self.hessian=torch.func.jacrev(self.gradient,argnums=0)
        self.identity=torch.eye(self.nodes.numel()*3,device=self.nodes.device,dtype=torch.float64)

    def _values(self,r,context):
        edge=r[self.rim_gpu]
        center=r[self.rim_gpu[:,0]].mean(0)
        tri=torch.cat((edge,center.expand(len(self.rim),1,3)),1)
        points=torch.cat((tri,r.new_zeros((len(self.rim),MAX_VERTICES-3,3))),1)[None].expand(len(self.bounds),-1,-1,-1)
        guard=torch.ones((),device=r.device,dtype=torch.bool)
        for stage in range(4):
            axis=self.other[stage//2]
            value=self.bounds_tensor[:,stage//2,stage%2,None,None]
            observed=points[...,axis]>=value if stage%2==0 else points[...,axis]<=value
            guard=guard & ((observed==context.inside[stage]) | ~context.active[stage]).all()
            p=points.gather(2,context.p[stage,...,None].expand(-1,-1,-1,3))
            q=points.gather(2,context.q[stage,...,None].expand(-1,-1,-1,3))
            denominator=q[...,axis]-p[...,axis]
            cross=context.edge[stage]
            safe=torch.where(cross & (denominator!=0),denominator,torch.ones_like(denominator))
            intersection=p+(value-p[...,axis])[...,None]/safe[...,None]*(q-p)
            points=torch.where(cross[...,None],intersection,p)
            points=torch.where(context.valid[stage,...,None],points,torch.zeros_like(points))
        p0=points[:,:,0,None,:];p1=points[:,:,1:-1,:];p2=points[:,:,2:,:]
        a=p1-p0;b=p2-p0
        area=(a[...,self.other[0]]*b[...,self.other[1]]-a[...,self.other[1]]*b[...,self.other[0]])/2
        valid=context.valid[-1,:, :,0,None] & context.valid[-1,:,:,1:-1] & context.valid[-1,:,:,2:]
        area=torch.where(valid,area,torch.zeros_like(area))
        height=self.top_coordinate_cm-(p0[...,self.domain.axis]+p1[...,self.domain.axis]+p2[...,self.domain.axis])/3
        return self.fixed_volume_ml+(area*height).sum(),area.sum((1,2)),guard

    def _scalar(self,r,context):return self._values(r,context)[0]

    def _ensure(self,r,context):
        if not context.ready:context.build(r)
        for attempt in range(2):
            value,covered,guard=self._values(r,context)
            if bool(guard):
                if not bool((abs(covered-self.expected)<=1e-9*self.expected.clamp_min(1.)).all()):
                    raise ValueError('moving basal cap no longer covers the fixed tract section')
                return True,value
            context.build(r)
        self.fallback_count+=1
        return False,None

    def rim_value(self,x):return x[self.nodes].detach()

    def stable_increment(self,r0,r1):
        """Algebraic volume difference without subtracting two large volumes.

        Propagate endpoint differences through the SAME clipping expression.
        Product and quotient difference identities avoid cancellation when
        the moving cap travels only a few micrometres. A clipping topology
        change explicitly returns None for the original general fallback.
        """
        old,new=self.contexts[:2]
        same=((old.inside==new.inside)&(old.active==new.active)
              &(old.valid==new.valid)&(old.edge==new.edge)).all()
        if not bool(same):return None
        dr=r1-r0
        edge=r0[self.rim_gpu];dedge=dr[self.rim_gpu]
        center=r0[self.rim_gpu[:,0]].mean(0)
        dcenter=dr[self.rim_gpu[:,0]].mean(0)
        tri=torch.cat((edge,center.expand(len(self.rim),1,3)),1)
        dtri=torch.cat((dedge,dcenter.expand(len(self.rim),1,3)),1)
        padding=r0.new_zeros((len(self.rim),MAX_VERTICES-3,3))
        points=torch.cat((tri,padding),1)[None].expand(len(self.bounds),-1,-1,-1)
        delta=torch.cat((dtri,padding),1)[None].expand(len(self.bounds),-1,-1,-1)
        for stage in range(4):
            axis=self.other[stage//2]
            value=self.bounds_tensor[:,stage//2,stage%2,None,None]
            ids=old.p[stage,...,None].expand(-1,-1,-1,3)
            jds=old.q[stage,...,None].expand(-1,-1,-1,3)
            p=points.gather(2,ids);q=points.gather(2,jds)
            dp=delta.gather(2,ids);dq=delta.gather(2,jds)
            cross=old.edge[stage]
            direction=q-p;ddirection=dq-dp
            den=direction[...,axis];dden=ddirection[...,axis]
            den=torch.where(cross,den,torch.ones_like(den))
            end_den=torch.where(cross,den+dden,torch.ones_like(den))
            ratio=(value-p[...,axis])/den
            # (a1/b1-a0/b0)=(delta_a-(a0/b0)*delta_b)/b1.
            dratio=(-dp[...,axis]-ratio*dden)/end_den
            intersection=p+ratio[...,None]*direction
            dinters=dp+(ratio+.5*dratio)[...,None]*ddirection+(direction+.5*ddirection)*dratio[...,None]
            points=torch.where(cross[...,None],intersection,p)
            delta=torch.where(cross[...,None],dinters,dp)
            valid=old.valid[stage,...,None]
            points=torch.where(valid,points,torch.zeros_like(points))
            delta=torch.where(valid,delta,torch.zeros_like(delta))
        p0=points[:,:,0,None,:];p1=points[:,:,1:-1,:];p2=points[:,:,2:,:]
        d0=delta[:,:,0,None,:];d1=delta[:,:,1:-1,:];d2=delta[:,:,2:,:]
        a=p1-p0;b=p2-p0;da=d1-d0;db=d2-d0
        i,j=self.other
        area=(a[...,i]*b[...,j]-a[...,j]*b[...,i])/2
        am=a+.5*da;bm=b+.5*db
        darea=(am[...,i]*db[...,j]-am[...,j]*db[...,i]
               +da[...,i]*bm[...,j]-da[...,j]*bm[...,i])/2
        axis=self.domain.axis
        height=self.top_coordinate_cm-(p0[...,axis]+p1[...,axis]+p2[...,axis])/3
        dheight=-(d0[...,axis]+d1[...,axis]+d2[...,axis])/3
        valid=old.valid[-1,:,:,0,None]&old.valid[-1,:,:,1:-1]&old.valid[-1,:,:,2:]
        difference=(area+.5*darea)*dheight+(height+.5*dheight)*darea
        return torch.where(valid,difference,torch.zeros_like(difference)).sum()

    def volume(self,x):
        with torch.no_grad():
            valid,value=self._ensure(self.rim_value(x),self.contexts[3])
            return float(value) if valid else self.reference.volume(x)

    def discrete(self,old,x):
        r0,r1=self.rim_value(old),self.rim_value(x)
        c=self.cached
        if (c is not None and not c.get('reference',False)
                and torch.equal(r0,c['r0']) and torch.equal(r1,c['r1'])):
            out=torch.zeros_like(x);out[self.nodes]=c['gradient'];return out
        mid=(r0+r1)/2
        good0,v0=self._ensure(r0,self.contexts[0])
        good1,v1=self._ensure(r1,self.contexts[1])
        goodm,_=self._ensure(mid,self.contexts[2])
        if not (good0 and good1 and goodm):
            self.cached=dict(reference=True)
            return self.reference.discrete(old,x)
        gm=self.gradient(mid,self.contexts[2]).detach()
        dy=r1-r0;square=dy.square().sum();work=(gm*dy).sum()
        increment=self.stable_increment(r0,r1)
        stable=increment is not None
        if stable:
            numerator=increment-work
            # The increment identities have the scale of the increment,
            # not the two ~10 mL endpoint volumes. Use their roundoff scale
            # to avoid an artificial on/off jump in the secant correction.
            floor=64*torch.finfo(r0.dtype).eps*(abs(increment)+abs(work)).clamp_min(torch.finfo(r0.dtype).tiny)
        else:
            increment=v1-v0;numerator=increment-work
            floor=64*torch.finfo(r0.dtype).eps*torch.maximum(torch.maximum(abs(v0),abs(v1)),v0.new_tensor(1.))
        corrected=(square>1e-20) & (abs(numerator)>floor)
        safe=torch.where(square>1e-20,square,torch.ones_like(square))
        coefficient=torch.where(corrected,numerator/safe,torch.zeros_like(square))
        g=gm+coefficient*dy
        self.cached=dict(reference=False,r0=r0,r1=r1,gm=gm,dy=dy,square=square,safe_square=safe,
            numerator=numerator,coefficient=coefficient,corrected=corrected,gradient=g,
            volume_old=v0,volume_new=v1,volume_increment=increment,stable_increment=stable,jacobian=None)
        out=torch.zeros_like(x);out[self.nodes]=g;return out

    def derivative(self,old,x):
        self.discrete(old,x);c=self.cached
        if c.get('reference',False):return self.reference.derivative(old,x)
        if c['jacobian'] is None:
            mid=(c['r0']+c['r1'])/2
            goodm,_=self._ensure(mid,self.contexts[2]);good1,_=self._ensure(c['r1'],self.contexts[1])
            if not (goodm and good1):return self.reference.derivative(old,x)
            h=self.hessian(mid,self.contexts[2]).reshape(mid.numel(),mid.numel()).detach()
            g1=self.gradient(c['r1'],self.contexts[1]).reshape(-1).detach()
            dy=c['dy'].reshape(-1);gm=c['gm'].reshape(-1);sq=c['safe_square']
            nv=g1-gm-.5*h@dy
            addition=c['coefficient']*self.identity+torch.outer(dy,nv)/sq-2*c['numerator']*torch.outer(dy,dy)/sq**2
            c['jacobian']=.5*h+torch.where(c['corrected'],addition,torch.zeros_like(addition))
        matrix=c['jacobian']
        def action(v):
            out=torch.zeros_like(v);local=v[self.nodes]
            out[self.nodes]=(matrix@local.reshape(matrix.shape[1],-1)).reshape_as(local)
            return out
        return action

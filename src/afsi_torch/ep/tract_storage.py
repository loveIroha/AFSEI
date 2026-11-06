"""Geometric storage between a moving P1 basal cap and fixed tract cut planes.

Only the small basal rim is used for CPU clipping/autodifferentiation. Fluid,
EP and constitutive solves remain on their original GPU backends. The exact
clipped height integral is paired with a secant discrete gradient, so its
finite-step work equals the actual volume increment, including cap motion.
"""
import torch


def _clip(poly, axis, value, high):
    if not poly:return []
    out=[]
    for p,q in zip(poly,poly[1:]+poly[:1]):
        ip=bool(p[axis].detach()>=value) if high else bool(p[axis].detach()<=value)
        iq=bool(q[axis].detach()>=value) if high else bool(q[axis].detach()<=value)
        if ip:out.append(p)
        if ip!=iq:out.append(p+(value-p[axis])/(q[axis]-p[axis])*(q-p))
    return out


class TractStorage:
    def __init__(self,cap,domain):
        self.cap,self.domain=cap,domain
        self.nodes=torch.unique(cap.rim)
        self.rim=torch.searchsorted(self.nodes,cap.rim).cpu().tolist()
        self.other=[a for a in range(3) if a!=domain.axis]
        self.bounds=getattr(domain,'storage_bounds',domain.bounds)
        self.cached=None

    @property
    def top_coordinate_cm(self):
        return getattr(self.domain,'storage_top_coordinate_cm',self.domain.metadata['cut_coordinate_cm'])

    @property
    def fixed_volume_ml(self):return getattr(self.domain,'storage_fixed_volume_ml',0.)

    def scalar(self,r):
        center=r[[a for a,b in self.rim]].mean(0)
        volume=r.new_tensor(self.fixed_volume_ml)
        for bounds in self.bounds:
            covered=r.new_zeros(())
            for a,b in self.rim:
                poly=[r[a],r[b],center]
                for axis,(lo,hi) in zip(self.other,bounds):
                    poly=_clip(poly,axis,lo,True)
                    poly=_clip(poly,axis,hi,False)
                for j in range(1,len(poly)-1):
                    tri=torch.stack((poly[0],poly[j],poly[j+1]))
                    edge=tri[1:,self.other]-tri[0,self.other]
                    area=(edge[0,0]*edge[1,1]-edge[0,1]*edge[1,0])/2
                    covered=covered+area
                    volume=volume+area*(self.top_coordinate_cm-tri[:,self.domain.axis].mean())
            expected=(bounds[0][1]-bounds[0][0])*(bounds[1][1]-bounds[1][0])
            if abs(float(covered.detach())-expected)>1e-9*max(expected,1.):
                raise ValueError('moving basal cap no longer covers the fixed tract section')
        return volume

    def rim_value(self,x):return x[self.nodes].detach().cpu()

    def volume(self,x):
        with torch.no_grad():return float(self.scalar(self.rim_value(x)))

    def discrete(self,old,x):
        r0,r1=self.rim_value(old),self.rim_value(x)
        if self.cached is None or not torch.equal(r0,self.cached['r0']) or not torch.equal(r1,self.cached['r1']):
            with torch.enable_grad():
                mid=((r0+r1)/2).requires_grad_(True)
                vm=self.scalar(mid)
                gm=torch.autograd.grad(vm,mid)[0].detach()
            v0,v1=self.scalar(r0).detach(),self.scalar(r1).detach()
            dy=r1-r0; square=dy.square().sum()
            numerator=v1-v0-(gm*dy).sum()
            # Near equilibrium, subtracting two ~10 mL volumes is only
            # accurate to their floating-point scale. Dividing that noise
            # by a tiny displacement makes the gradient/Jacobian erratic.
            # Keep the midpoint gradient when the secant defect is within
            # roundoff; its chain-rule defect is then roundoff as well.
            floor=64*torch.finfo(r0.dtype).eps*max(abs(float(v0)),abs(float(v1)),1.)
            corrected=float(square)>1e-20 and abs(float(numerator))>floor
            coefficient=numerator/square if corrected else square.new_zeros(())
            g=gm+coefficient*dy
            self.cached=dict(r0=r0,r1=r1,gm=gm,dy=dy,square=square,numerator=numerator,
                coefficient=coefficient,corrected=corrected,gradient=g,volume_old=v0,volume_new=v1,jacobian=None)
        out=torch.zeros_like(x)
        out[self.nodes]=self.cached['gradient'].to(x.device)
        return out

    def derivative(self,old,x):
        self.discrete(old,x)
        c=self.cached
        if c['jacobian'] is None:
            with torch.enable_grad():
                mid=((c['r0']+c['r1'])/2).requires_grad_(True)
                h=torch.autograd.functional.hessian(self.scalar,mid,vectorize=True).reshape(mid.numel(),mid.numel())
                end=c['r1'].clone().requires_grad_(True)
                g1=torch.autograd.grad(self.scalar(end),end)[0].reshape(-1)
            j=.5*h
            if c['corrected']:
                dy=c['dy'].reshape(-1);gm=c['gm'].reshape(-1)
                nv=g1-gm-.5*h@dy
                j=j+c['coefficient']*torch.eye(len(dy),dtype=j.dtype)
                j=j+torch.outer(dy,nv)/c['square']-2*c['numerator']*torch.outer(dy,dy)/c['square']**2
            c['jacobian']=j.detach().to(x.device)
        matrix=c['jacobian']
        def action(v):
            out=torch.zeros_like(v)
            local=v[self.nodes]
            out[self.nodes]=(matrix@local.reshape(matrix.shape[1],-1)).reshape_as(local)
            return out
        return action

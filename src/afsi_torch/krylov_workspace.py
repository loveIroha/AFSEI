"""Caller-owned GMRES scratch, reusable across restarts and Newton steps.

No global pool: concurrent/nested solvers use distinct workspaces. Published
solutions remain independently owned. V/Z need no clearing because Arnoldi
writes each column before using it; the small accumulated H is reset.
"""
import torch


class KrylovWorkspace:
    def __init__(self,*,fused=True):
        self.fused=fused; self.key=None; self.capacity=0

    def ensure(self,like,width):
        key=(like.numel(),like.dtype,like.device)
        if self.key==key and self.capacity>=width: return
        self.key,self.capacity=key,width
        n=like.numel()
        self.V=like.new_empty((n,width+1)); self.Z=like.new_empty((n,width))
        self.H=like.new_empty((width+1,width)); self.rotated=torch.empty_like(self.H)
        self.cosines=like.new_empty(width); self.sines=torch.empty_like(self.cosines)
        self.g=like.new_empty(width+1); self.column=torch.empty_like(self.g)
        self.estimate=like.new_empty(())
        self.base=like.new_empty(n); self.product=torch.empty_like(self.base)
        self.work=torch.empty_like(self.base); self.coefficients=like.new_empty(width)

    def reset(self,width,beta):
        self.H[:width+1,:width].zero_()
        self.g[:width+1].zero_(); self.g[0].copy_(beta)

    @property
    def storage_bytes(self):
        if self.key is None: return 0
        return sum(v.numel()*v.element_size() for v in (
            self.V,self.Z,self.H,self.rotated,self.cosines,self.sines,self.g,
            self.column,self.estimate,self.base,self.product,self.work,self.coefficients))

    def rotate(self,H,rotated,cosines,sines,g,j):
        if H.is_cuda and self.fused:
            from ._triton_givens import rotate
            rotate(H,rotated,cosines,sines,g,self.estimate,j)
        else:
            column=self.column[:j+2]; column.copy_(H[:j+2,j])
            for k in range(j):
                a,c=column[k].clone(),column[k+1].clone()
                column[k]=cosines[k]*a+sines[k]*c
                column[k+1]=-sines[k]*a+cosines[k]*c
            radius=torch.hypot(column[j],column[j+1])
            denominator=radius.clamp_min(torch.finfo(H.dtype).tiny)
            cosines[j],sines[j]=column[j]/denominator,column[j+1]/denominator
            column[j]=radius; column[j+1]=0
            rotated[:j+2,j].copy_(column)
            previous=g[j].clone()
            g[j+1]=-sines[j]*previous; g[j]=cosines[j]*previous
            self.estimate.copy_(g[j+1].abs())
        return self.estimate

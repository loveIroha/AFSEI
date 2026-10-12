"""One GPU launch for the original sequential Givens column update."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=['column'])
def _rotate(H,R,C,S,G,E,column,HS0:tl.constexpr,HS1:tl.constexpr,
            RS0:tl.constexpr,RS1:tl.constexpr,TINY:tl.constexpr):
    for k in range(column+2):
        tl.store(R+k*RS0+column*RS1,tl.load(H+k*HS0+column*HS1))
    for k in range(column):
        a=tl.load(R+k*RS0+column*RS1)
        b=tl.load(R+(k+1)*RS0+column*RS1)
        c=tl.load(C+k); s=tl.load(S+k)
        tl.store(R+k*RS0+column*RS1,c*a+s*b)
        tl.store(R+(k+1)*RS0+column*RS1,-s*a+c*b)
    a=tl.load(R+column*RS0+column*RS1)
    b=tl.load(R+(column+1)*RS0+column*RS1)
    radius=libdevice.hypot(a,b)
    denominator=tl.maximum(radius,TINY)
    cosine=a/denominator; sine=b/denominator
    tl.store(C+column,cosine); tl.store(S+column,sine)
    tl.store(R+column*RS0+column*RS1,radius)
    tl.store(R+(column+1)*RS0+column*RS1,0.)
    previous=tl.load(G+column)
    next_value=-sine*previous
    tl.store(G+column+1,next_value); tl.store(G+column,cosine*previous)
    tl.store(E,tl.abs(next_value))


def rotate(H,R,cosines,sines,g,estimate,column):
    import torch
    _rotate[(1,)](H,R,cosines,sines,g,estimate,column,*H.stride(),*R.stride(),
                   torch.finfo(H.dtype).tiny,num_warps=1,enable_fp_fusion=False)

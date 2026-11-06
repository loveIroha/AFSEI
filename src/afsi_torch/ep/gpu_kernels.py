"""Buffered FP32/FP64 EP kernels; no changes to quadrature or time integration."""
import torch
import triton
import triton.language as tl


@triton.jit
def _csr(R, C, W, X, Y, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)*16+tl.arange(0,16)
    j = tl.load(R + row,row<N,0)[:,None] + tl.arange(0, B)[None,:]
    end = tl.load(R + row + 1,row<N,0)[:,None]
    mask=(row[:,None]<N)&(j<end)
    col = tl.load(C + j, mask, 0)
    w = tl.load(W + j, mask, 0)
    x = tl.load(X + col, mask, 0)
    tl.store(Y + row, tl.sum(w * x, 1),row<N)


class CSRAction:
    def __init__(self, matrix, *, backend='torch'):
        self.matrix = matrix
        if backend not in ('torch','triton'):raise ValueError('unknown CSR action backend')
        self.backend=backend
        self.rows, self.cols, self.values = matrix.crow_indices(), matrix.col_indices(), matrix.values()
        self.width = (triton.next_power_of_2(int((self.rows[1:]-self.rows[:-1]).max()))
                      if backend=='triton' else None)

    def __call__(self, x, out):
        if self.backend=='torch':
            torch.mv(self.matrix,x,out=out)
            return out
        _csr[(triton.cdiv(self.matrix.shape[0],16),)](self.rows, self.cols, self.values, x, out,
                                     self.matrix.shape[0], self.width, num_warps=4)
        return out


@triton.jit
def _rhs(V, I, S, D, KV, OUT, R0, N: tl.constexpr, A: tl.constexpr,
         CHI: tl.constexpr, T: tl.constexpr, NODAL: tl.constexpr, B: tl.constexpr):
    j=tl.program_id(0)*B+tl.arange(0,B)
    mask=j<N
    v=tl.load(V+j,mask,0); i=tl.load(I+j,mask,0); s=tl.load(S+j,mask,0)
    z=A*v+CHI*(s-i)
    if NODAL:
        d=tl.load(D+j,mask,0); kv=tl.load(KV+j,mask,0)
        z=d*z-T*kv
        tl.store(R0+j,CHI*d*(s-i)-kv,mask)
    tl.store(OUT+j,z,mask)


class BufferedRHS:
    """Owns buffers: returned RHS valid until the next call, never aliases V/I/S."""
    def __init__(self, mass, diagonal, stiffness, alpha, chi, theta, nodal):
        self.diagonal, self.alpha, self.chi = diagonal, alpha, chi
        self.theta, self.nodal = theta, nodal
        self.mass = CSRAction(mass) if not nodal else None
        self.stiffness = CSRAction(stiffness)
        self.source, self.kv, self.out, self.r0 = [torch.empty_like(diagonal) for _ in range(4)]

    @property
    def initial_residual(self):
        # Valid only for x0=V and a previously computed K*V (CN).
        return self.r0 if self.nodal and self.theta != 1 else None

    def __call__(self, v, i, s):
        if self.theta != 1:
            self.stiffness(v, self.kv)
        else:
            self.kv.zero_()
        target=self.out if self.nodal else self.source
        _rhs[(triton.cdiv(v.numel(),256),)](v,i,s,self.diagonal,self.kv,target,self.r0,v.numel(),
            self.alpha,self.chi,1-self.theta,self.nodal,256,enable_fp_fusion=False)
        if not self.nodal:
            self.mass(self.source,self.out)
            self.out.add_(self.kv,alpha=-(1-self.theta))
        return self.out


@triton.jit
def _update(X,R,P,AP,ALPHA,N:tl.constexpr,B:tl.constexpr):
    j=tl.program_id(0)*B+tl.arange(0,B); m=j<N
    a=tl.load(ALPHA)
    tl.store(X+j,tl.load(X+j,m,0)+a*tl.load(P+j,m,0),m)
    tl.store(R+j,tl.load(R+j,m,0)-a*tl.load(AP+j,m,0),m)


def update(x,r,p,ap,alpha):
    _update[(triton.cdiv(x.numel(),256),)](x,r,p,ap,alpha,x.numel(),256,enable_fp_fusion=False)


@triton.jit
def _update_stats(X,R,Z,P,AP,D,ALPHA,RR,RZ,N:tl.constexpr,B:tl.constexpr):
    j=tl.program_id(0)*B+tl.arange(0,B); m=j<N
    a=tl.load(ALPHA)
    r=tl.load(R+j,m,0)-a*tl.load(AP+j,m,0)
    z=r/tl.load(D+j,m,1)
    tl.store(X+j,tl.load(X+j,m,0)+a*tl.load(P+j,m,0),m)
    tl.store(R+j,r,m);tl.store(Z+j,z,m)
    tl.store(RR+tl.program_id(0),tl.sum(r*r,0))
    tl.store(RZ+tl.program_id(0),tl.sum(r*z,0))


def update_stats(x,r,z,p,ap,diagonal,alpha,rr,rz):
    _update_stats[(triton.cdiv(x.numel(),256),)](x,r,z,p,ap,diagonal,alpha,rr,rz,
        x.numel(),256,enable_fp_fusion=False)


@triton.jit
def _direction(P,Z,BETA,ACTIVE,N:tl.constexpr,B:tl.constexpr):
    j=tl.program_id(0)*B+tl.arange(0,B);m=j<N
    value=tl.load(Z+j,m,0)+tl.load(BETA)*tl.load(P+j,m,0)
    tl.store(P+j,tl.where(tl.load(ACTIVE),value,0.),m)


def direction(p,z,beta,active):
    _direction[(triton.cdiv(p.numel(),256),)](p,z,beta,active,p.numel(),256,enable_fp_fusion=False)

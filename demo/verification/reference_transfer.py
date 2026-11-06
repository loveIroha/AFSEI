"""Independent dense dual FE/IB coupling and spectral fluid mobility."""
from itertools import product
import numpy as np,basix
from scipy.special import roots_jacobi
from scipy.linalg import solve


def delta(r):
    r=np.abs(r)
    return np.where(r<1,(3-2*r+np.sqrt(np.maximum(0,1+4*r-4*r*r)))/8,
                    np.where(r<2,(5-2*r-np.sqrt(np.maximum(0,-7+12*r-4*r*r)))/8,0.))


class ReferenceTransfer:
    def __init__(self,X,cells,flow):
        self.X,self.cells,self.flow=X,cells,flow;self.n=len(X);self.rules={}
        self.vol=np.abs(np.linalg.det((X[cells[:,1:]]-X[cells[:,:1]]).transpose(0,2,1)))/6
        M=np.zeros((self.n,self.n))
        for C,v in zip(cells,self.vol):M[np.ix_(C,C)]+=v/20*(np.ones((4,4))+np.eye(4))
        self.inverse=solve(M,np.eye(self.n),assume_a='pos')

    def rule(self,n):
        if n not in self.rules:
            if n<=8:
                q,w=basix.make_quadrature(basix.CellType.tetrahedron,2*n-1,rule=basix.QuadratureType.xiao_gimbutas)
            else:
                tables=[]
                for alpha in (2,1,0):
                    x,wj=roots_jacobi(n,alpha,0);tables.append(((x+1)/2,wj/2**(alpha+1)))
                u,v,z=np.meshgrid(*[t[0] for t in tables],indexing='ij')
                a,b,d=np.meshgrid(*[t[1] for t in tables],indexing='ij')
                q=np.stack((u,(1-u)*v,(1-u)*(1-v)*z),-1).reshape(-1,3);w=(a*b*d).ravel()
            self.rules[n]=(np.column_stack((1-q.sum(1),q)),w)
        return self.rules[n]

    def prepare(self,x):
        from itertools import combinations
        pairs=np.array(list(combinations(range(4),2)))
        edge=x[self.cells[:,pairs[:,0]]]-x[self.cells[:,pairs[:,1]]]
        orders=np.maximum(2,np.ceil(2*np.linalg.norm(edge,axis=-1).max(1)/self.flow.h.min()).astype(int))
        self.A=[np.zeros((self.n,np.prod(s))) for s in self.flow.face_shapes]
        for order in np.unique(orders):
            ids=np.flatnonzero(orders==order);C=self.cells[ids];N,w=self.rule(int(order))
            pts=np.einsum('qa,eai->eqi',N,x[C]).reshape(-1,3)
            W=(6*self.vol[ids,None]*w).ravel()
            nodal=[np.repeat(C[:,j],len(w)) for j in range(4)]
            nw=[np.tile(N[:,j],len(C))*W for j in range(4)]
            for k in range(3):
                scaled=(pts-self.flow.origins[k])/self.flow.h;base=np.floor(scaled-1).astype(int)
                shape=self.flow.face_shapes[k]
                for offset in product(range(4),repeat=3):
                    ind=base+np.array(offset)
                    if (ind<0).any() or (ind>=np.array(shape)).any():raise ValueError('independent IB support left box')
                    phi=np.prod(delta(scaled-ind),axis=1)
                    face=(ind[:,0]*shape[1]+ind[:,1])*shape[2]+ind[:,2]
                    for j in range(4):np.add.at(self.A[k],(nodal[j],face),phi*nw[j])
        self.I=[self.inverse@A for A in self.A]
        self.S=[A.T@self.inverse/np.prod(self.flow.h) for A in self.A]
        self.orders=orders

    def check_support(self,x):
        scaled=(x-self.flow.origin)/self.flow.h
        if (scaled>=2.125).all() and (scaled<np.array(self.flow.shape)-2.125).all():return
        from itertools import combinations
        pairs=np.array(list(combinations(range(4),2)))
        lengths=np.linalg.norm(x[self.cells[:,pairs[:,0]]]-x[self.cells[:,pairs[:,1]]],axis=-1).max(1)
        orders=np.maximum(2,np.ceil(2*lengths/self.flow.h.min()).astype(int))
        for order in np.unique(orders):
            C=self.cells[orders==order];N,_=self.rule(int(order));p=np.einsum('qa,eai->eqi',N,x[C])
            q=(p-self.flow.origin)/self.flow.h
            if not np.isfinite(q).all() or (q<2).any() or (q>=np.array(self.flow.shape)-2).any():
                raise ValueError('independent endpoint IB support left box')

    def interpolate(self,v):return np.column_stack([self.I[k]@v[k].ravel() for k in range(3)])
    def spread(self,f):return [(self.S[k]@f[:,k]).reshape(self.flow.face_shapes[k]) for k in range(3)]

    def mobility(self,batch=16):
        # Component-major vectors. Reference matrices are constructed here,
        # never supplied by AFSEI or used by its production solver.
        H=np.empty((3*self.n,3*self.n))
        for component in range(3):
            for lo in range(0,self.n,batch):
                hi=min(lo+batch,self.n);b=hi-lo;rhs=self.flow.zeros(b)
                rhs[component][:]=(self.flow.dt/self.flow.rho*self.S[component][:,lo:hi]).reshape((*self.flow.face_shapes[component],b))
                v,_=self.flow.apply(rhs)
                H[:,component*self.n+lo:component*self.n+hi]=np.concatenate([self.I[k]@v[k].reshape(-1,b) for k in range(3)],axis=0)
        return H

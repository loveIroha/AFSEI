"""Independent NumPy/SciPy BE MAC solve with arbitrary trailing RHS batches."""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.fft import dct,idct,dstn,idstn


class ReferenceFlow:
    def __init__(self,c):
        f=c['fluid'];self.shape=tuple(f['shape']);self.lengths=np.array(f['lengths_cm'])
        self.origin=np.array(f['origin_cm']);self.h=self.lengths/np.array(self.shape)
        self.dt=c['time']['fsi_dt_s'];self.rho=f['density_g_per_cm3'];self.mu=f['viscosity_dyn_s_per_cm2']
        self.face_shapes=[tuple(n+int(j==i) for j,n in enumerate(self.shape)) for i in range(3)]
        self.origins=[self.origin+np.where(np.arange(3)==i,0,self.h/2) for i in range(3)]
        self.coords=[np.stack(np.meshgrid(*[self.origins[i][j]+self.h[j]*np.arange(self.face_shapes[i][j]) for j in range(3)],indexing='ij'),-1) for i in range(3)]
        self.helmholtz=[]
        for i,s in enumerate(self.face_shapes):
            lam=np.zeros(s)
            for j,n in enumerate(s):
                eigen=4*np.sin(np.pi*np.arange(n)/(2*(n-1 if j==i else n)))**2/self.h[j]**2
                sh=[1]*3;sh[j]=n;lam+=eigen.reshape(sh)
            self.helmholtz.append(1+self.dt*self.mu/self.rho*lam)
        self.poisson=np.zeros(self.shape)
        for j,n in enumerate(self.shape):
            eigen=4*np.sin(np.pi*(np.arange(n)+1)/(2*n))**2/self.h[j]**2
            sh=[1]*3;sh[j]=n;self.poisson+=eigen.reshape(sh)

    def zeros(self,batch=None):
        return [np.zeros(s+(() if batch is None else (batch,))) for s in self.face_shapes]

    def sample(self,value,points,component):
        index=(points-self.origins[component])/self.h
        return map_coordinates(value,np.moveaxis(index,-1,0),order=1,mode='nearest',prefilter=False)

    def advect(self,velocity):
        return [self.sample(velocity[i],p-self.dt*np.stack([velocity[i] if j==i else self.sample(velocity[j],p,j) for j in range(3)],-1),i) for i,p in enumerate(self.coords)]

    def apply(self,rhs):
        star=[];batched=rhs[0].ndim==4
        for i,b in enumerate(rhs):
            q=b.copy()
            for j in range(3):q=dct(q,type=1 if j==i else 2,axis=j,norm=None)
            q/=self.helmholtz[i][...,None] if batched else self.helmholtz[i]
            for j in reversed(range(3)):q=idct(q,type=1 if j==i else 2,axis=j,norm=None)
            star.append(q)
        div=sum(np.diff(star[i],axis=i)/self.h[i] for i in range(3))
        p=dstn(-self.rho/self.dt*div,type=2,norm='ortho',axes=(0,1,2))
        p/=self.poisson[...,None] if batched else self.poisson
        p=idstn(p,type=2,norm='ortho',axes=(0,1,2))
        velocity=[]
        for i in range(3):
            g=np.concatenate((2*np.take(p,[0],axis=i)/self.h[i],np.diff(p,axis=i)/self.h[i],-2*np.take(p,[-1],axis=i)/self.h[i]),axis=i)
            velocity.append(star[i]-self.dt/self.rho*g)
        return velocity,p

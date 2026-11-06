"""Independent NumPy/SciPy MAC and IB reference. No AFSEI/Torch imports."""
from pathlib import Path
from itertools import product
import json,numpy as np
from scipy.ndimage import map_coordinates
from scipy.fft import dct,idct,dstn,idstn
out=Path('/work/independent_mac_ib')
a=np.load(out/'mac_inputs.npz');shape=a['shape'];lengths=a['lengths'];h=lengths/shape
dt=float(a['dt']);rho=float(a['rho']);mu=float(a['mu'])
old=[a[f'old_{c}'] for c in range(3)]
origins=[np.where(np.arange(3)==c,0,h/2) for c in range(3)]
def sample(field,points,component):
    coords=(points-origins[component])/h
    return map_coordinates(field,np.moveaxis(coords,-1,0),order=1,mode='nearest',prefilter=False)
dep=[];star=[]
for c in range(3):
    coords=np.stack(np.meshgrid(*[origins[c][j]+h[j]*np.arange(old[c].shape[j]) for j in range(3)],indexing='ij'),-1)
    speed=np.stack([old[c] if k==c else sample(old[k],coords,k) for k in range(3)],-1)
    dep.append(sample(old[c],coords-dt*speed,c))
    b=dep[c]+dt/rho*a[f'density_{c}'];lam=np.zeros(b.shape)
    for j in range(3):
        n=b.shape[j];kind=1 if j==c else 2
        b=dct(b,type=kind,axis=j,norm=None)
        l=4*np.sin(np.pi*np.arange(n)/(2*(n-1 if j==c else n)))**2/h[j]**2
        ss=[1]*3;ss[j]=n;lam+=l.reshape(ss)
    b/=1+dt*mu/rho*lam
    for j in reversed(range(3)):b=idct(b,type=1 if j==c else 2,axis=j,norm=None)
    star.append(b)
div=sum(np.diff(star[c],axis=c)/h[c] for c in range(3))
eigen=np.zeros(shape)
for j in range(3):
    l=4*np.sin(np.pi*(np.arange(shape[j])+1)/(2*shape[j]))**2/h[j]**2
    ss=[1]*3;ss[j]=shape[j];eigen+=l.reshape(ss)
p=idstn(dstn(-rho/dt*div,type=2,norm='ortho')/eigen,type=2,norm='ortho')
v=[]
for c in range(3):
    lo=np.take(p,[0],axis=c)*2/h[c];hi=-np.take(p,[-1],axis=c)*2/h[c]
    grad=np.concatenate((lo,np.diff(p,axis=c)/h[c],hi),axis=c)
    v.append(star[c]-dt/rho*grad)
def rel(x,y):return float(np.linalg.norm(np.ravel(x-y))/max(np.linalg.norm(np.ravel(y)),1e-30))
mac=dict(departure_relative=max(rel(dep[c],a[f'departure_{c}']) for c in range(3)),
         velocity_relative=max(rel(v[c],a[f'velocity_{c}']) for c in range(3)),
         pressure_relative=rel(p,a['pressure']),max_divergence=float(abs(sum(np.diff(v[c],axis=c)/h[c] for c in range(3))).max()))

a=np.load(out/'ib_inputs.npz');X=a['X'];cells=a['cells'];x=a['x'];n=len(X)
M=np.zeros((n,n));vol=abs(np.linalg.det((X[cells[:,1:]]-X[cells[:,:1]]).transpose(0,2,1)))/6
for tet,w in zip(cells,vol):M[np.ix_(tet,tet)]+=w/20*(np.ones((4,4))+np.eye(4))
coeff=np.linalg.solve(M,a['force']);rhs=np.zeros((n,3));spread=[np.zeros_like(a[f'velocity_{c}']) for c in range(3)]
def delta(r):
    r=abs(r)
    return np.where(r<1,(3-2*r+np.sqrt(np.maximum(0,1+4*r-4*r*r)))/8,
                    np.where(r<2,(5-2*r-np.sqrt(np.maximum(0,-7+12*r-4*r*r)))/8,0))
for gi in range(int(a['group_count'])):
    C=a[f'cells_{gi}'];N=a[f'N_{gi}'];W=a[f'W_{gi}']
    pts=np.einsum('qa,eai->eqi',N,x[C]).reshape(-1,3)
    Fq=(np.einsum('qa,eai->eqi',N,coeff[C])*W[...,None]).reshape(-1,3)
    uq=np.zeros_like(pts)
    for c in range(3):
        scaled=(pts-origins[c])/h;base=np.floor(scaled-1).astype(int)
        for offset in product(range(4),repeat=3):
            ind=base+np.array(offset);phi=np.prod(delta(scaled-ind),axis=1)
            ii=tuple(ind.T)
            np.add.at(spread[c],ii,phi*Fq[:,c]/np.prod(h))
            uq[:,c]+=phi*a[f'velocity_{c}'][ii]
    local=np.einsum('qa,eq,eqi->eai',N,W,uq.reshape(*W.shape,3))
    np.add.at(rhs,C.ravel(),local.reshape(-1,3))
interpolated=np.linalg.solve(M,rhs)
ib=dict(spread_relative=max(rel(spread[c],a[f'spread_{c}']) for c in range(3)),
        interpolation_relative=rel(interpolated,a['interpolated']))
ps=float(np.sum(interpolated*a['force']));pf=float(np.prod(h)*sum(np.sum(spread[c]*a[f'velocity_{c}']) for c in range(3)))
ib['power_relative']=abs(ps-pf)/max(abs(ps),abs(pf),1e-30)
passed=(mac['departure_relative']<1e-10 and mac['velocity_relative']<1e-8 and mac['pressure_relative']<1e-8
        and max(ib.values())<1e-9)
report=dict(passed=passed,MAC=mac,IB=ib,
    scope='MAC: full 32x32x64 nonzero saved-velocity step with specified forcing; IB: 24 actual deformed tetrahedra, adaptive quadrature',
    shared_inputs='geometry, physical fields, quadrature nodes/weights only; no AFSEI assembled operator or kernel weights used by oracle',
    full_coupled_trajectory_reference=False)
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

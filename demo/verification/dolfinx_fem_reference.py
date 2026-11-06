"""Independent DOLFINx/UFL assembly; never imports AFSEI or its operators."""
from pathlib import Path
from itertools import permutations
import json,numpy as np,ufl,basix,basix.ufl
from mpi4py import MPI
from dolfinx import mesh,fem
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

out=Path('/work/independent_fem');a=np.load(out/'inputs.npz')
c=json.loads((out/'config.json').read_text());X=a['X'];cells=a['cells']
domain=mesh.create_mesh(MPI.COMM_SELF,cells,basix.ufl.element('Lagrange','tetrahedron',1,shape=(3,)),X)
V=fem.functionspace(domain,('Lagrange',1,(3,)));S=fem.functionspace(domain,('Lagrange',1))
def mapping(space):
    distance,ids=cKDTree(X).query(space.tabulate_dof_coordinates())
    assert distance.max()<1e-12 and len(np.unique(ids))==len(X)
    return ids
vm,sm=mapping(V),mapping(S)
def scalar(values):
    f=fem.Function(S);f.x.array[:]=values[sm];return f
u=fem.Function(V);u.x.array[:]=(a['x']-X)[vm].ravel()
probe=fem.Function(V);probe.x.array[:]=a['probe'][vm].ravel()
xs=scalar(a['xs']);rest=scalar(a['resting'])
# Same stated degree-five symmetric Keast rule, generated independently.
orbits=[((.25,)*4,.030283678097089182),((0.,1/3,1/3,1/3),.006026785714285717),
        ((8/11,1/11,1/11,1/11),.011645249086028967),
        ((.4334498464263357,)*2+(.0665501535736643,)*2,.010949141561386453)]
bary=[];weights=[]
for point,w in orbits:
    orbit=sorted(set(permutations(point)));bary.extend(orbit);weights.extend([w]*len(orbit))
dx=ufl.Measure('dx',domain=domain,metadata={'quadrature_rule':'custom','quadrature_points':np.array(bary)[:,1:],'quadrature_weights':np.array(weights)})
Q=fem.functionspace(domain,('DG',0,(3,)))
fiber=fem.Function(Q);centers=Q.tabulate_dof_coordinates()
r=centers[:,:2]-.7;r/=np.linalg.norm(r,axis=1)[:,None]
fiber.x.array[:]=np.column_stack((-r[:,1],r[:,0],np.zeros(len(r)))).ravel()
F=ufl.Identity(3)+ufl.grad(u);J=ufl.det(F);FiT=ufl.inv(F).T
E=c['solid']['young_modulus_dyn_cm2'];nu=c['solid']['poisson_ratio']
mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
Pp=mu*(F-FiT)+lam*ufl.ln(J)*FiT
coord=ufl.SpatialCoordinate(domain);z=coord[2]-.4
q=ufl.min_value(1.,ufl.max_value(0.,ufl.min_value(z,2-z)/c['active']['end_taper_cm']))
Ta=ufl.max_value(4.8e6*xs-rest,0.)*ufl.sin(np.pi*.5*q)**2
ff=F*fiber;stretch=ufl.sqrt(ufl.max_value(ufl.dot(ff,ff),1.))
Pa=Ta*(1+4.9*(stretch-1))*ufl.outer(ff,fiber)
facets=mesh.locate_entities_boundary(domain,2,lambda x:np.isclose(x[2],.4)|np.isclose(x[2],2.4))
tags=mesh.meshtags(domain,2,np.sort(facets),np.ones(len(facets),dtype=np.int32))
ds=ufl.Measure('ds',domain=domain,subdomain_data=tags,metadata={'quadrature_degree':2})
test=ufl.TestFunction(V)
L=-ufl.inner(Pp+Pa,ufl.grad(test))*dx-c['solid']['end_spring_dyn_cm3']*ufl.inner(u,test)*ds(1)
def vec(form):
    v=fem.assemble_vector(fem.form(form)).array.copy().reshape(-1,3)
    res=np.empty_like(v);res[vm]=v;return res
force=vec(L);jvp=vec(ufl.derivative(L,u,probe))

# Vertex quadrature for BOTH mass and the spatially varying conductivity.
verts=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
dxv=ufl.Measure('dx',domain=domain,metadata={'quadrature_rule':'custom','quadrature_points':verts,'quadrature_weights':np.full(4,1/24)})
sheet=fem.Function(V);sheet.x.array[:]=a['ep_sheet'][vm].ravel()
f=ufl.as_vector([0.,0.,1.]);normal=ufl.cross(f,sheet)
gf,gs,gn=c['ep']['parameters']['conductivity_us_per_cm']
D=gf*ufl.outer(f,f)+gs*ufl.outer(sheet,sheet)+gn*ufl.outer(normal,normal)
trial=ufl.TrialFunction(S);test=ufl.TestFunction(S)
def mat(form):
    A=fem.assemble_matrix(fem.form(form));A.scatter_reverse()
    local=A.to_scipy();inv=np.argsort(sm);return local[inv,:][:,inv].tocsr()
M=mat(trial*test*dxv);K=mat(ufl.inner(D*ufl.grad(trial),ufl.grad(test))*dxv)
alpha=.1/5e-5
rhs=M@(alpha*a['voltage']-.1*a['ionic'])-.5*(K@a['voltage'])
voltage=spsolve(alpha*M+.5*K,rhs)
def rel(x,y):return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
metrics=dict(force_relative=rel(force,a['force']),force_to_saved_gpu_relative=rel(force,a['force_saved_gpu']),
    force_jvp_relative=rel(jvp,a['force_jvp']),cn_solution_relative=rel(voltage,a['next_voltage']),
    cn_solution_max_mv=float(np.max(abs(voltage-a['next_voltage']))*1000))
for key,A in [('M',M),('K',K)]:
    B=csr_matrix((a[key+'_data'],a[key+'_indices'],a[key+'_indptr']),shape=A.shape)
    metrics[key+'_relative']=float(np.linalg.norm((A-B).data)/max(np.linalg.norm(B.data),1e-30))
limits=dict(force_relative=1e-9,force_to_saved_gpu_relative=1e-8,force_jvp_relative=1e-8,
            cn_solution_relative=1e-8,cn_solution_max_mv=1e-6,M_relative=1e-11,K_relative=1e-10)
report=dict(reference='DOLFINx 0.10 / UFL independent weak forms; SciPy direct CN solve',
    scope='same actual P1 tube at 120 ms: force, tangent action, nodal diffusion matrices and one frozen-source CN step',
    metrics=metrics,limits=limits,passed=all(metrics[k]<=v for k,v in limits.items()),
    independent_gpb_land_ode=False,full_coupled_trajectory_reference=False)
(out/'report.json').write_text(json.dumps(report,indent=2))
np.savez_compressed(out/'reference_outputs.npz',force=force,jvp=jvp,voltage=voltage)
print(json.dumps(report,indent=2))

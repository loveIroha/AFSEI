"""Independent DOLFINx/UFL P1 wall and nodal-CN operator, no AFSEI imports."""
from itertools import permutations
import numpy as np,ufl,basix.ufl
from mpi4py import MPI
from dolfinx import mesh,fem
from scipy.spatial import cKDTree
from scipy.sparse.linalg import splu


class ReferenceSolid:
    def __init__(self,X,cells,rest,c):
        self.X=X;self.c=c;self.n=len(X)
        self.domain=mesh.create_mesh(MPI.COMM_SELF,cells,basix.ufl.element('Lagrange','tetrahedron',1,shape=(3,)),X)
        self.V=fem.functionspace(self.domain,('Lagrange',1,(3,)));self.S=fem.functionspace(self.domain,('Lagrange',1))
        def ids(space):
            distance,index=cKDTree(X).query(space.tabulate_dof_coordinates())
            if distance.max()>1e-12 or len(np.unique(index))!=len(X):raise ValueError('oracle dof mapping mismatch')
            return index
        self.vm,self.sm=ids(self.V),ids(self.S)
        self.u=fem.Function(self.V);self.xs=fem.Function(self.S);ref=fem.Function(self.S);ref.x.array[:]=rest[self.sm]
        orbits=[((.25,)*4,.030283678097089182),((0.,1/3,1/3,1/3),.006026785714285717),
                ((8/11,1/11,1/11,1/11),.011645249086028967),
                ((.4334498464263357,)*2+(.0665501535736643,)*2,.010949141561386453)]
        bary=[];weights=[]
        for point,w in orbits:
            orbit=sorted(set(permutations(point)));bary.extend(orbit);weights.extend([w]*len(orbit))
        dx=ufl.Measure('dx',domain=self.domain,metadata={'quadrature_rule':'custom','quadrature_points':np.array(bary)[:,1:],'quadrature_weights':np.array(weights)})
        Q=fem.functionspace(self.domain,('DG',0,(3,)));f=fem.Function(Q)
        center=np.array(c['geometry']['center_xy_cm']);p=Q.tabulate_dof_coordinates()[:,:2]-center;p/=np.linalg.norm(p,axis=1)[:,None]
        f.x.array[:]=np.column_stack((-p[:,1],p[:,0],np.zeros(len(p)))).ravel()
        F=ufl.Identity(3)+ufl.grad(self.u);J=ufl.det(F);FiT=ufl.inv(F).T
        E=c['solid']['young_modulus_dyn_cm2'];nu=c['solid']['poisson_ratio'];mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
        Pp=mu*(F-FiT)+lam*ufl.ln(J)*FiT
        coord=ufl.SpatialCoordinate(self.domain);z=coord[2]-c['geometry']['axial_start_cm'];L=c['geometry']['length_cm']
        q=ufl.min_value(1.,ufl.max_value(0.,ufl.min_value(z,L-z)/c['active']['end_taper_cm']))
        Ta=ufl.max_value(4.8e6*self.xs-ref,0.)*ufl.sin(np.pi*.5*q)**2*c['active']['tension_scale']
        ff=F*f;stretch=ufl.sqrt(ufl.max_value(ufl.dot(ff,ff),1.))
        Pa=Ta*(1+4.9*(stretch-1))*ufl.outer(ff,f)
        z0=c['geometry']['axial_start_cm']
        facets=mesh.locate_entities_boundary(self.domain,2,lambda x:np.isclose(x[2],z0)|np.isclose(x[2],z0+L))
        tags=mesh.meshtags(self.domain,2,np.sort(facets),np.ones(len(facets),dtype=np.int32))
        ds=ufl.Measure('ds',domain=self.domain,subdomain_data=tags,metadata={'quadrature_degree':2})
        v=ufl.TestFunction(self.V)
        form=-ufl.inner(Pp+Pa,ufl.grad(v))*dx-c['solid']['end_spring_dyn_cm3']*ufl.inner(self.u,v)*ds(1)
        self.force_form=fem.form(form);self.jac_form=fem.form(ufl.derivative(form,self.u,ufl.TrialFunction(self.V)))
        local_nodes=np.argsort(self.vm);interleaved=(3*local_nodes[:,None]+np.arange(3)).ravel()
        component=np.arange(3*self.n).reshape(self.n,3).T.ravel();self.jorder=interleaved[component]
        # Exact parent gradients used only for validity, independently from FE assembly.
        inv=np.linalg.inv((X[cells[:,1:]]-X[cells[:,:1]]).transpose(0,2,1))
        self.grad=np.concatenate((-inv.sum(1,keepdims=True),inv),axis=1);self.cells=cells
        g=c['geometry'];self.rings=np.arange(len(X)).reshape(g['axial_elements']+1,g['thickness_elements']+1,g['circumferential_elements'])[:,0]
        self.initial_areas=self.areas(X)
        # Original nodal FEM diffusion rule and CN time scheme.
        verts=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        dxv=ufl.Measure('dx',domain=self.domain,metadata={'quadrature_rule':'custom','quadrature_points':verts,'quadrature_weights':np.full(4,1/24)})
        radial=X[:,:2]-center;radial/=np.linalg.norm(radial,axis=1)[:,None]
        sheet=fem.Function(self.V);sheet.x.array[:]=np.column_stack((radial,np.zeros(self.n)))[self.vm].ravel()
        axial=ufl.as_vector([0.,0.,1.]);normal=ufl.cross(axial,sheet)
        gf,gs,gn=c['ep']['parameters']['conductivity_us_per_cm']
        D=gf*ufl.outer(axial,axial)+gs*ufl.outer(sheet,sheet)+gn*ufl.outer(normal,normal)
        trial,test=ufl.TrialFunction(self.S),ufl.TestFunction(self.S)
        def matrix(form):
            A=fem.assemble_matrix(fem.form(form));A.scatter_reverse();A=A.to_scipy();order=np.argsort(self.sm)
            return A[order,:][:,order].tocsr()
        self.M=matrix(trial*test*dxv);self.K=matrix(ufl.inner(D*ufl.grad(trial),ufl.grad(test))*dxv)
        p=c['ep']['parameters'];self.sv=p['surface_to_volume_per_cm'];self.alpha=self.sv*p['capacitance_uf_per_cm2']/c['time']['ep_dt_s']
        self.ep_lu=splu((self.alpha*self.M+.5*self.K).tocsc())

    def ep_step(self,voltage,current):
        return self.ep_lu.solve(self.M@(self.alpha*voltage-self.sv*current)-.5*(self.K@voltage))

    def stage(self,land):self.xs.x.array[:]=land[self.sm,0]
    def set_x(self,x):self.u.x.array[:]=(x-self.X)[self.vm].ravel()
    def areas(self,x):
        p=x[self.rings,:2];p=p-p.mean(1,keepdims=True);q=np.roll(p,-1,axis=1)
        return .5*(p[...,0]*q[...,1]-p[...,1]*q[...,0]).sum(1)
    def valid(self,x):
        return bool(np.isfinite(x).all() and (np.linalg.det(np.einsum('eai,eaj->eij',x[self.cells],self.grad))>0).all()
                    and (self.areas(x)>self.c['active']['minimum_radius_ratio']**2*self.initial_areas).all())

    def force(self,x):
        self.set_x(x);v=fem.assemble_vector(self.force_form).array.copy().reshape(-1,3)
        result=np.empty_like(v);result[self.vm]=v
        return result.T.ravel()

    def jacobian(self,x):
        self.set_x(x);A=fem.assemble_matrix(self.jac_form);A.scatter_reverse();dense=A.to_scipy().toarray()
        return dense[np.ix_(self.jorder,self.jorder)]

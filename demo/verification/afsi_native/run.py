"""Original AFSI Chorin + C++ IB on a shared-input single-layer tube.

Only numerical AFSI modules are loaded; original numerical sources and binary
are not modified. This entry does not invoke demo telemetry or notifications.
The declared fixed translation makes the native IB grid origin zero.
"""
from pathlib import Path
import argparse,hashlib,importlib,json,sys,time,types
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl,ufl
from dolfinx import fem,mesh
from dolfinx.fem.petsc import create_vector,assemble_vector
from scipy.spatial import cKDTree

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--inputs',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();c=json.loads(args.config.read_text());comm=MPI.COMM_WORLD;out=args.output
    error=None
    if comm.rank==0:
        if out.exists() and any(out.iterdir()):error='new empty AFSI output required'
        else:out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(c,indent=2))
    error=comm.bcast(error,root=0)
    if error:raise ValueError(error)
    comm.Barrier();setup_start=time.perf_counter()
    package=types.ModuleType('afsic');package.__path__=[c['afsi_runtime_directory']];sys.modules['afsic']=package
    original=importlib.import_module('afsic.afsic_ext')
    chorin_module=importlib.import_module('afsic.euler.ChorinSolver')
    with np.load(args.inputs) as a:data={k:a[k].copy() for k in a.files}
    shift=np.array([.7,.7,0.]);X=data['reference'];shifted=X+shift;cells=data['cells']
    domain=mesh.create_mesh(comm,cells if comm.rank==0 else np.empty((0,4),np.int64),
        basix.ufl.element('Lagrange','tetrahedron',1,shape=(3,)),shifted if comm.rank==0 else np.empty((0,3)))
    Vs=fem.functionspace(domain,('Lagrange',1,(3,)));S=fem.functionspace(domain,('Lagrange',1))
    coords=fem.Function(Vs,name='coordinates');coords.interpolate(lambda x:x)
    velocity_s=fem.Function(Vs,name='structural_velocity');force_s=fem.Function(Vs,name='integrated_nodal_force')
    def mapping(space):
        distance,index=cKDTree(shifted).query(space.tabulate_dof_coordinates())
        if distance.max(initial=0)>1e-12:raise ValueError('material coordinate mapping failed')
        return index
    solid_map=mapping(Vs);scalar_map=mapping(S)
    xs=fem.Function(S);xs.x.array[:]=data['land_xs'][scalar_map]
    rest=fem.Function(S);rest.x.array[:]=data['resting_tension_dyn_cm2'][scalar_map]
    Qf=fem.functionspace(domain,('DG',0,(3,)));fiber=fem.Function(Qf)
    centroids=shifted[cells].mean(1);distance,which=cKDTree(centroids).query(Qf.tabulate_dof_coordinates())
    if distance.max(initial=0)>1e-12:raise ValueError('DG0 fiber mapping failed')
    fiber.x.array[:]=data['fiber'][which].ravel()
    F=ufl.grad(coords);J=ufl.det(F);FiT=ufl.inv(F).T
    s=c['solid'];E=s['young_modulus_dyn_cm2'];nu=s['poisson_ratio'];mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
    P=mu*(F-FiT)+lam*ufl.ln(J)*FiT
    material=ufl.SpatialCoordinate(domain);z=material[2]-c['geometry']['axial_start_cm'];L=c['geometry']['length_cm']
    q=ufl.min_value(1.,ufl.max_value(0.,ufl.min_value(z,L-z)/c['active']['end_taper_cm']))
    Ta=ufl.max_value(4.8e6*xs-rest,0.)*ufl.sin(.5*np.pi*q)**2*c['active']['tension_scale']
    ff=F*fiber;stretch=ufl.sqrt(ufl.max_value(ufl.dot(ff,ff),1.))
    P+=Ta*(1+4.9*(stretch-1))*ufl.outer(ff,fiber)
    kappa=s['volumetric_stabilization']['bulk_modulus_dyn_cm2']
    P+=(kappa*ufl.ln(J)-ufl.inner(P,F)/3)*FiT
    # Same degree-five 15-point volume rule as the current AFSEI wall adapter.
    from itertools import permutations
    bary=[];weight=[]
    for point,w in [((.25,)*4,.030283678097089182),((0.,1/3,1/3,1/3),.006026785714285717),
                   ((8/11,1/11,1/11,1/11),.011645249086028967),
                   ((.4334498464263357,)*2+(.0665501535736643,)*2,.010949141561386453)]:
        orbit=sorted(set(permutations(point)));bary.extend(orbit);weight.extend([w]*len(orbit))
    dx=ufl.Measure('dx',domain=domain,metadata={'quadrature_rule':'custom','quadrature_points':np.array(bary)[:,1:],'quadrature_weights':np.array(weight)})
    z0=c['geometry']['axial_start_cm']
    facets=mesh.locate_entities_boundary(domain,2,lambda x:np.isclose(x[2],z0)|np.isclose(x[2],z0+L))
    facets=np.sort(facets);tags=mesh.meshtags(domain,2,facets,np.ones(len(facets),np.int32))
    ds=ufl.Measure('ds',domain=domain,subdomain_data=tags,metadata={'quadrature_degree':2})
    test=ufl.TestFunction(Vs)
    force_form=fem.form(-ufl.inner(P,ufl.grad(test))*dx-s['end_spring_dyn_cm3']*ufl.inner(coords-material,test)*ds(1))
    force_vector=create_vector(force_form)
    Jexpr=fem.Expression(J,np.array([[.25,.25,.25]]))
    local_cells=np.arange(domain.topology.index_map(3).size_local,dtype=np.int32)

    n=c['fluid_elements_per_axis'];length=2.8;dt=c['dt_s'];end=c['end_time_s'];steps=round(end/dt)
    bg=mesh.create_box(comm,[np.zeros(3),np.full(3,length)],[n]*3,cell_type=mesh.CellType.hexahedron)
    V=fem.functionspace(bg,('Lagrange',2,(3,)));Q=fem.functionspace(bg,('Lagrange',1))
    boundary=mesh.locate_entities_boundary(bg,2,lambda x:np.any(np.isclose(x,0.)|np.isclose(x,length),axis=0))
    pressure_dofs=fem.locate_dofs_topological(Q,2,boundary)
    bcp=[fem.dirichletbc(PETSc.ScalarType(0),pressure_dofs,Q)]
    rho=c['fluid']['density_g_per_cm3']
    if rho!=1.:raise ValueError('this original AFSI Chorin benchmark fixes rho=1 consistently')
    ns=chorin_module.ChorinSolver(V,Q,[],bcp,dt,rho,c['fluid']['viscosity_dyn_s_per_cm2'])
    for solver in (ns.solver1,ns.solver2,ns.solver3):solver.setTolerances(rtol=1e-10,atol=1e-12,max_it=2000)
    ibmesh=original.IBMesh3D(0.,length,0.,length,0.,length,n,n,n,2)
    ib=original.IBInterpolation3D(ibmesh);bg_coords=fem.Function(V);bg_coords.interpolate(lambda x:x)
    ibmesh.build_map(bg_coords._cpp_object);ib.evaluate_current_points(coords._cpp_object)
    owned_s=Vs.dofmap.index_map.size_local;owned_u=V.dofmap.index_map.size_local;owned_p=Q.dofmap.index_map.size_local
    ucoords=V.tabulate_dof_coordinates()[:owned_u];pcoords=Q.tabulate_dof_coordinates()[:owned_p]
    frames=[];blocks=[];last_time=0.;last_step=0;max_iterations=[0,0,0]
    def output(step,t):
        owned_ids=solid_map[:owned_s];points=coords.x.array[:3*owned_s].reshape(-1,3)-shift
        parcels=comm.gather((owned_ids.copy(),points.copy(),ucoords.copy(),ns.u_.x.array[:3*owned_u].reshape(-1,3).copy(),
            pcoords.copy(),ns.p_.x.array[:owned_p].copy()),root=0)
        if comm.rank==0:
            x=np.empty_like(X);u=np.empty((2*n+1,2*n+1,2*n+1,3));p=np.empty((n+1,n+1,n+1))
            seen_x=np.zeros(len(X),bool);seen_u=np.zeros(u.shape[:3],bool);seen_p=np.zeros(p.shape,bool)
            for ids,xx,uc,uv,pc,pv in parcels:
                x[ids]=xx;iu=np.rint(uc/(length/(2*n))).astype(int);ip=np.rint(pc/(length/n)).astype(int)
                u[tuple(iu.T)]=uv;p[tuple(ip.T)]=pv
                seen_x[ids]=True;seen_u[tuple(iu.T)]=True;seen_p[tuple(ip.T)]=True
            if not (seen_x.all() and seen_u.all() and seen_p.all()):raise ValueError('global output field map incomplete')
            if step==0 and np.max(np.abs(x-X))>1e-12:raise ValueError('initial geometry changed after mesh import')
            name=f'frame_{step:07d}.npz';np.savez_compressed(out/name,x=x,velocity_q2=u,pressure_q1=p,time_s=t)
            frames.append(dict(time_s=t,file=name));(out/'frames.json').write_text(json.dumps(frames,indent=2))
            D=(X[cells[:,1:]]-X[cells[:,:1]]).transpose(0,2,1);d=(x[cells[:,1:]]-x[cells[:,:1]]).transpose(0,2,1)
            jj=np.linalg.det(d)/np.linalg.det(D);vol=abs(np.linalg.det(D))/6
            print(json.dumps(dict(step=step,time_ms=t*1000,min_J=float(jj.min()),max_J=float(jj.max()),
                wall_volume_change_pct=float(100*(np.sum(vol*jj)/vol.sum()-1)),max_displacement_mm=float(np.linalg.norm(x-X,axis=1).max()*10))),flush=True)
    try:
        output(0,0.);comm.Barrier();setup_seconds=time.perf_counter()-setup_start;tick=time.perf_counter()
        block_steps=c['log_every'];field_every=c['field_every']
        for step in range(1,steps+1):
            # Original AFSI order: fluid -> interpolate -> move -> assemble -> spread.
            ns.solve_one_step()
            for j,solver in enumerate((ns.solver1,ns.solver2,ns.solver3)):
                if solver.getConvergedReason()<=0:raise RuntimeError(f'original AFSI fluid linear solve {j+1} did not converge')
                max_iterations[j]=max(max_iterations[j],solver.getIterationNumber())
            ib.fluid_to_solid(ns.u_._cpp_object,velocity_s._cpp_object)
            coords.x.array[:]+=dt*velocity_s.x.array;coords.x.scatter_forward()
            jlocal=Jexpr.eval(domain,local_cells).ravel();minimum=comm.allreduce(float(jlocal.min(initial=np.inf)),op=MPI.MIN)
            if not np.isfinite(minimum) or minimum<=0:raise RuntimeError('original AFSI accepted geometry would have nonpositive J')
            ib.evaluate_current_points(coords._cpp_object)
            with force_vector.localForm() as local:local.set(0.)
            assemble_vector(force_vector,force_form);force_vector.ghostUpdate(addv=PETSc.InsertMode.ADD,mode=PETSc.ScatterMode.REVERSE)
            values=force_vector.array;force_s.x.array[:len(values)]=values
            ib.solid_to_fluid(ns.f._cpp_object,force_s._cpp_object);ns.f.x.scatter_forward()
            last_step=step;last_time=step*dt
            if step%block_steps==0 or step==steps:
                comm.Barrier();seconds=comm.allreduce(time.perf_counter()-tick,op=MPI.MAX)
                blocks.append(dict(end_step=step,seconds=seconds))
                if step%field_every==0 or step==steps:output(step,last_time)
                tick=time.perf_counter()
        if comm.rank==0:
            report=dict(status='completed',steps=last_step,time_s=last_time,setup_seconds=setup_seconds,
                compute_seconds=sum(b['seconds'] for b in blocks),timing_blocks=blocks,
                max_linear_iterations=max_iterations,all_fluid_solves_converged=True,
                mpi_processes=comm.size,fluid_elements=[n]*3,velocity_nodes=[2*n+1]*3,
                original_Chorin_sha256=hashlib.sha256(Path(chorin_module.__file__).read_bytes()).hexdigest(),
                original_CPP_binary_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
                coordinate_translation_cm=shift.tolist(),EP_advanced=False,ODE_advanced=False,
                source='original AFSI Chorin and C++ IB, unchanged numerical modules',
                initial_force='original zero-force startup',formal_matched_accuracy_speedup_established=False)
            (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    except Exception as exc:
        if comm.rank==0:(out/'report.json').write_text(json.dumps(dict(status='failed',steps=last_step,time_s=last_time,error=repr(exc),settings_relaxed=False),indent=2))
        raise
    finally:ns.post_process();force_vector.destroy()

if __name__=='__main__':main()

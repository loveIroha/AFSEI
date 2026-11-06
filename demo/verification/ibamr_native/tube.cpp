// Native IBAMR/libMesh application for a fixed-model AFSEI comparison.
// Library initialization and hierarchy setup follow IBAMR IBFE/explicit/ex2
// (IBAMR 0.19.0, BSD-3-Clause). No AFSEI numerical operators are called.
#include <SAMRAI_config.h>
#include <petscsys.h>
#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>
#include <PatchLevel.h>
#include <Patch.h>
#include <VariableDatabase.h>
#include <CellData.h>
#include <CellIndex.h>
#include <SideData.h>
#include <SideIndex.h>
#include <libmesh/replicated_mesh.h>
#include <libmesh/elem.h>
#include <libmesh/equation_systems.h>
#include <libmesh/system.h>
#include <libmesh/numeric_vector.h>
#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFEMethod.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>
#include <ibamr/INSStaggeredVelocityBcCoef.h>
#include <ibamr/INSStaggeredPressureBcCoef.h>
#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/PETScKrylovLinearSolver.h>
#include <ibtk/libmesh_utilities.h>
#include <ibtk/muParserRobinBcCoefs.h>
#include <ibamr/app_namespaces.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace benchmark {
// AFSEI's open projection problem uses homogeneous velocity-Neumann and
// pressure-Dirichlet operators. Native IBAMR defaults instead interpret the
// same user Robin numbers as traction. Use its boundary extension interface
// to supply the intended operators; do not modify IBAMR solver source.
class OpenVelocityBc final : public INSStaggeredVelocityBcCoef {
public:
    OpenVelocityBc(int component,const INSStaggeredHierarchyIntegrator* solver,
        const std::vector<RobinBcCoefStrategy<NDIM>*>& bc)
        :INSStaggeredVelocityBcCoef(component,solver,bc,TRACTION) {}
    void setBcCoefs(Pointer<ArrayData<NDIM,double>>& a,Pointer<ArrayData<NDIM,double>>& b,
        Pointer<ArrayData<NDIM,double>>& g,const Pointer<SAMRAI::hier::Variable<NDIM>>&,
        const Patch<NDIM>&,const BoundaryBox<NDIM>&,double=0.) const override {
        if(a)a->fillAll(0.);if(b)b->fillAll(1.);if(g)g->fillAll(0.);
    }
};
class OpenPressureBc final : public INSStaggeredPressureBcCoef {
public:
    OpenPressureBc(const INSStaggeredHierarchyIntegrator* solver,
        const std::vector<RobinBcCoefStrategy<NDIM>*>& bc)
        :INSStaggeredPressureBcCoef(solver,bc,TRACTION) {}
    void setBcCoefs(Pointer<ArrayData<NDIM,double>>& a,Pointer<ArrayData<NDIM,double>>& b,
        Pointer<ArrayData<NDIM,double>>& g,const Pointer<SAMRAI::hier::Variable<NDIM>>&,
        const Patch<NDIM>&,const BoundaryBox<NDIM>&,double=0.) const override {
        if(a)a->fillAll(1.);if(b)b->fillAll(0.);if(g)g->fillAll(0.);
    }
};
class PressureOpenIntegrator final : public INSStaggeredHierarchyIntegrator {
public:
    PressureOpenIntegrator(const std::string& name,Pointer<Database> database)
        :INSStaggeredHierarchyIntegrator(name,database) {
        for(int j=0;j<3;++j) {delete d_U_bc_coefs[j];d_U_bc_coefs[j]=new OpenVelocityBc(j,this,d_bc_coefs);}
        delete d_P_bc_coef;d_P_bc_coef=new OpenPressureBc(this,d_bc_coefs);
    }
};
template<class T> void read(std::ifstream& f,T* x,std::size_t n=1) {
    f.read(reinterpret_cast<char*>(x),sizeof(T)*n);
    if(!f) throw std::runtime_error("truncated benchmark input");
}
template<class T> void write(std::ofstream& f,const T* x,std::size_t n=1) {
    f.write(reinterpret_cast<const char*>(x),sizeof(T)*n);
    if(!f) throw std::runtime_error("benchmark output failed");
}
std::array<double,9> inverse(const std::array<double,9>& A) {
    const double d=A[0]*(A[4]*A[8]-A[5]*A[7])-A[1]*(A[3]*A[8]-A[5]*A[6])+A[2]*(A[3]*A[7]-A[4]*A[6]);
    if(std::abs(d)<1e-20) throw std::runtime_error("degenerate reference tetrahedron");
    return {{(A[4]*A[8]-A[5]*A[7])/d,(A[2]*A[7]-A[1]*A[8])/d,(A[1]*A[5]-A[2]*A[4])/d,
      (A[5]*A[6]-A[3]*A[8])/d,(A[0]*A[8]-A[2]*A[6])/d,(A[2]*A[3]-A[0]*A[5])/d,
      (A[3]*A[7]-A[4]*A[6])/d,(A[1]*A[6]-A[0]*A[7])/d,(A[0]*A[4]-A[1]*A[3])/d}};
}
struct Model {
    std::uint64_t nn=0,ne=0,nc=0,ec=0,factor=0,nt=0;
    double tape_dt=0.,mu=0.,lambda=0.,kappa=0.,spring=0.,taper=0.,z0=0.,length=0.,scale=1.;
    std::vector<double> X,fiber,coarseX,rest,xs;
    std::vector<std::uint64_t> cells,coarseCells;
    std::vector<std::array<double,9>> inv;
    void load(const std::string& mesh,const std::string& active) {
        std::ifstream f(mesh,std::ios::binary);if(!f)throw std::runtime_error("cannot open mesh input");
        read(f,&nn);read(f,&ne);read(f,&nc);read(f,&ec);read(f,&factor);
        X.resize(nn*3);cells.resize(ne*4);fiber.resize(ne*3);coarseX.resize(nc*3);coarseCells.resize(ec*4);
        read(f,X.data(),X.size());read(f,cells.data(),cells.size());read(f,fiber.data(),fiber.size());
        read(f,coarseX.data(),coarseX.size());read(f,coarseCells.data(),coarseCells.size());
        std::ifstream a(active,std::ios::binary);if(!a)throw std::runtime_error("cannot open active input");
        std::uint64_t count;read(a,&count);read(a,&nt);read(a,&tape_dt);
        if(count!=nc || !nt || tape_dt<=0)throw std::runtime_error("active tape contract mismatch");
        rest.resize(nc);xs.resize(nc*nt);read(a,rest.data(),rest.size());read(a,xs.data(),xs.size());
        inv.reserve(ec);
        for(std::size_t e=0;e<ec;++e) {
            std::array<double,9> D;
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)
                D[3*i+j]=coarseX[3*coarseCells[4*e+j+1]+i]-coarseX[3*coarseCells[4*e]+i];
            inv.push_back(inverse(D));
        }
    }
    double active(const libMesh::Point& s,std::size_t element,double t) const {
        const auto parent=element/(factor*factor*factor);
        if(parent>=ec)throw std::runtime_error("libMesh element IDs changed");
        std::array<double,3> q;
        for(int j=0;j<3;++j)q[j]=s(j)-coarseX[3*coarseCells[4*parent]+j];
        double b[4]={1.,0.,0.,0.};
        for(int j=0;j<3;++j)for(int k=0;k<3;++k)b[j+1]+=inv[parent][3*j+k]*q[k];
        b[0]-=b[1]+b[2]+b[3];
        if(*std::min_element(b,b+4)<-1e-8)throw std::runtime_error("coarse material field map mismatch");
        double x=0.,r=0.;const double step=std::max(0.,t/tape_dt);
        if(nt>1 && step>nt-1+1e-6)throw std::runtime_error("active tape ended before simulation");
        const auto lo=std::min<std::size_t>(nt-1,std::size_t(std::floor(step)));
        const auto hi=std::min<std::size_t>(nt-1,lo+1);const double a=std::min(1.,step-lo);
        for(int j=0;j<4;++j) {
            const auto id=coarseCells[4*parent+j];
            x+=b[j]*((1-a)*xs[lo*nc+id]+a*xs[hi*nc+id]);r+=b[j]*rest[id];
        }
        const double d=std::max(0.,std::min(1.,std::min(s(2)-z0,z0+length-s(2))/taper));
        const double shape=std::sin(.5*std::acos(-1.)*d);
        return std::max(4.8e6*x-r,0.)*shape*shape*scale;
    }
};
void stress(TensorValue<double>& P,const TensorValue<double>& F,const libMesh::Point&,
 const libMesh::Point& s,Elem* const e,const std::vector<const std::vector<double>*>&,
 const std::vector<const std::vector<VectorValue<double>>*>&,double t,void* ptr) {
    const auto& m=*static_cast<Model*>(ptr);const double J=F.det();
    if(!(J>0) || !std::isfinite(J))throw std::runtime_error("nonpositive structural J");
    const auto FiT=tensor_inverse_transpose(F,3);
    P=m.mu*(F-FiT)+m.lambda*std::log(J)*FiT;
    VectorValue<double> f(m.fiber[3*e->id()],m.fiber[3*e->id()+1],m.fiber[3*e->id()+2]);
    const auto ff=F*f;const double stretch=std::max(1.,ff.norm());
    const double Ta=m.active(s,e->id(),t)*(1+4.9*(stretch-1));
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)P(i,j)+=Ta*ff(i)*f(j);
    double trace=0.;for(int i=0;i<3;++i)for(int j=0;j<3;++j)trace+=P(i,j)*F(i,j);
    P+=(m.kappa*std::log(J)-trace/3)*FiT;
}
void end_force(VectorValue<double>& force,const VectorValue<double>&,const VectorValue<double>&,
 const TensorValue<double>&,const libMesh::Point& x,const libMesh::Point& s,Elem* const,
 unsigned short,const std::vector<const std::vector<double>*>&,
 const std::vector<const std::vector<VectorValue<double>>*>&,double,void* ptr) {
    const auto& m=*static_cast<Model*>(ptr);force.zero();
    if(std::abs(s(2)-m.z0)<1e-10 || std::abs(s(2)-m.z0-m.length)<1e-10)force=m.spring*(s-x);
}

void snapshot(Pointer<PatchHierarchy<NDIM>> hierarchy,Pointer<INSStaggeredHierarchyIntegrator> flow,
 libMesh::ReplicatedMesh& mesh,IBFEMethod& ib,const Model& m,int n,int step,double t,const std::string& output) {
    auto& system=ib.getFEDataManager()->getEquationSystems()->get_system(ib.getCurrentCoordinatesSystemName());
    std::vector<Number> values;system.solution->localize(values);
    const std::size_t count=std::size_t(n)*n*n;
    std::vector<double> p(count,0.),u(3*count,0.),globalp(count),globalu(3*count);
    auto* db=VariableDatabase<NDIM>::getDatabase();
    const int pi=db->mapVariableAndContextToIndex(flow->getPressureVariable(),flow->getCurrentContext());
    const int ui=db->mapVariableAndContextToIndex(flow->getVelocityVariable(),flow->getCurrentContext());
    Pointer<PatchLevel<NDIM>> level=hierarchy->getPatchLevel(0);
    for(PatchLevel<NDIM>::Iterator it(level);it;it++) {
        Pointer<Patch<NDIM>> patch=level->getPatch(it());
        Pointer<CellData<NDIM,double>> pp=patch->getPatchData(pi);
        Pointer<SideData<NDIM,double>> vv=patch->getPatchData(ui);
        for(Box<NDIM>::Iterator cell(patch->getBox());cell;cell++) {
            CellIndex<NDIM> index(cell());const std::size_t k=(std::size_t(index(0))*n+index(1))*n+index(2);
            p[k]=(*pp)(index);
            for(int j=0;j<3;++j)u[3*k+j]=.5*((*vv)(SideIndex<NDIM>(index,j,SideIndex<NDIM>::Lower))+(*vv)(SideIndex<NDIM>(index,j,SideIndex<NDIM>::Upper)));
        }
    }
    MPI_Reduce(p.data(),globalp.data(),int(count),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(u.data(),globalu.data(),int(3*count),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank==0) {
        std::filesystem::create_directories(output);std::ostringstream name;
        name<<output<<"/frame_"<<std::setw(7)<<std::setfill('0')<<step<<".bin";
        std::ofstream f(name.str(),std::ios::binary);std::uint64_t shape[4]={1,std::uint64_t(n),m.nn,std::uint64_t(step)};
        write(f,shape,4);write(f,&t);
        std::vector<double> x(m.nn*3);
        for(std::size_t i=0;i<m.nn;++i)for(int j=0;j<3;++j)x[3*i+j]=values[mesh.node_ptr(i)->dof_number(system.number(),j,0)];
        write(f,x.data(),x.size());write(f,globalp.data(),globalp.size());write(f,globalu.data(),globalu.size());
    }
}
} // namespace benchmark

int main(int argc,char** argv) {
    IBTKInit init(argc,argv,MPI_COMM_WORLD);
    const double start=MPI_Wtime();
    {
        Pointer<AppInitializer> app=new AppInitializer(argc,argv,"native_ibamr.log");
        auto input=app->getInputDatabase();benchmark::Model m;
        m.load(input->getString("MESH_FILE"),input->getString("ACTIVE_FILE"));
        const double E=input->getDouble("WALL_E"),nu=input->getDouble("WALL_NU");
        m.mu=E/(2*(1+nu));m.lambda=E*nu/((1+nu)*(1-2*nu));m.kappa=input->getDouble("WALL_KAPPA");
        m.spring=input->getDouble("END_SPRING");m.taper=input->getDouble("END_TAPER");
        m.z0=input->getDouble("TUBE_START");m.length=input->getDouble("TUBE_LENGTH");
        const int n=input->getInteger("N"),dump=input->getInteger("FIELD_EVERY"),log_every=input->getInteger("LOG_EVERY");
        const std::string output=input->getString("OUTPUT_DIR");
        libMesh::ReplicatedMesh mesh(init.getLibMeshInit().comm(),3);mesh.allow_renumbering(false);
        for(std::size_t i=0;i<m.nn;++i)mesh.add_point(libMesh::Point(m.X[3*i],m.X[3*i+1],m.X[3*i+2]),i);
        for(std::size_t i=0;i<m.ne;++i) {
            auto e=libMesh::Elem::build(libMesh::TET4);e->set_id(i);
            for(int j=0;j<4;++j)e->set_node(j)=mesh.node_ptr(m.cells[4*i+j]);
            mesh.add_elem(e.release());
        }
        mesh.prepare_for_use();
        const bool pressure_open=input->getStringWithDefault("OPEN_BOUNDARY","TRACTION")=="AFSEI_PRESSURE_NEUMANN";
        auto fluid_db=app->getComponentDatabase("INSStaggeredHierarchyIntegrator");
        fluid_db->putString("stokes_solver_type",StaggeredStokesSolverManager::PETSC_KRYLOV_SOLVER);
        auto linear_db=fluid_db->putDatabase("stokes_solver_db");
        linear_db->putString("ksp_type","fgmres");linear_db->putDouble("rel_residual_tol",1e-10);
        linear_db->putDouble("abs_residual_tol",1e-12);linear_db->putInteger("max_iterations",500);
        Pointer<INSStaggeredHierarchyIntegrator> flow;
        if(pressure_open)flow=new benchmark::PressureOpenIntegrator("INSStaggeredHierarchyIntegrator",fluid_db);
        else flow=new INSStaggeredHierarchyIntegrator("INSStaggeredHierarchyIntegrator",fluid_db);
        Pointer<IBFEMethod> ib=new IBFEMethod("IBFEMethod",app->getComponentDatabase("IBFEMethod"),&mesh,1);
        IBFEMethod::PK1StressFcnData stress(benchmark::stress,{},&m,libMesh::QGAUSS,libMesh::FIFTH);
        ib->registerPK1StressFunction(stress);ib->registerLagSurfaceForceFunction(IBFEMethod::LagSurfaceForceFcnData(benchmark::end_force,{},&m));
        Pointer<IBHierarchyIntegrator> integrator=new IBExplicitHierarchyIntegrator("IBHierarchyIntegrator",app->getComponentDatabase("IBHierarchyIntegrator"),ib,flow);
        Pointer<CartesianGridGeometry<NDIM>> geometry=new CartesianGridGeometry<NDIM>("CartesianGeometry",app->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM>> hierarchy=new PatchHierarchy<NDIM>("PatchHierarchy",geometry);
        Pointer<StandardTagAndInitialize<NDIM>> detector=new StandardTagAndInitialize<NDIM>("StandardTagAndInitialize",integrator,app->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM>> generator=new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM>> balancer=new LoadBalancer<NDIM>("LoadBalancer",app->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM>> gridder=new GriddingAlgorithm<NDIM>("GriddingAlgorithm",app->getComponentDatabase("GriddingAlgorithm"),detector,generator,balancer);
        // IBFEMethod queries its registered hierarchy integrator here.
        ib->initializeFEEquationSystems();
        std::vector<RobinBcCoefStrategy<NDIM>*> bc(3);
        for(int j=0;j<3;++j)bc[j]=new muParserRobinBcCoefs("open_"+std::to_string(j),app->getComponentDatabase("VelocityBcCoefs_"+std::to_string(j)),geometry);
        flow->registerPhysicalBoundaryConditions(bc);
        ib->initializeFEData();integrator->initializePatchHierarchy(hierarchy,gridder);
        int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);std::filesystem::create_directories(output);
        if(rank==0) {
            std::ofstream contract(output+"/native_contract.json");
            contract<<"{\"software\":\"IBAMR 0.19.0 / native IBFE\",\"full_three_field_solver\":false,\"active_drive\":\"shared EP-derived field\",\"open_boundary\":\""<<(pressure_open?"homogeneous velocity Neumann and pressure Dirichlet through boundary coefficient subclasses":"native homogeneous Robin traction")<<"\",\"formal_speedup_established\":false}";
        }
        benchmark::snapshot(hierarchy,flow,mesh,*ib,m,n,0,0.,output);
        const double setup=MPI_Wtime()-start;double compute=0.,t=integrator->getIntegratorTime(),last_dt=0.;int step=0,max_linear_iterations=0;
        while(integrator->stepsRemaining() && t<integrator->getEndTime()-1e-12) {
            const double dt=integrator->getMaximumTimeStepSize();const double tick=MPI_Wtime();
            integrator->advanceHierarchy(dt);compute+=MPI_Wtime()-tick;t=integrator->getIntegratorTime();step=integrator->getIntegratorStep();
            last_dt=dt;
            auto* krylov=dynamic_cast<IBTK::PETScKrylovLinearSolver*>(flow->getStokesSolver().getPointer());
            if(!krylov)throw std::runtime_error("native Stokes solver does not expose its convergence reason");
            KSPConvergedReason reason;KSPGetConvergedReason(krylov->getPETScKSP(),&reason);
            if(reason<=0)throw std::runtime_error("native Stokes solve did not converge");
            max_linear_iterations=std::max(max_linear_iterations,flow->getStokesSolver()->getNumIterations());
            if(step%log_every==0)pout<<"NATIVE step="<<step<<" time="<<t<<" compute_seconds="<<compute<<std::endl;
            if(step%dump==0 || t>=integrator->getEndTime()-1e-12)benchmark::snapshot(hierarchy,flow,mesh,*ib,m,n,step,t,output);
        }
        double global_compute;MPI_Reduce(&compute,&global_compute,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        if(rank==0) {
            std::ofstream report(output+"/report.json");
            report<<std::setprecision(17)<<"{\"status\":\"completed\",\"time_s\":"<<t<<",\"pressure_time_s\":"<<t-.5*last_dt<<",\"steps\":"<<step<<",\"setup_seconds\":"<<setup<<",\"compute_seconds\":"<<global_compute<<",\"mean_compute_seconds_per_step\":"<<global_compute/step<<",\"max_stokes_iterations\":"<<max_linear_iterations<<",\"all_stokes_solves_converged\":true,\"stokes_rtol\":1e-10,\"formal_performance_accuracy_gate_passed\":false}";
        }
        integrator.setNull();ib.setNull();flow.setNull();for(auto* b:bc)delete b;
    }
    return 0;
}

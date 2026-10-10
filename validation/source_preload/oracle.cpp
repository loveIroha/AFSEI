// Assemble pinned regenerated UFL at arbitrary supplied preload positions.
// No original project files or simulations are edited; no fluid cap is loaded.
#include <dolfin.h>
#include "Active_land.h"
#include "cavity_volume.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <map>
#include <cmath>

using Key=std::array<long long,3>;
using Row=std::array<double,9>;
Key key(const double* x) {return {{std::llround(x[0]*1e10),std::llround(x[1]*1e10),std::llround(x[2]*1e10)}};}

class SuppliedField:public dolfin::Expression {
public:
    const std::map<Key,Row>& data; int offset;
    SuppliedField(const std::map<Key,Row>& d,int offset):Expression(3),data(d),offset(offset) {}
    void eval(Eigen::Ref<Eigen::VectorXd> v,Eigen::Ref<const Eigen::VectorXd> x) const override {
        auto found=data.find(key(x.data()));
        if(found==data.end()) throw std::runtime_error("supplied reference vertex not found");
        for(int j=0;j<3;++j) {
            if(std::abs(found->second[j]-x[j])>1e-10) throw std::runtime_error("supplied reference mismatch");
            v[j]=found->second[offset+j];
        }
    }
};
class Reference:public dolfin::Expression {
public:
    Reference():Expression(3) {}
    void eval(Eigen::Ref<Eigen::VectorXd> v,Eigen::Ref<const Eigen::VectorXd> x) const override {v=x;}
};
class CellDirections:public dolfin::Expression {
public:
    std::shared_ptr<dolfin::MeshFunction<double>> c[3];
    CellDirections(const std::string& dir,const std::string& name,std::shared_ptr<dolfin::Mesh> mesh):Expression(3) {
        for(int i=0;i<3;++i) c[i]=std::make_shared<dolfin::MeshFunction<double>>(mesh,dir+"/"+name+"_"+std::to_string(i)+".xml");
    }
    void eval(Eigen::Ref<Eigen::VectorXd> v,Eigen::Ref<const Eigen::VectorXd>,const ufc::cell& cell) const override {
        for(int i=0;i<3;++i) v[i]=(*c[i])[cell.index];
    }
};
template<class Form>
void bind_passive_source_form(Form& form,std::shared_ptr<const dolfin::GenericFunction> f,std::shared_ptr<const dolfin::GenericFunction> s,
          std::shared_ptr<dolfin::Function> x,std::shared_ptr<dolfin::Function> ref,
          std::shared_ptr<dolfin::Function> zero,std::shared_ptr<dolfin::Constant> pressure,
          std::shared_ptr<dolfin::MeshFunction<std::size_t>> tags) {
    form.f0=f; form.s0=s; form.X=x; form.x_start=ref; form.X_previous=x;
    form.XS=zero; form.XW=zero; form.Zetas_prev=zero; form.Zetaw_prev=zero;
    form.X_ta_anchor=x; form.X_ta_previous=x;
    form.XS_ta_anchor=zero; form.XW_ta_anchor=zero; form.Zetas_ta_anchor=zero; form.Zetaw_ta_anchor=zero;
    form.dt_mech=std::make_shared<dolfin::Constant>(.005); form.dt_ta_anchor=std::make_shared<dolfin::Constant>(.005);
    form.ta_use_land=std::make_shared<dolfin::Constant>(0.); form.ta_ramp_factor=std::make_shared<dolfin::Constant>(0.);
    form.pressure=pressure; form.ds=tags;
}
int main(int argc,char** argv) {
    dolfin::init(argc,argv); dolfin::set_log_active(false);
    if(argc!=5 || dolfin::MPI::size(MPI_COMM_WORLD)!=1) return 2;
    const std::string dir=argv[1],input=argv[2],output=argv[3]; const double p=std::stod(argv[4]);
    if(!std::isfinite(p) || p<=0) return 2;
    std::map<Key,Row> data; std::ifstream file(input); std::string line;
    while(std::getline(file,line)) {
        std::replace(line.begin(),line.end(),',',' '); std::istringstream stream(line); Row r;
        for(auto& a:r) if(!(stream>>a) || !std::isfinite(a)) throw std::runtime_error("bad input CSV");
        if(!data.emplace(key(r.data()),r).second) throw std::runtime_error("duplicate input vertex");
    }
    auto mesh=std::make_shared<dolfin::Mesh>();
    {dolfin::XDMFFile in(dir+"/mesh_scale.xdmf"); in.read(*mesh);}
    if(data.size()!=mesh->num_vertices()) throw std::runtime_error("input vertex count mismatch");
    auto V=std::make_shared<Active_land::Form_H_FunctionSpace_0>(mesh);
    auto S=std::make_shared<Active_land::Form_a_ep_FunctionSpace_0>(mesh);
    auto ref=std::make_shared<dolfin::Function>(V),x=std::make_shared<dolfin::Function>(V),v=std::make_shared<dolfin::Function>(V);
    Reference reference; ref->interpolate(reference);
    SuppliedField position(data,3),direction(data,6); x->interpolate(position); v->interpolate(direction);
    auto zero=std::make_shared<dolfin::Function>(S); zero->interpolate(dolfin::Constant(0.));
    auto f=std::make_shared<CellDirections>(dir,"fibers",mesh),s=std::make_shared<CellDirections>(dir,"sheets",mesh);
    auto tags=std::make_shared<dolfin::MeshFunction<std::size_t>>(mesh,dir+"/boundaries.xml");
    auto pressure=std::make_shared<dolfin::Constant>(0.);
    Active_land::Form_H H(V); Active_land::Form_J_mech J(V,V);
    bind_passive_source_form(H,f,s,x,ref,zero,pressure,tags); bind_passive_source_form(J,f,s,x,ref,zero,pressure,tags);
    std::vector<std::vector<double>> forces(2),actions(2);
    std::array<double,2> norms;
    for(int k=0;k<2;++k) {
        *pressure=(k?p*1333.22368421:0.);
        dolfin::Vector residual,action; dolfin::Matrix matrix;
        std::cerr<<"Assembling H/J pressure mode "<<k<<std::endl;
        dolfin::assemble(residual,H); dolfin::assemble(matrix,J);
        matrix.init_vector(action,0); matrix.mult(*v->vector(),action); norms[k]=residual.norm("l2");
        dolfin::Function force(V),tangent(V); *force.vector()=residual; *tangent.vector()=action;
        force.compute_vertex_values(forces[k]); tangent.compute_vertex_values(actions[k]);
    }
    Active_land::Form_L_proj_lmbda L(S); L.f0=f; L.X=x;
    std::cerr<<"Assembling stretch projection"<<std::endl;
    dolfin::Vector rhs; dolfin::assemble(rhs,L); dolfin::Function stretch_rhs(S); *stretch_rhs.vector()=rhs;
    std::vector<double> X,xx,rr; ref->compute_vertex_values(X); x->compute_vertex_values(xx); stretch_rhs.compute_vertex_values(rr);
    std::ofstream out(output); out<<std::setprecision(17); const auto count=mesh->num_vertices();
    for(std::size_t i=0;i<count;++i) {
        bool first=true; auto write=[&](double a){if(!first) out<<','; first=false; out<<a;};
        for(int j=0;j<3;++j) write(X[j*count+i]);
        for(int j=0;j<3;++j) write(xx[j*count+i]);
        for(int k=0;k<2;++k) for(int j=0;j<3;++j) write(-forces[k][j*count+i]);
        for(int k=0;k<2;++k) for(int j=0;j<3;++j) write(-actions[k][j*count+i]);
        write(rr[i]); out<<'\n';
    }
    lvcavity::ParallelVolume cavity(mesh,*tags,2); const auto volume=cavity.evaluate(*x);
    Active_land::Form_V_cav_form open(mesh); open.X=x; open.ds=tags;
    std::cerr<<"Assembling open volume"<<std::endl;
    const double raw=-dolfin::assemble(open);
    std::ofstream metrics(output+".json"); metrics<<std::setprecision(17)
        <<"{\"pressure_mmhg\":"<<p<<",\"passive_force_norm_dyn\":"<<norms[0]<<",\"source_equilibrium_residual_norm_dyn\":"<<norms[1]
        <<",\"volume_ml\":"<<volume.closed<<",\"open_endo_ml\":"<<volume.open_endo<<",\"cap_ml\":"<<volume.cap
        <<",\"open_ufl_ml\":"<<raw<<",\"closure_error_ml\":"<<volume.closure_error<<"}\n";
    return out && metrics?0:3;
}

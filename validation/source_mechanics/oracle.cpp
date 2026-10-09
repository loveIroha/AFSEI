// Free weak residual and Jacobian action from regenerated, pinned source UFL.
// No original-project files or runs are changed; no equilibrium is solved.
#include <dolfin.h>
#include "Active_land.h"
#include <fstream>
#include <iomanip>
#include <cmath>

class CellDirections : public dolfin::Expression {
public:
    std::shared_ptr<dolfin::MeshFunction<double>> c[3];
    CellDirections(const std::string& dir,const std::string& name,
                   std::shared_ptr<dolfin::Mesh> mesh):Expression(3) {
        for (int i=0;i<3;++i) c[i]=std::make_shared<dolfin::MeshFunction<double>>(
            mesh,dir+"/"+name+"_"+std::to_string(i)+".xml");
    }
    void eval(Eigen::Ref<Eigen::VectorXd> v,Eigen::Ref<const Eigen::VectorXd>,const ufc::cell& cell) const override {
        for (int i=0;i<3;++i) v[i]=(*c[i])[cell.index];
    }
};
class VectorField : public dolfin::Expression {
public:
    int which;
    explicit VectorField(int which):Expression(3),which(which) {}
    void eval(Eigen::Ref<Eigen::VectorXd> v,Eigen::Ref<const Eigen::VectorXd> x) const override {
        double a=x[0]-7.3,b=x[1]-7.65,c=x[2]-7.5;
        if (which==0) { v=x; return; }
        if (which==3) {
            v[0]=.02*std::sin(.6*x[0])+.005*x[1];
            v[1]=.01*std::cos(.4*x[1])-.002*x[2];
            v[2]=.015*std::sin(.7*x[2])+.003*x[0];
            return;
        }
        double factor=which==1?1.:(which==2?.55:.75);
        v[0]=x[0]+factor*(.035*a+.007*b+.0009*std::sin(.6*x[0]));
        v[1]=x[1]+factor*(.003*a-.015*b+.004*c+.0007*std::cos(.5*x[1]));
        v[2]=x[2]+factor*(.002*b+.01*c+.0008*std::sin(.4*x[2]));
    }
};
class ScalarField : public dolfin::Expression {
public:
    int which; bool passive;
    ScalarField(int which,bool passive):which(which),passive(passive) {}
    void eval(Eigen::Ref<Eigen::VectorXd> v,Eigen::Ref<const Eigen::VectorXd> x) const override {
        if (passive && which<2) { v[0]=0.; return; }
        if (which==0) v[0]=.03+.002*x[0]/13.;
        if (which==1) v[0]=.05+.003*x[1]/13.;
        if (which==2) v[0]=.01+.002*x[2]/13.;
        if (which==3) v[0]=-.01+.001*x[0]/13.;
    }
};

template<class Form>
void bind_source_form(Form& form,std::shared_ptr<const dolfin::GenericFunction> f,
          std::shared_ptr<const dolfin::GenericFunction> s,
          std::shared_ptr<dolfin::Function> x,std::shared_ptr<dolfin::Function> ref,
          std::shared_ptr<dolfin::Function> prev,std::shared_ptr<dolfin::Function> anchor,
          const std::vector<std::shared_ptr<dolfin::Function>>& states,
          std::shared_ptr<dolfin::MeshFunction<std::size_t>> tags,bool frozen) {
    form.f0=f; form.s0=s; form.X=x; form.x_start=ref; form.X_previous=prev;
    form.XS=states[0]; form.XW=states[1];
    form.Zetas_prev=states[2]; form.Zetaw_prev=states[3];
    form.dt_mech=std::make_shared<dolfin::Constant>(.001);
    form.X_ta_anchor=anchor; form.X_ta_previous=prev;
    form.XS_ta_anchor=states[0]; form.XW_ta_anchor=states[1];
    form.Zetas_ta_anchor=states[2]; form.Zetaw_ta_anchor=states[3];
    form.dt_ta_anchor=std::make_shared<dolfin::Constant>(.001);
    form.ta_use_land=std::make_shared<dolfin::Constant>(frozen?0.:1.);
    form.ta_ramp_factor=std::make_shared<dolfin::Constant>(frozen?.37:0.);
    form.pressure=std::make_shared<dolfin::Constant>(0.);
    form.ds=tags;
}

int main(int argc,char** argv) {
    dolfin::init(argc,argv); dolfin::set_log_active(false);
    if (argc!=5 || dolfin::MPI::size(MPI_COMM_WORLD)!=1) return 2;
    std::string dir=argv[1],mode=argv[3],directions=argv[4];
    if ((mode!="passive" && mode!="land" && mode!="frozen") ||
        (directions!="expression" && directions!="cg1")) return 2;
    auto mesh=std::make_shared<dolfin::Mesh>();
    { dolfin::XDMFFile input(dir+"/mesh_scale.xdmf"); input.read(*mesh); }
    auto V=std::make_shared<Active_land::Form_H_FunctionSpace_0>(mesh);
    auto S=std::make_shared<Active_land::Form_a_ep_FunctionSpace_0>(mesh);
    auto ef=std::make_shared<CellDirections>(dir,"fibers",mesh);
    auto es=std::make_shared<CellDirections>(dir,"sheets",mesh);
    std::shared_ptr<const dolfin::GenericFunction> f=ef,s=es;
    if (directions=="cg1") {
        auto ff=std::make_shared<dolfin::Function>(V),ss=std::make_shared<dolfin::Function>(V);
        ff->interpolate(*ef); ss->interpolate(*es); f=ff; s=ss;
    }
    auto tags=std::make_shared<dolfin::MeshFunction<std::size_t>>(mesh,dir+"/boundaries.xml");
    std::vector<std::shared_ptr<dolfin::Function>> vectors,states;
    for (int i=0;i<5;++i) {
        auto function=std::make_shared<dolfin::Function>(V); VectorField field(i);
        function->interpolate(field); vectors.push_back(function);
    }
    for (int i=0;i<4;++i) {
        auto function=std::make_shared<dolfin::Function>(S); ScalarField field(i,mode=="passive");
        function->interpolate(field); states.push_back(function);
    }
    Active_land::Form_H H(V); Active_land::Form_J_mech J(V,V);
    bind_source_form(H,f,s,vectors[1],vectors[0],vectors[2],vectors[4],states,tags,mode=="frozen");
    bind_source_form(J,f,s,vectors[1],vectors[0],vectors[2],vectors[4],states,tags,mode=="frozen");
    dolfin::Vector residual,action; dolfin::Matrix matrix;
    dolfin::assemble(residual,H); dolfin::assemble(matrix,J);
    matrix.init_vector(action,0); matrix.mult(*vectors[3]->vector(),action);
    dolfin::Function force(V),tangent(V);
    *force.vector()=residual; *tangent.vector()=action;
    // Export RHS instead of solving: comparison tests the actual projection
    // form independently of any DOLFIN mass solver tolerance/preconditioner.
    Active_land::Form_L_proj_lmbda projection(S);
    projection.f0=f; projection.X=vectors[1];
    dolfin::Vector rhs; dolfin::assemble(rhs,projection);
    dolfin::Function stretch_rhs(S); *stretch_rhs.vector()=rhs;
    std::vector<std::vector<double>> vv(5),sv(4);
    for (int i=0;i<5;++i) vectors[i]->compute_vertex_values(vv[i]);
    for (int i=0;i<4;++i) states[i]->compute_vertex_values(sv[i]);
    std::vector<double> fv,jv,rv;
    force.compute_vertex_values(fv); tangent.compute_vertex_values(jv); stretch_rhs.compute_vertex_values(rv);
    std::ofstream output(argv[2]); output<<std::setprecision(17);
    auto count=mesh->num_vertices();
    for (dolfin::VertexIterator vertex(*mesh);!vertex.end();++vertex) {
        auto i=vertex->index();
        bool first=true;
        auto write=[&](double value) { if (!first) output<<','; first=false; output<<value; };
        for (int k=0;k<5;++k) for (int j=0;j<3;++j) write(vv[k][j*count+i]);
        for (int k=0;k<4;++k) write(sv[k][i]);
        for (int j=0;j<3;++j) write(-fv[j*count+i]);
        for (int j=0;j<3;++j) write(-jv[j*count+i]);
        write(rv[i]); output<<'\n';
    }
    return output?0:3;
}

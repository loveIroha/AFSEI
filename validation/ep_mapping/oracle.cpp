// Export actual SERIAL source-DOLFIN CG1 interpolation and mesh indices.
// Active_land.h is read from a fixed git commit in AFSEI's reference cache.
#include <dolfin.h>
#include "Active_land.h"
#include <fstream>
#include <iomanip>
#include <set>

class CellVector : public dolfin::Expression {
public:
    std::shared_ptr<dolfin::MeshFunction<double>> c[3];
    explicit CellVector(const std::string& dir,const std::string& name,
                        std::shared_ptr<dolfin::Mesh> mesh):Expression(3) {
        for (int i=0;i<3;++i) c[i]=std::make_shared<dolfin::MeshFunction<double>>(
            mesh,dir+"/"+name+"_"+std::to_string(i)+".xml");
    }
    void eval(Eigen::Ref<Eigen::VectorXd> v,Eigen::Ref<const Eigen::VectorXd>,const ufc::cell& cell) const override {
        for (int i=0;i<3;++i) v[i]=(*c[i])[cell.index];
    }
};
class CellScalar : public dolfin::Expression {
public:
    bool owner;
    std::set<unsigned int> endo;
    explicit CellScalar(bool owner):owner(owner) {}
    void eval(Eigen::Ref<Eigen::VectorXd> v,Eigen::Ref<const Eigen::VectorXd>,const ufc::cell& cell) const override {
        v[0]=owner?cell.index:(endo.count(cell.index)?1.:0.);
    }
};
int main(int argc,char** argv) {
    dolfin::init(argc,argv); dolfin::set_log_active(false);
    if (argc!=3 || dolfin::MPI::size(MPI_COMM_WORLD)!=1) return 2;
    std::string dir=argv[1];
    auto mesh=std::make_shared<dolfin::Mesh>();
    { dolfin::XDMFFile input(dir+"/mesh_scale.xdmf"); input.read(*mesh); }
    auto V=std::make_shared<Active_land::Form_H_FunctionSpace_0>(mesh);
    auto S=std::make_shared<Active_land::Form_a_ep_FunctionSpace_0>(mesh);
    CellVector ef(dir,"fibers",mesh),es(dir,"sheets",mesh);
    CellScalar eo(true),em(false);
    dolfin::MeshFunction<std::size_t> boundaries(mesh,dir+"/boundaries.xml");
    for (dolfin::FacetIterator face(*mesh);!face.end();++face)
        if (boundaries[face->index()]==2)
            for (dolfin::CellIterator cell(*face);!cell.end();++cell) em.endo.insert(cell->index());
    dolfin::Function f(V),s(V),o(S),m(S);
    f.interpolate(ef); s.interpolate(es); o.interpolate(eo); m.interpolate(em);
    std::vector<double> fv,sv,ov,mv;
    f.compute_vertex_values(fv); s.compute_vertex_values(sv);
    o.compute_vertex_values(ov); m.compute_vertex_values(mv);
    std::ofstream output(argv[2]); output<<std::setprecision(17);
    auto count=mesh->num_vertices();
    for (dolfin::VertexIterator v(*mesh);!v.end();++v) {
        auto i=v->index(); auto p=v->point();
        output<<i<<','<<p.x()<<','<<p.y()<<','<<p.z();
        for (int j=0;j<3;++j) output<<','<<fv[j*count+i];
        for (int j=0;j<3;++j) output<<','<<sv[j*count+i];
        output<<','<<ov[i]<<','<<mv[i]<<'\n';
    }
    return output?0:3;
}

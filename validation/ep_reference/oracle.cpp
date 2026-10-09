// CPU oracle uses the ORIGINAL GPB and SDIRK implementations, not device RHS.
#include "GPB_cell_model.h"
#include "SDIRK3_Solver.h"
#include "../../src/afsi_torch/ep/csrc/land_device.h"
#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char** argv) {
    bool rhs_only=argc>1 && std::string(argv[1])=="rhs";
    int nodes,steps,land_live;
    double time,dt;
    if (!(std::cin>>nodes>>steps>>land_live>>time>>dt)) return 2;
    std::cout<<std::setprecision(17);
    for (int i=0;i<nodes;++i) {
        double voltage,stretch,zs,zw;
        LandState l;
        std::vector<double> y(57);
        std::cin>>voltage>>stretch>>zs>>zw>>l.XS>>l.XW>>l.CaTrpn>>l.TmB;
        for (auto& v:y) std::cin>>v;
        if (!std::cin) return 3;
        if (rhs_only) {
            auto r=GPBCellModel::compute_ionic_current(time,y,voltage);
            for (auto v:r.ydot) std::cout<<v<<' ';
            std::cout<<r.I_ion<<'\n';
            continue;
        }
        using Solver=SDIRK3Solver<std::vector<double>,GPBCellModel::Result>;
        Solver::Workspace workspace;
        Solver::Config cfg; cfg.max_iter=3; cfg.adaptive_iter=true; cfg.tol=1e-6;
        auto rhs=[voltage](double t,std::vector<double>& values,double) {
            return GPBCellModel::compute_ionic_current(t,values,voltage);
        };
        double sum=0;
        for (int k=0;k<steps;++k) {
            double current=Solver::step_with_workspace(rhs,time+k*dt,y,voltage,dt,workspace,cfg);
            if (!std::isfinite(current)) current=0;
            sum+=current;
            if (land_live) update_land_ode(l,y[37]*1000,stretch,zs,zw,dt);
        }
        for (auto v:y) std::cout<<v<<' ';
        std::cout<<l.XS<<' '<<l.XW<<' '<<l.CaTrpn<<' '<<l.TmB<<' '<<sum/steps<<'\n';
    }
}

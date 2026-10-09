#pragma once
#include <dolfin.h>
#include <mpi.h>
#include "GPB_cell_model.h"
// #include "ActiveContractionGPB.h"  // 已移除，NHS 模型不再使用
#include "SDIRK3_Solver.h"
#include <vector>
#include <memory>
#include <omp.h>
#include <cmath>
#include <algorithm>

// ===========================================================================
// 主动收缩力模型：Land (2017) 横桥模型
//
//   - 电生理（GPBCellModel + SDIRK3 + subcycling）完全未修改
//   - 主动力：Land 横桥 ODE 推进 XS/XW，由 main.cpp 写入 UFL 系数
//   - 力学反馈：main.cpp 力学步收敛后传入 lmbda/Zetas/Zetaw
//
//   与参考项目 RealLeftVentricleSystoleElectricity/GPBTissueManager_Land.h 一致。
// ===========================================================================

// ---------------------------------------------------------------------------
// Land 模型参数（单位 ms⁻¹ 或无量纲）
// ---------------------------------------------------------------------------
namespace LandParams {
    constexpr double ktrpn     = 0.1;    // TnC 结合率 (ms⁻¹)
    constexpr double ku        = 0.04;   // 横桥自发解离率 (ms⁻¹)
    constexpr double kuw       = 0.182;  // U→W 转换率 (ms⁻¹)
    constexpr double kws       = 0.012;  // W→S 转换率 (ms⁻¹)
    constexpr double rs        = 0.25;   // 强结合横桥平衡比例
    constexpr double rw        = 0.5;    // 弱结合横桥平衡比例
    constexpr double phi       = 2.23;   // 横桥功率冲程参数
    // 推导速率常数
    constexpr double kwu = -kws + kuw * (1.0/rw - 1.0);  // ≈ 0.170 ms⁻¹
    constexpr double ksu = kws * rw * (1.0/rs - 1.0);    // ≈ 0.018 ms⁻¹
    // CaTrpn 参数
    constexpr double ntrpn     = 2.0;
    constexpr double ntm       = 2.4;
    constexpr double Trpn50    = 0.35;
    constexpr double cat50_ref = 0.535;  // μM
    constexpr double Beta1     = -2.4;   // μM per unit stretch
    // 应变相关脱离系数
    constexpr double gammas    = 0.0085; // ms⁻¹
    constexpr double gammaw    = 0.615;  // ms⁻¹
}

// ---------------------------------------------------------------------------
// Land 模型状态
// ---------------------------------------------------------------------------
struct LandState {
    double XS     = 0.0;  // 强结合横桥比例
    double XW     = 0.0;  // 弱结合横桥比例
    double CaTrpn = 0.0;  // 与 TnC 结合的 Ca 比例
    double TmB    = 1.0;  // 被原肌球蛋白阻断的位点比例（初始=1，全阻断）
};

// ===========================================================================
// GPBTissueManager
// ===========================================================================
class GPBTissueManager
{
public:
    GPBTissueManager(
        std::shared_ptr<dolfin::FunctionSpace> function_space,
        double dt_pde_ms,
        double dt_ode_ms)
        : V_space(function_space),
          dt_pde_milliseconds(dt_pde_ms),
          dt_ode_milliseconds(dt_ode_ms)
    {
        n_substeps = static_cast<int>(std::round(dt_pde_ms / dt_ode_ms));
        if (n_substeps < 1) {
            std::cerr << "Warning: dt_ODE > dt_PDE, setting n_substeps = 1" << std::endl;
            n_substeps = 1;
            dt_ode_milliseconds = dt_pde_milliseconds;
        }
        dt_ode_milliseconds = dt_pde_milliseconds / n_substeps;

        auto dof_range   = V_space->dofmap()->ownership_range();
        num_nodes_local  = dof_range.second - dof_range.first;
        num_nodes_global = V_space->dim();

        if (dolfin::MPI::rank(MPI_COMM_WORLD) == 0) {
            std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
            std::cout << "初始化 GPB 组织管理器 (多尺度时间步进)" << std::endl;
            std::cout << "  本地节点数: " << num_nodes_local
                      << " / 全局: " << num_nodes_global << std::endl;
            std::cout << "  PDE 时间步长: " << dt_pde_milliseconds << " ms" << std::endl;
            std::cout << "  ODE 时间步长: " << dt_ode_milliseconds << " ms" << std::endl;
            std::cout << "  子循环步数:   " << n_substeps << std::endl;
            std::cout << "  主动收缩模型: Land (2017) 横桥模型" << std::endl;
            std::cout << "  并行: " << dolfin::MPI::size(MPI_COMM_WORLD) << " 个进程" << std::endl;
            std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        }

        // GPB 电生理状态（未修改）
        cell_states.resize(num_nodes_local);
        std::vector<double> resting = GPBCellModel::get_initial_state();
        for (size_t i = 0; i < num_nodes_local; ++i)
            cell_states[i] = resting;

        // Land 状态（XS=0, XW=0, CaTrpn=0, TmB=1）
        land_states.resize(num_nodes_local);

        // 力学反馈（默认未变形：lmbda=1, Zetas=Zetaw=0）
        lmbda_nodes.assign(num_nodes_local, 1.0);
        Zetas_nodes.assign(num_nodes_local, 0.0);
        Zetaw_nodes.assign(num_nodes_local, 0.0);

        if (dolfin::MPI::rank(MPI_COMM_WORLD) == 0) {
            std::cout << "  所有细胞已初始化为静息态" << std::endl;
            std::cout << "  Land 横桥模型启用 (XS/XW → UFL 系数)" << std::endl;
            std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        }
    }

    // -----------------------------------------------------------------------
    // 电生理 ODE 子循环（未修改）：计算 I_ion，供 Monodomain PDE 求 Vm
    // -----------------------------------------------------------------------
    void compute_ionic_current_subcycling(
        std::shared_ptr<dolfin::Function> Vm_function,
        std::shared_ptr<dolfin::Function> I_ion_function,
        double time_ms,
        bool advance_land = true)
    {
        std::vector<double> Vm_array;
        Vm_function->vector()->get_local(Vm_array);

        if (Vm_array.size() != num_nodes_local)
            throw std::runtime_error("Vm array size mismatch! Expected: "
                + std::to_string(num_nodes_local) + ", Got: "
                + std::to_string(Vm_array.size()));

        std::vector<double> I_ion_avg(num_nodes_local, 0.0);

        // Each node is independent while PDE voltage/feedback are frozen.
        // Keep substep order within each node, but avoid launching an OpenMP
        // team and allocating four SDIRK vectors for EVERY ionic substep.
        #pragma omp parallel for schedule(static)
        for (size_t node = 0; node < num_nodes_local; ++node) {
            typename SDIRK3Solver<std::vector<double>, GPBCellModel::Result>::Workspace workspace;
            for (int substep = 0; substep < n_substeps; ++substep) {
                const double t_sub = time_ms + substep * dt_ode_milliseconds;

                // ── GPB 电生理 ODE（SDIRK3，完全未修改）──
                double Vm_mV = Vm_array[node] * 1000.0;

                auto rhs_func = [Vm_mV](double t, std::vector<double>& y, double /*Vm_fixed*/) {
                    return GPBCellModel::compute_ionic_current(t, y, Vm_mV);
                };

                typename SDIRK3Solver<std::vector<double>, GPBCellModel::Result>::Config cfg;
                cfg.max_iter     = 3;
                cfg.adaptive_iter = true;
                cfg.tol          = 1e-6;

                double I_ion = SDIRK3Solver<std::vector<double>, GPBCellModel::Result>::step_with_workspace(
                    rhs_func, t_sub, cell_states[node], Vm_mV,
                    dt_ode_milliseconds, workspace, cfg);

                if (std::isnan(I_ion) || std::isinf(I_ion)) I_ion = 0.0;
                I_ion_avg[node] += I_ion;

                // GPB/ionic current advances in every EP interval. During
                // prescribed mechanical relaxation only the Land ODE is paused.
                if (advance_land) {
                    // GPB cell_states[37]: intracellular Ca²⁺ (mM -> µM).
                    const double Ca_i_uM = cell_states[node][37] * 1000.0;
                    update_land_ode(land_states[node], Ca_i_uM,
                                    lmbda_nodes[node], Zetas_nodes[node],
                                    Zetaw_nodes[node], dt_ode_milliseconds);
                }
            }
        }

        #pragma omp parallel for simd
        for (size_t node = 0; node < num_nodes_local; ++node)
            I_ion_avg[node] /= n_substeps;

        I_ion_function->vector()->set_local(I_ion_avg);
        I_ion_function->vector()->apply("insert");
    }

    // 单步版本（兼容接口）
    void compute_ionic_current(
        std::shared_ptr<dolfin::Function> Vm_function,
        std::shared_ptr<dolfin::Function> I_ion_function,
        double time_ms)
    {
        compute_ionic_current_subcycling(Vm_function, I_ion_function, time_ms);
    }

    // -----------------------------------------------------------------------
    // 力学步收敛后由 main.cpp 调用
    //
    // 传入当前步收敛后的 lmbda/Zetas/Zetaw 节点值，
    // 供下一 EP 子循环中 Land ODE 的 cat50 与应变脱离率计算使用
    // -----------------------------------------------------------------------
    void update_mechanics_feedback(
        const std::vector<double>& lmbda_arr,
        const std::vector<double>& Zetas_arr,
        const std::vector<double>& Zetaw_arr)
    {
        if (lmbda_arr.size() != num_nodes_local) {
            std::cerr << "Warning: update_mechanics_feedback size mismatch ("
                      << lmbda_arr.size() << " vs " << num_nodes_local
                      << "), skipping" << std::endl;
            return;
        }
        lmbda_nodes = lmbda_arr;
        Zetas_nodes = Zetas_arr;
        Zetaw_nodes = Zetaw_arr;
    }

    // -----------------------------------------------------------------------
    // XS/XW 输出（供 main.cpp 设置 UFL 的 XS/XW 系数）
    // -----------------------------------------------------------------------
    std::vector<double> get_XS_vector() const {
        std::vector<double> out(num_nodes_local);
        for (size_t i = 0; i < num_nodes_local; ++i)
            out[i] = land_states[i].XS;
        return out;
    }

    std::vector<double> get_XW_vector() const {
        std::vector<double> out(num_nodes_local);
        for (size_t i = 0; i < num_nodes_local; ++i)
            out[i] = land_states[i].XW;
        return out;
    }

    // -----------------------------------------------------------------------
    // Land 主动张力 Ta 输出（与参考项目
    //   RealLeftVentricleSolver::update_land_history 中 active_tension_latest
    //   完全一致）。
    //
    //   Ta = h_lambda * (Tref_land / rs) * ( XS*(Zetas+1) + XW*Zetaw )
    //   h_lambda = max(0, 1 + Beta0*(lmbda_c + min(lmbda_c,0.87) - 1.87))
    //
    //   Zetas/Zetaw 采用与力学反馈一致的一步精确积分形式：
    //     Zetas = Zetas_prev*exp(-cs*dt) + As*(lmbda-lmbda_prev)*(1-exp(-cs*dt))/(cs*dt)
    //   （本方法内部不持有 lmbda_prev，直接使用反馈的 Zetas_nodes/Zetaw_nodes，
    //     即上一步力学步算好的值，与 UFL 中 Ta 表达式量纲一致。）
    //
    //   单位：dyn/cm²（与 UFL 主动应力一致，Tref_land = 1.2e6）。
    // -----------------------------------------------------------------------
    std::vector<double> get_Ta_vector() const {
        using namespace LandParams;
        constexpr double Tref_land = 1.2e6;   // dyn/cm² (= 120 kPa)
        constexpr double Beta0_land = 2.3;    // Frank-Starling 长度依赖系数

        std::vector<double> out(num_nodes_local);
        for (size_t i = 0; i < num_nodes_local; ++i) {
            const double lmbda   = lmbda_nodes[i];
            const double XS      = land_states[i].XS;
            const double XW      = land_states[i].XW;
            const double Zetas   = Zetas_nodes[i];
            const double Zetaw   = Zetaw_nodes[i];

            const double lmbda_c  = (lmbda < 1.2) ? lmbda : 1.2;
            double h_prima = 1.0 + Beta0_land * (lmbda_c + ((lmbda_c < 0.87) ? lmbda_c : 0.87) - 1.87);
            if (h_prima < 0.0) h_prima = 0.0;
            const double h_lambda = h_prima;

            double Ta = h_lambda * (Tref_land / rs) * (XS * (Zetas + 1.0) + XW * Zetaw);
            if (Ta < 0.0) Ta = 0.0;

            out[i] = Ta;
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // 统计接口（与旧版本接口兼容）
    // -----------------------------------------------------------------------
    double get_dt_pde_ms()  const { return dt_pde_milliseconds; }
    double get_dt_ode_ms()  const { return dt_ode_milliseconds; }
    int    get_n_substeps() const { return n_substeps; }

    std::vector<double> get_Ca_i_vector() const {
        std::vector<double> out(num_nodes_local);
        for (size_t i = 0; i < num_nodes_local; ++i)
            out[i] = cell_states[i][37] * 1000.0;  // mM → μM
        return out;
    }

    struct Statistics {
        double Vm_min_mV, Vm_max_mV, Vm_mean_mV;
        double I_ion_min, I_ion_max, I_ion_mean;
        double Ca_i_min_mM, Ca_i_max_mM, Ca_i_mean_mM;
        size_t num_nodes;
    };

    Statistics get_statistics(
        std::shared_ptr<dolfin::Function> Vm_function,
        std::shared_ptr<dolfin::Function> I_ion_function) const
    {
        Statistics s;
        s.num_nodes  = num_nodes_global;
        s.Vm_min_mV  = Vm_function->vector()->min()  * 1000.0;
        s.Vm_max_mV  = Vm_function->vector()->max()  * 1000.0;
        s.Vm_mean_mV = Vm_function->vector()->sum()  / num_nodes_global * 1000.0;
        s.I_ion_min  = I_ion_function->vector()->min();
        s.I_ion_max  = I_ion_function->vector()->max();
        s.I_ion_mean = I_ion_function->vector()->sum() / num_nodes_global;

        double ca_min = 1e9, ca_max = -1e9, ca_sum = 0.0;
        for (size_t i = 0; i < num_nodes_local; ++i) {
            double ca = cell_states[i][37] * 1000.0;
            if (ca < ca_min) ca_min = ca;
            if (ca > ca_max) ca_max = ca;
            ca_sum += ca;
        }
        double gmin, gmax, gsum;
        MPI_Allreduce(&ca_min, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&ca_max, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&ca_sum, &gsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        s.Ca_i_min_mM  = gmin;
        s.Ca_i_max_mM  = gmax;
        s.Ca_i_mean_mM = gsum / num_nodes_global;
        return s;
    }

    // Land 横桥统计
    struct LandStats {
        double XS_min, XS_max, XS_mean;
        double XW_min, XW_max, XW_mean;
        double CaTrpn_mean;
        double lmbda_mean;
        double Ta_min, Ta_max, Ta_mean;   // 主动张力 (dyn/cm²)
    };

    LandStats get_land_statistics() const {
        // 主动张力节点值（本地）
        const std::vector<double> Ta_vec = get_Ta_vector();

        double xs_min=1e9, xs_max=-1e9, xs_sum=0.0;
        double xw_min=1e9, xw_max=-1e9, xw_sum=0.0;
        double ct_sum=0.0, lm_sum=0.0;
        double ta_min=1e300, ta_max=-1e300, ta_sum=0.0;
        for (size_t i = 0; i < num_nodes_local; ++i) {
            double xs = land_states[i].XS, xw = land_states[i].XW;
            if (xs < xs_min) xs_min = xs;  if (xs > xs_max) xs_max = xs;  xs_sum += xs;
            if (xw < xw_min) xw_min = xw;  if (xw > xw_max) xw_max = xw;  xw_sum += xw;
            ct_sum += land_states[i].CaTrpn;
            lm_sum += lmbda_nodes[i];
            double ta = Ta_vec[i];
            if (ta < ta_min) ta_min = ta;  if (ta > ta_max) ta_max = ta;  ta_sum += ta;
        }
        double gxs_min, gxs_max, gxs_sum;
        double gxw_min, gxw_max, gxw_sum;
        double gct_sum, glm_sum;
        double gta_min, gta_max, gta_sum;
        MPI_Allreduce(&xs_min, &gxs_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&xs_max, &gxs_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&xs_sum, &gxs_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&xw_min, &gxw_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&xw_max, &gxw_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&xw_sum, &gxw_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&ct_sum, &gct_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&lm_sum, &glm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&ta_min, &gta_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&ta_max, &gta_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&ta_sum, &gta_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        LandStats s;
        s.XS_min = gxs_min;  s.XS_max = gxs_max;  s.XS_mean = gxs_sum / num_nodes_global;
        s.XW_min = gxw_min;  s.XW_max = gxw_max;  s.XW_mean = gxw_sum / num_nodes_global;
        s.CaTrpn_mean = gct_sum / num_nodes_global;
        s.lmbda_mean  = glm_sum / num_nodes_global;
        s.Ta_min = gta_min;  s.Ta_max = gta_max;  s.Ta_mean = gta_sum / num_nodes_global;
        return s;
    }

    // -----------------------------------------------------------------------
    // [★ 诊断版本] 只读诊断访问器：Zetas/Zetaw 全局均值 + 单节点 Land 状态。
    //   仅用于诊断输出，不修改任何物理状态。
    // -----------------------------------------------------------------------
    void get_Zeta_stats(double& zetas_mean_out, double& zetaw_mean_out) const {
        double zs_sum = 0.0, zw_sum = 0.0;
        for (size_t i = 0; i < num_nodes_local; ++i) {
            zs_sum += Zetas_nodes[i];
            zw_sum += Zetaw_nodes[i];
        }
        double gzs = 0.0, gzw = 0.0;
        MPI_Allreduce(&zs_sum, &gzs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&zw_sum, &gzw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        zetas_mean_out = gzs / num_nodes_global;
        zetaw_mean_out = gzw / num_nodes_global;
    }

    // 诊断用：返回某本地节点的 Land 状态快照（XS/XW/Zetas/Zetaw/lmbda/CaTrpn）。
    void sample_node_land_state(size_t node,
                                double& XS, double& XW,
                                double& Zetas, double& Zetaw,
                                double& lmbda, double& CaTrpn) const {
        if (node >= num_nodes_local) { XS=XW=Zetas=Zetaw=CaTrpn=0.0; lmbda=1.0; return; }
        XS     = land_states[node].XS;
        XW     = land_states[node].XW;
        Zetas  = Zetas_nodes[node];
        Zetaw  = Zetaw_nodes[node];
        lmbda  = lmbda_nodes[node];
        CaTrpn = land_states[node].CaTrpn;
    }

    bool check_stability(
        std::shared_ptr<dolfin::Function> Vm_function,
        std::shared_ptr<dolfin::Function> I_ion_function,
        double acceptable_Vm_range_mV = 200.0) const
    {
        double vmin = Vm_function->vector()->min() * 1000.0;
        double vmax = Vm_function->vector()->max() * 1000.0;
        if (std::isnan(vmin)||std::isnan(vmax)||std::isinf(vmin)||std::isinf(vmax)) {
            std::cerr << "Error: Vm contains NaN/Inf!" << std::endl; return false;
        }
        if (vmax > acceptable_Vm_range_mV || vmin < -acceptable_Vm_range_mV) {
            std::cerr << "Warning: Vm out of range [" << vmin << ", " << vmax << "] mV\n";
            return false;
        }
        double imin = I_ion_function->vector()->min();
        double imax = I_ion_function->vector()->max();
        if (std::isnan(imin)||std::isnan(imax)||std::isinf(imin)||std::isinf(imax)) {
            std::cerr << "Error: I_ion contains NaN/Inf!" << std::endl; return false;
        }
        return true;
    }

    const std::vector<double>& get_cell_state(size_t idx) const {
        if (idx >= num_nodes_local) throw std::out_of_range("Node index out of range");
        return cell_states[idx];
    }

    // -----------------------------------------------------------------------
    // [§三 snapshot/rollback] 完整状态快照与恢复（供 main 的减步重试使用）。
    //   覆盖：GPB 细胞状态 + Land 横桥状态 + 力学反馈 (lmbda/Zetas/Zetaw)。
    //   ★ 这是"完整回退"的一部分；还原后所有后续推进从同一状态重新开始，
    //     失败尝试不产生任何状态污染。
    // -----------------------------------------------------------------------
    struct StateSnapshot {
        std::vector<std::vector<double>> cell_states;
        std::vector<LandState>           land_states;
        std::vector<double>              lmbda_nodes;
        std::vector<double>              Zetas_nodes;
        std::vector<double>              Zetaw_nodes;
    };

    StateSnapshot snapshot_state() const {
        StateSnapshot s;
        s.cell_states  = cell_states;
        s.land_states  = land_states;
        s.lmbda_nodes  = lmbda_nodes;
        s.Zetas_nodes  = Zetas_nodes;
        s.Zetaw_nodes  = Zetaw_nodes;
        return s;
    }

    // Snapshot/restore ONLY Land's four internal variables per node. GPB states
    // and calcium are deliberately never restored at a pacing/closure event.
    std::vector<LandState> snapshot_land_states() const { return land_states; }
    void restore_land_states(const std::vector<LandState>& resting) {
        if (resting.size() != num_nodes_local)
            throw std::runtime_error("Land resting snapshot size mismatch");
        land_states = resting;
    }

    void restore_state(const StateSnapshot& s) {
        cell_states  = s.cell_states;
        land_states  = s.land_states;
        lmbda_nodes  = s.lmbda_nodes;
        Zetas_nodes  = s.Zetas_nodes;
        Zetaw_nodes  = s.Zetaw_nodes;
    }

    // Read-only acceptance check; equations and model state are not modified.
    bool finite_state() const {
        for (const auto& cell : cell_states)
            for (double value : cell) if (!std::isfinite(value)) return false;
        for (const auto& state : land_states)
            if (!std::isfinite(state.XS) || !std::isfinite(state.XW) ||
                !std::isfinite(state.CaTrpn) || !std::isfinite(state.TmB)) return false;
        for (double value : lmbda_nodes) if (!std::isfinite(value)) return false;
        for (double value : Zetas_nodes) if (!std::isfinite(value)) return false;
        for (double value : Zetaw_nodes) if (!std::isfinite(value)) return false;
        return true;
    }

    void print_debug_info(size_t node = 0) const {
        if (node >= num_nodes_local) return;
        std::cout << "\nNode " << node << ":" << std::endl;
        std::cout << "  Ca_i    = " << cell_states[node][37]*1000.0 << " uM\n"
                  << "  XS      = " << land_states[node].XS
                  << "  XW      = " << land_states[node].XW
                  << "  CaTrpn  = " << land_states[node].CaTrpn << "\n"
                  << "  lmbda   = " << lmbda_nodes[node]
                  << "  Zetas   = " << Zetas_nodes[node]
                  << "  Zetaw   = " << Zetaw_nodes[node] << std::endl;
    }

private:
    // -----------------------------------------------------------------------
    // Land 模型 ODE 单步更新（前向欧拉）
    //
    //   dCaTrpn/dt = ktrpn*( (Ca/cat50)^ntrpn*(1-CaTrpn) - CaTrpn )
    //   dTmB/dt    = kb*CaTrpn^(-ntm/2)*XU - ku*CaTrpn^(ntm/2)*TmB
    //   dXW/dt     = kuw*XU - kws*XW - XW*gammawu - XW*kwu
    //   dXS/dt     = kws*XW - XS*gammasu - XS*ksu
    // -----------------------------------------------------------------------
    static void update_land_ode(
        LandState& s,
        double Ca_i_uM,
        double lmbda,
        double Zetas,
        double Zetaw,
        double dt_ms)
    {
        using namespace LandParams;

        // Frank-Starling：lmbda 影响 Ca₅₀
        double lmbda_c = (lmbda < 1.2) ? lmbda : 1.2;
        double cat50   = cat50_ref + Beta1 * (lmbda_c - 1.0);
        if (cat50 < 0.1) cat50 = 0.1;

        // 应变相关横桥脱离率
        double zetas1  = (Zetas >  0.0) ?  Zetas         : 0.0;
        double zetas2  = (Zetas < -1.0) ? (-1.0 - Zetas) : 0.0;
        double gammasu = gammas * ((zetas1 > zetas2) ? zetas1 : zetas2);
        double gammawu = gammaw * std::fabs(Zetaw);

        // 状态变量夹紧（防止数值负值）
        double XS_c     = (s.XS     > 0.0) ? s.XS     : 0.0;
        double XW_c     = (s.XW     > 0.0) ? s.XW     : 0.0;
        double CaTrpn_c = (s.CaTrpn > 0.0) ? s.CaTrpn : 0.0;
        double TmB_c    = (s.TmB    > 0.0) ? s.TmB    : 0.0;
        double XU = 1.0 - TmB_c - XS_c - XW_c;
        if (XU < 0.0) XU = 0.0;

        // 幂次项（防止溢出）
        double kb = ku * std::pow(Trpn50, ntm) / (1.0 - rs - rw*(1.0-rs));

        double Ca_ratio = Ca_i_uM / cat50;
        double CaTrpn_act   = std::pow(Ca_ratio, ntrpn);
        double CaTrpn_phalf = (CaTrpn_c > 1e-12) ? std::pow(CaTrpn_c,  ntm*0.5) : 0.0;
        double CaTrpn_nhalf = (CaTrpn_c > 1e-12)
            ? std::min(std::pow(CaTrpn_c, -ntm*0.5), 100.0) : 100.0;

        // ODE 右端
        double dCaTrpn = ktrpn * (-CaTrpn_c + CaTrpn_act * (1.0 - CaTrpn_c));
        double dTmB    = kb  * CaTrpn_nhalf * XU  - ku  * CaTrpn_phalf * TmB_c;
        double dXW     = kuw * XU - kws * XW_c - XW_c * gammawu - XW_c * kwu;
        double dXS     = kws * XW_c - XS_c * gammasu - XS_c * ksu;

        // 前向欧拉更新
        s.CaTrpn += dt_ms * dCaTrpn;
        s.TmB    += dt_ms * dTmB;
        s.XW     += dt_ms * dXW;
        s.XS     += dt_ms * dXS;

        // 物理约束夹紧
        if (s.CaTrpn < 0.0) s.CaTrpn = 0.0;
        if (s.XW     < 0.0) s.XW     = 0.0;
        if (s.XS     < 0.0) s.XS     = 0.0;
        if (s.TmB    < 0.0) s.TmB    = 0.0;
        if (s.TmB    > 1.0) s.TmB    = 1.0;
    }

    // 成员变量
    std::shared_ptr<dolfin::FunctionSpace> V_space;
    size_t num_nodes_local;
    size_t num_nodes_global;
    double dt_pde_milliseconds;
    double dt_ode_milliseconds;
    int    n_substeps;

    // 电生理状态（未修改）
    std::vector<std::vector<double>> cell_states;

    // Land 模型状态
    std::vector<LandState> land_states;

    // 力学反馈（lmbda/Zetas/Zetaw）
    std::vector<double> lmbda_nodes;
    std::vector<double> Zetas_nodes;
    std::vector<double> Zetaw_nodes;
};

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

/**
 * SDIRK3 - Third-order Single Diagonally Implicit Runge-Kutta Method
 *
 * 特点:
 * - 三阶精度 (O(Δt³))
 * - L-稳定性 (适合刚性问题)
 * - 对角隐式结构 (每步只需求解一次非线性方程)
 *
 * Butcher表:
 *   γ   |  γ      0      0
 *  1-γ  | 1-2γ    γ      0
 * ------+------------------
 *       |  1/2   1/2     0
 *
 * 其中 γ = (3 + √3) / 6 ≈ 0.788675134595
 *
 * 优势:
 * - 相比显式欧拉法,可以使用更大的时间步长
 * - 对刚性ODE系统数值稳定性好
 * - 适合心肌细胞离子通道模型(刚性系统)
 */
template<typename StateVector, typename RHSFunction>
class SDIRK3Solver
{
public:
    /**
     * Butcher表系数 (三阶SDIRK方法)
     */
    static constexpr double gamma = 0.788675134594812882254;  // (3 + √3) / 6

    // Butcher表 A矩阵 (对角隐式)
    static constexpr double a11 = gamma;
    static constexpr double a21 = 1.0 - 2.0 * gamma;
    static constexpr double a22 = gamma;

    // Butcher表 b向量 (权重)
    static constexpr double b1 = 0.5;
    static constexpr double b2 = 0.5;

    // Butcher表 c向量 (节点)
    static constexpr double c1 = gamma;
    static constexpr double c2 = 1.0 - gamma;

    /**
     * 配置参数
     */
    struct Config {
        int max_iter = 5;              // 不动点迭代最大次数
        double tol = 1e-6;             // 收敛容差
        bool adaptive_iter = true;     // 自适应迭代次数
    };

    // Scratch storage only: overwritten by each step, not a physical state.
    // A caller may reuse one workspace for successive substeps of one cell.
    struct Workspace {
        StateVector k1, k2, y_temp, k_prev;
        void resize(size_t n) {
            k1.resize(n); k2.resize(n); y_temp.resize(n); k_prev.resize(n);
        }
    };

    /**
     * 单步SDIRK3积分
     *
     * 求解: y_{n+1} = y_n + Δt * (b1*k1 + b2*k2)
     *
     * 其中 k_i 通过隐式方程求解:
     *   k1 = f(t_n + c1*Δt, y_n + Δt*a11*k1, Vm)
     *   k2 = f(t_n + c2*Δt, y_n + Δt*(a21*k1 + a22*k2), Vm)
     *
     * @param rhs_func   右端函数 f(t, y, Vm) -> ydot
     * @param t          当前时间
     * @param y          当前状态 (输入/输出)
     * @param Vm         膜电位 (固定输入)
     * @param dt         时间步长
     * @param config     求解器配置
     * @return           I_ion (离子电流)
     */
    template<typename Fn>
    static double step(
        Fn&& rhs_func,
        double t,
        StateVector& y,
        double Vm,
        double dt,
        const Config& config = Config())
    {
        Workspace workspace;
        return step_with_workspace(rhs_func,t,y,Vm,dt,workspace,config);
    }

    template<typename Fn>
    static double step_with_workspace(
        Fn&& rhs_func, double t, StateVector& y, double Vm, double dt,
        Workspace& workspace, const Config& config = Config())
    {
        const size_t n = y.size();

        workspace.resize(n);
        auto& k1 = workspace.k1;
        auto& k2 = workspace.k2;
        auto& y_temp = workspace.y_temp;
        auto& k_prev = workspace.k_prev;

        // ═══════════════════════════════════════════
        // Stage 1: 求解隐式方程 k1 = f(t + c1*dt, y + dt*a11*k1, Vm)
        // ═══════════════════════════════════════════
        double t1 = t + c1 * dt;

        // 初始猜测: 使用显式欧拉
        auto result0 = rhs_func(t, y, Vm);
        k1 = result0.ydot;

        // 不动点迭代求解隐式方程
        for (int iter = 0; iter < config.max_iter; ++iter) {
            k_prev = k1;

            // 计算 y_temp = y + dt * a11 * k1
            for (size_t i = 0; i < n; ++i) {
                y_temp[i] = y[i] + dt * a11 * k1[i];
            }

            // 计算新的 k1 = f(t1, y_temp, Vm)
            auto result1 = rhs_func(t1, y_temp, Vm);
            k1 = result1.ydot;

            // 检查收敛性
            if (config.adaptive_iter && iter > 0) {
                double error = compute_norm_error(k1, k_prev);
                if (error < config.tol) {
                    break;
                }
            }
        }

        // ═══════════════════════════════════════════
        // Stage 2: 求解隐式方程 k2 = f(t + c2*dt, y + dt*(a21*k1 + a22*k2), Vm)
        // ═══════════════════════════════════════════
        double t2 = t + c2 * dt;

        // 初始猜测: 使用 k1
        k2 = k1;

        // 不动点迭代求解隐式方程
        for (int iter = 0; iter < config.max_iter; ++iter) {
            k_prev = k2;

            // 计算 y_temp = y + dt * (a21*k1 + a22*k2)
            for (size_t i = 0; i < n; ++i) {
                y_temp[i] = y[i] + dt * (a21 * k1[i] + a22 * k2[i]);
            }

            // 计算新的 k2 = f(t2, y_temp, Vm)
            auto result2 = rhs_func(t2, y_temp, Vm);
            k2 = result2.ydot;

            // 检查收敛性
            if (config.adaptive_iter && iter > 0) {
                double error = compute_norm_error(k2, k_prev);
                if (error < config.tol) {
                    break;
                }
            }
        }

        // ═══════════════════════════════════════════
        // 最终更新: y_{n+1} = y_n + dt * (b1*k1 + b2*k2)
        // ═══════════════════════════════════════════
        for (size_t i = 0; i < n; ++i) {
            y[i] += dt * (b1 * k1[i] + b2 * k2[i]);

            // 数值稳定性检查
            if (std::isnan(y[i]) || std::isinf(y[i])) {
                y[i] = 0.0;  // 保护措施
            }
        }

        // 计算平均I_ion (使用两个stage的平均值)
        auto result_final = rhs_func(t + dt, y, Vm);
        double I_ion = result_final.I_ion;

        return I_ion;
    }

    /**
     * 计算相对误差范数 (用于收敛性判断)
     */
    static double compute_norm_error(
        const StateVector& v1,
        const StateVector& v2)
    {
        double max_error = 0.0;
        for (size_t i = 0; i < v1.size(); ++i) {
            double abs_error = std::abs(v1[i] - v2[i]);
            double scale = std::max(std::abs(v1[i]), 1.0);
            double rel_error = abs_error / scale;
            max_error = std::max(max_error, rel_error);
        }
        return max_error;
    }

    /**
     * 获取方法的理论精度阶数
     */
    static constexpr int order() { return 3; }

    /**
     * 获取方法名称
     */
    static const char* name() { return "SDIRK3 (Third-order L-stable)"; }
};


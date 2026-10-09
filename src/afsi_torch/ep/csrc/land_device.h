// Generated from GPBTissueManager_Land.h at 5bdde73c6be10cefadb005a0fcec5fdb6521f336; do not hand edit.
#pragma once
#include <cmath>
#include <algorithm>
#ifdef __CUDACC__
#define AFSEI_LAND_DEVICE __host__ __device__
#else
#define AFSEI_LAND_DEVICE
#endif
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
struct LandState {

    double XS     = 0.0;  // 强结合横桥比例
    double XW     = 0.0;  // 弱结合横桥比例
    double CaTrpn = 0.0;  // 与 TnC 结合的 Ca 比例
    double TmB    = 1.0;  // 被原肌球蛋白阻断的位点比例（初始=1，全阻断）

};
AFSEI_LAND_DEVICE inline void update_land_ode(LandState& s, double Ca_i_uM, double lmbda, double Zetas, double Zetaw, double dt_ms) {

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
#undef AFSEI_LAND_DEVICE

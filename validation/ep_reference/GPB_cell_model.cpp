
#include "GPB_cell_model.h"
#include <algorithm>

// Optimized tissue simulation interface
GPBCellModel::Result GPBCellModel::compute_ionic_current(
    double t,
    std::vector<double>& y,
    double Vm)
{
    Result result;
    result.ydot.resize(CELL_STATE_DIM, 0.0);
    
    // Create temporary full state vector for internal calculations
    // This is just for convenience - we copy to/from the 57-state vector
    std::vector<double> y_full(FULL_ODE_DIM, 0.0);
    
    // Copy cellular states into full vector
    for (int i = 0; i < 38; ++i) {
        y_full[i] = y[i];  // States before Vm
    }
    y_full[38] = Vm;  // Insert Vm at index 38
    for (int i = 38; i < CELL_STATE_DIM; ++i) {
        y_full[i + 1] = y[i];  // States after Vm (shifted)
    }
    
    // Call the full model to get derivatives
    // We pass I_app = 0 since stimulus is handled by PDE
    Result full_result = f_heartfailure_1(t, y_full, Vm, 0.0);
    
    // Extract derivatives for cellular states (excluding Vm at index 38)
    for (int i = 0; i < 38; ++i) {
        result.ydot[i] = full_result.ydot[i];  // States before Vm
    }
    for (int i = 38; i < CELL_STATE_DIM; ++i) {
        result.ydot[i] = full_result.ydot[i + 1];  // States after Vm (shifted)
    }
    
    // Copy ionic current
    result.I_ion = full_result.I_ion;
    
    return result;
}

// Get initial cellular states (excluding Vm)
std::vector<double> GPBCellModel::get_initial_state()
{
    // Full initial conditions (58 states including Vm)
    std::vector<double> y0_full = {
        0.00371917311175298, 0.629563145653617, 0.628156366619018, 2.87807501948292e-06, 0.995191678834277,
        0.0256455386561491, 0.0166702462776996, 0.000437233937773851, 0.789865222262672, 0.000437226960888769,
        0.999995922401368, 0.0182692097720730, 0.00425180082489623, 0.911423257257461, 1.10337211796779e-06,
        1.07229840398973e-07, 3.53795718160598, 0.772022201763961, 0.0106472771801263, 0.124747660339223,
        0.00709016718964759, 0.000361565169272969, 0.00291593396950060, 0.136552392811308, 0.00259137579210590,
        0.00770065038544987, 0.0108451851114268, 0.0754357237706355, 0.122427602372974, 1.33525927996781,
        0.686242333225035, 8.78601782663918, 8.78614841263188, 8.78640538366833, 120.0, 0.000182997516145308,
        0.000117027510847066, 0.000107032812487738, -81.5455936324844, 0.9946, 1.0, 0.00273690302369662,
        0.168523952444531, 0.1494, 0.4071, 0.4161, -0.314529004, 0.0001, 0.0006, 0.0008,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };
    
    // Extract cellular states (excluding Vm at index 38)
    std::vector<double> y0(CELL_STATE_DIM);
    for (int i = 0; i < 38; ++i) {
        y0[i] = y0_full[i];  // States before Vm
    }
    for (int i = 38; i < CELL_STATE_DIM; ++i) {
        y0[i] = y0_full[i + 1];  // States after Vm (skip index 38)
    }
    
    return y0;
}

// State names for debugging
const char* GPBCellModel::get_state_name(int tissue_idx)
{
    static const char* names[FULL_ODE_DIM] = {
        "m", "h", "j",                    // 0-2: Na channel
        "d", "f", "fCa_junc", "fCa_sl",   // 3-6: L-type Ca channel
        "xtos", "ytos", "xtof", "ytof",   // 7-10: Ito
        "xkr", "xks",                     // 11-12: IKr, IKs
        "RyR_R", "RyR_O", "RyR_I",        // 13-15: RyR states
        "NaBj", "NaBsl",                  // 16-17: Na buffers
        "TnCL", "TnCHc", "TnCHm",         // 18-20: Troponin
        "CaM",                            // 21: Calmodulin
        "Myosin_ca", "Myosin_mg",         // 22-23: Myosin
        "SRB",                            // 24: SR buffer
        "SLLj", "SLLsl", "SLHj", "SLHsl", // 25-28: SL buffers
        "Csqn", "Ca_sr",                  // 29-30: SR calcium
        "Na_junc", "Na_sl", "Na_i",       // 31-33: Na concentrations
        "K_i",                            // 34: K concentration
        "Ca_junc", "Ca_sl", "Ca_i",       // 35-37: Ca concentrations
        "Vm",                             // 38: Membrane potential (not in tissue state)
        "unused1", "unused2",             // 39-40: Unused
        "mL", "hL",                       // 41-42: Late Na channel
        "markov_iks_0", "markov_iks_1", "markov_iks_2", "markov_iks_3", "markov_iks_4",
        "markov_iks_5", "markov_iks_6", "markov_iks_7", "markov_iks_8", "markov_iks_9",
        "markov_iks_10", "markov_iks_11", "markov_iks_12", "markov_iks_13", "markov_iks_14"
    };
    
    int full_idx = tissue_to_full_index(tissue_idx);
    if (full_idx >= 0 && full_idx < FULL_ODE_DIM) {
        return names[full_idx];
    }
    return "unknown";
}

GPBCellModel::Result GPBCellModel::f_heartfailure_1(
    double t,
    std::vector<double>& y,
    double Vm,
    double I_app)
{
    Result result;
    result.ydot.resize(FULL_ODE_DIM, 0.0);
    
    // Model Parameters
    const int epi = 1;
    const int heartfailureConditions = 2;
    
    double I_NaL_factor, tau_hL_factor, I_to_factor, I_Ki_factor, I_NaK_factor;
    double I_Nab_factor, I_Cab_factor, I_NCX_factor, J_serca_factor, I_leak_factor, EC_50sr_factor;
    
    if (heartfailureConditions == 1) {
        I_NaL_factor = 2.0;
        tau_hL_factor = 2.0;
        I_to_factor = 1.0 - 0.6;
        I_Ki_factor = 1.0 - 0.32;
        I_NaK_factor = 1.0 - 0.5;
        I_Nab_factor = 0.0;
        I_Cab_factor = 1.53;
        I_NCX_factor = 1.75;
        J_serca_factor = 1.0 - 0.5;
        I_leak_factor = 5.0;
        EC_50sr_factor = 1.0 - 0.11;
    } else {
        I_NaL_factor = 1.0;
        tau_hL_factor = 1.0;
        I_to_factor = 1.0;
        I_Ki_factor = 1.0;
        I_NaK_factor = 1.0;
        I_Nab_factor = 1.0;
        I_Cab_factor = 1.0;
        I_NCX_factor = 1.0;
        J_serca_factor = 1.0;
        I_leak_factor = 1.0;
        EC_50sr_factor = 1.0;
    }
    
    // Constants
    const double R = 8314.0;
    const double Frdy = 96485.0;
    const double Temp = 310.0;
    const double FoRT = Frdy / R / Temp;
    const double Cmem = 1.3810e-10;
    const double Qpow = (Temp - 310.0) / 10.0;
    
    // Cell geometry
    const double cellLength = 100.0;
    const double cellRadius = 10.25;
    const double Vcell = M_PI * cellRadius * cellRadius * cellLength * 1e-15;
    const double Vmyo = 0.65 * Vcell;
    const double Vsr = 0.035 * Vcell;
    const double Vsl = 0.02 * Vcell;
    const double Vjunc = 0.0539 * 0.01 * Vcell;
    
    const double J_ca_juncsl = 1.0 / 1.2134e12 * 1.5;
    const double J_ca_slmyo = 1.0 / 2.68510e11 * 2.0;
    const double J_na_juncsl = 1.0 / (1.6382e12 / 3.0 * 100.0);
    const double J_na_slmyo = 1.0 / (1.8308e10 / 3.0 * 100.0);
    
    // Fractional currents
    const double Fjunc = 0.11;
    const double Fsl = 1.0 - Fjunc;
    const double Fjunc_CaL = 0.9;
    const double Fsl_CaL = 1.0 - Fjunc_CaL;
    
    // Fixed ion concentrations
    const double Cli = 15.0;
    const double Clo = 150.0;
    const double Ko = 5.4;
    const double Nao = 140.0;
    const double Cao = 1.8;
    const double Mgi = 1.0;
    
    // Nernst Potentials
    double ena_junc = (1.0 / FoRT) * log(Nao / y[31]);
    double ena_sl = (1.0 / FoRT) * log(Nao / y[32]);
    double ek = (1.0 / FoRT) * log(Ko / y[34]);
    double eca_junc = (1.0 / FoRT / 2.0) * log(Cao / y[35]);
    double eca_sl = (1.0 / FoRT / 2.0) * log(Cao / y[36]);
    double ecl = (1.0 / FoRT) * log(Cli / Clo);
    
    // Na transport parameters
    const double GNa = 23.0;
    const double GNaB = 0.597e-3;
    const double IbarNaK = 1.0 * 1.8;
    const double KmNaip = 11.0;
    const double KmKo = 1.5;
    
    // K current parameters
    const double pNaK = 0.01833;
    const double gkp = 2.0 * 0.001;
    
    // Cl current parameters
    const double GClCa = 0.5 * 0.109625;
    const double GClB = 1.0 * 9e-3;
    const double KdClCa = 100e-3;
    
    // I_Ca parameters
    const double pNa = 0.50 * 1.5e-8;
    const double pCa = 0.50 * 5.4e-4;
    const double pK = 0.50 * 2.7e-7;
    const double Q10CaL = 1.8;
    
    // Ca transport parameters
    const double IbarNCX = 1.0 * 4.5;
    const double KmCai = 3.59e-3;
    const double KmCao = 1.3;
    const double KmNai = 12.29;
    const double KmNao = 87.5;
    const double ksat = 0.32;
    const double nu = 0.27;
    const double Kdact = 0.150e-3;
    const double Q10NCX = 1.57;
    const double IbarSLCaP = 0.0673;
    const double KmPCa = 0.5e-3;
    const double GCaB = 5.513e-4;
    const double Q10SLCaP = 2.35;
    
    // SR flux parameters
    const double Q10SRCaP = 2.6;
    const double Vmax_SRCaP = 1.0 * 5.3114e-3;
    const double Kmf = 0.246e-3;
    const double Kmr = 1.7;
    const double hillSRCaP = 1.787;
    const double ks = 25.0;
    const double koCa = 10.0;
    const double kom = 0.06;
    const double kiCa = 0.5;
    const double kim = 0.005;
    const double ec50SR = 0.45 * EC_50sr_factor;
    
    // Buffering parameters
    const double Bmax_Naj = 7.561;
    const double Bmax_Nasl = 1.65;
    const double koff_na = 1e-3;
    const double kon_na = 0.1e-3;
    const double Bmax_TnClow = 70e-3;
    const double koff_tncl = 19.6e-3;
    const double kon_tncl = 32.7;
    const double Bmax_TnChigh = 140e-3;
    const double koff_tnchca = 0.032e-3;
    const double kon_tnchca = 2.37;
    const double koff_tnchmg = 3.33e-3;
    const double kon_tnchmg = 3e-3;
    const double Bmax_CaM = 24e-3;
    const double koff_cam = 238e-3;
    const double kon_cam = 34.0;
    const double Bmax_myosin = 140e-3;
    const double koff_myoca = 0.46e-3;
    const double kon_myoca = 13.8;
    const double koff_myomg = 0.057e-3;
    const double kon_myomg = 0.0157;
    const double Bmax_SR = 19.0 * 0.9e-3;
    const double koff_sr = 60e-3;
    const double kon_sr = 100.0;
    const double Bmax_SLlowsl = 37.4e-3 * Vmyo / Vsl;
    const double Bmax_SLlowj = 4.6e-3 * Vmyo / Vjunc * 0.1;
    const double koff_sll = 1300e-3;
    const double kon_sll = 100.0;
    const double Bmax_SLhighsl = 13.4e-3 * Vmyo / Vsl;
    const double Bmax_SLhighj = 1.65e-3 * Vmyo / Vjunc * 0.1;
    const double koff_slh = 30e-3;
    const double kon_slh = 100.0;
    const double Bmax_Csqn = 140e-3 * Vmyo / Vsr;
    const double koff_csqn = 65.0;
    const double kon_csqn = 100.0;
    
    // Use Vm instead of y[38] (voltage is passed as parameter)
    y[38] = Vm;
    
    // I_Na: Fast Na Current
    double mss = 1.0 / pow((1.0 + exp(-(56.86 + y[38]) / 9.03)), 2);
    double taum = 0.1292 * exp(-pow((y[38] + 45.79) / 15.54, 2)) + 
                  0.06487 * exp(-pow((y[38] - 4.823) / 51.12, 2));
    
    double ah = (y[38] >= -40.0) ? 0.0 : (0.057 * exp(-(y[38] + 80.0) / 6.8));
    double bh = (y[38] >= -40.0) ? (0.77 / (0.13 * (1.0 + exp(-(y[38] + 10.66) / 11.1)))) :
                ((2.7 * exp(0.079 * y[38]) + 3.1e5 * exp(0.3485 * y[38])));
    double tauh = 1.0 / (ah + bh);
    double hss = 1.0 / pow((1.0 + exp((y[38] + 71.55) / 7.43)), 2);
    
    double aj = (y[38] >= -40.0) ? 0.0 :
                (((-2.5428e4 * exp(0.2444 * y[38]) - 6.948e-6 * exp(-0.04391 * y[38])) * 
                  (y[38] + 37.78)) / (1.0 + exp(0.311 * (y[38] + 79.23))));
    double bj = (y[38] >= -40.0) ? 
                ((0.6 * exp(0.057 * y[38])) / (1.0 + exp(-0.1 * (y[38] + 32.0)))) :
                ((0.02424 * exp(-0.01052 * y[38])) / (1.0 + exp(-0.1378 * (y[38] + 40.14))));
    double tauj = 1.0 / (aj + bj);
    double jss = 1.0 / pow((1.0 + exp((y[38] + 71.55) / 7.43)), 2);
    
    result.ydot[0] = (mss - y[0]) / taum;
    result.ydot[1] = (hss - y[1]) / tauh;
    result.ydot[2] = (jss - y[2]) / tauj;
    
    double I_Na_junc = Fjunc * GNa * pow(y[0], 3) * y[1] * y[2] * (y[38] - ena_junc);
    double I_Na_sl = Fsl * GNa * pow(y[0], 3) * y[1] * y[2] * (y[38] - ena_sl);
    double I_Na = I_Na_junc + I_Na_sl;
    
    // I_Na_L: Late Na Current
    double gNaL = 0.008;
    double tauhL = 233.0 * tau_hL_factor;
    double a_na_L = 0.32 * (y[38] + 47.13) / (1.0 - exp(-0.1 * (y[38] + 47.13)));
    double b_na_L = 0.08 * exp(-y[38] / 11.0);
    double h_na_L = 1.0 / (1.0 + exp((y[38] + 91.0) / 6.1));
    result.ydot[41] = a_na_L * (1.0 - y[41]) - b_na_L * y[41];
    result.ydot[42] = (h_na_L - y[42]) / tauhL;
    double I_Na_L_junc = I_NaL_factor * Fjunc * gNaL * pow(y[41], 3) * y[42] * (y[38] - ena_junc);
    double I_Na_L_sl = I_NaL_factor * Fsl * gNaL * pow(y[41], 3) * y[42] * (y[38] - ena_sl);
    
    // I_nabk: Na Background Current
    double I_nabk_junc = Fjunc * GNaB * (y[38] - ena_junc) * I_Nab_factor;
    double I_nabk_sl = Fsl * GNaB * (y[38] - ena_sl) * I_Nab_factor;
    double I_nabk = I_nabk_junc + I_nabk_sl;
    
    // I_nak: Na/K Pump Current
    double sigma = (exp(Nao / 67.3) - 1.0) / 7.0;
    double fnak = 1.0 / (1.0 + 0.1245 * exp(-0.1 * y[38] * FoRT) + 0.0365 * sigma * exp(-y[38] * FoRT));
    double I_nak_junc = Fjunc * IbarNaK * fnak * Ko / (1.0 + pow(KmNaip / y[31], 4)) / (Ko + KmKo) * I_NaK_factor;
    double I_nak_sl = Fsl * IbarNaK * fnak * Ko / (1.0 + pow(KmNaip / y[32], 4)) / (Ko + KmKo) * I_NaK_factor;
    double I_nak = I_nak_junc + I_nak_sl;
    
    // I_kr: Rapidly Activating K Current
    double gkr = 1.0 * 0.035 * sqrt(Ko / 5.4);
    double xrss = 1.0 / (1.0 + exp(-(y[38] + 10.0) / 5.0));
    double tauxr = 550.0 / (1.0 + exp((-22.0 - y[38]) / 9.0)) * 6.0 / (1.0 + exp((y[38] - (-11.0)) / 9.0)) + 
                   230.0 / (1.0 + exp((y[38] - (-40.0)) / 20.0));
    result.ydot[11] = (xrss - y[11]) / tauxr;
    double rkr = 1.0 / (1.0 + exp((y[38] + 74.0) / 24.0));
    double I_kr = gkr * y[11] * rkr * (y[38] - ek);
    
    // I_ks: Slowly Activating K Current
    double eks = (1.0 / FoRT) * log((Ko + pNaK * Nao) / (y[34] + pNaK * y[33]));
    double gks_junc = 1.0 * 0.0035;
    double gks_sl = 1.0 * 0.0035;
    double xsss = 1.0 / (1.0 + exp(-(y[38] + 3.8) / 14.25));
    double tauxs = 990.1 / (1.0 + exp(-(y[38] + 2.436) / 14.12));
    result.ydot[12] = (xsss - y[12]) / tauxs;
    double I_ks_junc = Fjunc * gks_junc * pow(y[12], 2) * (y[38] - eks);
    double I_ks_sl = Fsl * gks_sl * pow(y[12], 2) * (y[38] - eks);
    double I_ks = I_ks_junc + I_ks_sl;
    
    // I_kp: Plateau K current
    double kp_kp = 1.0 / (1.0 + exp(7.488 - y[38] / 5.98));
    double I_kp_junc = Fjunc * gkp * kp_kp * (y[38] - ek);
    double I_kp_sl = Fsl * gkp * kp_kp * (y[38] - ek);
    double I_kp = I_kp_junc + I_kp_sl;
    
    // I_to: Transient Outward K Current
    double GtoSlow, GtoFast;
    if (epi == 1) {
        GtoSlow = 1.0 * 0.13 * 0.12;
        GtoFast = 1.0 * 0.13 * 0.88;
    } else {
        GtoSlow = 0.13 * 0.3 * 0.964;
        GtoFast = 0.13 * 0.3 * 0.036;
    }
    
    double xtoss = 1.0 / (1.0 + exp(-(y[38] - 19.0) / 13.0));
    double ytoss = 1.0 / (1.0 + exp((y[38] + 19.5) / 5.0));
    double tauxtos = 9.0 / (1.0 + exp((y[38] + 3.0) / 15.0)) + 0.5;
    double tauytos = 800.0 / (1.0 + exp((y[38] + 60.0) / 10.0)) + 30.0;
    result.ydot[7] = (xtoss - y[7]) / tauxtos;
    result.ydot[8] = (ytoss - y[8]) / tauytos;
    double I_tos = GtoSlow * y[7] * y[8] * (y[38] - ek) * I_to_factor;
    
    double tauxtof = 8.5 * exp(-pow((y[38] + 45.0) / 50.0, 2)) + 0.5;
    double tauytof = 85.0 * exp(-(pow(y[38] + 40.0, 2) / 220.0)) + 7.0;
    result.ydot[9] = (xtoss - y[9]) / tauxtof;
    result.ydot[10] = (ytoss - y[10]) / tauytof;
    double I_tof = GtoFast * y[9] * y[10] * (y[38] - ek) * I_to_factor;
    double I_to = I_tos + I_tof;
    
    // I_ki: Time-Independent K Current
    double aki = 1.02 / (1.0 + exp(0.2385 * (y[38] - ek - 59.215)));
    double bki = (0.49124 * exp(0.08032 * (y[38] + 5.476 - ek)) + 
                  exp(0.06175 * (y[38] - ek - 594.31))) / 
                 (1.0 + exp(-0.5143 * (y[38] - ek + 4.753)));
    double kiss = aki / (aki + bki);
    double I_ki = 1.0 * 0.35 * sqrt(Ko / 5.4) * kiss * (y[38] - ek) * I_Ki_factor;
    
    // I_ClCa: Ca-activated Cl Current, I_Clbk: background Cl Current
    double I_ClCa_junc = Fjunc * GClCa / (1.0 + KdClCa / y[35]) * (y[38] - ecl);
    double I_ClCa_sl = Fsl * GClCa / (1.0 + KdClCa / y[36]) * (y[38] - ecl);
    double I_ClCa = I_ClCa_junc + I_ClCa_sl;
    double I_Clbk = GClB * (y[38] - ecl);
    
    // I_Ca: L-type Calcium Current
    double dss = 1.0 / (1.0 + exp(-(y[38] + 5.0) / 6.0));
    double taud = dss * (1.0 - exp(-(y[38] + 5.0) / 6.0)) / (0.035 * (y[38] + 5.0));
    double fss = 1.0 / (1.0 + exp((y[38] + 35.0) / 9.0)) + 0.6 / (1.0 + exp((50.0 - y[38]) / 20.0));
    double tauf = 1.0 / (0.0197 * exp(-pow(0.0337 * (y[38] + 14.5), 2)) + 0.02);
    result.ydot[3] = (dss - y[3]) / taud;
    result.ydot[4] = (fss - y[4]) / tauf;
    result.ydot[5] = 1.7 * y[35] * (1.0 - y[5]) - 11.9e-3 * y[5];
    result.ydot[6] = 1.7 * y[36] * (1.0 - y[6]) - 11.9e-3 * y[6];
    
    double fcaCaMSL = 0.0;
    double fcaCaj = 0.0;
    double ibarca_j = pCa * 4.0 * (y[38] * Frdy * FoRT) * 
                      (0.341 * y[35] * exp(2.0 * y[38] * FoRT) - 0.341 * Cao) / 
                      (exp(2.0 * y[38] * FoRT) - 1.0);
    double ibarca_sl = pCa * 4.0 * (y[38] * Frdy * FoRT) * 
                       (0.341 * y[36] * exp(2.0 * y[38] * FoRT) - 0.341 * Cao) / 
                       (exp(2.0 * y[38] * FoRT) - 1.0);
    double ibark = pK * (y[38] * Frdy * FoRT) * 
                   (0.75 * y[34] * exp(y[38] * FoRT) - 0.75 * Ko) / 
                   (exp(y[38] * FoRT) - 1.0);
    double ibarna_j = pNa * (y[38] * Frdy * FoRT) * 
                      (0.75 * y[31] * exp(y[38] * FoRT) - 0.75 * Nao) / 
                      (exp(y[38] * FoRT) - 1.0);
    double ibarna_sl = pNa * (y[38] * Frdy * FoRT) * 
                       (0.75 * y[32] * exp(y[38] * FoRT) - 0.75 * Nao) / 
                       (exp(y[38] * FoRT) - 1.0);
    
    double I_Ca_junc = (Fjunc_CaL * ibarca_j * y[3] * y[4] * ((1.0 - y[5]) + fcaCaj) * 
                        pow(Q10CaL, Qpow)) * 0.45 * 1.0;
    double I_Ca_sl = (Fsl_CaL * ibarca_sl * y[3] * y[4] * ((1.0 - y[6]) + fcaCaMSL) * 
                      pow(Q10CaL, Qpow)) * 0.45 * 1.0;
    double I_Ca = I_Ca_junc + I_Ca_sl;
    double I_CaK = (ibark * y[3] * y[4] * (Fjunc_CaL * (fcaCaj + (1.0 - y[5])) + 
                    Fsl_CaL * (fcaCaMSL + (1.0 - y[6]))) * pow(Q10CaL, Qpow)) * 0.45 * 1.0;
    double I_CaNa_junc = (Fjunc_CaL * ibarna_j * y[3] * y[4] * ((1.0 - y[5]) + fcaCaj) * 
                          pow(Q10CaL, Qpow)) * 0.45 * 1.0;
    double I_CaNa_sl = (Fsl_CaL * ibarna_sl * y[3] * y[4] * ((1.0 - y[6]) + fcaCaMSL) * 
                        pow(Q10CaL, Qpow)) * 0.45 * 1.0;
    double I_CaNa = I_CaNa_junc + I_CaNa_sl;
    double I_Catot = I_Ca + I_CaK + I_CaNa;
    
    // I_ncx: Na/Ca Exchanger flux
    double Ka_junc = 1.0 / (1.0 + pow(Kdact / y[35], 2));
    double Ka_sl = 1.0 / (1.0 + pow(Kdact / y[36], 2));
    double s1_junc = exp(nu * y[38] * FoRT) * pow(y[31], 3) * Cao;
    double s1_sl = exp(nu * y[38] * FoRT) * pow(y[32], 3) * Cao;
    double s2_junc = exp((nu - 1.0) * y[38] * FoRT) * pow(Nao, 3) * y[35];
    double s3_junc = KmCai * pow(Nao, 3) * (1.0 + pow(y[31] / KmNai, 3)) + 
                     pow(KmNao, 3) * y[35] * (1.0 + y[35] / KmCai) + 
                     KmCao * pow(y[31], 3) + pow(y[31], 3) * Cao + pow(Nao, 3) * y[35];
    double s2_sl = exp((nu - 1.0) * y[38] * FoRT) * pow(Nao, 3) * y[36];
    double s3_sl = KmCai * pow(Nao, 3) * (1.0 + pow(y[32] / KmNai, 3)) + 
                   pow(KmNao, 3) * y[36] * (1.0 + y[36] / KmCai) + 
                   KmCao * pow(y[32], 3) + pow(y[32], 3) * Cao + pow(Nao, 3) * y[36];
    
    double I_ncx_junc = I_NCX_factor * Fjunc * IbarNCX * pow(Q10NCX, Qpow) * Ka_junc * 
                        (s1_junc - s2_junc) / s3_junc / (1.0 + ksat * exp((nu - 1.0) * y[38] * FoRT));
    double I_ncx_sl = I_NCX_factor * Fsl * IbarNCX * pow(Q10NCX, Qpow) * Ka_sl * 
                      (s1_sl - s2_sl) / s3_sl / (1.0 + ksat * exp((nu - 1.0) * y[38] * FoRT));
    double I_ncx = I_ncx_junc + I_ncx_sl;
    
    // I_pca: Sarcolemmal Ca Pump Current
    double I_pca_junc = Fjunc * pow(Q10SLCaP, Qpow) * IbarSLCaP * pow(y[35], 1.6) / 
                        (pow(KmPCa, 1.6) + pow(y[35], 1.6));
    double I_pca_sl = Fsl * pow(Q10SLCaP, Qpow) * IbarSLCaP * pow(y[36], 1.6) / 
                      (pow(KmPCa, 1.6) + pow(y[36], 1.6));
    double I_pca = I_pca_junc + I_pca_sl;
    
    // I_cabk: Ca Background Current
    double I_cabk_junc = Fjunc * GCaB * (y[38] - eca_junc) * I_Cab_factor;
    double I_cabk_sl = Fsl * GCaB * (y[38] - eca_sl) * I_Cab_factor;
    double I_cabk = I_cabk_junc + I_cabk_sl;
    
    // SR fluxes: Calcium Release, SR Ca pump, SR Ca leak
    double MaxSR = 15.0;
    double MinSR = 1.0;
    double kCaSR = MaxSR - (MaxSR - MinSR) / (1.0 + pow(ec50SR / y[30], 2.5));
    double koSRCa = koCa / kCaSR;
    double kiSRCa = kiCa * kCaSR;
    double RI = 1.0 - y[13] - y[14] - y[15];
    result.ydot[13] = (kim * RI - kiSRCa * y[35] * y[13]) - (koSRCa * pow(y[35], 2) * y[13] - kom * y[14]);
    result.ydot[14] = (koSRCa * pow(y[35], 2) * y[13] - kom * y[14]) - (kiSRCa * y[35] * y[14] - kim * y[15]);
    result.ydot[15] = (kiSRCa * y[35] * y[14] - kim * y[15]) - (kom * y[15] - koSRCa * pow(y[35], 2) * RI);
    double J_SRCarel = ks * y[14] * (y[30] - y[35]);
    
    double J_serca = (1.0 * pow(Q10SRCaP, Qpow) * Vmax_SRCaP * 
                      (pow(y[37] / Kmf, hillSRCaP) - pow(y[30] / Kmr, hillSRCaP)) /
                      (1.0 + pow(y[37] / Kmf, hillSRCaP) + pow(y[30] / Kmr, hillSRCaP))) * J_serca_factor;
    double J_SRleak = (5.348e-6 * (y[30] - y[35])) * I_leak_factor;
    
    // Sodium and Calcium Buffering
    result.ydot[16] = kon_na * y[31] * (Bmax_Naj - y[16]) - koff_na * y[16];
    result.ydot[17] = kon_na * y[32] * (Bmax_Nasl - y[17]) - koff_na * y[17];
    
    // Cytosolic Ca Buffers
    result.ydot[18] = kon_tncl * y[37] * (Bmax_TnClow - y[18]) - koff_tncl * y[18];
    result.ydot[19] = kon_tnchca * y[37] * (Bmax_TnChigh - y[19] - y[20]) - koff_tnchca * y[19];
    result.ydot[20] = kon_tnchmg * Mgi * (Bmax_TnChigh - y[19] - y[20]) - koff_tnchmg * y[20];
    result.ydot[21] = kon_cam * y[37] * (Bmax_CaM - y[21]) - koff_cam * y[21];
    result.ydot[22] = kon_myoca * y[37] * (Bmax_myosin - y[22] - y[23]) - koff_myoca * y[22];
    result.ydot[23] = kon_myomg * Mgi * (Bmax_myosin - y[22] - y[23]) - koff_myomg * y[23];
    result.ydot[24] = kon_sr * y[37] * (Bmax_SR - y[24]) - koff_sr * y[24];
    
    double J_CaB_cytosol = result.ydot[18] + result.ydot[19] + result.ydot[20] + 
                           result.ydot[21] + result.ydot[22] + result.ydot[23] + result.ydot[24];
    
    // Junctional and SL Ca Buffers
    result.ydot[25] = kon_sll * y[35] * (Bmax_SLlowj - y[25]) - koff_sll * y[25];
    result.ydot[26] = kon_sll * y[36] * (Bmax_SLlowsl - y[26]) - koff_sll * y[26];
    result.ydot[27] = kon_slh * y[35] * (Bmax_SLhighj - y[27]) - koff_slh * y[27];
    result.ydot[28] = kon_slh * y[36] * (Bmax_SLhighsl - y[28]) - koff_slh * y[28];
    
    double J_CaB_junction = result.ydot[25] + result.ydot[27];
    double J_CaB_sl = result.ydot[26] + result.ydot[28];
    
    // Ion concentrations
    // SR Ca Concentrations
    result.ydot[29] = kon_csqn * y[30] * (Bmax_Csqn - y[29]) - koff_csqn * y[29];
    result.ydot[30] = J_serca - (J_SRleak * Vmyo / Vsr + J_SRCarel) - result.ydot[29];
    
    // Sodium Concentrations
    double I_Na_tot_junc = I_Na_junc + I_nabk_junc + 3.0 * I_ncx_junc + 3.0 * I_nak_junc + I_CaNa_junc;
    double I_Na_tot_sl = I_Na_sl + I_nabk_sl + 3.0 * I_ncx_sl + 3.0 * I_nak_sl + I_CaNa_sl;
    
    result.ydot[31] = -I_Na_tot_junc * Cmem / (Vjunc * Frdy) + J_na_juncsl / Vjunc * (y[32] - y[31]) - result.ydot[16];
    result.ydot[32] = -I_Na_tot_sl * Cmem / (Vsl * Frdy) + J_na_juncsl / Vsl * (y[31] - y[32]) +
                      J_na_slmyo / Vsl * (y[33] - y[32]) - result.ydot[17];
    result.ydot[33] = J_na_slmyo / Vmyo * (y[32] - y[33]);
    
    // Potassium Concentration
    double I_K_tot = I_to + I_kr + I_ks + I_ki - 2.0 * I_nak + I_CaK + I_kp;
    result.ydot[34] = 0.0;  // K concentration held constant
    
    // Calcium Concentrations
    double I_Ca_tot_junc = I_Ca_junc + I_cabk_junc + I_pca_junc - 2.0 * I_ncx_junc + I_Na_L_junc;
    double I_Ca_tot_sl = I_Ca_sl + I_cabk_sl + I_pca_sl - 2.0 * I_ncx_sl + I_Na_L_sl;
    
    result.ydot[35] = -I_Ca_tot_junc * Cmem / (Vjunc * 2.0 * Frdy) + J_ca_juncsl / Vjunc * (y[36] - y[35]) -
                      J_CaB_junction + (J_SRCarel) * Vsr / Vjunc + J_SRleak * Vmyo / Vjunc;
    result.ydot[36] = -I_Ca_tot_sl * Cmem / (Vsl * 2.0 * Frdy) + J_ca_juncsl / Vsl * (y[35] - y[36]) +
                      J_ca_slmyo / Vsl * (y[37] - y[36]) - J_CaB_sl;
    result.ydot[37] = -J_serca * Vsr / Vmyo - J_CaB_cytosol + J_ca_slmyo / Vmyo * (y[36] - y[37]);
    
    // Membrane Potential (index 38 - but this is passed as Vm parameter)
    double I_Na_tot = I_Na_tot_junc + I_Na_tot_sl;
    double I_Cl_tot = I_ClCa + I_Clbk;
    double I_Ca_tot = I_Ca_tot_junc + I_Ca_tot_sl;
    result.I_ion = I_Na_tot + I_Cl_tot + I_Ca_tot + I_K_tot;
    // result.ydot[38] = -(result.I_tot - I_app); // cell model 
    result.ydot[38] = 0.0; //tissue model
    
    // States 39-40 are unused in this configuration
    result.ydot[39] = 0.0;
    result.ydot[40] = 0.0;
    
    // States 43-57 are for Markov IKs model (not used here)
    for (int i = 43; i < 58; i++) {
        result.ydot[i] = 0.0;
    }
    
    return result;
}
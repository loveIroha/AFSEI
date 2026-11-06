// Generated from unchanged gpb_device.h via generate_cached_rhs.py + build_compact.py.

#pragma once
#include <cmath>

struct GPBVoltageCache {
 double mss;
 double taum;
 double tauh;
 double hss;
 double tauj;
 double jss;
 double a_na_L;
 double b_na_L;
 double h_na_L;
 double fnak;
 double xrss;
 double tauxr;
 double rkr;
 double xsss;
 double tauxs;
 double kp_kp;
 double xtoss;
 double ytoss;
 double tauxtos;
 double tauytos;
 double tauxtof;
 double tauytof;
 double I_Clbk;
 double dss;
 double taud;
 double fss;
 double tauf;
 double math_0;
 double math_1;
 double math_2;
 double math_3;
};
__device__ inline GPBVoltageCache make_gpb_voltage_cache(double Vm) {
 double epi=1;
 double heartfailureConditions=2;
 double R=8314.0;
 double Frdy=96485.0;
 double Temp=310.0;
 double FoRT=Frdy / R / Temp;
 double Cmem=1.3810e-10;
 double Qpow=(Temp - 310.0) / 10.0;
 double cellLength=100.0;
 double cellRadius=10.25;
 double Vcell=M_PI * cellRadius * cellRadius * cellLength * 1e-15;
 double Vmyo=0.65 * Vcell;
 double Vsr=0.035 * Vcell;
 double Vsl=0.02 * Vcell;
 double Vjunc=0.0539 * 0.01 * Vcell;
 double J_ca_juncsl=1.0 / 1.2134e12 * 1.5;
 double J_ca_slmyo=1.0 / 2.68510e11 * 2.0;
 double J_na_juncsl=1.0 / (1.6382e12 / 3.0 * 100.0);
 double J_na_slmyo=1.0 / (1.8308e10 / 3.0 * 100.0);
 double Fjunc=0.11;
 double Fsl=1.0 - Fjunc;
 double Fjunc_CaL=0.9;
 double Fsl_CaL=1.0 - Fjunc_CaL;
 double Cli=15.0;
 double Clo=150.0;
 double Ko=5.4;
 double Nao=140.0;
 double Cao=1.8;
 double Mgi=1.0;
 double ecl=(1.0 / FoRT) * log(Cli / Clo);
 double GNa=23.0;
 double GNaB=0.597e-3;
 double IbarNaK=1.0 * 1.8;
 double KmNaip=11.0;
 double KmKo=1.5;
 double pNaK=0.01833;
 double gkp=2.0 * 0.001;
 double GClCa=0.5 * 0.109625;
 double GClB=1.0 * 9e-3;
 double KdClCa=100e-3;
 double pNa=0.50 * 1.5e-8;
 double pCa=0.50 * 5.4e-4;
 double pK=0.50 * 2.7e-7;
 double Q10CaL=1.8;
 double IbarNCX=1.0 * 4.5;
 double KmCai=3.59e-3;
 double KmCao=1.3;
 double KmNai=12.29;
 double KmNao=87.5;
 double ksat=0.32;
 double nu=0.27;
 double Kdact=0.150e-3;
 double Q10NCX=1.57;
 double IbarSLCaP=0.0673;
 double KmPCa=0.5e-3;
 double GCaB=5.513e-4;
 double Q10SLCaP=2.35;
 double Q10SRCaP=2.6;
 double Vmax_SRCaP=1.0 * 5.3114e-3;
 double Kmf=0.246e-3;
 double Kmr=1.7;
 double hillSRCaP=1.787;
 double ks=25.0;
 double koCa=10.0;
 double kom=0.06;
 double kiCa=0.5;
 double kim=0.005;
 double Bmax_Naj=7.561;
 double Bmax_Nasl=1.65;
 double koff_na=1e-3;
 double kon_na=0.1e-3;
 double Bmax_TnClow=70e-3;
 double koff_tncl=19.6e-3;
 double kon_tncl=32.7;
 double Bmax_TnChigh=140e-3;
 double koff_tnchca=0.032e-3;
 double kon_tnchca=2.37;
 double koff_tnchmg=3.33e-3;
 double kon_tnchmg=3e-3;
 double Bmax_CaM=24e-3;
 double koff_cam=238e-3;
 double kon_cam=34.0;
 double Bmax_myosin=140e-3;
 double koff_myoca=0.46e-3;
 double kon_myoca=13.8;
 double koff_myomg=0.057e-3;
 double kon_myomg=0.0157;
 double Bmax_SR=19.0 * 0.9e-3;
 double koff_sr=60e-3;
 double kon_sr=100.0;
 double Bmax_SLlowsl=37.4e-3 * Vmyo / Vsl;
 double Bmax_SLlowj=4.6e-3 * Vmyo / Vjunc * 0.1;
 double koff_sll=1300e-3;
 double kon_sll=100.0;
 double Bmax_SLhighsl=13.4e-3 * Vmyo / Vsl;
 double Bmax_SLhighj=1.65e-3 * Vmyo / Vjunc * 0.1;
 double koff_slh=30e-3;
 double kon_slh=100.0;
 double Bmax_Csqn=140e-3 * Vmyo / Vsr;
 double koff_csqn=65.0;
 double kon_csqn=100.0;
 double mss=1.0 / pow((1.0 + exp(-(56.86 + Vm) / 9.03)), 2);
 double taum=0.1292 * exp(-pow((Vm + 45.79) / 15.54, 2)) + 
                  0.06487 * exp(-pow((Vm - 4.823) / 51.12, 2));
 double ah=(Vm >= -40.0) ? 0.0 : (0.057 * exp(-(Vm + 80.0) / 6.8));
 double bh=(Vm >= -40.0) ? (0.77 / (0.13 * (1.0 + exp(-(Vm + 10.66) / 11.1)))) :
                ((2.7 * exp(0.079 * Vm) + 3.1e5 * exp(0.3485 * Vm)));
 double tauh=1.0 / (ah + bh);
 double hss=1.0 / pow((1.0 + exp((Vm + 71.55) / 7.43)), 2);
 double aj=(Vm >= -40.0) ? 0.0 :
                (((-2.5428e4 * exp(0.2444 * Vm) - 6.948e-6 * exp(-0.04391 * Vm)) * 
                  (Vm + 37.78)) / (1.0 + exp(0.311 * (Vm + 79.23))));
 double bj=(Vm >= -40.0) ? 
                ((0.6 * exp(0.057 * Vm)) / (1.0 + exp(-0.1 * (Vm + 32.0)))) :
                ((0.02424 * exp(-0.01052 * Vm)) / (1.0 + exp(-0.1378 * (Vm + 40.14))));
 double tauj=1.0 / (aj + bj);
 double jss=1.0 / pow((1.0 + exp((Vm + 71.55) / 7.43)), 2);
 double gNaL=0.008;
 double a_na_L=0.32 * (Vm + 47.13) / (1.0 - exp(-0.1 * (Vm + 47.13)));
 double b_na_L=0.08 * exp(-Vm / 11.0);
 double h_na_L=1.0 / (1.0 + exp((Vm + 91.0) / 6.1));
 double sigma=(exp(Nao / 67.3) - 1.0) / 7.0;
 double fnak=1.0 / (1.0 + 0.1245 * exp(-0.1 * Vm * FoRT) + 0.0365 * sigma * exp(-Vm * FoRT));
 double gkr=1.0 * 0.035 * sqrt(Ko / 5.4);
 double xrss=1.0 / (1.0 + exp(-(Vm + 10.0) / 5.0));
 double tauxr=550.0 / (1.0 + exp((-22.0 - Vm) / 9.0)) * 6.0 / (1.0 + exp((Vm - (-11.0)) / 9.0)) + 
                   230.0 / (1.0 + exp((Vm - (-40.0)) / 20.0));
 double rkr=1.0 / (1.0 + exp((Vm + 74.0) / 24.0));
 double gks_junc=1.0 * 0.0035;
 double gks_sl=1.0 * 0.0035;
 double xsss=1.0 / (1.0 + exp(-(Vm + 3.8) / 14.25));
 double tauxs=990.1 / (1.0 + exp(-(Vm + 2.436) / 14.12));
 double kp_kp=1.0 / (1.0 + exp(7.488 - Vm / 5.98));
 double xtoss=1.0 / (1.0 + exp(-(Vm - 19.0) / 13.0));
 double ytoss=1.0 / (1.0 + exp((Vm + 19.5) / 5.0));
 double tauxtos=9.0 / (1.0 + exp((Vm + 3.0) / 15.0)) + 0.5;
 double tauytos=800.0 / (1.0 + exp((Vm + 60.0) / 10.0)) + 30.0;
 double tauxtof=8.5 * exp(-pow((Vm + 45.0) / 50.0, 2)) + 0.5;
 double tauytof=85.0 * exp(-(pow(Vm + 40.0, 2) / 220.0)) + 7.0;
 double I_Clbk=GClB * (Vm - ecl);
 double dss=1.0 / (1.0 + exp(-(Vm + 5.0) / 6.0));
 double taud=dss * (1.0 - exp(-(Vm + 5.0) / 6.0)) / (0.035 * (Vm + 5.0));
 double fss=1.0 / (1.0 + exp((Vm + 35.0) / 9.0)) + 0.6 / (1.0 + exp((50.0 - Vm) / 20.0));
 double tauf=1.0 / (0.0197 * exp(-pow(0.0337 * (Vm + 14.5), 2)) + 0.02);
 double fcaCaMSL=0.0;
 double fcaCaj=0.0;
 double MaxSR=15.0;
 double MinSR=1.0;
GPBVoltageCache cache;
 cache.mss=mss;
 cache.taum=taum;
 cache.tauh=tauh;
 cache.hss=hss;
 cache.tauj=tauj;
 cache.jss=jss;
 cache.a_na_L=a_na_L;
 cache.b_na_L=b_na_L;
 cache.h_na_L=h_na_L;
 cache.fnak=fnak;
 cache.xrss=xrss;
 cache.tauxr=tauxr;
 cache.rkr=rkr;
 cache.xsss=xsss;
 cache.tauxs=tauxs;
 cache.kp_kp=kp_kp;
 cache.xtoss=xtoss;
 cache.ytoss=ytoss;
 cache.tauxtos=tauxtos;
 cache.tauytos=tauytos;
 cache.tauxtof=tauxtof;
 cache.tauytof=tauytof;
 cache.I_Clbk=I_Clbk;
 cache.dss=dss;
 cache.taud=taud;
 cache.fss=fss;
 cache.tauf=tauf;
 cache.math_0=exp(2.0 * Vm * FoRT);
 cache.math_1=exp(Vm * FoRT);
 cache.math_2=exp(nu * Vm * FoRT);
 cache.math_3=exp((nu - 1.0) * Vm * FoRT);
return cache;
}
__device__ inline double gpb_nodal_rhs(double t, const double* __restrict__ y, double Vm, double I_app, double* __restrict__ ydot,const GPBVoltageCache& cache,int64_t stride,double* relative_error=nullptr) {

    double I_ion;
    
    
    
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
    
    
    const double R = 8314.0;
    const double Frdy = 96485.0;
    const double Temp = 310.0;
    const double FoRT = Frdy / R / Temp;
    const double Cmem = 1.3810e-10;
    const double Qpow = (Temp - 310.0) / 10.0;
    
    
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
    
    
    const double Fjunc = 0.11;
    const double Fsl = 1.0 - Fjunc;
    const double Fjunc_CaL = 0.9;
    const double Fsl_CaL = 1.0 - Fjunc_CaL;
    
    
    const double Cli = 15.0;
    const double Clo = 150.0;
    const double Ko = 5.4;
    const double Nao = 140.0;
    const double Cao = 1.8;
    const double Mgi = 1.0;
    
    
    double ena_junc = (1.0 / FoRT) * log(Nao / y[31*stride]);
    double ena_sl = (1.0 / FoRT) * log(Nao / y[32*stride]);
    double ek = (1.0 / FoRT) * log(Ko / y[34*stride]);
    double eca_junc = (1.0 / FoRT / 2.0) * log(Cao / y[35*stride]);
    double eca_sl = (1.0 / FoRT / 2.0) * log(Cao / y[36*stride]);
    double ecl = (1.0 / FoRT) * log(Cli / Clo);
    
    
    const double GNa = 23.0;
    const double GNaB = 0.597e-3;
    const double IbarNaK = 1.0 * 1.8;
    const double KmNaip = 11.0;
    const double KmKo = 1.5;
    
    
    const double pNaK = 0.01833;
    const double gkp = 2.0 * 0.001;
    
    
    const double GClCa = 0.5 * 0.109625;
    const double GClB = 1.0 * 9e-3;
    const double KdClCa = 100e-3;
    
    
    const double pNa = 0.50 * 1.5e-8;
    const double pCa = 0.50 * 5.4e-4;
    const double pK = 0.50 * 2.7e-7;
    const double Q10CaL = 1.8;
    
    
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
    
    

    
    
    double mss=cache.mss;
    double taum=cache.taum;
    
    double ah = (Vm >= -40.0) ? 0.0 : (0.057 * exp(-(Vm + 80.0) / 6.8));
    double bh = (Vm >= -40.0) ? (0.77 / (0.13 * (1.0 + exp(-(Vm + 10.66) / 11.1)))) :
                ((2.7 * exp(0.079 * Vm) + 3.1e5 * exp(0.3485 * Vm)));
    double tauh=cache.tauh;
    double hss=cache.hss;
    
    double aj = (Vm >= -40.0) ? 0.0 :
                (((-2.5428e4 * exp(0.2444 * Vm) - 6.948e-6 * exp(-0.04391 * Vm)) * 
                  (Vm + 37.78)) / (1.0 + exp(0.311 * (Vm + 79.23))));
    double bj = (Vm >= -40.0) ? 
                ((0.6 * exp(0.057 * Vm)) / (1.0 + exp(-0.1 * (Vm + 32.0)))) :
                ((0.02424 * exp(-0.01052 * Vm)) / (1.0 + exp(-0.1378 * (Vm + 40.14))));
    double tauj=cache.tauj;
    double jss=cache.jss;
    
    {double next=((mss - y[0*stride]) / taum);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[0*stride])/fmax(fabs(next),1.));ydot[0*stride]=next;}
    {double next=((hss - y[1*stride]) / tauh);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[1*stride])/fmax(fabs(next),1.));ydot[1*stride]=next;}
    {double next=((jss - y[2*stride]) / tauj);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[2*stride])/fmax(fabs(next),1.));ydot[2*stride]=next;}
    
    double I_Na_junc = Fjunc * GNa * pow(y[0*stride], 3) * y[1*stride] * y[2*stride] * (Vm - ena_junc);
    double I_Na_sl = Fsl * GNa * pow(y[0*stride], 3) * y[1*stride] * y[2*stride] * (Vm - ena_sl);
    double I_Na = I_Na_junc + I_Na_sl;
    
    
    double gNaL = 0.008;
    double tauhL = 233.0 * tau_hL_factor;
    double a_na_L=cache.a_na_L;
    double b_na_L=cache.b_na_L;
    double h_na_L=cache.h_na_L;
    {double next=(a_na_L * (1.0 - y[40*stride]) - b_na_L * y[40*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[40*stride])/fmax(fabs(next),1.));ydot[40*stride]=next;}
    {double next=((h_na_L - y[41*stride]) / tauhL);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[41*stride])/fmax(fabs(next),1.));ydot[41*stride]=next;}
    double I_Na_L_junc = I_NaL_factor * Fjunc * gNaL * pow(y[40*stride], 3) * y[41*stride] * (Vm - ena_junc);
    double I_Na_L_sl = I_NaL_factor * Fsl * gNaL * pow(y[40*stride], 3) * y[41*stride] * (Vm - ena_sl);
    
    
    double I_nabk_junc = Fjunc * GNaB * (Vm - ena_junc) * I_Nab_factor;
    double I_nabk_sl = Fsl * GNaB * (Vm - ena_sl) * I_Nab_factor;
    double I_nabk = I_nabk_junc + I_nabk_sl;
    
    
    double sigma = (exp(Nao / 67.3) - 1.0) / 7.0;
    double fnak=cache.fnak;
    double I_nak_junc = Fjunc * IbarNaK * fnak * Ko / (1.0 + pow(KmNaip / y[31*stride], 4)) / (Ko + KmKo) * I_NaK_factor;
    double I_nak_sl = Fsl * IbarNaK * fnak * Ko / (1.0 + pow(KmNaip / y[32*stride], 4)) / (Ko + KmKo) * I_NaK_factor;
    double I_nak = I_nak_junc + I_nak_sl;
    
    
    double gkr = 1.0 * 0.035 * sqrt(Ko / 5.4);
    double xrss=cache.xrss;
    double tauxr=cache.tauxr;
    {double next=((xrss - y[11*stride]) / tauxr);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[11*stride])/fmax(fabs(next),1.));ydot[11*stride]=next;}
    double rkr=cache.rkr;
    double I_kr = gkr * y[11*stride] * rkr * (Vm - ek);
    
    
    double eks = (1.0 / FoRT) * log((Ko + pNaK * Nao) / (y[34*stride] + pNaK * y[33*stride]));
    double gks_junc = 1.0 * 0.0035;
    double gks_sl = 1.0 * 0.0035;
    double xsss=cache.xsss;
    double tauxs=cache.tauxs;
    {double next=((xsss - y[12*stride]) / tauxs);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[12*stride])/fmax(fabs(next),1.));ydot[12*stride]=next;}
    double I_ks_junc = Fjunc * gks_junc * pow(y[12*stride], 2) * (Vm - eks);
    double I_ks_sl = Fsl * gks_sl * pow(y[12*stride], 2) * (Vm - eks);
    double I_ks = I_ks_junc + I_ks_sl;
    
    
    double kp_kp=cache.kp_kp;
    double I_kp_junc = Fjunc * gkp * kp_kp * (Vm - ek);
    double I_kp_sl = Fsl * gkp * kp_kp * (Vm - ek);
    double I_kp = I_kp_junc + I_kp_sl;
    
    
    double GtoSlow, GtoFast;
    if (epi == 1) {
        GtoSlow = 1.0 * 0.13 * 0.12;
        GtoFast = 1.0 * 0.13 * 0.88;
    } else {
        GtoSlow = 0.13 * 0.3 * 0.964;
        GtoFast = 0.13 * 0.3 * 0.036;
    }
    
    double xtoss=cache.xtoss;
    double ytoss=cache.ytoss;
    double tauxtos=cache.tauxtos;
    double tauytos=cache.tauytos;
    {double next=((xtoss - y[7*stride]) / tauxtos);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[7*stride])/fmax(fabs(next),1.));ydot[7*stride]=next;}
    {double next=((ytoss - y[8*stride]) / tauytos);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[8*stride])/fmax(fabs(next),1.));ydot[8*stride]=next;}
    double I_tos = GtoSlow * y[7*stride] * y[8*stride] * (Vm - ek) * I_to_factor;
    
    double tauxtof=cache.tauxtof;
    double tauytof=cache.tauytof;
    {double next=((xtoss - y[9*stride]) / tauxtof);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[9*stride])/fmax(fabs(next),1.));ydot[9*stride]=next;}
    {double next=((ytoss - y[10*stride]) / tauytof);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[10*stride])/fmax(fabs(next),1.));ydot[10*stride]=next;}
    double I_tof = GtoFast * y[9*stride] * y[10*stride] * (Vm - ek) * I_to_factor;
    double I_to = I_tos + I_tof;
    
    
    double aki = 1.02 / (1.0 + exp(0.2385 * (Vm - ek - 59.215)));
    double bki = (0.49124 * exp(0.08032 * (Vm + 5.476 - ek)) + 
                  exp(0.06175 * (Vm - ek - 594.31))) / 
                 (1.0 + exp(-0.5143 * (Vm - ek + 4.753)));
    double kiss = aki / (aki + bki);
    double I_ki = 1.0 * 0.35 * sqrt(Ko / 5.4) * kiss * (Vm - ek) * I_Ki_factor;
    
    
    double I_ClCa_junc = Fjunc * GClCa / (1.0 + KdClCa / y[35*stride]) * (Vm - ecl);
    double I_ClCa_sl = Fsl * GClCa / (1.0 + KdClCa / y[36*stride]) * (Vm - ecl);
    double I_ClCa = I_ClCa_junc + I_ClCa_sl;
    double I_Clbk=cache.I_Clbk;
    
    
    double dss=cache.dss;
    double taud=cache.taud;
    double fss=cache.fss;
    double tauf=cache.tauf;
    {double next=((dss - y[3*stride]) / taud);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[3*stride])/fmax(fabs(next),1.));ydot[3*stride]=next;}
    {double next=((fss - y[4*stride]) / tauf);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[4*stride])/fmax(fabs(next),1.));ydot[4*stride]=next;}
    {double next=(1.7 * y[35*stride] * (1.0 - y[5*stride]) - 11.9e-3 * y[5*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[5*stride])/fmax(fabs(next),1.));ydot[5*stride]=next;}
    {double next=(1.7 * y[36*stride] * (1.0 - y[6*stride]) - 11.9e-3 * y[6*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[6*stride])/fmax(fabs(next),1.));ydot[6*stride]=next;}
    
    double fcaCaMSL = 0.0;
    double fcaCaj = 0.0;
    double ibarca_j = pCa * 4.0 * (Vm * Frdy * FoRT) * 
                      (0.341 * y[35*stride] * cache.math_0 - 0.341 * Cao) / 
                      (cache.math_0 - 1.0);
    double ibarca_sl = pCa * 4.0 * (Vm * Frdy * FoRT) * 
                       (0.341 * y[36*stride] * cache.math_0 - 0.341 * Cao) / 
                       (cache.math_0 - 1.0);
    double ibark = pK * (Vm * Frdy * FoRT) * 
                   (0.75 * y[34*stride] * cache.math_1 - 0.75 * Ko) / 
                   (cache.math_1 - 1.0);
    double ibarna_j = pNa * (Vm * Frdy * FoRT) * 
                      (0.75 * y[31*stride] * cache.math_1 - 0.75 * Nao) / 
                      (cache.math_1 - 1.0);
    double ibarna_sl = pNa * (Vm * Frdy * FoRT) * 
                       (0.75 * y[32*stride] * cache.math_1 - 0.75 * Nao) / 
                       (cache.math_1 - 1.0);
    
    double I_Ca_junc = (Fjunc_CaL * ibarca_j * y[3*stride] * y[4*stride] * ((1.0 - y[5*stride]) + fcaCaj) * 
                        pow(Q10CaL, Qpow)) * 0.45 * 1.0;
    double I_Ca_sl = (Fsl_CaL * ibarca_sl * y[3*stride] * y[4*stride] * ((1.0 - y[6*stride]) + fcaCaMSL) * 
                      pow(Q10CaL, Qpow)) * 0.45 * 1.0;
    double I_Ca = I_Ca_junc + I_Ca_sl;
    double I_CaK = (ibark * y[3*stride] * y[4*stride] * (Fjunc_CaL * (fcaCaj + (1.0 - y[5*stride])) + 
                    Fsl_CaL * (fcaCaMSL + (1.0 - y[6*stride]))) * pow(Q10CaL, Qpow)) * 0.45 * 1.0;
    double I_CaNa_junc = (Fjunc_CaL * ibarna_j * y[3*stride] * y[4*stride] * ((1.0 - y[5*stride]) + fcaCaj) * 
                          pow(Q10CaL, Qpow)) * 0.45 * 1.0;
    double I_CaNa_sl = (Fsl_CaL * ibarna_sl * y[3*stride] * y[4*stride] * ((1.0 - y[6*stride]) + fcaCaMSL) * 
                        pow(Q10CaL, Qpow)) * 0.45 * 1.0;
    double I_CaNa = I_CaNa_junc + I_CaNa_sl;
    double I_Catot = I_Ca + I_CaK + I_CaNa;
    
    
    double Ka_junc = 1.0 / (1.0 + pow(Kdact / y[35*stride], 2));
    double Ka_sl = 1.0 / (1.0 + pow(Kdact / y[36*stride], 2));
    double s1_junc = cache.math_2 * pow(y[31*stride], 3) * Cao;
    double s1_sl = cache.math_2 * pow(y[32*stride], 3) * Cao;
    double s2_junc = cache.math_3 * pow(Nao, 3) * y[35*stride];
    double s3_junc = KmCai * pow(Nao, 3) * (1.0 + pow(y[31*stride] / KmNai, 3)) + 
                     pow(KmNao, 3) * y[35*stride] * (1.0 + y[35*stride] / KmCai) + 
                     KmCao * pow(y[31*stride], 3) + pow(y[31*stride], 3) * Cao + pow(Nao, 3) * y[35*stride];
    double s2_sl = cache.math_3 * pow(Nao, 3) * y[36*stride];
    double s3_sl = KmCai * pow(Nao, 3) * (1.0 + pow(y[32*stride] / KmNai, 3)) + 
                   pow(KmNao, 3) * y[36*stride] * (1.0 + y[36*stride] / KmCai) + 
                   KmCao * pow(y[32*stride], 3) + pow(y[32*stride], 3) * Cao + pow(Nao, 3) * y[36*stride];
    
    double I_ncx_junc = I_NCX_factor * Fjunc * IbarNCX * pow(Q10NCX, Qpow) * Ka_junc * 
                        (s1_junc - s2_junc) / s3_junc / (1.0 + ksat * cache.math_3);
    double I_ncx_sl = I_NCX_factor * Fsl * IbarNCX * pow(Q10NCX, Qpow) * Ka_sl * 
                      (s1_sl - s2_sl) / s3_sl / (1.0 + ksat * cache.math_3);
    double I_ncx = I_ncx_junc + I_ncx_sl;
    
    
    double I_pca_junc = Fjunc * pow(Q10SLCaP, Qpow) * IbarSLCaP * pow(y[35*stride], 1.6) / 
                        (pow(KmPCa, 1.6) + pow(y[35*stride], 1.6));
    double I_pca_sl = Fsl * pow(Q10SLCaP, Qpow) * IbarSLCaP * pow(y[36*stride], 1.6) / 
                      (pow(KmPCa, 1.6) + pow(y[36*stride], 1.6));
    double I_pca = I_pca_junc + I_pca_sl;
    
    
    double I_cabk_junc = Fjunc * GCaB * (Vm - eca_junc) * I_Cab_factor;
    double I_cabk_sl = Fsl * GCaB * (Vm - eca_sl) * I_Cab_factor;
    double I_cabk = I_cabk_junc + I_cabk_sl;
    
    
    double MaxSR = 15.0;
    double MinSR = 1.0;
    double kCaSR = MaxSR - (MaxSR - MinSR) / (1.0 + pow(ec50SR / y[30*stride], 2.5));
    double koSRCa = koCa / kCaSR;
    double kiSRCa = kiCa * kCaSR;
    double RI = 1.0 - y[13*stride] - y[14*stride] - y[15*stride];
    {double next=((kim * RI - kiSRCa * y[35*stride] * y[13*stride]) - (koSRCa * pow(y[35*stride], 2) * y[13*stride] - kom * y[14*stride]));if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[13*stride])/fmax(fabs(next),1.));ydot[13*stride]=next;}
    {double next=((koSRCa * pow(y[35*stride], 2) * y[13*stride] - kom * y[14*stride]) - (kiSRCa * y[35*stride] * y[14*stride] - kim * y[15*stride]));if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[14*stride])/fmax(fabs(next),1.));ydot[14*stride]=next;}
    {double next=((kiSRCa * y[35*stride] * y[14*stride] - kim * y[15*stride]) - (kom * y[15*stride] - koSRCa * pow(y[35*stride], 2) * RI));if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[15*stride])/fmax(fabs(next),1.));ydot[15*stride]=next;}
    double J_SRCarel = ks * y[14*stride] * (y[30*stride] - y[35*stride]);
    
    double J_serca = (1.0 * pow(Q10SRCaP, Qpow) * Vmax_SRCaP * 
                      (pow(y[37*stride] / Kmf, hillSRCaP) - pow(y[30*stride] / Kmr, hillSRCaP)) /
                      (1.0 + pow(y[37*stride] / Kmf, hillSRCaP) + pow(y[30*stride] / Kmr, hillSRCaP))) * J_serca_factor;
    double J_SRleak = (5.348e-6 * (y[30*stride] - y[35*stride])) * I_leak_factor;
    
    
    {double next=(kon_na * y[31*stride] * (Bmax_Naj - y[16*stride]) - koff_na * y[16*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[16*stride])/fmax(fabs(next),1.));ydot[16*stride]=next;}
    {double next=(kon_na * y[32*stride] * (Bmax_Nasl - y[17*stride]) - koff_na * y[17*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[17*stride])/fmax(fabs(next),1.));ydot[17*stride]=next;}
    
    
    {double next=(kon_tncl * y[37*stride] * (Bmax_TnClow - y[18*stride]) - koff_tncl * y[18*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[18*stride])/fmax(fabs(next),1.));ydot[18*stride]=next;}
    {double next=(kon_tnchca * y[37*stride] * (Bmax_TnChigh - y[19*stride] - y[20*stride]) - koff_tnchca * y[19*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[19*stride])/fmax(fabs(next),1.));ydot[19*stride]=next;}
    {double next=(kon_tnchmg * Mgi * (Bmax_TnChigh - y[19*stride] - y[20*stride]) - koff_tnchmg * y[20*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[20*stride])/fmax(fabs(next),1.));ydot[20*stride]=next;}
    {double next=(kon_cam * y[37*stride] * (Bmax_CaM - y[21*stride]) - koff_cam * y[21*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[21*stride])/fmax(fabs(next),1.));ydot[21*stride]=next;}
    {double next=(kon_myoca * y[37*stride] * (Bmax_myosin - y[22*stride] - y[23*stride]) - koff_myoca * y[22*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[22*stride])/fmax(fabs(next),1.));ydot[22*stride]=next;}
    {double next=(kon_myomg * Mgi * (Bmax_myosin - y[22*stride] - y[23*stride]) - koff_myomg * y[23*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[23*stride])/fmax(fabs(next),1.));ydot[23*stride]=next;}
    {double next=(kon_sr * y[37*stride] * (Bmax_SR - y[24*stride]) - koff_sr * y[24*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[24*stride])/fmax(fabs(next),1.));ydot[24*stride]=next;}
    
    double J_CaB_cytosol = ydot[18*stride] + ydot[19*stride] + ydot[20*stride] + 
                           ydot[21*stride] + ydot[22*stride] + ydot[23*stride] + ydot[24*stride];
    
    
    {double next=(kon_sll * y[35*stride] * (Bmax_SLlowj - y[25*stride]) - koff_sll * y[25*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[25*stride])/fmax(fabs(next),1.));ydot[25*stride]=next;}
    {double next=(kon_sll * y[36*stride] * (Bmax_SLlowsl - y[26*stride]) - koff_sll * y[26*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[26*stride])/fmax(fabs(next),1.));ydot[26*stride]=next;}
    {double next=(kon_slh * y[35*stride] * (Bmax_SLhighj - y[27*stride]) - koff_slh * y[27*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[27*stride])/fmax(fabs(next),1.));ydot[27*stride]=next;}
    {double next=(kon_slh * y[36*stride] * (Bmax_SLhighsl - y[28*stride]) - koff_slh * y[28*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[28*stride])/fmax(fabs(next),1.));ydot[28*stride]=next;}
    
    double J_CaB_junction = ydot[25*stride] + ydot[27*stride];
    double J_CaB_sl = ydot[26*stride] + ydot[28*stride];
    
    
    
    {double next=(kon_csqn * y[30*stride] * (Bmax_Csqn - y[29*stride]) - koff_csqn * y[29*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[29*stride])/fmax(fabs(next),1.));ydot[29*stride]=next;}
    {double next=(J_serca - (J_SRleak * Vmyo / Vsr + J_SRCarel) - ydot[29*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[30*stride])/fmax(fabs(next),1.));ydot[30*stride]=next;}
    
    
    double I_Na_tot_junc = I_Na_junc + I_nabk_junc + 3.0 * I_ncx_junc + 3.0 * I_nak_junc + I_CaNa_junc;
    double I_Na_tot_sl = I_Na_sl + I_nabk_sl + 3.0 * I_ncx_sl + 3.0 * I_nak_sl + I_CaNa_sl;
    
    {double next=(-I_Na_tot_junc * Cmem / (Vjunc * Frdy) + J_na_juncsl / Vjunc * (y[32*stride] - y[31*stride]) - ydot[16*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[31*stride])/fmax(fabs(next),1.));ydot[31*stride]=next;}
    {double next=(-I_Na_tot_sl * Cmem / (Vsl * Frdy) + J_na_juncsl / Vsl * (y[31*stride] - y[32*stride]) +
                      J_na_slmyo / Vsl * (y[33*stride] - y[32*stride]) - ydot[17*stride]);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[32*stride])/fmax(fabs(next),1.));ydot[32*stride]=next;}
    {double next=(J_na_slmyo / Vmyo * (y[32*stride] - y[33*stride]));if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[33*stride])/fmax(fabs(next),1.));ydot[33*stride]=next;}
    
    
    double I_K_tot = I_to + I_kr + I_ks + I_ki - 2.0 * I_nak + I_CaK + I_kp;
    {double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[34*stride])/fmax(fabs(next),1.));ydot[34*stride]=next;}  
    
    
    double I_Ca_tot_junc = I_Ca_junc + I_cabk_junc + I_pca_junc - 2.0 * I_ncx_junc + I_Na_L_junc;
    double I_Ca_tot_sl = I_Ca_sl + I_cabk_sl + I_pca_sl - 2.0 * I_ncx_sl + I_Na_L_sl;
    
    {double next=(-I_Ca_tot_junc * Cmem / (Vjunc * 2.0 * Frdy) + J_ca_juncsl / Vjunc * (y[36*stride] - y[35*stride]) -
                      J_CaB_junction + (J_SRCarel) * Vsr / Vjunc + J_SRleak * Vmyo / Vjunc);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[35*stride])/fmax(fabs(next),1.));ydot[35*stride]=next;}
    {double next=(-I_Ca_tot_sl * Cmem / (Vsl * 2.0 * Frdy) + J_ca_juncsl / Vsl * (y[35*stride] - y[36*stride]) +
                      J_ca_slmyo / Vsl * (y[37*stride] - y[36*stride]) - J_CaB_sl);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[36*stride])/fmax(fabs(next),1.));ydot[36*stride]=next;}
    {double next=(-J_serca * Vsr / Vmyo - J_CaB_cytosol + J_ca_slmyo / Vmyo * (y[36*stride] - y[37*stride]));if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[37*stride])/fmax(fabs(next),1.));ydot[37*stride]=next;}
    
    
    double I_Na_tot = I_Na_tot_junc + I_Na_tot_sl;
    double I_Cl_tot = I_ClCa + I_Clbk;
    double I_Ca_tot = I_Ca_tot_junc + I_Ca_tot_sl;
    I_ion = I_Na_tot + I_Cl_tot + I_Ca_tot + I_K_tot;
    

    
    
    {double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[38*stride])/fmax(fabs(next),1.));ydot[38*stride]=next;}
    {double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[39*stride])/fmax(fabs(next),1.));ydot[39*stride]=next;}
    
    
    {double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[42*stride])/fmax(fabs(next),1.));ydot[42*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[43*stride])/fmax(fabs(next),1.));ydot[43*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[44*stride])/fmax(fabs(next),1.));ydot[44*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[45*stride])/fmax(fabs(next),1.));ydot[45*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[46*stride])/fmax(fabs(next),1.));ydot[46*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[47*stride])/fmax(fabs(next),1.));ydot[47*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[48*stride])/fmax(fabs(next),1.));ydot[48*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[49*stride])/fmax(fabs(next),1.));ydot[49*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[50*stride])/fmax(fabs(next),1.));ydot[50*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[51*stride])/fmax(fabs(next),1.));ydot[51*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[52*stride])/fmax(fabs(next),1.));ydot[52*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[53*stride])/fmax(fabs(next),1.));ydot[53*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[54*stride])/fmax(fabs(next),1.));ydot[54*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[55*stride])/fmax(fabs(next),1.));ydot[55*stride]=next;}
{double next=(0.0);if(relative_error) *relative_error=fmax(*relative_error,fabs(next-ydot[56*stride])/fmax(fabs(next),1.));ydot[56*stride]=next;}
    
    return I_ion;

}

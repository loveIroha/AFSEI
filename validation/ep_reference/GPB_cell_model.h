#pragma once
#include <vector>
#include <cmath>

/**
 * GPB Cell Model - Optimized for Tissue Simulation
 * 
 * In tissue simulation, this model:
 * 1. Takes Vm as INPUT from the monodomain PDE (not from state vector)
 * 2. Updates only cellular states (gates, concentrations, buffers)
 * 3. Returns I_ion as OUTPUT for the PDE right-hand side
 * 
 * Key difference from single-cell mode:
 * - Single cell: Vm is part of ODE system (58 states including Vm)
 * - Tissue:      Vm is external input (57 states excluding Vm)
 */
class GPBCellModel
{
public:
    struct Result {
        std::vector<double> ydot;  // State derivatives (size = CELL_STATE_DIM)
        double I_ion;              // Total ionic current (uA/cm^2)
    };
    
    // Number of cellular states (excluding Vm)
    static constexpr int CELL_STATE_DIM = 57;
    
    // Total ODE dimension in original model (for reference)
    static constexpr int FULL_ODE_DIM = 58;  // Including Vm at index 38
    
    /**
     * Compute ionic current and cell state derivatives for tissue simulation
     * 
     * This is the optimized interface for operator splitting:
     * - Vm is provided externally by the PDE solver
     * - Only cellular states (excluding Vm) are updated
     * - Returns I_ion for use in monodomain equation
     * 
     * State vector layout (size 57):
     * y[0-37]:  States before Vm (gates, concentrations, buffers)
     * y[38-56]: States after Vm (shifted down by 1 from original indices 39-57)
     * 
     * @param t         Current time (ms)
     * @param y         Cellular state variables (size CELL_STATE_DIM = 57)
     *                  Does NOT include Vm - Vm is separate parameter
     * @param Vm        Membrane potential (mV) - provided by PDE solver
     * @return Result   Contains:
     *                  - ydot: derivatives for cellular states (size 57)
     *                  - I_ion: total ionic current for PDE
     */
    static Result compute_ionic_current(
        double t,
        std::vector<double>& y,
        double Vm
    );
    
    /**
     * Legacy interface for backward compatibility with single-cell simulations
     * 
     * This keeps the original 58-state interface where y[38] = Vm
     * Use compute_ionic_current() for tissue simulation instead
     * 
     * @param t         Current time (ms)
     * @param y         Full state vector (size 58) including y[38] = Vm
     * @param Vm        Membrane potential (mV) - overrides y[38]
     * @param I_app     Applied current (ignored in tissue mode)
     * @return Result   Contains ydot (size 58 with ydot[38]=0) and I_ion
     */
    static Result f_heartfailure_1(
        double t,
        std::vector<double>& y,
        double Vm,
        double I_app
    );
    
    /**
     * Get initial conditions for cellular states (excluding Vm)
     * 
     * @return Initial state vector (size 57) for resting conditions
     */
    static std::vector<double> get_initial_state();
    
    /**
     * Map between tissue state indices [0-56] and original full ODE indices [0-57]
     * 
     * Tissue index i maps to:
     * - Original index i     if i < 38  (before Vm)
     * - Original index i+1   if i >= 38 (after Vm, shifted)
     * 
     * @param tissue_idx Index in 57-state tissue vector
     * @return Corresponding index in 58-state full vector
     */
    static inline int tissue_to_full_index(int tissue_idx) {
        return tissue_idx < 38 ? tissue_idx : tissue_idx + 1;
    }
    
    /**
     * Get state variable name for debugging/output
     * 
     * @param tissue_idx Index in 57-state tissue vector
     * @return Name of the state variable
     */
    static const char* get_state_name(int tissue_idx);
};
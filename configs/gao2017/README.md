# Gao et al. IB/FE circulation adaptation

Reference: Gao et al. (2017), *A coupled mitral valve–left ventricle model with
fluid–structure interaction*, DOI 10.1016/j.medengphy.2017.06.042, section 2.4.
Author manuscript: https://arxiv.org/html/1704.01960

This is an LV-only adaptation with idealized immersed basal ports, not a
reproduction of anatomical mitral-valve dynamics or the paper's entire cycle.

## Pressure and flow model

- During accepted filling, inlet supply traction rises linearly from 0 to
  8 mmHg. An independent endocardial follower load rises from 0 to 12 mmHg
  at the filling midpoint and falls to zero at the next stimulus.
- The paper's 800 ms filling waveform is rescaled to the available interval
  from the existing unloading/filling event to the next scheduled stimulus.
  Initial preload, 500/1300 ms pacing and 80 ms artificial tension unloading
  remain the established AFSEI choices. EP/GPB/Land and solid stress are not
  replaced with the paper's prescribed calcium or constitutive calibration.
- The inlet uses an ideal inward-only complementarity law. With outward
  relative flow q, q<=0, port_pressure-supply_pressure>=0, and their product
  is zero. Closure is solved jointly; flow is never clipped after the solve.
- AV is closed during filling. During ejection the actual relative AV flux
  drives the existing BE-RCR update. Rc is counted once. During closed-valve
  steps the arterial capacitor continues to discharge; it is not reset.
- Extra wall load is applied only on the endocardium. The previous additional
  prescribed cap-reference force is disabled for this model. Necessary paired
  constraint reactions on sealed cap regions are retained.

The circulation JSON retains the previously aligned AFSEI afterload values:
Rc=0.0345, Rp=0.7245 mmHg*s/mL, C=1.3 mL/mmHg, peripheral reference 8 mmHg,
initial capacitor pressure 85 mmHg. These are retained project parameters,
not claimed to be a recovered parameter table from Gao's paper.

The internal cap multiplier represents effective traction against the
background-fluid reference. It is NOT a verified pointwise absolute fluid
pressure. Supply and extra wall pressure are separate prescribed actuators;
the output records their powers separately. The profile does not cancel
negative fluid pressure and cannot guarantee a prescribed EDV/ESV.

## Configuration and restart

`two_cycles.json` retains the previous production MAC/IB selection, 128^3 MAC
grid, solid mesh, nodal-CN EP solver, ODE step and tolerances. The current job uses
the unrefined EP mesh (factor 1; 26,889 nodes and 135,430 cells), as requested
on 2026-10-21; the ODE step remains 0.001 ms.
The referenced spatial file explicitly selects `legacy-balanced`; this job
does not silently switch it to the separate `griffith-luo-unified` branch.
Both spatial implementations support the new inlet law; tests exercise their
open/closed derivatives and paired pressure work. The circulation adaptation
does not establish equivalence of either complete numerical scheme to the
paper's explicit time stepping or its anatomical valve geometry.

Start from the specified validated preload, not an old mid-cycle checkpoint.
All new filling parameters and their interpretation are stored in new
checkpoints. Old checkpoints without these keys keep the original source
model. Resuming with a different circulation JSON remains forbidden.

From `/mnt/large2/gjh/AFSEI`, with the afsei environment active:

```bash
python scripts/run_afsei_job.py configs/gao2017/two_cycles.json
```

The output directory must not already exist. Edit only the new job's output
path for another run. Full-cycle validation is pending until a new run finishes.

## Validation

`tests/test_gao_filling.py` covers the separate waveforms and units/config,
complementarity signs, independent GPU finite-difference Jacobians for both
IB branches, pressure-work adjoint identity, an actual coupled GPU short step,
bitwise checkpoint restoration, continuation within physical solver scales
and failure rollback. Existing cycle/wall-loading
tests check that old settings retain their behavior.

`scripts/check_afsei_gao.py` performs independent passive port probes on the
real production LV and MAC grid. These probes start from the same preload,
do not advance EP/ODE and are not a cardiac trajectory. They test blocked
reverse flow, a separately raised inlet-pressure positive control, and the
end-diastolic loading endpoint. An instantaneous mid-filling wall load applied
to an end-diastolic preload is not required to open the inlet in one step. GPU results
are recorded in `results/gao2017_validation_20261021/`.

The optional `spatial_unified_probe.json` is a diagnostic comparison only. The
real 128^3 mid-filling probe on that branch failed the existing finite-step mass
acceptance check; tolerances were not relaxed. The production job therefore
continues to select its explicit pre-existing `legacy-balanced` spatial file.
This known limitation must not be hidden by calling the job a full reproduction
of Griffith--Luo's spatial scheme.

#!/usr/bin/env python3
"""Source fixed-volume validation using EP/GPB/Land and original MAC FSI."""
import argparse
import json
from afsi_torch.simulation.afsei import run

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',required=True,help='source-preloaded or isovolumic AFSEI checkpoint')
    p.add_argument('--output',required=True)
    p.add_argument('--duration-ms',type=float,default=2.)
    p.add_argument('--device',default='cuda')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--pressure-solver',choices=('spectral','multigrid'),
                   help='numerical backend only; new runs default to spectral, restart retains saved choice')
    p.add_argument('--reference-responses',action='store_true',
                   help='disable equivalent-residual cache and pressure-unit reuse for execution comparisons')
    a = p.parse_args()
    report = run(resume=a.resume,output=a.output,duration_s=a.duration_ms/1000.,
                 device=a.device,profile=a.profile,isovolumic=True,pressure_backend=a.pressure_solver,
                 response_optimizations=False if a.reference_responses else None)
    summary = {k:report[k] for k in ('status','accepted_steps','reached_time_s','ep_steps',
        'ode_substeps','rejected_attempts','fluid_shape','fsi_solver','cavity_mode','performance',
        'isovolumic_constraint_implemented','isovolumic_validation_only',
        'isovolumic_execution',
        'windkessel_closed_state_advanced','windkessel_ports_connected','three_field_integration_implemented')}
    summary['isovolumic'] = {k:report['last_state'][k] for k in
        ('cavity_pressure_mmhg','isovolumic_target_ml','isovolumic_error_ml','arterial_pc_mmhg')}
    print(json.dumps(summary,indent=2,allow_nan=False))

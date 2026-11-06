"""3D EP-GPB-Land / circular Neo-Hookean tube / original MAC-IB demo."""
import argparse
import json
from pathlib import Path
from afsi_torch.simulation.circular_tube_config import config_summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-config', action='store_true', help='validate JSON only; do not build or run a simulation')
    p.add_argument('--end-ms', type=float, help='override physical duration in milliseconds')
    p.add_argument('--device', default='cuda:0', help='logical device after CUDA_VISIBLE_DEVICES')
    p.add_argument('--passive', action='store_true', help='zero active stress; keep EP/GPB/Land and passive FSI')
    p.add_argument('--no-stimulus', action='store_true', help='zero stimulus; keep all three solvers')
    p.add_argument('--resume', type=Path, help='continue a compatible open-tube accepted checkpoint at the same dt')
    p.add_argument('--resume-refine-time', action='store_true', help='diagnostic: refine EP/FSI dt together on an aligned checkpoint; keep ODE substep and physics')
    args = p.parse_args(argv)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.end_ms is not None:
        config['time']['end_time_s'] = args.end_ms/1000
    if args.passive:
        config['active']['tension_scale'] = 0.
    if args.no_stimulus:
        config['stimulus']['amplitude_ua_per_cm2'] = 0.
    if args.check_config:
        result = config_summary(config)
    else:
        from afsi_torch.simulation.circular_tube import run
        result = run(config, args.output, device=args.device, resume=args.resume, resume_refine_time=args.resume_refine_time)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()

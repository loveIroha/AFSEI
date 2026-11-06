"""Continue an existing filling pair with the standard AFSEI checkpoint driver.

Preserves a checkpoint before the original stimulus; changes no solver or pacing.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def publish(path, data):
    tmp = path.with_suffix(path.suffix + '.working')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive['metadata']))


def preflight(config, root):
    if config['physical_gpu'] not in (0, 1):
        raise ValueError('physical_gpu must be 0 or 1')
    if config['end_time_ms'] != 1300. or config['pre_stimulus_time_ms'] != 1299.9:
        raise ValueError('this diagnostic preserves the original 1300 ms stimulus')
    source_configs = []
    for case in config['cases']:
        m = metadata(root / case['resume'])
        if (m['producer'] != 'afsei-mac-four-phase-cycle'
                or m['cardiac']['phase'] != 'FILLING'
                or m['cardiac']['beat'] != 1
                or abs(m['time']*1000 - config['start_time_ms']) > 1e-7):
            raise ValueError('source is not the completed first filling segment')
        circulation = m['cardiac_execution']
        if (circulation['filling_model'] != 'gao2017'
                or circulation['gao_filling'].get('inlet_ramp_fraction', 1.) != case['inlet_ramp_fraction']):
            raise ValueError('source inlet profile differs from the requested branch')
        pulses = m['config']['protocol']['pulse_starts_s']
        if pulses != [.5, 1.3]:
            raise ValueError('unexpected pacing schedule')
        source_configs.append(m['config'])
    if source_configs[0] != source_configs[1]:
        raise ValueError('the paired solver configurations must agree')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.configuration.read_text())
    preflight(config, root)
    if args.check_only:
        print(json.dumps(dict(preflight='passed', configuration=str(args.configuration),
            source_times_ms=[config['start_time_ms']]*2, final_time_ms=config['end_time_ms'],
            pre_stimulus_time_ms=config['pre_stimulus_time_ms'], solver_changes=False,
            pacing_changes=False)), flush=True)
        return
    out = root / config['output']
    out.mkdir(parents=True, exist_ok=False)
    publish(out/'experiment.json', config)
    publish(out/'status.json', dict(status='running', pid=os.getpid(), stage='initializing'))
    completed = []
    try:
        for case in config['cases']:
            source = case['resume']
            stages = (
                ('pre_stimulus', config['start_time_ms'], config['pre_stimulus_time_ms'], 'FILLING', 1),
                ('at_stimulus', config['pre_stimulus_time_ms'], config['end_time_ms'], 'ISOVOLUMIC_CONTRACTION', 2),
            )
            for stage, start, target, phase, beat in stages:
                destination = out/case['name']/stage
                job = dict(physical_gpu=config['physical_gpu'], resume=source,
                    checkpoint_every=1000, output=str(destination.relative_to(root)))
                job_path = out/f"{case['name']}_{stage}.json"
                publish(job_path, job)
                command = [sys.executable, '-u', str(root/'scripts/run_afsei_job.py'),
                    str(job_path), '--duration-ms', format(target-start, '.10g')]
                publish(out/'status.json', dict(status='running', case=case['name'], stage=stage,
                    start_time_ms=start, target_time_ms=target, completed=completed, pid=os.getpid()))
                print(json.dumps(dict(case=case['name'], stage=stage, command=command)), flush=True)
                subprocess.run(command, cwd=root, check=True)
                report = json.loads((destination/'report.json').read_text())
                if (report['status'] != 'completed' or report['failure'] is not None
                        or abs(report['reached_time_s']*1000-target) > 1e-7
                        or report['phase'] != phase or report['beat'] != beat):
                    raise RuntimeError('continuation did not reach the expected accepted state')
                saved = metadata(destination/'checkpoint.npz')
                if saved['cardiac_execution']['gao_filling'].get('inlet_ramp_fraction', 1.) != case['inlet_ramp_fraction']:
                    raise RuntimeError('restored inlet profile changed unexpectedly')
                completed.append(dict(case=case['name'], stage=stage, target_time_ms=target,
                    checkpoint=str(destination/'checkpoint.npz')))
                source = str((destination/'checkpoint.npz').relative_to(root))
        subprocess.run([sys.executable, str(root/'scripts/analyze_afsei_filling_followup.py'),
                        str(out)], cwd=root, check=True)
        publish(out/'status.json', dict(status='completed', completed=completed))
    except BaseException as exc:
        publish(out/'status.json', dict(status='failed', failure=repr(exc), completed=completed))
        raise


if __name__ == '__main__':
    main()

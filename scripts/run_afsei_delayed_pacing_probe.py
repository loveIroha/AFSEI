"""Forward-only pacing migration and short continuation through the delayed pace.

Calls the standard checkpoint/EP/FSI driver and retains the old pressure anchors.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def publish(path, data):
    tmp = path.with_suffix(path.suffix+'.working')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive['metadata']))


def preflight(root, config):
    m = metadata(root/config['resume'])
    if config['physical_gpu'] not in (0, 1):
        raise ValueError('physical GPU must be 0 or 1')
    if (m['producer'] != 'afsei-mac-four-phase-cycle'
            or abs(m['time']*1000-config['source_time_ms']) > 1e-7
            or m['cardiac']['phase'] != 'FILLING' or m['cardiac']['beat'] != 1
            or m['cycle']['mode'] != 'zero'):
        raise ValueError('an accepted pre-stimulus filling checkpoint is required')
    if (m['config']['protocol']['pulse_starts_s'] != [.5, 1.3]
            or config['source_time_ms'] != 1299.9
            or config['original_second_stimulus_ms'] != 1300.
            or config['delayed_second_stimulus_ms'] != 1400.
            or config['pre_stimulus_time_ms'] != 1399.9):
        raise ValueError('this diagnostic implements only the authorized 100 ms delay')
    c = m['cardiac_execution']
    if (c['cycles'] != 2 or c['filling_model'] != 'gao2017'
            or c['gao_filling'].get('inlet_ramp_fraction', 1.) != .1
            or c['fill_target_mmhg'] != 8.
            or m['cardiac']['fill_end_tick'] != 26000):
        raise ValueError('expected the completed early-inlet branch and original waveform anchors')
    return m


def migrate(root, config):
    import torch
    from afsi_torch.ep import mac_balanced_checkpoint as restart
    from afsi_torch.ep.protocol import ticks
    from afsi_torch.ep.gao_filling import filling_fraction
    source = root/config['resume']
    out = root/config['output']/'migration'
    out.mkdir(exist_ok=False)
    with torch.no_grad():
        d, state, progress = restart.load(source, cardiac=True)
        old_protocol = asdict(d.protocol)
        old_execution = d.checkpoint_metadata(state)['cardiac_execution']
        old_anchors = (state.fill_start_tick, state.fill_end_tick)
        new_time = config['delayed_second_stimulus_ms']/1000
        protocol = replace(d.protocol, pulse_starts_s=(d.protocol.pulse_starts_s[0], new_time))
        d.protocol = protocol
        d.config = replace(d.config, protocol=protocol)
        # The existing two-pulse controller requires its period metadata to match
        # the 500 -> 1400 ms interval. This probe stops at 1400 ms.
        d.period_ticks = protocol.pulse_ticks[1]-protocol.pulse_ticks[0]
        d.end_tick = protocol.pulse_ticks[0]+d.cycles*d.period_ticks
        assert old_anchors == (state.fill_start_tick, state.fill_end_tick)
        for at_ms in (1300., 1350., 1399.9, 1400.):
            fraction = filling_fraction(state, ticks(at_ms/1000, protocol.dt_ep_s))
            assert d.gao_filling.pressures(fraction, d.fill_target_mmhg) == (8., 0.)
        mask = torch.ones(1, dtype=torch.bool)
        like = torch.zeros(1)
        width = ticks(protocol.pulse_width_s, protocol.dt_ep_s)
        old_tick = ticks(1.3, protocol.dt_ep_s)
        new_tick = ticks(1.4, protocol.dt_ep_s)
        assert all(protocol.stimulus(t, mask, like).item() == 0. for t in range(old_tick, old_tick+width))
        assert all(protocol.stimulus(t, mask, like).item() == protocol.amplitude_ua_per_cm2
                   for t in range(new_tick, new_tick+width))
        assert protocol.stimulus(new_tick+width, mask, like).item() == 0.
        d.validate_control(state)
        torch.testing.assert_close(d.force_for_state(state), state.fsi.force, rtol=1e-9, atol=1e-7)
        migration = dict(source=str(source), at_time_ms=state.fsi.time*1000,
            previous_protocol=old_protocol, new_protocol=asdict(protocol),
            previous_execution=old_execution,
            new_execution=d.checkpoint_metadata(state)['cardiac_execution'],
            filling_anchors_preserved=list(old_anchors), waveform_stretched=False,
            authorized_extra_filling_ms=100., simulation_target_ms=1400.,
            controller_period_metadata_ms=d.period_ticks*protocol.dt_ep_s*1000,
            controller_end_ms=d.end_tick*protocol.dt_ep_s*1000,
            controller_end_not_simulated=True,
            solver_changes=False, active_unloading_changes=False)
        progress.setdefault('numerical_migrations', []).append(dict(
            setting='second_stimulus_time', previous_ms=1300., current_ms=1400.,
            time_ms=state.fsi.time*1000, filling_anchors_preserved=list(old_anchors)))
        saved = out/'checkpoint.npz'
        restart.save(saved, d, state, progress)
    # All physical arrays must be bit-for-bit unchanged by the metadata migration.
    h = hashlib.sha256()
    with np.load(source, allow_pickle=False) as before, np.load(saved, allow_pickle=False) as after:
        keys = sorted(set(before.files)-{'metadata'})
        assert set(before.files) == set(after.files)
        for key in keys:
            a, b = before[key], after[key]
            if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                raise RuntimeError('pacing migration changed physical array '+key)
            h.update(key.encode()); h.update(a.tobytes())
    migration.update(physical_arrays_bitwise_unchanged=True, physical_payload_sha256=h.hexdigest(),
                     checked_array_count=len(keys), stimulus_width_ep_steps=width,
                     no_stimulus_at_1300_verified=True, stimulus_at_1400_verified=True)
    publish(out/'migration.json', migration)
    print(json.dumps(migration), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--migrate-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.configuration.read_text())
    preflight(root, config)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['physical_gpu'])
    if args.check_only:
        print(json.dumps(dict(preflight='passed', source_time_ms=1299.9,
            second_stimulus_ms=1400., extra_filling_ms=100., waveform_stretched=False)), flush=True)
        return
    if args.migrate_only:
        return migrate(root, config)
    out = root/config['output']
    out.mkdir(parents=True, exist_ok=False)
    publish(out/'experiment.json', config)
    completed = []
    try:
        publish(out/'status.json', dict(status='running', stage='migration', pid=os.getpid()))
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
            str(args.configuration.resolve()), '--migrate-only'], cwd=root, check=True)
        source = str((out/'migration/checkpoint.npz').relative_to(root))
        stages = [('original_boundary', 1299.9, 1300., 'FILLING', 1),
            ('extended_filling', 1300., 1399.9, 'FILLING', 1),
            ('delayed_stimulus', 1399.9, 1400., 'ISOVOLUMIC_CONTRACTION', 2)]
        for name, start, end, phase, beat in stages:
            destination = out/name
            job = dict(physical_gpu=config['physical_gpu'], resume=source,
                checkpoint_every=1000, output=str(destination.relative_to(root)))
            job_path = out/f'{name}.json'; publish(job_path, job)
            publish(out/'status.json', dict(status='running', stage=name, start_time_ms=start,
                target_time_ms=end, completed=completed, pid=os.getpid()))
            command = [sys.executable, '-u', str(root/'scripts/run_afsei_job.py'), str(job_path),
                '--duration-ms', format(end-start, '.10g')]
            subprocess.run(command, cwd=root, check=True)
            r = json.loads((destination/'report.json').read_text())
            if (r['status'] != 'completed' or r['failure'] is not None
                    or abs(r['reached_time_s']*1000-end) > 1e-7
                    or r['phase'] != phase or r['beat'] != beat):
                raise RuntimeError('pacing continuation reached an unexpected state')
            m = metadata(destination/'checkpoint.npz')
            if (m['config']['protocol']['pulse_starts_s'] != [.5, 1.4]
                    or m['cardiac']['fill_end_tick'] != 26000):
                raise RuntimeError('pacing schedule or original pressure anchors changed')
            completed.append(dict(stage=name, time_ms=end, checkpoint=str(destination/'checkpoint.npz')))
            source = str((destination/'checkpoint.npz').relative_to(root))
        subprocess.run([sys.executable, str(root/'scripts/analyze_afsei_delayed_pacing_probe.py'),
                        str(out)], cwd=root, check=True)
        publish(out/'status.json', dict(status='completed', completed=completed))
    except BaseException as exc:
        publish(out/'status.json', dict(status='failed', failure=repr(exc), completed=completed))
        raise


if __name__ == '__main__':
    main()

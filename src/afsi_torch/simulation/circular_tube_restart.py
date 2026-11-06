"""Clock/ledger validation for open-tube restart diagnostics (no physics)."""
from copy import deepcopy
import math


def restart_plan(old, new, saved, *, refine_time=False):
    def comparable(config):
        c = deepcopy(config)
        c.pop('description', None)
        c.pop('output', None)
        c['time'].pop('end_time_s')
        if refine_time:
            c['time'].pop('fsi_dt_s')
            c['time'].pop('ep_dt_s')
        return c

    if comparable(old) != comparable(new):
        raise ValueError('restart requires unchanged model, space, solvers and ODE substep')
    old_dt, dt = old['time']['fsi_dt_s'], new['time']['fsi_dt_s']
    for c in (old, new):
        d = c['time']['fsi_dt_s']
        if d <= 0 or not math.isclose(c['time']['ep_dt_s'], d, rel_tol=0, abs_tol=1e-14):
            raise ValueError('restart requires one EP update per FSI step')
        ratio = d*1000/c['time']['ode_dt_ms']
        if ratio < 1 or abs(ratio-round(ratio)) > 1e-9:
            raise ValueError('restart requires integral ODE substeps')
    if refine_time and dt > old_dt*(1+1e-12):
        raise ValueError('diagnostic restart may only keep or reduce the time step')
    clock = float(saved['time_s'])
    old_index = int(saved['step'])
    if old_index < 0 or abs(clock-old_index*old_dt) > 1e-12:
        raise ValueError('checkpoint clock does not match its source configuration')
    index = round(clock/dt)
    if abs(index*dt-clock) > 1e-12:
        raise ValueError('checkpoint time must lie on the target time grid')
    if clock > new['time']['end_time_s']+1e-12:
        raise ValueError('end time precedes checkpoint')
    ep_steps = int(saved['ep_steps'])
    fsi_steps = int(saved.get('fsi_steps', old_index))
    if ep_steps != fsi_steps or ep_steps < 0:
        raise ValueError('accepted EP/FSI update ledgers disagree')
    ode_steps = int(saved.get('ode_substeps', ep_steps*round(old_dt*1000/old['time']['ode_dt_ms'])))
    previous_dt = float(saved.get('previous_dt_s', old_dt))
    if previous_dt <= 0:
        raise ValueError('previous displacement interval must be positive')
    return dict(clock_s=clock, solver_step_index=index,
                accepted_fsi_steps=fsi_steps, accepted_ep_steps=ep_steps,
                accepted_ode_substeps=ode_steps, predictor_scale=dt/previous_dt, previous_dt_s=previous_dt,
                source_dt_s=old_dt, target_dt_s=dt,
                simultaneous_ep_fsi_refinement=dt != old_dt)

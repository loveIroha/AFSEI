"""Sparse accepted-state snapshots for offline cycle diagnostics.

The observer never changes forces, boundary conditions, or solver options.
History fields permit exact linear BE response decomposition after the run.
"""
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from . import mac_balanced_checkpoint as restart
from .protocol import ticks


class CycleDiagnostics:
    def __init__(self, config, output, driver, state, target_tick):
        keys={'snapshot_times_ms','sample_offsets_cells'}
        if not isinstance(config,dict) or set(config)-keys or 'snapshot_times_ms' not in config:
            raise ValueError('diagnostics require snapshot_times_ms and optional sample_offsets_cells')
        times=config['snapshot_times_ms']
        offsets=config.get('sample_offsets_cells',[1,2,3,4])
        if (not isinstance(times,list) or not times or
                any(type(v) not in (int,float) or not isfinite(v) or v<0 for v in times)):
            raise ValueError('snapshot times must be a nonempty list of finite milliseconds')
        if (not isinstance(offsets,list) or not offsets or
                any(type(v) is not int or not 1<=v<=8 for v in offsets)
                or offsets!=sorted(set(offsets))):
            raise ValueError('sample offsets must be unique increasing integers in 1..8')
        self.targets=tuple(ticks(t*.001,driver.protocol.dt_ep_s) for t in times)
        if self.targets!=tuple(sorted(set(self.targets))):
            raise ValueError('snapshot times must increase on the EP clock')
        if any(t<=state.tick or t>target_tick for t in self.targets):
            raise ValueError('snapshot times must lie after the start and within this run')
        self.directory=Path(output)/'diagnostics'
        self.directory.mkdir(parents=True,exist_ok=False)
        self.config=config; self.index=[]; self.elapsed_seconds=0.
        self._publish()

    def step_target(self, current_tick, final_tick):
        return min((t for t in self.targets if t>current_tick),default=final_tick)

    def _publish(self):
        path=self.directory/'index.json'; temporary=path.with_suffix('.json.working')
        temporary.write_text(json.dumps(dict(config=self.config,snapshots=self.index,
            observer_seconds=self.elapsed_seconds,solver_or_model_changed=False),indent=2)+'\n')
        temporary.replace(path)

    @torch.no_grad()
    def record(self, driver, previous, state, progress, info):
        if state.tick not in self.targets: return
        started=perf_counter()
        interval=state.tick-previous.tick
        dt=interval*driver.protocol.dt_ep_s
        if abs(dt-state.force_load.dt_s)>1e-12 or not torch.equal(previous.fsi.x,state.cap_geometry):
            raise ValueError('diagnostic history is not the previous accepted physical state')
        endpoint=driver._fsi_stepper(interval)
        path=self.directory/f'tick_{state.tick:08d}'
        path.mkdir(exist_ok=False)
        restart.save(path/'checkpoint.npz',driver,state,progress)
        # The production transport is deterministic and fixed by the previous
        # accepted velocity and actual dt. Re-evaluate it only at saved times.
        departure=endpoint.flow.advect(previous.fsi.velocity)
        data={f'previous_velocity_{c}':u.detach().cpu().numpy()
              for c,u in enumerate(previous.fsi.velocity)}
        data.update({f'departure_velocity_{c}':u.detach().cpu().numpy()
                     for c,u in enumerate(departure)})
        metadata=dict(schema=1,previous_tick=previous.tick,tick=state.tick,
            previous_time_s=previous.fsi.time,time_s=state.fsi.time,dt_s=dt,
            sample_offsets_cells=self.config.get('sample_offsets_cells',[1,2,3,4]),
            transport_recomputed_from_actual_previous_velocity=True,
            accepted_balanced=info['balanced'])
        with (path/'history.npz').open('wb') as stream:
            np.savez_compressed(stream,metadata=json.dumps(metadata,allow_nan=False),**data)
        self.elapsed_seconds+=perf_counter()-started
        self.index.append(dict(tick=state.tick,time_ms=state.fsi.time*1000,
            checkpoint=str(path/'checkpoint.npz'),history=str(path/'history.npz')))
        self._publish()

"""Inspect only frames already written by the current run; never invoke solvers."""
from pathlib import Path
import json
import numpy as np
root=Path('/mnt/large2/gjh/AFSEI/results');ours=root/'ep_afsei_nodal_opencarp_matched_20261030'
prog=json.loads((ours/'progress.json').read_text());last=max(0,int(prog['time_ms'])-1)
N=1777377
A=np.memmap(ours/'vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,N))
B=np.memmap(root/'opencarp_gpb_h0p0625_20261030/simulation_v2/vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,N))
x=np.arange(321)*.0625
def front(v):
    line=np.asarray(v[:321]);ids=np.flatnonzero(line>=0)
    if not len(ids):return None
    i=int(ids[-1])
    return float(x[i] if i==320 else x[i]-.0625*line[i]/(line[i+1]-line[i]))
print(json.dumps(dict(written_time_ms=last,initial_pointwise_max_difference_mv=float(abs(A[0]-B[0]).max()),
    stimulus_2ms_peak_mv=dict(AFSEI=float(A[2].max()),openCARP=float(B[2].max())),
    axis_wavefront_at_written_time_mm=dict(AFSEI=front(A[last]),openCARP=front(B[last])),
    note='Preliminary frame inspection; final metrics use complete timestep-resolved activation times, not these positions.'),indent=2))

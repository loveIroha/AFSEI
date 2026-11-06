"""Independent CPU EP/FSI path; shared cell ODE results arrive via JSON RPC."""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy.linalg import solve
from reference_flow import ReferenceFlow
from reference_transfer import ReferenceTransfer
from reference_solid import ReferenceSolid

out=Path('/work');c=json.loads((out/'model.json').read_text());a=np.load(out/'inputs.npz')
X=a['X'];n=len(X);flow=ReferenceFlow(c);solid=ReferenceSolid(X,a['cells'],a['resting_tension'],c)
transfer=ReferenceTransfer(X,a['cells'],flow);x=X.copy();previous_x=None
velocity=flow.zeros();pressure=np.zeros(flow.shape);solid.stage(a['land'])
transfer.prepare(x);H=transfer.mobility()
history=dict(x=[x.copy()],pressure=[pressure.copy()],voltage=[a['voltage'].copy()],orders=[],force=[solid.force(x).reshape(3,n).T])
for k in range(3):history[f'velocity_{k}']=[velocity[k].copy()]
diagnostics=[]
def send(data):print('AFSEI_RPC '+json.dumps(data,separators=(',',':')),flush=True)
probe=(H@a['probe'].T.ravel()).reshape(3,n).T
send(dict(status='ready',initial_force=history['force'][0].tolist(),initial_probe_response=probe.tolist()))
for line in sys.stdin:
    request=json.loads(line)
    if request['command']=='finish':
        np.savez_compressed(out/'reference_fields.npz',**{k:np.array(v) for k,v in history.items()})
        (out/'reference_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        send(dict(status='finished',accepted_steps=len(diagnostics)));break
    if request['command']!='advance':raise ValueError('unknown reference command')
    step=int(request['step']);started=time.perf_counter()
    if step!=len(diagnostics)+1:raise ValueError('reference step order mismatch')
    land=np.array(request['land']);solid.stage(land)
    vm=solid.ep_step(np.array(request['voltage']),np.array(request['current']))
    if step>1:transfer.prepare(x);H=transfer.mobility()
    departure=flow.advect(velocity);advected,_=flow.apply(departure)
    base=transfer.interpolate(advected).T.ravel()
    old=x.copy();y=np.zeros(3*n) if previous_x is None else (x-previous_x).T.ravel()
    def residual(d):return d-flow.dt*(base+H@solid.force(old+d.reshape(3,n).T))
    r=residual(y);r0=np.linalg.norm(r)
    opts=c['fsi']['nonlinear'];tol=max(opts['atol'],opts['rtol']*r0)
    iterations=0
    while np.linalg.norm(r)>tol:
        if iterations>=opts['max_iterations']:raise RuntimeError('independent Newton budget exceeded')
        J=np.eye(3*n)-flow.dt*(H@solid.jacobian(old+y.reshape(3,n).T))
        delta=solve(J,-r,assume_a='gen');alpha=1.
        for _ in range(16):
            trial=y+alpha*delta;xx=old+trial.reshape(3,n).T
            if solid.valid(xx):
                candidate=residual(trial)
                if np.linalg.norm(candidate)<np.linalg.norm(r):break
            alpha*=.5
        else:raise RuntimeError('independent Newton line search failed')
        y,r=trial,candidate;iterations+=1
    x=old+y.reshape(3,n).T;force=solid.force(x).reshape(3,n).T
    density=transfer.spread(force)
    velocity,pressure=flow.apply([departure[k]+flow.dt/flow.rho*density[k] for k in range(3)])
    true_residual=np.linalg.norm(y-flow.dt*transfer.interpolate(velocity).T.ravel())
    if true_residual>tol*1.02+1e-14:raise RuntimeError('independent final coupled residual mismatch')
    # Endpoint kernel support, same four-point complete-support predicate.
    transfer_old_orders=transfer.orders.copy();transfer.check_support(x)
    previous_x=old
    history['x'].append(x.copy());history['pressure'].append(pressure.copy());history['voltage'].append(vm.copy())
    history['orders'].append(transfer_old_orders);history['force'].append(force.copy())
    for k in range(3):history[f'velocity_{k}'].append(velocity[k].copy())
    diagnostics.append(dict(step=step,newton_iterations=iterations,residual=true_residual,tolerance=tol,elapsed_seconds=time.perf_counter()-started))
    send(dict(status='accepted',step=step,voltage=vm.tolist(),residual=true_residual,newton_iterations=iterations))

"""Matched multi-step three-field check: native GPU versus external EP/FSI.

GPB/Land CUDA kernels are shared, but each path evolves its own ODE state
using its own voltage. External DOLFINx/SciPy never imports AFSEI operators.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,os,subprocess,time,hashlib
import numpy as np,torch
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.ep.circular_tube import CircularTubeSolid
from afsi_torch.ep.monodomain import Monodomain
from afsi_torch.ep.tissue import TissueState,GPBLand
from afsi_torch.mac.paper_coupling import BEIBStepper


def compare(out,X,candidate,reference,ode_c,ode_r):
    metrics={};limits=dict(displacement_relative=2e-5,velocity_relative=2e-5,pressure_relative=2e-5,
                          force_relative=2e-5,voltage_max_mv=1e-5)
    def relative(a,b):return float(np.linalg.norm((a-b).ravel())/max(np.linalg.norm(b.ravel()),1e-30))
    for key in ['x','pressure','force']:
        a,b=candidate[key][1:],reference[key][1:]
        if key=='x':a=a-X;b=b-X
        name={'x':'displacement_relative','pressure':'pressure_relative','force':'force_relative'}[key]
        metrics[name]=max(relative(aa,bb) for aa,bb in zip(a,b))
    metrics['velocity_relative']=max(relative(np.concatenate([candidate[f'velocity_{i}'][j].ravel() for i in range(3)]),
                                            np.concatenate([reference[f'velocity_{i}'][j].ravel() for i in range(3)]))
                                     for j in range(1,len(candidate['x'])))
    metrics['voltage_max_mv']=float(abs(candidate['voltage']-reference['voltage']).max()*1000)
    orders_equal=bool(np.array_equal(candidate['orders'],reference['orders']))
    components={}
    for name in ['gpb','land']:
        aa,bb=np.array(ode_c[name]),np.array(ode_r[name])
        for j in range(aa.shape[-1]):
            diff=np.linalg.norm((aa[...,j]-bb[...,j]).ravel());scale=np.linalg.norm(bb[...,j].ravel())
            threshold=1e-10*np.sqrt(aa[...,j].size)+2e-5*scale
            components[f'{name}_{j}']=dict(relative=float(diff/max(scale,1e-30)),passed=bool(diff<=threshold))
    finite=all(np.isfinite(v).all() for case in (candidate,reference) for v in case.values())
    passed=finite and orders_equal and all(metrics[k]<=v for k,v in limits.items()) and all(v['passed'] for v in components.values())
    return dict(passed=passed,metrics=metrics,limits=limits,ode_component_checks=components,
                adaptive_orders_identical=orders_equal,all_fields_finite=finite)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=40)
    p.add_argument('--config',type=Path,help='exact same small-case JSON; default keeps the verified built-in setup')
    p.add_argument('--initial-state',type=Path,help='archived complete initial state, independent of the large-tube seed files')
    p.add_argument('--check-inputs',action='store_true',help='check configuration and initial arrays on CPU; do not run PDE/ODE/FSI')
    args=p.parse_args();out=args.output
    if args.steps<2 or (out.exists() and any(out.iterdir())):raise ValueError('positive multi-step case and empty output required')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[2]
    c=json.loads((root/'demo/active_tube_3d/configs/tube_single_isometric_full_gpu.json').read_text())
    c['description']='Small multi-step implementation verification; nonzero electrical initial condition, common cell ODE kernels, independent external EP/FSI'
    c['geometry'].update(axial_elements=8,circumferential_elements=16,thickness_elements=1)
    c['fluid']['shape']=[16,16,32];c['time']['end_time_s']=args.steps*c['time']['fsi_dt_s']
    c['stimulus'].update(start_s=0.,amplitude_ua_per_cm2=0.)
    c['fsi']['nonlinear'].update(rtol=1e-8,atol=1e-11)
    c['ep']['numerics']['solver'].update(rtol=1e-11,atol=1e-14)
    if args.config is not None:
        c=json.loads(args.config.read_text())
        if abs(c['time']['end_time_s']-args.steps*c['time']['fsi_dt_s'])>1e-12:
            raise ValueError('configured duration must equal steps*dt')
    (out/'model.json').write_text(json.dumps(c,indent=2))
    g,fsi,par,num=resolve_config(c)
    if (g.axial_elements,g.circumferential_elements,g.thickness_elements)!=(8,16,1):
        raise ValueError('this archived validation case requires the verified 8/16/1 small mesh')
    if args.check_inputs:
        if args.initial_state is None:raise ValueError('--check-inputs requires --initial-state')
        m0=generate_circular_tube(g)
        with np.load(args.initial_state) as data:
            assert np.array_equal(data['X'],m0.X.numpy()) and np.array_equal(data['cells'],m0.cells.numpy())
            for name,shape in [('voltage',(288,)),('gpb',(288,57)),('land',(288,4)),('zs',(288,)),('zw',(288,)),('resting_tension',(288,))]:
                assert data[name].shape==shape and np.isfinite(data[name]).all(),name
            assert (data['gpb'][:,30:38]>0).all()
        (out/'input_check.json').write_text(json.dumps(dict(passed=True,scope='CPU input/mesh validation only; no numerical advancement'),indent=2))
        print('input check passed; no equations advanced');return
    m=generate_circular_tube(g,device='cuda:0')
    seed=root/'results/domain_space_campaign_20261025/coarse_box2/prefix_60ms/final_state.npz'
    if args.initial_state is None:
        with np.load(seed) as a:
            select=np.arange(41*3*32).reshape(41,3,32)[::5,::2,::2].ravel()
            assert np.max(abs(a['reference'][select]-m.X.cpu().numpy()))<1e-12
            get=lambda k:torch.as_tensor(a[k][select].copy(),device=m.X.device,dtype=m.X.dtype)
            tissue=TissueState(get('voltage_v'),get('gpb'),get('land'));rest=get('resting_tension_dyn_cm2');zs=get('zs');zw=get('zw')
            absolute_start=float(a['time_s'])
    else:
        seed=args.initial_state.resolve()
        with np.load(seed) as a:
            assert np.array_equal(a['X'],m.X.cpu().numpy()) and np.array_equal(a['cells'],m.cells.cpu().numpy())
            get=lambda k:torch.as_tensor(a[k].copy(),device=m.X.device,dtype=m.X.dtype)
            tissue=TissueState(get('voltage'),get('gpb'),get('land'));rest=get('resting_tension');zs=get('zs');zw=get('zw')
            absolute_start=float(a['absolute_ode_start_s'])
    reference_tissue=TissueState(*(v.clone() for v in (tissue.voltage_v,tissue.gpb,tissue.land)))
    one=torch.ones_like(zs)
    solid=CircularTubeSolid(m,g,**{k:v for k,v in c['solid'].items() if k!='end_condition'},
        **{k:v for k,v in c['active'].items() if k not in ('mechanism','rest_reference')},
        dt_s=fsi.time.dt,end_support=c['solid']['end_condition'])
    solid.set_reference(rest);solid.stage(m.X,tissue,zs,zw)
    driver=BEIBStepper(solid,fsi,m.X.device);state=driver.initialize(m.X)
    f=m.X.new_tensor([0.,0.,1.]).expand_as(m.X)
    radial=m.X[:,:2]-m.X.new_tensor(g.center_xy_cm);radial=radial/radial.norm(dim=1,keepdim=True)
    sheet=torch.cat((radial,torch.zeros_like(radial[:,:1])),1)
    ep=Monodomain(m.X,m.cells,f,sheet,parameters=par,numerics=num);ep.prepare(fsi.time.dt)
    ode=GPBLand(fsi.time.dt,c['time']['ode_dt_ms'])
    rng=np.random.default_rng(20261025);probe=rng.normal(size=tuple(m.X.shape))
    np.savez_compressed(out/'inputs.npz',X=m.X.cpu().numpy(),cells=m.cells.cpu().numpy(),
                        voltage=tissue.voltage_v.cpu().numpy(),land=tissue.land.cpu().numpy(),
                        resting_tension=rest.cpu().numpy(),probe=probe)
    contract=dict(nodes=len(m.X),cells=len(m.cells),fluid_shape=list(driver.grid.shape),steps=args.steps,
        dt_s=fsi.time.dt,ode_dt_ms=c['time']['ode_dt_ms'],ode_absolute_start_s=absolute_start,
        initial_mechanics='reference geometry, zero fluid velocity and pressure',
        initial_electrical_state='restriction of saved 60 ms activation; no new stimulus',
        independent_ep_fsi=True,shared_ode_implementation=True,independent_ode_implementation=False,
        full_cardiac_cycle=False,full_size_tube=False,physical_gpu=os.environ.get('AFSEI_GPU'),gpu=torch.cuda.get_device_name(),
        seed=str(seed),seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest())
    (out/'contract.json').write_text(json.dumps(contract,indent=2))
    stderr=(out/'reference_stderr.log').open('w');trace=(out/'reference_stdout.log').open('w')
    cmd=['docker','run','--rm','-i','--network','none','--cpus','2','--memory','6g','--user',f'{os.getuid()}:{os.getgid()}',
         '-w','/work','-e','XDG_CACHE_HOME=/work/reference_cache','-e','XDG_CONFIG_HOME=/tmp/afsei-reference-config',
         '-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2','-v',str(out.resolve())+':/work',
         '-v',str(root/'demo/verification')+':/code:ro','mapengfei/dolfinx:local','python3','-u','/code/independent_coupled_reference.py']
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1)
    def receive():
        while True:
            line=process.stdout.readline()
            if not line:raise RuntimeError('independent reference exited; see reference_stderr.log')
            trace.write(line);trace.flush()
            if line.startswith('AFSEI_RPC '):return json.loads(line[len('AFSEI_RPC '):])
    def rpc(data):process.stdin.write(json.dumps(data,separators=(',',':'))+'\n');process.stdin.flush();return receive()
    candidate=dict(x=[],voltage=[],pressure=[],force=[],orders=[])
    for k in range(3):candidate[f'velocity_{k}']=[]
    ode_c=dict(gpb=[],land=[]);ode_r=dict(gpb=[],land=[]);native_info=[];completed=0
    def record():
        candidate['x'].append(state.x.cpu().numpy().copy());candidate['voltage'].append(tissue.voltage_v.cpu().numpy().copy())
        candidate['pressure'].append(state.pressure.cpu().numpy().copy());candidate['force'].append(state.force.cpu().numpy().copy())
        for k in range(3):candidate[f'velocity_{k}'].append(state.velocity[k].cpu().numpy().copy())
        for name in ['gpb','land']:
            ode_c[name].append(getattr(tissue,name).cpu().numpy().copy());ode_r[name].append(getattr(reference_tissue,name).cpu().numpy().copy())
    try:
        ready=receive();force_ref=np.array(ready['initial_force']);probe_ref=np.array(ready['initial_probe_response'])
        rel=lambda a,b:float(np.linalg.norm((a-b).ravel())/max(np.linalg.norm(b.ravel()),1e-30))
        initial_force_error=rel(state.force.cpu().numpy(),force_ref)
        stencil=driver.transfer.prepare(state.x)
        density,_=driver.transfer.spread(torch.as_tensor(probe,device=m.X.device),stencil)
        response=driver.flow.response(density);nodal,_=driver.transfer.interpolate(response,stencil)
        probe_error=rel(nodal.cpu().numpy(),probe_ref);driver.transfer.reset_warm_start()
        del stencil,density,response,nodal
        preflight=dict(initial_force_relative=initial_force_error,coupled_response_relative=probe_error)
        (out/'preflight.json').write_text(json.dumps(preflight,indent=2));print('preflight',json.dumps(preflight),flush=True)
        if initial_force_error>1e-9 or probe_error>2e-8:raise RuntimeError('initial coupled operator mismatch')
        record();started=time.perf_counter()
        for step in range(1,args.steps+1):
            time_s=absolute_start+(step-1)*fsi.time.dt
            next_tissue,_,_=ode.step(ep,tissue,one,zs,zw,time_s,torch.zeros_like(one))
            reference_ode,current=ode.advance_ode(reference_tissue,one,zs,zw,time_s)
            answer=rpc(dict(command='advance',step=step,voltage=reference_tissue.voltage_v.cpu().numpy().tolist(),
                            current=current.cpu().numpy().tolist(),land=reference_ode.land.cpu().numpy().tolist()))
            if answer['status']!='accepted' or answer['step']!=step:raise RuntimeError('reference protocol mismatch')
            reference_tissue=replace(reference_ode,voltage_v=torch.as_tensor(answer['voltage'],device=m.X.device,dtype=m.X.dtype))
            solid.stage(state.x,next_tissue,zs,zw);state,info=driver.step(state,diagnostics=True)
            tissue=next_tissue;completed=step;record();candidate['orders'].append(driver.transfer._last_prepared_rule.orders.cpu().numpy().copy())
            ni=info['nonlinear']
            native_info.append(dict(step=step,iterations=ni['iterations'],residual_norm=ni['residual_norm'],tolerance=ni['tolerance'],fluid_solves=ni['fluid_solves']))
            if step==1 or step%10==0:print('accepted paired step',step,'of',args.steps,flush=True)
        end=rpc(dict(command='finish'));assert end['accepted_steps']==args.steps
        process.wait(timeout=30)
        candidate={k:np.array(v) for k,v in candidate.items()};reference=dict(np.load(out/'reference_fields.npz'))
        np.savez_compressed(out/'candidate_fields.npz',**candidate)
        np.savez_compressed(out/'ode_fields.npz',**{f'candidate_{k}':np.array(v) for k,v in ode_c.items()},
                            **{f'reference_{k}':np.array(v) for k,v in ode_r.items()})
        report=compare(out,m.X.cpu().numpy(),candidate,reference,ode_c,ode_r)
        report.update(status='completed',accepted_steps=completed,elapsed_window_ms=completed*fsi.time.dt*1000,
            wall_seconds_including_reference_wait=time.perf_counter()-started,contract=contract,preflight=preflight,
            performance_comparison=False,ode_substeps_per_path=completed*ode.substeps)
        (out/'report.json').write_text(json.dumps(report,indent=2));(out/'native_diagnostics.json').write_text(json.dumps(native_info,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='ode_component_checks'},indent=2),flush=True)
        if not report['passed']:raise RuntimeError('paired trajectory comparison failed; no reliability claim accepted')
    except Exception as exc:
        if not (out/'report.json').exists():(out/'report.json').write_text(json.dumps(dict(status='failed',paired_accepted_steps=completed,error=str(exc),contract=contract),indent=2))
        raise
    finally:
        ep.close()
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill()
        stderr.close();trace.close()

if __name__=='__main__':main()

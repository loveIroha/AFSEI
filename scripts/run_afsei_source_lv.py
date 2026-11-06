"""Original LV-only source-Windkessel cycle. Usage: python THIS.py config.json."""
import argparse
import csv
import json
from dataclasses import asdict,replace
from pathlib import Path
from time import perf_counter
import traceback
import xml.etree.ElementTree as ET
import torch
import meshio
from afsi_torch.config import _decode
from afsi_torch.paper_lv import imported_model
from afsi_torch.ep.config import AFSEIConfig
from afsi_torch.ep.mac_source_cycle import SourceLVStepper
from afsi_torch.ep.windkessel import WindkesselParameters
from afsi_torch.ep.protocol import ticks
from afsi_torch.ep.checkpoint import save,load


def write_frame(root,driver,state):
    name = f'wall_{state.tick:09d}.vtu'
    array = lambda v:v.detach().cpu().numpy()
    recorder=getattr(driver,'active_recorder',None)
    extra_points={};extra_cells={}
    if recorder is not None:
        recorder.record(state,field=True)
        extra_points['active_force_dyn']=array(recorder.last_fields['active_force_dyn'])
        extra_cells['active_tension_mean_dyn_cm2']=[array(recorder.last_fields['active_tension_mean_dyn_cm2'])]
    meshio.write_points_cells(root/name,array(state.fsi.x),[('tetra',array(driver.mesh.cells))],
        point_data=dict(displacement_cm=array(state.fsi.x-driver.mesh.X),
                        voltage_mV=array(state.tissue.voltage_v*1000),
                        land_stretch=array(state.strain.stretch),**extra_points),cell_data=extra_cells)
    path = root/'wall.pvd'
    if path.exists(): tree = ET.parse(path); collection = tree.getroot().find('Collection')
    else:
        top = ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
        collection = ET.SubElement(top,'Collection'); tree = ET.ElementTree(top)
    if not any(item.get('file') == name for item in collection):
        ET.SubElement(collection,'DataSet',timestep=str(state.fsi.time),group='',part='0',file=name)
    temporary = path.with_suffix('.pvd.tmp');tree.write(temporary,encoding='utf-8',xml_declaration=True)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config');parser.add_argument('--output');parser.add_argument('--max-steps',type=int)
    parser.add_argument('--resume');args = parser.parse_args()
    request = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if request.get('schema') != 'afsei-source-lv-v1': raise ValueError('source LV JSON schema required')
    root = Path(args.output or request['output']);root.mkdir(parents=True,exist_ok=True)
    if not args.resume and (root/'history.csv').exists():
        raise ValueError('output already has history; use --resume or a new output directory')
    started = perf_counter()
    if args.resume:
        driver,state,progress = load(args.resume)
        if not isinstance(driver,SourceLVStepper): raise ValueError('cannot resume an old tract/port experiment')
        # A restart uses stored physics, never silently substitutes this JSON's physics.
    else:
        cfg = _decode(AFSEIConfig,request['config'])
        if (cfg.direction_mapping != 'l2-cg1' or cfg.ep_refinement_factor != 1
                or cfg.ep_numerics.backend == 'amgx_pcg_amg'):
            raise ValueError('this LV configuration requires L2 directions, original EP mesh, no AmgX')
        mesh = imported_model(cfg.fsi,device='cuda:0').mesh
        driver = SourceLVStepper(mesh,cfg);driver.set_execution(**request['execution'])
        state = driver.initialize()
        state = replace(state,windkessel=WindkesselParameters(**request['windkessel']))
        progress = {}
    driver.profile = bool(request.get('profile',False))
    driver.validate_control(state)
    (root/'configuration.json').write_text(json.dumps(dict(request=request,config=asdict(driver.config),
        execution=driver.execution_settings(),windkessel=asdict(state.windkessel)),indent=2))
    output = request['reporting'];stop = ticks(request['maximum_time_s'],driver.protocol.dt_ep_s)
    if stop <= state.tick: raise ValueError('maximum_time_s must exceed restart time')
    recording=request.get('active_recording',{})
    if recording.get('enabled',False):
        from afsi_torch.ep.active_recording import ActiveStressRecorder
        for key in ('statistics_every_steps','fields_every_steps'):
            if type(recording.get(key)) is not int or recording[key]<1:raise ValueError('positive recording step intervals required')
        driver.active_recorder=ActiveStressRecorder(root,driver,recording)
        driver.active_recorder.record(state,force=True)
    if output['write_vtk']: write_frame(root,driver,state)
    prepared = perf_counter(); print(f'Prepared original LV {len(driver.mesh.X)} nodes, '
        f'{len(driver.mesh.cells)} cells; MAC {driver.config.fsi.fluid.shape}',flush=True)
    count = 0;status = 'running';error = None
    history = root/'history.csv';append = history.exists() and history.stat().st_size > 0
    columns = ('step','time_s','phase','beat','volume_ml','blood_volume_ml','volume_balance_error_ml',
        'endocardial_load_mmhg','arterial_pressure_mmhg','pc_mmhg','q_out_0d_ml_s',
        'q_in_volume_ml_s','cumulative_in_ml','cumulative_out_ml','closure_tick','filling_start_tick',
        'accepted_dt_s','wall_seconds','newton_iterations','gmres_iterations','rejected_attempts')
    try:
        with history.open('a',newline='') as stream,(root/'events.jsonl').open('a') as event_stream:
            writer = csv.DictWriter(stream,fieldnames=columns)
            if not append: writer.writeheader()
            while not driver.completed(state) and state.tick < stop:
                t0 = perf_counter();state,info = driver.step(state,target_tick=stop,
                    diagnostics=(state.fsi.step % output['log_every'] == 0))
                count += 1;circ = info['circulation'];nlin = info['nonlinear']
                row = dict(circ,step=state.fsi.step,time_s=state.fsi.time,
                    accepted_dt_s=info['accepted_dt_s'],wall_seconds=perf_counter()-t0,
                    newton_iterations=nlin['iterations'],gmres_iterations=nlin['gmres_iterations'],
                    rejected_attempts=len(info['rejected_attempts']))
                writer.writerow({k:row[k] for k in columns});stream.flush()
                event = circ['phase'] != circ['previous_phase']
                if hasattr(driver,'active_recorder'):driver.active_recorder.record(state,force=event)
                if event:
                    event_stream.write(json.dumps(dict(time_s=state.fsi.time,**circ))+'\n');event_stream.flush()
                if event or state.fsi.step % output['log_every'] == 0:
                    print(f"step={state.fsi.step} t={state.fsi.time*1000:.3f} ms {state.phase} "
                          f"V={circ['volume_ml']:.5f} mL p_load={state.cavity_pressure_mmhg:.4f} mmHg "
                          f"Q_0D={state.outlet_flow_ml_s:.5f} mL/s dt={info['accepted_dt_s']:.6g} "
                          f"seconds/step={row['wall_seconds']:.3f}",flush=True)
                if output['write_vtk'] and (event or state.fsi.step % output['vtk_every'] == 0):
                    write_frame(root,driver,state)
                if event or state.fsi.step % output['checkpoint_every'] == 0:
                    save(root/'checkpoint.npz',driver,state,dict(total_steps=state.fsi.step))
                if args.max_steps is not None and count >= args.max_steps: break
        status = 'completed' if driver.completed(state) else 'step_limit' if args.max_steps else 'time_limit'
    except Exception:
        status,error = 'failed',traceback.format_exc()
    finally:
        if hasattr(driver,'active_recorder'):driver.active_recorder.record(state,force=True)
        save(root/'checkpoint.npz',driver,state,dict(total_steps=state.fsi.step))
        if output['write_vtk']: write_frame(root,driver,state)
        report = dict(status=status,error=error,accepted_steps=count,time_s=state.fsi.time,
            beat=state.beat,phase=state.phase,preparation_seconds=prepared-started,
            run_seconds=perf_counter()-prepared,pulse_starts_s=driver.protocol.pulse_starts_s,
            circulation='source pressure-driven valve/RCR + volume balance',
            fluid_ports_connected=False,anatomical_valves_present=False,
            endocardial_pressure_is_fluid_cavity_mean=False)
        (root/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
        if hasattr(driver,'active_recorder'):driver.active_recorder.close()
    if status == 'failed': raise SystemExit(1)


if __name__ == '__main__': main()

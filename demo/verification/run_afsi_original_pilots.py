"""Host-side queue of original AFSI pilots; preserve every attempted case.

Uses the existing verified runtime without installing or changing packages.
These runs share a host with grid studies and are not formal isolated timings.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    a=p.parse_args();root=a.root;folder=root/'afsi_original'
    work='/tmp/afsei_afsi_original_20261025';container='afsi_dev_ljy'
    status=root/'afsi_original_pilots_status.json';results={}
    def record(stage,**kw):
        tmp=status.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(stage=stage,cases=results,
            formal_matched_accuracy_speedup_established=False,**kw),indent=2));tmp.replace(status)
    agreement=json.loads((folder/'mpi_agreement.json').read_text())
    if any(v['relative_l2']>1e-8 for v in agreement.values()):
        raise RuntimeError('single/MPI numerical agreement failed')
    base=json.loads((folder/'smoke_config.json').read_text())
    cases=[('n16_dt50_mpi4',16,5e-5,4),('n16_dt20_mpi4',16,2e-5,4),
           ('n32_dt20_mpi4',32,2e-5,4),('n32_dt10_mpi4',32,1e-5,4)]
    for name,n,dt,ranks in cases:
        saved=folder/name/'report.json'
        if saved.exists():
            results[name]=json.loads(saved.read_text());continue
        # Never silently overwrite an earlier partial run or relax its settings.
        logpath=folder/(name+'.log')
        if logpath.exists():
            results[name]=dict(status='previous-attempt-needs-review',log=str(logpath));continue
        config=dict(base,fluid_elements_per_axis=n,dt_s=dt,end_time_s=.002,
            log_every=round(.0005/dt),field_every=round(.0005/dt),
            description='Original AFSI constant EP-derived load FSI pilot, not a full EP/ODE solve')
        cp=folder/(name+'.json');cp.write_text(json.dumps(config,indent=2))
        subprocess.run(['docker','cp',str(cp),container+':'+work+'/'+cp.name],check=True)
        command=['docker','exec',container,'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
            'MKL_NUM_THREADS=1','mpirun','-n',str(ranks),'/dolfinx-env/bin/python',work+'/run.py',
            '--config',work+'/'+cp.name,'--inputs',work+'/model_inputs.npz','--output',work+'/'+name]
        record('running',active_case=name,command=command)
        with logpath.open('w') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        copied=subprocess.run(['docker','cp',container+':'+work+'/'+name,str(folder/name)],
            capture_output=True,text=True)
        results[name]=(json.loads(saved.read_text()) if saved.exists() else
            dict(status='failed',returncode=proc.returncode,copy_error=copied.stderr[-1000:]))
        results[name]['process_returncode']=proc.returncode
        record('between-cases')
    record('pilots-completed')
    analysis=Path(__file__).with_name('analyze_afsi_original_pilots.py')
    if analysis.exists():
        with (folder/'comparison.log').open('w') as log:
            code=subprocess.run([sys.executable,str(analysis),'--root',str(root)],stdout=log,stderr=subprocess.STDOUT).returncode
        record('pilots-and-analysis-completed' if code==0 else 'pilots-completed-analysis-failed',analysis_returncode=code)

if __name__=='__main__':main()

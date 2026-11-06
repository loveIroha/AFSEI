from pathlib import Path
import json,time,subprocess,sys,os
R=Path('/mnt/large2/gjh/AFSEI');O=R/'results/tube_cg_euler_wave_20261031';full=O/'full'
while True:
    try:
        report=json.loads((full/'report.json').read_text())
        if (full/'wave_sections.npz').exists() and (full/'final_state.npz').exists():break
    except (OSError,ValueError):pass
    time.sleep(5)
env=os.environ.copy();env['MPLCONFIGDIR']=str(R/'.cache/matplotlib')
subprocess.run([sys.executable,str(R/'demo/active_tube_2d/cg_euler_wave/plot_wave.py'),'--source',str(full)],env=env,check=True)
(O/'postprocessing_completed.json').write_text(json.dumps(dict(status='completed',simulation_status=report['status'],time_s=report['time_s']),indent=2))
print('WAVE_POSTPROCESSING_COMPLETED')

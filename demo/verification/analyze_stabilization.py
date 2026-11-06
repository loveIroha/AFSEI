"""Compare new switch windows with existing baseline data; never advance."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np,meshio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/large2/gjh/AFSEI/results');OUT=ROOT/'volumetric_stabilization_20261025'


def load_baseline(factor):
    folder=ROOT/'domain_space_campaign_20261025'/('coarse_box2' if factor==1 else 'fine_box2')
    a=np.load(folder/'prefix_60ms/final_state.npz');X=a['reference'];cells=a['cells'];vertices=X[cells]
    vol=np.linalg.det((vertices[:,1:]-vertices[:,:1]).transpose(0,2,1))/6
    rows=[]
    for h in json.loads((folder/'continued_300ms/history.json').read_text()):
        t=h['time_s']*1000
        if t<60-1e-8 or t>100+1e-8:continue
        # Existing field frames are every 10ms; log-only 5ms samples have no
        # independent J/EP field file. Use all five preselected common frames.
        if abs(t/10-round(t/10))>1e-8:continue
        name=f"solid_{h['step']:06d}.vtu";path=folder/'continued_300ms/vtk'/name
        if not path.exists():path=folder/'prefix_60ms/vtk'/name
        m=meshio.read(path);assert np.array_equal(m.cells_dict['tetra'],cells)
        J=m.cell_data_dict['J_min']['tetra'];centers=m.points[a['inner_rings']].mean(1)
        ep=meshio.read(path.with_name(f"ep_{h['step']:06d}.vtu"))
        rows.append(dict(absolute_ode_time_ms=t,wall_volume_change_percent=100*(h['wall_volume_cm3']/vol.sum()-1),
            J_rms_difference_from_one=float(np.sqrt((vol*(J-1)**2).sum()/vol.sum())),
            minimum_detF=h['minimum_detF'],maximum_detF=h['maximum_detF'],
            minimum_projected_diameter_mm=20*np.sqrt(h['minimum_section_area_cm2']/np.pi),
            maximum_centerline_transverse_displacement_mm=float(np.linalg.norm(centers[:,:2]-.7,axis=1).max()*10),
            maximum_applied_tension_kpa=h['maximum_applied_tension_dyn_cm2']/1e4,
            ep_voltage_mv=ep.point_data['voltage_mV'],ep_calcium_um=ep.point_data['calcium_uM']))
    return rows


def main():
    design=json.loads((OUT/'design.json').read_text());results={};data={};checks={}
    for factor in (1,2):
        folder=OUT/f'switch_solid{factor}';report=json.loads((folder/'report.json').read_text())
        if report['status']!='completed':raise RuntimeError(f'Not completed: {folder}')
        history=json.loads((folder/'history.json').read_text());base=load_baseline(factor)
        lookup={round(h['absolute_ode_time_ms'],7):h for h in history}
        pairs=[(b,lookup[round(b['absolute_ode_time_ms'],7)]) for b in base]
        bmax=max(abs(b['wall_volume_change_percent']) for b,s in pairs)
        smax=max(abs(s['wall_volume_change_percent']) for b,s in pairs)
        bend=max(s['maximum_centerline_transverse_displacement_mm'] for b,s in pairs)
        epmax=camax=0.
        for b,s in pairs:
            ep=meshio.read(folder/'vtk'/f"ep_{s['step']:06d}.vtu")
            epmax=max(epmax,float(abs(ep.point_data['voltage_mV']-b['ep_voltage_mv']).max()))
            camax=max(camax,float(abs(ep.point_data['calcium_uM']-b['ep_calcium_um']).max()))
        ta_error=max(abs(b['maximum_applied_tension_kpa']-s['maximum_applied_tension_kpa']) for b,s in pairs)
        # Initial reference polygon diameter is common to both physical meshes.
        initial_diameter=20*np.sqrt(.5*32*np.sin(2*np.pi/32)*.3**2/np.pi)
        check=dict(all_steps_accepted=report['accepted_steps']==design['acceptance']['accepted_steps'],
            positive_J=report['minimum_detF']>0,nonlinear_accepted=report['max_sampled_residual_ratio']<=1,
            volume_reduction=1-smax/bmax>=design['acceptance']['common_sampled_volume_error_reduction_fraction_min'],
            nontrivial_contraction=report['minimum_projected_diameter_mm']/initial_diameter<=.9,
            unchanged_EP_voltage=epmax<=design['acceptance']['ep_voltage_max_difference_mv'],
            unchanged_calcium=camax<=design['acceptance']['ep_calcium_max_difference_um'])
        # Ensure every saved PVD references existing files, and all three fields.
        pvd={}
        for kind in ('solid','fluid','ep'):
            entries=ET.parse(folder/'vtk'/f'{kind}.pvd').findall('.//DataSet')
            pvd[kind]=len(entries)
            if not all((folder/'vtk'/e.attrib['file']).is_file() for e in entries):raise AssertionError('Broken PVD')
        check['pvd_complete']=len(set(pvd.values()))==1 and pvd['ep']==report['vtk_frames']
        checks[f'solid{factor}']={k:bool(v) for k,v in check.items()}
        summary=dict(common_times_ms=[b['absolute_ode_time_ms'] for b,s in pairs],
            baseline_max_common_wall_volume_error_percent=bmax,stabilized_max_common_wall_volume_error_percent=smax,
            volume_error_reduction_percent=100*(1-smax/bmax),
            baseline_final_wall_volume_percent=pairs[-1][0]['wall_volume_change_percent'],
            stabilized_final_wall_volume_percent=pairs[-1][1]['wall_volume_change_percent'],
            baseline_final_J_rms=pairs[-1][0]['J_rms_difference_from_one'],stabilized_final_J_rms=pairs[-1][1]['J_rms_difference_from_one'],
            baseline_final_projected_diameter_mm=pairs[-1][0]['minimum_projected_diameter_mm'],
            stabilized_final_projected_diameter_mm=pairs[-1][1]['minimum_projected_diameter_mm'],
            max_common_transverse_centerline_displacement_mm=bend,maximum_voltage_difference_mv=epmax,
            maximum_calcium_difference_um=camax,maximum_tension_difference_kpa=ta_error,
            stabilized_compute_seconds_per_step=report['warmed_seconds_per_step'],
            minimum_detF=report['minimum_detF'],maximum_detF=report['maximum_detF'],
            max_sampled_nonlinear_residual_ratio=report['max_sampled_residual_ratio'],pvd_frames=pvd)
        results[f'solid{factor}']=summary
        cleanbase=[{k:v for k,v in b.items() if not k.startswith('ep_')} for b in base]
        data[f'solid{factor}']=dict(baseline=cleanbase,stabilized=history)
    independent=json.loads((OUT/'independent_stress/report.json').read_text())
    passed=all(all(c.values()) for c in checks.values()) and independent['passed']
    result=dict(status='completed',screening_passed=passed,design=design,results=results,checks=checks,
        independent_stress=independent,full_trajectory_spatial_convergence_established=False,
        official_default_changed=False,performance_speedup_established=False)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2));(OUT/'comparison_data.json').write_text(json.dumps(data,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11.7,7.8),layout='constrained');colors=['#176d9c','#be602b']
    specs=[('wall_volume_change_percent','Wall-volume change from reference (%)'),
           ('J_rms_difference_from_one','Reference-volume RMS of J - 1'),
           ('minimum_projected_diameter_mm','Minimum x-y projected diameter (mm)'),
           ('maximum_applied_tension_kpa','Maximum Land input tension (kPa)')]
    for ax,(key,label) in zip(axes.flat,specs):
        for factor,color in zip((1,2),colors):
            for name,ls in [('baseline','--'),('stabilized','-')]:
                rows=data[f'solid{factor}'][name]
                ax.plot([r['absolute_ode_time_ms'] for r in rows],[r[key] for r in rows],ls=ls,color=color,
                    marker='o' if name=='baseline' else None,ms=3,lw=1.7,label=f'Solid {factor} / {name}')
        ax.set(xlabel='Absolute electrical time (ms)',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('IB/FE deviatoric + logarithmic-volume stabilization: short switch test',fontsize=13)
    fig.supxlabel('Switch at saved 60 ms state; existing baselines reused. Same MAC/IB, BE-BE, EP and active amplitude. Not a full-trajectory convergence test.',fontsize=8)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'comparison.{ext}',dpi=190)
    plt.close(fig);print(json.dumps(result,indent=2),flush=True)
    if not passed:raise AssertionError('Short screening criteria not all satisfied')


if __name__=='__main__':main()

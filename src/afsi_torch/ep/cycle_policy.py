"""Physical filling clocks and accepted-state periodicity; no force fitting."""
from dataclasses import dataclass, asdict
from math import isfinite, sqrt
import torch


@dataclass(frozen=True)
class FillingClock:
    duration_s: float = .8
    inlet_ramp_s: float = .8
    wall_peak_s: float = .4
    hold_s: float = .1
    loading_mode: str = 'gao-extra-wall'
    pacing_mode: str = 'delay-only'

    def __post_init__(self):
        values=(self.duration_s,self.inlet_ramp_s,self.wall_peak_s,self.hold_s)
        if any(type(v) not in (int,float) or not isfinite(v) for v in values):
            raise ValueError('finite physical filling times required')
        if (self.duration_s<=0 or not 0<self.inlet_ramp_s<=self.duration_s
                or not 0<self.wall_peak_s<self.duration_s or self.hold_s<0):
            raise ValueError('invalid physical filling clock')
        if self.loading_mode not in ('gao-extra-wall','source-wall','inlet-only'):
            raise ValueError('unknown filling load placement')
        if self.pacing_mode not in ('delay-only','after-filling'):
            raise ValueError('unknown physical pacing clock')

    def pressures(self,elapsed_s,parameters,target_mmhg):
        if not isfinite(elapsed_s):raise ValueError('finite elapsed filling time required')
        t=max(0.,elapsed_s)
        if self.loading_mode=='source-wall':
            return 0.,target_mmhg*min(1.,t/self.duration_s)
        if self.loading_mode=='inlet-only':
            return target_mmhg*min(1.,t/self.duration_s),0.
        inlet=parameters.inlet_start_mmhg+(target_mmhg-parameters.inlet_start_mmhg)*min(1.,t/self.inlet_ramp_s)
        wall=(t/self.wall_peak_s if t<=self.wall_peak_s else
              max(0.,(self.duration_s-t)/(self.duration_s-self.wall_peak_s)))
        return inlet,parameters.wall_peak_mmhg*wall


@dataclass(frozen=True)
class PeriodicCriteria:
    volume_relative: float = .01
    pc_mmhg: float = 1.
    rr_relative: float = .01
    voltage_rms_mv: float = 1.
    calcium_relative_rms: float = .02
    land_rms: float = .01
    stretch_rms: float = .005
    position_rms_cm: float = .005
    velocity_relative_rms: float = .02
    pressure_rms_mmhg: float = 1.
    consecutive: int = 2

    def __post_init__(self):
        if type(self.consecutive) is not int or self.consecutive<1:
            raise ValueError('positive consecutive periodic comparisons required')
        if any(type(v) not in (float,int) or not isfinite(v) or v<=0
               for k,v in asdict(self).items() if k!='consecutive'):
            raise ValueError('positive finite periodic tolerances required')


class PeriodicMonitor:
    """Observe only accepted pre-stimulus states; persist exact reference fields.

    Field reductions occur once per beat, not in residual/Jacobian calls.
    Two nearly equal EDVs alone never establish a periodic coupled state.
    """
    NAMES=('voltage','calcium','land','stretch','position','fluid_0','fluid_1','fluid_2','pressure')

    def __init__(self,criteria=None):
        self.criteria=PeriodicCriteria(**criteria) if isinstance(criteria,dict) else criteria or PeriodicCriteria()
        self.reference={};self.samples=[];self.comparisons=[];self.clean=0

    def capture(self,state):
        if self.samples and state.tick<=self.samples[-1]['tick']:
            raise ValueError('periodic samples must have increasing accepted ticks')
        fields=dict(voltage=state.tissue.voltage_v*1000.,calcium=state.tissue.gpb[:,37],
                    land=state.tissue.land,stretch=state.strain.stretch,position=state.fsi.x,
                    pressure=state.fsi.pressure)
        fields.update({'fluid_'+str(i):u for i,u in enumerate(state.fsi.velocity)})
        sample=dict(tick=state.tick,time_s=state.fsi.time,beat=state.beat,
                    volume_ml=state.edv_ml,pc_mmhg=state.arterial_pc_mmhg)
        if self.samples:
            old=self.samples[-1];rr=sample['time_s']-old['time_s'];sample['rr_s']=rr
            ref=self.reference
            count=sum(fields['fluid_'+str(i)].numel() for i in range(3))
            velocity_error=sum((fields['fluid_'+str(i)]-ref['fluid_'+str(i)]).square().sum() for i in range(3))/count
            velocity_scale=sum(ref['fluid_'+str(i)].square().sum() for i in range(3))/count
            metrics=torch.stack(((fields['voltage']-ref['voltage']).square().mean().sqrt(),
                (fields['calcium']-ref['calcium']).square().mean().sqrt()/ref['calcium'].square().mean().sqrt().clamp_min(1e-12),
                (fields['land']-ref['land']).square().mean().sqrt(),
                (fields['stretch']-ref['stretch']).square().mean().sqrt(),
                (fields['position']-ref['position']).square().mean().sqrt(),
                velocity_error.sqrt()/velocity_scale.sqrt().clamp_min(1.),
                (fields['pressure']-ref['pressure']).square().mean().sqrt()/1333.22368421)).cpu().tolist()
            names=('voltage_rms_mv','calcium_relative_rms','land_rms','stretch_rms',
                   'position_rms_cm','velocity_relative_rms','pressure_rms_mmhg')
            diff=dict(zip(names,metrics),volume_relative=abs(sample['volume_ml']-old['volume_ml'])/old['volume_ml'],
                pc_mmhg=abs(sample['pc_mmhg']-old['pc_mmhg']),
                rr_relative=(abs(rr-old['rr_s'])/old['rr_s'] if 'rr_s' in old else None))
            passed=all(v is not None and v<=getattr(self.criteria,k) for k,v in diff.items())
            self.clean=self.clean+1 if passed else 0
            self.comparisons.append(dict(beat=state.beat,passed=passed,metrics=diff,
                consecutive_passes=self.clean,periodic=self.clean>=self.criteria.consecutive))
        self.samples.append(sample)
        self.reference={k:v.detach().clone() for k,v in fields.items()}
        return self.summary()

    def summary(self):
        tail=[]
        for comparison in reversed(self.comparisons):
            if not comparison['periodic']:break
            tail.append(comparison['beat'])
        return dict(periodic=self.clean>=self.criteria.consecutive,consecutive_passes=self.clean,
                    samples=len(self.samples),last=self.comparisons[-1] if self.comparisons else None,
                    completed_periodic_cycles=(max(tail)-min(tail) if tail else 0))

    def metadata(self):
        return dict(version=2,criteria=asdict(self.criteria),samples=self.samples,comparisons=self.comparisons,clean=self.clean)

    def restore(self,metadata,arrays):
        if metadata.get('version',1)==1:
            # Early development checkpoints lacked fluid/mechanical fields.
            # Keep the physical state, require new full-state comparisons.
            self.reference={};self.samples=[];self.comparisons=[];self.clean=0
            return
        if metadata['criteria']!=asdict(self.criteria):raise ValueError('periodic restart criteria changed')
        if set(arrays)!=(set(self.NAMES) if metadata['samples'] else set()):
            raise ValueError('periodic checkpoint reference fields missing')
        self.samples=metadata['samples'];self.comparisons=metadata['comparisons'];self.clean=metadata['clean']
        self.reference=arrays
        if self.clean<0 or len(self.comparisons)!=max(0,len(self.samples)-1):
            raise ValueError('invalid periodic checkpoint history')


def frozen_flow_initial_pc(dt_s,flow_ml_s,parameters):
    """BE periodic fixed point for a COMPLETE supplied flow period.

    Diagnostic/optional fresh initialization only. This does not solve coupled
    periodic AFSEI and must never reset Pc at a paced beat or on resume.
    """
    dt=list(dt_s);flow=list(flow_ml_s)
    if not dt or len(dt)!=len(flow):raise ValueError('one flow per positive time interval required')
    if any(not isfinite(t) or t<=0 for t in dt) or any(not isfinite(q) or q<0 for q in flow):
        raise ValueError('finite forward outlet flow and positive intervals required')
    cap=parameters.compliance_ml_per_mmhg;rp=parameters.rp_mmhg_s_per_ml
    A=1.;B=0.
    for step,q in zip(dt,flow):
        a=cap/(cap+step/rp);A*=a;B=a*B+step*q/(cap+step/rp)
    if 1-A<1e-12:raise ValueError('flow period too short to initialize RCR')
    return dict(pc_mmhg=parameters.venous_mmhg+B/(1-A),period_s=sum(dt),A=A,
                mean_outflow_ml_s=sum(t*q for t,q in zip(dt,flow))/sum(dt),
                coupled_periodic_state=False)

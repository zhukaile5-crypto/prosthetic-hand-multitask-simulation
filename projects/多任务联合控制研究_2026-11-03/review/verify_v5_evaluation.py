"""Independent evaluator witnesses. Pure arrays only, never dynamics.

Pass --evaluation and --protocol only after the producer declares a stable
revision. Default CLI merely describes prepared witnesses, not a green gate.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,runpy,sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fixture():
    t=np.arange(3201)*.005;n=np.ones((len(t),5));n[t>=12.1]=0.;n[np.argmin(abs(t-12.))]=1.1
    dt=.0005;st=np.arange(32000)*dt;duration=np.full(len(st),dt)
    sf=np.column_stack([np.interp(st+duration,t,n[:,i]) for i in range(5)])
    target=np.tile([.1,0.,0.],(len(t),1));position=np.tile([0.,0.,.1],(len(t),1))
    trace=dict(time=t,normal=n.copy(),skin_normal=n.copy(),palm=np.zeros(len(t)),support=np.zeros(len(t)),
        position=position,relative=np.zeros((len(t),3)),relative_velocity=np.zeros((len(t),3)),angular_velocity=np.zeros((len(t),3)),
        lift=np.full(len(t),.1),lift_ref=np.full(len(t),.1),all_hand_object=n.sum(1),controller_active=t>=7.5,
        transport_paused=np.zeros(len(t),bool),warning_count=np.zeros(len(t),int),collision=np.zeros(len(t)),
        actual_motion_target=target,executed_motion_reference=target.copy(),wrist_position=np.zeros((len(t),2)),requested_wrist_reference=np.zeros((len(t),2)),
        substep_time=st,substep_duration=duration,substep_normal=sf.copy(),substep_skin_normal=sf.copy(),substep_palm=np.zeros(len(st)))
    metadata=dict(method='synthetic_evaluator_witness',scenario='nominal',seed=0,research_phase='development',physics_timestep=dt,
        config=dict(timestep=dt),limits=[5.]*5,budget=18.,handoff_passed=True,initial_position=[0.,0.,0.],reference_relative=[0.,0.,0.],
        release_at=12.,mechanical_release_at=12.,preparation_failure=None)
    return trace,metadata

def run_witnesses(api,protocol):
    checks={};details={};trace,metadata=fixture();r=api['evaluate_arrays'](trace,metadata,protocol)
    q=r['entire_holding_quality'];checks['whole_holding_includes_12s_preaction_truth_endpoint']=bool(q['eligible'] and q['window_s']==[7.5,12.] and q['raw_200Hz']['samples']==901 and abs(q['raw_200Hz']['duration_s']-4.5)<1e-10 and abs(q['raw_200Hz']['finger_TV_N']-.5)<1e-9)
    details['whole_holding']=dict(window=q['window_s'],eligible=q['eligible'],raw=q['raw_200Hz'])
    # The zero-error prefix ends at10s; it cannot prove perfect tracking until12.
    t=trace['time'];short={k:v[t<=10.].copy() for k,v in trace.items() if isinstance(v,np.ndarray) and len(v)==len(t)}
    partial=api['motion_tracking'](short,7.5,12.);checks['truncated_fixed_window_unavailable']=partial['available'] is False;details['truncated_tracking']=partial
    keep=np.arange(len(t))!=2000;gap={k:v[keep].copy() for k,v in trace.items() if isinstance(v,np.ndarray) and len(v)==len(t)}
    missing=api['motion_tracking'](gap,7.5,12.);checks['timestamp_gap_unavailable']=missing['available'] is False;details['gap_tracking']=missing
    # Paused commands and matching encoders still violate a nonzero user task.
    wrist=deepcopy(trace);request=(t>=9.3)&(t<10.5);wrist['actual_motion_target'][request,1]=.03;wrist['requested_wrist_reference'][request,0]=.03
    m=api['motion_tracking'](wrist,9.3,11.7);full=api['evaluate_arrays'](wrist,metadata,protocol)
    checks['nominal_reference_cannot_be_replaced_by_executed_reference']=m['available'] and m['rmse'][1]>.02
    checks['new_wrist_guard_does_not_relabel_legacy_mission']=full['mission_success'] is True and full['full_motion_mission_success'] is False and full['wrist_tracking_guard']['failure_at'] is not None
    details['wrist_target_counterexample']=dict(tracking=m,legacy_success=full['mission_success'],full_motion_success=full['full_motion_mission_success'],guard=full['wrist_tracking_guard'])
    # A last instantaneous sample is still part of the16s raw guard.
    terminal=deepcopy(trace);terminal['skin_normal'][-1,0]=6.;end=api['evaluate_arrays'](terminal,metadata,protocol)
    checks['terminal_16s_force_excess_not_censored']=end['safety']['guard_pass'] is False and end['safety']['finger_peak_N'][0]>=6.
    # A linear force has an exact interior timestamp; a half-ms shift is visible.
    linear=deepcopy(trace);linear['normal']=np.tile((t*.1)[:,None],(1,5));endclock=linear['substep_time']+linear['substep_duration'];linear['substep_normal']=np.tile((endclock*.1)[:,None],(1,5));linear['substep_skin_normal']=linear['substep_normal'].copy()
    band,contract=api['force_band'](linear,metadata,protocol);middle=(t>=1.)&(t<=15.)
    checks['substep_force_aligned_to_end_timestamp']=contract['valid'] and np.max(abs(band[middle]-linear['normal'][middle]))<1e-10
    details['force_clock_contract']=contract
    checks={k:bool(v) for k,v in checks.items()}
    return dict(passed=all(checks.values()),checks=checks,details=details,physics_steps_executed=0,
        scope='Real candidate evaluator on independently constructed arithmetic witnesses. No hand, controller, solver or dynamics imported; not a performance or physical-admission certificate.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path);p.add_argument('--protocol',type=Path);a=p.parse_args()
    if not a.evaluation or not a.protocol:
        print(json.dumps(dict(production_verified=False,prepared_witnesses=['truncated fixed window','interior timestamp gap','12s pre-action force edge','nominal target versus paused executed target','legacy versus full wrist mission','16s terminal force guard','substep end clock alignment'],physics_steps_executed=0)));return
    before={str(x.resolve()):sha(x) for x in [a.evaluation,a.protocol]};api=runpy.run_path(str(a.evaluation));protocol=json.loads(a.protocol.read_text(encoding='utf-8'));out=run_witnesses(api,protocol)
    out['source_sha256']=before;out['sources_unchanged']=all(sha(k)==v for k,v in before.items());out['reviewer_sha256']=sha(__file__)
    print(json.dumps(out,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()

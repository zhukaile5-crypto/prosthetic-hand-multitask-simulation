"""Pure synthetic research-design tests; this is never physical acceptance."""
from pathlib import Path
from datetime import datetime,timezone
import copy
import hashlib
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
from evaluation import (read_protocol,admission_errors,evaluate_arrays,force_variation,force_band,motion_tracking,
                        first_dwell,event_recovery)

ROOT=Path(__file__).resolve().parent


def synthetic():
    t=np.arange(3201)*.005;st=np.arange(32000)*.0005
    n=np.full((len(t),5),2.);n[t>=12.5]=0.
    physical=np.full((len(st),5),2.);physical[st>=12.5]=0.
    pos=np.zeros((len(t),3));pos[t>=1.,2]=.08
    lift=np.full(len(t),.1)
    trace=dict(time=t,normal=n,skin_normal=n.copy(),palm=np.zeros(len(t)),support=np.zeros(len(t)),
        position=pos,relative=np.tile([0.,0.,.08],(len(t),1)),relative_velocity=np.zeros((len(t),3)),
        angular_velocity=np.zeros((len(t),3)),lift=lift,lift_ref=lift.copy(),all_hand_object=n.sum(axis=1),
        controller_active=t>=7.5,transport_paused=np.zeros(len(t),bool),warning_count=np.zeros(len(t)),
        collision=np.zeros(len(t)),actual_motion_target=np.c_[lift,np.zeros((len(t),2))],
        executed_motion_reference=np.c_[lift,np.zeros((len(t),2))],
        wrist_position=np.zeros((len(t),2)),requested_wrist_reference=np.zeros((len(t),2)),
        substep_time=st,substep_duration=np.full(len(st),.0005),substep_normal=physical,
        substep_skin_normal=physical.copy(),substep_palm=np.zeros(len(st)))
    metadata=dict(method='ratio',research_phase='development',scenario='nominal',seed=610101,
        physics_timestep=.0005,limits=[5.]*5,budget=18.,initial_position=[0.,0.,0.],
        reference_relative=[0.,0.,.08],handoff_passed=True,release_at=12.,mechanical_release_at=12.,
        config=dict(mass=.12,friction=.6,size=1.,solref=.008))
    return trace,metadata


def main():
    p=read_protocol();checks={}
    checks['draft_rejects_confirmation']=bool(admission_errors(p,'confirmation'))
    checks['draft_allows_development']=not admission_errors(p,'development')
    t,m=synthetic();result=evaluate_arrays(t,m,p)
    checks['complete_clean_synthetic_mission']=result['record_complete'] and result['mission_success'] and result['safety']['guard_pass']
    checks['new_initial_stability_gate']=abs(result['initial_stability_wait_s']-.2)<1e-9
    checks['physical_release_and_task_release_clean']=result['release']['physical_success'] and result['release']['task_qualified_success'] and abs(result['release']['physical_clearance_latency_s']-.7)<1e-9
    checks['static_and_dynamic_fixed_window_coverage']=result['static_quality']['eligible'] and result['dynamic_quality']['eligible']
    checks['actual_wrist_encoder_contract']=result['progress']['wrist_tracking_observed'] and result['progress']['wrist_actual_tracking_max_rad']==[0.,0.]
    checks['full_motion_clean_and_tracking_duration']=result['full_motion_mission_success'] and abs(result['progress']['holding_motion_tracking']['duration_s']-4.5)<1e-10
    bad_wrist=copy.deepcopy(t);bad_wrist['wrist_position'][(t['time']>=9.3)&(t['time']<10.),0]=.03
    wrist_failed=evaluate_arrays(bad_wrist,m,p)
    checks['wrist_guard_does_not_rewrite_legacy_mission']=wrist_failed['mission_success'] and not wrist_failed['full_motion_mission_success'] and wrist_failed['wrist_tracking_guard']['failure_at'] is not None
    checks['actual_wrist_continuous_error_positive']=wrist_failed['progress']['dynamic_motion_tracking']['rmse'][1]>0
    missing_wrist=copy.deepcopy(t);missing_wrist.pop('wrist_position')
    checks['missing_wrist_not_perfect_motion']=not evaluate_arrays(missing_wrist,m,p)['full_motion_mission_success']
    checks['new_force_filter_uses_substep_end_clock']=result['filter_contract']['physical_force_clock']=='substep_end_with_initial_sample'
    ramp_clock=copy.deepcopy(t)
    ramp_clock['normal']=np.repeat(t['time'][:,None],5,axis=1)
    ramp_clock['substep_normal']=np.repeat((t['substep_time']+t['substep_duration'])[:,None],5,axis=1)
    rb,unused_contract=force_band(ramp_clock,m,p);interior=(t['time']>.1)&(t['time']<15.9)&(abs(t['time']-7.5)>.1)&(abs(t['time']-12.)>.1)
    checks['physical_linear_ramp_has_no_half_millisecond_phase_bias']=bool(np.max(abs(rb[interior]-ramp_clock['normal'][interior]))<1e-10)
    late_release=copy.deepcopy(t)
    late_release['substep_normal'][t['substep_time']>=12.,0]=100.
    a,_=force_band(t,m,p);b,bc=force_band(late_release,m,p)
    holding_mask=(t['time']>=7.5-1e-9)&(t['time']<=12.+1e-9)
    checks['future_release_cannot_change_any_holding_filter_sample']=bool(np.array_equal(a[holding_mask],b[holding_mask]))
    checks['phase_filter_keeps_release_endpoint_and_version']=bool(bc['version']==3 and np.isfinite(b[holding_mask]).all() and holding_mask.sum()==901)
    early_release=copy.deepcopy(t)
    early_release['substep_normal'][t['substep_time']>=8.,0]=100.
    early_metadata=dict(m,release_at=8.)
    a,_=force_band(t,early_metadata,p);b,bc=force_band(early_release,early_metadata,p)
    before_early_release=(t['time']>=7.5-1e-9)&(t['time']<=8.+1e-9)
    checks['actual_early_release_boundary_prevents_future_leak']=bool(np.array_equal(a[before_early_release],b[before_early_release]) and bc['phase_boundaries_s']==[0.,7.5,8.,16.])
    checks['whole_holding_window_retains_initial_redistribution']=result['entire_holding_quality']['eligible']
    checks['whole_holding_includes_pre_release_endpoint']=abs(result['entire_holding_quality']['bandlimited']['duration_s']-4.5)<1e-9
    cut={k:v[:2001] for k,v in t.items() if k in ('time','lift','wrist_position','actual_motion_target')}
    censored=motion_tracking(cut,7.5,12.)
    checks['missing_tracking_tail_not_perfect_score']=not censored['available'] and censored['censored'] and censored['rmse'] is None and censored['coverage_fraction']<1.
    gap={k:np.delete(t[k],1950,axis=0) for k in ('time','lift','wrist_position','actual_motion_target')}
    checks['tracking_time_gap_not_complete']=not motion_tracking(gap,7.5,12.)['available']
    dynamic=copy.deepcopy(t);dynamic['lift_ref'][(t['time']>=9.3)&(t['time']<12.)]=.084
    failed=evaluate_arrays(dynamic,m,p)
    checks['transport_failure_distinguished_from_drop']=not failed['mission_success'] and failed['failure_reason']=='transport_reference_unmet'
    checks['transport_failure_retains_physical_release']=failed['release']['physical_success'] and failed['release']['physical_exposure'] and not failed['release']['task_qualified_success']
    checks['force_window_not_selected_on_mission_success']=failed['dynamic_quality']['eligible']
    no_policy=copy.deepcopy(t);no_policy['controller_active'][:]=False
    unadmitted=evaluate_arrays(no_policy,dict(m,handoff_passed=False),p)
    checks['no_handoff_never_wins_initial_wait']=unadmitted['initial_stability_wait_s']==1.4 and not unadmitted['static_quality']['eligible'] and not unadmitted['dynamic_quality']['eligible']
    peak=copy.deepcopy(t);peak['substep_skin_normal'][18001,0]=7.
    unsafe=evaluate_arrays(peak,m,p)
    checks['substep_peak_not_hidden_by_filter']=not unsafe['safety']['guard_pass'] and unsafe['safety']['local_excess_integral_Ns']>0
    terminal=copy.deepcopy(t);terminal['skin_normal'][-1,0]=5.03
    checks['terminal_endpoint_guard']=not evaluate_arrays(terminal,m,p)['safety']['guard_pass']
    partial=copy.deepcopy(t);partial['substep_duration']=partial['substep_duration'][:-1]
    checks['missing_raw_coverage_never_complete']=not evaluate_arrays(partial,m,p)['record_complete']
    # A two-finger redistribution can have nonzero finger TV and zero total TV.
    wave=np.zeros((3,5));wave[:,0]=[0.,1.,0.];wave[:,1]=[1.,0.,1.]
    var=force_variation(np.array([0.,.5,1.]),wave)
    checks['finger_vs_total_force_TV_distinct']=var['finger_TV_N']==4. and var['total_force_TV_N']==0.
    checks['path_decomposition_exact']=var['finger_net_drift_N']==0. and var['finger_reversal_N']==2. and var['path_identity_residual_N']<1e-12
    ramp=np.zeros((3,5));ramp[:,0]=[0.,.5,1.]
    var=force_variation(np.array([0.,.5,1.]),ramp)
    checks['monotone_trend_not_mislabeled_reversal']=var['finger_TV_N']==1. and var['finger_net_drift_N']==1. and var['finger_reversal_N']==0.
    clock=t['time'];good=np.ones(len(clock),bool)
    event=dict(id='test',start=9.3,end=9.6)
    r=event_recovery(clock,good,[event],t,None,True,12.)[0]
    checks['never_unstable_null_not_zero']=r['status']=='never_unstable' and r['latency_from_instability_s'] is None
    late=good.copy();late[(clock>=9.3-1e-9)&(clock<11.8-1e-9)]=False
    r=event_recovery(clock,late,[event],t,None,True,12.)[0]
    checks['release_boundary_exclusive']=r['status']=='release_preempt'
    collision=copy.deepcopy(t);collision['collision'][(clock>=9.8)&(clock<10.)]=.1
    early=good.copy();early[(clock>=9.3)&(clock<9.35)]=False
    r=event_recovery(clock,early,[dict(event,requires_actual_contact=True)],collision,None,True,12.)[0]
    checks['collision_uses_actual_contact_onset']=r['analysis_start']==9.8 and r['status']=='never_unstable'
    rows=event_recovery(clock,good,[event,dict(id='second',start=10.3,end=10.6)],t,None,False,12.)
    checks['invalid_record_keeps_every_event']=len(rows)==2 and all(r['status']=='invalid_record' for r in rows)
    checks['old_256_explicitly_seen_diagnostic']=p['disclosure']['old_256_confirmation_seen'] and p['confirmation_gate']['old_256_may_not_enter_confirmation']
    checks['no_new_physical_domain_admitted']='not admitted' in p['scope']['new_size_pose_solref_domains']
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    certificate=dict(passed=all(checks.values()),checks=checks,check_count=len(checks),
        failed=[k for k,v in checks.items() if not v],created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(ROOT/'protocol.json'),evaluation_sha256=digest(ROOT/'evaluation.py'),
        validator_sha256=digest(Path(__file__)),
        scope='Synthetic arithmetic/interface/design checks only. No MuJoCo, original module, controller, simulation or confirmation performance was imported/run. Not physical or policy acceptance.')
    folder=ROOT/'review';folder.mkdir(exist_ok=True)
    out=folder/('design_validation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.write_text(json.dumps(certificate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=certificate['passed'],checks=len(checks),failed=certificate['failed'],path=str(out)),ensure_ascii=False))
    if not certificate['passed']:raise SystemExit(1)


if __name__=='__main__':main()

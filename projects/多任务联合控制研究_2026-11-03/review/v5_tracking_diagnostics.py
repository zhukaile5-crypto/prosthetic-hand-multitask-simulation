"""Independent continuous tracking diagnostics; never runs a simulator.

Keeps nominal-user target distinct from allocated/executed references. Existing
results are not edited. Missing data do not become zero error or a short run.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.dont_write_bytecode=True

def first_persistent(time,violation,lo,hi,dwell=.3):
    since=None;first=None
    for i in np.flatnonzero((time>=lo-1e-9)&(time<hi-1e-9)):
        if violation[i]:
            if since is None:since=float(time[i])
            if time[i]-since>=dwell-1e-9:first=float(time[i]);break
        else:since=None
    return dict(first_confirmed_at=first,ongoing_unconfirmed_at_cutoff=first is None and since is not None,
                ongoing_since=since if first is None else None,cutoff_s=hi,
                scope='Exclusive release/task cutoff; no missing sample or censored tail is counted as successful recovery')

def nominal_references(trace):
    if 'task_request_json' in trace:
        requests=[json.loads(str(x)) for x in trace['task_request_json']]
        return np.asarray([x['motion_target'] for x in requests],float),'public task_request_json.motion_target'
    if 'actual_motion_target' in trace:return np.asarray(trace['actual_motion_target'],float),'legacy logged nominal actual_motion_target'
    return np.column_stack([trace['lift_ref'],trace['requested_wrist_reference']]),'legacy nominal lift_ref/requested_wrist_reference'

def summarize_tracking(trace,window=(7.5,12.),tolerances=(.01,.015,.015),dwell=.3):
    time=np.asarray(trace['time'],float);nominal,source=nominal_references(trace)
    actual=np.column_stack([trace['lift'],trace['wrist_position']]);lo,hi=window
    selected=(time>=lo-1e-9)&(time<=hi+1e-9);idx=np.flatnonzero(selected);t=time[selected]
    finite=np.isfinite(actual).all(1)&np.isfinite(nominal).all(1)
    complete=bool(len(t)==round((hi-lo)/.005)+1 and len(t)>1 and abs(t[0]-lo)<1e-8 and abs(t[-1]-hi)<1e-8 and np.allclose(np.diff(t),.005,atol=1e-8,rtol=0) and finite[selected].all())
    result=dict(window_s=list(window),reference_source=source,complete_window=complete,samples=int(selected.sum()),
        finite_fraction=float(finite[selected].mean()) if selected.any() else None,nominal_tracking_rmse=None,
        nominal_tracking_time_weighted_rmse=None,nominal_tracking_IAE=None,nominal_tracking_maximum_error=None,
        units=['m','rad','rad'],sample_endpoint='Both endpoints establish coverage; left intervals define primary RMSE/IAE and exclude release-time retargeting')
    if not complete:
        result['censor_reason']='Missing, nonfinite or incomplete nominal/encoder window; no zero fill and no success-only cropping'
        result['guard_confirmations']=None;return result
    e=actual-nominal;window_e=e[selected];seconds=hi-lo
    weights=np.diff(t);left=window_e[:-1]
    result.update(nominal_tracking_rmse=np.sqrt(np.sum(left**2*weights[:,None],axis=0)/seconds).tolist(),
        sampled_RMSE_including_endpoint_sensitivity=np.sqrt(np.mean(window_e**2,axis=0)).tolist(),
        nominal_tracking_time_weighted_rmse=np.sqrt(np.trapezoid(window_e**2,t,axis=0)/seconds).tolist(),
        nominal_tracking_IAE=np.sum(abs(left)*weights[:,None],axis=0).tolist(),
        nominal_tracking_maximum_error=abs(left).max(axis=0).tolist())
    tol=np.asarray(tolerances,float);flags=abs(e)>tol
    result['guard_confirmations']={name:first_persistent(time,flags[:,i],lo,hi,dwell) for i,name in enumerate(['lift','WRJ2','WRJ1'])}
    result['guard_confirmations']['either_wrist']=first_persistent(time,flags[:,1:].any(1),lo,hi,dwell)
    result['guard_contract']=dict(tolerances=tol.tolist(),dwell_s=dwell,classification='Engineering diagnostic; does not overwrite legacy task_success')
    if 'executed_motion_reference' in trace:
        executed=np.asarray(trace['executed_motion_reference'],float)
        if executed.shape==nominal.shape and np.isfinite(executed[selected]).all():
            result['reference_allocation_rmse']=np.sqrt(np.mean((executed[selected]-nominal[selected])**2,axis=0)).tolist()
            result['encoder_servo_rmse']=np.sqrt(np.mean((actual[selected]-executed[selected])**2,axis=0)).tolist()
    return result

def variation_vectors(time,force):
    time=np.asarray(time,float);force=np.asarray(force,float)
    if len(time)<2 or force.shape!=(len(time),5) or not np.isfinite(force).all():return None
    tv=abs(np.diff(force,axis=0)).sum(0);net=force[-1]-force[0];retrace=.5*(tv-abs(net))
    return dict(per_finger_TV=tv.tolist(),per_finger_net_signed=net.tolist(),per_finger_net_absolute=abs(net).tolist(),
        per_finger_retrace=retrace.tolist(),TV_sum=float(tv.sum()),duration_s=float(time[-1]-time[0]),
        identity_maximum_error=float(abs(tv-abs(net)-2*retrace).max()),
        interpretation='Retrace is a path property; purposeful contact adaptation and transient unload/recovery are not automatically self-excited oscillation')

def self_test():
    t=np.arange(3201)*.005;target=np.tile([.1,.03,-.02],(len(t),1));trace=dict(time=t,lift=target[:,0].copy(),wrist_position=target[:,1:].copy(),actual_motion_target=target.copy(),executed_motion_reference=target.copy())
    checks={};r=summarize_tracking(trace)
    checks['zero_tracking_exact']=r['complete_window'] and np.array_equal(r['nominal_tracking_rmse'],[0.,0.,0.])
    bad={k:v.copy() for k,v in trace.items()};bad['wrist_position'][:,0]=0.;bad['executed_motion_reference'][:,1]=0.;r=summarize_tracking(bad)
    checks['executed_reference_cannot_hide_nominal_task_pause']=abs(r['nominal_tracking_rmse'][1]-.03)<1e-12 and r['encoder_servo_rmse'][1]==0.
    checks['continuous_wrist_guard_is_distinct_from_legacy_lift_guard']=r['guard_confirmations']['either_wrist']['first_confirmed_at'] is not None and r['guard_confirmations']['lift']['first_confirmed_at'] is None
    brief={k:v.copy() for k,v in trace.items()};brief['wrist_position'][(t>=8.)&(t<8.3),0]+=.016;r=summarize_tracking(brief)
    checks['subdwell_transient_not_failure']=r['guard_confirmations']['either_wrist']['first_confirmed_at'] is None
    cutoff={k:v.copy() for k,v in trace.items()};cutoff['wrist_position'][(t>=11.7)&(t<12.),0]+=.016;r=summarize_tracking(cutoff)
    checks['exact_release_cutoff_censors_unconfirmed_dwell']=r['guard_confirmations']['either_wrist']['first_confirmed_at'] is None and r['guard_confirmations']['either_wrist']['ongoing_unconfirmed_at_cutoff']
    missing={k:v.copy() for k,v in trace.items()};missing['wrist_position'][2000,0]=np.nan;r=summarize_tracking(missing)
    checks['missing_packet_does_not_become_zero_error']=not r['complete_window'] and r['nominal_tracking_rmse'] is None and r['guard_confirmations'] is None
    truncated={k:v[t<=10.].copy() for k,v in trace.items()};r=summarize_tracking(truncated)
    checks['truncated_fixed_window_is_censored']=not r['complete_window'] and r['nominal_tracking_rmse'] is None
    keep=np.arange(len(t))!=2000;gap={k:v[keep].copy() for k,v in trace.items()};r=summarize_tracking(gap)
    checks['missing_timestamp_gap_is_censored']=not r['complete_window'] and r['nominal_tracking_rmse'] is None
    trend=np.arange(6)[:,None]*np.ones((1,5));v=variation_vectors(np.arange(6),trend);checks['pure_trend_has_no_retrace']=np.array_equal(v['per_finger_retrace'],[0.]*5)
    back=np.array([0.,1.,0.])[:,None]*np.ones((1,5));v=variation_vectors(np.arange(3),back);checks['closed_path_retrace_preserved']=np.array_equal(v['per_finger_retrace'],[1.]*5)
    edge=np.zeros((3,5));edge[-1]=1.;v=variation_vectors(np.array([11.99,11.995,12.]),edge)
    checks['preaction_release_endpoint_last_difference_not_omitted']=v['TV_sum']==5. and variation_vectors(np.array([11.99,11.995]),edge[:-1])['TV_sum']==0.
    return dict(checker_self_tests_passed=all(checks.values()),checks=checks,production_verified=False,physics_steps_executed=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path);a=p.parse_args();out=self_test()
    if a.trace:
        keys=['time','lift','wrist_position','task_request_json','actual_motion_target','lift_ref','requested_wrist_reference','executed_motion_reference']
        with np.load(a.trace,allow_pickle=False) as z:trace={k:z[k] for k in keys if k in z.files}
        out['diagnostics']=[summarize_tracking(trace,w) for w in [(7.5,12.),(9.3,11.7)]]
    print(json.dumps(out,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()

"""Independent research metrics on saved truth; never runs a controller/plant.

The old canonical evaluator is preserved. This separate vector of outcomes
separates mission reliability, raw safety, physical release and force quality.
Draft protocols allow development only; confirmation requires explicit freeze.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT/'protocol.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value,np.ndarray):return jsonable(value.tolist())
    if isinstance(value,np.generic):return jsonable(value.item())
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def read_protocol(path=PROTOCOL):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def admission_errors(protocol,phase,metadata=None):
    if phase!='confirmation':return []
    p=protocol;c=p['splits']['confirmation'];gate=p['confirmation_gate'];errors=[]
    if p.get('status')!='frozen':errors.append('protocol_not_frozen')
    if c.get('status')!='frozen' or not c.get('physical_seeds') or not c.get('scenarios') or not c.get('methods'):
        errors.append('confirmation_matrix_not_frozen')
    if p['statistics'].get('status')!='frozen' or p['statistics'].get('noninferiority_margins') is None:
        errors.append('statistical_contract_not_frozen')
    if not gate.get('physical_validation'):errors.append('physical_validation_missing')
    sources=gate.get('frozen_source_sha256',{})
    if not sources:errors.append('source_freeze_missing')
    elif any(not (ROOT/name).is_file() or sha(ROOT/name)!=expected for name,expected in sources.items()):
        errors.append('source_identity_mismatch')
    if metadata:
        seed=metadata.get('physical_seed',metadata.get('seed'))
        if seed not in c.get('physical_seeds',[]):errors.append('seed_not_in_holdout')
        if metadata.get('research_method',metadata.get('method')) not in c.get('methods',[]):errors.append('method_not_in_holdout')
        if metadata.get('research_phase')!='confirmation':errors.append('missing_explicit_research_confirmation_phase')
    return errors


def first_dwell(t,flag,start,end,duration,exclusive=False):
    t=np.asarray(t);flag=np.asarray(flag,bool);since=None
    selected=(t>=start-1e-9)&(t<end-1e-9 if exclusive else t<=end+1e-9)
    for i in np.flatnonzero(selected):
        if flag[i]:
            if since is None:since=float(t[i])
            if t[i]-since>=duration-1e-9:return float(t[i])
        else:since=None
    return None


def force_variation(t,force):
    """Exact path decomposition; reversal is retracing, not a noise label."""
    t=np.asarray(t,float);force=np.asarray(force,float)
    if len(t)<2 or force.shape!=(len(t),5) or not np.isfinite(force).all():return None
    seconds=float(t[-1]-t[0])
    if seconds<=0:return None
    per_tv=np.abs(np.diff(force,axis=0)).sum(axis=0)
    per_drift=np.abs(force[-1]-force[0])
    reversal=np.maximum(0.,.5*(per_tv-per_drift))
    total=force.sum(axis=1);total_tv=float(np.abs(np.diff(total)).sum())
    total_drift=float(abs(total[-1]-total[0]))
    return dict(duration_s=seconds,samples=len(t),finger_TV_N=float(per_tv.sum()),
        finger_net_drift_N=float(per_drift.sum()),finger_reversal_N=float(reversal.sum()),
        finger_TV_rate_N_s=float(per_tv.sum()/seconds),finger_net_drift_rate_N_s=float(per_drift.sum()/seconds),
        finger_reversal_rate_N_s=float(reversal.sum()/seconds),total_force_TV_N=total_tv,
        total_force_net_drift_N=total_drift,total_force_reversal_N=max(0.,.5*(total_tv-total_drift)),
        total_force_TV_rate_N_s=total_tv/seconds,finger_SD_mean_N=float(force.std(axis=0).mean()),
        total_force_SD_N=float(total.std()),
        path_identity_residual_N=float(abs(per_tv.sum()-per_drift.sum()-2.*reversal.sum())),
        interpretation='Reversal includes purposeful nonmonotonic task response; it is not automatically unwanted oscillation.')


def force_band(trace,metadata,protocol):
    t=trace['time'];st=trace.get('substep_time',np.array([]));dur=trace.get('substep_duration',np.array([]))
    force=trace.get('substep_normal',np.empty((0,5)));skin=trace.get('substep_skin_normal',np.empty((0,5)))
    duration=protocol['schedule']['duration_s'];expected_dt=float(metadata.get('physics_timestep',metadata.get('config',{}).get('timestep',0.)))
    complete=bool(len(st)>0 and expected_dt>0 and dur.shape==st.shape and force.shape==skin.shape==(len(st),5) and
        np.isfinite(force).all() and np.isfinite(skin).all() and np.allclose(dur,expected_dt,atol=1e-9,rtol=0) and
        abs(st[0])<1e-8 and np.allclose(st,np.arange(len(st))*expected_dt,atol=1e-7,rtol=0) and abs(dur.sum()-duration)<1e-8)
    if not complete:return None,dict(full_substep_coverage=False,valid=False)
    half=max(1,int(round(.05/(2*expected_dt))));kt=np.arange(-half,half+1)*expected_dt
    kernel=2.*50.*expected_dt*np.sinc(2.*50.*kt)*np.hanning(2*half+1);kernel/=kernel.sum()
    # Each substep record is the force AFTER integration over [st,st+dur].
    # V1 aligned it to st. Preserve V1 archives, but use the actual end clock
    # for all new V2 records, with the measured initial force at time zero.
    actual=np.vstack([trace['normal'][0],force])
    grid=np.r_[0.,st+dur]
    # V3 filters each public lifecycle stage independently. A symmetric filter
    # over the whole episode leaked release forces into the last 25ms of hold
    # in V2. Retain every holding sample, including its pre-action endpoint;
    # reflection uses only that stage. Static/dynamic interior windows keep the
    # same cutoff/kernel and their V2 values. Raw safety is always unfiltered.
    handoff=float(protocol['schedule']['handoff_s'])
    requested_release=metadata.get('release_at')
    release=float(protocol['schedule']['release_s'] if requested_release is None else requested_release)
    if not np.isfinite(release) or release<0. or release>duration:
        return None,dict(full_substep_coverage=True,valid=False,reason='invalid_public_release_clock')
    boundaries=sorted(set((0.,handoff,release,float(duration))))
    band=np.full((len(t),5),np.nan)
    for lo,hi in zip(boundaries,boundaries[1:]):
        physical=(grid>=lo-1e-9)&(grid<=hi+1e-9)
        stage_grid=grid[physical];stage_force=actual[physical]
        if len(stage_grid)<2 or abs(stage_grid[0]-lo)>1e-7 or abs(stage_grid[-1]-hi)>1e-7:
            return None,dict(full_substep_coverage=True,valid=False,reason='phase_boundary_not_observed')
        samples=(t>=lo-1e-9)&(t<=hi+1e-9)
        if lo==release and lo>0.:samples&=t>lo+1e-9 # release force belongs to pre-action hold
        for digit in range(5):
            filtered=np.convolve(np.pad(stage_force[:,digit],(half,half),mode='reflect'),kernel,mode='valid')
            band[samples,digit]=np.interp(t[samples],stage_grid,filtered)
    return band,dict(valid=True,full_substep_coverage=True,cutoff_Hz=50.,span_s=.05,
        physical_interval_s=expected_dt,filtered_before_window=True,raw_safety_filtered=False,
        physical_force_clock='substep_end_with_initial_sample',version=3,
        phase_boundaries_s=boundaries,
        boundary_filter='reflect within public lifecycle stage; no future release force in holding',
        holding_endpoint='pre-action force at actual public release instant retained')


def motion_tracking(trace,lo,hi):
    """Fixed requested-vs-actual robot tracking, independent of task success.

    Left-sample integration gives every stored control interval its duration;
    the release-time target is excluded from the preceding transport stage.
    Missing encoders/references are unavailable, never a perfect zero error.
    """
    t=np.asarray(trace['time'],float)
    fields=('actual_motion_target','wrist_position','lift')
    empty=dict(available=False,window_s=[lo,hi],duration_s=0.,expected_duration_s=max(0.,hi-lo),
        complete_window=False,censored=True,coverage_fraction=0.,rmse=None,
        max_abs_error=None,absolute_error_integral=None,
        axes=['lift_m','WRJ2_rad','WRJ1_rad'])
    if hi<=lo or any(k not in trace for k in fields):return empty
    if np.asarray(trace['lift']).shape!=(len(t),) or np.asarray(trace['wrist_position']).shape!=(len(t),2):return empty
    actual=np.column_stack((trace['lift'],trace['wrist_position']))
    target=np.asarray(trace['actual_motion_target'],float)
    if actual.shape!=(len(t),3) or target.shape!=actual.shape or not np.isfinite(np.r_[actual.ravel(),target.ravel()]).all():return empty
    duration=np.maximum(0.,np.minimum(np.r_[t[1:],t[-1]],hi)-np.maximum(t,lo))
    seconds=float(duration.sum())
    continuous=bool(len(t)>1 and np.isfinite(t).all() and np.all(np.diff(t)>0.)
        and np.allclose(np.diff(t),np.median(np.diff(t)),rtol=0.,atol=1e-8))
    complete=bool(continuous and t[0]<=lo+1e-9 and t[-1]>=hi-1e-9 and abs(seconds-(hi-lo))<1e-8)
    empty.update(duration_s=seconds,coverage_fraction=min(1.,seconds/(hi-lo)))
    if seconds<=0 or not complete:return empty
    err=actual-target;mask=duration>0
    return dict(available=True,window_s=[lo,hi],duration_s=seconds,expected_duration_s=hi-lo,
        complete_window=True,censored=False,coverage_fraction=1.,samples=int(mask.sum()),
        axes=empty['axes'],rmse=np.sqrt(np.sum(err**2*duration[:,None],axis=0)/seconds).tolist(),
        max_abs_error=np.max(abs(err[mask]),axis=0).tolist(),
        absolute_error_integral=np.sum(abs(err)*duration[:,None],axis=0).tolist(),
        integration='unfiltered actual robot encoders; left sample over fixed user-task intervals',
        selection='not conditioned on mission success, force guard or release success')


def quality_window(trace,band,support,raised,policy_active,complete,lo,hi):
    t=trace['time'];mask=(t>=lo-1e-9)&(t<=hi+1e-9)
    dt=float(np.median(np.diff(t))) if len(t)>1 else .005
    coverage=float(min(1.,(t[mask][-1]-t[mask][0])/(hi-lo))) if mask.sum()>1 else 0.
    valid=bool(complete and band is not None and mask.sum()==round((hi-lo)/dt)+1 and coverage>=1.-1e-8 and
        np.all(support[mask]&raised[mask]&(trace['support'][mask]<=.05)&policy_active[mask]))
    return dict(window_s=[lo,hi],eligible=valid,coverage=coverage,
        physical_support_fraction=float(support[mask].mean()) if mask.any() else None,
        policy_exposure_fraction=float(policy_active[mask].mean()) if mask.any() else None,
        mean_actual_finger_force_N=float(trace['normal'][mask].sum(axis=1).mean()) if valid else None,
        bandlimited=force_variation(t[mask],band[mask]) if valid else None,
        raw_200Hz=force_variation(t[mask],trace['normal'][mask]) if valid else None,
        cohort='Complete physical support and actual admitted policy in fixed window; no whole-task-success or requested-motion-success selection.')


def event_recovery(t,stable,events,trace,failure_at,complete,release_at):
    rows=[];release=12. if release_at is None else float(release_at)
    for j,spec in enumerate(events):
        command=float(spec['start']);end=min(events[j+1]['start'] if j+1<len(events) else release,release)
        reason='next_event' if j+1<len(events) and events[j+1]['start']<release else 'release'
        actual=None;exposed=True
        if spec.get('requires_actual_contact'):
            cutoff=min(end,failure_at) if failure_at is not None else end
            hit=np.flatnonzero((t>=command-1e-9)&(t<cutoff-1e-9)&(np.asarray(trace.get('collision',np.zeros(len(t))))>.02))
            actual=float(t[hit[0]]) if len(hit) else None;exposed=actual is not None
        start=max(command,actual) if actual is not None else command
        if failure_at is not None and start<failure_at<end:end=failure_at;reason='task_failure'
        r={**spec,'analysis_start':start,'observation_end':end,'cutoff_reason':reason,'actual_exposure_at':actual,
            'exposed':bool(exposed),'stable_before':False,'first_instability':None,'stable_confirmation':None,
            'latency_from_event_s':None,'latency_from_instability_s':None,'recovered':False,'censored':False}
        if not complete:r['status']='invalid_record'
        elif failure_at is not None and failure_at<=start+1e-9:r['status']='failed_before'
        elif not exposed:r['status']='not_exposed'
        else:
            before=(t>=start-.2-1e-9)&(t<start-1e-9)
            r['stable_before']=bool(before.sum()==40 and stable[before].all())
            if not r['stable_before']:r['status']='not_stable_before'
            else:
                bad=np.flatnonzero((t>=start-1e-9)&(t<end-1e-9)&~stable)
                if not len(bad):r['status']='never_unstable'
                else:
                    onset=float(t[bad[0]]);r['first_instability']=onset
                    done=first_dwell(t,stable,onset,end,.2,exclusive=True)
                    if done is None:r.update(status='release_preempt' if reason=='release' else 'unrecovered',censored=True)
                    else:r.update(status='recovered_during' if spec['end'] is None or done<spec['end']-1e-9 else 'recovered_after',
                        recovered=True,stable_confirmation=done,latency_from_event_s=done-command,latency_from_instability_s=done-onset)
        rows.append(r)
    return rows


def evaluate_arrays(trace,metadata,protocol=None):
    p=read_protocol() if protocol is None else protocol;th=p['thresholds'];schedule=p['schedule'];m=metadata
    phase=m.get('research_phase',m.get('split','development'))
    errors=admission_errors(p,phase,m)
    if errors:raise RuntimeError('Confirmation blocked: '+', '.join(errors))
    t=np.asarray(trace['time'],float);normal=np.asarray(trace['normal'],float);skin=np.asarray(trace['skin_normal'],float)
    dt=schedule['sensor_period_s'];handoff=schedule['handoff_s'];duration=schedule['duration_s']
    truth_fields=('normal','skin_normal','position','relative','relative_velocity','angular_velocity','lift','lift_ref','support')
    full_truth=bool(len(t)==round(duration/dt)+1 and np.allclose(t,np.arange(len(t))*dt,atol=1e-8,rtol=0) and
        all(np.isfinite(trace[k]).all() for k in truth_fields))
    band,filter_contract=force_band(trace,m,p);complete=full_truth and filter_contract['valid']
    contact=normal>th['finger_contact_min_N'];support=contact[:,0]&(contact.sum(axis=1)>=th['minimum_contacts_including_thumb'])
    initial=np.asarray(m.get('initial_position',trace['position'][0]));raised=trace['position'][:,2]>=initial[2]+th['minimum_object_lift_m']
    lift_error=abs(trace['lift']-trace['lift_ref']);motion=(lift_error<=th['lift_tracking_tolerance_m'])&(trace['support']<=th['table_support_max_N'])
    relative=m.get('reference_relative');relative_error=np.full(len(t),np.inf) if relative is None else np.linalg.norm(trace['relative']-relative,axis=1)
    limits=np.asarray(m['limits'],float);budget=float(m['budget']);tol=th['force_numerical_tolerance_N']
    safe=(skin<=limits+tol).all(axis=1)&(skin.sum(axis=1)<=budget+tol)
    stable=support&raised&motion&safe&(relative_error<=th['stable_relative_drift_m'])&(np.linalg.norm(trace['relative_velocity'],axis=1)<=th['stable_relative_speed_m_s'])&(np.linalg.norm(trace['angular_velocity'],axis=1)<=th['stable_angular_speed_rad_s'])
    policy=np.asarray(trace.get('controller_active',np.full(len(t),bool(m.get('handoff_passed',False))))).astype(bool)
    admitted=bool(m.get('handoff_passed',False));release_at=m.get('release_at');hold_end=schedule['release_s'] if release_at is None else float(release_at)
    established=first_dwell(t,support&raised&motion,0.,handoff,.1)
    failure=m.get('preparation_failure');failure_at=None
    if not admitted or established is None:failure=failure or 'preparation_or_handoff_failed';failure_at=handoff
    else:
        lost=~support|~raised|(relative_error>th['retention_relative_drift_m'])
        drop=first_dwell(t,lost,handoff,hold_end,th['retention_failure_dwell_s'],exclusive=True)
        tracking=first_dwell(t,lift_error>th['lift_tracking_tolerance_m'],handoff,hold_end,th['transport_failure_dwell_s'],exclusive=True)
        if drop is not None and (tracking is None or drop<=tracking):failure='support_or_object_lost';failure_at=drop
        elif tracking is not None:failure='transport_reference_unmet';failure_at=tracking
    if not complete:failure=failure or 'incomplete_truth_or_physical_force'
    first_stable=first_dwell(t,stable,handoff,handoff+p['endpoints']['initial_stability']['restricted_horizon_s'],th['stable_dwell_s']) if admitted else None
    physical_exposure=False;clearance=None
    if release_at is not None and complete:
        i=int(np.argmin(abs(t-release_at)))
        physical_exposure=bool(support[i] and raised[i] and trace['support'][i]<=th['table_support_max_N'] and trace['all_hand_object'][i]>th['all_hand_clearance_N'])
        clearance=first_dwell(t,trace['all_hand_object']<=th['all_hand_clearance_N'],float(release_at),min(duration,float(release_at)+th['release_deadline_s']),th['release_clearance_dwell_s'])
    physical_released=bool(physical_exposure and clearance is not None)
    task_release_qualified=bool(physical_exposure and admitted and (failure_at is None or failure_at>hold_end))
    task_released=bool(physical_released and task_release_qualified)
    warnings=np.asarray(trace.get('warning_count',np.zeros(len(t))))
    mission=bool(complete and admitted and failure is None and task_released and warnings.max()==0)
    if not physical_released and failure is None:failure='physical_release_failed_or_not_exposed'
    safety=None;st=np.asarray(trace.get('substep_time',[]));dur=np.asarray(trace.get('substep_duration',[]))
    if filter_contract['full_substep_coverage']:
        f=np.asarray(trace['substep_skin_normal']);local=np.maximum(f-limits,0.);total=np.maximum(f.sum(axis=1)-budget,0.)
        safety=dict(guard_pass=bool(complete and (f<=limits+tol).all() and (f.sum(axis=1)<=budget+tol).all() and safe[-1]),
            finger_peak_N=np.maximum(f.max(axis=0),skin[-1]).tolist(),local_excess_integral_Ns=float(np.sum(local*dur[:,None])),
            total_excess_integral_Ns=float(total@dur),palm_peak_N=float(max(np.max(trace.get('substep_palm',[0.])),trace.get('palm',np.zeros(len(t)))[-1])))
        mechanical=m.get('mechanical_release_at',release_at)
        masks={'shared_preparation':st<handoff-1e-9,
            'policy_holding_transport' if admitted else 'post_preparation_without_policy':(st>=handoff-1e-9)&(st<(mechanical if mechanical is not None else duration)-1e-9)}
        if mechanical is not None:masks['mechanical_release_and_after']=st>=mechanical-1e-9
        safety['stages']={name:dict(duration_s=float(dur[mask].sum()),finger_peak_N=f[mask].max(axis=0).tolist(),
            local_excess_integral_Ns=float(np.sum(local[mask]*dur[mask,None])),total_excess_integral_Ns=float(total[mask]@dur[mask])) for name,mask in masks.items() if mask.any()}
    hold=(t>=handoff-1e-9)&(t<hold_end-1e-9)
    progress=dict(lift_absolute_error_integral_m_s=float(np.trapezoid(lift_error[hold],t[hold])) if hold.sum()>1 else None,
        lift_max_error_m=float(lift_error[hold].max()) if hold.any() else None,
        pause_fraction=float(np.asarray(trace.get('transport_paused',np.zeros(len(t))))[hold].mean()) if hold.any() else None)
    if 'actual_motion_target' in trace and 'executed_motion_reference' in trace:
        progress['requested_vs_executed_command_max']=np.max(abs(trace['actual_motion_target'][hold]-trace['executed_motion_reference'][hold]),axis=0).tolist() if hold.any() else None
    if 'wrist_position' in trace and 'requested_wrist_reference' in trace:
        progress['wrist_actual_tracking_max_rad']=np.max(abs(trace['wrist_position'][hold]-trace['requested_wrist_reference'][hold]),axis=0).tolist() if hold.any() else None
        progress['wrist_tracking_observed']=True
    else:progress['wrist_tracking_observed']=False
    progress['holding_motion_tracking']=motion_tracking(trace,handoff,min(hold_end,schedule['release_s']))
    progress['dynamic_motion_tracking']=motion_tracking(trace,*p['endpoints']['dynamic_quality']['window_s'])
    wrist_failure=None;wrist_contract=p.get('motion_tracking',{})
    wrist_available=bool('wrist_position' in trace and 'requested_wrist_reference' in trace
        and np.asarray(trace['wrist_position']).shape==(len(t),2)
        and np.asarray(trace['requested_wrist_reference']).shape==(len(t),2)
        and np.isfinite(trace['wrist_position']).all() and np.isfinite(trace['requested_wrist_reference']).all())
    wrist_guard_available=bool(wrist_available and wrist_contract.get('wrist_tolerance_rad') is not None)
    if wrist_guard_available:
        wrist_error=abs(np.asarray(trace['wrist_position'])-np.asarray(trace['requested_wrist_reference']))
        wrist_failure=first_dwell(t,np.any(wrist_error>np.asarray(wrist_contract['wrist_tolerance_rad']),axis=1),
            handoff,hold_end,float(wrist_contract['failure_dwell_s']),exclusive=True)
    full_motion_success=bool(mission and wrist_guard_available and wrist_failure is None)
    scene=m.get('scenario','unknown');scene=scene.get('name','unknown') if isinstance(scene,dict) else scene
    events=p['event_recovery']['events'].get(scene,[])
    record=dict(schema='joint-hand-research-evaluation-v3',method=m.get('research_method',m.get('method')),scenario=scene,
        physical_seed=m.get('physical_seed',m.get('seed')),noise_seed=m.get('noise_seed',m.get('seed')),research_phase=phase,
        record_complete=bool(complete),preparation_passed=admitted,policy_exposed=bool(policy[hold].any()) if hold.any() else False,
        mission_success=mission,failure_reason=failure,failure_at=failure_at,
        full_motion_mission_success=full_motion_success,
        wrist_tracking_guard=dict(available=wrist_guard_available,encoders_available=wrist_available,failure_at=wrist_failure,
            tolerance_rad=wrist_contract.get('wrist_tolerance_rad'),
            role='new development engineering endpoint; legacy mission_success retained separately'),
        initial_stability_wait_s=float(first_stable-handoff) if first_stable is not None else p['endpoints']['initial_stability']['restricted_horizon_s'],
        release=dict(scheduled=True,actual_command_at=release_at,physical_exposure=physical_exposure,physical_success=physical_released,
            task_qualified_exposure=task_release_qualified,task_qualified_success=task_released,
            physical_clearance_latency_s=float(clearance-release_at) if physical_released else None),
        safety=safety,progress=progress,filter_contract=filter_contract,
        static_quality=quality_window(trace,band,support,raised,policy,complete,*p['endpoints']['static_quality']['window_s']),
        dynamic_quality=quality_window(trace,band,support,raised,policy,complete,*p['endpoints']['dynamic_quality']['window_s']),
        entire_holding_quality=quality_window(trace,band,support,raised,policy,complete,handoff,schedule['release_s']),
        event_recovery=event_recovery(t,stable,events,trace,failure_at,complete,release_at),
        physical_parameters=m.get('config',{}),interpretation='Separate outcome vector; no force-quality benefit compensates mission or safety failure.')
    return jsonable(record)


def evaluate_research(path,protocol_path=PROTOCOL):
    path=Path(path);protocol=read_protocol(protocol_path)
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
        wanted=('time','normal','skin_normal','palm','support','position','relative','relative_velocity','angular_velocity',
            'lift','lift_ref','all_hand_object','controller_active','transport_paused','warning_count','collision',
            'actual_motion_target','executed_motion_reference','wrist_position','requested_wrist_reference',
            'substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm')
        trace={k:archive[k] for k in wanted if k in archive.files}
    sidecar=path.with_name('metadata.json')
    if sidecar.is_file():metadata.update(json.loads(sidecar.read_text(encoding='utf-8')))
    record=evaluate_arrays(trace,metadata,protocol)
    record.update(trace_sha256=sha(path),protocol_sha256=sha(protocol_path),evaluation_source_sha256=sha(__file__))
    return record


def summarize(records):
    rows=[]
    for method in sorted({r['method'] for r in records}):
        rr=[r for r in records if r['method']==method];events=[e for r in rr for e in r['event_recovery']]
        rows.append(dict(method=method,assigned=len(rr),complete_records=sum(r['record_complete'] for r in rr),
            missions_completed=sum(r['mission_success'] for r in rr),force_guard_passes=sum(bool(r['safety'] and r['safety']['guard_pass']) for r in rr),
            policy_exposed=sum(r['policy_exposed'] for r in rr),static_windows=sum(r['static_quality']['eligible'] for r in rr),
            dynamic_windows=sum(r['dynamic_quality']['eligible'] for r in rr),physical_release_exposures=sum(r['release']['physical_exposure'] for r in rr),
            physical_releases=sum(r['release']['physical_success'] for r in rr),task_qualified_releases=sum(r['release']['task_qualified_success'] for r in rr),
            planned_events=len(events),event_status_counts=dict(Counter(e['status'] for e in events)),
            failures=dict(Counter(r['failure_reason'] for r in rr if r['failure_reason']))))
    return dict(rows=rows,records=records,interpretation='Descriptive development readout; confirmation inference requires the separately frozen physical-seed statistical contract.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trace',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out).resolve()
    if not out.is_relative_to(ROOT) or out.exists():raise ValueError('Choose a fresh output inside this research directory')
    result=evaluate_research(args.trace);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(out),mission_success=result['mission_success'],record_complete=result['record_complete']),ensure_ascii=False))


if __name__=='__main__':main()

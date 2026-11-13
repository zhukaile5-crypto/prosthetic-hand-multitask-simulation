"""Saved V8 gate20 raw/metric audit. Never imports plant/controller/runtime."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import hashlib,json,runpy,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'output/development_v8_counterexamples_v1'
sys.dont_write_bytecode=True

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if a is None or b is None:return a is b
    if isinstance(a,(bool,str)) or isinstance(b,(bool,str)):return a==b
    try:return bool(np.isclose(a,b,atol=1e-9,rtol=1e-10,equal_nan=True))
    except (TypeError,ValueError):return a==b

def dwell(t,flag,start,end,seconds,exclusive=False):
    first=None
    for i in np.flatnonzero((t>=start-1e-9)&(t<(end-1e-9) if exclusive else t<=end+1e-9)):
        if flag[i]:
            if first is None:first=t[i]
            if t[i]-first>=seconds-1e-9:return float(t[i])
        else:first=None
    return None

def variation(t,f):
    duration=float(t[-1]-t[0]);tv=np.abs(np.diff(f,axis=0)).sum(0);net=np.abs(f[-1]-f[0])
    total=f.sum(1)
    return dict(finger_TV_rate_N_s=float(tv.sum()/duration),finger_net_drift_rate_N_s=float(net.sum()/duration),
        finger_reversal_rate_N_s=float((tv-net).sum()/(2*duration)),total_force_TV_rate_N_s=float(np.abs(np.diff(total)).sum()/duration),
        finger_SD_mean_N=float(f.std(axis=0).mean()),total_force_SD_N=float(total.std()),
        per_finger_TV_N=tv.tolist(),per_finger_absolute_net_N=net.tolist(),per_finger_retrace_N=((tv-net)/2).tolist())

def phase_wave(z,m,p):
    dt=float(m['physics_timestep']);half=round(.025/dt);grid=np.arange(-half,half+1)*dt
    kernel=np.sinc(100*grid)*np.hanning(2*half+1)*100*dt;kernel/=kernel.sum()
    ft=np.r_[0.,z['substep_time']+z['substep_duration']];force=np.vstack([z['normal'][0],z['substep_normal']]);t=z['time'];out=np.full((len(t),5),np.nan)
    release=m.get('release_at');release=12. if release is None else release
    boundaries=sorted(set([0.,p['schedule']['handoff_s'],release,p['schedule']['duration_s']]))
    for lo,hi in zip(boundaries,boundaries[1:]):
        src=(ft>=lo-1e-9)&(ft<=hi+1e-9);dst=(t>=lo-1e-9)&(t<=hi+1e-9)
        if lo==release and lo>0:dst&=t>lo+1e-9
        for digit in range(5):out[dst,digit]=np.interp(t[dst],ft[src],np.convolve(np.pad(force[src,digit],half,mode='reflect'),kernel,mode='valid'))
    return out

def motion_tracking(z,lo,hi):
    t=z['time'];duration=np.maximum(0.,np.minimum(np.r_[t[1:],t[-1]],hi)-np.maximum(t,lo))
    error=np.column_stack([z['lift'],z['wrist_position']])-z['actual_motion_target']
    active=duration>0
    return dict(rmse=np.sqrt(np.sum(error**2*duration[:,None],axis=0)/(hi-lo)).tolist(),
        absolute_error_integral=np.sum(abs(error)*duration[:,None],axis=0).tolist(),
        max_abs_error=np.max(abs(error[active]),axis=0).tolist(),duration_s=float(duration.sum()))

def outcome(z,m,p):
    th=p['thresholds'];t=z['time'];n=z['normal'];skin=z['skin_normal'];release=m['release_at']
    contact=n>th['finger_contact_min_N'];support=contact[:,0]&(contact.sum(1)>=th['minimum_contacts_including_thumb'])
    raised=z['position'][:,2]>=m['initial_position'][2]+th['minimum_object_lift_m']
    le=abs(z['lift']-z['lift_ref']);motion=(le<=th['lift_tracking_tolerance_m'])&(z['support']<=th['table_support_max_N'])
    relative=np.linalg.norm(z['relative']-np.asarray(m['reference_relative']),axis=1)
    limits=np.asarray(m['limits']);budget=m['budget'];tol=th['force_numerical_tolerance_N']
    safe=(skin<=limits+tol).all(1)&(skin.sum(1)<=budget+tol)
    stable=support&raised&motion&safe&(relative<=th['stable_relative_drift_m'])&(np.linalg.norm(z['relative_velocity'],axis=1)<=th['stable_relative_speed_m_s'])&(np.linalg.norm(z['angular_velocity'],axis=1)<=th['stable_angular_speed_rad_s'])
    established=dwell(t,support&raised&motion,0.,7.5,.1)
    loss=dwell(t,~support|~raised|(relative>th['retention_relative_drift_m']),7.5,release,th['retention_failure_dwell_s'],True)
    tracking=dwell(t,le>th['lift_tracking_tolerance_m'],7.5,release,th['transport_failure_dwell_s'],True)
    failure=None;when=None
    if not m['handoff_passed'] or established is None:failure=m.get('preparation_failure') or 'preparation_or_handoff_failed';when=7.5
    elif loss is not None and (tracking is None or loss<=tracking):failure='support_or_object_lost';when=loss
    elif tracking is not None:failure='transport_reference_unmet';when=tracking
    at=int(np.argmin(abs(t-release)));exposed=bool(support[at] and raised[at] and z['support'][at]<=th['table_support_max_N'] and z['all_hand_object'][at]>th['all_hand_clearance_N'])
    clear=dwell(t,z['all_hand_object']<=th['all_hand_clearance_N'],release,min(16.,release+th['release_deadline_s']),th['release_clearance_dwell_s'])
    physical=bool(exposed and clear is not None);qualified=bool(exposed and m['handoff_passed'] and (when is None or when>release))
    first=dwell(t,stable,7.5,8.9,th['stable_dwell_s']);wait=1.4 if first is None else first-7.5
    wrist=dwell(t,(abs(z['wrist_position']-z['requested_wrist_reference'])>.015).any(1),7.5,release,.3,True)
    f=z['substep_skin_normal'];duration=z['substep_duration'];breach=(f>limits+tol).any(1)|(f.sum(1)>budget+tol)
    guard=bool(not breach.any() and safe[-1]);mission=bool(m['handoff_passed'] and failure is None and physical and qualified and z['warning_count'].max()==0)
    return dict(mission_success=mission,full_motion_mission_success=mission and wrist is None,
        failure_reason=failure,failure_at=when,initial_stability_wait_s=wait,physical_exposure=exposed,physical_success=physical,
        task_qualified_success=physical and qualified,release_latency_s=None if not physical else clear-release,
        guard_pass=guard,local_excess_integral_Ns=float(np.sum(np.maximum(f-limits,0)*duration[:,None])),
        total_excess_integral_Ns=float(np.maximum(f.sum(1)-budget,0)@duration),
        first_guard_breach_time_end_s=None if not breach.any() else float((z['substep_time']+duration)[np.flatnonzero(breach)[0]]),
        whole_guard_breach_duration_s=float(duration[breach].sum()),supported=support,raised=raised)

def main():
    if not (FOLDER/'execution_complete.json').is_file() or not read(FOLDER/'execution_complete.json').get('execution_complete'):
        print('WAITING: execution_complete true is required before reading completed results');return 2
    rows=read(FOLDER/'trials.json');manifest=read(FOLDER/'manifest.json');finish=read(FOLDER/'execution_complete.json')
    snap=FOLDER/'sources';p=read(snap/'protocol.json');evaluate=runpy.run_path(str(snap/'evaluation.py'))['evaluate_arrays']
    checks={};records=[]
    def check(k,v):checks[k]=bool(v)
    check('development_only',finish['phase']=='development' and finish['performance_claim_allowed'] is False)
    check('terminal_complete20',finish['execution_complete'] and finish['trials']==finish['planned']==len(rows)==20 and not finish['runtime_failures'])
    check('trials_file_identity',sha(FOLDER/'trials.json')==finish['trials_sha256'])
    check('zero_new_physics_import',not any(n=='mujoco' or n.startswith('mujoco.') for n in sys.modules))
    wanted=['time','normal','skin_normal','palm','support','position','relative','relative_velocity','angular_velocity','lift','lift_ref','all_hand_object','controller_active','transport_paused','warning_count','collision','actual_motion_target','executed_motion_reference','wrist_position','requested_wrist_reference','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm','qpos','qvel','ctrl','alpha','executed_action','prediction_available','predicted_normal','prediction_at','model_invalid','observation_unavailable']
    for row in sorted(rows,key=lambda x:(x['scenario'],x['method'])):
        key=f"{row['scenario']}/{row['method']}/{row['seed']}";path=Path(row['trace_path']);m=read(path.with_name('metadata.json'))
        with np.load(path,allow_pickle=False) as archive:
            z={k:archive[k] for k in wanted if k in archive.files};inline=json.loads(str(archive['metadata']))
            if row['method']=='joint':z['debug_json']=archive['debug_json']
        check(key+'/trace_byte_hash',sha(path)==row['trace_sha256']==m['trace_sha256'])
        check(key+'/sidecar_inline_metadata',inline=={k:v for k,v in m.items() if k!='trace_sha256'})
        check(key+'/result_dict',read(path.with_name('result.json'))==row)
        t=z['time'];st=z['substep_time'];dur=z['substep_duration'];dt=float(m['physics_timestep'])
        check(key+'/full_clocks',len(t)==3201 and np.allclose(t,np.arange(3201)*.005,atol=1e-8,rtol=0) and len(st)==round(16/dt) and np.allclose(st,np.arange(len(st))*dt,atol=1e-7,rtol=0) and np.allclose(dur,dt,atol=1e-12,rtol=0))
        check(key+'/finite_actual_truth_and_actions',all(np.isfinite(z[k]).all() for k in ('normal','skin_normal','qpos','qvel','ctrl','alpha','executed_action','substep_normal','substep_skin_normal')))
        r=row['research_metrics'];fresh=evaluate(z,m,p);fresh.update(trace_sha256=sha(path),protocol_sha256=sha(snap/'protocol.json'),evaluation_source_sha256=sha(snap/'evaluation.py'))
        check(key+'/all_frozen_v3_metrics',same(fresh,r))
        own=outcome(z,m,p)
        check(key+'/independent_task_release_and_initial_settling',all(same(own[a],r[b]) for a,b in [('mission_success','mission_success'),('full_motion_mission_success','full_motion_mission_success'),('failure_reason','failure_reason'),('failure_at','failure_at'),('initial_stability_wait_s','initial_stability_wait_s')]) and all(same(own[a],r['release'][b]) for a,b in [('physical_exposure','physical_exposure'),('physical_success','physical_success'),('task_qualified_success','task_qualified_success'),('release_latency_s','physical_clearance_latency_s')]))
        check(key+'/independent_whole16s_raw_safety',all(same(own[k],r['safety'][k]) for k in ('guard_pass','local_excess_integral_Ns','total_excess_integral_Ns')))
        band=phase_wave(z,m,p);qualities={}
        for name,lo,hi in [('static_quality',8.2,8.9),('dynamic_quality',9.3,11.7),('entire_holding_quality',7.5,12.)]:
            win=(t>=lo-1e-9)&(t<=hi+1e-9);elig=bool(win.sum()==round((hi-lo)/.005)+1 and np.all(own['supported'][win]&own['raised'][win]&(z['support'][win]<=.05)&z['controller_active'][win].astype(bool)))
            check(key+'/'+name+'/coverage',elig==r[name]['eligible'])
            if elig:
                qualities[name]={sample:variation(t[win],signal[win]) for sample,signal in [('bandlimited',band),('raw_200Hz',z['normal'])]}
                check(key+'/'+name+'/independent_phase_FIR_raw_TV_net_retrace',all(same(v,r[name][sample][k]) for sample,q in qualities[name].items() for k,v in q.items() if not k.startswith('per_finger_')))
        tracking={}
        for name,lo,hi in [('holding_motion_tracking',7.5,12.),('dynamic_motion_tracking',9.3,11.7)]:
            tr=motion_tracking(z,lo,hi);tracking[name]=tr
            check(key+'/'+name+'/independent_fixed_window_encoders',r['progress'][name]['available'] and r['progress'][name]['complete_window'] and not r['progress'][name]['censored'] and all(same(v,r['progress'][name][k]) for k,v in tr.items()))
        force=z['substep_skin_normal'];limits=np.asarray(m['limits']);mechanical=m.get('mechanical_release_at');mechanical=12. if mechanical is None else mechanical
        stages={}
        for name,mask in [('shared_preparation',st<7.5-1e-9),('policy_holding_transport',(st>=7.5-1e-9)&(st<mechanical-1e-9)),('mechanical_release_and_after',st>=mechanical-1e-9)]:
            f=force[mask];duration=dur[mask];stages[name]=dict(duration_s=float(duration.sum()),finger_peak_N=f.max(0).tolist(),local_excess_integral_Ns=float(np.sum(np.maximum(f-limits,0)*duration[:,None])),total_excess_integral_Ns=float(np.maximum(f.sum(1)-m['budget'],0)@duration))
        check(key+'/all_raw_stage_integrals',same(stages,r['safety']['stages']))
        check(key+'/prediction_availability_clock',np.isnan(z['prediction_at'][~z['prediction_available']]).all() and not z['prediction_available'][t>=12.-1e-9].any())
        recovery=None
        if row['method']=='joint':
            check(key+'/V8_empirical_predictions_all_unavailable',not z['prediction_available'].any() and np.isnan(z['predicted_normal']).all())
            hold=np.flatnonzero((t>=7.5-1e-9)&(t<12.-1e-9)&z['controller_active'].astype(bool))
            debug=[json.loads(str(z['debug_json'][i])) for i in hold]
            flags=np.array([bool(d.get('recovery_motion_hold',False)) for d in debug])
            prior=np.maximum(0,hold-1);delta=z['executed_motion_reference'][hold]-z['executed_motion_reference'][prior]
            check(key+'/recovery_holds_reference_not_falsely_physical_encoder_stop',not flags.any() or np.max(abs(delta[flags]))<=1e-9)
            episodes=[];start=None
            for i,flag in zip(hold,flags):
                if flag and start is None:start=float(t[i])
                if not flag and start is not None:episodes.append(dict(start_s=start,end_s=float(t[i]),duration_s=float(t[i])-start,censored_by_release=False));start=None
            if start is not None:episodes.append(dict(start_s=start,end_s=12.,duration_s=12.-start,censored_by_release=True))
            recovery=dict(holding_ticks=len(hold),active_hold_ticks=int(flags.sum()),holding_fraction=float(flags.mean()) if len(flags) else None,
                episodes=episodes,material_risk_ticks=sum(bool(d.get('recovery_evidence',{}).get('material_risk')) for d in debug),
                slip_risk_ticks=sum(bool(d.get('recovery_evidence',{}).get('slip_risk')) for d in debug),
                legacy_transport_paused_fraction=float(z['transport_paused'][hold].mean()) if len(hold) else None,
                interpretation='motion-command hold from controller debug, not physical encoder arrest; original target/deadline retained and tracking costs counted')
        for k in ('supported','raised'):own.pop(k)
        records.append(dict(scenario=row['scenario'],method=row['method'],seed=row['seed'],trace_sha256=sha(path),
            physical_config=m['config'],outcome=own,qualities=qualities,tracking=tracking,stages=stages,recovery_motion=recovery,
            initial_stability_wait_s=r['initial_stability_wait_s'],mean_actual_total_finger_force_N={n:r[n]['mean_actual_finger_force_N'] for n in qualities},
            model_invalid_ticks=int(z['model_invalid'].sum()),observation_unavailable_ticks=int(z['observation_unavailable'].sum()),prediction_available_ticks=int(z['prediction_available'].sum()),
            quality_filter_contract=r['filter_contract']))
        print(key,'raw_metric_verified',flush=True)
    methods=sorted({x['method'] for x in rows});scenes=sorted({x['scenario'] for x in rows});indexed={(x['scenario'],x['method']):x for x in records}
    summaries=[];contrasts=[]
    for method in methods:
        group=[x for x in records if x['method']==method]
        summary=dict(method=method,assigned=len(group),missions=sum(x['outcome']['mission_success'] for x in group),full_motion_missions=sum(x['outcome']['full_motion_mission_success'] for x in group),whole16s_guard_passes=sum(x['outcome']['guard_pass'] for x in group),physical_releases=sum(x['outcome']['physical_success'] for x in group),
            holding_local_excess_total_Ns=sum(x['stages']['policy_holding_transport']['local_excess_integral_Ns'] for x in group),release_latency_mean_s=float(np.mean([x['outcome']['release_latency_s'] for x in group])),initial_stability_wait_mean_s=float(np.mean([x['initial_stability_wait_s'] for x in group])))
        summary['quality']={}
        for name in ('static_quality','dynamic_quality','entire_holding_quality'):
            selected=[x for x in group if name in x['qualities']]
            summary['quality'][name]=dict(coverage=len(selected),bandlimited_TV_mean_N_s=float(np.mean([x['qualities'][name]['bandlimited']['finger_TV_rate_N_s'] for x in selected])) if selected else None,raw_TV_mean_N_s=float(np.mean([x['qualities'][name]['raw_200Hz']['finger_TV_rate_N_s'] for x in selected])) if selected else None)
        summaries.append(summary)
    for baseline in ('ratio','passive','ratio_preview','passive_preview'):
        for name in ('static_quality','dynamic_quality','entire_holding_quality'):
            common=[s for s in scenes if name in indexed[(s,baseline)]['qualities'] and name in indexed[(s,'joint')]['qualities']]
            for sampling in ('bandlimited','raw_200Hz'):
                a=[indexed[(s,baseline)]['qualities'][name][sampling]['finger_TV_rate_N_s'] for s in common];b=[indexed[(s,'joint')]['qualities'][name][sampling]['finger_TV_rate_N_s'] for s in common]
                contrasts.append(dict(baseline=baseline,candidate='joint',window=name,sampling=sampling,assigned_pairs=len(scenes),common_pairs=len(common),scenarios=common,baseline_mean=float(np.mean(a)) if a else None,joint_mean=float(np.mean(b)) if b else None,joint_minus_baseline=float(np.mean(np.array(b)-a)) if a else None,relative_change=float(np.mean(b)/np.mean(a)-1.) if a and np.mean(a)>0 else None))
    result=dict(schema='v8_gate20_independent_raw_metrics_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],records=records,method_summaries=summaries,paired_quality_contrasts=contrasts,
        input_sha256={n:sha(FOLDER/n) for n in ('trials.json','manifest.json','plan.json','execution_complete.json')},reviewer_sha256=sha(__file__),
        physics_steps_executed=0,scope='All20 actual trace byte identities, complete numeric raw clocks, full frozenV3 evaluations plus independent phase-FIR/TV, whole16s and stage force guards/integrals, release, settling, encoder tracking; paired-source/preview audit is a separate artifact. V8 intentionally publishes no empirical force forecast and may pause actual motion; pause is independently derived from recovery debug, not legacy pause_fraction. No new dynamics or performance superiority is asserted.',
        statistical_scope='One already-seen development seed610212 reused across4 selected scenarios; five algorithms are paired treatments, not20 independent physical draws. Four scenario strata also reuse the same underlying standardized physical/noise stream, with scene transforms. No confidence/NI/superiority test.')
    out=ROOT/'review/v8_gate20_raw_independent_v1.json'
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],output=str(out),sha256=sha(out)),ensure_ascii=False))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

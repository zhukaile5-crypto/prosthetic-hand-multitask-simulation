"""Read complete pinned pilot only; no policy, plant, physics or source edits."""
from pathlib import Path
from collections import Counter
import argparse,gzip,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def counts(values):return dict(Counter(values))
def summary(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return dict(n=len(a),median=None if not len(a) else float(np.median(a)),p95=None if not len(a) else float(np.quantile(a,.95)),max=None if not len(a) else float(a.max()))
def signal(t,x):
    dx=np.diff(x,axis=0);tv=abs(dx).sum(0);net=abs(x[-1]-x[0]);freq=np.fft.rfftfreq(len(dx),float(np.median(np.diff(t))))
    power=abs(np.fft.rfft(dx-dx.mean(0),axis=0))**2;dom=[];band=[];reversals=[]
    for j in range(x.shape[1]):
        total=power[1:,j].sum();k=int(np.argmax(power[1:,j]))+1
        dom.append(None if total==0 else float(freq[k]));band.append(None if total==0 else float(power[(freq>=20)&(freq<=30),j].sum()/total))
        signs=np.sign(dx[:,j]);signs=signs[signs!=0];reversals.append(int(np.count_nonzero(signs[1:]!=signs[:-1])))
    return dict(samples=len(t),duration_s=float(t[-1]-t[0]),per_axis_TV=tv.tolist(),per_axis_net=net.tolist(),per_axis_retracing=(.5*np.maximum(0.,tv-net)).tolist(),
        positive_change=np.maximum(dx,0).sum(0).tolist(),negative_change=(-np.minimum(dx,0).sum(0)).tolist(),direction_reversals=reversals,
        dominant_increment_frequency_Hz=dom,increment_power_fraction20to30Hz=band,
        scope='Descriptive fixed-window command/force changes; retracing is not automatically unwanted oscillation, spectral association is not a causal proof.')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='output/development_v13_joint_pilot_v1');ap.add_argument('--out',required=True);args=ap.parse_args()
    folder=ROOT/args.root;out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    complete_path=folder/'execution_complete.json'
    if not complete_path.exists() or not read(complete_path).get('execution_complete'):
        print(json.dumps(dict(status='PENDING',source=str(folder))));return 2
    issues=[];checks=[]
    def check(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:issues.append(dict(name=name,detail=detail))
    done=read(complete_path);manifest=read(folder/'manifest.json');plan=read(folder/'plan.json');trials=read(folder/'trials.json')
    check('complete_source_freeze',done['sources_unchanged'] and not done['runtime_failures'])
    for name in ('trials','assignment_ledger','plan','manifest'):check(name+'_identity',sha(folder/(name+'.json'))==done[name+'_sha256'])
    identity=manifest['research_source_sha256'];snap=read(folder/'source_snapshot_manifest.json')
    check('exact99_snapshot_set',len(identity)==99 and set(snap['files'])==set(identity))
    for key,expected in identity.items():
        p=folder/snap['files'][key]['snapshot_path'];check('snapshot/'+key,sha(p)==expected)
    check('three_assigned_unique_records',len(trials)==len(plan['jobs'])==3 and len({(x['scenario'],x['method'],x['seed']) for x in trials})==3)
    # Compile a static declared robot only, for named control-address mapping
    # and its original robot gravity-bias expression. No MjData/plant created.
    import research_imports,mujoco,hand_plant
    from foundation_config import FoundationConfig
    from v11.robot_servo_effort import ACTUATORS21
    from v13_a1.codec import ReferenceActor21
    from v13_joint.design import reference_rates
    forbidden=[]
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_kinematics','mj_comPos','mj_collision','mj_contactForce'):
        def deny(*a,_name=name,**k):forbidden.append(_name);raise AssertionError(_name)
        setattr(mujoco,name,deny)
    hand_plant.HandPlant.__init__=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No plant'))
    model,_=hand_plant._model(FoundationConfig());actor=ReferenceActor21(model);ids=actor.indices21
    rows=[]
    for job in plan['jobs']:
        path=Path(job['destination']);method=job['method'];result=read(path/'result.json');meta=read(path/'metadata.json');jm=read(path/'joint_decision_manifest.json')
        check(method+'/source99',meta['research_source_sha256']==identity)
        check(method+'/result_trace_hash',sha(path/'trace.npz')==result['trace_sha256'])
        check(method+'/joint_log_hash',sha(path/jm['file'])==jm['sha256'])
        keys=('time','ctrl_before','ctrl','controller_active','control_ms','full_control_loop_ms','observation_ms','physics_advance_ms','actor_callback_ms',
            'requested_reference21','executed_reference21','reference_increment21','mode_generation','robot_actuator_target21','robot_control_time_s',
            'debug_json','predicted_normal','prediction_available','normal','skin_normal','substep_time','substep_duration','substep_skin_normal','substep_normal')
        with np.load(path/'trace.npz',allow_pickle=False) as z:d={k:z[k] for k in keys}
        t=d['time'];active=d['controller_active'].astype(bool);debug=[json.loads(str(v)) for v in d['debug_json']]
        check(method+'/complete_3201_clock',len(t)==3201 and np.allclose(t,np.arange(3201)*.005,atol=1e-9,rtol=0))
        check(method+'/complete_32000_force_clock',len(d['substep_time'])==32000 and np.isfinite(d['substep_skin_normal']).all() and np.allclose(d['substep_time']+d['substep_duration'],np.arange(1,32001)*.0005,atol=1e-8,rtol=0))
        check(method+'/unavailable_predictions',not d['prediction_available'].any() and np.isnan(d['predicted_normal']).all())
        port=d['executed_reference21'].copy();port[:,18]+=actor.lift_bias_m
        check(method+'/actual21_engine_encoding',port.tobytes()==d['ctrl'][:,ids].tobytes())
        check(method+'/actual_reference_finite',np.isfinite(d['executed_reference21']).all() and np.isfinite(d['reference_increment21']).all())
        requested=d['requested_reference21'];executed=d['executed_reference21'];changed=(requested.view(np.uint64)!=executed.view(np.uint64)).any(1)
        modes=[x.get('mode') for x in debug];rate_excess=np.zeros((len(t),21));flag_mismatch=[]
        for i in np.flatnonzero(active):
            maximum=reference_rates(modes[i])*.005;rate_excess[i]=np.maximum(0.,abs(d['reference_increment21'][i])-maximum)
            if changed[i]!=bool(debug[i].get('execution_changed_plan',False)):flag_mismatch.append(int(i))
        check(method+'/declared_5ms_rates',np.max(rate_excess,initial=0.)<1e-12,float(rate_excess.max()))
        check(method+'/receipt_changed_exact',not flag_mismatch,flag_mismatch[:10])
        expected_port=d['robot_actuator_target21'];check(method+'/robot_clock',np.allclose(d['robot_control_time_s'],t,atol=1e-9,rtol=0))
        check(method+'/observed_before_engine',expected_port.tobytes()==d['ctrl_before'][:,ids].tobytes())
        status=Counter();native_status=Counter();recheck=Counter();by_mode={};decision_indices=[];decision_stats=[];native_finite_calls=0
        with gzip.open(path/jm['file'],'rt',encoding='utf-8') as stream:
            for k,line in enumerate(stream):
                rec=json.loads(line);i=int(round(float(rec['time_s'])/.005));mode=rec['mode'];dd=debug[i]
                check(method+f'/decision_id_clock/{k}',rec['decision_id']==k and abs(t[i]-rec['time_s'])<1e-9 and dd.get('decision_id')==k and abs(dd.get('decision_time_s',-1)-t[i])<1e-9)
                decision_indices.append(i);q=rec['qp'];n=rec['native'];label='not_enabled' if q is None else q['status'];status[label]+=1
                entry=by_mode.setdefault(mode,dict(decisions=0,qp=Counter(),native=Counter(),conditional_support_active=0,conditional_support_unknown=0))
                entry['decisions']+=1;entry['qp'][label]+=1
                if dd.get('conditional_support_active'):entry['conditional_support_active']+=1
                elif mode=='holding':entry['conditional_support_unknown']+=1
                ns='not_called' if n is None else n['status'];native_status[ns]+=1;entry['native'][ns]+=1
                if n is not None:
                    nr=n['diagnostics'].get('prepared_value_final_full_recheck');native_finite_calls+=n['diagnostics'].get('native_evaluations',0)
                    if nr is None:recheck['not_present']+=1
                    elif nr['passed'] and all(v is True for key,v in nr.items() if key.endswith('_equal')):recheck['passed']+=1
                    else:recheck['failed']+=1
                    if n['reference21'] is not None:
                        check(method+f'/available_native_full_check/{k}',nr is not None and nr['passed'])
                        check(method+f'/native_fixed_p0/{k}',np.asarray(n['p0_effort21']).tobytes()==np.asarray(rec['domain']['p0_effort21']).tobytes())
                decision_stats.append(dd.get('solver_timing_ms',{}))
        check(method+'/all_decisions_retained',len(decision_indices)==jm['records'])
        decision_mask=np.zeros(len(t),bool);decision_mask[decision_indices]=True
        debug_ids=[int(debug[i]['decision_id']) for i in np.flatnonzero(active)]
        check(method+'/all_active_rows_resolve_to_recorded_decision',set(debug_ids)==set(range(len(decision_indices))))
        timing={}
        for name in ('control_ms','full_control_loop_ms','observation_ms','physics_advance_ms','actor_callback_ms'):
            timing[name]=dict(all_5ms_records=summary(d[name]),active_5ms_records=summary(d[name][active]),actual_20ms_decisions=summary(d[name][decision_mask]),between_decisions=summary(d[name][active&~decision_mask]))
        for key in ('qp','native_inverse','whole_decision'):timing['solver_'+key+'_ms']=summary([x[key] for x in decision_stats if key in x])
        st=d['substep_time'];dt=d['substep_duration'];skin=d['substep_skin_normal'];limits=np.asarray(meta['limits']);budget=float(meta['budget'])
        local=np.maximum(skin-limits,0.);total=np.maximum(skin.sum(1)-budget,0.);release=float(meta['mechanical_release_at']);stages={}
        for name,mask in (('preparation',st<7.5-1e-9),('holding',(st>=7.5-1e-9)&(st<release-1e-9)),('release',st>=release-1e-9)):
            ix=np.flatnonzero(mask);a=skin[ix];j,axis=np.unravel_index(int(np.argmax(a)),a.shape);row=int(ix[j])
            stages[name]=dict(duration_s=float(dt[mask].sum()),peak_N=float(a.max()),peak_digit=int(axis),peak_interval_start_s=float(st[row]),peak_interval_end_s=float(st[row]+dt[row]),
                local_excess_Ns=float(np.sum(local[mask]*dt[mask,None])),total_excess_Ns=float(total[mask]@dt[mask]))
        research=result['research_metrics'];check(method+'/raw_dose_matches_frozen_metric',abs(float(np.sum(local*dt[:,None]))-research['safety']['local_excess_integral_Ns'])<1e-11)
        windows={}
        for name,a,b in (('static',8.2,8.9),('holding',7.5,12.),('dynamic',9.3,11.7)):
            mask=(t>=a-1e-9)&(t<=b+1e-9);windows[name]=dict(window_s=[a,b],reference21=signal(t[mask],executed[mask]),pressure5_200Hz=signal(t[mask],d['normal'][mask]))
            rawt=st+dt;rm=(rawt>=a-1e-9)&(rawt<=b+1e-9);x=d['substep_normal'][rm];windows[name]['raw_integration_TV_rate_N_s']=float(abs(np.diff(x,axis=0)).sum()/(rawt[rm][-1]-rawt[rm][0]))
        rows.append(dict(method=method,trace_sha256=result['trace_sha256'],decision_log_sha256=jm['sha256'],rows=len(t),active_5ms_rows=int(active.sum()),decisions=len(decision_indices),
            qp_status=dict(status),native_status=dict(native_status),full_native_recheck=dict(recheck),by_mode=by_mode,decision_labels=counts(debug[i].get('label') for i in decision_indices),
            native_finite_evaluations=native_finite_calls,changed_receipts_active=int(changed[active].sum()),stale_receipts_active=sum(bool(debug[i].get('stale_plan_replaced')) for i in np.flatnonzero(active)),
            declared_rate_max_excess=float(rate_excess.max()),timing_ms=timing,safety_stages_raw=stages,windows=windows,
            result=dict(task_success=result['task_success'],full_motion_mission_success=research['full_motion_mission_success'],force_guard=research['safety']['guard_pass'],
                local_excess_Ns=research['safety']['local_excess_integral_Ns'],finger_peak_N=research['safety']['finger_peak_N'],static_quality=research['static_quality'],
                release=research['release'],initial_stability_wait_s=research['initial_stability_wait_s'])))
    check('no_forbidden_pipeline',not forbidden,forbidden)
    report=dict(schema='v13_joint_pilot_technical_readonly_v1',passed=not issues,issues=issues,checks=checks,source_count=len(identity),source_identity=identity,
        input=dict(root=str(folder),execution_complete_sha256=sha(complete_path),trials_sha256=sha(folder/'trials.json')),records=rows,
        timing_scope=dict(control_ms='Control branch after sensing/task construction; includes state/decision/commit/reconcile/preparation, not raw public sampling or physics.',
            full_control_loop_ms='Observation, task/event construction, control and trace/history serialization before physics. Excludes set_environment, advance, postadvance truth, return snapshot and save/evaluation.',
            actor_callback_ms='Overlapping accumulated writer callback cost spanning control AND following physics advance; never add to full_control_loop/physics.',
            decisions='JSONL decision_id/time matched to unique trace rows; repeated solver timings on intervening5ms rows are not independent calls.',
            realtime='Parallel single-host development; observed wall times, not hardware deadline guarantee.'),
        numerical_failure_scope='QP nonconvergence, missing conditional support, native unavailability and full-native failure reported separately; none silently recoded as physical loss.',
        physical_scope='All3 assigned retained. One seen nominal/hold physical seed, not independent superiority confirmation. Static compile maps robot addresses/bias only; no control replay or new physical trajectory.',
        spectrum_scope='Fixed official windows; increment spectrum and reference retracing are exploratory mechanism diagnostics, not a causal force-response model.',
        physics_steps=0,source_edits=False,checker_sha256=sha(Path(__file__)))
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),issues=issues,path=str(out),sha256=sha(out),counts=[dict(method=r['method'],qp=r['qp_status'],native=r['native_status'],full=r['full_native_recheck']) for r in rows]),ensure_ascii=False))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

"""Complete four-batch/48-record V12-native read-only acceptance; no dynamics.

Only registration and terminal markers are read until ALL batches terminate.
Each compressed NPZ column is loaded once; only three numeric trial caches
are retained per context. Public packet prefix hashes preserve literal bytes.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/'review/v12_native_development_registration_v1.json'
DIRECTORIES=None
EXPECTED_SOURCE_COUNT=80
PILOT=ROOT/'output/development_v12_native_pilot_v1'
spec=importlib.util.spec_from_file_location('v12_pilot_read_helpers',ROOT/'review/verify_v12_pilot_v1.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
PHYSICAL=helpers.PHYSICAL;PRE=helpers.HANDOFF_PRE
hash_cache={}


def timing_stats(value):
    """The actual 20ms decision grid only; never averaged over sensor ticks."""
    a=np.asarray(value,float)
    valid=np.isfinite(a)&(a>=0)
    good=a[valid]
    return dict(count=int(len(a)),finite_nonnegative_count=int(valid.sum()),
        median_ms=float(np.median(good)) if good.size else None,
        p95_ms=float(np.percentile(good,95)) if good.size else None,
        max_ms=float(good.max()) if good.size else None,values_ms=a.tolist(),
        scope='actual runtime control_ms at active t=7.5+k*.020, t<12; includes public sampling/control/execution overhead, not isolated solver time')


def inspect_native_record(e):
    """Check recorded native residual/loss arithmetic, not a new plant/model run.

    The separate 675-input audit independently re-evaluated the f067 robot law.
    Here every new decision's saved final vectors and candidate-score bytes are
    cross-checked, while genuine numerical/model/feasibility failure stays an
    outcome category rather than being silently called a physical failure.
    """
    problems=[];d=e.get('debug',{});status=str(e.get('status','unavailable'))
    available=e.get('increment8') is not None and e.get('port_target21') is not None
    if (e.get('increment8') is None)!=(e.get('port_target21') is None):problems.append('asymmetric reference/port availability')
    for flag in ('empirical_pressure_B_used','future_q_or_pressure_predicted','physical_safety_certified','global_model_admitted'):
        if d.get(flag) is not False:problems.append('scope flag '+flag+' missing or true')
    if not available:
        return dict(passed=not problems,available=False,status=status,reason=d.get('reason'),
            category='numerical_failure' if status=='NUMERICAL_FAILURE' else 'model_or_input_or_reference_unavailable',
            checked_scalar_count=0,problems=problems)
    worst_direct=None;scalar_count=0
    try:
        values={name:np.asarray(e.get(name),dtype=np.float64) for name in
            ('residual25','desired_torque25','achieved_torque25','reference8','increment8','port_target21')}
        for name,n in [('residual25',25),('desired_torque25',25),('achieved_torque25',25),('reference8',8),('increment8',8),('port_target21',21)]:
            if values[name].shape!=(n,) or not np.isfinite(values[name]).all():raise ValueError(name+' not finite expected shape')
        residual=values['residual25'];direct=values['achieved_torque25']-values['desired_torque25']
        worst_direct=float(np.max(abs(direct-residual)))
        if worst_direct>1e-10:problems.append('saved actual-minus-fixed-goal inconsistent with canonical residual')
        if d.get('true_step_reference_bounds_checked') is not True or d.get('true_actuator_port_bounds_checked') is not True:
            problems.append('actual-domain check missing/false')
        solutions=d.get('scalar_solutions',[]);rechecks=d.get('scalar_native_recheck',[])
        passthrough=d.get('exact_zero_correction_nominal_passthrough') is True
        if passthrough:
            if solutions or rechecks:problems.append('exact passthrough has unexpected scalar result')
        else:
            if len(solutions)!=8 or len(rechecks)!=8:problems.append('non-passthrough result lacks eight scalar records')
            recheck_by_axis={r['axis']:r for r in rechecks}
            if set(recheck_by_axis)!=set(range(8)):problems.append('scalar recheck axis coverage')
            if {r['axis'] for r in solutions}!=set(range(8)):problems.append('scalar solution axis coverage')
            for s in solutions:
                axis=s['axis'];ids=np.asarray(s['torque_support25'],int)
                if len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=25):raise ValueError('bad torque support')
                subset=np.asarray(residual[ids],dtype=np.float64);loss=float(subset@subset)
                expected_residual=s['selected_native_residual_bytes_hex'];expected_loss=s['selected_native_loss_bytes_hex']
                if subset.tobytes().hex()!=expected_residual:problems.append(f'axis{axis} final residual bytes disagree')
                if np.float64(loss).tobytes().hex()!=expected_loss:problems.append(f'axis{axis} final loss bytes disagree')
                if not np.array_equal(subset,np.asarray(s['selected_native_residual'],float)):problems.append(f'axis{axis} scalar residual values disagree')
                candidates=s['candidates']
                if not candidates:raise ValueError('empty scalar candidate pool')
                for c in candidates:
                    if not np.isfinite([c['loss'],c['distance_to_nominal'],c['value']]).all() or c['loss']<0 or c['distance_to_nominal']<0:
                        raise ValueError('nonfinite/negative canonical scalar score')
                selected=min(candidates,key=lambda x:(x['loss'],x['distance_to_nominal'],x['value']))
                if selected!=s['selected']:problems.append(f'axis{axis} recorded selection not canonical lexicographic minimum')
                if np.float64(selected['loss']).tobytes().hex()!=expected_loss:problems.append(f'axis{axis} score authority differs from final loss')
                r=recheck_by_axis[axis]
                if r.get('residual_bytes_equal') is not True or r.get('loss_bytes_equal') is not True:problems.append(f'axis{axis} combined native recheck false')
                if np.float64(r['native_actual_loss']).tobytes()!=np.float64(loss).tobytes():problems.append(f'axis{axis} recheck actual loss mismatch')
                scalar_count+=1
    except (TypeError,ValueError,KeyError,IndexError) as exc:problems.append(type(exc).__name__+': '+str(exc))
    return dict(passed=not problems,available=True,status=status,category='native_available_with_real_residual',
        checked_scalar_count=scalar_count,direct_goal_residual_max_error=worst_direct,problems=problems)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    path=Path(path).resolve()
    if path not in hash_cache:hash_cache[path]=helpers.sha(path)
    return hash_cache[path]


def inventory(reg):
    answer=[]
    for path in (DIRECTORIES or [ROOT/g['output'] for g in reg['groups']]):
        marker=path/'execution_complete.json'
        if not marker.exists():answer.append(dict(directory=str(path),status='pending',reason='terminal marker absent'));continue
        f=read(marker)
        # Runner writes this marker only at final source/preflight failure or
        # after workers have joined. Complete=False here is a retained terminal
        # failure, not an inferred failure from an unfinished directory.
        status='terminal_complete' if f.get('execution_complete') is True else 'terminal_failure'
        answer.append(dict(directory=str(path),status=status,finish=f))
    return answer


def key(job):return (job['scenario'],job['motion_mode'],int(job['seed']),job['method'])


def load_trial(job,result,source,protocol,check):
    folder=Path(job['destination']);path=folder/'trace.npz';method=job['method']
    if result is None or not path.is_file():
        check(str(key(job))+'/assigned_record_available',False,'retained missing/runtime-failed assignment')
        return dict(job=job,status='unavailable'),None
    wanted=set(PHYSICAL)|{'requested_action','controller_active','estimated_normal','tactile_acquisition_time',
        'estimated_load','estimated_load_available','auxiliary_support','auxiliary_support_available',
        'prediction_available','predicted_normal','prediction_at','posterior_mean_time_s','contact_model_available',
        'contact_model_active','validity','debug_json','observation_json','legacy_common_sensor_json','control_ms'}
    with np.load(path,allow_pickle=False) as z:
        missing=wanted-set(z.files)
        check(str(key(job))+'/required_columns',not missing,sorted(missing))
        if missing:return dict(job=job,status='unavailable',reason='missing required columns'),None
        # Every value below is a cached ndarray, not another NPZ decompression.
        data={name:z[name] for name in wanted};meta=json.loads(str(z['metadata']))
    check(str(key(job))+'/result_matches_trial',helpers.equal(result,read(folder/'result.json')))
    outer=read(folder/'metadata.json')
    check(str(key(job))+'/trace_hash',sha(path)==outer['trace_sha256'])
    check(str(key(job))+'/source_identity',meta['research_source_sha256']==source)
    t=data['time'];st=data['substep_time'];dur=data['substep_duration']
    complete=bool(len(t)==3201 and len(st)==32000 and np.allclose(t,np.arange(3201)*.005,rtol=0,atol=1e-8)
        and np.allclose(st,np.arange(32000)*.0005,rtol=0,atol=1e-7) and np.allclose(dur,.0005,rtol=0,atol=1e-9))
    check(str(key(job))+'/full16s_physical_coverage',complete)
    policy=(t>=7.5-1e-9)&(t<12.-1e-9)&data['controller_active'].astype(bool)
    decision=policy&(abs((t-7.5)/.02-np.round((t-7.5)/.02))<1e-7)
    before_ref=np.full((len(t),3),np.nan);rate=np.full((len(t),3),np.nan)
    bounds_lo=np.full((len(t),5),np.nan);bounds_hi=bounds_lo.copy();permission=np.zeros((len(t),5),bool)
    bad_models=[];bad_robot_clock=[];public_samples=np.full(len(t),np.nan)
    for i,value in enumerate(data.pop('observation_json')):
        f=json.loads(str(value));before_ref[i]=f['executed_motion_reference'];rate[i]=f['motion_reference_rate_limits']
        bounds_lo[i]=f['closure_min'];bounds_hi[i]=f['closure_max'];permission[i]=f['closure_permission'];public_samples[i]=f['sample_time']
        if set(f)&{'force_matrix','force_tau','force_gain','model_reference_alpha','response_error_bound'}:bad_models.append(i)
        if len(f['robot_actuator_target21'])!=21 or abs(f['robot_control_time_s']-t[i])>1e-8:bad_robot_clock.append(i)
    check(str(key(job))+'/no_legacy_force_model_policy_fields',not bad_models,bad_models[:5])
    check(str(key(job))+'/current_21_engine_telemetry',not bad_robot_clock,bad_robot_clock[:5])
    raw_common=data.pop('legacy_common_sensor_json')
    base_packet=json.loads(str(raw_common[0]));mech_lo=np.asarray(base_packet['closure_min']);mech_hi=np.asarray(base_packet['closure_max'])
    check(str(key(job))+'/mechanical_permission_not_old_B_bounds',np.array_equal(bounds_lo[policy],np.tile(mech_lo,(int(policy.sum()),1))) and
        np.array_equal(bounds_hi[policy],np.tile(mech_hi,(int(policy.sum()),1))))
    prefix_hashes=[hashlib.sha256(str(value).encode('utf-8')).hexdigest() for value in raw_common[t<7.5-1e-9]]
    del raw_common
    stamp=data['tactile_acquisition_time'];estimate=data['estimated_normal'];valid=data['validity'].astype(bool)
    unknown=~np.isfinite(stamp)|~valid;known=np.isfinite(estimate)
    check(str(key(job))+'/unknown_pressure_not_zero',np.isnan(estimate[unknown]).all())
    clock=np.broadcast_to(t[:,None],stamp.shape)
    check(str(key(job))+'/valid_pressure_acquisition_clock',np.all(np.isfinite(stamp[known])) and np.all(stamp[known]<=clock[known]+1e-8) and
        np.all(clock[known]-stamp[known]<=.12+1e-8))
    check(str(key(job))+'/no_pressure_prediction',not data['prediction_available'].any() and np.isnan(data['predicted_normal']).all() and np.isnan(data['prediction_at']).all())
    check(str(key(job))+'/no_fictional_old_contact_model',not data['contact_model_available'].any() and not data['contact_model_active'].any() and np.isnan(data['posterior_mean_time_s']).all())
    check(str(key(job))+'/unknown_load_aux_not_zero',np.isnan(data['estimated_load'][~data['estimated_load_available'].astype(bool)]).all() and
        np.isnan(data['auxiliary_support'][~data['auxiliary_support_available'].astype(bool)]).all())
    statuses=Counter();available=conditional=numerical=late=0
    failure_reasons=Counter();native_checks=[]
    for i,value in enumerate(data.pop('debug_json')):
        if not policy[i]:continue
        d=json.loads(str(value));late+=bool(d.get('execution_changed_plan'))
        if decision[i]:
            e=d.get('effort_tracking',{});status=str(e.get('status','unavailable'));statuses[status]+=1
            available+=e.get('increment8') is not None and e.get('port_target21') is not None
            numerical+=status=='NUMERICAL_FAILURE'
            if status=='NUMERICAL_FAILURE':failure_reasons[str(e.get('debug',{}).get('reason','reason_not_logged'))]+=1
            native=inspect_native_record(e)
            native_checks.append(dict(time_s=float(t[i]),**native))
            check(str(key(job))+f'/native_scalar/{t[i]:.3f}',native['passed'],native['problems'])
            conditional+=d.get('force_intent',{}).get('fusion_mode')=='conditional_object_contact_allocation'
    mismatch=abs(data['requested_action'][policy]-data['executed_action'][policy])
    action=data['executed_action'][policy];motion=data['executed_motion_reference'][policy]-before_ref[policy]
    blocked=int(((action>1e-10)&~permission[policy]).sum())
    # These are declared existing control rates, not new task/safety endpoints.
    close=int((action>.45*.005+1e-10).sum());opening=int((action<-.60*.005-1e-10).sum())
    move=int((abs(motion)>rate[policy]*.005+1e-10).sum())
    check(str(key(job))+'/declared_rates_permission',not(close+opening+move+blocked),dict(close=close,opening=opening,motion=move,blocked=blocked))
    decision_cost=timing_stats(data['control_ms'][decision])
    check(str(key(job))+'/decision_grid_control_cost_finite',decision_cost['count']==decision_cost['finite_nonnegative_count'])
    m=result['research_metrics']
    f=data['substep_skin_normal'];limits=np.asarray(meta['limits']);budget=float(meta['budget'])
    tol=protocol['thresholds']['force_numerical_tolerance_N']
    peak=np.maximum(f.max(0),data['skin_normal'][-1])
    local_Ns=float(np.sum(np.maximum(f-limits,0)*dur[:,None]))
    total_Ns=float(np.maximum(f.sum(1)-budget,0)@dur)
    raw_guard=bool(m['record_complete'] and complete and (f<=limits+tol).all() and (f.sum(1)<=budget+tol).all()
        and (data['skin_normal'][-1]<=limits+tol).all() and data['skin_normal'][-1].sum()<=budget+tol)
    check(str(key(job))+'/original_safety_recomputed_from_raw',np.allclose(peak,m['safety']['finger_peak_N'],rtol=0,atol=1e-12)
        and abs(local_Ns-m['safety']['local_excess_integral_Ns'])<1e-12 and abs(total_Ns-m['safety']['total_excess_integral_Ns'])<1e-12
        and raw_guard==m['safety']['guard_pass'])
    if meta['handoff_passed']:check(str(key(job))+'/actual_holding_exposure',int(policy.sum())==900 and int(decision.sum())==225)
    row=dict(job=job,status='complete',trace_sha256=sha(path),source_state='same frozen V12-native source80 observer/intent/native-effort comparator',
        handoff_physical_ready=meta['handoff_passed'],legacy_model_domain_supported=meta['legacy_model_domain_supported'],
        old_model_domain_does_not_gate_new_method=bool(meta['handoff_passed'] or meta['preparation_failure']=='preparation_not_ready'),
        physical_parameters=meta['config'],noise=meta['noise'],gains=meta['actual_ratio_task_gains'],
        policy_ticks=int(policy.sum()),decision_slots=int(decision.sum()),effort_available_decisions=int(available),
        effort_numerical_failure_decisions=int(numerical),effort_status=dict(statuses),fusion_conditional_decisions=int(conditional),
        numerical_failure_reasons=dict(failure_reasons),native_scalar_audit=native_checks,
        control_ms_decision_grid=decision_cost,
        requested_vs_actual_alpha_max=float(mismatch.max()) if mismatch.size else None,
        late_projection_ticks=int(late),hidden_ctrl_clip=meta.get('max_hidden_ctrl_clip'),
        max_positive_alpha_rate=(np.maximum(action,0).max(0)/.005).tolist() if len(action) else None,
        max_negative_alpha_rate=(np.maximum(-action,0).max(0)/.005).tolist() if len(action) else None,
        max_motion_rate=(abs(motion).max(0)/.005).tolist() if len(motion) else None,
        unknown_pressure_cells=int(unknown.sum()),metrics_original=m,legacy_task_success=result['task_success'],
        npz_unique_columns_loaded=len(wanted)+1,
        clock_semantics='Measurement pressure remains acquisition filtered; raw force uses physical END=start+duration; not a predictor')
    numeric={k:data[k] for k in PHYSICAL}
    return row,dict(numeric=numeric,prefix_sensor_sha256=prefix_hashes)


def main_execute(out):
    reg=read(REG);inv=inventory(reg)
    if any(x['status']=='pending' for x in inv):
        print(json.dumps(dict(status='pending',batches=inv,physics_steps=0),ensure_ascii=False));return 3
    if out.exists():raise FileExistsError(out)
    checks=[];issues=[]
    def check(name,ok,detail=None):
        c=dict(name=name,passed=bool(ok),detail=detail);checks.append(c)
        if not ok:issues.append(c)
    source=reg['source_sha256'];check('registered_expected_source_keys',len(source)==EXPECTED_SOURCE_COUNT==reg['source_count'])
    assignments={};results={};batch_info=[]
    for group,batch in zip(reg['groups'],inv):
        run=Path(batch['directory']);finish=batch['finish']
        required=('manifest.json','trials.json','source_snapshot_manifest.json')
        missing=[name for name in required if not (run/name).is_file()]
        check(run.name+'/terminal_batch_records_exist',not missing,missing)
        if missing:
            batch_info.append(dict(directory=str(run),terminal_status=batch['status'],retained_missing_files=missing))
            continue
        manifest=read(run/'manifest.json');trials=read(run/'trials.json')
        check(run.name+'/terminal_hash',sha(run/'trials.json')==finish['trials_sha256'])
        check(run.name+'/same_registered_sources',manifest['research_source_sha256']==source and finish.get('sources_unchanged') is True)
        snap=read(run/'source_snapshot_manifest.json');check(run.name+'/snapshot_set',set(snap['files'])==set(source))
        for k,v in snap['files'].items():check(run.name+'/snapshot/'+k,sha(run/v['snapshot_path'])==v['sha256']==source[k])
        for job in manifest['jobs']:
            k=key(job);check('unique_assignment/'+str(k),k not in assignments);assignments[k]=job
        for result in trials:
            k=(result['scenario'],manifest['motion_mode'],int(result['seed']),result['method'])
            check('unique_result/'+str(k),k not in results);results[k]=result
        batch_expected={(s,group['motion_mode'],int(seed),method) for s in group['scenarios'] for seed in group['seeds'] for method in reg['methods']}
        check(run.name+'/registered_batch_assignment',{key(j) for j in manifest['jobs']}==batch_expected and len(manifest['jobs'])==group['records'])
        batch_info.append(dict(directory=str(run),terminal_status=batch['status'],trials=len(trials),
            manifest_sha256=sha(run/'manifest.json'),trials_sha256=sha(run/'trials.json')))
    expected={(s,m,seed,method) for s in reg['scenarios'] for m in reg['motion_modes'] for seed in reg['seeds'] for method in reg['methods']}
    check('exact48_registered_unique_matrix',set(assignments)==expected and len(expected)==48)
    check('all48_results_or_retained_failure',set(results)==expected)
    rows=[];contexts=[];protocol=read(ROOT/'protocol.json')
    check('current_metric_protocol_matches_registered_source',sha(ROOT/'protocol.json')==source['protocol.json'])
    for seed in reg['seeds']:
        for scene in reg['scenarios']:
            for motion in reg['motion_modes']:
                # Release the three-trial cache after each context, O(n) NPZ IO.
                cache={};context_rows=[]
                for method in reg['methods']:
                    k=(scene,motion,seed,method);job=assignments.get(k)
                    if job is None:rows.append(dict(job=dict(scenario=scene,motion_mode=motion,seed=seed,method=method),status='unavailable'));continue
                    row,c=load_trial(job,results.get(k),source,protocol,check);rows.append(row);context_rows.append(row);cache[method]=c
                reference=cache.get('ratio_effort');details=[]
                for method in ('fixed_effort','fusion_effort'):
                    candidate=cache.get(method)
                    if reference is None or candidate is None:details.append(dict(method=method,paired=False,reason='unavailable assigned member'));continue
                    a=reference['numeric'];b=candidate['numeric'];different=[]
                    for name in PHYSICAL:
                        clock=a['substep_time'] if name.startswith('substep_') else a['time'];mask=clock<7.5-1e-9
                        x,y=a[name][mask],b[name][mask];ok=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
                        check(str((scene,motion,seed,method))+'/pre75/'+name,ok)
                        if not ok:different.append(name)
                    sensor=reference['prefix_sensor_sha256']==candidate['prefix_sensor_sha256']
                    check(str((scene,motion,seed,method))+'/pre75_sensor_exact',sensor)
                    at=int(np.argmin(abs(a['time']-7.5)));instant=[]
                    for name in PRE:
                        x,y=a[name][at],b[name][at];ok=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
                        check(str((scene,motion,seed,method))+'/handoff_preaction/'+name,ok)
                        if not ok:instant.append(name)
                    details.append(dict(method=method,paired=not different and sensor and not instant,
                        pre75_physical_differences=different,pre75_sensor_byte_exact=sensor,handoff_preaction_differences=instant))
                same_config=len({json.dumps(row.get('physical_parameters'),sort_keys=True) for row in context_rows})==1
                same_noise=len({json.dumps(row.get('noise'),sort_keys=True) for row in context_rows})==1
                check(str((scene,motion,seed))+'/paired_physical_noise_config',same_config and same_noise)
                contexts.append(dict(scenario=scene,motion_mode=motion,seed=seed,comparisons=details,same_physical_noise=same_config and same_noise))
                del cache
                print(json.dumps(dict(contexts_checked=len(contexts),records_checked=len(rows),planned_records=48,
                    seed=seed,scenario=scene,motion_mode=motion)),flush=True)
    aggregate={}
    for method in reg['methods']:
        rr=[r for r in rows if r['job']['method']==method];complete=[r for r in rr if r['status']=='complete']
        aggregate[method]=dict(assigned=16,complete=len(complete),unavailable=len(rr)-len(complete),
            mission_evaluable=len(complete),
            physical_ready=sum(r['handoff_physical_ready'] for r in complete),legacy_B_supported=sum(r['legacy_model_domain_supported'] is True for r in complete),
            full_motion_success=sum(r['metrics_original']['full_motion_mission_success'] for r in complete),
            legacy_mission_success=sum(r['metrics_original']['mission_success'] for r in complete),
            full16_forceguard=sum(r['metrics_original']['safety']['guard_pass'] for r in complete),
            physical_release200Hz=sum(r['metrics_original']['release']['physical_success'] for r in complete),
            effort_available_decisions=sum(r['effort_available_decisions'] for r in complete),decision_slots=sum(r['decision_slots'] for r in complete),
            effort_numerical_failure_decisions=sum(r['effort_numerical_failure_decisions'] for r in complete),
            numerical_failure_reasons=dict(sum((Counter(r['numerical_failure_reasons']) for r in complete),Counter())),
            native_scalar_records=sum(len(r['native_scalar_audit']) for r in complete),
            native_scalar_records_valid=sum(sum(v['passed'] for v in r['native_scalar_audit']) for r in complete),
            control_ms_decision_grid=timing_stats(np.concatenate([np.asarray(r['control_ms_decision_grid']['values_ms']) for r in complete]) if complete else np.array([])),
            fusion_conditional_decisions=sum(r['fusion_conditional_decisions'] for r in complete),
            local_excess_sum_Ns=sum(r['metrics_original']['safety']['local_excess_integral_Ns'] for r in complete),
            quality_available={name:sum(r['metrics_original'][name]['eligible'] for r in complete) for name in ('static_quality','dynamic_quality','entire_holding_quality')})
    pair_metrics=[]
    indexed={key(r['job']):r for r in rows}
    for c in contexts:
        s,m,seed=c['scenario'],c['motion_mode'],c['seed'];base=indexed[(s,m,seed,'ratio_effort')]
        for method in ('fixed_effort','fusion_effort'):
            candidate=indexed[(s,m,seed,method)];valid=base['status']==candidate['status']=='complete'
            p=dict(scenario=s,motion_mode=m,seed=seed,candidate=method,available=valid)
            if valid:
                a=base['metrics_original'];b=candidate['metrics_original']
                p.update(local_excess_delta_Ns=b['safety']['local_excess_integral_Ns']-a['safety']['local_excess_integral_Ns'],
                    peak_delta_N=max(b['safety']['finger_peak_N'])-max(a['safety']['finger_peak_N']),
                    full_motion_outcome=[a['full_motion_mission_success'],b['full_motion_mission_success']],
                    force_guard_outcome=[a['safety']['guard_pass'],b['safety']['guard_pass']],
                    common_quality={name:bool(a[name]['eligible'] and b[name]['eligible']) for name in ('static_quality','dynamic_quality','entire_holding_quality')})
            pair_metrics.append(p)
    report=dict(schema='v12_native_complete48_readonly_acceptance_v1',passed=not issues,checks=checks,issues=issues,
        batch_status=inv,batches=batch_info,registered_records=48,contexts=contexts,physical_seed_clusters=2,
        rows=rows,aggregate_full_assigned=aggregate,same_context_comparisons=pair_metrics,
        input_registration_path=str(REG),expected_source_count=EXPECTED_SOURCE_COUNT,source_sha256=source,
        input_registration_sha256=sha(REG),checker_sha256=sha(Path(__file__)),helper_source_sha256=sha(ROOT/'review/verify_v12_pilot_v1.py'),
        provenance=dict(role='runtime author separate read-only acceptance; independent stored-metric readout performed by peer',
            prefix_scope='Same-context methods t<7.5 all40 physical columns/public packet bytes and t7.5 pre-action measurements; no old posthandoff equality claim',
            noise_scope='Same seed/noise config and prebranch packet exactness; fixed-draw sensor source shared. Postbranch noise innovations not independently regenerated.',
            npz_io='Each selected compressed column read once; three numeric caches percontext, no z[column][i] repeated decompression',
            native_scope='Every available new decision: saved achieved-minus-fixed-goal, canonical scalar residual/loss bytes and recorded candidate ranking cross-checked. No native model/plant is constructed; independent real-robot-law reevaluation is separately documented by the 675-input audit.',
            runtime_cost_scope='Median/p95/max of actual control_ms on active 20ms decision-grid samples only. Runtime includes shared public sampling/control/execution overhead; host load/wall-clock conditions can confound old/new speed comparison.',
            old_new_scope='This evidence audits the new48 alone. Old/new per-method trajectories and outcome differences require a separate explicit comparison; no old48 or new48 is pooled as independent96.',
            material_success='Original evaluation thresholds/NULL/failures preserved, source/IO pass never means forceguard or global safety pass'),
        new_physics_steps=0,performance_claim_allowed=False)
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False,default=helpers.scalar_json)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:f.write(payload)
    lines=['# V12-native完整48记录：运行接入只读验收','',f"检查{len(checks)-len(issues)}/{len(checks)}；问题{len(issues)}。16情境×3方法，只有2物理/噪声种子cluster；不作总体优势结论。",'',
        '|方法|完整/16|物理接管/16|旧B支持/16|全运动任务/16|全16s限力/16|原释放/16|effort可用/决策|numerical fail|conditional fusion|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method,a in aggregate.items():lines.append(f"|{method}|{a['complete']}|{a['physical_ready']}|{a['legacy_B_supported']}|{a['full_motion_success']}|{a['full16_forceguard']}|{a['physical_release200Hz']}|{a['effort_available_decisions']}/{a['decision_slots']}|{a['effort_numerical_failure_decisions']}|{a['fusion_conditional_decisions']}|")
    lines+=['','计算耗时只取20ms决策格点的实际control_ms，不以全部5ms采样稀释；包含公共采样/控制/执行开销，不是单独求解器基准。','',
        '|方法|决策样本|median ms|p95 ms|max ms|','|---|---:|---:|---:|---:|']
    for method,a in aggregate.items():
        c=a['control_ms_decision_grid'];lines.append(f"|{method}|{c['count']}|{c['median_ms']}|{c['p95_ms']}|{c['max_ms']}|")
    lines+=['','所有48原评价向量、NULL品质与失败按行保留；同条件配对共同可用窗口在JSON单列。effort数值失败不是物理无解、目标可用不是实际材料安全。末端释放与持握阶段成本仍按原评价分列。','',
        '本批与旧V12是同条件开发比较、不是新的独立确认。新基准同公共观测与原生机器人effort执行层；不是把旧B基准改名。完整原始后验性能解读由另一个只读metric报告处理，不临时改权重/阈值或筛成功。']
    with out.with_suffix('.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(out),passed=not issues,checks=len(checks),issues=len(issues),sha256=sha(out),new_physics_steps=0)))
    return 0 if not issues else 2


def self_test():
    expected={(s,m,k,method) for s in ('a','b','c','d') for m in ('hold','transport') for k in (1,2) for method in ('r','x','f')}
    assert len(expected)==48 and len({v[:3] for v in expected})==16
    t=np.array([7.495,7.5,7.505]);assert (t<7.5-1e-9).tolist()==[True,False,False]
    a=np.array([0.,-0.]);b=np.array([0.,0.]);assert a.tobytes()!=b.tobytes()
    assert all(k not in PRE for k in ('alpha','ctrl','phase'))
    flags={k:False for k in ('empirical_pressure_B_used','future_q_or_pressure_predicted','physical_safety_certified','global_model_admitted')}
    flags.update(true_step_reference_bounds_checked=True,true_actuator_port_bounds_checked=True)
    residual=np.arange(25,dtype=float)/100
    solutions=[];rechecks=[]
    for axis in range(8):
        subset=residual[[axis]];loss=float(subset@subset);candidate=dict(value=.1,loss=loss,distance_to_nominal=.1)
        solutions.append(dict(axis=axis,torque_support25=[axis],selected_native_residual=subset.tolist(),
            selected_native_residual_bytes_hex=subset.tobytes().hex(),selected_native_loss_bytes_hex=np.float64(loss).tobytes().hex(),
            candidates=[candidate],selected=candidate))
        rechecks.append(dict(axis=axis,residual_bytes_equal=True,loss_bytes_equal=True,native_actual_loss=loss))
    e=dict(status='KNOWN_CURRENT_STATE_EFFORT_LS_WITH_RESIDUAL',increment8=np.zeros(8).tolist(),port_target21=np.zeros(21).tolist(),
        reference8=np.zeros(8).tolist(),desired_torque25=np.zeros(25).tolist(),achieved_torque25=residual.tolist(),residual25=residual.tolist(),
        debug=dict(flags,scalar_solutions=solutions,scalar_native_recheck=rechecks,exact_zero_correction_nominal_passthrough=False))
    assert inspect_native_record(e)['passed']
    from copy import deepcopy
    broken=deepcopy(e);broken['debug']['scalar_solutions'][0]['selected_native_loss_bytes_hex']=np.float64(1.).tobytes().hex()
    assert not inspect_native_record(broken)['passed']
    broken=deepcopy(e);broken['achieved_torque25'][20]+=.001
    assert not inspect_native_record(broken)['passed']
    failure=dict(status='NUMERICAL_FAILURE',increment8=None,port_target21=None,debug=dict(flags,reason='retained real numerical failure'))
    check=inspect_native_record(failure);assert check['passed'] and not check['available'] and check['category']=='numerical_failure'
    broken=deepcopy(failure);broken['increment8']=[0.]*8;assert not inspect_native_record(broken)['passed']
    c=timing_stats([2.,3.,4.]);assert c['count']==3 and c['median_ms']==3. and abs(c['p95_ms']-3.9)<1e-12
    assert timing_stats([])['median_ms'] is None
    ticks=7.5+np.arange(900)*.005
    decision=abs((ticks-7.5)/.02-np.round((ticks-7.5)/.02))<1e-7
    assert int(decision.sum())==225
    assert timing_stats([float('nan'),-1.,2.])['finite_nonnegative_count']==1
    return dict(passed=True,checks=13,actual_data_read=False,new_physics_steps=0,
        cases=['48 matrix','preaction mask','signed-zero bytes','preaction fields','native scalar bytes',
            'incorrect loss bytes rejected','incorrect final goal residual rejected','numerical failure retained',
            'asymmetric available rejected','decision cost summary','missing cost unknown','225 decision grid','invalid cost not hidden'])


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--inventory',action='store_true');g.add_argument('--self-test',action='store_true');g.add_argument('--execute',action='store_true')
    p.add_argument('--registration',type=Path,default=REG)
    p.add_argument('--directories',nargs=4,type=Path,metavar=('PILOT3','HOLD901_9','HOLD902_12','TRANSPORT24'),help='Four complete batch paths, in registered group order')
    p.add_argument('--expected-source-count',type=int,default=80)
    p.add_argument('--out',type=Path,default=ROOT/'review/v12_native_full48_readonly_acceptance_v1.json');a=p.parse_args()
    REG=a.registration.resolve();EXPECTED_SOURCE_COUNT=a.expected_source_count
    DIRECTORIES=[p.resolve() for p in a.directories] if a.directories else None
    if a.self_test:print(json.dumps(self_test(),ensure_ascii=False))
    elif a.inventory:
        batches=inventory(read(REG));print(json.dumps(dict(status='pending' if any(x['status']=='pending' for x in batches) else 'all_terminal',batches=batches),ensure_ascii=False))
    else:raise SystemExit(main_execute(a.out.resolve()))

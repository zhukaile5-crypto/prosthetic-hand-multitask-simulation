"""Read-only V12-native pilot acceptance, zero new dynamics.

The runtime author supplies this separate verifier. It is not proof of physical
safety or algorithm superiority; root/peer reviews remain separate evidence.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD_ROOT=ROOT/'output/development_v12_pilot_v1'
REG=ROOT/'review/v12_native_development_registration_v1.json'
spec=importlib.util.spec_from_file_location('native48_pilot_helpers',ROOT/'review/verify_v12_native_full48_v1.py')
native_helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(native_helpers)
METHODS=('ratio_effort','fixed_effort','fusion_effort')
PHYSICAL=('time','normal','skin_normal','shear','palm','hand_other','all_hand_object','support','position',
    'relative','rotation','relative_velocity','angular_velocity','lift','lift_ref','wrist_position',
    'requested_wrist_reference','qpos','qvel','ctrl_before','ctrl','alpha_before','alpha','executed_action',
    'warning_count','collision','actual_motion_target','executed_motion_reference','penetration','world_penetration',
    'substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm','substep_hand_other',
    'substep_all_hand_object','substep_support','substep_penetration','substep_world_penetration')
HANDOFF_PRE=('qpos','qvel','ctrl_before','alpha_before','normal','skin_normal','lift','wrist_position')


def required_columns():
    return set(PHYSICAL)|{'metadata','legacy_common_sensor_json','observation_json','debug_json','requested_action',
        'controller_active','prediction_available','predicted_normal','prediction_at','tactile_acquisition_time',
        'estimated_normal','validity','estimated_load_available','estimated_load','auxiliary_support_available',
        'auxiliary_support','contact_model_available','contact_model_active','posterior_mean_time_s','control_ms'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def equal(a,b):
    if isinstance(a,dict) or isinstance(b,dict):
        return isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list) or isinstance(b,list):
        return isinstance(a,list) and isinstance(b,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and isinstance(b,float) and np.isnan(a) and np.isnan(b):return True
    return a==b


def sensor_subset_equal(common,old):
    # New JSON is captured immediately after common skin acquisition and before
    # V12 additions/model stripping. Old contains four later public extras.
    if not set(common).issubset(old):return False
    # Preserve numeric JSON representation too (-0.0 is not silently 0.0).
    return json.dumps(common,sort_keys=True,separators=(',',':'))==json.dumps(
        {k:old[k] for k in common},sort_keys=True,separators=(',',':'))


def scalar_json(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)


def execute(run,out):
    run=run.resolve();out=out.resolve()
    if not run.is_relative_to((ROOT/'output').resolve()):raise ValueError('Research/output run required')
    marker=run/'execution_complete.json'
    if not marker.is_file():raise RuntimeError('NOT_READY: no terminal marker; no partial trace read')
    finish=read(marker)
    if not finish.get('ended_utc') or not finish.get('trials_sha256'):
        raise RuntimeError('NOT_READY: final runner ledger required')
    if out.exists() or not out.is_relative_to((ROOT/'review').resolve()):raise ValueError('Fresh review file required')
    manifest=read(run/'manifest.json');plan=read(run/'plan.json');trials=read(run/'trials.json')
    jobs=manifest['jobs'];sources=manifest['research_source_sha256'];snap=read(run/'source_snapshot_manifest.json')
    checks=[];issues=[];rows=[]
    def check(name,ok,detail=None):
        c=dict(name=name,passed=bool(ok),detail=detail);checks.append(c)
        if not ok:issues.append(c)
    check('terminal3_retained',finish.get('execution_complete') is True and finish.get('planned')==3 and finish.get('trials')==3)
    check('source_execution_unchanged',finish.get('sources_unchanged') is True)
    check('trials_terminal_hash',sha(run/'trials.json')==finish['trials_sha256'])
    check('exact_predeclared_pilot_assignment',len(jobs)==3 and {j['method'] for j in jobs}==set(METHODS)
        and all(j['scenario']=='nominal' and j['seed']==617901 and j['motion_mode']=='hold' and j['split']=='robustness' for j in jobs))
    check('no_empirical_servo_bank_sources',not any('servo_bank' in k or 'robot_servo_response_' in k for k in sources))
    check('source_snapshot_set_exact',set(snap['files'])==set(sources))
    for key,record in snap['files'].items():
        p=run/record['snapshot_path'];check('source_snapshot/'+key,p.is_file() and sha(p)==record['sha256']==sources[key])
    old_paths=[]
    registration=read(REG)
    check('exact_registered_source80',len(sources)==80 and sources==registration['source_sha256'])
    for job in jobs:
        OLD=OLD_ROOT/'runs'/('nominal__'+job['method']+'__617901')/'trace.npz'
        old_paths.append(OLD)
        with np.load(OLD,allow_pickle=False) as old_archive:
            old={k:old_archive[k] for k in PHYSICAL+('legacy_common_sensor_json',)};oldmeta=json.loads(str(old_archive['metadata']))
        mechanical_packet=json.loads(str(old['legacy_common_sensor_json'][0]))
        mechanical_lower=np.asarray(mechanical_packet['closure_min']);mechanical_upper=np.asarray(mechanical_packet['closure_max'])
        folder=Path(job['destination']);path=folder/'trace.npz'
        result=next((x for x in trials if x['method']==job['method'] and x['seed']==job['seed'] and x['scenario']==job['scenario']),None)
        if result is None or not path.is_file():
            check(job['method']+'/complete_assigned_trial',False,'assigned failure/missing trace retained')
            rows.append(dict(method=job['method'],status='unavailable',job=job));continue
        check(job['method']+'/run_result_exact',equal(result,read(folder/'result.json')))
        check(job['method']+'/saved_trace_hash',sha(path)==read(folder/'metadata.json')['trace_sha256'])
        with np.load(path,allow_pickle=False) as archive:
            z={name:archive[name] for name in required_columns()}
            meta=json.loads(str(z['metadata']));t=z['time'];st=z['substep_time'];dur=z['substep_duration']
            check(job['method']+'/full16s_frame_and_raw_coverage',len(t)==3201 and len(st)==32000 and
                np.allclose(t,np.arange(3201)*.005,rtol=0,atol=1e-8) and np.allclose(st,np.arange(32000)*.0005,rtol=0,atol=1e-7)
                and np.allclose(dur,.0005,rtol=0,atol=1e-9))
            check(job['method']+'/trial_source_lock_exact',meta['research_source_sha256']==sources)
            check(job['method']+'/same_physical_configuration',meta['config']==oldmeta['config'])
            before=t<7.5-1e-9;pre_diffs=[]
            for field in PHYSICAL:
                clock=st if field.startswith('substep_') else t
                mask=clock<7.5-1e-9
                a,b=z[field][mask],old[field][mask]
                exact=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
                check(job['method']+'/pre75/'+field,exact)
                if not exact:pre_diffs.append(field)
            common=[json.loads(str(v)) for v in z['legacy_common_sensor_json'][before]]
            oldobs=[json.loads(str(v)) for v in old['legacy_common_sensor_json'][before]]
            sensor_bad=[i for i,(a,b) in enumerate(zip(common,oldobs)) if not sensor_subset_equal(a,b)]
            check(job['method']+'/common_sensor_packets_pre75_exact',len(common)==len(oldobs) and not sensor_bad,sensor_bad[:10])
            at=int(np.argmin(abs(t-7.5)))
            for field in HANDOFF_PRE:
                a,b=z[field][at],old[field][at]
                check(job['method']+'/75_preaction/'+field,a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes())
            check(job['method']+'/75_preaction_public_sensor',sensor_subset_equal(
                json.loads(str(z['legacy_common_sensor_json'][at])),json.loads(str(old['legacy_common_sensor_json'][at]))))
            check(job['method']+'/same_noise_and_gains',meta['noise']==oldmeta['noise'] and meta['actual_ratio_task_gains']==oldmeta['actual_ratio_task_gains'])
            check(job['method']+'/physical_ready_separated_from_old_model_domain',meta['handoff_passed'] is True and meta['legacy_model_domain_supported'] is False)
            observations=[json.loads(str(v)) for v in z['observation_json']]
            policy=(t>=7.5-1e-9)&(t<12.-1e-9)&z['controller_active'].astype(bool)
            debug=[json.loads(str(v)) for v in z['debug_json']]
            # A fresh planner/intent call every20ms, four actuator ticks per plan.
            decision=policy&(np.abs((t-7.5)/.02-np.round((t-7.5)/.02))<1e-7)
            drows=[debug[i] for i in np.flatnonzero(decision)]
            effort_status=Counter(str(d.get('effort_tracking',{}).get('status','unavailable')) for d in drows)
            # The immutable result's available PROPERTY is not serialized by
            # to_dict(vars). Use its exact definition, not a missing bool key.
            effort_available=sum(d.get('effort_tracking',{}).get('increment8') is not None and
                d.get('effort_tracking',{}).get('port_target21') is not None for d in drows)
            native_audit=[native_helpers.inspect_native_record(d.get('effort_tracking',{})) for d in drows]
            native_bad=[dict(decision_index=i,detail=v) for i,v in enumerate(native_audit) if not v['passed']]
            check(job['method']+'/all_available_native_scalar_residual_loss_bytes',not native_bad,native_bad[:3])
            failure_reasons=Counter(str(d.get('effort_tracking',{}).get('debug',{}).get('reason','missing_reason'))
                for d in drows if d.get('effort_tracking',{}).get('status')=='NUMERICAL_FAILURE')
            costs=native_helpers.timing_stats(z['control_ms'][decision])
            check(job['method']+'/decision_grid_cost_finite',costs['count']==costs['finite_nonnegative_count'])
            allocation=sum(d.get('force_intent',{}).get('fusion_mode')=='conditional_object_contact_allocation' for d in drows)
            action_error=abs(z['requested_action'][policy]-z['executed_action'][policy])
            max_action_error=float(action_error.max()) if action_error.size else None
            check(job['method']+'/holding900ticks225decision_slots',int(policy.sum())==900 and int(decision.sum())==225)
            pre_reference=np.asarray([f['executed_motion_reference'] for f in observations],float)
            delta8=np.column_stack((z['executed_action'],z['executed_motion_reference']-pre_reference))
            observed_rate_max=(np.max(abs(delta8[policy]),axis=0)/.005).tolist() if policy.any() else None
            limits=np.asarray([f['motion_reference_rate_limits'] for f in observations],float)
            valid_rates=bool((delta8[policy,:5]<=.45*.005+1e-10).all() and
                (delta8[policy,:5]>=-.60*.005-1e-10).all() and
                (abs(delta8[policy,5:])<=limits[policy]*.005+1e-10).all())
            check(job['method']+'/actual_holding_reference_rates',valid_rates)
            release=t>=12.-1e-9
            check(job['method']+'/release_only_opens_with_native_rate',bool((z['executed_action'][release]<=1e-10).all() and
                (z['executed_action'][release]>=-.8*.005-1e-10).all()))
            check(job['method']+'/no_hidden_control_clip',meta.get('max_hidden_ctrl_clip') is not None and float(meta['max_hidden_ctrl_clip'])<=1e-12)
            check(job['method']+'/actual_request_realized_to_roundoff',max_action_error is not None and max_action_error<=1e-10)
            permission=np.asarray([f['closure_permission'] for f in observations],bool)
            check(job['method']+'/no_closing_through_observed_blocked_permission',not np.any((z['executed_action'][policy]>1e-10)&~permission[policy]))
            check(job['method']+'/all_predictions_unavailable',not z['prediction_available'].any() and np.isnan(z['predicted_normal']).all() and np.isnan(z['prediction_at']).all())
            stamp=z['tactile_acquisition_time'];estimated=z['estimated_normal'];inputvalid=z['validity'].astype(bool)
            unavailable=~np.isfinite(stamp)|~inputvalid
            check(job['method']+'/unknown_pressure_is_nan_not_zero',np.isnan(estimated[unavailable]).all())
            known=np.isfinite(estimated)
            check(job['method']+'/measurement_acquisition_is_causal',np.isfinite(stamp[known]).all() and
                np.all(stamp[known]<=np.broadcast_to(t[:,None],stamp.shape)[known]+1e-8))
            load_available=z['estimated_load_available'].astype(bool)
            check(job['method']+'/unknown_load_is_nan',np.isnan(z['estimated_load'][~load_available]).all())
            aux_available=z['auxiliary_support_available'].astype(bool)
            check(job['method']+'/unknown_aux_support_is_nan',np.isnan(z['auxiliary_support'][~aux_available]).all())
            check(job['method']+'/old_model_mask_unavailable',not z['contact_model_available'].any() and not z['contact_model_active'].any() and np.isnan(z['posterior_mean_time_s']).all())
            check(job['method']+'/no_legacy_model_inputs_in_policy_packet',not any(set(f)&{'force_matrix','force_tau','force_gain','model_reference_alpha','response_error_bound'} for f in observations))
            check(job['method']+'/21_actual_engine_telemetry_current_clock',all(len(f['robot_actuator_target21'])==21 and abs(f['robot_control_time_s']-float(t[i]))<1e-8 for i,f in enumerate(observations)))
            check(job['method']+'/native_release_prediction_remains_unknown',all(debug[i].get('release_authority') is True and not np.isfinite(z['predicted_normal'][i]).any() for i in np.flatnonzero(t>=12.-1e-9)))
            active_bad=[int(i) for i in np.flatnonzero(policy) if
                not np.array_equal(np.asarray(observations[i]['closure_min']),mechanical_lower) or
                not np.array_equal(np.asarray(observations[i]['closure_max']),mechanical_upper)]
            check(job['method']+'/permission_bounds_are_mechanical_not_old_B',not active_bad,active_bad[:10])
            metrics=result['research_metrics']
            rows.append(dict(method=job['method'],status='complete',trace_sha256=sha(path),job=job,
                handoff_physical_ready=meta['handoff_passed'],legacy_model_domain_supported=meta['legacy_model_domain_supported'],
                pre75_physical_differences=pre_diffs,pre75_sensor_mismatched_packets=sensor_bad,
                policy_ticks=int(policy.sum()),decision_slots=int(decision.sum()),effort_available_decisions=effort_available,
                effort_status_decisions=dict(effort_status),fusion_conditional_allocation_decisions=allocation,
                native_scalar_consistency=native_audit,numerical_failure_reasons=dict(failure_reasons),
                control_ms_decision_grid=costs,
                native_action_max_requested_vs_executed_delta=max_action_error,
                actual_max_abs_reference_rate8=observed_rate_max,
                rate_units8=['alpha/s']*5+['m/s','rad/s','rad/s'],
                actual_mechanical_lower5=mechanical_lower.tolist(),actual_mechanical_upper5=mechanical_upper.tolist(),
                late_projection_ticks=sum(bool(debug[i].get('execution_changed_plan')) for i in np.flatnonzero(policy)),
                metadata_hidden_ctrl_clip=meta.get('max_hidden_ctrl_clip'),
                unknown_pressure_cells=int(unavailable.sum()),finite_pressure_cells=int(known.sum()),
                source_state='same frozen new runtime/observer/actor; not exact old post-handoff baseline',
                old_compatibility_mask_scope='allfalse model mask is unavailable, observed_contact is separate',
                actual_engine_mapping_scope='axes21/clock checked from packet; no model recompile in this reader',
                original_evaluation=dict(task_success=result['task_success'],metrics=metrics)))
    report=dict(schema='v12_native_pilot_v1_readonly_acceptance',passed=not issues,checks=checks,issues=issues,rows=rows,
        planned=3,physical_seed_clusters=1,phase='development_single_seen_context',new_physics_steps=0,
        role='separate read-only verification by runtime author, not independent physical replication or algorithm efficacy proof',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (marker,run/'manifest.json',run/'plan.json',run/'trials.json',REG,*old_paths)},
        checker_sha256=sha(Path(__file__)),native_helper_sha256=sha(ROOT/'review/verify_v12_native_full48_v1.py'),source_sha256=sources,
        comparison_scope='t<7.5 all40 physical plus raw public sensor packets; t=7.5 pre-action q/v/ctrl_before/alpha_before and measurements only; post-action difference intentional',
        limitations=['No new mission/material/tracking threshold or aggregate score is introduced; metrics read exactly from frozen evaluator.',
            'A single seed/nominal hold pilot checks integration and exposure, not comprehensive fusion advantage.',
            'Effort availability is software/model availability, not measured pressure response or material safety.',
            'Native release remains a shared physical cost; late clip and unavailable observations remain explicit.'])
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False,default=scalar_json)
    lines=['# V12-native 三法 pilot 只读验收','',f"检查 {len(checks)-len(issues)}/{len(checks)}；问题 {len(issues)}。实际物理重跑0。",
        '', '|方法|物理接管|旧B域|holding ticks|effort可用/decision|fusion conditional|完整任务|全程力限|超限N·s|峰N|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['status']!='complete':lines.append('|'+r['method']+'|NULL|NULL|NULL|NULL|NULL|0|NULL|NULL|NULL|');continue
        m=r['original_evaluation']['metrics'];s=m['safety']
        lines.append(f"|{r['method']}|{int(r['handoff_physical_ready'])}|{int(r['legacy_model_domain_supported'])}|{r['policy_ticks']}|{r['effort_available_decisions']}/{r['decision_slots']}|{r['fusion_conditional_allocation_decisions']}|{int(m['full_motion_mission_success'])}|{int(s['guard_pass'])}|{s['local_excess_integral_Ns']:.8g}|{max(s['finger_peak_N']):.8g}|")
    lines+=['','完整原评价（含NULL品质、运动与稳定时间）保留在JSON；不另定阈值、不把effort模型可用当压力安全。7.5s前共用流程逐值比较；7.5s只比较动作前状态，旧V12与新native在接管后使用不同求解算术，后续动作不要求相同。','',
        '本验收由运行层作者提供，独立peer/root证据另列；单seed名义hold接入验收不能证明完整fusion综合优势。']
    with out.with_suffix('.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(out),passed=not issues,checks=len(checks),issues=len(issues),sha256=sha(out))))


def self_test():
    assert equal(dict(x=float('nan')),dict(x=float('nan')))
    assert sensor_subset_equal(dict(a=[1.,2.]),dict(a=[1.,2.],new_extra=0))
    assert not sensor_subset_equal(dict(a=[1.,3.]),dict(a=[1.,2.]))
    t=np.array([7.495,7.5,7.505]);assert (t<7.5-1e-9).tolist()==[True,False,False]
    # A changed post-action control at handoff is not among pre-action fields.
    assert 'ctrl' not in HANDOFF_PRE and 'alpha' not in HANDOFF_PRE and 'phase' not in HANDOFF_PRE
    return dict(passed=True,checks=5,new_physics_steps=0,actual_results_read=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--self-test',action='store_true');g.add_argument('--execute',action='store_true')
    p.add_argument('--run',type=Path,default=ROOT/'output/development_v12_native_pilot_v1')
    p.add_argument('--out',type=Path,default=ROOT/'review/v12_native_pilot_v1_readonly_acceptance_v1.json')
    a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()))
    else:execute(a.run,a.out)

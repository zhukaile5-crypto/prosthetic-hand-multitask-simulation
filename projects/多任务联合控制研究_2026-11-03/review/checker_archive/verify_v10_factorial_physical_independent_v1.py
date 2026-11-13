"""Read saved complete V10 hold20+transport20 only; never imports a plant.

Pure V3/raw arithmetic comes from the previous independent V9 reviewer, pinned
by exact bytes. Added V10 checks cover retained baselines, typed robot geometry
evidence and all142 actual copied sources including62 empirical model artifacts.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import runpy
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PURE=ROOT/'review/verify_v9_factorial_independent.py'
PURE_SHA='8089aac7d9a474351016a82a4c58b3c23d40c867a2cb148968134c27cdcdcad7'
METHODS=('ratio','passive','ratio_preview','passive_preview','joint')
BASELINES=METHODS[:-1]
SCENES=('nominal','friction_drop','fragile_conflict','collision')
NUMERIC_EXCLUSIONS={'control_ms'}
PHYSICAL=('time','normal','skin_normal','shear','palm','hand_other','all_hand_object','support','position','relative','rotation','relative_velocity','angular_velocity','lift','lift_ref','wrist_position','requested_wrist_reference','qpos','qvel','ctrl_before','ctrl','alpha_before','alpha','executed_action','warning_count','collision','actual_motion_target','executed_motion_reference','penetration','world_penetration','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm','substep_hand_other','substep_all_hand_object','substep_support','substep_penetration','substep_world_penetration')
EXTRA=('controller_active','transport_paused','prediction_available','predicted_normal','prediction_at','model_invalid','observation_unavailable','validity','local_domain_lower','local_domain_upper','observed_normal','observed_age','estimated_normal','observer_uncertainty','contact_model_active','estimated_load','auxiliary_support','posterior_mean_time_s','requested_action')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def numeric_keys(path):
    keys=set()
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                version=np.lib.format.read_magic(stream)
                reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:raise ValueError('Unknown NPY version')
                _,_,dtype=reader(stream)
            if dtype.kind in 'biufc':keys.add(name.removesuffix('.npy'))
    return keys


def bit_equal(x,y):return x.dtype==y.dtype and x.shape==y.shape and x.tobytes(order='C')==y.tobytes(order='C')


def expected_model_data():
    model='output/robot_servo_response_all15_v2';short='output/robot_servo_short_all15_v1'
    expected={model+'/execution_complete.json',short+'/execution_complete.json'}
    for shape in ('capsule','box','sphere'):
        for i in range(5):
            cid=f'{shape}/checkpoint_{i:02d}'
            for name in ('model.json','robot_response_raw.npz','actual_input_footprints.json'):expected.add(model+'/'+cid+'/'+name)
            expected.add(short+'/'+cid+'/comparison.json')
    return expected


def load_pure():
    if sha(PURE)!=PURE_SHA:raise ValueError('Independent arithmetic source changed')
    return runpy.run_path(str(PURE))


def require_terminal(modes):
    # Read completion flags first; never inspect partial trial performance.
    for mode in modes:
        run=ROOT/f'output/development_v10_{mode}_v1';finish=run/'execution_complete.json'
        if not finish.is_file() or read(finish).get('execution_complete') is not True:
            raise RuntimeError('WAIT_FOR_COMPLETE_BOTH_MODES:'+mode)


def baseline_pair(new_path,old_path):
    nk=numeric_keys(new_path)-NUMERIC_EXCLUSIONS;ok=numeric_keys(old_path)-NUMERIC_EXCLUSIONS
    differences=[];hashes={};extra_fields=sorted(nk-ok);missing=sorted(ok-nk)
    with np.load(new_path,allow_pickle=False) as a,np.load(old_path,allow_pickle=False) as b:
        for field in sorted(nk&ok):
            x=a[field];y=b[field]
            hashes[field]=hashlib.sha256(x.tobytes(order='C')).hexdigest()
            if not bit_equal(x,y):differences.append(dict(field=field,shape_new=x.shape,shape_old=y.shape,numeric_equal=bool(np.array_equal(x,y,equal_nan=True))))
        physical_exact=all(field in nk&ok and bit_equal(a[field],b[field]) for field in PHYSICAL)
        observation=[];old_packets=b['observation_json'];new_packets=a['observation_json'];qvel=a['qvel']
        for i,(left,right) in enumerate(zip(new_packets,old_packets)):
            n=json.loads(str(left));o=json.loads(str(right));extra=set(n)-set(o);missing_obs=set(o)-set(n)
            lift=n.pop('lift_velocity_m_s',None)
            if extra!={'lift_velocity_m_s'} or missing_obs or n!=o or lift!=float(qvel[i,0]):
                observation.append(dict(index=i,extra_keys=sorted(extra),missing_keys=sorted(missing_obs),legacy_fields_equal=n==o,lift_velocity_matches_current_robot_qvel=lift==float(qvel[i,0])))
        packets={field:bool(np.array_equal(a[field],b[field])) for field in ('raw_patches_json','environment_json')}
    return dict(common_numeric_fields=len(nk&ok),all_shared_numeric_arrays_bit_exact=not differences,numeric_field_set_equal=nk==ok,
        extra_numeric_fields=extra_fields,missing_numeric_fields=missing,mismatches=differences,physical40_fields_bit_exact=physical_exact,
        common_array_sha256=hashes,legacy_public_observation_fields_equal_except_added_lift_velocity=not observation and len(new_packets)==len(old_packets),
        observation_mismatches=observation[:20],observation_mismatch_count=len(observation),physical_packet_strings_exact=packets,
        added_observation_scope='public current robot lift encoder velocity only, no extra RNG draws/object state; metadata identity changes are expected')


def joint_evidence(z,check,key):
    t=z['time'];ids=np.flatnonzero((t>=7.5-1e-9)&(t<12.-1e-9)&z['controller_active'].astype(bool))
    debug={int(i):json.loads(str(z['debug_json'][i])) for i in ids}
    decisions=[int(i) for i in ids if abs((float(t[i])-7.5)/.02-round((float(t[i])-7.5)/.02))<1e-7]
    status_ticks=Counter(d.get('status','missing') for d in debug.values());status_decisions=Counter(debug[i].get('status','missing') for i in decisions)
    active=[i for i in decisions if debug[i].get('status')=='EXPERIMENTAL_DIRECTIONAL_FINITE_RAMP']
    check(key+'/joint_force_forecast_unavailable',not z['prediction_available'].any() and np.isnan(z['predicted_normal']).all())
    check(key+'/joint_no_extra_binary_risk_freeze',all(d.get('extra_motion_freeze') is False for d in debug.values()))
    check(key+'/joint_debug_actions_equal_executed',all(np.allclose(d.get('actual_subinterval_delta_closure'),z['executed_action'][i],atol=1e-10,rtol=0) for i,d in debug.items()))
    check(key+'/joint_pause_is_retained_stabilizer_pause',all(bool(z['transport_paused'][i])==bool(d.get('pause_transport',False)) and not d.get('stabilizer_pause_replaced_by_directional_plan',False) for i,d in debug.items()))
    intervals=[]
    for i in active:
        d=debug[i];end=i+4;delta=np.asarray(d['applied_input_increment_8'],float)
        obs=json.loads(str(z['observation_json'][i]));start=np.r_[z['alpha_before'][i],obs['executed_motion_reference']]
        actual=np.column_stack((z['alpha'][i:end],z['executed_motion_reference'][i:end]))
        wanted=start+np.arange(1,5)[:,None]/4*delta
        mismatch=float(np.max(abs(actual-wanted)));changed=any(debug[j].get('planned_finite_ramp_realization_changed',False) for j in range(i,end))
        scope=d.get('candidate_future_component_check',{});allocation=d.get('allocation',{});selection=d.get('servo_selection',{})
        check(key+f'/active_{t[i]:.3f}/realized_finite_ramp_or_reported_change',mismatch<=1e-9 or changed)
        check(key+f'/active_{t[i]:.3f}/not_claimed_force_or_global_certificate',allocation.get('pressure_safety_certified') is False and allocation.get('global_model_admitted') is False and selection.get('global_model_admitted') is False and d.get('pressure_prediction_available') is False and d.get('actual_baseline_future_sequence_predicted') is False)
        check(key+f'/active_{t[i]:.3f}/candidate_component_scope_and_checked_matrix',scope.get('status')=='EMPIRICAL_COMPONENT_CANDIDATE' and allocation.get('independently_checked') is True and allocation.get('real_bound_primal_residual',np.inf)<=1e-9 and d.get('candidate_finite_fk_accepted') is True)
        check(key+f'/active_{t[i]:.3f}/arbitrary_mixed_not_certified',d.get('empirical_l1_proposal_region',{}).get('certifies_arbitrary_mixed') is False)
        before=d.get('exact_model_geometry_cost_before');after=d.get('exact_model_geometry_cost_after')
        check(key+f'/active_{t[i]:.3f}/reported_model_geometric_cost_improved',after is not None and (before is None or after<before-1e-16))
        intervals.append(dict(start_s=float(t[i]),context_id=selection.get('context_id'),actual_ramp_max_abs_reference_error=mismatch,
            changed_realization_reported=changed,model_geometric_cost_before=before,model_geometric_cost_after=after,
            robot_state_distance=selection.get('robot_state_distance'),reference_outside_component_footprint=selection.get('reference_outside_component_footprint'),
            nearest_recorded_mixed_point_normalized_distance=selection.get('nearest_recorded_mixed_point_normalized_distance'),
            original_user_target3=d.get('original_user_target3'),future_component_scope=scope.get('status'),
            force_change_over_next20ms_N=(z['skin_normal'][i+4]-z['skin_normal'][i]).tolist(),
            interpretation='conditional actual subsequent force change; not a causal gradient or force prediction'))
    return dict(sensor_ticks=len(ids),decision_ticks=len(decisions),status_sensor_tick_counts=dict(status_ticks),status_decision_counts=dict(status_decisions),
        active_intervals=len(active),active_sensor_ticks=status_ticks.get('EXPERIMENTAL_DIRECTIONAL_FINITE_RAMP',0),
        late_permission_or_bound_override_ticks=sum(bool(d.get('planned_finite_ramp_realization_changed',False)) for d in debug.values()),
        reference_memory_rollback_ticks=sum(bool(d.get('reference_memory_rollback',False)) for d in debug.values()),
        reported_pause_ticks=int(z['transport_paused'][ids].sum()),intervals=intervals,
        model_selection_scope='empirical anchor/input/mixed-point diagnostics only; no force/novel-contact/sequence certification')


def audit_mode(mode,pure,check):
    run=ROOT/f'output/development_v10_{mode}_v1';old=ROOT/f'output/development_v9_{mode}_v1'
    finish=read(run/'execution_complete.json');rows=read(run/'trials.json');manifest=read(run/'manifest.json');plan=read(run/'plan.json');ledger=read(run/'assignment_ledger.json');snap=read(run/'source_snapshot_manifest.json')
    key=lambda x:(x['scenario'],x['method'],x['seed'])
    expected={(s,m,610212) for s in SCENES for m in METHODS};assigned={key(x):x for x in manifest['jobs']};bykey={key(x):x for x in rows}
    check(mode+'/exact20_assignments_results_and_ledger',len(rows)==len(ledger)==len(manifest['jobs'])==20 and set(assigned)==set(bykey)==expected and Counter(key(x['job']) for x in ledger)==Counter(assigned.keys()) and all(x['status']=='evaluated' for x in ledger))
    check(mode+'/terminal_hash_links',finish['trials']==finish['planned']==20 and not finish['runtime_failures'] and finish['sources_unchanged'] and all(finish[k]==sha(run/name) for k,name in [('trials_sha256','trials.json'),('assignment_ledger_sha256','assignment_ledger.json'),('plan_sha256','plan.json'),('manifest_sha256','manifest.json')]))
    check(mode+'/development_scope_and_registered_factorial',plan['motion_mode']==mode and set(plan['methods'])==set(METHODS) and set(plan['scenarios'])==set(SCENES) and plan['seeds']==[610212] and plan['physical_randomization'] is True and plan['performance_claim_allowed'] is False and finish['performance_claim_allowed'] is False)
    check(mode+'/source_snapshot_manifest_link',manifest['source_snapshot_manifest_sha256']==sha(run/'source_snapshot_manifest.json') and manifest['plan_sha256']==sha(run/'plan.json'))
    source_records=[]
    for name,h in manifest['research_source_sha256'].items():
        entry=snap['files'][name];copy=run/entry['snapshot_path'];origin=Path(entry['original_path'])
        matches=copy.resolve().is_relative_to((run/'sources').resolve()) and sha(copy)==sha(origin)==h==entry['sha256'] and copy.stat().st_size==origin.stat().st_size==entry['bytes']
        source_records.append(dict(name=name,sha256=h,snapshot_path=str(copy),original_path=str(origin),copied_matches=matches))
    check(mode+'/all142_actual_source_copies_and_current_hashes',len(source_records)==142 and all(x['copied_matches'] for x in source_records))
    copied={p.relative_to(run/'sources').as_posix() for p in (run/'sources').rglob('*') if p.is_file()}
    check(mode+'/exact_snapshot_file_set',set(snap['files'])==set(manifest['research_source_sha256'])==copied)
    check(mode+'/all62_empirical_model_data_truly_copied',expected_model_data()<=copied and len(expected_model_data())==62)
    protocol=read(run/'sources/protocol.json');evaluate=runpy.run_path(str(run/'sources/evaluation.py'))['evaluate_arrays']
    check(mode+'/no_physics_runtime_import',not any(n=='mujoco' or n.startswith('mujoco.') for n in sys.modules))
    records=[];baselines=[];metadata={};paths={}
    wanted=set(PHYSICAL+EXTRA)|{'debug_json','observation_json','raw_patches_json','environment_json'}
    for identity in sorted(expected):
        scenario,method,seed=identity;row=bykey[identity];folder=Path(assigned[identity]['destination']);path=folder/'trace.npz';m=read(folder/'metadata.json');tag=mode+'/'+scenario+'/'+method
        metadata[(scenario,method)]=m;paths[(scenario,method)]=path
        with np.load(path,allow_pickle=False) as archive:
            z={k:archive[k].copy() for k in wanted if k in archive.files};inline=json.loads(str(archive['metadata']))
        check(tag+'/source_trace_meta_result_identity',str(path)==row['trace_path'] and sha(path)==row['trace_sha256']==m['trace_sha256'] and inline=={k:v for k,v in m.items() if k!='trace_sha256'} and read(folder/'result.json')==row)
        check(tag+'/metadata_parameters_and_sources',m['method']==method and m['scenario']['name']==scenario and m['seed']==seed and m['motion_mode']==mode and m['research_source_sha256']==manifest['research_source_sha256'] and m['config']==row['research_metrics']['physical_parameters'] and m['observer_warmup_s']==.2)
        u=np.random.default_rng(seed+511000).normal(size=4)
        check(tag+'/actual_randomization_and_material_contract',m['config']['mass']==float(.12*np.clip(1+.06*u[0],.85,1.15)) and m['config']['friction']==float(.6*np.clip(1+.08*u[1],.8,1.2)) and m['limits']==([3.5,2.8,2.8,2.8,2.8] if scenario=='fragile_conflict' else [5.]*5) and m['budget']==(12. if scenario=='fragile_conflict' else 18.))
        t=z['time'];st=z['substep_time'];dur=z['substep_duration'];dt=float(m['physics_timestep'])
        check(tag+'/complete_clocks_and_finite_actual_record',len(t)==3201 and np.allclose(t,np.arange(3201)*.005,atol=1e-8,rtol=0) and len(st)==round(16/dt) and np.allclose(st,np.arange(len(st))*dt,atol=1e-7,rtol=0) and np.allclose(dur,dt,atol=1e-12,rtol=0) and all(np.isfinite(z[k]).all() for k in ('normal','skin_normal','qpos','qvel','alpha','ctrl','executed_action','substep_skin_normal')))
        reported=row['research_metrics'];fresh=evaluate(z,m,protocol);fresh.update(trace_sha256=sha(path),protocol_sha256=sha(run/'sources/protocol.json'),evaluation_source_sha256=sha(run/'sources/evaluation.py'))
        check(tag+'/complete_frozen_V3_evaluation_recomputed',pure['same'](fresh,reported))
        own=pure['outcome'](z,m,protocol)
        check(tag+'/independent_task_and_full_motion_outcomes',all(pure['same'](own[k],reported[k]) for k in ('mission_success','full_motion_mission_success','failure_reason','failure_at','initial_stability_wait_s')))
        check(tag+'/independent_release',all(pure['same'](own[a],reported['release'][b]) for a,b in [('physical_exposure','physical_exposure'),('physical_success','physical_success'),('task_qualified_success','task_qualified_success'),('release_latency_s','physical_clearance_latency_s')]))
        check(tag+'/independent_full_raw_safety',all(pure['same'](own[k],reported['safety'][k]) for k in ('guard_pass','local_excess_integral_Ns','total_excess_integral_Ns')))
        band=pure['phase_wave'](z,m,protocol);qualities={}
        for window,lo,hi in [('static_quality',8.2,8.9),('dynamic_quality',9.3,11.7),('entire_holding_quality',7.5,12.)]:
            mask=(t>=lo-1e-9)&(t<=hi+1e-9);eligible=bool(mask.sum()==round((hi-lo)/.005)+1 and np.all(own['supported'][mask]&own['raised'][mask]&(z['support'][mask]<=.05)&z['controller_active'][mask].astype(bool)))
            check(tag+'/'+window+'/failure_coverage_retained',eligible==reported[window]['eligible'])
            if eligible:
                qualities[window]={name:pure['variation'](t[mask],signal[mask]) for name,signal in [('bandlimited',band),('raw_200Hz',z['normal'])]}
                check(tag+'/'+window+'/independent_phase_filtered_and_raw_variation',all(pure['same'](value,reported[window][sampling][k]) for sampling,record in qualities[window].items() for k,value in record.items() if not k.startswith('per_finger_')))
        tracking={}
        for name,lo,hi in [('holding_motion_tracking',7.5,12.),('dynamic_motion_tracking',9.3,11.7)]:
            tracking[name]=pure['motion_tracking'](z,lo,hi)
            check(tag+'/'+name+'/actual_encoders_to_original_user_target',reported['progress'][name]['available'] and reported['progress'][name]['complete_window'] and all(pure['same'](value,reported['progress'][name][k]) for k,value in tracking[name].items()))
        stages={};force=z['substep_skin_normal'];caps=np.asarray(m['limits']);mechanical=m.get('mechanical_release_at');mechanical=12. if mechanical is None else mechanical
        for stage,mask in [('shared_preparation',st<7.5-1e-9),('policy_holding_transport',(st>=7.5-1e-9)&(st<mechanical-1e-9)),('mechanical_release_and_after',st>=mechanical-1e-9)]:
            f=force[mask];duration=dur[mask];peaks=np.argmax(f,axis=0);endclock=(st+dur)[mask]
            canonical=dict(duration_s=float(duration.sum()),finger_peak_N=f.max(0).tolist(),local_excess_integral_Ns=float(np.sum(np.maximum(f-caps,0)*duration[:,None])),total_excess_integral_Ns=float(np.maximum(f.sum(1)-m['budget'],0)@duration))
            check(tag+'/'+stage+'/raw_stage_integrals',pure['same'](canonical,reported['safety']['stages'][stage]))
            stages[stage]=dict(canonical,finger_peak_times_end_s=endclock[peaks].tolist(),stage_guard_pass=bool(np.all(f<=caps+protocol['thresholds']['force_numerical_tolerance_N']) and np.all(f.sum(1)<=m['budget']+protocol['thresholds']['force_numerical_tolerance_N'])))
        check(tag+'/prediction_clock_honesty',np.isnan(z['prediction_at'][~z['prediction_available']]).all() and not z['prediction_available'][t>=12.-1e-9].any())
        active=joint_evidence(z,check,tag) if method=='joint' else None
        for k in ('supported','raised'):own.pop(k)
        static=(t>=8.2-1e-9)&(t<=8.9+1e-9)
        records.append(dict(motion_mode=mode,scenario=scenario,method=method,seed=seed,trace_path=str(path),trace_sha256=sha(path),
            outcome=own,qualities=qualities,tracking=tracking,stages=stages,active_evidence=active,
            pause_fraction=reported['progress']['pause_fraction'],initial_stability_wait_s=reported['initial_stability_wait_s'],
            mean_actual_total_finger_force_N={name:reported[name]['mean_actual_finger_force_N'] for name in qualities},
            static_raw_force_profile_sha256=hashlib.sha256(z['normal'][static].tobytes()).hexdigest(),
            event_recovery=reported['event_recovery'],actual_physical_parameters=m['config']))
        if method in BASELINES:
            pair=baseline_pair(path,old/'runs'/f'{scenario}__{method}__{seed}'/'trace.npz');baselines.append(dict(motion_mode=mode,scenario=scenario,method=method,**pair))
            check(tag+'/retained_V9_baseline_physics_bitexact',pair['physical40_fields_bit_exact'])
            check(tag+'/retained_V9_baseline_all_shared_numeric_bitexact',pair['all_shared_numeric_arrays_bit_exact'] and pair['numeric_field_set_equal'])
            check(tag+'/retained_V9_public_observation_only_adds_liftvelocity',pair['legacy_public_observation_fields_equal_except_added_lift_velocity'] and all(pair['physical_packet_strings_exact'].values()))
        print(tag,'independent_raw_checked',flush=True)
    pairing=[]
    for scenario in SCENES:
        base=metadata[(scenario,'ratio')];fields=('config','noise','limits','budget','model_sha256','initial_position')
        equal={field:all(metadata[(scenario,method)][field]==base[field] for method in METHODS) for field in fields}
        public={field:all(metadata[(scenario,method)]['task_contract'][field]==base['task_contract'][field] for method in METHODS) for field in ('user_motion_trajectory','motion_reference_bounds','sensor_characteristics','tracking_tolerances')}
        gains=metadata[(scenario,'joint')]['actual_ratio_task_gains']==base['gains']
        check(mode+'/'+scenario+'/five_method_pairing_same_actual_physics_noise_public_task',all(equal.values()) and all(public.values()) and gains)
        with np.load(paths[(scenario,'ratio')],allow_pickle=False) as ref:target=ref['actual_motion_target']
        for method in METHODS[1:]:
            with np.load(paths[(scenario,method)],allow_pickle=False) as candidate:check(mode+'/'+scenario+'/'+method+'/same_original_user_motion_target',np.array_equal(target,candidate['actual_motion_target']))
        pairing.append(dict(scenario=scenario,physical_noise_and_caps=equal,public_task=public,actual_ratio_gains_equal=gains))
    return dict(mode=mode,source_records=source_records,source_count=len(source_records),model_data_copies_count=len(expected_model_data()),records=records,baseline_exact_comparisons=baselines,pairing=pairing,
        input_sha256={name:sha(run/name) for name in ('execution_complete.json','manifest.json','plan.json','trials.json','assignment_ledger.json','source_snapshot_manifest.json')},
        frozen_controller_sha256=manifest['research_source_sha256']['v10/controller.py'])


def summaries(records):
    grouped=[];contrasts=[];index={(r['motion_mode'],r['scenario'],r['method']):r for r in records};modes=sorted({r['motion_mode'] for r in records})
    for method in METHODS:
        group=[r for r in records if r['method']==method]
        result=dict(method=method,assigned=len(group),mission_successes=sum(r['outcome']['mission_success'] for r in group),full_motion_successes=sum(r['outcome']['full_motion_mission_success'] for r in group),full16s_raw_guard_passes=sum(r['outcome']['guard_pass'] for r in group),physical_releases=sum(r['outcome']['physical_success'] for r in group),holding_local_excess_total_Ns=sum(r['stages']['policy_holding_transport']['local_excess_integral_Ns'] for r in group),release_local_excess_total_Ns=sum(r['stages']['mechanical_release_and_after']['local_excess_integral_Ns'] for r in group),unique_static_raw_force_profiles=len({r['static_raw_force_profile_sha256'] for r in group}),quality={})
        for window in ('static_quality','dynamic_quality','entire_holding_quality'):
            selected=[r for r in group if window in r['qualities']]
            result['quality'][window]=dict(coverage=len(selected),assigned=len(group),bandlimited_TV_mean_N_s=float(np.mean([r['qualities'][window]['bandlimited']['finger_TV_rate_N_s'] for r in selected])) if selected else None,raw_TV_mean_N_s=float(np.mean([r['qualities'][window]['raw_200Hz']['finger_TV_rate_N_s'] for r in selected])) if selected else None)
        grouped.append(result)
    for baseline in BASELINES:
        for window in ('static_quality','dynamic_quality','entire_holding_quality'):
            for sample in ('bandlimited','raw_200Hz'):
                common=[(mode,s) for mode in modes for s in SCENES if window in index[(mode,s,baseline)]['qualities'] and window in index[(mode,s,'joint')]['qualities']]
                a=[index[(mode,s,baseline)]['qualities'][window][sample]['finger_TV_rate_N_s'] for mode,s in common];b=[index[(mode,s,'joint')]['qualities'][window][sample]['finger_TV_rate_N_s'] for mode,s in common]
                contrasts.append(dict(baseline=baseline,candidate='joint',window=window,sampling=sample,assigned_pairs=4*len(modes),common_pairs=len(common),included_mode_scenarios=common,baseline_mean=float(np.mean(a)) if a else None,joint_mean=float(np.mean(b)) if b else None,joint_minus_baseline=float(np.mean(np.asarray(b)-a)) if a else None,relative_change=float(np.mean(b)/np.mean(a)-1.) if a and np.mean(a)>0 else None))
    return grouped,contrasts


def self_test():
    check=[]
    if len(expected_model_data())!=62:raise AssertionError('Expected62 model artifacts')
    check.append('exact62_data_dependency_keys')
    x=np.array([np.nan,1.]);y=x.copy()
    if not bit_equal(x,y):raise AssertionError('NaN payload byte equality')
    check.append('numeric_bit_comparison_preserves_NaN_payload')
    y[1]=np.nextafter(1.,2.)
    if bit_equal(x,y):raise AssertionError('Tiny difference must fail')
    check.append('bit_comparison_does_not_hide_tiny_difference')
    pure=load_pure()
    t=np.array([0.,.005,.010,.015]);f=np.tile(np.array([0.,1.,0.,1.])[:,None],(1,5));v=pure['variation'](t,f)
    if not np.isclose(v['finger_TV_rate_N_s'],1000.) or not np.isclose(v['finger_reversal_rate_N_s'],1000./3):raise AssertionError('Independent variation toy')
    check.append('pinned_independent_raw_TV_net_reversal_toy')
    if any(n=='mujoco' or n.startswith('mujoco.') for n in sys.modules):raise AssertionError('Physics library imported')
    check.append('no_physics_import')
    return dict(passed=True,checks=check,new_physics=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--self-test',action='store_true');p.add_argument('--modes',nargs='+',choices=('hold','transport'),default=['hold','transport']);args=p.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2));return 0
    if len(set(args.modes))!=len(args.modes):raise ValueError('Duplicate modes forbidden')
    require_terminal(args.modes)
    if args.out is None:raise ValueError('Fresh review --out required')
    if args.out.exists():raise FileExistsError('Prior audit retained; choose a new result')
    checks={}
    def check(name,value):checks[name]=bool(value)
    pure=load_pure();modes=[audit_mode(mode,pure,check) for mode in args.modes];records=[r for mode in modes for r in mode['records']]
    grouped,contrasts=summaries(records)
    if len(modes)==2:check('same142_frozen_source_identity_across_hold_and_transport',{r['name']:r['sha256'] for r in modes[0]['source_records']}=={r['name']:r['sha256'] for r in modes[1]['source_records']})
    result=dict(schema='v10_factorial_complete_physical_independent_v1',passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],created_utc=datetime.now(timezone.utc).isoformat(),
        audit_source_sha256=sha(__file__),independent_arithmetic_source_sha256=PURE_SHA,registration_path=str(ROOT/'review/v10_factorial_gate_plan_v1.json'),registration_sha256=sha(ROOT/'review/v10_factorial_gate_plan_v1.json'),
        cohort_modes=args.modes,full_registered40trial_cohort=len(modes)==2,modes=modes,method_summaries=grouped,paired_quality_contrasts=contrasts,new_physics=False,production_sources_modified=False,
        interpretation='One-seen-seed paired development; each included mode is its complete20trial cohort, combined40trials only after bothterminal. Mission and raw guard remain separate from filtered/raw force-quality. Static duplicates and missing quality windows are explicit. Model/FK objectives are not measured pressure forecasts or global/novel-contact certificates; retained baselines are checked against V9, not assumed.')
    with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],output=str(args.out),sha256=sha(args.out),method_summaries=grouped),ensure_ascii=False,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

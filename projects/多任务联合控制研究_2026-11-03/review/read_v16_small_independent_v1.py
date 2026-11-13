"""Terminal-gated four-run saved-interface audit; no physical replay.

One optional static MjModel compile reads robot port names/bias. MjData,
forward/FK/contact/step/plant construction are forbidden. Old scores stay fixed.
Unicode payload and literal archive width failures are reported separately.
"""
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import argparse,hashlib,importlib.util,json,sys,gzip,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(FOUNDATION));sys.dont_write_bytecode=True
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
base=module('v16_saved_interface_helpers',ROOT/'review/read_v15_active9_independent_v1.py')
prefix=module('v16_exact_prefix_helpers',ROOT/'review/v16_small_prefix_storage_independent_v1.py')
read,sha,equal,load_npz=base.read,base.sha,base.equal,base.load_npz
PLAN=ROOT/'review/v16_release_small_plan_v1.json'
def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def gate(plan,reader=read):
    states=[]
    for s in plan['stages']:
        try:d=reader(ROOT/s['output']/'execution_complete.json')
        except FileNotFoundError:d=None
        states.append(dict(stage=s['id'],complete=d is not None and d.get('execution_complete') is True,completion=d))
    return dict(ready=all(x['complete'] for x in states),states=states,pending_is_failure=False,branch_results_read=False)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--out',default='review/v16_small_independent_v1.json');args=ap.parse_args()
    if args.self_test:
        seen=[]
        def absent(p):seen.append(str(p));raise FileNotFoundError(p)
        g=gate(dict(stages=[dict(id='a',output='output/NO_BRANCH_READ')]),absent)
        assert not g['ready'] and not g['pending_is_failure'] and len(seen)==1 and seen[0].endswith('execution_complete.json')
        a=np.array(['same','much_longer']);b=np.array(['same','x']);assert np.array_equal(a[:1],b[:1]) and a[:1].tobytes()!=b[:1].tobytes()
        assert len(prefix.COLS)==37
        print(json.dumps(dict(passed=True,checks=3,branch_results_read=False,physics_steps=0)));return 0
    if not args.execute:raise ValueError('Explicit --execute required')
    plan=read(PLAN);terminal=gate(plan)
    if not terminal['ready']:print(json.dumps(dict(status='PENDING',terminal=terminal)));return 2
    out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    checks=[];issues=[];rows=[];cache={};oldcache={};actor=None;forbidden=[]
    def ck(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:issues.append(dict(name=name,detail=detail))
    def h(path):
        p=Path(path).resolve()
        if p not in cache:cache[p]=sha(p)
        return cache[p]
    expected=[]
    for s in plan['stages']:
        reg=read(ROOT/s['registration']);assignment=dict(methods=['ratio_effort'],scenarios=s['scene'],seeds=[619732],motion_mode=s['motion'],physical_randomization=True,task_policy='integrated',receiver_policy=s['policy'])
        ck(s['id']+'/preregistered_assignment',reg['assignment']==assignment and reg['source_sha256']==plan['source_sha256'] and reg['environment']==plan['environment'] and reg['master_plan_sha256']==sha(PLAN))
        expected.extend((s['id'],x,s['motion'],619732,'ratio_effort') for x in s['scene'])
    ck('all4_unique_assignments_1blind_3coordinated',len(expected)==len(set(expected))==4 and sum(x[0]=='blind_bridge' for x in expected)==1)
    binaries=plan['environment']['native_binaries_sha256']
    ck('123_current_sources_and42_binaries',len(plan['source_sha256'])==123 and len(binaries)==42 and all(h((FOUNDATION/k[11:]) if k.startswith('foundation/') else ROOT/k)==v for k,v in plan['source_sha256'].items()) and all(h(Path(k))==v for k,v in binaries.items()) and h(Path(plan['environment']['executable']))==plan['environment']['executable_sha256'])
    oldplan=read(ROOT/'review/v15A_prospective_multiscene_plan_v2.json')
    ck('frozen114_subset_exactly_preserved',all(plan['source_sha256'].get(k)==v for k,v in oldplan['source_sha256'].items()) and len(oldplan['source_sha256'])==114)
    import research_imports,mujoco,hand_plant
    for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_comPos','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):
            def deny(*a,_n=name,**kw):forbidden.append(_n);raise AssertionError('Forbidden physical pipeline '+_n)
            setattr(mujoco,name,deny)
    from foundation_config import FoundationConfig,OPEN
    from v13_a1.codec import ReferenceActor21
    from v14_coord.design import reference_rates
    from evaluate_runs import evaluate_trace
    from evaluation import evaluate_research
    open18=np.concatenate(OPEN)
    for stage in plan['stages']:
      folder=ROOT/stage['output'];tag=stage['id']
      try:
        done=read(folder/'execution_complete.json');manifest=read(folder/'manifest.json');execution_plan=read(folder/'plan.json');trials=read(folder/'trials.json');failures=read(folder/'runtime_failures.json');assigned=read(folder/'assignment_ledger.json');snap=read(folder/'source_snapshot_manifest.json')
        key=lambda r:(r['scenario'],r['method'],int(r['seed']))
        wanted={(s,'ratio_effort',619732) for s in stage['scene']}
        ck(tag+'/complete_and_all_failures_retained',done['execution_complete'] and done['sources_unchanged'] and done['planned']==done['trials']==stage['planned'] and not failures and len(trials)==len(assigned)==len(execution_plan['jobs'])==stage['planned'] and {key(x) for x in trials}=={key(x['job']) for x in assigned}==wanted)
        ck(tag+'/terminal_file_hashes',all(done[k+'_sha256']==h(folder/(k+'.json')) for k in ('trials','assignment_ledger','plan','manifest')))
        ck(tag+'/same_preregistered_source_environment',manifest['research_source_sha256']==plan['source_sha256'] and read(folder/'environment.json')['pinned_environment']==plan['environment'] and set(snap['files'])==set(plan['source_sha256']) and manifest['source_snapshot_manifest_sha256']==h(folder/'source_snapshot_manifest.json'))
        bad=[]
        for k,v in plan['source_sha256'].items():
            rec=snap['files'][k];sp=(folder/rec['snapshot_path']).resolve();lp=Path(rec['original_path']).resolve()
            if not sp.is_relative_to(folder.resolve()) or not lp.is_relative_to(ROOT.parent.resolve()):raise ValueError('Source path outside declared roots')
            if h(sp)!=v or h(lp)!=v or rec['sha256']!=v:bad.append(k)
        ck(tag+'/all123_snapshot_and_current_bytes',not bad,bad)
        for scenario in stage['scene']:
          label=tag+'/'+scenario;run=folder/'runs'/f'{scenario}__ratio_effort__619732';row=dict(stage=tag,scenario=scenario,motion=stage['motion'],seed=619732,method='ratio_effort',policy=stage['policy'],status='unassessed')
          rows.append(row)
          try:
            result=read(run/'result.json');meta=read(run/'metadata.json');path=run/'trace.npz';a=load_npz(path);t=a['time'];active=a['controller_active'].astype(bool)
            ck(label+'/result_trials_source_identity',equal(result,next(x for x in trials if key(x)==(scenario,'ratio_effort',619732))) and meta['research_source_sha256']==plan['source_sha256'] and result['trace_sha256']==meta['trace_sha256']==h(path) and meta['task_policy']=='integrated' and meta['receiver_policy']==stage['policy'])
            ck(label+'/full16s3201_and32000_raw',meta['completed'] and len(t)==3201 and len(a['substep_time'])==32000 and np.max(abs(t-np.arange(3201)*.005))<1e-12 and np.max(abs(a['substep_time']+a['substep_duration']-np.arange(1,32001)*.0005))<1e-8)
            oldstage=next(x for x in oldplan['stages'] if x['arm']=='candidate_A' and x['motion_mode']==stage['motion']);oldrun=ROOT/oldstage['output']/'runs'/f'{scenario}__ratio_effort__619732'
            oldkey=str(oldrun)
            if oldkey not in oldcache:
                with np.load(oldrun/'trace.npz',allow_pickle=False) as z:oldcache[oldkey]={k:z[k] for k in prefix.COLS}
            b=oldcache[oldkey];comparison=prefix.compare(a,b,t,b['time'],stage['policy']!='blind')
            ck(label+'/37column_exact_physical_and_instrument_payload',all(x['exact_payload_equal'] for x in comparison),[x for x in comparison if not x['exact_payload_equal']])
            if stage['policy']=='blind':ck(label+'/blind37literal_storage_exact',all(x['literal_storage_equal'] for x in comparison))
            physical=('shape','scenario','seed','split','config','initial_position','limits','budget','noise','model_sha256','motion_mode','observer_warmup_s');oldmeta=read(oldrun/'metadata.json')
            ck(label+'/same_oldA_physical_sensor_context',all(meta.get(k)==oldmeta.get(k) for k in physical))
            # Canonical old score definitions, only requested critical vectors.
            def cached_load(p,*args,**kw):
                if Path(p).resolve()==path.resolve():return base.CachedArchive(a)
                raise AssertionError('Unexpected NPZ reread '+str(p))
            with patch.object(np,'load',cached_load):old_eval=evaluate_trace(path);research_eval=evaluate_research(path)
            oldkeys=[k for k in old_eval if k=='task_success' or 'release' in k]
            missionkeys=[k for k in research_eval if k=='full_motion_mission_success' or ('mission' in k and isinstance(research_eval[k],(bool,list,str,type(None))))]
            ck(label+'/original_task_release_scores_recomputed',all(equal(old_eval[k],result[k]) for k in oldkeys),oldkeys)
            ck(label+'/full_motion_and_force_guard_recomputed',all(equal(research_eval[k],result['research_metrics'][k]) for k in missionkeys) and equal(research_eval['safety'],result['research_metrics']['safety']),missionkeys+['safety'])
            if actor is None:model,_=hand_plant._model(FoundationConfig(**meta['config']));actor=ReferenceActor21(model)
            ids=actor.indices21;ports=a['executed_reference21'].copy();ports[:,18]+=actor.lift_bias_m
            ck(label+'/all200Hz_actual21_engine_ports_and_public_before',ports.tobytes()==a['ctrl'][:,ids].tobytes() and a['robot_actuator_target21'].tobytes()==a['ctrl_before'][:,ids].tobytes())
            obs=[json.loads(str(x)) for x in a['observation_json']];debug=[json.loads(str(x)) for x in a['debug_json']];graphs=[json.loads(str(x)) for x in a['receiver_release_tasks_json']]
            errors=[];phase_counts=Counter();transitions=[];last_version=None;last_gen=None
            for i in np.flatnonzero(active):
                frame=obs[i];before=np.asarray(frame['robot_reference21']);ex=a['executed_reference21'][i];lo=np.asarray(frame['reference_lower21']);hi=np.asarray(frame['reference_upper21']);mode=debug[i]['mode'];generation=int(a['mode_generation'][i])
                if np.max(abs(ex-before)-reference_rates(mode)*.005)>1e-12 or np.max(lo-ex)>1e-12 or np.max(ex-hi)>1e-12:errors.append((int(i),'actual_permission_or_rate'))
                if generation!=debug[i]['mode_generation'] or frame['mode_generation']!=generation or (ex-before).tobytes()!=a['reference_increment21'][i].tobytes():errors.append((int(i),'clock_generation_or_actual_delta'))
                changed=a['requested_reference21'][i].tobytes()!=ex.tobytes()
                if changed!=bool(debug[i].get('execution_changed_plan')):errors.append((int(i),'receipt_changed_flag'))
                g=graphs[i]
                if stage['policy']=='coordinated' and t[i]>=12.-1e-9:
                    if g is None:errors.append((int(i),'missing_release_process'));continue
                    phase_counts[g['phase']]+=1
                    if abs(g['time_s']-t[i])>1e-9 or g['holding_force_tasks_active'] or np.any(a['desired'][i]!=0) or g['placement_certified']:errors.append((int(i),'release_process_or_holding_leak'))
                    dd=ex[:18]-before[:18];toward=open18-before[:18]
                    if np.any(dd*toward < -1e-12) or np.any(abs(dd)>abs(toward)+1e-12) or ex[18]>before[18]+1e-12:errors.append((int(i),'5ms_release_direction'))
                    if g['first_receiver_event_latched'] and not g['coordinated_retreat_started'] and abs(ex[18]-before[18])>1e-12:errors.append((int(i),'latched_lift_not_frozen'))
                    if last_version is not None and g['version']!=last_version and generation<=last_gen:errors.append((int(i),'phase_generation_not_invalidated'))
                    if g['version']!=last_version:transitions.append(dict(time_s=float(t[i]),phase=g['phase'],version=g['version'],mode_generation=generation))
                    last_version=g['version'];last_gen=generation
            ck(label+'/all_active5ms_permissions_receipts_lifecycle',not errors,errors[:12])
            joint=read(run/'joint_decision_manifest.json');release=read(run/'release_decision_manifest.json');decision_indices=[];counts=Counter();decision_bad=[]
            for kind,manifest0 in [('joint',joint),('release',release)]:
                ck(label+'/'+kind+'_decision_file_hash',h(run/manifest0['file'])==manifest0['sha256'])
                records=[]
                with gzip.open(run/manifest0['file'],'rt',encoding='utf-8') as f:records=[json.loads(x) for x in f]
                if len(records)!=manifest0['records']:decision_bad.append((kind,'manifest_count'))
                for k,q in enumerate(records):
                    i=round(q['time_s']/.005);decision_indices.append(i);counts[kind+'_decisions']+=1
                    generation=q['generation'] if kind=='joint' else q['mode_generation']
                    if q['decision_id']!=k or abs(t[i]-q['time_s'])>1e-9 or generation!=int(a['mode_generation'][i]):decision_bad.append((kind,k,'clock_generation'))
                    if kind=='joint':
                        counts['joint_mode:'+q['mode']]+=1
                        acc=q.get('proposal_acceptance') or {};accepted=bool(acc.get('accepted'));native=q.get('native')
                        if acc.get('different_force_feedback_problem_used') or (accepted and not acc.get('objective_bytes_equal')):decision_bad.append((kind,k,'hidden_goal_substitution'))
                        if stage['policy']=='coordinated' and q['mode']!='holding':decision_bad.append((kind,k,'old_controller_during_coordinated_release'))
                        if native is not None and native.get('reference21') is not None:
                            full=native['diagnostics'].get('prepared_value_final_full_recheck',{});counts['native_final_pass']+=bool(full.get('passed'))
                            if not accepted or not full.get('passed') or np.asarray(native['requested_delta_effort21']).tobytes()!=np.asarray(q['qp']['d21']).tobytes():decision_bad.append((kind,k,'native_goal_or_fullcheck'))
                    else:
                        counts['release_hierarchy:'+str(q['hierarchy_status'])]+=1;accepted=q['geometry_proposal_accepted']
                        if accepted!=(q['hierarchy_status']=='CONVERGED_DECLARED_LINEAR_HIERARCHY') or (not accepted and not q['fallback']) or q['public_process']['holding_force_tasks_active']:decision_bad.append((kind,k,'hierarchy_or_fallback'))
            ck(label+'/saved_joint_native_and_release_decisions_honest',not decision_bad,decision_bad[:12])
            ledger=load_npz(run/'delivery_intervals.npz');ls=read(run/'delivery_summary.json');lm=read(run/'delivery_manifest.json');starts=ledger['interval_start_s'];ends=ledger['interval_end_s'];valid=ledger['contact_valid'].astype(bool)
            ck(label+'/all32000_contiguous_cached_ledger_and_hashes',len(ends)==32000 and starts.tobytes()==a['substep_time'].tobytes() and np.array_equal(starts[1:],ends[:-1]) and all(h(run/lm[k]['file'])==lm[k]['sha256'] for k in ('arrays','summary')) and ls['evaluation_only'] and not ls['controller_input_permitted'])
            dt=np.maximum(0.,np.minimum(ends,16.)-np.maximum(starts,12.));wm=dt>0;complete=bool(ls['window_coverage_complete'] and valid[wm].all());bad=[]
            for name,s in ls['target_contact_classes'].items():
                force=ledger['target_'+name+'_normal_N']
                if s['available']!=complete:bad.append((name,'availability'))
                if complete and not np.allclose([force[wm].max(),np.sum(force[wm]*dt[wm]),dt[wm&(force>0)].sum()],[s['peak_N'],s['impulse_Ns'],s['positive_contact_duration_s']],rtol=1e-10,atol=1e-10):bad.append((name,'statistics'))
                if not complete and any(s[k] is not None for k in ('peak_N','impulse_Ns','positive_contact_duration_s')):bad.append((name,'unknown_filled'))
            ck(label+'/release_window_ledger_stats_and_floor_recomputed',not bad and ls['floor_event']==(bool(np.any(ledger['target_floor_normal_N'][wm]>0)) if complete else None),bad)
            support=sum(ledger['target_'+k+'_normal_N'] for k in ('receiver_top','receiver_otherface','floor','other_nonhand_world','other_nonworld'))
            ck(label+'/ledger_matches_2kHz_original_contact',np.allclose(support[valid],a['substep_support'][valid],atol=1e-10,rtol=0) and np.allclose(ledger['target_hand_normal_N'][valid],a['substep_all_hand_object'][valid],atol=1e-10,rtol=0))
            ii=np.rint(ends/.005).astype(int);pick=np.flatnonzero((ii>=1)&(ii<=3200)&(abs(ends-ii*.005)<1e-9)&ledger['object_state_valid'].astype(bool));qa=ls['object_qpos_address'];va=ls['object_qvel_address']
            ck(label+'/ledger_integration_end_qv_exact',np.array_equal(ledger['object_free_qpos7_end'][pick],a['qpos'][ii[pick],qa:qa+7]) and np.array_equal(ledger['object_free_qvel6_end'][pick],a['qvel'][ii[pick],va:va+6]))
            first=lambda f:None if not np.any(wm&(f>0)) else [float(starts[np.flatnonzero(wm&(f>0))[0]]),float(ends[np.flatnonzero(wm&(f>0))[0]])]
            dec=np.zeros(len(t),bool);dec[decision_indices]=True
            row.update(status='audited',result=result,trace_sha256=h(path),metadata_sha256=h(run/'metadata.json'),old_run=str(oldrun),old_trace_sha256=h(oldrun/'trace.npz'),bridge_or_prefix37=comparison,decision_counts=dict(counts),release_phases_5ms=dict(phase_counts),release_phase_transitions=transitions,delivery_summary=ls,delivery_diagnostics=dict(first_target_top_interval_s=first(ledger['target_receiver_top_normal_N']),first_floor_interval_s=first(ledger['target_floor_normal_N']),anyactor_target_top_exact=ledger['receiver_any_actor_top_normal_N'].tobytes()==ledger['target_receiver_top_normal_N'].tobytes(),valid_contact_rows=int(valid.sum()),valid_object_rows=int(ledger['object_state_valid'].sum())),raw_skin_peak5_N=np.max(a['substep_skin_normal'],axis=0),timing_ms={k:dict(actual_decision_ticks=base.summary(a[k][dec]),active5ms=base.summary(a[k][active]),between_decisions=base.summary(a[k][active&~dec])) for k in ('control_ms','full_control_loop_ms')})
            print(json.dumps(dict(audited=sum(x['status']=='audited' for x in rows),assigned=4,stage=tag,scenario=scenario),ensure_ascii=False),flush=True)
          except Exception as exc:row.update(status='reader_error',error=repr(exc));issues.append(dict(name=label+'/reader_exception',error=repr(exc),traceback=traceback.format_exc()))
      except Exception as exc:issues.append(dict(name=tag+'/stage_exception',error=repr(exc),traceback=traceback.format_exc()))
    observed={(x['stage'],x['scenario']) for x in rows}
    for s in plan['stages']:
        for scene in s['scene']:
            if (s['id'],scene) not in observed:rows.append(dict(stage=s['id'],scenario=scene,motion=s['motion'],seed=619732,method='ratio_effort',policy=s['policy'],status='unassessed_due_to_reader_error'))
    ck('all4_assigned_denominators_retained',len(rows)==4 and {(x['stage'],x['scenario']) for x in rows}=={(x[0],x[1]) for x in expected})
    ck('source123_unchanged_after_readout',all(sha((FOUNDATION/k[11:]) if k.startswith('foundation/') else ROOT/k)==v for k,v in plan['source_sha256'].items()))
    ck('zero_forbidden_physics_calls',not forbidden,forbidden)
    summary=[]
    for policy in ('blind','coordinated'):
        group=[x for x in rows if x['policy']==policy];ready=[x for x in group if x['status']=='audited']
        summary.append(dict(policy=policy,assigned=len(group),audited=len(ready),task_success=sum(bool(x['result']['task_success']) for x in ready),full_motion=sum(bool(x['result']['research_metrics']['full_motion_mission_success']) for x in ready),force_guard=sum(bool(x['result']['research_metrics']['safety']['guard_pass']) for x in ready),floor_event=sum(x['delivery_summary']['floor_event'] is True for x in ready),quality_eligible={k:sum(bool(x['result']['research_metrics'][k]['eligible']) for x in ready) for k in ('static_quality','dynamic_quality','entire_holding_quality')}))
    report=dict(schema='v16-small-independent-v1',passed=not issues,checks=checks,issues=issues,rows=rows,summary=summary,assigned=4,terminal_gate=terminal,master_plan_sha256=sha(PLAN),source_sha256=plan['source_sha256'],environment=plan['environment'],checker_sha256=sha(Path(__file__)),shared_helper_sha256=sha(ROOT/'review/read_v15_active9_independent_v1.py'),prefix_helper_sha256=sha(ROOT/'review/v16_small_prefix_storage_independent_v1.py'),retained_stopped_v1_progress=read(ROOT/'review/v16_release_small_progress_v1.json'),resume_progress=read(ROOT/'review/v16_release_small_progress_v2.json'),scope=dict(actual_new_runs=4,blind_literal_columns=37,prefix_columns=37,static_model_compilations=0 if actor is None else 1,model_data_or_plant_constructions=0,physics_steps=0,solver_replays=0),limitations=['Three known failure cases/one seen physical seed, not prospective confirmation or population superiority.','Old task/full motion/guard labels and all quality NULLs remain; no weighted score or success-only selection.','Unicode storage dtype/padding mismatch stays explicit beside exact payload equality.','21 engine ports verified at200Hz; no new2kHz control telemetry is invented.','Ledger cached contact solve and integrated object END state are distinct from boundary-forward public receiver packets.'])
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# V16 四条小矩阵独立物理记录验收','',f"技术核验{'PASS' if report['passed'] else 'FAIL'}；{len(checks)}项检查，{len(issues)}问题。1条blind桥与3条coordinated已见失败开发案例分列，不是4个独立种子。",'', '|policy|assigned|audited|task|full motion|force guard|floor|','|---|---|---|---|---|---|---|']
    for s in summary:lines.append('|'+ '|'.join(map(str,[s['policy'],s['assigned'],s['audited'],s['task_success'],s['full_motion'],s['force_guard'],s['floor_event']]))+'|')
    lines+=['','完整向量、原quality可评/NULL、37列literal与payload、所有执行/账本检查及末态见JSON。原v1字符串padding停止和resume记录保留；没有重跑前两条或改原评分。','',f'JSON SHA256 `{sha(out)}`。本审查0积分/FK/接触重算，最多1次静态模型编译；技术PASS不代表释放或交付成功。']
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),issues=issues[:8],summary=summary,path=str(out),sha256=sha(out)),ensure_ascii=False));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

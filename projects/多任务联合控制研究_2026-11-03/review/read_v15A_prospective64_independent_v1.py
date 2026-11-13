"""Prospective64 terminal-gated audit; no physical replay or score revision.

--check-plan opens registrations and historical manifests only. --execute first
reads ONLY completion markers for all four batches. A pending cohort is never a
failed cohort. A terminal failure retains every assigned denominator.
"""
from pathlib import Path
from collections import Counter,defaultdict
from itertools import product
from unittest.mock import patch
import argparse,gzip,hashlib,importlib.util,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(FOUNDATION));sys.dont_write_bytecode=True
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
base=module('v15_completed_audit_helpers',ROOT/'review/read_v15_active9_independent_v1.py')
read,sha,equal,load_npz,summary=base.read,base.sha,base.equal,base.load_npz,base.summary
PLAN=ROOT/'review/v15A_prospective_multiscene_plan_v2.json'
DIAG=ROOT/'review/continuous_diagnostics_v1.py'
DIAG_SHA='05e86a5869f144f990dbb9883add20d002c4384f402cfb07b15db333e9bd8cfa'
SCENES={'nominal','friction_drop','fragile_conflict','collision'};SEEDS={619731,619732};METHODS={'ratio_effort','fixed_effort','fusion_effort'}
def write(path,value):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def numeric_leaves(value,prefix=''):
    """Unsigned preference: auxiliary deltas are lower/higher, never overall wins."""
    skip={'original_result','identity','execution_context','window_s','samples','expected_samples','duration_s','threshold_N','query_time_s'}
    if isinstance(value,dict):
        out={}
        for k,v in value.items():
            if k not in skip:out.update(numeric_leaves(v,prefix+'.'+k if prefix else k))
        return out
    if isinstance(value,list):
        out={}
        for i,v in enumerate(value):out.update(numeric_leaves(v,prefix+'.'+str(i)))
        return out
    if isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value):return {prefix:float(value)}
    return {}
def pair_readout(rows):
    scalar=module('existing_scalar_metric_comparisons',ROOT/'review/read_v12_native_multiscene_comparison_independent_v1.py')
    index={(r['arm'],r['motion'],r['scenario'],r['seed'],r['method']):r for r in rows}
    original=[];auxiliary=[]
    for method in sorted(METHODS):
      pairs=[]
      for motion,scenario,seed in product(('hold','transport'),sorted(SCENES),sorted(SEEDS)):
        c=index.get(('candidate_A',motion,scenario,seed,'ratio_effort'));b=index.get(('baseline',motion,scenario,seed,method));context=dict(motion=motion,scenario=scenario,seed=seed)
        cr=None if c is None or c['status']!='audited' else c['result']['research_metrics'];br=None if b is None or b['status']!='audited' else b['result']['research_metrics']
        ca={} if c is None or c['status']!='audited' else numeric_leaves(c['auxiliary']);ba={} if b is None or b['status']!='audited' else numeric_leaves(b['auxiliary'])
        pairs.append((context,cr,br,ca,ba))
      for spec in scalar.metric_specs():
        results=[dict(context=context,**scalar.comparison(cr,br,spec)) for context,cr,br,_,_ in pairs]
        original.append(dict(baseline_method=method,metric=spec,summary=scalar.summarize_pairs(results),by_seed={str(seed):scalar.summarize_pairs([r for r in results if r['context']['seed']==seed]) for seed in sorted(SEEDS)},pairs=results))
      for key in sorted(set().union(*(set(c)|set(b) for _,_,_,c,b in pairs))):
        detail=[]
        for context,_,_,c,b in pairs:
          cv=c.get(key);bv=b.get(key);delta=None if cv is None or bv is None else cv-bv
          kind='unavailable' if delta is None else 'same' if abs(delta)<=1e-12+1e-10*max(abs(cv),abs(bv)) else 'lower' if delta<0 else 'higher'
          percent=None if delta is None or abs(bv)<=1e-9 else 100*delta/abs(bv)
          detail.append(dict(context=context,candidate=cv,baseline=bv,absolute_delta=delta,relative_change_percent=percent,direction=kind))
        def collect(rr):
          count=Counter(r['direction'] for r in rr);d=[r['absolute_delta'] for r in rr if r['absolute_delta'] is not None];p=[r['relative_change_percent'] for r in rr if r['relative_change_percent'] is not None]
          return dict(planned=len(rr),counts=dict(count),median_absolute_delta=None if not d else float(np.median(d)),relative_evaluable=len(p),median_relative_change_percent=None if not p else float(np.median(p)))
        auxiliary.append(dict(baseline_method=method,metric_path=key,summary=collect(detail),by_seed={str(seed):collect([r for r in detail if r['context']['seed']==seed]) for seed in sorted(SEEDS)},pairs=detail))
    return dict(original_metrics=original,all_assigned_auxiliary=auxiliary,
        interpretation='Original metric directions are per-component only. Auxiliary lower/higher changes are descriptive, not success, global benefit, utility weights or implicit preference for zero force after loss.',
        scalar_helper_sha256=sha(ROOT/'review/read_v12_native_multiscene_comparison_independent_v1.py'))
def plan_check(p):
    checks=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    stages=p['stages'];seen=[]
    for s in stages:
        expected={k:s[k] for k in ('methods','scenarios','seeds','motion_mode','physical_randomization','task_policy','receiver_policy')}
        reg=read(ROOT/s['registration']);count=len(s['methods'])*len(s['scenarios'])*len(s['seeds'])
        ck(s['name']+'/role',s['receiver_policy']=='blind' and s['motion_mode'] in ('hold','transport') and set(s['scenarios'])==SCENES and set(s['seeds'])==SEEDS and s['physical_randomization'])
        ck(s['name']+'/arm',((s['arm']=='candidate_A' and s['methods']==['ratio_effort'] and s['task_policy']=='integrated') or (s['arm']=='baseline' and set(s['methods'])==METHODS and s['task_policy']=='legacy')))
        ck(s['name']+'/registration',reg['assignment']==expected and reg['source_sha256']==p['source_sha256'] and reg['environment']==p['environment'] and reg['software_gate_passed'])
        ck(s['name']+'/planned',count==s['planned'])
        seen += [(s['arm'],s['motion_mode'],c,int(seed),m) for c,seed,m in product(s['scenarios'],s['seeds'],s['methods'])]
    ck('64_unique_assignments',len(seen)==len(set(seen))==p['total_physical_trials']==64)
    ck('16contexts_2clusters',len({(m,c,k) for _,m,c,k,_ in seen})==p['paired_contexts']==16 and p['physical_seed_clusters']==len(SEEDS)==2)
    groups=Counter((arm,m) for arm,m,*_ in seen);ck('8_24_8_24',groups==Counter({('candidate_A','hold'):8,('baseline','hold'):24,('candidate_A','transport'):8,('baseline','transport'):24}))
    ck('114_sources_42binaries',len(p['source_sha256'])==114 and len(p['environment']['native_binaries_sha256'])==42)
    oldscenes=set();oldseeds=set();historical=[]
    for name,h in p['original_context_evidence'].items():
        path=ROOT/name;ck('old_context_manifest/'+name,sha(path)==h);old=read(path)
        for job in old['jobs']:oldscenes.add(job['scenario']);oldseeds.add(int(job['seed']));historical.append(job)
    ck('original_48_and_scene_set_restored',len(historical)==48 and oldscenes==SCENES)
    ck('new_seed_pair_disjoint_original48',not (oldseeds&SEEDS) and set(p['physical_seed_inventory']['seeds'])==SEEDS)
    ck('v1_superseded_before_execution_declared',p['supersedes_before_execution']['new_physics_started'] is False)
    ck('continuous_reader_frozen',sha(DIAG)==DIAG_SHA)
    return dict(schema='v15A_prospective64_registration_review_v1',passed=all(x['passed'] for x in checks),checks=checks,plan_sha256=sha(PLAN),continuous_reader_sha256=DIAG_SHA,
        denominators=dict(assigned=64,paired_contexts=16,arms_per_context=4,independent_physical_seed_clusters=2),branch_results_read=False,
        limits=['Seed novelty is the explicitly bounded preregistration inventory claim; the original inventory search stdout is not independently reproduced after registration.',
            'Candidate A was selected from seen617901 data; this is new-condition development, not independent confirmation. V16 is not present.'])

def main():
    ap=argparse.ArgumentParser();mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--check-plan',action='store_true');mode.add_argument('--execute',action='store_true');ap.add_argument('--out');args=ap.parse_args()
    p=read(PLAN);pre=plan_check(p)
    if args.check_plan:
        out=ROOT/(args.out or 'review/v15A_prospective64_preflight_independent_v1.json');write(out,pre);print(json.dumps(dict(passed=pre['passed'],checks=len(pre['checks']),branch_results_read=False,path=str(out),sha256=sha(out)),ensure_ascii=False));return 0 if pre['passed'] else 1
    gate=base.terminal_gate(p['stages'])
    if not gate['ready']:print(json.dumps(dict(status='PENDING',gate=gate)));return 2
    if not pre['passed']:raise RuntimeError('Registration inconsistency; inspect preflight')
    out=ROOT/(args.out or 'review/v15A_prospective64_independent_v1.json')
    if out.exists():raise FileExistsError(out)
    diag=module('frozen_continuous_measurement',DIAG);checks=[];issues=[];rows=[];cache={};pairs=defaultdict(list);forbidden=[];actor=None
    def ck(n,v,detail=None):
        checks.append(dict(name=n,passed=bool(v),detail=detail))
        if not v:issues.append(dict(name=n,detail=detail))
    def h(path):
        path=Path(path).resolve()
        if path not in cache:cache[path]=sha(path)
        return cache[path]
    import research_imports,mujoco,hand_plant
    for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_kinematics','mj_comPos','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):
            def deny(*a,_n=name,**k):forbidden.append(_n);raise AssertionError('Forbidden pipeline '+_n)
            setattr(mujoco,name,deny)
    from evaluate_runs import evaluate_trace
    from evaluation import evaluate_research
    from foundation_config import FoundationConfig
    from v13_a1.codec import ReferenceActor21
    from v14_coord.design import reference_rates
    try:
      for stage in p['stages']:
        folder=ROOT/stage['output'];done=read(folder/'execution_complete.json');man=read(folder/'manifest.json');jobs=read(folder/'plan.json')['jobs'];trials=read(folder/'trials.json');failures=read(folder/'runtime_failures.json');assignment=read(folder/'assignment_ledger.json');tag=stage['name']
        ck(tag+'/all_denominators_and_failures',done['planned']==stage['planned']==len(jobs)==len(assignment)
            and len(trials)==sum(x['status']=='evaluated' for x in assignment)
            and failures==done['runtime_failures']
            and all(x['status'] in ('evaluated','runtime_failed','not_executed_due_to_preflight_or_dispatch_failure','not_executed_due_to_source_snapshot_failure') for x in assignment))
        ck(tag+'/execution_terminal_complete',done['execution_complete'] and done['sources_unchanged'] and len(trials)==stage['planned'] and not failures)
        expected={(c,int(k),m) for c,k,m in product(stage['scenarios'],stage['seeds'],stage['methods'])};key=lambda r:(r['scenario'],int(r['seed']),r['method'])
        ck(tag+'/exact_plan_and_assignment_keys',len({key(j) for j in jobs})==len(jobs) and {key(j) for j in jobs}==expected and {key(x['job']) for x in assignment}==expected)
        for n in ('trials','plan','manifest','assignment_ledger'):ck(tag+'/'+n+'_hash',sha(folder/(n+'.json'))==done[n+'_sha256'])
        ident=man['research_source_sha256'];snap=read(folder/'source_snapshot_manifest.json');env=read(folder/'environment.json')['pinned_environment'];bad=[]
        for name,value in ident.items():
          rec=snap['files'][name];sp=(folder/rec['snapshot_path']).resolve();live=Path(rec['original_path']).resolve()
          if not sp.is_relative_to(folder.resolve()) or not live.is_relative_to(ROOT.parent.resolve()):raise ValueError('Source path escaped project')
          if h(sp)!=value or h(live)!=value or rec['sha256']!=value:bad.append(name)
        ck(tag+'/same114_sources_snapshots_live',ident==p['source_sha256'] and set(snap['files'])==set(ident) and not bad,bad[:6])
        ck(tag+'/same42_native_binaries',env==p['environment'] and len(env['native_binaries_sha256'])==42 and all(h(Path(k))==v for k,v in env['native_binaries_sha256'].items()) and h(Path(env['executable']))==env['executable_sha256'])
        tm={key(t):t for t in trials}
        for job in jobs:
          d=Path(job['destination']);label=tag+'/'+d.name;record=dict(arm=stage['arm'],motion=stage['motion_mode'],scenario=job['scenario'],seed=int(job['seed']),method=job['method'],run=str(d))
          if not (d/'result.json').exists():
            record.update(status='missing_or_runtime_failed',runtime_error=read(d/'runtime_error.json') if (d/'runtime_error.json').exists() else None);rows.append(record);ck(label+'/retained_missing_assignment',False);continue
          result=read(d/'result.json');meta=read(d/'metadata.json');a=load_npz(d/'trace.npz');t=a['time'];active=a['controller_active'].astype(bool);N=len(t)
          ck(label+'/full16_and_original_result',N==3201 and len(a['substep_time'])==32000 and meta['completed'] and key(job) in tm and equal(result,tm[key(job)]))
          ck(label+'/trace_and_source_identity',sha(d/'trace.npz')==meta['trace_sha256']==result['trace_sha256'] and meta['research_source_sha256']==ident and meta['task_policy']==stage['task_policy'] and meta['receiver_policy']=='blind')
          ck(label+'/clock',np.allclose(t,np.arange(3201)*.005,atol=1e-12,rtol=0) and np.allclose(a['substep_time']+a['substep_duration'],np.arange(1,32001)*.0005,atol=1e-8,rtol=0))
          def cached(path,*x,**kw):
            if Path(path).resolve()==(d/'trace.npz').resolve():return base.CachedArchive(a)
            raise AssertionError('Unexpected second archive load')
          with patch.object(np,'load',cached):old=evaluate_trace(d/'trace.npz');research=evaluate_research(d/'trace.npz')
          ck(label+'/both_original_evaluators_exact',all(k in result and equal(v,result[k]) for k,v in old.items()) and equal(research,result['research_metrics']))
          if actor is None:model,_=hand_plant._model(FoundationConfig(**meta['config']));actor=ReferenceActor21(model)
          port=a['executed_reference21'].copy();port[:,18]+=actor.lift_bias_m;debug=[json.loads(str(s)) for s in a['debug_json']];ids=actor.indices21
          ck(label+'/actual21_and_public_before_engine',port.tobytes()==a['ctrl'][:,ids].tobytes() and a['robot_actuator_target21'].tobytes()==a['ctrl_before'][:,ids].tobytes())
          rate=0.;bad=[]
          for i in np.flatnonzero(active):
            rate=max(rate,float(np.max(abs(a['reference_increment21'][i])-reference_rates(debug[i]['mode'])*.005)))
            if int(debug[i]['mode_generation'])!=int(a['mode_generation'][i]):bad.append((int(i),'generation'))
          ck(label+'/5ms_rates_generation',rate<1e-12 and not bad,dict(rate_excess=rate,issues=bad[:4]))
          ck(label+'/blind_graph_unused',all(json.loads(str(s)) is None for s in a['receiver_release_tasks_json']))
          instrument=meta['receiver_instrument'];ck(label+'/receiver_independent_clock_draws',instrument['acquisitions']==N and instrument['gaussian_draws']==6*N and instrument['uniform_draws']==N and instrument['seed']==job['seed'])
          jm=read(d/'joint_decision_manifest.json');ck(label+'/decision_hash',sha(d/jm['file'])==jm['sha256']);counts=Counter();errors=[];di=[]
          with gzip.open(d/jm['file'],'rt',encoding='utf-8') as f:
            for k,line in enumerate(f):
              q=json.loads(line);i=int(round(q['time_s']/.005));di.append(i);counts['decisions']+=1;counts[q['mode']+'_decisions']+=1;n=q['native'];qp=q['qp'];acc=q.get('proposal_acceptance') or {};accepted=bool(acc.get('accepted'))
              counts['qp:'+('not_called' if qp is None else qp['status'])]+=1;counts['goal:'+('certified' if acc.get('optimality_certified') else 'provisional') if accepted else 'goal:none']+=1
              if q['decision_id']!=k or debug[i].get('decision_id')!=k or abs(t[i]-q['time_s'])>1e-9 or q['generation']!=int(a['mode_generation'][i]):errors.append((k,'clock'))
              if acc.get('different_force_feedback_problem_used'):errors.append((k,'hidden_alternate_problem'))
              if accepted and (not acc.get('objective_bytes_equal') or acc['primal_violation']>q['qp_problem']['atol']):errors.append((k,'accepted_feasibility'))
              if q['mode']=='release' and (q.get('task_labels') or (q.get('qp_problem') or {}).get('extra_task_rows') or q.get('native_tasks',{}).get('rows',0)!=0):errors.append((k,'holding_task_during_release'))
              if n is not None:
                counts['native:'+n['status']]+=1
                if n.get('reference21') is not None:
                  full=n['diagnostics'].get('prepared_value_final_full_recheck',{});counts['native_full_pass']+=bool(full.get('passed'))
                  if not accepted or not full.get('passed') or np.asarray(n['p0_effort21']).tobytes()!=np.asarray(q['domain']['p0_effort21']).tobytes() or np.asarray(n['requested_delta_effort21']).tobytes()!=np.asarray(qp['d21']).tobytes():errors.append((k,'native_goal_or_fullcheck'))
              elif accepted:errors.append((k,'accepted_goal_without_native'))
          ck(label+'/all_QP_native_decisions_honest',len(di)==jm['records'] and not errors,errors[:8])
          ledger=load_npz(d/'delivery_intervals.npz');ls=read(d/'delivery_summary.json');dm=read(d/'delivery_manifest.json');starts=ledger['interval_start_s'];ends=ledger['interval_end_s'];valid=ledger['contact_valid'].astype(bool)
          ck(label+'/32k_ledger_identity_and_offline',len(ends)==32000 and starts.tobytes()==a['substep_time'].tobytes() and np.array_equal(starts[1:],ends[:-1]) and ls['evaluation_only'] and not ls['controller_input_permitted'] and all(sha(d/dm[k]['file'])==dm[k]['sha256'] for k in ('arrays','summary')))
          support=sum(ledger['target_'+k+'_normal_N'] for k in ('receiver_top','receiver_otherface','floor','other_nonhand_world','other_nonworld'))
          ck(label+'/ledger_reconstructs_raw_contact',np.allclose(support[valid],a['substep_support'][valid],atol=1e-10,rtol=0) and np.allclose(ledger['target_hand_normal_N'][valid],a['substep_all_hand_object'][valid],atol=1e-10,rtol=0))
          dt=np.maximum(0.,np.minimum(ends,16.)-np.maximum(starts,12.));wm=dt>0;complete=ls['window_coverage_complete'] and bool(valid[wm].all());bad=[]
          for name,st in ls['target_contact_classes'].items():
            force=ledger['target_'+name+'_normal_N']
            if st['available']!=complete:bad.append((name,'availability'))
            if complete:
              v=[float(force[wm].max()),float(np.sum(force[wm]*dt[wm])),float(dt[wm&(force>0)].sum())]
              if not np.allclose(v,[st['peak_N'],st['impulse_Ns'],st['positive_contact_duration_s']],atol=1e-10,rtol=1e-10):bad.append((name,'statistics'))
            elif any(st[k] is not None for k in ('peak_N','impulse_Ns','positive_contact_duration_s')):bad.append((name,'unknown_filled'))
          ck(label+'/ledger_window_stats_and_nulls',not bad,bad)
          ii=np.rint(ends/.005).astype(int);w=np.flatnonzero((ii>=1)&(ii<=3200)&(abs(ends-ii*.005)<1e-9)&ledger['object_state_valid'].astype(bool));qa=ls['object_qpos_address'];va=ls['object_qvel_address']
          ck(label+'/ledger_qv_end_trace_exact',np.array_equal(ledger['object_free_qpos7_end'][w],a['qpos'][ii[w],qa:qa+7]) and np.array_equal(ledger['object_free_qvel6_end'][w],a['qvel'][ii[w],va:va+6]))
          auxiliary=diag.plain(diag.measure_arrays(a,ledger,meta,original_result=result,delivery_metadata=ls))
          prefix={k:base.array_sha(a[k][t<7.5-1e-9]) for k in base.PREP};boundary={k:base.array_sha(a[k][abs(t-7.5)<1e-9]) for k in base.BOUNDARY}
          physical={k:meta.get(k) for k in ('shape','scenario','seed','split','config','initial_position','limits','budget','noise','model_sha256','motion_mode','observer_warmup_s')}
          compare=dict(prefix=prefix,boundary=boundary,physical=physical,user_target_sha256=base.array_sha(a['actual_motion_target']))
          pairs[(stage['motion_mode'],job['scenario'],int(job['seed']))].append((stage['arm'],job['method'],compare))
          mask=np.zeros(N,bool);mask[di]=True;record.update(status='audited',result=result,auxiliary=auxiliary,technical_counts=dict(counts),delivery_summary=ls,trace_sha256=result['trace_sha256'],decision_sha256=jm['sha256'],pairing=compare,
              timing_ms={name:dict(active5ms=summary(a[name][active]),decision_records=summary(a[name][mask]),between_decisions=summary(a[name][active&~mask])) for name in ('control_ms','full_control_loop_ms','observation_ms','actor_callback_ms')})
          rows.append(record);print(json.dumps(dict(audited=len(rows),assigned=64,stage=stage['name'],scenario=job['scenario'],seed=job['seed'],method=job['method']),ensure_ascii=False),flush=True)
      for context,arms in pairs.items():
        ck(str(context)+'/all4_arms',len(arms)==4 and {(x,y) for x,y,_ in arms}=={('candidate_A','ratio_effort'),*(('baseline',m) for m in METHODS)})
        a0=arms[0][2];ck(str(context)+'/same_physical_sensor_and_user_task',all(a[2]['physical']==a0['physical'] and a[2]['user_target_sha256']==a0['user_target_sha256'] for a in arms))
        ck(str(context)+'/commonprep_preaction_bitexact',all(a[2]['prefix']==a0['prefix'] and a[2]['boundary']==a0['boundary'] for a in arms))
      ck('all64_assignments_retained',len(rows)==64);ck('no_physics_calls',not forbidden,forbidden)
      # Current source check again after readout, without using the earlier cache.
      ck('source_freeze_through_audit',all(sha((FOUNDATION/name[11:]) if name.startswith('foundation/') else ROOT/name)==value for name,value in p['source_sha256'].items()))
    except Exception as exc:issues.append(dict(name='reader_exception',error=repr(exc),traceback=traceback.format_exc()))
    # Even a reader exception must retain the remaining registered assignments as
    # unassessed, never silently shrink a denominator or call them task failures.
    observed={(r['arm'],r['motion'],r['scenario'],r['seed'],r['method']) for r in rows}
    for stage in p['stages']:
      for scenario,seed,method in product(stage['scenarios'],stage['seeds'],stage['methods']):
        key=(stage['arm'],stage['motion_mode'],scenario,seed,method)
        if key not in observed:rows.append(dict(arm=key[0],motion=key[1],scenario=key[2],seed=key[3],method=key[4],status='unassessed_due_to_reader_error'))
    # Full outcome vectors retained; summaries keep original NULLs separate.
    group=[]
    for arm,method in [('candidate_A','ratio_effort'),*(('baseline',m) for m in sorted(METHODS))]:
      rr=[r for r in rows if r['arm']==arm and r['method']==method];ready=[r for r in rr if r['status']=='audited'];byseed=[]
      for seed in sorted(SEEDS):
        rseed=[r for r in ready if r['seed']==seed];byseed.append(dict(seed=seed,assigned=8,audited=len(rseed),task_success=sum(bool(r['result']['task_success']) for r in rseed),full_motion=sum(bool(r['result']['research_metrics']['full_motion_mission_success']) for r in rseed),guard=sum(bool(r['result']['research_metrics']['safety']['guard_pass']) for r in rseed)))
      group.append(dict(arm=arm,method=method,assigned=16,audited=len(ready),unassessed=16-len(ready),task_success=sum(bool(r['result']['task_success']) for r in ready),full_motion=sum(bool(r['result']['research_metrics']['full_motion_mission_success']) for r in ready),guard=sum(bool(r['result']['research_metrics']['safety']['guard_pass']) for r in ready),
          quality_eligible={key:sum(bool(r['result']['research_metrics'][key]['eligible']) for r in ready) for key in ('static_quality','dynamic_quality','entire_holding_quality')},by_seed=byseed))
    comparison=pair_readout(rows)
    report=dict(schema='v15A_prospective64_independent_v1',passed=not issues,checks=checks,issues=issues,rows=rows,summary=group,paired_comparisons=comparison,plan_review=pre,terminal_gate=gate,
        source_identity=p['source_sha256'],environment=p['environment'],continuous_measurement_source_sha256=sha(DIAG),checker_sha256=sha(__file__),shared_audit_helper_sha256=sha(ROOT/'review/read_v15_active9_independent_v1.py'),
        scope=dict(assigned64=True,paired_contexts=16,physical_seed_clusters=2,static_robot_model_compilations=0 if actor is None else 1,physics_steps=0,plant_constructions=0,solver_replays=0,continuous_diagnostics='Frozen all-assigned additive measurements, never replacement labels'),
        limitations=['Candidate A was selected from seen pilot results; fresh conditions are development, not confirmation.', '16 contexts share only2 physical seed clusters; no population superiority claim or posthoc utility weights.', 'All failures and original ineligible NULLs remain; auxiliary unselected force and pose diagnostics do not become task success.', 'This is saved physical-interface/record consistency, not a fresh F067/geometry/kernel mathematical audit.'])
    write(out,report);md=['# V15A 前瞻64条独立终态审计','',f"技术审计{'PASS' if report['passed'] else 'FAIL'}；{len(checks)}项检查，{len(issues)}问题。16配对context、2物理seed clusters，非64个独立样本。",'','|arm|method|任务 /16|全运动 /16|力限 /16|静态可评 /16|动态可评 /16|全持握可评 /16|','|---|---|---|---|---|---|---|---|']
    for g in group:md.append('|'+ '|'.join(map(str,[g['arm'],g['method'],g['task_success'],g['full_motion'],g['guard'],*g['quality_eligible'].values()]))+'|')
    md+=['','原始完整指标、各seed分母、全assigned连续测量、floor/receiver/末态和实际执行计数见同名JSON。NULL未补0，未使用成功筛选或新增综合分。','',f'JSON SHA256 `{sha(out)}`。静态模型仅提取机器人端口/偏置，没有Data/plant/积分/FK或物理重放。'];out.with_suffix('.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),issues=issues[:6],assigned=len(rows),path=str(out),sha256=sha(out)),ensure_ascii=False));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

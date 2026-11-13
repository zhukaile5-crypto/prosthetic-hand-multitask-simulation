"""Terminal-gated V19 four-run cross-component execution audit, no physics.

Reviewer authored V19 task_qp.py, not governor/runtime/supervisor. This audits
saved execution interfaces, not an independent proof of that math or policy
performance. It does not repeat component mathematical gates. --prepare reads no runs.
One static model compile per saved config identifies actual21 ports/bias; no
Data, step, forward, FK, contact query, writer or optimizer replay is allowed.
"""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch
import argparse,hashlib,importlib.util,json,gzip,sys,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1];FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(FOUNDATION));sys.dont_write_bytecode=True
PLAN=ROOT/'review/v19_small_plan_v1.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def source_path(k):return FOUNDATION/k[11:] if k.startswith('foundation/') else ROOT/k
def equal(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def gate(plan,reader=read):
    states=[]
    for s in plan['stages']:
        try:d=reader(ROOT/s['output']/'execution_complete.json')
        except FileNotFoundError:d=None
        states.append(dict(stage=s['name'],complete=d is not None and d.get('execution_complete') is True,completion=d))
    return dict(ready=all(s['complete'] for s in states),states=states,pending_is_failure=False,branch_results_read=False)
def task_interval(g,q,v,L,U):
    """Independent saved-task interval arithmetic; never re-run governor."""
    band=(q-g['recovery_reference_budget_m'],q+g['downward_recovery_reference_budget_m'])
    a,b=max(L,band[0]),min(U,band[1]);reachable=a<=b
    if not reachable:a=b=min(U,max(L,q))
    elif g['braking_intent_active']:
        brake=min(b,max(a,q))
        if v<0:a=brake
        elif v>0:b=brake
        else:a=b=brake
    return np.array([a,b]),reachable,band

def passive_complete_bridge(a,b,timing_helper):
    """Every column except provenance; EXACT v3 timing leaves, UTF8 text."""
    records=[]
    for name in sorted(set(a)|set(b)):
        if name=='metadata':continue
        if name not in a or name not in b:
            records.append(dict(column=name,passed=False,reason='missing_column'));continue
        x,y=a[name],b[name];same=x.shape==y.shape
        literal=same and x.dtype==y.dtype and x.tobytes()==y.tobytes()
        if name in timing_helper.TIMING_COLUMNS:
            ok=same and x.dtype==y.dtype and x.dtype.kind in 'fi' and np.isfinite(x).all() and np.isfinite(y).all() and (x>=0).all() and (y>=0).all()
            rule='v3_exact_five_finite_nonnegative_walltime_columns'
        elif name=='debug_json':
            ok=same and x.dtype.kind=='U' and y.dtype.kind=='U' and all(timing_helper.debug_payload_without_declared_timing(u)==timing_helper.debug_payload_without_declared_timing(v) for u,v in zip(x.flat,y.flat))
            rule='all_debug_fields_except_exact_v3_three_validated_walltime_paths'
        elif x.dtype.kind in 'US' and y.dtype.kind in 'US':
            ok=same and all(str(u).encode('utf-8')==str(v).encode('utf-8') for u,v in zip(x.flat,y.flat))
            rule='complete_exact_UTF8_payload_no_sensor_or_algorithm_fields_removed'
        else:ok=literal;rule='exact_dtype_shape_bytes'
        records.append(dict(column=name,passed=bool(ok),rule=rule,literal_storage_equal=bool(literal),actual_dtype=str(x.dtype),original_dtype=str(y.dtype)))
    return dict(passed=set(a)==set(b) and all(r['passed'] for r in records),same_column_keys=set(a)==set(b),columns_checked=len(records),columns=records,
                excluded_provenance_column='metadata',timing_columns=list(timing_helper.TIMING_COLUMNS),debug_timing_keys=list(timing_helper.DEBUG_TIMING_KEYS))

def restore_declared_empty_J(problem):
    """JSON [] loses only the second axis of a DECLARED P=0 by25 matrix.

    Require every independently declared normal-axis vector to be exactly the
    empty JSON list. None/missing/other malformed shapes stay unchanged. No
    unknown coefficient or physical observation is replaced with zero.
    """
    if (isinstance(problem,dict) and problem.get('Jpx25')==[]
        and all(type(problem.get(k)) is list and len(problem[k])==0
                for k in ('y_p','patch_digit_p'))
        and isinstance(problem.get('scales'),dict)
        and type(problem['scales'].get('intentP')) is list and len(problem['scales']['intentP'])==0):
        restored=dict(problem);restored['Jpx25']=np.empty((0,25))
        return restored,True
    return problem,False


def audit_v19_task(q,executed,actor,evaluate_task):
    """Cross-interface arithmetic only; no optimizer/native evaluator call.

    Full F067 values here are recorded upstream evidence. These checks do not
    independently re-execute F067 or certify unknown native law branches.
    """
    errors=[];counts=Counter()
    def ck(name,condition):
        if not bool(condition):errors.append(name)
    def same(a,b):return bits(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    current=np.asarray(q['current_reference21'],float)
    lower=np.asarray(q['actual_reference_lower21'],float);upper=np.asarray(q['actual_reference_upper21'],float)
    motion=np.asarray(q['motion_reference21'],float);opened=np.asarray(q['open_reference21'],float)
    preferred=np.asarray(q['preferred_reference21'],float);proposed=np.asarray(q['proposed_reference21'],float)
    domain=np.asarray(q['gamma_domain'],float);phase=q['phase'];rho=q['reaction_share']
    if phase in ('SEARCH','CAPTURE'):expected_gamma=[0.,0.];unload=0.
    elif phase=='TRANSFER':expected_gamma=[0.,0. if rho is None else rho];unload=1.
    elif phase=='UNSUPPORTED_EXIT':expected_gamma=[1.,1.];unload=1.
    elif phase=='SEPARATE':expected_gamma=[0.,1.];unload=1.
    else:expected_gamma=None;unload=None
    ck('phase_gamma_domain',expected_gamma is not None and same(domain,expected_gamma))
    ck('reaction_share_not_filled_or_outside',rho is None or type(rho) in (float,int) and np.isfinite(rho) and 0<=rho<=1)
    ck('preferred_original_goal',same(preferred,np.minimum(upper,np.maximum(lower,motion+domain[1]*(opened-motion)))))
    ck('proposal_in_actual_task_domain',np.isfinite(proposed).all() and np.all(proposed>=lower) and np.all(proposed<=upper))
    branch=q['branch'];known=branch['status']=='KNOWN_CURRENT_DIRECTION_NATIVE_BRANCH'
    counts['branch:'+branch['status']]+=1;counts['phase:'+phase]+=1
    problem,restored=restore_declared_empty_J(q['qp_problem']);counts['declared_empty_J_shape_restored']+=restored
    qp=q['qp'];acceptance=q['proposal_acceptance'];native=q['native_actual'];score=q['actual_full_task_score']
    accepted=q['reference_effort_goal_accepted'];counts['accepted' if accepted else 'fallback']+=1
    counts['rounding_projection_changed']+=q['rounding_projection_changed']
    if known:
        ck('branch_authoritative_clock_reference',branch['time_s']==q['time_s'] and same(branch['current_reference21'],current) and same(branch['preferred_reference21'],preferred))
        bl=np.asarray(branch['reference_lower21'],float);bu=np.asarray(branch['reference_upper21'],float)
        ck('selected_branch_is_subset_of_task',np.all(bl>=lower) and np.all(bu<=upper) and np.all(bl<=bu))
        ck('branch_delta_original_reference',same(branch['delta_lower21'],bl-current) and same(branch['delta_upper21'],bu-current))
        ck('known_branch_has_full_problem',problem is not None)
    else:
        ck('unknown_branch_no_fabricated_problem',problem is None and qp is None and acceptance is None and native is None and score is None and not accepted)
    if problem is not None:
        for a,b in (('p0_21','p0_effort21'),('gain21','gain21'),('offset21','offset21'),('A21x25','transmission21x25'),('delta_lower21','delta_lower21'),('delta_upper21','delta_upper21')):
            ck('original_problem_branch_'+a,same(problem[a],branch[b]))
        ck('problem_original_reference_goals',same(problem['motion_step21'],motion-current) and same(problem['open_step21'],opened-motion))
        ck('problem_gamma_and_unload_not_retargeted',same([problem['gamma_lower'],problem['gamma_upper']],domain) and problem['unload_fraction']==unload)
    ev=None;candidate=False
    if qp is not None:
        counts['qp:'+qp['status']]+=1
        candidate=all(qp.get(k) is not None for k in ('du21','zP','gamma','d21','p21','objective'))
        if candidate:
            ev=evaluate_task(problem,np.asarray(qp['du21']),np.asarray(qp['zP']),qp['gamma'])
            ck('raw_candidate_available',ev.get('available',False))
            if ev.get('available'):
                ck('raw_qp_objective_bytes',same(ev['objective'],qp['objective']))
                ck('affine_offset_gain_delta_and_absolute_p',same(ev['d21'],qp['d21']) and same(np.asarray(problem['p0_21'])+ev['d21'],qp['p21']))
                ck('gamma_candidate_in_declared_domain',domain[0]<=qp['gamma']<=domain[1])
                ck('no_false_converged_KKT',qp['status']!='converged' or ev['direct_KKT_met'])
                if acceptance is not None and acceptance.get('accepted'):
                    anchor=acceptance['anchor']
                    av=evaluate_task(problem,np.asarray(anchor['du21']),np.asarray(anchor['zP']),anchor['gamma'])
                    ck('anchor_same_problem_available_feasible',av.get('available',False) and av.get('feasible',False))
                    ck('anchor_score_and_original_score_bytes',same(av['objective'],acceptance['anchor_objective']) and same(ev['objective'],acceptance['original_objective']))
                    ck('accepted_candidate_original_primal_and_merit',ev['feasible'] and ev['objective']<=av['objective']+acceptance['objective_roundoff_allowance'])
                    ck('certified_claim_matches_fresh_KKT',not acceptance.get('optimality_certified') or ev['direct_KKT_met'])
        else:
            ck('unknown_candidate_not_accepted',not accepted and qp['status']!='converged' and not (acceptance or {}).get('accepted'))
    if native is not None and native.get('available'):
        ck('native_full_value_authority_marker',native['status']=='FULL_NATIVE_REFERENCE_REEVALUATED' and native['original_p0_preserved'] and native['full_native_is_value_authority'] and native['affine_error_is_not_silently_zeroed'])
        ck('native_no_false_certificate_or_inverse',not native['writer_called'] and not native['inverse_called'] and not native['physical_safety_certified'] and not native['pressure_prediction_available'] and not native['future_response_available'])
        nr=np.asarray(native['reference21'],float);d=np.asarray(native['native_effort21'])-np.asarray(problem['p0_21'])
        affine=np.asarray(problem['offset21'])+np.asarray(problem['gain21'])*(nr-current)
        ck('native_original_p0_actual_delta_affine_error',same(native['p0_effort21'],problem['p0_21']) and same(native['actual_delta_effort21'],d) and same(native['affine_delta_effort21'],affine) and same(native['affine_error21'],d-affine))
        encoded=nr.copy();encoded[18]+=actor.lift_bias_m
        ck('native_engine_exactly_one_lift_bias',same(encoded,native['engine_port21']))
    if accepted:
        ck('accepted_requires_all_same_goal_gates',known and candidate and ev is not None and ev.get('available',False) and (acceptance or {}).get('accepted',False) and (native or {}).get('available',False) and (score or {}).get('available',False) and (score or {}).get('feasible',False))
        if not (known and candidate and (native or {}).get('available') and (score or {}).get('available')):return errors,counts
        raw=current+np.asarray(qp['du21']);actual=np.minimum(bu,np.maximum(bl,raw));correction=actual-raw
        allowance=64*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(raw),abs(actual)))
        ck('rounding_projection_exact_and_within_declared64eps',same(correction,q['rounding_projection21']) and np.all(abs(correction)<=allowance) and q['rounding_projection_changed']==(actual.tobytes()!=raw.tobytes()))
        ck('native_reference_is_rounded_QP_reference',same(actual,native['reference21']) and same(proposed,actual))
        fresh=evaluate_task(problem,actual-current,np.asarray(qp['zP']),qp['gamma'],actual_delta21=np.asarray(native['actual_delta_effort21']))
        ck('actual_full_task_available_and_primal',fresh.get('available',False) and fresh.get('feasible',False))
        if fresh.get('available'):
            for key in ('objective','residual25','d21','affine_d21','correction25','du21','zP','gamma','affine_realization_error21'):
                ck('actual_full_task_'+key+'_bytes',same(fresh[key],score[key]))
            ck('actual_score_substitutes_native_delta',score['native_delta_substituted'] is True and not score['physical_safety_certified'] and not score['future_pressure_prediction_available'])
            value=float(fresh['objective']);anchor=float(acceptance['anchor_objective']);merit_allowance=64*np.finfo(float).eps*max(1.,abs(value),abs(anchor))
            ck('actual_full_task_same_anchor_merit',value<=anchor+merit_allowance)
        # The recorded native score covers its exact proposed reference. A late
        # changed receipt cannot silently inherit that earlier task certificate.
        ck('accepted_native_score_is_final_executed_reference',same(executed,native['reference21']))
        ck('accepted_fallback_absent',q['fallback'] is None and q['numerical_error'] is None)
    else:
        ck('fallback_label_and_original_legal_preference_retained',q['fallback'] is not None and same(proposed,preferred))
        counts['fallback:'+str(q['fallback'])]+=1
        if q.get('numerical_error') is not None:counts['error:'+str(q['numerical_error'])]+=1
    intents=q['contact_intents']
    ck('contact_scope_does_not_invent_target_or_twist',intents.get('target_identity_unknown') is True and intents.get('current_external_twist_unknown') is True)
    for rec in intents.get('motion_records',[]):
        if not rec['external_projection_available']:
            ck('external_projection_unknown_not_zero_filled',rec['relative_slip2_m_s'] is None and rec['measured_external_projection2_m_s'] is None)
        ck('external_sample_age_labels',rec['acquisition_s']<=rec['read_time_s']+1e-9 and abs((rec['read_time_s']-rec['acquisition_s'])-rec['age_s'])<=1e-9)
    return errors,counts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--prepare',action='store_true');ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--out',default='review/v19_small_execution_independent_v1.json');args=ap.parse_args()
    if args.self_test:
        seen=[]
        def absent(p):seen.append(str(p));raise FileNotFoundError(p)
        assert not gate(dict(stages=[dict(name='a',output='NO_BRANCH')]),absent)['ready'] and len(seen)==1
        g=dict(recovery_reference_budget_m=.0007,downward_recovery_reference_budget_m=.0007,braking_intent_active=True)
        x,ready,_=task_interval(g,.05,-.1,.049825,.050175);assert ready and bits(x,np.array([.05,.050175]))
        x,ready,_=task_interval(g,.05,0.,.047825,.048175);assert not ready and x[0]==x[1]==.048175
        p=dict(Jpx25=[],y_p=[],patch_digit_p=[],scales=dict(intentP=[]))
        restored,yes=restore_declared_empty_J(p);assert yes and restored['Jpx25'].shape==(0,25) and p['Jpx25']==[]
        bad=dict(p,Jpx25=None);assert restore_declared_empty_J(bad)==(bad,False)
        bad=dict(p,y_p=[1.]);assert restore_declared_empty_J(bad)==(bad,False)
        print(json.dumps(dict(passed=True,pure_checks=6,branch_results_read=False,physics_steps=0)));return 0
    if args.prepare and not PLAN.exists():
        out=ROOT/'review/v19_small_execution_readiness_v1.json'
        if out.exists():raise FileExistsError(out)
        record=dict(schema='v19_execution_cross_review_readiness_v1',prepared=True,
            final_source_bound=False,registration_pending=True,reader_sha256=sha(__file__),
            branch_results_read=False,physics_steps=0,
            reviewer_role='V19 task_qp author cross-execution review; not independent math proof',
            execute_requires='Final registration/source identity and explicit root all4 terminal notification')
        out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(path=str(out),sha256=sha(out),**record)));return 0
    plan=read(PLAN);pin=sha(PLAN)
    if args.prepare:
        out=ROOT/'review/v19_small_execution_readiness_v1.json'
        if out.exists():raise FileExistsError(out)
        record=dict(schema='v19_execution_cross_review_readiness_v1',prepared=True,final_source_bound=True,plan_sha256=pin,
            source_count=len(plan['source_sha256']),current152_match=len(plan['source_sha256'])==152 and all(sha(source_path(k))==v for k,v in plan['source_sha256'].items()),
            reader_sha256=sha(__file__),branch_results_read=False,physics_steps=0,
            reviewer_role='V19 task_qp author cross-execution review, not independent mathematical kernel audit',
            scope=['all4 source/snapshot/assignment/completion retained','blind full37 and active pre12 completeJSON prefixes',
                   'all active5ms actual21/rate/mode/receipt','every coordinated5ms decision current clock and no interpolation',
                   'governor TASK interval and FINAL actual receipt/debt','originalp0/offset/gain/du/z/gamma, rounded actualdu/native total goal and fallback unknown'])
        out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(path=str(out),sha256=sha(out),prepared=True,branch_results_read=False)));return 0
    if not args.execute:raise ValueError('Explicit --prepare or terminal --execute required')
    terminal=gate(plan)
    if not terminal['ready']:print(json.dumps(dict(status='PENDING',terminal=terminal)));return 2
    out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    spec=importlib.util.spec_from_file_location('v18_saved_prefix',ROOT/'review/v16_small_prefix_storage_independent_v1.py');prefix=importlib.util.module_from_spec(spec);spec.loader.exec_module(prefix)
    timing_path=ROOT/'review/v18_passive_trace_comparison_v3.py'
    spec=importlib.util.spec_from_file_location('v18_exact_timing_contract',timing_path);timing=importlib.util.module_from_spec(spec);spec.loader.exec_module(timing)
    if timing.TIMING_COLUMNS!=('control_ms','actor_callback_ms','full_control_loop_ms','observation_ms','physics_advance_ms') or timing.DEBUG_TIMING_KEYS!=('qp','native_inverse','whole_decision'):
        raise ValueError('Unexpected timing whitelist; no generic timing-field omission allowed')
    from v19_contact_reference.task_qp import evaluate_task
    from v14_coord.design import reference_rates
    import research_imports,mujoco,hand_plant
    from foundation_config import FoundationConfig
    from v13_a1.codec import ReferenceActor21
    checks=[];issues=[];rows=[];inputs={};forbidden=[]
    def ck(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:issues.append(dict(name=name,detail=detail))
    def h(p):
        p=str(Path(p).resolve())
        if p not in inputs:inputs[p]=sha(p)
        return inputs[p]
    parent=read(ROOT/'review/v18_small_plan_v1.json')['source_sha256']
    ck('frozen parent144 unchanged',len(parent)==144 and all(plan['source_sha256'].get(k)==v for k,v in parent.items()))
    ck('all152 actual sources and42 binaries',len(plan['source_sha256'])==152 and all(h(source_path(k))==v for k,v in plan['source_sha256'].items()) and len(plan['environment']['native_binaries_sha256'])==42 and all(h(k)==v for k,v in plan['environment']['native_binaries_sha256'].items()))
    def deny(name):
        def f(*a,**kw):forbidden.append(name);raise AssertionError('Forbidden '+name)
        return f
    with ExitStack() as guards:
        for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_comPos','mj_collision','mj_contactForce'):
            if hasattr(mujoco,name):guards.enter_context(patch.object(mujoco,name,deny(name)))
        guards.enter_context(patch.object(hand_plant.HandPlant,'__init__',deny('HandPlant')))
        guards.enter_context(patch.object(ReferenceActor21,'apply',deny('actor.apply')))
        for stage in plan['stages']:
            tag=stage['name'];assignment=stage['assignment'];folder=ROOT/stage['output'];row=dict(stage=tag,assignment=assignment,status='unassessed');rows.append(row)
            try:
                done=read(folder/'execution_complete.json');reg=read(ROOT/stage['registration']);manifest=read(folder/'manifest.json');snap=read(folder/'source_snapshot_manifest.json')
                trials=read(folder/'trials.json');jobs=read(folder/'assignment_ledger.json');failures=read(folder/'runtime_failures.json')
                ck(tag+'/all assigned retained and unchanged',done['execution_complete'] and done['sources_unchanged'] and done['planned']==done['trials']==stage['planned']==1 and len(trials)==len(jobs)==1 and not failures and reg['assignment']==assignment and reg['source_sha256']==plan['source_sha256'] and reg['environment']==plan['environment'])
                ck(tag+'/snapshot152 and pinned env',manifest['research_source_sha256']==plan['source_sha256'] and set(snap['files'])==set(plan['source_sha256']) and all(h(folder/snap['files'][k]['snapshot_path'])==v for k,v in plan['source_sha256'].items()) and read(folder/'environment.json')['pinned_environment']==plan['environment'])
                ck(tag+'/terminal manifest hashes',all(done[k+'_sha256']==h(folder/(k+'.json')) for k in ('trials','assignment_ledger','plan','manifest')) and manifest['source_snapshot_manifest_sha256']==h(folder/'source_snapshot_manifest.json'))
                scene,method,seed=assignment['scenarios'][0],assignment['methods'][0],assignment['seeds'][0]
                run=folder/'runs'/f'{scene}__{method}__{seed}';meta=read(run/'metadata.json');result=read(run/'result.json')
                ck(tag+'/result assignment source and tracehash',equal(result,trials[0]) and result['scenario']==scene and result['method']==method and result['seed']==seed and meta['research_source_sha256']==plan['source_sha256'] and result['trace_sha256']==meta['trace_sha256']==h(run/'trace.npz'))
                with np.load(run/'trace.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
                t=a['time'];active=a['controller_active'].astype(bool);ck(tag+'/full16s clocks',meta['completed'] and np.array_equal(t,np.arange(3201)*.005) and len(a['substep_time'])==32000)
                old=ROOT/stage['original_same_condition'];ck(tag+'/original comparator trace retained',h(old/'trace.npz')==stage['original_trace_sha256'])
                with np.load(old/'trace.npz',allow_pickle=False) as z:b={k:z[k] for k in (z.files if assignment['receiver_policy']=='blind' else prefix.COLS)}
                compare=prefix.compare(a,b,t,b['time'],assignment['receiver_policy']!='blind')
                ck(tag+'/strict complete37 payload no field removal',all(v['exact_payload_equal'] for v in compare),[v for v in compare if not v['exact_payload_equal']])
                bridge=None
                if assignment['receiver_policy']=='blind':
                    bridge=passive_complete_bridge(a,b,timing)
                    ck(tag+'/complete16s all columns exact except declared v3 timing',bridge['passed'],[x for x in bridge['columns'] if not x['passed']])
                model,xmlsha=hand_plant._model(FoundationConfig(**meta['config']));actor=ReferenceActor21(model)
                encoded=a['executed_reference21'].copy();encoded[:,18]+=actor.lift_bias_m
                ck(tag+'/actual21 sole actor encoding and before snapshot',xmlsha==meta['model_sha256'] and bits(encoded,a['ctrl'][:,actor.indices21]) and bits(a['robot_actuator_target21'],a['ctrl_before'][:,actor.indices21]))
                obs=[json.loads(str(v)) for v in a['observation_json']];debug=[json.loads(str(v)) for v in a['debug_json']]
                actual_errors=[]
                for i in np.flatnonzero(active):
                    f=obs[i];ex=a['executed_reference21'][i];before=np.asarray(f['robot_reference21']);mode=debug[i]['mode'];gen=int(a['mode_generation'][i])
                    if np.any(ex<np.asarray(f['reference_lower21'])-1e-12) or np.any(ex>np.asarray(f['reference_upper21'])+1e-12) or np.max(abs(ex-before)-reference_rates(mode)*.005)>1e-12:actual_errors.append((int(i),'permission_rate'))
                    if gen!=f['mode_generation'] or gen!=debug[i]['mode_generation'] or abs(f['robot_control_time_s']-t[i])>1e-9 or not bits(ex-before,a['reference_increment21'][i]):actual_errors.append((int(i),'generation_clock_delta'))
                ck(tag+'/all active5ms actual authority mode clock',not actual_errors,actual_errors[:10])
                dm=read(run/'release_decision_manifest.json');ck(tag+'/decision archive hash',h(run/dm['file'])==dm['sha256'])
                with gzip.open(run/dm['file'],'rt',encoding='utf-8') as f:records=[json.loads(v) for v in f]
                expected=np.flatnonzero([active[i] and debug[i].get('architecture')=='v19_joint_reference_contact_progress' for i in range(len(t))])
                indices=np.array([round(q['time_s']/.005) for q in records],int)
                ck(tag+'/every coordinated action has unique5ms record',len(records)==dm['records'] and np.array_equal(indices,expected))
                errors=[];counts=Counter();debt=[]
                for k,q in enumerate(records):
                    i=indices[k];f=obs[i];current=np.asarray(q['current_reference21']);ex=a['executed_reference21'][i];g=q['governor'];receipt=q.get('execution_receipt')
                    if q['decision_id']!=k or q['time_s']!=t[i] or q['mode_generation']!=int(a['mode_generation'][i]) or q['process_version']!=q['public_process']['version'] or not bits(current,np.asarray(f['robot_reference21'])):errors.append((k,'decision_current_clock_gen'))
                    if q['public_process']['holding_force_tasks_active'] or q['future_motion_or_pressure_prediction_available'] or q['placement_certified']:errors.append((k,'false_claim_or_holding_leak'))
                    lower=np.asarray(f['reference_lower21']);upper=np.asarray(f['reference_upper21']);rates=reference_rates('release')
                    L=np.maximum(lower,current-rates*.005);U=np.minimum(upper,current+rates*.005)
                    if not bits(L,np.asarray(q['original_mechanical_rate_lower21'])) or not bits(U,np.asarray(q['original_mechanical_rate_upper21'])):errors.append((k,'original_authority_retargeted'))
                    qlift=float(f['robot_position25'][24]);vlift=float(f['robot_velocity25'][24]);task,reachable,band=task_interval(g,qlift,vlift,float(L[18]),float(U[18]))
                    if not bits(task,np.asarray(q['governor_task_domain']['task_reference_interval_m'])) or q['governor_task_domain']['budget_band_reachable_this_tick']!=reachable or g['q_m']!=qlift or g['actual_velocity_m_s']!=vlift or g['committed_reference_m']!=current[18]:errors.append((k,'governor_current_or_task_domain'))
                    if not bits(task,np.array([q['actual_reference_lower21'][18],q['actual_reference_upper21'][18]])) or not task[0]<=ex[18]<=task[1]:errors.append((k,'final_actual_outside_task_interval'))
                    if receipt is None:errors.append((k,'missing_final_receipt'));continue
                    d=qlift-float(ex[18]);debt.append(d);budget=(-g['downward_recovery_reference_budget_m']<=d<=g['recovery_reference_budget_m'])
                    changed=not bits(np.asarray(q['proposed_reference21']),ex)
                    if not bits(np.asarray(receipt['actual_reference21']),ex) or receipt['same_snapshot_q_lift_m']!=qlift or receipt['same_snapshot_reference_debt_m']!=d or receipt['current_q_budget_respected']!=budget or receipt['actual_task_interval_respected']!=(task[0]<=ex[18]<=task[1]) or receipt['execution_changed']!=changed:errors.append((k,'receipt_debt_or_flag_not_actual'))
                    counts['task_budget_outside']+=not budget;counts['execution_changed']+=changed;counts['stale']+=receipt['stale_plan_replaced']
                    extra,local_counts=audit_v19_task(q,ex,actor,evaluate_task)
                    errors.extend((k,reason) for reason in extra);counts.update(local_counts)
                ck(tag+'/all explicit reference branch tasks full native score and governor receipts honest',not errors,errors[:16])
                ck(tag+'/pressure forecast unknown remainsNaN',not np.any(a['prediction_available']) and np.isnan(a['predicted_normal']).all())
                row.update(status='audited',trace_sha256=h(run/'trace.npz'),metadata_sha256=h(run/'metadata.json'),result_sha256=h(run/'result.json'),decisions=len(records),counts=dict(counts),
                    actual_active5ms=int(active.sum()),final_debt_abs_max_m=None if not debt else float(np.max(abs(np.asarray(debt)))),
                    original_task_success=result['task_success'],original_full_motion=result['research_metrics']['full_motion_mission_success'],original_force_guard=result['research_metrics']['safety']['guard_pass'],
                    prefix37=compare,complete_passive_bridge=bridge,QP_residual_scope='Full saved problem and du/z/gamma are rescored; QP residual itself is not separately logged. Native full-task residual/objective and actual du are checked bytewise.',native_value_scope='Uses recorded upstream fullF067 marker; no independent native model evaluation here.')
            except Exception as exc:row.update(status='reader_error',error=repr(exc));issues.append(dict(name=tag+'/reader_exception',error=repr(exc),traceback=traceback.format_exc()))
    ck('all4 assigned independently assessed including failures',len(rows)==4 and all(r['status']=='audited' for r in rows))
    ck('source152 unchanged after read',all(sha(source_path(k))==v for k,v in plan['source_sha256'].items()))
    ck('no forbidden physical calls',not forbidden,forbidden)
    all_checked=len(rows)==4 and all(r['status']=='audited' for r in rows)
    result=dict(schema='v19_small_execution_cross_review_v1',passed=not issues and all_checked,assigned_count=4,all_assigned_checked=all_checked,checks=checks,issues=issues,rows=rows,terminal_gate=terminal,plan_sha256=pin,source_sha256=plan['source_sha256'],input_sha256=inputs,reader_sha256=sha(__file__),
        exact_timing_whitelist_helper_sha256=sha(timing_path),retained_initial_all_column_failure_sha256=sha(ROOT/'review/v18_passive_full16_bridge_v1.json'),
        retained_V18_reader_empty_J_failure_sha256=sha(ROOT/'review/v18_small_execution_independent_v1.json'),
        reader_correction='Only JSON empty J=[] with J=[], y=[], digit=[], and intentP=[] declarations explicitly empty is restored to declared(0,25) shape. No values, objective, tolerance, source, data or unknown fields changed.',
        reviewer_role='V19 task_qp author cross-execution review; not independent math or native model audit',scope=dict(assigned=4,physics_steps=0,plant_or_data_constructions=0,solver_replays=0,all_coordinated5ms_records_checked=all_checked),
        limitations=['No performance superiority or future/global certificate; all original outcome labels and failures retained.','Native effort values come from recorded fullF067 verification; this reader recomputes same-task arithmetic, not model dynamics.','Known software TASK debt is not stopping-time/velocity safety certification.'])
    out.write_text(json.dumps(plain(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    out.with_suffix('.md').write_text(f"# V19四条执行交叉审查\n\n{'PASS' if result['passed'] else 'FAIL'}；{sum(c['passed'] for c in checks)}/{len(checks)}，问题{len(issues)}。审者是V19 task_qp作者，本门为跨运行/执行证据核查，不构成该数学核的独立证明。\n\n逐5ms核21ports、mode/rate/clock、governor TASK域与最终receipt/debt、原p0/offset/gain/du/z/γ与实际原生完整同式总评分、fallback/unknown、原任务标签保留。完整分母/失败/范围见JSON。没有运行新物理，未改152源/旧144或结果，不作性能胜负/停止或放置证书。\n",encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),issues=issues[:8],out=str(out),sha256=sha(out)),ensure_ascii=False));return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

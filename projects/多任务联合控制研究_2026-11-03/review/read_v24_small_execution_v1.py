"""Terminal-only V24 nine-run saved-execution cross review. No optimizer.

Runtime integration author reviewing actual saved execution, not independent
proof of authored glue or a policy-performance certificate. Finite KKT is
rechecked at EACH historical cert.x using its complete saved dual witness;
the final emitted point is checked for preserved attainment/domain separately.
Only static MjModel compilation for canonical robot-port indices is allowed;
no MjData/plant/FK/forward/contact/step/actor writer is constructed or invoked.
"""
from pathlib import Path
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch
import argparse,gzip,hashlib,importlib.util,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(FOUNDATION));sys.dont_write_bytecode=True
PLAN=ROOT/'review/v24_small_plan_v1.json'
CONTEXTS=(('nominal','transport',619741),('fragile_conflict','hold',619741),('friction_drop','transport',619742))
VARIANTS=('v23_joint','canonical_backend','canonical_joint')
FORMULATIONS={'v23_joint':'joint','canonical_backend':'backend','canonical_joint':'joint'}
CANONICAL_LEVEL='canonical_minimum_normalized_reference_witness'
FROZEN_SOLVE_BUDGET=dict(max_iterations=400,atol=1e-9,rtol=1e-8)
PARENT_READER_SHA256='148f17ff62bea9f477df670b7177a2ae7d16d130c8dd9b47521ae73e5a4f612d'
HELPERS={
 'review/read_v23_small_execution_v1.py':'8c261bf571e4ff49d0490abd3d93f144234fcabda6767d2c3db13d19d09c4129',
 'review/read_v22_small_execution_v1.py':PARENT_READER_SHA256,
 'review/read_v21_small_execution_v1.py':'0e0d6d668260e86cf56151fd2f8b41a4886094d3af267d4f9cab6665e7ba3810',
 'review/read_v20_small_execution_v1.py':'042085fe6e2cc462af7eee3ec5a091441fa0d6624c377c89c51cff400d19fa0a',
 'review/v16_small_prefix_storage_independent_v1.py':'18934c10daec73cb58cefab5d7114436a311bbe6e676a68f45c6094e1a93bac7',
 'review/v18_passive_trace_comparison_v3.py':'fbbdabefaa0f7c9a893d535b2aab8fe0520395d58851aeb2e032f70e23bb3d0b',
 'review/v21_original_matrix_kkt_verifier_v1.py':'d7ae425d0963bdad1ba5842c00a42419479408a93d304994c2f24956e1daf8c5'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def matrix(a,m,n):
    if a is None:raise ValueError('unknown matrix cannot be zero-filled')
    x=np.asarray(a,float)
    if x.shape==(0,) and m==0:x=np.empty((0,n))
    if x.shape!=(m,n) or not np.isfinite(x).all():raise ValueError('declared matrix shape/finite mismatch')
    return x
def bits(a,b):
    a=np.asarray(a);b=np.asarray(b);return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def fbits(a,b):return bits(np.asarray(a,float),np.asarray(b,float))
def near(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return bool(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.all(abs(a-b)<=256*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(a),abs(b)))))
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def barrier(plan,markers):
    keys=[(s['variant'],*s['context_key']) for s in plan['stages']]
    if len(keys)!=9 or set(keys)!={(v,*c) for v in VARIANTS for c in CONTEXTS}:raise ValueError('exact9 original assignments required')
    pending=[]
    for s in plan['stages']:
        m=markers.get(s['name']);a=s['assignment']
        if (s['variant'],*s['context_key'])!=(a['release_variant'],a['scenarios'][0],a['motion_mode'],a['seeds'][0]):raise ValueError('stage/assignment mismatch')
        if not (m and m.get('execution_complete') is True and m.get('sources_unchanged') is True and m.get('numerical_unchanged') is True and m.get('runtime_failures')==[] and m.get('planned')==m.get('trials')==1):pending.append(s['name'])
    return dict(ready=not pending,assigned=9,pending=pending,outcomes_read=False)

def layer_route_checks(levels,scope='task'):
    """V23 has TWO branches, one face per branch, at most two per layer.

    Explicit original-box candidate generation is not another face solve.
    Failed numerical attempts are counted too, never promoted to certificates.
    """
    errors=[];counts=Counter()
    for level in levels:
        counts[scope+':numerical_layers_logged']+=1
        attempts=level.get('attempts')
        if not isinstance(attempts,(list,tuple)):errors.append('missing_actual_numerical_attempts');continue
        faces=Counter();boxes=Counter();seen_slsqp=False
        for a in attempts:
            candidate=a.get('candidate','');g=a.get('generation') or {}
            if candidate.startswith('SLSQP'):seen_slsqp=True
            if candidate in ('SLSQP_branch_rejected_generation_or_check','FW_branch_rejected_generation_or_check'):
                counts[scope+':'+candidate]+=1
            if candidate=='FW_fallback_gradient_LP_exact_segment':
                counts[scope+':FW_direction_attempts']+=1
                if not seen_slsqp:errors.append('FW_before_declared_SLSQP_branch')
            if candidate.endswith('_tight_face'):
                branch=candidate[:-len('_tight_face')];faces[branch]+=1
                if branch not in ('SLSQP_branch','FW_branch') or g.get('candidate_branch')!=branch or g.get('face_threshold_attempts')!=1:errors.append('unknown_or_retried_face_branch')
            if candidate.endswith('_explicit_original_coordinate_box_candidate'):
                branch=candidate[:-len('_explicit_original_coordinate_box_candidate')];boxes[branch]+=1
                if branch not in ('SLSQP_branch','FW_branch') or g.get('candidate_branch')!=branch or faces[branch]!=1:errors.append('box_candidate_not_bound_to_once_face_branch')
        if any(v>1 for v in faces.values()) or sum(faces.values())>2:errors.append('more_than_one_face_per_branch_or_two_per_layer')
        if any(v>1 for v in boxes.values()):errors.append('repeated_original_box_candidate_in_branch')
        counts[scope+':recorded_SLSQP_face_refinements']+=faces['SLSQP_branch'];counts[scope+':recorded_FW_face_refinements']+=faces['FW_branch']
        counts[scope+':two_face_layers']+=sum(faces.values())==2;counts[scope+':explicit_box_candidates']+=sum(boxes.values())
    return errors,counts

def witness_certificate_checks(cert,n,atol,rtol,verifier):
    """Check an UNsubmitted normal witness only at its own saved point."""
    point=np.asarray(cert['x'],float);A=np.asarray(cert['matrix'],float);b=np.asarray(cert['target'],float);k=cert['KKT'];dom=cert['hard_domain']
    G=matrix(dom['normalized_inequality_matrix'],len(dom['normalized_inequality_upper']),n);h=np.asarray(dom['normalized_inequality_upper'])
    E=matrix(k['full_equality_matrix'],len(k['full_equality_target']),n);f=np.asarray(k['full_equality_target']);prior=cert['previous_frozen']
    base=matrix(dom['fixed_equality_matrix'],len(dom['fixed_equality_target']),n);target=np.asarray(dom['fixed_equality_target'])
    originalE=np.vstack([base]+[np.asarray(p['matrix']) for p in prior]);originalf=np.concatenate([target]+[np.asarray(p['attained']) for p in prior])
    problem=dict(A=A.tolist(),b=b.tolist(),G=G.tolist(),h=h.tolist(),E=E.tolist(),f=f.tolist(),previous_frozen=prior,atol=atol,rtol=rtol)
    proof=verifier.verify_stage(problem,point,duals=dict(inequality_unit=k['full_normalized_inequality_dual'],equality_raw=k['equality_raw_dual'],provenance='unsubmitted_normal_witness_at_its_own_historical_point'))
    residual=A@point-b
    return bool(point.shape==(n,) and fbits(E,originalE) and fbits(f,originalf) and proof.get('kkt_passed') is True
        and np.all(point>=np.asarray(dom['lower'])) and np.all(point<=np.asarray(dom['upper']))
        and near(A@point,cert['attained']) and near(residual,cert['residual']) and near(residual@residual,cert['squared_loss']))

def canonical_checks(q,P,verifier):
    """The selector certifies its OWN historical point, never the emitted task.

    The immutable supervisor calls use the declared 400/1e-9/1e-8 budget even
    when an early failed-witness diagnostic omits the successful-task budget.
    This is a source-declared numerical contract, not unknown sensor padding.
    """
    errors=[];counts=Counter();h=q['hierarchy'];d=h['diagnostics'];actual=h['certified_prefix']
    def ck(name,good):
        if not good:errors.append(name)
    pair=d.get('normal_witness_certified_prefix',[]);levels=d.get('normal_witness_levels',[])
    es,cs=layer_route_checks(levels,'canonical_witness');errors.extend(es);counts.update(cs)
    if P==0:
        ck('known_P0_has_no_canonical_witness_or_extra_layer',not pair and not levels
            and d.get('canonical_witness_required') is False and d.get('canonical_witness_available') is False
            and d.get('canonical_witness_certified') is False and d.get('canonical_witness_not_required_knownP0') is True
            and d.get('normal_first_certificate_is_submitted_prefix') is False)
        counts['known_P0_canonical_not_required']+=1
    elif P:
        if d.get('canonical_witness_required') is not True:
            ck('input_unavailable_never_has_actual_certificate_or_budget',not actual and not h['levels']
               and h['completed_prefix']==0 and not h['fully_converged'] and h['failed_layer'] is None
               and not d.get('normal_witness_budget_defined',False))
            counts['task_input_unavailable_before_witness']+=1
        else:
            budget=d.get('numerical_budget',FROZEN_SOLVE_BUDGET)
            ck('source_declared_numerical_budget_not_changed',budget==FROZEN_SOLVE_BUDGET)
            pressure=np.asarray(q['contact_tasks']['pressure_P_N'],float)
            relative=pressure/pressure.max();shares=relative/relative.sum()
            n=21+P;cost=np.zeros((P,n));cost[:,21:]=np.diag(np.sqrt(shares));selector=np.zeros((21,n));selector[:,:21]=np.eye(21)
            ck('canonical_prefix_length_and_status_honest',len(pair)==d.get('witness_completed_prefix') and len(pair)<=2)
            for i,c in enumerate(pair):
                ck('canonical_historical_cert_original_KKT',len(c['x'])==n and witness_certificate_checks(c,n,budget['atol'],budget['rtol'],verifier))
                ck('canonical_witness_indices_and_roles',c.get('witness_index')==i and c.get('solver_local_index')==i
                   and c.get('task_stage')=='canonical_normal_witness'
                   and c.get('certificate_role')==('NORMAL_OPTIMUM_WITNESS' if i==0 else 'CANONICAL_SELECTOR_WITNESS_ONLY_NEVER_SUBMITTED'))
                ck('canonical_original_objective_not_forged',c['name']==('normal_inward_correction_slack' if i==0 else CANONICAL_LEVEL)
                   and fbits(c['matrix'],cost if i==0 else selector) and fbits(c['target'],np.zeros(P if i==0 else 21)))
                if i==1:
                    ck('selector_preserves_only_normal_attainment',len(c['previous_frozen'])==1
                       and fbits(c['previous_frozen'][0]['matrix'],pair[0]['matrix'])
                       and fbits(c['previous_frozen'][0]['attained'],pair[0]['attained'])
                       and fbits(c['hard_domain']['lower'],pair[0]['hard_domain']['lower'])
                       and fbits(c['hard_domain']['upper'],pair[0]['hard_domain']['upper']))
                counts['canonical_witness_certificates_checked']+=1
                counts['canonical_selector_witnesses_checked']+=i==1
            ready=d.get('canonical_witness_available') is True
            ck('pair_never_claimed_as_actual_prefix',d.get('normal_witness_is_submitted_prefix') is False
               and d.get('canonical_witness_certificate_submitted') is False)
            if ready:
                ck('ready_witness_is_both_certificates',len(pair)==len(levels)==2 and d.get('canonical_witness_certified') is True
                   and d.get('witness_failed_layer') is None)
                if len(pair)==2:
                    ck('motion_budget_anchor_is_actual_canonical_selector_point',fbits(d['normal_witness_final_extended_anchor'],pair[1]['x'])
                       and fbits(d['certified_L1_extended_anchor'],pair[1]['x'])
                       and fbits(d['normal_witness_optimal_eta_P'],np.asarray(pair[1]['x'])[21:]))
                counts['canonical_witness_ready']+=1
                backend=q['formulation']=='backend'
                ck('only_original_normal_can_be_submitted',d.get('normal_first_certificate_is_submitted_prefix') is backend
                   and d.get('normal_optimal_slack_frozen_before_motion') is backend)
                if backend:
                    ck('backend_first_actual_is_original_normal_only',bool(actual) and actual[0]['name']==pair[0]['name']
                       and fbits(actual[0]['matrix'],pair[0]['matrix']) and fbits(actual[0]['x'],pair[0]['x']))
                else:ck('joint_never_submits_old_normal_optimum',all(c['name']!='normal_inward_correction_slack' for c in actual))
            else:
                ck('failed_witness_never_defines_budget_or_actual_layer',not actual and not h['levels']
                   and h['completed_prefix']==0 and not h['fully_converged'] and h['failed_layer'] is None
                   and d.get('canonical_witness_certified') is False and d.get('normal_witness_budget_defined') is False
                   and d.get('actual_increment_available') is False and d.get('actual_submitted_certificate_count')==0
                   and d.get('motion_task',{}).get('available') is False and 'second_stage_hard_matrix' not in d)
                failure=d.get('witness_failed_layer')
                ck('witness_failed_index_is_witness_local_not_actual',failure is None or failure.get('index')==len(pair))
                counts['canonical_witness_unavailable']+=1
                counts['canonical_witness_failed_after_normal_only']+=len(pair)==1
                if failure:counts['witness_failed_layer:'+str(failure.get('name'))+':'+str(failure.get('status'))]+=1
    ck('selector_never_enters_actual_tasks_or_frozen_history',all(c['name']!=CANONICAL_LEVEL
       and all(p.get('name')!=CANONICAL_LEVEL for p in c.get('previous_frozen',[])) for c in actual)
       and all(l['name']!=CANONICAL_LEVEL for l in h['levels']))
    ck('actual_prefix_global_indices_contiguous',all(c.get('index')==i for i,c in enumerate(actual))
       and all(l.get('index')==i for i,l in enumerate(h['levels'])))
    if h['failed_layer'] is not None:
        ck('actual_failed_index_is_global_completed_count',h['failed_layer']['index']==h['completed_prefix'])
    if d.get('canonical_witness_available') is True or P==0:
        offset=int(bool(P) and q['formulation']=='backend')
        ck('actual_solver_local_indices_separate_from_global',all(c.get('solver_local_index')==(0 if i<offset else i-offset)
           for i,c in enumerate(actual)) and all(l.get('solver_local_index')==(0 if i<offset else i-offset)
           for i,l in enumerate(h['levels'])))
    return errors,counts

def finite_certificates(q,frame,actual,rates,verifier,actual_engine21):
    """No production solve/dual recovery. Historical certs and emitted x differ."""
    errors=[];count=Counter();h=q.get('hierarchy');accepted=q['geometry_proposal_accepted'];ct=q.get('contact_tasks')
    def ck(name,good):
        if not good:errors.append(name)
    P=None if not ct or ct.get('available') is not True else len(ct['pressure_P_N'])
    count['P_unknown' if P is None else 'P0' if P==0 else 'positive_pressure']+=1
    if P==0:ck('P0_explicit_empty_geometry_not_unknown_padding',ct.get('normal_jacobian_Px25')==[] and ct.get('Npx21')==[] and ct['diagnostics'].get('normal_task_available') is False and ct['diagnostics'].get('partial_geometry') is False)
    if h is None:
        count['direct' if q['task_role']=='direct_frozen_search_capture_governor' else 'adapter_unknown_or_unavailable']+=1
        if q['task_role']!='direct_frozen_search_capture_governor':ck('unknown_legal_anchor',not accepted and fbits(actual,q['feasible_recovery_anchor21']))
        return errors,count
    d=h['diagnostics'];certs=h['certified_prefix'];raw=np.asarray(d.get('extended_normalized_x',h['x']),float);n=len(raw)
    count['task_decisions']+=1;ck('prefix_count_not_fake_full',len(certs)==h['completed_prefix'] and (not h['fully_converged'] or h['failed_layer'] is None))
    es,cs=layer_route_checks(h['levels'],'task_or_early_failure');errors.extend(es);count.update(cs)
    es,cs=canonical_checks(q,P,verifier);errors.extend(es);count.update(cs)
    if h.get('failed_layer'):
        failure=h['failed_layer'];count['failed_layer:'+str(failure.get('name'))+':'+str(failure.get('status'))]+=1
    if not certs:
        # A legitimate early L1 failure has no optimum-eta or extended task
        # certificate fields. An unsubmitted joint witness is different from
        # a submitted prefix. Neither may be upgraded to execution success.
        count['no_certified_task']+=1;count['candidate_not_committed_fallback']+=1
        witness=bool(d.get('normal_witness_certified_prefix'))
        count['witness_only_unsubmitted']+=q['formulation']=='joint' and d.get('canonical_witness_available') is True
        count['witness_incomplete_unsubmitted']+=witness and d.get('canonical_witness_available') is not True
        count['normal_witness_unavailable']+=not witness
        ck('empty_prefix_explicit_actual_legal_anchor_not_task_certificate',h['completed_prefix']==0 and not h['fully_converged'] and not accepted and q['fallback']=='legal_current_q_recovery_anchor_without_exit_certificate' and fbits(actual,q['feasible_recovery_anchor21']))
        return errors,count
    if q['formulation']=='joint':
        ck('joint_no_submitted_old_L1_optimum',d['normal_witness_is_submitted_prefix'] is False and d['normal_optimal_slack_frozen_before_motion'] is False and all(c['name']!='normal_inward_correction_slack' for c in certs))
        count['witness_only_unsubmitted']+=bool(d.get('normal_witness_certified_prefix')) and not bool(certs)
    elif P:ck('backend_keeps_original_eta_optimum',d['normal_witness_is_submitted_prefix'] is False and d['normal_first_certificate_is_submitted_prefix'] is True and d['normal_optimal_slack_frozen_before_motion'] is True)
    current=np.asarray(q['current_reference21']);scale=rates*.005;emitted=raw.copy();emitted[:21]=(np.asarray(q['proposed_reference21'])-current)/scale;actualx=raw.copy();actualx[:21]=(actual-current)/scale
    coord=np.zeros(n);coord[:21]=128*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(current),abs(np.asarray(q['proposed_reference21'],float))))/scale
    ck('real_eta_coordinate_dimensions',n==21+(P or 0) and np.asarray(h['x']).shape==(21,))
    if P:
        ck('real_joint_or_fixed_backend_eta',near(raw[21:],d['certified_slack_eta_P']) and near(raw[21:]*np.asarray(d['normal_effective_report_units_P_m']),d['final_slack_P_m']))
    score=q.get('emitted_hierarchy_score')
    if accepted:ck('accepted_score_available',score is not None and score['available'] and len(score['levels'])==len(certs))
    for i,c in enumerate(certs):
        point=np.asarray(c['x'],float);A=np.asarray(c['matrix'],float);b=np.asarray(c['target'],float);k=c['KKT'];dom=c['hard_domain'];tol=k['coordinate_tolerance']
        lo=np.asarray(dom['lower']);hi=np.asarray(dom['upper']);G=matrix(dom['normalized_inequality_matrix'],len(dom['normalized_inequality_upper']),n);gh=np.asarray(dom['normalized_inequality_upper']);E=matrix(k['full_equality_matrix'],len(k['full_equality_target']),n);f=np.asarray(k['full_equality_target'])
        baseE=matrix(dom['fixed_equality_matrix'],len(dom['fixed_equality_target']),n);basef=np.asarray(dom['fixed_equality_target']);prior=c['previous_frozen'];expectedE=np.vstack([baseE]+[np.asarray(z['matrix']) for z in prior]);expectedf=np.concatenate([basef]+[np.asarray(z['attained']) for z in prior])
        ck('all_original_equalities_and_frozen_rows_present',fbits(E,expectedE) and fbits(f,expectedf))
        problem=dict(kind='least_squares',A=A.tolist(),b=b.tolist(),G=G.tolist(),h=gh.tolist(),E=E.tolist(),f=f.tolist(),previous_frozen=prior,atol=d['numerical_budget']['atol'],rtol=d['numerical_budget']['rtol'])
        # At this layer's own historical certified point, NOT final command.
        proof=verifier.verify_stage(problem,point,duals=dict(inequality_unit=k['full_normalized_inequality_dual'],equality_raw=k['equality_raw_dual'],provenance='saved_current_layer_complete_dual_witness'))
        ck('original_finite_KKT_at_historical_cert_x',proof.get('kkt_passed') is True and np.all(point>=lo) and np.all(point<=hi));count['finite_certificates_checked']+=1
        att=A@point;res=att-b;ck('saved_original_attainment',near(att,c['attained']) and near(res,c['residual']) and near(res@res,c['squared_loss']))
        returned=A@raw;ck('returned_point_preserves_completed_layer',np.all(abs(returned-np.asarray(c['attained']))<=tol))
        if accepted:
            ae=A@emitted;ax=A@actualx;allow=abs(A)@coord+128*np.finfo(float).eps*np.maximum(1.,abs(A)@abs(emitted))
            ck('emitted_and_actual_preserve_accepted_attainment',np.all(abs(ae-returned)<=allow) and np.all(abs(ax-ae)<=allow))
            for z in (emitted,actualx):ck('actual_hard_domain_not_only_old_cert_point',np.all(G@z-gh<=tol+abs(G)@coord) and np.all(abs(E@z-f)<=tol+abs(E)@coord))
            recorded=score['levels'][i];ck('actual_same_objective_score',near(ae,recorded['attained']) and near(ae-b,recorded['residual']) and near((ae-b)@(ae-b),recorded['squared_loss']))
    if accepted:
        count['accepted_full' if h['fully_converged'] else 'accepted_partial_prefix']+=1
        ck('full_native_fixed_snapshot_available',q['native_current']['status']==q['native_proposed']['status']=='KNOWN_ROBOT_ACTUATION_ALGEBRA')
        ck('incoming_native_command_is_public21',fbits(q['native_current']['commanded_target21'],frame['robot_actuator_target21']))
        ck('proposed_native_command_is_actual_engine21',fbits(q['native_proposed']['commanded_target21'],actual_engine21))
        for name in ('actuator_length21','actuator_velocity21','transmission21x25'):ck('native_current_and_proposed_same_robot_'+name,fbits(q['native_current'][name],q['native_proposed'][name]))
    else:count['candidate_not_committed_fallback']+=1;ck('unaccepted_candidate_not_fake_emitted_certificate',q['fallback']=='legal_current_q_recovery_anchor_without_exit_certificate' and fbits(actual,q['feasible_recovery_anchor21']))
    return errors,count

def execute(plan,out):
    markers={s['name']:read(ROOT/s['output']/'execution_complete.json') for s in plan['stages'] if (ROOT/s['output']/'execution_complete.json').exists()}
    state=barrier(plan,markers)
    if not state['ready']:raise ValueError('all9terminal barrier pending; no result/metadata/raw read')
    if out.exists():raise FileExistsError(out)
    for name,s in HELPERS.items():
        if sha(ROOT/name)!=s:raise ValueError('frozen execution helper changed:'+name)
    frozen_v23=module('v24_frozen_V23_exec',ROOT/'review/read_v23_small_execution_v1.py')
    helper=module('v24_old_exec',ROOT/'review/read_v20_small_execution_v1.py');oldprefix=module('v23_old_prefix',ROOT/'review/read_v21_small_execution_v1.py');prefix=module('v23_prefix37',ROOT/'review/v16_small_prefix_storage_independent_v1.py');timing=module('v23_timing_v3',ROOT/'review/v18_passive_trace_comparison_v3.py');verifier=module('v23_saved_KKT',ROOT/'review/v21_original_matrix_kkt_verifier_v1.py')
    checks=[];issues=[];rows=[];inputs={};source=plan['source_sha256'];num=plan['numerical_identity'];forbidden=[];static_compiles=0
    def ck(name,good,detail=None):
        checks.append(dict(name=name,passed=bool(good),detail=detail))
        if not good:issues.append(dict(name=name,detail=detail))
    def h(p):
        key=str(Path(p).resolve())
        if key not in inputs:inputs[key]=sha(key)
        return inputs[key]
    def sourcepath(k):return FOUNDATION/k[11:] if k.startswith('foundation/') else ROOT/k
    ck('all192_current_sources',len(source)==192 and all(h(sourcepath(k))==v for k,v in source.items()))
    parent186=read(ROOT/'review/v23_small_plan_v1.json')['source_sha256'];ck('parent186_frozen_exact',len(parent186)==186 and parent186==plan['reference_sources']['v23'] and all(source.get(k)==v for k,v in parent186.items()))
    parent=read(ROOT/'review/v22_small_plan_v1.json')['source_sha256'];ck('parent179_frozen_exact',len(parent)==179 and parent==plan['reference_sources']['v22'] and all(source.get(k)==v for k,v in parent.items()))
    frozen=read(ROOT/'review/v21_complete16_plan_v1.json')['source_sha256'];ck('original170_and42_unchanged',len(frozen)==170 and frozen==plan['reference_sources']['v21'] and all(source.get(k)==v for k,v in frozen.items()) and plan['environment']==num['legacy_pinned_environment'] and len(plan['environment']['native_binaries_sha256'])==42)
    deps=dict(plan['environment']['native_binaries_sha256']);deps[plan['environment']['executable']]=plan['environment']['executable_sha256'];deps.update(num['scipy_native_binaries_sha256']);deps.update(num['necessary_runtime_dll_sha256']);deps.update({v['path']:v['sha256'] for v in num['scipy_python_implementation_sha256'].values()})
    def declared_python_files(value):
        if isinstance(value,dict):
            path=value.get('path',value.get('file'));digest=value.get('sha256',value.get('init_sha256'))
            if isinstance(path,str) and isinstance(digest,str) and len(digest)==64:deps[path]=digest
            for child in value.values():declared_python_files(child)
        elif isinstance(value,list):
            for child in value:declared_python_files(child)
    declared_python_files(num['setup']);ck('all_declared_numeric_dependency_bytes',all(h(k)==v for k,v in deps.items()))
    import research_imports,mujoco,hand_plant
    from foundation_config import FoundationConfig
    from v13_a1.codec import ReferenceActor21
    from v14_coord.design import reference_rates
    def deny(*a,**kw):forbidden.append('forbidden_model_pipeline_or_writer');raise AssertionError(forbidden[-1])
    refs={tuple(r['context_key']):r for r in plan['original_v23_joint_bridges']};models={}
    with ExitStack() as guards:
        for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_comPos','mj_collision','mj_contactForce'):
            if hasattr(mujoco,name):guards.enter_context(patch.object(mujoco,name,deny))
        guards.enter_context(patch.object(hand_plant.HandPlant,'__init__',deny));guards.enter_context(patch.object(ReferenceActor21,'apply',deny))
        for s in plan['stages']:
            tag=s['name'];variant=s['variant'];key=tuple(s['context_key']);folder=ROOT/s['output'];row=dict(stage=tag,variant=variant,context_key=list(key),status='unassessed');rows.append(row)
            try:
                mark=markers[tag];reg=read(ROOT/s['registration']);manifest=read(folder/'manifest.json');snap=read(folder/'source_snapshot_manifest.json');trials=read(folder/'trials.json');jobs=read(folder/'assignment_ledger.json')
                ck(tag+'/registered_assignment_and_terminal_hashes',h(ROOT/s['registration'])==s['registration_sha256'] and reg['assignment']==s['assignment'] and all(reg[k]==plan[k] for k in ('source_sha256','environment','numerical_identity')) and all(mark[field]==h(folder/file) for file,field in [('trials.json','trials_sha256'),('assignment_ledger.json','assignment_ledger_sha256'),('plan.json','plan_sha256'),('manifest.json','manifest_sha256')]))
                ck(tag+'/all192_snapshots_and_extra_numeric_identity',set(snap['files'])==set(source) and all(h(folder/snap['files'][k]['snapshot_path'])==v for k,v in source.items()) and manifest['source_snapshot_manifest_sha256']==h(folder/'source_snapshot_manifest.json') and manifest['research_source_sha256']==source and manifest['numerical_identity']==num and read(folder/'environment.json')['pinned_environment']==plan['environment'] and read(folder/'environment.json')['numerical_identity']==num)
                run=folder/'runs'/f'{key[0]}__ratio_effort__{key[2]}';meta=read(run/'metadata.json');result=read(run/'result.json');scene=meta['scenario']['name'] if isinstance(meta['scenario'],dict) else meta['scenario']
                ck(tag+'/one_assigned_actual_result_all_failures_retained',len(trials)==len(jobs)==1 and jobs[0]['status']=='evaluated' and Path(jobs[0]['job']['destination']).resolve()==run.resolve() and result==trials[0] and (scene,meta['motion_mode'],meta['seed'])==key and meta['release_variant']==variant and meta['research_source_sha256']==source and meta['numerical_identity']==num and result['trace_sha256']==meta['trace_sha256']==h(run/'trace.npz'))
                original=Path(refs[key]['run']);ck(tag+'/originalV23joint_reference_pins',all(h(original/n)==v for n,v in refs[key]['input_sha256'].items()));om=read(original/'metadata.json');ck(tag+'/same_physical_noise_material',all(meta[k]==om[k] for k in ('config','noise','limits','budget','motion_mode','seed')))
                with np.load(run/'trace.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
                with np.load(original/'trace.npz',allow_pickle=False) as z:
                    compare=prefix.compare(a,z,a['time'],z['time'],variant!='v23_joint')
                    mask=a['time']<12.-1e-9 if variant!='v23_joint' else np.ones(len(a['time']),bool)
                    public_equal=all(str(x).encode('utf-8')==str(y).encode('utf-8') for x,y in zip(a['observation_json'][mask],z['observation_json'][mask]))
                    bridge=timing.compare_arrays(a,z) if variant=='v23_joint' else None
                ck(tag+'/complete37_numeric_and_UTF8_public',all(c['exact_payload_equal'] for c in compare) and public_equal,[c for c in compare if not c['exact_payload_equal']])
                if bridge is not None:ck(tag+'/entire16s_A_bridge_only_fixed_timing_whitelist',bridge['passed'],[k for k,v in bridge['checks'].items() if not v['passed']])
                time=a['time'];active=a['controller_active'].astype(bool);ck(tag+'/full16_sensor_raw_clock',meta['completed'] and np.array_equal(time,np.arange(3201)*.005) and len(a['substep_time'])==32000)
                if key not in models:
                    model,xml=hand_plant._model(FoundationConfig(**meta['config']));models[key]=(ReferenceActor21(model),xml);static_compiles+=1
                actor,xml=models[key];encoded=a['executed_reference21'].copy();encoded[:,18]+=actor.lift_bias_m
                ck(tag+'/sole_actual21_port_and_unbiased_lift_once',xml==meta['model_sha256'] and bits(encoded,a['ctrl'][:,actor.indices21]) and bits(a['robot_actuator_target21'],a['ctrl_before'][:,actor.indices21]))
                obs=[json.loads(str(x)) for x in a['observation_json']];debug=[json.loads(str(x)) for x in a['debug_json']];errors=[]
                for i in np.flatnonzero(active):
                    f=obs[i];ex=a['executed_reference21'][i];cur=np.asarray(f['robot_reference21']);rates=reference_rates(debug[i]['mode']);gen=int(a['mode_generation'][i])
                    if np.any(ex<np.asarray(f['reference_lower21'])-1e-12) or np.any(ex>np.asarray(f['reference_upper21'])+1e-12) or np.max(abs(ex-cur)-rates*.005)>1e-12:errors.append((int(i),'actual_mode_mechanical_rate'))
                    if gen!=f['mode_generation'] or gen!=debug[i]['mode_generation'] or abs(f['robot_control_time_s']-time[i])>1e-9 or not fbits(ex-cur,a['reference_increment21'][i]):errors.append((int(i),'same_clock_generation_increment'))
                ck(tag+'/all5ms_actual21_permissions',not errors,errors[:20]);dm=read(run/'release_decision_manifest.json');ck(tag+'/original_decision_gzip_identity',h(run/dm['file'])==dm['sha256'])
                counts=Counter();errors=[];indices=[]
                with gzip.open(run/dm['file'],'rt',encoding='utf-8') as f:
                    for k,line in enumerate(f):
                        q=json.loads(line);i=round(q['time_s']/.005);indices.append(i);frame=obs[i];ex=a['executed_reference21'][i];cur=np.asarray(q['current_reference21']);g=q['governor'];rates=reference_rates('release');receipt=q.get('execution_receipt')
                        if q['decision_id']!=k or q['time_s']!=time[i] or q['mode_generation']!=int(a['mode_generation'][i]) or not fbits(cur,frame['robot_reference21']):errors.append((k,'saved_decision_same_clock_current'))
                        if q['public_process']['holding_force_tasks_active'] or q['placement_certified'] or q['future_motion_or_pressure_prediction_available']:errors.append((k,'false_prediction_or_holding'))
                        if q.get('formulation')!=FORMULATIONS[variant]:errors.append((k,'actual_formulation'))
                        L=np.maximum(np.asarray(frame['reference_lower21']),cur-rates*.005);U=np.minimum(np.asarray(frame['reference_upper21']),cur+rates*.005)
                        if not fbits(L,q['original_mechanical_rate_lower21']) or not fbits(U,q['original_mechanical_rate_upper21']):errors.append((k,'original_authority'))
                        ql=float(frame['robot_position25'][24]);vl=float(frame['robot_velocity25'][24]);interval,reach=helper.independent_lift_interval(g,ql,vl,L[18],U[18],q['phase'] in helper.EXIT_PHASES)
                        if not fbits(interval,q['governor_task_domain']['task_reference_interval_m']) or q['governor_task_domain']['budget_band_reachable_this_tick']!=reach:errors.append((k,'original_governor_TASK_domain'))
                        if receipt is None:errors.append((k,'missing_actual_receipt'));continue
                        debt=ql-float(ex[18]);budget=-g['downward_recovery_reference_budget_m']<=debt<=g['recovery_reference_budget_m'];changed=not fbits(q['proposed_reference21'],ex)
                        if not fbits(receipt['actual_reference21'],ex) or receipt['same_snapshot_reference_debt_m']!=debt or receipt['current_q_budget_respected']!=budget or receipt['actual_task_interval_respected']!=(interval[0]<=ex[18]<=interval[1]) or receipt['execution_changed']!=changed:errors.append((k,'actual_receipt_debt_domain'))
                        counts['execution_changed']+=changed;counts['stale_plan_replaced']+=receipt['stale_plan_replaced'];counts['phase_'+q['phase']]+=1
                        es,cs=frozen_v23.finite_certificates(q,frame,ex,rates,verifier,a['ctrl'][i,actor.indices21]) if variant=='v23_joint' else finite_certificates(q,frame,ex,rates,verifier,a['ctrl'][i,actor.indices21]);errors.extend((k,e) for e in es);counts.update(cs)
                expected=np.flatnonzero([active[i] and debug[i].get('architecture')==('v23_task_execution_exit' if variant=='v23_joint' else 'v24_canonical_task_fusion_exit') for i in range(len(time))]);ck(tag+'/all_actual_release_decisions_retained',len(indices)==dm['records'] and np.array_equal(indices,expected));ck(tag+'/same_problem_KKT_canonical_witness_actual_emission_receipts',not errors,errors[:30]);ck(tag+'/no_false_prediction_zero_fill',not np.any(a['prediction_available']) and np.isnan(a['predicted_normal']).all())
                row.update(status='audited',run=str(run),decisions=len(indices),counts=dict(counts),original_task_success=result['task_success'],original_full_motion=result['research_metrics']['full_motion_mission_success'],original_guard=result['research_metrics']['safety']['guard_pass'],prefix37=compare,full_A_bridge=bridge,input_sha256={n:h(run/n) for n in ('metadata.json','result.json','trace.npz','release_decisions.jsonl.gz')})
            except Exception as exc:row.update(status='reader_error',error=repr(exc));issues.append(dict(stage=tag,error=repr(exc),traceback=traceback.format_exc()))
    ck('all9_assignments_assessed_no_selection',len(rows)==9 and all(r['status']=='audited' for r in rows));ck('192_still_unchanged',all(sha(sourcepath(k))==v for k,v in source.items()));ck('no_forbidden_pipeline',not forbidden)
    result=dict(schema='v24_small_execution_cross_v1',passed=not issues,planned=9,all_assigned_checked=len(rows)==9 and all(r['status']=='audited' for r in rows),rows=rows,checks=checks,issues=issues,production_source_sha256=source,numerical_identity=num,environment=plan['environment'],plan_sha256=sha(PLAN),reader_sha256=sha(__file__),helper_sha256=HELPERS,input_sha256=inputs,physics_steps=0,optimizer_calls=0,static_model_compiles=static_compiles,plant_or_data_constructions=0,forbidden_calls=forbidden,scope='Integration-author cross execution review; full nine assignment/source/prefix/port and saved-current finite KKT. No new task-math independence or performance/safety certificate.')
    out.write_text(json.dumps(plain(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(path=str(out),sha256=sha(out),passed=result['passed'],issues=issues[:3]),ensure_ascii=False));return 0 if result['passed'] else 1

def selftest():
    stages=[dict(name=v+str(i),variant=v,context_key=list(c),assignment=dict(release_variant=v,scenarios=[c[0]],motion_mode=c[1],seeds=[c[2]])) for v in VARIANTS for i,c in enumerate(CONTEXTS)];p=dict(stages=stages);good=dict(execution_complete=True,sources_unchanged=True,numerical_unchanged=True,runtime_failures=[],planned=1,trials=1);m={s['name']:dict(good) for s in stages};checks=[barrier(p,m)['ready']]
    for key,val in [('execution_complete',False),('numerical_unchanged',False),('runtime_failures',[dict(retained=True)])]:
        changed={k:dict(v) for k,v in m.items()};changed[stages[-1]['name']][key]=val;checks.append(not barrier(p,changed)['ready'])
    missing=dict(m);missing.pop(stages[-1]['name']);checks.append(not barrier(p,missing)['ready'])
    try:matrix(None,0,21);checks.append(False)
    except ValueError:checks.append(True)
    checks.append(matrix([],0,21).shape==(0,21))
    v=module('v23_reader_pure_KKT',ROOT/'review/v21_original_matrix_kkt_verifier_v1.py');problem=dict(A=[[1.,0.]],b=[.25],G=[[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],h=[1.,0.,1.,0.],E=[],f=[],previous_frozen=[],atol=1e-9,rtol=1e-8);dual=dict(inequality_unit=[0.]*4,equality_raw=[]);checks.append(v.verify_stage(problem,[.25,.5],duals=dual)['kkt_passed'] and not v.verify_stage(problem,[0.,.5],duals=dual)['kkt_passed'])
    # All fixtures are synthetic JSON-compatible data. No V24 result opened.
    def face(branch):return dict(candidate=branch+'_tight_face',generation=dict(candidate_branch=branch,face_threshold_attempts=1))
    two=dict(name='synthetic',attempts=[dict(candidate='SLSQP'),face('SLSQP_branch'),dict(candidate='FW_fallback_gradient_LP_exact_segment'),face('FW_branch')])
    e,c=layer_route_checks(json.loads(json.dumps([two])));checks.append(not e and c['task:two_face_layers']==1)
    bad=dict(two,attempts=[*two['attempts'],face('FW_branch')]);checks.append(bool(layer_route_checks([bad])[0]))
    bad=dict(two,attempts=[dict(candidate='SLSQP'),face('SLSQP_branch'),face('SLSQP_branch')]);checks.append(bool(layer_route_checks([bad])[0]))
    bad=dict(name='synthetic',attempts=[dict(candidate='FW_fallback_gradient_LP_exact_segment')]);checks.append(bool(layer_route_checks([bad])[0]))
    boxed=dict(two,attempts=[dict(candidate='SLSQP'),face('SLSQP_branch'),dict(candidate='SLSQP_branch_explicit_original_coordinate_box_candidate',generation=dict(candidate_branch='SLSQP_branch'))]);checks.append(not layer_route_checks([boxed])[0])
    checks.append(not layer_route_checks([dict(name='honest_failed_stage',attempts=[dict(candidate='anchor_check_error',error='synthetic')])])[0])
    cert=dict(x=[.25,.5],matrix=[[1.,0.]],target=[.25],attained=[.25],residual=[0.],squared_loss=0.,previous_frozen=[],hard_domain=dict(lower=[0.,0.],upper=[1.,1.],normalized_inequality_matrix=problem['G'],normalized_inequality_upper=problem['h'],fixed_equality_matrix=[],fixed_equality_target=[]),KKT=dict(full_equality_matrix=[],full_equality_target=[],full_normalized_inequality_dual=[0.]*4,equality_raw_dual=[]))
    cert=json.loads(json.dumps(cert));checks.append(witness_certificate_checks(cert,2,1e-9,1e-8,v))
    bad=dict(cert,x=[0.,.5]);checks.append(not witness_certificate_checks(bad,2,1e-9,1e-8,v))
    zero=[0.]*21
    sparse=dict(formulation='backend',geometry_proposal_accepted=False,contact_tasks=dict(available=True,pressure_P_N=[.3]),fallback='legal_current_q_recovery_anchor_without_exit_certificate',feasible_recovery_anchor21=zero,
        hierarchy=dict(diagnostics=dict(reason='synthetic_canonical_L1_unavailable',canonical_witness_required=True,canonical_witness_available=False,canonical_witness_certified=False,normal_witness_budget_defined=False,actual_increment_available=False,actual_submitted_certificate_count=0,normal_witness_is_submitted_prefix=False,canonical_witness_certificate_submitted=False,witness_completed_prefix=0,normal_witness_certified_prefix=[],normal_witness_levels=[dict(name='normal_inward_correction_slack',index=0,attempts=[dict(candidate='anchor_check_error',error='synthetic')])],witness_failed_layer=dict(index=0,name='normal_inward_correction_slack',status='UNAVAILABLE'),motion_task=dict(available=False)),certified_prefix=[],x=zero,completed_prefix=0,fully_converged=False,failed_layer=None,levels=[]))
    ee,cc=finite_certificates(json.loads(json.dumps(sparse)),{},np.zeros(21),np.ones(21),v,np.zeros(21));checks.append(not ee and cc['no_certified_task']==1 and cc['normal_witness_unavailable']==1)
    checks.extend(canonical_selftest(v))
    return dict(schema='v24_small_execution_reader_software_v1',passed=all(checks),checks=len(checks),physics_steps=0,optimizer_calls=0,new_results_read=False,scope='Synthetic all-nine terminal barriers, complete canonical pair KKT, actual-vs-witness semantics/global indices and frozen once-per-branch face bounds. No new rollout/result/optimizer.',reader_sha256=sha(__file__))
def canonical_selftest(verifier):
    from copy import deepcopy
    n=22;cost=np.zeros((1,n));cost[0,-1]=1.;selector=np.zeros((21,n));selector[:,:21]=np.eye(21)
    G=np.vstack((np.eye(n),-np.eye(n)));gh=np.r_[np.ones(n),np.ones(21),0.]
    dom=dict(lower=[-1.]*21+[0.],upper=[1.]*n,normalized_inequality_matrix=G.tolist(),normalized_inequality_upper=gh.tolist(),fixed_equality_matrix=[],fixed_equality_target=[])
    def cert(i):
        A=cost if i==0 else selector;prior=[] if i==0 else [dict(name='normal_inward_correction_slack',matrix=cost.tolist(),attained=[0.])]
        E=[] if i==0 else cost.tolist();f=[] if i==0 else [0.]
        return dict(name='normal_inward_correction_slack' if i==0 else CANONICAL_LEVEL,index=i,witness_index=i,solver_local_index=i,task_stage='canonical_normal_witness',certificate_role='NORMAL_OPTIMUM_WITNESS' if i==0 else 'CANONICAL_SELECTOR_WITNESS_ONLY_NEVER_SUBMITTED',x=[0.]*n,matrix=A.tolist(),target=[0.]*len(A),attained=[0.]*len(A),residual=[0.]*len(A),squared_loss=0.,previous_frozen=prior,hard_domain=deepcopy(dom),KKT=dict(certified=True,coordinate_tolerance=1e-9,full_equality_matrix=E,full_equality_target=f,full_normalized_inequality_dual=[0.]*len(G),equality_raw_dual=f))
    def fixture(formulation):
        pair=[cert(0),cert(1)];actual=deepcopy(pair[:1]);actual[0].update(task_stage='normal_slack',certificate_role='ORIGINAL_NORMAL_PREFIX_ONLY')
        if formulation=='joint':actual[0].update(name='joint_normal_and_observed_projection')
        levels=[dict(name=c['name'],index=i,attempts=[dict(candidate='SLSQP')]) for i,c in enumerate(pair)]
        d=dict(numerical_budget=FROZEN_SOLVE_BUDGET,normal_witness_levels=levels,normal_witness_certified_prefix=pair,witness_completed_prefix=2,canonical_witness_required=True,canonical_witness_available=True,canonical_witness_certified=True,witness_failed_layer=None,normal_witness_is_submitted_prefix=False,canonical_witness_certificate_submitted=False,normal_witness_final_extended_anchor=[0.]*n,certified_L1_extended_anchor=[0.]*n,normal_witness_optimal_eta_P=[0.],normal_first_certificate_is_submitted_prefix=formulation=='backend',normal_optimal_slack_frozen_before_motion=formulation=='backend')
        h=dict(diagnostics=d,certified_prefix=actual,levels=[dict(name=actual[0]['name'],index=0,solver_local_index=0,attempts=[dict(candidate='SLSQP')])],completed_prefix=1,fully_converged=True,failed_layer=None)
        return dict(formulation=formulation,contact_tasks=dict(pressure_P_N=[.3]),hierarchy=h)
    checks=[]
    for formulation in ('backend','joint'):
        q=fixture(formulation);e,c=canonical_checks(json.loads(json.dumps(q)),1,verifier);checks.append(not e and c['canonical_witness_certificates_checked']==2)
        bad=deepcopy(q);bad['hierarchy']['certified_prefix'][0]['index']=1;checks.append(bool(canonical_checks(bad,1,verifier)[0]))
        bad=deepcopy(q);bad['hierarchy']['levels'][0]['solver_local_index']=1;checks.append(bool(canonical_checks(bad,1,verifier)[0]))
        bad=deepcopy(q);bad['hierarchy']['diagnostics']['normal_witness_certified_prefix'][1]['matrix'][0][0]=0.;checks.append(bool(canonical_checks(bad,1,verifier)[0]))
        bad=deepcopy(q);bad['hierarchy']['certified_prefix'][0]['name']=CANONICAL_LEVEL;checks.append(bool(canonical_checks(bad,1,verifier)[0]))
        bad=deepcopy(q);bad['hierarchy']['diagnostics']['normal_witness_is_submitted_prefix']=True;checks.append(bool(canonical_checks(bad,1,verifier)[0]))
    failure=fixture('backend');h=failure['hierarchy'];d=h['diagnostics'];d.update(canonical_witness_available=False,canonical_witness_certified=False,normal_witness_budget_defined=False,actual_increment_available=False,actual_submitted_certificate_count=0,witness_completed_prefix=1,witness_failed_layer=dict(index=1,name=CANONICAL_LEVEL,status='UNAVAILABLE'),motion_task=dict(available=False));d['normal_witness_certified_prefix']=d['normal_witness_certified_prefix'][:1];d['normal_witness_levels']=d['normal_witness_levels'][:1];d.pop('numerical_budget');h.update(certified_prefix=[],levels=[],completed_prefix=0,fully_converged=False,failed_layer=None)
    e,c=canonical_checks(json.loads(json.dumps(failure)),1,verifier);checks.append(not e and c['canonical_witness_failed_after_normal_only']==1)
    bad=deepcopy(failure);bad['hierarchy']['failed_layer']=dict(index=1);checks.append(bool(canonical_checks(bad,1,verifier)[0]))
    bad=deepcopy(failure);bad['hierarchy']['diagnostics']['normal_witness_budget_defined']=True;checks.append(bool(canonical_checks(bad,1,verifier)[0]))
    empty=fixture('joint');empty['hierarchy']['diagnostics'].update(normal_witness_levels=[],normal_witness_certified_prefix=[],canonical_witness_required=False,canonical_witness_available=False,canonical_witness_certified=False,canonical_witness_not_required_knownP0=True,normal_first_certificate_is_submitted_prefix=False)
    checks.append(not canonical_checks(empty,0,verifier)[0])
    bad=deepcopy(empty);bad['hierarchy']['diagnostics']['canonical_witness_available']=True;checks.append(bool(canonical_checks(bad,0,verifier)[0]))
    return checks

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--execute',action='store_true');ap.add_argument('--out',required=True);a=ap.parse_args();out=ROOT/a.out
    if out.exists():raise FileExistsError(out)
    if a.execute:raise SystemExit(execute(read(PLAN),out))
    if a.self_test:r=selftest()
    elif a.prepare:
        r=dict(schema='v24_small_execution_readiness_v1',plan_present=PLAN.exists(),outcomes_read=False,physics_steps=0,reader_sha256=sha(__file__))
        if PLAN.exists():
            p=read(PLAN);m={s['name']:read(ROOT/s['output']/'execution_complete.json') for s in p['stages'] if (ROOT/s['output']/'execution_complete.json').exists()};r['barrier']=barrier(p,m);r['plan_sha256']=sha(PLAN)
    else:raise ValueError('explicit mode required')
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(path=str(out),sha256=sha(out),passed=r.get('passed'),outcomes_read=r.get('outcomes_read')),ensure_ascii=False))

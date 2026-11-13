"""Create the terminal-only V24 execution reader from frozen V23 audit glue."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
old = ROOT / 'review/read_v23_small_execution_v1.py'
new = ROOT / 'review/read_v24_small_execution_v1.py'
assert not new.exists(), 'preserve all prior reader attempts'
assert hashlib.sha256(old.read_bytes()).hexdigest() == '8c261bf571e4ff49d0490abd3d93f144234fcabda6767d2c3db13d19d09c4129'
text = old.read_text(encoding='utf-8')
text = text.replace('Terminal-only V23', 'Terminal-only V24').replace('v23_small', 'v24_small')
text = text.replace("VARIANTS=('v21','backend','joint')", "VARIANTS=('v23_joint','canonical_backend','canonical_joint')\nFORMULATIONS={'v23_joint':'joint','canonical_backend':'backend','canonical_joint':'joint'}\nCANONICAL_LEVEL='canonical_minimum_normalized_reference_witness'\nFROZEN_SOLVE_BUDGET=dict(max_iterations=400,atol=1e-9,rtol=1e-8)")
text = text.replace("HELPERS={", "HELPERS={\n 'review/read_v23_small_execution_v1.py':'8c261bf571e4ff49d0490abd3d93f144234fcabda6767d2c3db13d19d09c4129',")
text = text.replace("def finite_certificates(q,frame,actual,rates,verifier,actual_engine21):", '''def canonical_checks(q,P,verifier):
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

def finite_certificates(q,frame,actual,rates,verifier,actual_engine21):''')

start=text.index("    if q['formulation']=='joint':\n        es,cs=layer_route_checks(d.get('normal_witness_levels'")
end=text.index("    if h.get('failed_layer'):",start)
text=text[:start]+"    es,cs=canonical_checks(q,P,verifier);errors.extend(es);count.update(cs)\n"+text[end:]
text=text.replace("        count['witness_only_unsubmitted']+=q['formulation']=='joint' and witness", "        count['witness_only_unsubmitted']+=q['formulation']=='joint' and d.get('canonical_witness_available') is True\n        count['witness_incomplete_unsubmitted']+=witness and d.get('canonical_witness_available') is not True")
text=text.replace("d['normal_witness_is_submitted_prefix'] is True and d['normal_optimal_slack_frozen_before_motion'] is True", "d['normal_witness_is_submitted_prefix'] is False and d['normal_first_certificate_is_submitted_prefix'] is True and d['normal_optimal_slack_frozen_before_motion'] is True")
text=text.replace("    helper=module('v23_old_exec'", "    frozen_v23=module('v24_frozen_V23_exec',ROOT/'review/read_v23_small_execution_v1.py')\n    helper=module('v24_old_exec'")
text=text.replace("len(source)==186", "len(source)==192").replace('all186_current_sources','all192_current_sources').replace('all186_snapshots','all192_snapshots').replace('186_still_unchanged','192_still_unchanged')
needle="    parent=read(ROOT/'review/v22_small_plan_v1.json')['source_sha256'];ck('parent179_frozen_exact',len(parent)==179 and parent==plan['parent179_source_sha256'] and all(source.get(k)==v for k,v in parent.items()))"
replace="    parent186=read(ROOT/'review/v23_small_plan_v1.json')['source_sha256'];ck('parent186_frozen_exact',len(parent186)==186 and parent186==plan['reference_sources']['v23'] and all(source.get(k)==v for k,v in parent186.items()))\n    parent=read(ROOT/'review/v22_small_plan_v1.json')['source_sha256'];ck('parent179_frozen_exact',len(parent)==179 and parent==plan['reference_sources']['v22'] and all(source.get(k)==v for k,v in parent.items()))"
assert needle in text
text=text.replace(needle,replace)
text=text.replace("len(frozen)==170 and all(source", "len(frozen)==170 and frozen==plan['reference_sources']['v21'] and all(source")
text=text.replace("plan['original_v21_bridges']", "plan['original_v23_joint_bridges']").replace('originalV21_reference_pins','originalV23joint_reference_pins')
text=text.replace("variant!='v21'", "variant!='v23_joint'").replace("variant=='v21'", "variant=='v23_joint'")
text=text.replace("if variant!='v23_joint' and q.get('formulation')!=variant", "if q.get('formulation')!=FORMULATIONS[variant]")
needle="oldprefix.audit_prefix(q,'v21',frame,ex,rates) if variant=='v23_joint' else finite_certificates(q,frame,ex,rates,verifier,a['ctrl'][i,actor.indices21])"
assert needle in text
text=text.replace(needle,"frozen_v23.finite_certificates(q,frame,ex,rates,verifier,a['ctrl'][i,actor.indices21]) if variant=='v23_joint' else finite_certificates(q,frame,ex,rates,verifier,a['ctrl'][i,actor.indices21])")
text=text.replace("('v21_stable_handoff_exit' if variant=='v23_joint' else 'v23_task_execution_exit')", "('v23_task_execution_exit' if variant=='v23_joint' else 'v24_canonical_task_fusion_exit')")
text=text.replace("same_problem_KKT_eta_actual_emission_receipts", "same_problem_KKT_canonical_witness_actual_emission_receipts")
text=text.replace("# All fixtures are synthetic JSON-compatible data. No V23 result opened.", "# All fixtures are synthetic JSON-compatible data. No V24 result opened.")
start=text.index("    sparse=dict(")
end=text.index("    ee,cc=finite_certificates",start)
text=text[:start]+'''    sparse=dict(formulation='backend',geometry_proposal_accepted=False,contact_tasks=dict(available=True,pressure_P_N=[.3]),fallback='legal_current_q_recovery_anchor_without_exit_certificate',feasible_recovery_anchor21=zero,
        hierarchy=dict(diagnostics=dict(reason='synthetic_canonical_L1_unavailable',canonical_witness_required=True,canonical_witness_available=False,canonical_witness_certified=False,normal_witness_budget_defined=False,actual_increment_available=False,actual_submitted_certificate_count=0,normal_witness_is_submitted_prefix=False,canonical_witness_certificate_submitted=False,witness_completed_prefix=0,normal_witness_certified_prefix=[],normal_witness_levels=[dict(name='normal_inward_correction_slack',index=0,attempts=[dict(candidate='anchor_check_error',error='synthetic')])],witness_failed_layer=dict(index=0,name='normal_inward_correction_slack',status='UNAVAILABLE'),motion_task=dict(available=False)),certified_prefix=[],x=zero,completed_prefix=0,fully_converged=False,failed_layer=None,levels=[]))
'''+text[end:]
text=text.replace("    return dict(schema='v24_small_execution_reader_software_v1'", "    checks.extend(canonical_selftest(v))\n    return dict(schema='v24_small_execution_reader_software_v1'")
text=text.replace("Synthetic all-nine terminal barriers, original JSON witness KKT and V23 once-per-branch/two-per-layer face bounds only.", "Synthetic all-nine terminal barriers, complete canonical pair KKT, actual-vs-witness semantics/global indices and frozen once-per-branch face bounds. No new rollout/result/optimizer.")
marker="if __name__=='__main__':"
assert text.count(marker)==1
tests='''def canonical_selftest(verifier):
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

'''
text=text.replace(marker,tests+marker)
new.write_text(text,encoding='utf-8')
print(new)

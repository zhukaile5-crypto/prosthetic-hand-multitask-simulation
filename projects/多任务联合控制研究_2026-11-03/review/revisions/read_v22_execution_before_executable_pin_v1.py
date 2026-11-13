"""Terminal-only V22 nine-run saved-execution cross review. No optimizer.

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
PLAN=ROOT/'review/v22_small_plan_v1.json'
CONTEXTS=(('nominal','transport',619741),('fragile_conflict','hold',619741),('friction_drop','transport',619742))
VARIANTS=('v21','backend','joint')
HELPERS={
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
    if q['formulation']=='joint':
        ck('joint_no_submitted_old_L1_optimum',d['normal_witness_is_submitted_prefix'] is False and d['normal_optimal_slack_frozen_before_motion'] is False and all(c['name']!='normal_inward_correction_slack' for c in certs))
        count['witness_only_unsubmitted']+=bool(d.get('normal_witness_certified_prefix')) and not bool(certs)
    elif P:ck('backend_keeps_original_eta_optimum',d['normal_witness_is_submitted_prefix'] is True and d['normal_optimal_slack_frozen_before_motion'] is True)
    if not certs:
        count['no_certified_task']+=1;ck('empty_cert_not_accepted',not accepted)
        return errors,count
    current=np.asarray(q['current_reference21']);scale=rates*.005;emitted=raw.copy();emitted[:21]=(np.asarray(q['proposed_reference21'])-current)/scale;actualx=raw.copy();actualx[:21]=(actual-current)/scale
    coord=np.zeros(n);coord[:21]=128*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(current),abs(q['proposed_reference21'])))/scale
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
    helper=module('v22_old_exec',ROOT/'review/read_v20_small_execution_v1.py');oldprefix=module('v22_old_prefix',ROOT/'review/read_v21_small_execution_v1.py');prefix=module('v22_prefix37',ROOT/'review/v16_small_prefix_storage_independent_v1.py');timing=module('v22_timing_v3',ROOT/'review/v18_passive_trace_comparison_v3.py');verifier=module('v22_saved_KKT',ROOT/'review/v21_original_matrix_kkt_verifier_v1.py')
    checks=[];issues=[];rows=[];inputs={};source=plan['source_sha256'];num=plan['numerical_identity'];forbidden=[];static_compiles=0
    def ck(name,good,detail=None):
        checks.append(dict(name=name,passed=bool(good),detail=detail))
        if not good:issues.append(dict(name=name,detail=detail))
    def h(p):
        key=str(Path(p).resolve())
        if key not in inputs:inputs[key]=sha(key)
        return inputs[key]
    def sourcepath(k):return FOUNDATION/k[11:] if k.startswith('foundation/') else ROOT/k
    ck('all179_current_sources',len(source)==179 and all(h(sourcepath(k))==v for k,v in source.items()))
    frozen=read(ROOT/'review/v21_complete16_plan_v1.json')['source_sha256'];ck('original170_and42_unchanged',len(frozen)==170 and all(source.get(k)==v for k,v in frozen.items()) and plan['environment']==num['legacy_pinned_environment'] and len(plan['environment']['native_binaries_sha256'])==42)
    deps=dict(plan['environment']['native_binaries_sha256']);deps.update(num['scipy_native_binaries_sha256']);deps.update(num['necessary_runtime_dll_sha256']);deps.update({v['path']:v['sha256'] for v in num['scipy_python_implementation_sha256'].values()});ck('all_declared_numeric_dependency_bytes',all(h(k)==v for k,v in deps.items()))
    import research_imports,mujoco,hand_plant
    from foundation_config import FoundationConfig
    from v13_a1.codec import ReferenceActor21
    from v14_coord.design import reference_rates
    def deny(*a,**kw):forbidden.append('forbidden_model_pipeline_or_writer');raise AssertionError(forbidden[-1])
    refs={tuple(r['context_key']):r for r in plan['original_v21_bridges']};models={}
    with ExitStack() as guards:
        for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_comPos','mj_collision','mj_contactForce'):
            if hasattr(mujoco,name):guards.enter_context(patch.object(mujoco,name,deny))
        guards.enter_context(patch.object(hand_plant.HandPlant,'__init__',deny));guards.enter_context(patch.object(ReferenceActor21,'apply',deny))
        for s in plan['stages']:
            tag=s['name'];variant=s['variant'];key=tuple(s['context_key']);folder=ROOT/s['output'];row=dict(stage=tag,variant=variant,context_key=list(key),status='unassessed');rows.append(row)
            try:
                mark=markers[tag];reg=read(ROOT/s['registration']);manifest=read(folder/'manifest.json');snap=read(folder/'source_snapshot_manifest.json');trials=read(folder/'trials.json');jobs=read(folder/'assignment_ledger.json')
                ck(tag+'/registered_assignment_and_terminal_hashes',h(ROOT/s['registration'])==s['registration_sha256'] and reg['assignment']==s['assignment'] and all(reg[k]==plan[k] for k in ('source_sha256','environment','numerical_identity')) and all(mark[field]==h(folder/file) for file,field in [('trials.json','trials_sha256'),('assignment_ledger.json','assignment_ledger_sha256'),('plan.json','plan_sha256'),('manifest.json','manifest_sha256')]))
                ck(tag+'/all179_snapshots_and_extra_numeric_identity',set(snap['files'])==set(source) and all(h(folder/snap['files'][k]['snapshot_path'])==v for k,v in source.items()) and manifest['source_snapshot_manifest_sha256']==h(folder/'source_snapshot_manifest.json') and manifest['research_source_sha256']==source and manifest['numerical_identity']==num and read(folder/'environment.json')['pinned_environment']==plan['environment'] and read(folder/'environment.json')['numerical_identity']==num)
                run=folder/'runs'/f'{key[0]}__ratio_effort__{key[2]}';meta=read(run/'metadata.json');result=read(run/'result.json');scene=meta['scenario']['name'] if isinstance(meta['scenario'],dict) else meta['scenario']
                ck(tag+'/one_assigned_actual_result_all_failures_retained',len(trials)==len(jobs)==1 and jobs[0]['status']=='evaluated' and Path(jobs[0]['job']['destination']).resolve()==run.resolve() and result==trials[0] and (scene,meta['motion_mode'],meta['seed'])==key and meta['release_variant']==variant and meta['research_source_sha256']==source and meta['numerical_identity']==num and result['trace_sha256']==meta['trace_sha256']==h(run/'trace.npz'))
                original=Path(refs[key]['run']);ck(tag+'/originalV21_reference_pins',all(h(original/n)==v for n,v in refs[key]['input_sha256'].items()));om=read(original/'metadata.json');ck(tag+'/same_physical_noise_material',all(meta[k]==om[k] for k in ('config','noise','limits','budget','motion_mode','seed')))
                with np.load(run/'trace.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
                with np.load(original/'trace.npz',allow_pickle=False) as z:
                    compare=prefix.compare(a,z,a['time'],z['time'],variant!='v21')
                    mask=a['time']<12.-1e-9 if variant!='v21' else np.ones(len(a['time']),bool)
                    public_equal=all(str(x).encode('utf-8')==str(y).encode('utf-8') for x,y in zip(a['observation_json'][mask],z['observation_json'][mask]))
                    bridge=timing.compare_arrays(a,z) if variant=='v21' else None
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
                        if variant!='v21' and q.get('formulation')!=variant:errors.append((k,'actual_formulation'))
                        L=np.maximum(np.asarray(frame['reference_lower21']),cur-rates*.005);U=np.minimum(np.asarray(frame['reference_upper21']),cur+rates*.005)
                        if not fbits(L,q['original_mechanical_rate_lower21']) or not fbits(U,q['original_mechanical_rate_upper21']):errors.append((k,'original_authority'))
                        ql=float(frame['robot_position25'][24]);vl=float(frame['robot_velocity25'][24]);interval,reach=helper.independent_lift_interval(g,ql,vl,L[18],U[18],q['phase'] in helper.EXIT_PHASES)
                        if not fbits(interval,q['governor_task_domain']['task_reference_interval_m']) or q['governor_task_domain']['budget_band_reachable_this_tick']!=reach:errors.append((k,'original_governor_TASK_domain'))
                        if receipt is None:errors.append((k,'missing_actual_receipt'));continue
                        debt=ql-float(ex[18]);budget=-g['downward_recovery_reference_budget_m']<=debt<=g['recovery_reference_budget_m'];changed=not fbits(q['proposed_reference21'],ex)
                        if not fbits(receipt['actual_reference21'],ex) or receipt['same_snapshot_reference_debt_m']!=debt or receipt['current_q_budget_respected']!=budget or receipt['actual_task_interval_respected']!=(interval[0]<=ex[18]<=interval[1]) or receipt['execution_changed']!=changed:errors.append((k,'actual_receipt_debt_domain'))
                        counts['execution_changed']+=changed;counts['stale_plan_replaced']+=receipt['stale_plan_replaced'];counts['phase_'+q['phase']]+=1
                        es,cs=oldprefix.audit_prefix(q,'v21',frame,ex,rates) if variant=='v21' else finite_certificates(q,frame,ex,rates,verifier,a['ctrl'][i,actor.indices21]);errors.extend((k,e) for e in es);counts.update(cs)
                expected=np.flatnonzero([active[i] and debug[i].get('architecture')==('v21_stable_handoff_exit' if variant=='v21' else 'v22_joint_task_fusion_exit') for i in range(len(time))]);ck(tag+'/all_actual_release_decisions_retained',len(indices)==dm['records'] and np.array_equal(indices,expected));ck(tag+'/same_problem_KKT_eta_actual_emission_receipts',not errors,errors[:30]);ck(tag+'/no_false_prediction_zero_fill',not np.any(a['prediction_available']) and np.isnan(a['predicted_normal']).all())
                row.update(status='audited',run=str(run),decisions=len(indices),counts=dict(counts),original_task_success=result['task_success'],original_full_motion=result['research_metrics']['full_motion_mission_success'],original_guard=result['research_metrics']['safety']['guard_pass'],prefix37=compare,full_A_bridge=bridge,input_sha256={n:h(run/n) for n in ('metadata.json','result.json','trace.npz','release_decisions.jsonl.gz')})
            except Exception as exc:row.update(status='reader_error',error=repr(exc));issues.append(dict(stage=tag,error=repr(exc),traceback=traceback.format_exc()))
    ck('all9_assignments_assessed_no_selection',len(rows)==9 and all(r['status']=='audited' for r in rows));ck('179_still_unchanged',all(sha(sourcepath(k))==v for k,v in source.items()));ck('no_forbidden_pipeline',not forbidden)
    result=dict(schema='v22_small_execution_cross_v1',passed=not issues,planned=9,all_assigned_checked=len(rows)==9 and all(r['status']=='audited' for r in rows),rows=rows,checks=checks,issues=issues,production_source_sha256=source,numerical_identity=num,environment=plan['environment'],plan_sha256=sha(PLAN),reader_sha256=sha(__file__),helper_sha256=HELPERS,input_sha256=inputs,physics_steps=0,optimizer_calls=0,static_model_compiles=static_compiles,plant_or_data_constructions=0,forbidden_calls=forbidden,scope='Integration-author cross execution review; full nine assignment/source/prefix/port and saved-current finite KKT. No new task-math independence or performance/safety certificate.')
    out.write_text(json.dumps(plain(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(path=str(out),sha256=sha(out),passed=result['passed'],issues=issues[:3]),ensure_ascii=False));return 0 if result['passed'] else 1

def selftest():
    stages=[dict(name=v+str(i),variant=v,context_key=list(c),assignment=dict(release_variant=v,scenarios=[c[0]],motion_mode=c[1],seeds=[c[2]])) for v in VARIANTS for i,c in enumerate(CONTEXTS)];p=dict(stages=stages);good=dict(execution_complete=True,sources_unchanged=True,numerical_unchanged=True,runtime_failures=[],planned=1,trials=1);m={s['name']:dict(good) for s in stages};checks=[barrier(p,m)['ready']]
    for key,val in [('execution_complete',False),('numerical_unchanged',False),('runtime_failures',[dict(retained=True)])]:
        changed={k:dict(v) for k,v in m.items()};changed[stages[-1]['name']][key]=val;checks.append(not barrier(p,changed)['ready'])
    missing=dict(m);missing.pop(stages[-1]['name']);checks.append(not barrier(p,missing)['ready'])
    try:matrix(None,0,21);checks.append(False)
    except ValueError:checks.append(True)
    checks.append(matrix([],0,21).shape==(0,21))
    v=module('v22_reader_pure_KKT',ROOT/'review/v21_original_matrix_kkt_verifier_v1.py');problem=dict(A=[[1.,0.]],b=[.25],G=[[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],h=[1.,0.,1.,0.],E=[],f=[],previous_frozen=[],atol=1e-9,rtol=1e-8);dual=dict(inequality_unit=[0.]*4,equality_raw=[]);checks.append(v.verify_stage(problem,[.25,.5],duals=dual)['kkt_passed'] and not v.verify_stage(problem,[0.,.5],duals=dual)['kkt_passed'])
    return dict(schema='v22_small_execution_reader_software_v1',passed=all(checks),checks=len(checks),physics_steps=0,new_results_read=False,reader_sha256=sha(__file__))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--execute',action='store_true');ap.add_argument('--out',required=True);a=ap.parse_args();out=ROOT/a.out
    if out.exists():raise FileExistsError(out)
    if a.execute:raise SystemExit(execute(read(PLAN),out))
    if a.self_test:r=selftest()
    elif a.prepare:
        r=dict(schema='v22_small_execution_readiness_v1',plan_present=PLAN.exists(),outcomes_read=False,physics_steps=0,reader_sha256=sha(__file__))
        if PLAN.exists():
            p=read(PLAN);m={s['name']:read(ROOT/s['output']/'execution_complete.json') for s in p['stages'] if (ROOT/s['output']/'execution_complete.json').exists()};r['barrier']=barrier(p,m);r['plan_sha256']=sha(PLAN)
    else:raise ValueError('explicit mode required')
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(path=str(out),sha256=sha(out),passed=r.get('passed'),outcomes_read=r.get('outcomes_read')),ensure_ascii=False))

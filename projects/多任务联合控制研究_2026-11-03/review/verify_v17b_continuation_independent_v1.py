"""Static + extracted pure receipt guards; never prepare/execute/subprocess."""
from pathlib import Path
from copy import deepcopy
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
SUBJECT=ROOT/'review/continue_v17b_prospective80_v1.py'
PIN='d2229449a1e3ec075ff6066cd3ec8219486f9f545ac73ea263634e6f24932cea'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    before=sha(SUBJECT)
    if before!=PIN:raise ValueError('Final candidate SHA changed; rereview required')
    tree=ast.parse(SUBJECT.read_text(encoding='utf-8'));defs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    master=json.loads((ROOT/'review/v17b_prospective80_plan_v1.json').read_text(encoding='utf-8'))
    masterpin='b08fe0d57b2c43a8abcf58e9ff869d1fdc01ae0f2ed5cfe1546028e7268a6a06'
    env=dict(MASTER_SHA=masterpin,Path=Path,ROOT=Path('FAKE_ROOT'),RESUME=Path('FAKE_RESUME'),STOP=Path('FAKE_STOP'),HELPER=Path('FAKE_HELPER'),__file__='FAKE_SCRIPT')
    exec(compile(ast.Module(body=[defs['validate_resume_plan'],defs['resume_invariant']],type_ignores=[]),str(SUBJECT),'exec'),env)
    plan=dict(schema='v17b_same_cohort_resume_remaining64_v1',original_plan_sha256=masterpin,
        source_sha256=deepcopy(master['source_sha256']),environment=deepcopy(master['environment']),
        retained_trials=16,remaining_trials=64,total_planned=80,retained_stages=deepcopy(master['stages'][:2]),remaining_stages=deepcopy(master['stages'][2:]),
        old16_repeated=False,algorithm_model_gain_measurements_metrics_thresholds_assignments_changed=False,
        scheduler_sha256='SCRIPT_SHA',corrected_reader_sha256='HELPER_SHA',stopped_progress_sha256='STOP_SHA',
        original_registration_sha256={'registration.json':'REG_SHA'},incident_gate='incident.json',incident_gate_sha256='INCIDENT_SHA',reader_gate='reader.json',reader_gate_sha256='READER_SHA')
    hashes={'FAKE_RESUME':'RESUME_SHA','FAKE_STOP':'STOP_SHA','FAKE_SCRIPT':'SCRIPT_SHA','FAKE_HELPER':'HELPER_SHA',
        str(Path('FAKE_ROOT')/'registration.json'):'REG_SHA',str(Path('FAKE_ROOT')/'incident.json'):'INCIDENT_SHA',str(Path('FAKE_ROOT')/'reader.json'):'READER_SHA'}
    payloads={str(Path('FAKE_ROOT')/'incident.json'):{'passed':True},str(Path('FAKE_ROOT')/'reader.json'):{'passed':True}}
    env['sha']=lambda p:hashes[str(p)];env['read']=lambda p:payloads[str(p)];env['invariant']=lambda m:None
    checks=[]
    def ck(n,v,detail=None):checks.append(dict(name=n,passed=bool(v),detail=detail))
    def rejected(fn):
        try:fn()
        except (ValueError,RuntimeError):return True
        return False
    env['validate_resume_plan'](master,plan);env['resume_invariant'](master,plan,'RESUME_SHA')
    ck('valid_exact_original16_plus64_receipt_accepted',True)
    variants=[]
    p=deepcopy(plan);p['remaining_stages']=p['remaining_stages'][:-1];variants.append(('dropped_stage',p))
    p=deepcopy(plan);p['remaining_stages'][0]['seeds']=[999999];variants.append(('changed_seed',p))
    p=deepcopy(plan);p['remaining_stages'][0]['output']='other_output';variants.append(('changed_output',p))
    p=deepcopy(plan);p['remaining_stages'][0]['methods']=['ratio_effort'];variants.append(('changed_methods',p))
    p=deepcopy(plan);p['retained_stages']=list(reversed(p['retained_stages']));variants.append(('changed_retained_order',p))
    p=deepcopy(plan);p['source_sha256'][next(iter(p['source_sha256']))]='changed';variants.append(('changed_source',p))
    p=deepcopy(plan);p['environment']['executable_sha256']='changed';variants.append(('changed_environment',p))
    p=deepcopy(plan);p['remaining_trials']=63;variants.append(('changed_count',p))
    p=deepcopy(plan);p['old16_repeated']=True;variants.append(('old16_repeat',p))
    p=deepcopy(plan);p['algorithm_model_gain_measurements_metrics_thresholds_assignments_changed']=True;variants.append(('algorithm_changed',p))
    for name,bad in variants:ck('reject_'+name,rejected(lambda bad=bad:env['validate_resume_plan'](master,bad)))
    for key in list(hashes):
        original=hashes[key];hashes[key]='MUTATED'
        ck('reject_changed_pin_'+key,rejected(lambda:env['resume_invariant'](master,plan,'RESUME_SHA')))
        hashes[key]=original
    for key in payloads:
        payloads[key]={'passed':False};ck('reject_failed_evidence_'+key,rejected(lambda:env['resume_invariant'](master,plan,'RESUME_SHA')));payloads[key]={'passed':True}
    calls=[n for n in ast.walk(defs['execute']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='resume_invariant']
    dispatch=[n for n in ast.walk(defs['execute']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess' and n.func.attr=='run']
    final_loop=next(n for n in defs['execute'].body if isinstance(n,ast.Try));loop=next(n for n in final_loop.body if isinstance(n,ast.For))
    loopguard=[n for n in loop.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='resume_invariant']
    afterloop=[n for n in final_loop.body if getattr(n,'lineno',0)>loop.end_lineno and isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='resume_invariant']
    ck('guards_before_dispatch_before_after_eachstage_and_final',len(calls)==4 and len(dispatch)==1 and min(n.lineno for n in calls)<dispatch[0].lineno and len(loopguard)==2 and loopguard[0].lineno<dispatch[0].lineno<loopguard[1].lineno and len(afterloop)==1)
    targets=[]
    for name in ('prepare','execute'):
        for n in ast.walk(defs[name]):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='write':targets.append(n.args[0].id if isinstance(n.args[0],ast.Name) else ast.dump(n.args[0]))
    ck('write_targets_only_new_receipts_not_oldmaster_stop_or_data',set(targets)=={'RESUME','PROGRESS','CHAIN'},targets)
    ck('original_remaining_assignments_count64',sum(s['planned'] for s in master['stages'][2:])==64 and [s['name'] for s in master['stages'][:2]]==['holding_A_hold','full_V17_hold'])
    ck('candidate_source_unchanged_during_review',sha(SUBJECT)==before)
    report=dict(schema='v17b_continuation_independent_v1',passed=all(c['passed'] for c in checks),checks=checks,check_count=len(checks),source_path=str(SUBJECT.relative_to(ROOT)),source_sha256=before,
        original_plan_sha256=masterpin,checker_sha256=sha(__file__),initial_review='review/v17b_continue_static_review_initial_v1.json',
        scope=dict(extracted_actual_pure_functions=['validate_resume_plan','resume_invariant'],mock_hash_and_read_only=True,
            prepare_calls=0,execute_calls=0,subprocess_calls=0,physics_steps=0,registration_calls=0,new_performance_reads=0,
            real_runtime_source_environment_rehash='root preflight and actual invariant retain this responsibility; mocked here to isolate receipt guards'))
    out=ROOT/'review/v17b_continuation_independent_v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],out=str(out),sha256=sha(out))))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

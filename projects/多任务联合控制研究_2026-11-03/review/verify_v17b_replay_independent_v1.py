"""Synthetic/AST viewer review; no server, model, renderer or actual80 reads."""
from pathlib import Path
from copy import deepcopy
from tempfile import TemporaryDirectory
from unittest.mock import patch
import argparse,ast,hashlib,importlib.util,json,sys
import re,shutil,subprocess
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    from v17b_replay_v1 import reader as r,server as s
    checks=[];forbidden=[]
    def ck(n,v,d=None):checks.append(dict(name=n,passed=bool(v),detail=d))
    def no_npz(*a,**k):forbidden.append('NPZ');raise AssertionError('New cohort NPZ access forbidden')
    plan=r.read(r.DEFAULT_PLAN);sources={str(q.relative_to(ROOT)):sha(q) for q in (ROOT/'v17b_replay_v1').rglob('*') if q.suffix in ('.py','.html')}
    ck('three_ratio_methods_remain_distinct_roles',len({r.role_for(a,'ratio_effort') for a in ('full_V17','holding_A','baselines')})==3)
    for finalkind in ('missing','source_bad'):
        with TemporaryDirectory(prefix='independent_replay_pending_') as temp:
            root=Path(temp);access=[]
            for st in plan['stages'][:-1]:
                q=root/st['output'];q.mkdir(parents=True);(q/'execution_complete.json').write_text('{}')
            if finalkind=='source_bad':
                q=root/plan['stages'][-1]['output'];q.mkdir(parents=True);(q/'execution_complete.json').write_text('{}')
            def read_fake(path):
                path=Path(path);access.append(path.name)
                if path.resolve()==r.DEFAULT_PLAN.resolve():return deepcopy(plan)
                if path.name!='execution_complete.json':raise AssertionError('Incomplete viewer accessed '+str(path))
                st=next(x for x in plan['stages'] if path.parent==root/x['output'])
                return dict(execution_complete=True,sources_unchanged=not(finalkind=='source_bad' and st==plan['stages'][-1]),runtime_failures=[],trials=st['planned'],planned=st['planned'])
            with patch.object(r,'ROOT',root),patch.object(r,'read',read_fake),patch.object(np,'load',no_npz):
                store=r.ResultStore();status=store.catalog()['status']
            ck('five_complete_one_'+finalkind+'_reads_no_results_orNPZ',status==('pending' if finalkind=='missing' else 'blocked') and not store.records and set(access)<={'v17b_prospective80_plan_v1.json','execution_complete.json'},access)
    metric=r.original_metrics(dict(task_success=False,research_metrics=dict(full_motion_mission_success=False,safety=dict(guard_pass=False),static_quality=dict(eligible=False,mean_actual_finger_force_N=0.,raw_200Hz=dict(finger_TV_rate_N_s=0.)))))
    ck('original_false_and_ineligible_NULL_not_zero',metric['full_task'] is False and metric['force_guard'] is False and metric['static']['mean_N'] is None and metric['static']['tv_N_per_s'] is None)
    # All audit/receipt data here are invented fixtures. No new cohort data.
    audit_path=Path('FAKE_AUDIT');chain_path=s.ROOT/'review/v17b_prospective80_completion_chain_v1.json'
    fake=dict(schema='v17b_prospective80_execution_independent_v1',passed=True,plan_sha256=r.PLAN_SHA,assigned_count=80,all_assigned_checked=True,
        source_sha256=plan['source_sha256'],environment=plan['environment'],
        retained16_resume64_chain=dict(completion_chain_sha256='CHAIN_SHA',retained_data_unchanged=True))
    chain=dict(complete=True,original_plan_sha256=r.PLAN_SHA,source_sha256=plan['source_sha256'],environment=plan['environment'],
        retained_trials=16,continued_trials=64,total_completed=80,stopped_progress_sha256='STOP_SHA',resume_plan_sha256='RESUME_SHA',resume_progress_sha256='PROGRESS_SHA')
    pins={str(s.DEFAULT_PLAN):r.PLAN_SHA,str(chain_path):'CHAIN_SHA',
        str(s.ROOT/'review/v17b_prospective80_progress_v1.json'):'STOP_SHA',str(s.ROOT/'review/v17b_prospective80_resume_plan_v1.json'):'RESUME_SHA',str(s.ROOT/'review/v17b_prospective80_resume_progress_v1.json'):'PROGRESS_SHA'}
    def read_fake(path):
        if Path(path)==audit_path:return fake
        if Path(path)==s.DEFAULT_PLAN:return plan
        if Path(path)==chain_path:return chain
        raise AssertionError('Unexpected actual audit read: '+str(path))
    def rejected(fn):
        try:fn()
        except (ValueError,KeyError,OSError):return True
        return False
    with patch.object(s,'read',read_fake),patch.object(s,'sha',lambda path:pins[str(path)]):
        ck('correct136_environment_and_receipts_audit_accepted',s.check_audit(audit_path)==fake)
        for field,wrong in [('source_sha256',{'wrong':'136'}),('environment',{'wrong':'42'})]:
            original=fake[field];fake[field]=wrong;ck('wrong_audit_'+field+'_rejected',rejected(lambda:s.check_audit(audit_path)));fake[field]=original
        original=fake['retained16_resume64_chain']['completion_chain_sha256'];fake['retained16_resume64_chain']['completion_chain_sha256']='WRONG'
        ck('wrong_current_chain_audit_rejected',rejected(lambda:s.check_audit(audit_path)));fake['retained16_resume64_chain']['completion_chain_sha256']=original
        for name in ('stopped_progress_sha256','resume_plan_sha256','resume_progress_sha256'):
            original=chain[name];chain[name]='WRONG';ck('changed_current_receipt_'+name+'_rejected',rejected(lambda:s.check_audit(audit_path)));chain[name]=original
    rows=[dict(run=str((Path('FAKE_ROWS')/str(i)).resolve()),status='audited',metadata_sha256='META',result_sha256='RESULT') for i in range(80)]
    store=SimpleNamespace(records={i:dict(folder=Path(row['run'])) for i,row in enumerate(rows)})
    with patch.object(s,'sha',lambda path:'META' if Path(path).name=='metadata.json' else 'RESULT'):
        s.bind_audit(store,dict(rows=rows));ck('audit_binds_exact80_record_directories',len(store.audit_fingerprints)==80)
        rows[0]['metadata_sha256']='WRONG';ck('changed_audited_metadata_rejected',rejected(lambda:s.bind_audit(store,dict(rows=rows))));rows[0]['metadata_sha256']='META'
        ck('missing_audited_record_rejected',rejected(lambda:s.bind_audit(store,dict(rows=rows[:-1]))))
    # Actual old schema, no old trace/model/controller is loaded.
    old_metadata=r.read(ROOT/'output/v17_transfer_contact_hold_v1/runs/fragile_conflict__ratio_effort__619732/metadata.json')
    r.check_metadata_context(old_metadata,'fragile_conflict','hold',619732)
    ck('actual_rich_scenario_metadata_context_accepted',True)
    legacy=deepcopy(old_metadata);legacy['scenario']='fragile_conflict';r.check_metadata_context(legacy,'fragile_conflict','hold',619732)
    ck('legacy_string_metadata_context_accepted',True)
    for name,context in [('scenario',('nominal','hold',619732)),('motion',('fragile_conflict','transport',619732)),('seed',('fragile_conflict','hold',619741))]:
        ck('wrong_actual_metadata_'+name+'_rejected',rejected(lambda context=context:r.check_metadata_context(old_metadata,*context)))
    html=(ROOT/'v17b_replay_v1/web/index.html').read_text(encoding='utf-8')
    expr=re.search(r'knownFloor===undefined\?d\.delivery\?\.floor_contact:knownFloor',html)
    node=shutil.which('node')
    if expr is None or node is None:ck('actual_floor_expression_true_false_null_missing',False,'expression or Node unavailable; not simulated as pass')
    else:
        code='const choose=(knownFloor,d)=>('+expr.group(0)+'); const rows=[[true,false,true],[false,true,false],[null,true,null],[undefined,true,true]]; if(rows.some(([a,b,w])=>!Object.is(choose(a,{delivery:{floor_contact:b}}),w))) process.exit(1); process.stdout.write("four_values_passed");'
        result=subprocess.run([node,'-e',code],capture_output=True,text=True,timeout=20)
        ck('actual_floor_expression_true_false_null_missing',result.returncode==0 and result.stdout=='four_values_passed',result.stderr)
    for name in ('reader.py','server.py','rendering.py'):
        q=ROOT/'v17b_replay_v1'/name;tree=ast.parse(q.read_text(encoding='utf-8'))
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        ck('no_step_forward_contactForce_'+name,not set(calls)&{'mj_step','mj_step1','mj_step2','mj_forward','mj_collision','mj_contactForce'})
    render_tree=ast.parse((ROOT/'v17b_replay_v1/rendering.py').read_text(encoding='utf-8'))
    ck('renderer_dependencies_lazy_not_module_top',not any(isinstance(n,ast.Import) and any(k.name in ('mujoco','PIL') for k in n.names) for n in render_tree.body))
    ck('no_imported_renderer_or_mujoco','v17b_replay_v1.rendering' not in sys.modules and 'mujoco' not in sys.modules)
    ck('forbiddenNPZ_calls_zero',not forbidden)
    after={str(q.relative_to(ROOT)):sha(q) for q in (ROOT/'v17b_replay_v1').rglob('*') if q.suffix in ('.py','.html')}
    ck('viewer_source_unchanged_during_gate',sources==after)
    report=dict(schema='v17b_replay_independent_v1',passed=all(c['passed'] for c in checks),checks=checks,check_count=len(checks),source_sha256=sources,checker_sha256=sha(__file__),
        plan_sha256=r.PLAN_SHA,scope=dict(physics_steps=0,server_started=False,models_created=0,renderer_started=False,browser_or_DOM=False,new80_results_read=0,
            synthetic_pending_markers_only=True,static_actual_renderer_call_audit=True,original_viewers_modified=False))
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],out=str(out),sha256=sha(out))))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

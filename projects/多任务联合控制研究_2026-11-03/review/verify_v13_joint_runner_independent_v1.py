"""Independent runner-only fault fixtures. No simulator import or integration."""
from pathlib import Path
import ast
import hashlib
import json
import multiprocessing
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'v13_joint/run_development.py'
OLD=ROOT/'review/revisions/joint_runner_1207_before_failure_preservation.py'
EXPECTED='e055d576690331e4bdc15c3b2af43677e5cdc25ed5f4366956fa9b850f0da1c5'
OUT=ROOT/'review/v13_joint_runner_independent_v1'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def extract(path):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ('run_one','output_path')]
    ns=dict(Path=Path,ROOT=ROOT,time=time,traceback=traceback,write=write)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
    return tree,ns

def main():
    OUT.mkdir(exist_ok=False)
    checks=[];cases=[]
    def check(name,passed,detail=None):
        checks.append(dict(name=name,passed=bool(passed),detail=detail))
    check('new_source_identity',digest(NEW)==EXPECTED,digest(NEW))
    check('old_source_identity_retained',digest(OLD)=='1207f4713170d9f705799176e138f87eac3716831bfbfcea8eda597c25281669')
    tree,new=extract(NEW);_,old=extract(OLD)
    check('guarded_main_only',any(isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'" for n in tree.body))
    check('windows_standard_spawn',sys.platform=='win32' and multiprocessing.get_start_method()=='spawn',multiprocessing.get_start_method())
    check('runner_import_has_no_top_level_call_to_main_or_worker',not any(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id in ('main','run_one') for n in tree.body))
    source=NEW.read_text(encoding='utf-8-sig')
    for name,text in {
        'registration_exact_source_environment_assignment':"registration['source_sha256']!=identity or registration['environment']!=pinned_environment() or registration['assignment']!=expected",
        'snapshot_before_dispatch':'snapshot_sources(destination,identity,args.methods)',
        'worker_identity_before_and_after':'Frozen V13Joint worker sources changed during execution',
        'all_assignment_ledger':'assignment_ledger.json',
        'no_confirmation_performance_claim':"performance_claim_allowed=False",
        'both_compatible_evaluators':"result['research_metrics']=evaluate_research(destination/'trace.npz')",
    }.items():check(name,text in source)
    runtime=(ROOT/'v13_joint/runtime.py').read_text(encoding='utf-8-sig')
    foundation=ROOT.parent/'五指策略接入_2026-11-03'
    session=(foundation/'session.py').read_text(encoding='utf-8-sig')
    check('runtime_full_duration_branch','if t>=DURATION-1e-9:' in runtime and 'self.running=False;self.completed=True' in runtime and 'DURATION=16.' in session)
    for value in ('output','../outside','output/../escaped'):
        try:new['output_path'](value);ok=False
        except (ValueError,FileExistsError):ok=True
        check('reject_output_'+value,ok)
    check('fresh_output_accepted',new['output_path']('output/__runner_pure_not_created__').is_relative_to(ROOT/'output'))
    identity={'fixture.py':'fixed'}

    def one(ns,label,mode):
        destination=OUT/label
        class Session:
            def __init__(self,**kw):
                if mode=='construction':raise ValueError('ORIGINAL_CONSTRUCTOR')
                self.running=True;self.completed=False;self.time=0.;self.trace=[{'fixture':1}];self.research_sources=identity
            def step_one_sensor_tick(self):
                if mode in ('step','step_save','old_step_save'):raise RuntimeError('ORIGINAL_STEP')
                self.running=False;self.completed=True;self.time=16.
            def save(self,path):
                if mode in ('step_save','old_step_save'):raise OSError('SECONDARY_SAVE')
                Path(path).mkdir(parents=True,exist_ok=False)
                (Path(path)/'trace.npz').write_bytes(b'PURE_FIXTURE_NOT_NPZ')
        def primary(path):
            if mode=='evaluation':raise ArithmeticError('ORIGINAL_EVALUATION')
            return dict(task_success=True,record_complete=True,scenario='nominal',method='fixture',seed=617901)
        calls=[]
        def identity_fn(methods):
            calls.append(list(methods))
            return identity if mode!='post_source' or len(calls)==1 else {'fixture.py':'CHANGED'}
        ns.update(JointSession=Session,evaluate_trace=primary,evaluate_research=lambda p:{'fixture':True},source_identity=identity_fn)
        job=dict(destination=str(destination),method='fixture',scenario='nominal',seed=617901,split='development',motion_mode='hold')
        try:returned=ns['run_one'](job,identity);exception=None
        except Exception as exc:returned=None;exception=dict(type=type(exc).__name__,error=str(exc))
        error=destination/'runtime_error.json'
        record=dict(label=label,exception=exception,returned=returned,runtime_error=json.loads(error.read_text(encoding='utf-8')) if error.exists() else None,trace_exists=(destination/'trace.npz').exists())
        cases.append(record);return record

    prior=one(old,'old_step_save','old_step_save')
    check('old_bug_reproduced_secondary_masks_original',prior['exception']==dict(type='OSError',error='SECONDARY_SAVE') and prior['runtime_error'] is None)
    r=one(new,'success','success')
    check('success_result_retained',r['exception'] is None and r['returned']['record_complete'] and (OUT/'success/result.json').is_file())
    for mode,expected_type,expected_error in (
        ('construction','ValueError','ORIGINAL_CONSTRUCTOR'),
        ('step','RuntimeError','ORIGINAL_STEP'),
        ('step_save','RuntimeError','ORIGINAL_STEP'),
        ('evaluation','ArithmeticError','ORIGINAL_EVALUATION'),
        ('post_source','RuntimeError','Frozen V13Joint worker sources changed during execution')):
        r=one(new,mode,mode);e=r['runtime_error']
        check(mode+'_original_raised',r['exception']==dict(type=expected_type,error=expected_error))
        check(mode+'_original_ledger',e is not None and e['type']==expected_type and e['error']==expected_error and expected_error in e['traceback'])
        if mode=='step_save':check('secondary_save_separate',e['partial_save_error']['type']=='OSError' and e['partial_save_error']['error']=='SECONDARY_SAVE' and 'SECONDARY_SAVE' in e['partial_save_error']['traceback'])
        if mode in ('step','evaluation','post_source'):check(mode+'_partial_or_complete_trace_retained',r['trace_exists'])
    result=dict(schema='v13_joint_runner_independent_pure_v1',passed=all(x['passed'] for x in checks),checks=len(checks),failed=[x for x in checks if not x['passed']],source_sha256={str(NEW):digest(NEW),str(OLD):digest(OLD),str(Path(__file__)):digest(__file__)},scope='AST+worker fault fixtures only; no simulator import, no plant construction, no physics, no spawned physical worker; normal 16s/evaluator linkage read only',cases=cases,details=checks)
    write(OUT/'audit.json',result)
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],evidence=str(OUT/'audit.json')),ensure_ascii=False))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

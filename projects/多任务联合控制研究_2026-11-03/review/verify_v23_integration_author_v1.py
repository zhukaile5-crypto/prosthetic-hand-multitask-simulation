"""V23 integration AUTHOR gate, not independent task-math certification.

No physical rollout. Static action-delta review, actual imported lifecycle
routing through mocked sole parent/observer constructors, CLI/source identity,
and optional genuine zero-step session construction. Initial native forward
is counted if --zero-step; no step/advance is permitted. New math is separately
owned and certified. Only fresh report paths are allowed.
"""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib,json,sys,traceback
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def function(path,name):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    return next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
class NormalizeAction(ast.NodeTransformer):
    def visit_Call(self,n):
        self.generic_visit(n)
        if isinstance(n.func,ast.Name) and n.func.id=='solve_contact_exit':n.keywords=[k for k in n.keywords if k.arg!='formulation']
        if isinstance(n.func,ast.Name) and n.func.id=='dict':
            if any(k.arg=='decision_id' for k in n.keywords):n.keywords=[k for k in n.keywords if k.arg not in ('release_variant','formulation')]
            if any(k.arg=='architecture' for k in n.keywords):
                n.keywords=[k for k in n.keywords if k.arg!='formulation']
                next(k for k in n.keywords if k.arg=='architecture').value=ast.Constant('v21_stable_handoff_exit')
        return n
    def visit_Assign(self,n):
        self.generic_visit(n)
        if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='role' and isinstance(n.value,ast.IfExp):n.value=ast.Constant('contact_exit_hierarchy_current_public_snapshot')
        return n

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zero-step',action='store_true');ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    assert not out.exists(),'fresh gate output required'
    checks=[]
    def ck(name,good,**more):checks.append(dict(name=name,passed=bool(good),**more))
    old=ROOT/'v21_stable_handoff/supervisor.py';new=ROOT/'v23_task_execution/supervisor.py'
    ck('mechanical_governor_snapshot_native_actor_receipt_action_AST_retained',dump(function(old,'action'))==dump(NormalizeAction().visit(copy.deepcopy(function(new,'action')))))
    for name in ('nominal','reconcile'):ck(name+'_AST_identical_to_frozenV21',dump(function(old,name))==dump(function(new,name)))
    text=new.read_text(encoding='utf-8');ck('actual_solve_call_passes_only_explicit_self_formulation',"formulation=self.formulation" in text and "from v21_stable_handoff.prefix_supervisor import contact_exit_lift_interval, score_emitted_reference" in text)
    mod=importlib.import_module('v23_task_execution.runtime');runner=importlib.import_module('v23_task_execution.run_development');design=importlib.import_module('v23_task_execution.design')
    inherited=mod.frozen_v22_files(['ratio_effort']);files=mod.source_files(['ratio_effort']);identity=mod.source_identity(['ratio_effort'])
    oldplan=json.loads((ROOT/'review/v22_small_plan_v1.json').read_text(encoding='utf-8'))
    ck('all179_frozen_sources_exact_and_all7_new_required',len(inherited)==179 and len(files)==186 and {k:identity[k] for k in inherited}==oldplan['source_sha256'] and set(files)-set(inherited)=={'v23_task_execution/'+n for n in design.SOURCE_NAMES})
    ck('existing_v23_init_and_finite_unchanged',identity['v23_task_execution/__init__.py']=='250d0014fd3ec50cf96c161bc6a649fb57f4a1e834b20cbe85f0152b5578466e' and identity['v23_task_execution/finite_solver.py']=='4b2ac6a4d00896729aded914bece9efa21421643c0cc9b2e598f7be46c099dbf')
    for variant in design.RELEASE_VARIANTS:
        calls=[];bound=[];base=[]
        def frozen_factory(*a,**kw):calls.append(('v21',kw));return SimpleNamespace(kind='v21')
        def new_factory(*a,**kw):calls.append(('new',kw));return SimpleNamespace(kind=kw['formulation'],formulation=kw['formulation'])
        entry=SimpleNamespace(started_s=12.)
        obj=mod.ReleaseSession.__new__(mod.ReleaseSession);obj.release_variant=variant;obj.v16_release_policy='coordinated';obj.v16_supervisor=None
        obj.plant=SimpleNamespace(joint_release_entry=entry,model=object(),open_pose=(np.zeros(5),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(4)),reference_authority21=lambda **k:(np.zeros(21),np.ones(21)),bind_supervisor=lambda value:bound.append(value))
        obj.release_adapter=SimpleNamespace(begin_release=lambda t:dict(prior=True),receiver_half_size_xy_m=np.array([.1,.1]),receiver_up_axis3=np.array([0.,0.,1.]))
        obj.controller=SimpleNamespace(limits=np.ones(5),budget=5.,g=SimpleNamespace(force_margin=.1))
        def parent_step(self,obs):base.append(obs);return 'parent_actual_actor_path'
        with patch.object(mod,'FrozenV21Supervisor',frozen_factory),patch.object(mod,'ReleaseSupervisor',new_factory),patch.object(mod,'TransferObserver',lambda **kw:SimpleNamespace(**kw)),patch.object(mod,'release_clearance_goal',lambda *a,**kw:SimpleNamespace(reference18=np.zeros(18))),patch.object(mod.FrozenSession,'_lifecycle_step',parent_step):
            a=obj._lifecycle_step(object());b=obj._lifecycle_step(object())
        ck('actual_lifecycle_'+variant+'_constructor_once_then_same_parent_actor',a==b=='parent_actual_actor_path' and len(calls)==len(bound)==1 and len(base)==2 and obj.release_driver is bound[0] and calls[0][0]==('v21' if variant=='v21' else 'new') and ('formulation' not in calls[0][1] if variant=='v21' else calls[0][1]['formulation']==variant))
    try:runner.output_path('output');bad=False
    except ValueError:bad=True
    ck('runner_strict_output_root_rejected',bad)
    try:runner.output_path('review/invalid');bad=False
    except ValueError:bad=True
    ck('runner_nonoutput_rejected',bad)
    parser=function(ROOT/'v23_task_execution/run_development.py','main');argument=next(n for n in ast.walk(parser) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--release-variant')
    ck('CLI_release_variant_explicit_required_three_arms',any(k.arg=='required' and isinstance(k.value,ast.Constant) and k.value.value is True for k in argument.keywords) and any(k.arg=='choices' and ast.literal_eval(k.value)==('v21','backend','joint') for k in argument.keywords))
    forbidden_constructor=[];written=[]
    with patch.object(runner,'SupportSession',lambda **k:forbidden_constructor.append(k)),patch.object(runner,'write',lambda p,v:written.append((str(p),v))),patch.object(Path,'mkdir',lambda *a,**k:None):
        try:runner.run_one(dict(destination=str(ROOT/'review/virtual_v22_bad_identity'),method='ratio_effort'),identity,dict(wrong='numeric-pin'));rejected=False
        except RuntimeError as exc:rejected='numerical dependency identity mismatch' in str(exc)
    ck('worker_rejects_numeric_pin_before_constructor_and_keeps_error',rejected and not forbidden_constructor and len(written)==1 and 'numerical dependency identity mismatch' in written[0][1]['error'])
    forwards=0;steps=0;time_records=[]
    if args.zero_step:
        import mujoco,hand_plant
        original_forward=mujoco.mj_forward
        def forward(*a,**k):
            nonlocal forwards
            forwards+=1;return original_forward(*a,**k)
        def forbidden(*a,**k):
            nonlocal steps
            steps+=1;raise RuntimeError('integration is forbidden in author gate')
        with patch.object(mujoco,'mj_forward',forward),patch.object(mujoco,'mj_step',forbidden),patch.object(mujoco,'mj_step1',forbidden),patch.object(mujoco,'mj_step2',forbidden),patch.object(hand_plant.HandPlant,'advance',forbidden):
            for variant in design.RELEASE_VARIANTS:
                session=mod.ReleaseSession(method='ratio_effort',scenario='nominal',seed=619741,split='robustness',motion_mode='transport',task_policy='integrated',receiver_policy='coordinated',release_variant=variant)
                meta=session.metadata();time_records.append(dict(variant=variant,time=session.time,tick=session.tick,samples=session.sensors.samples))
                ck('genuine_zero_step_'+variant+'_metadata_and_identity',session.time==0. and session.tick==0 and session.sensors.samples==0 and not session.trace and meta['release_variant']==variant and meta['research_source_sha256']==identity and meta['numerical_identity']==session.numerical_sources==mod.numerical_identity() and meta['v23_release_design']['declared_task_formulation']==(None if variant=='v21' else variant))
    final=mod.source_identity(['ratio_effort']);ck('executed_source_identity_unchanged_during_gate',final==identity)
    result=dict(schema='v23_integration_author_v1',passed=all(c['passed'] for c in checks),checks=checks,source_sha256=identity,production_source_sha256=identity,numerical_identity=mod.numerical_identity(),source_count=len(identity),zero_step_requested=args.zero_step,initialization_forward_calls=forwards,integration_entry_calls=steps,time_records=time_records,physics_steps=0,scope='Integration author software gate; mocked lifecycle constructor routing and optional actual zero-step only, not independent task mathematics or a rollout.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(path=str(out),sha256=sha(out),passed=result['passed'],checks=len(checks),source_count=len(identity),initialization_forward_calls=forwards,integration_entry_calls=steps),ensure_ascii=False));return 0 if result['passed'] else 1
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        out=ROOT/sys.argv[sys.argv.index('--out')+1]
        result=dict(schema='v23_integration_author_v1',passed=False,error=repr(exc),traceback=traceback.format_exc(),physics_steps=0,source_gate_complete=False)
        if not out.exists():out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(result['traceback']);raise SystemExit(1)

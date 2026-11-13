"""Minimal V16b delta: real zero-integration constructor and actual lifecycle entry.

Reuses only the exact pure fake-port helper of the previous independent gate.
No physical tick, solver replay, data acquisition or plant advance is executed.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
import sys,ast,hashlib,json,inspect,traceback,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CHECKS=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def ck(name,value,detail=None):CHECKS.append(dict(name=name,passed=bool(value),detail=detail))

def main():
    import research_imports,mujoco
    from v16b_release_goal import runtime as rt
    from v16_release_geometry.plant import ReleasePlant
    from v13_a1.codec import Reference21,EnginePort21
    from v13_joint.lifecycle import ReleaseEntry,release_nominal
    from v13_joint.design import NONTHUMB_CLEAR_TIME_S,THUMB_TUCK_UNTIL_S,THUMB_UNFOLD_TIME_S,LIFT_DESCENT_TIME_S
    from foundation_config import OPEN
    from hand_plant import HandPlant
    old=read(ROOT/'review/v16_release_small_plan_v1.json')['source_sha256'];before=rt.source_identity(rt.METHODS)
    ck('exact126_old123_plus_only3_new',len(old)==123 and len(before)==126 and {k:v for k,v in before.items() if not k.startswith('v16b_release_goal/')}==old and set(k.split('/')[-1] for k in before if k.startswith('v16b_release_goal/'))=={'goal.py','runtime.py','run_development.py'})
    paths=rt.source_files(rt.METHODS)
    ck('all126_actual_bytes_bound',all(sha(paths[k])==v for k,v in before.items()))
    mro=[c.__module__+'.'+c.__name__ for c in rt.ReleaseSession.__mro__]
    ck('cooperative_MRO_extends_frozen_V16',mro[:3]==['v16b_release_goal.runtime.ReleaseSession','v16_release_geometry.runtime.ReleaseSession','v15_support.runtime.SupportSession'],mro)
    calls={name:0 for name in ('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','advance')};forwards=[]
    def deny(name):
        def f(*a,**kw):calls[name]+=1;raise AssertionError('Forbidden integration '+name)
        return f
    original=mujoco.mj_forward
    def initial_forward(*a,**kw):
        stack=[(Path(f.filename).name,f.function) for f in inspect.stack()]
        allowed=any(name=='hand_plant.py' and function=='reset' for name,function in stack)
        forbidden_owner=any(name in ('public_adapter.py','controller.py') for name,function in stack)
        forwards.append(dict(initial_reset=allowed,forbidden_owner=forbidden_owner))
        if not allowed or forbidden_owner:raise AssertionError('Noninitial forward')
        return original(*a,**kw)
    actual=None;error=None
    try:
        with ExitStack() as stack:
            for name in calls:
                if name!='advance' and hasattr(mujoco,name):stack.enter_context(patch.object(mujoco,name,deny(name)))
            stack.enter_context(patch.object(HandPlant,'advance',deny('advance')))
            stack.enter_context(patch.object(mujoco,'mj_forward',initial_forward))
            actual=rt.ReleaseSession(method='ratio_effort',scenario='nominal',seed=619732,split='robustness',motion_mode='hold',task_policy='integrated',receiver_policy='coordinated')
    except Exception as exc:error=dict(error=repr(exc),traceback=traceback.format_exc())
    ck('one_actual_constructor_zero_integration',actual is not None and error is None,error)
    ck('all_step_advance_zero_and_counted_initial_forward',all(x==0 for x in calls.values()) and all(f['initial_reset'] and not f['forbidden_owner'] for f in forwards),dict(integration_calls=calls,initial_forward_calls=forwards))
    if actual is not None:
        meta=actual.metadata()
        ck('real_factory_and_time0_no_sensor_sample',type(actual.plant) is ReleasePlant and actual.time==actual.plant.time==0 and actual.tick==0 and not actual.trace and actual.last_obs is None)
        ck('parent_init_preserves_goal_and_policy_flags',actual.release_clearance_goal is None and actual.v16_release_policy=='coordinated' and actual.v16_supervisor is None and actual.release_adapter is not None)
        ck('metadata126_and_true_parent_scopes',meta['research_source_sha256']==before and meta['schema']=='v16b_explicit_release_clearance_goal_trial_v1' and meta['release_clearance_goal'] is None and meta['task_goal_correction_is_not_algorithm_contribution'] and meta['frozen_v16_source_sha256']==old and len(meta['frozen_v15_source_sha256'])==114)
    # Extract the previous pure helper, not its main/constructor or old tests.
    helper=ROOT/'review/v16_runtime_bridge_independent_v2.py'
    ck('reused_pure_helper_exact_source',sha(helper)=='4ac03746a8ad35abe037fa57b249daee5cee90202ab1adfc2035375d5e8aa517')
    tree=ast.parse(helper.read_text(encoding='utf-8-sig'));fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='make_fake')
    env=dict(np=np,SimpleNamespace=SimpleNamespace,ReleasePlant=ReleasePlant,Reference21=Reference21,EnginePort21=EnginePort21,ReleaseEntry=ReleaseEntry)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(helper),'exec'),env)
    plant,unused,start=env['make_fake']();plant.open_pose=[np.array(x,float) for x in OPEN];plant.v16_supervisor=None;plant.v16_last_version=None
    lo,hi=plant.reference_authority21(include_mode=False);preparation=np.concatenate(plant.open_pose)
    terminal=max(NONTHUMB_CLEAR_TIME_S,THUMB_TUCK_UNTIL_S+THUMB_UNFOLD_TIME_S,LIFT_DESCENT_TIME_S)
    query=float(np.nextafter(plant.joint_release_entry.started_s+terminal,np.inf))
    expected=release_nominal(plant.joint_release_entry,query,preparation,lo,hi).values[:18]
    ck('fixture_distinguishes_true_goal_from_preparation_OPEN',not np.array_equal(expected,preparation))
    order=[];original_goal=rt.release_clearance_goal;captured={}
    def goal(entry,**kw):
        order.append('goal');captured.update(entry=entry,lower=kw['mechanical_lower21'].copy(),upper=kw['mechanical_upper21'].copy());return original_goal(entry,**kw)
    def prior(t):order.append('prior');return None
    def observer(**kw):order.append('observer');return SimpleNamespace(request=kw['request_time_s'])
    class Supervisor:
        def __init__(self,entry,*,open18,observer):
            order.append('supervisor');self.entry=entry;self.open18=np.asarray(open18).copy();self.version=0;self.public_state={};self.was_clear=False;self.reconciles=0
        def lift_interval(self,current):return current.values[18],current.values[18]
        def nominal(self,t,lo,hi):return Reference21(np.clip(np.r_[self.open18,start[18:]],lo,hi),t)
        def observe(self,**kw):order.append('observe');self.version+=1;self.public_state=dict(version=self.version)
        def action(self,*,frame,current21,lower21,upper21,generation,**kw):
            order.append('action');captured.update(action_time=frame.time,action_lower=np.asarray(lower21).copy(),action_upper=np.asarray(upper21).copy(),action_generation=generation)
            return Reference21(np.clip(np.r_[self.open18,current21[18:]],lower21,upper21),frame.time),dict(mode='release',mode_generation=generation)
        def reconcile(self,r):order.append('reconcile');self.reconciles+=1;self.receipt=r
    oldapply=plant.actor21.apply
    def apply(*a,**kw):order.append('actor_apply');return oldapply(*a,**kw)
    plant.actor21.apply=apply
    s=rt.ReleaseSession.__new__(rt.ReleaseSession);s.plant=plant;s.v16_release_policy='coordinated';s.v16_supervisor=None;s.release_clearance_goal=None;s.release_driver=None;s.latest_receiver_measurement=None
    s.release_adapter=SimpleNamespace(begin_release=prior,receiver_half_size_xy_m=np.array([.2,.15]),receiver_up_axis3=np.array([0.,0.,1.]),transfer_inputs=lambda x:{'typed_mock':True})
    state=SimpleNamespace(_last_time=None,time=0.,updates=0)
    def update(frame,dt):state._last_time=state.time=frame.time;state.updates+=1
    state.update=update
    def no_holding(*a,**kw):raise AssertionError('Holding controller must not run')
    s.controller=SimpleNamespace(state=state,step=no_holding,desired=np.ones(5),debug={},motion_reference=np.zeros(3));frame=SimpleNamespace(time=plant.time,dt=.005)
    with patch.object(rt,'release_clearance_goal',goal),patch.object(rt,'TransferObserver',observer),patch.object(rt,'ReleaseSupervisor',Supervisor):desired,prediction,debug=s._lifecycle_step(frame)
    g=s.v16_supervisor;r=g.receipt
    ck('actual_lifecycle_builds_real_clearance_goal_before_parent_execution',order[:4]==['goal','prior','observer','supervisor'] and order.index('action')<order.index('actor_apply')<order.index('reconcile'),order)
    ck('goal_bound_once_into_supervisor_exact_terminal18',np.array_equal(s.release_clearance_goal.reference18,expected) and np.array_equal(g.open18,expected) and s.release_driver is g and plant.v16_supervisor is g)
    ck('goal_uses_actual_entry_and_mechanical_authority',captured['entry'] is plant.joint_release_entry and np.array_equal(captured['lower'],lo) and np.array_equal(captured['upper'],hi))
    ck('future_extraction_query_is_not_issued_command_clock',s.release_clearance_goal.extraction_query_s>frame.time and r.requested.time_s==r.executed.time_s==plant.time==frame.time)
    ck('parent_generation_current_permissions_and_sole_actor',captured['action_generation']==plant.mode_generation==frame.mode_generation and plant.actor21.applications==1 and np.all(r.executed.values>=captured['action_lower']-1e-12) and np.all(r.executed.values<=captured['action_upper']+1e-12))
    ck('actual_receipt_reconciled_and_public_execution_memory',g.reconciles==1 and debug['actual_reference21']==r.executed.values.tolist() and np.array_equal(s.controller.motion_reference,r.executed.values[18:]) and np.array_equal(plant.data.ctrl[:21],r.engine.values))
    ck('holding_preempted_unknown_prediction_task_not_retargeted',state.updates==1 and np.array_equal(desired,np.zeros(5)) and np.isnan(prediction).all() and debug['task_request_modified'] is False)
    # Blind policy takes the identical parent path and does not create a goal.
    blind=rt.ReleaseSession.__new__(rt.ReleaseSession);blind.v16_release_policy='blind';blind.v16_supervisor=None;blind.release_clearance_goal=None;sentinel=object()
    with patch.object(rt.FrozenReleaseSession,'_lifecycle_step',lambda self,obs:sentinel):answer=blind._lifecycle_step(SimpleNamespace())
    ck('blind_policy_no_goal_parent_path_exact',answer is sentinel and blind.release_clearance_goal is None)
    after=rt.source_identity(rt.METHODS);ck('all126_sources_unchanged_through_gate',before==after)
    report=dict(schema='v16b-runtime-bridge-independent-v1',passed=all(x['passed'] for x in CHECKS),counts=dict(total=len(CHECKS),passed=sum(x['passed'] for x in CHECKS),failed=sum(not x['passed'] for x in CHECKS)),checks=CHECKS,source_sha256=before,source_after_sha256=after,checker_sha256=sha(Path(__file__)),reused_helper_sha256=sha(helper),scope=dict(real_session_constructions=1,method='ratio_effort',scenario='nominal',motion='hold',seed=619732,initialization_forward_calls=len(forwards),integration_calls=calls,physics_steps=0,mock_actor=True,real_lifecycle_entry=True,no_runtime_tick=True),limitations=['Finite bridge gate; goal mathematics separately audited, not duplicated.','Fake supervisor isolates goal binding and actual parent commit/reconcile; no real pressure/geometry/physics result.','Initial forward is explicitly allowed frozen initialization, not zero forward.','Does not establish full16s/blind equivalence or improvement.'])
    out=ROOT/'review/v16b_runtime_bridge_independent_v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],counts=report['counts'],forward_initialization=len(forwards),path=str(out),sha256=sha(out)),ensure_ascii=False));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

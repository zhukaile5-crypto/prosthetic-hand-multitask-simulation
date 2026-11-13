"""Finite MRO/source/sole21 bridge gate. One actual zero-integration constructor.

Only frozen foundation initialization mj_forward is allowed and counted.
Every integration API/advance is forbidden; no runtime tick is executed.
Other lifecycle checks use __new__ + a strict fake robot port, not a model.
"""
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch
import sys,ast,hashlib,json,datetime,inspect,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CHECKS=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name,value,detail=None):CHECKS.append(dict(name=name,passed=bool(value),detail=detail))
def constructor(path,cls):
    text=path.read_text(encoding='utf-8-sig');tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    init=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    return ast.dump(init,include_attributes=False),ast.get_source_segment(text,init)

def main():
    plan=json.loads((ROOT/'review/v15A_prospective_multiscene_plan_v2.json').read_text(encoding='utf-8-sig'))
    old=plan['source_sha256'];snapshot=json.loads((ROOT/plan['stages'][0]['output']/'source_snapshot_manifest.json').read_text(encoding='utf-8-sig'))['files']
    old_actual={k:sha(snapshot[k]['original_path']) for k in old}
    check('all_frozen114_actual_bytes_unchanged',old_actual==old,dict(count=len(old),mismatches=[k for k in old if old[k]!=old_actual[k]]))
    oldctor=constructor(ROOT/'v15_support/bootstrap.py','SupportBootstrap');newctor=constructor(ROOT/'v16_release_geometry/bootstrap.py','ReleaseBootstrap')
    check('bootstrap_constructor_AST_and_source_segment_exact',oldctor==newctor)
    import research_imports,mujoco
    from v16_release_geometry import runtime as rt
    from v16_release_geometry.plant import ReleasePlant
    from v16_release_geometry.bootstrap import ReleaseBootstrap
    from v15_support.runtime import SupportSession
    from v15_support.bootstrap import SupportBootstrap
    from v13_a1.codec import Reference21,EnginePort21
    from v13_joint.lifecycle import ReleaseEntry
    from hand_plant import HandPlant
    before=rt.source_identity(rt.METHODS);paths=rt.source_files(rt.METHODS)
    new={k:v for k,v in before.items() if k.startswith('v16_release_geometry/')}
    check('exact_old114_plus_declared_new9_closure',len(before)==123 and set(before)==set(old)|set(new) and len(new)==9 and all(before[k]==v for k,v in old.items()),dict(count=len(before),new_files=list(new)))
    check('frozen_public_adapter_bound',new.get('v16_release_geometry/public_adapter.py')=='0a6d172297e410fccf6a0ce1f386b73157b0c10f59bdf428dfda00f6c7575d44')
    missing=[]
    for key in new:
        path=paths[key]
        for n in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(n,ast.ImportFrom):
                base=path.parent if n.level else ROOT
                for _ in range(max(0,n.level-1)):base=base.parent
                stem=base.joinpath(*(n.module or '').split('.'))
                opts=[stem.with_suffix('.py'),stem/'__init__.py']
            elif isinstance(n,ast.Import):opts=[ROOT.joinpath(*x.name.split('.')).with_suffix('.py') for x in n.names]
            else:continue
            for q in opts:
                if q.is_file() and q.is_relative_to(ROOT) and q.relative_to(ROOT).as_posix() not in before:missing.append(dict(source=key,dependency=str(q),line=n.lineno))
    check('all_new_direct_project_imports_in_source_pin',not missing,missing)
    mro=[c.__name__ for c in rt.ReleaseSession.__mro__]
    check('actual_cooperative_MRO_inserts_ReleaseBootstrap',mro[:5]==['ReleaseSession','SupportSession','ReleaseBootstrap','SupportBootstrap','TrialSession'],mro)
    writes=[]
    for f in ('runtime.py','plant.py','bootstrap.py'):
        tree=ast.parse((ROOT/'v16_release_geometry'/f).read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            if isinstance(n,(ast.Assign,ast.AugAssign)):
                targets=n.targets if isinstance(n,ast.Assign) else [n.target]
                if any('data.ctrl' in ast.unparse(x) for x in targets):writes.append((f,n.lineno))
    check('new_bridge_adds_no_direct_engine_ctrl_writer',not writes,writes)

    # One real constructor. Wrappers count actual calls and forbid integration.
    calls={name:0 for name in ('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','advance')};forward=[]
    def forbidden(name):
        def f(*a,**kw):calls[name]+=1;raise AssertionError('Forbidden integration: '+name)
        return f
    original_forward=mujoco.mj_forward
    def allowed_initial_forward(*a,**kw):
        frames=inspect.stack();locations=[(str(Path(x.filename).resolve()),x.function) for x in frames]
        allowed=any(Path(p).name=='hand_plant.py' and f=='reset' for p,f in locations)
        forbidden_owner=any('/v16_release_geometry/' in p.replace('\\','/') or Path(p).name=='controller.py' for p,f in locations)
        forward.append(dict(initial_reset=allowed,forbidden_owner=forbidden_owner))
        if not allowed or forbidden_owner:raise AssertionError('Forward outside frozen initialization')
        return original_forward(*a,**kw)
    actual=None;construction_error=None
    try:
        with ExitStack() as stack:
            for name in tuple(calls):
                if name!='advance' and hasattr(mujoco,name):stack.enter_context(patch.object(mujoco,name,forbidden(name)))
            stack.enter_context(patch.object(mujoco,'mj_forward',allowed_initial_forward))
            stack.enter_context(patch.object(HandPlant,'advance',forbidden('advance')))
            actual=rt.ReleaseSession(method='ratio_effort',scenario='nominal',seed=619732,split='robustness',motion_mode='hold',task_policy='integrated',receiver_policy='coordinated')
    except Exception as exc:construction_error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    check('one_actual_zero_integration_constructor',actual is not None and construction_error is None,construction_error)
    check('all_integration_advance_calls_zero',all(v==0 for v in calls.values()),calls)
    check('forward_only_frozen_initial_reset',all(x['initial_reset'] and not x['forbidden_owner'] for x in forward),dict(count=len(forward),calls=forward))
    if actual is not None:
        check('factory_is_actual_ReleasePlant',type(actual.plant) is ReleasePlant,type(actual.plant).__name__)
        check('parent_init_preserves_v16_flags_and_adapter',actual.v16_release_policy=='coordinated' and actual.v16_supervisor is None and actual.release_adapter is not None and actual.receiver_policy=='blind' and not actual.plant.receiver_release_enabled)
        check('construction_clock_tick_trace_zero',actual.time==0. and actual.plant.time==0. and actual.tick==0 and not actual.trace and actual.last_obs is None,dict(time=actual.time,plant_time=actual.plant.time,tick=actual.tick,trace=len(actual.trace)))
        meta=actual.metadata()
        check('actual_metadata_and_allmethod_source_identity',meta['schema']=='v16_public_transfer_geometry_trial_v1' and meta['receiver_policy']=='coordinated' and meta['research_source_sha256']==before and actual.research_sources==before)

    # Pure actor buffer: execute real inherited commit with new virtual dispatch.
    def make_fake():
        p=ReleasePlant.__new__(ReleasePlant);p.data=SimpleNamespace(time=1.,ctrl=np.zeros(22))
        p.config=SimpleNamespace(control_dt=.005,lift_target=1.)
        p.model=SimpleNamespace(joint=lambda name:SimpleNamespace(id=0 if name=='rh_WRJ2' else 1),jnt_limited=np.zeros(2,bool),jnt_range=np.zeros((2,2)))
        p.open_pose=[np.zeros(5),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(4)]
        value=np.r_[np.full(18,.3),.05,.1,-.1]
        class Actor:
            lift_bias_m=.1;indices21=np.arange(21)
            ctrl_range21=np.tile([-2.,2.],(21,1));ctrl_range21[18]=[.1,1.1]
            applications=0
            def encode(self,r):
                v=r.values.copy();v[18]+=self.lift_bias_m;return EnginePort21(v,r.time_s,self.lift_bias_m)
            def apply(self,data,r):
                self.applications+=1;e=self.encode(r);data.ctrl[self.indices21]=e.values;return e
        p.actor21=Actor();p._joint_committed=Reference21(value,1.);p._joint_mode='release';p.mode_generation=7
        p.joint_release_entry=ReleaseEntry(.5,Reference21(value,.5),7);p.release_started=.5
        p.receiver_release_tasks=None;p.receiver_release_version=None;p.receiver_release_enabled=False
        p._joint_anchor_time=None;p._joint_anchor_reference=None
        p.actor_callback_total_ns=0;p.actor_callback_calls=0;p.obstacle_motor=21;p.env={'obstacle_x':0.}
        p.a1_last_engine_port21=p.actor21.encode(p._joint_committed);p.data.ctrl[:21]=p.a1_last_engine_port21.values
        p.v16_supervisor=None;p.v16_last_version=None;p.lift_ref=.05;p.wrist_ref=np.array([.1,-.1])
        class Graph:
            version=0;was_clear=False;public_state={};nominal_calls=0;reconciles=0
            def lift_interval(self,ref):return float(ref.values[18]),float(ref.values[18])
            def nominal(self,t,lo,hi):
                self.nominal_calls+=1;v=value.copy();v[:18]-=.1;v[18]=0.;return Reference21(np.clip(v,lo,hi),t)
            def observe(self,**kw):self.version+=1;self.public_state=dict(version=self.version)
            def action(self,*,frame,**kw):return Reference21(value-np.r_[np.full(18,.1),0.,0.,0.],frame.time),dict(test_graph=True)
            def reconcile(self,receipt):self.reconciles+=1;self.receipt=receipt
        g=Graph();p.bind_supervisor(g);return p,g,value
    fake,g,start=make_fake();g.version+=1;fake.sync_supervisor_generation();generation=fake.mode_generation;fake.sync_supervisor_generation()
    check('graph_version_invalidates_generation_exactly_once',generation==8 and fake.mode_generation==8)
    r=fake.commit_reference21(Reference21(start+.1,1.),mode='holding',expected_generation=7)
    check('stale_holding_commit_uses_new_graph_nominal',r.stale_plan_replaced and r.mode=='release' and g.nominal_calls==1 and not np.array_equal(r.executed.values,start+.1))
    check('new_lift_permission_effective_in_commit',r.executed.values[18]==start[18])
    check('one_inherited_sole_robot21_writer',fake.actor21.applications==1 and fake.actor_callback_calls==1 and np.array_equal(fake.data.ctrl[:21],r.engine.values))
    try:fake.set_closure(np.ones(5));alpha_blocked=False
    except RuntimeError:alpha_blocked=True
    check('legacy_alpha_cannot_overwrite_full21',alpha_blocked)
    # Current lifecycle body: public filter only, graph -> commit -> reconcile.
    fake,g,start=make_fake();s=rt.ReleaseSession.__new__(rt.ReleaseSession);s.v16_release_policy='coordinated';s.plant=fake;s.v16_supervisor=g;s.release_driver=g
    s.release_adapter=SimpleNamespace(transfer_inputs=lambda x:{'typed_inputs':True});s.latest_receiver_measurement=None
    state=SimpleNamespace(_last_time=None,time=0.,updates=0)
    def update(obs,dt):state._last_time=obs.time;state.time=obs.time;state.updates+=1
    state.update=update
    def holding_forbidden(*a,**kw):raise AssertionError('Old holding controller executed in release')
    s.controller=SimpleNamespace(state=state,step=holding_forbidden,desired=np.ones(5),debug={},motion_reference=np.zeros(3))
    obs=SimpleNamespace(time=1.,dt=.005)
    desired,predicted,debug=s._lifecycle_step(obs)
    check('release_bypasses_old_controller_and_reconciles_actual_receipt',g.reconciles==1 and fake.actor21.applications==1 and state.updates==1 and np.array_equal(desired,np.zeros(5)) and np.isnan(predicted).all())
    check('release_attachment_coherent_mode_and_actual_reference',obs.mode_generation==fake.mode_generation and np.array_equal(s.controller.motion_reference,fake.reference_snapshot().values[18:]) and debug['actual_reference21']==g.receipt.executed.values.tolist())
    check('release_lifecycle_original_task_unmodified',debug['task_request_modified'] is False)
    after=rt.source_identity(rt.METHODS)
    check('source123_stable_through_all_checks',before==after)
    report=dict(schema='v16-runtime-bridge-independent-v1',role='independent_source_MRO_and_execution_bridge_gate',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=before,source_after_sha256=after,test_source_sha256=sha(Path(__file__)),reference_frozen_plan_sha256=sha(ROOT/'review/v15A_prospective_multiscene_plan_v2.json'),scope=dict(real_session_constructions=1,methods_constructed=['ratio_effort'],physics_steps=0,initialization_forward_calls=len(forward),mock_native_ports=True,no_runtime_ticks=True,no_performance_or_physical_claim=True),checks=CHECKS,counts=dict(total=len(CHECKS),passed=sum(x['passed'] for x in CHECKS),failed=sum(not x['passed'] for x in CHECKS)),passed=all(x['passed'] for x in CHECKS))
    out=ROOT/'review/v16_runtime_bridge_independent_v1.json'
    if out.exists():raise FileExistsError('Preserve evidence')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(report=str(out),sha256=sha(out),counts=report['counts'],passed=report['passed'],initialization_forward_calls=len(forward)),ensure_ascii=False))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

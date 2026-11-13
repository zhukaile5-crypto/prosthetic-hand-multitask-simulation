"""Logging-only source/constructor and actual lifecycle differential gate.

Three baseline constructors, no sensor/physics tick. A strict fake robot port
then drives parent and child actual lifecycle methods on identical public data.
One small CONTACT kernel problem per decision; no physical replay or model FK.
"""
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch
import sys,ast,inspect,json,hashlib,traceback,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CHECKS=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def ck(name,value,detail=None):CHECKS.append(dict(name=name,passed=bool(value),detail=detail))
def norm(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:norm(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [norm(v) for v in x]
    return x
def same(a,b):return json.dumps(norm(a),sort_keys=True,allow_nan=True)==json.dumps(norm(b),sort_keys=True,allow_nan=True)
def strip_timing(x):
    if isinstance(x,dict):return {k:strip_timing(v) for k,v in x.items() if k not in ('solver_ms','solver_timing_ms','elapsed_ms','degenerate_face_elapsed_ms','motion_task_public','motion_evidence_role')}
    if isinstance(x,(list,tuple)):return [strip_timing(v) for v in x]
    return norm(x)
def runner_tree(path,child=False):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    if child:
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module=='v17b_full_evidence.runtime':node.module='v17_contact_transfer.runtime'
            if isinstance(node,ast.Constant) and isinstance(node.value,str):
                node.value=node.value.replace('V17bEvidenceOnly','V17ContactTransfer').replace('v17b_complete_motion_evidence','v17_contact_task').replace('v17b_full_evidence','v17_contact_transfer')
    return ast.dump(tree,include_attributes=False)
def method_tree(path,cls,name):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'));c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return ast.dump(next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)

def main():
    import research_imports,mujoco
    from v17_contact_transfer import runtime as Parent
    from v17b_full_evidence import runtime as Child
    from v17_contact_transfer.motion_domain_sensors import MotionDomainSensors
    from v17_contact_transfer.motion_adapter import PublicMotionAdapter,MotionTask
    from v15_support.receiver_sensor import ReceiverMeasurement
    from v16_release_geometry.transfer_observer import HandSupportSample
    from v16_release_geometry.plant import ReleasePlant
    from v13_a1.codec import Reference21,EnginePort21
    from v13_joint.lifecycle import ReleaseEntry
    from foundation_config import OPEN
    from hand_plant import HandPlant
    parent_pin=read(ROOT/'review/v17_transfer_small_plan_v1.json')['source_sha256'];before=Child.source_identity(Child.METHODS)
    ck('current136_and_frozen_parent133_exact',len(before)==136 and len(parent_pin)==133 and all(before.get(k)==v for k,v in parent_pin.items()) and {k.split('/')[-1] for k in before if k.startswith('v17b_full_evidence/')}=={'runtime.py','supervisor.py','run_development.py'})
    ck('runner_AST_namespace_schema_only',runner_tree(ROOT/'v17_contact_transfer/run_development.py')==runner_tree(ROOT/'v17b_full_evidence/run_development.py',True))
    ck('runtime_lifecycle_AST_same_construction_parent_path',method_tree(ROOT/'v17_contact_transfer/runtime.py','ReleaseSession','_lifecycle_step')==method_tree(ROOT/'v17b_full_evidence/runtime.py','ReleaseSession','_lifecycle_step'))
    forbidden=[];forwards=[]
    def deny(name):
        def f(*a,**kw):forbidden.append(name);raise AssertionError('Forbidden '+name)
        return f
    original_forward=mujoco.mj_forward
    def initialization_forward(*a,**kw):
        stack=[(Path(f.filename).name,f.function) for f in inspect.stack()]
        allowed=('hand_plant.py','reset') in stack and not any(n in ('public_adapter.py','motion_adapter.py','controller.py') for n,f in stack)
        forwards.append(dict(frozen_reset=allowed))
        if not allowed:raise AssertionError('Forward outside frozen initialization')
        return original_forward(*a,**kw)
    models=[];construction=[];global_rng=np.random.get_state()
    with ExitStack() as guards:
        for name in ('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta'):
            if hasattr(mujoco,name):guards.enter_context(patch.object(mujoco,name,deny(name)))
        guards.enter_context(patch.object(HandPlant,'advance',deny('advance')))
        guards.enter_context(patch.object(mujoco,'mj_forward',initialization_forward))
        for method in Child.METHODS:
            try:
                s=Child.ReleaseSession(method=method,scenario='nominal',seed=619732,split='robustness',motion_mode='hold',task_policy='legacy',receiver_policy='blind')
                models.append(s);meta=s.metadata()
                ok=(s.time==s.plant.time==0 and s.tick==0 and not s.trace and s.last_obs is None and s.sensors.samples==s.sensors.inner.samples==0 and type(s.sensors) is MotionDomainSensors and type(s.release_adapter) is PublicMotionAdapter and s.sensors.inner.__class__.__name__=='HandSensors' and s.v16_supervisor is None and meta['research_source_sha256']==before and meta['frozen_v17_source_sha256']==parent_pin and meta['schema']=='v17b_complete_motion_evidence_trial_v1' and meta['decision_algorithm']=='unchanged_V17_contact_transfer')
                ck('actual_blind_constructor_shared_instrument_'+method,ok)
                construction.append(dict(method=method,time_s=s.time,samples=s.sensors.samples,receiver_acquisitions=s.receiver_sensor._acquisitions,sensor_rng=s.sensors.inner.rng.bit_generator.state,receiver_rng=s.receiver_sensor._rng.bit_generator.state))
            except Exception as exc:ck('actual_blind_constructor_'+method,False,dict(error=repr(exc),traceback=traceback.format_exc()))
    ck('three_baseline_RNG_states_and_no_extra_acquisition',len(construction)==3 and all(same(x['sensor_rng'],construction[0]['sensor_rng']) and same(x['receiver_rng'],construction[0]['receiver_rng']) and x['receiver_acquisitions']==0 for x in construction))
    ck('legacy_global_numpy_rng_unchanged',same(global_rng,np.random.get_state()))
    ck('initialization_forward_counted_no_step',len(forwards)==3 and all(x['frozen_reset'] for x in forwards) and not forbidden,dict(forwards=forwards,forbidden=forbidden))
    # Exact previous fake port helper; no old main/constructors or writer edits.
    helper=ROOT/'review/v16_runtime_bridge_independent_v2.py';assert sha(helper)=='4ac03746a8ad35abe037fa57b249daee5cee90202ab1adfc2035375d5e8aa517'
    node=next(n for n in ast.walk(ast.parse(helper.read_text(encoding='utf-8-sig'))) if isinstance(n,ast.FunctionDef) and n.name=='make_fake')
    env=dict(np=np,SimpleNamespace=SimpleNamespace,ReleasePlant=ReleasePlant,Reference21=Reference21,EnginePort21=EnginePort21,ReleaseEntry=ReleaseEntry);exec(compile(ast.Module(body=[node],type_ignores=[]),str(helper),'exec'),env)
    class Adapter:
        receiver_half_size_xy_m=np.array([.2,.15]);receiver_up_axis3=np.array([0.,0.,1.])
        def __init__(self):self._motion_task=None;self.productions=0;self.tasks={};self.frame=None
        def begin_release(self,t):return None
        def transfer_inputs(self,r):
            t=self.frame.time
            return dict(time_s=t,hand=HandSupportSample(t,t-.01,True,1.,.02),receiver=r)
        @property
        def motion_task(self):return self._motion_task
        def current_geometry_tasks(self,frame,current,desired,lo,hi):
            self.productions+=1;C=np.zeros((1,21));C[0,0]=-.01;M=np.zeros((2,21));M[0,1]=.01;M[1,2]=.01
            self._motion_task=MotionTask(frame.time,M,np.array([-.00001,.00002]),dict(role='pure_public_motion_fixture',read_time_s=frame.time));self.tasks[frame.time]=self._motion_task
            return SimpleNamespace(status='EXPERIMENTAL_GEOMETRY_TASKS',normal_matrix_m_per_reference=C,normal_offset_m=np.zeros(1),step_lower21=lo,step_upper21=hi,tangent_matrix_m_per_reference=np.empty((0,21)),tangent_target_m=np.empty(0),open_target_increment21=np.clip(desired-current,lo,hi),diagnostics=dict(current_time_s=frame.time,patches=[dict(observed_normal_N=1.)]))
    def session(cls):
        plant,_,_=env['make_fake']();plant.open_pose=[np.array(v,float) for v in OPEN];plant.v16_supervisor=None;plant.v16_last_version=None
        s=cls.__new__(cls);s.plant=plant;s.v16_release_policy='coordinated';s.v16_supervisor=None;s.release_clearance_goal=None;s.release_driver=None;s.release_adapter=Adapter();s.latest_receiver_measurement=None
        state=SimpleNamespace(_last_time=None,time=0.)
        def update(frame,dt):state._last_time=state.time=frame.time
        state.update=update
        s.controller=SimpleNamespace(state=state,step=deny('holding step in release'),desired=np.zeros(5),debug={},motion_reference=np.zeros(3));return s
    parent=session(Parent.ReleaseSession);child=session(Child.ReleaseSession);commands=[];sigma=np.array([.01]*3+[.0005]*3)
    for tick in (200,201,202):
        for s in (parent,child):
            s.plant.control_steps=tick;s.plant.data.time=s.plant.time;t=s.plant.time
            frame=SimpleNamespace(time=t,dt=.005,normal=np.ones(5),valid=np.ones(5,bool),age=.01,sample_time=t-.01,auxiliary_pressure=0.,auxiliary_valid=True,auxiliary_age=.01)
            s.release_adapter.frame=frame;s.release_adapter._motion_task=None
            s.latest_receiver_measurement=ReceiverMeasurement(t,t-.01,.01,True,np.array([0.,0.,-.2,0.,0.,0.]),.2,sigma,3*sigma,None,'boundary_after_forward',None)
            with patch.object(mujoco,'mj_forward',deny('mock forward')),patch.object(HandPlant,'advance',deny('mock advance')):s._lifecycle_step(frame)
        commands.append(dict(time_s=t,parent=parent.plant.last_execution_receipt21.executed.values.copy(),child=child.plant.last_execution_receipt21.executed.values.copy()))
    ck('first_actual_lifecycle_binds_logging_child',type(child.v16_supervisor) is Child.ReleaseSupervisor and type(parent.v16_supervisor) is Parent.ReleaseSupervisor and child.release_driver is child.v16_supervisor and child.plant.v16_supervisor is child.v16_supervisor)
    ck('APPROACH_and_CONTACT_actual_commands_byte_equal',all(x['parent'].tobytes()==x['child'].tobytes() for x in commands),[dict(time_s=x['time_s'],equal=x['parent'].tobytes()==x['child'].tobytes()) for x in commands])
    ck('no_extra_geometry_production_decision_or_actor',parent.release_adapter.productions==child.release_adapter.productions and len(parent.v16_supervisor.decision_evidence)==len(child.v16_supervisor.decision_evidence) and parent.plant.actor21.applications==child.plant.actor21.applications==3)
    ck('parent_decision_algorithm_evidence_unchanged_except_wall_and_added_copy',same(strip_timing(parent.v16_supervisor.decision_evidence),strip_timing(child.v16_supervisor.decision_evidence)))
    log=child.v16_supervisor.decision_evidence;motion=[q for q in log if q['motion_task_available']];bad=[]
    for q in log:
        task=q['motion_task_public']
        if q['motion_task_available']:
            original=child.release_adapter.tasks[q['time_s']]
            if task is None or task['time_s']!=q['time_s'] or np.asarray(task['matrix_Mx21']).tobytes()!=original.matrix_Mx21.tobytes() or np.asarray(task['target_M']).tobytes()!=original.target_M.tobytes():bad.append(q['decision_id'])
        elif task is not None:bad.append(q['decision_id'])
    ck('all_logged_M_target_bind_same_new_decision_not_stale_reuse',bool(motion) and not bad,dict(records=len(log),motion_records=len(motion),bad=bad))
    q=motion[0];original=child.release_adapter.tasks[q['time_s']];oldbytes=original.matrix_Mx21.tobytes();saved=q['motion_task_public']['matrix_Mx21'][0][0];q['motion_task_public']['matrix_Mx21'][0][0]+=1.
    ck('evidence_is_a_copy_not_mutated_task_input',original.matrix_Mx21.tobytes()==oldbytes);q['motion_task_public']['matrix_Mx21'][0][0]=saved
    ck('actual_current_clocks_generation_receipts_reconciled',child.plant.last_execution_receipt21.executed.time_s==child.plant.time and child.controller.debug['actual_reference21']==child.plant.reference_snapshot().values.tolist() and child.controller.debug['mode_generation']==child.plant.mode_generation)
    after=Child.source_identity(Child.METHODS);ck('all136_stable_after_gate',before==after);ck('no_physics_or_holding_calls',not forbidden,forbidden)
    report=dict(schema='v17b-logging-independent-v1',passed=all(x['passed'] for x in CHECKS),counts=dict(total=len(CHECKS),passed=sum(x['passed'] for x in CHECKS),failed=sum(not x['passed'] for x in CHECKS)),checks=CHECKS,source_sha256=before,source_after_sha256=after,checker_sha256=sha(Path(__file__)),reused_pure_helper_sha256=sha(helper),scope=dict(real_blind_constructors=3,initial_forward_calls=len(forwards),physics_steps=0,actual_physical_ticks=0,pure_lifecycle_ticks=3,pure_contact_kernel_used=True,recorded_commands_compared=len(commands),author65_replay_not_reused=True),limitations=['Source/constructor/pure lifecycle gate; no new full16s equivalence or performance experiment.','Logging adds serialization work and memory; runtime wall timing may change.','Clock equality checked for already-created legal producer MotionTasks; no new public measurement or native authority.'])
    out=ROOT/'review/v17b_logging_independent_v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(norm(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],counts=report['counts'],source_count=len(before),path=str(out),sha256=sha(out)),ensure_ascii=False));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

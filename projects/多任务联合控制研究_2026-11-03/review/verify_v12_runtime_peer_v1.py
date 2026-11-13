"""Independent captured-runtime contract oracles: no import of plant/controller.

Actual runtime methods are executed with explicit inert interfaces. No real
MjModel, MjData, preparation trajectory or dynamics equivalence is tested here.
"""
from pathlib import Path
from types import SimpleNamespace as NS
from copy import deepcopy
from numbers import Real
import ast, hashlib, json, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
OUT=ROOT/'review/v12_runtime_bridge_independent_v1.json'

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def main():
    p=ROOT/'v12/runtime.py';raw=p.read_bytes();tree=ast.parse(raw)
    keep=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.Assign))]
    env=dict(Path=Path,deepcopy=deepcopy,Real=Real,ast=ast,hashlib=hashlib,json=json,sys=sys,time=time,np=np,
             ROOT=ROOT,FOUNDATION=FOUNDATION,TrialSession=type('InertTrialSession',(),{}),SENSOR_PERIOD=.005,
             jsonable=plain)
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(p),'exec'),env)
    Session=env['V12Session'];results=[]
    def check(name,ok,detail):results.append(dict(name=name,passed=bool(ok),detail=plain(detail)))
    def handoff(normal,valid):
        c=Session.__new__(Session);c.plant=NS(phase='HOLDING',closure_min=np.zeros(5),closure_max=np.ones(5))
        c.config=NS(lift_target=.1);c.calibration=dict(reference_alpha=np.full(5,.5));c.trust_radius=.05;c.time=7.5
        c.event=lambda message:None;c.failure_reason=None
        ob=NS(valid=np.array(valid,bool),normal=np.array(normal,float),alpha=np.full(5,.9),lift=.1)
        c._handoff(ob,dict(relative=np.zeros(3)))
        return c
    c=handoff([1.]*5,[True]*5)
    check('physical_ready_outside_legacy_B_is_admitted',c.handoff_passed and not c.legacy_model_domain_supported and
          np.array_equal(c.local_lower,np.zeros(5)) and np.array_equal(c.local_upper,np.ones(5)),
          dict(handoff=c.handoff_passed,legacy_supported=c.legacy_model_domain_supported,actual_lower=c.local_lower,actual_upper=c.local_upper))
    c=handoff([0.,1.,1.,1.,1.],[True]*5)
    check('TH_absence_is_physical_failure_despite_four_other_contacts',not c.handoff_passed and c.failure_reason=='preparation_not_ready',dict(handoff=c.handoff_passed,failure=c.failure_reason))
    c=handoff([1.,1.,1.,1.,1.],[True,True,False,False,False])
    check('two_valid_contacts_do_not_meet_readiness',not c.handoff_passed,dict(handoff=c.handoff_passed))

    calls=[];c=Session.__new__(Session)
    c.plant=NS(alpha=np.full(5,.5),closure_min=np.zeros(5),execute_delta=lambda delta:calls.append(('execute',delta.copy())))
    c.controller=NS(state=NS(update=lambda frame,dt:calls.append(('observer',dt))),desired=np.ones(5),
        step=lambda *args:(_ for _ in ()).throw(AssertionError('holding controller must not run')))
    desired,pred,debug=c._lifecycle_step(NS())
    check('native_release_helper_bypasses_holding_controller',len(calls)==2 and np.allclose(calls[0][1],-.004) and
          np.array_equal(desired,np.zeros(5)) and np.isnan(pred).all() and debug['release_authority'],
          dict(calls=calls,desired=desired,prediction_is_all_unknown=bool(np.isnan(pred).all())))

    model=NS(body_subtreemass=np.array([.5]),joint=lambda name:NS(id={'transport_lift':0,'rh_WRJ2':1,'rh_WRJ1':2}[name]),
        actuator=lambda name:NS(id={'lift_motor':0,'rh_A_WRJ2':1,'rh_A_WRJ1':2}[name]),
        jnt_limited=np.ones(3,bool),jnt_range=np.array([[0.,.2],[-.2,.2],[-.25,.25]]),
        actuator_ctrllimited=np.ones(3,bool),actuator_ctrlrange=np.array([[0.,.11],[-.3,.3],[-.4,.4]]))
    bounds=env['public_motion_reference_bounds'](NS(model=model,config=NS(lift_target=.1),hand=0))
    check('motion_authority_intersects_native_reference_joint_and_port_bounds',np.allclose(bounds['lower'],[0.,-.2,-.25]) and np.allclose(bounds['upper'],[.1,.2,.25]),bounds)

    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='V12Session')
    step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step_one_sensor_tick')
    # Execute only the actual sensor/telemetry prefix, before any task/plant step.
    prefix=[n for n in step.body if 280<=n.lineno<=297]
    ob=NS(raw_joint=np.arange(24.)/100.,raw_joint_velocity=np.arange(24.)/1000.,lift=.08,
        **{key:np.array([987.]) for key in env['LEGACY_MODEL_FIELDS']})
    current_ctrl=np.arange(22.)/10.;axis=np.r_[np.arange(18),20,18,19]
    c=Session.__new__(Session);c.time=7.3;c.plant=NS(time=7.3,truth=lambda:dict(),raw_sensors=lambda:dict(lift_velocity=.015),
        data=NS(ctrl=current_ctrl.copy()),lift_ref=.08,wrist_ref=np.array([.01,-.02]))
    c.sensors=NS(sample=lambda raw:ob);c.skin_geometry=NS(acquire=lambda raw,obs:None)
    c.robot_actuator_ids21=axis;c.robot_actuator_axes21=['a'+str(i) for i in axis];c.sensor_characteristics=dict(force_sigma_N=.01)
    scope={**env,'self':c}
    exec(compile(ast.Module(body=prefix,type_ignores=[]),str(p),'exec'),scope)
    check('all_legacy_model_fields_deleted_before_new_observer',not any(hasattr(ob,key) for key in env['LEGACY_MODEL_FIELDS']),
          dict(removed_fields=env['LEGACY_MODEL_FIELDS'],legacy_common_sensor_kept_for_prefix_comparison='common_sensor_json' in scope))
    check('current_actual_robot21_port_and_clock_telemetry',np.array_equal(ob.robot_actuator_target21,current_ctrl[axis]) and ob.robot_control_time_s==7.3,
          dict(actual_target21=ob.robot_actuator_target21,clock=ob.robot_control_time_s,selected_axis_ids21=axis))
    check('robot_q25_v25_and_executed_reference_are_public_robot_only',np.array_equal(ob.robot_position25,np.r_[np.arange(24.)/100.,.08]) and
          np.array_equal(ob.robot_velocity25,np.r_[np.arange(24.)/1000.,.015]) and np.array_equal(ob.executed_motion_reference,[.08,.01,-.02]),
          dict(q25=ob.robot_position25,v25=ob.robot_velocity25,executed_reference=ob.executed_motion_reference))
    state=NS(load=None,auxiliary_support=None)
    assignments=[n for n in ast.walk(step) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('estimated_load','auxiliary_support') for t in n.targets)]
    scope={**env,'state':state};exec(compile(ast.Module(body=assignments,type_ignores=[]),str(p),'exec'),scope)
    check('unknown_load_and_aux_support_logged_NaN',np.isnan(scope['estimated_load']) and np.isnan(scope['auxiliary_support']),
          dict(estimated_load_unknown=bool(np.isnan(scope['estimated_load'])),auxiliary_support_unknown=bool(np.isnan(scope['auxiliary_support']))))

    # Actual source walker, without importing executable runtime/controller.
    sources=env['source_files'](env['METHODS']);identities={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sources.items()}
    check('source_closure_has_all_new_core_and_no_empirical_servo_bank',
        all(k in sources for k in ('v12/runtime.py','v12/controller.py','v12/observation.py','v12/intents.py','v12/effort_tracker.py','v11/robot_servo_effort.py','v11/object_belief.py','v10/robot_contact_space.py')) and
        not any('servo_bank' in k or k.endswith('.npz') and 'kernel' in k for k in sources),
        dict(count=len(sources),source_keys=list(sources)))
    init=ast.parse((ROOT/'v12/__init__.py').read_bytes())
    check('v12_package_init_has_no_execution_side_effect',all(isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) for n in init.body),dict(init_sha256=hashlib.sha256((ROOT/'v12/__init__.py').read_bytes()).hexdigest()))
    stable=p.read_bytes()==raw
    output=dict(schema='v12_runtime_bridge_independent_v1',role='independent_read_only_peer',physics_steps=0,plant_constructors=0,
        real_model_constructed=False,source_sha256=hashlib.sha256(raw).hexdigest(),source_file=str(p),runtime_stable_during_check=stable,
        actual_source_closure_sha256=identities,source_closure_count=len(sources),fixtures=results,all_assertions_passed=all(t['passed'] for t in results),
        line_references=dict(source_closure=[27,97],physical_not_B_ready=[212,230],legacy_fields_and_current_robot_port=[278,297],
                             native_release_priority=[315,339,359,368],mechanical_bounds=[225,357,358],unknowns=[400,424]),
        findings=[],limitations=[
            'Source closure is observed now, not a complete V12 freeze; controller/tracker author may revise before root freeze.',
            'Pre7.5 physical prefix bit equality, 7.5 pre-action integration/RNG/FK and actual pressure/retention/release remain untested here; root bridge/physical pilot required.',
            'Native release helper is tested with inert execute_delta; no assertion of mechanical clearance or force safety.',
            'public_motion_reference_bounds assumes the frozen plant lift control gain 1500 and actual native reference clipping; not a generic changed-robot authority adapter.',
            'Old foundation observer/model code remains in bootstrap/sensor initialization dependency closure; six model fields are removed before new observer/controller calls.',
            'Source79 number is an identity description, not an algorithm efficacy denominator or global admission.'
        ])
    OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
    print(json.dumps(dict(path=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),fixtures=len(results),passed=output['all_assertions_passed'],runtime_stable=stable,source_count=len(sources)),ensure_ascii=False))
    if not stable or not output['all_assertions_passed']:raise SystemExit(1)

if __name__=='__main__':main()

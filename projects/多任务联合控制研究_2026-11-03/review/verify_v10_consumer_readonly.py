"""Read-only consumer review: real method ASTs, pure fixtures, no dynamics.

No constructor/plant is instantiated. Only frozen robot dataclasses and Bank
data are loaded; controller methods are compiled unchanged from their AST.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parent/'五指策略接入_2026-11-03'))
from v10.robot_contact_space import RobotJointSample,PublicSkinPatch
from v10.servo_bank import Bank


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method(source_path,class_name,function_name,globals_dict):
    tree=ast.parse(Path(source_path).read_text(encoding='utf-8'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==class_name)
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==function_name)
    namespace=dict(globals_dict);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source_path),'exec'),namespace)
    return namespace[function_name]


def verify(out):
    if out.exists():raise ValueError('Fresh review directory required')
    out.mkdir(parents=True)
    cp=ROOT/'v10/controller.py';rp=ROOT/'v10/runtime.py';gp=ROOT/'v10/robot_contact_space.py';bp=ROOT/'v10/servo_bank.py'
    sources={str(p):sha(p) for p in (cp,rp,gp,bp)}
    # Preserve the reviewed bytes independently from any ongoing root edits.
    (out/'controller_reviewed_source.py').write_bytes(cp.read_bytes())
    (out/'runtime_reviewed_source.py').write_bytes(rp.read_bytes())
    global_ns=dict(np=np,RobotJointSample=RobotJointSample,PublicSkinPatch=PublicSkinPatch)
    motion=method(cp,'JointTaskController','_motion_plan',global_ns)
    groups=method(cp,'JointTaskController','_patch_groups',global_ns)
    exact=method(cp,'JointTaskController','_exact_geometry',global_ns)
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    first=json.loads((ROOT/'output/robot_servo_response_all15_v2/capsule/checkpoint_00/model.json').read_text(encoding='utf-8'))
    axes=first['output_axes25'];names=axes[:24]
    frame=NS(time=8.,sample_time=7.99,raw_joint=np.arange(24)*.01,raw_joint_velocity=np.arange(24)*.002,lift=.08,lift_ref=.1)
    sample=RobotJointSample.from_public_frame(frame,names,lift_velocity_m_s=.006)
    check('robot_q25_lift_uses_actual_encoder_not_reference',sample.q25[-1]==.08)
    check('robot_velocity25_lift_explicit',sample.velocity25[-1]==.006)
    check('current_robot_sample_clock_not_delayed_tactile_clock',sample.time_s==frame.time and sample.time_s!=frame.sample_time)
    bank=Bank(axes);selection=bank.select('capsule',sample.q25,sample.velocity25,np.asarray(first['base_reference8']))
    check('bank_free_uses_current_full25_state',np.array_equal(selection.free25,sample.q25+.02*sample.velocity25))
    missing=RobotJointSample.from_public_frame(frame,names)
    check('missing_lift_velocity_fail_closed',missing.velocity25 is None and bank.select('capsule',missing.q25,missing.velocity25,np.asarray(first['base_reference8'])).response25x8 is None)
    target=np.array([.0798,.001,.001])
    state=NS(alpha=np.full(5,.5),closure_min=np.zeros(5),closure_max=np.ones(5),valid=np.ones(5,bool),closure_permission=np.ones(5,bool))
    owner=NS(latest_frame=NS(time=8.,motion_reference_rate_limits=np.array([.035,.12,.12])),g=NS(decision_period=.02,open_rate=.5,close_rate=.3),state=state,
        nominal=NS(interval_action=np.full(5,.004)),executed_motion=np.array([.08,0.,0.]),task_request=dict(motion_target=np.array([.08,0.,0.]),
        motion_preview=dict(future_reference_available=True,requested_times_s=[8.,8.02],motion_targets=[ [.08,0.,0.],target.tolist()]),
        motion_reference_bounds=dict(lower=[0.,-.5,-.5],upper=[.1,.5,.5])))
    chosen,lo,hi,nominal=motion(owner)
    check('future_target_is_declared_now_plus20ms_preview',np.allclose(chosen,target))
    check('motion_delta_is_from_actual_executed_reference',np.allclose(nominal[5:],target-owner.executed_motion))
    check('future_endpoint_within_actual_motion_bounds',np.all(owner.executed_motion+nominal[5:]>=np.asarray(owner.task_request['motion_reference_bounds']['lower'])) and np.all(owner.executed_motion+nominal[5:]<=np.asarray(owner.task_request['motion_reference_bounds']['upper'])))
    check('motion_step_rate_bounds_respected',np.all(nominal[5:]>=lo[5:]) and np.all(nominal[5:]<=hi[5:]))
    # Actual method applies an old-acquisition normal at current robot FK. A
    # synthetic row isolates the dimensional projection, not a new FK test.
    row=np.zeros(25);row[0]=.02;row[-1]=1.
    patch=dict(digit=0,robot_body=1,link_point=[0.,0.,0.],link_normal=[1.,0.,0.],reference_position=[0.,0.,0.],reference_normal=[1.,0.,0.],normal=2.,sample_time=7.99,age=.01)
    def linearize(s,p,maximum_age_s):
        check('patch_acquisition_time_retained',p.sample_time_s==7.99 and maximum_age_s==.12)
        return NS(available=True,normal_jacobian25=row,kinematics=NS(point_world=np.zeros(3)),acquisition_normal_world=np.array([1.,0.,0.]))
    geometry=NS(linearize=linearize)
    local=NS(latest_frame=NS(skin_patches=[patch]),geometry=geometry)
    G,c,error,w,saved=groups(local,sample,np.array([.2,0.,0.,0.,0.]),selection)
    check('geometric_R_maps_robot25_to_normal_meters',np.array_equal(G[0],row@selection.response25x8))
    check('geometric_free_is_qfree_minus_qnow',abs(c[0]-float(row@(selection.free25-sample.q25)))<1e-15)
    check('empirical_error_units_rad_and_m_to_meter_without_length_rescaling',abs(error[0]-float(abs(row)@selection.error25))<1e-15)
    captured=[]
    def fk(future,p):
        captured.append(future)
        return NS(point_world=np.array([float(row@(future.q25-sample.q25)),0.,0.]))
    local.geometry=NS(point_kinematics=fk)
    qfuture=selection.free25+selection.response25x8@np.zeros(8)
    result=exact(local,sample,qfuture,saved)
    check('finiteFK_receives_true25_axis_order',np.array_equal(captured[0].q25,qfuture))
    check('finiteFK_future_robot_clock_plus20ms',captured[0].time_s==sample.time_s+.02)
    check('finiteFK_uses_acquired_world_normal_and_current_point_difference',abs(result[0]-float(row@(qfuture-sample.q25)))<1e-15)
    # Real frozen baseline method demonstrates the hypothetical-intent vs
    # actual future execution distinction; this is a diagnostic, not a failure.
    vendor=ROOT.parent/'五指策略接入_2026-11-03/vendor/frozen_controllers.py'
    clip=method(vendor,'CommonController','_clip_first',dict(np=np))
    cs=NS(alpha=np.array([.996,.5,.5,.5,.5]),closure_min=np.zeros(5),closure_max=np.ones(5),valid=np.ones(5,bool),closure_permission=np.ones(5,bool))
    old=NS(g=NS(decision_period=.02,open_rate=.5,close_rate=.3),state=cs)
    action=np.array([.004,0.,0.,0.,0.]);executed=[]
    for _ in range(4):
        d=clip(old,action)*.25;executed.append(d.copy());cs.alpha+=d
    sum_delta=np.sum(executed,axis=0)
    check('baseline_5ms_boundary_reclip_is_not_uniform_intent_ramp',abs(sum_delta[0]-.002734375)<1e-14 and abs(sum_delta[0]-action[0])>.001)
    after={str(p):sha(p) for p in (cp,rp,gp,bp)}
    record=dict(schema='v10_bank_geometry_consumer_readonly_v1',passed=True,new_physics=False,checks=checks,
        sources_reviewed=sources,sources_after=after,source_unchanged_during_pure_checks=sources==after,
        nominal_comparison_counterexample=dict(interval_intent8=action.tolist(),actual_5ms_increment_sequence=np.asarray(executed).tolist(),actual20ms_total=sum_delta.tolist(),uniform_intent_total=action.tolist(),
            meaning='intent uniform-ramp is hypothetical and differs from future retained-baseline commands; no actual closed-loop counterfactual asserted'),
        limits=['geometric error projection is first-order empirical, not hard bound','one anchor/point/component/L1 support is not a certified mixed trajectory','exact FK is exact predicted robot geometry only, not true future geometry or external motion','only public current robot encoders/velocity and delayed acquisition skin fields used in fixtures'])
    with (out/'result.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();r=verify(args.out)
    print(json.dumps({key:value for key,value in r.items() if key not in ('sources_after','sources_reviewed')},ensure_ascii=False,indent=2))

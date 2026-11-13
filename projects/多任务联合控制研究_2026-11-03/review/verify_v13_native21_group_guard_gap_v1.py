"""Independent private-model mode-guard counterexample; zero dynamics."""
from pathlib import Path
from copy import copy
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    sys.path.insert(0,str(ROOT));import research_imports,mujoco,hand_plant
    from foundation_config import FoundationConfig
    from v13_joint.native_effort21 import NativeEffort21
    from v13_a1.codec import Reference21
    from v11.robot_servo_effort import RobotServoEffort
    source=ROOT/'v13_joint/native_effort21.py';expected='6f097f88b19a80b8692be6d8a937c7a640d2ad6fe17a12d2f4d87b48c7e148f3'
    if sha(source)!=expected:raise ValueError('Specific6f target changed; keep this historical counterexample separate')
    out=ROOT/'review/v13_native21_group_guard_gap_independent_v1.json'
    if out.exists():raise FileExistsError('Fresh counterexample evidence required')
    guarded={};counts={}
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_inverse','mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation','mj_kinematics','mj_comPos','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):
            guarded[name]=getattr(mujoco,name);counts[name]=0
            def forbidden(*a,_name=name,**k):counts[_name]+=1;raise AssertionError('No physical pipeline: '+_name)
            setattr(mujoco,name,forbidden)
    constructor=hand_plant.HandPlant.__init__
    hand_plant.HandPlant.__init__=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No actual plant construction'))
    try:
        original,_=hand_plant._model(FoundationConfig());fixture=copy(original)
        fixture.opt.disableactuator=1<<5
        core=NativeEffort21(fixture);q=fixture.qpos0[core.servo.qadr].copy();v=np.zeros(25)
        values=np.zeros(21);values[0]=.5;values[18]=.1;ref=Reference21(values,1.)
        domain=core.build_domain(q25=q,v25=v,robot_axes25=core.robot_axes25,time_s=1.,current_reference=ref,
            nominal_reference=ref,actual_engine=core.actor.encode(ref),reference_lower21=values-.001,reference_upper21=values+.001)
        before=core._model_signature();axis=int(core.servo.aids[0]);old_group=int(core.servo.model.actuator_group[axis])
        core.servo.model.actuator_group[axis]=5;after=core._model_signature()
        fresh=RobotServoEffort(core.servo.model)
        result=core.solve(domain,p0_effort21=domain.p0_effort21,delta_effort21=np.zeros(21),correction25=np.zeros(25),residual_scale25=np.ones(25),delta_scale21=np.ones(21))
        reproduced=bool(domain.available and before==after and result.available and 'robot_actuator_group_disabled' in fresh.unavailable)
        value=dict(schema='v13_native21_group_signature_gap_v1',source_sha256=expected,checker_sha256=sha(__file__),
            reproduced=reproduced,full_model_mutation_gate_passed=not reproduced,
            initial_disabled_group_mask=32,changed_robot_actuator='rh_A_THJ5',initial_group=old_group,changed_group=5,
            original_compiled_model_disable_mask=int(original.opt.disableactuator),
            signature_before=before,signature_after=after,initial_domain_available=domain.available,
            stale_bound_core_status=result.status,stale_bound_core_available=result.available,
            stale_bound_core_TH_effort=None if result.native_effort21 is None else float(result.native_effort21[0]),
            fresh_frozen_F067_unavailable=fresh.unavailable,physics_steps=0,forbidden_calls=counts,
            standard_immutable_model_scope_not_rejected=True,
            note='Does not invalidate8020 nominal-frozen-model checks. Missing actuator_group allows a relevant mode mutation to evade a claimed prepared-domain model-change guard; no physical simulation or pressure claim.')
        if sha(source)!=expected:raise ValueError('6f changed during counterexample')
        with out.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        print(json.dumps(dict(reproduced=reproduced,physics_steps=0,out=str(out),sha256=sha(out)),ensure_ascii=False))
    finally:
        hand_plant.HandPlant.__init__=constructor
        for name,function in guarded.items():setattr(mujoco,name,function)
if __name__=='__main__':main()

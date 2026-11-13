"""Author algebra/known-model check only; never an integration experiment."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import research_imports
from hand_plant import _model
from foundation_config import FoundationConfig
from v18_effort_transfer.governor import LiftProgressGovernor
from v13_a1.codec import ReferenceActor21,Reference21
from v11.robot_servo_effort import RobotServoEffort


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError('fresh output required')
    checks=[]
    def check(name,condition,detail=None):
        checks.append(dict(name=name,passed=bool(condition),detail=detail))
    m=_model(FoundationConfig())[0]
    forbidden=('mj_step','mj_forward','mj_kinematics','mj_comPos','mj_collision','mj_contactForce')
    def fail(*a,**kw):raise RuntimeError('forbidden physics/FK/contact pipeline')
    patches=[patch('mujoco.'+n,side_effect=fail) for n in forbidden]
    for p in patches:p.start()
    try:
        g=LiftProgressGovernor(m)
        common=dict(time_s=12.02,q_m=.08,v_m_s=0.,current_reference_m=.08,
                    reference_lower_m=0.,reference_upper_m=.1,
                    reference_rate_lower_m_s=-.18,reference_rate_upper_m_s=.18,
                    requested_velocity_m_s=-.18,dt_s=.005,deadline_s=15.5)
        def solve(**kw):return g.step(**dict(common,**kw))
        x=solve()
        check('known lift kp/kv/passive damping and tau',g.kp==1500. and g.kv==140. and g.passive_damping==25. and g.tau_s==.11)
        check('search command bounded by actual one-tick rate',x.available and -.00090000000001<=x.increment_m<=0)
        check('20ms budget derives 3.6mm and velocity intention',x.diagnostics['recovery_reference_budget_m']==.0036 and
              abs(x.diagnostics['governed_velocity_intent_m_s']+.0036/.11)<1e-15)
        check('requested descent is preserved as request, not relabelled actual speed',
              x.diagnostics['requested_velocity_m_s']==-.18 and x.diagnostics['actual_velocity_m_s']==0.)
        debt=solve(q_m=.08,current_reference_m=.061,v_m_s=-.16)
        check('19mm old debt cannot be frozen and called braking',debt.status=='BRAKE_RECOVER' and debt.increment_m>0 and
              debt.diagnostics['remaining_brake_reference_debt_m']>.018)
        check('old debt not claimed recoverable in 20ms',not debt.diagnostics['current_q_brake_reference_reachable_within_budget'])
        overspeed=solve(v_m_s=-.16)
        check('actual descent alone requests recovery even at zero q-u debt',overspeed.diagnostics['measured_descent_exceeds_envelope'] and overspeed.reference_m==.08)
        outward=solve(q_m=.08,current_reference_m=.061,v_m_s=-.16,requested_velocity_m_s=.18)
        check('positive unloading intention survives prior descent recovery',outward.increment_m>0 and outward.diagnostics['governed_velocity_intent_m_s']>0)
        explicit=solve(requested_velocity_m_s=.18,brake_requested=True)
        check('explicit brake has strict current-q target',explicit.diagnostics['governed_velocity_intent_m_s']==0 and explicit.reference_m==.08)
        upper=solve(current_reference_m=.099,q_m=.08,v_m_s=.16,requested_velocity_m_s=.18)
        check('upward debt and actual ascent recover symmetrically',upper.increment_m<0 and upper.diagnostics['measured_ascent_exceeds_envelope'])
        reverse=solve(current_reference_m=.099,q_m=.08,v_m_s=.16,requested_velocity_m_s=-.18)
        check('negative recovery intention allowed during excessive ascent',reverse.diagnostics['governed_velocity_intent_m_s']<0)
        fixed=solve(current_reference_m=.061,q_m=.08,v_m_s=-.16,reference_lower_m=.061,reference_upper_m=.061,
                    reference_rate_lower_m_s=0.,reference_rate_upper_m_s=0.)
        check('fixed authority is legal but recovery remains unavailable',fixed.available and fixed.reference_m==.061 and
              fixed.diagnostics['minimum_reference_return_time_at_current_q_s'] is None and
              not fixed.diagnostics['current_q_brake_reference_reachable_within_budget'])
        side=solve(reference_rate_upper_m_s=0.)
        check('no upward return permission removes new downward debt',side.reference_m==.08 and side.diagnostics['governed_velocity_intent_m_s']==0.)
        absolute=solve(reference_lower_m=.0798)
        check('absolute authority is not widened by velocity task',absolute.reference_m>=.0798)
        empty=solve(reference_lower_m=.09,reference_upper_m=.1)
        check('empty actual intersection unavailable, not inverse then clip',not empty.available and empty.reference_m is None)
        late=solve(time_s=15.5)
        check('deadline requests supervisor exit and current-q brake',late.diagnostics['supervisor_exit_required'] and late.reference_m==.08)
        repeated=solve()
        check('same public inputs do not accumulate another hidden step',x==repeated)
        for key,value in (('v_m_s',None),('q_m',True),('current_reference_m','0.08'),('reference_rate_upper_m_s',np.bool_(False))):
            bad=solve(**{key:value});check('strict public numeric '+key,not bad.available and bad.reference_m is None)
        check('20ms planning interval cannot masquerade as5ms execution',not solve(dt_s=.02).available)
        qadr=np.asarray(m.jnt_qposadr[[m.joint(n).id for n in
            tuple(m.joint(j).name for j in range(m.njnt) if (m.joint(j).name or '').startswith('rh_'))+('transport_lift',)]],int)
        servo=RobotServoEffort(m);actor=ReferenceActor21(m)
        for idx,(qv,vv,uu) in enumerate(((.08,0.,.08),(.08,-.16,.061),(.08,.16,.099),(.0,-.5,.1))):
            result=solve(q_m=qv,v_m_s=vv,current_reference_m=uu)
            q25=np.asarray(m.qpos0[qadr]).copy();q25[24]=qv
            v25=np.zeros(25);v25[24]=vv
            ref=np.zeros(21);ref[18]=result.reference_m
            oracle=servo.evaluate(q25,v25,actor.encode(Reference21(ref,12.02)).values,time_s=12.02)
            got=result.diagnostics['current_native_effort_of_emitted_reference']['effort_N']
            check('full F067 native effort value same bits '+str(idx),np.float64(got).tobytes()==oracle.predicted_effort21[18].tobytes())
        a=m.actuator('lift_motor').id;old=int(m.actuator_group[a]);m.actuator_group[a]=5
        check('bound mode/group mutation invalidates proposal',solve().status=='UNAVAILABLE_MODEL_CHANGED')
        m.actuator_group[a]=old
        check('no pressure/safety/future stopping certificate',all(not x.diagnostics[k] for k in
              ('contact_force_prediction_available','physics_safety_certified','future_robot_response_predicted','stopping_time_or_distance_certified')))
        json.dumps([x.__dict__,debt.__dict__,fixed.__dict__],allow_nan=False)
        check('unknown uses null and strict JSON remains finite',True)
    finally:
        for p in reversed(patches):p.stop()
    sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
             (ROOT/'v18_effort_transfer/governor.py',ROOT/'v18_effort_transfer/design.py',Path(__file__))}
    payload=dict(schema='v18_progress_governor_author_algebra_v1',passed=all(c['passed'] for c in checks),
        checks=checks,check_count=len(checks),source_sha256=sources,
        scope=dict(model_compile_only=True,physics_steps=0,world_FK_calls=0,contact_reads=0,
                   F067_fixed_transmission_algebra_allowed=True,no_empirical_force_or_future_response_claim=True),
        failed_checks=[c['name'] for c in checks if not c['passed']])
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=payload['passed'],checks=len(checks),failed=payload['failed_checks'],out=str(out),
                         sha256=hashlib.sha256(out.read_bytes()).hexdigest()),ensure_ascii=False))
    return 0 if payload['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

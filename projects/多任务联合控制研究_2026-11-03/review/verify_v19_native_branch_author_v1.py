"""Known-model author QA for native branches only; no plant/integration."""
from pathlib import Path
from copy import copy
from dataclasses import replace
from unittest.mock import patch
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
from hand_plant import _model
from foundation_config import FoundationConfig
from v19_contact_reference.native_branch import NativeBranchBuilder21,MAX_BOUNDARY_BISECTIONS
from v13_a1.codec import Reference21,ReferenceActor21
from v11.robot_servo_effort import RobotServoEffort
import mujoco

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    source=ROOT/'v19_contact_reference/native_branch.py';before=hashlib.sha256(source.read_bytes()).hexdigest()
    model=_model(FoundationConfig())[0];checks=[];cases=[]
    def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def fail(*a,**k):raise RuntimeError('forbidden world/physics/writer call')
    guards=[patch('mujoco.'+n,side_effect=fail) for n in ('mj_step','mj_forward','mj_kinematics','mj_comPos','mj_collision','mj_contactForce')]
    guards.append(patch.object(ReferenceActor21,'apply',side_effect=fail))
    for g in guards:g.start()
    try:
        def setup(gain=2.,bias0=0.,forcelimited=True):
            m=copy(model);a=int(m.actuator('rh_A_THJ5').id);c=int(m.actuator_ctrladr[a]);o=int(m.actuator_outadr[a])
            m.actuator_gainprm[a,:]=0.;m.actuator_gainprm[a,0]=gain;m.actuator_gaintype[a]=mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_biasprm[a,:]=0.;m.actuator_biasprm[a,0]=bias0;m.actuator_biastype[a]=mujoco.mjtBias.mjBIAS_AFFINE
            m.actuator_ctrllimited[c]=True;m.actuator_ctrlrange[c]=[-1.,1.]
            m.actuator_forcelimited[o]=forcelimited;m.actuator_forcerange[o]=[-.2,.2]
            builder=NativeBranchBuilder21(m);q=m.qpos0[builder.servo.qadr].copy();v=np.zeros(25);q[-1]=.08
            return builder,q,v
        def build(builder,q,v,cur,pref,lo=-.5,hi=.5):
            u=np.zeros(21);u[0]=cur;u[18]=.08;p=u.copy();p[0]=pref
            lower=u.copy();upper=u.copy();lower[0]=lo;upper[0]=hi
            current=Reference21(u,12.02);preferred=Reference21(p,12.02)
            return builder.build(q25=q,v25=v,robot_axes25=builder.robot_axes25,time_s=12.02,current_reference=current,
                actual_engine=builder.actor.encode(current),preferred_reference=preferred,reference_lower21=lower,reference_upper21=upper)
        def verify_samples(name,builder,q,v,branch):
            ck(name+'/available',branch.available,branch.reason)
            if not branch.available:return
            oracle=RobotServoEffort(builder.servo.model);errors=[]
            for scalar in (branch.reference_lower21[0],branch.anchor_reference21[0],branch.reference_upper21[0],
                           (branch.reference_lower21[0]+branch.reference_upper21[0])/2.):
                u=branch.anchor_reference21.copy();u[0]=scalar;ref=Reference21(u,12.02)
                result=builder.evaluate_reference(branch,ref)
                full=oracle.evaluate(q,v,builder.actor.encode(ref).values,time_s=12.02)
                good=(result['available'] and result['native_effort21'].tobytes()==full.predicted_effort21.tobytes() and
                      result['actual_delta_effort21'].tobytes()==(full.predicted_effort21-branch.p0_effort21).tobytes())
                ck(name+'/fullnative sample '+repr(float(scalar)),good)
                if result['available']:errors.append(float(np.max(abs(result['affine_error21']))))
            ck(name+'/affine model error retained finite',all(np.isfinite(errors)) and max(errors,default=0)<1e-12,errors)
            ck(name+'/exact absolute bounds not reconstructed from rounded delta',
               branch.absolute_reference_lower21.tobytes()==branch.reference_lower21.tobytes() and
               branch.absolute_reference_upper21.tobytes()==branch.reference_upper21.tobytes())
            ck(name+'/fixed scalar bisection budget',max(max(x) for x in branch.diagnostics['scalar_boundary_bisection_counts21'])<=MAX_BOUNDARY_BISECTIONS)
            cases.append(dict(name=name,kind=branch.branch_kind21[0],gain=float(branch.gain21[0]),offset=float(branch.offset21[0]),
                p0=float(branch.p0_effort21[0]),current=float(branch.current_reference21[0]),preferred=float(branch.preferred_reference21[0]),
                branch_reference=[float(branch.reference_lower21[0]),float(branch.reference_upper21[0])],
                construction_ms=branch.diagnostics['construction_ms'],full_calls=branch.diagnostics['native_full_evaluations_delta'],
                predicate_calls=branch.diagnostics['scalar_raw_predicate_evaluations'],max_affine_error=max(errors,default=0)))
        b,q,v=setup();active=build(b,q,v,0.,.4);verify_samples('positive active',b,q,v,active)
        ck('current branch gain and original p0',active.gain21[0]==2. and active.offset21[0]==0. and active.p0_effort21[0]==0.)
        ck('preferred can lie beyond selected kink but is not silently retargeted',active.preferred_reference21[0]==.4 and
           not active.diagnostics['preferred_inside_selected_branch21'][0] and active.reference_upper21[0]<=.1)
        plateau=build(b,q,v,.3,.4);verify_samples('known force platform',b,q,v,plateau)
        ck('platform zero slope leaves reference free',plateau.gain21[0]==0. and plateau.reference_upper21[0]>plateau.reference_lower21[0] and
           plateau.diagnostics['known_zero_slope_but_reference_not_fixed21'][0])
        right=build(b,q,v,.1,.2);left=build(b,q,v,.1,0.)
        ck('exact upper-force kink caller-positive chooses plateau',right.available and right.gain21[0]==0. and right.reference_lower21[0]>=.1)
        ck('same kink caller-negative chooses affine one-sided',left.available and left.gain21[0]==2. and left.reference_upper21[0]<=.1)
        verify_samples('kink affine side',b,q,v,left)
        neg,nq,nv=setup(-2.);negative=build(neg,nq,nv,0.,.4);verify_samples('negative known gain',neg,nq,nv,negative)
        ck('negative affine gain not absolute-valued',negative.gain21[0]==-2.)
        zero,zq,zv=setup(0.,.07);constant=build(zero,zq,zv,0.,.4);verify_samples('zero known gain',zero,zq,zv,constant)
        ck('known zero gain is not an unknown/missing model',constant.available and constant.gain21[0]==0. and
           constant.reference_lower21[0]==-.5 and constant.reference_upper21[0]==.5)
        outside=build(b,q,v,.3,0.,-.08,.08);verify_samples('current outside TASK, active offset',b,q,v,outside)
        ck('nonzero offset preserves raw current p0 instead of anchor retarget',outside.p0_effort21[0]==.2 and
           outside.anchor_effort21[0]==0. and abs(outside.offset21[0]-.4)<1e-15 and
           not outside.diagnostics['current_inside_actual_domain21'][0])
        fixed=build(b,q,v,.3,0.,0.,0.);verify_samples('TASK debt-repayment singleton',b,q,v,fixed)
        ck('singleton has explicit constant offset and legal nonzero du',fixed.gain21[0]==0. and fixed.offset21[0]==-.2 and
           fixed.delta_lower21[0]==-.3 and fixed.delta_upper21[0]==-.3)
        rawclip=build(b,q,v,1.5,0.,-.08,.08);verify_samples('rawcurrent controlclip fixed p0',b,q,v,rawclip)
        ck('rawcurrent clip is admitted without reanchoring p0',rawclip.p0_effort21[0]==.2 and rawclip.anchor_effort21[0]==0. and rawclip.available)
        invalid=build(b,q,v,0.,.6);ck('illegal preferred rejected rather than clipped',not invalid.available and invalid.status=='UNAVAILABLE_PREFERRED_OUTSIDE_DOMAIN')
        ck('read-only gain/domain arrays',not active.gain21.flags.writeable and not active.reference_lower21.flags.writeable and not active.p0_effort21.flags.writeable)
        ck('foreign-owner branch rejected',not neg.evaluate_reference(active,Reference21(active.anchor_reference21,12.02))['available'])
        ck('wrong-clock final reference rejected',not b.evaluate_reference(active,Reference21(active.anchor_reference21,12.025))['available'])
        cross=active.anchor_reference21.copy();cross[0]=np.nextafter(active.reference_upper21[0],np.inf)
        ck('one ULP outside chosen branch rejected',not b.evaluate_reference(active,Reference21(cross,12.02))['available'])
        mutated=active.gain21.copy();mutated[0]=3.;mutated.setflags(write=False)
        tampered=replace(active,gain21=mutated)
        ck('modified descriptor rejected',not b.evaluate_reference(tampered,Reference21(active.anchor_reference21,12.02))['available'])
        aid=int(b.servo.model.actuator('rh_A_THJ5').id);b.servo.model.actuator_group[aid]=5
        ck('model group/mode fingerprint rejects stale branch',not b.evaluate_reference(active,Reference21(active.anchor_reference21,12.02))['available'])
        original=NativeBranchBuilder21(model);q0=model.qpos0[original.servo.qadr].copy();q0[24]=.08;v0=np.zeros(25)
        std=build(original,q0,v0,0.,.001,0.,.005);verify_samples('unmodified known robot native law',original,q0,v0,std)
        ck('no pressure/future/safety/inverse/writer claim',all(not std.diagnostics[k] for k in ('pressure_prediction_available','future_robot_response_available','physical_safety_certified','writer_called','inverse_called')))
        json.dumps(std.to_dict(),allow_nan=False);ck('strict finite descriptor JSON',True)
    finally:
        for g in reversed(guards):g.stop()
    after=hashlib.sha256(source.read_bytes()).hexdigest()
    payload=dict(schema='v19_native_branch_author_v1',passed=all(x['passed'] for x in checks) and before==after,
        check_count=len(checks),checks=checks,cases=cases,source_sha256=after,reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physics_steps=0,forward_calls=0,world_FK_calls=0,contact_reads=0,actor_writes=0,
        model_compile_only=True,unmodified_F067_value_oracle=True,
        scope='author finite current-state native branch algebra, no control or contact-force performance proof')
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=payload['passed'],checks=len(checks),failures=[x['name'] for x in checks if not x['passed']],
                         out=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source=after),ensure_ascii=False))
    return 0 if payload['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

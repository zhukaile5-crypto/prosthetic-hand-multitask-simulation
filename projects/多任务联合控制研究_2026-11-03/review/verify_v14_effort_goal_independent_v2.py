"""Independent small effort-goal realization oracles; no dynamics/FK/writers."""
from pathlib import Path
from copy import copy
from dataclasses import replace
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/v14_effort_goal_independent_v2.json'
PIN='7450ea567362d5675d9ca8228bccd7c79b8362d25541ca14d1abc86c80589a78'

def main():
    names=('v14_coord/effort_realization.py','v13_joint/prepared_value_effort21.py','v13_joint/prepared_native_effort21.py','v13_joint/native_effort21.py','v11/robot_servo_effort.py','v13_a1/codec.py')
    saved={n:(ROOT/n).read_bytes() for n in names};assert hashlib.sha256(saved[names[0]]).hexdigest()==PIN
    sys.path.insert(0,str(ROOT));import research_imports,mujoco
    guards={};calls={}
    for n in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_fwdPosition','mj_fwdVelocity','mj_kinematics','mj_comPos','mj_camlight','mj_collision','mj_contactForce'):
        if hasattr(mujoco,n):
            guards[n]=getattr(mujoco,n);calls[n]=0
            def reject(*a,_n=n,**k):calls[_n]+=1;raise AssertionError('forbidden '+_n)
            setattr(mujoco,n,reject)
    import hand_plant
    from foundation_config import FoundationConfig
    from v13_a1.codec import Reference21,ReferenceActor21
    from v11.robot_servo_effort import RobotServoEffort
    from v14_coord.effort_realization import EffortGoalRealizer21
    old_init=hand_plant.HandPlant.__init__;old_apply=ReferenceActor21.apply;plants=[];writers=[]
    def no_plant(*a,**k):plants.append(True);raise AssertionError('plant constructor forbidden')
    def no_writer(*a,**k):writers.append(True);raise AssertionError('writer forbidden')
    hand_plant.HandPlant.__init__=no_plant;ReferenceActor21.apply=no_writer
    checks=[];cases={}
    def check(n,v,detail=None):checks.append(dict(name=n,passed=bool(v),detail=detail))
    try:
        model,xml_sha=hand_plant._model(FoundationConfig());probe=EffortGoalRealizer21(model);axis=6;aid=int(probe.servo.aids[axis]);oid=int(probe.servo.outadr[axis])
        q=model.qpos0[probe.servo.qadr].copy();v=np.zeros(25)
        def setup(m,current=.12,nominal=.1,free_wrist=False):
            c=EffortGoalRealizer21(m);cur=np.zeros(21);cur[18]=.03;cur[axis]=current;nom=cur.copy();nom[axis]=nominal
            lo=cur.copy();hi=cur.copy();lo[axis]=0.;hi[axis]=.2
            if free_wrist:lo[19]=-.03;hi[19]=.03
            cr=Reference21(cur,4.25);nr=Reference21(nom,4.25)
            d=c.build_domain(q25=q,v25=v,robot_axes25=c.robot_axes25,time_s=4.25,current_reference=cr,nominal_reference=nr,actual_engine=c.actor.encode(cr),reference_lower21=lo,reference_upper21=hi)
            assert d.available,d.reason
            return c,d
        def solve(c,d,delta,correction=None,sr=None,sd=None):
            return c.solve(d,p0_effort21=d.p0_effort21,delta_effort21=delta,correction25=np.zeros(25) if correction is None else correction,residual_scale25=np.ones(25) if sr is None else sr,delta_scale21=np.ones(21) if sd is None else sd)
        def full_check(m,c,d,result,correction,sr,sd):
            full=RobotServoEffort(m).evaluate(q,v,result.engine_port21.values,time_s=d.time_s)
            actual=full.predicted_effort21-d.p0_effort21;residual=full.transmission21x25.T@actual-correction;objective=float((sr*residual)@(sr*residual)+(sd*actual)@(sd*actual))
            return full,all((full.predicted_effort21.tobytes()==result.native_effort21.tobytes(),actual.tobytes()==result.actual_delta_effort21.tobytes(),residual.tobytes()==result.residual25.tobytes(),np.float64(objective).tobytes()==np.float64(result.objective).tobytes()))
        #1 Negative gain, multi-axis composition and selection independence from correction/weights.
        negative=copy(model);negative.actuator_gainprm[aid,:3]=[-.75,0.,0.];negative.actuator_biasprm[aid,:3]=0.
        c,d=setup(negative,free_wrist=True);delta=np.zeros(21);delta[axis]=.015;delta[19]=.00001
        correction=d.transmission21x25.T@delta;sr=np.linspace(.7,1.3,25);sd=np.linspace(.5,1.5,21)
        a=solve(c,d,delta,correction,sr,sd);opposed=solve(c,d,delta,-100.*correction,np.full(25,17.),np.full(21,1e4))
        check('negative_gain_known_and_goal_realized',a.available and abs(a.reference21.values[axis]-.08)<1e-14 and np.max(abs(a.actual_delta_effort21-delta))<1e-12)
        check('correction_and_regularizer_cannot_select_second_goal',opposed.available and a.reference21.values.tobytes()==opposed.reference21.values.tobytes() and a.objective!=opposed.objective)
        check('original_p0_and_accepted_delta_preserved',a.p0_effort21.tobytes()==d.p0_effort21.tobytes() and a.requested_delta_effort21.tobytes()==delta.tobytes())
        full,okay=full_check(negative,c,d,a,correction,sr,sd)
        check('combined_multi_axis_native_delta_residual_objective_bytes',okay and a.diagnostics['prepared_value_final_full_recheck']['passed'])
        check('combined_derivative_status_is_full_f067',a.diagnostics['native_derivative_status']==full.derivative_status and 'NOT_COMPUTED_VALUE_ONLY' not in full.derivative_status.values())
        check('nine_fixed_vector_families_no_global_optimum_claim',len(a.diagnostics['finite_candidate_families'])==9 and a.diagnostics['finite_family_only'] and not a.diagnostics['all_float_global_optimum_certified'])
        cases['negative_gain']=dict(reference=float(a.reference21.values[axis]),goal_delta=float(delta[axis]),actual_delta=float(a.actual_delta_effort21[axis]),opposed_objective=opposed.objective,original_objective=a.objective)
        #2 A known saturated platform has many equal effort solutions: choose current before nominal.
        clipped=copy(model);clipped.actuator_gainprm[aid,:3]=[1.,0.,0.];clipped.actuator_biasprm[aid,:3]=0.;clipped.actuator_forcelimited[oid]=True;clipped.actuator_forcerange[oid]=[-.02,.02]
        cc,dc=setup(clipped,current=.15,nominal=.18);zero=np.zeros(21);plateau=solve(cc,dc,zero)
        check('force_clipped_platform_selects_exact_current_not_nominal',plateau.available and plateau.reference21.values[axis]==.15 and plateau.diagnostics['axis_selections'][axis]['family']=='current')
        check('platform_unreachable_delta_reported_not_claimed_realized',abs(solve(cc,dc,np.eye(21)[axis]*.005).diagnostics['actual_effort_minus_requested_delta21'][axis]+.005)<1e-15)
        ck,dk=setup(clipped,current=.02,nominal=.18);kink=solve(ck,dk,zero)
        check('known_force_kink_values_available_derivatives_unknown',kink.available and kink.reference21.values[axis]==.02 and kink.diagnostics['native_derivative_status']['reference_jacobian']=='UNAVAILABLE_CLIP_KINK' and kink.diagnostics['native_derivative_status']['actuation_stiffness']=='UNAVAILABLE_FORCE_CLIP_KINK')
        cases['platform']=dict(selected=plateau.diagnostics['axis_selections'][axis]['family'],reference=float(plateau.reference21.values[axis]),nominal=.18,kink_status=kink.diagnostics['native_derivative_status'])
        #3 Small delta swallowed by huge absolute p0 must still remain an explicit error.
        huge=copy(model);huge.actuator_gainprm[aid,:3]=0.;huge.actuator_biasprm[aid,:3]=[float(2**54),0.,0.];huge.actuator_forcelimited[oid]=False
        ch,dh=setup(huge,current=.12,nominal=.16);tiny=np.zeros(21);tiny[axis]=.25;h=solve(ch,dh,tiny,dh.transmission21x25.T@tiny)
        check('huge_p0_addition_really_swallows_delta',dh.p0_effort21[axis]+tiny[axis]==dh.p0_effort21[axis])
        check('swallowed_delta_not_forged_as_zero_error',h.available and h.actual_delta_effort21[axis]==0. and h.diagnostics['absolute_addition_lost_delta21'][axis]==.25 and h.diagnostics['actual_effort_minus_requested_delta21'][axis]==-.25 and h.diagnostics['scaled_realization_linf']==.25)
        expected_huge_objective=.25**2*float(dh.transmission21x25[axis]@dh.transmission21x25[axis])
        check('flat_known_gain_uses_current_and_residual_honest',h.reference21.values[axis]==.12 and np.max(abs(h.residual25))==.25 and h.objective==expected_huge_objective,dict(actual_axis=ch.servo.aids[axis].item(),transmission_row=dh.transmission21x25[axis].tolist(),expected_objective=expected_huge_objective))
        cases['huge_p0']=dict(p0=float(dh.p0_effort21[axis]),delta=.25,actual_delta=float(h.actual_delta_effort21[axis]),error=h.diagnostics['actual_effort_minus_requested_delta21'][axis],objective=h.objective)
        #4 Owner/domain/model failures are unavailable, not fabricated zero effort.
        foreign=EffortGoalRealizer21(negative);bad=solve(foreign,d,delta)
        check('foreign_domain_rejected_without_command',not bad.available and bad.status=='UNAVAILABLE_SOURCE' and bad.native_effort21 is None)
        altered=d.p0_effort21.copy();altered[axis]+=1.;tampered=replace(d,p0_effort21=altered)
        bad=solve(c,tampered,delta);check('tampered_domain_seal_rejected',not bad.available and bad.status=='UNAVAILABLE_SOURCE')
        changed,dd=setup(negative);changed.servo.model.actuator_gainprm[aid,0]*=1.1;bad=solve(changed,dd,zero)
        check('private_bound_model_mutation_rejected',not bad.available and bad.status=='UNAVAILABLE_MODEL' and bad.native_effort21 is None)
        coupled=c.build_domain(q25=q,v25=v,robot_axes25=c.robot_axes25,time_s=4.25,current_reference=Reference21(d._context['current'],4.25),nominal_reference=Reference21(d._context['nominal'],4.25),actual_engine=c.actor.encode(Reference21(d._context['current'],4.25)),reference_lower21=d.reference_lower21,reference_upper21=d.reference_upper21,coupled_reference_constraints={'sum':1.})
        check('hard_coupled_domain_not_box_enveloped',not coupled.available and coupled.status=='UNAVAILABLE_COUPLED_DOMAIN')
        #5 Deliberately corrupt intermediate candidate values: real full F067 combination must reject.
        corrupt,dd=setup(model);corrupt.servo._value_data[4][axis,0]+=.125
        bad=solve(corrupt,dd,np.eye(21)[axis]*.01)
        check('full_combination_recheck_rejects_candidate_cache_corruption',not bad.available and bad.status=='NUMERICAL_FAILURE' and bad.reason=='combined_goal_realization_full_recheck_failed' and bad.diagnostics['native_bytes_equal'] is False)
        # The newly declared absolute-goal overflow is an explicit numerical failure.
        overflow=copy(model);overflow.actuator_gainprm[aid,:3]=0.;overflow.actuator_biasprm[aid,:3]=[1e308,0.,0.];overflow.actuator_forcelimited[oid]=False
        co,do=setup(overflow);large=np.zeros(21);large[axis]=1e308;bad=solve(co,do,large)
        check('overflowed_absolute_goal_rejected_without_false_value',not bad.available and bad.status=='NUMERICAL_FAILURE' and bad.native_effort21 is None and bad.requested_delta_effort21.tobytes()==large.tobytes())
    finally:
        hand_plant.HandPlant.__init__=old_init;ReferenceActor21.apply=old_apply
        for n,fn in guards.items():setattr(mujoco,n,fn)
    stable=all((ROOT/n).read_bytes()==v for n,v in saved.items())
    check('production_source_bytes_unchanged',stable)
    check('no_dynamics_FK_contact_plant_or_writer_calls',not any(calls.values()) and not plants and not writers)
    report=dict(schema='v14_effort_goal_independent_v2',role='independent_small_goal_mapping_oracles_not_author_QA_replay',passed=all(x['passed'] for x in checks),count=len(checks),checks=checks,cases=cases,
                source_sha256={n:hashlib.sha256(v).hexdigest() for n,v in saved.items()},checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_stable=stable,
                static_model_compile_count=1,static_xml_sha256=xml_sha,physics_steps=0,plant_calls=len(plants),writer_calls=len(writers),forbidden_calls=calls,
                previous_fixture_error=dict(report='review/v14_effort_goal_independent_v1.json',sha256=hashlib.sha256((ROOT/'review/v14_effort_goal_independent_v1.json').read_bytes()).hexdigest(),passed=19,total=20,reason='The first test expected a two-joint J0 row for actuator index6; actual index6 is single-joint FFJ3. Correct closed form uses the actual native transmission row squared norm. Production source unchanged.'),
                limitations=['Finite9-vector family only, not an all-float inverse/global optimum proof; actual goal error and diagnostic residual remain authoritative.',
                  'Model computed actuator effort is not observed contact force, future pressure or a safety certificate.',
                  'No V14QP kernel, runtime, actuator writer, physical trajectory, broad public-packet performance or all-source closure audit.'])
    assert not OUT.exists();OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),passed=report['passed'],checks=len(checks),failed=[x['name'] for x in checks if not x['passed']]),ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()

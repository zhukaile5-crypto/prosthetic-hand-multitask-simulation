"""Actual-model zero-step author checks; integration/FK/contact calls forbidden."""
from pathlib import Path
from copy import copy
from dataclasses import replace
import hashlib,json,sys,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1];FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(FOUNDATION));sys.dont_write_bytecode=True


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
    import research_imports,mujoco
    forbidden=[]
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_fwdPosition','mj_fwdVelocity','mj_kinematics','mj_comPos','mj_camlight','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):
            def deny(*a,_name=name,**k):forbidden.append(_name);raise AssertionError('forbidden '+_name)
            setattr(mujoco,name,deny)
    from foundation_config import FoundationConfig
    import hand_plant
    hand_plant.HandPlant.__init__=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No plant construction'))
    from v13_a1.codec import Reference21,EnginePort21
    from v13_joint.native_effort21 import NativeEffort21,NEXTAFTER_PER_SIDE
    model,_=hand_plant._model(FoundationConfig());original_gain=model.actuator_gainprm.copy()
    core=NativeEffort21(model);q=model.qpos0[core.servo.qadr].copy();v=np.zeros(25)
    r=np.zeros(21);axis=5;current=Reference21(r,1.);actual=core.actor.encode(current)
    lo=r.copy();hi=r.copy();lo[axis]=-.2;hi[axis]=.2
    checks=[];cases={}
    def check(name,ok,detail=None):
        item=dict(name=name,passed=bool(ok),detail=detail);checks.append(item)
        if not ok:raise AssertionError(name+' '+str(detail))
    def domain(c=core,**kw):
        defaults=dict(q25=q,v25=v,robot_axes25=c.robot_axes25,time_s=1.,current_reference=current,nominal_reference=current,
            actual_engine=c.actor.encode(current),reference_lower21=lo,reference_upper21=hi)
        defaults.update(kw);return c.build_domain(**defaults)
    def solve(d,c=core,request=None,correction=None,**kw):
        if request is None:request=np.zeros(21)
        if correction is None:correction=np.zeros(25)
        defaults=dict(p0_effort21=d.p0_effort21,delta_effort21=request,correction25=correction,
            residual_scale25=np.ones(25),delta_scale21=np.ones(21));defaults.update(kw)
        return c.solve(d,**defaults)
    d=domain();check('actual_model_domain_available',d.available,d.reason)
    check('readonly_original_p0_and_intervals',not d.p0_effort21.flags.writeable and not d.reference_lower21.flags.writeable)
    zero=solve(d);check('zero_correction_legal_nominal_exact',zero.available and zero.reference21.values.tobytes()==current.values.tobytes())
    check('same_multiplicative_objective',zero.objective==0. and zero.diagnostics['scale_convention'].startswith('positive multipliers'))
    A=d.transmission21x25;row=A[axis];correction=.1*row;request=np.zeros(21);request[axis]=.05
    sr=np.ones(25);sd=np.ones(21);sr[row!=0.]=3.;sd[axis]=2.
    expected=float(np.sum(sr**2*row*correction)/(np.sum(sr**2*row**2)+sd[axis]**2))
    optimum=solve(d,request=request,correction=correction,residual_scale25=sr,delta_scale21=sd)
    check('actual_native_closed_form_multiplicative_scales',optimum.available and abs(optimum.actual_delta_effort21[axis]-expected)<1e-13,
        dict(expected=expected,actual=None if not optimum.available else float(optimum.actual_delta_effort21[axis]),reason=optimum.reason))
    native=core.servo.evaluate(q,v,optimum.engine_port21.values);dr=native.predicted_effort21-d.p0_effort21;rr=A.T@dr-correction
    expected_obj=float((sr*rr)@(sr*rr)+(sd*dr)@(sd*dr))
    check('same_native_full_residual_and_objective_bytes',rr.tobytes()==optimum.residual25.tobytes() and expected_obj==optimum.objective)
    check('final_selected_native_bytes_match',all(x['canonical_native_bytes_equal'] for x in optimum.diagnostics['combined_native_rechecks']))
    check('fixed_coordinates_not_inverse_then_clip',np.array_equal(optimum.reference21.values[np.arange(21)!=axis],r[np.arange(21)!=axis]))
    one_lo=r.copy();one_hi=r.copy();one_hi[axis]=.001
    limited=domain(reference_lower21=one_lo,reference_upper21=one_hi)
    saturated=solve(limited,request=request,correction=row*10.)
    check('one_sided_saturation_preserves_requested_goal',saturated.available and saturated.reference21.values[axis]==one_hi[axis] and
        saturated.requested_delta_effort21[axis]==request[axis] and np.linalg.norm(saturated.residual25)>0)
    free_lo=r.copy();free_hi=r.copy();free_lo[axis]=core.actor.ctrl_range21[axis,0];free_hi[axis]=core.actor.ctrl_range21[axis,1]
    wide=domain(reference_lower21=free_lo,reference_upper21=free_hi)
    plateau=solve(wide,request=request*10000,correction=row*1000.)
    check('native_interval_endpoint_value_accepted',plateau.available and plateau.native_effort21[axis]==wide.effort_upper21[axis])
    check('fixed_finite_nextafter_family',NEXTAFTER_PER_SIDE==2 and plateau.diagnostics['finite_family_only'] and not plateau.diagnostics['all_float_global_optimum_certified'])
    nominal_values=r.copy();nominal_values[axis]=.3
    shifted=domain(nominal_reference=Reference21(nominal_values,1.))
    raw=core.servo.evaluate(q,v,core.actor.encode(Reference21(nominal_values,1.)).values)
    check('p0_fixed_before_permission_projection',shifted.available and shifted.p0_effort21.tobytes()==raw.predicted_effort21.tobytes() and
        shifted.reference_upper21[axis]<nominal_values[axis])
    check('retargeted_p0_rejected',not solve(d,p0_effort21=np.nextafter(d.p0_effort21,np.inf)).available)
    other=NativeEffort21(model);check('foreign_domain_rejected',not other.solve(d,p0_effort21=d.p0_effort21,delta_effort21=request,
        correction25=correction,residual_scale25=sr,delta_scale21=sd).available)
    check('tampered_domain_rejected',not solve(replace(d,p0_effort21=d.p0_effort21+1.)).available)
    check('coupled_domain_not_box_enclosed',not domain(coupled_reference_constraints=dict(A=np.ones((1,21)),b=[0.])).available)
    check('unknown_observation_not_zero',not domain(q25=None).available)
    check('robot_axis_order_rejected',not domain(robot_axes25=core.robot_axes25[::-1]).available)
    check('snapshot_clock_mismatch_rejected',not domain(time_s=1.005).available)
    badactual=actual.values.copy();badactual[18]+=core.actor.lift_bias_m
    check('double_lift_bias_rejected',not domain(actual_engine=EnginePort21(badactual,1.,core.actor.lift_bias_m)).available)
    check('invalid_scale_rejected',not solve(d,residual_scale25=np.zeros(25)).available)
    empty=domain(reference_lower21=r+.1,reference_upper21=r-.1)
    check('empty_domain_keeps_original_p0',not empty.available and empty.p0_effort21.tobytes()==d.p0_effort21.tobytes())
    # Mutant model clones are negative/degenerate fixtures only, not modifications
    # to the bound production robot or frozen XML/PD parameters.
    flat_model=copy(model);aid=int(core.servo.aids[axis]);flat_model.actuator_gainprm[aid,:3]=0.;flat_model.actuator_biasprm[aid,:3]=0.
    flat_core=NativeEffort21(flat_model);flat=domain(flat_core)
    need=np.zeros(21);need[axis]=1.
    flat_result=solve(flat,flat_core,request=need,correction=row)
    check('known_zero_gain_is_known_flat_not_unknown',flat.available and flat.effort_lower21[axis]==flat.effort_upper21[axis] and
        flat_result.available and flat_result.actual_delta_effort21[axis]==0. and np.linalg.norm(flat_result.residual25)>0)
    clip_model=copy(model);clip_model.actuator_forcerange[int(core.servo.outadr[axis])]=[-.01,.01]
    clip_core=NativeEffort21(clip_model);clipped_domain=domain(clip_core)
    clipped=solve(clipped_domain,clip_core,request=need,correction=row*1000.)
    check('known_force_clip_kink_value_not_unknown_or_zero_derivative',clipped.available and clipped.native_effort21[axis]==.01 and
        clipped.diagnostics['native_derivative_status']['reference_jacobian']=='UNAVAILABLE_CLIP_KINK')
    huge_model=copy(model);huge_model.actuator_gainprm[aid,:3]=[1.,0.,0.];huge_model.actuator_biasprm[aid,:3]=[1e16,0.,0.]
    huge_model.actuator_forcelimited[int(core.servo.outadr[axis])]=False
    huge_core=NativeEffort21(huge_model);huge=domain(huge_core)
    huge_result=solve(huge,huge_core,request=need,correction=row,requested_target_effort21=huge.p0_effort21+need)
    check('p0_plus_microdelta_rounding_does_not_fake_zero_error',huge_result.available and huge.p0_effort21[axis]+1.==huge.p0_effort21[axis] and
        huge_result.requested_delta_effort21[axis]==1. and huge_result.actual_delta_effort21[axis]==0. and
        np.linalg.norm(huge_result.residual25)>0. and huge_result.diagnostics['absolute_addition_lost_delta21'][axis]==1.)
    check('original_model_PD_unchanged',np.array_equal(model.actuator_gainprm,original_gain))
    core.servo.model.actuator_gainprm[aid,0]+=1.
    check('post_binding_model_mutation_rejected',not domain().available and not solve(d).available)
    check('no_writer_used',all(c.actor.write_count==0 for c in (core,other,flat_core,clip_core,huge_core)))
    check('forbidden_runtime_calls_zero',not forbidden,forbidden)
    cases.update(optimum=optimum.to_dict(),saturation=saturated.to_dict(),known_flat=flat_result.to_dict(),known_clip=clipped.to_dict(),swallowed_delta=huge_result.to_dict())
    return dict(schema='v13_native_effort21_author_zero_step_v1',passed=True,checks=checks,cases=cases,
        forbidden_calls=forbidden,physics_steps=0,writer_calls=0,fixture_scope='Actual frozen robot model plus isolated declared zero/huge-bias model clones; no physical dynamics or baseline parameter edits')


if __name__=='__main__':
    out=ROOT/'review/v13_native_effort21_author_v2.json'
    if out.exists():raise FileExistsError(out)
    try:report=run()
    except Exception as exc:report=dict(schema='v13_native_effort21_author_zero_step_v1',passed=False,error=repr(exc),traceback=traceback.format_exc(),physics_steps=0)
    report['source_sha256']={p:sha(ROOT/p) for p in ['v13_joint/native_effort21.py','v11/robot_servo_effort.py','v13_a1/codec.py']}
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(passed=report['passed'],checks=len(report.get('checks',[])),output=str(out),sha256=sha(out))))
    raise SystemExit(0 if report['passed'] else 1)

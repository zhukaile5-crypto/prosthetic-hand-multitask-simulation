"""Five small independent actual-native inverse oracles; zero dynamics/writes."""
from pathlib import Path
from copy import copy
from dataclasses import replace
import hashlib,importlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
EXPECTED='6f097f88b19a80b8692be6d8a937c7a640d2ad6fe17a12d2f4d87b48c7e148f3'
OUT=ROOT/'review/v13_native_effort21_independent_v2.json'

def main():
    source=ROOT/'v13_joint/native_effort21.py';payload=source.read_bytes();assert hashlib.sha256(payload).hexdigest()==EXPECTED
    sys.path.insert(0,str(ROOT));import research_imports,mujoco
    counts={};restore={}
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_fwdPosition','mj_fwdVelocity','mj_kinematics','mj_comPos','mj_camlight','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):
            restore[name]=getattr(mujoco,name);counts[name]=0
            def reject(*a,_name=name,**kw):counts[_name]+=1;raise AssertionError('forbidden '+_name)
            setattr(mujoco,name,reject)
    from foundation_config import FoundationConfig
    import hand_plant
    old_init=hand_plant.HandPlant.__init__;plant_calls=[]
    def no_plant(*a,**k):plant_calls.append(True);raise AssertionError('HandPlant constructor forbidden')
    hand_plant.HandPlant.__init__=no_plant
    module=importlib.import_module('v13_joint.native_effort21');codec=importlib.import_module('v13_a1.codec')
    old_apply=codec.ReferenceActor21.apply;writer_calls=[]
    def no_write(*a,**k):writer_calls.append(True);raise AssertionError('writer forbidden')
    codec.ReferenceActor21.apply=no_write
    checks=[];cases=[]
    def check(name,yes,detail=None):checks.append(dict(name=name,passed=bool(yes),detail=detail))
    try:
        model,xml_sha=hand_plant._model(FoundationConfig());original_model_gain=model.actuator_gainprm.copy();original_model_bias=model.actuator_biasprm.copy()
        core=module.NativeEffort21(model);axis=module.ACTUATORS21.index('rh_A_FFJ0');axis2=module.ACTUATORS21.index('rh_A_RFJ0')
        q=model.qpos0[core.servo.qadr].copy();v=np.zeros(25);clock=2.345
        ref=np.zeros(21);ref[axis]=.1
        def domain(c,values=ref,lower=None,upper=None,nominal=None,**extra):
            lo=values.copy() if lower is None else lower;hi=values.copy() if upper is None else upper
            if lower is None:lo[axis]=0.
            if upper is None:hi[axis]=.2
            current=codec.Reference21(values,clock);nom=codec.Reference21(values if nominal is None else nominal,clock)
            args=dict(q25=q,v25=v,robot_axes25=c.robot_axes25,time_s=clock,current_reference=current,nominal_reference=nom,
                actual_engine=c.actor.encode(current),reference_lower21=lo,reference_upper21=hi);args.update(extra)
            return c.build_domain(**args)
        def solve(c,d,delta=None,correction=None,sr=None,sd=None,**kw):
            return c.solve(d,p0_effort21=d.p0_effort21,delta_effort21=np.zeros(21) if delta is None else delta,
                correction25=np.zeros(25) if correction is None else correction,residual_scale25=np.ones(25) if sr is None else sr,
                delta_scale21=np.ones(21) if sd is None else sd,**kw)
        # 1: p0+d rounding cannot hide the original requested micro-delta.
        start=len(checks);huge_model=copy(model);aid=int(core.servo.aids[axis]);out=int(core.servo.outadr[axis])
        huge_model.actuator_gainprm[aid,:3]=[1.,0.,0.];huge_model.actuator_biasprm[aid,:3]=[float(2**54),0.,0.];huge_model.actuator_forcelimited[out]=False
        huge=module.NativeEffort21(huge_model);d=domain(huge);delta=np.zeros(21);delta[axis]=.25;corr=d.transmission21x25[axis]*.25
        result=solve(huge,d,delta,corr,requested_target_effort21=d.p0_effort21+delta)
        check('micro_delta_native_domain_known',d.available,d.reason)
        check('micro_delta_not_lost_as_goal',result.available and result.requested_delta_effort21[axis]==.25 and result.actual_delta_effort21[axis]==0.)
        check('micro_delta_loss_disclosed',result.diagnostics['absolute_addition_lost_delta21'][axis]==.25 and result.diagnostics['actual_effort_minus_requested_delta21'][axis]==-.25)
        check('micro_delta_true_residual_and_cost',np.array_equal(result.residual25,-corr) and result.objective==float(corr@corr),dict(objective=result.objective,residual=result.residual25.tolist()))
        cases.append(dict(id='requested_delta_survives_absolute_rounding',passed=all(x['passed'] for x in checks[start:])))
        # 2: known zero native gain stays known/flat with honest residual.
        start=len(checks);flat_model=copy(model);fa=int(core.servo.aids[axis2]);flat_model.actuator_gainprm[fa,:3]=0.;flat_model.actuator_biasprm[fa,:3]=[.03125,0.,0.]
        flat=module.NativeEffort21(flat_model);fr=ref.copy();fr[axis2]=.08;lo=fr.copy();hi=fr.copy();lo[axis2]=0.;hi[axis2]=.2
        d=domain(flat,fr,lo,hi);delta=np.zeros(21);delta[axis2]=.3;corr=d.transmission21x25[axis2]*.3;result=solve(flat,d,delta,corr)
        check('known_zero_gain_domain_and_finite_value',d.available and d.effort_lower21[axis2]==d.effort_upper21[axis2] and d.effort_lower21[axis2]!=0.)
        check('known_zero_gain_not_fake_realization',result.available and result.actual_delta_effort21[axis2]==0. and np.linalg.norm(result.residual25)>0.)
        check('known_zero_gain_nominal_tie_preference',result.reference21.values[axis2]==fr[axis2] and result.diagnostics['finite_candidate_solutions'][axis2]['known_reference_gain']==0.)
        cases.append(dict(id='known_constant_gain_is_not_unknown_or_achieved_delta',passed=all(x['passed'] for x in checks[start:])))
        # 3: one-sided authority keeps original nominal p0; clip/kink values known.
        start=len(checks);lo=ref.copy();hi=ref.copy();lo[axis]=.099;hi[axis]=.1;nom=ref.copy();nom[axis]=.15
        d=domain(core,lower=lo,upper=hi,nominal=nom);delta=np.zeros(21);delta[axis]=.2;corr=d.transmission21x25[axis]*2.;result=solve(core,d,delta,corr)
        low_ulp=np.nextafter(np.nextafter(hi[axis],-np.inf),-np.inf)
        raw=core.servo.evaluate(q,v,core.actor.encode(codec.Reference21(nom,clock)).values)
        check('one_sided_original_p0_not_retargeted',d.p0_effort21.tobytes()==raw.predicted_effort21.tobytes() and d.reference_upper21[axis]<nom[axis])
        check('one_sided_native_finite_endpoint_family',result.available and low_ulp<=result.reference21.values[axis]<=hi[axis] and np.linalg.norm(result.residual25)>0.)
        clipped_model=copy(model);clipped_model.actuator_forcerange[out]=[-.02,.02];clipped=module.NativeEffort21(clipped_model)
        zero=np.zeros(21);lo=zero.copy();hi=zero.copy();hi[axis]=.2;d=domain(clipped,zero,lo,hi)
        delta=np.zeros(21);delta[axis]=.3;corr=d.transmission21x25[axis]*10.;result=solve(clipped,d,delta,corr)
        kink=zero.copy();kink[axis]=.02/float(clipped.servo.model.actuator_gainprm[aid,0]);snapshot=clipped.servo.evaluate(q,v,clipped.actor.encode(codec.Reference21(kink,clock)).values)
        check('known_clip_kink_value_unknown_derivative',snapshot.status=='KNOWN_ROBOT_ACTUATION_ALGEBRA' and snapshot.predicted_effort21[axis]==.02 and snapshot.reference_jacobian25x21 is None and snapshot.actuation_stiffness25x25 is None and snapshot.diagnostics['force_at_clip_kink21'][axis])
        check('plateau_finite_family_and_no_global_claim',result.available and result.native_effort21[axis]<=.02 and result.native_effort21[axis]>=np.nextafter(np.nextafter(.02,-np.inf),-np.inf) and not result.diagnostics['all_float_global_optimum_certified'])
        cases.append(dict(id='one_sided_and_known_clip_kink',passed=all(x['passed'] for x in checks[start:])))
        # 4: ownership/model/clock and coupled domains cannot be silently boxed.
        start=len(checks);d=domain(core);other=module.NativeEffort21(model)
        foreign=other.solve(d,p0_effort21=d.p0_effort21,delta_effort21=np.zeros(21),correction25=np.zeros(25),residual_scale25=np.ones(25),delta_scale21=np.ones(21))
        check('foreign_owner_rejected',not foreign.available and foreign.status=='UNAVAILABLE_SOURCE')
        coupled=domain(core,coupled_reference_constraints=dict(A=np.ones((1,21)),b=[.15]))
        check('coupled_reference_domain_not_box_enclosed',not coupled.available and coupled.status=='UNAVAILABLE_COUPLED_DOMAIN')
        check('different_snapshot_clock_rejected',not domain(core,time_s=clock+.005).available)
        altered=replace(d,p0_effort21=d.p0_effort21+1.)
        check('changed_prepared_domain_rejected',not solve(core,altered).available)
        owned=module.NativeEffort21(model);owned_domain=domain(owned);owned.servo.model.actuator_gainprm[int(owned.servo.aids[axis]),0]*=1.0001
        changed=solve(owned,owned_domain)
        check('bound_model_mutation_rejected',not changed.available and changed.status=='UNAVAILABLE_MODEL')
        delayed_model=copy(model);delayed_model.actuator_delay[aid]=.005;delayed=module.NativeEffort21(delayed_model)
        check('unknown_native_model_returns_unavailable',not domain(delayed).available)
        cases.append(dict(id='ownership_clock_model_and_coupled_domain',passed=all(x['passed'] for x in checks[start:])))
        # 5: two actuator supports, multiplier scales and whole-native score bytes.
        start=len(checks);lo=ref.copy();hi=ref.copy();lo[[axis,19]]=0.;hi[[axis,19]]=[.3,.1]
        d=domain(core,lower=lo,upper=hi);A=d.transmission21x25;delta=np.zeros(21);delta[axis]=.07;delta[19]=.02
        corr=.12*A[axis]+.035*A[19];sr=np.linspace(.75,1.75,25);sd=np.linspace(.8,1.8,21)
        result=solve(core,d,delta,corr,sr,sd);native=core.servo.evaluate(q,v,result.engine_port21.values);dr=native.predicted_effort21-d.p0_effort21
        residual=A.T@dr-corr;weighted=sr*residual;regular=sd*dr;cost=float(weighted@weighted+regular@regular)
        check('full_native_canonical_residual_and_cost_bytes',result.available and residual.tobytes()==result.residual25.tobytes() and np.float64(cost).tobytes()==np.float64(result.objective).tobytes())
        check('all21_combined_local_bytes_agree',all(x['canonical_native_bytes_equal'] for x in result.diagnostics['combined_native_rechecks']))
        expected=float(np.sum(sr**2*A[axis]*corr)/(np.sum(sr**2*A[axis]**2)+sd[axis]**2))
        check('independent_weighted_tendon_closed_form',abs(result.actual_delta_effort21[axis]-expected)<1e-12,dict(expected=expected,actual=float(result.actual_delta_effort21[axis])))
        check('finite_native_bounds_and_proposal_only',result.diagnostics['actual_reference_bounds_ok'] and result.diagnostics['actual_engine_bounds_ok'] and result.diagnostics['writer_called'] is False and result.diagnostics['nextafter_neighbours_per_side']==2)
        cases.append(dict(id='weighted_two_axis_native_combination',passed=all(x['passed'] for x in checks[start:])))
        check('original_public_model_not_mutated',np.array_equal(model.actuator_gainprm,original_model_gain) and np.array_equal(model.actuator_biasprm,original_model_bias))
    finally:
        codec.ReferenceActor21.apply=old_apply;hand_plant.HandPlant.__init__=old_init
        for name,fn in restore.items():setattr(mujoco,name,fn)
    check('no_forbidden_physics_FK_contact_or_writer',not any(counts.values()) and not plant_calls and not writer_calls)
    check('frozen_native_module_unchanged',payload==source.read_bytes())
    report=dict(schema='v13_native_effort21_independent_v2',role='independent_small_native_oracle_peer',passed=all(x['passed'] for x in checks),cases=cases,checks=checks,first_fixture_failure_evidence='review/v13_native_effort21_independent_v1.json',fixture_corrections='FFJ0 actual gain0.5 sets kink reference; unbounded weighted closed-form has explicitly widened legal u range',
        checks_count=len(checks),source_sha256=EXPECTED,checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        static_public_model_compiled=1,public_model_xml_sha256=xml_sha,physics_steps=0,plant_constructor_calls=len(plant_calls),writer_calls=len(writer_calls),forbidden_calls=counts,
        native_primitives_permitted=['mj_resetData','mj_tendon','mj_transmission'],
        limitations=['Five selected algebra/domain cases; not full replay, new plant integration, native response calibration or policy benefit.',
            'Zero/huge-bias/clipped/delayed model copies are isolated fixtures, not modifications of the admitted production model.',
            'Native finite-family proposal availability is not achievement of requested effort, future pressure prediction, physical safety or all-float global optimality.'])
    assert not OUT.exists();OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),passed=report['passed'],checks=len(checks),failed=[x['name'] for x in checks if not x['passed']]),ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()

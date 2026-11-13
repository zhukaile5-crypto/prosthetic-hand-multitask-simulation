"""Independent finite-native inverse audit. Private tendon/transmission only.

Uses trusted frozen F067 as the value authority, independent effort-space
closed forms / endpoint images / native rescoring of all declared candidates.
No author helper, HandPlant construction, FK/contact/forward or integration.
"""
from pathlib import Path
from copy import copy
from dataclasses import replace
import argparse,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
EXPECTED='6f097f88b19a80b8692be6d8a937c7a640d2ad6fe17a12d2f4d87b48c7e148f3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError('Fresh independent native21 evidence required')
    source=ROOT/'v13_joint/native_effort21.py'
    if sha(source)!=EXPECTED:raise ValueError('Expected final native21 SHA changed')
    sys.path.insert(0,str(ROOT));import research_imports,mujoco
    forbidden={};calls={};checks=[];cases=[]
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_inverse','mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation','mj_kinematics','mj_comPos','mj_camlight','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):
            forbidden[name]=getattr(mujoco,name);calls[name]=0
            def reject(*a,_name=name,**k):calls[_name]+=1;raise AssertionError('Forbidden pipeline '+_name)
            setattr(mujoco,name,reject)
    import hand_plant
    old_constructor=hand_plant.HandPlant.__init__
    def no_plant(*a,**k):raise AssertionError('Actual HandPlant construction forbidden')
    hand_plant.HandPlant.__init__=no_plant
    try:
        from foundation_config import FoundationConfig
        from v11.robot_servo_effort import RobotServoEffort
        from v13_a1.codec import Reference21,EnginePort21,ReferenceActor21
        spec=importlib.util.spec_from_file_location('independent_native21_'+EXPECTED[:12],source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        def check(name,passed,detail=None):checks.append(dict(name=name,passed=bool(passed),detail=clean(detail)))
        model,model_sha=hand_plant._model(FoundationConfig())
        base_gain=model.actuator_gainprm.copy();base_bias=model.actuator_biasprm.copy()
        core=module.NativeEffort21(model);oracle=RobotServoEffort(model);encoder=ReferenceActor21(model)
        def native(q,v,reference):
            result=oracle.evaluate(q,v,encoder.encode(reference).values,time_s=reference.time_s)
            if result.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA':raise AssertionError('Independent frozen value authority unavailable')
            return np.asarray(result.predicted_effort21),np.asarray(result.transmission21x25)
        def make_domain(c,q,v,ref,lo,hi,nom=None,**extra):
            kw=dict(q25=q,v25=v,robot_axes25=c.robot_axes25,time_s=ref.time_s,current_reference=ref,
                nominal_reference=ref if nom is None else nom,actual_engine=c.actor.encode(ref),reference_lower21=lo,reference_upper21=hi)
            kw.update(extra);return c.build_domain(**kw)
        def run(c,domain,request,correction,sr,sd,**extra):
            return c.solve(domain,p0_effort21=domain.p0_effort21,delta_effort21=request,correction25=correction,
                residual_scale25=sr,delta_scale21=sd,**extra)
        def score(p,p0,A,correction,sr,sd):
            d=p-p0;r=A.T@d-correction;wr=sr*r;wd=sd*d
            return float(wr@wr+wd@wd),d,r
        bridge=ROOT/'output/v13_a1_native_behavior_bridge_v1';done=read(bridge/'execution_complete.json');jobs=read(bridge/'manifest.json')['jobs']
        check('complete_all9_actual_bridge',done['execution_complete'] and done['all9_exact'] and done['sources_unchanged'] and len(jobs)==9)
        selected=[]
        for job in jobs:
            path=Path(job['destination'])/'trace.npz';fingerprint=sha(path);meta=read(path.with_name('metadata.json'))
            check('source_trace_hash_'+str(len(selected)),fingerprint==meta['trace_sha256'] and meta['completed'])
            with np.load(path,allow_pickle=False) as z:
                time=z['time'];engine=z['robot_actuator_target21']
                # Extract actual named25 before passing any value to model API.
                q25=z['qpos'][:,oracle.qadr].copy();v25=z['qvel'][:,oracle.dadr].copy()
            for timestamp in (7.5,9.435,12.21):
                k=int(np.argmin(abs(time-timestamp)))
                if time[k]!=timestamp:raise AssertionError('Predetermined public packet not found exactly')
                ref=encoder.decode(EnginePort21(engine[k],float(time[k]),encoder.lift_bias_m))
                selected.append(dict(identity={key:job[key] for key in ('scenario','method','seed','motion_mode')},trace=str(path),trace_sha256=fingerprint,time=float(time[k]),q=q25[k],v=v25[k],reference=ref))
            check('source_unchanged_after_read_'+str(len(selected)),sha(path)==fingerprint)
        rng=np.random.default_rng(572109);candidate_evaluations=0;max_closed_gap=0.;dense_diagnostics=[]
        for index,packet in enumerate(selected):
            q,v,ref=packet['q'],packet['v'],packet['reference'];width=np.r_[np.ones(18)*.005,.00003,.00015,.00015]
            domain=make_domain(core,q,v,ref,ref.values-width,ref.values+width)
            check('actual_domain_'+str(index),domain.available,domain.reason)
            if not domain.available:continue
            p0,A=native(q,v,ref);pl,_=native(q,v,Reference21(domain.reference_lower21,ref.time_s));pu,_=native(q,v,Reference21(domain.reference_upper21,ref.time_s))
            check('actual_p0_fixed_raw_nominal_'+str(index),same(domain.p0_effort21,p0))
            check('actual_endpoint_box_independent_'+str(index),same(domain.effort_lower21,np.minimum(pl,pu)) and same(domain.effort_upper21,np.maximum(pl,pu)))
            sr=rng.uniform(.5,2.,25);sd=rng.uniform(.5,2.,21);correction=rng.normal(0,.03,25)
            # Independent actuator-EFFORT space minimizer, not reference PD inversion.
            request=np.array([(np.dot(sr*row,sr*correction))/(np.dot(sr*row,sr*row)+sd[i]**2) for i,row in enumerate(A)])
            request=np.clip(request,domain.effort_lower21-p0,domain.effort_upper21-p0)
            result=run(core,domain,request,correction,sr,sd)
            check('actual_available_finite_candidate_'+str(index),result.available,result.reason)
            if not result.available:continue
            p,_=native(q,v,result.reference21);cost,d,r=score(p,p0,A,correction,sr,sd)
            check('actual_final_native_value_bytes_'+str(index),same(result.native_effort21,p) and same(result.actual_delta_effort21,d) and same(result.residual25,r) and same([result.objective],[cost]))
            check('actual_original_request_preserved_'+str(index),same(result.requested_delta_effort21,request))
            check('actual_legal_reference_and_port_'+str(index),np.all(result.reference21.values>=domain.reference_lower21) and np.all(result.reference21.values<=domain.reference_upper21) and result.reference21.time_s==ref.time_s and result.engine_port21.time_s==ref.time_s)
            analytic_residual=A.T@request-correction;weighted_analytic=sr*analytic_residual;regular_analytic=sd*request
            lower_cost=float(weighted_analytic@weighted_analytic+regular_analytic@regular_analytic)
            gap=cost-lower_cost;max_closed_gap=max(max_closed_gap,abs(gap))
            # Inspect, independently native-score and order every declared family.
            seed=np.minimum(domain.reference_upper21,np.maximum(domain.reference_lower21,ref.values))
            for solution in result.diagnostics['finite_candidate_solutions']:
                axis=solution['axis'];support=np.flatnonzero(A[axis]);evaluated=[]
                for c in solution['candidates']:
                    vector=seed.copy();vector[axis]=c['value'];pp,_=native(q,v,Reference21(vector,ref.time_s));cc,dd,rr=score(pp,p0,A,correction,sr,sd);wr=sr[support]*rr[support]
                    local=float(wr@wr+(sd[axis]*float(dd[axis]))**2);candidate_evaluations+=1
                    check('candidate_native_byte_'+str(index)+'_'+str(axis)+'_'+str(len(evaluated)),same([local],[c['local_native_objective']]) and same([cc],[c['full_seed_native_objective']]) and rr[support].tobytes().hex()==c['residual_bytes_hex'] and np.float64(dd[axis]).tobytes().hex()==c['delta_bytes_hex'])
                    evaluated.append((local,abs(c['value']-ref.values[axis]),abs(c['value']-ref.values[axis]),c['value']))
                winner=min(evaluated)
                check('candidate_lexicographic_minimum_'+str(index)+'_'+str(axis),np.float64(winner[-1]).tobytes()==np.float64(solution['selected']['value']).tobytes())
            axis=(0,7,18,19,20)[index%5];best_grid=float('inf')
            for value in np.linspace(domain.reference_lower21[axis],domain.reference_upper21[axis],101):
                point=result.reference21.values.copy();point[axis]=value;pp,_=native(q,v,Reference21(point,ref.time_s));cc,_,_=score(pp,p0,A,correction,sr,sd);best_grid=min(best_grid,cc)
            dense_diagnostics.append(dict(packet=index,axis=axis,selected_minus_grid_objective=cost-best_grid,
                note='Grid is diagnostic only, not a required finite-family/global-float optimality gate'))
            cases.append(dict(case='actual_public_packet',packet=index,identity=packet['identity'],time_s=ref.time_s,trace=packet['trace'],trace_sha256=packet['trace_sha256'],objective=cost,effort_space_closed_form_gap=gap,native_evaluations=result.diagnostics['native_evaluations']))
        # Raw-nominal baseline beyond permission: retain original p0 and disclose gap.
        packet=selected[0];q,v,current=packet['q'],packet['v'],packet['reference'];lo=current.values.copy();hi=current.values.copy();lo[0]-=.01;hi[0]+=.01
        nominal_values=current.values.copy();nominal_values[0]+=.3;nominal=Reference21(nominal_values,current.time_s)
        domain=make_domain(core,q,v,current,lo,hi,nom=nominal);expected,_=native(q,v,nominal);anchor=Reference21(np.minimum(domain.reference_upper21,np.maximum(domain.reference_lower21,nominal.values)),current.time_s);anchor_effort,_=native(q,v,anchor)
        check('preprojection_p0_is_raw_native_not_anchor',domain.available and same(domain.p0_effort21,expected) and not same(domain.p0_effort21,anchor_effort))
        result=run(core,domain,np.zeros(21),np.zeros(25),np.ones(25),np.ones(21))
        check('unreachable_preprojection_target_has_real_residual',result.available and np.max(abs(result.actual_delta_effort21))>0 and not np.array_equal(result.residual25,np.zeros(25)))
        domain=make_domain(core,q,v,current,current.values-.001,current.values+.001)
        tiny=np.zeros(21);tiny[18]=np.spacing(domain.p0_effort21[18])/4
        result=run(core,domain,tiny,np.zeros(25),np.ones(25),np.ones(21))
        check('raw_delta_lost_by_absolute_addition_preserved',result.available and same(result.requested_delta_effort21,tiny) and result.diagnostics['absolute_addition_lost_delta21'][18]==tiny[18])
        # Source/clock/domain invalidities fail closed, never substitute a zero snapshot.
        foreign=module.NativeEffort21(model)
        check('foreign_domain_rejected',not foreign.solve(domain,p0_effort21=domain.p0_effort21,delta_effort21=np.zeros(21),correction25=np.zeros(25),residual_scale25=np.ones(25),delta_scale21=np.ones(21)).available)
        altered=replace(domain,p0_effort21=domain.p0_effort21+1.)
        check('modified_domain_rejected',not run(core,altered,np.zeros(21),np.zeros(25),np.ones(25),np.ones(21)).available)
        check('retargeted_p0_rejected',not core.solve(domain,p0_effort21=domain.p0_effort21+1.,delta_effort21=np.zeros(21),correction25=np.zeros(25),residual_scale25=np.ones(25),delta_scale21=np.ones(21)).available)
        check('unknown_q_rejected_not_zero',not make_domain(core,None,v,current,current.values-.01,current.values+.01).available)
        check('clock_mismatch_rejected',not make_domain(core,q,v,current,current.values-.01,current.values+.01,nom=Reference21(current.values,current.time_s+.005)).available)
        check('coupled_domain_not_outer_box',not make_domain(core,q,v,current,current.values-.01,current.values+.01,coupled_reference_constraints=[]).available)
        check('empty_requested_set_rejected',not make_domain(core,q,v,current,current.values+.01,current.values-.01).available)
        check('unknown_axis_order_rejected',not make_domain(core,q,v,current,current.values-.01,current.values+.01,robot_axes25=tuple(reversed(core.robot_axes25))).available)
        wrong_values=core.actor.encode(current).values.copy();wrong_values[0]=np.nextafter(wrong_values[0],np.inf)
        check('mismatched_actual_engine_byte_rejected',not make_domain(core,q,v,current,current.values-.01,current.values+.01,actual_engine=EnginePort21(wrong_values,current.time_s,core.actor.lift_bias_m)).available)
        check('foreign_lift_bias_rejected',not make_domain(core,q,v,current,current.values-.01,current.values+.01,actual_engine=EnginePort21(core.actor.encode(current).values,current.time_s,core.actor.lift_bias_m+.001)).available)
        check('untyped_engine_rejected',not make_domain(core,q,v,current,current.values-.01,current.values+.01,actual_engine=core.actor.encode(current).values).available)
        check('nonfinite_velocity_unknown_rejected',not make_domain(core,q,np.full(25,np.nan),current,current.values-.01,current.values+.01).available)
        check('zero_multiplier_not_division_fallback',not run(core,domain,np.zeros(21),np.zeros(25),np.zeros(25),np.ones(21)).available)
        check('omitted_finite_permissions_rejected',not core.build_domain(q25=q,v25=v,robot_axes25=core.robot_axes25,time_s=current.time_s,current_reference=current,nominal_reference=current,actual_engine=core.actor.encode(current)).available)
        # Actual model-derived, zero-integration coefficient fixtures. Never simulate
        # negative/zero-gain models or use these as real hardware parameter claims.
        for mode in ('negative_gain','zero_gain','force_plateau','clip_kink'):
            fixture=copy(model);axis=2 if mode in ('negative_gain','zero_gain') else 0
            aid=int(core.servo.aids[axis]);outadr=int(core.servo.outadr[axis]);fixture.actuator_biasprm[aid,:3]=0.
            if mode=='negative_gain':fixture.actuator_gainprm[aid,0]=-.5
            if mode=='zero_gain':fixture.actuator_gainprm[aid,0]=0.;fixture.actuator_biasprm[aid,0]=.07
            if mode in ('force_plateau','clip_kink'):fixture.actuator_forcerange[outadr]=[-.05,.05]
            c=module.NativeEffort21(fixture);value=current.values.copy();value[axis]=.24 if mode=='force_plateau' else .125 if mode=='clip_kink' else 0.
            ref=Reference21(value,current.time_s);lower=value.copy();upper=value.copy();lower[axis]=.2 if mode=='force_plateau' else .1 if mode=='clip_kink' else -.1;upper[axis]=.3 if mode=='force_plateau' else .15 if mode=='clip_kink' else .1
            dd=make_domain(c,q,v,ref,lower,upper);check(mode+'_domain_available',dd.available,dd.reason)
            correction=np.zeros(25);support=np.flatnonzero(dd.transmission21x25[axis]);correction[support]=.02
            wanted=np.zeros(21);wanted[axis]=.01;res=run(c,dd,wanted,correction,np.ones(25),np.ones(21))
            check(mode+'_finite_proposal_available',res.available,res.reason)
            check(mode+'_not_physical_or_global_certificate',res.available and res.diagnostics['all_float_global_optimum_certified'] is False and res.diagnostics['writer_called'] is False)
            if mode in ('zero_gain','force_plateau'):
                check(mode+'_constant_known_effort_domain',dd.effort_lower21[axis]==dd.effort_upper21[axis] and dd.diagnostics['flat_effort_axes21'][axis])
                check(mode+'_tie_prefers_nominal',res.available and res.reference21.values[axis]==ref.values[axis])
            if mode=='negative_gain':check('negative_gain_ordered_effort_image',dd.effort_lower21[axis]<dd.effort_upper21[axis] and res.reference21.values[axis]<0.)
            if mode=='clip_kink':check('kink_derivative_unknown_does_not_destroy_native_values',dd._context['state'].reference_jacobian25x21 is None and res.available)
            before=c.servo.model.actuator_gainprm[aid,0];c.servo.model.actuator_gainprm[aid,0]=before+.1
            check(mode+'_model_mutation_rejected',not run(c,dd,wanted,correction,np.ones(25),np.ones(21)).available)
        check('original_model_parameters_unchanged',same(model.actuator_gainprm,base_gain) and same(model.actuator_biasprm,base_bias))
        check('writer_free_actual_and_oracle',core.actor.write_count==0 and encoder.write_count==0)
        check('all_forbidden_pipeline_calls_zero',all(x==0 for x in calls.values()))
        check('kernel_source_unchanged',sha(source)==EXPECTED)
        result=dict(schema='v13_native_effort21_independent_zero_dynamics_v1',passed=all(c['passed'] for c in checks),checks=checks,failures=[c for c in checks if not c['passed']],
            source_sha256=EXPECTED,checker_sha256=sha(__file__),trusted_value_authority_sha256=sha(ROOT/'v11/robot_servo_effort.py'),nominal_robot_model_sha256=model_sha,
            physics_steps=0,forbidden_pipeline_calls=calls,public_packet_count=len(selected),actual_packet_cases=cases,candidate_rescores=candidate_evaluations,
            max_effort_space_closed_form_objective_gap=max_closed_gap,dense_grid_diagnostics=dense_diagnostics,
            limitations=['Trusted F067 value oracle, not a new independent torque sensor or pressure predictor.',
                'No all-float/full Cartesian finite-family optimality claim: scalar-family minima are checked; complete final canonical value is separately rescored.',
                'Domain owner binds a prepared snapshot but does not prove runtime freshness/generation.',
                'Requested delta is retained as candidate/diagnostic; returned optimum minimizes fixed-correction residual+regularizer, not arbitrary delta-target tracking.',
                'Private internal context tampering beyond documented public readonly fields is not a security boundary; immutable runtime ownership still required.'])
        with out.open('x',encoding='utf-8') as stream:json.dump(clean(result),stream,ensure_ascii=False,indent=2,allow_nan=False)
        print(json.dumps(dict(passed=result['passed'],checks=len(checks),failures=len(result['failures']),packets=len(selected),candidate_rescores=candidate_evaluations,physics_steps=0,out=str(out),sha256=sha(out)),ensure_ascii=False))
    finally:
        hand_plant.HandPlant.__init__=old_constructor
        for name,function in forbidden.items():setattr(mujoco,name,function)

if __name__=='__main__':main()

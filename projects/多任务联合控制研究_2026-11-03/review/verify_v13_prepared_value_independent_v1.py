"""Frozen6f vs value-only5b95 independent bit regression; zero physics.

No author helper, value-only or zero shortcuts. Complete inverse public result
excludes only a separately validated final-full recheck record. Domain excludes
only declared model fingerprint. Intermediate J/K/D must explicitly be unknown.
"""
from pathlib import Path
from dataclasses import fields
from copy import copy
import argparse,hashlib,json,struct,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE_SHA='6f097f88b19a80b8692be6d8a937c7a640d2ad6fe17a12d2f4d87b48c7e148f3'
PREP_SHA='5b95ec9f1614415ec75b91ecb02629f4316d844960b6111932ffb8d1c76f81ac'
CACHE_SHA='3ae5efa293ad7f7987e07c51722c94943a3549bcd11cf639ea13366abd4d151b'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def bits(value):
    if isinstance(value,np.ndarray):return b'A'+str(value.dtype).encode()+repr(value.shape).encode()+value.tobytes()
    if isinstance(value,(float,np.floating)):return b'F'+struct.pack('<d',float(value))
    if isinstance(value,(bool,np.bool_)):return b'B'+bytes([bool(value)])
    if isinstance(value,(int,np.integer)):return b'I'+str(int(value)).encode()+b';'
    if value is None:return b'N'
    if isinstance(value,str):return b'S'+str(len(value.encode())).encode()+b':'+value.encode()
    if isinstance(value,dict):return b'D'+b''.join(bits(k)+bits(v) for k,v in value.items())
    if isinstance(value,(tuple,list)):return b'L'+b''.join(bits(x) for x in value)
    raise TypeError('Unregistered public result type '+str(type(value)))
def digest(value):return hashlib.sha256(bits(value)).hexdigest()
def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError('Fresh value-only independent output required')
    baseline_path=ROOT/'v13_joint/native_effort21.py';prepared_path=ROOT/'v13_joint/prepared_value_effort21.py';cache_path=ROOT/'v13_joint/prepared_native_effort21.py'
    if sha(baseline_path)!=BASE_SHA or sha(prepared_path)!=PREP_SHA or sha(cache_path)!=CACHE_SHA:raise ValueError('Exact separate baseline/cache/value SHAs required')
    sys.path.insert(0,str(ROOT));import research_imports,mujoco,hand_plant
    guarded={};calls={};checks=[];records=[];original_constructor=hand_plant.HandPlant.__init__
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_inverse','mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation','mj_kinematics','mj_comPos','mj_camlight','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):
            guarded[name]=getattr(mujoco,name);calls[name]=0
            def reject(*a,_name=name,**k):calls[_name]+=1;raise AssertionError('Forbidden '+_name)
            setattr(mujoco,name,reject)
    hand_plant.HandPlant.__init__=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No actual plant'))
    try:
        from foundation_config import FoundationConfig
        from v11.robot_servo_effort import RobotServoEffort
        from v13_a1.codec import Reference21,EnginePort21
        from v13_joint.native_effort21 import NativeEffort21
        from v13_joint.prepared_value_effort21 import PreparedValueEffort21
        def check(name,okay,detail=None):checks.append(dict(name=name,passed=bool(okay),detail=clean(detail)))
        model,_=hand_plant._model(FoundationConfig());baseline=NativeEffort21(model);prepared=PreparedValueEffort21(model)
        packets=[];bridge=ROOT/'output/v13_a1_native_behavior_bridge_v1';done=read(bridge/'execution_complete.json')
        jobs=read(bridge/'manifest.json')['jobs'];check('complete9_original_packet_sources',done['execution_complete'] and done['all9_exact'] and len(jobs)==9)
        for job in jobs:
            path=Path(job['destination'])/'trace.npz';before=sha(path);meta=read(path.with_name('metadata.json'))
            check('completed_source_identity_'+str(len(packets)),meta['completed'] and before==meta['trace_sha256'])
            with np.load(path,allow_pickle=False) as z:
                t=z['time'];u=z['robot_actuator_target21'];q=z['qpos'][:,baseline.servo.qadr].copy();v=z['qvel'][:,baseline.servo.dadr].copy()
            for timestamp in (7.5,9.435,12.21):
                k=int(np.argmin(abs(t-timestamp)))
                if float(t[k])!=timestamp:raise ValueError('Predetermined packet clock missing')
                reference=baseline.actor.decode(EnginePort21(u[k],timestamp,baseline.actor.lift_bias_m))
                packets.append(dict(q=q[k],v=v[k],reference=reference,source=str(path),source_sha256=before,time_s=timestamp,identity={key:job[key] for key in ('method','scenario','seed','motion_mode')}))
            check('source_still_unchanged_'+str(len(packets)),sha(path)==before)
        def domain(core,packet,lo,hi,nominal=None):
            ref=packet['reference'];return core.build_domain(q25=packet['q'],v25=packet['v'],robot_axes25=core.robot_axes25,time_s=ref.time_s,
                current_reference=ref,nominal_reference=ref if nominal is None else nominal,actual_engine=core.actor.encode(ref),reference_lower21=lo,reference_upper21=hi)
        def compare_case(name,b,p,packet,lo,hi,correction,sr,sd,nominal=None,raw_delta=None):
            began=time.perf_counter();db=domain(b,packet,lo,hi,nominal);base_domain_ms=(time.perf_counter()-began)*1000
            began=time.perf_counter();dp=domain(p,packet,lo,hi,nominal);prep_domain_ms=(time.perf_counter()-began)*1000
            check(name+'_both_domains_available',db.available and dp.available,dict(base=db.reason,prepared=dp.reason))
            if not db.available or not dp.available:return
            bdict=db.to_dict();pdict=dp.to_dict();bsig=bdict['diagnostics'].pop('model_signature');psig=pdict['diagnostics'].pop('model_signature')
            check(name+'_explicit_domain_fingerprints',bsig==b._initial_model_signature and psig==p._initial_model_signature and len(bsig)==len(psig)==64)
            check(name+'_domain_public_bits_except_fingerprint',bits(bdict)==bits(pdict))
            if raw_delta is None:
                A=np.asarray(db.transmission21x25);raw_delta=np.array([(np.dot(sr*row,sr*correction))/(np.dot(sr*row,sr*row)+sd[i]**2) for i,row in enumerate(A)])
                raw_delta=np.clip(raw_delta,db.effort_lower21-db.p0_effort21,db.effort_upper21-db.p0_effort21)
            kw=dict(delta_effort21=raw_delta,correction25=correction,residual_scale25=sr,delta_scale21=sd)
            began=time.perf_counter();rb=b.solve(db,p0_effort21=db.p0_effort21,**kw);base_solve_ms=(time.perf_counter()-began)*1000
            full_before=p.cache_diagnostics()['full_evaluations']
            began=time.perf_counter();rp=p.solve(dp,p0_effort21=dp.p0_effort21,**kw);prep_solve_ms=(time.perf_counter()-began)*1000
            complete_base=rb.to_dict();complete_prepared=rp.to_dict();recheck=complete_prepared['diagnostics'].pop('prepared_value_final_full_recheck',None)
            required=('passed','native_bytes_equal','delta_bytes_equal','residual_bytes_equal','objective_bytes_equal','transmission_bytes_equal','no_zero_shortcut')
            check(name+'_mandatory_final_full_recheck',isinstance(recheck,dict) and all(recheck.get(key) is True for key in required) and recheck.get('derivative_source')=='unmodified_F067_full_evaluate',recheck)
            check(name+'_full_native_recheck_actually_called',p.cache_diagnostics()['full_evaluations']>full_before)
            check(name+'_complete_inverse_bits_ONLY_VALIDATED_RECHECK_EXCLUDED',bits(complete_base)==bits(complete_prepared))
            check(name+'_available_status_unchanged',rb.available and rp.available,dict(base=rb.status,prepared=rp.status))
            records.append(dict(name=name,time_s=packet['reference'].time_s,identity=packet.get('identity'),source=packet.get('source'),source_sha256=packet.get('source_sha256'),
                domain_public_digest=digest(bdict),inverse_complete_digest=digest(rb.to_dict()),validated_final_full_recheck=recheck,
                native_candidates=sum(len(x['candidates']) for x in rb.diagnostics.get('finite_candidate_solutions',[])),
                base_domain_ms=base_domain_ms,prepared_domain_ms=prep_domain_ms,base_solve_ms=base_solve_ms,prepared_solve_ms=prep_solve_ms,
                cache_after=p.cache_diagnostics()))
        rng=np.random.default_rng(572109)
        for index,packet in enumerate(packets):
            width=np.r_[np.ones(18)*.005,.00003,.00015,.00015];ref=packet['reference'].values
            sr=rng.uniform(.5,2.,25);sd=rng.uniform(.5,2.,21);correction=rng.normal(0,.03,25)
            compare_case('actual_packet_'+str(index),baseline,prepared,packet,ref-width,ref+width,correction,sr,sd)
        packet=packets[0];current=packet['reference'];ref=current.values
        compare_case('zero_correction_legal_nominal_no_shortcut',baseline,prepared,packet,ref-.001,ref+.001,np.zeros(25),np.ones(25),np.ones(21),raw_delta=np.zeros(21))
        lo=ref.copy();hi=ref.copy();lo[0]-=.01;hi[0]+=.01;nominal=ref.copy();nominal[0]+=.3
        compare_case('fixed_raw_nominal_before_projection',baseline,prepared,packet,lo,hi,np.zeros(25),np.ones(25),np.ones(21),Reference21(nominal,current.time_s),np.zeros(21))
        tiny=np.zeros(21);db=domain(baseline,packet,ref-.001,ref+.001);tiny[18]=np.spacing(db.p0_effort21[18])/4
        compare_case('lost_absolute_addition_preserves_raw_delta',baseline,prepared,packet,ref-.001,ref+.001,np.zeros(25),np.ones(25),np.ones(21),raw_delta=tiny)
        for mode in ('negative_gain','zero_gain','force_plateau','clip_kink'):
            fixture=copy(model);axis=2 if mode in ('negative_gain','zero_gain') else 0;aid=int(baseline.servo.aids[axis]);outadr=int(baseline.servo.outadr[axis]);fixture.actuator_biasprm[aid,:3]=0.
            if mode=='negative_gain':fixture.actuator_gainprm[aid,0]=-.5
            if mode=='zero_gain':fixture.actuator_gainprm[aid,0]=0.;fixture.actuator_biasprm[aid,0]=.07
            if mode in ('force_plateau','clip_kink'):fixture.actuator_forcerange[outadr]=[-.05,.05]
            b=NativeEffort21(fixture);p=PreparedValueEffort21(fixture);value=ref.copy();value[axis]=.24 if mode=='force_plateau' else .125 if mode=='clip_kink' else 0.
            lower=value.copy();upper=value.copy();lower[axis]=.2 if mode=='force_plateau' else .1 if mode=='clip_kink' else -.1;upper[axis]=.3 if mode=='force_plateau' else .15 if mode=='clip_kink' else .1
            changed=dict(packet,reference=Reference21(value,current.time_s));correction=np.zeros(25);correction[b.servo._transmission(packet['q'],packet['v'])[0][axis]!=0]=.02
            wanted=np.zeros(21);wanted[axis]=.01;compare_case(mode,b,p,changed,lower,upper,correction,np.ones(25),np.ones(21),raw_delta=wanted)
        # Original F067 full snapshot, repeated q/v with fresh controls/clips/time.
        raw=RobotServoEffort(model);cached=PreparedValueEffort21(model).servo;servo_records=[]
        def compare_servo(name,q,v,port,*,control_layout='robot21',time_s=0.):
            sb=raw.evaluate(q,v,port,control_layout=control_layout,time_s=time_s)
            sp=cached.evaluate(q,v,port,control_layout=control_layout,time_s=time_s)
            bpub={f.name:getattr(sb,f.name) for f in fields(sb)};ppub={f.name:getattr(sp,f.name) for f in fields(sp)}
            bd=dict(bpub['diagnostics']);pd=dict(ppub['diagnostics']);hit=pd.get('prepared_transmission_cache_hit',False);value_only=pd.get('prepared_value_only',False)
            if value_only:
                derivatives=('reference_jacobian25x21','actuation_stiffness25x25','actuation_damping25x25')
                check(name+'_derivatives_explicitly_NOT_COMPUTED',all(ppub[key] is None for key in derivatives) and set(sp.derivative_status.values())=={'NOT_COMPUTED_VALUE_ONLY'} and pd.get('derivative_computation')=='NOT_COMPUTED_VALUE_ONLY')
                for key in derivatives+('derivative_status',):bpub.pop(key);ppub.pop(key)
            for key in ('native_value_arithmetic','prepared_transmission_cache_hit','prepared_cache_private_data_scope'):pd.pop(key,None)
            if value_only:pd.pop('prepared_value_only');pd.pop('derivative_computation')
            if hit or value_only:bd.pop('pipeline',None);pd.pop('pipeline',None)
            bpub['diagnostics']=bd;ppub['diagnostics']=pd
            check(name+'_all_non_derivative_native_bits',bits(bpub)==bits(ppub),dict(hit=hit,value_only=value_only,time_s=sp.time_s))
            servo_records.append(dict(name=name,cache_hit=hit,value_only=value_only,actual_time_s=sp.time_s,public_digest=digest(bpub)))
        for index in range(12):
            values=ref.copy()
            if index%4==1:values[0]+=.003
            elif index%4==2:values[18]+=.0001
            elif index%4==3:values[19]=1.0  # raw control clipping fixture, no write
            port=baseline.actor.encode(Reference21(values,current.time_s+index*.005)).values
            compare_servo('F067_snapshot_'+str(index),packet['q'],packet['v'],port,time_s=current.time_s+index*.005)
        signed_zero=np.zeros(21);signed_zero[[3,7,18]]=np.float64(-0.)
        compare_servo('signed_zero_robot21',packet['q'],packet['v'],signed_zero,time_s=current.time_s+.1)
        model22=np.zeros(22);model22[baseline.servo.ctrladr]=signed_zero;model22[21]=np.nan
        compare_servo('model22_nonrobot_nan_ignored',packet['q'],packet['v'],model22,control_layout='model22',time_s=current.time_s+.105)
        compare_servo('invalid_cached_q_not_zero',None,packet['v'],signed_zero,time_s=current.time_s+.11)
        compare_servo('invalid_cached_time_priority',packet['q'],packet['v'],signed_zero,time_s=True)
        check('repeated_qv_really_used_value_path',cached.value_evaluations>=13 and cached.transmission_misses==1,cached.cache_diagnostics())
        # Unsupported native model has priority over malformed input, as F067.
        bad=copy(model);bad.actuator_dyntype[baseline.servo.aids[2]]=mujoco.mjtDyn.mjDYN_FILTER
        raw_bad=RobotServoEffort(bad);value_bad=PreparedValueEffort21(bad).servo
        sb=raw_bad.evaluate(None,np.full(25,np.nan),np.ones(21,bool));sp=value_bad.evaluate(None,np.full(25,np.nan),np.ones(21,bool))
        check('initial_unsupported_mode_priority_preserved',sb.status==sp.status=='UNAVAILABLE' and sb.reasons==sp.reasons and sp.predicted_effort21 is None and value_bad.value_evaluations==0)
        # Declared expected behavior difference: the6f missing group guard is NOT copied.
        fixture=copy(model);fixture.opt.disableactuator=1<<5;b=NativeEffort21(fixture);p=PreparedValueEffort21(fixture)
        bd=domain(b,packet,ref-.001,ref+.001);pd=domain(p,packet,ref-.001,ref+.001)
        b.servo.model.actuator_group[b.servo.aids[0]]=5;p.servo.model.actuator_group[p.servo.aids[0]]=5
        kw=dict(delta_effort21=np.zeros(21),correction25=np.zeros(25),residual_scale25=np.ones(25),delta_scale21=np.ones(21))
        rb=b.solve(bd,p0_effort21=bd.p0_effort21,**kw);rp=p.solve(pd,p0_effort21=pd.p0_effort21,**kw)
        check('group_mutation_predeclared_corrected_rejection',rb.available and not rp.available,dict(base_status=rb.status,prepared_status=rp.status,prepared_reason=rp.reason))
        check('cache_group_mutation_has_invalidated',p.cache_diagnostics()['model_invalidations']>=1,p.cache_diagnostics())
        check('group_mutation_clears_value_snapshot',not p.cache_diagnostics()['value_snapshot_populated'])
        check('no_robot_writer',baseline.actor.write_count==prepared.actor.write_count==0)
        check('all_forbidden_pipeline_calls_zero',all(x==0 for x in calls.values()),calls)
        check('frozen_base_prepared_sources_unchanged',sha(baseline_path)==BASE_SHA and sha(prepared_path)==PREP_SHA and sha(cache_path)==CACHE_SHA)
        result=dict(schema='prepared_value5b95_independent_bit_regression_v1',passed=all(c['passed'] for c in checks),checks=checks,failures=[c for c in checks if not c['passed']],
            baseline_sha256=BASE_SHA,prepared_sha256=PREP_SHA,cache_dependency_sha256=CACHE_SHA,checker_sha256=sha(__file__),physics_steps=0,forbidden_pipeline_calls=calls,
            original_input_math_reference='review/v13_native_effort21_hand_foundation_independent_v1.json',actual_packets=27,
            complete_inverse_cases=records,complete_inverse_field_exclusions=['diagnostics.prepared_value_final_full_recheck ONLY AFTER every mandatory full native byte check and no_zero_shortcut flag is true'],
            domain_field_exclusions=['diagnostics.model_signature (each separately validated, stronger prepared guard)'],
            raw_servo_diagnostic_exclusions=['native_value_arithmetic','prepared_transmission_cache_hit','prepared_cache_private_data_scope','pipeline only if prepared cache/value hit','prepared_value_only and derivative_computation only for explicit value-only snapshot'],
            raw_servo_derivative_exclusions=['J/K/D and derivative_status only if all are explicitly NOT_COMPUTED_VALUE_ONLY; final inverse status must match original F067'],
            original_F067_snapshot_cases=servo_records,predeclared_expected_difference='actuator_group/disableactuator mode mutation now rejected',
            timing_scope='Direct private build_domain/solve calls only; excludes IO/sensing/controller/physics, not a20ms/hard-real-time claim',
            not_checked='No zero shortcut; no new physical performance evidence, no hard-real-time guarantee, no blanket all-float optimum or pressure certificate')
        with out.open('x',encoding='utf-8') as stream:json.dump(clean(result),stream,ensure_ascii=False,indent=2,allow_nan=False)
        print(json.dumps(dict(passed=result['passed'],checks=len(checks),failures=len(result['failures']),inverse_cases=len(records),physics_steps=0,out=str(out),sha256=sha(out)),ensure_ascii=False))
    finally:
        hand_plant.HandPlant.__init__=original_constructor
        for name,function in guarded.items():setattr(mujoco,name,function)
if __name__=='__main__':main()

"""Zero-physics cache-only regression and serial public-snapshot profiling."""
from pathlib import Path
from copy import copy
from dataclasses import fields
import argparse,hashlib,json,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];F=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(F));sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out);out=out if out.is_absolute() else ROOT/out
    if out.exists():raise FileExistsError(out)
    watched=['v13_joint/prepared_value_effort21.py','v13_joint/prepared_native_effort21.py','v13_joint/native_effort21.py','v11/robot_servo_effort.py','v13_a1/codec.py']
    before={p:sha(ROOT/p) for p in watched};checks=[];profiles=[];forbidden=[]
    def check(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:raise AssertionError(name+' '+str(detail))
    def same(a,b):
        if isinstance(a,dict) and isinstance(b,dict) and 'prepared_value_final_full_recheck' in b:
            if not all(v is True for k,v in b['prepared_value_final_full_recheck'].items() if k.endswith('_equal') or k=='passed'):return False
            b={k:v for k,v in b.items() if k!='prepared_value_final_full_recheck'}
        if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
            return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
        if isinstance(a,(float,np.floating)) and isinstance(b,(float,np.floating)):return np.float64(a).tobytes()==np.float64(b).tobytes()
        if type(a)!=type(b):return False
        if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    try:
        import research_imports,mujoco,hand_plant
        for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_fwdPosition','mj_fwdVelocity','mj_kinematics','mj_comPos','mj_camlight','mj_collision','mj_contactForce'):
            if hasattr(mujoco,name):
                def deny(*a,_name=name,**k):forbidden.append(_name);raise AssertionError('Forbidden '+_name)
                setattr(mujoco,name,deny)
        hand_plant.HandPlant.__init__=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No plant construction'))
        from foundation_config import FoundationConfig
        from v13_a1.codec import Reference21,EnginePort21,ReferenceActor21
        ReferenceActor21.apply=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No writer'))
        import v13_joint.native_effort21 as base_module
        from v13_joint.native_effort21 import NativeEffort21
        from v13_joint.prepared_value_effort21 import PreparedValueEffort21
        model,_=hand_plant._model(FoundationConfig())
        # Reuse the exact pre-existing29 author assertions as additional coverage,
        # with only constructor substitution; no production body is modified.
        import review.verify_v13_native_effort21_author_v1 as original_author
        base_module.NativeEffort21=PreparedValueEffort21
        try:legacy_checks=original_author.run()
        finally:base_module.NativeEffort21=NativeEffort21
        check('all_original_author_assertions_on_cached_servo',legacy_checks['passed'],len(legacy_checks['checks']))
        # Explicit fixture ledger is selected before comparisons or timings.
        proto=NativeEffort21(model);q=model.qpos0[proto.servo.qadr].copy();v=np.zeros(25)
        axis=5;aid=int(proto.servo.aids[axis]);oid=int(proto.servo.outadr[axis]);fixtures=[]
        for label in ('standard','negative_gain','zero_constant','force_kink','large_bias_cancellation'):
            m=copy(model)
            if label=='negative_gain':m.actuator_gainprm[aid,:3]=[-1.,0.,0.]
            elif label=='zero_constant':m.actuator_gainprm[aid,:3]=0.;m.actuator_biasprm[aid,:3]=[.7,0.,0.]
            elif label=='force_kink':m.actuator_forcerange[oid]=[-.01,.01]
            elif label=='large_bias_cancellation':m.actuator_gainprm[aid,:3]=[1.,0.,0.];m.actuator_biasprm[aid,:3]=[1e16,-1.,.5];m.actuator_forcelimited[oid]=False
            fixtures.append((label,m))
        for label,m in fixtures:
            b=NativeEffort21(m);p=PreparedValueEffort21(m);ref=np.zeros(21);lo=ref.copy();hi=ref.copy();lo[axis]=-.2;hi[axis]=.2
            for x in (0.,-0.,.01,np.nextafter(.01,-np.inf),np.nextafter(.01,np.inf),-.2,.2):
                r=ref.copy();r[axis]=x;engine=b.actor.encode(Reference21(r,1.)).values
                for repeat in range(2):
                    bs=b.servo.evaluate(q,v,engine,time_s=1.);ps=p.servo.evaluate(q,v,engine,time_s=1.)
                    for f in fields(bs):
                        if f.name=='diagnostics':continue
                        if ps.diagnostics.get('prepared_value_only') and f.name in ('reference_jacobian25x21','actuation_stiffness25x25','actuation_damping25x25','derivative_status'):
                            check(label+f'/value_only/{x.hex()}/{repeat}/{f.name}',all(v=='NOT_COMPUTED_VALUE_ONLY' for v in ps.derivative_status.values()) if f.name=='derivative_status' else getattr(ps,f.name) is None)
                            continue
                        check(label+f'/sample/{x.hex()}/{repeat}/{f.name}',same(getattr(bs,f.name),getattr(ps,f.name)))
                    bd=bs.diagnostics;pd={k:val for k,val in ps.diagnostics.items() if k not in ('native_value_arithmetic','prepared_transmission_cache_hit','prepared_cache_private_data_scope','prepared_value_only','derivative_computation')}
                    pd['pipeline']=bd['pipeline']
                    check(label+f'/sample/{x.hex()}/{repeat}/diagnostics_except_cache_pipeline',same(bd,pd))
            current=Reference21(ref,1.);domargs=dict(q25=q,v25=v,robot_axes25=b.robot_axes25,time_s=1.,current_reference=current,nominal_reference=current,
                actual_engine=b.actor.encode(current),reference_lower21=lo,reference_upper21=hi)
            bd=b.build_domain(**domargs);pd=p.build_domain(**domargs)
            check(label+'/domain_available',bd.available and pd.available,(bd.reason,pd.reason))
            for f in fields(bd):
                if f.name.startswith('_') or f.name=='diagnostics':continue
                check(label+'/domain/'+f.name,same(getattr(bd,f.name),getattr(pd,f.name)))
            bdiag=dict(bd.diagnostics);pdiag=dict(pd.diagnostics);bdiag.pop('model_signature');pdiag.pop('model_signature')
            check(label+'/domain/only_model_signature_diff',same(bdiag,pdiag))
            request=np.zeros(21);request[axis]=.05;corr=.1*bd.transmission21x25[axis];sr=np.ones(25);sd=np.ones(21);sr[corr!=0]=3.;sd[axis]=2.
            kw=dict(p0_effort21=bd.p0_effort21,delta_effort21=request,correction25=corr,residual_scale25=sr,delta_scale21=sd)
            br=b.solve(bd,**kw);pr=p.solve(pd,**kw)
            # to_dict contains all candidate values/scores/selection, native
            # bytes, clock and domain outcomes. There is NO result exclusion.
            check(label+'/all_inverse_result_and_candidate_fields_exact',same(br.to_dict(),pr.to_dict()),(br.status,pr.status))
            check(label+'/cached_only_one_transmission_per_qv',p.cache_diagnostics()['transmission_misses']==1,p.cache_diagnostics())
        disabled=copy(model);disabled.opt.disableactuator=1<<5;disabled.actuator_group[aid]=5
        db=NativeEffort21(disabled);dp=PreparedValueEffort21(disabled)
        check('initial_unavailable_mode_preserves_F067_reason_priority',same(db.servo.evaluate(None,None,None).reasons,dp.servo.evaluate(None,None,None).reasons))
        # Explicit q/v byte changes must miss. Independent returned snapshots
        # remain readonly; no subsequent command changes a previous snapshot.
        p=PreparedValueEffort21(model);u=np.zeros(21);a=p.servo.evaluate(q,v,u);saved=a.predicted_effort21.tobytes()
        q2=q.copy();q2[0]=np.nextafter(q2[0],np.inf);p.servo.evaluate(q2,v,u)
        v2=v.copy();v2[0]=.1;p.servo.evaluate(q2,v2,u)
        check('q_and_v_changes_invalidate_cached_transmission',p.cache_diagnostics()['transmission_misses']==3,p.cache_diagnostics())
        check('earlier_returned_native_array_not_mutated',saved==a.predicted_effort21.tobytes() and not a.predicted_effort21.flags.writeable)
        check('cached_array_readonly',not any(x.flags.writeable for x in p.servo._cache_value))
        # Fresh F067 refuses this mode change; base6f missing-group guard is an
        # explicitly expected stricter refusal, never excluded numeric drift.
        m=copy(model);m.opt.disableactuator=1<<5;p=PreparedValueEffort21(m);ref=Reference21(np.zeros(21),1.)
        d=p.build_domain(q25=q,v25=v,robot_axes25=p.robot_axes25,time_s=1.,current_reference=ref,nominal_reference=ref,
            actual_engine=p.actor.encode(ref),step_lower21=np.full(21,-.001),step_upper21=np.full(21,.001))
        check('initial_enabled_group_domain_available',d.available,d.reason)
        p.servo.model.actuator_group[p.servo.aids[0]]=5
        rr=p.solve(d,p0_effort21=d.p0_effort21,delta_effort21=np.zeros(21),correction25=np.zeros(25),residual_scale25=np.ones(25),delta_scale21=np.ones(21))
        check('changed_group_mode_refuses_domain_and_clears_cache',not rr.available and rr.status=='UNAVAILABLE_MODEL' and not p.cache_diagnostics()['cache_populated'],rr.reason)
        # Public states: all observations are loaded once per saved archive;
        # same actual q/v/native engine and finite20ms envelope for both solvers.
        selections=[(rel,t) for rel in ('output/development_v12_native_pilot_v1/runs/nominal__fusion_effort__617901/trace.npz','output/development_v12_native_hold_seed617902_v1/runs/collision__fusion_effort__617902/trace.npz','output/development_v12_native_transport_v1/runs/friction_drop__ratio_effort__617902/trace.npz') for t in (7.5,8.,8.5,9.,9.5,10.,10.5,11.,11.5)]
        arrays={}
        for rel,t in selections:
            if rel not in arrays:
                with np.load(ROOT/rel,allow_pickle=False) as z:arrays[rel]=(z['time'],z['observation_json'])
            ts,obs=arrays[rel];i=int(np.argmin(abs(ts-t)));frame=json.loads(str(obs[i]));qreal=np.r_[frame['raw_joint'],frame['lift']];vreal=np.r_[frame['raw_joint_velocity'],frame['lift_velocity_m_s']]
            for workload in ('zero','nonzero'):
                timings={};results={};counts={}
                for label,cls in (('base',NativeEffort21),('prepared',PreparedValueEffort21)):
                    c=cls(model);engine=EnginePort21(frame['robot_actuator_target21'],float(ts[i]),c.actor.lift_bias_m);ref=c.actor.decode(engine)
                    step=np.r_[np.full(18,1.2),.035,.12,.12]*.020
                    start=time.perf_counter_ns();d=c.build_domain(q25=qreal,v25=vreal,robot_axes25=c.robot_axes25,time_s=float(ts[i]),current_reference=ref,nominal_reference=ref,
                        actual_engine=engine,step_lower21=-step,step_upper21=step);build=(time.perf_counter_ns()-start)/1e6
                    check(f'public/{t}/{workload}/{label}/domain_available',d.available,d.reason)
                    req=np.zeros(21);corr=np.zeros(25)
                    if workload=='nonzero':req[5]=.05;corr=d.transmission21x25[5]*.1
                    start=time.perf_counter_ns();r=c.solve(d,p0_effort21=d.p0_effort21,delta_effort21=req,correction25=corr,residual_scale25=np.ones(25),delta_scale21=np.ones(21));solve=(time.perf_counter_ns()-start)/1e6
                    timings[label]=dict(build_ms=build,solve_ms=solve,total_ms=build+solve);results[label]=r
                    counts[label]=dict(native_evaluations_solve=r.diagnostics.get('native_evaluations'),candidates=sum(len(x['candidates']) for x in r.diagnostics.get('finite_candidate_solutions',[])),
                        cache=c.cache_diagnostics() if label=='prepared' else None)
                check(f'public/{t}/{workload}/full_inverse_result_exact',same(results['base'].to_dict(),results['prepared'].to_dict()))
                profiles.append(dict(source=rel,time_s=float(ts[i]),index=i,workload=workload,timings=timings,counts=counts))
        check('no_forbidden_physics_FK_contact_calls',not forbidden,forbidden)
        check('all_watched_sources_unchanged',before=={p:sha(ROOT/p) for p in watched})
        report=dict(schema='prepared_value_effort21_author_v1',passed=True,checks=checks,profiles=profiles,original_author_checks=legacy_checks['checks'])
    except Exception as exc:report=dict(schema='prepared_value_effort21_author_v1',passed=False,checks=checks,profiles=profiles,error=repr(exc),traceback=traceback.format_exc())
    report.update(source_sha256=before,forbidden_calls=forbidden,physics_steps=0,writer_calls=0,
        scope='No zero shortcut. First and final snapshot use unchanged full F067; intermediate values use exact native operation order. Only explicit final full-recheck field excluded from result comparison, and required passed. All finite candidate values and scores unchanged.',
        timing_scope='Serial one-host cold solver per public state/workload; setup/IO/hash/compile excluded; all calls retained. No repeat-best selection or real-time guarantee.')
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),output=str(out),sha256=sha(out),profiles=profiles),ensure_ascii=False))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

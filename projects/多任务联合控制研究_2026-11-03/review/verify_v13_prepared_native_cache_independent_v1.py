"""Small cache/domain negative oracles; F067 reset/tendon/transmission only."""
from pathlib import Path
from copy import copy
import hashlib,importlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/v13_prepared_native_cache_independent_v1.json'

def main():
    names=('v13_joint/prepared_native_effort21.py','v13_joint/native_effort21.py','v11/robot_servo_effort.py','v13_a1/codec.py')
    payloads={name:(ROOT/name).read_bytes() for name in names}
    sys.path.insert(0,str(ROOT));import research_imports,mujoco
    original={};calls={}
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_fwdPosition','mj_fwdVelocity','mj_kinematics','mj_comPos','mj_camlight','mj_collision','mj_contactForce'):
        if hasattr(mujoco,name):
            original[name]=getattr(mujoco,name);calls[name]=0
            def deny(*a,_name=name,**kw):calls[_name]+=1;raise AssertionError('forbidden '+_name)
            setattr(mujoco,name,deny)
    from foundation_config import FoundationConfig
    import hand_plant
    old_init=hand_plant.HandPlant.__init__;plant_calls=[]
    def deny_plant(*a,**k):plant_calls.append(True);raise AssertionError('plant constructor forbidden')
    hand_plant.HandPlant.__init__=deny_plant
    from v13_a1.codec import Reference21,ReferenceActor21
    old_apply=ReferenceActor21.apply;writer_calls=[]
    def deny_writer(*a,**k):writer_calls.append(True);raise AssertionError('writer forbidden')
    ReferenceActor21.apply=deny_writer
    prepared=importlib.import_module('v13_joint.prepared_native_effort21');base=importlib.import_module('v13_joint.native_effort21')
    checks=[];cases=[]
    def check(name,yes,detail=None):checks.append(dict(name=name,passed=bool(yes),detail=detail))
    try:
        model,xml_sha=hand_plant._model(FoundationConfig());core=prepared.PreparedNativeEffort21(model);frozen=base.NativeEffort21(model)
        q=model.qpos0[core.servo.qadr].copy();v=np.zeros(25);r=np.zeros(21);r[6]=.08;reference=Reference21(r,3.25);u=core.actor.encode(reference).values
        snapshot=core.servo.evaluate(q,v,u,time_s=3.25);again=core.servo.evaluate(q,v,u,time_s=3.255);truth=frozen.servo.evaluate(q,v,u,time_s=3.255)
        check('same_qv_reuses_transmission_not_value_surrogate',core.cache_diagnostics()['transmission_misses']==1 and core.cache_diagnostics()['transmission_hits']==1 and again.predicted_effort21.tobytes()==truth.predicted_effort21.tobytes())
        check('snapshot_arrays_do_not_alias_private_cache',not np.shares_memory(again.transmission21x25,core.servo._cache_value[0]) and not again.transmission21x25.flags.writeable)
        before=again.predicted_effort21.copy();again.transmission21x25.setflags(write=True);again.transmission21x25[:]=123.
        after=core.servo.evaluate(q,v,u)
        check('returned_snapshot_mutation_cannot_poison_cache',after.predicted_effort21.tobytes()==before.tobytes() and not np.any(after.transmission21x25==123.))
        q2=q.copy();q2[2]+=.007;changed_q=core.servo.evaluate(q2,v,u);expected=frozen.servo.evaluate(q2,v,u)
        check('q_change_invalidates_cache',core.cache_diagnostics()['qv_invalidations']>=1 and changed_q.predicted_effort21.tobytes()==expected.predicted_effort21.tobytes())
        v2=v.copy();v2[2]=.019;changed_v=core.servo.evaluate(q2,v2,u);expected=frozen.servo.evaluate(q2,v2,u)
        check('v_change_invalidates_cache',core.cache_diagnostics()['qv_invalidations']>=2 and changed_v.predicted_effort21.tobytes()==expected.predicted_effort21.tobytes())
        bad=core.servo.evaluate(None,v,u)
        check('unknown_q_not_cached_zero_or_old_snapshot',bad.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA' and bad.predicted_effort21 is None)
        # Each mutation is a new isolated model/solver fixture, never production.
        mutations={
            'robot_actuator_group':lambda c:setattr(c.servo.model,'actuator_group',c.servo.model.actuator_group),
        }
        guard_results=[]
        for kind in ('actuator_group','disableactuator','disableflags','external_routing'):
            c=prepared.PreparedNativeEffort21(model);c.servo.evaluate(q,v,c.actor.encode(reference).values)
            m=c.servo.model;aid=int(c.servo.aids[6])
            if kind=='actuator_group':m.actuator_group[aid]=int(m.actuator_group[aid])+1
            elif kind=='disableactuator':m.opt.disableactuator=int(m.opt.disableactuator)|(1<<int(m.actuator_group[aid]))
            elif kind=='disableflags':m.opt.disableflags=int(m.opt.disableflags)|int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            else:
                other=next(i for i in range(m.nactuator) if i not in set(c.servo.aids));m.actuator_trnid[other,0]=int(c.servo.jids[0])
            blocked=c.servo.evaluate(q,v,c.actor.encode(reference).values)
            okay=blocked.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA' and blocked.predicted_effort21 is None and c.cache_diagnostics()['model_invalidations']>=1 and not c.cache_diagnostics()['cache_populated']
            guard_results.append(dict(mutation=kind,blocked=bool(okay),reason=list(blocked.reasons)))
        check('all_model_admission_mutations_invalidate',all(x['blocked'] for x in guard_results),guard_results)
        external=copy(model);isolated=prepared.PreparedNativeEffort21(external);first=isolated.servo.evaluate(q,v,isolated.actor.encode(reference).values)
        external.actuator_gainprm[int(isolated.servo.aids[6]),0]*=1.1;second=isolated.servo.evaluate(q,v,isolated.actor.encode(reference).values)
        check('source_model_clone_mutation_not_private_alias',first.predicted_effort21.tobytes()==second.predicted_effort21.tobytes() and isolated.servo.model is not external)
        c=prepared.PreparedNativeEffort21(model);lo=r.copy();hi=r.copy();lo[6]=0.;hi[6]=.16
        d=c.build_domain(q25=q,v25=v,robot_axes25=c.robot_axes25,time_s=3.25,current_reference=reference,nominal_reference=reference,
            actual_engine=c.actor.encode(reference),reference_lower21=lo,reference_upper21=hi)
        other=prepared.PreparedNativeEffort21(model);result=other.solve(d,p0_effort21=d.p0_effort21,delta_effort21=np.zeros(21),correction25=np.zeros(25),residual_scale25=np.ones(25),delta_scale21=np.ones(21))
        check('prepared_domains_cannot_cross_instance',d.available and not result.available and result.status=='UNAVAILABLE_SOURCE')
        check('inherits_base_candidate_build_and_solve_body',prepared.PreparedNativeEffort21.build_domain is base.NativeEffort21.build_domain and prepared.PreparedNativeEffort21.solve is base.NativeEffort21.solve)
        check('f067_native_evaluate_is_delegated_not_overridden_math','super().evaluate' in payloads[names[0]].decode('utf-8') and not any(x in payloads[names[0]].decode('utf-8') for x in ('beta =','beta=','unbounded_effort21=')))
        cases=dict(cache=core.cache_diagnostics(),mutation_guards=guard_results)
    finally:
        ReferenceActor21.apply=old_apply;hand_plant.HandPlant.__init__=old_init
        for name,fn in original.items():setattr(mujoco,name,fn)
    stable=all((ROOT/name).read_bytes()==data for name,data in payloads.items())
    check('sources_stable_during_actual_checks',stable)
    check('no_forbidden_calls_or_writers',not any(calls.values()) and not plant_calls and not writer_calls)
    value=dict(schema='v13_prepared_native_cache_independent_v1',role='independent_small_cache_negative_peer',passed=all(x['passed'] for x in checks),checks=checks,count=len(checks),
        source_sha256={name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()},checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_stable=stable,static_model_compiled=1,static_xml_sha256=xml_sha,physics_steps=0,plant_calls=len(plant_calls),writer_calls=len(writer_calls),forbidden_calls=calls,cases=cases,
        limitations=['Cache/domain negatives and selected F067 native value bytes only; parent owns broad bit regression/profile and foundation owns all-candidate review.',
            'Private model/observation snapshots are isolated; runtime Python hot-patching or editing source files is outside this model guard and must be prevented by the outer frozen-source identity contract.',
            'Known q/v/model identity is required; no future robot/pressure, physical feasibility or real-time claim.'])
    assert not OUT.exists();OUT.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),passed=value['passed'],checks=len(checks),failed=[x['name'] for x in checks if not x['passed']],prepared_sha=value['source_sha256'][names[0]]),ensure_ascii=False))
    if not value['passed']:raise SystemExit(1)

if __name__=='__main__':main()

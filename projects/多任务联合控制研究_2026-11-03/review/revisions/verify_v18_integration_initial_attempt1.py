"""Independent few-packet V18 wiring checks. Never advances a plant.

Constructors may initialize/forward a model; archived public replay then denies
forward/contact/writer calls. Native algebra and robot-only FK remain allowed.
Governor's analytic dynamics scope is reviewed separately, not duplicated here.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import fields
from copy import deepcopy
from unittest.mock import patch
import argparse, hashlib, json, sys, traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    checks=[];calls=[];forbidden=[];truth=[];before={};records=[];constructors=[]
    def ck(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=plain(detail)))
        if not ok:raise AssertionError(name+' '+str(detail))
    def reject(fn):
        try:fn()
        except (ValueError,TypeError):return True
        return False
    class PublicFrame(SimpleNamespace):
        def __getattribute__(self,name):
            if name in {'object_position','object_pose','object_velocity','object_mass','object_friction','target_id','target_contact_id','is_target','true_qpos','raw_qpos'}:
                truth.append(name);raise AssertionError('private external truth '+name)
            return super().__getattribute__(name)
    try:
        import research_imports,mujoco
        from v18_effort_transfer.runtime import ReleaseSession,source_identity
        from v18_effort_transfer.adapter import EffortContactAdapter
        import v18_effort_transfer.supervisor as sm
        from v18_effort_transfer.supervisor import ReleaseSupervisor
        from v18_effort_transfer.effort_task import evaluate_effort_task
        from v17b_full_evidence.runtime import source_identity as parent_identity
        from v13_a1.codec import ReferenceActor21,Reference21
        from v13_joint.lifecycle import ReleaseEntry
        from v13_joint.intent_qp import IntentQPResult
        from v16b_release_goal.goal import release_clearance_goal
        from v16_release_geometry.transfer_observer import TransferObserver
        from v15_support.receiver_sensor import ReceiverMount,ReceiverMeasurement
        from v14_coord.design import reference_rates
        import hand_plant
        from foundation_config import FoundationConfig,OPEN
        aux=json.loads((ROOT/'review/v17b_delivery_auxiliary_readout_v1.json').read_text('utf-8'))
        before=source_identity()
        ck('parent136_same_as_executed80',parent_identity()==aux['execution_source_sha256'])
        ck('new_closure_contains_exact_parent',all(before[k]==v for k,v in parent_identity().items()))
        for name in ('mj_step','mj_step1','mj_step2'):
            if hasattr(mujoco,name):
                def deny(*a,_n=name,**k):forbidden.append(_n);raise AssertionError(_n)
                setattr(mujoco,name,deny)
        for method in ('ratio_effort','fixed_effort','fusion_effort'):
            s=ReleaseSession(method=method,scenario='nominal',seed=619741,split='robustness',motion_mode='hold',task_policy='integrated',receiver_policy='coordinated')
            ck('constructor_no_advance/'+method,s.tick==0 and s.time==0 and s.sensors.samples==0 and len(s.trace)==0)
            ck('constructor_new_adapter/'+method,type(s.release_adapter) is EffortContactAdapter)
            constructors.append(dict(method=method,tick=s.tick,time_s=s.time,sensor_samples=s.sensors.samples))
        del s
        for name in ('mj_forward','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
            if hasattr(mujoco,name):
                def deny(*a,_n=name,**k):forbidden.append(_n);raise AssertionError(_n)
                setattr(mujoco,name,deny)
        def no_writer(*a,**k):forbidden.append('actor.apply');raise AssertionError('writer')
        ReferenceActor21.apply=no_writer
        model,_=hand_plant._model(FoundationConfig());actor=ReferenceActor21(model)
        lo=actor.ctrl_range21[:,0].copy();hi=actor.ctrl_range21[:,1].copy()
        lo[18]-=actor.lift_bias_m;hi[18]-=actor.lift_bias_m;lo[18]=max(0.,lo[18]);hi[18]=min(.1,hi[18])
        for j,n in ((19,'rh_WRJ2'),(20,'rh_WRJ1')):
            k=model.joint(n).id
            if model.jnt_limited[k]:lo[j]=max(lo[j],model.jnt_range[k,0]);hi[j]=min(hi[j],model.jnt_range[k,1])
        hz=float(model.geom_size[model.geom('support').id,2])
        mount=ReceiverMount('support',('floor','support_post'),np.array([0.,0.,hz]),hz,float(np.cos(np.pi/4)))
        run=Path(next(r['run'] for r in aux['records'] if r['record_key']==['full_V17','nominal','hold',619741]))
        meta=json.loads((run/'metadata.json').read_text('utf-8'))
        with np.load(run/'trace.npz',allow_pickle=False) as z:
            times=z['time'];observations=z['observation_json'];receivers=z['receiver_measurement_json']
        receiver_fields={f.name for f in fields(ReceiverMeasurement)}
        def packet(i):
            f=PublicFrame(**json.loads(str(observations[i])))
            for k,v in list(vars(f).items()):
                if isinstance(v,list) and k not in ('skin_patches','robot_actuator_axes21'):setattr(f,k,np.asarray(v))
            raw=json.loads(str(receivers[i]));r=ReceiverMeasurement(**{k:v for k,v in raw.items() if k in receiver_fields})
            return f,r
        adapter=EffortContactAdapter(model,open18=np.concatenate(OPEN),receiver_mount=mount)
        graph=None;saved={};selected={12.02,12.35,13.25}
        for i in np.flatnonzero((times>=11.8-1e-9)&(times<=13.25+1e-9)):
            f,r=packet(int(i));adapter.observe(f)
            if f.time<12.-1e-9:continue
            current=np.asarray(f.robot_reference21,float)
            ck('archived_engine_reference_exact/'+str(f.time),actor.encode(Reference21(current,f.time)).values.tobytes()==np.asarray(f.robot_actuator_target21,float).tobytes())
            if graph is None:
                entry=ReleaseEntry(12.,Reference21(current,12.),1)
                goal=release_clearance_goal(entry,preparation_open18=np.concatenate(OPEN),mechanical_lower21=lo,mechanical_upper21=hi)
                prior=adapter.begin_release(12.)
                observer=TransferObserver(request_time_s=12.,receiver_half_size_xy_m=adapter.receiver_half_size_xy_m,receiver_up_axis3=adapter.receiver_up_axis3,prior=prior)
                graph=ReleaseSupervisor(entry,open18=goal.reference18,observer=observer,robot_model=model,limits5=meta['limits'],budget=meta['budget'],force_margin=.02)
            transfer=adapter.transfer_inputs(r);public=graph.observe(frame=f,transfer_inputs=transfer,current21=current)
            if not any(abs(f.time-t)<1e-9 for t in selected):continue
            count=len(graph.decision_evidence);ref,debug=graph.action(frame=f,adapter=adapter,current21=current,lower21=lo,upper21=hi,generation=1)
            ev=graph.decision_evidence[-1];rates=reference_rates('release')*.005
            ck('actual5ms_domain/'+str(f.time),np.all(ref.values>=np.maximum(lo,current-rates)) and np.all(ref.values<=np.minimum(hi,current+rates)))
            ck('no_future_pressure_or_placement/'+str(f.time),not ev['future_pressure_prediction_available'] and not ev['placement_certified'])
            if ev['effort_goal_realized']:
                n=ev['native'];q=np.asarray(f.robot_position25,float);v=np.asarray(f.robot_velocity25,float)
                full=graph.native.servo.evaluate_full(q,v,actor.encode(ref).values,time_s=f.time)
                np0=np.asarray(ev['domain']['p0_effort21']);native=np.asarray(n['native_effort21']);pr=ev['qp_problem'];zz=np.asarray(ev['qp']['zP']);delta=native-np0
                ck('fresh_F067_final_native/'+str(f.time),full.predicted_effort21.tobytes()==native.tobytes())
                A=np.asarray(pr['A21x25']);J=np.asarray(pr['Jpx25']).reshape((-1,25));y=np.asarray(pr['y_p']);tau=np.asarray(pr['tau_extra25']);sr=np.asarray(pr['scales']['residual25']);sd=np.asarray(pr['scales']['delta21']);sf=np.asarray(pr['scales']['intentP']);E=np.asarray(pr['extra_task_rows']['matrix']);b=np.asarray(pr['extra_task_rows']['target'])
                residual=A.T@delta-(J.T@(zz-y)+tau)
                cost=float((sr*residual)@(sr*residual)+(sf*(zz-np.asarray(pr['z_intent_p'])))@(sf*(zz-np.asarray(pr['z_intent_p'])))+(sd*delta)@(sd*delta)+(E@np.r_[delta,zz]-b)@(E@np.r_[delta,zz]-b))
                ck('independent_fixed_p0_canonical_residual/'+str(f.time),residual.tobytes()==np.asarray(n['residual25']).tobytes())
                ck('independent_full_same_task_cost/'+str(f.time),np.float64(cost).tobytes()==np.float64(ev['native_full_task_score']['objective']).tobytes())
            else:ck('honest_unaccepted_fallback/'+str(f.time),ev['fallback']=='legal_governed_posture_without_joint_task_certificate')
            records.append(dict(time_s=f.time,phase=graph.phase,patches=len(ev['qp_problem']['y_p']) if ev['qp_problem'] else None,native=ev['effort_goal_realized'],fallback=ev['fallback'],qp=None if ev['qp'] is None else ev['qp']['status']))
            if abs(f.time-12.02)<1e-9:
                repeat=graph.observe(frame=f,transfer_inputs=transfer,current21=current)
                ck('supervisor_duplicate_idempotent',json.dumps(plain(repeat),sort_keys=True)==json.dumps(plain(public),sort_keys=True))
                bad=deepcopy(f);bad.normal=bad.normal.copy();bad.normal[0]+=.1
                ck('supervisor_changed_duplicate_rejected',reject(lambda:graph.observe(frame=bad,transfer_inputs=transfer,current21=current)))
                oldframe=deepcopy(f);oldframe.time-=.005
                ck('supervisor_time_rewind_rejected',reject(lambda:graph.observe(frame=oldframe,transfer_inputs=transfer,current21=current)))
                with patch.object(sm,'solve_effort_task',return_value=IntentQPResult('unknown','independent_failure_injection')):
                    fallback,fd=graph.action(frame=f,adapter=adapter,current21=current,lower21=lo,upper21=hi,generation=1)
                ck('forced_unknown_solver_is_honest_legal_fallback',not fd['effort_goal_realized'] and fd['fallback']=='legal_governed_posture_without_joint_task_certificate' and np.all(fallback.values>=np.maximum(lo,current-rates)) and np.all(fallback.values<=np.minimum(hi,current+rates)))
                saved=dict(frame=deepcopy(f),receiver=r,transfer=transfer,current=current.copy(),graph=graph,adapter=adapter,public=deepcopy(public),reference=ref,decision=deepcopy(ev))
        ck('three_real_public_decisions',len(records)==3)
        ck('at_least_one_real_native_full_checked',any(r['native'] for r in records))
        # Independent task-level geometry fixtures use exactly the saved public inputs.
        f=saved['frame'];current=saved['current'];ad=EffortContactAdapter(model,open18=np.concatenate(OPEN),receiver_mount=mount)
        # Prime causal acquisition history with public samples only.
        for i in np.flatnonzero((times>=11.8-1e-9)&(times<=f.time+1e-9)):ad.observe(packet(int(i))[0])
        base=ad.current_contact_intents(f);ck('real_pressure_geometry_present',len(base['y'])>0)
        changed=deepcopy(f);digit=int(base['digits'][0]);changed.valid=changed.valid.copy();changed.valid[digit]=False
        partial=ad.current_contact_intents(changed)
        ck('partial_invalid_omitted_not_zero',len(partial['y'])<len(base['y']) and not np.any(partial['digits']==digit) and len(partial['y'])==np.count_nonzero(base['digits']!=digit))
        flat=deepcopy(f);flat.robot_reference21=np.asarray(f.robot_reference21,float).copy();flat.robot_reference21[:18]=hi[:18];flat.robot_actuator_target21=actor.encode(Reference21(flat.robot_reference21,flat.time)).values
        snap=graph.native.servo.evaluate(np.asarray(f.robot_position25),np.asarray(f.robot_velocity25),flat.robot_actuator_target21,time_s=f.time)
        flat_task=ad.current_contact_intents(flat)
        ck('real_flat_force_clip_is_present',any(snap.diagnostics['force_clipped21']))
        ck('flat_motorclip_does_not_erase_pressure_J',np.array_equal(flat_task['J'],base['J']) and np.array_equal(flat_task['y'],base['y']))
        split=deepcopy(f);index=base['diagnostics']['patch_indices'][0];raw=deepcopy(split.skin_patches[index]);raw['normal']/=2.;split.skin_patches[index]=raw;split.skin_patches.insert(index+1,deepcopy(raw))
        sp=ad.current_contact_intents(split)
        ck('split_tangent_one_device_not_repeated',np.allclose(sp['tau_tangent25'],base['tau_tangent25'],rtol=2e-14,atol=2e-14) and sp['diagnostics']['tangent_active_digits']==base['diagnostics']['tangent_active_digits'])
        ck('split_normal_JTy_invariant',np.allclose(sp['J'].T@sp['y'],base['J'].T@base['y'],rtol=2e-14,atol=2e-14))
        bp=deepcopy(saved['decision']['qp_problem']);spp=deepcopy(bp)
        def sf(y,digits):
            return np.array([1./(max(1.,bp['caps5'][k])*np.sqrt(y[i]/sum(y[j] for j in range(len(y)) if digits[j]==k))) for i,k in enumerate(digits)])
        ck('production_normal_scale_is_pressure_share_normalized',np.allclose(bp['scales']['intentP'],sf(base['y'],base['digits']),rtol=8*np.finfo(float).eps,atol=0.))
        spp.update(Jpx25=sp['J'],y_p=sp['y'],patch_digit_p=sp['digits'])
        # This first SEARCH packet has unchanged normal prior; no future load is used.
        ck('split_fixture_is_current_normal_prior',np.array_equal(bp['z_intent_p'],base['y']))
        spp['z_intent_p']=sp['y'];spp['scales']['intentP']=sf(sp['y'],sp['digits'])
        E=np.zeros((21,21+len(sp['y'])));E[:,:21]=np.asarray(bp['extra_task_rows']['matrix'])[:,:21]
        spp['extra_task_rows']['matrix']=E
        oldz=np.asarray(saved['decision']['qp']['zP']);baseids=base['diagnostics']['patch_indices'];pos=baseids.index(index)
        newz=np.r_[oldz[:pos],oldz[pos]/2.,oldz[pos]/2.,oldz[pos+1:]]
        oldeval=evaluate_effort_task(bp,np.asarray(saved['decision']['qp']['d21']),oldz)
        neweval=evaluate_effort_task(spp,np.asarray(saved['decision']['qp']['d21']),newz)
        ck('split_full_same_problem_objective_invariant',abs(oldeval['objective']-neweval['objective'])<=128*np.finfo(float).eps*max(1.,abs(oldeval['objective'])))
        spqp=sm.solve_effort_task(**spp);accepted=sm.check_same_problem_candidate(spp,spqp)
        ck('split_same_problem_has_accepted_goal',accepted['accepted'])
        ck('split_effort_and_digit_totals_invariant',np.allclose(spqp.d21,saved['decision']['qp']['d21'],atol=2e-8,rtol=2e-7) and all(np.isclose(sum(spqp.zP[sp['digits']==k]),sum(oldz[base['digits']==k]),atol=2e-8,rtol=2e-7) for k in range(5)))
        domain=graph.native.build_domain(q25=f.robot_position25,v25=f.robot_velocity25,robot_axes25=graph.native.robot_axes25,time_s=f.time,current_reference=Reference21(current,f.time),nominal_reference=Reference21(current,f.time),actual_engine=actor.encode(Reference21(current,f.time)),reference_lower21=np.asarray(saved['decision']['actual_reference_lower21']),reference_upper21=np.asarray(saved['decision']['actual_reference_upper21']))
        spnative=graph.native.solve(domain,p0_effort21=domain.p0_effort21,delta_effort21=spqp.d21,correction25=sm.native_correction25(spp,spqp.zP),residual_scale25=np.asarray(spp['scales']['residual25']),delta_scale21=np.asarray(spp['scales']['delta21']),requested_target_effort21=spqp.p21)
        ck('split_realized_reference_invariant_with_declared_numerical_bound',spnative.available and np.allclose(spnative.reference21.values,saved['reference'].values,rtol=2e-7,atol=2e-8),dict(atol=2e-8,rtol=2e-7,max_abs=float(np.max(abs(spnative.reference21.values-saved['reference'].values))) if spnative.available else None))
        ck('adapter_duplicate_idempotent',ad.observe(f) is ad.hand_sample)
        bad=deepcopy(f);bad.normal=bad.normal.copy();bad.normal[0]+=.1
        ck('adapter_changed_duplicate_rejected',reject(lambda:ad.observe(bad)))
        rewind=packet(int(np.flatnonzero(abs(times-(f.time-.005))<1e-9)[0]))[0]
        ck('adapter_time_rewind_rejected',reject(lambda:ad.observe(rewind)))
        ck('forbidden_truth_and_pipeline_calls_zero',not truth and not forbidden)
        ck('all_executed_parent_and_new_sources_unchanged',source_identity()==before)
        result=dict(passed=True)
    except Exception as exc:result=dict(passed=False,error=repr(exc),traceback=traceback.format_exc())
    result.update(schema='v18_integration_independent_v1',checks=checks,records=records,constructors=constructors,source_sha256=before,checker_sha256=sha(__file__),forbidden_calls=forbidden,forbidden_truth_reads=truth,physics_steps=0,scope='Three genuine zero-tick constructors; one archived public stream cached once, three off-policy release decisions; no physics/outcome/stop certificate; governor analytic gates reviewed separately.')
    with out.open('x',encoding='utf-8') as h:json.dump(plain(result),h,ensure_ascii=False,indent=2,allow_nan=False);h.write('\n')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),records=len(records),out=str(out)),ensure_ascii=False));print(result.get('traceback',''))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())

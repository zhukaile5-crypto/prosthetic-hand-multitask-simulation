"""Finite V23 integration-author archived-public replay; not closed-loop data.

Original legacy interpreter, static known robot, no plant/step/forward/contact
query. Real public prehistory, adapter/private robot FK, task solver, native
algebra, exact emitted-task score, real actor into a private plain22 buffer,
and supervisor receipt. Unknown geometry is an explicit software injection.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import fields
from unittest.mock import patch
import argparse,hashlib,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bits(a,b):return np.asarray(a,float).shape==np.asarray(b,float).shape and np.asarray(a,float).tobytes()==np.asarray(b,float).tobytes()
def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out;assert not out.exists()
    checks=[];records=[];forbidden=[];truth=[];source={};numeric={};inputs={}
    def ck(name,good,**extra):
        checks.append(dict(name=name,passed=bool(good),**plain(extra)))
        if not good:raise AssertionError(name)
    def deny(*a,**kw):forbidden.append('forbidden_pipeline');raise AssertionError(forbidden[-1])
    class Frame(SimpleNamespace):
        def __getattribute__(self,key):
            if key in {'object_position','object_velocity','object_pose','object_mass','object_friction','target_id','is_target','target_contact_id','true_qpos','raw_qpos'}:truth.append(key);raise AssertionError('truth:'+key)
            return super().__getattribute__(key)
    try:
        from v22_joint_task_fusion.numerics import ensure_numeric_environment,numerical_identity
        ensure_numeric_environment()
        import research_imports,mujoco,hand_plant
        from foundation_config import FoundationConfig,OPEN
        from v13_a1.codec import ReferenceActor21,Reference21
        from v13_joint.lifecycle import ReleaseEntry
        from v15_support.receiver_sensor import ReceiverMount,ReceiverMeasurement
        from v16_release_geometry.transfer_observer import TransferObserver
        from v16b_release_goal.goal import release_clearance_goal
        from v20_contact_exit.adapter import ContactExitAdapter
        from v23_task_execution.supervisor import ReleaseSupervisor
        from v23_task_execution.runtime import source_identity
        source=source_identity(['ratio_effort']);numeric=numerical_identity()
        pipeline=('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce')
        with patch.multiple(mujoco,**{name:deny for name in pipeline if hasattr(mujoco,name)}):
            model,_=hand_plant._model(FoundationConfig());actor=ReferenceActor21(model)
            lo=actor.ctrl_range21[:,0].copy();hi=actor.ctrl_range21[:,1].copy();lo[18]-=actor.lift_bias_m;hi[18]-=actor.lift_bias_m;lo[18]=max(0.,lo[18]);hi[18]=min(.1,hi[18])
            for index,name in ((19,'rh_WRJ2'),(20,'rh_WRJ1')):
                j=model.joint(name).id
                if model.jnt_limited[j]:lo[index]=max(lo[index],model.jnt_range[j,0]);hi[index]=min(hi[index],model.jnt_range[j,1])
            h=float(model.geom_size[model.geom('support').id,2]);mount=ReceiverMount('support',('floor','support_post'),np.array([0.,0.,h]),h,float(np.cos(np.pi/4)))
            run=ROOT/'output/v20_small_nominal_transport_v1/runs/nominal__ratio_effort__619741';meta=read(run/'metadata.json')
            for name in ('trace.npz','metadata.json'):inputs[str(run/name)]=sha(run/name)
            with np.load(run/'trace.npz',allow_pickle=False) as z:t=z['time'];encoded=z['observation_json'];recv=z['receiver_measurement_json']
            packets={};rf={f.name for f in fields(ReceiverMeasurement)}
            for i in np.flatnonzero((t>=11.8-1e-9)&(t<=15.+1e-9)):
                raw=json.loads(str(encoded[i]));rr=json.loads(str(recv[i]));f=Frame(**{k:np.asarray(v) if isinstance(v,list) and k not in ('skin_patches','robot_actuator_axes21') else v for k,v in raw.items()});packets[int(i)]=(f,ReceiverMeasurement(**{k:v for k,v in rr.items() if k in rf}))
            empty=next(i for i,(f,_) in packets.items() if f.time>=13.935-1e-9 and np.all(f.valid&f.distributed_valid) and not np.any(f.contact_present) and not f.skin_patches)
            positive=max(i for i,(f,_) in packets.items() if i<empty and np.any(f.contact_present) and f.skin_patches);unknown=empty+1
            ad=ContactExitAdapter(model,open18=np.concatenate(OPEN),receiver_mount=mount);graphs={}
            selected={positive:'positive',empty:'P0',unknown:'unknown'}
            for i,(f,receiver) in packets.items():
                ad.observe(f)
                if f.time<12.-1e-9:continue
                current=np.asarray(f.robot_reference21,float)
                if not graphs:
                    entry=ReleaseEntry(12.,Reference21(current,12.),1);goal=release_clearance_goal(entry,preparation_open18=np.concatenate(OPEN),mechanical_lower21=lo,mechanical_upper21=hi);prior=ad.begin_release(12.)
                    for variant in ('backend','joint'):graphs[variant]=ReleaseSupervisor(entry,open18=goal.reference18,observer=TransferObserver(request_time_s=12.,receiver_half_size_xy_m=ad.receiver_half_size_xy_m,receiver_up_axis3=ad.receiver_up_axis3,prior=prior),robot_model=model,limits5=meta['limits'],budget=meta['budget'],force_margin=.02,formulation=variant)
                transfer=ad.transfer_inputs(receiver)
                for g in graphs.values():g.observe(frame=f,transfer_inputs=transfer,current21=current)
                if i not in selected:continue
                kind=selected[i]
                for variant,g in graphs.items():
                    if kind=='unknown':
                        with patch.object(ad,'current_contact_tasks',return_value=dict(available=False,diagnostics=dict(status='INJECTED_UNKNOWN_GEOMETRY'))):ref,debug=g.action(frame=f,adapter=ad,current21=current,lower21=lo,upper21=hi,generation=int(f.mode_generation))
                    else:ref,debug=g.action(frame=f,adapter=ad,current21=current,lower21=lo,upper21=hi,generation=int(f.mode_generation))
                    r=g.decision_evidence[-1];accepted=r['geometry_proposal_accepted'];hr=r['hierarchy']
                    records.append(dict(variant=variant,case=kind,time_s=f.time,phase=g.phase,accepted=accepted,status=None if hr is None else hr['status'],completed_prefix=None if hr is None else hr['completed_prefix'],certified_names=[] if hr is None else [c['name'] for c in hr['certified_prefix']],fallback=r['fallback'],numerical_error=r['numerical_error'],contact_diagnostics=None if r['contact_tasks'] is None else r['contact_tasks'].get('diagnostics'),failed_layer=None if hr is None else hr['failed_layer'],hierarchy_diagnostics=None if hr is None else hr['diagnostics'],emitted_hierarchy_score=r['emitted_hierarchy_score']))
                    ck(variant+'/'+kind+'/true_formulation_domain_clock',r['formulation']==variant and r['time_s']==f.time and ref.time_s==f.time and np.all(ref.values>=r['actual_reference_lower21']) and np.all(ref.values<=r['actual_reference_upper21']))
                    if kind!='unknown':
                        ck(variant+'/'+kind+'/actual_task_and_emission_accepted',accepted and r['emitted_hierarchy_score']['available'] and hr is not None)
                        p=len(r['contact_tasks']['pressure_P_N']);ck(variant+'/'+kind+'/positive_or_known_empty_shape',(p>0 if kind=='positive' else p==0))
                        if kind=='positive':ck(variant+'/genuine_formulation_certificate',hr['diagnostics']['normal_witness_is_submitted_prefix']==(variant=='backend') and hr['diagnostics']['normal_optimal_slack_frozen_before_motion']==(variant=='backend') and (variant=='backend' or all(c['name']!='normal_inward_correction_slack' for c in hr['certified_prefix'])))
                        native=g.native.servo.evaluate(np.asarray(f.robot_position25),np.asarray(f.robot_velocity25),actor.encode(ref).values,time_s=f.time)
                        ck(variant+'/'+kind+'/actual_native_value_matches_saved',native.status=='KNOWN_ROBOT_ACTUATION_ALGEBRA' and bits(native.predicted_effort21,r['native_proposed']['predicted_effort21']))
                    else:ck(variant+'/unknown_is_explicit_unaccepted_anchor',not accepted and hr is None and r['fallback']=='legal_current_q_recovery_anchor_without_exit_certificate' and bits(ref.values,r['feasible_recovery_anchor21']))
                    buffer=SimpleNamespace(ctrl=np.full(actor.control_width,.123));before=buffer.ctrl.copy();writes_before=actor.write_count;port=actor.apply(buffer,ref);nonrobot=np.setdiff1d(np.arange(actor.control_width),actor.indices21)
                    ck(variant+'/'+kind+'/real_actor_private_buffer_exact_once',actor.write_count==writes_before+1 and bits(buffer.ctrl[actor.indices21],port.values) and bits(port.values,actor.encode(ref).values) and bits(buffer.ctrl[nonrobot],before[nonrobot]))
                    g.reconcile(SimpleNamespace(executed=ref,execution_changed=False,stale_plan_replaced=False,mode_generation=int(f.mode_generation)));receipt=g.decision_evidence[-1]['execution_receipt']
                    ck(variant+'/'+kind+'/receipt_actual_debt_permission',bits(receipt['actual_reference21'],ref.values) and receipt['actual_task_interval_respected'] and receipt['same_snapshot_reference_debt_m']==float(f.robot_position25[24])-ref.values[18])
            ck('six_archived_cases_complete',len(records)==6 and not forbidden and not truth)
        ck('source_and_numerical_identity_stable',source==source_identity(['ratio_effort']) and numeric==numerical_identity())
        passed=True
    except Exception as exc:passed=False;error=repr(exc);trace=traceback.format_exc()
    result=dict(schema='v23_public_route_author_v1',passed=passed,public_cases_assessed=len(records),checks=checks,records=records,production_source_sha256=source,numerical_identity=numeric,input_sha256=inputs,physics_steps=0,forbidden_calls=forbidden,forbidden_truth_reads=truth,checker_sha256=sha(__file__),scope='Integration author replay of one archived public stream,2formulations×positive/P0/unknown. Static robot, private FK/native algebra and private plain actor buffer only; no model dynamics/forward/contact query/new rollout or independent kernel proof.')
    if not passed:result.update(error=error,traceback=trace)
    out.write_text(json.dumps(plain(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(path=str(out),sha256=sha(out),passed=passed,checks=len(checks),records=len(records)),ensure_ascii=False));print(result.get('traceback',''));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())

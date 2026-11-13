"""Few real archived public packets through JointController; no plant/step.

Private robot-only FK/Jacobian calls are necessary and counted. Integration,
forward dynamics, contacts, plant construction, engine writes are forbidden.
"""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import argparse,hashlib,json,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];F=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(F));sys.dont_write_bytecode=True
EXPECTED={'v13_joint/controller.py':'b9981e8fec70d9fcaf0cb368f89ada8d0e9396a9391b700030eb4fcadc1f9663',
 'v13_joint/prepared_value_effort21.py':'5b95ec9f1614415ec75b91ecb02629f4316d844960b6111932ffb8d1c76f81ac'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    checks=[];records=[];forbidden=[];fk_calls={};truth_reads=[];sources={}
    def check(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:raise AssertionError(name+' '+str(detail))
    class PublicFrame(SimpleNamespace):
        def __getattribute__(self,name):
            if name in ('object_position','object_velocity','object_pose','object_mass','object_friction','target_id','target_contact_id','is_target','raw_qpos','true_qpos','true_mass','true_friction'):
                truth_reads.append(name);raise AssertionError('forbidden runtime truth '+name)
            return super().__getattribute__(name)
    try:
        watched=list(EXPECTED)+['v13_joint/design.py','v13_joint/lifecycle.py','v13_joint/native_effort21.py','v13_joint/prepared_native_effort21.py','v13_joint/intent_qp.py','v12/observation.py','v12/intents.py']
        sources={p:sha(ROOT/p) for p in watched}
        check('final_controller_and_native_binding',all(sources[k]==v for k,v in EXPECTED.items()))
        import research_imports,mujoco,hand_plant
        for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
            if hasattr(mujoco,name):
                def deny(*a,_name=name,**k):forbidden.append(_name);raise AssertionError('forbidden '+_name)
                setattr(mujoco,name,deny)
        for name in ('mj_kinematics','mj_comPos','mj_jac'):
            original=getattr(mujoco,name)
            def counted(*a,_name=name,_original=original,**k):fk_calls[_name]=fk_calls.get(_name,0)+1;return _original(*a,**k)
            setattr(mujoco,name,counted)
        hand_plant.HandPlant.__init__=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No plant construction'))
        from foundation_config import FoundationConfig,OPEN
        from v13_a1.codec import Reference21,ReferenceActor21,EnginePort21
        from v13_joint.controller import create_controller
        from v13_joint.lifecycle import ReleaseEntry,release_nominal
        from v13_joint.design import reference_rates
        from v13_joint.prepared_value_effort21 import PreparedValueEffort21
        from v10.robot_contact_space import PublicSkinPatch
        ReferenceActor21.apply=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No engine writer'))
        model,_=hand_plant._model(FoundationConfig());actor=ReferenceActor21(model)
        lo=actor.ctrl_range21[:,0].copy();hi=actor.ctrl_range21[:,1].copy();lo[18]-=actor.lift_bias_m;hi[18]-=actor.lift_bias_m
        lo[18]=max(0.,lo[18]);hi[18]=min(.1,hi[18])
        for k,n in ((19,'rh_WRJ2'),(20,'rh_WRJ1')):
            j=model.joint(n).id
            if model.jnt_limited[j]:lo[k]=max(lo[k],model.jnt_range[j,0]);hi[k]=min(hi[k],model.jnt_range[j,1])
        open18=np.concatenate(OPEN);cache={}
        def load(rel):
            if rel not in cache:
                p=ROOT/rel
                with np.load(p/'trace.npz',allow_pickle=False) as z:cache[rel]=(z['time'],z['observation_json'],z['task_request_json'],json.loads((p/'metadata.json').read_text(encoding='utf-8')))
            return cache[rel]
        def current_reference(frame):
            r=np.r_[frame.robot_actuator_target21[:18],frame.executed_motion_reference]
            check('public_actual_reference_encodes_exact/'+str(frame.time),actor.encode(Reference21(r,frame.time)).values.tobytes()==np.asarray(frame.robot_actuator_target21,float).tobytes())
            return r
        def frame_at(rel,t):
            times,obs,tasks,meta=load(rel);i=int(np.argmin(abs(times-t)));f=PublicFrame(**json.loads(str(obs[i])))
            for k,v in list(vars(f).items()):
                if isinstance(v,list) and k not in ('skin_patches','robot_actuator_axes21'):setattr(f,k,np.asarray(v))
            return f,json.loads(str(tasks[i])),meta,i
        def prepare(rel,t,release=False):
            f,task,meta,index=frame_at(rel,t);r=current_reference(f)
            f.robot_reference21=r.copy();f.reference_lower21=lo.copy();f.reference_upper21=hi.copy();f.mode_generation=1 if release else 0
            f.release_entry_reference21=None;f.release_nominal_reference21=None
            if release:
                entryframe,_,_,_=frame_at(rel,12.);entry=current_reference(entryframe);f.release_entry_reference21=entry.copy()
                f.release_nominal_reference21=release_nominal(ReleaseEntry(12.,Reference21(entry,12.),1),f.time+.020,open18,lo,hi).values.copy()
                for k in range(5,18):
                    if entry[k]>open18[k]:f.reference_lower21[k]=max(lo[k],open18[k]);f.reference_upper21[k]=min(hi[k],r[k])
                    elif entry[k]<open18[k]:f.reference_lower21[k]=max(lo[k],r[k]);f.reference_upper21[k]=min(hi[k],open18[k])
                    else:f.reference_lower21[k]=f.reference_upper21[k]=open18[k]
            return f,task,meta,index
        selections=[('holding','output/development_v12_native_pilot_v1/runs/nominal__fusion_effort__617901',8.5,False),
            ('transport','output/development_v12_native_transport_v1/runs/friction_drop__ratio_effort__617902',10.,False),
            ('release','output/development_v12_native_pilot_v1/runs/nominal__fusion_effort__617901',12.020,True)]
        def controller_for(method,rel,t,meta):
            c=create_controller(method,dict(config=dict(shape=meta['shape'])),meta['limits'],meta['budget'],meta['actual_ratio_task_gains'],model)
            times=load(rel)[0]
            for j in np.flatnonzero((times>=t-.2-1e-9)&(times<t-1e-9)):
                warm,_,_,_=frame_at(rel,float(times[j]));c.state.update(warm,.005)
            return c
        for label,rel,t,release in selections:
            for method in ('fixed_effort','ratio_effort','fusion_effort'):
                f,task,meta,index=prepare(rel,t,release);c=controller_for(method,rel,float(f.time),meta);beforetask=deepcopy(task)
                c.set_task_request(task);start=time.perf_counter_ns();action=c.step(f,'RELEASING' if release else 'HOLDING','release' if release else 'hold');elapsed=(time.perf_counter_ns()-start)/1e6
                tag=label+'/'+method;check(tag+'/prepared_value_factory',type(c.native) is PreparedValueEffort21)
                check(tag+'/typed_actual21_finite_clock',type(action.reference21) is Reference21 and action.reference21.time_s==f.time and np.isfinite(action.reference21.values).all())
                check(tag+'/phase_generation',action.mode==('release' if release else 'holding') and action.mode_generation==f.mode_generation)
                step=reference_rates(action.mode)*.005
                check(tag+'/actual_five_ms_authority',np.all(action.reference21.values>=f.reference_lower21-1e-15) and np.all(action.reference21.values<=f.reference_upper21+1e-15) and np.all(abs(action.reference21.values-f.robot_reference21)<=step+1e-15))
                check(tag+'/not_pressure_prediction',not c.debug['pressure_prediction_available'] and np.isnan(c.predicted).all())
                check(tag+'/public_task_not_retargeted',task==beforetask)
                evidence=c.decision_evidence[-1];native=evidence['native'];dom=evidence['domain']
                if native and native['reference21'] is not None:
                    recheck=native['diagnostics']['prepared_value_final_full_recheck']
                    check(tag+'/frozen_full_native_final_check',recheck['passed'] and all(v is True for k,v in recheck.items() if k.endswith('_equal')))
                    check(tag+'/fixed_original_p0',np.asarray(native['p0_effort21']).tobytes()==np.asarray(dom['p0_effort21']).tobytes())
                    check(tag+'/raw_delta_not_absolute_retarget',np.asarray(native['requested_delta_effort21']).tobytes()==np.asarray(native['diagnostics']['requested_delta21']).tobytes())
                else:check(tag+'/unavailable_honestly_labelled',c.debug['label']=='native_unavailable_legal_nominal_reference')
                if release:check(tag+'/release_holding_task_cancelled',c.release_latched and c.debug['holding_load_task_inactive'] and np.array_equal(c.debug['force_prior5'],np.zeros(5)))
                records.append(dict(case=label,method=method,source=rel,time_s=float(f.time),index=index,step_ms=elapsed,mode=action.mode,generation=action.mode_generation,
                    public_valid5=c.state.valid.tolist(),patch_count=c.debug['public_patch_count'],label=c.debug['label'],qp_status=c.debug['qp_status'],
                    domain_status=dom['status'],native_status=None if native is None else native['status'],native_final_full_recheck=None if native is None else native['diagnostics'].get('prepared_value_final_full_recheck'),
                    increment21=(action.reference21.values-f.robot_reference21).tolist(),pressure_prediction_available=False,cache=c.native.cache_diagnostics()))
        # Explicit observation loss is unknown, not a zero pressure estimate.
        rel=selections[0][1];f,task,meta,index=prepare(rel,8.5);f.valid[:]=False;f.distributed_valid[:]=False;f.auxiliary_valid=False;f.load_valid=False
        c=controller_for('fusion_effort',rel,8.5,meta);c.set_task_request(task);unknown=c.step(f,'HOLDING','hold')
        check('missing_pressure_remains_nan_and_invalid',not c.state.valid.any() and np.isnan(c.state.normal).all() and not c.state.load_valid)
        check('unknown_is_honest_and_command_legal',not c.debug['conditional_support_active'] and not c.debug['pressure_prediction_available'] and np.isfinite(unknown.reference21.values).all())
        # Patches with object identity fields must not be accepted by geometry.
        bad=deepcopy(frame_at(rel,8.5)[0].skin_patches[0]);bad['target_id']=17
        try:PublicSkinPatch.from_public(bad)
        except ValueError:check('target_contact_ID_patch_rejected',True)
        else:check('target_contact_ID_patch_rejected',False)
        check('no_environment_truth_read',not truth_reads,truth_reads)
        check('no_physics_forward_contact_plant_or_writer',not forbidden,forbidden)
        check('execution_sources_unchanged',sources=={p:sha(ROOT/p) for p in watched})
        report=dict(schema='v13_joint_real_public_packet_zero_physics_v1',passed=True,checks=checks,records=records)
    except Exception as exc:report=dict(schema='v13_joint_real_public_packet_zero_physics_v1',passed=False,error=repr(exc),traceback=traceback.format_exc(),checks=checks,records=records)
    report.update(source_sha256=sources,forbidden_calls=forbidden,robot_only_FK_calls=fk_calls,forbidden_truth_reads=truth_reads,physics_steps=0,writer_calls=0,
        scope='Nine single-decision off-policy calls from saved public robot/skin/task packets, not trajectories or closed-loop success. Public shape only; no actual mass/friction/object state supplied. Committed21 and release entry derived from archived actual command telemetry. Private robot FK is counted, not physical integration. Software five_ms rates are declared policy, not hardware guarantee.')
    with out.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),records=len(records),path=str(out),sha256=sha(out)),ensure_ascii=False))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

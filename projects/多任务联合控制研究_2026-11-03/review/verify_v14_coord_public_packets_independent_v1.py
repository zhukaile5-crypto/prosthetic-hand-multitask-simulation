"""Few archived public packets through CoordinatedController; no plant/step.

Private robot-only FK/Jacobian calls are necessary and counted. Integration,
forward dynamics, contacts, plant construction, engine writes are forbidden.
"""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import argparse,hashlib,json,sys,time,traceback
from dataclasses import replace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];F=ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(F));sys.dont_write_bytecode=True
EXPECTED={'v14_coord/controller.py':'1220f4fc9bd4ea1004e7164ffbd2c99f6e806b661478994c13aaa871adffda09',
 'v14_coord/runtime.py':'f078924964cf2791971d9ee9b21d5d3f515f267bc797d793467a8587a60b89ec',
 'v14_coord/intent_qp.py':'1f56174b10af062dea161513874607e6971788d1a9589b2e2c414a5694eeeebc',
 'v14_coord/effort_realization.py':'7450ea567362d5675d9ca8228bccd7c79b8362d25541ca14d1abc86c80589a78',
 'v14_coord/effort_domain.py':'254b4a95dbaa7557bd7e767b5db7692b26287c1dfcf898341df655dbf91d5060'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    checks=[];records=[];forbidden=[];fk_calls={};truth_reads=[];sources={};inputs={};injections=[]
    def check(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:raise AssertionError(name+' '+str(detail))
    class PublicFrame(SimpleNamespace):
        def __getattribute__(self,name):
            if name in ('object_position','object_velocity','object_pose','object_mass','object_friction','target_id','target_contact_id','is_target','raw_qpos','true_qpos','true_mass','true_friction'):
                truth_reads.append(name);raise AssertionError('forbidden runtime truth '+name)
            return super().__getattribute__(name)
    try:
        watched=list(EXPECTED)+['v14_coord/plant.py','v14_coord/bootstrap.py','v14_coord/design.py','v13_joint/plant.py','v13_joint/lifecycle.py','v13_joint/native_effort21.py','v13_joint/prepared_native_effort21.py','v13_joint/prepared_value_effort21.py','v13_joint/intent_qp.py','v12/observation.py','v12/intents.py']
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
        import v14_coord.controller as cm
        from v14_coord.controller import create_controller,check_same_problem_candidate
        from v13_joint.lifecycle import ReleaseEntry,release_nominal
        from v14_coord.design import reference_rates
        from v14_coord.effort_domain import FastEffortGoal21
        from v14_coord.runtime import CoordinatedSession
        from v14_coord.intent_qp import solve_intent_qp
        from v13_joint.intent_qp import IntentQPResult
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
                inputs[rel]=dict(metadata_sha256=sha(p/'metadata.json'),trace_sha256_declared=json.loads((p/'result.json').read_text(encoding='utf-8'))['trace_sha256'])
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
                tag=label+'/'+method;check(tag+'/fast_domain_accepted_goal_factory',type(c.native) is FastEffortGoal21)
                check(tag+'/typed_actual21_finite_clock',type(action.reference21) is Reference21 and action.reference21.time_s==f.time and np.isfinite(action.reference21.values).all())
                check(tag+'/phase_generation',action.mode==('release' if release else 'holding') and action.mode_generation==f.mode_generation)
                step=reference_rates(action.mode)*.005
                check(tag+'/actual_five_ms_authority',np.all(action.reference21.values>=f.reference_lower21-1e-15) and np.all(action.reference21.values<=f.reference_upper21+1e-15) and np.all(abs(action.reference21.values-f.robot_reference21)<=step+1e-15))
                check(tag+'/not_pressure_prediction',not c.debug['pressure_prediction_available'] and np.isnan(c.predicted).all())
                check(tag+'/public_task_not_retargeted',task==beforetask)
                evidence=c.decision_evidence[-1];native=evidence['native'];dom=evidence['domain']
                problem=evidence['qp_problem'];acceptance=evidence['proposal_acceptance']
                expected_keys={'p0_21','A21x25','Jpx25','y_p','z_intent_p','effort_lower21','effort_upper21','patch_digit_p','caps5','total_budget','support_coeff_p','load_proxy_N','scales','max_iterations','atol','rtol'}
                check(tag+'/complete_problem_is_saved',set(problem)==expected_keys and set(problem['scales'])=={'residual25','intentP','delta21','load'})
                check(tag+'/accepted_one_problem_only',acceptance['accepted'] and acceptance['different_force_feedback_problem_used'] is False)
                restored={k:np.asarray(v) if isinstance(v,list) else v for k,v in problem.items()};restored['Jpx25']=np.asarray(problem['Jpx25']).reshape((-1,25));restored['scales']={k:np.asarray(v) if isinstance(v,list) else v for k,v in problem['scales'].items()}
                replay=solve_intent_qp(**restored)
                check(tag+'/saved_problem_replays_original_goal_and_output',replay.available and np.asarray(evidence['qp']['d21']).tobytes()==replay.d21.tobytes() and np.asarray(evidence['qp']['zP']).tobytes()==replay.zP.tobytes() and np.float64(evidence['qp']['objective']).tobytes()==np.float64(replay.objective).tobytes())
                nominal=f.release_nominal_reference21.copy() if release else f.robot_reference21.copy()
                if not release:
                    nominal[18:]=np.asarray(task['motion_target'],float)
                    preview=task.get('motion_preview')
                    if preview and preview['future_reference_available']:
                        nominal[18:]=[np.interp(f.time+.020,preview['requested_times_s'],np.asarray(preview['motion_targets'])[:,i]) for i in range(3)]
                    nominal+=np.asarray(c.debug['nominal_contact_search21'])
                raw=c.native.servo.evaluate_full(np.r_[f.raw_joint,f.lift],np.r_[f.raw_joint_velocity,f.lift_velocity_m_s],actor.encode(Reference21(nominal,f.time)).values,time_s=f.time)
                check(tag+'/p0_independent_raw_nominal_full_native',np.asarray(dom['p0_effort21']).tobytes()==raw.predicted_effort21.tobytes())
                if native and native['reference21'] is not None:
                    recheck=native['diagnostics']['prepared_value_final_full_recheck']
                    check(tag+'/frozen_full_native_final_check',recheck['passed'] and all(v is True for k,v in recheck.items() if k.endswith('_equal')))
                    check(tag+'/fixed_original_p0',np.asarray(native['p0_effort21']).tobytes()==np.asarray(dom['p0_effort21']).tobytes())
                    check(tag+'/accepted_delta_not_reoptimized',np.asarray(native['requested_delta_effort21']).tobytes()==np.asarray(evidence['qp']['d21']).tobytes())
                    check(tag+'/sole_goal_realization_role',native['diagnostics']['role']=='realize_the_single_accepted_effort_goal_not_second_force_optimization')
                else:check(tag+'/unavailable_honestly_labelled',c.debug['label']=='no_accepted_effort_goal_legal_nominal_fallback')
                if release:check(tag+'/release_holding_task_cancelled',c.release_latched and c.debug['holding_load_task_inactive'] and np.array_equal(c.debug['force_prior5'],np.zeros(5)))
                records.append(dict(case=label,method=method,source=rel,time_s=float(f.time),index=index,step_ms=elapsed,mode=action.mode,generation=action.mode_generation,
                    public_valid5=c.state.valid.tolist(),patch_count=c.debug['public_patch_count'],label=c.debug['label'],qp_status=c.debug['qp_status'],
                    domain_status=dom['status'],native_status=None if native is None else native['status'],native_final_full_recheck=None if native is None else native['diagnostics'].get('prepared_value_final_full_recheck'),
                    increment21=(action.reference21.values-f.robot_reference21).tolist(),pressure_prediction_available=False,cache=c.native.cache_diagnostics()))
        # One genuinely truncated solve may be provisionally used only with the same problem.
        rel=selections[0][1];f,task,meta,index=prepare(rel,8.5);c=controller_for('fixed_effort',rel,8.5,meta);c.set_task_request(task);limited_results=[]
        def limited(**kwargs):
            result=solve_intent_qp(**{**kwargs,'max_iterations':1});limited_results.append(result);return result
        with patch.object(cm,'solve_intent_qp',side_effect=limited):provisional=c.step(f,'HOLDING','hold')
        evidence=c.decision_evidence[-1];proposal=evidence['proposal_acceptance'];r=limited_results[0]
        check('one_iteration_real_nonconvergence',not r.available and r.status=='nonconvergence' and r.iterations==1)
        check('same_problem_feasible_merit_candidate_explicitly_provisional',proposal['accepted'] and proposal['classification']=='feasible_nonoptimal_same_problem' and proposal['optimality_certified'] is False and not proposal['different_force_feedback_problem_used'])
        check('provisional_goal_native_receives_original_delta',evidence['native'] is not None and np.asarray(evidence['native']['requested_delta_effort21']).tobytes()==r.d21.tobytes() and c.debug['label']=='same_problem_feasible_nonoptimal_explicit')
        injections.append(dict(case='one_iteration',kernel_status=r.status,iterations=r.iterations,acceptance=proposal,native_status=evidence['native']['status'],original_problem_budget=evidence['qp_problem']['max_iterations'],injected_solver_budget=1))
        # Directly exercise rejection of feasible-worse, bad-primal, forged-score and NaN iterates.
        problem=evidence['qp_problem'];problem={k:np.asarray(v) if isinstance(v,list) else v for k,v in problem.items()};problem['Jpx25']=np.asarray(problem['Jpx25']).reshape((-1,25));problem['scales']={k:np.asarray(v) if isinstance(v,list) else v for k,v in problem['scales'].items()}
        score_bad=replace(r,objective=np.nextafter(r.objective,np.inf));check('one_ULP_forged_objective_rejected',not check_same_problem_candidate(problem,score_bad)['accepted'])
        dd=r.d21.copy();dd[0]=problem['effort_upper21'][0]-problem['p0_21'][0]+1.
        def objective(p,d,z):
            residual=p['A21x25'].T@d-p['Jpx25'].T@(z-p['y_p']);s=p['scales'];a=s['residual25']*residual;b=s['intentP']*(z-p['z_intent_p']);c=s['delta21']*d
            return float(a@a+b@b+c@c+(s['load']*max(0.,p['load_proxy_N']-p['support_coeff_p']@z))**2)
        primal_bad=replace(r,d21=dd,objective=objective(problem,dd,r.zP));check('infeasible_even_honestly_scored_candidate_rejected',not check_same_problem_candidate(problem,primal_bad)['accepted'])
        nan_bad=replace(r,d21=np.full(21,np.nan));check('nonfinite_candidate_rejected',not check_same_problem_candidate(problem,nan_bad)['accepted'])
        toy=dict(p0_21=np.zeros(21),A21x25=np.zeros((21,25)),Jpx25=np.zeros((0,25)),y_p=np.zeros(0),z_intent_p=np.zeros(0),effort_lower21=np.full(21,-10.),effort_upper21=np.full(21,10.),patch_digit_p=np.zeros(0,int),caps5=np.full(5,5.),total_budget=18.,support_coeff_p=np.zeros(0),load_proxy_N=0.,scales=dict(residual25=np.ones(25),intentP=np.zeros(0),delta21=np.ones(21),load=1.),atol=1e-8)
        worse=IntentQPResult('nonconvergence','independent_feasible_worse_fixture',d21=np.ones(21),zP=np.zeros(0),objective=21.)
        check('feasible_but_worse_than_same_problem_anchor_rejected',not check_same_problem_candidate(toy,worse)['accepted'])
        # A rejected score must never reach the native realizer in the actual controller.
        f,task,meta,index=prepare(rel,8.5);c=controller_for('fixed_effort',rel,8.5,meta);c.set_task_request(task)
        def forged(**kwargs):
            x=solve_intent_qp(**kwargs);return replace(x,objective=np.nextafter(x.objective,np.inf))
        with patch.object(cm,'solve_intent_qp',side_effect=forged),patch.object(c.native,'solve',side_effect=AssertionError('rejected candidate reached native')) as native_call:action=c.step(f,'HOLDING','hold')
        check('rejected_candidate_does_not_reach_native_or_claim_accepted_goal',native_call.call_count==0 and c.decision_evidence[-1]['native'] is None and not c.decision_evidence[-1]['proposal_acceptance']['accepted'] and c.debug['label']=='no_accepted_effort_goal_legal_nominal_fallback')
        # Explicit observation loss is unknown, not a zero pressure estimate.
        rel=selections[0][1];f,task,meta,index=prepare(rel,8.5);f.valid[:]=False;f.distributed_valid[:]=False;f.auxiliary_valid=False;f.load_valid=False
        c=controller_for('fusion_effort',rel,8.5,meta);c.set_task_request(task);unknown=c.step(f,'HOLDING','hold')
        check('missing_pressure_remains_nan_and_invalid',not c.state.valid.any() and np.isnan(c.state.normal).all() and not c.state.load_valid)
        check('unknown_is_honest_and_command_legal',not c.debug['conditional_support_active'] and not c.debug['pressure_prediction_available'] and np.isfinite(unknown.reference21.values).all())
        e=c.decision_evidence[-1];activation=e['task_activation']
        check('unknown_load_is_not_zero_observation',c.state.load is None and not activation['load_observation_available'] and not activation['conditional_load'] and activation['missing_support_is_zero_observation'] is False and c.debug['support_demand_N'] is None)
        check('unknown_support_disables_only_the_task_explicitly',e['qp_problem']['load_proxy_N']==0. and not np.any(e['qp_problem']['support_coeff_p']) and activation['conditional_load_reason']=='public_support_unavailable_task_inactive')
        # Same-clock release must preempt a just-created holding plan, with no new sensor draw.
        for method in ('fixed_effort','ratio_effort','fusion_effort'):
            f,task,meta,index=prepare(rel,8.5);c=controller_for(method,rel,8.5,meta);c.set_task_request(task);c.step(f,'HOLDING','hold')
            incoming=f.robot_reference21.copy();f.mode_generation+=1;f.release_entry_reference21=incoming.copy();f.release_nominal_reference21=release_nominal(ReleaseEntry(f.time,Reference21(incoming,f.time),f.mode_generation),f.time+.020,open18,lo,hi).values.copy()
            for k in range(5,18):
                if incoming[k]>open18[k]:f.reference_lower21[k]=max(lo[k],open18[k]);f.reference_upper21[k]=min(hi[k],incoming[k])
                elif incoming[k]<open18[k]:f.reference_lower21[k]=max(lo[k],incoming[k]);f.reference_upper21[k]=min(hi[k],open18[k])
                else:f.reference_lower21[k]=f.reference_upper21[k]=open18[k]
            # Public task can still be hold: lifecycle command/phase has priority.
            release=c.step(f,'RELEASING','release');e=c.decision_evidence[-1]
            check('same_clock_release_preempts_holding/'+method,release.mode=='release' and release.mode_generation==f.mode_generation and e['mode']=='release' and np.array_equal(e['force_prior5'],np.zeros(5)) and e['task_activation']['holding_load_cancelled'])
        # Patches with object identity fields must not be accepted by geometry.
        bad=deepcopy(frame_at(rel,8.5)[0].skin_patches[0]);bad['target_id']=17
        try:PublicSkinPatch.from_public(bad)
        except ValueError:check('target_contact_ID_patch_rejected',True)
        else:check('target_contact_ID_patch_rejected',False)
        # Actual runtime method, without constructing or advancing a session:
        # one held command-memory snapshot replaces deliberately stale engine.
        f,_,_,_=prepare(rel,8.5);actual_ref=f.robot_reference21.copy();actual_engine=np.asarray(f.robot_actuator_target21).copy()
        sensor_before=(f.normal.tobytes(),f.sample_time,json.dumps(f.skin_patches,sort_keys=True))
        f.robot_actuator_target21=actual_engine+.25;f.robot_control_time_s=f.time-.005
        fake=SimpleNamespace(time=f.time,mode_generation=0,joint_release_entry=None,
            engine_snapshot=lambda:EnginePort21(actual_engine,f.time,actor.lift_bias_m),reference_snapshot=lambda:Reference21(actual_ref,f.time),
            reference_authority21=lambda:(lo.copy(),hi.copy()))
        session=CoordinatedSession.__new__(CoordinatedSession);session.plant=fake;session._attach_execution_state(f)
        check('runtime_refreshes_reference_and_engine_one_current_clock',f.robot_actuator_target21.tobytes()==actual_engine.tobytes() and f.robot_reference21.tobytes()==actual_ref.tobytes() and f.robot_control_time_s==f.time)
        check('runtime_refresh_does_not_resample_or_modify_skin',sensor_before==(f.normal.tobytes(),f.sample_time,json.dumps(f.skin_patches,sort_keys=True)))
        check('no_environment_truth_read',not truth_reads,truth_reads)
        check('no_physics_forward_contact_plant_or_writer',not forbidden,forbidden)
        check('execution_sources_unchanged',sources=={p:sha(ROOT/p) for p in watched})
        root_binding=ROOT/'review/v14_coord_runtime_root_zero_v1.json';binding=json.loads(root_binding.read_text(encoding='utf-8'))['source_sha256']
        check('selected_versions_match_root105_binding',len(binding)==105 and all(binding[k]==v for k,v in sources.items()))
        report=dict(schema='v14_coord_real_public_packet_zero_physics_independent_v1',passed=True,checks=checks,records=records)
    except Exception as exc:report=dict(schema='v14_coord_real_public_packet_zero_physics_independent_v1',passed=False,error=repr(exc),traceback=traceback.format_exc(),checks=checks,records=records)
    report.update(source_sha256=sources,checker_sha256=sha(Path(__file__)),input_provenance=inputs,injections=injections,forbidden_calls=forbidden,robot_only_FK_calls=fk_calls,forbidden_truth_reads=truth_reads,physics_steps=0,writer_calls=0,
        scope='Nine single-decision off-policy public packet calls plus targeted candidate/unknown/release/runtime negatives; not trajectories or closed-loop success. Public shape only, no actual mass/friction/object state. Release entry uses incoming actual commands. Counted private robot FK is geometry only. No plant, integration, contact solve or writer. Provisional merit acceptance does not certify KKT. V14 release remains the frozen V13 geometric path; no release performance improvement is claimed. Source105 map is only selected-version binding, not an independent all105 closure audit. Empty J JSON arrays are reshaped by the declared P-by25 schema for replay.')
    with out.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),records=len(records),path=str(out),sha256=sha(out)),ensure_ascii=False))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

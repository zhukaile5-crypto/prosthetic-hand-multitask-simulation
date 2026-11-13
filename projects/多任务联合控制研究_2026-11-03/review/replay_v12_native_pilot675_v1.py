"""Frozen-public-input effort replay; no plant construction or dynamics.

All 675 original goals/domains/final native residuals are checked. Candidate
pool scores are independently re-evaluated only at each method's first, last,
and first old-failure decision (nine fixed representatives). This is not a
pressure forecast or a closed-loop performance comparison.
"""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from datetime import datetime, timezone
import argparse, ast, hashlib, importlib.util, json, sys, traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'output/development_v12_pilot_v1'
FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
EXPECTED='160eb7129237d695240d0639a6562dd7fb21c6bacbaea1c06414cf61ea098915'
METHODS=('ratio_effort','fixed_effort','fusion_effort')
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(FOUNDATION))
sys.dont_write_bytecode=True


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=serial),encoding='utf-8')
def same(a,b):return np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True)
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def recovery(check):
    """Recover original input, including original pre-intersection step bounds."""
    snap=PILOT/'sources'
    observer=load_module('pilot_frozen_observer_for_native_replay',snap/'v12/observation.py')
    controller_text=(snap/'v12/controller.py').read_text(encoding='utf-8')
    controller_ast=ast.parse(controller_text)
    bounds=next(m for c in controller_ast.body if isinstance(c,ast.ClassDef) and c.name=='TaskController'
                for m in c.body if isinstance(m,ast.FunctionDef) and m.name=='_bounds')
    namespace={'np':np,'array':observer.array}
    exec(compile(ast.Module(body=[bounds],type_ignores=[]),'<frozen original _bounds>','exec'),namespace)
    gt=ast.parse((snap/'foundation/vendor/frozen_controllers.py').read_text(encoding='utf-8'))
    gc=next(n for n in gt.body if isinstance(n,ast.ClassDef) and n.name=='Gains')
    defaults={n.target.id:ast.literal_eval(n.value) for n in gc.body if isinstance(n,ast.AnnAssign)}
    manifest=read(PILOT/'manifest.json');sources=manifest['research_source_sha256']
    recovered=[];summaries=[]
    for method in METHODS:
        folder=PILOT/'runs'/('nominal__'+method+'__617901');path=folder/'trace.npz'
        names=('time','observation_json','task_request_json','debug_json','estimated_normal',
               'tactile_acquisition_time','observer_uncertainty','validity')
        with np.load(path,allow_pickle=False) as z:
            data={n:z[n] for n in names};meta=json.loads(str(z['metadata']))
        check(method+'/source_identity',meta['research_source_sha256']==sources)
        check(method+'/trace_identity',sha(path)==read(folder/'metadata.json')['trace_sha256'])
        gains=SimpleNamespace(**dict(defaults,**meta['actual_ratio_task_gains']))
        state=observer.PublicTactileState(gains);dummy=SimpleNamespace(g=gains,state=state)
        counts=Counter();failures=[];method_rows=[];observer_errors=[];clock_errors=[]
        for i,t in enumerate(data['time']):
            if not 7.3-1e-8<=t<12.-1e-8:continue
            frame=SimpleNamespace(**json.loads(str(data['observation_json'][i])))
            state.update(frame,.005)
            if not (same(state.normal,data['estimated_normal'][i]) and
                    same(state.normal_time,data['tactile_acquisition_time'][i]) and
                    same(state.force_uncertainty,data['observer_uncertainty'][i]) and
                    same(state.valid,data['validity'][i])):observer_errors.append(int(i))
            if t<7.5-1e-8 or abs((t-7.5)/.02-round((t-7.5)/.02))>=1e-7:continue
            task=json.loads(str(data['task_request_json'][i]));debug=json.loads(str(data['debug_json'][i]))
            old=debug['effort_tracking'];lo,hi=namespace['_bounds'](dummy,frame,.02)
            q=np.r_[frame.raw_joint,frame.lift];v=np.r_[frame.raw_joint_velocity,frame.lift_velocity_m_s]
            check(method+f'/{t:.3f}/robot_sample_public_exact',same(q,frame.robot_position25) and same(v,frame.robot_velocity25))
            if abs(frame.robot_control_time_s-t)>1e-8 or abs(task['time_s']-t)>1e-8:clock_errors.append(int(i))
            kwargs=dict(q25=q,v25=v,current_reference8=np.r_[frame.alpha,task['executed_motion_reference']],
                actual_target21=np.asarray(frame.robot_actuator_target21),
                desired_torque_correction25=np.asarray(debug['virtual_force_error_torque25']),
                nominal_motion_reference3=np.asarray(debug['nominal_motion_reference3']),
                step_lower8=lo,step_upper8=hi,motion_reference_bounds=task['motion_reference_bounds'],
                tracking_tolerances3=task['tracking_tolerances'])
            available=old['increment8'] is not None and old['port_target21'] is not None
            counts[old['status']]+=1
            if not available:failures.append(len(method_rows))
            complete=old.get('desired_torque25') is not None and all(np.isfinite(kwargs[n]).all() for n in
                ('q25','v25','current_reference8','actual_target21','desired_torque_correction25','nominal_motion_reference3','step_lower8','step_upper8'))
            check(method+f'/{t:.3f}/complete_original_input',complete)
            if not complete:raise ValueError('Original goal/input unavailable; no replacement allowed')
            row=dict(method=method,time=float(t),trace_index=i,kwargs=kwargs,old=old,
                     observed_valid5=state.valid.copy(),old_available=available)
            method_rows.append(row)
        check(method+'/225_original_decisions',len(method_rows)==225)
        check(method+'/observer_exact_all_warmup_and_holding_samples',not observer_errors,observer_errors[:8])
        check(method+'/same_sample_robot_target_and_request_clock',not clock_errors,clock_errors[:8])
        selected={0,len(method_rows)-1}
        if failures:selected.add(failures[0])
        for k,row in enumerate(method_rows):row['candidate_pool_representative']=k in selected
        recovered.extend(method_rows)
        summaries.append(dict(method=method,decisions=len(method_rows),old_statuses=dict(counts),
                              candidate_pool_representative_times=[method_rows[k]['time'] for k in sorted(selected)]))
    return recovered,summaries


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);args=ap.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=args.out or ROOT/'review'/('v12_native_pilot675_replay_'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    checks=[];issues=[];rows=[]
    def check(name,ok,detail=None):
        rec=dict(name=name,passed=bool(ok))
        if detail is not None:rec['detail']=detail
        checks.append(rec)
        if not ok:issues.append(rec)
    watch=[Path(__file__),ROOT/'v12_native/effort_tracker.py',ROOT/'v11/robot_servo_effort.py',
           ROOT/'v10/robot_contact_space.py',FOUNDATION/'hand_plant.py',FOUNDATION/'foundation_config.py',
           PILOT/'sources/v12/controller.py',PILOT/'sources/v12/observation.py',
           PILOT/'sources/foundation/vendor/frozen_controllers.py']
    before={str(p):sha(p) for p in watch};physics_calls=Counter();started=datetime.now(timezone.utc).isoformat()
    report=dict(schema='independent_public_input_native_effort_replay_v1',started_utc=started,
        closed_loop_performance_claim=False,pressure_prediction_test=False,physics_steps=0,
        source_sha256=before,checks=checks,issues=issues,rows=rows)
    try:
        check('native_tracker_final_identity',sha(ROOT/'v12_native/effort_tracker.py')==EXPECTED)
        if issues:raise ValueError('Final source identity mismatch')
        recovered,summaries=recovery(check);report['input_recovery']=summaries
        check('675_complete_original_inputs',len(recovered)==675)
        check('old_available599_failure76',sum(r['old_available'] for r in recovered)==599 and
              sum(r['old']['status']=='NUMERICAL_FAILURE' for r in recovered)==76)
        write(out/'input_recovery.json',dict(source_sha256=before,summaries=summaries,
            records=[dict(method=r['method'],time=r['time'],original_kwargs=r['kwargs'],
                old_status=r['old']['status'],old_goal25=r['old']['desired_torque25'],
                old_final_lower8=r['old']['debug']['candidate_lower_reference8'],
                old_final_upper8=r['old']['debug']['candidate_upper_reference8']) for r in recovered]))
        if issues:raise ValueError('Original input recovery failed; new solves not executed')
        print('Recovered 675 original inputs; old available=599 numerical_failure=76',flush=True)
        import research_imports, mujoco, hand_plant
        from foundation_config import FoundationConfig
        def forbidden(name):
            def deny(*a,**k):physics_calls[name]+=1;raise AssertionError('Forbidden dynamics/plant operation '+name)
            return deny
        for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation'):
            if hasattr(mujoco,name):setattr(mujoco,name,forbidden(name))
        hand_plant.HandPlant.__init__=forbidden('HandPlant.__init__')
        from v12_native.effort_tracker import EffortTracker
        from v11.robot_servo_effort import RobotServoEffort
        model,model_hash=hand_plant._model(FoundationConfig())
        report['public_nominal_model_xml_sha256']=model_hash
        tracker=EffortTracker(model);oracle=RobotServoEffort(model)
        basis=tracker.basis
        def port(ref):
            return np.r_[np.concatenate([basis.open_pose[j]+ref[j]*(basis.closed_pose[j]-basis.open_pose[j]) for j in range(5)]),
                         ref[5]+basis.lift_motor_offset,ref[6:]]
        totals=Counter();residual_error=goal_error=candidate_score_error=0.;max_angular=max_lift=0.
        for number,r in enumerate(recovered):
            kw=r['kwargs'];old=r['old'];name=r['method']+f"/{r['time']:.3f}"
            result=tracker.solve(**kw);d=result.debug;totals[result.status]+=1
            current=kw['current_reference8'];nominal=np.r_[current[:5],kw['nominal_motion_reference3']]
            nominal_snapshot=oracle.evaluate(kw['q25'],kw['v25'],port(nominal))
            target=nominal_snapshot.generalized_actuator_effort25+kw['desired_torque_correction25']
            lower=np.maximum(current+kw['step_lower8'],basis.held_lower8)
            upper=np.minimum(current+kw['step_upper8'],basis.held_upper8)
            lower[5:]=np.maximum(lower[5:],kw['motion_reference_bounds']['lower'])
            upper[5:]=np.minimum(upper[5:],kw['motion_reference_bounds']['upper'])
            for j,tol in enumerate(kw['tracking_tolerances3']):
                if tol is not None:
                    lower[j+5]=max(lower[j+5],nominal[j+5]-tol);upper[j+5]=min(upper[j+5],nominal[j+5]+tol)
            check(name+'/goal_fixed_original_bytes',same(result.desired_torque25,old['desired_torque25']) and same(target,old['desired_torque25']))
            check(name+'/original_reference_intersection_unchanged',same(lower,old['debug']['candidate_lower_reference8']) and
                  same(upper,old['debug']['candidate_upper_reference8']) and same(lower,d.get('candidate_lower_reference8')) and same(upper,d.get('candidate_upper_reference8')))
            check(name+'/new_available',result.available,dict(status=result.status,reason=d.get('reason')))
            item=dict(method=r['method'],time=r['time'],old_status=old['status'],old_available=r['old_available'],
                new_status=result.status,available=result.available,candidate_pool_checked=r['candidate_pool_representative'])
            if result.available:
                ref=np.asarray(result.reference8);actual_port=port(ref);step=ref-current
                check(name+'/real_step_reference_and_port_bounds',np.all(ref>=lower-1e-12) and np.all(ref<=upper+1e-12)
                    and np.all(step>=kw['step_lower8']-1e-12) and np.all(step<=kw['step_upper8']+1e-12)
                    and same(step,result.increment8) and same(actual_port,result.port_target21)
                    and np.all(actual_port[oracle.ctrl_limited]>=oracle.ctrl_range[oracle.ctrl_limited,0]-1e-12)
                    and np.all(actual_port[oracle.ctrl_limited]<=oracle.ctrl_range[oracle.ctrl_limited,1]+1e-12))
                actual=oracle.evaluate(kw['q25'],kw['v25'],actual_port)
                A=actual.transmission21x25
                canonical=A.T@(actual.predicted_effort21-nominal_snapshot.predicted_effort21)-kw['desired_torque_correction25']
                direct=actual.generalized_actuator_effort25-target
                e=float(np.max(abs(canonical-result.residual25)));residual_error=max(residual_error,e)
                check(name+'/independent_native_final_residual_bytes',same(canonical,result.residual25) and
                    same(actual.generalized_actuator_effort25,result.achieved_torque25) and np.max(abs(direct-canonical))<=1e-10
                    and not any(actual.diagnostics['control_clipped21']))
                check(name+'/canonical_final_matches_selected_scalar_bytes',all(s['residual_bytes_equal'] and s['loss_bytes_equal'] for s in d['scalar_native_recheck']))
                max_angular=max(max_angular,float(np.linalg.norm(canonical[:24])));max_lift=max(max_lift,abs(float(canonical[24])))
                if r['candidate_pool_representative']:
                    seed=np.minimum(upper,np.maximum(lower,nominal))
                    for solution in d['scalar_solutions']:
                        axis=solution['axis'];ids=np.asarray(solution['torque_support25'],int)
                        observed=[]
                        for c in solution['candidates']:
                            candidate=seed.copy();candidate[axis]=c['value']
                            s=oracle.evaluate(kw['q25'],kw['v25'],port(candidate))
                            residual=A.T@(s.predicted_effort21-nominal_snapshot.predicted_effort21)-kw['desired_torque_correction25']
                            subset=np.asarray(residual[ids],dtype=np.float64);loss=float(subset@subset)
                            candidate_score_error=max(candidate_score_error,abs(loss-c['loss']))
                            check(name+f'/candidate_axis{axis}/{len(observed)}/native_score_bytes',
                                np.float64(loss).tobytes()==np.float64(c['loss']).tobytes())
                            observed.append((loss,abs(c['value']-nominal[axis]),c['value']))
                        best=min(observed)
                        check(name+f'/candidate_axis{axis}/lexicographic_native_minimum',
                            (solution['selected']['loss'],solution['selected']['distance_to_nominal'],solution['selected']['value'])==best)
                        chosen=np.asarray(canonical[ids],dtype=np.float64)
                        check(name+f'/candidate_axis{axis}/final_residual_bytes',
                            chosen.tobytes().hex()==solution['selected_native_residual_bytes_hex'] and
                            np.float64(chosen@chosen).tobytes().hex()==solution['selected_native_loss_bytes_hex'])
                item.update(reference8=ref,increment8=step,residual25=canonical,
                    angular_residual_l2_Nm=float(np.linalg.norm(canonical[:24])),lift_residual_N=float(canonical[24]),
                    old_residual_l2=(float(np.linalg.norm(old['residual25'])) if old['residual25'] is not None else None))
            check(name+'/no_pressure_or_safety_claim',d.get('empirical_pressure_B_used') is False and
                  d.get('future_q_or_pressure_predicted') is False and d.get('physical_safety_certified') is False and d.get('global_model_admitted') is False)
            rows.append(item)
            if issues and not (out/'first_failed_decision.json').exists():write(out/'first_failed_decision.json',dict(original=r,new=result.to_dict(),issues=issues))
            if (number+1)%75==0:print('Native replay',number+1,'/675 issues',len(issues),flush=True)
        check('675_new_available',len(rows)==675 and all(r['available'] for r in rows))
        check('zero_dynamics_and_plant_calls',not physics_calls,dict(physics_calls))
        check('private_robot_time_zero',float(tracker.servo.data.time)==0. and float(oracle.data.time)==0.)
        report.update(new_statuses=dict(totals),old_unavailable_recovered=sum(not r['old_available'] and r['available'] for r in rows),
            old_available_retained=sum(r['old_available'] and r['available'] for r in rows),
            representative_candidate_pools=sum(r['candidate_pool_checked'] for r in rows),
            independent_residual_max_absolute_error=residual_error,independent_candidate_score_max_absolute_error=candidate_score_error,
            max_angular_native_residual_l2_Nm=max_angular,max_absolute_lift_native_residual_N=max_lift)
    except Exception as exc:
        report['exception']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        check('completed_without_exception',False,str(exc))
    after={str(p):sha(p) for p in watch};check('watched_execution_and_review_sources_unchanged',before==after)
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(),passed=not issues,completed=len(rows)==675,
        check_count=len(checks),physics_calls=dict(physics_calls),source_sha256_after=after,
        limitations=['Offline re-solve at saved public robot states, not new closed-loop performance.',
            'No pressure response, future encoder, contact-force or safety validation.',
            'All 675 final goals/domains/native outputs checked; candidate enumeration independently rescored at nine preselected representatives.',
            'Public default robot structure compiled; no per-trial mass/friction/object pose enters the solver.'])
    write(out/'audit.json',report)
    text=f"675 frozen-input native effort replay: passed={report['passed']}, completed={report['completed']}, checks={len(checks)}, issues={len(issues)}\n"
    text+=f"New statuses: {report.get('new_statuses')}\nOld available retained={report.get('old_available_retained')}; old failures recovered={report.get('old_unavailable_recovered')}\nNo plant construction/dynamics/forward/contact solve; no pressure forecast or performance claim.\n"
    (out/'readout.txt').write_text(text,encoding='utf-8');print(text,flush=True);print(str(out/'audit.json'),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

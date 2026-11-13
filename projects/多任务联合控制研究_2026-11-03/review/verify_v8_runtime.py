"""V8 runtime pure-reference/source and optional zero-physics construction QA.

No plant is advanced and no policy step/QP is called. Outputs are new review
evidence only. Pass --zero-step for all6 constructors and retained force-policy comparisons.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from datetime import datetime,timezone
from dataclasses import asdict
import argparse,ast,hashlib,json
import numpy as np
from v8.runtime import PublicPreviewFollower,V8Session,source_identity,source_files
from v8.runtime import V8_REFERENCE_RATE_LIMITS
from task_contract import build_task_request
from task_trajectory import public_transport_trajectory,unknown_future_point


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(zero_step=False):
    checks=[]
    def check(name,value,detail=None):
        checks.append(dict(name=name,passed=bool(value),detail=detail))
    trajectory=public_transport_trajectory()
    bounds=dict(lower=[0.,-.5,-.7],upper=[.1,.2,.5])
    def request(t,actual,mode='transport',known=True):
        target=trajectory.position_at(t)
        return build_task_request(t,mode,target,actual,12. if mode not in ('release','monitor') else t+3.5,
            [5.]*5,18.,tracking_tolerances=[.01,.015,.015],
            motion_preview=trajectory.preview(t) if known else unknown_future_point(t,target),motion_reference_bounds=bounds)
    follower=PublicPreviewFollower();actual=np.asarray(trajectory.position_at(9.3));origin=actual.copy()
    endpoint=np.asarray(trajectory.position_at(9.32));commands=[]
    for index in range(4):
        actual=follower.step(request(9.3+index*.005,actual));commands.append(actual.copy())
    check('20ms_endpoint_reached_after_four5ms_reference_commands',np.allclose(actual,endpoint,rtol=0,atol=1e-12))
    check('finite_ramp_quarters_exact',all(np.allclose(v,origin+(j+1)/4.*(endpoint-origin),rtol=0,atol=1e-12) for j,v in enumerate(commands)))
    first=np.asarray(commands[0]);check('follower_uses_known_next20ms_not_instantaneous_target',not np.array_equal(first,origin))
    hold=follower.step(request(9.32,actual),pause_transport=True)
    check('force_strategy_pause_retains_actual_reference',np.array_equal(hold,actual) and follower.start_time is None)
    lagged=np.array([.08,0.,0.]);delayed_request=request(9.3,lagged)
    original_request=json.loads(json.dumps(delayed_request))
    lagged_follower=PublicPreviewFollower();limited=lagged_follower.step(delayed_request)
    max_delta=np.asarray(V8_REFERENCE_RATE_LIMITS)*.005
    check('lagged_entry_cannot_jump_to20ms_waypoint_beyond_shared_rate',np.all(abs(limited-lagged)<=max_delta+1e-12) and lagged_follower.last_diagnostics['rate_limited_axes'][0])
    check('reference_rate_guard_preserves_original_user_request',delayed_request==original_request and lagged_follower.last_diagnostics['task_request_modified'] is False)
    lagged_follower.step(request(9.305,limited),pause_transport=True)
    recovered=lagged_follower.step(request(9.310,limited))
    check('pause_recovery_is_rate_bounded',np.all(abs(recovered-limited)<=max_delta+1e-12))
    jump_request=request(9.3,np.array([.03,0.,0.]),known=False)
    point_follower=PublicPreviewFollower();jumped=point_follower.step(jump_request)
    check('unknown_future_point_jump_is_execution_rate_bounded_without_fabricated_future',np.all(abs(jumped-np.array([.03,0.,0.]))<=max_delta+1e-12) and not jump_request['motion_preview']['future_reference_available'])
    no_preview=PublicPreviewFollower();point=no_preview.step(request(9.3,origin,known=False))
    check('unknown_future_keeps_current_explicit_point_not_fabricated_preview',np.array_equal(point,origin) and no_preview.start_time is None)
    releasing=PublicPreviewFollower();released=releasing.step(request(9.3,origin,mode='release'))
    check('release_preempts_reference_follower',np.array_equal(released,origin) and releasing.start_time is None)
    terminal=trajectory.preview(11.995)
    check('pre_release_preview_retains_continuous_terminal_user_endpoint',all(v==trajectory.position_at(12.) for t,v in zip(terminal['requested_times_s'],terminal['motion_targets']) if t>=12.))
    bad=request(9.3,origin);bad['object_mass']=.12
    try:PublicPreviewFollower().step(bad);rejected=False
    except ValueError:rejected=True
    check('extra_object_truth_rejected_by_strict_task_schema',rejected)
    bad=request(9.3,origin);bad['motion_preview']['target_object_pose']=[0.,0.,0.]
    try:PublicPreviewFollower().step(bad);rejected=False
    except ValueError:rejected=True
    check('nested_future_object_truth_rejected',rejected)
    bad=request(9.3,origin);bad['motion_reference_bounds']['upper'][1]=0.
    try:PublicPreviewFollower().step(bad);rejected=False
    except ValueError:rejected=True
    check('infeasible_preview_is_not_silently_clipped',rejected)
    f=PublicPreviewFollower();f.step(request(9.3,origin))
    try:f.step(request(9.2,origin));rejected=False
    except ValueError:rejected=True
    check('unrestored_clock_rewind_rejected',rejected)
    runtime=(ROOT/'v8/runtime.py').read_text(encoding='utf-8-sig');tree=ast.parse(runtime)
    factories=[n for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and any(a.name=='create_joint' for a in n.names)]
    check('new_joint_factory_is_v8_only',[n.module for n in factories]==['v8.controller'])
    check('runtime_never_imports_old_joint_or_coupled_state',not any(isinstance(n,ast.ImportFrom) and n.module in ('coordinator','state_estimator') for n in ast.walk(tree)))
    check('every_observation_gets_public_actual_reference_and_device_spec',runtime.index('obs.executed_motion_reference=')<runtime.index('if not all(np.isfinite(truth') and 'obs.sensor_characteristics=deepcopy(self.sensor_characteristics)' in runtime)
    from v8.run_development import output_path
    rejected_paths=[]
    for path in ('output','runtime.py','../outside'):
        try:output_path(path);rejected_paths.append(False)
        except (ValueError,FileExistsError):rejected_paths.append(True)
    check('runner_refuses_existing_root_source_and_outside_output_paths',all(rejected_paths))
    required_sources={
        'v8/__init__.py','v8/runtime.py','v8/run_development.py','v8/controller.py','v8/feedback.py','v7/belief.py','v7/__init__.py',
        'evaluation.py','protocol.json','task_contract.py','task_trajectory.py','research_imports.py',
        'foundation/review/admission_manifest.json','foundation/session.py','foundation/hand_plant.py','foundation/sensors.py'}
    identity=source_identity(['ratio','passive']);paths=source_files(['ratio','passive'])
    check('v8_identity_has_actual_v8_parent_and_foundation_sources',required_sources<=set(identity))
    check('all_resolved_source_hashes_match_actual_files',set(identity)==set(paths) and all(identity[k]==sha(p) for k,p in paths.items()))
    check('identity_is_not_old_v6_runtime_snapshot','runtime.py' not in identity and identity['v8/runtime.py']==sha(ROOT/'v8/runtime.py'))
    original_names=['runtime.py','run_development.py','protocol.json','evaluation.py']
    originals={p:sha(ROOT/p) for p in original_names}
    construction=[]
    if zero_step:
        from runtime import ResearchSession
        for method in ['ratio','passive','unified','ratio_preview','passive_preview']:
            native_method=method.removesuffix('_preview')
            old=ResearchSession(method=native_method,scenario='nominal',seed=610231,split='robustness')
            new=V8Session(method=method,scenario='nominal',seed=610231,split='robustness')
            exact={
                'physical_config':asdict(old.config)==asdict(new.config),
                'sensor_noise_profile':asdict(old.sensors.noise)==asdict(new.sensors.noise),
                'initial_sensor_rng_state':old.sensors.rng.bit_generator.state==new.sensors.rng.bit_generator.state,
                'initial_qpos':np.array_equal(old.plant.data.qpos,new.plant.data.qpos),
                'initial_qvel':np.array_equal(old.plant.data.qvel,new.plant.data.qvel),
                'initial_ctrl':np.array_equal(old.plant.data.ctrl,new.plant.data.ctrl),
                'actual_gains':old.gains==new.gains,
                'same_force_controller_type':type(old.controller) is type(new.controller),
                'warmup_original_baseline_02s':new.observer_warmup_s==.2,
                'zero_physics':old.time==new.time==0. and old.tick==new.tick==0 and not old.trace and not new.trace and not old.plant.substep_records and not new.plant.substep_records,
                'native_release':new.release_driver is None and new.lifecycle_enabled is False}
            check(method+'_zero_step_matches_retained_platform',all(exact.values()),exact);construction.append({'method':method,'exact':exact})
        old=ResearchSession(method='ratio',scenario='nominal',seed=610231,split='robustness')
        joint=V8Session(method='joint',scenario='nominal',seed=610231,split='robustness')
        exact=dict(physical_config=asdict(old.config)==asdict(joint.config),sensor_noise_profile=asdict(old.sensors.noise)==asdict(joint.sensors.noise),
                   actual_ratio_task_gains=old.gains==joint.actual_ratio_task_gains,
                   unique_new_joint_state=type(joint.controller.state).__module__=='v8.controller',
                   declared_half_second_warmup=joint.observer_warmup_s==.5,
                   initial_qpos=np.array_equal(old.plant.data.qpos,joint.plant.data.qpos),
                   initial_ctrl=np.array_equal(old.plant.data.ctrl,joint.plant.data.ctrl),
                   zero_physics=joint.time==0. and joint.tick==0 and not joint.trace and not joint.plant.substep_records)
        check('joint_zero_step_uses_new_factory_unique_context_and_exact_ratio_task',all(exact.values()),exact)
        construction.append({'method':'joint','exact':exact})
    check('original_runtime_runner_protocol_evaluator_unchanged',originals=={p:sha(ROOT/p) for p in original_names})
    check('all_v8_parent_foundation_identities_stable_through_validation',source_identity(['ratio','passive'])==identity)
    return dict(schema='v8_runtime_pure_and_zero_step_validation_v1',created_utc=datetime.now(timezone.utc).isoformat(),
                checks=checks,all_checks_passed=all(c['passed'] for c in checks),zero_step_constructions=construction,
                physics_steps_executed=0,policy_steps_or_qp_calls=0,source_identity=identity,
                original_source_hashes=originals,scope='Pure reference/schema/source checks and optional all6-method zero-step construction only; no prepared grasp, full policy, runtime deadline, safety or performance claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--zero-step',action='store_true');args=parser.parse_args()
    result=run(args.zero_step);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'review'/('v8_runtime_validation_'+stamp+'.json')
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(out),checks=len(result['checks']),all_checks_passed=result['all_checks_passed'],failed=[x for x in result['checks'] if not x['passed']],physics_steps=0,policy_steps_or_qp_calls=0),ensure_ascii=False))
    return 0 if result['all_checks_passed'] else 1


if __name__=='__main__':raise SystemExit(main())

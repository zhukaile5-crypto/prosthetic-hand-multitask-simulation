"""Validate the existing 16x8 response along a public robot task trajectory.

No kernel is fitted, no production model is changed, and no wider control domain
is admitted. --prepare-plan and --self-test do not construct or step a plant.
Only --run starts physics; it requires a fresh output and unchanged frozen inputs.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / 'calibration'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CALIBRATION))
from public_synergy_calibration import (sha, serial, write_json, public_packet,
    capture_state, restore_state, guard, ProbeStop, arrays)
from research_imports import FOUNDATION
from sensors import aggregate_patches

PLAN_PATH = CALIBRATION / 'plan_phase3_path_validation_v3.json'
MODEL_PATH = CALIBRATION / 'public_contact_model_phase2_v1.json'
INPUT_NAMES = ['thumb_alpha', 'index_alpha', 'middle_alpha', 'ring_alpha',
               'little_alpha', 'lift_reference_m', 'WRJ2_reference_rad', 'WRJ1_reference_rad']
OUTPUT_NAMES = [f'finger_pressure_{i}_N' for i in range(5)] + [f'finger_shear_{i}_N' for i in range(5)] + ['Fx_N', 'Fy_N', 'Fz_N', 'Mx_Nm', 'My_Nm', 'Mz_Nm']


def references(plant):
    return np.r_[plant.alpha, plant.lift_ref, plant.wrist_ref]


def _source_files():
    """Include imported code, the admitted plant assets/evidence, and input model."""
    paths = {'research/' + str(p.relative_to(ROOT)).replace('\\', '/'): p for p in
             (Path(__file__).resolve(), CALIBRATION/'public_synergy_calibration.py',
              ROOT/'research_imports.py', MODEL_PATH)}
    admission_path = FOUNDATION/'review/admission_manifest.json'
    admission = json.loads(admission_path.read_text(encoding='utf-8-sig'))
    names = set(admission['source_sha256']) | {'review/admission_manifest.json'}
    names.update(item['path'] for item in admission['evidence'].values())
    calibration = json.loads((FOUNDATION/admission['evidence']['calibration']['path']).read_text(encoding='utf-8-sig'))
    for entry in calibration['shapes'].values():
        entry = {**entry, **entry.get('response', {})}
        evidence = entry.get('selected_holdout', {})
        if not evidence:
            selected = float(entry.get('selected_radius', entry.get('trust_radius')))
            evidence = next((v for v in entry.get('holdout_by_radius', {}).values()
                             if abs(float(v.get('radius', -1))-selected) < 1e-10), {})
        for key in ('source_path', 'source', 'summary_path'):
            if evidence.get(key):
                p = Path(evidence[key])
                p = p if p.is_absolute() else FOUNDATION/p
                if not p.resolve().is_relative_to(FOUNDATION.resolve()):
                    raise ValueError('Selected foundation evidence must remain in preserved foundation')
                names.add(p.relative_to(FOUNDATION).as_posix())
    paths.update({'foundation/' + name.replace('\\', '/'): FOUNDATION/name for name in names})
    return paths


def make_plan():
    files = _source_files()
    return dict(schema='public_path_response_validation_phase3_v3',
        status='frozen_before_phase3_physical_data', created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Validate the unchanged phase2 16x8 kernel after cumulative robot-state transport; not model fitting or strategy performance.',
        shapes=['capsule', 'box', 'sphere'], mass_setup_kg=.12, seed=630301,
        physical_randomization=False, preparation_end_s=7.5, transport_start_s=9.3,
        checkpoint_elapsed_s=[0., .6, 1.2, 1.8, 2.4], interval_s=.02, control_dt_s=.005,
        branch_duration_s=.5, causal_window_s=.12, forecast_horizons_s=[.02, .04, .06, .1],
        micro_amplitude=[.004]*5 + [.0005, .002, .002],
        mixed_amplitude=[.006]*5 + [.0008, .003, .003], mixed_sequences=8,
        branch_order='zero, zero_replay, axis0 plus/minus through axis7 plus/minus, mixed00..07',
        planned_checkpoint_count=15, planned_branches_per_checkpoint=26, planned_branches=390,
        public_motion=dict(lift_origin_m=.1, lift_cosine_excursion_m=.008, lift_omega_rad_s=3.,
            WRJ2_amplitude_rad=.04, WRJ2_omega_rad_s=3., WRJ1_amplitude_rad=.025, WRJ1_omega_rad_s=2.),
        task_path_scope='Analytic, publicly declared user transport only; no scenario event_schedule, future force/friction/obstacle events, object pose, or object identity is used to generate commands.',
        finger_policy='Hold the five actual common-preparation synergy references fixed along the reference path; no ratio/joint policy acts after preparation.',
        model_gates=dict(pressure_rmse_N=.35, pressure_p95_N=.75,
            shear_rmse_N=.35, shear_p95_N=.75, force_axis_rmse_N=.05, moment_axis_rmse_Nm=.005),
        shear_gate_disclosure='These shear thresholds are newly declared before phase3 data; they are not retroactive phase2 preregistration.',
        stop_gates=dict(per_finger_pressure_N=5.02, sum_finger_pressure_N=18.02,
            penetration_m=.002, table_support_N=.05, minimum_object_lift_m=.07, maximum_warning_count=0),
        task_height_gate=dict(absolute_minimum_object_lift_m=.025, initial_static_minimum_object_lift_m=.07,
            relative_retention_tolerance_m=.005,
            formula='max(0.025, 0.070 + z_actual_robot_marker(t) - z_actual_robot_marker(handoff))',
            reference_marker='One fixed initial object-in-palm marker measured ONLY by the independent acceptance gate at common preparation end. Actual end-clock robot FK transports it for retention-height acceptance; user target tracking is checked separately. The marker never commands the hand or predicts contact outputs.',
            geometry_clock='Clone complete current MjData into independent gate scratch and call mj_kinematics only; object and palm geometry both at the same physical end qpos/time. Real plant, solver and warmstart are untouched.',
            preparation_rule='Common preparation must still meet the original static 0.070m gate at handoff before the task-conditioned gate is constructed.',
            extra_retention_rule='The same end-clock object-in-palm displacement from the handoff marker must remain <=0.005m, the existing task stability drift threshold.'),
        robot_tracking_gate=dict(tolerances=[.01,.015,.015],failure_dwell_s=.3,
            axes=['lift_m','WRJ2_rad','WRJ1_rad'],
            rule='Actual robot end-qpos minus the CURRENT public reference. Each axis must be outside its tolerance continuously for >=0.3s to fail; a short excursion is separately recorded.',
            dwell_clock='First observed bad physical-end time; duration is absolute time minus onset, never packet counts. Identical timestamps cannot accumulate dwell. Each physical branch restores the exact independent gate timer snapshot from its checkpoint.',
            source='Current public research engineering task tolerances: 10mm lift, 15mrad per wrist, 0.3s dwell; no numeric threshold was adjusted after the phase3 failure.'),
        revision_disclosure=dict(previous_plan=str(CALIBRATION/'plan_phase3_path_validation_v2.json'),
            previous_plan_sha256=sha(CALIBRATION/'plan_phase3_path_validation_v2.json'),
            previous_execution=str(CALIBRATION/'phase3_path_validation_v2'),
            previous_execution_summary_sha256=sha(CALIBRATION/'phase3_path_validation_v2/summary.json'),
            previous_stopped_trace_sha256=sha(CALIBRATION/'phase3_path_validation_v2/capsule/reference_segment_03/trace.npz'),
            reason='V2 used commanded robot FK as though it had already been reached. It stopped on legal finite-servo lag while grasp-relative drift was <0.086mm and all observed tracking errors stayed within 10mm/15mrad. V3 separates robot task tracking from object retention relative to the actual robot. V1/V2 are seen development data, preserved unchanged; this is physical-acceptance semantics, not an algorithm improvement.',
            archived_v2_script=str(CALIBRATION/'phase3_path_validation_revisions/v2/phase3_path_validation.py'),
            archived_v2_script_sha256=sha(CALIBRATION/'phase3_path_validation_revisions/v2/phase3_path_validation.py')),
        stop_entire_sweep_on_first_safety_violation=True,
        bounds='Finger commands remain in the original calibrated/reference/handoff intersection. Lift and wrist requests remain inside actual robot actuator/reference bounds; those mechanical bounds do not admit the phase2 model at new cumulative states.',
        unavailable_directions='Keep requested direction and explicit reason, attempted_physics=false; never fill an unmeasured response with zero or shrink its amplitude silently.',
        prediction='Use the frozen existing dense current_palm kernel, all known reference history from 7.5s, and only available past public output residuals. Future zero output is forbidden in causal prediction; it is used only for a separately labelled differential error.',
        coordinate='Validate current_palm outputs; checkpoint-fixed-frame raw wrenches are diagnostic and are not confused with the phase2 7.5s fixed frame.',
        clone_rule='Complete mj_copyData plus all foundation side state; zero/zero_replay compare every saved physical/control field including raw patches.',
        model_admitted=False, input_model=dict(path=str(MODEL_PATH), sha256=sha(MODEL_PATH)),
        information_boundary='Object lift/table support/contact identity may be saved or checked ONLY by independent physical gates/diagnostics, never by reference generation or causal predictor.',
        data_scope='One nominal public preparation per shape, one fixed standard mass and independent development seed; no robustness or performance holdout claim.',
        runtime_estimate=dict(simulated_branch_seconds=195., physical_substeps_approx=450000,
            output_note='Raw contact packets plus full numeric substeps may require hundreds of MB; no existing results are overwritten.'),
        sources={key:dict(path=str(p.resolve()), sha256=sha(p), bytes=p.stat().st_size)
                 for key,p in sorted(files.items())})


def check_sources(plan):
    changed = [key for key, entry in plan['sources'].items()
               if not Path(entry['path']).is_file() or sha(entry['path']) != entry['sha256']]
    if changed:
        raise RuntimeError('Frozen phase3 inputs changed: ' + ', '.join(changed))


def snapshot_sources(plan, out):
    records = {}
    for key, entry in plan['sources'].items():
        payload = Path(entry['path']).read_bytes()
        import hashlib
        if hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise RuntimeError('Source changed while snapshotting: ' + key)
        path = out/'sources'/key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records[key] = dict(**entry, snapshot_path=str(path.relative_to(out)), snapshot_sha256=sha(path))
    write_json(out/'source_snapshot_manifest.json', dict(files=records, all_bytes_verified=True))


def public_motion(elapsed, plan):
    p = plan['public_motion']; t = float(elapsed)
    return np.array([p['lift_origin_m']-p['lift_cosine_excursion_m']*(1.-math.cos(p['lift_omega_rad_s']*t)),
        p['WRJ2_amplitude_rad']*math.sin(p['WRJ2_omega_rad_s']*t),
        p['WRJ1_amplitude_rad']*math.sin(p['WRJ1_omega_rad_s']*t)])


def path_designs(base, lower, upper, plan):
    n = round(plan['branch_duration_s']/plan['interval_s'])+1
    zero = np.tile(base, (n,1)); designs = [('zero', zero), ('zero_replay', zero.copy())]
    for axis, amplitude in enumerate(plan['micro_amplitude']):
        for sign, label in ((1., 'plus'), (-1., 'minus')):
            path = zero.copy(); path[1:,axis] += sign*amplitude
            designs.append((f'micro_{axis}_{label}', path))
    H = np.ones((1,1))
    for _ in range(3): H = np.block([[H,H],[H,-H]])
    t = np.linspace(0.,1.,n)
    for sequence in range(plan['mixed_sequences']):
        path = zero.copy()
        for axis, amplitude in enumerate(plan['mixed_amplitude']):
            sign = H[sequence, [1,2,3,4,5,0,6,7][axis]]
            if base[axis]+sign*amplitude > upper[axis]+1e-12 or base[axis]+sign*amplitude < lower[axis]-1e-12:
                sign = -sign
            wave = np.sin(np.pi*t)**2*(.8+.2*np.cos(2*np.pi*(1+axis%3)*t+sequence*.37+axis*.23))
            path[:,axis] += sign*amplitude*wave
        designs.append((f'mixed_{sequence:02d}', path))
    return designs


def load_kernel(plan, shape):
    document = json.loads(Path(plan['input_model']['path']).read_text(encoding='utf-8'))
    if sha(plan['input_model']['path']) != plan['input_model']['sha256']:
        raise RuntimeError('Input model changed')
    entry = next(e for e in document['shapes'] if e['shape']==shape)
    coordinate = entry['coordinates']['current_palm']
    times = np.asarray(coordinate['response_kernel_times_s'],float)
    K = np.asarray(coordinate['response_to_20ms_interval_increment'],float)
    if K.shape != (len(times),16,8) or not np.isfinite(K).all() or times[0] != 0.:
        raise ValueError('Expected finite existing dense 16x8 kernel')
    return dict(times=times, K=K, original_input_availability=entry['input_availability'],
                original_measured_delta_min=entry['measured_input_delta_min'],
                original_measured_delta_max=entry['measured_input_delta_max'])


def input_effect(kernel, reference_times, reference_values, query):
    """Known 20ms finite-ramp response with persistent old step effects.

    No output measurement or future zero trajectory enters this function. Values
    after query only describe a known task command; interval starts >=query have
    exactly no effect. The last kernel value is a declared persistence assumption.
    """
    ts = np.asarray(reference_times,float); values = np.asarray(reference_values,float)
    if len(ts)<2 or query<=ts[0]+1e-12:return np.zeros(16)
    starts = ts[:-1]; mask = starts < query-1e-10
    starts = starts[mask]; delta = np.diff(values,axis=0)[mask]
    ages = query-starts
    flat = kernel['K'].reshape(len(kernel['times']),128)
    response = np.stack([np.interp(ages,kernel['times'],flat[:,i],left=0.,right=flat[-1,i])
                         for i in range(128)],axis=-1).reshape(len(ages),16,8)
    return np.einsum('koi,ki->o',response,delta)


def error_summary(errors, plan):
    errors = np.asarray(errors,float)
    if not len(errors):return dict(available=False, vectors=0, all_engineering_gates_passed=False)
    rmse = np.sqrt(np.mean(errors**2,axis=0)); p95 = np.quantile(abs(errors),.95,axis=0)
    gates = plan['model_gates']
    threshold = np.r_[np.full(5,gates['pressure_rmse_N']),np.full(5,gates['shear_rmse_N']),
                      np.full(3,gates['force_axis_rmse_N']),np.full(3,gates['moment_axis_rmse_Nm'])]
    passed = rmse <= threshold
    passed[:5] &= p95[:5] <= gates['pressure_p95_N']
    passed[5:10] &= p95[5:10] <= gates['shear_p95_N']
    return dict(available=True, vectors=len(errors), rmse_per_output=rmse,
        p95_abs_per_output=p95, max_abs_per_output=np.max(abs(errors),axis=0),
        engineering_gate_per_output=passed, all_engineering_gates_passed=bool(passed.all()),
        output_order=OUTPUT_NAMES, units=['N']*13+['Nm']*3,
        engineering_thresholds_not_statistical_or_hard_error_bounds=True)


def causal_errors(kernel, reference_times, reference_values, past_rows, rows, plan):
    """Complete rolling predictions. Signature intentionally has no zero branch."""
    selected = rows[::round(plan['interval_s']/plan['control_dt_s'])]
    times = np.array([r['time'] for r in selected]); measured = np.stack([r['y_palm'] for r in selected])
    modeled = np.stack([input_effect(kernel,reference_times,reference_values,t) for t in times])
    history = [r for r in past_rows if r['time'] < times[0]-1e-10]
    ht = np.array([r['time'] for r in history]); hy = np.asarray([r['y_palm'] for r in history]).reshape(-1,16)
    he = np.asarray([input_effect(kernel,reference_times,reference_values,t) for t in ht]).reshape(-1,16)
    result = {str(float(h)):[] for h in plan['forecast_horizons_s']}; details=[]
    for origin, now in enumerate(times):
        past_t = np.r_[ht,times[:origin+1]]
        residual = np.vstack((hy-he,measured[:origin+1]-modeled[:origin+1]))
        keep = (past_t>=now-plan['causal_window_s']-1e-10)&(past_t<=now+1e-10)
        tt = past_t[keep]; yy = residual[keep]
        if len(tt)>=4:
            x=tt-tt.mean(); slope=x@(yy-yy.mean(axis=0))/max(1e-12,float(x@x))
            level=yy.mean(axis=0)+slope*(now-tt.mean())
        else: slope=np.zeros(16); level=yy[-1]
        for horizon in plan['forecast_horizons_s']:
            target=origin+round(horizon/plan['interval_s'])
            if target>=len(times):continue
            prediction=level+horizon*slope+modeled[target]
            error=prediction-measured[target]
            result[str(float(horizon))].append(error)
            details.append(dict(origin_time_s=float(now),horizon_s=horizon,error=error))
    return dict(complete_causal_by_horizon={key:error_summary(value,plan) for key,value in result.items()},
                error_vectors=details, no_future_zero_used=True)


def task_height_requirement(static_minimum, absolute_minimum, public_task_delta_z):
    return max(float(absolute_minimum), float(static_minimum)+float(public_task_delta_z))


class RobotTrackingDwell:
    """Branch-clonable physical-time task monitor, independent of object state."""
    def __init__(self,tolerances,dwell_s):
        self.tolerances=np.asarray(tolerances,float)
        self.dwell_s=float(dwell_s)
        if self.tolerances.shape!=(3,) or not np.isfinite(self.tolerances).all() or np.any(self.tolerances<=0.) or not np.isfinite(self.dwell_s) or self.dwell_s<=0.:
            raise ValueError('Explicit positive finite robot tracking tolerances/dwell required')
        self.last_time=None;self.outside_since=[None]*3

    def capture_state(self):
        return dict(last_time=self.last_time,outside_since=self.outside_since.copy())

    def restore_state(self,state):
        self.last_time=state['last_time'];self.outside_since=state['outside_since'].copy()

    def update(self,time_s,error):
        now=float(time_s);error=np.asarray(error,float)
        if not np.isfinite(now) or error.shape!=(3,) or not np.isfinite(error).all():
            raise ValueError('Finite robot time/error required')
        if self.last_time is not None and now<self.last_time-1e-10:
            raise RuntimeError('Independent tracking clock rewound without restoring its checkpoint')
        outside=abs(error)>self.tolerances;duration=np.zeros(3)
        for axis,bad in enumerate(outside):
            if bad:
                if self.outside_since[axis] is None:self.outside_since[axis]=now
                duration[axis]=max(0.,now-self.outside_since[axis])
            else:self.outside_since[axis]=None
        self.last_time=now
        return dict(gate_robot_tracking_error=error.copy(),gate_robot_tracking_outside=outside,
            gate_robot_tracking_dwell_s=duration,gate_robot_tracking_failed=duration>=self.dwell_s-1e-10)


class RigidTransportHeightGate:
    """Independent physical acceptance geometry, never a controller sensor.

    Object generalized coordinates are permitted ONLY here for offline physical
    acceptance. Public command generation and causal response prediction never
    receive this object or its initial marker. Kinematics is recomputed in cloned
    scratch data, so mj_step's old derived geometry cannot mix physical clocks.
    """
    def __init__(self,plant,plan,_api=None):
        if _api is None:
            import mujoco
        else:mujoco=_api
        self.mj=mujoco;self.plant=plant;self.config=plan['task_height_gate']
        self.geometry=mujoco.MjData(plant.model)
        self.reference_robot=mujoco.MjData(plant.model)
        self.robot_anchor=plant.data.qpos[plant.raw_qadr].copy()
        self.wrist_qadr=[int(plant.model.joint(name).qposadr[0]) for name in ('rh_WRJ2','rh_WRJ1')]
        obj,palm,R=self._actual_geometry()
        self.marker=R.T@(obj-palm)
        self.anchor_reference=references(plant).copy()
        self.anchor_reference_marker_z=self._reference_marker_height(self.anchor_reference)
        self.anchor_actual_marker_z=float((palm+R@self.marker)[2])
        self.anchor_time=float(plant.data.time)
        tracking=plan['robot_tracking_gate']
        self.tracking=RobotTrackingDwell(tracking['tolerances'],tracking['failure_dwell_s'])

    def capture_state(self):return dict(tracking=self.tracking.capture_state())

    def restore_state(self,state):self.tracking.restore_state(copy.deepcopy(state['tracking']))

    def _actual_geometry(self):
        p=self.plant
        self.mj.mj_copyData(self.geometry,p.model,p.data)
        self.mj.mj_kinematics(p.model,self.geometry)
        return (self.geometry.xpos[p.object].copy(),self.geometry.xpos[p.palm].copy(),
                self.geometry.xmat[p.palm].reshape(3,3).copy())

    def _reference_marker_height(self,request):
        p=self.plant;d=self.reference_robot
        # Only known robot encoder configuration and public task references enter
        # this FK. No current object coordinate is copied into reference_robot.
        d.qpos[p.raw_qadr]=self.robot_anchor
        d.qpos[p.lift_qadr]=float(request[5])
        d.qpos[self.wrist_qadr]=np.asarray(request[6:])
        self.mj.mj_kinematics(p.model,d)
        return float((d.xpos[p.palm]+d.xmat[p.palm].reshape(3,3)@self.marker)[2])

    def measure(self):
        p=self.plant;obj,palm,R=self._actual_geometry()
        relative=R.T@(obj-palm)
        delta=float((palm+R@self.marker)[2])-self.anchor_actual_marker_z
        command_delta=self._reference_marker_height(references(p))-self.anchor_reference_marker_z
        floor=task_height_requirement(self.config['initial_static_minimum_object_lift_m'],
            self.config['absolute_minimum_object_lift_m'],delta)
        actual=np.r_[p.data.qpos[p.lift_qadr],p.data.qpos[self.wrist_qadr]]
        tracking=self.tracking.update(float(p.data.time),actual-references(p)[5:])
        return dict(object_lift_m=float(obj[2]-p.initial_position[2]),
            gate_object_position_end=obj,gate_palm_position_end=palm,gate_palm_rotation_end=R,
            gate_object_relative_position_end=relative,gate_relative_drift_m=float(np.linalg.norm(relative-self.marker)),
            gate_actual_robot_delta_z_m=delta,gate_task_reference_delta_z_m=command_delta,
            gate_minimum_object_lift_m=floor,gate_actual_robot_motion_end=actual,
            gate_geometry_time_s=float(p.data.time),gate_geometry_clock='physical_end_qpos_independent_scratch_FK',**tracking)

    def metadata(self):
        return dict(anchor_time_s=self.anchor_time,initial_object_in_palm_marker_m=self.marker,
            public_anchor_reference=self.anchor_reference,public_anchor_reference_marker_z_m=self.anchor_reference_marker_z,
            actual_anchor_marker_z_m=self.anchor_actual_marker_z,
            scope='Independent acceptance truth only; not available to path generator, online controller or causal predictor.',
            real_plant_or_solver_state_modified=False)


def packet(plant, contacts, R0, p0, height_gate=None):
    row = public_packet(plant,contacts,R0,p0)
    _, diagnostic = aggregate_patches(contacts['patches'],support_axis=plant.data.xmat[plant.palm].reshape(3,3).T@np.array([0.,0.,1.]))
    shear=np.asarray(diagnostic['shear_magnitude']); row['shear_magnitude']=shear
    row['slip_magnitude']=np.asarray(diagnostic['slip_magnitude'])
    row['contact_present']=np.asarray(diagnostic['contact_present'],bool)
    row['y']=np.r_[row['pressure'],shear,row['wrench_reference']]
    row['y_palm']=np.r_[row['pressure'],shear,row['wrench_palm']]
    row['input_reference']=references(plant)
    row['robot_lift_position_m']=float(plant.data.qpos[plant.lift_qadr])
    row['robot_wrist_position_rad']=np.array([plant.data.qpos[plant.model.joint(name).qposadr[0]] for name in ('rh_WRJ2','rh_WRJ1')])
    row['warning_count']=int(sum(int(x.number) for x in plant.data.warning))
    row['non_target_contact_pressure_N']=float(sum(p['normal'] for p in contacts['patches'] if not p.get('is_target',False)))
    if height_gate is not None:
        row['legacy_solver_geometry_object_lift_m']=row['object_lift_m']
        row.update(height_gate.measure())
    return row


def physical_guard(row,plan):
    effective_plan=plan
    if 'gate_minimum_object_lift_m' in row:
        effective_plan={**plan,'stop_gates':{**plan['stop_gates'],
            'minimum_object_lift_m':row['gate_minimum_object_lift_m']}}
    guard(row,effective_plan)
    if 'gate_relative_drift_m' in row and row['gate_relative_drift_m']>plan['task_height_gate']['relative_retention_tolerance_m']:
        raise ProbeStop('object_relative_retention')
    if 'gate_robot_tracking_failed' in row and np.any(row['gate_robot_tracking_failed']):
        raise ProbeStop('robot_task_tracking_dwell')
    if row['warning_count']>plan['stop_gates']['maximum_warning_count']:raise ProbeStop('physics_warning')
    if not np.isfinite(np.r_[row['qpos'],row['qvel'],row['ctrl']]).all():raise ProbeStop('nonfinite_robot_or_physical_state')


def execute_path(plant, state, R0, p0, path, lower, upper, plan, folder, height_gate=None,height_gate_state=None):
    independent_state=(height_gate.capture_state() if height_gate_state is None else height_gate_state) if height_gate is not None else None
    restore_state(plant,state);folder.mkdir(parents=True,exist_ok=False)
    if height_gate is not None:height_gate.restore_state(independent_state)
    path=np.asarray(path,float);base=references(plant);physical=[];control=[];failure=None;last=-np.inf
    if np.max(abs(path[0]-base))>1e-9:raise ValueError('Path must start at the actual restored reference')
    outside=np.flatnonzero(np.any((path<lower[None,:]-1e-12)|(path>upper[None,:]+1e-12),axis=0))
    if len(outside):
        metadata=dict(available=False,complete=False,attempted_physics=False,
            unavailable_reason='requested_path_outside_original_finger_or_mechanical_motion_bounds',unavailable_inputs=outside,
            requested_path=path,actual_base=base,lower=lower,upper=upper)
        write_json(folder/'metadata.json',metadata);return None,None,metadata
    original=plant._contacts
    def hooked(patches=True):
        nonlocal last
        contacts=original(patches)
        if float(plant.data.time)>last+1e-12:
            row=packet(plant,contacts if patches else original(True),R0,p0,height_gate)
            row['interval_start']=float(plant.data.time-plant.model.opt.timestep)
            row['duration']=float(plant.model.opt.timestep)
            physical.append(row);last=float(plant.data.time);physical_guard(row,plan)
        return contacts
    control.append(packet(plant,original(True),R0,p0,height_gate))
    try:
        physical_guard(control[0],plan);plant._contacts=hooked
        for interval in range(len(path)-1):
            for sub in range(4):
                target=path[interval]+(sub+1)/4.*(path[interval+1]-path[interval])
                plant.execute_delta(target[:5]-plant.alpha)
                plant.set_motion(float(target[5]),target[6:])
                if np.max(abs(references(plant)-target))>1e-9:raise ProbeStop('requested_actual_reference_mismatch_or_rate_saturation')
                plant.advance(plan['control_dt_s'])
                control.append(packet(plant,original(True),R0,p0,height_gate));physical_guard(control[-1],plan)
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc))
    finally:plant._contacts=original
    rows=physical or control
    metadata=dict(available=True,attempted_physics=True,complete=failure is None,failure=failure,
        requested_path=path,executed_path=np.stack([r['input_reference'] for r in control]),actual_base=base,
        lower=lower,upper=upper,maximum_pressure_N=np.max([r['pressure'] for r in rows],axis=0),
        maximum_shear_N=np.max([r['shear_magnitude'] for r in rows],axis=0),
        maximum_slip_m_s=np.max([r['slip_magnitude'] for r in rows],axis=0),
        maximum_penetration_m=max(r['penetration'] for r in rows),
        maximum_non_target_contact_pressure_N=max(r['non_target_contact_pressure_N'] for r in rows),
        maximum_warning_count=max(r['warning_count'] for r in rows),
        control_samples=len(control),physical_substeps=len(physical),raw_gate_scope='all_external_finger_skin',
        object_truth_scope='independent gates and saved diagnostics only; never path generation or predictor input')
    if height_gate is not None:
        metadata.update(minimum_observed_object_lift_m=min(r['object_lift_m'] for r in rows),
            minimum_task_height_margin_m=min(r['object_lift_m']-r['gate_minimum_object_lift_m'] for r in rows),
            maximum_relative_drift_m=max(r['gate_relative_drift_m'] for r in rows),
            maximum_robot_tracking_error=np.max([abs(r['gate_robot_tracking_error']) for r in rows],axis=0),
            maximum_robot_tracking_dwell_s=np.max([r['gate_robot_tracking_dwell_s'] for r in rows],axis=0),
            height_gate='actual robot rigid-transport marker plus absolute floor and grasp-relative retention; public-task tracking separate',
            geometry_clock='independent same-end-clock object/palm FK; solver state untouched')
    packed={**arrays(control),**arrays(physical,'physical_')}
    packed['metadata']=json.dumps(serial(metadata),ensure_ascii=False,allow_nan=False)
    np.savez_compressed(folder/'trace.npz',**packed)
    metadata['trace_sha256']=sha(folder/'trace.npz');write_json(folder/'metadata.json',metadata)
    return control,physical,metadata


def replay_equal(first,second):
    for left,right in zip(first[:2],second[:2]):
        if left is None or right is None or len(left)!=len(right):return False
        for a,b in zip(left,right):
            if set(a)!=set(b) or any(not np.array_equal(np.asarray(a[k]),np.asarray(b[k])) for k in a):return False
    return True


def input_bounds(plant,calibration,handoff):
    reference=np.asarray(calibration['reference_alpha']);radius=float(calibration['selected_radius'])
    lower=np.maximum.reduce([plant.closure_min,reference-radius,handoff[:5]-radius])
    upper=np.minimum.reduce([plant.closure_max,reference+radius,handoff[:5]+radius])
    wrists=plant.model.actuator_ctrlrange[[plant.aid(k) for k in ('rh_A_WRJ2','rh_A_WRJ1')]]
    return np.r_[lower,0.,wrists[:,0]],np.r_[upper,plant.config.lift_target,wrists[:,1]]


def validate_checkpoint(plant,calibration,handoff,prefix_times,prefix_values,past_rows,plan,folder,shape,height_gate):
    check_sources(plan)
    base=references(plant);lower,upper=input_bounds(plant,calibration,handoff)
    R0=plant.data.xmat[plant.palm].reshape(3,3).copy();p0=plant.data.xpos[plant.palm].copy()
    state=capture_state(plant);gate_state=height_gate.capture_state();kernel=load_kernel(plan,shape);folder.mkdir()
    write_json(folder/'checkpoint.json',dict(time_s=plant.time,base_reference=base,
        cumulative_reference_offset_from_handoff=base-handoff,
        actual_lift_position_m=float(plant.data.qpos[plant.lift_qadr]),
        actual_wrist_position_rad=[float(plant.data.qpos[plant.model.joint(n).qposadr[0]]) for n in ('rh_WRJ2','rh_WRJ1')],
        actual_robot_joint_position=plant.data.qpos[plant.raw_qadr].copy(),
        actual_robot_joint_velocity=plant.data.qvel[plant.raw_dadr].copy(),
        lower=lower,upper=upper,original_model_availability=kernel['original_input_availability'],
        original_model_measured_delta_min=kernel['original_measured_delta_min'],
        original_model_measured_delta_max=kernel['original_measured_delta_max'],
        phase3_is_validation_not_model_admission=True,reference_history_times=prefix_times,
        independent_gate_timer_checkpoint=gate_state,
        reference_history_values=prefix_values,causal_history=[dict(time=r['time'],y_palm=r['y_palm']) for r in past_rows[-60:]]))
    results={};scores=[]
    try:
        for label,path in path_designs(base,lower,upper,plan):
            rows,physical,metadata=execute_path(plant,state,R0,p0,path,lower,upper,plan,folder/label,height_gate,gate_state)
            results[label]=(rows,physical,metadata)
            print(shape,folder.name,label,'available',metadata['available'],'complete',metadata['complete'],flush=True)
            if metadata['available'] and not metadata['complete']:
                raise ProbeStop(f'{shape}/{folder.name}/{label}: {metadata["failure"]}')
            if label=='zero_replay':
                equal=replay_equal(results['zero'],results['zero_replay'])
                write_json(folder/'restore_replay_check.json',dict(passed=equal,
                    control_samples=len(rows or []),physical_substeps=len(physical or []),
                    comparison='all saved control and physical fields, including qpos/qvel/ctrl/raw patches'))
                if not equal:raise ProbeStop('complete_MjData_restore_replay_mismatch')
            if not metadata['available'] or label in ('zero','zero_replay'):continue
            bt=rows[0]['time']+np.arange(len(path))*plan['interval_s']
            all_times=np.r_[prefix_times,bt[1:]];all_values=np.vstack((prefix_values,path[1:]))
            score=causal_errors(kernel,all_times,all_values,past_rows[-60:],rows,plan)
            sampled=rows[::4];zero=results['zero'][0][::4]
            zero_values=np.vstack((prefix_values,np.tile(base,(len(path)-1,1))))
            differential=[]
            for actual,neutral in zip(sampled,zero):
                command_effect=input_effect(kernel,all_times,all_values,actual['time'])-input_effect(kernel,all_times,zero_values,actual['time'])
                differential.append(actual['y_palm']-neutral['y_palm']-command_effect)
            score.update(branch=label,differential_response_error=error_summary(differential,plan),
                differential_uses_future_zero_offline_only=True,original_kernel_unchanged=True,model_admitted=False)
            write_json(folder/label/'validation.json',score)
            scores.append(dict(branch=label,complete_causal_by_horizon=score['complete_causal_by_horizon'],
                differential_response_error=score['differential_response_error']))
        summary=dict(complete=True,planned_branches=26,executed_branches=sum(v[2]['available'] for v in results.values()),
            unavailable_branches=[key for key,value in results.items() if not value[2]['available']],
            all_available_complete_causal_engineering_gates_passed=bool(scores and all(v['all_engineering_gates_passed']
                for score in scores for v in score['complete_causal_by_horizon'].values())),
            branch_scores=scores,model_admitted=False,scope='This checkpoint and nominal public path only; no domain expansion or performance evidence.')
        write_json(folder/'summary.json',summary);return summary
    finally:
        restore_state(plant,state);height_gate.restore_state(gate_state)


def run(plan,out,plan_path):
    check_sources(plan)
    if out.exists() or not out.resolve().is_relative_to(CALIBRATION.resolve()) or out.resolve()==CALIBRATION.resolve():
        raise ValueError('Choose a fresh output strictly inside calibration')
    out.mkdir(parents=True,exist_ok=False);write_json(out/'plan.json',plan);snapshot_sources(plan,out)
    import mujoco
    from session import TrialSession
    write_json(out/'environment.json',dict(python=sys.version,executable=sys.executable,numpy=np.__version__,
        mujoco=mujoco.__version__,platform=platform.platform(),plan_path=str(plan_path),plan_sha256=sha(plan_path)))
    summary=dict(execution_complete=False,model_admitted=False,calibration_not_performance=True,
        input_kernel_modified=False,checkpoints=[],started_utc=datetime.now(timezone.utc).isoformat())
    try:
        for shape in plan['shapes']:
            session=TrialSession(method='ratio',shape=shape,scenario='nominal',seed=plan['seed'],mode='manual',split='demonstration')
            plant=session.plant;case=out/shape;case.mkdir();preparation=[]
            while session.time<plan['preparation_end_s']-1e-9:
                session.step_one_sensor_tick()
                R=plant.data.xmat[plant.palm].reshape(3,3);p=plant.data.xpos[plant.palm]
                preparation.append(packet(plant,plant._contacts(True),R,p))
            if session.handoff_attempted or abs(plant.config.mass-plan['mass_setup_kg'])>1e-12:
                raise ProbeStop('unexpected_preparation_or_nonstandard_mass')
            # Preparation acceptance retains the original static 70mm floor.
            R=plant.data.xmat[plant.palm].reshape(3,3);p=plant.data.xpos[plant.palm]
            physical_guard(packet(plant,plant._contacts(True),R,p),plan)
            np.savez_compressed(case/'preparation_trace.npz',**arrays(preparation),
                foundation_substep_records=np.asarray(plant.substep_records))
            handoff=references(plant);lower,upper=input_bounds(plant,session.calibration,handoff)
            height_gate=RigidTransportHeightGate(plant,plan)
            write_json(case/'independent_height_gate_reference.json',height_gate.metadata())
            write_json(case/'setup.json',dict(config=session.config.to_dict(),calibration=session.calibration,
                handoff_reference=handoff,preparation_controller_acted=False,preparation_trace_sha256=sha(case/'preparation_trace.npz'),
                handoff_actual_lift_m=float(plant.data.qpos[plant.lift_qadr]),
                handoff_actual_wrist_rad=[float(plant.data.qpos[plant.model.joint(n).qposadr[0]]) for n in ('rh_WRJ2','rh_WRJ1')],
                handoff_actual_robot_joint_position=plant.data.qpos[plant.raw_qadr].copy(),
                all_future_commands_use_only_public_transport_definition=True,
                reference_path_finger_commands='fixed at actual common-preparation handoff'))
            past_rows=preparation[-60:];prefix_times=[plant.time];prefix_values=[handoff.copy()]
            for number,elapsed in enumerate(plan['checkpoint_elapsed_s']):
                target_time=plan['transport_start_s']+elapsed
                steps=round((target_time-plant.time)/plan['interval_s'])
                times=plant.time+np.arange(steps+1)*plan['interval_s']
                path=np.array([np.r_[handoff[:5], public_motion(max(0.,t-plan['transport_start_s']),plan)] for t in times])
                path[0]=references(plant)
                if steps:
                    state=capture_state(plant);R=plant.data.xmat[plant.palm].reshape(3,3).copy();p=plant.data.xpos[plant.palm].copy()
                    rows,physical,metadata=execute_path(plant,state,R,p,path,lower,upper,plan,case/f'reference_segment_{number:02d}',height_gate)
                    if not metadata['available'] or not metadata['complete']:
                        raise ProbeStop(f'reference_path_failed_at_checkpoint_{number}: {metadata.get("failure",metadata.get("unavailable_reason"))}')
                    past_rows.extend(rows[1:]);past_rows=past_rows[-60:]
                    prefix_times.extend(times[1:].tolist());prefix_values.extend(path[1:].copy())
                result=validate_checkpoint(plant,session.calibration,handoff,np.asarray(prefix_times),
                    np.asarray(prefix_values),past_rows,plan,case/f'checkpoint_{number:02d}',shape,height_gate)
                summary['checkpoints'].append(dict(shape=shape,index=number,transport_elapsed_s=elapsed,
                    complete=result['complete'],executed_branches=result['executed_branches'],
                    unavailable_branches=result['unavailable_branches'],
                    all_available_complete_causal_engineering_gates_passed=result['all_available_complete_causal_engineering_gates_passed']))
                write_json(out/'summary.json',summary)
        summary['execution_complete']=True
    except Exception as exc:
        summary['failure']=dict(type=type(exc).__name__,message=str(exc))
    finally:
        changed=[key for key,item in plan['sources'].items() if not Path(item['path']).is_file() or sha(item['path'])!=item['sha256']]
        summary.update(sources_unchanged=not changed,changed_sources=changed,ended_utc=datetime.now(timezone.utc).isoformat())
        if changed:summary['execution_complete']=False
        summary['all_checkpoint_engineering_gates_passed']=bool(summary['execution_complete'] and all(
            c['all_available_complete_causal_engineering_gates_passed'] for c in summary['checkpoints']))
        write_json(out/'summary.json',summary)
        write_json(out/'execution_complete.json',summary)
    return summary


def self_test():
    """Only algebra/contract checks; intentionally no MuJoCo objects or steps."""
    plan=dict(branch_duration_s=.5,interval_s=.02,control_dt_s=.005,micro_amplitude=[.004]*5+[.0005,.002,.002],
        mixed_amplitude=[.006]*5+[.0008,.003,.003],mixed_sequences=8,causal_window_s=.12,forecast_horizons_s=[.02,.04,.06,.1],
        model_gates=dict(pressure_rmse_N=.35,pressure_p95_N=.75,shear_rmse_N=.35,shear_p95_N=.75,force_axis_rmse_N=.05,moment_axis_rmse_Nm=.005),
        public_motion=dict(lift_origin_m=.1,lift_cosine_excursion_m=.008,lift_omega_rad_s=3.,WRJ2_amplitude_rad=.04,WRJ2_omega_rad_s=3.,WRJ1_amplitude_rad=.025,WRJ1_omega_rad_s=2.))
    times=np.arange(101)*.005;K=np.zeros((101,16,8));K[1:,:8,:]=np.eye(8)[None,:,:];kernel=dict(times=times,K=K)
    rt=np.arange(26)*.02;rv=np.zeros((26,8));rv[1:,0]=.004
    checks={}
    checks['step_effect_persists_after_kernel_memory']=bool(abs(input_effect(kernel,rt,rv,2.)[0]-.004)<1e-12)
    checks['future_commands_do_not_affect_past_outputs']=bool(np.all(input_effect(kernel,rt,rv,0.)==0))
    rv2=rv.copy();rv2[10:,1]=.123
    checks['future_changed_commands_leave_prior_prediction_identical']=bool(np.array_equal(input_effect(kernel,rt,rv,.1),input_effect(kernel,rt,rv2,.1)))
    base=np.r_[np.full(5,.5),.1,0.,0.];lo=np.r_[np.zeros(5),0.,-.5,-.5];hi=np.r_[np.ones(5),.1,.5,.5]
    designs=path_designs(base,lo,hi,plan)
    checks['checkpoint_declares_all_26_branches']=bool(len(designs)==26)
    checks['unavailable_positive_lift_not_filled_with_zero']=bool(designs[12][1][1,5]>.1 and designs[13][1][1,5]<.1)
    checks['mixed_sequences_start_and_end_at_clone']=bool(all(np.allclose(path[[0,-1]],base) for label,path in designs if label.startswith('mixed')))
    checks['public_path_cumulative_excursion_exceeds_micro_not_confused']=bool(np.max(abs(public_motion(1.2,plan)-public_motion(0.,plan)))>.01)
    rows=[dict(time=float(t),y_palm=np.r_[np.full(16,2.)]+input_effect(kernel,rt,rv,t)) for t in np.arange(101)*.005]
    result=causal_errors(kernel,rt,rv,[],rows,plan)
    checks['causal_prediction_matches_known_synthetic_step']=bool(all(np.max(v['max_abs_per_output'])<1e-10 for v in result['complete_causal_by_horizon'].values()))
    changed=[dict(time=row['time'],y_palm=row['y_palm'].copy()) for row in rows]
    for row in changed:
        if row['time']>.2:row['y_palm']+=.1
    late=causal_errors(kernel,rt,rv,[],changed,plan)
    first=[v for v in result['error_vectors'] if v['origin_time_s']+v['horizon_s']<=.2+1e-10]
    second=[v for v in late['error_vectors'] if v['origin_time_s']+v['horizon_s']<=.2+1e-10]
    checks['future_measured_outputs_do_not_change_earlier_forecasts']=bool(all(np.array_equal(a['error'],b['error']) for a,b in zip(first,second)))
    errors=np.zeros((20,16));errors[:,5]=.36
    checks['separate_shear_gate_is_enforced']=bool(not error_summary(errors,plan)['all_engineering_gates_passed'])
    errors=np.zeros((20,16));errors[:,13]=.006
    checks['moment_unit_gate_not_mixed_with_newtons']=bool(not error_summary(errors,plan)['all_engineering_gates_passed'])
    checks['dynamic_height_preserves_original_static_floor_at_zero_motion']=bool(task_height_requirement(.07,.025,0.)==.07)
    checks['dynamic_height_follows_public_downward_displacement']=bool(abs(task_height_requirement(.07,.025,-.016)-.054)<1e-12)
    checks['dynamic_height_never_passes_below_absolute_floor']=bool(task_height_requirement(.07,.025,-.2)==.025)
    # A pure-array kinematics stub checks object/robot clock isolation. It is not
    # a substitute for the separately planned physical validation.
    from types import SimpleNamespace
    class KinematicStub:
        @staticmethod
        def MjData(model):
            return SimpleNamespace(qpos=np.zeros(6),xpos=np.zeros((2,3)),
                xmat=np.tile(np.eye(3).reshape(1,9),(2,1)),qacc_warmstart=np.array([3.]))
        @staticmethod
        def mj_copyData(target,model,source):
            for key in ('qpos','xpos','xmat','qacc_warmstart'):
                getattr(target,key)[:]=getattr(source,key)
        @staticmethod
        def mj_kinematics(model,data):
            a,b=data.qpos[1:3]
            Rx=np.array([[1.,0.,0.],[0.,np.cos(a),-np.sin(a)],[0.,np.sin(a),np.cos(a)]])
            Ry=np.array([[np.cos(b),0.,np.sin(b)],[0.,1.,0.],[-np.sin(b),0.,np.cos(b)]])
            data.xpos[0]=[0.,0.,data.qpos[0]];data.xmat[0]=(Ry@Rx).reshape(9)
            data.xpos[1]=data.qpos[3:6]
    robot=SimpleNamespace(model=SimpleNamespace(joint=lambda name:SimpleNamespace(qposadr=[1 if name=='rh_WRJ2' else 2])),
        raw_qadr=np.array([1,2]),lift_qadr=0,object=1,palm=0,alpha=np.ones(5)*.5,
        lift_ref=.1,wrist_ref=np.zeros(2),initial_position=np.array([.01,.02,.1]))
    robot.data=KinematicStub.MjData(robot.model);robot.data.time=7.5
    robot.data.qpos[:]=[.1,0.,0.,.01,.02,.172];KinematicStub.mj_kinematics(robot.model,robot.data)
    gate_plan=dict(task_height_gate=dict(initial_static_minimum_object_lift_m=.07,
        absolute_minimum_object_lift_m=.025,relative_retention_tolerance_m=.005),
        robot_tracking_gate=dict(tolerances=[.01,.015,.015],failure_dwell_s=.3))
    gate=RigidTransportHeightGate(robot,gate_plan,_api=KinematicStub)
    robot.lift_ref=.084;robot.wrist_ref[:]=[.04,.02];robot.data.qpos[:3]=[.084,.04,.02]
    fresh=KinematicStub.MjData(robot.model);fresh.qpos[:]=robot.data.qpos;KinematicStub.mj_kinematics(robot.model,fresh)
    robot.data.qpos[3:]=fresh.xpos[0]+fresh.xmat[0].reshape(3,3)@gate.marker
    original_data={key:getattr(robot.data,key).copy() for key in ('qpos','xpos','xmat','qacc_warmstart')}
    measurement=gate.measure()
    checks['scratch_FK_recomputes_end_geometry_without_touching_plant']=bool(all(np.array_equal(getattr(robot.data,key),value) for key,value in original_data.items()) and not np.array_equal(measurement['gate_palm_position_end'],robot.data.xpos[0]))
    checks['rigid_public_lift_and_wrist_motion_preserves_relative_marker']=bool(measurement['gate_relative_drift_m']<1e-12)
    checks['height_gate_includes_wrist_rotation_not_only_lift']=bool(abs(measurement['gate_actual_robot_delta_z_m']-(-.016))>1e-5)
    checks['correct_task_motion_keeps_original_two_mm_height_margin']=bool(abs(measurement['object_lift_m']-measurement['gate_minimum_object_lift_m']-.002)<1e-12)
    robot.data.qpos[3]+=.006;slipped=gate.measure()
    checks['current_object_slip_cannot_relax_actual_robot_height_gate']=bool(abs(slipped['gate_minimum_object_lift_m']-measurement['gate_minimum_object_lift_m'])<1e-12 and slipped['gate_relative_drift_m']>.005)
    tracker=RobotTrackingDwell([.01,.015,.015],.3)
    tracker.update(0.,[0.,.02,0.]);checkpoint=tracker.capture_state()
    first=tracker.update(.299,[0.,.02,0.]);repeat=tracker.update(.299,[0.,.02,0.])
    checks['tracking_repeated_packets_do_not_accumulate_dwell']=bool(np.array_equal(first['gate_robot_tracking_dwell_s'],repeat['gate_robot_tracking_dwell_s']) and not np.any(repeat['gate_robot_tracking_failed']))
    checks['tracking_continuous_three_tenths_fails_at_original_threshold']=bool(tracker.update(.3,[0.,.02,0.])['gate_robot_tracking_failed'][1])
    tracker.restore_state(checkpoint);replayed=tracker.update(.299,[0.,.02,0.])
    checks['tracking_branch_restore_repeats_original_timer']=bool(np.array_equal(replayed['gate_robot_tracking_dwell_s'],first['gate_robot_tracking_dwell_s']))
    tracker.update(.3,[0.,0.,0.]);new_episode=tracker.update(.4,[0.,.02,0.])
    checks['good_tracking_breaks_failure_dwell']=bool(new_episode['gate_robot_tracking_dwell_s'][1]==0. and not np.any(new_episode['gate_robot_tracking_failed']))
    try:tracker.update(.1,[0.,0.,0.]);rewind_rejected=False
    except RuntimeError:rewind_rejected=True
    checks['unrestored_tracking_time_rewind_is_rejected']=rewind_rejected
    # A legal servo lag changes command-FK height but not actual retention floor.
    robot.data.qpos[3]-=.006;robot.lift_ref=.087;robot.wrist_ref[:]+=[.004,.006]
    lagged=gate.measure()
    checks['legal_servo_lag_is_separate_from_actual_robot_retention']=bool(abs(lagged['gate_minimum_object_lift_m']-measurement['gate_minimum_object_lift_m'])<1e-12 and np.all(abs(lagged['gate_robot_tracking_error'])<=np.array([.01,.015,.015])))
    guard_plan={**gate_plan,'stop_gates':dict(per_finger_pressure_N=5.02,sum_finger_pressure_N=18.02,
        penetration_m=.002,table_support_N=.05,minimum_object_lift_m=.07,maximum_warning_count=0)}
    test_row={**lagged,'y':np.zeros(16),'pressure':np.ones(5),'penetration':0.,'table_support_N':0.,
              'warning_count':0,'qpos':np.zeros(3),'qvel':np.zeros(3),'ctrl':np.zeros(3)}
    test_row['gate_relative_drift_m']=.006
    try:physical_guard(test_row,guard_plan);retention_rejected=False
    except ProbeStop as exc:retention_rejected=str(exc)=='object_relative_retention'
    checks['horizontal_object_loss_still_fails_relative_retention_gate']=retention_rejected
    test_row['gate_relative_drift_m']=0.;test_row['gate_robot_tracking_failed']=np.array([False,True,False])
    try:physical_guard(test_row,guard_plan);tracking_rejected=False
    except ProbeStop as exc:tracking_rejected=str(exc)=='robot_task_tracking_dwell'
    checks['stable_object_does_not_excuse_failed_robot_task_tracking']=tracking_rejected
    # Exercise raw-record persistence and unavailable-branch semantics with a
    # tiny fake clock/decoder. No MuJoCo model/data/step is constructed here.
    import tempfile
    class FakePlant:
        def __init__(self):
            self.data=SimpleNamespace(time=0.)
            self.model=SimpleNamespace(opt=SimpleNamespace(timestep=.0005))
            self.alpha=base[:5].copy();self.lift_ref=float(base[5]);self.wrist_ref=base[6:].copy()
            self.advance_calls=0;self._contacts=lambda patches=True: {}
        def execute_delta(self,delta):self.alpha+=np.asarray(delta)
        def set_motion(self,lift,wrist):self.lift_ref=float(lift);self.wrist_ref=np.asarray(wrist).copy()
        def advance(self,dt):
            self.advance_calls+=1
            for _ in range(round(dt/.0005)):
                self.data.time+=.0005;self._contacts(False)
    def fake_restore(plant,state):
        plant.data.time=0.;plant.alpha=base[:5].copy();plant.lift_ref=float(base[5]);plant.wrist_ref=base[6:].copy()
    def fake_packet(plant,*unused):
        return dict(time=float(plant.data.time),input_reference=references(plant),
            pressure=np.full(5,1.),shear_magnitude=np.full(5,.3),slip_magnitude=np.zeros(5),
            y=np.zeros(16),y_palm=np.zeros(16),penetration=0.,warning_count=0,
            non_target_contact_pressure_N=0.,qpos=references(plant),qvel=np.zeros(8),ctrl=references(plant))
    previous=(globals()['restore_state'],globals()['packet'],globals()['physical_guard'])
    try:
        globals()['restore_state']=fake_restore;globals()['packet']=fake_packet;globals()['physical_guard']=lambda row,p:None
        with tempfile.TemporaryDirectory(prefix='phase3-pure-contract-') as temporary:
            folder=Path(temporary);plant=FakePlant();short=np.tile(base,(3,1));short[1:,0]+=.004
            first=execute_path(plant,None,np.eye(3),np.zeros(3),short,lo,hi,plan,folder/'valid')
            checks['fake_decoder_records_all_80_physical_substeps']=bool(first[2]['complete'] and len(first[0])==9 and len(first[1])==80)
            second=execute_path(plant,None,np.eye(3),np.zeros(3),short,lo,hi,plan,folder/'replay')
            checks['restored_fake_zero_records_are_compared_completely']=bool(replay_equal(first,second))
            impossible=short.copy();impossible[1:,5]=.101;calls=plant.advance_calls
            unavailable=execute_path(plant,None,np.eye(3),np.zeros(3),impossible,lo,hi,plan,folder/'unavailable')
            checks['unavailable_direction_never_steps_or_writes_fake_trace']=bool(not unavailable[2]['attempted_physics'] and plant.advance_calls==calls and not (folder/'unavailable/trace.npz').exists())
            def stop_on_raw_substep(row,p):
                if row['time']>.0009:raise ProbeStop('synthetic_substep_gate')
            globals()['physical_guard']=stop_on_raw_substep
            stopped=execute_path(plant,None,np.eye(3),np.zeros(3),short,lo,hi,plan,folder/'stopped')
            checks['physical_stop_preserves_triggering_substep_and_partial_trace']=bool(not stopped[2]['complete'] and len(stopped[1])==2 and (folder/'stopped/trace.npz').exists())
    finally:globals()['restore_state'],globals()['packet'],globals()['physical_guard']=previous
    if not all(checks.values()):raise AssertionError(checks)
    return dict(passed=True,checks=checks,synthetic_only=True,physics_started=False)


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test',action='store_true');group.add_argument('--prepare-plan',action='store_true');group.add_argument('--run',action='store_true')
    parser.add_argument('--plan',type=Path,default=PLAN_PATH)
    parser.add_argument('--out',type=Path,default=CALIBRATION/'phase3_path_validation_v3')
    args=parser.parse_args()
    if args.self_test:print(json.dumps(serial(self_test()),ensure_ascii=False));return
    plan_path=args.plan.resolve()
    if not plan_path.is_relative_to(CALIBRATION.resolve()):raise ValueError('Plan must remain inside new calibration directory')
    if args.prepare_plan:
        if plan_path.exists():raise FileExistsError('Refusing to overwrite an existing plan')
        write_json(plan_path,make_plan());print(str(plan_path));return
    plan=json.loads(plan_path.read_text(encoding='utf-8'));result=run(plan,args.out.resolve(),plan_path)
    print(json.dumps(serial(result),ensure_ascii=False))
    if not result['execution_complete']:raise SystemExit(1)


if __name__=='__main__':main()

"""Fixed actual-reference replay. Only root may call run_job: it integrates physics.

No new feedback calculation occurs, including preparation. The complete 0.5ms
bridge, not an assumption about a saved qpos row, is the reproduction test.
"""
from pathlib import Path
from dataclasses import asdict
import gzip
import json
import time
import traceback
import numpy as np
from .source_record import sha, canonical, exact, require_unchanged
from .reconstruction import imports, config_for, robot_ids, release_fields

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)

def plant_class():
    _, _, Plant = imports()
    class FixedInputPlant(Plant):
        def __init__(self, config, packets):
            self.replay_ctrl = None
            self.replay_inside_advance = False
            self.replay_last_micro_time = None
            self.replay_micro_states = []
            self.replay_packets = packets
            super().__init__(config)

        def _apply_ctrl(self):
            if self.replay_ctrl is None:
                return super()._apply_ctrl()
            self.data.ctrl[:] = self.replay_ctrl
            # Saved row.ctrl precedes current set_environment. The original
            # advance _apply_ctrl sets this one nonrobot port from current env.
            self.data.ctrl[self.obstacle_motor] = float(np.clip(self.env.get('obstacle_x', 0.), -.4, .1))

        def _contacts(self, patches=True):
            value = super()._contacts(patches)
            end = float(self.data.time)
            if self.replay_inside_advance and end != self.replay_last_micro_time:
                self.replay_last_micro_time = end
                self.replay_micro_states.append((end, self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(),
                                                int(sum(int(x.number) for x in self.data.warning))))
                if 9.2 - 1e-9 <= end <= 10.4 + 1e-9 or 12. - 1e-9 <= end <= 16. + 1e-9:
                    # These reads do NOT call forward/kinematics/step. Same
                    # cached integration solve, before the boundary mj_forward.
                    complete = super()._contacts(True)
                    from session import jsonable
                    self.replay_packets.write(json.dumps(jsonable(dict(end_time_s=end, start_time_s=end-self.model.opt.timestep,
                        phase=self.phase, patches=complete['patches'])), ensure_ascii=False) + '\n')
            return value

        def advance(self, step=None):
            self.replay_inside_advance = True
            try:
                return super().advance(step)
            finally:
                self.replay_inside_advance = False
    return FixedInputPlant

def arrays_from(rows, plant):
    values = {key: np.asarray([r[key] for r in rows]) for key in rows[0]} if rows else {}
    micro = np.asarray(plant.substep_records, float)
    if len(micro):
        for key, columns in dict(time=0, duration=1, normal=slice(2, 7), skin_normal=slice(7, 12), palm=12,
                                 hand_other=13, all_hand_object=14, support=15, penetration=16, world_penetration=17).items():
            values['substep_' + key] = micro[:, columns]
        states = plant.replay_micro_states
        values.update(substep_state_end_time=np.asarray([s[0] for s in states]),
                      substep_qpos=np.asarray([s[1] for s in states]), substep_qvel=np.asarray([s[2] for s in states]),
                      substep_actual_ctrl=np.asarray([s[3] for s in states]), substep_warning_count=np.asarray([s[4] for s in states]))
    return values

def bridge_comparison(values, record):
    result = {}
    for key, actual in values.items():
        if key not in record['arrays']:
            continue
        original = record['arrays'][key]
        matched = exact(actual, original)
        detail = dict(exact=matched, expected_shape=list(original.shape), actual_shape=list(actual.shape))
        if not matched and actual.shape == original.shape and actual.dtype.kind in 'fiu':
            detail['max_abs_difference'] = float(np.max(abs(actual - original)))
        result[key] = detail
    mandatory = ('qpos', 'qvel', 'ctrl_before', 'ctrl', 'alpha_before', 'alpha', 'normal', 'skin_normal',
                 'palm', 'hand_other', 'all_hand_object', 'support', 'position', 'relative',
                 'relative_velocity', 'angular_velocity', 'penetration', 'world_penetration', 'lift',
                 'wrist_position', 'warning_count', 'legacy_common_sensor_json', 'robot_actuator_target21')
    mandatory += tuple(key for key in record['arrays'] if key.startswith('substep_'))
    passed = all(key in result and result[key]['exact'] for key in mandatory)
    return dict(passed=passed, required_fields=list(mandatory), fields=result,
                scope='Complete saved physics and common instrument packets, not just peak equivalence. Original full initial integration state absent; any mismatch blocks refinement.')

def peak_diagnostics(values, metadata):
    t = values['substep_time']; duration = values['substep_duration']; force = values['substep_skin_normal']
    limits = np.asarray(metadata['limits']); end = t + duration; result = {}
    for phase, lo, hi in (('preparation', 0., 7.5), ('holding', 7.5, 12.), ('collision', 9.2, 10.4), ('release', 12., 16.)):
        selected = np.flatnonzero((t >= lo-1e-9) & (t < hi-1e-9))
        row, digit = np.unravel_index(np.argmax(force[selected]), force[selected].shape); k = selected[row]
        nontarget = np.maximum(force[selected] - values['substep_normal'][selected], 0).sum(axis=1)
        contact = np.flatnonzero(nontarget > 1e-10)
        result[phase] = dict(peak_N=float(force[k, digit]), digit=int(digit), peak_start_s=float(t[k]), peak_end_s=float(end[k]),
            target_same_digit_N=float(values['substep_normal'][k, digit]), nontarget_same_digit_N=float(max(0.,force[k,digit]-values['substep_normal'][k,digit])),
            target_palm_normal_sum_N=float(values['substep_palm'][k]), table_normal_sum_N=float(values['substep_support'][k]),
            local_excess_Ns=float(np.sum(np.maximum(force[selected]-limits,0)*duration[selected,None])),
            budget_excess_Ns=float(np.maximum(force[selected].sum(axis=1)-metadata['budget'],0)@duration[selected]),
            first_nontarget_finger_contact_end_s=None if not len(contact) else float(end[selected[contact[0]]]),
            maximum_hand_object_penetration_m=float(values['substep_penetration'][selected].max()),
            maximum_world_penetration_m=float(values['substep_world_penetration'][selected].max()),
            maximum_warning_count=int(values['substep_warning_count'][selected].max()),
            actual_ctrl_at_peak=values['substep_actual_ctrl'][k].tolist(),
            numerical_qpos_at_peak_evaluation_only=values['substep_qpos'][k].tolist())
    return result

def run_job(record, dt, target):
    """INTEGRATES. Deliberately not called by any author verification entrypoint."""
    target = Path(target); target.mkdir(parents=True, exist_ok=False)
    rows = []; plant = None; began = time.perf_counter(); outcome = None
    packet = None
    try:
        mj, _, _ = imports()
        from sensors import HandSensors, SensorNoise
        from skin_geometry import DelayedSkinGeometry
        from session import jsonable
        a = record['arrays']; meta = record['metadata']
        packet = gzip.open(target / 'cached_raw_contact_packets.jsonl.gz', 'xt', encoding='utf-8')
        plant = plant_class()(config_for(record, dt), packet)
        ids = robot_ids(plant); obstacle = int(plant.model.geom('integration_obstacle_geom').id)
        sensors = HandSensors(meta['calibration'], SensorNoise(**meta['noise']), seed=meta['seed'])
        skin = DelayedSkinGeometry(plant)
        spec = mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mj.mj_stateSize(plant.model, spec)); mj.mj_getState(plant.model, plant.data, state, spec)
        np.savez_compressed(target / 'reset_integration_state.npz', state=state, state_spec=int(spec), qpos=plant.data.qpos.copy(), qvel=plant.data.qvel.copy(), ctrl=plant.data.ctrl.copy())
        if not exact(plant.data.qpos, a['qpos'][0]) or not exact(plant.data.qvel, a['qvel'][0]) or not exact(plant.data.ctrl, a['ctrl_before'][0]):
            raise ValueError('Reconstructed reset does not reproduce saved initial state; no future state restoration allowed')
        release = release_fields(plant, record)
        for k, t in enumerate(a['time']):
            if plant.time != float(t):
                raise ValueError('Public control clock changed')
            truth = plant.truth(); raw = plant.raw_sensors()
            before_rng = canonical(sensors.rng.bit_generator.state)
            obs = sensors.sample(raw); skin.acquire(raw, obs)
            common = json.dumps(jsonable(vars(obs)), ensure_ascii=False)
            after_rng = canonical(sensors.rng.bit_generator.state)
            row = dict(time=float(t), **{key: truth[key] for key in ('normal', 'skin_normal', 'palm', 'hand_other', 'all_hand_object',
                'support', 'position', 'relative', 'rotation', 'relative_velocity', 'angular_velocity', 'lift', 'qpos', 'qvel',
                'penetration', 'world_penetration', 'warning_count')})
            row.update(ctrl_before=truth['ctrl'], alpha_before=truth['alpha'], robot_actuator_target21=truth['ctrl'][ids].copy(),
                       wrist_position=truth['qpos'][[plant.model.joint(n).qposadr[0] for n in ('rh_WRJ2', 'rh_WRJ1')]],
                       legacy_common_sensor_json=common, rng_state_before_sample=before_rng, rng_state_after_sample=after_rng,
                       collision=float(sum(p['normal'] for p in raw['patches'] if int(p['external_geom']) == obstacle)),
                       lift_ref=float(a['lift_ref'][k]), transport_paused=bool(a['transport_paused'][k]),
                       source_input_tick=k, source_environment_json=str(a['environment_json'][k]))
            # No controller/observer feedback or preparation action is recomputed.
            # References only; actual q/velocity never overwritten from source.
            if k == record['release_index']:
                for key, value in release.items():
                    setattr(plant, key, value.copy() if isinstance(value, np.ndarray) else value)
            plant.alpha = a['alpha'][k].copy()
            plant.lift_ref = float(a['executed_motion_reference'][k, 0]); plant.wrist_ref = a['executed_motion_reference'][k, 1:].copy()
            plant.phase = str(a['phase'][k]); plant.replay_ctrl = a['ctrl'][k].copy(); plant.data.ctrl[:] = plant.replay_ctrl
            row.update(ctrl=plant.data.ctrl.copy(), alpha=plant.alpha.copy(), phase=plant.phase,
                       executed_motion_reference=np.r_[plant.lift_ref, plant.wrist_ref])
            rows.append({key: value.copy() if isinstance(value, np.ndarray) else value for key, value in row.items()})
            if k < 3200:
                plant.set_environment(record['environments'][k]); plant.advance(.005)
        require_unchanged(record)
        values = arrays_from(rows, plant)
        replay_meta = dict(meta)
        replay_meta.update(schema='contact_peak_fixed_actual_input_replay_v1', role='numerical_diagnostic_not_policy_performance',
                           original_method=record['method'], original_trace=str(record['path']), original_trace_sha256=record['trace_sha256'],
                           config=asdict(plant.config), physics_timestep=dt, model_sha256=plant.model_sha256,
                           completed=True, new_feedback_decisions=0, actual_q_restored_after_reset=False,
                           unchanged_source_handoff_fields_are_historical_labels_not_new_admission=True,
                           actual_command_scope='Source-proven interval22 engine ports, obstacle refreshed from same saved environment',
                           saved_lift_ref_is_user_target_not_actual_command=True)
        values['metadata'] = json.dumps(replay_meta, ensure_ascii=False)
        np.savez_compressed(target / 'trace.npz', **values)
        write(target / 'metadata.json', {**replay_meta, 'trace_sha256': sha(target / 'trace.npz')})
        comparison = bridge_comparison(values, record) if dt == .0005 else None
        if comparison is not None:
            write(target / 'original_complete_bridge_comparison.json', comparison)
        from evaluate_runs import evaluate_trace
        # Consume the unchanged raw-force evaluator only. Task-success readout
        # would reuse historical handoff metadata and is deliberately not exported.
        force = evaluate_trace(target / 'trace.npz')['safety']
        write(target / 'original_force_guards.json', force)
        diagnostic = peak_diagnostics(values, meta)
        write(target / 'peak_diagnostics.json', diagnostic)
        outcome = dict(method=record['method'], dt=dt, complete=True, source_trace_sha256=record['trace_sha256'],
                       trace_sha256=sha(target / 'trace.npz'), source_inputs_unchanged=True,
                       original_bridge_passed=None if comparison is None else comparison['passed'],
                       force_guards=force, peak_diagnostics=diagnostic, wall_seconds=time.perf_counter()-began,
                       physics_steps=len(plant.substep_records), noise_rng_final=canonical(sensors.rng.bit_generator.state))
    except Exception as exc:
        outcome = dict(method=record['method'], dt=dt, complete=False, error=repr(exc), traceback=traceback.format_exc(),
                       actual_saved_rows=len(rows), wall_seconds=time.perf_counter()-began)
        if plant is not None:
            try:
                np.savez_compressed(target / 'partial_trace.npz', **arrays_from(rows, plant))
                outcome['partial_trace'] = 'partial_trace.npz'
            except Exception as save_exc:
                outcome['partial_save_error'] = repr(save_exc)
    finally:
        if packet is not None:
            packet.close()
        if outcome is not None:
            write(target / 'result.json', outcome)
    return outcome

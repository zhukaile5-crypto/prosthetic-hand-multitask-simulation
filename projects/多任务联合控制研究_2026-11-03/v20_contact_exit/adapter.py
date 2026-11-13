"""Public V20 robot/old-contact reference tasks; no sensor or plant access.

Parent observation history, signed support and receiver evidence are reused.
Only the geometric task builder changes. The optional motion signal is one
existing two-dimensional instrument projection per digit, not one per patch.
"""
from copy import deepcopy
import numpy as np
import mujoco

from v13_a1.codec import Reference21, REFERENCE_AXES21
from v10.robot_contact_space import RobotJointSample, PublicSkinPatch
from v16_release_geometry.public_adapter import _array, _mask, _real, _tick
from v17_contact_transfer.motion_adapter import PublicMotionAdapter
from v17_contact_transfer.motion_domain_sensors import SLIP_DOMAIN_ROLE
from .geometry import finite_array, position_reference_map, contact_geometry_rows


class ContactExitAdapter(PublicMotionAdapter):
    """Same public observe/begin_release/transfer_inputs as the frozen parent.

    current_contact_tasks takes absolute CURRENT legal reference bounds. It
    neither enlarges nor clips those permissions, creates a goal, nor writes.
    """
    version = 'V20_POSITION_REFERENCE_CONTACT_EXIT_V1'

    def _unknown(self, now, reason):
        diagnostics = dict(status='UNKNOWN_REFERENCE_GEOMETRY', reason=str(reason), current_time_s=now,
                           force_prediction_available=False, future_robot_motion_prediction_available=False,
                           placement_certified=False, motion_available=False)
        self.last_geometry_diagnostics = diagnostics
        return dict(available=False, time_s=now, Npx21=None, y_p=None, digits_p=None,
                    normal_jacobian_Px25=None, pressure_P_N=None, vnormal_p=None,
                    current_unbiased_debt21=None, a_normal_p=None, B25x21=None,
                    proposal_map25x21=None, encoder_reference21=None, robot_velocity25=None,
                    native_state=None, motion_Mx21=None, motion_e_M=None,
                    motion_matrix_Mx21=None, motion_error_M_m_s=None, motion_time_s=None,
                    diagnostics=diagnostics)

    def _snapshot(self, frame, current, lower, upper):
        _, now = _tick(frame.time)
        if self._last_time is None or abs(self._last_time - now) > 1e-9:
            raise ValueError('observe current public frame before task construction')
        if self._signature() != self._binding:
            raise ValueError('bound known robot definition changed')
        if abs(_real(frame.robot_control_time_s, 'robot command clock', nonnegative=True) - now) > 1e-9:
            raise ValueError('robot command clock mismatch')
        if _array(frame.robot_reference21, (21,), 'authoritative public reference').tobytes() != current.tobytes():
            raise ValueError('current21 is not the authoritative public reference')
        engine = _array(frame.robot_actuator_target21, (21,), 'actual public robot engine21')
        if self.actor.encode(Reference21(current, now)).values.tobytes() != engine.tobytes():
            raise ValueError('public reference/engine bytes mismatch')
        for value in (lower, upper):
            port = self.actor.encode(Reference21(value, now)).values
            limited = self.actor.ctrl_limited21
            if np.any(port[limited] < self.actor.ctrl_range21[limited, 0]) or np.any(port[limited] > self.actor.ctrl_range21[limited, 1]):
                raise ValueError('declared absolute reference domain exceeds native engine control domain')
        q = _array(frame.robot_position25, (25,), 'current public robot q25')
        v = _array(frame.robot_velocity25, (25,), 'current public robot v25')
        state = self.servo.evaluate(q, v, engine, time_s=now)
        if state.status != 'KNOWN_ROBOT_ACTUATION_ALGEBRA':
            raise ValueError('native robot algebra unavailable: ' + str(state.reasons))
        if any(state.diagnostics['control_clipped21']):
            raise ValueError('current engine control clipping invalidates current target-reference geometry')
        m = self.servo.model; aids = self.servo.aids
        B, info = position_reference_map(transmission21x25=state.transmission21x25,
            joint_scale25=self.joint_scale25, gain21x3=m.actuator_gainprm[aids, :3],
            bias21x3=m.actuator_biasprm[aids, :3],
            fixed_gain21=m.actuator_gaintype[aids] == mujoco.mjtGain.mjGAIN_FIXED,
            affine_bias21=m.actuator_biastype[aids] == mujoco.mjtBias.mjBIAS_AFFINE)
        sample = RobotJointSample(now, self.geometry.raw_joint_names, q[:24], q[24], v[:24], v[24])
        return state, B, info, sample, v

    def _motion_rows(self, frame, *, B, sample, accepted):
        """Acquired e, current-J proposal; no future or current external state."""
        now = float(frame.time); skipped = []; blocks = []; values = []; details = []
        try:
            acq = _real(frame.sample_time, 'motion acquisition', nonnegative=True)
            if getattr(frame, 'slip_domain_role', None) != SLIP_DOMAIN_ROLE or getattr(frame, 'slip_domain_available', None) is not True:
                raise ValueError('existing instrument projection domain unavailable')
            if abs(_real(frame.slip_domain_acquisition_time_s, 'projection acquisition', nonnegative=True) - acq) > 1e-9 or abs(_real(frame.slip_domain_time_s, 'projection read time', nonnegative=True) - now) > 1e-9:
                raise ValueError('projection domain clock mismatch')
            if not 0 <= now - acq <= .12 + 1e-9 or abs(_real(frame.age, 'public age', nonnegative=True) - (now - acq)) > 1e-9:
                raise ValueError('projection age unavailable/outside parent domain')
            old = self._motion_samples.get(int(round(acq / .005)))
            if old is None or old['reason'] is not None or abs(old['time_s'] - acq) > 1e-9:
                raise ValueError('same-acquisition public robot history unavailable')
            valid = _mask(frame.slip_valid5, 'projection validity') & _mask(frame.valid, 'hand validity')
            basis = np.asarray(frame.tangent_basis, dtype=object); slip = np.asarray(frame.slip_local, dtype=object)
            if basis.shape != (5, 2, 3) or slip.shape != (5, 2):
                raise ValueError('five device projection pairs required')
            acquired_sample = RobotJointSample(acq, self.geometry.raw_joint_names, old['q'][:24], old['q'][24], old['v'][:24], old['v'][24])
            for digit in range(5):
                group = [p for p in accepted if p['digit'] == digit]
                if not valid[digit] or not group:
                    skipped.append(dict(digit=digit, reason='no_valid_same_digit_projection_and_geometry'))
                    continue
                try:
                    if any(abs(p['acquisition_time_s'] - acq) > 1e-9 for p in group):
                        raise ValueError('patch/projection acquisition mismatch')
                    T = finite_array(basis[digit], (2, 3), 'registered device basis')
                    s = finite_array(slip[digit], (2,), 'observed relative slip')
                    if not np.allclose(T @ T.T, np.eye(2), atol=1e-8, rtol=0):
                        raise ValueError('registered orthonormal device projection required')
                    weights = np.asarray([p['normal_N'] for p in group]); weights = weights / weights.max(); weights = weights / weights.sum()
                    acquired_v = []
                    for p in group:
                        k = self.geometry.point_kinematics(acquired_sample, p['typed_patch'])
                        acquired_v.append(k.point_velocity_world)
                    meanv = weights @ np.asarray(acquired_v)
                    meanJ = np.einsum('p,pij->ij', weights, np.asarray([p['jacobian3x25'] for p in group]))
                    world_to_plane = T @ old['R'].T
                    measured_external = world_to_plane @ meanv + s
                    block = world_to_plane @ meanJ @ B
                    blocks.append(block); values.append(measured_external)
                    details.append(dict(digit=digit, patch_indices=[p['patch_index'] for p in group],
                        pressure_shares=weights.tolist(), acquisition_time_s=acq,
                        acquired_robot_projection_m_s=(world_to_plane @ meanv).tolist(),
                        observed_relative_slip_m_s=s.tolist(), external_absolute_projection_m_s=measured_external.tolist()))
                except (ValueError, TypeError, AttributeError, KeyError) as exc:
                    skipped.append(dict(digit=digit, reason=str(exc)))
            matrix = np.vstack(blocks) if blocks else np.empty((0, 21))
            error = np.concatenate(values) if values else np.empty(0)
            meta = dict(available=bool(blocks), status='ACQUIRED_PROJECTION_CURRENT_J_REFERENCE_INTENT' if blocks else 'UNKNOWN_NO_VALID_MOTION_ROWS',
                acquisition_time_s=acq, read_time_s=now, age_s=now-acq, rows_per_digit=2,
                digit_details=details, skipped=skipped, shared_finger_noise_not_independent_per_patch=True,
                unknown_actor_mixture=True, external_rigid_body_twist_available=False,
                matrix_units='metres per reference coordinate', error_units='metres per second',
                future_external_velocity_prediction=False, actual_dissipation_certified=False)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            matrix = np.empty((0, 21)); error = np.empty(0)
            meta = dict(available=False, status='UNKNOWN_PUBLIC_MOTION', reason=str(exc), skipped=skipped)
        self.last_motion_diagnostics = deepcopy(meta)
        return finite_array(matrix, matrix.shape, 'motion reference rows'), finite_array(error, error.shape, 'acquired external projection'), meta

    def current_contact_tasks(self, frame, *, current21, reference_lower21, reference_upper21):
        _, now = _tick(frame.time)
        current = finite_array(current21, (21,), 'current unbiased reference21')
        lower = finite_array(reference_lower21, (21,), 'actual absolute reference lower21')
        upper = finite_array(reference_upper21, (21,), 'actual absolute reference upper21')
        if np.any(lower > upper):
            raise ValueError('empty caller actual/TASK reference domain')
        try:
            state, B, map_info, sample, v = self._snapshot(frame, current, lower, upper)
            debt = finite_array(current - state.actuator_length21, (21,), 'unbiased geometric reference debt')
            valid = _mask(frame.valid, 'finger valid') & _mask(frame.distributed_valid, 'distributed geometry valid')
            present = _mask(frame.contact_present, 'public contact presence')
            finger_normal = np.asarray(frame.normal, dtype=object)
            if finger_normal.shape != (5,):
                raise ValueError('five public finger normal observations required')
            positive_present = []
            for digit in range(5):
                if valid[digit]:
                    normal = _real(finger_normal[digit], 'valid public finger normal', nonnegative=True)
                    if present[digit] and normal > 0:
                        positive_present.append(digit)
            accepted, skipped = [], []
            raw_patches = frame.skin_patches
            if not isinstance(raw_patches, (list, tuple)):
                raise ValueError('explicit public skin patch sequence required')
            for index, raw in enumerate(raw_patches):
                try:
                    patch = PublicSkinPatch.from_public(raw)
                    digit = patch.digit
                    if digit is None or not valid[digit]:
                        raise ValueError('finger geometry invalid or auxiliary geometry without registered digit')
                    pressure = _real(raw.get('normal'), 'measured positive pressure', nonnegative=True)
                    if pressure <= 0:
                        raise ValueError('nonpositive patch not an active measured normal task')
                    if now - patch.sample_time_s < -1e-9 or now - patch.sample_time_s > .12 + 1e-9:
                        raise ValueError('patch acquisition outside parent geometry age domain')
                    if abs(patch.sample_time_s - _real(frame.sample_time, 'public hand acquisition', nonnegative=True)) > 1e-9:
                        raise ValueError('patch/hand acquisition mismatch')
                    if patch.reference_normal is None or patch.reference_position is None:
                        raise ValueError('registered acquisition plane missing')
                    current_k = self.geometry.point_kinematics(sample, patch)
                    accepted.append(dict(patch_index=index, typed_patch=patch, digit=digit,
                        normal_N=pressure, acquisition_time_s=patch.sample_time_s, robot_time_s=now,
                        old_normal_world3=patch.reference_normal, jacobian3x25=current_k.jacobian3x25))
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    skipped.append(dict(patch_index=index, reason=str(exc)))
            accepted_digits = {p['digit'] for p in accepted}
            missing_geometry_digits = [d for d in positive_present if d not in accepted_digits]
            if not accepted and (skipped or not valid.all() or present.any()):
                raise ValueError('no usable normal rows with incomplete geometry is UNKNOWN, not known zero contact')
            rows = contact_geometry_rows(proposal_map25x21=B, current_unbiased_debt21=debt,
                                        robot_velocity25=v, patches=accepted)
            M, e, motion = self._motion_rows(frame, B=B, sample=sample, accepted=accepted)
            native = dict(status=state.status, time_s=state.time_s,
                effective_target21=state.effective_target21.tolist(), actuator_length21=state.actuator_length21.tolist(),
                actuator_velocity21=state.actuator_velocity21.tolist(), unbounded_effort21=state.unbounded_effort21.tolist(),
                predicted_effort21=state.predicted_effort21.tolist(), diagnostics=deepcopy(state.diagnostics),
                effort_is_model_computed_not_measured=True)
            diagnostics = dict(status='EXPERIMENTAL_POSITION_REFERENCE_GEOMETRY', current_time_s=now,
                map=map_info, patches=rows.pop('patches'), skipped_patches=skipped,
                valid_finger_geometry5=valid.tolist(), contact_present5=present.tolist(),
                missing_geometry_digits=missing_geometry_digits,
                partial_geometry=bool(skipped or not valid.all() or missing_geometry_digits),
                normal_task_available=bool(accepted), motion_available=motion['available'], motion=motion,
                normal_geometry_scope='observed finger patches only; auxiliary contact geometry remains unknown',
                debt_scope='unbiased current reference minus robot transmission length; actor lift gravity bias excluded',
                normal_sign='hand-to-external old acquisition normal; positive robot/reference projection is inward',
                normal_velocity_scope='CURRENT robot velocity in OLD acquired normal, not relative/external separation velocity',
                acquired_external_plane_not_updated_from_object_truth=True, real_domain_not_widened=True,
                force_prediction_available=False, future_robot_motion_prediction_available=False,
                placement_certified=False, reference_axes21=list(REFERENCE_AXES21), robot_axes25=list(self.axes25))
            self.last_geometry_diagnostics = deepcopy(diagnostics)
            return dict(available=True, time_s=now, **rows, current_unbiased_debt21=debt,
                B25x21=B, proposal_map25x21=B, encoder_reference21=state.actuator_length21,
                robot_velocity25=v, native_state=native, motion_Mx21=M, motion_e_M=e,
                motion_matrix_Mx21=M if motion['available'] else None,
                motion_error_M_m_s=e if motion['available'] else None,
                motion_time_s=now if motion['available'] else None,
                diagnostics=diagnostics)
        except (ValueError, TypeError, KeyError, AttributeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            return self._unknown(now, exc)


PublicExitAdapter = ContactExitAdapter

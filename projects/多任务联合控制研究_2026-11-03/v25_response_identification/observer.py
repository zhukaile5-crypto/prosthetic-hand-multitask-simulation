"""Immutable public motion observations and actual command receipts.

No device sampling, actor identity, native effort model, integration, or writer.
The two rows of one digit are ONE existing instrument pair, shared by all its
patches. C describes a conditional rigid velocity field; it proves no actor.
"""
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from numbers import Real, Integral
from types import MappingProxyType
import json
import numpy as np

from v10.robot_contact_space import PublicSkinPatch, RobotJointSample
from v13_a1.codec import Reference21, EnginePort21, REFERENCE_AXES21
from v11.robot_servo_effort import ACTUATORS21
from v17_contact_transfer.motion_domain_sensors import SLIP_DOMAIN_ROLE

PERIOD_S = .005
CLOCK_TOL_S = 1e-9
MAX_GEOMETRY_AGE_S = .12  # Same domain as the frozen public motion adapter.
SCHEMA = 'V25_PUBLIC_MOTION_PACKET_V1'


def _real(value, name, *, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(name + ' requires a finite real number')
    value = float(value)
    if nonnegative and value < 0:
        raise ValueError(name + ' must be nonnegative')
    return value


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(name + ' requires a nonnegative integer')
    return int(value)


def _array(value, shape, name):
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in raw.flat):
        raise ValueError(name + ' requires real numeric ' + str(shape))
    a = np.asarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(name + ' must be finite')
    # Bytes backing prevents a caller from re-enabling WRITEABLE on the view.
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(shape)


def _mask(value, name):
    a = np.asarray(value)
    if a.shape != (5,) or a.dtype != np.dtype(bool):
        raise ValueError(name + ' requires explicit bool5')
    return tuple(bool(x) for x in a)


def _tick(value, name):
    t = _real(value, name, nonnegative=True)
    tick = int(round(t / PERIOD_S))
    if abs(t - tick * PERIOD_S) > CLOCK_TOL_S:
        raise ValueError(name + ' is not on the existing 5ms public clock')
    return tick, t


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return _array(value, value.shape, 'immutable array')
    if isinstance(value, np.generic):
        return value.item()
    return value


def _plain(value):
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _identity_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' requires a nonempty string')
    return value


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


class _ImmutableOutput:
    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, _freeze(getattr(self, f.name)))


@dataclass(frozen=True)
class RobotAcquisitionSnapshot25:
    time_s: float
    position25: np.ndarray
    velocity25: np.ndarray
    rotation_world_from_palm3x3: np.ndarray
    axes25: tuple

    def __post_init__(self):
        _, t = _tick(self.time_s, 'robot acquisition time')
        object.__setattr__(self, 'time_s', t)
        object.__setattr__(self, 'position25', _array(self.position25, (25,), 'robot q25'))
        object.__setattr__(self, 'velocity25', _array(self.velocity25, (25,), 'robot v25'))
        R = _array(self.rotation_world_from_palm3x3, (3, 3), 'acquisition palm rotation')
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-8, rtol=0) or abs(np.linalg.det(R)-1) > 1e-8:
            raise ValueError('Acquisition rotation must be proper orthonormal')
        object.__setattr__(self, 'rotation_world_from_palm3x3', R)
        axes = tuple(self.axes25)
        if len(axes) != 25 or len(set(axes)) != 25 or any(not isinstance(s, str) or not s for s in axes):
            raise ValueError('Explicit unique robot axes25 required')
        object.__setattr__(self, 'axes25', axes)


@dataclass(frozen=True)
class DigitMotionRecord(_ImmutableOutput):
    digit: int
    status: str
    reasons: tuple
    sensor_valid: object
    distributed_valid: object
    contact_present: object
    instrument_valid: object
    acquisition_time_s: float
    read_time_s: float
    age_s: float
    observed_normal_N: object
    raw_relative_slip2_m_s: object
    basis_palm2x3: object
    projection_world2x3: object
    point_world3: object
    robot_jacobian_acquisition3x25: object
    robot_velocity_world3_m_s: object
    robot_projection2_m_s: object
    external_projection2_m_s: object
    conditional_C2x6: object
    patch_support: tuple
    pressure_shares: tuple
    noise_group: tuple
    diagnostics: object

    def to_dict(self):
        return _plain(self)


@dataclass(frozen=True)
class PublicMotionPacket(_ImmutableOutput):
    schema: str
    stream_id: str
    epoch_id: str
    read_tick: int
    acquisition_tick: int
    read_time_s: float
    acquisition_time_s: float
    available_at_s: float
    age_s: float
    status: str
    available: bool
    known_empty: bool
    anchor_world3: np.ndarray
    robot_snapshot: object
    digits: tuple
    row_keys: tuple
    external_projection_m_s: np.ndarray
    conditional_C_Mx6: np.ndarray
    diagnostics: object

    def to_dict(self):
        return _plain(self)


def build(frame, *, robot_snapshot, point_kinematics, stream_id, epoch_id,
          anchor_world3=(0., 0., 0.)):
    """Pure builder. callback(snapshot, PublicSkinPatch) returns PointKinematics.

    The callback must use ONLY known robot geometry and this public q/v sample.
    No current reference / B / control clipping / native admission is consumed.
    Unknown rows are absent, never padded with measured zero motion.
    """
    stream_id = _identity_text(stream_id, 'stream_id')
    epoch_id = _identity_text(epoch_id, 'epoch_id')
    read_tick, now = _tick(frame.time, 'read time')
    acq_tick, acq = _tick(frame.sample_time, 'hand acquisition time')
    anchor = _array(anchor_world3, (3,), 'fixed public world anchor')
    age = now - acq
    if age < -CLOCK_TOL_S:
        raise ValueError('Future hand acquisition cannot be public history')
    global_reasons = []
    try:
        declared_age = _real(frame.age, 'public hand age', nonnegative=True)
        if abs(declared_age-age) > CLOCK_TOL_S or age > MAX_GEOMETRY_AGE_S+CLOCK_TOL_S:
            raise ValueError('hand_age_mismatch_or_outside_parent_domain')
        if getattr(frame, 'slip_domain_role', None) != SLIP_DOMAIN_ROLE or getattr(frame, 'slip_domain_available', None) is not True:
            raise ValueError('instrument_acquisition_domain_unknown')
        if abs(_real(frame.slip_domain_acquisition_time_s, 'instrument acquisition', nonnegative=True)-acq) > CLOCK_TOL_S or abs(_real(frame.slip_domain_time_s, 'instrument read time', nonnegative=True)-now) > CLOCK_TOL_S:
            raise ValueError('instrument_domain_clock_mismatch')
    except (ValueError, AttributeError, TypeError) as exc:
        global_reasons.append(str(exc))
    masks = {}
    for key in ('valid', 'distributed_valid', 'contact_present', 'slip_valid5'):
        try:
            masks[key] = _mask(getattr(frame, key), key)
        except (ValueError, AttributeError, TypeError):
            masks[key] = (None,) * 5
    if robot_snapshot is not None:
        if type(robot_snapshot) is not RobotAcquisitionSnapshot25:
            raise TypeError('Typed robot acquisition snapshot required')
        if abs(robot_snapshot.time_s-acq) > CLOCK_TOL_S:
            global_reasons.append('robot_history_not_same_acquisition')
    else:
        global_reasons.append('same_acquisition_robot_history_unknown')
    if not callable(point_kinematics):
        global_reasons.append('robot_only_geometry_callback_unknown')

    grouped = [[] for _ in range(5)]
    group_errors = [[] for _ in range(5)]
    raw_patches = getattr(frame, 'skin_patches', None)
    if not isinstance(raw_patches, (tuple, list)):
        global_reasons.append('public_skin_patch_sequence_unknown')
        raw_patches = []
    for index, raw in enumerate(raw_patches):
        if not isinstance(raw, dict):
            global_reasons.append('unassigned_malformed_public_patch')
            continue
        # Strict public geometry boundary: unexpected truth/actor fields reject.
        allowed = {'robot_body','link_point','link_normal','sample_time',
                   'reference_position','reference_normal','digit','normal','age'}
        if set(raw)-allowed:
            raise ValueError('Unexpected public patch fields; external identity/truth is forbidden')
        try:
            d = _integer(raw.get('digit'), 'patch digit')
            if d >= 5:
                raise ValueError('patch digit outside five finger instrument')
        except ValueError:
            global_reasons.append('unassigned_public_patch_digit')
            continue
        try:
            p = PublicSkinPatch.from_public(raw)
            n = _real(raw.get('normal'), 'patch normal', nonnegative=True)
            if abs(p.sample_time_s-acq) > CLOCK_TOL_S:
                raise ValueError('patch_acquisition_mismatch')
            if 'age' in raw and abs(_real(raw['age'], 'patch age', nonnegative=True)-age) > CLOCK_TOL_S:
                raise ValueError('patch_age_mismatch')
            if n > 0:
                grouped[d].append((index, p, n))
        except (ValueError, KeyError, TypeError) as exc:
            group_errors[d].append(str(exc))

    def component(name, digit, shape):
        raw = np.asarray(getattr(frame, name, None), dtype=object)
        if raw.shape != (5,) + shape:
            raise ValueError(name + '_shape_unknown')
        return _array(raw[digit], shape, name)

    records, values, Cs, keys = [], [], [], []
    unresolved = False
    known_empty_digits = []
    specs = getattr(frame, 'sensor_characteristics', None)
    noise = {}
    if isinstance(specs, Mapping):
        for key in ('slip_sigma_m_s', 'position_sigma_m', 'force_sigma_N'):
            try:
                noise[key] = _real(specs[key], key, nonnegative=True)
            except (KeyError, ValueError, TypeError):
                pass
    for d in range(5):
        reasons = list(global_reasons) + list(group_errors[d])
        sv, dv, present, iv = (masks[k][d] for k in ('valid','distributed_valid','contact_present','slip_valid5'))
        s = T = P = pbar = Jbar = vbar = robot2 = e = C = None
        support, shares = (), ()
        normal = None
        try:
            normal = float(component('normal', d, ()))
            if normal < 0:
                raise ValueError('negative_finger_normal')
        except (ValueError, TypeError):
            reasons.append('finger_normal_unknown')
        try:
            if iv is True and sv is True:
                s = component('slip_local', d, (2,))
                T = component('tangent_basis', d, (2,3))
                if not np.allclose(T @ T.T, np.eye(2), atol=1e-8, rtol=0):
                    raise ValueError('device_projection_basis_not_orthonormal')
            else:
                reasons.append('no_valid_retained_instrument_pair')
        except (ValueError, TypeError):
            reasons.append('relative_pair_or_basis_unknown')
            s = T = None
        empty = (not global_reasons and not group_errors[d] and sv is True and dv is True
                 and iv is False and present is False and normal == 0. and not grouped[d])
        known_empty_digits.append(empty)
        if not empty:
            if sv is not True or dv is not True:
                reasons.append('finger_or_distributed_geometry_validity_unknown')
            if present is not True:
                reasons.append('contact_presence_not_confirmed')
            if not grouped[d]:
                reasons.append('positive_patch_geometry_missing')
            if grouped[d] and normal is not None:
                patch_sum = sum(x[2] for x in grouped[d])
                numeric_tol = 64*np.finfo(float).eps*max(1.,abs(normal),abs(patch_sum))
                if abs(patch_sum-normal) > numeric_tol:
                    reasons.append('public_patch_pressure_coverage_incomplete')
            if s is not None and T is not None and not reasons:
                try:
                    group = grouped[d]
                    weights = np.asarray([x[2] for x in group], dtype=float)
                    weights = weights / weights.max(); weights = weights / weights.sum()
                    points, velocities, jacobians = [], [], []
                    patch_info = []
                    for (index, patch, n), w in zip(group, weights):
                        k = point_kinematics(robot_snapshot, patch)
                        if abs(_real(k.time_s, 'point FK time', nonnegative=True)-acq) > CLOCK_TOL_S:
                            raise ValueError('robot_point_clock_mismatch')
                        if tuple(k.axes25) != robot_snapshot.axes25:
                            raise ValueError('robot_point_axis_order_mismatch')
                        points.append(_array(k.point_world, (3,), 'robot acquired skin point'))
                        velocity = _array(k.point_velocity_world, (3,), 'robot acquired point velocity')
                        J = _array(k.jacobian3x25, (3,25), 'robot acquired point Jacobian')
                        if not np.allclose(velocity, J@robot_snapshot.velocity25, atol=1e-12, rtol=1e-12):
                            raise ValueError('robot_callback_velocity_not_from_same_qv_snapshot')
                        velocities.append(velocity); jacobians.append(J)
                        patch_info.append(_freeze(dict(local_patch_index=index, robot_body=patch.robot_body,
                            link_point3=patch.link_point, acquisition_time_s=acq, normal_N=n,
                            pressure_share=float(w))))
                    pbar = _array(weights @ np.asarray(points), (3,), 'pressure weighted public point')
                    vbar = _array(weights @ np.asarray(velocities), (3,), 'pressure weighted robot velocity')
                    Jbar = _array(np.einsum('p,pij->ij', weights, np.asarray(jacobians)), (3,25), 'acquired mean J')
                    P = _array(T @ robot_snapshot.rotation_world_from_palm3x3.T, (2,3), 'world projection')
                    robot2 = _array(P @ vbar, (2,), 'robot projection')
                    e = _array(robot2 + s, (2,), 'external projection')
                    C = _array(np.c_[P, -P @ _skew(pbar-anchor)], (2,6), 'conditional rigid C')
                    support, shares = tuple(patch_info), tuple(float(x) for x in weights)
                    values.append(e); Cs.append(C); keys.extend(((d, 0), (d, 1)))
                except (ValueError, TypeError, AttributeError, KeyError) as exc:
                    reasons.append(str(exc))
                    P = pbar = Jbar = vbar = robot2 = e = C = None
        status = 'KNOWN_NO_RETAINED_CONTACT' if empty else ('OBSERVED_PROJECTION' if e is not None else 'UNKNOWN_PROJECTION')
        if status == 'UNKNOWN_PROJECTION':
            unresolved = True
        records.append(DigitMotionRecord(d, status, tuple(dict.fromkeys(reasons)), sv, dv, present, iv,
            acq, now, age, normal, s, T, P, pbar, Jbar, vbar, robot2, e, C, support, shares,
            (stream_id, epoch_id, acq_tick, d), _freeze(dict(
                actor_association='UNASSIGNED', same_actor_verified=False,
                contact_group_consistency='UNKNOWN_EXTERNAL_ACTOR_MEMBERSHIP',
                complete_patch_coverage_required=True, noise=noise,
                complete_error_covariance_available=False,
                shared_digit_position_and_slip_noise=True,
                patch_indices_are_packet_local_not_persistent_actor_ids=True))))
    e_stack = _array(np.concatenate(values) if values else np.empty(0), (2*len(values),), 'projection stack')
    C_stack = _array(np.vstack(Cs) if Cs else np.empty((0,6)), (2*len(Cs),6), 'conditional C stack')
    known_empty = all(known_empty_digits)
    status = ('KNOWN_NO_ACTIVE_MOTION_ROWS' if known_empty else
              ('AVAILABLE_PARTIAL' if unresolved else 'AVAILABLE') if values else 'UNKNOWN_NO_MOTION_ROWS')
    return PublicMotionPacket(SCHEMA, stream_id, epoch_id, read_tick, acq_tick, now, acq, now, age,
        status, bool(values), known_empty, anchor, robot_snapshot, tuple(records), tuple(keys), e_stack, C_stack,
        _freeze(dict(rows_per_digit=2, row_units='m/s', C_columns=('world_v_at_anchor_m_s','world_omega_rad_s'),
            anchor_role='declared fixed public coordinate; not object centre or centre of mass',
            rigid_equation_role='CONDITIONAL_COMMON_RIGID_VELOCITY_FIELD_NOT_ESTABLISHED',
            actor_association='UNASSIGNED', same_actor_verified=False,
            external_normal_velocity_available=False, full_twist_available=False,
            covariance_complete=False, current_proposal_admission_not_consumed=True,
            basis_changes_require_explicit_common_subspace_not_zero_filled_rotation=True,
            missing_patch_subset_not_renormalized_as_complete_instrument=True,
            delayed_measurement_not_current_external_state=True,
            future_response_prediction_available=False, force_safety_certified=False)))


@dataclass(frozen=True)
class PublicInputEvent(_ImmutableOutput):
    stream_id: str
    epoch_id: str
    issue_time_s: float
    available_at_s: float
    robot_position25: np.ndarray
    robot_velocity25: np.ndarray
    precommit_reference21: np.ndarray
    precommit_engine21: object
    requested_reference21: np.ndarray
    executed_reference21: np.ndarray
    executed_engine21: np.ndarray
    actual_reference_increment21: np.ndarray
    mode: str
    mode_generation: int
    execution_changed: bool
    stale_plan_replaced: bool
    diagnostics: object

    def to_dict(self):
        return _plain(self)


class PublicMotionObserver:
    """Causal packet/receipt owner. Does not call adapter.observe or any sensor.

    Caller first runs its unchanged adapter.observe(frame). Command memory may
    refresh on this tick without changing the consumed measurement packet.
    """
    def __init__(self, *, stream_id, epoch_id, anchor_world3=(0.,0.,0.)):
        self.stream_id = _identity_text(stream_id, 'stream_id')
        self.epoch_id = _identity_text(epoch_id, 'epoch_id')
        self.anchor_world3 = _array(anchor_world3, (3,), 'fixed public anchor')
        self._packets = []; self._packet_signatures = {}; self._events = []

    def observe(self, frame, adapter):
        tick, now = _tick(frame.time, 'public read time')
        if self._packets and now < self._packets[-1].read_time_s-CLOCK_TOL_S:
            raise ValueError('Public clock rewind requires a fresh observer epoch')
        _, acq = _tick(frame.sample_time, 'public acquisition time')
        history = getattr(adapter, '_motion_samples', {})
        old = history.get(int(round(acq/PERIOD_S))) if isinstance(history, dict) else None
        snapshot = None
        geometry = getattr(adapter, 'geometry', None)
        if getattr(adapter, '_last_time', None) is not None and abs(adapter._last_time-now) <= CLOCK_TOL_S:
            if old is not None and old.get('reason') is None:
                try:
                    snapshot = RobotAcquisitionSnapshot25(old['time_s'], old['q'], old['v'], old['R'],
                        tuple(geometry.raw_joint_names)+('transport_lift',))
                except (ValueError, TypeError, AttributeError, KeyError):
                    snapshot = None

        def robot_only(sample, patch):
            raw = RobotJointSample(sample.time_s, geometry.raw_joint_names,
                sample.position25[:24], sample.position25[24], sample.velocity25[:24], sample.velocity25[24])
            return geometry.point_kinematics(raw, patch)

        packet = build(frame, robot_snapshot=snapshot, point_kinematics=robot_only,
            stream_id=self.stream_id, epoch_id=self.epoch_id, anchor_world3=self.anchor_world3)
        consumed = ('time','sample_time','age','normal','valid','distributed_valid','contact_present',
                    'slip_local','tangent_basis','slip_valid5','slip_domain_role','slip_domain_available',
                    'slip_domain_acquisition_time_s','slip_domain_time_s','skin_patches',
                    'robot_position25','robot_velocity25')
        # Include source payload even where an unknown value is not in a derived
        # row. Command-memory refresh is intentionally outside this signature.
        source = {k:_plain(getattr(frame,k,None)) for k in consumed}
        signature = json.dumps(dict(packet=packet.to_dict(),consumed=source), sort_keys=True,
                               separators=(',', ':'), allow_nan=True)
        if tick in self._packet_signatures:
            if signature != self._packet_signatures[tick]:
                raise ValueError('Consumed public measurement changed on duplicate read tick')
            return self._packets[-1]
        self._packet_signatures[tick] = signature; self._packets.append(packet)
        return packet

    def on_receipt(self, receipt, precommit_snapshot):
        """Log one actual commit, not desired/QP input or an inferred old input.

        precommit_snapshot required keys: time_s, reference21, robot_position25,
        robot_velocity25. Optional keys documented below; all other keys reject.
        A receipt alone proves a command issue, not subsequent substep holding.
        """
        from v13_joint.plant import ExecutionReceipt21
        if not isinstance(receipt, ExecutionReceipt21):
            raise TypeError('Actual frozen ExecutionReceipt21 required')
        if not isinstance(precommit_snapshot, Mapping):
            raise TypeError('Public precommit snapshot mapping required')
        allowed = {'time_s','reference21','engine21','robot_position25','robot_velocity25',
                   'robot_actuator_axes21','reference_lower21','reference_upper21','mode_generation','phase'}
        if set(precommit_snapshot)-allowed:
            raise ValueError('Unexpected precommit fields; truth/environment is not an input')
        _, now = _tick(precommit_snapshot['time_s'], 'precommit time')
        if not isinstance(receipt.requested, Reference21) or not isinstance(receipt.executed, Reference21) or not isinstance(receipt.engine, EnginePort21):
            raise TypeError('Typed reference21/engine21 required on actual receipt')
        if any(abs(t-now)>CLOCK_TOL_S for t in (receipt.requested.time_s,receipt.executed.time_s,receipt.engine.time_s)):
            raise ValueError('Actual receipt/precommit clocks disagree')
        q = _array(precommit_snapshot['robot_position25'], (25,), 'precommit q25')
        v = _array(precommit_snapshot['robot_velocity25'], (25,), 'precommit v25')
        raw = precommit_snapshot['reference21']
        if isinstance(raw, Reference21):
            if abs(raw.time_s-now)>CLOCK_TOL_S:
                raise ValueError('Precommit reference clock mismatch')
            raw = raw.values
        pre = _array(raw, (21,), 'precommit unbiased reference21')
        mode = receipt.mode
        if mode not in ('holding','release','monitor'):
            raise ValueError('Unknown actual receipt mode')
        gen = _integer(receipt.mode_generation, 'mode generation')
        if any(not isinstance(x, (bool,np.bool_)) for x in (receipt.execution_changed,receipt.stale_plan_replaced)):
            raise ValueError('Explicit receipt flags required')
        if bool(receipt.execution_changed) != (receipt.requested.values.tobytes()!=receipt.executed.values.tobytes()):
            raise ValueError('execution_changed disagrees with actual receipt values')
        encoded = receipt.executed.values.copy(); encoded[18] += receipt.engine.lift_bias_m
        if encoded.tobytes()!=receipt.engine.values.tobytes():
            raise ValueError('Executed reference/engine authority disagree')
        pre_engine = None
        if 'engine21' in precommit_snapshot:
            pe = precommit_snapshot['engine21']
            if isinstance(pe, EnginePort21):
                if abs(pe.time_s-now)>CLOCK_TOL_S or pe.lift_bias_m!=receipt.engine.lift_bias_m:
                    raise ValueError('Precommit engine clock/bias mismatch')
                pe = pe.values
            pre_engine = _array(pe,(21,),'precommit engine21')
            encoded_pre=pre.copy();encoded_pre[18]+=receipt.engine.lift_bias_m
            if encoded_pre.tobytes()!=pre_engine.tobytes():
                raise ValueError('Precommit reference/engine authority disagree')
        if 'robot_actuator_axes21' in precommit_snapshot and tuple(precommit_snapshot['robot_actuator_axes21'])!=ACTUATORS21:
            raise ValueError('Precommit actuator axes unknown/reordered')
        bounds = {}
        for name in ('reference_lower21','reference_upper21'):
            if name in precommit_snapshot:
                bounds[name] = _array(precommit_snapshot[name],(21,),name)
        if len(bounds)==1:
            raise ValueError('Both actual reference bounds must be provided or neither')
        if bounds and np.any(bounds['reference_lower21']>bounds['reference_upper21']):
            raise ValueError('Empty declared actual reference bounds')
        pg = precommit_snapshot.get('mode_generation')
        if pg is not None:
            pg = _integer(pg,'precommit mode generation')
        phase = precommit_snapshot.get('phase')
        if phase is not None and not isinstance(phase,str):
            raise ValueError('Public phase must be a string or unknown')
        event = PublicInputEvent(self.stream_id,self.epoch_id,now,now,q,v,pre,pre_engine,
            _array(receipt.requested.values,(21,),'requested21'),
            _array(receipt.executed.values,(21,),'executed21'),
            _array(receipt.engine.values,(21,),'actual engine21'),
            _array(receipt.executed.values-pre,(21,),'actual increment21'),mode,gen,
            bool(receipt.execution_changed),bool(receipt.stale_plan_replaced),_freeze(dict(
                reference_axes21=REFERENCE_AXES21,engine_axes21=ACTUATORS21,
                known_actor_lift_bias_m=receipt.engine.lift_bias_m,
                precommit_engine_available=pre_engine is not None,actual_reference_bounds=bounds,
                precommit_mode_generation=pg,phase=phase,substep_hold_verified=False,
                input_end_time_s=None,old_acquisition_input_not_backfilled=True)))
        if self._events and now < self._events[-1].issue_time_s-CLOCK_TOL_S:
            raise ValueError('Receipt clock rewind requires a new observer epoch')
        if self._events and abs(now-self._events[-1].issue_time_s)<=CLOCK_TOL_S:
            if json.dumps(event.to_dict(),sort_keys=True,allow_nan=False)!=json.dumps(self._events[-1].to_dict(),sort_keys=True,allow_nan=False):
                raise ValueError('More than one changed actual commit on the same logged tick')
            return self._events[-1]
        self._events.append(event)
        return event

    def packets(self, *, available_at_s=None):
        t = float('inf') if available_at_s is None else _real(available_at_s,'available_at',nonnegative=True)
        return tuple(p for p in self._packets if p.available_at_s <= t+CLOCK_TOL_S)

    def receipts(self, *, available_at_s=None):
        t = float('inf') if available_at_s is None else _real(available_at_s,'available_at',nonnegative=True)
        return tuple(p for p in self._events if p.available_at_s <= t+CLOCK_TOL_S)

    def input_intervals(self, *, available_at_s=None):
        """Adjacent public command issues; not a substep execution certificate."""
        events = self.receipts(available_at_s=available_at_s)
        return tuple(_freeze(dict(start_s=a.issue_time_s,end_s=b.issue_time_s,
            available_at_s=b.available_at_s,executed_reference21=a.executed_reference21,
            consecutive_5ms=abs(b.issue_time_s-a.issue_time_s-PERIOD_S)<=CLOCK_TOL_S,
            substep_hold_verified=False,role='ADJACENT_ACTUAL_COMMAND_RECEIPTS_NOT_PHYSICS_PROOF'))
            for a,b in zip(events,events[1:]))

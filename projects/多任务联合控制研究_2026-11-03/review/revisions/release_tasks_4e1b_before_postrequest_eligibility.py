"""Pure receiver-aware release graph; reference tasks, never contact prediction.

Fingers always follow the frozen V13 release.  Only lift intent/authority and
phase are coordinated.  The receiver is a delayed instrument on any contacting
actor, and low hand pressure is a sampled proxy, not geometric clearance.
"""
from copy import deepcopy
from numbers import Real
import json
import numpy as np

from v13_a1.codec import Reference21, finite_array, finite_scalar
from v13_joint.lifecycle import ReleaseEntry, release_nominal
from .receiver_sensor import ReceiverMeasurement, PERIOD_S, DELAY_S

PHASES = ('APPROACH', 'UNKNOWN_HOLD', 'GUARDED_CLEAR', 'WITHDRAW', 'MONITOR')
HAND_CLEARANCE_N = .2
HAND_DWELL_S = .2
HAND_MAX_AGE_S = .12
DESCENT_DURATION_S = .6
WITHDRAW_DURATION_S = .6
RELEASE_TERMINAL_S = 2.2
CLOCK_TOL_S = 1e-9
NUMERIC_EPS = 64. * np.finfo(float).eps
_WIDE_LOWER = np.full(21, -np.finfo(float).max)
_WIDE_UPPER = np.full(21, np.finfo(float).max)


def _tick(value, name):
    t = finite_scalar(value, name)
    k = int(round(t / PERIOD_S))
    if t < 0. or abs(t - k * PERIOD_S) > CLOCK_TOL_S:
        raise ValueError(name + ' must be a nonnegative 200Hz public clock')
    return k, k * PERIOD_S


def _observed(value, n, name):
    """Nonfinite/missing measurements remain unknown, never substituted by zero."""
    if value is None:
        return np.full(n, np.nan)
    raw = np.asarray(value, dtype=object)
    if raw.shape != (n,) or any(isinstance(x, (bool, np.bool_)) or
            x is not None and not isinstance(x, Real) for x in raw):
        raise ValueError(name + ' requires real measurements or explicit unknowns')
    return np.array([np.nan if x is None else float(x) for x in raw])


def _valid(value, name):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (5,) or any(not isinstance(x, (bool, np.bool_)) for x in raw):
        raise ValueError(name + ' requires five explicit bool flags')
    return np.asarray(value, dtype=bool)


def _optional_scalar(value, name):
    if value is None:
        return np.nan
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + ' requires a real measurement or unknown')
    return float(value)


def _safe(value):
    if isinstance(value, np.ndarray):
        return [_safe(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(x) for x in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _equal_numeric(actual, target):
    scale = np.maximum(1., np.maximum(abs(actual), abs(target)))
    return bool(np.all(abs(actual - target) <= NUMERIC_EPS * scale))


class ReceiverReleaseTasks:
    """Stateful task graph with no model, plant, optimizer, or engine writer.

    ``current_ref21`` is the actual executed *unbiased* robot reference at this
    update's clock. ``robot_lift_m`` is its separately observed lift encoder.
    Call update at every 5ms public tick, then query intent and lift authority.
    Caller intersects this authority with global ports, rates, and other tasks.
    """
    def __init__(self, entry: ReleaseEntry, open18):
        if not isinstance(entry, ReleaseEntry):
            raise TypeError('Typed actual ReleaseEntry required')
        self._entry_tick, self.started_s = _tick(entry.started_s, 'release entry')
        if entry.reference21.values[18] < 0.:
            raise ValueError('This descent-to-zero graph requires nonnegative entry lift')
        self.entry = entry
        self.declared_open18 = finite_array(open18, (18,), 'declared open18')
        # Beyond the endpoint avoids binary subtraction accidentally evaluating
        # 2.199999999...; it does not change the actual nominal time trajectory.
        self.release_terminal18 = release_nominal(entry, entry.started_s + 3.,
            self.declared_open18, _WIDE_LOWER, _WIDE_UPPER).values[:18].copy()
        self.release_terminal18.setflags(write=False)
        self.phase = 'UNKNOWN_HOLD'
        self.version = int(entry.mode_generation)
        self._last_tick = None
        self._signature = None
        self._now = None
        self._current = None
        self._encoder_lift = None
        self._hold_anchor = None
        self._descent_start_s = None
        self._descent_start_lift = None
        self._withdraw_start_s = None
        self._withdraw_start_lift = None
        self._withdraw_goal = None
        self._withdraw_started_once = False
        self._receiver_latched = False
        self._receiver_streak = 0
        self._receiver_last_acq = None
        self._receiver_evidence = {'available': False, 'reason': 'not_updated'}
        self._low_since = None
        self._hand_last_acq = None
        self._low_confirmed = False
        self._hand_evidence = {'available': False, 'reason': 'not_updated'}
        self._terminal_reference = False
        self._history = []
        self._last_proposal = None

    def _structural(self):
        return (self.phase, self._receiver_latched, self._hold_anchor,
            self._descent_start_s, self._descent_start_lift,
            self._withdraw_start_s, self._withdraw_start_lift, self._withdraw_goal)

    def _hold(self, phase, current, encoder):
        if self.phase != phase or self._hold_anchor is None:
            self._hold_anchor = max(float(current[18]), encoder)
        self.phase = phase

    def _approach(self, current, now):
        if self.phase != 'APPROACH' or self._descent_start_s is None:
            self._descent_start_s = now
            self._descent_start_lift = float(current[18])
        self.phase = 'APPROACH'

    def _clear_hand_evidence(self):
        self._low_since = None
        self._hand_last_acq = None
        self._low_confirmed = False

    def _update_hand(self, now, normal, valid, age, aux, aux_valid, aux_age, gap):
        available = bool(valid.all() and np.isfinite(normal).all() and
            np.isfinite(age).all() and np.all(age >= 0.) and
            np.all(age <= HAND_MAX_AGE_S + CLOCK_TOL_S) and
            aux_valid and np.isfinite(aux) and np.isfinite(aux_age) and
            0. <= aux_age <= HAND_MAX_AGE_S + CLOCK_TOL_S and
            np.all(age <= now + CLOCK_TOL_S) and aux_age <= now + CLOCK_TOL_S)
        if not available:
            self._clear_hand_evidence()
            self._hand_evidence = dict(available=False, reason='missing_invalid_or_stale',
                load_proxy_N=None, oldest_acquisition_s=None, low=False,
                confirmed=False, dwell_s=None)
            return
        stamp = now - max(float(np.max(age)), aux_age)
        load = float(np.maximum(normal, 0.).sum() + max(aux, 0.))
        low = load <= HAND_CLEARANCE_N
        previous = self._hand_last_acq
        rewound = previous is not None and stamp < previous - CLOCK_TOL_S
        if gap or rewound:
            self._clear_hand_evidence()
            previous = None
        fresh = previous is None or stamp > previous + CLOCK_TOL_S
        continuous = previous is not None and stamp - previous <= PERIOD_S + CLOCK_TOL_S
        if not low:
            self._clear_hand_evidence()
        elif fresh:
            if self._low_since is None or not continuous:
                self._low_since = stamp
            self._hand_last_acq = stamp
            self._low_confirmed = stamp - self._low_since >= HAND_DWELL_S - CLOCK_TOL_S
        # A duplicate acquisition cannot advance dwell or create confirmation.
        dwell = None if self._low_since is None else max(0., stamp - self._low_since)
        self._hand_evidence = dict(available=not rewound,
            reason='acquisition_rewind' if rewound else None,
            load_proxy_N=load, oldest_acquisition_s=stamp, low=low and not rewound,
            fresh_acquisition=fresh, duplicate_acquisition=not fresh,
            confirmed=self._low_confirmed and not rewound, dwell_s=dwell)
        if rewound:
            self._clear_hand_evidence()

    def _update_receiver(self, receiver, now, gap):
        same_clock = abs(receiver.time_s - now) <= CLOCK_TOL_S
        available = bool(same_clock and receiver.valid and receiver.age_s is not None
            and abs(receiver.age_s - DELAY_S) <= CLOCK_TOL_S and
            receiver.acquisition_time_s is not None)
        if not available:
            self._receiver_streak = 0
            self._receiver_last_acq = None
            self._receiver_evidence = dict(available=False,
                reason='delivery_clock_mismatch' if not same_clock else receiver.invalid_reason,
                load_N=None, engineering_threshold_N=float(receiver.uncertainty6[2]),
                compressive_contact=None, consecutive_fresh_acquisitions=0,
                acquisition_time_s=receiver.acquisition_time_s)
            return
        acq = receiver.acquisition_time_s
        previous = self._receiver_last_acq
        fresh = previous is None or acq > previous + CLOCK_TOL_S
        adjacent = previous is not None and abs(acq - previous - PERIOD_S) <= CLOCK_TOL_S
        above = receiver.normal_load_N > float(receiver.uncertainty6[2])
        if gap or previous is not None and acq < previous - CLOCK_TOL_S:
            self._receiver_streak = 0
            adjacent = False
        if not above:
            self._receiver_streak = 0
        elif fresh:
            self._receiver_streak = self._receiver_streak + 1 if adjacent else 1
        if fresh:
            self._receiver_last_acq = acq
        if self._receiver_streak >= 2:
            self._receiver_latched = True
        self._receiver_evidence = dict(available=True, reason=None,
            load_N=receiver.normal_load_N, engineering_threshold_N=float(receiver.uncertainty6[2]),
            compressive_contact=above, fresh_acquisition=fresh,
            consecutive_fresh_acquisitions=self._receiver_streak,
            acquisition_time_s=acq, sampling_phase=receiver.sampling_phase)

    def update(self, *, time_s, current_ref21, robot_lift_m,
            normal5, valid5, age5, aux_pressure, aux_valid, aux_age,
            receiver: ReceiverMeasurement):
        tick, now = _tick(time_s, 'release update')
        if tick < self._entry_tick:
            raise ValueError('Release update precedes entry')
        if not isinstance(current_ref21, Reference21):
            raise TypeError('Actual unbiased Reference21 required, not EnginePort21/array')
        if abs(current_ref21.time_s - now) > CLOCK_TOL_S:
            raise ValueError('Actual executed reference/update clock mismatch')
        if not isinstance(receiver, ReceiverMeasurement):
            raise TypeError('Typed ReceiverMeasurement required')
        encoder = finite_scalar(robot_lift_m, 'public robot lift encoder')
        normal = _observed(normal5, 5, 'hand normal5')
        valid = _valid(valid5, 'hand valid5')
        age = _observed(age5, 5, 'hand age5')
        aux = _optional_scalar(aux_pressure, 'auxiliary pressure')
        aux_age = _optional_scalar(aux_age, 'auxiliary age')
        if not isinstance(aux_valid, (bool, np.bool_)):
            raise ValueError('Explicit auxiliary validity required')
        current = current_ref21.values
        signature = (current.tobytes(), encoder, normal.tobytes(), valid.tobytes(),
            age.tobytes(), aux, bool(aux_valid), aux_age,
            json.dumps(receiver.to_public(), sort_keys=True, allow_nan=False))
        # NaN is an explicit unknown but must still be idempotent by bit pattern.
        signature = tuple(np.float64(x).tobytes() if isinstance(x, float) else x for x in signature)
        if self._last_tick is not None:
            if tick < self._last_tick:
                raise ValueError('Release public clock rewound')
            if tick == self._last_tick:
                if signature != self._signature:
                    raise ValueError('Different payload on an already consumed release tick')
                return self.state_dict()
        gap = self._last_tick is not None and tick != self._last_tick + 1
        before = self._structural()
        self._now = now
        self._current = current.copy()
        self._encoder_lift = encoder
        self._update_hand(now, normal, valid, age, aux, bool(aux_valid), aux_age, gap)
        self._update_receiver(receiver, now, gap)
        self._terminal_reference = _equal_numeric(current[:18], self.release_terminal18)
        elapsed_terminal = tick - self._entry_tick >= int(round(RELEASE_TERMINAL_S / PERIOD_S))
        hand_usable_low = bool(self._hand_evidence['available'] and self._hand_evidence['low'])
        retreat_interrupted = self.phase in ('WITHDRAW', 'MONITOR') and not hand_usable_low
        if retreat_interrupted:
            self._clear_hand_evidence()
            self._hand_evidence['confirmed'] = False
            self._hold('GUARDED_CLEAR', current, encoder)
        elif self.phase == 'WITHDRAW':
            if now - self._withdraw_start_s >= WITHDRAW_DURATION_S - CLOCK_TOL_S and \
                    _equal_numeric(np.array([current[18]]), np.array([self._withdraw_goal])):
                self.phase = 'MONITOR'
        elif self.phase == 'MONITOR':
            pass
        elif self._low_confirmed and elapsed_terminal and self._terminal_reference:
            self.phase = 'WITHDRAW'
            self._withdraw_started_once = True
            self._withdraw_start_s = now
            self._withdraw_start_lift = float(current[18])
            self._withdraw_goal = max(float(self.entry.reference21.values[18]), float(current[18]))
        elif self._receiver_latched or self._withdraw_started_once:
            self._hold('GUARDED_CLEAR', current, encoder)
        elif not self._receiver_evidence['available']:
            self._hold('UNKNOWN_HOLD', current, encoder)
        elif self._receiver_evidence['compressive_contact']:
            # First above-threshold acquisition is not yet a receiver latch.
            # An existing approach continues; an unknown segment stays braked.
            if self.phase != 'APPROACH':
                self._hold('UNKNOWN_HOLD', current, encoder)
        else:
            self._approach(current, now)
        if before != self._structural():
            self.version += 1
            self._history.append(dict(time_s=now, phase=self.phase, version=self.version,
                receiver_contact_latched=self._receiver_latched))
        self._last_tick = tick
        self._signature = signature
        return self.state_dict()

    def _require_updated(self):
        if self._now is None:
            raise RuntimeError('No public release observation has been consumed')

    def _lift_nominal(self, query):
        if self.phase == 'APPROACH':
            speed = float(self.entry.reference21.values[18]) / DESCENT_DURATION_S
            return max(0., self._descent_start_lift - speed * (query - self._descent_start_s))
        if self.phase in ('UNKNOWN_HOLD', 'GUARDED_CLEAR'):
            return self._hold_anchor
        if self.phase == 'WITHDRAW':
            blend = float(np.clip((query - self._withdraw_start_s) / WITHDRAW_DURATION_S, 0., 1.))
            return self._withdraw_start_lift + blend * (self._withdraw_goal - self._withdraw_start_lift)
        return self._withdraw_goal

    def proposal(self, query_time, global_lower, global_upper):
        self._require_updated()
        query = finite_scalar(query_time, 'release intent query')
        if query < self._now - CLOCK_TOL_S:
            raise ValueError('Cannot query a release intent before its latest evidence')
        lo = finite_array(global_lower, (21,), 'global reference lower21')
        hi = finite_array(global_upper, (21,), 'global reference upper21')
        if np.any(lo > hi):
            raise ValueError('Global reference box is empty')
        raw = release_nominal(self.entry, query, self.declared_open18,
            _WIDE_LOWER, _WIDE_UPPER).values.copy()
        original = raw.copy()
        raw[18] = self._lift_nominal(query)
        # The original finger terminal is preserved after withdrawal, even if
        # query times become large.  No force/receiver condition re-closes it.
        projected = np.clip(raw, lo, hi)
        self._last_proposal = _safe(dict(query_time_s=query, evidence_time_s=self._now,
            phase=self.phase, version=self.version,
            raw_frozen_nominal21=original, raw_graph_nominal21=raw,
            original_descent_terminal_m=0., global_projected21=projected,
            global_projection_changed=raw != projected,
            global_projection_max_abs=float(np.max(abs(raw - projected))),
            mode_bounds_applied=False, rate_bounds_applied=False,
            lift_authority_interval_m=self.lift_interval(Reference21(self._current, self._now)),
            projection_role='global_box_only;caller_intersects_mode_and_rate_authority'))
        return Reference21(projected, query)

    def lift_interval(self, current_ref):
        self._require_updated()
        if not isinstance(current_ref, Reference21):
            raise TypeError('Typed current unbiased Reference21 required')
        if abs(current_ref.time_s - self._now) > CLOCK_TOL_S:
            raise ValueError('Lift authority must use the latest executed reference')
        current = float(current_ref.values[18])
        if self.phase == 'APPROACH':
            return 0., max(current, float(self.entry.reference21.values[18]), self._descent_start_lift)
        if self.phase in ('UNKNOWN_HOLD', 'GUARDED_CLEAR'):
            return current, max(current, self._hold_anchor)
        return current, max(current, self._withdraw_goal)

    def state_dict(self):
        return deepcopy(_safe(dict(schema='receiver-release-tasks-v1',
            phase=self.phase, version=self.version, started_s=self.started_s,
            current_time_s=self._now, entry_reference21=self.entry.reference21.values,
            current_reference21=self._current, public_lift_encoder_m=self._encoder_lift,
            declared_open18=self.declared_open18, release_terminal18=self.release_terminal18,
            terminal_reference_matched=self._terminal_reference,
            terminal_gate_role='executed_release_terminal_reference_not_geometric_gap',
            terminal_numeric_relative_tolerance=NUMERIC_EPS,
            receiver_contact_latched=self._receiver_latched,
            receiver=self._receiver_evidence, hand=self._hand_evidence,
            hand_low_since_acquisition_s=self._low_since,
            hand_last_oldest_acquisition_s=self._hand_last_acq,
            hand_low_confirmed=self._low_confirmed,
            hold_anchor_m=self._hold_anchor, descent_start_s=self._descent_start_s,
            descent_start_lift_m=self._descent_start_lift,
            descent_speed_m_s=float(self.entry.reference21.values[18]) / DESCENT_DURATION_S,
            withdraw_start_s=self._withdraw_start_s, withdraw_start_lift_m=self._withdraw_start_lift,
            withdraw_goal_m=self._withdraw_goal, withdraw_started_once=self._withdraw_started_once,
            thresholds=dict(receiver_consecutive_fresh_acquisitions=2,
                receiver_age_s=DELAY_S, receiver_gate='load_N > uncertainty6[2]',
                receiver_uncertainty_role='declared_three_sigma_engineering_proxy_not_hard_bound',
                hand_low_pressure_N=HAND_CLEARANCE_N, hand_dwell_s=HAND_DWELL_S,
                hand_max_age_s=HAND_MAX_AGE_S, terminal_elapsed_s=RELEASE_TERMINAL_S,
                descent_duration_s=DESCENT_DURATION_S, withdraw_duration_s=WITHDRAW_DURATION_S),
            semantics=dict(any_actor_receiver_contact_not_target_identity=True,
                hand_low_is_sampled_pressure_proxy_not_geometric_clearance=True,
                receiver_contact_not_placed_proof=True,
                brake_is_reference_not_instantaneous_physical_stop=True,
                fingers_always_frozen_release_no_holding_recovery=True,
                authority_separate_from_nominal_proposal=True,
                unknown_measurements_never_replaced_by_zero=True,
                no_force_prediction_no_safety_certificate=True),
            phase_history=self._history, last_proposal=self._last_proposal)))

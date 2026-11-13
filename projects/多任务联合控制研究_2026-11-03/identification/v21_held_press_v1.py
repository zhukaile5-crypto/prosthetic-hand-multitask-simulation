"""Independent one-axis load task on the existing five-finger holding controller."""

from collections import deque

import numpy as np

from task_trajectory import unknown_future_point
from v14_coord.design import DECISION_PERIOD_S, HOLDING_REFERENCE_RATE21
from v15_support.receiver_sensor import DELAY_S, PERIOD_S
from v21_stable_handoff.runtime import ReleaseSession

ARMS = ("ratio_pi", "current", "expanded")
START_S = 8.05
HOLD_START_S = 11.30
END_S = 12.
TARGET_N = 2.
CONTACT_N = .15
CONTACT_SAMPLES = 3
SLOW_LIFT_REFERENCE_M = .047
FAST_APPROACH_M_S = .032
SLOW_APPROACH_M_S = .015
GAIN_M_S_PER_N = .006
PRESS_MAX_M_S = .009
UNLOAD_MAX_M_S = .015
HIGH_LOAD_N = 3.5
FILTER_ALPHA = .25
LEAD_S = .025
RATE_ESTIMATE_SPAN_S = .020
RATE_CAP_N_S = 20.
CONTROL_TICKS = 2400


def force_rate(load, target=TARGET_N):
    return float(np.clip(GAIN_M_S_PER_N*(load-target), -PRESS_MAX_M_S, UNLOAD_MAX_M_S))


class PressLift:
    """A shared load task with an optional delay-aware force estimate."""

    def __init__(self, start_m, predictive):
        self.start_m = float(start_m)
        self.predictive = predictive
        self.target_m = self.start_m
        self.consecutive_contact = 0
        self.contacted = False
        self.filtered = None
        self.history = deque(maxlen=5)
        self.records = []

    def position_at(self, now):
        return [self.target_m, 0., 0.]

    def preview(self, now):
        return unknown_future_point(now, self.position_at(now))

    def as_dict(self):
        return dict(schema="independent_held_press_intent_v1", common_user_goal_force_N=TARGET_N,
                    approach_start_s=START_S, hold_score_start_s=HOLD_START_S, end_s=END_S,
                    dynamic_internal_lift_target=True, fixed_wrist_rad=[0.,0.],
                    known_future_sensor_values=False)

    def load_estimate(self, load):
        if not self.predictive:
            return load
        self.filtered = load if self.filtered is None else self.filtered + FILTER_ALPHA*(load-self.filtered)
        self.history.append(self.filtered)
        slope = 0. if len(self.history) < self.history.maxlen else (self.history[-1]-self.history[0])/RATE_ESTIMATE_SPAN_S
        return max(0., self.filtered+LEAD_S*float(np.clip(slope,-RATE_CAP_N_S,RATE_CAP_N_S)))

    def observe(self, *, now, reference, packet, hand_valid, lower, upper):
        if now < START_S:
            return
        known = (packet.valid and packet.time_s == now and packet.acquisition_time_s is not None
                 and abs(now-packet.acquisition_time_s-DELAY_S) < 1e-8)
        measured = float(packet.normal_load_N) if known else None
        used = None
        speed = 0.
        if known:
            self.consecutive_contact = self.consecutive_contact+1 if measured >= CONTACT_N else 0
            self.contacted |= self.consecutive_contact >= CONTACT_SAMPLES
            used = self.load_estimate(measured)
            if self.contacted:
                speed = force_rate(used)
            else:
                speed = -FAST_APPROACH_M_S if reference > SLOW_LIFT_REFERENCE_M else -SLOW_APPROACH_M_S
            if measured >= HIGH_LOAD_N:
                speed = UNLOAD_MAX_M_S
            if not hand_valid:
                speed = max(0.,speed)
        else:
            self.consecutive_contact = 0
            self.filtered = None
            self.history.clear()
        limit = HOLDING_REFERENCE_RATE21[18]
        if abs(speed) > limit:
            raise ValueError("Held press rate exceeds the registered lift authority")
        self.target_m = float(np.clip(reference+DECISION_PERIOD_S*speed, lower, upper))
        self.records.append(dict(time_s=now, acquisition_time_s=packet.acquisition_time_s,
            load_measurement_valid=known, measured_load_N=measured, feedback_load_N=used,
            hand_contact_and_force_permission=bool(hand_valid), admitted_contact=self.contacted,
            lift_rate_request_m_s=speed, committed_lift_reference_m=reference, internal_lift_target_m=self.target_m,
            goal_N=TARGET_N))


def fixture_identity(session):
    """Static fixture geometry and the named-hand collision exclusion."""
    model = session.plant.model
    receiver = model.geom("support")
    idx = int(receiver.id)
    mask = (model.geom_contype[idx] & model.geom_conaffinity) | (model.geom_conaffinity[idx] & model.geom_contype)
    if (int(model.geom_bodyid[idx]) != 0 or not np.array_equal(receiver.quat,[1.,0.,0.,0.])
            or any(mask[g] for g in session.plant.hand_geoms)):
        raise ValueError("Expected a horizontal static receiver with no direct hand collision")
    return dict(receiver_name="support", top_height_m=float(receiver.pos[2]+receiver.size[2]),
                half_sizes_m=receiver.size[:2].tolist(), hand_receiver_direct_collision_mask=False,
                attribution="offline target and any-actor contact ledgers must agree")


class PressSession(ReleaseSession):
    def __init__(self, *, arm, seed):
        if arm not in ARMS:
            raise ValueError("Unknown press arm")
        self.press_arm = arm
        self.press_finished = False
        super().__init__(method="ratio_effort", scenario="nominal", split="robustness", seed=seed,
                         motion_mode="hold", task_policy="legacy" if arm=="ratio_pi" else "integrated",
                         receiver_policy="blind" if arm=="ratio_pi" else "coordinated")
        self.press_fixture = fixture_identity(self)
        self.user_trajectory = PressLift(self.config.lift_target, predictive=arm=="expanded")

    def observe_public(self):
        result = super().observe_public()
        frame = result[2]
        normal = np.asarray(frame.normal)
        valid = np.asarray(frame.valid)
        touch = valid & np.asarray(frame.contact_present) & (normal>.2)
        permission = (valid.all() and touch[0] and np.count_nonzero(touch)>=3
                      and np.isfinite(normal).all() and np.all(normal <= np.asarray(self.limits)-.02)
                      and normal.sum() <= self.budget-.02)
        self.user_trajectory.observe(now=float(frame.time), reference=float(frame.robot_reference21[18]),
            packet=self.latest_receiver_measurement, hand_valid=permission,
            lower=self.motion_reference_bounds["lower"][0], upper=self.motion_reference_bounds["upper"][0])
        return result

    def metadata(self):
        meta = super().metadata()
        meta.update(schema="held_press_trial_v1", experiment_arm=self.press_arm, duration=END_S,
                    observed_until_s=self.time, completed=self.press_finished, scored_release=False,
                    held_press=dict(target_N=TARGET_N, press_start_s=START_S, hold_start_s=HOLD_START_S,
                        hold_end_s=END_S, fixture=self.press_fixture, receiver_period_s=PERIOD_S,
                        receiver_delay_s=DELAY_S, feedback="filtered_delay_lead" if self.press_arm=="expanded"
                        else "proportional_on_delayed_measurement"))
        return meta


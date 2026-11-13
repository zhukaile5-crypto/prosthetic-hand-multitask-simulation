"""Independent held-object wrist target and two-axis effort-aim coordinator."""

import math

import numpy as np

from v21_stable_handoff.runtime import ReleaseSession as FrozenV21Session
from task_trajectory import PublicTaskTrajectory
from v13_a1.codec import Reference21
from v14_coord.design import DECISION_PERIOD_S, HOLDING_REFERENCE_RATE21
from v14_coord.effort_domain import FastEffortGoal21
from v15_support.controller import UnifiedTaskController
from v15_support import runtime as held

START_S = 9.3
RAMP_END_S = 10.3
END_S = 12.
PULL_START_S = 9.85
PULL_END_S = 10.15
PULL_N = .9
TARGET_RAD = (.06, -.045)
CONTROL_TICKS = 2400
WRIST = slice(19, 21)
AXES = ("rh_WRJ2", "rh_WRJ1")
ARMS = ("ratio_pi", "current", "expanded")
CASES = ("normal", "midturn_pull")
BASE_SCHEDULE = held.event_schedule


def trajectory(lift):
    return PublicTaskTrajectory(START_S, RAMP_END_S, (lift, 0., 0.),
        cosine_amplitude=(0., *(-x / 2. for x in TARGET_RAD)),
        angular_frequency=(0., math.pi, math.pi))


def case_schedule(case):
    if case not in CASES:
        raise ValueError("Unknown wrist-alignment case")

    def schedule(name, now, friction, lift):
        if name != "nominal":
            raise ValueError("Wrist pairs require the same nominal physical scene")
        env, target, wrist = BASE_SCHEDULE(name, now, friction, lift)
        if case == "midturn_pull" and PULL_START_S <= now < PULL_END_S:
            env["force"][1] = PULL_N
        return env, target, wrist

    return schedule


def wrist_aim(nominal, position, velocity, lower, upper, step):
    """Compensate current tracking lag, inside the same feasible wrist box."""
    projected = position + DECISION_PERIOD_S * velocity
    lead = np.clip(nominal - projected, -step, step)
    return np.clip(nominal + lead, lower, upper)


class AlignmentEffortGoal21(FastEffortGoal21):
    """Change the desired wrist effort before the common force/motion solve."""

    def __init__(self, model):
        super().__init__(model)
        self.enable_aim = False
        self.aim = None

    def build_domain(self, **kwargs):
        self.aim = None
        if self.enable_aim:
            nominal = kwargs["nominal_reference"]
            current = kwargs["current_reference"].values[WRIST]
            step = np.asarray(HOLDING_REFERENCE_RATE21[WRIST]) * DECISION_PERIOD_S
            lower = np.maximum(kwargs["reference_lower21"][WRIST], current + kwargs["step_lower21"][WRIST])
            upper = np.minimum(kwargs["reference_upper21"][WRIST], current + kwargs["step_upper21"][WRIST])
            # If nominal is already outside the one-decision rate box, the
            # parent uses it as a raw objective. Preserve that behavior.
            if np.all(lower <= nominal.values[WRIST]) and np.all(nominal.values[WRIST] <= upper):
                axes = [kwargs["robot_axes25"].index(name) for name in AXES]
                q, v = np.asarray(kwargs["q25"])[axes], np.asarray(kwargs["v25"])[axes]
                if np.isfinite(q).all() and np.isfinite(v).all():
                    updated = nominal.values.copy()
                    updated[WRIST] = wrist_aim(nominal.values[WRIST], q, v, lower, upper, step)
                    self.aim = (updated[WRIST] - nominal.values[WRIST]).tolist()
                    kwargs["nominal_reference"] = Reference21(updated, nominal.time_s)
        return super().build_domain(**kwargs)


class AlignmentController(UnifiedTaskController):
    def __init__(self, method, context, limits, budget, gains, model):
        super().__init__(method, context, limits, budget, gains, model)
        self.native = AlignmentEffortGoal21(model)

    def _decide(self, frame, phase, mode, current, generation):
        normal = self.state.observed_normal
        touch = self.state.contact & (normal > .2)
        caps = np.minimum(self.limits, self.task_request["limits"])
        budget = min(self.budget, self.task_request["budget"])
        margin = self.g.force_margin
        guard = bool(self.state.valid.all() and touch[0] and np.count_nonzero(touch) >= 3
            and np.all(normal <= caps - margin) and normal.sum() <= budget - 2. * margin
            and np.isfinite(self.state.slip_speed) and self.state.slip_speed <= .004)
        self.native.enable_aim = mode == "holding" and START_S <= frame.time < END_S and guard
        super()._decide(frame, phase, mode, current, generation)
        evidence = dict(public_grasp_guard=guard, enabled=bool(self.native.enable_aim),
                        wrist_nominal_offset_rad=self.native.aim,
                        uses_original_reference_box=True, modifies_public_target=False)
        self.decision_evidence[-1]["wrist_alignment"] = evidence
        self.debug["wrist_alignment"] = evidence


class AlignmentSession(FrozenV21Session):
    def __init__(self, *, arm, case, seed):
        if arm not in ARMS or case not in CASES:
            raise ValueError("Unknown wrist-alignment assignment")
        self.alignment_arm = arm
        self.alignment_case = case
        self.alignment_finished = False
        super().__init__(method="ratio_effort", scenario="nominal", split="robustness",
                         motion_mode="transport", seed=seed,
                         task_policy="legacy" if arm == "ratio_pi" else "integrated",
                         receiver_policy="blind" if arm == "ratio_pi" else "coordinated")
        self.user_trajectory = trajectory(self.config.lift_target)
        if arm == "expanded":
            context = dict(config=dict(shape=self.shape), nominal_force_weights=self.calibration["nominal_force_weights"])
            self.controller = AlignmentController(self.method, context, self.limits, self.budget,
                                                  self.actual_ratio_task_gains, self.plant.model)

    def metadata(self):
        result = super().metadata()
        result.update(schema="held_wrist_alignment_trial_v1", benchmark_id="held_wrist_alignment_pilot",
                      duration=END_S, completed=self.alignment_finished, experiment_arm=self.alignment_arm,
                      alignment_case=self.alignment_case, scored_release=False,
                      observed_until_s=self.time, assessment_scope="held-object alignment up to the 12 s boundary",
                      pull_schedule_offline_only=dict(start_s=PULL_START_S, end_s=PULL_END_S, force_world_y_N=PULL_N)
                      if self.alignment_case == "midturn_pull" else None)
        return result


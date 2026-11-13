"""Run the existing ratio and macro controllers from their first observation.

Only the trial's start condition and public pickup task are new. No grip gain,
effort objective, contact search, release policy or physical model is retuned.
The task generator requests lift; it never writes a robot actuator reference.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import numpy as np
import research_imports

from .kernel import ColdStartKernel
from midterm_core_v1.runtime import CoreSession, source_files as core_sources
from v21_stable_handoff.runtime import ReleaseSession
from task_contract import build_task_request
from task_trajectory import unknown_future_point, AXES

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("baseline", "macro")
SCENES = ("nominal", "friction_drop", "fragile_conflict", "collision")
SEED = 619741


def source_files():
    files = dict(core_sources())
    for p in (ROOT / "cold_start_take_v1").glob("*.py"):
        files[p.relative_to(ROOT).as_posix()] = p
    p = ROOT / "v22_joint_task_fusion/numerics.py"
    files[p.relative_to(ROOT).as_posix()] = p
    return dict(sorted(files.items()))


def identity():
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in source_files().items()}


class DisabledPreparation:
    """Fail visibly if a legacy preparation controller is called by mistake."""
    def step(self, *args, **kwargs):
        raise RuntimeError("CommonPreparation is forbidden in the cold-start trial")


class ContactLiftTask:
    """Shared, causal task request from delayed public fingertip observations.

    These thresholds/ramp are the existing foundation task constants. Each
    method meets them using its own actions. There is no direct lift writer.
    """
    def __init__(self, config):
        self.contact_force = float(config.prelift_contact_force)
        self.contact_dwell = float(config.prelift_dwell)
        self.lift_duration = float(config.lift_duration)
        self.lift_target = float(config.lift_target)
        self.contact_since = None
        self.lift_started = None
        self.phase = "CLOSING"

    def update(self, obs):
        t = float(obs.time)
        if self.lift_started is None:
            valid = np.asarray(obs.valid, bool)
            force = np.asarray(obs.normal, float)
            contact = bool(np.all(valid & np.isfinite(force) & (force > self.contact_force)))
            if contact:
                if self.contact_since is None:
                    self.contact_since = t
                if t - self.contact_since >= self.contact_dwell - 1e-9:
                    self.lift_started = t
            else:
                self.contact_since = None
        if self.lift_started is not None:
            self.phase = "HOLDING" if t - self.lift_started >= self.lift_duration - 1e-9 else "LIFTING"

    def position_at(self, t):
        amount = 0. if self.lift_started is None else np.clip((t - self.lift_started) / self.lift_duration, 0., 1.)
        return [self.lift_target * float(amount), 0., 0.]

    def preview(self, t, transport):
        # The future contact-trigger time is unknown until contact dwell passes.
        if self.lift_started is None:
            return unknown_future_point(t, self.position_at(t))
        times = [t + i * .02 for i in range(4)]
        return dict(schema="public-user-motion-preview-v1", axes=list(AXES), time_s=t,
            requested_times_s=times, trajectory_times_s=[min(x, 12.) for x in times],
            motion_targets=[transport.position_at(x) if x >= 9.3 else self.position_at(x) for x in times],
            future_reference_available=True, available_through_s=12.,
            active_user_motion=[x < 12. for x in times],
            terminal_semantics="continuous pre-release user trajectory; contact gate already observed",
            information="public contact-triggered task and user motion; no object or future disturbance state")

    def specification(self):
        return dict(contact_force_N=self.contact_force, all_five_valid_contacts=True,
                    contact_dwell_s=self.contact_dwell, lift_duration_s=self.lift_duration,
                    lift_target_m=self.lift_target, actual_lift_request_start_s=self.lift_started,
                    observation="same delayed noisy fingertip observations supplied to both controllers",
                    role="public task request only; no grip or motion actuator writes")


class ColdStartMixin(ColdStartKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.mode != "scheduled" or self.motion_mode != "transport":
            raise ValueError("This pilot admits scheduled transport only")
        self.preparation = DisabledPreparation()
        self.pickup_task = ContactLiftTask(self.config)
        self.local_lower = self.plant.closure_min.copy()
        self.local_upper = self.plant.closure_max.copy()
        # This is an evaluation anchor, never sent to a controller.
        self.reference_relative = self.last_truth["relative"].copy()
        self.initial_reference21 = self.plant.reference_snapshot().values.copy()
        self.initial_qpos = self.plant.data.qpos.copy()
        self.initial_qvel = self.plant.data.qvel.copy()
        self.events = []
        self.event("独立冷启动：从张手初态由各自策略控制，未执行共同预抓。")
        self.research_sources = identity()
        self.observer_warmup_s = 0.

    def _pickup_motion_request(self, t, obs):
        previous = self.pickup_task.phase
        self.pickup_task.update(obs)
        if self.plant.release_started is None:
            self.plant.phase = self.pickup_task.phase
        if previous != self.pickup_task.phase:
            self.event("各自接触状态触发任务阶段：" + self.pickup_task.phase)
        return self.pickup_task.position_at(t)

    def _public_task(self, t, mode, target, executed, deadline, release_active):
        if release_active:
            preview = unknown_future_point(t, target)
        elif t < 9.3 - 1e-9:
            preview = self.pickup_task.preview(t, self.user_trajectory)
        else:
            preview = self.user_trajectory.preview(t)
        return build_task_request(t, mode, target, executed, deadline, self.limits, self.budget,
            tracking_tolerances=[.01, .015, .015], motion_preview=preview,
            motion_reference_bounds=self.motion_reference_bounds,
            sensor_characteristics=self.sensor_characteristics)

    def snapshot(self):
        value = super().snapshot()
        value.update(controller_active=bool(self.trace), handoff_passed=None,
            domain_label="从张手初态开始的独立全流程试验；抓取能力另行物理评价",
            message=self.failure_reason or "各自策略从 0 秒接入；无共同预抓。")
        return value

    def metadata(self):
        meta = super().metadata()
        meta.update(schema="cold_start_take_trial_v1", benchmark_id="cold_start_take_pilot_v1",
            experiment_arm=self.cold_arm, research_phase="development",
            controller_scope="existing controller active from first observation through pickup, transport and release",
            control_start_s=0., common_preparation_used=False, handoff_time=None, handoff_passed=None,
            handoff_alpha=None, preparation_failure=None,
            reference_kind="initial open-hand relative position; offline only; not a prepared grasp",
            original_preparation_certificate_scope="unchanged physical carrier only; not validation of this pickup algorithm",
            pickup_task=self.pickup_task.specification(),
            initial_reference21=self.initial_reference21.tolist(),
            initial_qpos=self.initial_qpos.tolist(), initial_qvel=self.initial_qvel.tolist(),
            pickup_deadline_s=7.5, transport_start_s=9.3, release_request_s=12.,
            pickup_force_prior="existing ratio-intent acquisition reference shared as a prior; independent native control actions",
            gains_changed=False, original_model_changed=False,
            original_results_reused_as_new_trials=False,
            old_post_preparation_success_labels_applicable=False)
        meta["task_contract"]["pickup_task"] = self.pickup_task.specification()
        meta["task_contract"]["trajectory_before_9_3_s"] = "contact-triggered pickup task; see pickup_task and per-tick requests"
        meta.update(self.run_provenance)
        meta['admission_sha256']=None
        meta['original_preparation_certificate_scope']='not included; exported parameters do not certify this environment'
        meta['execution_distribution']='public_source_release'
        return meta


class ColdStartBaseline(ColdStartMixin, ReleaseSession):
    cold_arm = "baseline"
    def __init__(self, **kwargs):
        super().__init__(method="ratio_effort", task_policy="legacy", receiver_policy="blind", **kwargs)


class ColdStartMacro(ColdStartMixin, CoreSession):
    cold_arm = "macro"
    def __init__(self, **kwargs):
        super().__init__(arm="core", **kwargs)


def make_session(arm, scene="nominal", seed=SEED, *, runtime_config=None):
    if arm not in ARMS or scene not in SCENES:
        raise ValueError("Unregistered pilot condition")
    cls = ColdStartBaseline if arm == "baseline" else ColdStartMacro
    return cls(scenario=scene, seed=seed, split="robustness", motion_mode="transport", runtime_config=runtime_config)

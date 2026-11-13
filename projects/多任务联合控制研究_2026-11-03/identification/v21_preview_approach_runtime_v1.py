"""Independent preview-approach plus motion-reserved V21 carrier.

SEARCH shapes only the velocity request to the existing lift governor.
TRANSFER retains the already tested motion-reserved action path. The normal
lifecycle still makes exactly one native reference commit per control tick.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from identification.v21_motion_reserved_runtime_v2 import (
    MotionReservedSession, MotionReservedSupervisor, source_files as parent_source_files,
)
from identification.v21_public_support_preview_v1 import (
    DeclaredShapeTemplate, StaticReceiverTop, SupportPreview,
)
from identification.v21_support_approach_schedule_v1 import ApproachSchedule, READ_DELAY_S, finite
from v16_release_geometry.transfer_observer import TransferObserver
from v16b_release_goal.goal import release_clearance_goal

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "v21_preview_approach_motion_reserved_runtime_v1"
EXTRA_SOURCES = (
    "identification/v21_preview_approach_runtime_v1.py",
    "identification/v21_public_support_preview_v1.py",
    "identification/v21_support_approach_schedule_v1.py",
)
TEMPLATE = DeclaredShapeTemplate("capsule", .028, .047, "same public nominal template as frozen V21")


def source_files(methods=None):
    files = dict(parent_source_files(methods))
    files.update({name:ROOT / name for name in EXTRA_SOURCES})
    return dict(sorted(files.items()))


def source_identity(methods=None):
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files(methods).items()}


class PreviewApproachGovernor:
    def __init__(self, delegate):
        self.delegate = delegate
        self.schedule = ApproachSchedule()
        self.prepared = None

    @property
    def tau_s(self):
        return self.delegate.tau_s

    def prepare(self, phase, now, preview):
        # Values are derived from the current public frame, not a second read.
        if not finite(now):
            self.prepared = None
            return
        if isinstance(preview, dict) and preview.get("available") is True:
            read, acquired = preview.get("read_s"), preview.get("acquisition_s")
            if (not finite(read) or not finite(acquired) or read != now
                    or abs(now - acquired - READ_DELAY_S) > 1e-9):
                preview = dict(available=False, reason="preview_acquisition_clock_mismatch", read_s=float(now))
        value = (phase, float(now), deepcopy(preview))
        if self.prepared is not None and now < self.prepared[1]:
            raise ValueError("Rewound approach read")
        if self.prepared is not None and now == self.prepared[1] and value != self.prepared:
            raise ValueError("Changed approach observation on the same tick")
        self.prepared = value

    def step(self, **kwargs):
        now = kwargs["time_s"]
        original = kwargs["requested_velocity_m_s"]
        prepared = self.prepared
        if prepared is None or prepared[1] != now:
            return self.delegate.step(**kwargs)
        phase, _, public = prepared
        if phase != "SEARCH":
            return self.delegate.step(**kwargs)
        tau = getattr(self.delegate, "tau_s", None)
        budget = getattr(self.delegate, "recovery_budget_s", None)
        rate = kwargs["reference_rate_upper_m_s"]
        deadline = kwargs.get("deadline_s")
        if (not all(finite(x) for x in (tau, budget, rate, deadline))
                or tau <= 0. or budget <= 0. or rate < 0.
                or kwargs.get("brake_requested", False) or now >= deadline):
            return self.delegate.step(**kwargs)
        envelope = rate * budget / tau
        shaped = self.schedule.update(now=now, phase=phase, preview=public,
            original_request_m_s=original, velocity_envelope_m_s=envelope, deadline_s=deadline)
        result = self.delegate.step(**{**kwargs, "requested_velocity_m_s":shaped["requested_velocity_m_s"]})
        telemetry = dict(shaped, original_requested_velocity_m_s=original,
                         request_was_shaped=shaped["requested_velocity_m_s"] != original,
                         preview=deepcopy(public), native_commit_owned_by_parent=True)
        return replace(result, diagnostics={**result.diagnostics, "approach_preview":telemetry})


def declared_horizontal_receiver(model, receiver_mount):
    """Use the declared device geometry only, with no dynamic data object."""
    if receiver_mount.geom_name != "support":
        return None
    geom = model.geom(receiver_mount.geom_name)
    i = int(geom.id)
    if (int(model.geom_bodyid[i]) != 0 or int(model.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_BOX)
            or not np.array_equal(geom.quat, [1., 0., 0., 0.])
            or not np.array_equal(receiver_mount.origin_in_geom_m, [0., 0., geom.size[2]])):
        return None
    return StaticReceiverTop(tuple(geom.pos[:2]), tuple(geom.size[:2]), float(geom.pos[2] + geom.size[2]))


class ApproachMotionSupervisor(MotionReservedSupervisor):
    def __init__(self, entry, *, approach_mount, **kwargs):
        super().__init__(entry, **kwargs)
        self.approach_preview = SupportPreview(approach_mount, TEMPLATE)
        self.governor = PreviewApproachGovernor(self.governor)

    def observe(self, *, frame, transfer_inputs, current21):
        super().observe(frame=frame, transfer_inputs=transfer_inputs, current21=current21)
        outcome = None
        if self.phase == "SEARCH":
            packet = {k:getattr(frame, k, None) for k in (
                "time", "sample_time", "skin_patches", "sensor_characteristics", "lift_velocity_m_s")}
            try:
                outcome = self.approach_preview.read(packet)
            except (ValueError, TypeError, KeyError, FloatingPointError, np.linalg.LinAlgError) as exc:
                outcome = dict(available=False, reason=type(exc).__name__ + ": " + str(exc), read_s=float(frame.time))
        self.governor.prepare(self.phase, float(frame.time), outcome)
        return deepcopy(self.public_state)

    def reconcile(self, receipt):
        super().reconcile(receipt)
        record = self.decision_evidence[-1]
        preview = record["governor"].get("approach_preview")
        if preview is None:
            return
        reference = np.asarray(record["proposed_reference21"])
        aligned = (receipt.requested.time_s == receipt.executed.time_s == receipt.engine.time_s == record["time_s"]
                   and receipt.mode == "release" and receipt.mode_generation == record["mode_generation"])
        encoded = self.native.actor.encode(receipt.executed)
        engine_matches = (receipt.engine.axes == encoded.axes and receipt.engine.lift_bias_m == encoded.lift_bias_m
                          and receipt.engine.values.tobytes() == encoded.values.tobytes())
        unchanged = (aligned and engine_matches and np.array_equal(receipt.requested.values, reference)
                     and np.array_equal(receipt.executed.values, reference)
                     and not receipt.execution_changed and not receipt.stale_plan_replaced)
        record["approach_execution"] = dict(same_clock_mode_and_generation=aligned,
                    engine_matches_executed_reference=engine_matches,
                    requested_lift_m=float(receipt.requested.values[18]), actual_lift_m=float(receipt.executed.values[18]),
                    shaped_request_retained=bool(preview["request_was_shaped"] and unchanged), retried=False)


class PreviewApproachSession(MotionReservedSession):
    """Opt-in combined carrier, distinct from the registered V2 pilot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.research_sources = source_identity([self.method])

    def _lifecycle_step(self, obs):
        eligible = (self.v16_release_policy == "coordinated" and self.v16_supervisor is None
                    and self.handoff_attempted and self.handoff_passed and not self.manual_release
                    and self.shape == "capsule")
        if eligible:
            entry = self.plant.joint_release_entry
            if entry is None:
                raise ValueError("Actual release must preempt holding")
            mount = declared_horizontal_receiver(self.plant.model, self.receiver_mount)
            if mount is not None and abs(entry.started_s - 12.) <= 1e-9:
                lower, upper = self.plant.reference_authority21(include_mode=False)
                self.release_clearance_goal = release_clearance_goal(
                    entry, preparation_open18=np.concatenate(self.plant.open_pose),
                    mechanical_lower21=lower, mechanical_upper21=upper)
                adapter = self.release_adapter
                prior = adapter.begin_release(entry.started_s)
                observer = TransferObserver(request_time_s=entry.started_s,
                    receiver_half_size_xy_m=adapter.receiver_half_size_xy_m,
                    receiver_up_axis3=adapter.receiver_up_axis3, prior=prior)
                self.v16_supervisor = ApproachMotionSupervisor(
                    entry, approach_mount=mount, observer=observer, robot_model=self.plant.model,
                    open18=self.release_clearance_goal.reference18, limits5=self.controller.limits,
                    budget=self.controller.budget, force_margin=self.controller.g.force_margin)
                self.release_driver = self.v16_supervisor
                self.plant.bind_supervisor(self.v16_supervisor)
        return super()._lifecycle_step(obs)

    def metadata(self):
        meta = super().metadata()
        meta.update(schema=SCHEMA, benchmark_id="v21_preview_approach_motion_reserved_development",
                    decision_algorithm=SCHEMA,
                    preview_approach=dict(scope="scheduled capsule SEARCH with declared horizontal receiver",
                        support_pose_at_acquisition_only=True, velocity_request_only=True,
                        same_native_commit=True, new_physical_claim=False))
        return meta

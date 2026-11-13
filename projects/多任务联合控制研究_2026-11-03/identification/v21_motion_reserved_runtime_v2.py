"""Opt-in V21 carrier for motion-reserved transfer, with one native commit.

The frozen action provides the common current task domain and fallback. This
subclass selects a different reference only before the original lifecycle's
single commit. It does not start a run or write to the plant.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np

from identification.v21_motion_reserved_observation_v2 import SupportObservation
from identification.v21_motion_reserved_transfer_v2 import solve_reserved_transfer
from identification.v21_receiver_transfer_evidence_v1 import number
from identification.v21_support_conditioned_exit_v1 import AXES, retained_domain, snapshot
from v12_native.controller import plain
from v13_a1.codec import Reference21
from v16_release_geometry.transfer_observer import TransferObserver
from v16b_release_goal.goal import release_clearance_goal
from v21_stable_handoff.runtime import ReleaseSession as FrozenV21Session
from v21_stable_handoff.runtime import source_files as frozen_source_files
from v21_stable_handoff.supervisor import ReleaseSupervisor as FrozenV21Supervisor
from v22_joint_task_fusion.numerics import numerical_identity

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "v21_motion_reserved_transfer_runtime_v2"
SOURCE_NAMES = (
    "identification/v21_motion_reserved_observation_v2.py",
    "identification/v21_motion_reserved_runtime_v2.py",
    "identification/v21_motion_reserved_transfer_v2.py",
    "identification/v21_receiver_transfer_evidence_v1.py",
    "identification/v21_support_conditioned_exit_v1.py",
    "v22_joint_task_fusion/numerics.py",
    "v23_task_execution/finite_solver.py",
)


def source_files(methods=None):
    files = dict(frozen_source_files(methods))
    files.update({name:ROOT / name for name in SOURCE_NAMES})
    return dict(sorted(files.items()))


def source_identity(methods=None):
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files(methods).items()}


@dataclass(frozen=True)
class ReferenceCheck:
    problem: object
    current: np.ndarray
    units: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    correction: np.ndarray
    stability: np.ndarray
    protected_motion: np.ndarray
    fraction: float
    capacity_m: float

    def inspect(self, values):
        """Check the actual encoded-domain value, even after the sole commit."""
        u = np.asarray(values, dtype=float)
        if u.shape != (AXES,) or not np.isfinite(u).all():
            return dict(satisfies=False, reason="invalid_reference_shape_or_value")
        problem = self.problem
        z = np.r_[(u - self.current) / self.units, problem.lower[AXES:]]
        hard, rhs = retained_domain(problem, self.stability, self.fraction)
        violations = dict(
            absolute_reference=float(max(0., np.max(self.lower - u), np.max(u - self.upper))),
            frozen_domain=problem.violation(z),
            inward_from_correction=float(np.max(problem.normal @ (self.correction - z), initial=0.)),
            outward_domain=float(np.max(hard @ z - rhs, initial=0.)),
            protected_motion=float(np.max(abs(problem.motion_level.matrix @ z - self.protected_motion), initial=0.)),
            overall_outward_m=float(max(0., problem.work_row @ (z - self.stability)[:AXES]
                                        - self.fraction * self.capacity_m)),
        )
        accepted = (violations["absolute_reference"] == 0. and violations["frozen_domain"] <= 1e-8
                    and violations["inward_from_correction"] <= 1e-8 and violations["outward_domain"] <= 1e-8
                    and violations["protected_motion"] <= 1e-9 and violations["overall_outward_m"] <= 1e-10)
        return dict(satisfies=accepted, violations=violations)


class MotionReservedSupervisor(FrozenV21Supervisor):
    def __init__(self, entry, *, observer, robot_model, **kwargs):
        super().__init__(entry, observer=observer, robot_model=robot_model, **kwargs)
        self.support_observation = SupportObservation(robot_model, observer.receiver_up, observer.prior, entry.started_s)
        self.support_state = None
        self._pending_motion_reserved = None

    def observe(self, *, frame, transfer_inputs, current21):
        super().observe(frame=frame, transfer_inputs=transfer_inputs, current21=current21)
        try:
            read_time = float(frame.time)
            delivery = transfer_inputs.get("time_s")
            if not number(delivery) or abs(delivery - read_time) > 1e-9:
                self.support_observation.update(self.phase, {"time_s":read_time})
                raise ValueError("Support and command clocks differ")
            support = self.support_observation.update(self.phase, transfer_inputs)
            self.support_state = dict(support, read_time_s=read_time)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self.support_state = dict(robot_direct_contact_excluded=False,
                                      unavailable_reason=type(exc).__name__ + ": " + str(exc))
        self.public_state["motion_reserved_support"] = deepcopy(self.support_state)
        return deepcopy(self.public_state)

    def action(self, *, frame, adapter, current21, lower21, upper21, generation):
        if self._pending_motion_reserved is not None:
            raise RuntimeError("A selected reference still needs its single execution receipt")
        reference, debug = super().action(frame=frame, adapter=adapter, current21=current21,
                                          lower21=lower21, upper21=upper21, generation=generation)
        if self.phase != "TRANSFER":
            return reference, debug
        record = self.decision_evidence[-1]
        began = time.perf_counter_ns()
        result = {"status":"frozen_v21_reference_selected", "reason":"support_topology_unavailable"}
        state = self.support_state
        if (state is not None and state.get("robot_direct_contact_excluded") is True
                and state.get("read_time_s") != reference.time_s):
            result["reason"] = "support_observation_clock_mismatch"
        elif state is not None and state.get("robot_direct_contact_excluded") is True:
            hierarchy = record["hierarchy"]
            diag = None if hierarchy is None else hierarchy["diagnostics"]
            if diag is None or "second_stage_hard_matrix" not in diag:
                result["reason"] = "frozen_post_correction_domain_unavailable"
            else:
                try:
                    selected = self._select(frame, generation, reference, record, diag,
                                            state["sustained_proxy_fraction"])
                    if selected["status"] == "solved":
                        reference = selected.pop("reference")
                        result = selected
                    else:
                        result.update(reason=selected["status"], solver=selected)
                except (ValueError, TypeError, KeyError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    result.update(reason="new_reference_rejected", detail=type(exc).__name__ + ": " + str(exc))
        added_ms = (time.perf_counter_ns() - began) / 1e6
        result.update(added_solver_ms=added_ms, requested_reference21=reference.values.tolist())
        record["motion_reserved"] = result
        record["solver_ms"] += added_ms
        self.debug["motion_reserved"] = deepcopy(result)
        self.debug["solver_timing_ms"]["whole_decision"] = record["solver_ms"]
        if result["status"] == "solved":
            self.debug.update(architecture=SCHEMA, geometry_proposal_accepted=True, fallback=None)
        return reference, deepcopy(self.debug)

    def _select(self, frame, generation, original, record, diag, fraction):
        now = original.time_s
        if diag["time_s"] != now or record["mode_generation"] != generation:
            raise ValueError("Frozen domain is from a different reference generation or time")
        problem = snapshot(diag)
        if problem.motion_level is None or not len(problem.normal):
            return dict(status="no_current_motion_and_contact_domain")
        result = solve_reserved_transfer(problem, fraction)
        if result["status"] != "solved":
            return result
        current = np.asarray(record["current_reference21"], dtype=float)
        units = np.asarray(diag["reference_step_units21"], dtype=float)
        lower = np.asarray(record["actual_reference_lower21"], dtype=float)
        upper = np.asarray(record["actual_reference_upper21"], dtype=float)
        raw = current + np.asarray(result["final_reference_normalized"]) * units
        proposed = np.minimum(upper, np.maximum(lower, raw))
        rounding = 64 * np.finfo(float).eps * np.maximum(1., np.maximum(abs(raw), abs(proposed)))
        if np.any(abs(proposed - raw) > rounding):
            raise ValueError("Candidate requires more than frozen reference rounding")
        check = ReferenceCheck(
            problem, current, units, lower, upper,
            np.r_[result["correction_reference_normalized"], problem.lower[AXES:]],
            np.r_[result["stability_reference_normalized"], problem.lower[AXES:]],
            np.asarray(result["protected_motion_vector"]), result["fraction"], result["optional_capacity_proxy_m"])
        certificate = check.inspect(proposed)
        if not certificate["satisfies"]:
            raise ValueError("Emitted reference did not preserve the new domain and motion optimum")
        reference = Reference21(proposed, now)
        q, v = frame.robot_position25, frame.robot_velocity25
        current_effort = self.native.servo.evaluate(q, v, frame.robot_actuator_target21, time_s=now)
        effort = self.native.servo.evaluate(q, v, self.native.actor.encode(reference).values, time_s=now)
        if any(x.status != "KNOWN_ROBOT_ACTUATION_ALGEBRA" for x in (current_effort, effort)):
            raise ValueError("Native current-configuration actuation value unavailable")
        # Keep the frozen proposal evidence separate from the selected proposal.
        original_fields = ("proposed_reference21", "geometry_proposal_accepted", "fallback",
                           "emitted_hierarchy_score", "native_current", "native_proposed",
                           "rounding_projection21", "numerical_error", "task_role", "hierarchy")
        record["frozen_v21_proposal"] = {k:record[k] for k in original_fields}
        record.update(task_role=SCHEMA, proposed_reference21=proposed.tolist(),
                      geometry_proposal_accepted=True, fallback=None, numerical_error=None,
                      emitted_hierarchy_score=certificate,
                      hierarchy=None, hierarchy_scope="new_motion_reserved_certificate_in_motion_reserved",
                      native_current=plain(vars(current_effort)), native_proposed=plain(vars(effort)),
                      rounding_projection21=(proposed - raw).tolist())
        self._pending_motion_reserved = (check, reference, int(generation), record["decision_id"])
        task_scope = ("mandatory_and_full_optional" if result["full_optional_tasks"]
                      else "mandatory_and_certified_optional_prefix" if result["optional_task_prefix"]
                      else "mandatory_only")
        return dict(status="solved", reference=reference, reference_certificate=certificate,
                    accepted_task_scope=task_scope, solver=result)

    def reconcile(self, receipt):
        super().reconcile(receipt)
        pending = self._pending_motion_reserved
        if pending is None:
            return
        self._pending_motion_reserved = None
        check, expected, generation, decision = pending
        record = self.decision_evidence[-1]
        if record["decision_id"] != decision:
            raise RuntimeError("Receipt does not correspond to the pending decision")
        stamps_ok = (receipt.requested.time_s == receipt.executed.time_s == expected.time_s
                     and receipt.mode_generation == generation and receipt.mode == "release")
        requested_ok = np.array_equal(receipt.requested.values, expected.values)
        actual = check.inspect(receipt.executed.values)
        preserved = bool(stamps_ok and requested_ok and actual["satisfies"]
                         and not receipt.execution_changed and not receipt.stale_plan_replaced
                         and np.array_equal(receipt.executed.values, expected.values))
        outcome = dict(same_snapshot_and_generation=stamps_ok, request_matches=requested_ok,
                       selected_reference_certificate_preserved=preserved,
                       actual_reference_check=actual, retried=False)
        record["motion_reserved"]["execution"] = outcome
        self.debug["motion_reserved"]["execution"] = deepcopy(outcome)


class MotionReservedSession(FrozenV21Session):
    """Opt-in independent scheduled carrier. Other V21 entry points are intact."""

    def __init__(self, *args, **kwargs):
        if kwargs.get("release_variant", "v21") != "v21":
            raise ValueError("The motion-reserved carrier requires frozen V21 as its parent")
        super().__init__(*args, **kwargs)
        if (self.method != "ratio_effort" or self.task_policy != "integrated"
                or self.v16_release_policy != "coordinated" or self.mode != "scheduled"):
            raise ValueError("This carrier requires the registered scheduled V21 strategy")
        self.research_sources = source_identity([self.method])
        self.motion_reserved_numerics = numerical_identity()

    def _lifecycle_step(self, obs):
        eligible = (self.v16_release_policy == "coordinated" and self.v16_supervisor is None
                    and self.handoff_attempted and self.handoff_passed and not self.manual_release)
        if eligible:
            entry = self.plant.joint_release_entry
            if entry is None:
                raise ValueError("Actual release must preempt holding")
            if abs(entry.started_s - 12.) > 1e-9:
                return super()._lifecycle_step(obs)
            lower, upper = self.plant.reference_authority21(include_mode=False)
            self.release_clearance_goal = release_clearance_goal(
                entry, preparation_open18=np.concatenate(self.plant.open_pose),
                mechanical_lower21=lower, mechanical_upper21=upper)
            adapter = self.release_adapter
            prior = adapter.begin_release(entry.started_s)
            observer = TransferObserver(request_time_s=entry.started_s,
                receiver_half_size_xy_m=adapter.receiver_half_size_xy_m,
                receiver_up_axis3=adapter.receiver_up_axis3, prior=prior)
            self.v16_supervisor = MotionReservedSupervisor(
                entry, observer=observer, robot_model=self.plant.model,
                open18=self.release_clearance_goal.reference18, limits5=self.controller.limits,
                budget=self.controller.budget, force_margin=self.controller.g.force_margin)
            self.release_driver = self.v16_supervisor
            self.plant.bind_supervisor(self.v16_supervisor)
        return super()._lifecycle_step(obs)

    def metadata(self):
        meta = super().metadata()
        meta.update(schema=SCHEMA, benchmark_id="v21_motion_reserved_transfer_development",
                    decision_algorithm=SCHEMA, numerical_identity=self.motion_reserved_numerics,
                    frozen_v21_source_sha256={name:self.research_sources[name]
                                             for name in frozen_source_files([self.method])},
                    motion_reserved=dict(scope="admitted TRANSFER, same V21 correction/motion domain",
                                         mandatory_stability_reserved=True, new_physical_claim=False))
        return meta

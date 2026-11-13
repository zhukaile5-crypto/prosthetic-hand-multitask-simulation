"""Exercise the runtime adapter against one frozen decision without a plant."""

from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONTEXT = ("collision", "hold", 619741)
ANCHOR = "first_sustained_support"
SNAPSHOT = ROOT / "review/v21_motion_reserved_transfer_snapshot_v2.json"
OUTPUT = ROOT / "review/v21_motion_reserved_runtime_software_v2.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fixture():
    import replay_v21_support_conditioned_exit_v1 as archive

    snapshot = archive.read(SNAPSHOT)
    expected = next(row for row in snapshot["rows"]
                    if tuple(row["context"]) == CONTEXT and row["label"] == ANCHOR)
    assert expected["status"] == "solved"
    evidence = archive.read(archive.verify(archive.SUPPORT, archive.SUPPORT_SHA))
    auxiliary = archive.read(archive.verify(archive.AUX, archive.AUX_SHA))
    case = next(row for row in evidence["cases"] if tuple(row["context"]) == CONTEXT)
    meta = next(row for row in auxiliary["records"] if tuple(row["context_key"]) == CONTEXT)
    label, reading, decision, executed, digest = next(
        item for item in archive.inputs(case, meta) if item[0] == ANCHOR)
    assert decision["time_s"] == reading["time_s"] == expected["time_s"]
    assert digest == expected["release_log_sha256"]
    assert np.array_equal(executed, decision["execution_receipt"]["actual_reference21"])
    assert .25 + .75*reading["sustained_proxy_fraction"] == expected["result"]["fraction"]
    trace_path = Path(meta["run"]) / "trace.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        public = json.loads(trace["observation_json"][archive.tick(decision["time_s"])])
    assert public["time"] == public["robot_control_time_s"] == decision["time_s"]
    assert np.array_equal(public["robot_reference21"], decision["current_reference21"])
    pins = {path.relative_to(ROOT).as_posix(): sha(path)
            for path in (SNAPSHOT, archive.SUPPORT, archive.AUX)}
    pins[trace_path.relative_to(ROOT).as_posix()] = meta["input_sha256"]["trace.npz"]
    pins[(trace_path.parent / "release_decisions.jsonl.gz").relative_to(ROOT).as_posix()] = digest
    return public, reading, case["topology"], decision, expected, pins


def raw_checks(emitted, row, solution):
    """Independently rebuild the certificate from the public-unit rows."""
    diag = row["hierarchy"]["diagnostics"]
    current = np.asarray(row["current_reference21"])
    step = emitted - current
    units = np.asarray(diag["reference_step_units21"])
    lower = np.asarray(diag["second_stage_lower"])
    upper = np.asarray(diag["second_stage_upper"])
    normalized = np.r_[step / units, lower[21:]]
    raw_normal = np.asarray(diag["normal_matrix_N_Px21"])
    report_units = np.asarray(diag["normal_effective_report_units_P_m"])
    active = np.asarray(diag["normal_row_units_P_m"]) > 0.
    outward = -raw_normal[active] * units / report_units[active, None]
    correction = np.asarray(solution["correction_reference_normalized"])
    stability = np.asarray(solution["stability_reference_normalized"])
    actual = normalized[:21]
    extra = outward @ (actual - stability)
    ceiling = np.maximum(0., np.maximum(outward, 0.) @ upper[:21]
                         + np.minimum(outward, 0.) @ lower[:21] - outward @ stability)
    fraction = solution["fraction"]
    motion = diag["motion_task"]
    motion_m = np.asarray(motion["matrix_Mx21"]) @ step
    expected_motion_m = np.asarray(solution["protected_motion_vector"]) * motion["effective_row_units_M"]
    motion_error_m = abs(motion_m - expected_motion_m)
    original_hard = np.asarray(diag["second_stage_hard_matrix"])
    original_upper = np.asarray(diag["second_stage_hard_upper"])
    optional_m = float(np.asarray(diag["scalar_normal_work_row"]) @ (actual - stability))
    return dict(
        absolute_reference_inside=bool(np.all(emitted >= row["actual_reference_lower21"])
                                       and np.all(emitted <= row["actual_reference_upper21"])),
        normalized_box_error=float(max(0., np.max(lower-normalized), np.max(normalized-upper))),
        original_hard_error=float(np.max(original_hard @ normalized-original_upper, initial=0.)),
        original_raw_normal_error=float(np.max((raw_normal @ step-diag["normal_one_sided_bound_P_m"])
                                              / report_units, initial=0.)),
        inward_from_correction=float(np.max(outward @ (correction-actual), initial=0.)),
        inward_from_stability=float(np.max(-extra, initial=0.)),
        per_row_pace_error=float(np.max(extra-fraction*ceiling, initial=0.)),
        optional_withdrawal_m=optional_m,
        optional_capacity_ceiling_m=fraction*solution["optional_capacity_proxy_m"],
        maximum_motion_drift_m=float(np.max(motion_error_m, initial=0.)),
        maximum_motion_drift_normalized=float(np.max(motion_error_m/motion["effective_row_units_M"], initial=0.)))


def main(output=OUTPUT):
    if output.exists():
        raise FileExistsError(output)
    from identification import v21_motion_reserved_runtime_v2 as runtime
    from v13_a1.codec import EnginePort21, Reference21
    from v13_joint.plant import ExecutionReceipt21
    import mujoco

    before = runtime.source_identity()
    checks = []
    calls_to_physics = []
    error = None
    pins = {}

    def check(name, passed, detail=None):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))

    def deny(*args, **kwargs):
        calls_to_physics.append("physical operation attempted")
        raise AssertionError(calls_to_physics[-1])

    try:
        names = ("MjModel", "MjData", "mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_contactForce")
        with patch.multiple(mujoco, **{name: deny for name in names if hasattr(mujoco, name)}):
            public, reading, topology, archived, expected, pins = fixture()
            now, generation = archived["time_s"], archived["mode_generation"]
            nominal = Reference21(archived["proposed_reference21"], now)
            current = np.asarray(archived["current_reference21"])
            lower, upper = (np.asarray(public[f"reference_{side}21"]) for side in ("lower", "upper"))
            state = {**topology, **reading, "read_time_s": now}
            frame = SimpleNamespace(**public)
            engine_bias = public["robot_actuator_target21"][18] - current[18]

            def encode(reference):
                values = reference.values.copy()
                values[18] += engine_bias
                return EnginePort21(values, reference.time_s, engine_bias)

            def make_supervisor(phase="TRANSFER", *, admitted=True, read_time=now):
                obj = object.__new__(runtime.MotionReservedSupervisor)
                obj.phase = phase
                obj.decision_evidence = []
                obj.debug = {}
                obj.support_state = dict(state, robot_direct_contact_excluded=admitted, read_time_s=read_time)
                obj._pending_motion_reserved = None
                obj._fixture_actions = []
                obj._fixture_receipts = []
                obj._fixture_values = []

                def evaluate(q, v, target, *, time_s):
                    obj._fixture_values.append((q, v, np.asarray(target).copy(), time_s))
                    return SimpleNamespace(status="KNOWN_ROBOT_ACTUATION_ALGEBRA", time_s=time_s,
                                           commanded_target21=np.asarray(target).tolist())

                obj.native = SimpleNamespace(actor=SimpleNamespace(encode=encode),
                                             servo=SimpleNamespace(evaluate=evaluate))
                return obj

            def frozen_action(self, **kwargs):
                self._fixture_actions.append(kwargs)
                record = deepcopy(archived)
                record["phase"] = self.phase
                record.pop("execution_receipt", None)
                self.decision_evidence.append(record)
                self.debug = dict(architecture="frozen-fixture", phase=self.phase,
                                  solver_timing_ms={"whole_decision": record["solver_ms"]})
                return nominal, deepcopy(self.debug)

            def frozen_reconcile(self, receipt):
                self._fixture_receipts.append(receipt)
                self.decision_evidence[-1]["execution_receipt"] = dict(
                    actual_reference21=receipt.executed.values.tolist(),
                    stale_plan_replaced=receipt.stale_plan_replaced,
                    execution_changed=receipt.execution_changed)

            def action(obj):
                return obj.action(frame=frame, adapter=None, current21=current,
                                  lower21=lower, upper21=upper, generation=generation)

            with patch.object(runtime.FrozenV21Supervisor, "action", frozen_action), \
                    patch.object(runtime.FrozenV21Supervisor, "reconcile", frozen_reconcile):
                for phase in ("SEARCH", "CAPTURE", "SEPARATE", "UNSUPPORTED_EXIT"):
                    obj = make_supervisor(phase)
                    with patch.object(runtime, "solve_reserved_transfer", side_effect=deny):
                        ref, debug = action(obj)
                    check(f"{phase} delegates exactly once without replacement",
                          ref is nominal and debug["phase"] == phase and len(obj._fixture_actions) == 1
                          and "motion_reserved" not in obj.decision_evidence[-1]
                          and not obj._fixture_values and obj._pending_motion_reserved is None)

                obj = make_supervisor(admitted=False)
                with patch.object(runtime, "solve_reserved_transfer", side_effect=deny):
                    ref, debug = action(obj)
                check("unsupported receiver topology keeps the original proposal",
                      ref is nominal and debug["motion_reserved"]["reason"] == "support_topology_unavailable"
                      and len(obj._fixture_actions) == 1 and not obj._fixture_values
                      and obj._pending_motion_reserved is None)

                obj = make_supervisor(read_time=now-.005)
                with patch.object(runtime, "solve_reserved_transfer", side_effect=deny):
                    ref, debug = action(obj)
                check("stale support cannot enter the new solver",
                      ref is nominal and debug["motion_reserved"]["status"] == "frozen_v21_reference_selected"
                      and len(obj._fixture_actions) == 1 and not obj._fixture_values
                      and obj._pending_motion_reserved is None, debug["motion_reserved"]["reason"])

                obj = make_supervisor()
                with patch.object(runtime, "solve_reserved_transfer", return_value={"status": "correction_not_certified"}) as fail:
                    ref, debug = action(obj)
                check("uncertified solver result falls back once without native replacement",
                      ref is nominal and debug["motion_reserved"]["reason"] == "correction_not_certified"
                      and len(obj._fixture_actions) == fail.call_count == 1
                      and not obj._fixture_values and obj._pending_motion_reserved is None)

                obj = make_supervisor()
                ref, debug = action(obj)
                entry = obj.decision_evidence[-1]
                selected = entry["motion_reserved"]
                check("real archived decision emits a separately certified typed replacement",
                      isinstance(ref, Reference21) and ref.time_s == now
                      and selected["status"] == "solved" and selected["reference_certificate"]["satisfies"]
                      and not np.array_equal(ref.values, nominal.values)
                      and np.max(abs(ref.values-expected["emitted_reference_check"]["reference21"])) <= 1e-12
                      and np.array_equal(entry["frozen_v21_proposal"]["proposed_reference21"], nominal.values)
                      and entry["frozen_v21_proposal"]["hierarchy"] == archived["hierarchy"]
                      and entry["hierarchy"] is None
                      and selected["accepted_task_scope"] == "mandatory_and_full_optional"
                      and np.array_equal(entry["proposed_reference21"], ref.values)
                      and len(obj._fixture_actions) == 1)
                native = obj._fixture_values
                check("native value checks use the same archived q/v and the two correct engine targets",
                      len(native) == 2 and all(t == now and q is frame.robot_position25 and v is frame.robot_velocity25
                                               for q, v, _, t in native)
                      and np.array_equal(native[0][2], frame.robot_actuator_target21)
                      and np.array_equal(native[1][2], encode(ref).values))
                metrics = raw_checks(ref.values, archived, selected["solver"])
                check("replacement preserves raw motion and normal rows plus per-row and total outward limits",
                      metrics["absolute_reference_inside"]
                      and metrics["maximum_motion_drift_normalized"] <= 1e-9
                      and all(metrics[key] <= 1e-8 for key in ("normalized_box_error", "original_hard_error",
                           "original_raw_normal_error", "inward_from_correction", "inward_from_stability", "per_row_pace_error"))
                      and metrics["optional_withdrawal_m"] <= metrics["optional_capacity_ceiling_m"] + 1e-10,
                      metrics)

                try:
                    action(obj)
                    duplicate_rejected = False
                except RuntimeError:
                    duplicate_rejected = True
                check("pending selected reference cannot produce a second action before its receipt",
                      duplicate_rejected and len(obj._fixture_actions) == 1)
                for kind in ("unchanged", "changed", "stale", "wrong generation"):
                    if kind != "unchanged":
                        obj = make_supervisor()
                        ref, _ = action(obj)
                    executed = nominal if kind == "changed" else ref
                    receipt_generation = generation+1 if kind == "wrong generation" else generation
                    receipt = ExecutionReceipt21(ref, executed, encode(executed), "release", receipt_generation,
                                                 kind == "changed", kind == "stale")
                    obj.reconcile(receipt)
                    record = obj.decision_evidence[-1]
                    outcome = record["motion_reserved"]["execution"]
                    check(f"{kind} receipt is consumed once with actual values and no corrective retry",
                          obj._fixture_receipts == [receipt] and len(obj._fixture_actions) == 1
                          and obj._pending_motion_reserved is None and not outcome["retried"]
                          and np.array_equal(record["execution_receipt"]["actual_reference21"], executed.values)
                          and outcome["selected_reference_certificate_preserved"] is (kind == "unchanged"), outcome)

            # The already-bound session must hand the same owner to the inherited
            # lifecycle. No plant or supervisor constructor is invoked here.
            existing = make_supervisor()
            session = object.__new__(runtime.MotionReservedSession)
            session.v16_release_policy = "coordinated"
            session.v16_supervisor = session.release_driver = existing
            session.plant = SimpleNamespace(v16_supervisor=existing, bind_supervisor=deny)
            lifecycle_calls = []

            def parent_lifecycle(self, obs):
                lifecycle_calls.append((obs, self.v16_supervisor, self.release_driver, self.plant.v16_supervisor))
                return "frozen-lifecycle-result"

            with patch.object(runtime.FrozenV21Session, "_lifecycle_step", parent_lifecycle):
                result = session._lifecycle_step(frame)
            check("already-bound session delegates once and retains all three owner aliases",
                  result == "frozen-lifecycle-result" and lifecycle_calls == [(frame, existing, existing, existing)])

            def fresh_session(*, attempted=True, passed=True, manual=False, start=12.):
                obj = object.__new__(runtime.MotionReservedSession)
                obj.v16_release_policy = "coordinated"
                obj.v16_supervisor = obj.release_driver = None
                obj.handoff_attempted, obj.handoff_passed, obj.manual_release = attempted, passed, manual
                obj.plant = SimpleNamespace(v16_supervisor=None, joint_release_entry=SimpleNamespace(started_s=start),
                                            model=object(), open_pose=[np.zeros(18)])
                obj.plant.reference_authority21 = lambda *, include_mode: (lower, upper)
                obj.release_adapter = SimpleNamespace(begin_release=lambda stamp: "public-prior",
                    receiver_half_size_xy_m=(.2, .2), receiver_up_axis3=(0., 0., 1.))
                obj.controller = SimpleNamespace(limits=[2.]*5, budget=8., g=SimpleNamespace(force_margin=.02))
                return obj

            for label, settings in (("failed handoff", {"passed": False}),
                                    ("not attempted", {"attempted": False}),
                                    ("manual release", {"manual": True}),
                                    ("other release time", {"start": 11.995})):
                session = fresh_session(**settings)
                session.plant.bind_supervisor = deny
                lifecycle_calls.clear()
                with patch.object(runtime.FrozenV21Session, "_lifecycle_step", parent_lifecycle), \
                        patch.object(runtime, "MotionReservedSupervisor", side_effect=deny):
                    result = session._lifecycle_step(frame)
                check(f"{label} uses the frozen release path without variant binding",
                      result == "frozen-lifecycle-result" and lifecycle_calls == [(frame, None, None, None)])

            session = fresh_session()
            lifecycle_calls.clear()
            binds = []

            def bind(value):
                binds.append(value)
                session.plant.v16_supervisor = value

            session.plant.bind_supervisor = bind
            with patch.object(runtime.FrozenV21Session, "_lifecycle_step", parent_lifecycle), \
                    patch.object(runtime, "MotionReservedSupervisor", return_value=existing) as make, \
                    patch.object(runtime, "release_clearance_goal", return_value=SimpleNamespace(reference18=np.zeros(18))), \
                    patch.object(runtime, "TransferObserver", return_value=object()):
                result = session._lifecycle_step(frame)
            check("scheduled eligible release binds one variant before one frozen lifecycle call",
                  result == "frozen-lifecycle-result" and make.call_count == 1 and binds == [existing]
                  and lifecycle_calls == [(frame, existing, existing, existing)])
    except Exception:
        error = traceback.format_exc()
    check("production sources unchanged and no construction, forward, contact or physics calls",
          runtime.source_identity() == before and not calls_to_physics)
    result = dict(schema="v21_motion_reserved_runtime_software_v2", passed=error is None and
                  all(row["passed"] for row in checks), checks=checks, error=error,
                  physics_steps=0, plant_instances=0, model_constructions=0,
                  forbidden_calls=calls_to_physics, fixture_context=list(CONTEXT), fixture_anchor=ANCHOR,
                  fixture_input_sha256=pins, production_source_sha256=before, checker_sha256=sha(Path(__file__)))
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "checks": len(checks), "error": error,
                      "output": str(output)}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    raise SystemExit(main(parser.parse_args().out))

"""Independent archived SEARCH and actual-receipt integration gate, with no model."""

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "review/v21_preview_approach_runtime_software_v1.json"
CONTEXT = ["nominal", "transport", 619741]
PINNED = {
    "review/v21_support_approach_schedule_replay_v1.json": "f6a253350ec742d15805839cb90c0b513ab27565d293a1c8d0200a4057e5b019",
    "review/v21_public_support_preview_archive_v1.json": "5228711fc2da64e1a775b831826459d19c53fbca4ed8abe4858333fec3e82635",
    "review/v21_stage_attribution_v1.json": "dd45c6a1fbcd70920c587b5696bf56ef1137b69e3a8f8a5d09304ae335dafc48",
}
PREVIEW_KEYS = ("available", "read_s", "acquisition_s", "conditional_plane_gap_m", "engineering_gap_m")
FRAME_KEYS = {"time", "sample_time", "skin_patches", "sensor_characteristics", "lift_velocity_m_s"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def case(rows):
    return next(item for item in rows if item["context"] == CONTEXT)


def archive():
    blobs = {}
    for name, digest in PINNED.items():
        path = ROOT / name
        assert sha(path) == digest, name
        blobs[name] = read(path)
    replay, preview, stages = blobs.values()
    expected = case(replay["cases"])
    public = case(preview["cases"])
    old = case(stages["rows"])["metrics"]["full_V21"]
    trace_path = ROOT / old["trace"]
    log = trace_path.with_name("release_decisions.jsonl.gz")
    assert sha(trace_path) == old["trace_sha256"] == public["trace_sha256"]
    assert sha(log) == expected["frozen_release_log_sha256"]
    with gzip.open(log, "rt", encoding="utf-8") as stream:
        count = expected["precontact_samples"]
        decisions = [json.loads(next(stream)) for _ in range(count)]
    with np.load(trace_path, allow_pickle=False) as saved:
        frames = saved["observation_json"][2400:2400 + count]
    rows = []
    for d, text, p, e in zip(decisions, frames, public["precontact_rows"],
                             expected["precontact_rows"], strict=True):
        frame = json.loads(text)
        assert d["phase"] == "SEARCH" and frame["time"] == p["read_s"] == e["time_s"] == d["time_s"]
        assert frame["robot_position25"][24] == e["q_m"]
        assert frame["robot_velocity25"][24] == e["actual_v_m_s"]
        assert frame["robot_reference21"] == d["current_reference21"]
        rows.append((SimpleNamespace(**frame), d, {k:p[k] for k in PREVIEW_KEYS if k in p}, e))
    pins = {**PINNED, trace_path.relative_to(ROOT).as_posix(): old["trace_sha256"],
            log.relative_to(ROOT).as_posix(): expected["frozen_release_log_sha256"]}
    return rows, pins


def command(frame, decision, *, time=None, requested=None, brake=False):
    g = decision["governor"]
    rate = g["recovery_reference_budget_m"] / g["recovery_budget_s"]
    return dict(time_s=frame.time if time is None else time,
                q_m=frame.robot_position25[24], v_m_s=frame.robot_velocity25[24],
                current_reference_m=frame.robot_reference21[18],
                reference_lower_m=frame.reference_lower21[18], reference_upper_m=frame.reference_upper21[18],
                reference_rate_lower_m_s=-rate, reference_rate_upper_m_s=rate,
                requested_velocity_m_s=g["requested_velocity_m_s"] if requested is None else requested,
                dt_s=.005, brake_requested=brake, deadline_s=15.5)


def run():
    from identification import v21_preview_approach_runtime_v1 as runtime
    from identification.v21_motion_reserved_runtime_v2 import MotionReservedSupervisor
    from v13_a1.codec import EnginePort21, Reference21
    from v13_joint.plant import ExecutionReceipt21
    from v18_effort_transfer.governor import LiftProgressGovernor
    from v18_effort_transfer.supervisor import governor_task_interval
    from v21_stable_handoff.path_memory import ExitPathMemory
    import mujoco

    before = runtime.source_identity(["ratio_effort"])
    checks, blocked = [], []
    pins = {}
    count = 0
    error = None

    def check(name, ok, detail=None):
        checks.append(dict(name=name, passed=bool(ok), detail=detail))

    def deny(*args, **kwargs):
        blocked.append("model or physics call")
        raise AssertionError(blocked[-1])

    class GovernorWithoutModel(LiftProgressGovernor):
        """Run the frozen scalar governor code with archived constants."""
        def __init__(self, g):
            self._reasons = ()
            self._bound_signature = "archived_constants"
            self.tau_s = g["servo_velocity_scale_s"]
            self.recovery_budget_s = g["recovery_budget_s"]
            self.kp, self.kv, self.passive_damping = (g[k] for k in ("kp_N_m", "kv_N_s_m", "passive_damping_N_s_m"))
            self.calls = []
            self.result = None

        def _signature(self):
            return "archived_constants"

        def _native_effort(self, reference, q, v):
            return {"evaluation_scope": "native effort deliberately not evaluated in this gate"}

        def step(self, **kwargs):
            self.calls.append(kwargs)
            self.result = super().step(**kwargs)
            return self.result

    blocked_names = ("MjModel", "MjData", "mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_contactForce")
    try:
        with patch.multiple(mujoco, **{k:deny for k in blocked_names if hasattr(mujoco, k)}):
            rows, pins = archive()
            count = len(rows)
            frame, decision, preview, expected = rows[-1]
            delegate = GovernorWithoutModel(decision["governor"])
            governor = runtime.PreviewApproachGovernor(delegate)
            check("preserves original governor time constant", governor.tau_s == delegate.tau_s)
            original_ok = forwarded_ok = requested_ok = True
            max_error = 0.
            for f, d, p, archived in rows:
                args = command(f, d)
                raw = LiftProgressGovernor.step(delegate, **args)
                original_ok &= raw.reference_m == d["governor"]["emitted_reference_m"]
                governor.prepare("SEARCH", f.time, p)
                before_call = len(delegate.calls)
                proposal = governor.step(**args)
                diag = proposal.diagnostics["approach_preview"]
                received = delegate.calls[-1]
                requested_ok &= (received["requested_velocity_m_s"] == archived["requested_velocity_m_s"]
                                 == diag["requested_velocity_m_s"])
                forwarded_ok &= (len(delegate.calls) == before_call + 1
                                 and all(received[key] == value for key, value in args.items()
                                         if key != "requested_velocity_m_s"))
                lo, hi, _ = governor_task_interval(proposal, q_m=args["q_m"], v_m_s=args["v_m_s"],
                    actual_lower_m=d["original_mechanical_rate_lower21"][18],
                    actual_upper_m=d["original_mechanical_rate_upper21"][18])
                max_error = max(max_error, abs(float(np.clip(proposal.reference_m, lo, hi))
                                               - archived["new_reference_m"]))
            check("all archive inputs reconstruct the original SEARCH governor", original_ok, count)
            check("all archived SEARCH requests match the frozen schedule", requested_ok, count)
            check("only velocity request changes and delegate is called once per archived tick", forwarded_ok, count)
            check("archived shaped reference agrees after original TASK projection", max_error <= 1e-12, max_error)
            original_request = command(frame, decision)["requested_velocity_m_s"]
            last_shaped = delegate.calls[-1]["requested_velocity_m_s"]
            now = round(frame.time/.005 + 1)*.005
            stale_args = command(frame, decision, time=now)
            stale = governor.step(**stale_args)
            check("stale prepared tick delegates unchanged", stale is delegate.result
                  and delegate.calls[-1] == stale_args and "approach_preview" not in stale.diagnostics)
            governor.prepare("SEARCH", now, {"available": False, "read_s": now})
            dropout = governor.step(**stale_args)
            check("fresh unavailable preview retains previously narrowed pace",
                  delegate.calls[-1]["requested_velocity_m_s"] == last_shaped
                  and dropout.diagnostics["approach_preview"]["status"] == "unknown_preview_no_acceleration")
            for label, update in (("changed duplicate", {"phase": "CAPTURE", "now": now, "preview": None}),
                                  ("rewound", {"phase": "SEARCH", "now": frame.time, "preview": preview})):
                try:
                    governor.prepare(**update)
                except ValueError:
                    ok = True
                else:
                    ok = False
                check(f"rejects {label} prepared read", ok)

            for phase in ("CAPTURE", "TRANSFER", "SEPARATE", "UNSUPPORTED_EXIT"):
                fresh = runtime.PreviewApproachGovernor(delegate)
                fresh.prepare(phase, frame.time, preview)
                args = command(frame, decision, requested=0., brake=phase == "CAPTURE")
                value = fresh.step(**args)
                check(f"{phase} delegates the unmodified request and original brake flag",
                      value is delegate.result and delegate.calls[-1] == args
                      and "approach_preview" not in value.diagnostics)
            fresh = runtime.PreviewApproachGovernor(delegate)
            fresh.prepare("SEARCH", frame.time, None)
            args = command(frame, decision)
            missing = fresh.step(**args)
            check("initial missing preview preserves SEARCH request",
                  delegate.calls[-1] == args and not missing.diagnostics["approach_preview"]["request_was_shaped"])
            for timestamp in ("read_s", "acquisition_s"):
                clock_case = runtime.PreviewApproachGovernor(delegate)
                bad = {**preview, timestamp: preview[timestamp] - .005}
                clock_case.prepare("SEARCH", frame.time, bad)
                result = clock_case.step(**command(frame, decision))
                info = result.diagnostics["approach_preview"]
                check(f"incorrect preview {timestamp} is unknown and cannot shape the request",
                      not info["request_was_shaped"]
                      and info["preview"]["reason"] == "preview_acquisition_clock_mismatch")
            for label, args in (
                ("explicit SEARCH brake", command(frame, decision, brake=True)),
                ("SEARCH deadline", {**command(frame, decision), "deadline_s": frame.time}),
            ):
                value = fresh.step(**args)
                check(f"{label} preserves original braking", value is delegate.result and delegate.calls[-1] == args
                      and value.diagnostics["braking_intent_active"] and "approach_preview" not in value.diagnostics)
            overspeed = {**command(frame, decision), "v_m_s": -.06}
            fresh = runtime.PreviewApproachGovernor(delegate)
            fresh.prepare("SEARCH", frame.time, preview)
            value = fresh.step(**overspeed)
            check("measured overspeed retains original governor brake despite nonzero preview request",
                  value.diagnostics["reason"] == "measured_descent_outside_command_recovery_envelope"
                  and value.diagnostics["governed_velocity_intent_m_s"] == 0.
                  and original_request < delegate.calls[-1]["requested_velocity_m_s"] < 0.)

            # Run the real inherited lifecycle with a typed mock actor receipt.
            # Mock only the parent evidence observation and external ports.
            for kind in ("accepted", "changed", "stale", "wrong generation", "wrong clock", "wrong engine", "wrong engine clock"):
                events = []
                actual_governor = GovernorWithoutModel(decision["governor"])
                graph = object.__new__(runtime.ApproachMotionSupervisor)
                graph.phase, graph.version = "SEARCH", 0
                graph.public_state, graph.debug, graph.decision_evidence = {}, {}, []
                graph.last_time = None
                graph._pending_motion_reserved = None
                graph.entry = SimpleNamespace(started_s=12.)
                graph.exit_path = ExitPathMemory()
                graph.governor = runtime.PreviewApproachGovernor(actual_governor)

                def preview_read(packet):
                    events.append(("preview", set(packet), packet["time"], packet["sample_time"]))
                    return deepcopy(preview)

                graph.approach_preview = SimpleNamespace(read=preview_read)
                session = object.__new__(runtime.PreviewApproachSession)
                session.release_variant = "v21"
                session.v16_release_policy = "coordinated"
                session.v16_supervisor = session.release_driver = graph
                session.release_adapter = SimpleNamespace(transfer_inputs=lambda rx: {"time_s": frame.time})
                session.latest_receiver_measurement = object()
                session.controller = SimpleNamespace(state=SimpleNamespace(_last_time=frame.time, time=frame.time),
                                                     desired=np.zeros(5))
                generation = decision["mode_generation"]
                current = Reference21(frame.robot_reference21, frame.time)
                bias = frame.robot_actuator_target21[18]-current.values[18]
                receipts = []

                def encode(reference):
                    values = reference.values.copy()
                    values[18] += bias
                    return EnginePort21(values, reference.time_s, bias)

                graph.native = SimpleNamespace(actor=SimpleNamespace(encode=encode))

                def commit(reference, *, mode, expected_generation):
                    events.append(("commit", mode, expected_generation))
                    values = reference.values.copy()
                    if kind == "changed":
                        values[18] = current.values[18]
                    actual = Reference21(values, reference.time_s+.005 if kind == "wrong clock" else reference.time_s)
                    engine = encode(actual)
                    if kind == "wrong engine":
                        engine_values = engine.values.copy()
                        engine_values[18] += bias
                        engine = EnginePort21(engine_values, engine.time_s, bias)
                    if kind == "wrong engine clock":
                        engine = EnginePort21(engine.values, engine.time_s + .005, bias)
                    receipt = ExecutionReceipt21(reference, actual, engine, mode,
                        generation + int(kind == "wrong generation"), kind == "changed", kind == "stale")
                    receipts.append(receipt)
                    return receipt

                session.plant = SimpleNamespace(joint_release_entry=graph.entry, v16_supervisor=graph,
                    reference_snapshot=lambda: current, mode_generation=generation,
                    sync_supervisor_generation=lambda: events.append(("sync",)), bind_supervisor=deny,
                    commit_reference21=commit)
                session._attach_execution_state = lambda obs: events.append(("attach", obs is frame))

                def parent_observe(self, *, frame, transfer_inputs, current21):
                    events.append(("observe", frame.time, transfer_inputs["time_s"], np.array_equal(current21, current.values)))
                    self.last_time = frame.time
                    return self.public_state

                with patch.object(MotionReservedSupervisor, "observe", parent_observe):
                    _, _, debug = session._lifecycle_step(frame)
                receipt = receipts[0]
                actual = graph.decision_evidence[-1]["approach_execution"]
                original = graph.decision_evidence[-1]["execution_receipt"]
                order = [event[0] for event in events]
                check(f"{kind} lifecycle: narrow same-frame preview, one unchanged native commit path",
                      order == ["observe", "preview", "sync", "attach", "commit"]
                      and events[1][1] == FRAME_KEYS and events[1][2:] == (frame.time, frame.sample_time)
                      and events[-1][1:] == ("release", generation) and len(receipts) == len(actual_governor.calls) == 1
                      and session.v16_supervisor is session.release_driver is session.plant.v16_supervisor is graph)
                check(f"{kind} lifecycle: actual typed receipt records retention without retry",
                      actual["shaped_request_retained"] is (kind == "accepted")
                      and actual["same_clock_mode_and_generation"] is (kind not in ("wrong generation", "wrong clock", "wrong engine clock"))
                      and actual["engine_matches_executed_reference"] is (kind != "wrong engine")
                      and actual["requested_lift_m"] == receipt.requested.values[18]
                      and actual["actual_lift_m"] == receipt.executed.values[18]
                      and original["actual_reference21"] == debug["actual_reference21"] == receipt.executed.values.tolist()
                      and np.array_equal(session.controller.motion_reference, receipt.executed.values[18:])
                      and not actual["retried"]
                      and abs(receipt.requested.values[18]-expected["new_reference_m"]) <= 1e-12,
                      actual)
            check("TRANSFER action remains the frozen motion-reserved path",
                  runtime.ApproachMotionSupervisor.action is MotionReservedSupervisor.action)
    except Exception as exc:
        import traceback
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    after = runtime.source_identity(["ratio_effort"])
    check("whole registered source closure unchanged", before == after and len(before) == 180, len(before))
    check("no plant, model, acquisition, or simulator call", not blocked)
    return dict(schema="v21_preview_approach_runtime_software_v1", passed=error is None and all(c["passed"] for c in checks),
                checks=checks, error=error, context=CONTEXT, archived_SEARCH_ticks=count,
                physics_steps=0, model_instances=0, plant_instances=0, blocked_calls=blocked,
                production_source_sha256=before, source_sha256=before, input_sha256=pins,
                checker_sha256=sha(Path(__file__)), scope="Archived causal public SEARCH and mock native receipts only.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = run()
    if not args.check_only:
        with OUTPUT.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k:result[k] for k in ("passed", "archived_SEARCH_ticks", "error")}))
    print(json.dumps([c for c in result["checks"] if not c["passed"]]))
    print(f"{sum(c['passed'] for c in result['checks'])}/{len(result['checks'])} checks")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


"""Registered V21 motion-reserved development comparison.

Registration and preflight are read-only with respect to models and runs.
Only the explicit execution mode can create a plant or dispatch a trial.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "review"))
from identification.v21_motion_reserved_runtime_v2 import (
    FrozenV21Session, MotionReservedSession, SOURCE_NAMES, source_files, source_identity,
)
from v13_a1.run_bridge import environment
from v21_stable_handoff.run_development import (
    check_admission, evaluate_research, evaluate_trace, output_path, write,
)
from v22_joint_task_fusion.numerics import numerical_identity
from verify_v21_fragile_hold_continuation_v1 import compare_saved_continuations

SCHEMA = "v21_motion_reserved_development_registration_v2"
DEFAULT_REGISTRATION = ROOT / "review/v21_motion_reserved_development_registration_v2.json"
SOFTWARE = {
    "review/v21_motion_reserved_runtime_software_v2.json": "06bb33c9f6f5ed8e210a1ee9706ac60790e4041de20ea70b5161b2a95f9a4bca",
    "review/v21_motion_reserved_observation_replay_v2.json": "ddb8e5329a5ca63fd62ba3bd3acb3a68e5a392b92478e205dbbca203a8f2a97d",
    "review/v21_motion_reserved_transfer_snapshot_v2.json": "a2984f1c70595d11dd8a4c02927790009e17d8ebd2c54edd5f1b823c1202a899",
}
STAGE = ROOT / "review/v21_stage_attribution_v1.json"
STAGE_SHA = "dd45c6a1fbcd70920c587b5696bf56ef1137b69e3a8f8a5d09304ae335dafc48"
FROZEN = ROOT / "review/v21_complete16_plan_v1.json"
FROZEN_SHA = "77c3670c906cd9b7951fe1ffc0e50dacc10b1e7e4908174ba7f63d0c9b41a9ed"
MOTION = "transport"
SEED = 619741
SCENES = ("nominal", "friction_drop", "fragile_conflict", "collision")
PASSIVE_CONTEXT = ("fragile_conflict", "hold", SEED)
METHODS = ["ratio_effort"]
ARCHIVE_FILES = ("metadata.json", "trace.npz", "result.json", "delivery_intervals.npz",
                 "delivery_summary.json", "delivery_manifest.json")
BRIDGE_FILES = ("joint_decisions.jsonl.gz", "joint_decision_manifest.json",
                "release_decisions.jsonl.gz", "release_decision_manifest.json")
HELPERS = (
    "identification/run_v21_motion_reserved_development_v2.py",
    "review/verify_v21_fragile_hold_continuation_v1.py",
    "review/v18_passive_trace_comparison_v3.py",
    "review/read_v21_motion_reserved_active_comparison_v2.py",
    "review/read_v21_contact_aligned_force_quality_v1.py",
    "review/delivery_auxiliary_metrics_v3.py",
    "review/delivery_auxiliary_contract_v3.json",
    "review/v21_complete16_release_variation_v1.json",
)


class PassiveCarrier(MotionReservedSession):
    """Use the identical new carrier and numeric imports with the old policy."""

    def _lifecycle_step(self, obs):
        return FrozenV21Session._lifecycle_step(self, obs)

    def metadata(self):
        meta = super().metadata()
        meta["motion_reserved"]["active"] = False
        meta["decision_algorithm"] = "frozen_v21_release_in_passive_carrier"
        return meta


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require_hash(path, digest):
    if sha(path) != digest:
        raise ValueError(f"Frozen evidence changed: {path}")
    return path


def pinned_archive(record, bridge=False):
    path = (ROOT / record["trace"]).resolve().parent
    if not path.is_relative_to((ROOT / "output").resolve()):
        raise ValueError("Archive outside the project output directory")
    require_hash(path / "trace.npz", record["trace_sha256"])
    require_hash(path / "delivery_intervals.npz", record["delivery_sha256"])
    names = ARCHIVE_FILES + (BRIDGE_FILES if bridge else ())
    return dict(path=path.relative_to(ROOT).as_posix(), sha256={n:sha(path / n) for n in names})


def assign(context, arm, references):
    scene, motion, seed = context
    return dict(id=f"{arm}__{scene}__{motion}__{seed}", context=list(context), arm=arm, references=references)


def build_registration():
    original = read(require_hash(FROZEN, FROZEN_SHA))
    stage = read(require_hash(STAGE, STAGE_SHA))
    evidence = {name:read(require_hash(ROOT / name, digest)) for name, digest in SOFTWARE.items()}
    current_sources = source_identity(METHODS)
    if {k:v for k, v in current_sources.items() if k not in SOURCE_NAMES} != original["source_sha256"]:
        raise ValueError("Original 170 source identities differ")
    gate, observation, model = evidence.values()
    if (gate["passed"] is not True or len(gate["checks"]) != 22 or gate["production_source_sha256"] != current_sources
            or observation["passed"] is not True or observation["public_readouts"] != 1600
            or model["summary"]["solved"] != 64 or any(e["physics_steps"] != 0 for e in evidence.values())):
        raise ValueError("Required scope-limited software gates do not match the runtime")
    archived = {tuple(row["context"]):row["metrics"] for row in stage["rows"]}
    if len(archived) != 16:
        raise ValueError("Expected the completed original roster")
    bridge = assign(PASSIVE_CONTEXT, "passive",
        {"v21":pinned_archive(archived[PASSIVE_CONTEXT]["full_V21"], bridge=True)})
    active = []
    for scene in SCENES:
        context = (scene, MOTION, SEED)
        records = archived[context]
        active.append(assign(context, "active", {"v21":pinned_archive(records["full_V21"]),
                       "pi":pinned_archive(records["same_prior_legacy_ratio"])}))
    admission = check_admission()
    if not admission["passed"]:
        raise ValueError("Original common platform admission failed")
    return dict(schema=SCHEMA, scope="fixed seen-seed development comparison", confirmation=False,
        selection="one fixed existing seed; all four available transport scenarios, irrespective of result",
        environment=environment(), numerical_identity=numerical_identity(),
        foundation_admission_sha256=admission["manifest_sha256"],
        runtime_source_sha256=current_sources,
        orchestration_source_sha256={k:sha(ROOT / k) for k in HELPERS},
        evidence_sha256={**SOFTWARE, STAGE.relative_to(ROOT).as_posix():STAGE_SHA,
                         FROZEN.relative_to(ROOT).as_posix():FROZEN_SHA},
        schedule=dict(duration_s=16., handoff_s=7.5, release_s=12., public_period_s=.005),
        methods=dict(method="ratio_effort", split="robustness", task_policy="integrated",
                     receiver_policy="coordinated", release_variant="v21"),
        gate="passive full16 exact trace, decision streams and delivery ledger first",
        jobs=[bridge, *active], registration_physics_steps=0)


def save_source_snapshot(out, registration):
    paths = {**source_files(METHODS), **{k:ROOT / k for k in HELPERS}}
    hashes = {**registration["runtime_source_sha256"], **registration["orchestration_source_sha256"]}
    if set(paths) != set(hashes):
        raise ValueError("Source set changed before execution")
    for name, path in paths.items():
        payload = require_hash(path, hashes[name]).read_bytes()
        target = out / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write(payload)
        require_hash(target, hashes[name])
    write(out / "source_snapshot_manifest.json", hashes)


def verify_archive(ref):
    path = ROOT / ref["path"]
    for name, digest in ref["sha256"].items():
        require_hash(path / name, digest)
    return path


def worker(job, methods, expected_sources, expected_numerics, destination):
    """Fresh process, normal V21 step/save/evaluation, with explicit failures."""
    started = time.perf_counter()
    session = None
    try:
        if source_identity(METHODS) != expected_sources or numerical_identity() != expected_numerics:
            raise ValueError("Worker environment or source drift before construction")
        cls = PassiveCarrier if job["arm"] == "passive" else MotionReservedSession
        scene, motion, seed = job["context"]
        session = cls(scenario=scene, motion_mode=motion, seed=seed, **methods)
        if (session.research_sources != expected_sources or session.motion_reserved_numerics != expected_numerics
                or session.tick != 0 or session.time != 0. or session.trace or session.sensors.samples != 0):
            raise ValueError("Fresh run does not match registration at time zero")
        while session.running:
            session.step_one_sensor_tick()
        session.save(destination)
        result = evaluate_trace(destination / "trace.npz")
        result["research_metrics"] = evaluate_research(destination / "trace.npz")
        result.update(wall_seconds=time.perf_counter()-started, motion_mode=motion, experiment_arm=job["arm"])
        write(destination / "result.json", result)
        if source_identity(METHODS) != expected_sources or numerical_identity() != expected_numerics:
            raise ValueError("Worker environment or source drift after execution")
        return dict(status="evaluated", task_success=bool(result["task_success"]), wall_seconds=result["wall_seconds"])
    except Exception as exc:
        failure = dict(status="runtime_failure", error=type(exc).__name__ + ": " + str(exc),
                       traceback=traceback.format_exc(), time_s=None if session is None else session.time)
        if session is not None and session.trace and not destination.exists():
            try:
                session.running = False
                session.completed = False
                session.save(destination)
            except Exception as save_error:
                failure["partial_save_error"] = str(save_error)
        destination.mkdir(parents=True, exist_ok=True)
        write(destination / "runtime_error.json", failure)
        return failure


STATE_COLUMNS = (
    "normal", "skin_normal", "shear", "palm", "hand_other", "all_hand_object",
    "support", "penetration", "world_penetration", "position", "relative",
    "rotation", "relative_velocity", "angular_velocity", "lift", "lift_ref",
    "qpos", "qvel", "wrist_position", "ctrl_before", "alpha_before", "phase",
    "robot_actuator_target21", "robot_control_time_s", "observed_normal",
    "validity", "observed_age", "observed_auxiliary_wrench", "observed_auxiliary_valid",
    "collision", "raw_patches_json", "receiver_measurement_json", "task_request_json",
)
ACTION_COLUMNS = ("executed_reference21", "requested_reference21", "ctrl", "reference_increment21")
SUBSTEP_COLUMNS = ("substep_time", "substep_duration", "substep_normal", "substep_skin_normal",
                   "substep_palm", "substep_hand_other", "substep_all_hand_object", "substep_support",
                   "substep_penetration", "substep_world_penetration")


def same_bytes(x, y):
    if x.shape != y.shape:
        return False
    if x.dtype.kind == y.dtype.kind == "U":
        return np.array_equal(x, y)
    return x.dtype == y.dtype and x.tobytes() == y.tobytes()


def precommand_prefix(left, right):
    """Compare truth and public state up to and including the first changed command."""
    a, b = left["time"], right["time"]
    if not np.array_equal(a, np.arange(3201)*.005) or not np.array_equal(a, b):
        return dict(passed=False, reason="no_matching_complete_control_clock", first_changed_reference_s=None)
    first = next((k for k, (x, y) in enumerate(zip(left["executed_reference21"],
                 right["executed_reference21"], strict=True)) if not same_bytes(x, y)), None)
    if first is not None and a[first] < 12.:
        return dict(passed=False, reason="executed_command_differs_before_release", first_changed_reference_s=float(a[first]))
    before = len(a) if first is None else first
    state_end = len(a) if first is None else first + 1
    checks = {name:same_bytes(left[name][:state_end], right[name][:state_end]) for name in STATE_COLUMNS}
    checks.update({name:same_bytes(left[name][:before], right[name][:before]) for name in ACTION_COLUMNS})
    def public(value, is_observation):
        packet = json.loads(str(value))
        process = packet.get("receiver_release_tasks") if is_observation else packet
        if isinstance(process, dict):
            process.pop("motion_reserved_support", None)
        return json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True)
    for name in ("observation_json", "receiver_release_tasks_json"):
        checks[name] = all(public(x, name == "observation_json") == public(y, name == "observation_json")
                           for x, y in zip(left[name][:state_end], right[name][:state_end], strict=True))
    full_substep = np.arange(32000)*.0005
    checks["complete_substep_clock"] = all(z["substep_time"].shape == full_substep.shape
        and np.allclose(z["substep_time"], full_substep, atol=1e-8, rtol=0.) for z in (left, right))
    physical = 32000 if first is None else round(float(a[first])/.0005)
    checks.update({name:same_bytes(left[name][:physical], right[name][:physical]) for name in SUBSTEP_COLUMNS})
    return dict(passed=all(checks.values()), checks=checks, first_changed_reference_s=None if first is None else float(a[first]),
                control_state_samples=state_end, control_action_samples=before, completed_physical_substeps=physical)


def verify_active_prefix(active, original):
    with np.load(active/"trace.npz", allow_pickle=False) as a, np.load(original/"trace.npz", allow_pickle=False) as b:
        prefix = precommand_prefix(a, b)
        first = prefix["first_changed_reference_s"]
        if first is not None:
            k = round(first/.005)
            changed_reference = a["executed_reference21"][k]
    if not prefix["passed"]:
        return prefix
    logs = active / "release_decisions.jsonl.gz"
    manifest = read(active / "release_decision_manifest.json")
    if manifest["file"] != logs.name or manifest["records"] != 801:
        raise ValueError("Missing full release-decision receipt ledger")
    require_hash(logs, manifest["sha256"])
    matched = first is None
    selected = preserved = fallbacks = 0
    i = -1
    with gzip.open(logs, "rt", encoding="utf-8") as f:
        for i, row in enumerate(map(json.loads, f)):
            if row["time_s"] != (2400+i)*.005 or row["decision_id"] != i:
                raise ValueError("Release-decision clock or sequence mismatch")
            decision = row.get("motion_reserved")
            if decision is None:
                continue
            if decision["status"] != "solved":
                fallbacks += 1
                continue
            selected += 1
            execution = decision.get("execution", {})
            okay = execution.get("selected_reference_certificate_preserved") is True
            preserved += okay
            if row["time_s"] == first:
                matched = okay and np.array_equal(changed_reference, row["execution_receipt"]["actual_reference21"])
    if i != 800:
        raise ValueError("Release ledger is truncated")
    prefix.update(passed=matched, first_change_matches_certified_receipt=matched if first is not None else None,
                  selected_candidates=selected, preserved_certificates=preserved, frozen_fallback_ticks=fallbacks,
                  changed_reference_observed=first is not None)
    return prefix

def stage_trials(jobs, launch, compare_bridge, readout, *, rows=None, checkpoint=None):
    """The passive gate is a dependency, never an optional check."""
    rows = [] if rows is None else rows
    bridge_passed = False
    for job in jobs:
        row = dict(job=job, status="dispatch_started")
        rows.append(row)
        if checkpoint is not None:
            checkpoint(rows)
        try:
            if job["arm"] != "passive" and not bridge_passed:
                row.update(status="not_run_passive_gate_failed")
                continue
            row.update(launch(job))
            if row["status"] != "evaluated":
                continue
            if job["arm"] == "passive":
                row["bridge"] = compare_bridge(job)
                bridge_passed = row["bridge"]["passed"] is True
                if not bridge_passed:
                    row["status"] = "passive_comparison_failed"
            else:
                row["paired_comparison"] = readout(job)
                if row["paired_comparison"].get("available") is not True:
                    row["status"] = "evaluated_comparison_unavailable"
        except Exception as exc:
            row.update(status="comparison_or_dispatch_failure",
                       error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
        finally:
            if checkpoint is not None:
                checkpoint(rows)
    return rows, bridge_passed


def execute(out, registration):
    from read_v21_motion_reserved_active_comparison_v2 import compare_active_run

    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", registration)
    jobs = registration["jobs"]
    rows = []
    bridge = False
    error = None
    final_pins = False
    try:
        save_source_snapshot(out, registration)
        def directory(job):
            return out / "runs" / job["id"]
        def compare_bridge(job):
            return compare_saved_continuations(directory(job), verify_archive(job["references"]["v21"]))
        def readout(job):
            active = directory(job)
            frozen = verify_archive(job["references"]["v21"])
            prefix = verify_active_prefix(active, frozen)
            if not prefix["passed"]:
                return dict(available=False, reason="precommand_prefix_or_receipt_mismatch", prefix=prefix)
            pi = verify_archive(job["references"]["pi"])
            result = compare_active_run(active, frozen, pi)
            result["prefix"] = prefix
            return result
        with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
            def launch(job):
                print(f"Running {job['id']}", flush=True)
                return pool.submit(worker, job, registration["methods"], registration["runtime_source_sha256"],
                                   registration["numerical_identity"], directory(job)).result()
            rows, bridge = stage_trials(jobs, launch, compare_bridge, readout, rows=rows,
                checkpoint=lambda evidence:write(out / "assignment_ledger.json", evidence))
        final_pins = build_registration() == registration
    except Exception as exc:
        error = dict(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
    finally:
        existing = {row["job"]["id"] for row in rows}
        rows.extend(dict(job=j, status="not_run_interrupted_or_initialization_failure")
                    for j in jobs if j["id"] not in existing)
        write(out / "assignment_ledger.json", rows)
        complete = error is None and bridge and final_pins and all(r["status"] == "evaluated" for r in rows)
        final = dict(execution_complete=complete, passive_bridge_passed=bridge, sources_and_environment_unchanged=final_pins,
                     physically_evaluated=sum("task_success" in r for r in rows),
                     assessed=sum(r["status"] == "evaluated" for r in rows), planned=len(jobs), error=error,
                     assignment_sha256=sha(out / "assignment_ledger.json"), confirmation=False)
        write(out / "execution_complete.json", final)
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-registration", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.execute != bool(args.out):
        parser.error("A fresh --out under output is required only for --execute")
    candidate = build_registration()
    if args.write_registration:
        with args.registration.open("x", encoding="utf-8") as f:
            json.dump(candidate, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        print(json.dumps(dict(registered=len(candidate["jobs"]), physics_steps=0)))
        return 0
    registration = read(args.registration)
    if registration != candidate:
        raise ValueError("Source, evidence, environment or assignment differs from registration")
    if args.preflight:
        print(json.dumps(dict(passed=True, jobs=[j["id"] for j in registration["jobs"]],
                              runtime_source_entries=len(registration["runtime_source_sha256"]), physics_steps=0)))
        return 0
    return execute(output_path(args.out), registration)


if __name__ == "__main__":
    raise SystemExit(main())

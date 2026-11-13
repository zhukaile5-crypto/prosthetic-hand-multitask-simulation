"""Run the four registered SEARCH preview ablations against the completed V2 pilot."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import time
import traceback

import numpy as np

from identification import run_v21_motion_reserved_development_v2 as v2
from identification import v21_preview_approach_runtime_v1 as preview
from identification.v21_support_approach_schedule_v1 import READ_DELAY_S
from review import read_v21_motion_reserved_active_comparison_v2 as metrics

ROOT = v2.ROOT
PILOT = ROOT / "output/v21_motion_reserved_development_v2_20261107"
REGISTRATION = ROOT / "review/v21_preview_approach_ablation_registration_v1.json"
SCHEMA = "v21_preview_approach_ablation_v1"
PILOT_REGISTRATION_SHA = "6329cf3bac2541df77e9a1269733018536ec9a61777293c766e4300fdb45dd1d"
PILOT_LEDGER_SHA = "93380014b02131b5fbb703e428cc1208f32d78775b95475c3c6d8d2783e72e0b"
PILOT_TERMINAL_SHA = "c3ac106192240e425b417183b3c29067b35e8cc60d9f6d620179f4d009ca0534"
SOFTWARE = {
    "review/v21_preview_approach_runtime_software_v1.json": "2e8ea7df4b96f34d2f912ace54823a0452b19c0271938d4050d707c2a676f0c2",
    "review/v21_preview_approach_runtime_request_replay_v1.json": "8575ab55d6f16bf4d8cc935e94109b5c51c405318567533517db9082f84ad343",
}
LEDGER = "release_decisions.jsonl.gz"
LEDGER_MANIFEST = "release_decision_manifest.json"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def decision_rows(run):
    manifest = v2.read(run / LEDGER_MANIFEST)
    require(manifest["file"] == LEDGER and manifest["records"] == 801, "Invalid release decision manifest")
    with gzip.open(v2.require_hash(run / LEDGER, manifest["sha256"]), "rt", encoding="utf-8") as f:
        rows = list(map(json.loads, f))
    require(len(rows) == 801 and all(row["decision_id"] == i and row["time_s"] == (2400 + i) * .005
                                    for i, row in enumerate(rows)), "Invalid release decision sequence")
    return rows


def build_registration():
    old = v2.read(v2.require_hash(v2.DEFAULT_REGISTRATION, PILOT_REGISTRATION_SHA))
    require(old == v2.build_registration() == v2.read(PILOT / "plan.json"), "V2 plan or environment drift")
    terminal = v2.read(v2.require_hash(PILOT / "execution_complete.json", PILOT_TERMINAL_SHA))
    require(terminal["execution_complete"] and terminal["passive_bridge_passed"]
            and terminal["sources_and_environment_unchanged"] and terminal["assessed"] == 5
            and terminal["assignment_sha256"] == PILOT_LEDGER_SHA, "Incomplete V2 pilot")
    rows = v2.read(v2.require_hash(PILOT / "assignment_ledger.json", PILOT_LEDGER_SHA))
    require([r["job"] for r in rows] == old["jobs"] and all(r["status"] == "evaluated" for r in rows)
            and rows[0]["bridge"]["passed"], "V2 roster or acceptance differs")

    sources = preview.source_identity(v2.METHODS)
    require(len(sources) == 180
            and {k:v for k,v in sources.items() if k not in preview.EXTRA_SOURCES} == old["runtime_source_sha256"],
            "Expected the frozen V2 sources plus three preview files")
    gate, replay = (v2.read(v2.require_hash(ROOT / n, s)) for n, s in SOFTWARE.items())
    require(gate["passed"] and gate["production_source_sha256"] == sources and len(gate["checks"]) == 36
            and replay["passed"] and replay["requests"] == 5471
            and gate["physics_steps"] == replay["physics_steps"] == 0, "Preview software gate differs")

    jobs = []
    for row in rows[1:]:
        context = row["job"]["context"]
        base = PILOT / "runs" / row["job"]["id"]
        comparison = row["paired_comparison"]
        reference = comparison["arms"]["active"]
        require(comparison["available"] and comparison["prefix"]["passed"] and comparison["context_key"] == context
                and Path(reference["run"]).resolve() == base.resolve(), "Missing assessed V2 comparison")
        meta, _, pins = metrics._load_record(base, "motion_reserved", reference["input_sha256"])
        require(meta["research_source_sha256"] == old["runtime_source_sha256"]
                and meta["numerical_identity"] == old["numerical_identity"]
                and metrics._context(meta) == tuple(context), "V2 run sources or context differ")
        decision_rows(base)
        extra_pins = {n:v2.sha(base / n) for n in (LEDGER, LEDGER_MANIFEST)}
        jobs.append(v2.assign(context, "preview", {"motion_reserved": {
            "path": base.relative_to(ROOT).as_posix(), "sha256": {**pins, **extra_pins}}}))
    orchestration = {**old["orchestration_source_sha256"], Path(__file__).relative_to(ROOT).as_posix(): v2.sha(Path(__file__))}
    return dict(schema=SCHEMA, scope="four fixed paired transport contexts, one existing development seed",
                planned_physics_episodes=4, confirmation=False, runtime_source_sha256=sources,
                orchestration_source_sha256=orchestration, evidence_sha256=SOFTWARE,
                pilot_registration_sha256=PILOT_REGISTRATION_SHA, pilot_ledger_sha256=PILOT_LEDGER_SHA,
                pilot_terminal_sha256=PILOT_TERMINAL_SHA, environment=old["environment"],
                numerical_identity=old["numerical_identity"], schedule=old["schedule"], methods=old["methods"],
                primary="target-top normal peak over the first 20 ms of each arm's own known contact onset",
                secondary=["20 ms impulse", "physical clearance latency", "0–200 ms finger force P95",
                           "200–500 ms finger force RMS", "full force guard", "local and total excess",
                           "task and floor contact", "first and final low-motion dwell"],
                jobs=jobs, registration_physics_steps=0)


def worker(job, registration, destination):
    started = time.perf_counter()
    session = None
    try:
        sources = registration["runtime_source_sha256"]
        numerics = registration["numerical_identity"]
        require(preview.source_identity(v2.METHODS) == sources and v2.numerical_identity() == numerics,
                "Preview worker environment drift")
        scene, motion, seed = job["context"]
        session = preview.PreviewApproachSession(scenario=scene, motion_mode=motion, seed=seed,
                                                **registration["methods"])
        require(session.research_sources == sources and session.motion_reserved_numerics == numerics
                and session.tick == 0 and session.time == 0. and not session.trace and session.sensors.samples == 0,
                "Preview run does not match registration at time zero")
        actor = session.plant.actor21
        native = dict(indices21=actor.indices21.tolist(), lift_bias_m=actor.lift_bias_m)
        while session.running:
            session.step_one_sensor_tick()
        session.save(destination)
        v2.write(destination / "preview_native_codec.json", native)
        result = v2.evaluate_trace(destination / "trace.npz")
        result.update(research_metrics=v2.evaluate_research(destination / "trace.npz"), motion_mode=motion,
                      experiment_arm=job["arm"], wall_seconds=time.perf_counter()-started)
        v2.write(destination / "result.json", result)
        return dict(status="evaluated", task_success=bool(result["task_success"]), wall_seconds=result["wall_seconds"])
    except Exception as exc:
        failure = dict(status="runtime_failure", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                       time_s=None if session is None else session.time)
        if session is not None and session.trace and not destination.exists():
            try:
                session.running = False
                session.completed = False
                session.save(destination)
            except Exception as save_error:
                failure["partial_save_error"] = str(save_error)
        destination.mkdir(parents=True, exist_ok=True)
        v2.write(destination / "runtime_error.json", failure)
        return failure


def first_change_checks(now, new, old, new_ref, old_ref, new_req, old_req, new_ctrl, old_ctrl, codec, generation):
    info = new["governor"].get("approach_preview", {})
    execution = new.get("approach_execution", {})
    public = info.get("preview") or {}
    new_receipt, old_receipt = (r["execution_receipt"] for r in (new, old))
    changed = np.flatnonzero(new_ref != old_ref).tolist()
    indices = np.asarray(codec["indices21"], dtype=int)
    bias = codec["lift_bias_m"]

    def encoded(reference):
        port = reference.copy()
        port[18] += bias
        return port

    def in_authority(row, reference):
        return (np.all(np.asarray(row["actual_reference_lower21"]) <= reference + 1e-12)
                and np.all(reference <= np.asarray(row["actual_reference_upper21"]) + 1e-12))

    original = info.get("original_requested_velocity_m_s")
    shaped = info.get("requested_velocity_m_s")
    read = public.get("read_s")
    acquired = public.get("acquisition_s")
    return dict(
        same_search_frame=new["time_s"] == old["time_s"] == now and new["phase"] == old["phase"] == "SEARCH"
            and new["mode_generation"] == old["mode_generation"] == generation,
        public_preview_is_current=public.get("available") is True and read == now and acquired is not None
            and abs(now - acquired - READ_DELAY_S) < 1e-9,
        velocity_reduced=info.get("request_was_shaped") is True and original is not None and shaped is not None
            and original == old["governor"]["requested_velocity_m_s"] and original < shaped <= 0.,
        lift_only_less_downward=changed == [18] and new_ref[18] > old_ref[18],
        approach_receipt=all(execution.get(k) is True for k in
            ("same_clock_mode_and_generation", "engine_matches_executed_reference", "shaped_request_retained"))
            and execution.get("retried") is False
            and execution.get("requested_lift_m") == new_req[18] and execution.get("actual_lift_m") == new_ref[18],
        original_receipts=all(receipt["execution_changed"] is False and receipt["stale_plan_replaced"] is False
            and np.array_equal(receipt["actual_reference21"], ref) and np.array_equal(req, ref)
            and np.array_equal(row["proposed_reference21"], req) and in_authority(row, ref)
            for row, receipt, ref, req in ((new, new_receipt, new_ref, new_req), (old, old_receipt, old_ref, old_req))),
        engine_postcommit=indices.shape == (21,) and np.unique(indices).size == 21
            and np.array_equal(new_ctrl[indices], encoded(new_ref)) and np.array_equal(old_ctrl[indices], encoded(old_ref))
            and indices[18] == 20 and new_ctrl[20] > old_ctrl[20]
            and np.flatnonzero(new_ctrl != old_ctrl).tolist() == [20],
    )


def audit_prefix(run, base, arms):
    new_log, old_log = decision_rows(run), decision_rows(base)
    with np.load(run / "trace.npz", allow_pickle=False) as a, np.load(base / "trace.npz", allow_pickle=False) as b:
        prefix = v2.precommand_prefix(a, b)
        if not prefix["passed"]:
            return dict(status="prefix_mismatch", prefix=prefix)
        now = prefix["first_changed_reference_s"]
        k = prefix["control_state_samples"] - 1
        extra = {name:v2.same_bytes(a[name][:k+1], b[name][:k+1]) for name in
                 ("mode_generation", "observation_json", "receiver_release_tasks_json",
                  "legacy_common_sensor_json", "environment_json")}
        if not all(extra.values()):
            return dict(status="prefix_mismatch", prefix=prefix, extra=extra)
        if now is None:
            return dict(status="no_changed_reference", prefix=prefix, extra=extra)
        n = k - 2400
        require(0 <= n < len(new_log), "First change not on the release decision clock")
        row = new_log[n]
        checks = first_change_checks(now, row, old_log[n], a["executed_reference21"][k], b["executed_reference21"][k],
            a["requested_reference21"][k], b["requested_reference21"][k], a["ctrl"][k], b["ctrl"][k],
            v2.read(run / "preview_native_codec.json"), int(a["mode_generation"][k]))
        touch = {name:arm["first_target_top_impact_start_s"] for name, arm in arms.items()}
        before_touch = all(t is not None and now < t for t in touch.values())
        def object_vertical_speed(directory, t):
            if t is None:
                return None
            with np.load(directory / "delivery_intervals.npz", allow_pickle=False) as intervals:
                matches = np.flatnonzero(np.isclose(intervals["interval_end_s"], t, rtol=0., atol=1e-8))
                if len(matches) == 1 and intervals["object_state_valid"][matches[0]]:
                    return float(intervals["object_free_qvel6_end"][matches[0], 2])
                return None
        return dict(status="attributed_precontact_search" if all(checks.values()) and before_touch else "unattributed",
                    prefix=prefix, extra=extra, first_change_checks=checks, before_both_contact_onsets=before_touch,
                    first_change_s=now, first_change_preview=row["governor"].get("approach_preview"),
                    first_lift_reference_delta_m=float(a["executed_reference21"][k,18] - b["executed_reference21"][k,18]),
                    first_touch_s=touch, precontact_object_vertical_velocity_m_s=dict(
                        preview=object_vertical_speed(run, touch["preview"]),
                        motion_reserved=object_vertical_speed(base, touch["motion_reserved"])))


def readout(run, job, registration):
    base = v2.verify_archive(job["references"]["motion_reserved"])
    candidate = metrics._load_record(run, "preview")
    control = metrics._load_record(base, "motion_reserved")
    a, b = candidate[0], control[0]
    require(metrics._context(a) == metrics._context(b) == tuple(job["context"])
            and all(a[k] == b[k] for k in metrics.MATCHED_FIELDS)
            and candidate[1]["research_metrics"]["noise_seed"] == control[1]["research_metrics"]["noise_seed"]
            and a["schema"] == preview.SCHEMA and a["research_source_sha256"] == registration["runtime_source_sha256"]
            and a["numerical_identity"] == registration["numerical_identity"], "Ablation pair is not matched")
    arms = {"preview":metrics._arm(run, candidate), "motion_reserved":metrics._arm(base, control)}
    audit = audit_prefix(run, base, arms)
    return dict(arms=arms, audit=audit, preview_minus_motion_reserved=metrics._contrasts(*arms.values()))


def execute(out, registration):
    out.mkdir(parents=True, exist_ok=False)
    v2.write(out / "plan.json", registration)
    rows = []
    error = None
    final_pins = False
    try:
        paths = {**preview.source_files(v2.METHODS),
                 **{k:ROOT / k for k in registration["orchestration_source_sha256"]}}
        pins = {**registration["runtime_source_sha256"], **registration["orchestration_source_sha256"]}
        require(set(paths) == set(pins), "Source snapshot roster differs")
        for name, path in paths.items():
            target = out / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(v2.require_hash(path, pins[name]).read_bytes())
            v2.require_hash(target, pins[name])
        v2.write(out / "source_snapshot_manifest.json", pins)
        with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
            for job in registration["jobs"]:
                row = dict(job=job, status="dispatch_started")
                rows.append(row)
                v2.write(out / "assignment_ledger.json", rows)
                run = out / "runs" / job["id"]
                print(f"Running {job['id']}", flush=True)
                try:
                    row.update(pool.submit(worker, job, registration, run).result())
                    if row["status"] == "evaluated":
                        row["paired_comparison"] = readout(run, job, registration)
                except Exception as exc:
                    row.update(status="comparison_or_dispatch_failure", error=f"{type(exc).__name__}: {exc}",
                               traceback=traceback.format_exc())
                v2.write(out / "assignment_ledger.json", rows)
        final_pins = build_registration() == registration
    except Exception as exc:
        error = dict(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    existing = {r["job"]["id"] for r in rows}
    rows.extend(dict(job=j, status="not_run_initialization_failure")
                for j in registration["jobs"] if j["id"] not in existing)
    ledger = out / "assignment_ledger.json"
    v2.write(ledger, rows)
    assessed = sum("paired_comparison" in r for r in rows)
    attributed = sum(r.get("paired_comparison", {}).get("audit", {}).get("status") ==
                     "attributed_precontact_search" for r in rows)
    complete = error is None and final_pins and assessed == len(rows)
    final = dict(execution_complete=complete, sources_environment_and_pilot_unchanged=final_pins,
                 physically_evaluated=sum("task_success" in r for r in rows), assessed=assessed,
                 attributed_precontact_search=attributed, planned=len(rows), error=error,
                 assignment_sha256=v2.sha(ledger), confirmation=False)
    v2.write(out / "execution_complete.json", final)
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-registration", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.execute != bool(args.out):
        parser.error("A fresh --out under output is required only for --execute")
    candidate = build_registration()
    if args.write_registration:
        with REGISTRATION.open("x", encoding="utf-8") as stream:
            json.dump(candidate, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        return 0
    registration = v2.read(REGISTRATION)
    require(candidate == registration, "Ablation registration no longer matches live prerequisites")
    if args.preflight:
        print(json.dumps(dict(passed=True, jobs=[j["id"] for j in registration["jobs"]],
                              runtime_source_entries=len(registration["runtime_source_sha256"]), physics_steps=0)))
        return 0
    return execute(v2.output_path(args.out), registration)


if __name__ == "__main__":
    raise SystemExit(main())


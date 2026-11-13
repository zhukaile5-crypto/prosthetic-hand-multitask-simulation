"""Frozen two-case, three-arm wrist alignment pilot with fresh output only."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np

from identification import run_v21_motion_reserved_development_v2 as frozen
from identification import v21_held_wrist_alignment_v1 as task
from identification.v21_preview_approach_runtime_v1 import source_files as base_sources
from review.read_wrist_alignment_v1 import evaluate

ROOT = frozen.ROOT
REGISTRATION = ROOT / "review/wrist_alignment_pilot_registration_v1.json"
SEEDS = (630173, 630197)
WORKERS = 3
DT = .005
COMMON = ("time", "qpos", "qvel", "ctrl_before", "normal", "skin_normal", "relative",
          "position", "wrist_position", "requested_wrist_reference")
TREATMENT_PINS = (
    "identification/run_held_wrist_alignment_v1.py",
    "identification/v21_held_wrist_alignment_v1.py",
    "review/read_wrist_alignment_v1.py",
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sources():
    return {**base_sources(frozen.METHODS), **{name: ROOT / name for name in TREATMENT_PINS}}


def identity():
    return {name: frozen.sha(path) for name, path in sources().items()}


def registration():
    acceptance = frozen.check_admission()
    require(acceptance["passed"], "Common platform admission failed")
    jobs = [dict(id=f"{case}__{arm}__{seed}", case=case, arm=arm, seed=seed)
            for seed in SEEDS for case in task.CASES for arm in task.ARMS]
    return dict(schema="held_wrist_alignment_pilot_registration_v1", scope="held object with two wrist axes; stop before release",
                jobs=jobs, physics_episodes=len(jobs), physical_seeds=list(SEEDS), workers=WORKERS,
                common_target_rad=list(task.TARGET_RAD), alignment_start_s=task.START_S, ramp_end_s=task.RAMP_END_S,
                observed_stop_s=task.END_S, control_samples=task.CONTROL_TICKS, control_period_s=DT, physics_period_s=.0005,
                pull=dict(start_s=task.PULL_START_S, end_s=task.PULL_END_S, world_y_N=task.PULL_N),
                method_convention=dict(ratio_pi="frozen legacy/blind holding port", current="frozen integrated holding port",
                    expanded="same integrated holding solve with bounded two-wrist nominal effort aim"),
                guard_scope="full 2 kHz pre-release and alignment windows, per-finger and total",
                outcomes=["wrist tracking", "pull recovery", "relative object displacement", "force guards and excess"],
                evidence_scope="two new development seeds shared within each case, no retuning from these runs",
                environment=frozen.environment(), numerics=frozen.numerical_identity(),
                sources_sha256=identity(), foundation_admission_sha256=acceptance["manifest_sha256"])


def software_preflight():
    times = np.arange(task.CONTROL_TICKS)*DT
    public = task.trajectory(.1)
    poses = np.array([public.position_at(t) for t in times])
    require(np.array_equal(poses[0,1:], np.zeros(2)) and np.array_equal(poses[-1,1:], task.TARGET_RAD)
            and np.max(abs(np.diff(poses[:,1:],axis=0))) < .12*DT, "Unexpected shared wrist path or rate")
    expect = (times >= task.PULL_START_S) & (times < task.PULL_END_S)
    for case in task.CASES:
        rule = task.case_schedule(case)
        commands = [rule("nominal", t, .6, .1)[0] for t in times]
        actual = np.asarray([env["force"] for env in commands])
        require(np.all(actual[:,[0,2]] == 0.)
                and np.array_equal(actual[:,1], task.PULL_N*expect if case == "midturn_pull" else np.zeros(len(times)))
                and all(env["friction"] == .6 and env["squeeze"] == 0. and env["obstacle_x"] == 0.
                        and env["torque"] == [0.,0.,0.] for env in commands), "Unexpected perturbation configuration")
    require(int(expect.sum()) == 60, "Pull must span exactly 60 control intervals")
    nominal = np.array([.03,-.02])
    delta = task.wrist_aim(nominal, nominal-[.005,-.006], np.zeros(2), nominal-.0024, nominal+.0024, .0024)-nominal
    require(np.allclose(delta,[.0024,-.0024],atol=1e-12,rtol=0), "Encoder aim direction or rate bound differs")
    require(np.array_equal(task.wrist_aim(nominal, nominal, np.zeros(2), nominal-.0024, nominal+.0024, .0024),
                           nominal), "Aim must vanish at a steady target")


def verify_trace(path, session, case):
    times = np.arange(task.CONTROL_TICKS)*DT
    with np.load(path,allow_pickle=False) as trace:
        require(np.array_equal(trace["time"], times) and len(trace["substep_time"]) == 24000
                and np.allclose(trace["substep_time"], np.arange(24000)*.0005, rtol=0., atol=1e-8),
                "Incomplete held alignment clocks")
        target = np.asarray([session.user_trajectory.position_at(t) for t in times])
        require(np.array_equal(trace["actual_motion_target"], target)
                and np.array_equal(trace["requested_wrist_reference"], target[:,1:]), "Recorded wrist request differs")
        mask = ((times >= task.PULL_START_S) & (times < task.PULL_END_S)) if case == "midturn_pull" else np.zeros(len(times),bool)
        force = np.asarray([json.loads(x)["force"] for x in trace["environment_json"]])
        require(np.array_equal(force[:,1], task.PULL_N*mask) and np.all(force[:,[0,2]] == 0.),
                "Recorded pull schedule differs")
        if session.handoff_passed:
            require(session.release_at is None and session.plant.release_started is None
                    and not np.any(trace["command"] == "release"), "An exposed alignment arm issued a release")
        return dict(request_samples=len(target), logged_pull_intervals=int(mask.sum()),
                    pull_passed_only_to_environment=True, completed_physical_substeps=24000,
                    warning_count=int(np.max(trace["warning_count"])))


def worker(job, plan, destination):
    started = time.perf_counter()
    session = None
    try:
        require(identity() == plan["sources_sha256"] and frozen.numerical_identity() == plan["numerics"],
                "Worker source or numeric environment drift")
        session = task.AlignmentSession(arm=job["arm"], case=job["case"], seed=job["seed"])
        require(session.tick == 0 and session.time == 0. and session.sensors.samples == 0,
                "Fresh alignment session required")
        case = job["case"]
        # One fresh spawned process per trial. The common runtime logs the
        # schedule and sends it only to the plant's environment input.
        with patch.object(task.held, "event_schedule", task.case_schedule(case)):
            for _ in range(task.CONTROL_TICKS):
                session.step_one_sensor_tick()
        require(session.tick == task.CONTROL_TICKS and session.time == task.END_S, "Did not reach held task boundary")
        session.running = False
        session.alignment_finished = True
        meta = session.save(destination)
        trace = destination / "trace.npz"
        proof = verify_trace(trace, session, case)
        report = evaluate(trace, pull_start_s=task.PULL_START_S if case=="midturn_pull" else None,
                          pull_end_s=task.PULL_END_S if case=="midturn_pull" else None)
        evidence = [r.get("wrist_alignment", {}) for r in session.controller.decision_evidence
                    if task.START_S <= r["time_s"] < task.END_S]
        evidence = [r for r in evidence if r]
        aim = dict(decisions=len(evidence), public_guard_met=sum(r["public_grasp_guard"] for r in evidence),
                   nonzero_nominal_offset_decisions=sum(any(abs(x)>1e-12 for x in (r["wrist_nominal_offset_rad"] or []))
                                                       for r in evidence))
        row = dict(status="evaluated", handoff_passed=meta["handoff_passed"], trace_sha256=meta["trace_sha256"],
                   proof=proof, wrist_aim=aim, metrics=report, wall_seconds=time.perf_counter()-started)
        frozen.write(destination / "result.json", row)
        return row
    except Exception as exc:
        failure = dict(status="runtime_or_evaluation_failure", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc(), time_s=None if session is None else session.time)
        if session is not None and session.trace and not destination.exists():
            try:
                session.running = False
                session.alignment_finished = False
                session.save(destination)
            except Exception as save_error:
                failure["partial_save_error"] = str(save_error)
        destination.mkdir(parents=True, exist_ok=True)
        frozen.write(destination / "runtime_error.json", failure)
        return failure


def equal_prefix(a, b, until):
    with np.load(a/"trace.npz", allow_pickle=False) as x, np.load(b/"trace.npz", allow_pickle=False) as y:
        k = round(until/DT)
        checks = {n:np.array_equal(x[n][:k+1],y[n][:k+1]) for n in COMMON}
        for n in ("ctrl", "executed_reference21", "substep_normal", "substep_skin_normal"):
            count = k*10 if n.startswith("substep") else k
            checks[n] = np.array_equal(x[n][:count],y[n][:count])
        return dict(passed=all(checks.values()), matched_through_s=until,
                    fields=checks, state_samples=k+1)


def pair_audit(out, rows):
    indexed = {(r["job"]["seed"],r["job"]["case"],r["job"]["arm"]):r for r in rows}
    audits = []
    for seed in SEEDS:
        def run(case, arm):
            row = indexed[seed,case,arm]
            if row["status"] != "evaluated":
                raise ValueError(f"Missing assessed matched arm {seed}/{case}/{arm}")
            path = out / "runs" / row["job"]["id"]
            frozen.require_hash(path/"trace.npz",row["trace_sha256"])
            return path
        pairs = [(case,"ratio_pi",case,"current",7.5) for case in task.CASES]
        pairs += [(case,"current",case,"expanded",task.START_S) for case in task.CASES]
        pairs += [("normal",arm,"midturn_pull",arm,task.PULL_START_S) for arm in task.ARMS]
        for case_a, arm_a, case_b, arm_b, until in pairs:
            a,b = run(case_a,arm_a),run(case_b,arm_b)
            ma,mb = (frozen.read(x/"metadata.json") for x in (a,b))
            matched = all(ma[n] == mb[n] for n in
                ("seed","config","noise","limits","budget","model_sha256","physics_timestep","initial_position"))
            audits.append(dict(seed=seed, first=[case_a,arm_a], second=[case_b,arm_b], same_configuration=matched,
                               **equal_prefix(a,b,until)))
    return audits


def execute(out, plan):
    out.mkdir(parents=True, exist_ok=False)
    frozen.write(out/"plan.json", plan)
    rows = [{"job":j,"status":"queued"} for j in plan["jobs"]]
    ledger = out/"assignment_ledger.json"
    frozen.write(ledger,rows)
    error = None
    audits = []
    pins_ok = False
    try:
        for name,path in sources().items():
            target = out / "sources" / name
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as stream:
                stream.write(frozen.require_hash(path, plan["sources_sha256"][name]).read_bytes())
            frozen.require_hash(target, plan["sources_sha256"][name])
        frozen.write(out/"source_snapshot_manifest.json",plan["sources_sha256"])
        with ProcessPoolExecutor(max_workers=WORKERS,max_tasks_per_child=1) as pool:
            pending={pool.submit(worker,r["job"],plan,out/"runs"/r["job"]["id"]):r for r in rows}
            for future in as_completed(pending):
                row=pending[future]
                try:
                    row.update(future.result())
                except Exception as exc:
                    row.update(status="dispatch_failure",error=f"{type(exc).__name__}: {exc}",
                               traceback=traceback.format_exc())
                frozen.write(ledger,rows)
                print(row["job"]["id"],row["status"],row.get("wall_seconds"),row.get("error",""),flush=True)
        audits = pair_audit(out,rows)
        pins_ok = registration() == plan
    except Exception as exc:
        error = dict(error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
    frozen.write(out/"pair_audit.json",audits)
    measured = sum(r["status"]=="evaluated" for r in rows)
    matched = len(audits) == 14 and all(x["passed"] and x["same_configuration"] for x in audits)
    final = dict(execution_complete=error is None and pins_ok and measured==len(rows) and matched,
                 assessed=measured, planned=len(rows), sources_and_environment_unchanged=pins_ok,
                 matched_pair_audits_passed=matched, assignment_sha256=frozen.sha(ledger), error=error)
    frozen.write(out/"execution_complete.json",final)
    print(json.dumps(final),flush=True)
    return 0 if final["execution_complete"] else 1


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes=cli.add_mutually_exclusive_group(required=True)
    modes.add_argument("--register",action="store_true")
    modes.add_argument("--preflight",action="store_true")
    modes.add_argument("--execute",action="store_true")
    cli.add_argument("--out")
    args=cli.parse_args()
    if args.execute != bool(args.out):
        cli.error("A fresh --out under output is required only for --execute")
    software_preflight()
    plan=registration()
    if args.register:
        with REGISTRATION.open("x",encoding="utf-8") as stream:
            json.dump(plan,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write("\n")
        return 0
    require(frozen.read(REGISTRATION)==plan,"Registration differs from live code or environment")
    if args.preflight:
        print(f"PASS: {len(plan['jobs'])} jobs, {len(plan['sources_sha256'])} pinned sources, 0 physics steps")
        return 0
    return execute(frozen.output_path(args.out),plan)


if __name__=="__main__":
    raise SystemExit(main())


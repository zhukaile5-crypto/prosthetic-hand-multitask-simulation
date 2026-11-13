"""Fixed paired pilot, new outputs only. Never retune or expand after outcomes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preserved_results(root, scenes, seed):
    files = {}
    for scene in scenes:
        runs = (root/"output/midterm_core_macro_ablation_v1_verified_20261111/runs"/f"core__{scene}__{seed}",
                root/"output/v17b_prospective80_baselines_transport_v1/runs"/f"{scene}__ratio_effort__{seed}")
        for folder in runs:
            for name in ("trace.npz", "metadata.json", "result.json", "delivery_intervals.npz"):
                path = folder/name
                files[path.relative_to(root).as_posix()] = sha(path)
    return files


def worker(job, plan, output):
    from v22_joint_task_fusion.numerics import ensure_numeric_environment, numerical_identity
    ensure_numeric_environment()
    import numpy as np
    from .runtime import identity, make_session
    from .metrics import evaluate
    output = Path(output)
    run_dir = output/"runs"/job["id"]
    began, session = time.perf_counter(), None
    try:
        if identity() != plan["sources_sha256"]:
            raise RuntimeError("Source bytes changed after freezing the pilot")
        if numerical_identity() != plan["numerics"]:
            raise RuntimeError("Numerical environment changed after freezing the pilot")
        session = make_session(job["arm"], job["scene"], job["seed"])
        while session.running:
            session.step_one_sensor_tick()
            if session.tick % 100 == 0 or not session.running:
                progress = dict(job=job, simulated_time_s=session.time, phase=session.plant.phase,
                    lift_request_start_s=session.pickup_task.lift_started,
                    object_rise_m=float(session.last_truth["position"][2]-session.plant.initial_position[2]),
                    total_finger_force_N=float(np.sum(session.last_truth["normal"])),
                    wall_seconds=time.perf_counter()-began, completed=session.completed)
                write(output/"progress"/(job["id"]+".json"), progress)
        if not session.completed or session.tick != 3200:
            raise RuntimeError("Incomplete scheduled trial")
        session.save(run_dir)
        result = evaluate(run_dir)
        if identity() != plan["sources_sha256"]:
            raise RuntimeError("Source bytes changed during the pilot")
        response = dict(status="evaluated", job=job, result=result, wall_seconds=time.perf_counter()-began)
        write(run_dir/"job.json", response)
        return response
    except Exception as error:
        response = dict(status="runtime_failure", job=job, error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(), simulation_time_s=None if session is None else session.time,
            wall_seconds=time.perf_counter()-began)
        if session is not None and session.trace and not run_dir.exists():
            try:
                session.running = False
                session.completed = False
                session.save(run_dir)
            except Exception as save_error:
                response["partial_save_error"] = str(save_error)
        write(run_dir/"runtime_error.json", response)
        return response


def pair_audit(output, scene, seed):
    import numpy as np
    paths = [Path(output)/"runs"/f"{arm}__{scene}__{seed}" for arm in ("baseline", "macro")]
    if not all((p/"result.json").is_file() for p in paths):
        return dict(passed=False, reason="missing evaluated arm")
    meta = [read(p/"metadata.json") for p in paths]
    equal_fields = ("config", "limits", "budget", "noise", "seed", "model_sha256",
                    "initial_qpos", "initial_qvel", "initial_reference21", "actual_ratio_task_gains")
    checks = {k: meta[0][k] == meta[1][k] for k in equal_fields}
    with np.load(paths[0]/"trace.npz", allow_pickle=False) as a, np.load(paths[1]/"trace.npz", allow_pickle=False) as b:
        for name in ("normal", "skin_normal", "qpos", "qvel", "ctrl_before"):
            checks["initial_"+name] = bool(np.array_equal(a[name][0], b[name][0]))
        checks["environment_schedule"] = bool(np.array_equal(a["environment_json"], b["environment_json"]))
        checks["both_controllers_active_at_zero"] = bool(a["controller_active"][0] and b["controller_active"][0])
        checks["fixed_transport_request"] = bool(np.array_equal(a["actual_motion_target"][a["time"] >= 9.3], b["actual_motion_target"][b["time"] >= 9.3]))
        different = np.any(a["executed_reference21"] != b["executed_reference21"], axis=1)
        first_action = float(a["time"][np.flatnonzero(different)[0]]) if different.any() else None
        different = np.any(a["normal"] != b["normal"], axis=1)
        first_force = float(a["time"][np.flatnonzero(different)[0]]) if different.any() else None
    return dict(passed=all(checks.values()), checks=checks,
        first_action_difference_s=first_action, first_physical_force_difference_s=first_force,
        pickup_command_rule_identical=True, actual_pickup_trigger_times_may_differ=True)


def main():
    from v22_joint_task_fusion.numerics import ensure_numeric_environment, numerical_identity
    ensure_numeric_environment()
    from .runtime import ROOT, ARMS, SCENES, SEED, source_files, identity
    from .metrics import METRIC_SPEC
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--preflight", required=True)
    p.add_argument("--workers", type=int, default=4, choices=(1, 2, 3, 4))
    args = p.parse_args()
    out, preflight = Path(args.out).resolve(), Path(args.preflight).resolve()
    if out.exists():
        raise ValueError("Refusing to overwrite an existing experiment")
    if not out.is_relative_to(ROOT/"output"):
        raise ValueError("Pilot must be stored in the independent research output folder")
    gate = read(preflight/"preflight.json")
    if not gate["passed"]:
        raise ValueError("Cold-start integration preflight has not passed")
    sources = identity()
    for arm in ARMS:
        tested = read(preflight/arm/"metadata.json")["research_source_sha256"]
        for name in ("cold_start_take_v1/kernel.py", "cold_start_take_v1/runtime.py"):
            if tested[name] != sources[name]:
                raise ValueError("Cold-start runtime changed after the preflight")
    jobs = [dict(id=f"{arm}__{scene}__{SEED}", arm=arm, scene=scene, seed=SEED)
            for scene in SCENES for arm in ARMS]
    plan = dict(schema="cold_start_take_plan_v1", status="frozen_development_pilot",
        created_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs, workers=args.workers,
        duration_s=16., control_period_s=.005, physics_period_s=.0005,
        preflight=str(preflight), preflight_sha256=sha(preflight/"preflight.json"),
        sources_sha256=sources, numerics=numerical_identity(), metric_spec=METRIC_SPEC,
        preserved_results_sha256=preserved_results(ROOT,SCENES,SEED),
        scope="activate existing baseline/macro from open/no-contact state; same physical carrier and gains",
        no_parameter_search=True, no_unplanned_retries=True, stop_after_fixed_eight_runs=True,
        seed_status="same already-seen midterm development seed, not a new holdout",
        no_pooling_with_archived_prepared_grasp_experiments=True)
    out.mkdir(parents=True,exist_ok=False)
    write(out/"plan.json",plan)
    for key, path in source_files().items():
        if path.suffix in (".py", ".json"):
            target = out/"source_snapshot"/key
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    print(json.dumps(dict(status="frozen_and_started", jobs=len(jobs), workers=args.workers,
                          plan_sha256=sha(out/"plan.json")),ensure_ascii=False),flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        tasks = {executor.submit(worker,job,plan,out):job for job in jobs}
        for future in as_completed(tasks):
            try:
                result = future.result()
            except Exception as error:
                result = dict(status="worker_failure",job=tasks[future],error=str(error))
            results.append(result)
            write(out/"interim_jobs.json",results)
            brief = {k:result[k] for k in ("status","job","wall_seconds","error") if k in result}
            if "result" in result:
                brief["outcome"] = {k:result["result"][k] for k in ("pickup_completed","full_task_completed","force_limit_pass","reasons")}
            print(json.dumps(brief,ensure_ascii=False),flush=True)
    audit = dict(sources_unchanged=identity()==sources,
        old_results_unchanged=preserved_results(ROOT,SCENES,SEED)==plan["preserved_results_sha256"],
        planned_jobs=len(jobs), retained_jobs=len(results),
        evaluated_jobs=sum(r["status"]=="evaluated" for r in results),
        pairs={scene:pair_audit(out,scene,SEED) for scene in SCENES})
    audit["passed"] = bool(audit["sources_unchanged"] and audit["old_results_unchanged"] and
        audit["planned_jobs"]==audit["retained_jobs"]==audit["evaluated_jobs"] and
        all(p["passed"] for p in audit["pairs"].values()) and
        all(all(r["result"]["physical_audit"].values()) for r in results if r["status"]=="evaluated"))
    write(out/"audit.json",audit)
    write(out/"summary.json",dict(plan_sha256=sha(out/"plan.json"),jobs=results,audit=audit))
    print(json.dumps(dict(status="fixed_pilot_finished",audit_passed=audit["passed"],
                         evaluated_jobs=audit["evaluated_jobs"]),ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()

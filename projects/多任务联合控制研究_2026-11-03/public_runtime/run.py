"""New paired trials with frozen inputs; no archived results or certificates."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def worker(job, plan, output, config):
    from .numerics import ensure_numeric_environment
    numerics = ensure_numeric_environment()
    from .config import sha
    from cold_start_take_v1.runtime import make_session, identity
    from cold_start_take_v1.metrics import evaluate
    run_dir = Path(output) / "runs" / job["id"]
    began, session = time.perf_counter(), None
    try:
        if identity() != plan["sources_sha256"] or sha(config) != plan["parameters_sha256"]:
            raise RuntimeError("Runtime inputs changed after the new run plan was frozen")
        if numerics != plan["numerics"]:
            raise RuntimeError("Worker numerical environment differs from its run plan")
        session = make_session(job["arm"], job["scene"], job["seed"], runtime_config=config)
        while session.running:
            session.step_one_sensor_tick()
            if session.tick % 100 == 0 or not session.running:
                write(Path(output) / "progress" / (job["id"] + ".json"), dict(
                    job=job, simulated_time_s=session.time,
                    wall_seconds=time.perf_counter()-began, completed=session.completed,
                ))
        if not session.completed or session.tick != 3200:
            raise RuntimeError("Incomplete scheduled 16-second trial")
        session.save(run_dir)
        result = evaluate(run_dir)
        if identity() != plan["sources_sha256"] or sha(config) != plan["parameters_sha256"]:
            raise RuntimeError("Runtime inputs changed during the trial")
        response = dict(status="evaluated", job=job, result=result,
                        wall_seconds=time.perf_counter()-began)
        write(run_dir / "job.json", response)
        return response
    except Exception as exc:
        response = dict(status="runtime_failure", job=job, error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(), wall_seconds=time.perf_counter()-began,
                        simulated_time_s=None if session is None else session.time)
        if session is not None and session.trace and not run_dir.exists():
            try:
                session.running = False
                session.completed = False
                session.save(run_dir)
            except Exception as save_error:
                response["partial_save_error"] = str(save_error)
        write(run_dir / "runtime_error.json", response)
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="A new output directory; existing directories are never overwritten")
    parser.add_argument("--config", help="Optional public_hand_runtime_parameters_v1 JSON")
    parser.add_argument("--scenes", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[12345])
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    from .numerics import ensure_numeric_environment
    numerics = ensure_numeric_environment()
    from .config import DEFAULT_CONFIG, load_context, sha
    from cold_start_take_v1.runtime import ARMS, SCENES, identity
    from cold_start_take_v1.metrics import METRIC_SPEC
    from cold_start_take_v1.run import pair_audit
    config = Path(args.config).resolve() if args.config else DEFAULT_CONFIG
    load_context(config, "capsule")
    scenes = list(SCENES) if args.scenes == ["all"] else args.scenes
    if not scenes or len(scenes) != len(set(scenes)) or set(scenes) - set(SCENES):
        parser.error("Choose distinct scenes: all, nominal, friction_drop, fragile_conflict, collision")
    if len(args.seeds) != len(set(args.seeds)) or any(s < 0 or s > 2**32 - 1 for s in args.seeds):
        parser.error("Seeds must be distinct integers in [0, 2**32 - 1]")
    output = Path(args.out).resolve()
    if output.exists():
        parser.error("Output already exists; choose a fresh directory")
    jobs = [dict(id=f"{arm}__{scene}__{seed}", arm=arm, scene=scene, seed=seed)
            for seed in args.seeds for scene in scenes for arm in ARMS]
    plan = dict(
        schema="public_paired_runtime_plan_v1", created_utc=datetime.now(timezone.utc).isoformat(),
        status="frozen_before_execution", jobs=jobs, seeds=args.seeds, scenes=scenes,
        sources_sha256=identity(), parameters_sha256=sha(config), numerics=numerics,
        metric_spec=METRIC_SPEC, duration_s=16., workers=args.workers,
        historical_results_included=False, historical_physical_acceptance_inherited=False,
        independent_physical_acceptance="not_run", failed_assignments_retained=True,
        interpretation="New paired descriptive experiment; no inherited performance conclusion or real-hardware certification",
    )
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan)
    print(json.dumps(dict(status="started", trials=len(jobs), output=str(output)), ensure_ascii=False), flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(worker, job, plan, str(output), str(config)): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                record = future.result()
            except Exception as exc:
                record = dict(status="worker_failure", job=job, error=f"{type(exc).__name__}: {exc}")
                write(output / "runs" / job["id"] / "runtime_error.json", record)
            results.append(record)
            write(output / "interim_jobs.json", results)
            print(json.dumps(dict(completed=len(results), total=len(jobs), job=job, status=record["status"]), ensure_ascii=False), flush=True)
    pairs = {}
    for seed in args.seeds:
        for scene in scenes:
            try:
                pairs[f"{scene}__{seed}"] = pair_audit(output, scene, seed)
            except Exception as exc:
                pairs[f"{scene}__{seed}"] = dict(passed=False, reason=f"{type(exc).__name__}: {exc}")
    evaluated = [r for r in results if r["status"] == "evaluated"]
    summary = dict(
        schema="public_paired_runtime_summary_v1", jobs=results, pairs=pairs,
        assigned_jobs=len(jobs), evaluated_jobs=len(evaluated), failed_jobs=len(results)-len(evaluated),
        sources_unchanged=identity() == plan["sources_sha256"],
        parameters_unchanged=sha(config) == plan["parameters_sha256"],
        paired_conditions_checks=all(p["passed"] for p in pairs.values()),
        per_trial_record_checks=all(all(r["result"]["physical_audit"].values()) for r in evaluated) and len(evaluated) == len(jobs),
        independent_physical_acceptance="not_run", historical_acceptance_inherited=False,
    )
    write(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("assigned_jobs", "evaluated_jobs", "failed_jobs", "paired_conditions_checks", "per_trial_record_checks", "independent_physical_acceptance")}), flush=True)
    if summary["failed_jobs"] or not summary["paired_conditions_checks"] or not summary["per_trial_record_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Retained paired development trials; never a confirmation experiment."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, os, time, shutil

import research_imports
from research_imports import ROOT, FOUNDATION, source_identity
from session import SCENE_BY_NAME, check_admission, sha, jsonable
from evaluate_runs import evaluate_trace


def run_one(job):
    from runtime import ResearchSession
    began = time.perf_counter()
    session = ResearchSession(method=job['method'], scenario=job['scenario'],
        seed=job['seed'], split=job['split'])
    destination = Path(job['destination'])
    try:
        while session.running:
            session.step_one_sensor_tick()
    except Exception as exc:
        session.running=False
        session.completed=False
        session.failure_reason='research_runtime_failure'
        session.failure_at=session.time
        if session.trace:
            session.save(destination)
            (destination/'runtime_error.json').write_text(json.dumps(dict(
                type=type(exc).__name__,error=str(exc),time=session.time),indent=2),encoding='utf-8')
        raise
    session.save(destination)
    result = evaluate_trace(destination / 'trace.npz')
    try:
        from evaluation import evaluate_research
    except ImportError:
        evaluate_research = None
    if evaluate_research is not None:
        result['research_metrics'] = evaluate_research(destination / 'trace.npz')
    result['wall_seconds'] = time.perf_counter() - began
    (destination / 'result.json').write_text(json.dumps(jsonable(result), ensure_ascii=False,
        indent=2, allow_nan=False), encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--methods', nargs='+', default=['ratio', 'unified', 'joint'])
    p.add_argument('--scenarios', nargs='+', default=['nominal', 'sphere', 'contact_robustness',
        'dynamic_transport', 'fragile_conflict', 'sensor_noise'])
    p.add_argument('--seeds', nargs='+', type=int, default=[610101])
    p.add_argument('--physical-randomization', action='store_true')
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    if any(s not in SCENE_BY_NAME for s in a.scenarios):
        raise ValueError('Unknown scenario')
    if any(not 610000 <= s < 620000 for s in a.seeds):
        raise ValueError('Development seeds must be in the dedicated 610000..619999 range')
    destination = (ROOT / a.out).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError('Research output must stay inside the independent research directory')
    if destination.exists():
        raise FileExistsError('All development attempts retained; choose a new output')
    admission = check_admission()
    if not admission['passed']:
        raise RuntimeError(admission['problems'])
    identity = source_identity()
    split = 'robustness' if a.physical_randomization else 'development'
    jobs = [dict(method=m, scenario=s, seed=seed, split=split,
        destination=str(destination / 'runs' / f'{s}__{m}__{seed}'))
        for seed in a.seeds for s in a.scenarios for m in a.methods]
    destination.mkdir(parents=True)
    (destination/'sources').mkdir()
    for filename in identity:
        shutil.copy2(ROOT/filename,destination/'sources'/filename)
    manifest = dict(schema='independent_joint_research_development_v1',
        phase='development', performance_claim_allowed=False, jobs=jobs,
        created_utc=datetime.now(timezone.utc).isoformat(), research_source_sha256=identity,
        foundation=str(FOUNDATION), foundation_admission_sha256=admission['manifest_sha256'],
        physical_randomization=a.physical_randomization,
        note='Prior 256 confirmation traces were seen for diagnosis. These are development trials.')
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    results, failures = [], []
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(json.dumps(dict(completed=len(results), total=len(jobs), method=job['method'],
                    scenario=job['scenario'], success=result['task_success'], tv=result.get('finger_tv_rate'),
                    failure=result.get('failure_reason')), ensure_ascii=False), flush=True)
            except Exception as exc:
                failures.append(dict(job=job, error=type(exc).__name__ + ': ' + str(exc)))
                print(json.dumps(failures[-1], ensure_ascii=False), flush=True)
    (destination / 'trials.json').write_text(json.dumps(jsonable(results), ensure_ascii=False,
        indent=2, allow_nan=False), encoding='utf-8')
    finish = dict(execution_complete=len(results)==len(jobs), trials=len(results), planned=len(jobs),
        runtime_failures=failures, sources_unchanged=identity==source_identity(),
        trials_sha256=sha(destination / 'trials.json'), phase='development', performance_claim_allowed=False)
    (destination / 'execution_complete.json').write_text(json.dumps(finish, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(finish, ensure_ascii=False), flush=True)
    if failures or not finish['sources_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

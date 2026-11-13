"""Root-only launcher: input preflight -> exact3 bridge -> matched6 refinements."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import sys
import time
import traceback
from .source_record import ROOT, METHODS, load, read, sha
from .source_pinning import identity, ensure_original_sources, pin, require_lock, snapshot
from .reconstruction import verify_native_commands
from .replay import run_job, write

def preflight(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Fresh preflight report required')
    reports = []; initial_identity = identity()
    for method in METHODS:
        try:
            record = load(method); ensure_original_sources([record]); reports.append(verify_native_commands(record))
        except Exception as exc:
            reports.append(dict(method=method, passed=False, error=repr(exc), traceback=traceback.format_exc()))
    value = dict(schema='contact_peak_actual_input_preflight_v1', integration_steps=0,
                 passed=all(r['passed'] for r in reports), reports=reports,
                 candidate_sources_sha256=initial_identity, sources_unchanged=initial_identity == identity())
    value['passed'] &= value['sources_unchanged']; write(out, value)
    return value

def worker(method, dt, out):
    record = load(method); require_lock([record])
    return run_job(record, dt, out)

def execute(args):
    records = [load(method) for method in METHODS]; lock = require_lock(records)
    before = identity(); started = time.perf_counter()
    if args.phase == 'bridge':
        jobs = [(method, .0005) for method in METHODS]
    else:
        bridge = read(Path(args.bridge) / 'execution_complete.json')
        if bridge['phase'] != 'bridge' or not bridge['execution_complete'] or not bridge['bridge_all_exact'] or bridge['source_lock_sha256'] != sha(Path(__file__).with_name('source_lock.json')):
            raise ValueError('All three full16s same-lock exact physical bridges are mandatory; no bypass')
        jobs = [(method, dt) for method in METHODS for dt in (.00025, .000125)]
    target = Path(args.out); target.mkdir(parents=True, exist_ok=False)
    snapshot(target, lock)
    manifest = dict(schema='contact_peak_fixed_actual_reference_jobs_v1', phase=args.phase,
                    source_lock_sha256=sha(Path(__file__).with_name('source_lock.json')),
                    input_sources=lock['source_inputs'], jobs=[dict(method=m, dt=dt, destination=str(target / f'{m}__dt_{dt:.6f}')) for m, dt in jobs],
                    matched_input_not_policy_performance=True)
    write(target / 'manifest.json', manifest)
    # Avoid holding large input arrays in the parent while worker processes run.
    del records
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, job['method'], job['dt'], job['destination']): job for job in manifest['jobs']}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(method=job['method'], dt=job['dt'], complete=False, worker_failure=repr(exc), traceback=traceback.format_exc())
            results.append(result)
    results.sort(key=lambda r: (METHODS.index(r['method']), r['dt']))
    complete = len(results) == len(jobs) and all(r['complete'] for r in results)
    same_rng = {}
    for method in METHODS:
        states = {r.get('noise_rng_final') for r in results if r['method'] == method and r['complete']}
        if args.phase == 'refine':
            states.add(next(r['noise_rng_final'] for r in bridge['results'] if r['method'] == method))
        same_rng[method] = len(states) == 1 and None not in states
    value = dict(phase=args.phase, execution_complete=complete, results=results,
                 source_lock_sha256=manifest['source_lock_sha256'], sources_unchanged=before == identity(),
                 identical_sensor_rng_final_per_method=same_rng,
                 bridge_all_exact=args.phase == 'bridge' and complete and all(r['original_bridge_passed'] for r in results),
                 wall_seconds=time.perf_counter()-started,
                 analysis_scope='Complete fixed-actual-input numerical diagnostic; no new closed-loop strategy outcome or physical admission')
    write(target / 'execution_complete.json', value)
    return value

def main():
    p = argparse.ArgumentParser(description=__doc__)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--preflight', action='store_true'); choice.add_argument('--pin-sources', action='store_true'); choice.add_argument('--execute', action='store_true')
    p.add_argument('--out'); p.add_argument('--preflight-result'); p.add_argument('--phase', choices=('bridge', 'refine')); p.add_argument('--bridge')
    p.add_argument('--workers', type=int, default=3); args = p.parse_args()
    if args.preflight:
        if not args.out: p.error('--preflight requires fresh --out file')
        result = preflight(args.out); print(json.dumps(dict(passed=result['passed'], integration_steps=0, reports=len(result['reports'])), ensure_ascii=False)); return 0 if result['passed'] else 1
    if args.pin_sources:
        if not args.preflight_result: p.error('--pin-sources requires a passing --preflight-result')
        check = read(args.preflight_result)
        if not check['passed'] or check['candidate_sources_sha256'] != identity(): raise ValueError('Preflight is stale or failed')
        records = [load(m) for m in METHODS]; value = pin(records)
        print(json.dumps(dict(sources=len(value['sources']), native_binaries=len(value['environment']['native_package_binaries_sha256']), lock_sha256=sha(Path(__file__).with_name('source_lock.json'))))); return 0
    if not args.out or not args.phase or args.workers < 1: p.error('--execute requires --phase, fresh --out directory and positive --workers')
    if args.phase == 'refine' and not args.bridge: p.error('--phase refine requires exact --bridge directory')
    result = execute(args)
    passed = result['execution_complete'] and result['sources_unchanged'] and all(result['identical_sensor_rng_final_per_method'].values())
    if args.phase == 'bridge': passed &= result['bridge_all_exact']
    print(json.dumps(dict(phase=args.phase, complete=result['execution_complete'], passed=passed, wall_seconds=result['wall_seconds']), ensure_ascii=False))
    return 0 if passed else 1

if __name__ == '__main__':
    sys.exit(main())

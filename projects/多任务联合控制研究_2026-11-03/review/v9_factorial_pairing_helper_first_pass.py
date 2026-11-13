"""Terminal-only retained-data audit. No simulator/controller imports or steps.

Run once both factorial batches are terminal. It never edits execution inputs.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {m: ROOT / f'output/development_v9_{m}_v1' for m in ('hold', 'transport')}
OLD = ROOT / 'output/development_v8_counterexamples_v1'
OUT = ROOT / 'review/v9_factorial_pairing_independent_v1.json'
REGISTERED = ROOT / 'review/v9_factorial_gate_plan_v1.json'
METHODS = ['ratio', 'passive', 'ratio_preview', 'passive_preview', 'joint']
SCENES = ['nominal', 'friction_drop', 'fragile_conflict', 'collision']
SEED = 610212
RATES = np.array([.035, .12, .12])
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def numeric_keys(path):
    names = set()
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}[version]
                _, _, dtype = reader(stream)
            if dtype.kind in 'biufc':
                names.add(name.removesuffix('.npy'))
    return names


checks, issues = [], []


def check(name, ok, detail=None):
    row = dict(name=name, passed=bool(ok), detail=detail)
    checks.append(row)
    if not ok:
        issues.append(row)


# Do not inspect a complete matrix before BOTH independently recorded terminals.
terminals = {}
for mode, run in RUNS.items():
    p = run / 'execution_complete.json'
    terminals[mode] = read(p) if p.exists() else None
pending = [m for m, f in terminals.items()
           if f is None or f.get('execution_complete') is not True]
if pending:
    print(json.dumps(dict(status='awaiting_terminal', pending_modes=pending,
                          complete_matrix_read=False, physics_steps=0)), flush=True)
    raise SystemExit(2)
if OUT.exists():
    raise FileExistsError('Independent evidence already exists; preserve it: ' + str(OUT))

registered = read(REGISTERED)
check('registered_40_seen_seed_factorial',
      registered['methods'] == METHODS and registered['environment_scenarios'] == SCENES
      and registered['public_motion_modes'] == list(RUNS)
      and registered['physical_seeds'] == [SEED]
      and registered['physical_randomization'] is True
      and registered['expected_trials'] == 40
      and registered['performance_claim_allowed'] is False)
key = lambda x: (x['scenario'], x['method'], x['seed'])
expected_keys = {(s, m, SEED) for s in SCENES for m in METHODS}
current_sha = {}
batches, metadata, trace_paths, selected = {}, {}, {}, {}
immutable_fields = ['time_s', 'motion_axes', 'motion_target', 'tracking_tolerances',
                    'deadline_s', 'release_at_s', 'release_deadline_s',
                    'motion_preview', 'motion_reference_bounds', 'sensor_characteristics']
trajectory_fields = ['time_s', 'motion_axes', 'motion_target', 'deadline_s',
                     'release_at_s', 'release_deadline_s', 'motion_preview']
public_summaries, prep, motion = [], [], []
for mode, run in RUNS.items():
    start = len(checks)
    manifest, plan, rows, ledger, snap = [read(run / f) for f in
        ['manifest.json', 'plan.json', 'trials.json', 'assignment_ledger.json',
         'source_snapshot_manifest.json']]
    jobs = manifest['jobs']
    bykey = {key(x): x for x in rows}
    finish = terminals[mode]
    check(mode + ':exact20planned_keys', len(jobs) == len(rows) == len(ledger) == 20
          and len(bykey) == 20 and set(map(key, jobs)) == expected_keys
          and Counter(map(key, jobs)) == Counter(map(key, rows))
          == Counter(key(x['job']) for x in ledger) and plan['jobs'] == jobs)
    check(mode + ':all_assignments_retained_and_evaluated',
          all(x['status'] == 'evaluated' for x in ledger))
    check(mode + ':terminal_links', finish['trials'] == finish['planned'] == 20
          and finish['runtime_failures'] == [] and finish['sources_unchanged'] is True
          and all(finish[k] == sha(run / f) for k, f in [
              ('trials_sha256', 'trials.json'), ('assignment_ledger_sha256', 'assignment_ledger.json'),
              ('plan_sha256', 'plan.json'), ('manifest_sha256', 'manifest.json')]))
    check(mode + ':mode_and_warmup_provenance', manifest['motion_mode'] == plan['motion_mode'] == mode
          and plan['methods'] == METHODS and plan['scenarios'] == SCENES
          and plan['seeds'] == [SEED] and plan['physical_randomization'] is True
          and plan['observer_warmup_s'] == dict(joint=.2, traditional=.2)
          and all(j['motion_mode'] == mode and j['split'] == 'robustness' for j in jobs))
    sources = []
    for name, expected in manifest['research_source_sha256'].items():
        entry = snap['files'][name]
        source, original = run / entry['snapshot_path'], Path(entry['original_path'])
        if str(original) not in current_sha:
            current_sha[str(original)] = sha(original)
        observed = sha(source)
        sources.append(dict(name=name, original_path=str(original), expected=expected,
                            snapshot=observed, current=current_sha[str(original)],
                            matches=expected == entry['sha256'] == observed == current_sha[str(original)]
                            and source.stat().st_size == original.stat().st_size == entry['bytes']))
    check(mode + ':all77_source_snapshot_current_identities',
          len(sources) == 77 and all(x['matches'] for x in sources))
    actual_source_files = {p.relative_to(run / 'sources').as_posix()
                           for p in (run / 'sources').rglob('*') if p.is_file()}
    check(mode + ':source_set_bindings_and_actual_dependency_namespace',
          set(snap['files']) == set(manifest['research_source_sha256']) == actual_source_files
          and {'v9/runtime.py', 'v9/run_development.py', 'v9/controller.py', 'v9/feedback.py',
               'v9/__init__.py'} <= actual_source_files
          and 'v7/belief.py' not in actual_source_files
          and 'v8/controller.py' not in actual_source_files
          and manifest['source_snapshot_manifest_sha256'] == sha(run / 'source_snapshot_manifest.json')
          and manifest['plan_sha256'] == sha(run / 'plan.json'))
    records = []
    groups = defaultdict(dict)
    for job in jobs:
        k = key(job)
        folder = Path(job['destination'])
        m, r = read(folder / 'metadata.json'), read(folder / 'result.json')
        metadata[(mode, *k)] = m
        trace_paths[(mode, *k)] = path = folder / 'trace.npz'
        groups[(k[0], k[2])][k[1]] = m
        u = np.random.default_rng(k[2] + 511000).normal(size=4)
        caps = [3.5, 2.8, 2.8, 2.8, 2.8] if k[0] == 'fragile_conflict' else [5.] * 5
        record = dict(key=list(k), result_equals_trial=r == bykey[k],
                      metadata_key_matches=m['method'] == k[1] and m['scenario']['name'] == k[0]
                      and m['seed'] == k[2] and m['split'] == 'robustness',
                      mode_recorded=m['motion_mode'] == r['motion_mode'] == mode,
                      source_matches=m['research_source_sha256'] == manifest['research_source_sha256'],
                      physical_parameters_match=m['config'] == r['research_metrics']['physical_parameters'],
                      rng_formula_matches=m['config']['mass'] == float(.12 * np.clip(1 + .06 * u[0], .85, 1.15))
                      and m['config']['friction'] == float(.6 * np.clip(1 + .08 * u[1], .8, 1.2)),
                      noise_profile_matches=all(m['noise'][a] == b for a, b in {
                          'force_sigma': .01, 'tangential_sigma': .005, 'position_sigma': .0001,
                          'slip_sigma': .0002, 'effort_sigma': .02, 'delay': .01,
                          'dropout_probability': 0., 'dropout_windows': []}.items()),
                      material_caps_match=m['limits'] == caps and m['budget'] == (12. if k[0] == 'fragile_conflict' else 18.),
                      warmup_matches=m['observer_warmup_s'] == .2,
                      trace_sha_linkage=r['trace_sha256'] == m['trace_sha256'] == r['research_metrics']['trace_sha256'],
                      metadata_sha256=sha(folder / 'metadata.json'), result_sha256=sha(folder / 'result.json'))
        records.append(record)
        # Cache each selected archive column ONCE; never repeatedly decompress JSON rows.
        with np.load(path, allow_pickle=False) as z:
            arrays = {f: z[f] for f in ['time', 'environment_json', 'actual_motion_target',
                                       'executed_motion_reference', 'task_request_json']}
        env = arrays['environment_json']
        task_full, trajectory_only = hashlib.sha256(), hashlib.sha256()
        for encoded in arrays.pop('task_request_json'):
            request = json.loads(str(encoded))
            for h, fields in [(task_full, immutable_fields), (trajectory_only, trajectory_fields)]:
                h.update(json.dumps({f: request[f] for f in fields}, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n')
        summary = dict(mode=mode, scenario=k[0], method=k[1], seed=k[2],
                       samples=len(arrays['time']), immutable_task_sha256=task_full.hexdigest(),
                       trajectory_clock_preview_sha256=trajectory_only.hexdigest(),
                       actual_motion_target_sha256=array_sha(arrays['actual_motion_target']),
                       environment_rows_sha256=hashlib.sha256('\n'.join(env.tolist()).encode('utf-8')).hexdigest())
        public_summaries.append(summary)
        selected[(mode, *k)] = arrays
    check(mode + ':all20_result_meta_parameters', all(all(v for k, v in x.items()
          if k not in ['key', 'metadata_sha256', 'result_sha256']) for x in records))
    paired = []
    for (scene, seed), mm in sorted(groups.items()):
        base = mm['ratio']
        fields = ['config', 'noise', 'limits', 'budget', 'model_sha256', 'initial_position']
        equality = {f: all(m[f] == base[f] for m in mm.values()) for f in fields}
        public = {f: all(m['task_contract'][f] == base['task_contract'][f] for m in mm.values())
                  for f in ['user_motion_trajectory', 'motion_reference_bounds', 'sensor_characteristics',
                            'tracking_tolerances']}
        ss = [x for x in public_summaries if x['mode'] == mode and x['scenario'] == scene]
        paired.append(dict(scenario=scene, seed=seed, physical_noise_material_exact=equality,
                           public_context_exact=public,
                           full_public_task_and_environment_exact=all(len({x[f] for x in ss}) == 1
                               for f in ['immutable_task_sha256', 'actual_motion_target_sha256', 'environment_rows_sha256']),
                           actual_baseline_gains_exact=mm['ratio']['gains'] == mm['ratio_preview']['gains']
                           and mm['passive']['gains'] == mm['passive_preview']['gains']
                           and mm['joint']['actual_ratio_task_gains'] == mm['ratio']['gains']))
    check(mode + ':all4_five_method_pairs', all(all(x['physical_noise_material_exact'].values())
          and all(x['public_context_exact'].values()) and x['full_public_task_and_environment_exact']
          and x['actual_baseline_gains_exact'] for x in paired))
    same_mode_public = [x for x in public_summaries if x['mode'] == mode]
    check(mode + ':public_trajectory_independent_of_environment_all20',
          len({x['trajectory_clock_preview_sha256'] for x in same_mode_public}) == 1
          and len({x['actual_motion_target_sha256'] for x in same_mode_public}) == 1)
    for scene in SCENES:
        p = selected[(mode, scene, 'ratio_preview', SEED)]
        j = selected[(mode, scene, 'joint', SEED)]
        t = j['time']; mask = (t >= 7.5 - 1e-9) & (t < 12 - 1e-9)
        a, b = j['executed_motion_reference'], p['executed_motion_reference']
        target = j['actual_motion_target']; delta = np.diff(a, axis=0, prepend=a[:1])
        target_delta = np.diff(target, axis=0, prepend=target[:1])
        moving = np.any(abs(target_delta) > 1e-10, axis=1) & mask
        stationary = np.all(abs(delta) <= 1e-10, axis=1)
        strong_rates = {}
        for method in ['ratio_preview', 'passive_preview', 'joint']:
            aa = selected[(mode, scene, method, SEED)]['executed_motion_reference']
            dd = np.diff(aa, axis=0, prepend=aa[:1])
            strong_rates[method] = (abs(dd[mask]) > RATES * .005 + 1e-9).sum(axis=0).tolist()
        motion.append(dict(mode=mode, scenario=scene, holding_samples=int(mask.sum()),
                           common_user_target_exact=np.array_equal(target, p['actual_motion_target']),
                           max_abs_joint_vs_ratio_preview_reference_difference=abs(a[mask] - b[mask]).max(axis=0).tolist(),
                           rms_joint_vs_ratio_preview_reference_difference=np.sqrt(np.mean((a[mask] - b[mask]) ** 2, axis=0)).tolist(),
                           joint_stationary_reference_fraction=float(stationary[mask].mean()),
                           user_moving_samples=int(moving.sum()),
                           joint_stationary_while_user_moves_fraction=None if not moving.any() else float(stationary[moving].mean()),
                           strong_methods_reference_rate_violations=strong_rates,
                           interpretation='Stationarity is an actual-reference measure, not a controller risk-hold flag or a safety/causal-benefit claim.'))
    for scene in ['nominal', 'fragile_conflict']:
        fields = ['qpos', 'qvel', 'alpha_before', 'alpha', 'ctrl_before', 'ctrl',
                  'lift', 'lift_ref', 'controller_active']
        with np.load(trace_paths[(mode, scene, 'ratio', SEED)], allow_pickle=False) as z:
            mask = z['time'] < 7.5 - 1e-9
            a = {f: z[f][mask] for f in fields}
        comparisons = []
        for method in METHODS[1:]:
            with np.load(trace_paths[(mode, scene, method, SEED)], allow_pickle=False) as z:
                eq = {f: bool(np.array_equal(a[f], z[f][mask], equal_nan=True)) for f in fields}
            comparisons.append(dict(method=method, fields_exact=eq, all_exact=all(eq.values())))
        prep.append(dict(mode=mode, scenario=scene, window='t<7.5-1e-9', samples=int(mask.sum()),
                         ratio_reference_array_sha256={f: array_sha(a[f]) for f in fields},
                         comparisons=comparisons))
    batches[mode] = dict(terminal=finish, input_sha256={f: sha(run / f) for f in [
        'execution_complete.json', 'manifest.json', 'plan.json', 'trials.json',
        'assignment_ledger.json', 'source_snapshot_manifest.json']},
        research_source_sha256=manifest['research_source_sha256'], source_checks=sources,
        result_checks=records, five_method_pairs=paired,
        outcomes_from_retained_result=dict(assigned=len(jobs), retained_results=len(rows),
            task_success=sum(x.get('task_success') is True for x in rows),
            prepared=sum(metadata[(mode, *key(x))]['handoff_passed'] is True for x in rows),
            completed=sum(metadata[(mode, *key(x))]['completed'] is True for x in rows)),
        audit_check_names=[x['name'] for x in checks[start:]])

check('both_batches_identical_77_execution_identities',
      batches['hold']['research_source_sha256'] == batches['transport']['research_source_sha256'])
cross_mode = []
for scene in SCENES:
    for method in METHODS:
        a, b = metadata[('hold', scene, method, SEED)], metadata[('transport', scene, method, SEED)]
        fields = ['config', 'noise', 'limits', 'budget', 'model_sha256', 'initial_position',
                  'gains', 'actual_ratio_task_gains', 'observer_warmup_s']
        eq = {f: a[f] == b[f] for f in fields}
        aa, bb = selected[('hold', scene, method, SEED)], selected[('transport', scene, method, SEED)]
        cross_mode.append(dict(scenario=scene, method=method, metadata_equal=eq,
                               full_environment_exact=bool(np.array_equal(aa['environment_json'], bb['environment_json'])),
                               full_clock_exact=bool(np.array_equal(aa['time'], bb['time']))))
check('all20_cross_motion_same_environment_physical_noise_material',
      all(all(x['metadata_equal'].values()) and x['full_environment_exact'] and x['full_clock_exact'] for x in cross_mode))
check('representative_preparation_exact_all_methods',
      all(c['all_exact'] for p in prep for c in p['comparisons']))
check('joint_public_target_preserved_and_all_strong_reference_rates_respected',
      all(x['common_user_target_exact'] and all(v == [0, 0, 0] for v in x['strong_methods_reference_rate_violations'].values()) for x in motion))

# Compare only genuine retained old tasks, after proving actual environment and target equality.
baseline = []
correspondence = [('hold', s, s) for s in ['nominal', 'fragile_conflict', 'collision']]
correspondence.append(('transport', 'nominal', 'dynamic_transport'))
for mode, scene, old_scene in correspondence:
    for method in METHODS[:-1]:
        new = trace_paths[(mode, scene, method, SEED)]
        old_folder = OLD / 'runs' / f'{old_scene}__{method}__{SEED}'
        old = old_folder / 'trace.npz'; old_meta = read(old_folder / 'metadata.json')
        new_meta = metadata[(mode, scene, method, SEED)]
        context_fields = ['config', 'noise', 'limits', 'budget', 'model_sha256', 'initial_position', 'gains']
        with np.load(old, allow_pickle=False) as z:
            context_exact = all(new_meta[f] == old_meta[f] for f in context_fields)
            target_exact = bool(np.array_equal(selected[(mode, scene, method, SEED)]['actual_motion_target'], z['actual_motion_target']))
            environment_exact = bool(np.array_equal(selected[(mode, scene, method, SEED)]['environment_json'], z['environment_json']))
        row = dict(mode=mode, scenario=scene, method=method, old_scenario=old_scene,
                   matched_context=context_exact, matched_user_target=target_exact,
                   matched_environment=environment_exact,
                   old_metadata_sha256=sha(old_folder / 'metadata.json'), excluded_numeric_fields=['control_ms'])
        if context_exact and target_exact and environment_exact:
            ak, bk = numeric_keys(new) - {'control_ms'}, numeric_keys(old) - {'control_ms'}
            differences, hashes = [], {}
            with np.load(new, allow_pickle=False) as a, np.load(old, allow_pickle=False) as b:
                for field in sorted(ak & bk):
                    x, y = a[field], b[field]
                    same = x.dtype == y.dtype and x.shape == y.shape and x.tobytes(order='C') == y.tobytes(order='C')
                    hashes[field] = array_sha(x)
                    if not same:
                        differences.append(dict(field=field, numeric_equal=bool(np.array_equal(x, y, equal_nan=True)),
                                                new_shape=list(x.shape), old_shape=list(y.shape)))
            row.update(numeric_fields=len(ak), numeric_field_set_exact=ak == bk,
                       all_selected_numeric_arrays_bit_exact=ak == bk and not differences,
                       mismatches=differences, new_numeric_array_sha256=hashes)
        else:
            row['comparison_status'] = 'not_a_matching_retained_old_task_no_exact_claim'
        baseline.append(row)
check('all16_verified_old_task_counterparts_bit_exact', len(baseline) == 16
      and all(x.get('all_selected_numeric_arrays_bit_exact') is True for x in baseline),
      [x for x in baseline if x.get('all_selected_numeric_arrays_bit_exact') is not True])

report = dict(schema='v9_factorial_pairing_independent_v1',
              created_utc=datetime.now(timezone.utc).isoformat(),
              checks=checks, issues=issues, registration_sha256=sha(REGISTERED),
              batches=batches, cross_motion_environment_pairs=cross_mode,
              full_public_task_schedule_checks=public_summaries,
              joint_reference_deviation_and_stationarity=motion,
              representative_preparation=prep, genuine_old_baseline_counterparts=baseline,
              scope=dict(full=['40 planned/result/ledger/meta identities and failure denominators',
                               '77 execution snapshot/current byte hashes per batch; both source maps identical',
                               '40 actual parameter/noise profile/material/warmup records and physical RNG mass/friction formula',
                               '40 full public task target/preview/clock sequences and current environment sequences',
                               '20 cross-motion environment/config/noise/material pairs',
                               '8 joint-vs-ratio-preview holding reference deviations, targets and three strong-method rate limits',
                               '16 genuine V8 traditional counterparts: all numeric/bool arrays excluding control_ms'],
                         representative=['2 environments per mode, all5 methods,9 preparation physics/command columns before7.5s'],
                         not_checked=['full compressed trace SHA; trace hash linkages alone are checked',
                                      'noise packet replay or independent generator reconstruction of every sample',
                                      'V3 FIR/TV/force safety/physical tracking metrics, owned by parent',
                                      'risk-hold debug state semantics; actual reference stationarity is separately measured',
                                      'new physical trials or control optimization']),
              physics_steps=0, production_or_old_evidence_modified=False,
              interpretation='Seen-seed development factorial, not independent confirmation. Different public motion is crossed with unchanged actual environment schedule. Exact old baseline claims are limited to proven same tasks. Joint is allowed to alter execution for risk; preserved user targets do not imply achieved tracking, safety, or benefit. No global model admission.')
with OUT.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps(dict(output=str(OUT), sha256=sha(OUT), checks=len(checks),
                      passed=sum(x['passed'] for x in checks), issues=issues,
                      outcomes={m: b['outcomes_from_retained_result'] for m, b in batches.items()},
                      motion=motion), ensure_ascii=False), flush=True)

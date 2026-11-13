"""Complete16 V20 execution review: retained3 receipts + terminal new13.

No simulation, no solver replay, no partial performance reads. The frozen
small-run reader supplies cheap task-expression checks; this file owns roster,
multi-job batches, original80 context pairing and explicit stage denominators.
Geometry/adapter author review of execution, not independent geometry proof.
"""
from pathlib import Path
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch
import argparse, gzip, hashlib, importlib.util, json, sys, traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOUNDATION = ROOT.parent/'五指策略接入_2026-11-03'
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(FOUNDATION)); sys.dont_write_bytecode = True
PLAN = ROOT/'review/v20_complete16_plan_v1.json'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def source_path(k): return FOUNDATION/k[11:] if k.startswith('foundation/') else ROOT/k
def load_module(name, p):
    spec = importlib.util.spec_from_file_location(name, p); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def terminal_gate(plan, reader=read):
    states = []
    for s in plan['stages']:
        try: done = reader(ROOT/s['output']/'execution_complete.json')
        except FileNotFoundError: done = None
        states.append(dict(stage=s['name'], planned=s['planned'], complete=done is not None and done.get('execution_complete') is True))
    return dict(ready=bool(states) and all(s['complete'] for s in states), new_records_planned=sum(s['planned'] for s in states),
                states=states, branch_results_read=False, pending_not_physical_failure=True)
def metadata_scene(meta):
    value = meta.get('scenario')
    return value.get('name') if isinstance(value, dict) else value
def full17_comparator(old_plan, scene, motion, seed):
    stages = [s for s in old_plan['stages'] if s['arm'] == 'full_V17' and s['motion_mode'] == motion and scene in s['scenarios'] and seed in s['seeds']]
    if len(stages) != 1: raise ValueError('Exactly one original FullV17 context required')
    return ROOT/stages[0]['output']/'runs'/f'{scene}__ratio_effort__{seed}'
def classify_decision(q, counts):
    if q['phase'] not in ('TRANSFER', 'SEPARATE', 'UNSUPPORTED_EXIT'):
        counts['direct_search_capture'] += 1; return
    t = q.get('contact_tasks'); h = q.get('hierarchy'); error = q.get('numerical_error')
    P = len(t['pressure_P_N']) if t is not None and t.get('available') else None
    role = 'positive_pressure' if P else 'known_P0' if P == 0 else 'geometry_unknown'
    counts[role+':attempted'] += 1
    if q['geometry_proposal_accepted']:
        counts[role+':accepted'] += 1; return
    if t is None or not t.get('available'): kind = 'adapter_unknown'
    elif error: kind = 'emitted_score' if 'emitted_reference' in error or 'roundoff' in error else 'native_value' if 'native actuation' in error else 'other_explicit_exception'
    elif h is not None and not h['diagnostics'].get('actual_increment_available'): kind = 'solver_uncertified'
    else: kind = 'other_unaccepted'
    counts[role+':fallback:'+kind] += 1
    counts['fallback_status:'+('None' if h is None else h['status'])] += 1


def execute(plan, out, md):
    readiness = terminal_gate(plan)
    if not readiness['ready']:
        print(json.dumps(dict(status='PENDING', terminal_gate=readiness))); return 2
    if out.exists() or md.exists(): raise FileExistsError('Fresh full16 evidence required')
    helper_path = ROOT/'review/read_v20_small_execution_v1.py'
    helper = load_module('v20_small_formula_review', helper_path)
    prefix_path = ROOT/'review/v16_small_prefix_storage_independent_v1.py'
    prefix = load_module('v20_full16_prefix37', prefix_path)
    checks, issues, rows, inputs, forbidden = [], [], [], {}, []
    def ck(name, ok, details=None):
        checks.append(dict(name=name, passed=bool(ok), details=details))
        if not ok: issues.append(dict(name=name, details=details))
    def h(p):
        p = str(Path(p).resolve())
        if p not in inputs: inputs[p] = sha(p)
        return inputs[p]
    source = plan['source_sha256']; pins = plan['evidence_pins']
    for name, pin in pins.items(): ck('pin/'+name, h(ROOT/pin['path']) == pin['sha256'])
    small = read(ROOT/pins['small_plan']['path']); audit = read(ROOT/pins['execution_gate']['path']); aux = read(ROOT/pins['small_auxiliary']['path'])
    ck('retained initial execution and v3 links', audit['passed'] and audit['all_assigned_checked'] and audit['assigned_count'] == len(audit['rows']) == 4 and not audit['issues'] and audit['source_sha256'] == source and audit['plan_sha256'] == pins['small_plan']['sha256'] and aux['complete'] and not aux['issues'] and aux['source_sha256'] == source and aux['plan_sha256'] == pins['small_plan']['sha256'])
    ck('same reviewed cheap formula helper', h(helper_path) == audit['reader_sha256'])
    parent = read(ROOT/'review/v19_small_plan_v1.json')['source_sha256']; nomotion = read(ROOT/'review/v19_motion_ablation_plan_v1.json')['source_sha256']
    ck('source160 and preserved parent152 NoMotion156', len(source) == 160 and len(parent) == 152 and len(nomotion) == 156 and all(source.get(k) == v for k, v in parent.items()) and all(h(source_path(k)) == v for k, v in source.items()) and all(h(source_path(k)) == v for k, v in nomotion.items()))
    ck('exact same42 native environment', plan['environment'] == small['environment'] and len(plan['environment']['native_binaries_sha256']) == 42 and all(h(k) == v for k, v in plan['environment']['native_binaries_sha256'].items()))
    roster = plan['roster']; expected = {tuple(x['record_key']) for x in roster}
    new_roster = [x for x in roster if x['origin'] == 'new_remaining13']; retained_roster = [x for x in roster if x['origin'] == 'retained_initial3']
    ck('all16 assigned unique retained3 plus13 no repeat', len(roster) == len(expected) == plan['combined_count'] == 16 and len(new_roster) == plan['new_records'] == 13 and len(retained_roster) == plan['retained_count'] == 3 and readiness['new_records_planned'] == 13)
    old_plan = read(ROOT/'review/v17b_prospective80_plan_v1.json')
    retained_stage_map = {x['stage']: x for x in audit['rows']}
    for retained in plan['retained']:
        row = dict(record_key=retained['record_key'], origin='retained_initial3', status='unassessed', run=retained['run'])
        rows.append(row); tag = '/'.join(map(str, retained['record_key']))
        try:
            ck(tag+'/retained original run files', retained['rerun'] is False and all(h(ROOT/k) == v for k, v in retained['files_sha256'].items()))
            ck(tag+'/retained completion and snapshot manifest', h(ROOT/retained['completion_path']) == retained['completion_sha256'] and h(ROOT/retained['source_snapshot_manifest_path']) == retained['source_snapshot_manifest_sha256'])
            snapshot = read(ROOT/retained['source_snapshot_manifest_path']); base = (ROOT/retained['source_snapshot_manifest_path']).parent
            ck(tag+'/retained actual full snapshot160', set(snapshot['files']) == set(source) and all(h(base/snapshot['files'][k]['snapshot_path']) == v for k, v in source.items()))
            inherited = retained_stage_map[retained['stage']]
            ck(tag+'/previous actual run hashes still exact', inherited['status'] == 'audited' and h(ROOT/retained['run']/'trace.npz') == inherited['trace_sha256'] and h(ROOT/retained['run']/'metadata.json') == inherited['metadata_sha256'] and h(ROOT/retained['run']/'result.json') == inherited['result_sha256'])
            row.update(status='audited_retained_by_exact_hash_and_passed_receipt', previous_stage=retained['stage'], previous_execution_gate=pins['execution_gate'],
                       previous_counts=inherited['counts'], original_task_success=inherited['original_task_success'], original_full_motion=inherited['original_full_motion'], original_force_guard=inherited['original_force_guard'],
                       raw_recomputed_this_run=False, original_actual5ms_recomputed_in_small_gate=True)
        except Exception as exc:
            row.update(status='reader_error', error=repr(exc)); issues.append(dict(name=tag+'/retained_reader_error', error=repr(exc), traceback=traceback.format_exc()))
    import research_imports, mujoco, hand_plant
    from foundation_config import FoundationConfig
    from v13_a1.codec import ReferenceActor21
    from v14_coord.design import reference_rates
    def deny(name):
        def stop(*a, **kw): forbidden.append(name); raise AssertionError('Forbidden '+name)
        return stop
    with ExitStack() as guards:
        for name in ('MjData', 'mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_forwardSkip', 'mj_kinematics', 'mj_comPos', 'mj_collision', 'mj_contactForce'):
            if hasattr(mujoco, name): guards.enter_context(patch.object(mujoco, name, deny(name)))
        guards.enter_context(patch.object(hand_plant.HandPlant, '__init__', deny('HandPlant')))
        guards.enter_context(patch.object(ReferenceActor21, 'apply', deny('actor.apply')))
        for stage in plan['stages']:
            tag = stage['name']; folder = ROOT/stage['output']; assignment = stage['assignment']; planned = stage['planned']
            try:
                done = read(folder/'execution_complete.json'); reg = read(ROOT/stage['registration']); manifest = read(folder/'manifest.json'); snapshot = read(folder/'source_snapshot_manifest.json')
                trials = read(folder/'trials.json'); ledger = read(folder/'assignment_ledger.json'); failures = read(folder/'runtime_failures.json')
                keys = {(s, m, int(se)) for s in assignment['scenarios'] for m in assignment['methods'] for se in assignment['seeds']}
                ck(tag+'/all assigned no failure filtering', done['execution_complete'] and done['sources_unchanged'] and done['planned'] == done['trials'] == planned == len(trials) == len(ledger) == len(keys) and not failures and {(t['scenario'], t['method'], t['seed']) for t in trials} == keys)
                ck(tag+'/registration source environment', reg['assignment'] == assignment and reg['source_sha256'] == source and reg['environment'] == plan['environment'] and manifest['research_source_sha256'] == source and read(folder/'environment.json')['pinned_environment'] == plan['environment'])
                ck(tag+'/snapshot160 bytes', set(snapshot['files']) == set(source) and all(h(folder/snapshot['files'][k]['snapshot_path']) == v for k, v in source.items()) and manifest['source_snapshot_manifest_sha256'] == h(folder/'source_snapshot_manifest.json'))
                ck(tag+'/terminal all manifest filehashes', all(done[k+'_sha256'] == h(folder/(k+'.json')) for k in ('trials', 'assignment_ledger', 'plan', 'manifest')))
            except Exception as exc:
                issues.append(dict(name=tag+'/batch_header', error=repr(exc), traceback=traceback.format_exc())); continue
            for scene, method, seed in sorted(keys):
                motion = assignment['motion_mode']; key = ('full_V20', scene, motion, seed)
                run = folder/'runs'/f'{scene}__{method}__{seed}'; label = '/'.join(map(str, key))
                row = dict(record_key=list(key), origin='new_remaining13', run=str(run.relative_to(ROOT)), batch=tag, status='unassessed'); rows.append(row)
                try:
                    assigned = [x for x in new_roster if tuple(x['record_key']) == key]
                    ck(label+'/exact assigned original output', len(assigned) == 1 and assigned[0]['run'] == run.relative_to(ROOT).as_posix())
                    meta = read(run/'metadata.json'); result = read(run/'result.json'); trial = next(t for t in trials if (t['scenario'], t['method'], t['seed']) == (scene, method, seed))
                    ck(label+'/result metadata input identities', result == trial and metadata_scene(meta) == scene and meta['seed'] == seed and meta['motion_mode'] == motion and meta['research_source_sha256'] == source and result['trace_sha256'] == meta['trace_sha256'] == h(run/'trace.npz'))
                    with np.load(run/'trace.npz', allow_pickle=False) as z: a = {k: z[k] for k in z.files}
                    time = a['time']; active = a['controller_active'].astype(bool)
                    ck(label+'/full16 raw clock coverage', meta['completed'] and np.array_equal(time, np.arange(3201)*.005) and len(a['substep_time']) == 32000 and np.isfinite(a['qpos']).all() and np.isfinite(a['qvel']).all() and np.isfinite(a['ctrl']).all())
                    comparator = full17_comparator(old_plan, scene, motion, seed); oldmeta = read(comparator/'metadata.json')
                    physical_keys = ('shape', 'scenario', 'seed', 'split', 'config', 'initial_position', 'limits', 'budget', 'noise', 'model_sha256', 'motion_mode', 'observer_warmup_s')
                    ck(label+'/same original physical randomization and device contract', all(meta.get(k) == oldmeta.get(k) for k in physical_keys))
                    with np.load(comparator/'trace.npz', allow_pickle=False) as z: b = {k: z[k] for k in tuple(prefix.COLS)+('observation_json',)}
                    comparison = prefix.compare(a, b, time, b['time'], True)
                    ck(label+'/strict37 pre12 payload', all(x['exact_payload_equal'] for x in comparison), [x for x in comparison if not x['exact_payload_equal']])
                    before = time < 12-1e-9; before_old = b['time'] < 12-1e-9
                    xobs, yobs = a['observation_json'][before], b['observation_json'][before_old]
                    full_obs_exact = xobs.shape == yobs.shape and all(str(x).encode('utf-8') == str(y).encode('utf-8') for x, y in zip(xobs, yobs))
                    ck(label+'/complete public observation JSON pre12 no field stripping', full_obs_exact)
                    model, xmlsha = hand_plant._model(FoundationConfig(**meta['config'])); actor = ReferenceActor21(model)
                    ports = a['executed_reference21'].copy(); ports[:, 18] += actor.lift_bias_m
                    ck(label+'/actual21 sole encoding and incoming actual21', xmlsha == meta['model_sha256'] and helper.bits(ports, a['ctrl'][:, actor.indices21]) and helper.bits(a['robot_actuator_target21'], a['ctrl_before'][:, actor.indices21]))
                    obs = [json.loads(str(s)) for s in a['observation_json']]; debug = [json.loads(str(s)) for s in a['debug_json']]
                    errors = []
                    for i in np.flatnonzero(active):
                        f = obs[i]; ex = a['executed_reference21'][i]; before_ref = np.asarray(f['robot_reference21']); gen = int(a['mode_generation'][i])
                        if np.any(ex < np.asarray(f['reference_lower21'])-1e-12) or np.any(ex > np.asarray(f['reference_upper21'])+1e-12) or np.max(abs(ex-before_ref)-reference_rates(debug[i]['mode'])*.005) > 1e-12: errors.append((int(i), 'actual_mode_rate'))
                        if gen != f['mode_generation'] or gen != debug[i]['mode_generation'] or abs(f['robot_control_time_s']-time[i]) > 1e-9 or not helper.fbits(ex-before_ref, a['reference_increment21'][i]): errors.append((int(i), 'generation_clock_delta'))
                    ck(label+'/all5ms actual permissions rates clocks', not errors, errors[:12])
                    dm = read(run/'release_decision_manifest.json'); ck(label+'/decision raw archive hash', h(run/dm['file']) == dm['sha256'])
                    with gzip.open(run/dm['file'], 'rt', encoding='utf-8') as f: records = [json.loads(s) for s in f]
                    indices = np.asarray([round(q['time_s']/.005) for q in records], int)
                    expected_indices = np.flatnonzero([active[i] and debug[i].get('architecture') == 'v20_contact_conditioned_whole_hand_exit' for i in range(len(time))])
                    ck(label+'/every actual release decision retained', len(records) == dm['records'] and np.array_equal(indices, expected_indices))
                    counts = Counter(); classified = Counter(); errors = []; max_debt = 0.
                    for k, q in enumerate(records):
                        i = indices[k]; f = obs[i]; ex = a['executed_reference21'][i]; current = np.asarray(q['current_reference21']); g = q['governor']; receipt = q.get('execution_receipt')
                        if q['decision_id'] != k or q['time_s'] != time[i] or q['mode_generation'] != int(a['mode_generation'][i]) or q['process_version'] != q['public_process']['version'] or not helper.fbits(current, f['robot_reference21']): errors.append((k, 'current_public_record'))
                        if q['public_process']['holding_force_tasks_active']: errors.append((k, 'holding_leak'))
                        rates = reference_rates('release'); L = np.maximum(np.asarray(f['reference_lower21']), current-rates*.005); U = np.minimum(np.asarray(f['reference_upper21']), current+rates*.005)
                        if not helper.fbits(L, q['original_mechanical_rate_lower21']) or not helper.fbits(U, q['original_mechanical_rate_upper21']): errors.append((k, 'original_authority'))
                        qlift, vlift = float(f['robot_position25'][24]), float(f['robot_velocity25'][24])
                        interval, reachable = helper.independent_lift_interval(g, qlift, vlift, L[18], U[18], q['phase'] in helper.EXIT_PHASES)
                        if not helper.fbits(interval, q['governor_task_domain']['task_reference_interval_m']) or q['governor_task_domain']['budget_band_reachable_this_tick'] != reachable or not interval[0] <= ex[18] <= interval[1]: errors.append((k, 'TASK_reference_domain'))
                        if receipt is None: errors.append((k, 'missing_actual_receipt')); continue
                        debt = qlift-float(ex[18]); max_debt = max(max_debt, abs(debt)); budget = -g['downward_recovery_reference_budget_m'] <= debt <= g['recovery_reference_budget_m']
                        changed = not helper.fbits(q['proposed_reference21'], ex)
                        if not helper.fbits(receipt['actual_reference21'], ex) or receipt['same_snapshot_q_lift_m'] != qlift or receipt['same_snapshot_reference_debt_m'] != debt or receipt['current_q_budget_respected'] != budget or receipt['actual_task_interval_respected'] != (interval[0] <= ex[18] <= interval[1]) or receipt['execution_changed'] != changed: errors.append((k, 'actual_receipt_flags'))
                        counts['exact_budget_false'] += not budget; counts['execution_changed'] += changed; counts['stale_plan_replaced'] += receipt['stale_plan_replaced']
                        local_errors, local_counts = helper.independent_tasks(q, f, ex, rates)
                        errors.extend((k, x) for x in local_errors); counts.update(local_counts); classify_decision(q, classified)
                    ck(label+'/all actual raw normal tasks layers fallback and receipt', not errors, errors[:20])
                    ck(label+'/future pressure always unavailable', not np.any(a['prediction_available']) and np.isnan(a['predicted_normal']).all())
                    row.update(status='audited_new_raw', trace_sha256=h(run/'trace.npz'), metadata_sha256=h(run/'metadata.json'), result_sha256=h(run/'result.json'),
                        original_full17_run=str(comparator.relative_to(ROOT)), original_full17_trace_sha256=h(comparator/'trace.npz'), decisions=len(records),
                        original_task_success=result['task_success'], original_full_motion=result['research_metrics']['full_motion_mission_success'], original_force_guard=result['research_metrics']['safety']['guard_pass'],
                        counts=dict(counts), stage_denominators_and_fallbacks=dict(classified), maximum_abs_current_q_reference_debt_m=max_debt,
                        complete_public_observation_prefix_exact=full_obs_exact, prefix37=comparison,
                        native_value_scope='New13 saved current/proposed fullF067 status and same q/v length/speed checked; native evaluator not replayed here. Initial3 representative re-evaluation belongs to pinned small audit.')
                except Exception as exc:
                    row.update(status='reader_error', error=repr(exc)); issues.append(dict(name=label+'/reader_error', error=repr(exc), traceback=traceback.format_exc()))
    actual_keys = {tuple(r['record_key']) for r in rows}
    ck('all assigned16 assessed with no selected winner filtering', len(rows) == len(actual_keys) == 16 and actual_keys == expected and all(r['status'] in ('audited_new_raw', 'audited_retained_by_exact_hash_and_passed_receipt') for r in rows))
    ck('source160 still unchanged', all(sha(source_path(k)) == v for k, v in source.items()))
    ck('zero forbidden calls', not forbidden, forbidden)
    report = dict(schema='v20_complete16_execution_independent_v1', passed=not issues, complete=not issues,
        checks=checks, issues=issues, assigned_count=16, retained_count=3, newly_audited_raw_count=13,
        rows=rows, plan_sha256=sha(PLAN), source_sha256=source, environment=plan['environment'], input_sha256=inputs,
        retained_receipt_chain=pins, reader_sha256=sha(__file__),
        helper_sha256={'review/read_v20_small_execution_v1.py': h(helper_path), 'review/v16_small_prefix_storage_independent_v1.py': h(prefix_path)},
        scope=dict(physics_steps=0, optimizer_replays=0, plant_constructions=0, data_constructions=0, all_new13_saved_coordinated_records_checked=True,
                   retained3_recomputed=False, retained3_full_run_hash_and_passed_actual_receipt_checked=True),
        limitations=['Execution PASS is not a task, force-safety, placement or performance win.', 'Original failures, qualified NULL and unexposed cases are retained.',
                    'Positive-pressure optimization, known P0 postclear actions and direct SEARCH/CAPTURE have separate denominators.',
                    'Exact budget comparisons can be false at IEEE boundary; magnitude and actual TASK-domain checks are separately retained.',
                    'No new13 native model replay; saved native same-state value checks reuse component and initial3 independent full-value evidence.',
                    'Two physical seed clusters and16 seen/development contexts are not16 independent confirmatory physical draws.'])
    out.parent.mkdir(parents=True, exist_ok=True); md.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(helper.plain(report), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    md.write_text('# V20完整16条件执行审查\n\n'+('PASS' if report['passed'] else 'FAIL')+f'；{sum(c["passed"] for c in checks)}/{len(checks)}项，问题{len(issues)}。\n\n'
        '原3条通过精确文件hash及原63项执行收据链保留、不重跑；新13全部终态后逐run读取所有需要数组一次，对应原FullV17同条件pre12完整37列与完整公共observation逐UTF8比较。实际21机械/mode/rate/代际/回执、TASK预算、每个release决策原法向任务与层级公式均复核；有压、P0、直接动作和未知/未认证回退分别计数。\n\n'
        '没有优化器重解、模型积分、forward或接触查询。此门只判执行证据，原task/fullmotion/guard失败不被删除，也不宣称总体胜利或稳定放置。完整分母、限制和来源见JSON。\n', encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], checks=len(checks), issues=issues[:8], out=str(out), sha256=sha(out)), ensure_ascii=False))
    return 0 if report['passed'] else 1


def main():
    p = argparse.ArgumentParser(); p.add_argument('--execute', action='store_true'); p.add_argument('--prepare', action='store_true')
    p.add_argument('--out', type=Path, default=ROOT/'review/v20_complete16_execution_v1.json'); p.add_argument('--md', type=Path, default=ROOT/'review/V20完整16条件执行独审_v1.md'); args = p.parse_args()
    if not PLAN.exists(): print(json.dumps(dict(status='PENDING_PLAN', branch_results_read=False))); return 2
    plan = read(PLAN)
    if not args.execute:
        print(json.dumps(dict(prepared=True, source_count=len(plan['source_sha256']), new13=sum(s['planned'] for s in plan['stages']), retained3=len(plan['retained']), plan_sha256=sha(PLAN), reader_sha256=sha(__file__), branch_results_read=False, physics_steps=0))); return 0
    return execute(plan, args.out, args.md)


if __name__ == '__main__': raise SystemExit(main())

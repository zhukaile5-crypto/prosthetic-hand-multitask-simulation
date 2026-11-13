"""All-nine terminal barrier, unchanged auxiliaryv3, same-context three arms.

No partial outcome selection, new thresholds, new success label or simulation.
Each assigned ledger is loaded once. Original failed/NULL values remain in the
record. Three contexts reuse two physical seeds, not independent sample n=9.
"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import argparse, hashlib, importlib.util, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'review/v21_small_plan_v1.json'
VARIANTS = ('v20', 'prefix', 'v21')
CONTEXTS = (('nominal', 'transport', 619741), ('fragile_conflict', 'hold', 619741), ('friction_drop', 'transport', 619742))
CALC_SHA = '0228b2526ace5f448a6f7f458ebeadb9023b79b120fe4ba16eccb3463eb5bc55'
CONTRACT_SHA = '8dcda359252014d509a8b26c102570309ad893b6a4ccad31015a36ca0cb7e979'
METRICS_SHA = '962f5b8c346bdfe7f044878c93246e3681b71201ee39abfe4294ade3a5d7d49e'
OLD80_SHA = '8a5e9a3cdfcffe4f484ded0c183ec72da718d7c29f30264fbea65beea459e116'
OLD20_SHA = '09c9768132943e14e01c77f2d1ee4c325881b5656b9806fbb61b23672a46278d'
HISTORICAL_ROLES = ('full_V17', 'holding_A', 'same_prior_legacy_ratio')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_module(name, p):
    spec = importlib.util.spec_from_file_location(name, p); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def require_equal(a, b, name):
    if a != b: raise ValueError(name+' differs')
def context(assignment):
    if (assignment.get('methods') != ['ratio_effort'] or len(assignment.get('scenarios', [])) != 1 or len(assignment.get('seeds', [])) != 1 or
        assignment.get('task_policy') != 'integrated' or assignment.get('receiver_policy') != 'coordinated' or
        assignment.get('physical_randomization') is not True or assignment.get('release_variant') not in VARIANTS):
        raise ValueError('explicit assigned release variant and single paired job required')
    key = (assignment['scenarios'][0], assignment['motion_mode'], assignment['seeds'][0])
    if key not in CONTEXTS: raise ValueError('unexpected registered context')
    return key
def validate_roster(plan):
    if plan.get('planned') != 9 or len(plan.get('stages', [])) != 9: raise ValueError('exact all-nine assignment required')
    keys = [(s['variant'], *context(s['assignment'])) for s in plan['stages']]
    if len(set(s['name'] for s in plan['stages'])) != 9 or len(set(keys)) != 9 or set(keys) != {(v, *c) for v in VARIANTS for c in CONTEXTS}:
        raise ValueError('duplicate/missing variant-context assignment')
    if any(s['variant'] != s['assignment']['release_variant'] or s['planned'] != 1 for s in plan['stages']): raise ValueError('descriptor variant/count mismatch')
    return keys
def terminal_barrier(plan, markers):
    validate_roster(plan); pending = []
    for s in plan['stages']:
        m = markers.get(s['name'])
        if m is None or m.get('execution_complete') is not True or m.get('sources_unchanged') is not True or m.get('runtime_failures') != [] or m.get('planned') != 1 or m.get('trials') != 1:
            pending.append(s['name'])
    return dict(ready_to_read_metrics=not pending, pending=pending, assigned=9, metrics_read=False)
def readiness():
    if not PLAN.exists(): return dict(ready_to_read_metrics=False, pending=['registered_plan_absent'], metrics_read=False), None, {}
    plan = read(PLAN); markers = {}
    for s in plan['stages']:
        p = ROOT/s['output']/'execution_complete.json'
        if p.exists(): markers[s['name']] = read(p)
    return terminal_barrier(plan, markers), plan, markers
def metadata_context(meta, assignment):
    scene = meta.get('scenario'); scene = scene.get('name') if isinstance(scene, dict) else scene
    require_equal((scene, meta.get('motion_mode'), meta.get('seed')), context(assignment), 'actual metadata context')
    require_equal((meta.get('method'), meta.get('task_policy'), meta.get('receiver_policy'), meta.get('release_variant')),
        ('ratio_effort', 'integrated', 'coordinated', assignment['release_variant']), 'actual metadata policy/variant')
def frozen_dependencies():
    paths = dict(calculator=ROOT/'review/delivery_auxiliary_metrics_v3.py', contract=ROOT/'review/delivery_auxiliary_contract_v3.json',
        metrics=ROOT/'review/read_v18_small_auxiliary_v1.py', old80=ROOT/'review/v17b_delivery_auxiliary_readout_v1.json', old20=ROOT/'review/v20_small_auxiliary_v1.json')
    expected = dict(calculator=CALC_SHA, contract=CONTRACT_SHA, metrics=METRICS_SHA, old80=OLD80_SHA, old20=OLD20_SHA)
    for k, p in paths.items(): require_equal(sha(p), expected[k], 'frozen dependency '+k)
    return paths, expected
def release_cost(result):
    x = result.get('safety', {}).get('stages', {}).get('mechanical_release_and_after')
    return None if x is None else dict(finger_peak_N=x.get('finger_peak_N'), local_excess_integral_Ns=x.get('local_excess_integral_Ns'), total_excess_integral_Ns=x.get('total_excess_integral_Ns'), duration_s=x.get('duration_s'))
def differences(a, b, paired=True):
    result = {}
    for k in sorted(set(a or {}) | set(b or {})):
        x, y = (a or {}).get(k), (b or {}).get(k)
        known = bool(paired and type(x) in (bool, int, float) and type(y) in (bool, int, float))
        result[k] = dict(candidate=x, reference=y, paired_available=known, delta=float(x)-float(y) if known else None,
            relative_delta=(float(x)-float(y))/float(y) if known and type(x) in (int, float) and type(y) in (int, float) and y != 0 else None)
    return result
def paired_metadata(a, b): return all(a[k] == b[k] for k in ('config', 'noise', 'limits', 'budget'))


def collect(plan, markers):
    # Independent inner barrier prevents bypass by directly calling collect.
    if not terminal_barrier(plan, markers)['ready_to_read_metrics']: raise ValueError('all nine terminal before results/NPZ')
    paths, pins = frozen_dependencies(); calc = load_module('v21_frozen_auxv3', paths['calculator']); projection = load_module('v21_original_metrics_projection', paths['metrics'])
    require_equal(plan['auxiliary_contract_sha256'], CONTRACT_SHA, 'plan contract pin'); require_equal(plan['auxiliary_calculator_sha256'], CALC_SHA, 'plan calculator pin')
    if len(plan.get('source_sha256', {})) != 170: raise ValueError('registered complete170 source closure required')
    old80 = read(paths['old80']); old20 = read(paths['old20'])
    if not old80.get('complete') or not old80.get('source_identity_passed') or not old20.get('complete') or old20.get('issues') != []: raise ValueError('complete old references required')
    refs = {(r['role'], r['scenario'], r['motion_mode'], r['physical_seed']): r for r in old80['records'] if r['role'] in HISTORICAL_ROLES}
    original20 = {tuple(r['context_key']): r for r in old20['records'] if r['stage'] != 'passive_ratio_transport'}
    rows, errors, external_pairs, within_pairs, bridge = [], [], [], [], []
    metas = {}; completion_pins = {}
    for stage in plan['stages']:
        folder = ROOT/stage['output']; ass = stage['assignment']; key = context(ass); variant = stage['variant']; mark = markers[stage['name']]
        completion_pins[stage['name']] = sha(folder/'execution_complete.json')
        trials = read(folder/'trials.json'); ledger = read(folder/'assignment_ledger.json'); manifest = read(folder/'manifest.json'); reg = read(ROOT/stage['registration'])
        require_equal(len(trials), 1, 'one original result'); require_equal(len(ledger), 1, 'one original assignment'); require_equal(ledger[0]['status'], 'evaluated', 'evaluated assignment')
        require_equal(reg['assignment'], ass, 'registration assignment'); require_equal(reg['source_sha256'], plan['source_sha256'], 'registered170'); require_equal(reg['environment'], plan['environment'], 'registered native environment')
        require_equal(manifest['research_source_sha256'], plan['source_sha256'], 'manifest170'); require_equal(read(folder/'environment.json')['pinned_environment'], plan['environment'], 'actual native environment')
        for file, field in (('trials.json', 'trials_sha256'), ('assignment_ledger.json', 'assignment_ledger_sha256'), ('plan.json', 'plan_sha256'), ('manifest.json', 'manifest_sha256')):
            require_equal(sha(folder/file), mark[field], 'completion '+file)
        run = ROOT/stage['output']/'runs'/f'{key[0]}__ratio_effort__{key[2]}'
        require_equal(Path(ledger[0]['job']['destination']).resolve(), run.resolve(), 'actual assigned directory')
        meta = read(run/'metadata.json'); result = read(run/'result.json'); metadata_context(meta, ass)
        require_equal(meta['research_source_sha256'], plan['source_sha256'], 'metadata170'); require_equal(result, trials[0], 'original evaluator result')
        release_at = meta.get('mechanical_release_at', meta.get('release_at')); auxiliary = metrics = None; row_errors = []
        if release_at is None: status = 'no_actual_release_clock'
        else:
            try:
                with np.load(run/'delivery_intervals.npz', allow_pickle=False) as z: arrays = {k: z[k] for k in calc.C['input_fields']}
                auxiliary = calc.calculate(arrays, release_at_s=release_at, episode_end_s=meta['duration'], physical_period_s=meta['physics_timestep'], original_result=result)
                metrics = projection.metrics(auxiliary); status = auxiliary['status']; del arrays
            except (ValueError, TypeError, KeyError) as exc: status = 'diagnostic_unavailable'; row_errors.append(type(exc).__name__+': '+str(exc))
        row = dict(stage=stage['name'], variant=variant, record_key=[variant, *key], context_key=list(key), run=str(run), original_result=result,
            original_release_cost=release_cost(result), handoff_passed=meta['handoff_passed'], auxiliary=auxiliary, metrics=metrics, status=status, issues=row_errors,
            input_sha256={n: sha(run/n) for n in ('metadata.json', 'result.json', 'delivery_intervals.npz')})
        rows.append(row); metas[(variant, *key)] = meta
        if row_errors: errors.append(stage['name']+'/diagnostic_unavailable')
        if variant == 'v20':
            old = original20[key]; oldrun = Path(old['run']); oldmeta = read(oldrun/'metadata.json')
            for name, digest in old['input_sha256'].items(): require_equal(sha(oldrun/name), digest, 'oldV20 preserved '+name)
            exact = auxiliary is not None and projection.same_non_original(auxiliary, old['auxiliary'])
            bridge.append(dict(stage=stage['name'], context_key=list(key), old_run=str(oldrun), same_physical_noise_limits=paired_metadata(meta, oldmeta), all_non_original_auxiliary_fields_exact=bool(exact)))
            if not exact: errors.append(stage['name']+'/originalV20_aux_differ')
        for role in HISTORICAL_ROLES:
            ref = refs[(role, *key)]; equal = all(meta[k] == ref[rk] for k, rk in (('config', 'physical_config'), ('noise', 'noise'), ('limits', 'limits'), ('budget', 'budget')))
            rm = projection.metrics(ref['auxiliary'])
            external_pairs.append(dict(variant=variant, context_key=list(key), reference_role=role, reference_record_id=ref['record_id'], same_physical_noise_limits=equal,
                candidate_metrics=metrics, reference_metrics=rm, differences=differences(metrics, rm, equal), candidate_original_release_cost=row['original_release_cost'], reference_original_release_cost=release_cost(ref['auxiliary']['original_result'])))
            if not equal: errors.append(stage['name']+'/'+role+'/unpaired_parameters')
    index = {(r['variant'], *r['context_key']): r for r in rows}
    for key in CONTEXTS:
        for candidate, reference in (('prefix', 'v20'), ('v21', 'v20'), ('v21', 'prefix')):
            a = index[(candidate, *key)]; b = index[(reference, *key)]; equal = paired_metadata(metas[(candidate, *key)], metas[(reference, *key)])
            within_pairs.append(dict(candidate_variant=candidate, reference_variant=reference, context_key=list(key), same_physical_noise_limits=equal,
                candidate_metrics=a['metrics'], reference_metrics=b['metrics'], differences=differences(a['metrics'], b['metrics'], equal),
                candidate_original_release_cost=a['original_release_cost'], reference_original_release_cost=b['original_release_cost']))
            if not equal: errors.append(candidate+'/'+reference+'/'+str(key)+'/unpaired_parameters')
    if len(rows) != 9 or len(index) != 9 or len(within_pairs) != 9 or len(external_pairs) != 27 or len(bridge) != 3: raise ValueError('complete nine-record pair denominators required')
    return dict(schema='v21_three_arm_small_auxiliary_v1', complete=not errors, planned=9, records=rows, issues=errors,
        status='computed_all9' if not errors else 'computed_with_issues', created_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(PLAN), source_sha256=plan['source_sha256'], environment=plan['environment'], reader_sha256=sha(__file__),
        calculator_sha256=CALC_SHA, contract_sha256=CONTRACT_SHA, metric_projection_sha256=METRICS_SHA, reference_old80_sha256=OLD80_SHA, original_v20_aux_sha256=OLD20_SHA,
        completion_marker_sha256=completion_pins, bridge_auxiliary_equivalence=bridge, within_variant_pairs=within_pairs, historical_pairs=external_pairs,
        definition_changed=False, original_metrics_changed=False, physics_steps=0,
        performance_scope='Three seen contexts, two shared physical seed clusters, three release variants. No independent n9, new success criterion, total score or selected-winner filtering.')


def text_report(d, out):
    def fmt(v):
        if v is None: return '不可评/未达到'
        if type(v) is bool: return '是' if v else '否'
        return f'{v:.6g}' if type(v) in (int, float) else str(v)
    fields = ('original_task_success', 'original_physical_clearance_success', 'original_200Hz_clearance_latency_s', 'final_low_motion_suffix_found', 'ever_floor',
        'end_linear_speed_m_s', 'end_angular_speed_rad_s', 'receiver_first20ms_peak_N', 'receiver_first20ms_impulse_Ns', 'original_full_guard_pass', 'original_full_max_finger_peak_N', 'original_full_local_excess_Ns')
    lines = ['# V21九条三臂开发辅助读数', '', '全部九条终态后计算；v20为同源复现臂，prefix只改变认证前缀提交，v21为新单侧/运动预算/路径任务。固定原v3定义，原失败与NULL保持；三个条件复用两个物理seed，不能称九个独立试验或全局优势。', '',
        '|条件/版本|原任务|原清离|清离s|末段驻留|曾floor|末线速m/s|末角速rad/s|触台峰N|冲量Ns|全guard|最大指峰N|局部超力Ns|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key in CONTEXTS:
        for variant in VARIANTS:
            row = next(r for r in d['records'] if r['variant'] == variant and tuple(r['context_key']) == key); m = row['metrics'] or {}
            lines.append('|'+str(key)+'/'+variant+'|'+'|'.join(fmt(m.get(k)) for k in fields)+'|')
    lines += ['', 'JSON保存9组内同条件对照、27组历史V17/A/力比PI对照、3条原V20辅助复现门、所有原结果/阶段力成本/暴露资格/first及final驻留/censor。未达到、无新撞击、未知和零值不相互替换。', '', f'完整JSON：`{out}`，SHA `{sha(out)}`。']
    return '\n'.join(lines)+'\n'


def self_test():
    stages = []
    for variant in VARIANTS:
        for i, (scene, motion, seed) in enumerate(CONTEXTS):
            stages.append(dict(name=variant+str(i), variant=variant, planned=1, assignment=dict(methods=['ratio_effort'], scenarios=[scene], seeds=[seed], motion_mode=motion, physical_randomization=True, task_policy='integrated', receiver_policy='coordinated', release_variant=variant)))
    plan = dict(planned=9, stages=stages); good = dict(execution_complete=True, sources_unchanged=True, runtime_failures=[], planned=1, trials=1); markers = {s['name']: dict(good) for s in stages}; checks = []
    checks.append(dict(category='complete_terminal', passed=terminal_barrier(plan, markers)['ready_to_read_metrics']))
    missing = deepcopy(markers); missing.pop(stages[-1]['name']); checks.append(dict(category='missing_terminal', passed=not terminal_barrier(plan, missing)['ready_to_read_metrics']))
    changed = deepcopy(markers); changed[stages[0]['name']]['sources_unchanged'] = False; checks.append(dict(category='changed_source', passed=not terminal_barrier(plan, changed)['ready_to_read_metrics']))
    bad = deepcopy(markers); bad[stages[0]['name']]['runtime_failures'] = [{'error': 'test'}]; bad[stages[0]['name']]['trials'] = 0; checks.append(dict(category='runtime_or_partial_ledger', passed=not terminal_barrier(plan, bad)['ready_to_read_metrics']))
    duplicate = deepcopy(plan); duplicate['stages'][-1] = deepcopy(duplicate['stages'][0])
    try: validate_roster(duplicate); rejected = False
    except ValueError: rejected = True
    checks.append(dict(category='duplicate_variant_context', passed=rejected))
    ass = stages[0]['assignment']; meta = dict(scenario={'name': 'nominal'}, motion_mode='transport', seed=619741, method='ratio_effort', task_policy='integrated', receiver_policy='coordinated', release_variant='v20')
    metadata_context(meta, ass); badmeta = dict(meta, scenario={'name': 'collision'})
    try: metadata_context(badmeta, ass); rejected = False
    except ValueError: rejected = True
    checks.append(dict(category='actual_dict_metadata_and_wrong_context', passed=rejected))
    badmeta = dict(meta, release_variant='prefix')
    try: metadata_context(badmeta, ass); rejected = False
    except ValueError: rejected = True
    checks.append(dict(category='same_method_distinct_variant', passed=rejected))
    paths, pins = frozen_dependencies(); helper = load_module('v21_old_metrics_selftest', paths['metrics']); old = read(ROOT/'review/v18_small_auxiliary_v1.json')
    delta = differences({'unknown': None, 'zero': 0., 'failed': False}, {'unknown': False, 'zero': 0., 'failed': True})
    checks.append(dict(category='frozen_definition_null_zero_failure', passed=all(helper.metrics(r['auxiliary']) == r['metrics'] for r in old['records']) and delta['unknown']['delta'] is None and delta['zero']['relative_delta'] is None and delta['failed']['delta'] == -1.))
    return dict(schema='v21_auxiliary_reader_protection_v1', passed=all(c['passed'] for c in checks), checks=checks, source_sha256={'review/read_v21_small_auxiliary_v1.py': sha(__file__)}, frozen_dependencies=pins,
        new_outcomes_read=False, physics_steps=0, original_V18_metrics_exact=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--execute', action='store_true'); p.add_argument('--prepare', action='store_true'); p.add_argument('--self-test', action='store_true')
    p.add_argument('--out', type=Path, required=True); p.add_argument('--md', type=Path); args = p.parse_args()
    if args.out.exists() or (args.md and args.md.exists()): raise FileExistsError('Fresh auxiliary outputs required')
    if args.self_test: payload = self_test()
    else:
        barrier, plan, markers = readiness()
        if not args.execute: payload = dict(schema='v21_auxiliary_readiness_v1', **barrier, reader_sha256=sha(__file__), physics_steps=0)
        else:
            if not barrier['ready_to_read_metrics']: raise ValueError('all nine terminal before any outcome/NPZ: '+str(barrier['pending']))
            payload = collect(plan, markers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as f: json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    if args.execute and args.md and payload.get('complete'):
        args.md.parent.mkdir(parents=True, exist_ok=True); args.md.write_text(text_report(payload, args.out), encoding='utf-8')
    print(json.dumps(dict(out=str(args.out), sha256=sha(args.out), passed=payload.get('passed'), complete=payload.get('complete'), ready=payload.get('ready_to_read_metrics'), physics_steps=0), ensure_ascii=False))
    return 1 if payload.get('passed') is False or payload.get('complete') is False else 0


if __name__ == '__main__': raise SystemExit(main())

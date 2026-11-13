"""Saved-record statistics only; does not instantiate a controller or plant."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from v22_joint_task_fusion.numerics import ensure_numeric_environment
ensure_numeric_environment()
import numpy as np

SCENES = ('nominal', 'friction_drop', 'fragile_conflict', 'collision')
SCENE_NAMES = dict(nominal='正常运输', friction_drop='摩擦下降',
                   fragile_conflict='低限力与拉拽', collision='碰撞扰动', overall='四场景合计')
ARMS = ('baseline', 'macro')
ARM_NAMES = dict(baseline='力比 PI 基线', macro='宏观版')
STAGES = ('full', 'pickup', 'holding_transport', 'release')
STAGE_NAMES = dict(full='全程', pickup='抓握', holding_transport='保持及运输', release='放下')
BINARY = ('terminal_success', 'stage_success', 'pickup_success', 'retained_to_release',
          'force_limit_pass', 'safe_terminal_success', 'steady_placement_attained')
LABELS = dict(terminal_success='完整任务成功率（最终留台且离手）',
    stage_success='阶段完成率（原口径）', pickup_success='稳定抓起率',
    retained_to_release='保持到放下指令比例', force_limit_pass='全程限力通过率',
    safe_terminal_success='完成且限力通过率', steady_placement_attained='放置稳定达成率',
    pickup_stable_time_s='抓握达到稳定时间（s，仅达成者）',
    pickup_deadline_capped_s='抓握达到稳定时间（s，未达成按7.5s封顶）',
    release_latency_s='放下指令到留台并离手时间（s，仅达成者）',
    release_deadline_capped_s='放下完成耗时（s，未达成按3.5s封顶）',
    steady_release_latency_s='放下指令到稳定放置时间（s，仅达成者）',
    steady_release_deadline_capped_s='稳定放置耗时（s，未达成按3.5s封顶）',
    force_exceedance_duration_s='超力累计时间（s）', maximum_force_excess_N='最大超力幅度（N）',
    force_excess_impulse_N_s='超力积分（N·s）', peak_total_external_finger_force_N='外部总指力峰值（N）')
CONTINUOUS = tuple(k for k in LABELS if k not in BINARY)
FORCE_METRICS = ('total_force_reversal_rate_N_s', 'total_force_SD_N', 'raw_total_mean_N', 'raw_total_peak_N')
FORCE_NAMES = dict(total_force_reversal_rate_N_s='总指力往复变化率（N/s）',
    total_force_SD_N='总指力标准差（N）', raw_total_mean_N='平均总指力（N）', raw_total_peak_N='总指力峰值（N）')
FORCE_KEYS = tuple(f'{s}__{m}' for s in STAGES for m in FORCE_METRICS)
LABELS.update({f'{s}__{m}': f'{STAGE_NAMES[s]}{FORCE_NAMES[m]}' for s in STAGES for m in FORCE_METRICS})
ALL_METRICS = BINARY + CONTINUOUS + FORCE_KEYS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_dump(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def flatten(record):
    job = record['job']
    row = dict(id=job['id'], seed=job['seed'], scene=job['scene'], arm=job['arm'],
               status=record['status'], wall_seconds=record.get('wall_seconds'))
    r, extra = record.get('result', {}), record.get('repeat_metrics', {})
    valid = record['status'] == 'evaluated' and all(r.get('physical_audit', {'missing': False}).values())
    row.update(valid_record=valid, functional_eligible=bool(valid and r['pickup_completed'] and r['retained_to_release']),
               failure_reasons=r.get('reasons', ['technical_failure']))
    row.update(terminal_success=bool(valid and extra.get('terminal_qualified_completion')),
        stage_success=bool(valid and r.get('full_task_completed')),
        pickup_success=bool(valid and r.get('pickup_completed')),
        retained_to_release=bool(valid and r.get('retained_to_release')),
        force_limit_pass=bool(valid and r.get('force_limit_pass')),
        safe_terminal_success=bool(valid and extra.get('safe_terminal_completion')),
        steady_placement_attained=bool(valid and r.get('steady_placement_time_s') is not None))
    for key in CONTINUOUS + FORCE_KEYS:
        row[key] = None
    if not valid:
        return row
    row.update({k: r[k] for k in ('pickup_stable_time_s', 'release_latency_s',
        'force_exceedance_duration_s', 'maximum_force_excess_N',
        'force_excess_impulse_N_s', 'peak_total_external_finger_force_N')})
    row['pickup_deadline_capped_s'] = r['pickup_stable_time_s'] if r['pickup_stable_time_s'] is not None else 7.5
    row['release_deadline_capped_s'] = r['release_latency_s'] if r['release_latency_s'] is not None else 3.5
    steady = r['steady_placement_time_s']
    row['steady_release_latency_s'] = None if steady is None else steady - 12.
    row['steady_release_deadline_capped_s'] = steady - 12. if steady is not None else 3.5
    row.update({f'{s}__{m}': r['total_force'][s][m] for s in STAGES for m in FORCE_METRICS})
    if row['stage_success'] and not row['terminal_success']:
        row['failure_reasons'] = row['failure_reasons'] + ['terminal_receiver_or_clearance_not_retained']
    return row


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p, den = k/n, 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0., center-half)), float(min(1., center+half))]


def cluster_ci(values, seeds, key, replicates=20000):
    """Resample complete seed clusters, retaining all represented scenarios."""
    values, seeds = np.asarray(values, float), np.asarray(seeds, int)
    unique = np.unique(seeds)
    if len(unique) < 2:
        return None
    sums = np.asarray([values[seeds == seed].sum() for seed in unique])
    counts = np.asarray([(seeds == seed).sum() for seed in unique])
    salt = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(2026111207 + salt)
    index = rng.integers(0, len(unique), size=(replicates, len(unique)))
    estimates = sums[index].sum(axis=1)/counts[index].sum(axis=1)
    return np.quantile(estimates, [.025, .975]).tolist()


def descriptive(rows, key, group):
    selected = [r for r in rows if r.get(key) is not None]
    values = np.asarray([r[key] for r in selected], float)
    result = dict(assigned_n=len(rows), measured_n=len(selected), missing_n=len(rows)-len(selected),
                  independent_seed_n=len({r['seed'] for r in selected}))
    if not len(values):
        return dict(result, mean=None, sd=None, median=None, min=None, max=None, ci95=None)
    ci = cluster_ci(values, [r['seed'] for r in selected], f'{group}/{key}/mean')
    result.update(mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values)>1 else None,
                  median=float(np.median(values)), min=float(values.min()), max=float(values.max()), ci95=ci)
    if key in BINARY:
        result.update(successes=int(values.sum()), trials=len(values))
        if group.split('/')[0] != 'overall':
            result['ci95'] = wilson(int(values.sum()), len(values))
            result['interval_method'] = 'Wilson 95%'
        else:
            result['interval_method'] = 'seed-cluster bootstrap 95%'
    return result


def pair_comparison(pairs, key, group):
    selected = [(a,b) for a,b in pairs if a.get(key) is not None and b.get(key) is not None]
    if not selected:
        return dict(paired_n=0, independent_seed_n=0, baseline_mean=None, macro_mean=None,
                    macro_minus_baseline=None, relative_change_pct=None, ci95=None,
                    lower_n=0, equal_n=0, higher_n=0)
    base = np.asarray([a[key] for a,b in selected], float)
    macro = np.asarray([b[key] for a,b in selected], float)
    delta = macro-base
    seeds = [a['seed'] for a,b in selected]
    return dict(paired_n=len(selected), independent_seed_n=len(set(seeds)),
        baseline_mean=float(base.mean()), macro_mean=float(macro.mean()),
        macro_minus_baseline=float(delta.mean()),
        relative_change_pct=float((macro.mean()/base.mean()-1)*100) if abs(base.mean())>1e-12 else None,
        ci95=cluster_ci(delta, seeds, f'{group}/{key}/paired'),
        lower_n=int(np.sum(delta < -1e-10)), equal_n=int(np.sum(abs(delta)<=1e-10)), higher_n=int(np.sum(delta>1e-10)))


def aggregate(rows):
    groups, compact, comparisons, functional_comparisons = {}, [], [], []
    lookup = {(r['seed'],r['scene'],r['arm']): r for r in rows}
    for group in (*SCENES, 'overall'):
        group_rows = [r for r in rows if group=='overall' or r['scene']==group]
        keys = sorted({(r['seed'],r['scene']) for r in group_rows})
        pairs = [(lookup[(s,c,'baseline')], lookup[(s,c,'macro')]) for s,c in keys]
        functional = [(a,b) for a,b in pairs if a['functional_eligible'] and b['functional_eligible']]
        arms = {}
        for arm in ARMS:
            selected = [r for r in group_rows if r['arm']==arm]
            metrics = {k: descriptive(selected, k, f'{group}/{arm}') for k in ALL_METRICS}
            failure_counts = dict(Counter(reason for r in selected for reason in r['failure_reasons']))
            arms[arm] = dict(metrics=metrics, failure_counts=failure_counts)
            for key, val in metrics.items():
                compact.append(dict(scene=group, arm=arm, metric=key, label=LABELS[key], **val))
        pair_stats = {k: pair_comparison(pairs, k, f'{group}/all') for k in ALL_METRICS}
        quality_stats = {k: pair_comparison(functional, k, f'{group}/functional') for k in FORCE_KEYS}
        for key,val in pair_stats.items():
            comparisons.append(dict(scene=group, metric=key, label=LABELS[key], **val))
        for key,val in quality_stats.items():
            functional_comparisons.append(dict(scene=group, metric=key, label=LABELS[key], **val))
        groups[group] = dict(arms=arms, pairs=pair_stats, functional_pair_n=len(functional),
                             functional_seed_n=len({a['seed'] for a,b in functional}), functional_pairs=quality_stats)
    return groups, compact, comparisons, functional_comparisons

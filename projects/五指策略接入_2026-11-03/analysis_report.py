"""Auditable static reporting of the preregistered 256-trial full-hand comparison.

First run the frozen evaluate_runs.py on the completed confirmation directory.
This separate report tool never runs a controller, changes an endpoint, or
imports runtime/physics code. Development results are deliberately rejected.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ('ratio', 'unified')
TITLES = {'ratio': '摩擦比基线', 'unified': '多任务融合策略'}
SHAPES = {'capsule': '胶囊体', 'box': '圆角盒', 'sphere': '球体'}
COLORS = {'ratio': '#31688e', 'unified': '#d98b39'}
STAGES = {'preparation': '共同准备', 'policy_holding': '策略持握',
          'post_preparation_without_policy': '准备失败后', 'mechanical_release_and_after': '机械释放及之后'}

METHODS_TEXT = """# 完整五指确认试验：分析方法

本工具只分析新平台预先登记的确认矩阵：16场景×8个种子×2种方法，共256次。旧五触点实验和六次开发烟测均不进入正式比较。共同准备截至7.5s；后续为校准域内持握策略迁移，12s计划释放，16s结束。共同准备不能归功于策略。

## 主指标保持原定义

- A_S：7.5s接管后，在1.4s限制窗内首次满足新的200ms稳定驻留所需等待，单位s；未达到贡献1.4s。它是首次稳定等待，不能称为扰动后恢复时间。200ms驻留使下限约0.2s。
- 逐指TV率：8.2–8.9s，实际物体接触力经已登记50Hz/50ms对称FIR及200Hz重采样后的五指绝对增量之和除以窗长，单位N/s。只比较双方共同具备完整支持窗口的场景-种子配对，并单列各自窗口与共同窗口覆盖。失抓、缺失或无效窗不填为0，也不因低TV获得收益。
- 主场景与预先登记的infeasible边界场景分开。A_S与TV的原始配对CI、精确种子符号置换及两端点Holm结果直接引用冻结evaluate_runs.py产生的canonical summary，并独立核对点估计与配对分母。

## 其他指标及分母

完整任务完成、完整16s逐子步力保护、准备通过均以所有已分配试验为分母，包含失败。所有场景的计划释放均保留分母，并另列真正承重释放暴露率；成功释放延迟仅在双方均成功的配对上描述。力保护覆盖五指全部外物皮肤接触；掌部与其他部位另记，不能声称已有其材料损伤阈值。

均力单位N，为有效窗内五指物体法向力之和的时间平均，使用双方共同有效窗。阶段安全分别列共同准备、实际策略持握、机械释放及之后；局部逐指超限积分与总力预算超限积分单位均为N·s，不能相加当作单一分数。阶段保护按保存的实际阶段峰与原限值及0.02N数值容差核对；缺失阶段保留缺失及覆盖分母，不填0。完整任务力保护仍是全程门槛。

局部动作域占用为实际策略持握中任一五协同坐标距当时域边界不超过1e-6的采样比例；静态窗域占用与请求-执行不一致比例另列。它不等于全部关节饱和；低TV伴随高域边界占用不能解释成无约束策略优势。

## 不确定性与分层

比较先按(场景,形状,种子)连接两方法，再对整个种子跨场景整簇有放回抽样5000次，报告差值“融合−基线”的百分位95%CI。8个独立种子簇不能当作256个独立样本；有效窗条件化后簇数及配对数重新显示。形状、阶段安全、域占用和恢复均为描述性次要分析，不新增主假设或事后综合分数。比例差单位为百分点，力、TV和时间差保留各自物理单位。

## 扰动恢复单独定义

事件恢复由独立预先登记的evaluate_recovery.py输出描述；保持事件前稳、实际暴露、从未失稳、无适用事件、恢复、截至观察窗仍未恢复等状态。未失稳不算0秒恢复，未暴露不算成功恢复，未恢复不以截断时长伪装完成。A_S不替代事件恢复。

## 可审计来源与解释边界

读取完整矩阵manifest、execution_complete、trials及canonical summary；核对256个计划键、128个方法配对、8个登记种子，核验每条metadata/result/原始NPZ及冻结源码SHA，最后再次核对输入未变化。每张图有chart_data.json/CSV分母、筛选和单位。脚本不重跑策略、动力学或调参。输出表保留各场景、形状、boundary状态、失败原因和覆盖；范围为简化完整手接触模型、虚拟皮肤、掌部被动承重与固定抓姿局部迁移，不是硬件或任意抓姿验证。
"""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def local(value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Analysis stays inside the independent integration directory')
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def key(row):
    return row['scenario'], row['shape'], int(row['seed']), row['method']


def finite(value):
    return value is not None and isinstance(value, (int, float)) and np.isfinite(value)


def field(row, name):
    if name == 'force_protection_pass':
        return bool(row.get('safety', {}).get(name, False))
    value = row
    for part in name.split('.'):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def cohorts(records):
    result = [('all', '全部场景', records)]
    for boundary, name, title in ((False, 'main', '主场景'), (True, 'boundary', '边界场景')):
        rows = [row for row in records if bool(row.get('boundary')) == boundary]
        if rows:
            result.append((name, title, rows))
            for shape in sorted({row['shape'] for row in rows}):
                result.append((f'{name}/{shape}', f'{title}·{SHAPES.get(shape, shape)}',
                               [row for row in rows if row['shape'] == shape]))
    return result


def paired(rows, endpoint, quality=False):
    maps = {method: {(row['scenario'], row['shape'], row['seed']): row for row in rows if row['method'] == method}
            for method in METHODS}
    result = []
    for identity in sorted(maps['ratio'].keys() & maps['unified'].keys()):
        baseline, candidate = maps['ratio'][identity], maps['unified'][identity]
        if quality and not (baseline.get('static_eligible') and candidate.get('static_eligible')):
            continue
        if endpoint == 'release_latency' and not (baseline.get('release_success') and candidate.get('release_success')):
            continue
        a, b = field(baseline, endpoint), field(candidate, endpoint)
        if finite(a) and finite(b):
            result.append((identity, float(a), float(b)))
    return result


def bootstrap_pairs(pairs, replicates=5000):
    if not pairs:
        return dict(pairs=0, clusters=0, baseline_mean=None, candidate_mean=None, difference=None, ci95=None)
    seeds = sorted({identity[2] for identity, _, _ in pairs})
    seed_values = {seed: [(b-a) for identity, a, b in pairs if identity[2] == seed] for seed in seeds}
    sums = np.asarray([sum(seed_values[seed]) for seed in seeds])
    counts = np.asarray([len(seed_values[seed]) for seed in seeds])
    rng = np.random.default_rng(20261103)
    selections = rng.integers(0, len(seeds), size=(replicates, len(seeds)))
    draws = sums[selections].sum(axis=1) / counts[selections].sum(axis=1)
    return dict(pairs=len(pairs), clusters=len(seeds),
                baseline_mean=float(np.mean([a for _, a, b in pairs])),
                candidate_mean=float(np.mean([b for _, a, b in pairs])),
                difference=float(np.mean([b-a for _, a, b in pairs])),
                ci95=np.quantile(draws, [.025, .975]).tolist(), bootstrap_replicates=replicates,
                difference_definition='unified minus ratio', resampling_unit='whole paired seed across selected scenarios')


def load_confirmation(folder):
    hashes = {}
    def read(path):
        path = local(path)
        hashes[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))
    manifest = read(folder / 'manifest.json')
    completion = read(folder / 'execution_complete.json')
    trials = read(folder / 'trials.json')
    canonical = read(folder / 'analysis/summary.json')
    if manifest.get('phase') != 'confirmation' or not manifest.get('complete_phase_requested'):
        raise ValueError('Only the complete registered confirmation phase is a formal reporting source')
    if not completion.get('execution_complete') or completion.get('trials') != 256 or len(trials) != 256:
        raise ValueError('Wait for all 256 assigned confirmation trials before formal reporting')
    if completion.get('manifest_sha256') != sha(folder / 'manifest.json') or completion.get('trials_sha256') != sha(folder / 'trials.json'):
        raise ValueError('Execution-completion hash chain does not match')
    expected = {key(row) for row in manifest['planned_jobs']}
    actual = {key(row) for row in trials}
    if len(expected) != 256 or len(actual) != 256 or actual != expected:
        raise ValueError('Missing, duplicated, or unexpected confirmation assignments')
    seeds = sorted({row['seed'] for row in trials})
    if len(seeds) != 8 or any(row.get('split') != 'confirmation' or row.get('mode') != 'scheduled' for row in trials):
        raise ValueError('Mixed splits, modes, or seed counts')
    records = canonical.get('records', [])
    if len(records) != 256 or {key(row) for row in records} != expected:
        raise ValueError('Canonical evaluation does not cover the same assigned matrix')
    original = {key(row): row for row in trials}
    for row in records:
        result = original[key(row)]
        for name in ('task_success', 'A_S', 'static_eligible', 'mean_force', 'finger_tv_rate',
                     'record_complete', 'release_success', 'local_domain_boundary_fraction', 'safety'):
            if row.get(name) != result.get(name):
                raise ValueError(f'Canonical evaluator and saved result disagree: {key(row)} / {name}')
    source_hashes = {}
    for name, expected_sha in manifest['source_sha256'].items():
        source = local(name)
        source_hashes[name] = sha(source)
        if source_hashes[name] != expected_sha:
            raise ValueError(f'Frozen source identity no longer matches: {name}')
    admission = read(ROOT / 'review/admission_manifest.json')
    if sha(ROOT / 'review/admission_manifest.json') != manifest['admission_sha256']:
        raise ValueError('Current admission does not match this confirmation run')
    plan_entry = admission['evidence']['experiment_plan']
    plan = read(ROOT / plan_entry['path'])
    if plan_entry['sha256'] != manifest['experiment_plan_sha256'] or sha(ROOT / plan_entry['path']) != plan_entry['sha256']:
        raise ValueError('Registered experiment-plan identity does not match')
    if plan['statistics']['bootstrap_replicates'] != 5000 or plan['phases']['confirmation']['expected_trials'] != 256:
        raise ValueError('Unexpected registered analysis design')
    phase = plan['phases']['confirmation']
    shape_by_scene = {scene['name']:scene['shape'] for scene in plan['scenarios']}
    registered = {(scene,shape_by_scene[scene],int(seed),method)
                  for scene in phase['scenarios'] for seed in phase['seeds'] for method in plan['methods']}
    if registered != expected or set(plan['methods']) != set(METHODS) or seeds != sorted(phase['seeds']):
        raise ValueError('Assigned matrix is not the exact registered scenario × seed × method Cartesian product')
    scene_specs = {scene['name']:scene for scene in plan['scenarios']}
    if any(bool(row.get('boundary')) != bool(scene_specs[row['scenario']].get('boundary',False)) for row in records):
        raise ValueError('Boundary labels differ from the registered scenario definitions')
    raw_missing = []
    for row in records:
        run = folder / 'runs' / f"{row['scenario']}__{row['shape']}__{row['method']}__{row['seed']}"
        saved_result = read(run / 'result.json')
        if saved_result != original[key(row)]:
            raise ValueError(f'Per-trial result differs from the matrix trials file: {run}')
        meta = read(run / 'metadata.json')
        limits, budget = np.asarray(meta.get('limits',[]),float), meta.get('budget')
        row['analysis_stage_substep_protection'] = {}
        if limits.shape == (5,) and np.isfinite(limits).all() and finite(budget):
            for stage, values in row.get('safety',{}).get('stages',{}).items():
                row['analysis_stage_substep_protection'][stage] = bool(row.get('record_complete') and
                    np.all(np.asarray(values['finger_peak_N']) <= limits+.02) and
                    values['total_finger_peak_N'] <= float(budget)+.02)
        trace = run / 'trace.npz'
        if not trace.exists():
            raw_missing.append(str(trace.relative_to(ROOT)))
            continue
        actual_sha = sha(trace)
        hashes[str(trace.relative_to(ROOT))] = actual_sha
        if actual_sha != meta.get('trace_sha256') or actual_sha != row.get('trace_sha256'):
            raise ValueError(f'Original physical trace hash does not match: {trace}')
        if meta.get('source_sha256') != manifest['source_sha256'] or meta.get('admission_sha256') != manifest['admission_sha256']:
            raise ValueError(f'Trial provenance is not the frozen paired source: {run}')
    return records, canonical, plan, dict(input_sha256_before=hashes, frozen_source_sha256=source_hashes,
        raw_missing=raw_missing, raw_trace_count=256-len(raw_missing), assigned_trials=256, paired_assignments=128,
        seeds=seeds, canonical_records_match_saved_results=True)


def load_recovery(path, records, audit):
    path = local(path)
    result = json.loads(path.read_text(encoding='utf-8'))
    plan_path, analyzer_path = ROOT/'review/recovery_plan_v1.json', ROOT/'evaluate_recovery.py'
    if result.get('schema') != 'secondary_event_recovery_results_v1' or result.get('role') != 'descriptive_secondary_only':
        raise ValueError('Recovery input must be the independent registered secondary analyzer output')
    if result.get('recovery_plan_sha256') != sha(plan_path) or result.get('analyzer_sha256') != sha(analyzer_path):
        raise ValueError('Independent recovery analyzer/plan identity does not match')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    primary = {key(row): row for row in records}
    seen, events = set(), []
    statuses = {'invalid_record','failed_before','not_exposed','not_stable_before','never_unstable',
                'recovered_during','recovered_after','unrecovered','release_preempt'}
    for row in result['records']:
        identity = key(row)
        if identity not in primary or identity in seen:
            raise ValueError('Recovery records are duplicated or use a different trial/trace')
        invalid = not row.get('record_complete') and all(event['status'] == 'invalid_record' for event in row['events'])
        if row.get('trace_sha256') != primary[identity].get('trace_sha256') and not (invalid and row.get('trace_sha256') is None):
            raise ValueError('Recovery source trace differs from the original assigned trial')
        seen.add(identity)
        expected_events = {event['event_id']:event for event in plan['events'][row['scenario']]}
        actual_ids = [event['event_id'] for event in row['events']]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_events):
            raise ValueError('Recovery trial dropped, duplicated, or inserted a registered event')
        for event in row['events']:
            spec = expected_events[event['event_id']]
            if any(event.get(name) != value for name,value in spec.items()):
                raise ValueError('Recovery event schedule/specification differs from its fixed plan')
            if event['status'] not in statuses:
                raise ValueError('Unknown recovery state')
            if event['status'] == 'never_unstable' or event.get('censored') or not event.get('stable_before'):
                if event.get('latency_from_event') is not None or event.get('latency_from_instability') is not None:
                    raise ValueError('Non-recovered/never-unstable events must not masquerade as zero latency')
            if event.get('recovered') and not finite(event.get('latency_from_instability')):
                raise ValueError('Recovered event needs an actual finite conditional latency')
            events.append({**{name: primary[identity][name] for name in ('method','shape','scenario','seed','boundary')}, **event})
    if seen != set(primary):
        raise ValueError('Recovery output must retain every assigned trial, including every invalid record')
    if len(result['rows']) != 2 or {row['method'] for row in result['rows']} != set(METHODS):
        raise ValueError('Recovery needs one full-denominator aggregate for each registered method')
    for row in result['rows']:
        selected = [event for event in events if event['method'] == row['method']]
        if len(selected) != row['scheduled_events'] or dict(Counter(event['status'] for event in selected)) != row['status_counts']:
            raise ValueError('Recovery aggregate status counts disagree with recorded events')
        eligible = [event for event in selected if event['stable_before'] and event['exposed'] and event['status'] not in ('failed_before','invalid_record')]
        unstable = [event for event in eligible if event['first_instability'] is not None]
        recovered = [event for event in unstable if event['recovered']]
        expected = dict(trials=sum(record['method'] == row['method'] for record in result['records']),
                        trials_with_events=sum(record['method'] == row['method'] and bool(record['events']) for record in result['records']),
                        eligible_events=len(eligible),unstable_eligible_events=len(unstable),recovered_events=len(recovered),
                        recovery_fraction=len(recovered)/len(unstable) if unstable else None,
                        recovered_latency_median=float(np.median([event['latency_from_instability'] for event in recovered])) if recovered else None)
        if any(row.get(name) != value for name,value in expected.items()):
            raise ValueError('Recovery eligibility/recovered/conditional-latency aggregate does not reconcile with full events')
    for source in (path, plan_path, analyzer_path):
        audit['input_sha256_before'][str(source.relative_to(ROOT))] = sha(source)
    return dict(**result, flattened_events=events, missing_primary_records=len(primary)-len(seen))


def self_test():
    checks = []
    synthetic = []
    for seed in (1, 2):
        for scenario in ('a', 'b'):
            for method in METHODS:
                synthetic.append(dict(scenario=scenario,shape='capsule',seed=seed,method=method,
                    static_eligible=not(seed == 2 and method == 'unified'),mean_force=2. if method == 'ratio' else 3.))
    values = paired(synthetic, 'mean_force', True)
    checks.append(len(values) == 2 and all(identity[2] == 1 for identity, _, _ in values))
    stats = bootstrap_pairs(values)
    checks.append(stats['pairs'] == 2 and stats['clusters'] == 1 and stats['ci95'] == [1., 1.])
    checks.append(bootstrap_pairs([])['difference'] is None)
    missing = [dict(row, mean_force=None) if row['method'] == 'unified' else row for row in synthetic]
    checks.append(not paired(missing, 'mean_force', True))
    release = [dict(row,release_latency=.5,release_success=row['method'] == 'unified') for row in synthetic]
    checks.append(not paired(release,'release_latency'))
    unbalanced = [((f's{j}','capsule',i),0.,float(i)) for i in range(8) for j in range(i+1)]
    resampled = bootstrap_pairs(unbalanced, 100)
    rng = np.random.default_rng(20261103); brute = []
    for _ in range(100):
        draw = rng.integers(0,8,size=8)
        delta = [b-a for seed in draw for identity,a,b in unbalanced if identity[2] == seed]
        brute.append(float(np.mean(delta)))
    checks.append(np.allclose(resampled['ci95'],np.quantile(brute,[.025,.975]),rtol=0,atol=1e-12))
    checks.append(resampled['clusters'] == 8 and resampled['pairs'] == 36)
    if not all(checks):
        raise AssertionError(f'Analysis cohort/cluster bootstrap checks failed: {checks}')
    return dict(passed=True, checks=len(checks), synthetic_only=True,
                checked='joint eligibility, missing-not-zero, empty cohort, whole-seed resampling with unequal eligible pair counts')


def project_overview():
    sources = {}
    def read(name):
        path = ROOT/name
        sources[name] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))
    design = read('review/foundation_design_record.json')
    physical = read('review/physical_independent_acceptance.json')
    gui = read('review/actual_browser_run_evidence_v1.json')
    deterministic = read('output/determinism_demo_v1/determinism_recheck.json')
    interface = read('review/runtime_static_contract.json')
    preserved = read('review/source_provenance.json')
    migration = read('review/policy_migration_provenance.json')
    if not all(value.get('passed') for value in (physical,gui,deterministic,interface)):
        raise ValueError('Static project overview must not claim acceptance from a failed source')
    stats = design['final_statistics']
    checks = sum(method['render_state_checks'] for method in deterministic['methods'])
    text = f"""## 整体接入架构与研究范围

独立接入版保留原Shadow Hand的24个手/腕关节、18个手指驱动加2个腕驱动、腱耦合、视觉网格与原惯量。物体具有独立自由体，执行器力有限，不焊接物体、不裁剪实际接触力。公开共享接触近似采用condim3点接触与随指节运动的光滑指端；盒体是27个同body几何组成的2mm圆角盒。它是明确声明的完整手接触近似，不等于原始视觉网格接触精确复现。

整个控制链为“共同准备 → 全皮肤多patch观测与状态估计 → 当前冻结基线/融合策略二选一 → 共享带符号耦合预测与可行动作约束 → 相同真实关节执行 → 原始数据和可视化”。两方法沿用当前策略上层规则和目标；迁移公开替换分布接触剪切/滑移量及共同耦合物理模型，不能称为旧五触点执行模型的原样复用。公共适配的实际变换另见policy_migration_provenance.json。

共同准备0–7.5s使用全部外物皮肤压力和公开固定抓姿，低力建立接触后以0.10m为抬升目标；只有公共建立与固定局部域交接检查均通过，才把持握交给两策略。尺寸、质量或扰动场景不保证交接必达，失败保留完整分母。控制器不读取物体身份、质量或真实物体位姿。胶囊体参考闭合坐标附近最大验证半径为±0.02，圆角盒和球体±0.04；都绑定各自独立holdout及误差界。12s统一请求释放，两法共享卸力速率和拇指清离轨迹；没有以持续持握放回台面替代主动释放。

## 物理与接口验收

在策略性能测试前完成三形状×接触solref0.008/0.012×积分步长0.5/0.25ms的12格共同物理验收，并通过独立证书。最大全程接触穿透{stats['max_penetration']*1000:.3f}mm，最小逐样本实际物体抬升{stats['minimum_object_lift']*1000:.3f}mm；步长减半最大静态均力相对差{stats['max_mean_step_difference']*100:.2f}%，完整生命周期峰力相对差{stats['max_peak_step_difference']*100:.2f}%。原深侵入、原网格/6D接触、尖角盒峰力不收敛等失败设计均保留，不计入通过集合。物理数值验收也不是材料损伤安全证明，正式任务力保护仍需按原限值逐子步评价。

运行接口静态合同{interface['check_count']}项通过。固定demo种子540101、胶囊体、两方法的4条完整16s轨迹做了有/无绘图逐值对照，物理、完整观测、原始patch、动作与子步力一致；{checks}次绘图前后状态签名不变。UUID、墙钟计时与相机显示状态明确排除。这是已测胶囊体同条件确定性，不外推为所有形状GUI轨迹测试。

实际浏览器操作证据7项通过：方法/形状切换、暂停、手动扰动/挤压/释放、重置及新局身份均记录。IAB截图工具超时，因此该证据不宣称整页截图布局验收通过；实际物理帧及真实DOM回归另存。GUI展示与批量比较共用固定5ms运行核，网页读取使用缓存快照，不因轮询推进物理。

## 原平台与历史证据保留

全部新增实现与输出位于独立接入目录。启动前已建立{preserved['preserved_files']}个旧文件、{preserved['preserved_bytes']:,}字节的逐文件保留清单，范围包括旧五指平台与冻结五触点实验；旧实验不与本轮256次混合。初始保留清单不代替任务结束后的全文件重核，最终保全结果应另附实际核验文件。旧展示端口与新接入平台分开；开发、demo及正式确认种子分离，所有失败原始结果保留。
"""
    return text, sources


def collect_metrics(records, canonical):
    comparisons, counts, stages = [], [], []
    canonical_map = {(row['domain'], row['endpoint']): row for row in canonical['comparisons'] if 'cohort' not in row}
    endpoints = [('task_success', '完整任务完成率', 'fraction', False),
                 ('force_protection_pass', '全程力保护率', 'fraction', False),
                 ('release_success', '计划释放成功率', 'fraction', False),
                 ('release_exposed', '真实承重释放暴露率', 'fraction', False),
                 ('A_S', '接管后首次稳定等待', 's', False),
                 ('finger_tv_rate', '有效窗逐指TV率', 'N/s', True),
                 ('mean_force', '有效窗平均总指力', 'N', True),
                 ('stable_hold_fraction', '持握稳定采样比例', 'fraction', False),
                 ('local_domain_boundary_fraction', '持握局部域边界占用', 'fraction', False),
                 ('static_window_domain_boundary_fraction', '静态窗局部域边界占用', 'fraction', True),
                 ('execution_request_mismatch_fraction', '持握请求执行不一致', 'fraction', False)]
    endpoints.append(('release_latency','成功释放延迟','s',False))
    for group, title, selected in cohorts(records):
        for method in METHODS:
            rows = [row for row in selected if row['method'] == method]
            counts.append(dict(group=group, title=title, method=method, assigned=len(rows),
                task_success=sum(bool(row.get('task_success')) for row in rows),
                record_complete=sum(bool(row.get('record_complete')) for row in rows),
                preparation_passed=sum(bool(row.get('preparation_passed')) for row in rows),
                physical_established=sum(row.get('established') is not None for row in rows),
                policy_epoch_trials=sum('policy_holding' in row.get('safety',{}).get('stages',{}) for row in rows),
                static_eligible=sum(bool(row.get('static_eligible')) for row in rows),
                force_protection_pass=sum(bool(field(row, 'force_protection_pass')) for row in rows),
                safety_records=sum(bool(row.get('safety')) for row in rows),
                release_planned=sum(bool(row.get('release_planned')) for row in rows),
                release_exposed=sum(bool(row.get('release_exposed')) for row in rows),
                release_success=sum(bool(row.get('release_success')) for row in rows),
                actual_fixture_exposure=sum(bool(row.get('actual_collision_exposed')) for row in rows),
                failures=dict(Counter(row.get('failure_reason') for row in rows if row.get('failure_reason')))))
        for endpoint, label, unit, quality in endpoints:
            values = paired(selected, endpoint, quality)
            stats = bootstrap_pairs(values)
            if (group, endpoint) in canonical_map:
                official = canonical_map[group, endpoint]
                for name in ('pairs', 'clusters', 'baseline_mean', 'candidate_mean', 'difference'):
                    if stats[name] is None or abs(stats[name] - official[name]) > 1e-10:
                        raise ValueError(f'Canonical primary point estimate or pairing mismatch: {group}/{endpoint}/{name}')
                stats = {**stats, **official, 'interval_source': 'frozen evaluate_runs canonical comparison'}
            comparisons.append(dict(group=group, title=title, endpoint=endpoint, label=label, unit=unit,
                cohort=('both methods successful release only; failed/nonexposed latency remains null' if endpoint == 'release_latency'
                        else 'joint complete supported static window' if quality else 'joint assigned rows with available value'),
                role='preregistered primary' if group in ('main', 'boundary') and endpoint in ('A_S', 'finger_tv_rate') else 'descriptive secondary', **stats))
        for stage in STAGES:
            for method in METHODS:
                rows = [row for row in selected if row['method'] == method]
                values = [row['safety']['stages'][stage] for row in rows if stage in row.get('safety', {}).get('stages', {})]
                if not values:
                    continue
                peak = [max(value['finger_peak_N']) for value in values]
                stage_rows = [row for row in rows if stage in row.get('safety', {}).get('stages', {})]
                guard_values = [row['analysis_stage_substep_protection'][stage] for row in stage_rows
                                if stage in row.get('analysis_stage_substep_protection',{})]
                stages.append(dict(group=group, title=title, method=method, stage=stage, stage_title=STAGES[stage],
                    observed_stages=len(values), assigned=len(rows),
                    stage_substep_protection_passes=sum(guard_values), stage_guard_records=len(guard_values),
                    stage_substep_protection_fraction=float(np.mean(guard_values)) if guard_values else None,
                    mean_duration_s=float(np.mean([value['duration_s'] for value in values])),
                    mean_local_excess_integral_Ns=float(np.mean([value['local_excess_integral_Ns'] for value in values])),
                    mean_total_excess_integral_Ns=float(np.mean([value['total_excess_integral_Ns'] for value in values])),
                    mean_per_trial_max_finger_N=float(np.mean(peak)), maximum_finger_N=float(max(peak)),
                    mean_per_trial_total_peak_N=float(np.mean([value['total_finger_peak_N'] for value in values]))))
    return comparisons, counts, stages


def plotting(out):
    os.environ['MPLCONFIGDIR'] = str(out / '.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    path = Path('C:/Windows/Fonts/msyh.ttc')
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(path)).get_name()
        plt.rcParams['font.family'] = family
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11, 'axes.titlesize': 13,
                         'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white', 'axes.axisbelow': True})
    return plt


def bars(ax, categories, series, ylabel, title, rate=False):
    x = np.arange(len(categories))
    for index, method in enumerate(METHODS):
        values = [row.get(method, (None, ''))[0] for row in series]
        heights = [np.nan if value is None else value * (100 if rate else 1) for value in values]
        artists = ax.bar(x + (index-.5)*.36, heights, .34, label=TITLES[method], color=COLORS[method],
                         hatch='//' if method == 'unified' else None, edgecolor='white', linewidth=.8)
        for bar, row, value in zip(artists, series, values):
            if value is not None:
                label = row.get(method, (None, ''))[1]
                ax.annotate(label, (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0, 4),
                            textcoords='offset points', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left')
    ax.set_ylim(bottom=0)
    if rate:
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
    else:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top*1.18 if top > 0 else 1)
    ax.grid(axis='y', alpha=.2)


def make_figures(out, comparisons, counts, stages):
    plt = plotting(out)
    plots = []
    compact = [(g, t) for g, t in [('main', '主场景'), ('boundary', '边界场景')] if any(r['group'] == g for r in counts)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, endpoint, title in zip(axes.flat,
            ['task_success', 'force_protection_pass', 'release_success', 'release_exposed'],
            ['完整任务完成', '完整16s逐子步力保护', '计划释放成功', '真实承重释放暴露']):
        series = []
        for group, _ in compact:
            row = {}
            for method in METHODS:
                source = next(r for r in counts if r['group'] == group and r['method'] == method)
                n = source['release_planned'] if endpoint.startswith('release') else source['assigned']
                numerator = source[endpoint]
                row[method] = (numerator/n if n else None, f'{numerator}/{n}')
            series.append(row)
        bars(ax, [t for _, t in compact], series, '%', title, True)
    axes[0, 0].legend(loc='lower left', frameon=False)
    fig.suptitle('任务与安全：失败保留在计划分母中', fontsize=16)
    fig.text(.02, .015, '力保护覆盖全部五指外物皮肤；掌部无损伤阈值。释放暴露单列，未暴露不能算完成释放。', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .95));fig.savefig(out/'01_task_safety.png', dpi=160);plt.close(fig)
    plots.append('01_task_safety.png')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, endpoint, unit, title in zip(axes, ['A_S', 'finger_tv_rate'], ['s', 'N/s'],
            ['接管后首次稳定等待 A_S', '共同有效窗逐指TV率']):
        rows = [next(r for r in comparisons if r['group'] == group and r['endpoint'] == endpoint) for group, _ in compact]
        for y, row in enumerate(rows):
            if row['difference'] is None:
                continue
            lo, hi = row['ci95'];value = row['difference']
            ax.hlines(y, lo, hi, color=COLORS['unified'], lw=1.5)
            ax.plot([lo,hi], [y,y], '|', color=COLORS['unified'], ms=9)
            ax.plot(value, y, 'D', color=COLORS['unified'])
            ax.annotate(f"Δ={value:+.3g} [{lo:+.3g}, {hi:+.3g}]", (value, y), xytext=(0, 13),
                        textcoords='offset points', ha='center', fontsize=10)
        ax.axvline(0, color='#405568', lw=1)
        ax.set_yticks(range(len(rows)), [f"{r['title']}  n={r['pairs']}配对 / k={r['clusters']}种子" for r in rows])
        ax.set_ylim(-.55, len(rows)-.25)
        ax.set_xlabel(f'融合 − 基线 / {unit}（负值表示更低）')
        ax.set_title(title, loc='left');ax.grid(axis='x', alpha=.2)
    fig.suptitle('预登记主指标：原始配对差与种子聚类95%CI', fontsize=15)
    fig.text(.02, .015, 'A_S未达到记1.4s；TV只用双方共同有效窗，覆盖另列。区间和Holm来自冻结评价器。', fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .92));fig.savefig(out/'02_primary_endpoints.png', dpi=160);plt.close(fig)
    plots.append('02_primary_endpoints.png')
    shape_groups = [(r['group'], r['title'].replace('·', '\n')) for r in counts if r['method'] == 'ratio' and '/' in r['group']]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, endpoint, unit, title in zip(axes, ['mean_force', 'finger_tv_rate', 'A_S'], ['N', 'N/s', 's'],
            ['共同有效窗平均总指力', '共同有效窗逐指TV率', '接管后首次稳定等待']):
        series = []
        for group, _ in shape_groups:
            row = next(r for r in comparisons if r['group'] == group and r['endpoint'] == endpoint)
            series.append({method: (row['baseline_mean' if method == 'ratio' else 'candidate_mean'],
                                   f"n={row['pairs']}") for method in METHODS})
        bars(ax, [t for _, t in shape_groups], series, unit, title)
    axes[0].legend(frameon=False)
    fig.suptitle('分形状与边界：描述性比较，不能替代主指标检验', fontsize=15)
    fig.text(.02, .015, '每柱n为同一条件配对数；力与TV仅共同有效窗，缺窗不填0。A_S未达到稳定计1.4s。', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .94));fig.savefig(out/'03_shape_quality.png', dpi=160);plt.close(fig)
    plots.append('03_shape_quality.png')
    selected_stages = [stage for stage in STAGES if any(r['group'] == 'all' and r['stage'] == stage for r in stages)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, metric, unit, title in zip(axes.flat,
            ['mean_local_excess_integral_Ns', 'mean_total_excess_integral_Ns', 'mean_per_trial_max_finger_N','stage_substep_protection_fraction'],
            ['N·s', 'N·s', 'N','%'], ['逐指局部超限积分', '总力预算超限积分', '每试验单指最大力的均值','阶段逐子步力保护']):
        series = []
        for stage in selected_stages:
            item = {}
            for method in METHODS:
                row = next((r for r in stages if r['group'] == 'all' and r['stage'] == stage and r['method'] == method), None)
                annotation = (f"{row['stage_substep_protection_passes']}/{row['stage_guard_records']}" if metric == 'stage_substep_protection_fraction'
                              else f"{row['observed_stages']}/{row['assigned']}") if row else ''
                item[method] = (row[metric], annotation) if row else (None, '')
            series.append(item)
        bars(ax, [STAGES[s].replace('及之后', '\n及之后') for s in selected_stages], series, unit, title,metric == 'stage_substep_protection_fraction')
    axes[0,0].legend(frameon=False)
    fig.suptitle('分阶段安全：共同准备、策略持握和机械释放', fontsize=15)
    fig.text(.02, .015, '前三图标注=可观测阶段/分配；保护标注=通过/有原限值阶段。缺失阶段不补0；全程含终点保护率见图1。', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .94));fig.savefig(out/'04_stage_safety.png', dpi=160);plt.close(fig)
    plots.append('04_stage_safety.png')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, endpoint, title in zip(axes,
            ['local_domain_boundary_fraction', 'static_window_domain_boundary_fraction', 'execution_request_mismatch_fraction'],
            ['实际持握域边界占用', '静态有效窗域边界占用', '持握请求与执行不一致']):
        series = []
        for group, _ in shape_groups:
            row = next(r for r in comparisons if r['group'] == group and r['endpoint'] == endpoint)
            series.append({method: (row['baseline_mean' if method == 'ratio' else 'candidate_mean'],
                                   f"n={row['pairs']}") for method in METHODS})
        bars(ax, [t for _, t in shape_groups], series, '%', title, True)
    axes[0].legend(frameon=False)
    fig.suptitle('动作域约束：低波动结果的共同解释条件', fontsize=15)
    fig.text(.02, .015, '仅实际策略持握采样；静态窗须双方共同有效。任一五协同坐标触边，不代表全部关节饱和。', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .94));fig.savefig(out/'05_domain_occupancy.png', dpi=160);plt.close(fig)
    plots.append('05_domain_occupancy.png')
    return plots


def recovery_figure(out, recovery):
    plt = plotting(out)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3), gridspec_kw={'width_ratios':[1,1.25]})
    fields = [('scheduled_events','计划事件'),('eligible_events','事件前稳且暴露'),
              ('unstable_eligible_events','其中曾失稳'),('recovered_events','其中已恢复')]
    series = [{method:(float(next(row for row in recovery['rows'] if row['method'] == method)[name]),
                       str(next(row for row in recovery['rows'] if row['method'] == method)[name]))
               for method in METHODS} for name,_ in fields]
    bars(axes[0],[title.replace('且','\n且') for _,title in fields],series,'事件数','事件覆盖与条件恢复分母')
    labels = {'invalid_record':'无效记录','failed_before':'事件前已失败','not_exposed':'未实际暴露',
              'not_stable_before':'事件前未稳定','never_unstable':'从未失稳','recovered_during':'扰动期间恢复',
              'recovered_after':'扰动结束后恢复','unrecovered':'未恢复/下事件截断','release_preempt':'释放截断'}
    y = np.arange(len(labels))
    for index,method in enumerate(METHODS):
        row = next(row for row in recovery['rows'] if row['method'] == method)
        counts = [row['status_counts'].get(status,0) for status in labels]
        bars_ = axes[1].barh(y+(index-.5)*.34,counts,.32,color=COLORS[method],
                            hatch='//' if method == 'unified' else None,label=TITLES[method])
        for artist,value in zip(bars_,counts):
            axes[1].text(value+.3,artist.get_y()+artist.get_height()/2,str(value),va='center',fontsize=9)
    axes[1].set_yticks(y,list(labels.values()));axes[1].invert_yaxis()
    axes[1].set_xlabel('事件数');axes[1].set_title('所有事件状态：未恢复和无适用条件保留',loc='left')
    axes[1].grid(axis='x',alpha=.2);axes[1].margins(x=.16)
    axes[0].legend(frameon=False)
    fig.suptitle('事件恢复：独立补充分析，不替代主A_S',fontsize=16)
    fig.text(.02,.015,'从未失稳/未暴露/截断均不是0秒恢复；延迟只在已恢复条件下描述。已见开发6条、未见确认结果时固定；不作新增主检验。',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(out/'06_event_recovery.png',dpi=160);plt.close(fig)
    return '06_event_recovery.png'


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def report_text(records, comparisons, counts, stages, plots, audit, recovery=None, overview=''):
    lines = ['# 新完整五指平台：确认试验结果草稿', '', overview, '',
             '本报告仅包含新平台正式256次确认试验。完整任务、安全和覆盖先列；力波动只在共同有效窗口中比较。', '',
             '## 全体分母与任务表现', '',
             '|分组|方法|完成/分配|力保护/分配|有效窗/分配|释放暴露/计划|成功释放/计划|',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in counts:
        lines.append(f"|{row['title']}|{TITLES[row['method']]}|{row['task_success']}/{row['assigned']}|{row['force_protection_pass']}/{row['assigned']}|{row['static_eligible']}/{row['assigned']}|{row['release_exposed']}/{row['release_planned']}|{row['release_success']}/{row['release_planned']}|")
    lines += ['', '## 公共准备、交接域与实际策略覆盖', '',
              '建立持握、通过交接和执行策略是不同层次。交接失败或局部域不满足不进入策略；不能把共同准备的失败说成已执行策略的失抓，也不能从任务失败分母删除。', '',
              '|分组|方法|公共建立/分配|交接准入/分配|实际策略持握阶段/分配|',
              '|---|---|---:|---:|---:|']
    for row in counts:
        lines.append(f"|{row['title']}|{TITLES[row['method']]}|{row['physical_established']}/{row['assigned']}|{row['preparation_passed']}/{row['assigned']}|{row['policy_epoch_trials']}/{row['assigned']}|")
    lines += ['', '## 预登记主指标', '',
              'Δ定义为融合−基线。A_S单位s，包含未达到1.4s贡献；TV单位N/s，按双方共同有效窗条件化。它们分别回答首次稳定等待和持握力涨落，不能相互替代。', '',
              '|分组|指标|基线|融合|Δ [聚类95%CI]|共同配对|种子簇|Holm p|',
              '|---|---|---:|---:|---|---:|---:|---:|']
    for row in comparisons:
        if row['role'] != 'preregistered primary':
            continue
        interval = '不可估计' if row['ci95'] is None else f"{number(row['difference'])} [{number(row['ci95'][0])}, {number(row['ci95'][1])}]"
        lines.append(f"|{row['title']}|{row['label']} ({row['unit']})|{number(row['baseline_mean'])}|{number(row['candidate_mean'])}|{interval}|{row['pairs']}|{row['clusters']}|{number(row.get('p_holm'),4)}|")
    lines += ['', '区间按整个种子跨场景配对抽样5000次；只有8个登记种子簇，不能把256次当作256个独立样本。分形状与其他次要差值CI见paired_comparisons.csv，不新增主显著性结论。', '',
              '## 力品质、阶段安全与动作域', '',
              '均力、TV和静态域占用均使用双方共同有效窗；失败和缺窗在上表分母中保留。所有实际阶段的局部/总超限积分及峰值分别列于stage_safety.csv。共同准备、策略持握和机械释放必须分开解释；释放执行轨迹两法共享。', '',
              '域边界占用与请求兑现用于解释受约束策略行为，不能把高占用时较低TV直接归因于更自由或更强的优化能力。', '']
    lines += ['成功释放延迟仅在双方均真正承重且完成释放的配对中比较，失败不记0秒：', '',
              '|分组|共同成功释放配对|基线延迟 s|融合延迟 s|Δ s [聚类95%CI]|',
              '|---|---:|---:|---:|---|']
    for row in comparisons:
        if row['group'] in ('all','main','boundary') and row['endpoint'] == 'release_latency':
            interval = '不可估计' if row['ci95'] is None else f"{number(row['difference'])} [{number(row['ci95'][0])}, {number(row['ci95'][1])}]"
            lines.append(f"|{row['title']}|{row['pairs']}|{number(row['baseline_mean'])}|{number(row['candidate_mean'])}|{interval}|")
    lines.append('')
    for name in plots:
        lines += [f'![{name}]({name})', '']
    lines += ['## 失败、覆盖与解释边界', '']
    for method in METHODS:
        selected = [row for row in records if row['method'] == method]
        failures = dict(Counter(row.get('failure_reason') for row in selected if row.get('failure_reason')))
        lines.append(f'- {TITLES[method]}：失败原因 {json.dumps(failures, ensure_ascii=False)}；完整原始记录 {sum(bool(row.get("record_complete")) for row in selected)}/{len(selected)}。')
    if recovery:
        lines += ['', '## 事件恢复（描述性补充）', '',
                  '该分析在看过六次开发烟测、尚未查看确认结果时独立固定，不属于新增主检验。A_S继续表示接管后首次稳定等待。', '',
                  '|方法|计划事件|前稳且暴露|其中曾失稳|其中已恢复|恢复率条件分母|已恢复延迟中位数 s|',
                  '|---|---:|---:|---:|---:|---|---:|']
        for row in recovery['rows']:
            lines.append(f"|{TITLES[row['method']]}|{row['scheduled_events']}|{row['eligible_events']}|{row['unstable_eligible_events']}|{row['recovered_events']}|{row['recovered_events']}/{row['unstable_eligible_events']}|{number(row['recovered_latency_median'])}|")
        lines += ['', '延迟只在已恢复事件条件下描述；never_unstable延迟为null，未恢复/释放或下一事件截断保留状态。全事件状态及形状/boundary来源见recovery_events.csv。']
    lines += ['', f"原始NPZ可核验 {audit['raw_trace_count']}/256；所有输入与冻结来源SHA见source_audit.json。",
              '现有范围为真实24关节/20手驱动和腱耦合的简化接触完整手模型、虚拟皮肤、掌部被动承重、共同固定抓姿准备及公开局部动作域。不能外推至硬件、任意抓姿规划或全任务自主抓取。', '',
              '事件恢复属于独立次要描述；接管后A_S约0.2s下限不代表扰动后0.2s恢复。', '',
              '本工具没有构造事后加权综合分数；项目应同时陈述完成、安全、稳定与用力代价，并保留各自的失败和适用分母。', '',
              '方法全文见analysis_methods.md；图形精确数据见chart_data.json及各CSV。']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    began = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', help='Complete confirmation run directory; canonical analysis/summary.json must already exist')
    parser.add_argument('--out', required=True, help='Fresh report/<name> directory')
    parser.add_argument('--methods-only', action='store_true')
    parser.add_argument('--overview-only', action='store_true', help='Write verified project architecture/acceptance front matter without reading confirmation performance')
    parser.add_argument('--recovery', help='Optional registered secondary_event_recovery_results_v1 JSON')
    args = parser.parse_args(argv)
    out = local(args.out)
    if not out.is_relative_to(ROOT/'report') or out == ROOT/'report' or out.exists():
        raise ValueError('Choose a fresh directory below report/')
    checks = self_test()
    if args.overview_only:
        text,sources = project_overview()
        out.mkdir(parents=True,exist_ok=False)
        (out/'project_overview.md').write_text(text,encoding='utf-8')
        write_json(out/'overview_source_sha256.json',dict(source_sha256=sources,contains_confirmation_performance=False,
            analysis_report_sha256=sha(Path(__file__))))
        print(json.dumps(dict(overview_only=True,out=str(out)),ensure_ascii=False));return
    if args.methods_only:
        out.mkdir(parents=True, exist_ok=False)
        (out/'analysis_methods.md').write_text(METHODS_TEXT, encoding='utf-8')
        write_json(out/'methods_provenance.json', dict(analysis_report_sha256=sha(Path(__file__)),
            contains_performance_results=False, primary_endpoints_unchanged=True,
            statistical_unit='paired whole seed across scenarios', planned_replicates=5000, analysis_self_test=checks))
        print(json.dumps(dict(methods_only=True, out=str(out)), ensure_ascii=False));return
    if not args.root:
        raise ValueError('--root is required for formal results')
    records, canonical, plan, audit = load_confirmation(local(args.root))
    overview,overview_sources = project_overview()
    audit['input_sha256_before'].update(overview_sources)
    recovery = load_recovery(args.recovery,records,audit) if args.recovery else None
    comparisons, counts, stages = collect_metrics(records, canonical)
    out.mkdir(parents=True, exist_ok=False)
    (out/'analysis_methods.md').write_text(METHODS_TEXT, encoding='utf-8')
    write_csv(out/'paired_comparisons.csv', comparisons)
    write_csv(out/'denominators.csv', counts)
    write_csv(out/'stage_safety.csv', stages)
    write_csv(out/'per_trial_metrics.csv', records)
    write_json(out/'chart_data.json', dict(comparisons=comparisons, denominators=counts, stages=stages))
    plots = make_figures(out, comparisons, counts, stages)
    if recovery:
        plots.append(recovery_figure(out,recovery))
        write_csv(out/'recovery_events.csv',recovery['flattened_events'])
        write_json(out/'recovery_descriptive_summary.json', {k:v for k,v in recovery.items() if k not in ('records','flattened_events')})
    for name, expected in audit['input_sha256_before'].items():
        if sha(ROOT/name) != expected:
            raise RuntimeError(f'Analysis input changed during reporting: {name}')
    for name, expected in audit['frozen_source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise RuntimeError(f'Frozen source changed during reporting: {name}')
    audit.update(inputs_unchanged=True, frozen_sources_unchanged=True, script_sha256=sha(Path(__file__)),
        created_utc=datetime.now(timezone.utc).isoformat(), elapsed_wall_seconds=perf_counter()-began,
        primary_statistics_source='frozen evaluator canonical summary; point estimates and pair counts independently checked',
        figures=plots, contains_development_results=False, analysis_self_test=checks)
    write_json(out/'source_audit.json', audit)
    (out/'results_report.md').write_text(report_text(records, comparisons, counts, stages, plots, audit,recovery,overview), encoding='utf-8')
    print(json.dumps(dict(passed=True, trials=len(records), out=str(out), figures=plots), ensure_ascii=False))


if __name__ == '__main__':
    main()

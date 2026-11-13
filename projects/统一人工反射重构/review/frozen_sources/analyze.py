"""Paired, seed-cluster analysis with gates and separate raw-unit endpoints."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import numpy as np


METHOD_LABELS = {
    "fixed": "固定握力", "friction_ratio": "摩擦比强基线", "ratio": "摩擦比强基线",
    "independent": "独立反射叠加", "unified": "统一协调",
    "unified_coordinator": "统一协调", "no_temporal": "无连续性代价",
    "no_geometry": "无接触几何耦合", "fixed_weights": "固定任务权重",
    "unified_no_continuity": "无连续性代价",
    "unified_no_geometry": "无接触几何耦合", "unified_fixed_stage": "固定任务权重",
}


def load_records(source):
    source = Path(source)
    if source.is_dir():
        paths = [source / "trials.json", source / "results.jsonl"]
        source = next((p for p in paths if p.exists()), paths[0])
    if source.suffix == ".jsonl":
        records = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        records = json.loads(source.read_text(encoding="utf-8-sig"))
        if isinstance(records, dict):
            records = records.get("trials", records.get("records", []))
    result = []
    for row in records:
        metrics = row.get("metrics")
        result.append({**row, **metrics} if isinstance(metrics, dict) else row)
    keys = [(r["split"], r["scenario"], r["seed"], r["method"]) for r in result]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate method/scenario/seed/split trials would corrupt pairing")
    return result


def _mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def _median(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def _metric(row, endpoint):
    if endpoint.startswith("static."):
        metrics = row.get("static_window", {}).get("metrics") or {}
        return metrics.get(endpoint.split(".", 1)[1])
    if endpoint.startswith("sensitivity."):
        metrics = row.get("static_window_sensitivity", {}).get("metrics") or {}
        return metrics.get(endpoint.split(".", 1)[1])
    if endpoint.startswith("safety."):
        return row.get("force_safety", {}).get(endpoint.split(".", 1)[1])
    if endpoint == "static_coverage":
        return row.get("static_window", {}).get("eligible", False)
    return row.get(endpoint)


DEFAULT_GATE_MARGINS = {
    "task_success": {"direction": "higher", "margin": -.05},
    "static_coverage": {"direction": "higher", "margin": -.05},
    "release_success": {"direction": "higher", "margin": -.05},
    "safety.force_protection_pass": {"direction": "higher", "margin": -.05},
    "safety.total_excess_integral": {"direction": "lower", "margin": .02},
    "safety.local_excess_integral": {"direction": "lower", "margin": .02},
}


def gate_judgement(estimate, definition):
    if estimate["pairs"] == 0:
        return "not_estimable"
    low, high = estimate["ci_low"], estimate["ci_high"]
    margin = definition["margin"]
    if definition["direction"] == "higher":
        return "supported" if low >= margin else "exceeds_margin" if high < margin else "inconclusive"
    return "supported" if high <= margin else "exceeds_margin" if low > margin else "inconclusive"


def summarize_method(records):
    count = len(records)
    eligible = [r for r in records if r["static_window"]["eligible"]]
    events = [e for r in records for e in r.get("events", [])]
    outcomes = Counter(e["outcome"] for e in events)
    release = [r for r in records if r.get("release_required")]
    return dict(
        title=records[0].get("scenario_title", records[0].get("title", records[0]["scenario"])) if records else None,
        trials=count,
        seeds=len({r["seed"] for r in records}),
        successes=sum(bool(r["task_success"]) for r in records),
        completion_rate=_mean([r["task_success"] for r in records]),
        establishment_failures=sum(bool(r["establishment_failure"]) for r in records),
        post_establishment_losses=sum(bool(r["post_establishment_loss"]) for r in records),
        failure_types=dict(Counter(r["end_reason"] for r in records if not r["task_success"])),
        force_protection_passes=sum(bool(r["force_safety"]["force_protection_pass"]) for r in records),
        force_protection_rate=_mean([r["force_safety"]["force_protection_pass"] for r in records]),
        complete_truth_records=sum(bool(r.get("truth_record_complete")) for r in records),
        complete_physical_force_records=sum(bool(r["force_safety"].get("full_substeps")) for r in records),
        valid_force_quality_records=sum(bool(r.get("force_quality_contract", {}).get("valid")) for r in records),
        complete_evidence_records=sum(bool(r.get("complete_evidence", r.get("truth_record_complete"))) for r in records),
        total_peak_max=max((_metric(r, "safety.total_peak") or 0.) for r in records),
        total_excess_integral_mean=_mean([_metric(r, "safety.total_excess_integral") for r in records]),
        A_S=_mean([r["restricted_unstable_time"] for r in records]),
        A_S_normalized=_mean([r["restricted_unstable_fraction"] for r in records]),
        stable_attained=sum(bool(r["first_stability_attained"]) for r in records),
        stable_attainment_rate=_mean([r["first_stability_attained"] for r in records]),
        first_stable_conditional_mean=_mean([r["first_stable_time"] for r in records]),
        first_stable_conditional_median=_median([r["first_stable_time"] for r in records]),
        stable_hold_fraction_mean=_mean([r["stable_hold_fraction"] for r in records]),
        static_eligible=len(eligible),
        static_coverage_rate=len(eligible) / count if count else None,
        static_low_contact_fraction=_mean([r["static_window"]["low_contact_fraction"] for r in records]),
        finger_tv_eligible_mean=_mean([_metric(r, "static.finger_tv_rate") for r in eligible]),
        finger_sd_eligible_mean=_mean([_metric(r, "static.finger_sd_mean") for r in eligible]),
        total_sd_eligible_mean=_mean([_metric(r, "static.total_sd") for r in eligible]),
        raw200_finger_tv_eligible_mean=_mean([_metric(r, "static.raw200_finger_tv_rate") for r in eligible]),
        substep_finger_tv_eligible_mean=_mean([_metric(r, "static.substep_finger_tv_rate") for r in eligible]),
        mean_force_eligible_mean=_mean([_metric(r, "static.mean_force") for r in eligible]),
        relative_drift_eligible_mean=_mean([_metric(r, "static.relative_drift_max") for r in eligible]),
        release_trials=len(release),
        release_successes=sum(bool(r["release_success"]) for r in release),
        release_completion_rate=_mean([r["release_success"] for r in release]),
        release_exposed_trials=sum(bool(r.get("release_exposed_support")) for r in release),
        release_physical_clearance_trials=sum(bool(r.get("release_physical_clearance_success")) for r in release),
        release_outcomes=dict(Counter(r.get("release_outcome", "unknown_old_contract") for r in release)),
        release_success_conditional_exposed_rate=_mean([r["release_success"] for r in release if r.get("release_exposed_support")]),
        release_time_conditional_mean=_mean([r["release_time"] for r in release]),
        scheduled_events=len(events), event_outcomes=dict(outcomes),
        recovery_conditional_mean=_mean([e["recovery_time"] for e in events]),
        boundary=bool(records[0].get("boundary")) if records else False,
    )


def paired_cluster_estimate(pairs, endpoint, bootstrap=5000, rng=None, cohort="all"):
    """Bootstrap complete seed clusters, never contact ticks or individual pulses."""
    rng = np.random.default_rng(2651901) if rng is None else rng
    observations = []
    candidate_values, baseline_values = [], []
    for candidate, baseline in pairs:
        if cohort == "window" and not (candidate["static_window"]["eligible"] and
                                        baseline["static_window"]["eligible"]):
            continue
        if cohort == "completed" and not (candidate["task_success"] and baseline["task_success"]):
            continue
        if cohort == "completed_window" and not (candidate["task_success"] and baseline["task_success"] and
                                                  candidate["static_window"]["eligible"] and baseline["static_window"]["eligible"]):
            continue
        if cohort == "sensitivity_window" and not (candidate.get("static_window_sensitivity", {}).get("eligible", False) and
                                                     baseline.get("static_window_sensitivity", {}).get("eligible", False)):
            continue
        if cohort == "release" and not (candidate["release_required"] and baseline["release_required"]):
            continue
        a, b = _metric(candidate, endpoint), _metric(baseline, endpoint)
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            continue
        observations.append((candidate["seed"], float(a) - float(b)))
        candidate_values.append(float(a))
        baseline_values.append(float(b))
    if not observations:
        return dict(endpoint=endpoint, cohort=cohort, pairs=0, clusters=0, candidate_mean=None,
                    baseline_mean=None, difference=None, ci_low=None, ci_high=None,
                    relative_percent=None, p=None)
    grouped = defaultdict(list)
    for seed, difference in observations:
        grouped[seed].append(difference)
    seeds = sorted(grouped)
    sums = np.array([np.sum(grouped[s]) for s in seeds])
    sizes = np.array([len(grouped[s]) for s in seeds])
    difference = float(np.sum(sums) / np.sum(sizes))
    index = rng.integers(0, len(seeds), (bootstrap, len(seeds)))
    replicate = np.sum(sums[index], axis=1) / np.sum(sizes[index], axis=1)
    lower, upper = np.quantile(replicate, [.025, .975])
    # Under paired exchangeability, swap method assignment for entire seed.
    # Exact enumeration for small seed counts, Monte Carlo otherwise.
    if len(seeds) <= 16:
        combinations = np.arange(2 ** len(seeds), dtype=np.uint64)[:, None]
        signs = 2 * ((combinations >> np.arange(len(seeds), dtype=np.uint64)) & 1).astype(float) - 1
        null = signs @ sums / np.sum(sizes)
        p = float(np.mean(np.abs(null) >= abs(difference) - 1e-12))
        p_method = "exact seed-cluster paired sign permutation"
    else:
        signs = rng.choice([-1., 1.], (50000, len(seeds)))
        null = signs @ sums / np.sum(sizes)
        p = float((np.sum(np.abs(null) >= abs(difference) - 1e-12) + 1) / (len(null) + 1))
        p_method = "Monte Carlo seed-cluster paired sign permutation"
    a_mean, b_mean = float(np.mean(candidate_values)), float(np.mean(baseline_values))
    return dict(endpoint=endpoint, cohort=cohort, pairs=len(observations), clusters=len(seeds),
                candidate_mean=a_mean, baseline_mean=b_mean, difference=difference,
                ci_low=float(lower), ci_high=float(upper),
                relative_percent=100 * difference / b_mean if abs(b_mean) > 1e-12 else None,
                p=p, p_method=p_method, bootstrap_replicates=bootstrap)


def holm_adjust(values):
    result = [None] * len(values)
    valid = [(i, p) for i, p in enumerate(values) if p is not None]
    ordered = sorted(valid, key=lambda value: value[1])
    previous = 0.
    for rank, (index, p) in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered) - rank) * p))
        result[index] = previous
    return result


def analyze_records(records, baseline="ratio", candidate="unified", bootstrap=5000, gate_margins=None):
    methods = sorted({r["method"] for r in records})
    if baseline not in methods:
        alternatives = [m for m in ("friction_ratio", "ratio") if m in methods]
        if not alternatives:
            raise ValueError(f"baseline {baseline!r} not found in {methods}")
        baseline = alternatives[0]
    if candidate not in methods:
        alternatives = [m for m in ("unified", "unified_coordinator") if m in methods]
        candidate = alternatives[0] if alternatives else candidate
    splits = sorted({r["split"] for r in records})
    method_summary, scenario_summary, comparisons = {}, {}, []
    rng = np.random.default_rng(2651901)
    gate_margins = DEFAULT_GATE_MARGINS if gate_margins is None else gate_margins
    endpoints = [
        ("restricted_unstable_time", "all", "primary", "s"),
        ("static.finger_tv_rate", "window", "primary", "N/s"),
        ("task_success", "all", "gate", "fraction"),
        ("safety.force_protection_pass", "all", "gate", "fraction"),
        ("release_success", "release", "gate", "fraction"),
        ("static_coverage", "all", "gate", "fraction"),
        ("static.mean_force", "window", "secondary", "N"),
        ("static.total_sd", "window", "secondary", "N"),
        ("static.finger_sd_mean", "window", "secondary", "N"),
        ("static.raw200_finger_tv_rate", "window", "bandwidth_diagnostic", "N/s"),
        ("static.substep_finger_tv_rate", "window", "bandwidth_diagnostic", "N/s"),
        ("first_stable_time", "all", "conditional", "s"),
        ("safety.total_excess_integral", "all", "gate", "N s"),
        ("safety.local_excess_integral", "all", "gate", "N s"),
        ("stable_hold_fraction", "all", "secondary", "fraction"),
        ("static.finger_tv_rate", "completed_window", "sensitivity", "N/s"),
        ("static.mean_force", "completed_window", "sensitivity", "N"),
        ("sensitivity.finger_tv_rate", "sensitivity_window", "sensitivity", "N/s"),
        ("sensitivity.mean_force", "sensitivity_window", "sensitivity", "N"),
    ]
    for split in splits:
        available = [r for r in records if r["split"] == split]
        method_summary[split] = {}
        scenario_summary[split] = {}
        for boundary in (False, True):
            subset = [r for r in available if bool(r.get("boundary", False)) == boundary]
            if not subset:
                continue
            domain = "boundary" if boundary else "main"
            method_summary[split][domain] = {}
            for method in methods:
                rows = [r for r in subset if r["method"] == method]
                if rows:
                    method_summary[split][domain][method] = summarize_method(rows)
            indexed = {(r["scenario"], r["seed"], r["method"]): r for r in subset}
            for method in methods:
                if method == baseline:
                    continue
                pairs = [(r, indexed[(r["scenario"], r["seed"], baseline)]) for r in subset
                         if r["method"] == method and (r["scenario"], r["seed"], baseline) in indexed]
                for endpoint, cohort, role, unit in endpoints:
                    estimate = paired_cluster_estimate(pairs, endpoint, bootstrap, rng, cohort)
                    estimate.update(split=split, domain=domain, method=method, baseline=baseline,
                                    role=role if method == candidate else "exploratory_" + role, unit=unit,
                                    planned_pairs=sum(r["method"] == baseline for r in subset),
                                    paired_window_denominator=sum(r["method"] == baseline for r in subset))
                    if endpoint in gate_margins:
                        estimate["engineering_margin"] = gate_margins[endpoint]["margin"]
                        estimate["gate_judgement"] = gate_judgement(estimate, gate_margins[endpoint])
                    comparisons.append(estimate)
        for scenario in sorted({r["scenario"] for r in available}):
            scenario_summary[split][scenario] = {}
            for method in methods:
                rows = [r for r in available if r["scenario"] == scenario and r["method"] == method]
                if rows:
                    scenario_summary[split][scenario][method] = summarize_method(rows)
    # Two frozen quality endpoints form one primary family in each parameter domain.
    # Gates have separate engineering interpretation, no invented noninferiority margins.
    for split in splits:
        primary = [r for r in comparisons if r["split"] == split and r["domain"] == "main"
                   and r["method"] == candidate and r["role"] == "primary"]
        for row, adjusted in zip(primary, holm_adjust([r["p"] for r in primary])):
            row["p_holm"] = adjusted
    coverage = {}
    for split in splits:
        rows = [r for r in records if r["split"] == split]
        expected = len({r["seed"] for r in rows}) * len({r["scenario"] for r in rows}) * len(methods)
        coverage[split] = dict(expected_cross_product=expected, observed=len(rows),
                               complete_cross_product=len(rows) == expected)
    force_contracts = {}
    for row in records:
        contract = row.get("force_quality_contract", {})
        definition = {k: contract.get(k) for k in ("filter_kind", "cutoff_hz", "kernel_span_seconds",
                                                   "evaluation_interval", "force_signal_kind")}
        key = json.dumps(definition, sort_keys=True)
        if key not in force_contracts:
            force_contracts[key] = dict(definition=definition, trials=0)
        force_contracts[key]["trials"] += 1
    return dict(baseline=baseline, candidate=candidate, trial_count=len(records),
                methods=methods, splits=splits, bootstrap_replicates=bootstrap,
                matrix_coverage=coverage,
                force_quality_definitions=list(force_contracts.values()),
                seed_unit="whole randomization seed across scenarios",
                primary_family="A_S and paired fixed-window finger TV per split; Holm two endpoints",
                sensitivity_family="joint whole-task completion + main eligible window; separate predeclared fixed [8.3,8.8] window; descriptive only",
                gate_inference="predeclared engineering margin checked using paired 95% seed-cluster bootstrap intervals",
                gate_margins=gate_margins,
                summaries=method_summary, scenarios=scenario_summary, comparisons=comparisons)


def _format(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def write_analysis(result, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                                       encoding="utf-8")
    columns = sorted({key for row in result["comparisons"] for key in row})
    with (out / "paired_comparisons.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result["comparisons"])
    rows = []
    for split, domains in result["summaries"].items():
        for domain, methods in domains.items():
            for method, value in methods.items():
                rows.append(dict(split=split, domain=domain, method=method,
                                 **{k: v for k, v in value.items() if not isinstance(v, dict)}))
    with (out / "methods.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 综合评价结果", "", "先检查任务、实际力保护、释放及窗口覆盖，再比较稳定速度、逐指力涨落与平均施力。所有单位与分母公开，未定义加权总分。", "",
             f"数据共 {result['trial_count']} 次试验。统计重采样单位为完整种子簇，主品质端点是首次稳定受限等待 A_S 与共同有效静态窗逐指 TV，分别在每个参数域作两端点 Holm 校正。A_S只衡量首次稳定达到，不是全任务失稳时长积分；首次稳定后失抓仍由任务失败门槛完整保留。",
             "", "主逐指TV是目标物实际接触力：完整物理子步全轨迹先经离线对称Hann窗sinc FIR（50 Hz截止、50 ms核跨度），再对齐共同200 Hz时钟，最后取预定静态窗。这是独立评价的带宽约定，不是传感器滤波或命令平滑。S、峰力和超限不滤波。SD、均力保留原始200 Hz定义，并同时保存原始200 Hz TV与完整子步TV作为带宽诊断。开发合成数据可显式fallback；正式要求完整真值和物理区间，缺失记录不能合格。",
             "", "预定工程门槛：完成、全子步力合格、释放和固定窗覆盖差异的95%区间下限至少−5个百分点；逐指超限积分之和、总力超限积分分别要求区间上限不超过+0.02 N·s。各项独立判断，不相加重复计量。差异不显著不表示非劣成立；少量种子簇使区间与检验分辨率有限。", ""]
    for split, domains in result["summaries"].items():
        for domain, methods in domains.items():
            lines.extend([f"## {split} / {domain}", "",
                          "| 方法 | 完成 | 力保护 | 首次稳定受限等待 A_S (s) | 首稳达成 | 静态窗覆盖 | TV (N/s) | 均力 (N) |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
            for method, s in methods.items():
                lines.append(f"| {METHOD_LABELS.get(method, method)} | {s['successes']}/{s['trials']} | {s['force_protection_passes']}/{s['trials']} | {_format(s['A_S'])} | {s['stable_attained']}/{s['trials']} | {s['static_eligible']}/{s['trials']} | {_format(s['finger_tv_eligible_mean'])} | {_format(s['mean_force_eligible_mean'])} |")
            lines.extend(["", "此表 TV 与均力是各方法有效窗描述值。方法差异的主比较使用双方共同有效队列，见下表，不能直接相减本表的不同队列均值。", ""])
            lines.append("记录验收（完整真值 / 完整物理力 / 全证据；全部预定任务分母）：")
            lines.extend(["", "| 方法 | 真值完整 | 物理力完整 | 全证据完整 |", "|---|---:|---:|---:|"])
            for method, s in methods.items():
                lines.append(f"| {METHOD_LABELS.get(method, method)} | {s['complete_truth_records']}/{s['trials']} | {s['complete_physical_force_records']}/{s['trials']} | {s['complete_evidence_records']}/{s['trials']} |")
            lines.append("")
            if any(s["release_trials"] for s in methods.values()):
                lines.extend(["释放以全部预定释放任务为分母。只有指令执行端点前已建立且仍有规定支持接触的对象才构成可执行释放暴露；指令前已掉落的零接触仅算物理clearance，不算成功释放。", "",
                              "| 方法 | 可执行释放完成 / 预定 | 暴露 / 预定 | 物理clearance / 预定 | 暴露后成功释放条件延迟(s) |", "|---|---:|---:|---:|---:|"])
                for method, s in methods.items():
                    if s["release_trials"]:
                        lines.append(f"| {METHOD_LABELS.get(method, method)} | {s['release_successes']}/{s['release_trials']} | {s['release_exposed_trials']}/{s['release_trials']} | {s['release_physical_clearance_trials']}/{s['release_trials']} | {_format(s['release_time_conditional_mean'])} |")
                lines.append("")
    lines.extend(["## 完整方案与强基线的配对差异", "", "差值=完整方案−强基线；A_S、TV、平均施力及超限积分越小越好，完成率和覆盖率越大越好。", "",
                  "| 参数域 | 指标 | 配对数 / 预定数 | 种子簇 | 差值 | 95%区间 | Holm p（仅主要品质） | 工程门槛 |", "|---|---|---:|---:|---:|---|---:|---|"])
    for row in result["comparisons"]:
        if row["method"] == result["candidate"] and row["domain"] == "main" and row["role"] != "sensitivity":
            lines.append(f"| {row['split']} | {row['endpoint']} ({row['unit']}) | {row['pairs']}/{row['planned_pairs']} | {row['clusters']} | {_format(row['difference'])} | [{_format(row['ci_low'])}, {_format(row['ci_high'])}] | {_format(row.get('p_holm'), 4)} | {row.get('gate_judgement', '—')} |")
    lines.extend(["", "恢复事件逐类报告 never_unstable、recovered_during_event、recovered_after_event、unrecovered、failed_before_recovery、not_exposed_failure、not_stable_before_event 和 preempted_by_release。未失稳不会作为零延迟恢复样本，用户释放中断恢复单列。所有恢复与条件首次稳定时间均不能覆盖完整任务失败。",
                  "", "能力边界场景单列，降级或失败不算原任务成功。指部限力保护优先使用所有已布设虚拟触觉区域的外界skin接触压力，不按目标物身份排除碰撞；S支持与持握TV使用目标物专属grasp接触。掌部及障碍碰撞负荷独立监测，未设材料损伤范围时不能称整只手的完整安全。试验定义和工程范围仍须结合模型验收，不等同于临床或实物安全保证。", ""])
    lines.extend(["## 预声明敏感性", "", "主比较保持固定8.2–8.9秒静态窗。以下两项只检查结论对队列/预定较短窗口的依赖，不替换主端点或挑最有利结果：双方全任务完成且主窗共同有效；固定8.3–8.8秒较短窗共同有效。均报告TV与平均力、全部预定分母及有效种子簇，不纳入主品质Holm家族。", "",
                  "| 参数域 | 敏感性队列 | 指标 | 有效/预定配对 | 种子簇 | 差值 | 95%区间 |", "|---|---|---|---:|---:|---:|---|"])
    for row in result["comparisons"]:
        if row["method"] == result["candidate"] and row["domain"] == "main" and row["role"] == "sensitivity":
            lines.append(f"| {row['split']} | {row['cohort']} | {row['endpoint']} ({row['unit']}) | {row['pairs']}/{row['planned_pairs']} | {row['clusters']} | {_format(row['difference'])} | [{_format(row['ci_low'])}, {_format(row['ci_high'])}] |")
    lines.extend(["## 实际力计量合同", ""])
    for contract in result["force_quality_definitions"]:
        lines.append(f"- {contract['trials']}次：{json.dumps(contract['definition'], ensure_ascii=False)}")
    (out / "综合评价结果.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--out")
    parser.add_argument("--baseline", default="ratio")
    parser.add_argument("--candidate", default="unified")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    result = analyze_records(load_records(args.source), args.baseline, args.candidate, args.bootstrap)
    output = args.out or str((Path(args.source) if Path(args.source).is_dir() else Path(args.source).parent) / "analysis")
    write_analysis(result, output)
    print(json.dumps(dict(trials=result["trial_count"], baseline=result["baseline"], candidate=result["candidate"], output=output), ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Read-only, post-hoc pulse recovery analysis of frozen formal trajectories.

Run --action plan before --action analyze. No simulator/controller is imported.
This proxy is distinct from force settling, force safety, and task completion.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import csv
import hashlib
import json

import numpy as np


METHODS = ("event", "ratio", "fusion", "coordinated",
           "coordinated_no_guard", "coordinated_no_redistribution")
SCENARIOS = ("three_pulses", "local_limit")
SUITES = {"test": "output/test_holdout_v2",
          "robustness": "output/robustness_holdout_v2"}
FOCALS = ("coordinated", "fusion")
DT = .005
PULSE_TICKS = ((1800, 1890, 2190), (2200, 2290, 2590), (2600, 2690, 3000))
STABLE_INTERVALS = 40
REQUIRED_POINTS = STABLE_INTERVALS + 1
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20261015
CORE_FILES = ("controllers.py", "physics.py", "scenarios.py",
              "run_experiments.py", "selected_gains.json")
PLAN_VERSION = "pulse_recovery_posthoc_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    require(bool(rows), "Cannot write empty records")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False)
                             if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def frozen_hashes(project):
    formal = read_json(project / "review/formal_plan.json")
    expected = {**formal["code_sha256"], **formal["supporting_sha256"]}
    actual = {name: sha256(project / name) for name in CORE_FILES}
    require(all(actual[name] == expected[name] for name in CORE_FILES),
            "Frozen core/source parameter hash differs from formal plan")
    return actual


def selected_inputs(project):
    """Validate identity/cardinality without computing any recovery endpoints."""
    selected, manifests, input_hashes = {}, {}, {}
    for suite, relative in SUITES.items():
        directory = project / relative
        manifest = read_json(directory / "manifest.json")
        require(manifest["finished"] and manifest["code_unchanged_during_run"],
                f"Incomplete/non-frozen suite: {suite}")
        require(tuple(manifest["methods"]) == METHODS, "Unexpected methods")
        definitions = {s["name"]: s for s in manifest["scenario_definitions"]}
        for scenario in SCENARIOS:
            s = definitions[scenario]
            require(s["event"] == "pulses" and s["release_at"] is None
                    and s["duration"] == 17., "Unexpected pulse task definition")
        rows = [r for r in read_json(directory / "trials.json")
                if r["scenario"] in SCENARIOS]
        keys = [(r["scenario"], r["method"], r["seed"]) for r in rows]
        wanted = {(s, m, seed) for s in SCENARIOS for m in METHODS
                  for seed in manifest["seeds"]}
        require(len(keys) == len(set(keys)) and set(keys) == wanted,
                f"Missing/duplicate pulse trials: {suite}")
        require(manifest["seeds"] == list(range(8001, 8021) if suite == "test"
                                         else range(9001, 9011)), "Unexpected seeds")
        selected[suite] = sorted(rows, key=lambda r: (r["scenario"], r["seed"], r["method"]))
        manifests[suite] = manifest
        for name in ("manifest.json", "trials.json"):
            input_hashes[f"{relative}/{name}"] = sha256(directory / name)
    require(sum(map(len, selected.values())) == 360, "Expected 360 pulse traces")
    return selected, manifests, input_hashes


def build_plan(project):
    selected, manifests, inputs = selected_inputs(project)
    return {
        "version": PLAN_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "事后探索；已知原任务结果后、计算恢复指标前固定定义",
        "no_simulation_or_tuning": True,
        "frozen_sources": frozen_hashes(project),
        "formal_plan_sha256": sha256(project / "review/formal_plan.json"),
        "analysis_script_sha256": sha256(__file__),
        "suites": SUITES, "methods": METHODS, "scenarios": SCENARIOS,
        "seeds": {s: m["seeds"] for s, m in manifests.items()},
        "input_manifest_hashes": inputs,
        "expected_traces": 360, "expected_scheduled_events": 1080,
        "pulses": [{"pulse": i + 1, "onset_s": a * DT,
                    "pulse_end_s": b * DT, "deadline_exclusive_s": c * DT}
                   for i, (a, b, c) in enumerate(PULSE_TICKS)],
        "raw_time_alignment": "trace.time 与 raw normal/slip_true/position_error_m 同刻；execute 更新后的 phase 不用于重建失抓",
        "exposure": "setup_ok=true 且脉冲 onset 严格早于 drop_time_s-0.05；开始前失败单列未暴露；脉冲开始后失抓仍属已暴露失败",
        "drop_boundary": "无释放情景下按原 drop_time_s 减 0.05 s 保守截断；不从位置或 phase 重建掉落；截断时刻端点不纳入",
        "criterion": {"normal_each_gt_N": .2, "minimum_contact_fingers": 2,
                      "slip_true_lt_mps": .005, "position_error_lt_m": .08,
                      "sustained_duration_s": .20, "control_sample_s": DT,
                      "required_consecutive_endpoints": REQUIRED_POINTS,
                      "window": "[onset+0.45,min(next_onset-0.05 or 15.0,drop_time-0.05))",
                      "latency": "首次合格连续 41 端点窗口起点减去 pulse_end；已稳定可为0；未恢复为 null+false，不能填0"},
        "descriptive": "所有6方法分 suite/scenario 汇总 scheduled/exposed/recovered/failed/not_exposed 及未暴露原因；恢复后仍可能在后续失抓",
        "paired": {"focals": FOCALS, "comparator": "ratio",
                   "strata": [*SCENARIOS, "both_pulse_scenarios"],
                   "recovery_rate": "只在双方均已暴露的同 seed/scenario/pulse 上比较，差=焦点方法- ratio；正值利于焦点方法",
                   "latency": "只在双方均暴露且均恢复事件上比较，差=焦点方法- ratio；负值更快；条件选择不能证明全任务优势",
                   "scheduled_attainment_diagnostic": "所有预定事件把未暴露计未达成，仅作综合代理诊断，不叫暴露后恢复率",
                   "bootstrap": {"repetitions": BOOTSTRAP_REPS,
                                 "unit": "同suite整组seed，包括全部所选场景和脉冲",
                                 "estimator": "重采样后差值总和/合格事件数总和；不把帧或脉冲当独立样本",
                                 "seed": BOOTSTRAP_SEED, "ci": "95% marginal percentile",
                                 "multiplicity": "探索性边际区间；不作验证性显著结论，不提供事后选择p值"}},
        "caveats": [
            "这是所选 5 mm/s、至少2指、200 ms 的工程代理；不等于实际力回到目标、姿态恢复、力安全或完整任务完成",
            "slip_true 是当前模型轴向/向下相对速度代理，非完整三维相对速度；离散41端点不能证明端点间持续稳定",
            "仅两类局部脉冲情景；不包含永久摩擦突降、释放或其他未选择场景",
            "jointly_exposed 及双方恢复延迟存在事后生存/恢复条件选择；报告完整分母与未暴露原因",
            "物理步尖峰、接触穿透和原安全/完成指标原样保留；恢复率不替代原正式结论",
            "后续独立评价才可验证；不能把落物后的零力视为恢复或平稳收益"],
    }


def first_sustained_start(mask, start_tick, end_tick):
    """Find earliest 0.20 s sampled interval wholly inside a half-open window."""
    count = 0
    for tick in range(start_tick, end_tick):
        count = count + 1 if mask[tick] else 0
        if count >= REQUIRED_POINTS:
            return tick - STABLE_INTERVALS
    return None


def event_record(stable_mask, trial, pulse_index, suite):
    onset, pulse_end, deadline = PULSE_TICKS[pulse_index]
    drop_time = trial["drop_time_s"]
    drop_tick = None
    if drop_time is not None:
        drop_tick = round(drop_time / DT)
        require(abs(drop_time - drop_tick * DT) < 1e-9, "Drop time not control-grid aligned")
    cutoff = len(stable_mask) if drop_tick is None else drop_tick - 10
    exposed = bool(trial["setup_ok"] and onset < cutoff)
    excluded = ("setup_failure_before_pulse" if not trial["setup_ok"]
                else "loss_before_pulse" if not exposed else None)
    end = min(deadline, cutoff)
    first = first_sustained_start(stable_mask, pulse_end, end) if exposed and end > pulse_end else None
    recovered = first is not None
    failure = None
    if exposed and not recovered:
        failure = ("lost_during_pulse" if cutoff <= pulse_end
                   else "lost_before_sustained_recovery" if cutoff < deadline
                   else "no_sustained_proxy_recovery_before_deadline")
    return {
        "suite": suite, "scenario": trial["scenario"], "method": trial["method"],
        "seed": trial["seed"], "pulse": pulse_index + 1,
        "onset_s": onset * DT, "pulse_end_s": pulse_end * DT,
        "nominal_deadline_exclusive_s": deadline * DT,
        "setup_ok": bool(trial["setup_ok"]), "drop_time_s": drop_time,
        "drop_cutoff_exclusive_s": cutoff * DT if drop_tick is not None else None,
        "exposed": exposed, "not_exposed_reason": excluded,
        "recovered": recovered, "failure_reason": failure,
        "recovery_start_s": first * DT if recovered else None,
        "recovery_confirm_s": (first + STABLE_INTERVALS) * DT if recovered else None,
        "latency_s": (first - pulse_end) * DT if recovered else None,
        "available_recovery_window_s": max(0, end - pulse_end) * DT if exposed else 0.,
        "pre_pulse_stable_proxy": bool(np.all(stable_mask[onset - REQUIRED_POINTS:onset])) if exposed else None,
        "stable_at_pulse_end": bool(stable_mask[pulse_end]) if exposed and cutoff > pulse_end else None,
        "recovered_then_lost_before_nominal_deadline": bool(recovered and cutoff < deadline),
        "episode_completed": bool(trial["completed"]),
        "episode_safe_completed": bool(trial["safe_completed"]),
        "episode_peak_limit_completed": bool(trial["peak_limit_completed"]),
    }


def read_trace(directory, trial):
    path = directory / trial["trace_file"]
    require(path.resolve().parent == (directory / "traces").resolve(), "Trace path outside formal traces")
    before = sha256(path)
    with np.load(path, allow_pickle=False) as data:
        trace = data["trace"]
        columns = data["columns"].tolist()
        metadata = json.loads(str(data["metadata"].item()))
        for key in ("scenario", "method", "seed", "setup_ok", "drop_time_s", "completed",
                    "safe_completed", "peak_limit_completed", "dropped", "nonfinite"):
            require(metadata[key] == trial[key], f"Metadata differs: {key}: {path.name}")
        require(trace.shape[0] == 3400, "Incomplete control trace")
        index = {name: i for i, name in enumerate(columns)}
        time = trace[:, index["time"]]
        require(np.max(np.abs(time - np.arange(3400) * DT)) < 1e-10, "Control clock differs")
        normal = trace[:, [index[f"normal_{i}"] for i in range(5)]]
        slip = trace[:, index["slip_true"]]
        position_error = trace[:, index["position_error_m"]]
        require(np.all(np.isfinite(normal)) and np.all(np.isfinite(slip))
                and np.all(np.isfinite(position_error)), "Nonfinite proxy inputs")
        require(not trial["nonfinite"], "Nonfinite episode")
        require(bool(trial["dropped"]) == (trial["drop_time_s"] is not None), "Drop fields inconsistent")
        stable = ((normal > .2).sum(axis=1) >= 2) & (slip < .005) & (position_error < .08)
    after = sha256(path)
    require(before == after, "Trace changed while reading")
    return stable, {"trace_file": str(path), "sha256_before": before,
                    "sha256_after": after, "unchanged": True,
                    "n_control_rows": 3400}


def summarize(events):
    rows = []
    for suite in SUITES:
        for scenario in (*SCENARIOS, "both_pulse_scenarios"):
            for method in METHODS:
                subset = [e for e in events if e["suite"] == suite and e["method"] == method
                          and (scenario == "both_pulse_scenarios" or e["scenario"] == scenario)]
                exposed = [e for e in subset if e["exposed"]]
                recovered = [e for e in exposed if e["recovered"]]
                latency = [e["latency_s"] for e in recovered]
                rows.append({"suite": suite, "scenario": scenario, "method": method,
                             "scheduled": len(subset), "exposed": len(exposed),
                             "recovered": len(recovered), "failed": len(exposed) - len(recovered),
                             "not_exposed": len(subset) - len(exposed),
                             "not_exposed_reasons": dict(Counter(e["not_exposed_reason"] for e in subset if not e["exposed"])),
                             "failure_reasons": dict(Counter(e["failure_reason"] for e in exposed if not e["recovered"])),
                             "exposed_recovery_rate": len(recovered) / len(exposed) if exposed else None,
                             "scheduled_proxy_attainment_rate": len(recovered) / len(subset),
                             "recovered_latency_mean_s": float(np.mean(latency)) if latency else None,
                             "recovered_latency_median_s": float(np.median(latency)) if latency else None,
                             "zero_latency_recovered": sum(x == 0. for x in latency),
                             "recovered_then_lost_before_nominal_deadline": sum(e["recovered_then_lost_before_nominal_deadline"] for e in recovered)})
    return rows


def clustered_effect(pairs, seeds, eligible, value, labels):
    sums = np.array([sum(value(a, b) for a, b in pairs if a["seed"] == seed and eligible(a, b))
                     for seed in seeds], dtype=float)
    counts = np.array([sum(eligible(a, b) for a, b in pairs if a["seed"] == seed)
                       for seed in seeds], dtype=int)
    total = int(counts.sum())
    seed_stats = [{"seed": seed, "eligible_events": int(n), "difference_sum": float(s)}
                  for seed, n, s in zip(seeds, counts, sums)]
    if not total:
        return {"eligible_events": 0, "contributing_seed_clusters": 0,
                "estimate": None, "ci95": None, "valid_bootstrap_draws": 0,
                "seed_cluster_statistics": seed_stats}
    digest = hashlib.sha256("|".join(labels).encode()).digest()
    rng = np.random.default_rng((BOOTSTRAP_SEED + int.from_bytes(digest[:8], "little")) % 2**64)
    draw = rng.integers(0, len(seeds), size=(BOOTSTRAP_REPS, len(seeds)))
    denominators = counts[draw].sum(axis=1)
    valid = denominators > 0
    boot = sums[draw].sum(axis=1)[valid] / denominators[valid]
    return {"eligible_events": total, "contributing_seed_clusters": int((counts > 0).sum()),
            "estimate": float(sums.sum() / total),
            "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
            "valid_bootstrap_draws": int(valid.sum()),
            "seed_cluster_statistics": seed_stats}


def paired_effects(events, plan):
    lookup = {(e["suite"], e["scenario"], e["seed"], e["pulse"], e["method"]): e for e in events}
    rows = []
    for suite in SUITES:
        for stratum in (*SCENARIOS, "both_pulse_scenarios"):
            selected = SCENARIOS if stratum == "both_pulse_scenarios" else (stratum,)
            for focal in FOCALS:
                pairs = [(lookup[(suite, s, seed, p, focal)], lookup[(suite, s, seed, p, "ratio")])
                         for s in selected for seed in plan["seeds"][suite] for p in (1, 2, 3)]
                jointly_exposed = lambda a, b: a["exposed"] and b["exposed"]
                both_recovered = lambda a, b: jointly_exposed(a, b) and a["recovered"] and b["recovered"]
                base = {"suite": suite, "scenario": stratum, "focal": focal,
                        "comparator": "ratio", "scheduled_pairs": len(pairs),
                        "jointly_exposed": sum(jointly_exposed(a, b) for a, b in pairs),
                        "focal_only_exposed": sum(a["exposed"] and not b["exposed"] for a, b in pairs),
                        "ratio_only_exposed": sum(b["exposed"] and not a["exposed"] for a, b in pairs),
                        "neither_exposed": sum(not a["exposed"] and not b["exposed"] for a, b in pairs),
                        "jointly_recovered": sum(both_recovered(a, b) for a, b in pairs),
                        "focal_only_recovered_when_jointly_exposed": sum(jointly_exposed(a, b) and a["recovered"] and not b["recovered"] for a, b in pairs),
                        "ratio_only_recovered_when_jointly_exposed": sum(jointly_exposed(a, b) and b["recovered"] and not a["recovered"] for a, b in pairs)}
                metrics = (
                    ("jointly_exposed_recovery_rate_difference", jointly_exposed,
                     lambda a, b: int(a["recovered"]) - int(b["recovered"])),
                    ("both_recovered_latency_difference_s", both_recovered,
                     lambda a, b: a["latency_s"] - b["latency_s"]),
                    ("scheduled_proxy_attainment_difference_diagnostic", lambda a, b: True,
                     lambda a, b: int(a["recovered"]) - int(b["recovered"])),
                )
                for name, eligible, value in metrics:
                    effect = clustered_effect(pairs, plan["seeds"][suite], eligible, value,
                                              (suite, stratum, focal, name))
                    rows.append({**base, "metric": name, **effect})
    return rows


def synthetic_checks():
    trial = {"scenario": "three_pulses", "method": "ratio", "seed": 8001,
             "setup_ok": True, "drop_time_s": None, "completed": True,
             "safe_completed": True, "peak_limit_completed": True}
    checks = []
    def check(name, predicate):
        require(bool(predicate), f"Synthetic recovery check failed: {name}")
        checks.append({"name": name, "passed": True})
    mask = np.ones(3400, dtype=bool)
    e = event_record(mask, trial, 0, "test")
    check("already_stable_has_true_zero_latency", e["recovered"] and e["latency_s"] == 0.)
    mask[:] = False
    mask[1890:1930] = True
    e = event_record(mask, trial, 0, "test")
    check("40_points_only_195ms_not_200ms", not e["recovered"] and e["latency_s"] is None)
    mask[1930] = True
    e = event_record(mask, trial, 0, "test")
    check("41_endpoints_span_200ms", e["recovered"] and e["recovery_confirm_s"] == 9.65)
    mask[:] = False
    mask[1900:1941] = True
    e = event_record(mask, trial, 0, "test")
    check("first_stable_window_start_not_confirmation", e["latency_s"] == .05 and e["recovery_start_s"] == 9.5)
    mask[:] = True
    e = event_record(mask, {**trial, "drop_time_s": 9.2}, 0, "test")
    check("loss_during_pulse_counts_exposed_failure", e["exposed"] and not e["recovered"] and e["failure_reason"] == "lost_during_pulse")
    e = event_record(mask, {**trial, "drop_time_s": 9.05}, 0, "test")
    check("loss_at_onset_not_exposed", not e["exposed"] and e["not_exposed_reason"] == "loss_before_pulse")
    e = event_record(mask, {**trial, "drop_time_s": 9.7}, 0, "test")
    check("cutoff_endpoint_excluded", not e["recovered"] and e["latency_s"] is None)
    e = event_record(mask, {**trial, "setup_ok": False}, 0, "test")
    check("setup_failure_not_exposed_or_zero_recovery", not e["exposed"] and not e["recovered"] and e["latency_s"] is None)
    mask[:] = False
    mask[2170:2211] = True
    e = event_record(mask, trial, 0, "test")
    check("next_pulse_deadline_disallows_crossing_window", not e["recovered"])
    normal = np.zeros((41, 5)); normal[:, :2] = .201
    slip = np.full(41, .00499); position = np.full(41, .07999)
    proxy = lambda n, s, p: ((n > .2).sum(axis=1) >= 2) & (s < .005) & (p < .08)
    check("two_contacts_qualify", np.all(proxy(normal, slip, position)))
    normal[:, 1] = .2
    check("zero_force_and_threshold_contact_do_not_recover", not np.any(proxy(normal, slip, position)))
    normal[:, 1] = .201
    check("strict_speed_and_position_thresholds", not np.any(proxy(normal, np.full(41, .005), position))
          and not np.any(proxy(normal, slip, np.full(41, .08))))
    pairs = [({"seed": 1, "x": 1}, {}), ({"seed": 1, "x": 3}, {}),
             ({"seed": 2, "x": 10}, {})]
    effect = clustered_effect(pairs, [1, 2], lambda a, b: True,
                              lambda a, b: a["x"], ("test", "synthetic"))
    check("cluster_estimator_weights_events_not_seed_means", abs(effect["estimate"] - 14 / 3) < 1e-12)
    return {"passed": True, "checks": checks, "count": len(checks)}


def analyze(project, output, plan_path):
    plan = read_json(plan_path)
    require(plan["version"] == PLAN_VERSION, "Unexpected recovery plan version")
    require(plan["analysis_script_sha256"] == sha256(__file__), "Analysis script changed since plan")
    require(plan["frozen_sources"] == frozen_hashes(project), "Frozen sources changed since plan")
    selected, manifests, input_hashes = selected_inputs(project)
    require(input_hashes == plan["input_manifest_hashes"], "Formal manifests changed since plan")
    checks = synthetic_checks()
    events, sources = [], []
    for suite, trials in selected.items():
        for trial in trials:
            mask, source = read_trace(project / SUITES[suite], trial)
            source.update(suite=suite, scenario=trial["scenario"], method=trial["method"], seed=trial["seed"])
            sources.append(source)
            events.extend(event_record(mask, trial, p, suite) for p in range(3))
        print(f"{suite}: {len(trials)} read-only traces / {len(trials) * 3} scheduled events", flush=True)
    require(len(events) == 1080 and len(sources) == 360, "Unexpected analysis coverage")
    for e in events:
        require((e["latency_s"] is not None) == e["recovered"], "Missing latency incorrectly imputed")
        require(not e["recovered"] or e["exposed"], "Excluded event marked recovered")
    summary, effects = summarize(events), paired_effects(events, plan)
    require(plan["frozen_sources"] == frozen_hashes(project), "Frozen sources changed during analysis")
    require(all(sha256(project / relative) == h for relative, h in input_hashes.items()), "Formal input changed during analysis")
    for source in sources:
        require(sha256(source["trace_file"]) == source["sha256_before"], "Formal trace changed during analysis")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "recovery_events.json", events)
    save_csv(output / "recovery_events.csv", events)
    save_json(output / "recovery_summary.json", summary)
    save_csv(output / "recovery_summary.csv", summary)
    save_json(output / "recovery_paired_effects.json", effects)
    save_csv(output / "recovery_paired_effects.csv", effects)
    save_json(output / "recovery_sources.json", sources)
    save_json(output / "recovery_checks.json", checks)
    audit = {"passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
             "status": "post-hoc exploratory; no simulation/tuning",
             "plan_file": str(plan_path), "plan_sha256": sha256(plan_path),
             "script_sha256": sha256(__file__), "frozen_sources": frozen_hashes(project),
             "formal_input_hashes_unchanged": True, "all_trace_hashes_unchanged": True,
             "traces": len(sources), "scheduled_events": len(events),
             "summary_groups": len(summary), "paired_effect_groups": len(effects),
             "synthetic_checks_passed": checks["count"],
             "metric_caveats": plan["caveats"]}
    save_json(output / "recovery_audit.json", audit)
    print(json.dumps({k: audit[k] for k in ("passed", "traces", "scheduled_events",
                                          "summary_groups", "paired_effect_groups")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "self-test", "analyze"), default="analyze")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "恢复结果")
    args = parser.parse_args()
    plan_path = Path(__file__).resolve().parent / "恢复计划.json"
    if args.action == "plan":
        require(not plan_path.exists(), "Recovery plan already saved; do not overwrite a pre-calculation plan")
        save_json(plan_path, build_plan(args.project))
        print(f"Plan saved before recovery calculation: {plan_path}")
    elif args.action == "self-test":
        checks = synthetic_checks()
        save_json(args.output / "recovery_checks.json", checks)
        print(json.dumps(checks))
    else:
        require(plan_path.is_file(), "Save fixed plan with --action plan before calculating endpoints")
        analyze(args.project, args.output, plan_path)


if __name__ == "__main__":
    main()

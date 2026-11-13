"""Write a Chinese second-round report only after evidence and audits are ready.

Defaults resolve against this script's directory. --test-audit and
--robustness-audit may override the usual <suite>/audit.json locations.
No simulation, retuning, significance selection, or first-round pooling occurs.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
import math

import numpy as np
import analyze_results as analysis


ROOT = Path(__file__).resolve().parent
METHODS = ("event", "ratio", "fusion", "coordinated",
           "coordinated_no_guard", "coordinated_no_redistribution")
BASELINES = ("event", "ratio")
COMPARATORS = (*BASELINES, "fusion")
ABLATIONS = ("coordinated_no_guard", "coordinated_no_redistribution")
SUITES = ("test", "robustness")
DOMAINS = ("main", "all", "boundary", "conflict")
CONFLICT = ("fragile_conflict", "local_limit", "release_slip", "collision")
LABELS = {"event": "事件规则", "ratio": "摩擦比基线", "fusion": "第一轮融合架构",
          "coordinated": "事件触发协调", "coordinated_no_guard": "协调去除力保护",
          "coordinated_no_redistribution": "协调去除接触重分配"}
SUITE_LABELS = {"test": "正式留出", "robustness": "扰动留出"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_evidence(path, sources):
    path = Path(path).resolve()
    require(path.is_file(), f"证据尚未就绪，拒绝生成报告：{path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"证据不是有效JSON：{path}: {exc}") from exc
    sources[str(path)] = sha256(path)
    return obj


def number(value, label):
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value), f"缺少有限数值：{label}")
    return float(value)


def same_hash(first, second):
    return isinstance(first, str) and isinstance(second, str) and first.lower() == second.lower()


def timestamp(value):
    require(isinstance(value, str), "缺少冻结运行时间")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def pct(count, n):
    return f"{count}/{n}（{100 * count / n:.1f}%）"


def fnum(value, digits=3):
    return f"{number(value, '报告数值'):.{digits}f}"


def pformat(value):
    if value is None:
        return "—（探索性）"
    value = number(value, "p值")
    require(0 <= value <= 1, "p值不在0–1范围")
    return f"{value:.4g}" if value >= .0001 else f"{value:.2e}"


def effect_text(row, digits=2):
    estimate = number(row.get("estimate"), "配对效应")
    ci = row.get("ci95")
    require(isinstance(ci, list) and len(ci) == 2, "缺少配对95%区间")
    low, high = (number(value, "配对区间") for value in ci)
    require(low <= high, "配对区间顺序错误")
    return f"{estimate:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def md_table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(cell, row)) + " |" for row in rows)])


def local_link(path, label):
    return f"[{label}](<{Path(path).resolve().as_posix()}>)"


def load_calibration(directory, stage, defaults, sources):
    directory = Path(directory).resolve()
    manifest = read_evidence(directory / "manifest.json", sources)
    scores = read_evidence(directory / "scores.json", sources)
    trials = read_evidence(directory / "trials.json", sources)
    selected = read_evidence(directory / "selected_gains.json", sources)
    require(manifest.get("stage") == stage and manifest.get("unchanged") is True
            and manifest.get("dynamics_unchanged") is True,
            f"校准未完成或未冻结：{directory}")
    require(manifest.get("code_sha256") == manifest.get("final_code_sha256"),
            f"校准代码运行中发生变化：{directory}")
    seeds, scenes, candidates = manifest.get("seeds"), manifest.get("scenes"), manifest.get("candidates")
    require(isinstance(seeds, list) and seeds and len(set(seeds)) == len(seeds)
            and set(seeds) <= set(range(1101, 1106)), "校准包含非开发种子")
    require(isinstance(scenes, list) and scenes and len(set(scenes)) == len(scenes), "校准场景不完整")
    expected_methods = {"event", "ratio", "fusion"} if stage == "baseline" else {"coordinated"}
    require(isinstance(candidates, dict) and set(candidates) == expected_methods
            and all(isinstance(items, list) and len(items) == 8 for items in candidates.values()),
            "本轮声明为三基线各8候选、协调8候选；校准证据不符")
    require(isinstance(scores, list) and len(scores) == sum(map(len, candidates.values()))
            and isinstance(trials, list) and trials, "校准分数或试验为空")
    expected_trials = len(seeds) * len(scenes) * sum(map(len, candidates.values()))
    require(len(trials) == manifest.get("expected_trials") == manifest.get("completed_trials") == expected_trials,
            "校准试验数量不完整")
    score_map, budget, best = {}, {}, {}
    for method, items in candidates.items():
        for index, gains in enumerate(items):
            key = method, index
            matches = [row for row in scores if (row.get("method"), row.get("index")) == key]
            require(len(matches) == 1 and matches[0].get("gains") == gains, f"校准分数候选不一致：{key}")
            score = matches[0]
            # Historical stages may have different defaults for parameters
            # unused by that controller. Their declared candidate overrides
            # identify the recorded cohort; the final suite loader separately
            # checks its full effective gains against the frozen source.
            rows = [row for row in trials if row.get("method") == method
                    and isinstance(row.get("gains"), dict)
                    and all(row["gains"].get(field) == value for field, value in gains.items())]
            expected_keys = {(name, seed) for name in scenes for seed in seeds}
            require(len(rows) == len(expected_keys)
                    and {(row.get("scenario"), row.get("seed")) for row in rows} == expected_keys,
                    f"校准候选缺少场景种子：{key}")
            safe = sum(row["safe_completed"] for row in rows)
            loss = sum(row["dropped"] or not row["setup_ok"] for row in rows)
            force = sum(number(row["total_force_excess_Ns"], str(key)) for row in rows)
            budget_excess = sum(number(row["total_budget_excess_Ns"], str(key)) for row in rows)
            shifts = [row["max_slip_mm"] for row in rows if row["max_slip_mm"] is not None]
            shift = float(np.median(shifts)) if shifts else 1e9
            require(score.get("trials") == len(rows) and score.get("safe") == safe
                    and score.get("drops") == loss, f"校准计数不能由试验重建：{key}")
            for name, value in (("force_excess_Ns", force), ("budget_excess_Ns", budget_excess),
                                ("excess_Ns", force + budget_excess), ("median_shift_mm", shift)):
                require(np.isclose(number(score.get(name), str(key)), value, atol=1e-9, rtol=1e-9),
                        f"校准分数不能由试验重建：{key}/{name}")
            score_map[key] = score
        winning = min(range(8), key=lambda index: (-score_map[(method, index)]["safe"],
                      score_map[(method, index)]["drops"], score_map[(method, index)]["excess_Ns"],
                      score_map[(method, index)]["median_shift_mm"], index))
        require(selected.get(method) == items[winning], f"选定参数不符合存档排序：{method}")
        best[method] = score_map[(method, winning)]
        budget[method] = {"candidates": len(items), "scenes": len(scenes), "seeds": len(seeds),
                          "trials": len(items) * len(scenes) * len(seeds)}
    require(same_hash(sha256(directory / "selected_gains.json"), manifest.get("selected_gains_sha256")),
            "校准选定参数哈希不符")
    return {"directory": directory, "manifest": manifest, "scores": scores,
            "trials": trials, "selected": selected, "budget": budget, "best": best}


def load_development(directory, sources, final_suite=None):
    directory = Path(directory).resolve()
    manifest = read_evidence(directory / "manifest.json", sources)
    trials = read_evidence(directory / "trials.json", sources)
    audit = read_evidence(directory / "audit.json", sources)
    require(manifest.get("suite") == "development" and manifest.get("code_unchanged_during_run") is True,
            f"开发验证不是完整冻结开发集合：{directory}")
    require(manifest.get("code_sha256") == manifest.get("final_code_sha256"), "开发验证代码运行中变化")
    seeds, scenes, methods = manifest.get("seeds"), manifest.get("scenarios"), manifest.get("methods")
    require(isinstance(seeds, list) and seeds and len(set(seeds)) == len(seeds)
            and set(seeds) <= set(range(1101, 1106)), "开发验证种子泄漏")
    require(isinstance(scenes, list) and len(scenes) == 16 and len(set(scenes)) == 16
            and isinstance(methods, list) and len(methods) == 6 and set(methods) == set(METHODS),
            "开发验证须完整16情景6方法")
    expected = {(scene, method, seed) for scene in scenes for method in methods for seed in seeds}
    require(isinstance(trials, list) and len(trials) == len(expected)
            == manifest.get("expected_trials") == manifest.get("completed_trials")
            and {(row.get("scenario"), row.get("method"), row.get("seed")) for row in trials} == expected,
            "开发验证缺少组合或有重复")
    require(audit.get("passed") is True and audit.get("results", {}).get("trials") == len(trials)
            and audit["results"].get("independently_recomputed_trace_integrals") == len(trials)
            and audit.get("repeatability", {}).get("identical_physical_and_control_results") is True,
            "开发逐轨迹审计/复跑尚未通过")
    if final_suite is not None:
        require(set(seeds) == {1105}, "最终开发内一致性检验应使用预定1105")
        for field in ("code_sha256", "gains", "gains_sha256", "scenario_definitions",
                      "physics_timestep", "control_timestep", "source_model_sha256"):
            require(manifest.get(field) == final_suite.manifest.get(field), f"最终开发验证与正式配置不同：{field}")
    definitions = {item["name"]: item for item in manifest["scenario_definitions"]}
    summaries = {}
    for domain in ("main", "all"):
        domain_names = scenes if domain == "all" else [name for name in scenes if not definitions[name]["boundary"]]
        for method in METHODS:
            rows = [row for row in trials if row["method"] == method and row["scenario"] in domain_names]
            counts = {"safe_completed": sum(row["safe_completed"] for row in rows),
                      "peak_limit_completed": sum(row["peak_limit_completed"] for row in rows),
                      "completed": sum(row["completed"] for row in rows),
                      "dropped": sum(row["dropped"] for row in rows),
                      "setup_failure": sum(not row["setup_ok"] for row in rows)}
            summaries[(domain, method)] = {"n": len(rows), "counts": counts}
    return {"directory": directory, "manifest": manifest, "trials": trials,
            "audit": audit, "summaries": summaries}


def load_probe(directory, sources):
    directory = Path(directory).resolve()
    manifest = read_evidence(directory / "manifest.json", sources)
    trials = read_evidence(directory / "trials.json", sources)
    summary = read_evidence(directory / "summary.json", sources)
    diagnostic = read_evidence(directory / "diagnostic_summary.json", sources)
    require(manifest.get("research_only") is True, "摩擦探索必须标为研究而非正式比较")
    immutable = manifest.get("immutable", {})
    methods, scenes, seeds = immutable.get("methods"), immutable.get("scenarios"), immutable.get("seeds")
    require(isinstance(methods, list) and isinstance(scenes, list) and isinstance(seeds, list)
            and set(seeds) <= set(range(1101, 1106)), "摩擦探索配置/种子不符")
    expected = {(scene, method, seed) for scene in scenes for method in methods for seed in seeds}
    require(isinstance(trials, list) and len(trials) == immutable.get("trials_expected") == len(expected)
            and {(row.get("scenario"), row.get("method"), row.get("seed")) for row in trials} == expected,
            "摩擦探索试验不完整")
    for filename in ("manifest.json", "trials.json", "summary.json"):
        require(same_hash(diagnostic.get("evidence_sha256", {}).get(filename), sha256(directory / filename)),
                f"摩擦探索诊断来源哈希不符：{filename}")
    for method in methods:
        rows = [row for row in trials if row["method"] == method]
        stored = summary.get("methods", {}).get(method, {})
        require(stored.get("episodes") == len(rows)
                and stored.get("safe_completed") == sum(row["safe_completed"] for row in rows)
                and stored.get("completed") == sum(row["completed"] for row in rows)
                and stored.get("dropped") == sum(row["dropped"] for row in rows),
                f"摩擦探索汇总不能重建：{method}")
        require(np.isclose(number(stored.get("mean_total_force_excess_Ns"), method),
                           np.mean([number(row["total_force_excess_Ns"], method) for row in rows]),
                           atol=1e-9, rtol=1e-9), f"摩擦探索力暴露汇总不符：{method}")
        for scene in scenes:
            chosen = [row for row in rows if row["scenario"] == scene]
            stored_scene = summary.get("scenarios", {}).get(scene, {}).get(method, {})
            require(stored_scene.get("safe_completed") == sum(row["safe_completed"] for row in chosen)
                    and stored_scene.get("dropped") == sum(row["dropped"] for row in chosen),
                    "摩擦探索场景汇总不符")
    return {"directory": directory, "manifest": manifest, "trials": trials,
            "summary": summary, "diagnostic": diagnostic}


def load_all(args):
    sources = {}
    summary = read_evidence(args.analysis / "summary.json", sources)
    paired = read_evidence(args.analysis / "paired_effects.json", sources)
    meta = summary.get("metadata")
    require(isinstance(meta, dict) and meta.get("analysis_version") == 2
            and paired.get("metadata") == meta and set(meta.get("methods", [])) == set(METHODS),
            "只接受完整第二轮分析；旧结果或不一致元数据不能生成本报告")
    require(same_hash(meta.get("analysis_script_sha256"), sha256(ROOT / "analyze_results.py")),
            "分析脚本与结果记录不一致")
    sources[str(ROOT / "analyze_results.py")] = sha256(ROOT / "analyze_results.py")
    require(isinstance(meta.get("inputs"), dict) and all(name in meta["inputs"] for name in SUITES),
            "分析缺少正式/扰动数据来源")
    suites, audits = {}, {}
    for name in SUITES:
        directory = Path(meta["inputs"][name]["directory"])
        # The analysis loader verifies full snapshots, pairing, metrics and
        # the exact predeclared seed split without running any simulation.
        suites[name] = analysis.load_suite(directory, name)
        for filename, key in (("manifest.json", "manifest_sha256"), ("trials.json", "trials_sha256")):
            read_evidence(directory / filename, sources)
            require(same_hash(sources[str((directory / filename).resolve())], meta["inputs"][name].get(key)),
                    f"分析来源与当前正式数据不一致：{name}/{filename}")
        audit_override = args.test_audit if name == "test" else args.robustness_audit
        audit_path = audit_override or directory / "audit.json"
        audit = read_evidence(audit_path, sources)
        require(audit.get("passed") is True and isinstance(audit.get("results"), dict),
                f"正式审计未通过：{audit_path}")
        required_flags = ("paired_parameters_verified", "exact_cartesian_pairing_verified",
                          "frozen_code_and_source_verified", "uninterrupted_force_safety_verified",
                          "retained_orientation_denominator_verified", "palm_and_collision_metrics_verified",
                          "release_outputs_verified")
        require(all(audit["results"].get(flag) is True for flag in required_flags)
                and audit["results"].get("trials") == len(suites[name].trials)
                and audit["results"].get("independently_recomputed_trace_integrals") == len(suites[name].trials)
                and audit.get("repeatability", {}).get("identical_physical_and_control_results") is True,
                f"正式逐轨迹审计或复跑证据不完整：{name}")
        require(audit["results"].get("seed_split_verified") == name, f"审计集合身份不符：{name}")
        audits[name] = {"path": Path(audit_path).resolve(), "value": audit}
    analysis.validate_suite_pair(suites["test"], suites["robustness"])
    rows = {}
    for row in summary.get("domains", []):
        key = row.get("suite"), row.get("domain"), row.get("method")
        require(key not in rows, f"重复总体汇总行：{key}")
        rows[key] = row
    required_rows = {(suite, domain, method) for suite in SUITES for domain in DOMAINS for method in METHODS}
    require(required_rows <= set(rows), "缺少主要、全16、边界或预定冲突域汇总")
    for name, suite in suites.items():
        require(set(suite.domain_names("conflict")) == set(CONFLICT), "冲突子域与预定4情景不符")
        for domain in DOMAINS:
            names = suite.domain_names(domain)
            for method in METHODS:
                row = rows[(name, domain, method)]
                trials = [trial for trial in suite.trials if trial["method"] == method and trial["scenario"] in names]
                regenerated = analysis.summarize_method(trials, suite.definitions)
                require(row.get("n") == len(trials) and row.get("counts") == regenerated["counts"],
                        f"总体计数不能由原始试验重建：{name}/{domain}/{method}")
                require(row.get("rates_pct") == regenerated["rates_pct"], "完成率与实际计数不符")
                for metric in ("physics_peak_force_N", "physics_peak_penetration_mm",
                               "physics_force_excess_Ns", "physics_budget_excess_Ns"):
                    require(row.get("continuous", {}).get(metric) == regenerated["continuous"][metric],
                            f"物理诊断汇总不一致：{name}/{domain}/{method}/{metric}")
    scenario_rows = {}
    for row in summary.get("scenarios", []):
        key = row.get("suite"), row.get("scenario"), row.get("method")
        require(key not in scenario_rows, f"重复场景汇总：{key}")
        scenario_rows[key] = row
    for name, suite in suites.items():
        expected = {(name, scene, method) for scene in suite.manifest["scenarios"] for method in METHODS}
        require(expected <= set(scenario_rows) and len(suite.manifest["scenarios"]) == 16,
                f"缺少完整16情景矩阵：{name}")
        for key in expected:
            _, scene, method = key
            trial_rows = [suite.indexed[(scene, method, seed)] for seed in suite.seeds]
            regenerated = analysis.summarize_method(trial_rows, suite.definitions,
                                                     raw_sphere_orientation=scene == "sphere")
            require(scenario_rows[key].get("counts") == regenerated["counts"]
                    and scenario_rows[key].get("n") == len(trial_rows), f"场景计数不一致：{key}")
    effects = {}
    for row in paired.get("effects", []):
        key = row.get("suite"), row.get("domain"), row.get("comparator"), row.get("metric"), row.get("condition")
        require(key not in effects and row.get("method") == "coordinated", f"重复或非第二轮配对效应：{key}")
        effects[key] = row
    for name in SUITES:
        for domain in DOMAINS:
            for comparator in (*COMPARATORS, *ABLATIONS):
                for metric in ("safe_completed", "peak_limit_completed", "completed", "drop", "setup_failure",
                               "total_force_excess_Ns", "total_budget_excess_Ns"):
                    key = name, domain, comparator, metric, "all"
                    require(key in effects, f"缺少配对效应：{key}")
                    effect_text(effects[key])
                    require(effects[key].get("n_pairs") == rows[(name, domain, "coordinated")]["n"],
                            f"配对主/诊断指标缺失样本：{key}")
                effect = effects[(name, domain, comparator, "safe_completed", "all")]
                raw_difference = (rows[(name, domain, "coordinated")]["rates_pct"]["safe_completed"]
                                  - rows[(name, domain, comparator)]["rates_pct"]["safe_completed"])
                require(np.isclose(effect["estimate"], raw_difference, atol=1e-9, rtol=0),
                        f"配对效应与计数不符：{name}/{domain}/{comparator}")
                if domain == "main" and comparator in BASELINES:
                    pvalue = number(effect.get("p_value_holm"), "主比较Holm p")
                    require(0 <= pvalue <= 1 and effect.get("holm_family")
                            == f"{name}.main.safe_completed.two_baselines", "主Holm家族不符")
                else:
                    require(effect.get("p_value_holm") is None, "探索性比较意外被当作主Holm比较")
    baseline = load_calibration(args.baseline_calibration, "baseline", suites["test"].defaults, sources)
    coordinated = load_calibration(args.coordinated_calibration, "coordinated", suites["test"].defaults, sources)
    reused_subset = load_calibration(args.coordinated_reused_subset, "coordinated", suites["test"].defaults, sources)
    baseline_history = [load_calibration(path, "baseline", suites["test"].defaults, sources)
                        for path in args.baseline_history]
    history = [load_calibration(path, "coordinated", suites["test"].defaults, sources)
               for path in args.coordination_history]
    require(coordinated["manifest"].get("baseline_gains") == baseline["selected"],
            "协调调参没有继承存档基线参数")
    require(reused_subset["manifest"].get("baseline_gains") == baseline["selected"]
            and reused_subset["manifest"].get("candidates") == coordinated["manifest"].get("candidates")
            and reused_subset["manifest"].get("scenes") == coordinated["manifest"].get("scenes")
            and set(reused_subset["manifest"]["seeds"]) < set(coordinated["manifest"]["seeds"]),
            "最终协调扩展不符合相同8候选/场景的冻结子集复用")
    def calibration_trial_key(row):
        return row["method"], row["scenario"], row["seed"], json.dumps(row["gains"], sort_keys=True)
    merged_index = {calibration_trial_key(row): row for row in coordinated["trials"]}
    reused_index = {calibration_trial_key(row): row for row in reused_subset["trials"]}
    require(set(reused_index) <= set(merged_index)
            and all(merged_index[key] == row for key, row in reused_index.items()),
            "最终协调结果没有原样复用已冻结子集；不能去重预算")
    if "resume_completed_trials" in coordinated["manifest"]:
        require(len(reused_index) <= coordinated["manifest"]["resume_completed_trials"] <= len(merged_index),
                "协调复用条数与resume记录不符")
    reused_subset["reused_trials"] = len(reused_index)
    reused_subset["new_trials"] = len(merged_index) - len(reused_index)
    ratio_core = baseline["selected"]["ratio"]
    for candidate in coordinated["manifest"]["candidates"]["coordinated"]:
        require(all(candidate.get(key) == value for key, value in ratio_core.items()),
                "协调候选改变了选定摩擦比核心")
    for method in (*BASELINES, "fusion"):
        require(coordinated["selected"].get(method) == baseline["selected"][method], "协调阶段改变基线参数")
    for method in ABLATIONS:
        require(coordinated["selected"].get(method) == coordinated["selected"]["coordinated"], "消融被独立调参")
    require(suites["test"].manifest["gains"] == coordinated["selected"], "正式参数不是开发选定参数")
    equivalence = read_evidence(args.review / "baseline_kernel_equivalence.json", sources)
    kernel_names = {"Common", "EventBaseline", "RatioBaseline", "AdaptiveConeBaseline", "Fusion"}
    require(equivalence.get("passed") is True and equivalence.get("classes_unchanged") is True,
            "复用基线校准需要通过AST等价核查")
    require(set(equivalence.get("before_class_sha256", {})) == kernel_names
            and equivalence.get("before_class_sha256") == equivalence.get("after_class_sha256"),
            "基线类AST等价证据不完整")
    before_path = args.review / "development_runtime_v3/controllers.py"
    after_path = ROOT / "controllers.py"
    for path, side in ((before_path, "before"), (after_path, "after")):
        require(path.is_file(), f"缺少AST核查实际源文件：{path}")
        sources[str(path.resolve())] = sha256(path)
        require(same_hash(sources[str(path.resolve())], equivalence.get(f"{side}_sha256")),
                f"基线AST证据的完整源文件哈希不符：{side}")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                   for node in tree.body if isinstance(node, ast.ClassDef) and node.name in kernel_names}
        require(classes == equivalence[f"{side}_class_sha256"], f"实际基线类AST不能重建证据：{side}")
    require(same_hash(equivalence["after_sha256"], suites["test"].manifest["code_sha256"]["controllers.py"]),
            "AST核查的最终控制源不是正式冻结版本")
    # Final calibrations must use the exact final runtime. Earlier iterations
    # are provenance/budget only and may use older clock or scenario versions.
    for stage in (baseline, reused_subset, coordinated):
        for code, expected_hash in suites["test"].manifest["code_sha256"].items():
            actual_hash = stage["manifest"]["code_sha256"].get(code)
            require(same_hash(actual_hash, expected_hash),
                    f"最终校准与正式完整动力学/控制代码不同：{stage['manifest']['stage']}/{code}")
    for stage in (*baseline_history, baseline, *history, reused_subset, coordinated):
        require(timestamp(stage["manifest"]["ended"]) <= min(timestamp(suite.manifest["created"]) for suite in suites.values()),
                "校准结束时间晚于正式运行开始")
    for stage in history:
        origins = [item for item in (*baseline_history, baseline)
                   if stage["manifest"].get("baseline_gains") == item["selected"]]
        require(len(origins) == 1, "历史协调阶段的基线来源不能唯一确认")
        stage["inherited_baseline"] = origins[0]
    review = {}
    for filename in ("physics_freeze_summary.json", "physics_model_v2.json", "preflight.json", "controller_checks.json",
                     "integer_clock_validation.json", "selected_timestep_validation.json"):
        review[filename] = read_evidence(args.review / filename, sources)
    physics = review["physics_freeze_summary.json"]
    clock = review["integer_clock_validation.json"]
    final_physics = suites["test"].manifest["code_sha256"]["physics.py"]
    require(same_hash(clock.get("canonical_source_sha256"), final_physics)
            and clock.get("models_and_solvers_unchanged") is True
            and clock.get("all_event_ticks_and_durations_match") is True
            and clock.get("raw_physics_clock_not_reset") is True,
            "共同整数时钟修订尚未在最终物理源上核验")
    require(same_hash(physics.get("frozen_physics_sha256"), final_physics)
            or same_hash(physics.get("frozen_physics_sha256"), clock.get("legacy_source_sha256")),
            "接触模型核查不能追溯到当前模型或仅改时钟的前一模型")
    timestep = review["selected_timestep_validation.json"]
    timestep_manifest = timestep.get("manifest", {})
    require(timestep.get("complete") is True and timestep.get("code_and_gains_unchanged") is True
            and timestep.get("errors") == [], "新的选定控制器步长敏感性检查未完整完成")
    for field in ("code_hashes_before", "code_hashes_after"):
        require(timestep_manifest.get(field) == suites["test"].manifest["code_sha256"],
                "步长敏感性检查还是旧浮点调度/旧控制版本")
    for field in ("gains_sha256_before", "gains_sha256_after"):
        require(same_hash(timestep_manifest.get(field), suites["test"].manifest["gains_sha256"]),
                "步长敏感性检查不是最终选定参数")
    require(timestep_manifest.get("seed") in range(1101, 1106)
            and set(timestep_manifest.get("methods", [])) == {"ratio", "coordinated"},
            "步长检查没有使用预定开发控制器对照")
    require(review["preflight.json"].get("passed") is True and review["controller_checks.json"].get("status") == "passed",
            "预检或控制机制核查未通过")
    require(same_hash(review["controller_checks.json"].get("sources", {}).get("controllers.py"),
                      suites["test"].manifest["code_sha256"]["controllers.py"]),
            "控制机制核查不是正式控制版本")
    require(same_hash(review["physics_model_v2.json"].get("source_model_sha256"),
                      suites["test"].manifest["source_model_sha256"]), "模型说明引用的源模型不符")
    extra_audits = []
    for path in args.extra_audit:
        value = read_evidence(path, sources)
        require(value.get("passed") is True or value.get("status") == "passed", f"附加审计未通过：{path}")
        extra_audits.append(Path(path).resolve())
    development_history = [load_development(path, sources) for path in args.development_history]
    development = load_development(args.development, sources, suites["test"])
    probe = load_probe(args.friction_probe, sources)
    research_path = ROOT / "research_round2.md"
    require(research_path.is_file() and research_path.stat().st_size > 0, "缺少本轮原论文研究笔记")
    sources[str(research_path)] = sha256(research_path)
    research_text = research_path.read_text(encoding="utf-8")
    for url in ("https://arxiv.org/html/2411.07833", "https://arxiv.org/html/1903.09816",
                "https://arxiv.org/html/2603.07040",
                "https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters",
                "https://mujoco.readthedocs.io/en/3.3.4/computation/"):
        require(url in research_text, f"本轮研究笔记缺少报告引用的原始来源：{url}")
    plan_path = args.review / "formal_plan.json"
    plan = read_evidence(plan_path, sources) if plan_path.is_file() else None
    return {"meta": meta, "suites": suites, "rows": rows, "scenario_rows": scenario_rows,
            "effects": effects, "baseline": baseline, "coordinated": coordinated,
            "baseline_history": baseline_history, "history": history, "equivalence": equivalence,
            "reused_subset": reused_subset,
            "development_history": development_history, "development": development, "probe": probe,
            "review": review, "audits": audits, "extra_audits": extra_audits,
            "sources": sources, "plan": plan}


def comparative_conclusion(row, label, secondary=False):
    estimate, pvalue = row["estimate"], row["p_value_holm"]
    if pvalue < .05 and estimate > 0:
        return f"相对{label}的合格完成率提高，{'次要验证' if secondary else '主比较'}的Holm检验支持这一差异"
    if pvalue < .05 and estimate < 0:
        return f"相对{label}的合格完成率降低，{'次要验证' if secondary else '主比较'}的Holm检验支持退步"
    return f"相对{label}的差异尚未获得{'次要验证' if secondary else '主比较'}Holm检验支持"


def ablation_conclusion(row):
    low, high = row["ci95"]
    if low > 0:
        return "未校正区间支持本冻结配置中的正向净效应；仍属探索性"
    if high < 0:
        return "删除模块后的完成率更高；本配置不支持该模块的净收益"
    return "区间含零，未能确定该模块的净收益"


def build_report(data, args):
    rows, effects, suites = data["rows"], data["effects"], data["suites"]
    get = lambda suite, domain, method: rows[(suite, domain, method)]
    eff = lambda suite, domain, comparator, metric="safe_completed", condition="all": effects[(suite, domain, comparator, metric, condition)]
    test_co = get("test", "main", "coordinated")
    rob_co = get("robustness", "main", "coordinated")
    test_ratio = eff("test", "main", "ratio")
    test_event = eff("test", "main", "event")
    wins = [method for method in BASELINES if eff("test", "main", method)["estimate"] > 0
            and eff("test", "main", method)["p_value_holm"] < .05]
    losses = [method for method in BASELINES if eff("test", "main", method)["estimate"] < 0
              and eff("test", "main", method)["p_value_holm"] < .05]
    if len(wins) == 2:
        lead = "本轮事件触发协调在预定15情景主域，对事件规则与摩擦比两种基线均得到经Holm校正支持的合格完成率优势。"
    elif wins:
        lead = f"本轮事件触发协调对{LABELS[wins[0]]}获得主比较支持的合格完成率优势，但尚未建立对全部强基线的总体优势。"
    elif losses:
        lead = f"本轮事件触发协调相对{LABELS[losses[0]]}出现经主比较支持的退步，不能宣称当前改进整体有效。"
    else:
        lead = "本轮事件触发协调尚未建立超过两种论文迁移基线的总体优势；当前应把结果解释为机制修订后的配对检验，而非已经证明全面领先。"
    parts = ["# 第二轮结果与改进路线", "",
             f"{lead} 正式主域协调合格完成为{pct(test_co['counts']['safe_completed'], test_co['n'])}；"
             f"相对摩擦比的配对差为{effect_text(test_ratio)}个百分点，Holm p={pformat(test_ratio['p_value_holm'])}。"
             f"扰动主域为{pct(rob_co['counts']['safe_completed'], rob_co['n'])}。",
             "",
             f"30 ms主指标允许短时高峰。更严格的逐物理子步不超限完成为：正式"
             f"{pct(test_co['counts']['peak_limit_completed'], test_co['n'])}，扰动"
             f"{pct(rob_co['counts']['peak_limit_completed'], rob_co['n'])}。"
             f"全16情景中，协调方法的子步最大指力分别为"
             f"{fnum(get('test', 'all', 'coordinated')['continuous']['physics_peak_force_N']['maximum'], 2)} / "
             f"{fnum(get('robustness', 'all', 'coordinated')['continuous']['physics_peak_force_N']['maximum'], 2)} N，"
             f"最大穿透为{fnum(get('test', 'all', 'coordinated')['continuous']['physics_peak_penetration_mm']['maximum'], 3)} / "
             f"{fnum(get('robustness', 'all', 'coordinated')['continuous']['physics_peak_penetration_mm']['maximum'], 3)} mm。"
             "因此合格完成率、瞬时峰值和接触模型可信度必须分别判断；这些数值不是真实材料损伤的验证。", ""]
    brief = []
    for method in ("event", "ratio", "fusion", "coordinated"):
        a, b = get("test", "main", method), get("robustness", "main", method)
        brief.append((LABELS[method], pct(a["counts"]["safe_completed"], a["n"]),
                      pct(b["counts"]["safe_completed"], b["n"]),
                      pct(a["counts"]["peak_limit_completed"], a["n"]),
                      pct(b["counts"]["peak_limit_completed"], b["n"])))
    parts += [md_table(("方法", "正式：30 ms合格", "扰动：30 ms合格", "正式：严格峰值合格", "扰动：严格峰值合格"), brief), "",
              "上述简表只含15个非边界情景。第一轮融合架构也在本轮公共模型上重新整定，不是把旧模型的第一轮百分比直接搬来对比。", "",
              "## 本轮改了什么，为什么这样改", "",
              "第一轮诊断暴露了三个控制问题：摩擦下降后的实际滑移可能伴随力比下降，原融合的上升力比门控会漏判；"
              "固定逐指摩擦比目标与较强积分会在不滑时持续增力；按总目标力守恒配力可能把需求分给已经失去接触的手指。"
              "最终第二轮方案保留低增益摩擦比外环：即使发生滑移，只要没有压力/观测冲突，仍使用基线输出；"
              "局部压力或总预算超限时采用温和比例卸载，监督也覆盖抓握建立与抬升。"
              "保护压力采用独立的三次有效原始观测中值，拒绝孤立峰值污染慢速力记忆；持续过力仍触发保护。"
              "只有在监督已经削去请求且持续滑移时，才把部分被削需求补偿给有效且有余量的接触；"
              "抬升阶段也允许受限的接触补偿。"
              "总请求不超过基线原请求，不再因滑移另加一份总力。补偿受速率与幅值限制，卸载后的执行目标回馈抗饱和；"
              "失真或过时通道不继续增力。", ""]
    checks = data["review"]["controller_checks.json"].get("checks", [])
    identity = [item for item in checks if item.get("check") == "no_conflict_baseline_identity"]
    if identity:
        parts += [f"合成触觉机制核查记录了{len(identity)}组无冲突同一性检查，共"
                  f"{sum(item['ticks'] for item in identity)}个控制步，最大输出差"
                  f"{max(item['max_absolute_difference'] for item in identity):.3g} N。"
                  "它验证实现关系，不能代替真实动力学上的安全或完成率证据。", ""]
    model = data["review"]["physics_model_v2.json"]
    parts += ["公共模型也经过修订：用简化指垫碰撞代理处理旧接触网格的深度重叠，清除初始手物穿透，统一接触层，"
              "并限制升降执行器的目标变化速率。所有方法使用同一修订；没有裁剪真实力值，也没有给控制器质量、真实摩擦、未来扰动或评价位姿。"
              f"模型记录的接触时间常数映射为`{model.get('contact_solref_map')}`。简化碰撞指垫尚未经过真实硬件标定。", "",
              "随后发现浮点累计时钟会使部分扰动边界偏移一个5 ms控制步。最终全部方法使用整数控制步生成事件时刻，"
              "保留MuJoCo自身积分时间且不改变碰撞几何、求解器或原载荷安排。新的整数时钟核查与选定控制器步长复测另行保存；"
              "旧步长比较含调度差异，不能用作最终版本的纯数值收敛证明。", "",
              "本轮的依据包括名义控制最小偏离的安全监督、接触负载反馈与配力平滑；这些已有文献先例。"
              f"完整核对见{local_link(ROOT / 'research_round2.md', '第二轮研究笔记')}。"
              "现有逐指标量观测不足以完整实现原论文的三维接触力/力矩平衡或鲁棒CBF形式保证。", "",
              "## 相同模型、开发预算与冻结", ""]
    baseline, coordinated = data["baseline"], data["coordinated"]
    budgets = {}
    for stage in (*data["baseline_history"], baseline):
        for method, values in stage["budget"].items():
            budgets[f"{method}:{stage['directory'].name}"] = values
    for stage in (*data["history"], coordinated):
        for method, values in stage["budget"].items():
            budgets[f"{method}:{stage['directory'].name}"] = values
    parts += [md_table(("调参对象", "候选", "开发情景", "开发种子", "本阶段试验"),
                      [(LABELS[method.split(':')[0]] + (f"（{method.split(':', 1)[1]}）" if ':' in method else ''),
                        values["candidates"], values["scenes"], values["seeds"], values["trials"])
                       for method, values in budgets.items()]), "",
              f"最终三基线各搜索8个核心候选，共{sum(item['trials'] for item in baseline['budget'].values())}次参数评估；"
              f"最终协调阶段8个候选、{sum(item['trials'] for item in coordinated['budget'].values())}次。"
              f"其中{data['reused_subset']['reused_trials']}条来自"
              f"{data['reused_subset']['directory'].name}的冻结复用子集，仅新增执行"
              f"{data['reused_subset']['new_trials']}次；没有增加候选设置。"
              f"历史基线阶段另有{sum(sum(item['trials'] for item in stage['budget'].values()) for stage in data['baseline_history'])}次，"
              f"历史协调阶段累计{sum(sum(item['candidates'] for item in stage['budget'].values()) for stage in data['history'])}"
              f"个阶段内设置、{sum(sum(item['trials'] for item in stage['budget'].values()) for stage in data['history'])}次。"
              f"全部存档参数评估合计{sum(item['trials'] for item in budgets.values())}次。"
              "报告入口逐条确认了复用子集与最终记录完全一致，因此没有把该子集再计一次。"
              "此预算为唯一存档参数评价条目，开发检查、数值复核及可能的恢复重试不充当额外独立样本。"
              "各协调阶段继承已经选定的摩擦比核心，仅搜索协调参数。"
              "协调有额外迭代且受益于前一阶段摩擦比核心搜索，因此不能宣称四方法总计算预算完全相等；参数维数与搜索密度也不同，"
              "不保证任何方法全局最优。消融直接使用完整协调的选定参数，不独立调参。", "",
              f"最终两阶段存档开发种子为{baseline['manifest']['seeds']}与{coordinated['manifest']['seeds']}，"
              f"正式种子为{suites['test'].seeds[0]}–{suites['test'].seeds[-1]}，扰动为"
              f"{suites['robustness'].seeds[0]}–{suites['robustness'].seeds[-1]}。"
              "最终协调选择覆盖15个非边界情景和1101–1105五个开发种子；1104/1105已经参加参数选择，"
              "不能再称独立开发验证。最终基线核心仍由1101–1103三种子搜索选定。"
              "报告入口验证了最终开发选定参数被正式运行原样继承、校准结束早于正式开始、"
              "最终两校准与正式集合使用逐字相同的控制/动力学/场景/运行器源码。"
              "存档校准中没有8001/9001新留出种子；只有正式8001–8020与扰动9001–9010提供独立性能证据。"
              "第一轮3001–3020 / 6501–6510仅用于诊断，不进入本轮独立结论。", "",
              "候选排序先比较完整任务合格完成，再比较失抓/建立失败，随后用逐指与预算超限积分之和作开发平局惩罚，最后比较保留位移。"
              "两种积分重叠，该和只是开发排序，不是一个独立物理冲量；正式结果分别报告。", ""]
    if data["history"]:
        history_rows = []
        for stage in data["history"]:
            baseline_best = stage["inherited_baseline"]["best"]["ratio"]
            best = stage["best"]["coordinated"]
            history_rows.append((stage["directory"].name, f"{best['safe']}/{best['trials']}", best["drops"],
                                 f"{baseline_best['safe']}/{baseline_best['trials']}", baseline_best["drops"]))
        parts += ["### 保留的开发失败与最终设计调整", "",
                  md_table(("历史协调阶段", "最佳候选30 ms合格", "掉落/建立失败", "继承摩擦比候选合格", "摩擦比掉落/建立失败"), history_rows), "",
                  "最初协调版本在滑移时给总力额外追加，同时较激进的绝对卸载会削弱原本足够的抓握。"
                  "v2取消了额外滑移总力，但完整形状验证又暴露抬升/建立失败：慢速力记忆被短峰污染，保护持续过度卸力。"
                  "这些失败结果与源码保留。最终版本改用独立三次有效原始压力中值、持续压力比例保护，并在抬升时允许受限接触补偿。"
                  "它与整数时钟修订均只依据开发诊断，正式新种子没有参与选择。"
                  "开发失败不是正式优势证据；是否有效以上述新留出结果为准。", "",
                  "最终前/后控制源中的Common、事件、摩擦比、逐指自适应及旧融合类AST逐项哈希一致，"
                  "报告生成器实读development_runtime_v3与最终源文件重建证据。最终参数仍在整数时钟模型上重新校准。"
                  "旧阶段允许不同物理/场景/运行器版本，只用于失败历史与开发预算，不混入正式总体。", ""]
    development_stages = (*data["development_history"], data["development"])
    parts += ["### 全场景开发诊断与一致性检验另计", "",
              f"选定参数后的完整开发验证合计{sum(len(stage['trials']) for stage in development_stages)}次，"
              "与候选参数评估分开记账。早期480次用于暴露形状/建立遗漏，随后1104/1105的192次与步长检查"
              "促发将这两个开发种子纳入协调选择；这些均属于开发证据。"
              "最终1105的96次只核查选定参数下的路径/轨迹/复跑一致性，仍是开发内检查。", ""]
    development_rows = []
    for stage in development_stages:
        for method in ("ratio", "coordinated"):
            values = stage["summaries"][("main", method)]
            counts = values["counts"]
            development_rows.append((stage["directory"].name, len(stage["trials"]), LABELS[method],
                                     pct(counts["safe_completed"], values["n"]), counts["completed"],
                                     counts["dropped"], counts["setup_failure"]))
    parts += [md_table(("开发验证阶段", "全方法试验数", "方法", "主域30 ms合格", "完成", "掉落", "建立失败"), development_rows), "",
              "这些开发结果不参加主Holm检验，也不增加正式留出样本数；旧开发审计证明其旧版本记录一致，"
              "不能证明旧调度下的数值就是最终版本性能。", ""]
    manifest = suites["test"].manifest
    total_trials = sum(len(suite.trials) for suite in suites.values())
    parts += [f"正式运行：16情景×{len(suites['test'].seeds)}种子×6方法={len(suites['test'].trials)}次；"
              f"扰动：16×{len(suites['robustness'].seeds)}×6={len(suites['robustness'].trials)}次，总计{total_trials}次。"
              f"物理步长{manifest['physics_timestep'] * 1000:g} ms，控制周期{manifest['control_timestep'] * 1000:g} ms，"
              f"MuJoCo {manifest.get('mujoco', '未记录')}。"
              "每个种子跨情景共享参数结构，统计独立单位是种子簇，不能把情景试验数当作独立人体样本。", "",
              "## 完成、掉落与建立失败", "",
              "主指标safe_completed：任务完成且任一指超过上限0.25 N或五指合力超过预算0.5 N，没有持续30 ms；包括建立抓握。"
              "严格指标peak_limit_completed是探索性指标：任务完成且每个物理子步及同步端点不超指力上限或总预算，"
              "只容许1e-6 N数值误差，没有30 ms宽限。两者都不代表掌部压力、软物变形或损伤风险已经验证。", ""]
    for suite_name in SUITES:
        tab = []
        for method in METHODS:
            row = get(suite_name, "main", method)
            counts, n = row["counts"], row["n"]
            tab.append((LABELS[method], pct(counts["safe_completed"], n),
                        pct(counts["peak_limit_completed"], n), counts["completed"],
                        counts["drop"], counts["setup_failure"], counts["completed_but_not_safe"]))
        parts += [f"### {SUITE_LABELS[suite_name]}：15情景主域", "",
                  md_table(("方法", "30 ms合格", "严格峰值合格", "完成", "掉落", "建立失败", "完成但30 ms不合格"), tab), ""]
    tab = []
    for suite_name in SUITES:
        for method in METHODS:
            row = get(suite_name, "all", method)
            tab.append((SUITE_LABELS[suite_name], LABELS[method], row["n"],
                        row["counts"]["safe_completed"], row["counts"]["peak_limit_completed"],
                        row["counts"]["completed"], row["counts"]["drop"], row["counts"]["setup_failure"]))
    parts += ["### 全16情景（含预定极端边界）", "",
              md_table(("集合", "方法", "次数", "30 ms合格", "严格峰值合格", "完成", "掉落", "建立失败"), tab), "",
              "掉落只在建立抓握后计数；建立失败另列。两项是互斥的失抓类型，不能只用掉落少来解释成功。"
              "释放评价是接触清除且在规定时间内完成，不等于物体被安全放回支撑面。", "",
              "## 配对效应与证据强度", "",
              "差值为协调减对照，单位为百分点；95%区间使用10000次种子簇配对重采样，区间未做多重比较校正。"
              "事件与摩擦比两项safe_completed主比较在正式与扰动集合分别作Holm校正；旧融合及其他指标属于探索性。"
              "双侧方法标签交换检验依赖零假设下种子内算法标签可交换的假设。区间不含零不能替代Holm p值。", ""]
    tab = []
    for suite_name in SUITES:
        for comparator in COMPARATORS:
            row = eff(suite_name, "main", comparator)
            tab.append((SUITE_LABELS[suite_name], LABELS[comparator], effect_text(row), pformat(row["p_value_holm"]),
                        "主比较" if suite_name == "test" and comparator in BASELINES
                        else "次要验证" if comparator in BASELINES else "探索性旧版本比较"))
    parts += [md_table(("集合", "对照", "合格完成差及95%区间", "Holm p", "性质"), tab), ""]
    for suite_name in SUITES:
        parts += [f"{SUITE_LABELS[suite_name]}：" + "；".join(comparative_conclusion(eff(suite_name, "main", method),
                   LABELS[method], suite_name == "robustness") for method in BASELINES) + "。", ""]
    tab = []
    for suite_name in SUITES:
        for comparator in COMPARATORS:
            tab.append((SUITE_LABELS[suite_name], LABELS[comparator],
                        effect_text(eff(suite_name, "main", comparator, "total_force_excess_Ns"), 4),
                        effect_text(eff(suite_name, "main", comparator, "total_budget_excess_Ns"), 4)))
    parts += ["### 全程超限暴露的配对差（探索性）", "",
              md_table(("集合", "对照", "逐指超限差及区间 Ns", "预算超限差及区间 Ns"), tab), "",
              "力积分涵盖全程且不筛选成功试验。失去接触/掉落可降低后续力，不能把低积分单独解释成保护更好。"
              "位移只在建立抓握的样本上有值；两算法都建立抓握或都完成任务的配对分析存在结果筛选偏差，不能替代总体完成率。", "",
              "### 预定4情景冲突子域（探索性）", "",
              "该域包含防滑与限力、局部受限配力、拉动中释放、碰撞与持握；在新留出运行前已预定。"
              "它用于检验方案针对任务竞争的表现，不能替代15情景主域，也不能据结果删除负向情景或称确认性显著。", ""]
    tab = []
    for suite_name in SUITES:
        for comparator in COMPARATORS:
            a, b = get(suite_name, "conflict", "coordinated"), get(suite_name, "conflict", comparator)
            tab.append((SUITE_LABELS[suite_name], LABELS[comparator],
                        pct(a["counts"]["safe_completed"], a["n"]),
                        pct(b["counts"]["safe_completed"], b["n"]),
                        effect_text(eff(suite_name, "conflict", comparator))))
    parts += [md_table(("集合", "对照", "协调合格", "对照合格", "差及未校正95%区间 pp"), tab), "",
              "### 两个消融是否支持机制", ""]
    tab = []
    for suite_name in SUITES:
        for comparator in ABLATIONS:
            row = eff(suite_name, "main", comparator)
            tab.append((SUITE_LABELS[suite_name], LABELS[comparator], effect_text(row),
                        effect_text(eff(suite_name, "main", comparator, "peak_limit_completed")),
                        ablation_conclusion(row)))
    parts += [md_table(("集合", "删除模块的对照", "30 ms合格差 pp", "严格峰值合格差 pp", "本配置证据"), tab), "",
              "差值仍为完整协调减消融。区间是未校正的探索性区间；消融没有重新调参，结果只检验本冻结参数下的净作用，"
              "不证明模块普遍有效或无效。", "",
              "## 完整16情景结果", "",
              "以下表格保留组合扰动和极低摩擦边界。每格是safe_completed的成功数/总数，判据仍为30 ms代理。", ""]
    definitions = suites["test"].manifest["scenario_definitions"]
    for suite_name in SUITES:
        tab = []
        for definition in definitions:
            name = definition["name"]
            label = definition["title"] + ("（边界）" if definition["boundary"] else "")
            tab.append((label, *[f"{data['scenario_rows'][(suite_name, name, method)]['counts']['safe_completed']}/"
                                f"{data['scenario_rows'][(suite_name, name, method)]['n']}" for method in METHODS]))
        parts += [f"### {SUITE_LABELS[suite_name]}", "", md_table(("情景", *[LABELS[method] for method in METHODS]), tab), ""]
    failure_rows = []
    for suite_name in SUITES:
        for definition in definitions:
            row = data["scenario_rows"][(suite_name, definition["name"], "coordinated")]
            failures = row["n"] - row["counts"]["safe_completed"]
            if failures:
                counts = row["counts"]
                failure_rows.append((SUITE_LABELS[suite_name], definition["title"], failures,
                                     counts["setup_failure"], counts["drop"], counts["completed_but_not_safe"],
                                     counts["other_failure_after_setup"]))
    parts += ["### 协调方法仍失败的情景", "",
              md_table(("集合", "情景", "未达30 ms合格", "建立失败", "掉落", "完成但限力不合格", "建立后其他未完成"), failure_rows)
              if failure_rows else "本轮协调在全部情景均达到30 ms代理合格；仍须检查严格峰值与模型边界。", "",
              "## 子步峰值、穿透和数值局限", ""]
    tab = []
    for suite_name in SUITES:
        for method in METHODS:
            metrics = get(suite_name, "all", method)["continuous"]
            tab.append((SUITE_LABELS[suite_name], LABELS[method],
                        fnum(metrics["physics_peak_force_N"]["median"], 2),
                        fnum(metrics["physics_peak_force_N"]["maximum"], 2),
                        fnum(metrics["physics_peak_penetration_mm"]["median"], 3),
                        fnum(metrics["physics_peak_penetration_mm"]["maximum"], 3),
                        fnum(metrics["physics_force_excess_Ns"]["mean"], 4),
                        fnum(metrics["physics_budget_excess_Ns"]["mean"], 4)))
    parts += [md_table(("集合", "方法", "子步力峰中位 N", "最大 N", "穿透峰中位 mm", "最大 mm", "子步逐指超限均值 Ns", "子步预算超限均值 Ns"), tab), "",
              "上表包含所有16情景和所有失败样本，没有裁剪尖峰或删除异常高值。物理子步记录与5 ms控制采样的峰值/积分范围不同，"
              "不能用后者替代前者。力值是碰撞与执行器模型的输出，零数值警告不等于接触力已经物理标定。", ""]
    freeze = data["review"]["physics_freeze_summary.json"]
    parts += [f"接触模型开发诊断记录{freeze.get('n_checks')}个检查，建立与完成分别为"
              f"{freeze.get('setup_success')} / {freeze.get('completed')}。核心诊断穿透范围"
              f"{freeze.get('core_penetration_mm_range')} mm，形状检查最大穿透"
              f"{freeze.get('shape_penetration_mm_max')} mm，诊断最大力{freeze.get('maximum_force_N')} N。"
              "这些记录用于追溯碰撞几何与深穿透修订；旧版本还使用浮点事件调度，"
              "其步长标签一致记录不作为最终版本的收敛依据。", ""]
    limitation_translations = {
        "All diagnostic seeds; not fresh statistical test.": "核查使用诊断种子，不是新的统计留出检验。",
        "Collision forces exceed 200 N despite 30 ms safe proxy passing.": "碰撞诊断存在超过200 N的力峰，同时30 ms代理仍可能合格。",
        "Cylinder transient peaks and penetration remain larger than capsule/box/sphere.": "柱体的瞬态峰值和穿透仍大于部分其他形状。",
        "Finer timestep changes continuous exposure and can still change labels for other cases.": "更细步长会改变连续暴露，其他情形的分类仍可能改变。",
    }
    for limitation in freeze.get("limitations", []):
        parts += ["- " + limitation_translations.get(limitation, limitation)]
    step = data["review"]["selected_timestep_validation.json"]
    step_manifest = step["manifest"]
    step_values = [(field, value) for field, value in step.get("max_absolute_numeric_differences", {}).items()]
    parts += ["", "### 最终整数时钟版本的步长复核", "",
              f"新的复核使用开发种子{step_manifest['seed']}、{step_manifest['methods']}、"
              f"物理步长{[value * 1000 for value in step_manifest['timesteps']]} ms，完成"
              f"{len(step.get('results', []))}/{step_manifest.get('expected_trials')}次；"
              f"{len(step.get('label_disagreements', []))}/{step_manifest.get('expected_pairs')}对存在至少一项离散标签差异。"
              f"记录的全面数值收敛认证为{step.get('numerical_convergence_certified')}。"
              "无标签差异也不等于所有连续力、穿透和位移已经收敛。", "",
              md_table(("复核指标（原字段单位）", "所检对子最大绝对差"),
                       [(field, "缺失" if value is None else f"{number(value, field):.6g}") for field, value in step_values]), "",
              str(step.get("conclusion", "本子集检查不属于正式性能或统计优势证据。")), "",
              "## 进一步摩擦控制探索的负证据", ""]
    probe = data["probe"]
    methods = probe["manifest"]["immutable"]["methods"]
    probe_rows = []
    for method in methods:
        stored = probe["summary"]["methods"][method]
        combined = probe["summary"]["scenarios"].get("contact_robustness", {}).get(method, {})
        count = sum(row["method"] == method and row["scenario"] == "contact_robustness" for row in probe["trials"])
        probe_rows.append((method, stored["episodes"], stored["safe_completed"], stored["completed"],
                           stored["dropped"], f"{combined.get('safe_completed')}/{count}",
                           fnum(stored["mean_total_force_excess_Ns"], 4)))
    parts += [f"另有{len(probe['trials'])}次开发专用摩擦探索（不属于正式留出或上述参数评估），"
              "比较继承控制与三个摩擦锥/摩擦估计原型。该探索保存的是早期时钟/控制版本，不能与最终正式率混合。", "",
              md_table(("探索方法", "次数", "30 ms合格", "完成", "掉落", "组合情景合格", "全程逐指超限均值 Ns"), probe_rows), ""]
    variants = [method for method in methods if method.startswith("cone_")]
    reference = probe["summary"]["methods"]["coordinated"]
    no_completion_gain = all(probe["summary"]["methods"][method]["completed"] <= reference["completed"] for method in variants)
    higher_exposure = [method for method in variants if probe["summary"]["methods"][method]["mean_total_force_excess_Ns"]
                       > reference["mean_total_force_excess_Ns"] + 1e-12]
    parts += [("这组探索没有原型提高相对协调控制的任务完成。" if no_completion_gain
               else "这组探索有原型改变了任务完成，但只属于开发证据。")
              + (f"{', '.join(higher_exposure)}增加了平均逐指超限暴露。" if higher_exposure else "")
              + "因此不能将增加压力或增加摩擦估计模块直接当作解决方案。", ""]
    contact_losses = []
    for record in probe["diagnostic"].get("combined_contact_loss", []):
        checkpoints = record.get("checkpoints", [])
        if checkpoints:
            first = checkpoints[0]
            last = min(checkpoints, key=lambda item: item["observed_contacts_above_.25N"])
            contact_losses.append((record["seed"], first["time"], first["observed_contacts_above_.25N"],
                                   last["time"], last["observed_contacts_above_.25N"]))
    if contact_losses:
        parts += ["组合情景的接触检查记录如下。力比估计依赖有效支持接触，快速接触丢失会先破坏估计条件；"
                  "这指向接触建立/维持与执行器响应问题，不能单凭当前控制器全部掉落就称物理不可行。", "",
                  md_table(("诊断种子", "前一记录时刻 s", "有效接触指", "较少接触记录时刻 s", "有效接触指"),
                           [(seed, f"{start:.3f}", initial, f"{end:.3f}", final)
                            for seed, start, initial, end, final in contact_losses]), ""]
    parts += ["## 后续改进路线", "",
              "1. 先处理实际接触峰值与执行器响应：逐指压入/卸载标定、限速/限扭矩与延迟测试，区分接触几何、运动命令和外力碰撞造成的峰值。"
              "碰撞、释放和持握发生冲突时，需要减速、退让或安全放置；单靠增握力无法保证兼顾。",
              "2. 补足接触位置/方向与局部力响应辨识，再检验接触力/力矩平衡分配；把失去接触与接触重建明确建模，避免用目标指力总和代替真实支持能力。",
              "3. 在摩擦、刚度、延迟具有可验证误差界后，再考虑鲁棒安全滤波或扰动观测器。当前工程监督不能直接宣称CBF形式安全保证。",
              "4. 扩展到随机扰动方向/时刻、偏心负载与更完整的局部接触滑移；先声明新的参数范围和统计比较，再用新的留出数据验证。",
              "5. 在真实触觉、软材料及假肢执行器上标定后再讨论损伤保护、意图融合或用户负担。仿真任务完成不证明人体或临床收益。", "",
              "若本轮主比较没有优势，下一步应针对以上失败类型开发；不能反复在当前留出集上调参后继续称它为独立证明。"
              "若主比较有优势，也仅适用于本轮模型、阈值和参数范围，严格峰值与步长敏感性仍需单独解决。", "",
              "## 原论文依据与迁移边界", "",
              "- [Kim等：Robust Adaptive Safe Robotic Grasping with Tactile Sensing](https://arxiv.org/html/2411.07833)：名义控制的最小偏离安全监督、接触力范围与摩擦锥；保证依赖动力学和误差假设。",
              "- [Shaw-Cortez等：Control Barrier Functions for Mechanical Systems](https://arxiv.org/html/1903.09816)：数字采样约束、无冲突时保留原控制、接触滚离与关节可行性。",
              "- [Ke等：TacDexGrasp](https://arxiv.org/html/2603.07040)：触觉接触负载、力分配和平滑；目标力可行不等于实际力可行。其三维观测与雅可比控制没有在本项目完整复现。",
              "- [MuJoCo官方Modeling](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)：软接触、穿透及位置伺服；数值稳定建议不代替力标定。",
              "- [MuJoCo官方Computation](https://mujoco.readthedocs.io/en/3.3.4/computation/)：接触距离与积分步长原则。版本化文档用于定义核查，最终运行版本以上述manifest为准。", "",
              "摩擦比和事件基线是原论文思想在共同五指仿真上的迁移，包含理想化触觉生成、不同执行器和工程限力/释放接口。"
              "不能把本轮数值称为原论文硬件、人体或材料实验的精确复现，也不能据此声称多任务融合世界首次。", "",
              "## 图、回放与独立审计", ""]
    artifact_paths = [(ROOT / "output/final_figures/completion_main.png", "主域合格完成图"),
                      (ROOT / "output/final_figures/paired_effect_main.png", "配对效应森林图"),
                      (ROOT / "output/final_figures/scenario_completion_test.png", "正式16情景矩阵"),
                      (ROOT / "output/final_figures/scenario_completion_robustness.png", "扰动16情景矩阵"),
                      (ROOT / "output/final_figures/physics_extremes.png", "全部试验子步极值分布")]
    linked = [local_link(path, label) for path, label in artifact_paths if path.is_file()]
    if linked:
        parts += ["图：" + "、".join(linked) + "。", ""]
    replay_dir = ROOT / "output/final_replay"
    if replay_dir.is_dir():
        replay_files = sorted(replay_dir.glob("*.gif"))
        if replay_files:
            parts += ["回放：" + "、".join(local_link(path, path.stem) for path in replay_files) + "。"
                      "预定种子8001的回放只解释机制，不作为总体优势证明。", ""]
    audit_table = []
    for suite_name in SUITES:
        audit = data["audits"][suite_name]
        values = audit["value"]
        audit_table.append((SUITE_LABELS[suite_name], values["results"]["trials"],
                            values["results"]["independently_recomputed_trace_integrals"],
                            "通过", local_link(audit["path"], "审计文件")))
    parts += [md_table(("集合", "试验记录", "独立轨迹重算", "代表试验复跑", "证据"), audit_table), "",
              "生成报告前读取了两份正式manifest/trials、汇总和配对效应、最终与历史完整校准分数/试验、全场景开发验证及其审计、"
              "公共接触模型核查、整数时钟核查、最终选定控制器步长复核、摩擦探索及其诊断、控制机制核查、基础预检及两份正式逐轨迹审计。"
              "审计通过只说明记录与实现一致；物理标定与真实损伤仍在范围之外。", ""]
    if data["extra_audits"]:
        parts += ["附加审计：" + "、".join(local_link(path, path.stem) for path in data["extra_audits"]) + "。", ""]
    parts += ["## 来源与复查", "",
              f"报告生成时间：{datetime.now(timezone.utc).isoformat()}。本报告独立于第一轮输出，未混合旧/新模型成功率。", "",
              md_table(("读取的本地证据", "SHA256"),
                       [(local_link(Path(path), Path(path).name), value) for path, value in data["sources"].items()]), ""]
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "output/final_analysis")
    parser.add_argument("--baseline-calibration", type=Path, default=ROOT / "output/baseline_calibration_v2_clock")
    parser.add_argument("--baseline-history", type=Path, nargs="*", default=[ROOT / "output/baseline_calibration_v1"])
    parser.add_argument("--coordinated-calibration", type=Path, default=ROOT / "output/coordination_calibration_v4_5seeds")
    parser.add_argument("--coordinated-reused-subset", type=Path, default=ROOT / "output/coordination_calibration_v3")
    parser.add_argument("--coordination-history", type=Path, nargs="*",
                        default=[ROOT / "output/coordination_calibration_v1", ROOT / "output/coordination_calibration_v2"])
    parser.add_argument("--development", type=Path, default=ROOT / "output/development_selected_v3_final")
    parser.add_argument("--development-history", type=Path, nargs="*",
                        default=[ROOT / "output/development_selected_v1", ROOT / "output/development_selected_v2"])
    parser.add_argument("--friction-probe", type=Path, default=ROOT / "output/friction_probe")
    parser.add_argument("--review", type=Path, default=ROOT / "review")
    parser.add_argument("--test-audit", type=Path)
    parser.add_argument("--robustness-audit", type=Path)
    parser.add_argument("--extra-audit", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=ROOT / "第二轮结果与改进路线.md")
    args = parser.parse_args()
    try:
        script_hash = sha256(__file__)
        data = load_all(args)
        data["sources"][str(Path(__file__).resolve())] = script_hash
        report = build_report(data, args)
        require(all(same_hash(sha256(path), expected) for path, expected in data["sources"].items()),
                "生成期间输入或报告脚本改变，拒绝发布报告")
        out = args.out.resolve()
        require(str(out) not in data["sources"], "报告输出不能覆盖输入证据")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(json.dumps({"report": str(out), "sha256": sha256(out),
                          "formal_trials": sum(len(suite.trials) for suite in data["suites"].values()),
                          "sources_verified": len(data["sources"]), "characters": len(report)}, ensure_ascii=False))
    except (ValueError, KeyError, OSError, AssertionError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

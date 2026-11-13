"""Frozen paired scientific comparison; standard library plus NumPy only.

No controller parameters are selected or modified here. The independent unit is
a simulation seed, clustered across all declared scenarios. Continuous effects
use only explicitly eligible pairs; absent measurements are never filled with 0.
"""
from pathlib import Path
from dataclasses import dataclass
import argparse
import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

BOOTSTRAP_REPS = 10000
SIGN_FLIP_REPS = 100000
DEFAULT_RANDOM_SEED = 20261013
ORIENTATION_EXCLUDED_SCENARIOS = ("sphere",)
BASELINES = ("constant", "event", "ratio", "adaptive")
ABLATIONS = ("fusion_no_prediction", "fusion_no_distribution", "fusion_no_filter")
EXPECTED_METHODS = ("constant", "event", "ratio", "adaptive", "fusion", *ABLATIONS)
CODE_FILES = ("controllers.py", "physics.py", "scenarios.py", "run_experiments.py")
DESCRIPTIVE_METRICS = (
    "max_slip_mm", "orientation_max_deg", "orientation_rms_deg",
    "orientation_drift_max_deg", "peak_force_N", "initial_force_peak_N",
    "mean_force_N", "peak_force_limit_ratio", "total_force_excess_Ns",
    "total_budget_excess_Ns", "initial_force_excess_Ns",
    "initial_budget_excess_Ns", "force_excess_Ns", "budget_excess_Ns",
    "over_limit_s", "release_latency_s", "grasp_time_s", "switches",
    "slip_events", "controller_p95_ms", "controller_mean_ms", "wall_seconds",
    "palm_peak_N", "palm_force_Ns", "collision_peak_N", "collision_force_Ns",
)
BOOL_FIELDS = ("setup_ok", "dropped", "completed", "safe_completed",
               "unsafe_proxy", "initial_unsafe_proxy", "hold_safe_completed",
               "nonfinite", "boundary", "robustness")
REQUIRED_TRIAL_FIELDS = (
    "scenario", "method", "seed", "parameters", "gains", "numerical_warnings",
    *BOOL_FIELDS, *DESCRIPTIVE_METRICS,
)
REQUIRED_MANIFEST_FIELDS = (
    "suite", "seeds", "methods", "scenarios", "scenario_definitions",
    "code_sha256", "final_code_sha256", "code_unchanged_during_run",
    "gains", "gains_sha256", "metric_version", "metric_definitions",
    "physics_timestep", "control_timestep", "source_model_sha256",
    "source_manifest", "expected_trials", "completed_trials", "finished",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stable_seed(base_seed, *labels):
    digest = hashlib.sha256("\0".join(map(str, labels)).encode()).digest()
    return (int.from_bytes(digest[:8], "little") + int(base_seed)) % (2**64)


def source_literals(path):
    """Read frozen dataclass defaults/titles without importing physics/MuJoCo."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    gains, methods, titles = {}, None, None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Gains":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    require(item.value is not None, "Gains has a required field")
                    gains[item.target.id] = ast.literal_eval(item.value)
        if isinstance(node, ast.Assign):
            names = [item.id for item in node.targets if isinstance(item, ast.Name)]
            if "METHODS" in names:
                methods = ast.literal_eval(node.value)
            if "METHOD_TITLES" in names:
                titles = ast.literal_eval(node.value)
    require(gains and methods and titles, "Frozen controller definitions unavailable")
    return gains, tuple(methods), titles


def frozen_scenario_definitions(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    fields, defaults, calls = [], {}, None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Scenario":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields.append(item.target.id)
                    if item.value is not None:
                        defaults[item.target.id] = ast.literal_eval(item.value)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SCENARIOS" for t in node.targets):
            require(isinstance(node.value, ast.List), "Frozen SCENARIOS must be a literal list")
            calls = node.value.elts
    require(fields and calls, "Frozen scenario definitions unavailable")
    result = []
    for call in calls:
        require(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "Scenario", "Unexpected frozen scenario constructor")
        definition = defaults.copy()
        definition.update({field: ast.literal_eval(value) for field, value in zip(fields, call.args)})
        definition.update({item.arg: ast.literal_eval(item.value) for item in call.keywords})
        require(set(definition) == set(fields), "Incomplete frozen scenario constructor")
        result.append(definition)
    return json.loads(json.dumps(result))


@dataclass
class Suite:
    name: str
    directory: Path
    manifest: dict
    trials: list
    indexed: dict
    definitions: dict
    defaults: dict
    titles: dict
    hashes: dict

    @property
    def seeds(self):
        return sorted(self.manifest["seeds"])

    def domain_names(self, domain):
        names = self.manifest["scenarios"]
        if domain == "all":
            return names
        is_boundary = domain == "boundary"
        return [name for name in names
                if self.definitions[name]["boundary"] is is_boundary]


def load_suite(directory, expected_suite):
    directory = Path(directory).resolve()
    manifest_path, trials_path = directory/"manifest.json", directory/"trials.json"
    manifest, trials = json_read(manifest_path), json_read(trials_path)
    missing = set(REQUIRED_MANIFEST_FIELDS) - manifest.keys()
    require(not missing, f"{directory}: manifest missing {sorted(missing)}")
    require(manifest["suite"] == expected_suite, f"{directory}: incorrect suite")
    expected_seeds = list(range(3001, 3021)) if expected_suite == "test" else list(range(6501, 6511))
    require(sorted(manifest["seeds"]) == expected_seeds, f"{expected_suite}: incorrect seeds")
    require(len(manifest["seeds"]) == len(set(manifest["seeds"])), "Duplicate seeds")
    methods = manifest["methods"]
    require(len(methods) == 8 and set(methods) == set(EXPECTED_METHODS),
            f"{expected_suite}: expected the declared 8 methods")
    require(len(methods) == len(set(methods)), "Duplicate methods")
    names = manifest["scenarios"]
    require(names and len(names) == len(set(names)), "Missing or duplicate scenarios")
    definitions_list = manifest["scenario_definitions"]
    require(isinstance(definitions_list, list), "scenario_definitions must be a list")
    definitions = {item["name"]: item for item in definitions_list}
    require(len(definitions) == len(definitions_list), "Duplicate scenario definitions")
    require(set(names) == set(definitions), "Not all declared scenarios were run")
    require(all(type(item.get("boundary")) is bool for item in definitions_list),
            "Boundary tags must be explicitly declared")
    require(sum(item["boundary"] for item in definitions_list) == 1,
            "Exactly one predeclared boundary scenario required")
    require(any(not item["boundary"] for item in definitions_list), "No main-domain scenarios")
    require(manifest["metric_version"] >= 5, "Metrics predating cross-phase/RMS repairs rejected")
    hashes, final_hashes = manifest["code_sha256"], manifest["final_code_sha256"]
    require(set(hashes) == set(CODE_FILES) and set(final_hashes) == set(CODE_FILES),
            "Code hash maps must contain the complete four-file key set")
    require(manifest["code_unchanged_during_run"] is True and hashes == final_hashes,
            "Code changed during frozen run")
    for flag in ("freeze", "frozen"):
        if flag in manifest:
            require(manifest[flag] is True, f"Manifest {flag} must be true")
    snapshot = directory/"snapshot"
    snapshot_hashes = {}
    for name in CODE_FILES:
        file_path = snapshot/name
        require(file_path.is_file(), f"Frozen snapshot absent: {name}")
        snapshot_hashes[name] = sha256(file_path)
        require(snapshot_hashes[name] == hashes[name], f"Snapshot hash mismatch: {name}")
    require(manifest["scenario_definitions"] == frozen_scenario_definitions(snapshot/"scenarios.py"),
            "Scenario definitions differ from frozen source")
    defaults, frozen_methods, titles = source_literals(snapshot/"controllers.py")
    require(set(frozen_methods) == set(EXPECTED_METHODS), "Frozen method definitions differ")
    source_model = snapshot/"source_model.xml"
    require(source_model.is_file() and sha256(source_model) == manifest["source_model_sha256"],
            "Frozen source model hash mismatch")
    gains_map = manifest["gains"]
    require(isinstance(gains_map, dict) and not(set(gains_map)-set(EXPECTED_METHODS)),
            "Unexpected gains method key")
    require(all(isinstance(value, dict) for value in gains_map.values()), "Invalid gains map")
    require(all(not(set(value)-set(defaults)) for value in gains_map.values()),
            "Unknown gains field")
    if manifest["gains_sha256"] is not None:
        gain_file = snapshot/"selected_gains.json"
        require(gain_file.is_file() and sha256(gain_file) == manifest["gains_sha256"],
                "Selected gains hash mismatch")
        require(json_read(gain_file) == gains_map, "Selected gains content differs")
    effective_gains = {name: defaults | gains_map.get(name, {}) for name in methods}
    for name in ABLATIONS:
        require(effective_gains[name] == effective_gains["fusion"],
                f"Ablation was independently retuned: {name}")
    expected_keys = {(scenario, method, seed)
                     for scenario in names for method in methods for seed in expected_seeds}
    require(isinstance(trials, list), "trials.json must be a list")
    require(len(trials) == manifest["expected_trials"] == manifest["completed_trials"] == len(expected_keys),
            "Frozen trial count incomplete")
    indexed, parameter_pairs = {}, {}
    for trial in trials:
        missing = set(REQUIRED_TRIAL_FIELDS) - trial.keys()
        require(not missing, f"Trial fields missing: {sorted(missing)}")
        key = trial["scenario"], trial["method"], trial["seed"]
        require(key in expected_keys and key not in indexed, f"Unexpected/duplicate trial: {key}")
        require(all(type(trial[field]) is bool for field in BOOL_FIELDS), f"Invalid boolean at {key}")
        require(trial["robustness"] is (expected_suite == "robustness"), f"Robustness flag at {key}")
        require(trial["boundary"] is definitions[key[0]]["boundary"], f"Boundary flag at {key}")
        require(trial["gains"] == effective_gains[key[1]], f"Trial gains differ: {key}")
        pair_key = key[0], key[2]
        previous_parameters = parameter_pairs.setdefault(pair_key, trial["parameters"])
        require(previous_parameters == trial["parameters"], f"Unpaired environment parameters: {key}")
        require(isinstance(trial["parameters"], dict) and trial["parameters"], f"Invalid parameters: {key}")
        for field in DESCRIPTIVE_METRICS:
            value = trial[field]
            require(value is None or (not isinstance(value, bool) and isinstance(value, (int, float))
                                     and np.isfinite(value)), f"Invalid {field} at {key}")
            if field not in ("max_slip_mm", "orientation_max_deg", "orientation_rms_deg",
                             "orientation_drift_max_deg", "release_latency_s", "grasp_time_s"):
                require(value is not None, f"Required measurement absent: {key} {field}")
        require(not trial["safe_completed"] or trial["completed"], f"Safe without completion: {key}")
        require(not trial["completed"] or (trial["setup_ok"] and not trial["dropped"] and not trial["nonfinite"]),
                f"Inconsistent completion: {key}")
        require(not trial["dropped"] or trial["setup_ok"], f"Drop counted before setup: {key}")
        require((trial["max_slip_mm"] is not None) is trial["setup_ok"], f"Slip missingness: {key}")
        for field in ("total_force_excess_Ns", "total_budget_excess_Ns"):
            require(trial[field] is not None and trial[field] >= 0, f"Missing/negative {field}: {key}")
        for full, initial, hold in (("total_force_excess_Ns", "initial_force_excess_Ns", "force_excess_Ns"),
                                    ("total_budget_excess_Ns", "initial_budget_excess_Ns", "budget_excess_Ns")):
            require(trial[initial] is not None and trial[hold] is not None
                    and np.isclose(trial[full], trial[initial]+trial[hold], atol=1e-9, rtol=1e-9),
                    f"Whole-task integral does not equal stage sum: {key} {full}")
        require(isinstance(trial["numerical_warnings"], int) and trial["numerical_warnings"] >= 0,
                f"Invalid numerical warning count: {key}")
        require(not trial["safe_completed"] or (
            not trial["unsafe_proxy"] and not trial["initial_unsafe_proxy"]
            and trial["numerical_warnings"] == 0), f"Invalid safe completion: {key}")
        indexed[key] = trial
    require(set(indexed) == expected_keys, "Full scenario-method-seed key set not present")
    return Suite(expected_suite, directory, manifest, trials, indexed, definitions, defaults, titles,
                 {"manifest_sha256": sha256(manifest_path), "trials_sha256": sha256(trials_path),
                  "snapshot_sha256": snapshot_hashes, "source_model_sha256": sha256(source_model)})


def validate_suite_pair(test, robustness):
    keys = ("methods", "scenarios", "scenario_definitions", "code_sha256", "final_code_sha256",
            "gains", "gains_sha256", "metric_version", "metric_definitions",
            "physics_timestep", "control_timestep", "source_model_sha256", "source_manifest")
    for key in keys:
        require(test.manifest[key] == robustness.manifest[key],
                f"Test/robustness frozen settings differ: {key}")
    require(test.defaults == robustness.defaults, "Frozen default gains differ")


def finite_values(trials, field, condition=lambda trial: True):
    return np.asarray([float(trial[field]) for trial in trials
                       if condition(trial) and trial.get(field) is not None
                       and np.isfinite(trial[field])], dtype=float)


def numeric_summary(values):
    if not len(values):
        return dict(n=0, mean=None, median=None, p95=None, minimum=None, maximum=None, sum=None)
    return dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), minimum=float(np.min(values)),
                maximum=float(np.max(values)), sum=float(np.sum(values)))


def summarize_method(trials, definitions, raw_sphere_orientation=False):
    n = len(trials)
    counts = {
        "setup": sum(t["setup_ok"] for t in trials),
        "setup_failure": sum(not t["setup_ok"] for t in trials),
        "drop": sum(t["dropped"] for t in trials),
        "loss_including_setup_failure": sum(not t["setup_ok"] or t["dropped"] for t in trials),
        "completed": sum(t["completed"] for t in trials),
        "safe_completed": sum(t["safe_completed"] for t in trials),
        "hold_safe_completed": sum(t["hold_safe_completed"] for t in trials),
        "initial_force_violation": sum(t["initial_unsafe_proxy"] for t in trials),
        "hold_force_violation": sum(t["unsafe_proxy"] for t in trials),
        "nonfinite": sum(t["nonfinite"] for t in trials),
        "numerical_warning_trials": sum(t["numerical_warnings"] > 0 for t in trials),
        "other_failure_after_setup": sum(t["setup_ok"] and not t["dropped"] and not t["completed"] for t in trials),
        "completed_but_not_safe": sum(t["completed"] and not t["safe_completed"] for t in trials),
    }
    release_trials = [t for t in trials if definitions[t["scenario"]].get("release_at") is not None]
    counts["release_requested"] = len(release_trials)
    counts["release_clearance_measured"] = sum(t.get("release_latency_s") is not None for t in release_trials)
    counts["release_within_3_5s"] = sum(t.get("release_latency_s") is not None
                                       and t["release_latency_s"] <= 3.5 for t in release_trials)
    rates = {key: 100*value/n if n else None for key, value in counts.items()
             if not key.startswith("release_")}
    release_n = len(release_trials)
    rates["release_within_3_5s"] = 100*counts["release_within_3_5s"]/release_n if release_n else None
    continuous = {}
    for field in DESCRIPTIVE_METRICS:
        if field.startswith("orientation_"):
            condition = lambda t: t["setup_ok"] and (raw_sphere_orientation or t["scenario"] not in ORIENTATION_EXCLUDED_SCENARIOS)
        elif field.startswith("max_slip"):
            condition = lambda t: t["setup_ok"]
        else:
            condition = lambda t: True
        selected = release_trials if field == "release_latency_s" else trials
        continuous[field] = numeric_summary(finite_values(selected, field, condition))
    return {"n": n, "counts": counts, "rates_pct": rates, "continuous": continuous,
            "orientation_sphere_excluded_trials": 0 if raw_sphere_orientation else sum(t["scenario"] in ORIENTATION_EXCLUDED_SCENARIOS for t in trials),
            "orientation_usage": "raw sphere full-SO3 diagnostic only" if raw_sphere_orientation else "sphere excluded; full-SO3 remains diagnostic for axial roll of axisymmetric objects"}


def bootstrap_cluster_mean(seed_sums, seed_counts, random_seed):
    total_n = int(np.sum(seed_counts))
    if not total_n:
        return {"estimate": None, "ci95": [None, None], "valid_bootstrap_reps": 0}
    estimate = float(np.sum(seed_sums)/total_n)
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, len(seed_counts), size=(BOOTSTRAP_REPS, len(seed_counts)))
    sums = np.sum(seed_sums[indices], axis=1)
    counts = np.sum(seed_counts[indices], axis=1)
    valid = counts > 0
    values = sums[valid]/counts[valid]
    ci = np.percentile(values, [2.5, 97.5]).tolist() if len(values) else [None, None]
    return {"estimate": estimate, "ci95": ci, "valid_bootstrap_reps": int(np.sum(valid))}


def cluster_sign_flip(seed_sums, random_seed):
    """Exchange both algorithms for an entire seed; retain all within-seed scenes."""
    values = np.asarray(seed_sums, float)
    values = values[np.abs(values) > 1e-12]
    if not len(values):
        return {"p_value": 1.0, "p_method": "exact_cluster_sign_flip", "permutations": 1,
                "nonzero_seed_clusters": 0}
    observed = abs(float(np.sum(values)))
    tolerance = 1e-10
    exceed = 0
    exact = len(values) <= 16
    repetitions = 2**len(values) if exact else SIGN_FLIP_REPS
    rng = np.random.default_rng(random_seed)
    for start in range(0, repetitions, 4096):
        size = min(4096, repetitions-start)
        if exact:
            masks = np.arange(start, start+size, dtype=np.uint64)[:, None]
            signs = 2*((masks >> np.arange(len(values), dtype=np.uint64)) & 1).astype(float)-1
        else:
            signs = rng.integers(0, 2, size=(size, len(values)), dtype=np.int8)*2-1
        statistics = np.abs(np.sum(signs*values, axis=1))
        exceed += int(np.count_nonzero(statistics >= observed-tolerance))
    p = exceed/repetitions if exact else (exceed+1)/(repetitions+1)
    return {"p_value": float(p), "p_method": "exact_cluster_sign_flip" if exact else "monte_carlo_cluster_sign_flip",
            "permutations": repetitions, "nonzero_seed_clusters": len(values)}


def holm_adjust(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.]*len(pvalues)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues)-rank)*pvalues[index])
        adjusted[index] = min(1., running)
    return adjusted


BINARY_METRICS = {
    "safe_completed": lambda t: t["safe_completed"],
    "completed": lambda t: t["completed"],
    "drop": lambda t: t["dropped"],
    "setup_failure": lambda t: not t["setup_ok"],
    "loss_including_setup_failure": lambda t: not t["setup_ok"] or t["dropped"],
    "hold_safe_completed": lambda t: t["hold_safe_completed"],
}


def paired_effect(suite, domain, comparator, metric, condition, base_seed):
    scenarios, seeds = suite.domain_names(domain), suite.seeds
    is_binary = metric in BINARY_METRICS
    get_value = BINARY_METRICS[metric] if is_binary else lambda t: t.get(metric)
    values_a, values_b, diffs, pair_seeds = [], [], [], []
    total_pairs = len(scenarios)*len(seeds)
    for seed in seeds:
        for scenario in scenarios:
            a, b = suite.indexed[(scenario, "fusion", seed)], suite.indexed[(scenario, comparator, seed)]
            eligible = True
            if condition == "both_setup":
                eligible = a["setup_ok"] and b["setup_ok"]
            elif condition == "jointly_completed":
                eligible = a["completed"] and b["completed"]
            elif condition == "both_release_measured":
                eligible = (suite.definitions[scenario].get("release_at") is not None
                            and a["setup_ok"] and b["setup_ok"])
            if metric.startswith("orientation_") and scenario in ORIENTATION_EXCLUDED_SCENARIOS:
                eligible = False
            va, vb = get_value(a), get_value(b)
            if not eligible or va is None or vb is None:
                continue
            if not np.isfinite(va) or not np.isfinite(vb):
                continue
            va, vb = float(va), float(vb)
            values_a.append(va)
            values_b.append(vb)
            diffs.append((va-vb)*(100 if is_binary else 1))
            pair_seeds.append(seed)
    diffs, values_a, values_b = map(lambda x: np.asarray(x, float), (diffs, values_a, values_b))
    sums = np.asarray([np.sum(diffs[np.asarray(pair_seeds) == seed]) for seed in seeds], float)
    counts = np.asarray([pair_seeds.count(seed) for seed in seeds], int)
    label = (suite.name, domain, comparator, metric, condition)
    boot_seed = stable_seed(base_seed, "bootstrap", *label)
    result = {
        "suite": suite.name, "domain": domain, "method": "fusion", "comparator": comparator,
        "comparison_type": "baseline" if comparator in BASELINES else "ablation",
        "metric": metric, "condition": condition,
        "unit": "percentage_points" if is_binary else metric.rsplit("_", 1)[-1],
        "effect_definition": "mean(fusion minus comparator) over eligible paired trials",
        "favorable_direction": "positive" if metric in ("safe_completed", "completed", "hold_safe_completed") else "negative",
        "n_total_pairs": total_pairs, "n_pairs": len(diffs), "n_excluded_pairs": total_pairs-len(diffs),
        "n_seed_clusters": len(seeds), "n_seed_clusters_with_pairs": int(np.count_nonzero(counts)),
        "scenarios": scenarios, "bootstrap_seed": boot_seed,
        "conditional_selection": condition != "all",
        "orientation_diagnostic_only": metric.startswith("orientation_"),
        "orientation_excluded_scenarios": list(ORIENTATION_EXCLUDED_SCENARIOS) if metric.startswith("orientation_") else [],
        "exploratory": not (suite.name == "test" and domain == "main" and comparator in BASELINES
                            and metric == "safe_completed" and condition == "all"),
        "fusion_mean": float(np.mean(values_a))*(100 if is_binary else 1) if len(diffs) else None,
        "comparator_mean": float(np.mean(values_b))*(100 if is_binary else 1) if len(diffs) else None,
        "fusion_median": float(np.median(values_a))*(100 if is_binary else 1) if len(diffs) else None,
        "comparator_median": float(np.median(values_b))*(100 if is_binary else 1) if len(diffs) else None,
        "median_paired_difference": float(np.median(diffs)) if len(diffs) else None,
        "seed_difference_sums": {str(seed): float(value) for seed, value in zip(seeds, sums)},
        "seed_eligible_counts": {str(seed): int(value) for seed, value in zip(seeds, counts)},
    }
    result.update(bootstrap_cluster_mean(sums, counts, boot_seed))
    result["ci95_low"], result["ci95_high"] = result["ci95"]
    result["p_value_holm"] = None
    result["holm_family"] = None
    if is_binary and len(diffs):
        result["sign_flip_seed"] = stable_seed(base_seed, "sign_flip", *label)
        result.update(cluster_sign_flip(sums, result["sign_flip_seed"]))
    else:
        result.update(p_value=None, p_method=None, permutations=0, nonzero_seed_clusters=None)
    return result


def generate_effects(suite, base_seed):
    effects = []
    for domain in ("main", "all", "boundary"):
        for comparator in (*BASELINES, *ABLATIONS):
            for metric in BINARY_METRICS:
                effects.append(paired_effect(suite, domain, comparator, metric, "all", base_seed))
            for metric in ("total_force_excess_Ns", "total_budget_excess_Ns",
                           "peak_force_N", "initial_force_peak_N", "mean_force_N",
                           "controller_p95_ms", "wall_seconds", "palm_peak_N",
                           "palm_force_Ns", "collision_peak_N", "collision_force_Ns"):
                effects.append(paired_effect(suite, domain, comparator, metric, "all", base_seed))
            for metric in ("max_slip_mm", "orientation_rms_deg", "orientation_max_deg"):
                effects.append(paired_effect(suite, domain, comparator, metric, "both_setup", base_seed))
            effects.append(paired_effect(suite, domain, comparator, "release_latency_s",
                                         "both_release_measured", base_seed))
            for metric in ("max_slip_mm", "total_force_excess_Ns",
                           "total_budget_excess_Ns", "orientation_rms_deg"):
                effects.append(paired_effect(suite, domain, comparator, metric, "jointly_completed", base_seed))
    primary_rows = [row for row in effects if row["domain"] == "main" and row["metric"] == "safe_completed"
                    and row["comparator"] in BASELINES and row["condition"] == "all"]
    require(len(primary_rows) == 4, "Holm baseline family must have four comparisons")
    for row, adjusted in zip(primary_rows, holm_adjust([row["p_value"] for row in primary_rows])):
        row["p_value_holm"] = adjusted
        row["holm_family"] = f"{suite.name}.main.safe_completed.four_baselines"
        row["family_status"] = "primary" if suite.name == "test" else "secondary_robustness"
    return effects


def flatten_summary(row):
    flat = {key: value for key, value in row.items() if key not in ("counts", "rates_pct", "continuous")}
    flat.update({f"{key}_count": value for key, value in row["counts"].items()})
    flat.update({f"{key}_rate_pct": value for key, value in row["rates_pct"].items()})
    for field, stats in row["continuous"].items():
        flat.update({f"{field}_{stat}": value for stat, value in stats.items()})
    return flat


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def generate_summary(suite):
    domain_rows, scenario_rows = [], []
    for domain in ("main", "all", "boundary"):
        names = suite.domain_names(domain)
        for method in EXPECTED_METHODS:
            trials = [t for t in suite.trials if t["method"] == method and t["scenario"] in names]
            row = dict(suite=suite.name, domain=domain, method=method, method_title=suite.titles[method],
                       n_scenarios=len(names), n_seed_clusters=len(suite.seeds), scenarios=names)
            row.update(summarize_method(trials, suite.definitions))
            domain_rows.append(row)
    for scenario in suite.manifest["scenarios"]:
        for method in EXPECTED_METHODS:
            trials = [suite.indexed[(scenario, method, seed)] for seed in suite.seeds]
            row = dict(suite=suite.name, scenario=scenario,
                       scenario_title=suite.definitions[scenario]["title"],
                       domain="boundary" if suite.definitions[scenario]["boundary"] else "main",
                       method=method, method_title=suite.titles[method], n_seed_clusters=len(suite.seeds))
            row.update(summarize_method(trials, suite.definitions,
                                       raw_sphere_orientation=scenario in ORIENTATION_EXCLUDED_SCENARIOS))
            scenario_rows.append(row)
    return domain_rows, scenario_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    analysis_hash = sha256(__file__)
    test, robustness = load_suite(args.test, "test"), load_suite(args.robustness, "robustness")
    validate_suite_pair(test, robustness)
    domains, scenarios, effects = [], [], []
    for suite in (test, robustness):
        domain_rows, scenario_rows = generate_summary(suite)
        domains.extend(domain_rows)
        scenarios.extend(scenario_rows)
        effects.extend(generate_effects(suite, args.bootstrap_seed))
    metadata = {
        "analysis_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script_sha256": analysis_hash,
        "inputs": {suite.name: {"directory": str(suite.directory), **suite.hashes,
                              "validation": "complete frozen snapshot, exact key set, seed split, paired parameters and gains"}
                   for suite in (test, robustness)},
        "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_base_seed": args.bootstrap_seed,
        "sign_flip_monte_carlo_reps": SIGN_FLIP_REPS, "ci_level": .95,
        "bootstrap_method": "paired seed-cluster percentile bootstrap; sample seed clusters with replacement and retain every scene within each sampled cluster; conditional effects divide resampled difference sums by resampled eligible pair counts",
        "sign_flip_method": "two-sided whole-seed algorithm-label exchange; exact when <=16 nonzero clusters, otherwise Monte Carlo with plus-one correction",
        "primary_hypothesis_family": "test.main.safe_completed fusion versus four prespecified baselines; Holm familywise correction",
        "secondary_hypothesis_family": "robustness.main.safe_completed four baselines, separately Holm adjusted",
        "other_inference": "all/boundary domains, other metrics and all three ablations are exploratory; confidence intervals are marginal, not multiplicity-adjusted",
        "continuous_missingness": "do not impute absent measurements; max_slip and orientation effects require both setups valid; jointly completed effects are additionally selected on outcomes and cannot establish population-wide superiority",
        "orientation_interpretation": "sphere full-SO3 is excluded from aggregate summaries and all paired orientation effects; its per-scenario raw values are diagnostics. Full-SO3 axial roll for capsules/cylinders is likewise diagnostic and does not establish functional orientation precision; no symmetry-invariant axis error is computed by this script",
        "orientation_aggregate_excluded_scenarios": list(ORIENTATION_EXCLUDED_SCENARIOS),
        "drop_definition": "confirmed unintended loss after grasp setup; setup failure reported separately; combined loss is their disjoint union",
        "force_integrals": "total_force_excess_Ns sums per-finger ceiling exceedance over the whole task; total_budget_excess_Ns integrates five-finger total-budget exceedance separately; never sum these overlapping quantities into a single score",
        "force_failure_caveat": "failed grasp or drop may lower subsequent contact force; lower unconditional force exposure alone does not demonstrate better control. Whole-task all-trial and jointly-completed paired effects are both retained; the latter is conditional on outcomes and subject to selection",
        "release_definition": "contact clearance, not supported placement; descriptive and paired latency are conditional on measured clearance; absent clearance retains failure in completion counts",
        "runtime_definition": "controller_p95_ms is an episode-level 95th percentile; its across-episode p95 is not a pooled tick-level percentile; wall_seconds is per-episode wall time and parallel episode sums are not suite elapsed time",
        "generalization_boundary": "randomized simulation seeds, not independent human participants; conclusions restricted to the tested hand/model/parameter ranges",
        "method_definitions": {
            "constant": "fixed grip target plus shared servo, bounds and release",
            "event": "literature-inspired threshold incremental force with shared bounds and release",
            "ratio": "Matos-inspired shear/normal ratio feedback transferred to common force servo",
            "adaptive": "Zhang-inspired per-finger friction-cone margin adaptation transferred to common servo",
            "fusion": "tactile task coordination with predictive force protection, confidence, redistribution and coupled anti-windup",
            "fusion_no_prediction": "fusion ablation disabling force prediction",
            "fusion_no_distribution": "fusion ablation replacing local allocation with common proportional grip",
            "fusion_no_filter": "fusion ablation disabling temporal median filtering",
        },
        "methods": list(EXPECTED_METHODS), "method_titles": test.titles,
        "metric_definitions_from_frozen_runner": test.manifest["metric_definitions"],
        "frozen_gains": test.manifest["gains"],
        "scenario_definitions": test.manifest["scenario_definitions"],
        "suite_elapsed_seconds": {suite.name: suite.manifest.get("wall_seconds") for suite in (test, robustness)},
    }
    out = args.out.resolve()
    require(out not in (test.directory, robustness.directory), "Analysis output cannot overwrite input directory")
    require(sha256(__file__) == analysis_hash, "Analysis script changed during analysis")
    for suite in (test, robustness):
        require(sha256(suite.directory/"manifest.json") == suite.hashes["manifest_sha256"]
                and sha256(suite.directory/"trials.json") == suite.hashes["trials_sha256"],
                f"{suite.name}: input data changed during analysis")
    out.mkdir(parents=True, exist_ok=True)
    for name, content in (("summary.json", {"metadata": metadata, "domains": domains, "scenarios": scenarios}),
                          ("paired_effects.json", {"metadata": metadata, "effects": effects})):
        (out/name).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(out/"summary.csv", [flatten_summary(row) for row in domains])
    write_csv(out/"scenario_summary.csv", [flatten_summary(row) for row in scenarios])
    write_csv(out/"paired_effects.csv", effects)
    print(json.dumps({"output": str(out), "domain_rows": len(domains),
                      "scenario_rows": len(scenarios), "paired_effect_rows": len(effects),
                      "input_trials": len(test.trials)+len(robustness.trials),
                      "analysis_script_sha256": metadata["analysis_script_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


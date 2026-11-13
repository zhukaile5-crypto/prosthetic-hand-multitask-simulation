"""Independent all25-axis micro refit/mixed causal replay. No author imports."""
import argparse
import json
from pathlib import Path
import numpy as np
from verify_robot_servo_response_v1 import (
    ROOT, AXES25, UNITS25, INPUT8, QADR25, Audit, read, sha, robot25,
    independent_micro_kernel, independent_ramp_convolution, error_summary,
    task_tolerance_consumption)

HORIZONS = (.02, .04, .06, .1)


def closed_form_past_weights(times, origin, horizon):
    """Affine LS using scalar centered moments, independent of author pinv."""
    t = np.asarray(times, float)
    indices = np.flatnonzero((np.arange(len(t)) <= origin) & (t >= t[origin] - .12 - 1e-10))
    weights = np.zeros(len(t))
    if len(indices) == 1:
        weights[indices[0]] = 1.
    else:
        x = t[indices] - t[origin]
        center = x.mean(); centered = x - center
        weights[indices] = 1 / len(x) + (horizon - center) * centered / (centered @ centered)
    return weights


def check_stats(audit, label, error, recorded):
    summary = error_summary(error)
    audit.check(label + "/count", summary["samples"] == recorded["samples"])
    for own, saved in (("rmse", "rmse_per_axis25"), ("p95_abs", "p95_abs_per_axis25"), ("max_abs", "max_abs_per_axis25")):
        delta = float(np.max(abs(np.asarray(summary[own]) - recorded[saved])))
        audit.check(label + "/" + own, delta < 1e-10, max_arithmetic_difference=delta)
    return summary


def load_selected_raw(base, context):
    directory = base / context
    branches = {}
    for child in sorted(directory.iterdir()):
        metadata_path = child / "metadata.json"
        if not child.is_dir() or not metadata_path.exists():continue
        m = read(metadata_path)
        if not m.get("available"):continue
        with np.load(child / "trace.npz", allow_pickle=False) as z:
            branches[child.name] = dict(q=robot25(z), u=z["input_reference"].copy(), t=z["time"].copy() - z["time"][0])
    return branches


def verify_context(model_directory, base, audit):
    model = read(model_directory / "model.json"); context = model["context_id"]
    audit.check(context + "/axes_units_whitelist", tuple(model["output_axes25"]) == AXES25 and tuple(model["input_axes8"]) == INPUT8
                and tuple(model["output_units"]) == UNITS25 and np.array_equal(model["output_qpos_addresses25"], QADR25))
    audit.check(context + "/object_excluded", set(model["free_object_qpos_addresses_excluded"]).isdisjoint(QADR25))
    audit.check(context + "/scope_not_global_or_safety", model["global_model_admitted"] is False and model["new_physics"] is False
                and model["adapter_export_status"].startswith("WITHHELD"))
    audit.check(context + "/dense_source_identity", sha(model["dense_kernel_npz_path"]) == model["dense_kernel_npz_sha256"])
    audit.check(context + "/input_source_identity", sha(model["source_manifest_path"]) == model["source_manifest_sha256"])
    audit.check(context + "/footprint_source_identity", sha(model["actual_footprints_path"]) == model["actual_footprints_sha256"])
    branches = load_selected_raw(base, context); zero = branches["zero"]["q"]
    training = [(b["q"], b["u"]) for name, b in branches.items() if name.startswith("micro_")]
    K, known, plus, minus = independent_micro_kernel(zero, training)
    training_names = {name for name in branches if name.startswith("micro_")}
    audit.check(context + "/training_micro_only_exact", {r["branch"] for r in model["training"]} == training_names)
    audit.check(context + "/actual_direction_flags", np.array_equal(plus, model["known_plus8"]) and np.array_equal(minus, model["known_minus8"]))
    if not known.all():
        audit.check(context + "/whole_unknown_column_unavailable", bool(model["missing_entire_input_columns"]))
        return dict(context_id=context, status="entire_column_unknown", model_sha256=sha(model_directory / "model.json"))
    pooled_differential = []; pooled_causal = {str(h): [] for h in HORIZONS}; branch_reports = []
    validation = model["validation"]
    audit.check(context + "/mixed8_exact_validation_set", {b["branch"] for b in validation["branches"]} == {f"mixed_{i:02d}" for i in range(8)})
    recorded_branches = {r["branch"]: r for r in validation["branches"]}
    with np.load(model["dense_kernel_npz_path"], allow_pickle=False) as saved:
        kernel_delta = float(np.max(abs(K - saved["response25x8"])))
        audit.check(context + "/micro_refit_all101x25x8", kernel_delta < 1e-12, max_arithmetic_difference=kernel_delta)
        audit.check(context + "/dense_origin_zero", np.array_equal(K[0], np.zeros((25, 8))))
        audit.check(context + "/declared_horizon_matrices", np.allclose(K[[4, 8, 12, 20]], model["response25x8_at_diagnostic_horizons"], atol=1e-12, rtol=0))
        for i in range(8):
            name = f"mixed_{i:02d}"; b = branches[name]; q = b["q"]; u = b["u"]; t = b["t"]
            effect = independent_ramp_convolution(K, u)
            effect_delta = float(np.max(abs(effect - saved[name + "_modeled_controlled_increment25"])))
            audit.check(context + "/" + name + "/actual_raw_robot_and_input", np.array_equal(q, saved[name + "_actual_q25"]) and np.array_equal(u, saved[name + "_actual_reference8"]))
            audit.check(context + "/" + name + "/command_convolution", effect_delta < 1e-10, max_arithmetic_difference=effect_delta)
            differential = q - zero - effect; pooled_differential.extend(differential)
            audit.check(context + "/" + name + "/differential_raw_errors", np.max(abs(differential - saved[name + "_paired_differential_error25"])) < 1e-10)
            check_stats(audit, context + "/" + name + "/paired_offline", differential, recorded_branches[name]["paired_zero_differential"])
            residual = q - effect; causal_summaries = {}
            for h in HORIZONS:
                offset = round(h / .005); origins = list(range(0, 101 - offset, 4))
                W = np.array([closed_form_past_weights(t, origin, t[origin + offset] - t[origin]) for origin in origins])
                predictions = effect[np.array(origins) + offset] + W @ residual
                actual = q[np.array(origins) + offset]; error = actual - predictions
                prefix = name + "_h" + str(round(h * 1000))
                weight_delta = float(np.max(abs(W - saved[prefix + "_causal_past_q_weights"])))
                pred_delta = float(np.max(abs(predictions - saved[prefix + "_prediction25"])))
                audit.check(context + "/" + prefix + "/independent_past_OLS", weight_delta < 1e-10 and pred_delta < 1e-10,
                            maximum_weight_difference=weight_delta, maximum_prediction_difference=pred_delta)
                audit.check(context + "/" + prefix + "/no_future_encoder_weight", all(not np.any(W[row, origin + 1:]) for row, origin in enumerate(origins)))
                # Explicit adversarial future encoder perturbation at one origin.
                altered = residual.copy(); altered[origins[0] + 1:] += 123.
                audit.check(context + "/" + prefix + "/future_encoder_perturbation_invariant", np.array_equal(W[0] @ residual, W[0] @ altered))
                audit.check(context + "/" + prefix + "/actual_targets_and_errors", np.array_equal(actual, saved[prefix + "_actual25"])
                            and np.max(abs(error - saved[prefix + "_error25"])) < 1e-10)
                pooled_causal[str(h)].extend(error)
                causal_summaries[str(h)] = check_stats(audit, context + "/" + prefix + "/scores", error, recorded_branches[name]["causal_by_horizon"][str(h)])
            branch_reports.append(dict(branch=name, differential=error_summary(differential), causal=causal_summaries))
    differential = check_stats(audit, context + "/pooled_paired_offline", pooled_differential, validation["paired_zero_differential"])
    causal = {h: check_stats(audit, context + "/pooled_causal" + h, e, validation["causal_by_horizon"][h]) for h, e in pooled_causal.items()}
    return dict(context_id=context, model_sha256=sha(model_directory / "model.json"), kernel_max_difference=kernel_delta,
        branches=branch_reports, paired_zero_differential=differential, causal_by_horizon=causal,
        task_tolerance_consumption={h: task_tolerance_consumption(summary["max_abs"], .01, (.015, .015)) for h, summary in causal.items()},
        scope="full selected context raw/micro/mixed; past-only affine residual plus publicly known future reference; no force safety certificate")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=ROOT / "calibration/phase3_path_validation_v3")
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    audit = Audit(); context = verify_context(args.model_directory, args.base, audit)
    result = dict(schema="independent_robot_servo_model_recompute_v1", passed=all(c["passed"] for c in audit.checks), checks=audit.checks, context=context,
        global_model_admitted=False, pressure_safety_certified=False, new_physics=False, analysis_source_sha256=sha(__file__))
    with args.output.open("x", encoding="utf-8") as stream:json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=result["passed"], checks=len(result["checks"]), failed=[c["name"] for c in result["checks"] if not c["passed"]], model=context["model_sha256"], physics_steps=0), ensure_ascii=False))
    if not result["passed"]:raise SystemExit(1)


if __name__ == "__main__":main()

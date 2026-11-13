"""Archived public-reference bridge audit, no runtime imports or physics."""
from pathlib import Path
import json
import numpy as np
from verify_robot_servo_response_v1 import ROOT, INPUT8, Audit, read, sha

ENVIRONMENTS = ("nominal", "friction_drop", "fragile_conflict", "collision")
EPSILONS = (1e-12, 1e-10, 1e-9)


def training_amplitudes():
    by_axis = [[] for _ in range(8)]
    base = ROOT / "output/robot_servo_response_all15_v2"
    for shape in ("capsule", "box", "sphere"):
        for i in range(5):
            model = read(base / shape / f"checkpoint_{i:02d}/model.json")
            for row in model["training"]:
                by_axis[row["input_axis"]].append(abs(row["denominator"]))
    return [dict(min=min(v), max=max(v), count=len(v)) for v in by_axis]


def diagnose(path, task, environment, amplitudes, audit):
    with np.load(path, allow_pickle=False) as z:
        time = z["time"].copy()
        indices = np.flatnonzero((time >= 7.5 - 1e-9) & (time <= 12. + 1e-9))
        frames = [json.loads(str(x)) for x in z["observation_json"][indices]]
        t = time[indices]
        u = np.array([f["alpha"] + f["executed_motion_reference"] for f in frames], float)
        requested = z["requested_action"][indices[:-1]].copy()
        executed = z["executed_action"][indices[:-1]].copy()
        alpha_before = z["alpha_before"][indices].copy()
        trace_motion_before = z["executed_motion_reference"][indices[:-1] - 1].copy()
    label = task + "/" + environment
    audit.check(label + "/901_preaction_observations", len(indices) == 901 and u.shape == (901, 8))
    audit.check(label + "/public_clock_5ms", np.allclose(t, 7.5 + np.arange(901) * .005, atol=2e-12, rtol=0))
    audit.check(label + "/public_alpha_is_current_not_future", np.array_equal(u[:, :5], alpha_before))
    # Motion observed at this time belongs to the previous completed execution.
    audit.check(label + "/public_motion_reference_is_current", np.array_equal(u[:-1, 5:], trace_motion_before))
    actual = np.diff(u[:, :5], axis=0)
    audit.check(label + "/actual_alpha_increment_clock", np.array_equal(actual, executed))
    segments = []
    fraction = np.arange(5)[:, None] / 4
    for k, start in enumerate(range(0, 900, 4)):
        block = u[start:start + 5]
        straight = block[0] + fraction * (block[4] - block[0])
        residual = block - straight
        max_abs = np.max(abs(residual), axis=0)
        max_where = np.argmax(abs(residual), axis=0)
        request_actual_delta = np.max(abs(requested[start:start + 4] - actual[start:start + 4]), axis=0)
        request_spread = np.ptp(requested[start:start + 4], axis=0)
        lower = np.array([f["closure_min"] for f in frames[start:start + 5]], float)
        upper = np.array([f["closure_max"] for f in frames[start:start + 5]], float)
        bound_touch = np.any((abs(block[:, :5] - lower) <= 1e-10) | (abs(block[:, :5] - upper) <= 1e-10), axis=0)
        segments.append(dict(interval=k, start_s=float(t[start]), end_s=float(t[start + 4]),
            max_deviation8=max_abs.tolist(), worst_time_s_by_axis=[float(t[start + j]) for j in max_where],
            relative_to_micro_amplitude_min=[float(max_abs[j] / amplitudes[j]["min"]) for j in range(8)],
            relative_to_micro_amplitude_max=[float(max_abs[j] / amplitudes[j]["max"]) for j in range(8)],
            requested_vs_actual_max_delta5=request_actual_delta.tolist(), requested_tick_spread5=request_spread.tolist(),
            observed_domain_boundary_touch5=bound_touch.tolist(),
            nonuniform_request_without_extra_execution_change5=((max_abs[:5] > 1e-10) & (request_spread > 1e-10) & (request_actual_delta <= 1e-12)).tolist()))
    errors = np.array([s["max_deviation8"] for s in segments])
    peaks = []
    for axis in range(8):
        k = int(np.argmax(errors[:, axis])); s = 4 * k
        peaks.append(dict(axis=INPUT8[axis], interval=k, time_s=[float(t[s]), float(t[s + 4])],
            deviation=float(errors[k, axis]), observed_reference_five_samples=u[s:s + 5, axis].tolist(),
            actual_four_increments=np.diff(u[s:s + 5, axis]).tolist(),
            requested_four_increments=requested[s:s + 4, axis].tolist() if axis < 5 else None,
            relative_to_micro_amplitude_range=[float(errors[k, axis] / amplitudes[axis]["max"]), float(errors[k, axis] / amplitudes[axis]["min"])]))
    return dict(task=task, environment=environment, trace_path=str(path), trace_sha256=sha(path), input_axes8=INPUT8,
        intervals=225, interval_deviation_max8=errors.max(axis=0).tolist(),
        segment_counts_by_numeric_diagnostic_threshold={str(eps): (errors > eps).sum(axis=0).tolist() for eps in EPSILONS},
        peak_segments=peaks, segments=segments,
        actual_request_difference_max5=np.max(abs(requested - actual), axis=0).tolist(),
        max_true_executed_alpha_tick5=np.max(abs(actual), axis=0).tolist(),
        classification="thresholds are numerical diagnostics, not performance/physics admission; true nonlinear requests and additional execution changes are separately recorded")


def main():
    audit = Audit(); amplitudes = training_amplitudes(); records = []
    for task in ("hold", "transport"):
        for environment in ENVIRONMENTS:
            path = ROOT / f"output/development_v9_{task}_v1/runs/{environment}__ratio_preview__610212/trace.npz"
            records.append(diagnose(path, task, environment, amplitudes, audit))
    result = dict(schema="robot_servo_runtime_bridge_diagnostic_v1", data_contract_passed=all(c["passed"] for c in audit.checks), checks=audit.checks,
        records=records, actual_training_micro_amplitude8=amplitudes, source_sha256=sha(__file__),
        scope="8 archived strongest-baseline observed streams, 7.5..12s pre-release/pre-action endpoints; no object state or actual mass input",
        inference="20ms endpoint subsampling does not certify the intervening5ms path as the calibrated finite ramp", new_physics=False)
    output = ROOT / "review/robot_servo_runtime_bridge_v1.json"
    with output.open("x", encoding="utf-8") as stream:json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(data_contract_passed=result["data_contract_passed"], checks=len(audit.checks), failed=[c["name"] for c in audit.checks if not c["passed"]],
        rows=[dict(task=r["task"], environment=r["environment"], max8=r["interval_deviation_max8"], counts_1e10=r["segment_counts_by_numeric_diagnostic_threshold"]["1e-10"], request_actual_max5=r["actual_request_difference_max5"]) for r in records], new_physics=False), ensure_ascii=False))
    if not result["data_contract_passed"]:raise SystemExit(1)


if __name__ == "__main__":main()

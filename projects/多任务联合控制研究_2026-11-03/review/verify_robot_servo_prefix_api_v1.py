"""Independent numerical-prefix API checks; imports no physical runtime."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from verify_robot_servo_response_v1 import ROOT, Audit, read, sha, independent_ramp_convolution
from verify_robot_servo_model_v1 import closed_form_past_weights


def main():
    path = ROOT / "identification/robot_servo_response_v1/causal_forecast.py"
    spec = importlib.util.spec_from_file_location("independent_servo_prefix_subject", path)
    subject = importlib.util.module_from_spec(spec); spec.loader.exec_module(subject)
    audit = Audit(); cp = ROOT / "output/robot_servo_response_all15_v2/capsule/checkpoint_00"
    max_difference = 0.; calls = 0
    with np.load(cp / "robot_response_raw.npz", allow_pickle=False) as z:
        K = z["response25x8"]
        for i in range(8):
            name = f"mixed_{i:02d}"; q = z[name + "_actual_q25"]; u = z[name + "_actual_reference8"]
            original = ROOT / "calibration/phase3_path_validation_v3/capsule/checkpoint_00" / name / "trace.npz"
            with np.load(original, allow_pickle=False) as raw:t = raw["time"] - raw["time"][0]
            effect = independent_ramp_convolution(K, u); residual = q - effect
            for origin in (0, 20, 40, 80):
                hs = np.array([.02, .04, .06, .1])
                result = subject.forecast_from_prefix(K, t[:origin + 1], q[:origin + 1],
                    t[:origin + 1:4], u[:origin + 1:4], t[origin::4], u[origin::4])
                expected = np.array([effect[origin + round(h / .005)] + closed_form_past_weights(t, origin, h) @ residual for h in hs])
                delta = float(np.max(abs(result["predicted_robot_q25"] - expected)))
                max_difference = max(max_difference, delta); calls += 1
                audit.check(name + "/prefix_origin_" + str(origin), delta < 1e-10, maximum_difference=delta)
                audit.check(name + "/explicit_experimental_scope_" + str(origin), result["global_model_admitted"] is False
                            and result["uses_zero_future"] is False and result["uses_future_robot_q"] is False
                            and result["sequence_support"].startswith("EXPERIMENTAL"))
        q = z["mixed_00_actual_q25"]; u = z["mixed_00_actual_reference8"]
        kwargs = dict(kernel101x25x8=K, observed_times_s=t[:21], observed_robot_q25=q[:21],
            actual_boundary_times_s=t[:21:4], actual_boundary_reference8=u[:21:4],
            declared_future_boundary_times_s=t[20::4], declared_future_boundary_reference8=u[20::4])
        bad_kernel = K.copy(); bad_kernel[:, :, 7] = np.nan
        changed_join = u[20::4].copy(); changed_join[0, 0] += .1
        cases = dict(unknown_column=dict(kernel101x25x8=bad_kernel), unscored_horizon=dict(horizons_s=(.03,)),
            future_rewrites_past=dict(declared_future_boundary_reference8=changed_join),
            future_schedule_missing=dict(declared_future_boundary_times_s=t[20:25:4], declared_future_boundary_reference8=u[20:25:4]),
            dropped_context_origin=dict(observed_times_s=t[4:21], observed_robot_q25=q[4:21]),
            nonboundary_origin=dict(observed_times_s=t[:20], observed_robot_q25=q[:20]))
        for name, change in cases.items():
            try:subject.forecast_from_prefix(**dict(kwargs, **change))
            except ValueError:audit.check("reject_" + name, True)
            else:audit.check("reject_" + name, False)
    result = dict(schema="independent_robot_servo_prefix_api_v1", passed=all(c["passed"] for c in audit.checks), checks=audit.checks,
        prefix_calls=calls, numerical_forecasts=calls * 4, maximum_independent_difference=max_difference,
        api_sha256=sha(path), identify_sha256=sha(path.with_name("identify.py")), source_sha256=sha(__file__),
        model_admitted=False, physics_steps=0,
        limitations="Numerical API does not enforce measured plus/minus directions, posture/velocity, input footprints, sequence support or force constraints; separate consumer required")
    out = ROOT / "review/robot_servo_prefix_api_independent_v1.json"
    with out.open("x", encoding="utf-8") as stream:json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=result["passed"], checks=len(audit.checks), prefix_calls=calls, maximum_independent_difference=max_difference,
        failed=[c["name"] for c in audit.checks if not c["passed"]], physics_steps=0), ensure_ascii=False))
    if not result["passed"]:raise SystemExit(1)


if __name__ == "__main__":main()

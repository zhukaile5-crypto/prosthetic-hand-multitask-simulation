"""New consumer assumption: drop history before current-.12s and reset origin.

Calls the FROZEN numerical API, without modifying it or running physics.
The model remains the old context; this reset is a new experimental assumption.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from verify_robot_servo_response_v1 import ROOT, AXES25, UNITS25, read, sha, error_summary

API = ROOT / "identification/robot_servo_response_v1/causal_forecast.py"
IDENTIFY = API.with_name("identify.py")
API_SHA = "09f38bf978b676981f461d08ae38e6c49c0133370c8d365be1acdc9e9474fa36"
IDENTIFY_SHA = "edf555e54a3f84826d0f0cfa9c1cfb0930b243d84e321c6e4652e2c5ee20b98f"
HORIZONS = (.02, .04, .06, .1)


def one_context(context):
    spec = importlib.util.spec_from_file_location("rolling_prefix_frozen_api", API)
    api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
    base = ROOT / "output/robot_servo_response_all15_v2" / context
    model = read(base / "model.json")
    rolling = {str(h): [] for h in HORIZONS}; original = {str(h): [] for h in HORIZONS}
    branches = []; calls = predictions = 0; future_weight_violations = 0
    with np.load(base / "robot_response_raw.npz", allow_pickle=False) as z:
        K = z["response25x8"]
        for i in range(8):
            name = f"mixed_{i:02d}"; q = z[name + "_actual_q25"]; u = z[name + "_actual_reference8"]
            raw = ROOT / "calibration/phase3_path_validation_v3" / context / name / "trace.npz"
            with np.load(raw, allow_pickle=False) as source:t = source["time"] - source["time"][0]
            branch_errors = {str(h): [] for h in HORIZONS}; branch_rows = []
            for origin in range(0, 97, 4):
                available = [h for h in HORIZONS if origin + round(h / .005) <= 100]
                if not available:continue
                start = max(0, origin - 24)  # exactly.12s on already checked5ms grid
                qt = t[start:origin + 1] - t[start]
                at = t[start:origin + 1:4] - t[start]; au = u[start:origin + 1:4]
                ft = t[origin::4] - t[start]; fu = u[origin::4]
                result = api.forecast_from_prefix(K, qt, q[start:origin + 1], at, au, ft, fu, horizons_s=available)
                calls += 1
                for row, h in enumerate(available):
                    end = origin + round(h / .005); error = q[end] - result["predicted_robot_q25"][row]
                    branch_errors[str(h)].append(error); rolling[str(h)].append(error); predictions += 1
                    branch_rows.append(dict(original_origin_s=float(t[origin]), retained_start_s=float(t[start]), retained_origin_s=float(qt[-1]),
                        horizon_s=h, actual_target_time_s=float(t[end]), error25=error.tolist(),
                        maximum_kernel_age_s=float(qt[-1] + h), observed_prefix_samples=len(qt)))
                    # API takes only q[start:origin+1]; future q was never passed.
                    future_weight_violations += int(len(result["residual_weights"][row]) > len(qt))
            reports = {}
            for h in HORIZONS:
                full = z[name + "_h" + str(round(h * 1000)) + "_error25"]
                original[str(h)].extend(full)
                reports[str(h)] = dict(rolling=error_summary(branch_errors[str(h)]), original_full_history=error_summary(full))
            branches.append(dict(branch=name, scores=reports, predictions=branch_rows))
    return dict(context_id=context, model_sha256=sha(base / "model.json"), rolling_by_horizon={h: error_summary(e) for h, e in rolling.items()},
        original_full_history_by_horizon={h: error_summary(e) for h, e in original.items()}, branches=branches,
        api_calls=calls, numerical_forecasts=predictions, future_weight_violations=future_weight_violations,
        maximum_kernel_age_s=max(r["maximum_kernel_age_s"] for b in branches for r in b["predictions"]))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "review/robot_servo_rolling_prefix_v1.json")
    args = parser.parse_args(); assert sha(API) == API_SHA and sha(IDENTIFY) == IDENTIFY_SHA
    contexts = [f"{shape}/checkpoint_{i:02d}" for shape in ("capsule", "box", "sphere") for i in range(5)]
    started = time.perf_counter(); rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(one_context, contexts):
            rows.append(row); print(json.dumps(dict(completed=len(rows), context=row["context_id"], api_calls=row["api_calls"])), flush=True)
    assert sha(API) == API_SHA and sha(IDENTIFY) == IDENTIFY_SHA
    maxima = {}
    for h in map(str, HORIZONS):
        rolling = np.array([r["rolling_by_horizon"][h]["max_abs"] for r in rows])
        full = np.array([r["original_full_history_by_horizon"][h]["max_abs"] for r in rows])
        rolling_rmse = np.array([r["rolling_by_horizon"][h]["rmse"] for r in rows])
        full_rmse = np.array([r["original_full_history_by_horizon"][h]["rmse"] for r in rows])
        maxima[h] = dict(rolling_max_abs25=rolling.max(axis=0).tolist(), full_history_max_abs25=full.max(axis=0).tolist(),
            rolling_max_context_rmse25=rolling_rmse.max(axis=0).tolist(), full_history_max_context_rmse25=full_rmse.max(axis=0).tolist())
    result = dict(schema="robot_servo_rolling_prefix_consumer_diagnostic_v1", complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
        contexts=rows, output_axes25=AXES25, units25=UNITS25, error_maxima=maxima, source_sha256=sha(__file__),
        frozen_api_sha256=API_SHA, frozen_identify_sha256=IDENTIFY_SHA, api_calls=sum(r["api_calls"] for r in rows),
        numerical_forecasts=sum(r["numerical_forecasts"] for r in rows), wall_s=time.perf_counter()-started,
        maximum_kernel_age_s=max(r["maximum_kernel_age_s"] for r in rows), future_weight_violations=sum(r["future_weight_violations"] for r in rows),
        new_physics=False, model_admitted=False,
        scope="new consumer assumption, fixed.12s prefix reset, all120 seen nominal mixed paths; original scored horizons/origins only; no selection of best window",
        limitations=["reset discards older actual command memory and reuses original context kernel at a new state", "local empirical errors are not pressure or global safety bounds", "does not validate indefinite online use beyond the original.5s heldout trajectories", "future references are explicitly declared mixed paths; future robot state/zero never passed to API"])
    with args.output.open("x", encoding="utf-8") as stream:json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(complete=True, contexts=len(rows), api_calls=result["api_calls"], forecasts=result["numerical_forecasts"], wall_s=result["wall_s"], output=str(args.output)), ensure_ascii=False))


if __name__ == "__main__":main()

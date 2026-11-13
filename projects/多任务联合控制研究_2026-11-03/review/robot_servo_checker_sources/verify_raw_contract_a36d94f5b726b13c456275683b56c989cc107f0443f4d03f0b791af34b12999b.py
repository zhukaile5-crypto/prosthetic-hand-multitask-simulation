"""Independent robot-servo data contract; NumPy only, no plant or physics.

The model-output adapter is intentionally separate from this raw-data layer.
No numerical servo admission threshold is invented by this checker.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW24 = ("rh_WRJ2", "rh_WRJ1", "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
         "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1", "rh_RFJ4", "rh_RFJ3",
         "rh_RFJ2", "rh_RFJ1", "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2",
         "rh_LFJ1", "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1")
AXES25 = RAW24 + ("transport_lift",)
UNITS25 = ("rad",) * 24 + ("m",)
INPUT8 = tuple("alpha_" + str(i) for i in range(5)) + (
    "lift_reference_m", "WRJ2_reference_rad", "WRJ1_reference_rad")
# Explicit whitelist for THIS pinned trace layout, verified against checkpoint
# raw24 plus separately recorded lift/wrist. Object-free qpos25:32 is never used.
QADR25 = np.r_[np.arange(1, 25), 0]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, **details):
        self.checks.append(dict(name=name, passed=bool(passed), **details))


def robot25(trace, prefix=""):
    q = np.asarray(trace[prefix + "qpos"], float)
    if q.ndim != 2 or q.shape[1] != 33:
        raise ValueError("This raw layout requires 33 qpos columns and an explicit robot whitelist")
    result = q[:, QADR25]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite robot encoder; object coordinates are not a replacement")
    return result


def error_summary(error):
    e = np.asarray(error, float)
    if e.ndim != 2 or e.shape[1] != 25 or not np.isfinite(e).all():
        raise ValueError("Finite sample-by-25 robot errors required")
    return dict(samples=len(e), rmse=np.sqrt(np.mean(e * e, axis=0)).tolist(),
                p95_abs=np.quantile(abs(e), .95, axis=0).tolist(),
                max_abs=np.max(abs(e), axis=0).tolist(), units=list(UNITS25))


def task_tolerance_consumption(error25, lift_tolerance_m, wrist_tolerance_rad):
    """Unit-aware engineering consumption, not a new admission cutoff."""
    error = np.asarray(error25, float)
    tolerance = np.r_[np.asarray(wrist_tolerance_rad, float), lift_tolerance_m]
    if error.shape != (25,) or tolerance.shape != (3,) or np.any(tolerance <= 0):
        raise ValueError("Explicit existing positive task tolerances and 25-axis errors required")
    actual = error[[0, 1, 24]]
    return dict(axes=["rh_WRJ2", "rh_WRJ1", "transport_lift"],
                actual_abs_error=actual.tolist(), existing_tolerance=tolerance.tolist(),
                fraction_of_existing_tolerance=(actual / tolerance).tolist(),
                scope="prediction-error consumption only; tracking lag and other errors remain separate")


def point_error_envelope(jacobian3x25, empirical_error25):
    """Triangle propagation at one FK linearization, not finite-motion safety."""
    J = np.asarray(jacobian3x25, float)
    error = np.asarray(empirical_error25, float)
    if J.shape != (3, 25) or error.shape != (25,) or np.any(error < 0):
        raise ValueError("Known local J and nonnegative complete robot error required")
    if not np.isfinite(J).all() or not np.isfinite(error).all():
        raise ValueError("Unknown geometry/error must remain unavailable")
    component = abs(J) @ error
    return dict(component_m=component.tolist(), norm_upper_m=float(np.linalg.norm(component)),
                scope="first-order robot-point empirical propagation; nonlinear remainder/external motion unknown")


def audit_clock(audit, label, z):
    t = np.asarray(z["time"], float)
    end = np.asarray(z["physical_time"], float)
    start = np.asarray(z["physical_interval_start"], float)
    duration = np.asarray(z["physical_duration"], float)
    valid = len(t) == 101 and len(end) == 1000 and len(start) == len(end) == len(duration)
    audit.check(label + "/clock_shapes", valid)
    if not valid:
        return
    audit.check(label + "/control_5ms", np.allclose(np.diff(t), .005, atol=2e-12, rtol=0))
    audit.check(label + "/physical_end_start", np.allclose(end - start, duration, atol=2e-12, rtol=0))
    audit.check(label + "/physical_half_ms", np.allclose(duration, .0005, atol=2e-12, rtol=0))
    audit.check(label + "/continuous_substeps", np.allclose(start[1:], end[:-1], atol=2e-12, rtol=0))
    audit.check(label + "/encoder_end_clock", np.allclose(t[1:], end[9::10], atol=2e-12, rtol=0)
                and abs(t[0] - start[0]) <= 2e-12)


def audit_executed_inputs(audit, label, z, metadata):
    u = np.asarray(z["input_reference"], float)
    physical_u = np.asarray(z["physical_input_reference"], float)
    audit.check(label + "/input_shape_finite", u.shape == (101, 8) and np.isfinite(u).all())
    audit.check(label + "/alpha_is_realized_reference", np.array_equal(u[:, :5], z["alpha"]))
    lo = np.asarray(metadata["lower"], float); hi = np.asarray(metadata["upper"], float)
    audit.check(label + "/cumulative_actual_bounds", np.all(u >= lo - 1e-12) and np.all(u <= hi + 1e-12))
    # Every physical substep in a 5ms tick carries the same already-ramped
    # reference; control sample at tick end records that same reference.
    audit.check(label + "/physical_reference_matches_actual_tick",
                physical_u.shape == (1000, 8) and np.allclose(physical_u, np.repeat(u[1:], 10, axis=0), atol=1e-12, rtol=0))
    requested = np.asarray(metadata["requested_path"], float)
    executed = np.asarray(metadata["executed_path"], float)
    audit.check(label + "/20ms_requested_and_5ms_executed_count", requested.shape == (26, 8) and executed.shape == (101, 8))
    if executed.shape == (101, 8):
        audit.check(label + "/executed_path_matches_realized", np.allclose(u, executed, atol=2e-12, rtol=0))
        knots = executed[::4]
        ramp = np.concatenate([np.array([knots[0]]), np.concatenate([
            knots[k] + np.arange(1, 5)[:, None] / 4 * (knots[k + 1] - knots[k])
            for k in range(25)], axis=0)])
        audit.check(label + "/four_tick_ramp", np.allclose(u, ramp, atol=2e-12, rtol=0))
    return dict(actual_increment_min=(u - u[0]).min(axis=0).tolist(),
                actual_increment_max=(u - u[0]).max(axis=0).tolist(),
                requested_executed_knots_max_abs=float(np.max(abs(requested - executed[::4]))))


def audit_raw_context(base, context, audit):
    directory = base / context
    checkpoint = read(directory / "checkpoint.json")
    result = dict(context=context, branches=[])
    zero_path = directory / "zero/trace.npz"
    with np.load(zero_path, allow_pickle=False) as zero:
        zero_robot = robot25(zero)
        audit.check(context + "/checkpoint_robot24", np.array_equal(zero_robot[0, :24], checkpoint["actual_robot_joint_position"]))
        audit.check(context + "/checkpoint_lift", zero_robot[0, 24] == checkpoint["actual_lift_position_m"])
        with np.load(directory / "zero_replay/trace.npz", allow_pickle=False) as replay:
            keys = ("time", "qpos", "qvel", "ctrl", "input_reference", "physical_time", "physical_ctrl", "physical_input_reference")
            # qpos comparison is intentionally ONLY the whitelisted robot25.
            exact = np.array_equal(zero_robot, robot25(replay)) and np.array_equal(robot25(zero, "physical_"), robot25(replay, "physical_"))
            exact &= all(np.array_equal(zero[k], replay[k]) for k in keys if k != "qpos")
            audit.check(context + "/zero_replay_robot_and_input_exact", exact)
        for child in sorted(directory.iterdir()):
            mpath = child / "metadata.json"
            if not child.is_dir() or not mpath.exists():
                continue
            metadata = read(mpath)
            if not metadata.get("attempted_physics"):
                audit.check(context + "/" + child.name + "/unavailable_not_fake_trace", not (child / "trace.npz").exists())
                result["branches"].append(dict(name=child.name, available=False, declared=metadata))
                continue
            label = context + "/" + child.name
            trace = child / "trace.npz"
            audit.check(label + "/raw_hash", sha(trace) == metadata["trace_sha256"])
            with np.load(trace, allow_pickle=False) as z:
                audit_clock(audit, label, z)
                q = robot25(z)
                audit.check(label + "/same_robot_initial_condition", np.array_equal(q[0], zero_robot[0]))
                audit.check(label + "/lift_column", np.array_equal(q[:, 24], z["robot_lift_position_m"]))
                audit.check(label + "/wrist_columns", np.array_equal(q[:, :2], z["robot_wrist_position_rad"]))
                footprint = audit_executed_inputs(audit, label, z, metadata)
                delta = q - zero_robot
                result["branches"].append(dict(name=child.name, available=True, role="train_micro" if child.name.startswith("micro_") else "heldout_mixed" if child.name.startswith("mixed_") else "zero_control",
                    robot_differential_range_min=delta.min(axis=0).tolist(), robot_differential_range_max=delta.max(axis=0).tolist(), **footprint))
    return result


def self_test():
    audit = Audit()
    z = {"qpos": np.arange(66.).reshape(2, 33)}
    expected = z["qpos"][:, QADR25].copy()
    audit.check("robot_whitelist_exact", np.array_equal(robot25(z), expected))
    z["qpos"][:, 25:] = np.nan
    audit.check("object_columns_never_model_output", np.array_equal(robot25(z), expected))
    summary = error_summary(np.tile(np.arange(25.) / 1000, (3, 1)))
    audit.check("unit_aware_error_no_whitening", np.allclose(summary["rmse"], np.arange(25.) / 1000))
    consumption = task_tolerance_consumption(np.r_[.001, .002, np.zeros(22), .003], .01, (.015, .015))
    audit.check("task_tolerance_order", np.allclose(consumption["fraction_of_existing_tolerance"], [1/15, 2/15, .3]))
    J = np.zeros((3, 25)); J[0, 0] = -2; J[1, 24] = 1
    envelope = point_error_envelope(J, np.r_[.001, np.zeros(23), .003])
    audit.check("geometry_triangle_units", np.allclose(envelope["component_m"], [.002, .003, 0]))
    try:
        point_error_envelope(J, np.full(25, np.nan))
    except ValueError:
        audit.check("unknown_geometry_error_not_zero_gain", True)
    else:
        audit.check("unknown_geometry_error_not_zero_gain", False)
    return audit.checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "calibration/phase3_path_validation_v3")
    parser.add_argument("--raw-context", default="capsule/checkpoint_00")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        checks = self_test(); result = dict(schema="independent_robot_servo_checker_selftest_v1", checks=checks, passed=all(c["passed"] for c in checks), physics_steps=0)
    else:
        audit = Audit(); inventory = read(ROOT / "review/phase3_robot_servo_header_inventory_v1.json")
        audit.check("15_checkpoint_inventory", len(inventory["groups"]) == 15)
        audit.check("inventory_no_missing", not inventory["incomplete_or_missing"])
        audit.check("phase3_plan_identity", sha(args.base / "plan.json") == inventory["plan_sha256"])
        audit.check("phase3_execution_identity", sha(args.base / "execution_complete.json") == inventory["complete_sha256"])
        raw = audit_raw_context(args.base, args.raw_context, audit)
        result = dict(schema="independent_robot_servo_raw_contract_v1", status="raw_contract_only_no_servo_model_inspected", output_axes25=AXES25, units=UNITS25,
            extraction_addresses=QADR25.tolist(), raw_context=raw, checks=audit.checks, passed=all(c["passed"] for c in audit.checks), model_admitted=False, physics_steps=0,
            scope="one preselected checkpoint full raw plus previously recorded 15-group header inventory; future model validation is separate",
            analysis_source_sha256=sha(__file__))
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=result["passed"], checks=len(result["checks"]), failed=[c["name"] for c in result["checks"] if not c["passed"]], physics_steps=0, output=str(args.output)), ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

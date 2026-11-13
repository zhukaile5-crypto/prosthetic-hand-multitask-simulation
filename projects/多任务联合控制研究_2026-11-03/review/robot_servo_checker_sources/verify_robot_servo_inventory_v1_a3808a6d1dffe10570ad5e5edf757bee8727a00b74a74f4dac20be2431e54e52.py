"""Independent all15 robot-servo evidence inventory, no simulation imports."""
import argparse
import json
from pathlib import Path
import numpy as np
from verify_robot_servo_response_v1 import ROOT, Audit, read, sha, AXES25, UNITS25


def audit_inventory(base):
    audit = Audit(); plan = read(base / "plan.json")
    snapshots = read(base / "source_snapshot_manifest.json")
    source_records = []
    for name, record in snapshots["files"].items():
        snapshot = base / record["snapshot_path"]
        actual = sha(snapshot)
        audit.check("historical_execution_source/" + name,
            actual == record["sha256"] == record["snapshot_sha256"] == plan["sources"][name]["sha256"])
        source_records.append(dict(name=name, path=str(snapshot), sha256=actual))
    expected = {f"{shape}/checkpoint_{i:02d}" for shape in ("capsule", "box", "sphere") for i in range(5)}
    found = {str(path.relative_to(base)).replace("\\", "/") for shape in ("capsule", "box", "sphere") for path in (base / shape).glob("checkpoint_*") if path.is_dir()}
    audit.check("15_exact_contexts", found == expected)
    groups = []; totals = dict(planned=0, executed=0, unavailable=0, train=0, mixed=0, zeros=0)
    for context in sorted(expected):
        directory = base / context; checkpoint = read(directory / "checkpoint.json")
        expected_names = {f"micro_{i}_{direction}" for i in range(8) for direction in ("minus", "plus")} | {f"mixed_{i:02d}" for i in range(8)} | {"zero", "zero_replay"}
        actual_names = {p.parent.name for p in directory.glob("*/metadata.json")}
        audit.check(context + "/exact_train_holdout_zero_names", actual_names == expected_names)
        branches = []; directional = {str(i): {} for i in range(8)}
        for name in sorted(expected_names):
            metadata_path = directory / name / "metadata.json"; m = read(metadata_path)
            totals["planned"] += 1
            trace = directory / name / "trace.npz"
            available = bool(m.get("available")); attempted = bool(m.get("attempted_physics"))
            audit.check(context + "/" + name + "/availability_ledger", available == attempted)
            audit.check(context + "/" + name + "/completed_or_unavailable", (attempted and bool(m.get("complete")) and trace.is_file() and not m.get("failure")) or (not attempted and not trace.exists()))
            if attempted:
                totals["executed"] += 1
                role = "train" if name.startswith("micro_") else "mixed" if name.startswith("mixed_") else "zeros"
                totals[role] += 1
                executed = np.asarray(m["executed_path"], float)
                base_input = np.asarray(m["actual_base"], float)
                lower = np.asarray(m["lower"], float); upper = np.asarray(m["upper"], float)
                audit.check(context + "/" + name + "/actual_reference_footprint", executed.shape == (101, 8) and np.isfinite(executed).all() and np.all(executed >= lower - 1e-12) and np.all(executed <= upper + 1e-12))
                record = dict(name=name, role=role, available=True, metadata_sha256=sha(metadata_path), declared_trace_sha256=m["trace_sha256"],
                    actual_increment_min=(executed - base_input).min(axis=0).tolist(), actual_increment_max=(executed - base_input).max(axis=0).tolist(),
                    cumulative_reference_offset_from_handoff=checkpoint["cumulative_reference_offset_from_handoff"])
                if name.startswith("micro_"):
                    _, axis, direction = name.split("_")
                    increment = executed[4] - executed[0]
                    nonzero = np.flatnonzero(abs(increment) > 1e-12)
                    audit.check(context + "/" + name + "/actual_single_axis_direction", np.array_equal(nonzero, [int(axis)]) and (increment[int(axis)] > 0) == (direction == "plus"))
                    directional[axis][direction] = dict(available=True, actual_step=float(increment[int(axis)]))
            else:
                totals["unavailable"] += 1
                record = dict(name=name, available=False, metadata_sha256=sha(metadata_path), declared=m)
                if name.startswith("micro_"):
                    _, axis, direction = name.split("_")
                    directional[axis][direction] = dict(available=False, reason=m)
            branches.append(record)
        zero = directory / "zero/trace.npz"; replay = directory / "zero_replay/trace.npz"
        # Identical NPZ bytes prove all robot/input arrays agree without reading
        # object values into an estimator or pretending a new restore experiment.
        zero_hash = sha(zero); replay_hash = sha(replay)
        audit.check(context + "/historical_zero_replay_bytes_exact", zero_hash == replay_hash)
        for path, digest in ((zero, zero_hash), (replay, replay_hash)):
            audit.check(context + "/" + path.parent.name + "/zero_trace_hash", digest == read(path.parent / "metadata.json")["trace_sha256"])
        groups.append(dict(context_id=context, checkpoint_sha256=sha(directory / "checkpoint.json"), directions=directional,
            zero_trace_sha256=zero_hash, zero_replay_trace_sha256=replay_hash, branches=branches))
    audit.check("390_planned_382_executed_8_unavailable", totals["planned"] == 390 and totals["executed"] == 382 and totals["unavailable"] == 8)
    audit.check("120_mixed_holdouts_all_retained", totals["mixed"] == 120)
    return dict(schema="independent_robot_servo_inventory_v1", passed=all(c["passed"] for c in audit.checks), checks=audit.checks,
        output_axes25=AXES25, output_units=UNITS25, totals=totals, groups=groups, historical_source_snapshots=source_records,
        model_admitted=False, physics_steps=0, scope="all15 metadata/actual reference support + 15 zero-replay byte hashes + historical source snapshots; micro/mixed raw only in separate preselected-context audit",
        analysis_source_sha256=sha(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--base", type=Path, default=ROOT / "calibration/phase3_path_validation_v3"); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = audit_inventory(args.base)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=result["passed"], checks=len(result["checks"]), totals=result["totals"], failures=[c["name"] for c in result["checks"] if not c["passed"]], physics_steps=0), ensure_ascii=False))
    if not result["passed"]:raise SystemExit(1)

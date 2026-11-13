"""Read-only V21 complete no-intervention continuation comparator.

This compares saved behavior. It never constructs a session, runs a model, or
certifies that either archive was produced by a native checkpoint fork.
"""

import argparse
from copy import deepcopy
import gzip
from itertools import zip_longest
import json
import math
from pathlib import Path

import numpy as np

from v18_passive_trace_comparison_v3 import compare_arrays, sha

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ORIGIN_S = 9.280
ARCHIVE_INDEX = HERE / "v23_original_V21_full16_bridges_v1.json"
ARCHIVE_INDEX_SHA = "7d00df8e10dac18e0d5bd3ef8f3e89542fc189c0d075e3ef6bee10cf1df9812a"
TRACE_HELPER = HERE / "v18_passive_trace_comparison_v3.py"
TRACE_HELPER_SHA = "fbbdabefaa0f7c9a893d535b2aab8fe0520395d58851aeb2e032f70e23bb3d0b"
CONTROL_CLOCK = np.arange(3201) * .005
PHYSICS_CLOCK = np.arange(32000) * .0005
TIMING_PATHS = {
    ("solver_ms",),
    ("hierarchy", "diagnostics", "first_stage_diagnostics", "degenerate_face_elapsed_ms"),
    ("hierarchy", "diagnostics", "second_stage_diagnostics", "degenerate_face_elapsed_ms"),
    ("hierarchy", "levels", "[]", "degenerate_face_repairs", "[]", "elapsed_ms"),
    ("hierarchy", "certified_prefix", "[]", "record", "degenerate_face_repairs", "[]", "elapsed_ms"),
    ("hierarchy", "failed_layer", "record", "degenerate_face_repairs", "[]", "elapsed_ms"),
    ("hierarchy", "diagnostics", "failed_layer", "record", "degenerate_face_repairs", "[]", "elapsed_ms"),
    ("hierarchy", "diagnostics", "second_stage_diagnostics", "failed_layer", "record", "degenerate_face_repairs", "[]", "elapsed_ms"),
}
METADATA_KEYS = (
    "method", "scenario", "seed", "shape", "mode", "split", "config", "noise",
    "limits", "budget", "duration", "sensor_period", "physics_timestep",
    "initial_position", "model_sha256", "source_sha256", "handoff_passed",
    "handoff_alpha", "motion_mode", "task_policy", "receiver_policy",
    "release_variant", "release_at", "mechanical_release_at", "commands", "events",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def strip_release_walltime(value, path=()):
    if path in TIMING_PATHS:
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid declared wall time at " + ".".join(path))
        return "<validated wall time>"
    if isinstance(value, dict):
        return {key: strip_release_walltime(item, path + (key,)) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_release_walltime(item, path + ("[]",)) for item in value]
    return value


def clock_matches(actual, expected, *, tolerance=0.):
    return bool(actual.shape == expected.shape and np.allclose(
        actual, expected, rtol=0., atol=tolerance, equal_nan=False))


def verify_file(path, digest):
    actual = sha(path)
    if actual != digest:
        raise ValueError(f"Evidence hash mismatch for {path}")
    return actual


def verify_trace_clock(data):
    return (
        clock_matches(data["time"], CONTROL_CLOCK)
        and clock_matches(data["robot_control_time_s"], CONTROL_CLOCK)
        and clock_matches(data["substep_time"], PHYSICS_CLOCK, tolerance=1e-9)
        and clock_matches(data["substep_duration"], np.full(32000, .0005))
    )


def compare_decisions(left, right, kind):
    stem = kind + "_decision"
    name = stem + "s.jsonl.gz"
    manifests = [read(folder / (stem + "_manifest.json")) for folder in (left, right)]
    for folder, manifest in zip((left, right), manifests):
        if manifest["file"] != name:
            raise ValueError("Unexpected decision evidence file")
        verify_file(folder / name, manifest["sha256"])
    count = 0
    first_difference = None
    times = []
    with gzip.open(left / name, "rt", encoding="utf-8") as a, gzip.open(right / name, "rt", encoding="utf-8") as b:
        for i, (x, y) in enumerate(zip_longest(a, b)):
            count += 1
            if x is None or y is None:
                first_difference = i if first_difference is None else first_difference
                continue
            x, y = json.loads(x), json.loads(y)
            times.append(x["time_s"])
            if kind == "release":
                x, y = strip_release_walltime(x), strip_release_walltime(y)
            equal = x["decision_id"] == y["decision_id"] == i and canonical(x) == canonical(y)
            if not equal and first_difference is None:
                first_difference = i
    t = np.asarray(times, dtype=float)
    if kind == "joint":
        clock_ok = (t.size > 0 and t[0] == 7.5 and t[-1] < 12.
                    and np.isfinite(t).all() and (np.diff(t) > 0.).all())
    else:
        clock_ok = clock_matches(t, CONTROL_CLOCK[2400:])
    passed = first_difference is None and count > 0 and clock_ok and all(
        count == m["records"] for m in manifests)
    return dict(passed=bool(passed), records=count, first_different_row=first_difference,
                clock_valid=bool(clock_ok), sha256=[m["sha256"] for m in manifests])


def compare_delivery(left, right):
    manifests = [read(folder / "delivery_manifest.json") for folder in (left, right)]
    for folder, manifest in zip((left, right), manifests):
        for key, filename in (("arrays", "delivery_intervals.npz"), ("summary", "delivery_summary.json")):
            if manifest[key]["file"] != filename:
                raise ValueError("Unexpected delivery evidence file")
            verify_file(folder / filename, manifest[key]["sha256"])
    with np.load(left / "delivery_intervals.npz", allow_pickle=False) as a, np.load(
            right / "delivery_intervals.npz", allow_pickle=False) as b:
        if "metadata" in a.files or "metadata" in b.files:
            raise ValueError("Delivery ledger does not carry a provenance-only column")
        comparison = compare_arrays(a, b)
        clocks_ok = True
        for z in (a, b):
            start, end = z["interval_start_s"], z["interval_end_s"]
            clocks_ok = (clocks_ok and clock_matches(start, PHYSICS_CLOCK, tolerance=1e-9)
                         and clock_matches(end, PHYSICS_CLOCK + .0005, tolerance=1e-9)
                         and np.array_equal(start[1:], end[:-1])
                         and np.array_equal(z["interval_duration_s"], end - start))
    summary_exact = canonical(read(left / "delivery_summary.json")) == canonical(read(right / "delivery_summary.json"))
    return dict(passed=comparison["passed"] and clocks_ok and summary_exact,
                clocks_valid=clocks_ok, intervals=32000 if clocks_ok else None,
                summary_exact=summary_exact, checks=comparison["checks"],
                sha256=[m["arrays"]["sha256"] for m in manifests])


def compare_saved_continuations(left, right):
    if left.resolve() == right.resolve():
        raise ValueError("Two distinct saved runs are required")
    metadata = [read(folder / "metadata.json") for folder in (left, right)]
    for folder, record in zip((left, right), metadata):
        verify_file(folder / "trace.npz", record["trace_sha256"])
        if not (record["completed"] is True and record["duration"] == 16.
                and record["sensor_period"] == .005 and record["physics_timestep"] == .0005
                and record["scenario"]["name"] == "fragile_conflict"
                and record["handoff_passed"] is True and record["method"] == "ratio_effort"
                and record["release_variant"] == "v21" and record["task_policy"] == "integrated"
                and record["receiver_policy"] == "coordinated" and record["mode"] == "scheduled"):
            raise ValueError("Expected a complete V21 fragile holding run")
    meta_equal = all(canonical(metadata[0][k]) == canonical(metadata[1][k]) for k in METADATA_KEYS)
    with np.load(left / "trace.npz", allow_pickle=False) as a, np.load(right / "trace.npz", allow_pickle=False) as b:
        clocks_ok = verify_trace_clock(a) and verify_trace_clock(b)
        trace = compare_arrays(a, b)
    joint = compare_decisions(left, right, "joint")
    release = compare_decisions(left, right, "release")
    delivery = compare_delivery(left, right)
    return dict(passed=meta_equal and clocks_ok and all(x["passed"] for x in (trace, joint, release, delivery)),
                context=["fragile_conflict", metadata[0]["motion_mode"], metadata[0]["seed"]],
                input_directories=[str(left), str(right)], metadata_equal=meta_equal,
                metadata_fields=list(METADATA_KEYS), full_trace_clock_valid=clocks_ok,
                trace_sha256=[m["trace_sha256"] for m in metadata], trace=trace,
                joint=joint, release=release, delivery=delivery)


def historical_inputs():
    verify_file(ARCHIVE_INDEX, ARCHIVE_INDEX_SHA)
    rows = read(ARCHIVE_INDEX)["rows"]
    row, = (r for r in rows if r["context_key"] == ["fragile_conflict", "hold", 619741])
    a, b = ROOT / row["original"], ROOT / row["actual"]
    verify_file(a / "trace.npz", row["original_trace_sha256"])
    verify_file(b / "trace.npz", row["actual_trace_sha256"])
    return a, b


def negative_controls(left):
    with np.load(left / "trace.npz", allow_pickle=False) as z:
        minimal = {k: z[k] for k in ("time", "robot_control_time_s", "substep_time", "substep_duration")}
    clock = minimal["robot_control_time_s"]
    clock[1856] = clock[1855]
    with gzip.open(left / "release_decisions.jsonl.gz", "rt", encoding="utf-8") as f:
        original = next(row for row in map(json.loads, f) if row.get("execution_receipt"))
    numeric = deepcopy(original)
    numeric["execution_receipt"]["actual_reference21"][0] += .001
    wrong_time = deepcopy(original)
    wrong_time["solver_ms"] = -1.
    try:
        strip_release_walltime(wrong_time)
    except ValueError:
        invalid_walltime_rejected = True
    else:
        invalid_walltime_rejected = False
    return dict(duplicate_control_tick_rejected=not verify_trace_clock(minimal),
                changed_executed_reference_rejected=canonical(strip_release_walltime(original))
                != canonical(strip_release_walltime(numeric)),
                invalid_declared_walltime_rejected=invalid_walltime_rejected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.historical and (args.left or args.right) or not args.historical and not (args.left and args.right):
        parser.error("Use --historical or supply both --left and --right")
    verify_file(TRACE_HELPER, TRACE_HELPER_SHA)
    a, b = historical_inputs() if args.historical else (args.left, args.right)
    result = compare_saved_continuations(a, b)
    negatives = negative_controls(a) if args.historical else {}
    passed = result["passed"] and all(negatives.values())
    report = dict(schema="v21_fragile_hold_continuation_comparator_v1",
                  passed=passed, comparison=result, historical_fixture=args.historical,
                  negative_controls=negatives, requested_future_origin_s=ORIGIN_S,
                  native_fork_validated=False, physics_steps=0,
                  source_sha256={Path(__file__).name: sha(__file__), TRACE_HELPER.name: TRACE_HELPER_SHA},
                  release_decision_walltime_paths=[".".join(p) for p in sorted(TIMING_PATHS)])
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(passed=passed, trace_fields=len(result["trace"]["checks"]),
                          joint_records=result["joint"]["records"], release_records=result["release"]["records"],
                          intervals=result["delivery"]["intervals"], negatives=negatives), ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

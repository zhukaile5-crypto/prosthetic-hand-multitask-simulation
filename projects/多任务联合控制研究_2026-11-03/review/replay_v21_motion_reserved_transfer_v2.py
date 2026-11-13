"""Same 64 frozen snapshots, V1 reference versus motion-reserved V2."""

import argparse
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import replay_v21_support_conditioned_exit_v1 as archive
from identification import v21_support_conditioned_exit_v1 as v1
from identification import v21_motion_reserved_transfer_v2 as v2

PRIOR = ROOT / "review/v21_support_conditioned_exit_snapshot_v1.json"
PRIOR_SHA = "7a2acb5a9e5bf0b38a1e2e9f1703db398d87199c288d65707401fa32572acfc6"
CHECK_OUTWARD_M = 1e-10
CHECK_MOTION = 1e-8
CHECK_EMITTED_MOTION = 1e-9


def verify_emitted(problem, row, diag, solved):
    """Test typed-reference arithmetic before any hypothetical actor call."""
    current = np.asarray(row["current_reference21"])
    units = np.asarray(diag["reference_step_units21"])
    a, b, u = (np.r_[np.asarray(solved[k]), problem.lower[21:]] for k in (
        "correction_reference_normalized", "stability_reference_normalized", "final_reference_normalized"))
    raw = current + u[:21] * units
    lower, upper = np.asarray(row["actual_reference_lower21"]), np.asarray(row["actual_reference_upper21"])
    emitted = np.minimum(upper, np.maximum(lower, raw))
    rounding = 64 * np.finfo(float).eps * np.maximum(1., np.maximum(abs(raw), abs(emitted)))
    if np.any(abs(emitted - raw) > rounding):
        raise ValueError("Reconstructed reference requires more than frozen rounding projection")
    physical_delta = emitted - current
    normalized = np.r_[physical_delta / units, problem.lower[21:]]
    if (problem.violation(normalized) > CHECK_MOTION
            or np.max(problem.normal @ (a - normalized), initial=0.) > CHECK_MOTION
            or np.max(problem.normal @ (b - normalized), initial=0.) > CHECK_MOTION):
        raise ValueError("Reconstructed reference violates the original or retention domain")
    cap = problem.normal @ normalized
    cap_matrix, cap_rhs = v1.retained_domain(problem, b, solved["fraction"])
    if np.max(cap_matrix @ normalized - cap_rhs, initial=0.) > CHECK_MOTION:
        raise ValueError("Reconstructed reference violates the per-contact pace")
    support_outward = float(problem.work_row @ (normalized - b)[:21])
    ceiling = solved["fraction"] * solved["optional_capacity_proxy_m"]
    if support_outward > ceiling + CHECK_OUTWARD_M:
        raise ValueError("Reconstructed reference violates the overall pace")
    protected = np.asarray(solved["protected_motion_vector"])
    motion_drift = float(np.max(abs(problem.motion_level.matrix @ normalized - protected), initial=0.))
    if motion_drift > CHECK_EMITTED_MOTION:
        raise ValueError("Reconstructed reference changed the protected motion vector")
    raw_normal_violation = np.asarray(diag["normal_matrix_N_Px21"]) @ physical_delta - diag["normal_one_sided_bound_P_m"]
    normal_units = np.asarray(diag["normal_effective_report_units_P_m"])
    if np.max(raw_normal_violation / normal_units, initial=0.) > CHECK_MOTION:
        raise ValueError("Reconstructed reference violates the original physical-unit normal row")
    motion = diag["motion_task"]
    raw_motion_debt = np.asarray(motion["matrix_Mx21"]) @ physical_delta + diag["dt_s"] * np.asarray(motion["error_M_m_s"])
    motion_units = np.asarray(motion["effective_row_units_M"])
    raw_motion_drift = raw_motion_debt - (protected - problem.motion_level.target) * motion_units
    if np.max(abs(raw_motion_drift / motion_units), initial=0.) > CHECK_EMITTED_MOTION:
        raise ValueError("Reconstructed reference changed the physical-unit motion task")
    return dict(reference21=emitted.tolist(), original_domain_error=problem.violation(normalized),
                maximum_reference_rounding_projection=float(np.max(abs(emitted - raw))),
                motion_vector_drift=motion_drift, maximum_motion_debt_drift_m=float(np.max(abs(raw_motion_drift))),
                maximum_raw_normal_violation_m=float(np.max(raw_normal_violation, initial=0.)),
                minimum_outward_from_stability=float(np.min(cap - problem.normal @ b)),
                optional_withdrawal_proxy_m=support_outward)


def replay(case, meta, prior):
    output = []
    for label, reading, record, executed, digest in archive.inputs(case, meta):
        key = (tuple(case["context"]), label)
        old = prior[key]
        if (old["phase"] != record["phase"] or old["time_s"] != reading["time_s"]
                or old["release_log_sha256"] != digest
                or old["sustained_fraction"] != reading["sustained_proxy_fraction"]
                or not np.array_equal(executed, record["execution_receipt"]["actual_reference21"])):
            raise ValueError("Motion-reserved replay does not match the frozen reference arm")
        info = dict(context=case["context"], label=label, time_s=record["time_s"], release_log_sha256=digest)
        if old["status"] != "solved":
            output.append(dict(info, status="scope_unavailable_in_original_pair"))
            continue
        diag = record["hierarchy"]["diagnostics"]
        problem = v1.snapshot(diag)
        new = v2.solve_reserved_transfer(problem, reading["sustained_proxy_fraction"])
        if new["status"] != "solved":
            output.append(dict(info, status=new["status"], result=new))
            continue
        verify = verify_emitted(problem, record, diag, new)
        ref = old["solve"]
        before = ref["uncapped"]
        capped = ref["support_conditioned"]
        output.append(dict(info, status="solved", result=new, emitted_reference_check=verify,
                           v1_uncapped_outward_proxy_m=before["optional_work_m"],
                           v1_capped_outward_proxy_m=capped["optional_work_m"],
                           v1_uncapped_motion_residual=before["motion_residual_norm"],
                           v1_capped_motion_residual=capped["motion_residual_norm"],
                           v1_motion_degraded=ref["motion_only_diagnostic"] is not None,
                           reference_difference_linf_vs_v1_capped=float(np.max(abs(
                               np.asarray(new["final_reference_normalized"]) - capped["normalized_reference"])))))
    return output


def directions(values, tol):
    x = np.asarray(values)
    return dict(lower=int((x < -tol).sum()), unchanged=int((abs(x) <= tol).sum()),
                higher=int((x > tol).sum()), median=None if not len(x) else float(np.median(x)))


def summarize(rows):
    done = [r for r in rows if r["status"] == "solved"]
    old_degraded = [r for r in done if r["v1_motion_degraded"]]
    return dict(assigned=len(rows), solved=len(done),
                complete_stability_witness=sum(r["result"]["canonical_stability_witness_certified"] for r in done),
                complete_optional_tasks=sum(r["result"]["full_optional_tasks"] for r in done),
                protected_motion_vs_v1_uncapped=directions([
                    r["result"]["final_motion_residual_norm"] - r["v1_uncapped_motion_residual"] for r in done], CHECK_MOTION),
                recovered_v1_degraded_frames=sum(r["result"]["final_motion_residual_norm"]
                    <= r["v1_uncapped_motion_residual"] + CHECK_MOTION for r in old_degraded),
                v1_degraded_frames=len(old_degraded),
                nonzero_required_withdrawal=sum(r["result"]["needed_withdrawal_proxy_m"] > CHECK_OUTWARD_M for r in done),
                nonzero_optional_capacity=sum(r["result"]["optional_capacity_proxy_m"] > CHECK_OUTWARD_M for r in done),
                total_withdrawal_vs_v1_capped=directions([
                    r["result"]["total_beyond_correction_proxy_m"] - r["v1_capped_outward_proxy_m"] for r in done],
                    CHECK_OUTWARD_M),
                total_withdrawal_vs_v1_uncapped=directions([
                    r["result"]["total_beyond_correction_proxy_m"] - r["v1_uncapped_outward_proxy_m"] for r in done],
                    CHECK_OUTWARD_M),
                max_final_motion_drift=max((r["emitted_reference_check"]["motion_vector_drift"] for r in done), default=None),
                changed_reference_vs_v1_capped=sum(r["reference_difference_linf_vs_v1_capped"] > CHECK_MOTION for r in done))


def self_check():
    n = 23
    lower, upper = np.r_[np.full(21, -1.), 0., 0.], np.r_[np.ones(21), 0., 0.]
    normal = np.eye(n)[:2]
    work = np.r_[.0001, .0001, np.zeros(19)]
    motion = v1.Level("observed_stability", normal[:1], np.array([.8]))
    objective = v1.Level("withdrawal", normal.sum(axis=0, keepdims=True), np.array([2.]))
    posture = v1.Level("posture", np.eye(n)[:21], np.zeros(21))
    problem = v1.Snapshot(lower, upper, np.empty((0, n)), np.empty(0), np.zeros(n),
                          normal, work, (objective, motion, posture), motion,
                          dict(atol=1e-9, rtol=1e-8, max_iterations=400))
    old = v1.solve_pair(problem, None)
    new = v2.solve_reserved_transfer(problem, None)
    assert old["support_conditioned"]["motion_residual_norm"] > .5
    assert new["status"] == "solved" and new["full_optional_tasks"]
    assert new["final_motion_residual_norm"] < 1e-9
    assert abs(new["needed_withdrawal_proxy_m"] - .00008) < 1e-12
    assert abs(new["optional_withdrawal_proxy_m"] - .000025) < 1e-12
    return ["motion_requires_outward_reference", "required_withdrawal_exempt_from_cap",
            "unconfirmed_support_caps_only_optional_movement"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checks = self_check()
    old = archive.read(archive.verify(PRIOR, PRIOR_SHA))
    prior = {(tuple(x["context"]), x["label"]):x for x in old["rows"]}
    assert len(prior) == 64 and old["physics_steps"] == 0
    meta = archive.read(archive.verify(archive.AUX, archive.AUX_SHA))
    source = archive.read(archive.verify(archive.SUPPORT, archive.SUPPORT_SHA))
    lookup = {tuple(x["context_key"]):x for x in meta["records"]}
    rows = []
    for case in source["cases"]:
        rows.extend(replay(case, lookup[tuple(case["context"])], prior))
    assert len(rows) == 64 and {(tuple(r["context"]), r["label"]) for r in rows} == set(prior)
    out = dict(schema="v21_motion_reserved_transfer_snapshot_v2", physics_steps=0,
               plant_instances=0, self_checks=checks, summary=summarize(rows),
               by_anchor={k:summarize([r for r in rows if r["label"] == k]) for k in archive.ANCHORS},
               source_sha256={PRIOR.name:PRIOR_SHA, archive.AUX.name:archive.AUX_SHA,
                 archive.SUPPORT.name:archive.SUPPORT_SHA,
                 "v23_task_execution/finite_solver.py":archive.sha(ROOT / "v23_task_execution/finite_solver.py"),
                 **{Path(p).name:archive.sha(Path(p)) for p in (__file__, archive.__file__, v1.__file__, v2.__file__)}},
               rows=rows)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()

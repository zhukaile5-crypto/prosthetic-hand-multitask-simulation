"""Re-solve archived V21 decisions in two matched post-correction domains."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from identification import v21_support_conditioned_exit_v1 as candidate

SUPPORT = ROOT / "review/v21_receiver_transfer_evidence_archive_v1.json"
SUPPORT_SHA = "0ae50080e60c4f450594ec73dae3044c0764506011013d46bbf5cf4a3c34dce7"
AUX = ROOT / "review/v21_complete16_auxiliary_v1.json"
AUX_SHA = "b894592f38a8ca199088fd41bda9331ab819486b995b7db08d35e898be12d933"
ANCHORS = ("first_transfer", "last_before_support", "first_sustained_support", "support_plus_20ms")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(path, digest):
    if sha(path) != digest:
        raise ValueError(f"Frozen data changed: {path}")
    return path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def tick(t):
    k = round(t / candidate.CONTROL_S)
    if abs(t - k * candidate.CONTROL_S) > 1e-8:
        raise ValueError("Archived decision is off the public clock")
    return k


def inputs(case, meta):
    k_first = tick(case["first_transfer_s"])
    k_support = tick(case["first_sustained_s"])
    ticks = dict(zip(ANCHORS, (k_first, k_support - 1, k_support, k_support + 4)))
    if len(set(ticks.values())) != len(ANCHORS):
        raise ValueError("Expected four distinct predeclared replay times")
    support = {tick(r["time_s"]):r for r in case["rows"]}
    folder = Path(meta["run"]).resolve()
    if not folder.is_relative_to((ROOT / "output").resolve()):
        raise ValueError("Invalid frozen archive path")
    manifest = read(folder / "release_decision_manifest.json")
    if manifest["file"] != "release_decisions.jsonl.gz" or manifest["records"] != 801:
        raise ValueError("Expected the full terminal release decision ledger")
    log = verify(folder / manifest["file"], manifest["sha256"])
    decisions = {}
    with gzip.open(log, "rt", encoding="utf-8") as f:
        for k, row in enumerate(map(json.loads, f), start=2400):
            if k != tick(row["time_s"]):
                raise ValueError("Decision ordering changed")
            if k in ticks.values():
                decisions[k] = row
    if k != 3200 or len(decisions) != len(ANCHORS):
        raise ValueError("Archived release decisions incomplete")
    digest = meta["input_sha256"]["trace.npz"]
    if digest != case["trace_sha256"]:
        raise ValueError("Evidence and decision archive differ")
    trace = verify(folder / "trace.npz", digest)
    with np.load(trace, allow_pickle=False) as z:
        actual = z["executed_reference21"]
    for label, k in ticks.items():
        yield label, support[k], decisions[k], actual[k], manifest["sha256"]


def replay(case, meta):
    out = []
    for label, evidence, d, native, digest in inputs(case, meta):
        if (d["phase"] != evidence["phase"] or d["phase"] != d["public_process"]["phase"]
                or d["time_s"] != evidence["time_s"]):
            raise ValueError("Paired observations are not from the same archived decision")
        receipt = d["execution_receipt"]
        if not np.array_equal(native, receipt["actual_reference21"]):
            raise ValueError("Frozen executed reference differs between decision and trace")
        info = dict(context=case["context"], label=label, time_s=d["time_s"], phase=d["phase"],
                    release_log_sha256=digest, sustained_fraction=evidence["sustained_proxy_fraction"],
                    original_status=None if d["hierarchy"] is None else d["hierarchy"]["status"])
        d2 = None if d["hierarchy"] is None else d["hierarchy"]["diagnostics"]
        if d["phase"] != "TRANSFER" or not d2 or "second_stage_hard_matrix" not in d2:
            out.append(dict(info, status="no_scoped_post_correction_domain"))
            continue
        if d2["time_s"] != d["time_s"] or d2["dt_s"] != candidate.CONTROL_S:
            raise ValueError("Frozen solver geometry from another tick")
        problem = candidate.snapshot(d2)
        if not len(problem.normal):
            out.append(dict(info, status="no_contact_row_authority"))
            continue
        pair = candidate.solve_pair(problem, info["sustained_fraction"])
        if "support_conditioned" not in pair:
            reason = "canonical_witness_unavailable" if pair["anchor"] is None else "uncapped_capacity_unavailable"
            out.append(dict(info, status=reason, solve=pair))
            continue
        units = np.asarray(d2["reference_step_units21"])
        archived_delta = (native - d["current_reference21"]) / units
        archived_outward = problem.normal[:, :candidate.AXES] @ (archived_delta - pair["anchor"])
        if (np.any(native < np.asarray(d["actual_reference_lower21"]) - 1e-12)
                or np.any(native > np.asarray(d["actual_reference_upper21"]) + 1e-12)):
            raise ValueError("Frozen output escaped its logged authority")
        out.append(dict(info, status="solved", solve=pair,
                        archived_outward_proxy=float(np.max(archived_outward, initial=0.)),
                        archived_any_inward_relative_to_witness=bool(np.any(archived_outward < -1e-8))))
    return out


def summarize(rows, comparison="support_conditioned"):
    solved = [r for r in rows if r["status"] == "solved"]
    paired = [r["solve"] for r in solved if r["solve"]["uncapped"]["candidate_available"]
              and r["solve"][comparison]["candidate_available"]]
    motion_cert = "public_external_projection_low_motion"
    motion = [r for r in paired if all(motion_cert in r[a]["certified_task_names"]
                                     for a in ("uncapped", comparison))]
    delta = np.array([r[comparison]["optional_work_m"] - r["uncapped"]["optional_work_m"]
                      for r in paired])
    motion_delta = np.array([r[comparison]["motion_residual_norm"] - r["uncapped"]["motion_residual_norm"]
                             for r in motion])
    return dict(assigned=len(rows), solved_with_witness=len(solved), paired_certified_withdrawal=len(paired),
                uncapped_full=sum(r["solve"]["uncapped"]["full"] for r in solved),
                comparison_full=sum(r["solve"][comparison]["full"] for r in solved),
                fewer_outward_geometry_m=int(np.sum(delta < -1e-10)), same_outward_geometry_m=int(np.sum(abs(delta) <= 1e-10)),
                greater_outward_geometry_m=int(np.sum(delta > 1e-10)),
                median_delta_outward_geometry_um=None if not len(delta) else float(1e6*np.median(delta)),
                jointly_certified_motion=len(motion),
                less_motion_proxy_residual=int(np.sum(motion_delta < -1e-8)),
                greater_motion_proxy_residual=int(np.sum(motion_delta > 1e-8)),
                median_delta_normalized_motion_residual=None if not len(motion) else float(np.median(motion_delta)),
                archived_inward_relative_to_new_witness=sum(r["archived_any_inward_relative_to_witness"] for r in solved))


def self_check():
    size = 23
    lower, upper = np.r_[np.full(21, -1.), 0., 0.], np.r_[np.ones(21), 0., 0.]
    n = np.eye(size)[:2]
    work = np.zeros(21)
    work[:2] = .0001
    proto = candidate.Snapshot(lower, upper, np.zeros((0, size)), np.zeros(0), np.zeros(size),
        n, work, (candidate.Level("work", n.sum(axis=0)[None, :], np.array([2.])),),
        None, dict(atol=1e-9, rtol=1e-8, max_iterations=400))
    a = candidate.solve_pair(proto, None)
    b = candidate.solve_pair(proto, 1.)
    assert a["uncapped"]["candidate_available"] and a["support_conditioned"]["candidate_available"]
    assert abs(a["uncapped"]["optional_work_m"] - .0002) < 1e-12
    assert abs(a["support_conditioned"]["optional_work_m"] - .00005) < 1e-12
    assert a["support_conditioned"]["maximum_original_and_new_domain_error"] < 1e-9
    assert b["uncapped"] == b["support_conditioned"]
    return ["uncertainty_retains_quarter_rate", "cap_limits_each_outward_row", "full_support_matches_uncapped"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checks = self_check()
    source = read(verify(SUPPORT, SUPPORT_SHA))
    aux = read(verify(AUX, AUX_SHA))
    lookup = {tuple(x["context_key"]):x for x in aux["records"]}
    if len(lookup) != 16 or len(source["cases"]) != 16 or source["physics_steps"] != 0:
        raise ValueError("Expected all matched V21 evidence cases")
    rows = []
    for case in source["cases"]:
        rows.extend(replay(case, lookup[tuple(case["context"])]))
    assert len(rows) == len({(tuple(r["context"]), r["label"]) for r in rows}) == 64
    motion_probes = [r["solve"]["motion_only_diagnostic"] for r in rows
                     if r["status"] == "solved" and r["solve"]["motion_only_diagnostic"] is not None]
    out = dict(schema="v21_support_conditioned_exit_snapshot_v1", physics_steps=0, simulated_outcomes=False,
               selected_time_anchors=list(ANCHORS), checks=checks, summary=summarize(rows),
               motion_only_diagnostics=dict(attempted=len(motion_probes),
                   certified=sum(p["certified"] for p in motion_probes),
                   cannot_match_uncapped=sum(p["still_above_uncapped"] is True for p in motion_probes)),
               by_anchor={key:summarize([r for r in rows if r["label"] == key]) for key in ANCHORS},
               mechanical_only_summary=summarize(rows, "mechanical_only"),
               mechanical_only_by_anchor={key:summarize([r for r in rows if r["label"] == key], "mechanical_only") for key in ANCHORS},
               source_sha256={SUPPORT.name:SUPPORT_SHA, AUX.name:AUX_SHA, Path(__file__).name:sha(Path(__file__)),
                   Path(candidate.__file__).name:sha(Path(candidate.__file__)),
                   "v23_task_execution/finite_solver.py":sha(ROOT / "v23_task_execution/finite_solver.py")},
               rows=rows)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()

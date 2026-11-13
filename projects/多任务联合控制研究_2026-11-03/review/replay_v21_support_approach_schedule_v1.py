"""Reconstruct the frozen SEARCH governor and compare preview-shaped commands."""

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import replay_v21_public_support_preview_v1 as preview_replay
from identification import v21_support_approach_schedule_v1 as schedule
from identification.v21_public_support_preview_v1 import SupportPreview
from foundation_config import FoundationConfig
from hand_plant import _model
from v14_coord.design import reference_rates
from v18_effort_transfer.governor import LiftProgressGovernor
from v18_effort_transfer.supervisor import governor_task_interval

PREVIEW = ROOT / "review/v21_public_support_preview_archive_v1.json"
PREVIEW_SHA = "5228711fc2da64e1a775b831826459d19c53fbca4ed8abe4858333fec3e82635"
CONTROL = .005
RATE = float(reference_rates("release")[18])
DEADLINE_AFTER_RELEASE = 3.5
TOL = 1e-12


def window(log, expected):
    manifest = preview_replay.read(log / "release_decision_manifest.json")
    if manifest["file"] != "release_decisions.jsonl.gz" or manifest["records"] != 801:
        raise ValueError("Expected a complete frozen release ledger")
    path = preview_replay.verify(log / manifest["file"], manifest["sha256"])
    if expected is not None and manifest["sha256"] != expected:
        raise ValueError("Unexpected frozen release log identity")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for r in map(json.loads, f):
            if r["phase"] != "SEARCH":
                break
            rows.append(r)
    if not rows:
        raise ValueError("Frozen run had no public SEARCH phase")
    return rows, manifest["sha256"]


def actual_reference(governed, q, v, lower, upper):
    if not governed.available:
        raise ValueError("Frozen governor could not form a legal reference")
    lo, hi, _ = governor_task_interval(governed, q_m=q, v_m_s=v,
                                       actual_lower_m=lower, actual_upper_m=upper)
    return float(np.clip(governed.reference_m, lo, hi))


def run_case(case, meta, mounted, expected_preview):
    archived = case["metrics"]["full_V21"]
    trace = preview_replay.verify(ROOT / archived["trace"], archived["trace_sha256"])
    if archived["trace_sha256"] != meta["input_sha256"]["trace.npz"]:
        raise ValueError("Frozen case trace differs between indices")
    folder = trace.parent
    record = preview_replay.read(preview_replay.verify(folder / "metadata.json", meta["input_sha256"]["metadata.json"]))
    config = {k:tuple(v) if isinstance(v, list) else v for k,v in record["config"].items()}
    model, model_sha = _model(FoundationConfig(**config))
    if model_sha != record["model_sha256"]:
        raise ValueError("Static lift model identity differs from archived run")
    gov = LiftProgressGovernor(model)
    public = SupportPreview(mounted, preview_replay.TEMPLATE)
    policy = schedule.ApproachSchedule()
    decisions, log_sha = window(folder, meta["input_sha256"].get("release_decisions.jsonl.gz"))
    old_preview = {r["read_s"]:r for r in expected_preview["precontact_rows"]}
    seq = []
    with np.load(trace, allow_pickle=False) as z:
        observation = z["observation_json"][2400:2400+len(decisions)]
        saved_refs = z["executed_reference21"][2400:2400+len(decisions)]
    for i, (d, packet, executed) in enumerate(zip(decisions, observation, saved_refs, strict=True)):
        now = (2400+i)*CONTROL
        frame = json.loads(packet)
        q, v = frame["robot_position25"][24], frame["robot_velocity25"][24]
        current = frame["robot_reference21"][18]
        if (frame["time"] != d["time_s"] or d["time_s"] != now
                or d["governor"]["q_m"] != q or d["governor"]["actual_velocity_m_s"] != v
                or d["governor"]["committed_reference_m"] != current):
            raise ValueError("Frozen current lift, velocity or command clock does not match")
        lower, upper = frame["reference_lower21"][18], frame["reference_upper21"][18]
        authority_lo = max(lower, current-RATE*CONTROL)
        authority_hi = min(upper, current+RATE*CONTROL)
        if (abs(authority_lo - d["original_mechanical_rate_lower21"][18]) > TOL
                or abs(authority_hi - d["original_mechanical_rate_upper21"][18]) > TOL):
            raise ValueError("Frozen original step domain does not reconstruct")
        original_request = d["governor"]["requested_velocity_m_s"]
        deadline = d["public_process"]["started_s"] + DEADLINE_AFTER_RELEASE
        common = dict(time_s=now, q_m=q, v_m_s=v, current_reference_m=current,
                      reference_lower_m=lower, reference_upper_m=upper,
                      reference_rate_lower_m_s=-RATE, reference_rate_upper_m_s=RATE,
                      deadline_s=deadline)
        old = gov.step(**common, requested_velocity_m_s=original_request)
        old_ref = actual_reference(old, q, v, authority_lo, authority_hi)
        if (old.reference_m != d["governor"]["emitted_reference_m"]
                or abs(old_ref - executed[18]) > TOL or abs(old_ref - d["proposed_reference21"][18]) > TOL
                or old.diagnostics["reason"] != d["governor"]["reason"]):
            raise ValueError("Cannot reproduce the original same-tick SEARCH action")
        preview = public.read(frame)
        frozen = old_preview.get(now)
        if frozen is not None and (preview["available"] != frozen["available"] or
                preview.get("engineering_gap_m") != frozen.get("engineering_gap_m")):
            raise ValueError("Preview differs from the earlier frozen public geometry")
        requested = policy.update(now=now, phase="SEARCH", preview=preview,
            original_request_m_s=original_request,
            velocity_envelope_m_s=old.diagnostics["source_derived_velocity_envelope_m_s"],
            deadline_s=deadline)
        new = gov.step(**common, requested_velocity_m_s=requested["requested_velocity_m_s"])
        reference = actual_reference(new, q, v, authority_lo, authority_hi)
        if not authority_lo <= reference <= authority_hi or reference < old_ref - TOL:
            raise ValueError("New same-tick reference exceeded authority or became more downward")
        seq.append(dict(time_s=now, q_m=q, actual_v_m_s=v, original_reference_m=old_ref,
                        new_reference_m=reference, reference_delta_m=reference-old_ref,
                        original_equivalent_velocity_m_s=(old_ref-q)/gov.tau_s,
                        new_equivalent_velocity_m_s=(reference-q)/gov.tau_s,
                        original_fixed_q_debt_m=q-old_ref, new_fixed_q_debt_m=q-reference,
                        original_native_effort_N=old.diagnostics["current_native_effort_of_emitted_reference"]["effort_N"],
                        new_native_effort_N=new.diagnostics["current_native_effort_of_emitted_reference"]["effort_N"],
                        old_governor_reason=old.diagnostics["reason"],
                        new_governor_reason=new.diagnostics["reason"], **requested))
    # Privileged first touch anchors are used only after causal generation.
    touch = archived["first_receiver_contact_s"]
    pre = [r for r in seq if r["time_s"] < touch - 1e-8]
    if len(pre) != len(old_preview):
        raise ValueError("Archived precontact denominator changed")
    changed = [r for r in pre if r["reference_delta_m"] > TOL]
    last = pre[-1]
    return dict(context=case["context"], first_physical_touch_s=touch, frozen_release_log_sha256=log_sha,
                original_reconstruction_samples=len(seq), precontact_samples=len(pre), changed_samples=len(changed),
                first_changed_lead_s=None if not changed else touch-changed[0]["time_s"],
                last_precontact=last, reference_authority_violations=0, precontact_rows=pre)


def med(values):
    return float(np.median(values))


def summarize(cases):
    affected = [r for r in cases if r["first_changed_lead_s"] is not None]
    lead = [1000.*r["first_changed_lead_s"] for r in affected]
    last = [r["last_precontact"] for r in cases]
    before = [-1000.*r["original_equivalent_velocity_m_s"] for r in last]
    after = [-1000.*r["new_equivalent_velocity_m_s"] for r in last]
    return dict(cases_with_changed_reference=len(affected), changed_precontact_rows=sum(r["changed_samples"] for r in cases),
                precontact_rows=sum(r["precontact_samples"] for r in cases),
                reconstructed_original_rows=sum(r["original_reconstruction_samples"] for r in cases),
                first_changed_lead_ms=[min(lead), med(lead), max(lead)],
                old_and_new_median_last_downward_reference_intent_mm_s=[med(before),med(after)],
                median_same_frame_last_downward_intent_change_mm_s=med([y-x for x,y in zip(before,after)]),
                median_last_fixed_q_debt_old_and_new_mm=[
                    med([1000.*r[k] for r in last]) for k in ("original_fixed_q_debt_m","new_fixed_q_debt_m")],
                max_reference_authority_violations=max(r["reference_authority_violations"] for r in cases),
                original_governor_brake_reasons_unchanged=sum(r["old_governor_reason"] == r["new_governor_reason"]
                    for c in cases for r in c["precontact_rows"]))


def self_check():
    controller = schedule.ApproachSchedule()
    def update(now, gap, phase="SEARCH", deadline=15.5):
        public = None if gap is None else dict(available=True, read_s=now,
            engineering_gap_m=(gap[0],gap[2]), conditional_plane_gap_m=[gap[1]-1e-5,gap[1]+1e-5])
        return controller.update(now=now, phase=phase, preview=public, original_request_m_s=-.18,
            velocity_envelope_m_s=.032, deadline_s=deadline)
    far = update(12., (.02,.025,.03))
    near = update(12.005, (-.005, 0., .005))
    dropout = update(12.01, None)
    recovers = update(12.015, (.02,.025,.03))
    assert far["requested_velocity_m_s"] == -.18
    assert 0. < near["effective_request_magnitude_m_s"] < .032
    assert dropout["requested_velocity_m_s"] == near["requested_velocity_m_s"]
    assert near["effective_request_magnitude_m_s"] < recovers["effective_request_magnitude_m_s"] < .032
    assert update(12.02, None, phase="CAPTURE")["requested_velocity_m_s"] == -.18
    assert update(15.5, None)["status"] == "original_phase_or_deadline"
    return ["far_original", "positive_crawl_at_near", "missing_no_acceleration",
            "bounded_return_to_speed", "capture_and_deadline_keep_original_request"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    pins = preview_replay.PINS
    stage, auxiliary, plan = [preview_replay.read(preview_replay.verify(ROOT / f, h)) for f,h in pins.items()]
    frozen = preview_replay.read(preview_replay.verify(PREVIEW, PREVIEW_SHA))
    preview_cases = {tuple(r["context"]):r for r in frozen["cases"]}
    archived = {tuple(r["context_key"]):r for r in auxiliary["records"]}
    if len(archived) != 16 or set(archived) != set(preview_cases):
        raise ValueError("Complete16 and preview rosters differ")
    mount, _ = preview_replay.mount_from_source(plan)
    checks = self_check()
    cases = [run_case(r, archived[tuple(r["context"])], mount, preview_cases[tuple(r["context"])])
             for r in stage["rows"]]
    result = dict(schema="v21_support_approach_schedule_replay_v1", physics_steps=0, plant_instances=0,
                  contexts=len(cases), read_from_static_compiled_model_only=True, tests=checks,
                  source_sha256={**pins, PREVIEW.name:PREVIEW_SHA, **{
                      Path(p).name:preview_replay.sha(Path(p)) for p in (__file__,schedule.__file__,preview_replay.__file__)}},
                  summary=summarize(cases), cases=cases)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()

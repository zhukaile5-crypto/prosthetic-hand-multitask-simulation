"""Replay causal public support clearance and score against archived truth."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from identification import v21_public_support_preview_v1 as preview
from research_imports import FOUNDATION
from v14_coord.design import reference_rates

PINS = {
    "review/v21_stage_attribution_v1.json": "dd45c6a1fbcd70920c587b5696bf56ef1137b69e3a8f8a5d09304ae335dafc48",
    "review/v21_complete16_auxiliary_v1.json": "b894592f38a8ca199088fd41bda9331ab819486b995b7db08d35e898be12d933",
    "review/v21_complete16_plan_v1.json": "77c3670c906cd9b7951fe1ffc0e50dacc10b1e7e4908174ba7f63d0c9b41a9ed",
}
DT = .0005
CONTROL_DT = .005
TOL = 1e-8
TEMPLATE = preview.DeclaredShapeTemplate("capsule", .028, .047, "same nominal capsule as V21 public holding")
LAST_WINDOW_S = .200
CUES = ("cue_lower", "cue_center", "cue_upper")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Frozen file changed: {path}")
    return path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def mount_from_source(original):
    file = verify(FOUNDATION / "assets/legacy_model.xml",
                  original["source_sha256"]["foundation/assets/legacy_model.xml"])
    model = ET.parse(file).getroot()
    geom = model.find("./worldbody/geom[@name='support']")
    if geom.attrib.get("type") != "box" or any(k in geom.attrib for k in ("quat", "axisangle", "euler", "xyaxes", "zaxis")):
        raise ValueError("Expected a fixed horizontal receiver box")
    pos, size = (np.fromstring(geom.attrib[k], sep=" ") for k in ("pos", "size"))
    obj = model.find("./worldbody/body[@name='object']/geom[@name='object_geom']")
    if obj.attrib.get("quat") != "1 1 0 0" or "pos" in obj.attrib:
        raise ValueError("Unknown archived local object axis for offline scoring")
    return preview.StaticReceiverTop(tuple(pos[:2]), tuple(size[:2]), float(pos[2] + size[2])), sha(file)


def archive_truth_bottom(state, radius, half_segment, mount):
    """Only used after the public preview has been computed."""
    q = state[3:]
    if abs(np.linalg.norm(q) - 1.) > 1e-8:
        raise ValueError("Invalid archived object quaternion")
    vector = np.array([0., -1., 0.])
    axis = vector + 2 * np.cross(q[1:], np.cross(q[1:], vector) + q[0] * vector)
    return float(state[2] - radius - half_segment * abs(axis[2]) - mount.height_m)


def run_case(case, record, mount):
    archive = case["metrics"]["full_V21"]
    trace = verify(ROOT / archive["trace"], archive["trace_sha256"])
    if not trace.resolve().is_relative_to((ROOT / "output").resolve()) or archive["trace_sha256"] != record["input_sha256"]["trace.npz"]:
        raise ValueError("Trace identity does not match the frozen complete16 roster")
    meta = read(verify(trace.parent / "metadata.json", record["input_sha256"]["metadata.json"]))
    device = meta["receiver_instrument"]
    if device["mount_geom_name"] != "support" or meta["config"]["shape"] != "capsule":
        raise ValueError("Known registered mount and capsule required")
    reader = preview.SupportPreview(mount, TEMPLATE)
    observations = []
    with np.load(trace, allow_pickle=False) as z:
        t = z["time"]
        if not np.array_equal(t, np.arange(3201) * CONTROL_DT):
            raise ValueError("Frozen trace does not have the complete public clock")
        phases = z["receiver_release_tasks_json"]
        packets = z["observation_json"]
        for k in range(2400, len(t)):
            phase = json.loads(phases[k])
            if not phase or phase["phase"] != "SEARCH":
                break
            frame = json.loads(packets[k])
            if frame["time"] != t[k]:
                raise ValueError("Public observation and decision clocks differ")
            result = reader.read(frame)
            ref = frame["robot_reference21"][18]
            q = frame["robot_position25"][24]
            result["fixed_q_reference_recovery_s"] = max(0., q - ref) / float(reference_rates("release")[18])
            observations.append(result)
    # Only here open privileged archived scoring labels.
    delivery = verify(trace.parent / "delivery_intervals.npz", archive["delivery_sha256"])
    with np.load(delivery, allow_pickle=False) as z:
        start, end = z["interval_start_s"], z["interval_end_s"]
        truth = z["object_free_qpos7_end"]
        target = z["target_receiver_top_normal_N"]
        valid = z["object_state_valid"]
        complete = (len(end) == 32000 and truth.shape == (32000, 7) and valid.all()
                    and np.allclose(end, (np.arange(32000)+1)*DT, atol=TOL, rtol=0.))
        if not complete:
            raise ValueError("Incomplete frozen object truth")
    first = float(start[np.flatnonzero((start >= 12. - TOL) & (target > 0.))[0]])
    if abs(first - archive["first_receiver_contact_s"]) > TOL:
        raise ValueError("Frozen contact anchor differs from first target-top ledger interval")
    radius = .028 * meta["config"]["size"]
    half_segment = .047 * meta["config"]["size"]
    pre = []
    for row in observations:
        if row["read_s"] >= first - TOL:
            continue
        if row["available"]:
            k = round(row["acquisition_s"] / DT) - 1
            if abs(end[k] - row["acquisition_s"]) > TOL:
                raise ValueError("Offline geometry was compared at a different physical time")
            actual = archive_truth_bottom(truth[k], radius, half_segment, mount)
            low, high = row["engineering_gap_m"]
            fit_low, fit_high = row["conditional_plane_gap_m"]
            row.update(scoring_gap_at_acquisition_m=actual, padded_interval_contains_actual=low <= actual <= high,
                       conditional_interval_contains_actual=fit_low <= actual <= fit_high,
                       midpoint_error_m=(fit_low + fit_high) / 2 - actual,
                       padded_width_m=high-low)
        pre.append(row)
    cues = {}
    for key in CUES:
        r = next((r for r in observations if r.get(key) is True), None)
        lead = None if r is None else first - r["read_s"]
        cues[key] = dict(read_s=None if r is None else r["read_s"], lead_s=lead,
                         earlier_than_200ms=bool(lead is not None and lead > LAST_WINDOW_S + TOL),
                         positive_lead=bool(lead is not None and lead > TOL),
                         lead_at_least_fixed_q_reference_recovery=bool(
                             lead is not None and lead >= r["fixed_q_reference_recovery_s"] > 0.),
                         lead_at_least_existing_20ms_command_horizon=bool(lead is not None and lead >= .020 - TOL))
    return dict(context=case["context"], first_physical_target_top_s=first,
                public_search_reads=len(observations), physical_precontact_reads=len(pre),
                geometry_available_before_contact=sum(r["available"] for r in pre),
                trace_sha256=archive["trace_sha256"], delivery_sha256=archive["delivery_sha256"],
                cues=cues, precontact_rows=pre)


def q50(xs):
    return None if not len(xs) else float(np.median(xs))


def summarize(cases):
    all_before = [r for c in cases for r in c["precontact_rows"]]
    near = [r for c in cases for r in c["precontact_rows"]
            if c["first_physical_target_top_s"] - r["read_s"] <= LAST_WINDOW_S + TOL]
    def accuracy(rows):
        valid = [r for r in rows if r["available"]]
        error = np.asarray([r["midpoint_error_m"] for r in valid])
        return dict(public_frames=len(rows), available=len(valid),
                    padded_interval_contains_actual=sum(r["padded_interval_contains_actual"] for r in valid),
                    conditional_interval_contains_actual=sum(r["conditional_interval_contains_actual"] for r in valid),
                    padded_lower_gap_too_high=sum(r["engineering_gap_m"][0] > r["scoring_gap_at_acquisition_m"] for r in valid),
                    median_signed_error_mm=None if not len(error) else float(1000.*np.median(error)),
                    p95_abs_error_mm=None if not len(error) else float(1000.*np.quantile(abs(error), .95)),
                    median_padded_width_mm=q50([1000.*r["padded_width_m"] for r in valid]))
    cue = {}
    for key in CUES:
        readings = [c["cues"][key] for c in cases]
        leads = [1000.*r["lead_s"] for r in readings if r["lead_s"] is not None]
        cue[key] = dict(available=len(leads), before_touch=sum(r["positive_lead"] for r in readings),
                        earlier_than_200ms=sum(r["earlier_than_200ms"] for r in readings),
                        at_least_20ms=sum(r["lead_at_least_existing_20ms_command_horizon"] for r in readings),
                        lead_covers_fixed_q_reference_recovery=sum(r["lead_at_least_fixed_q_reference_recovery"] for r in readings),
                        lead_ms_min_median_max=None if not leads else [min(leads), q50(leads), max(leads)])
    return dict(all_before_touch=accuracy(all_before), last_200ms=accuracy(near), cue=cue)


def self_check(mount):
    horizontal = SimpleNamespace(template=TEMPLATE, axis_reference=[0., 1., 0.],
        center_line_origin_reference=[.095, 0., .4], axial_center_interval_m=(-.01, .01))
    vertical = SimpleNamespace(template=TEMPLATE, axis_reference=[0., 0., 1.],
        center_line_origin_reference=[.095, 0., 0.], axial_center_interval_m=(.39, .41))
    assert np.allclose(preview.lowest_points(horizontal)[:, 2], .4-.028, atol=1e-15, rtol=0.)
    assert np.allclose(preview.lowest_points(vertical)[:, 2], np.array([.39, .41])-.047-.028, atol=1e-15, rtol=0.)
    q = np.r_[.095, 0., .4, 1., 0., 0., 0.]
    assert abs(archive_truth_bottom(q, .028, .047, mount) - (.4-.028-mount.height_m)) < 1e-15
    q[3:] = [2.**-.5, 2.**-.5, 0., 0.]
    assert abs(archive_truth_bottom(q, .028, .047, mount) - (.4-.028-.047-mount.height_m)) < 1e-15
    probe = preview.SupportPreview(mount, TEMPLATE)
    assert probe.read({"time":12., "sample_time":11.995})["reason"] == "wrong_acquisition_clock"
    return ["capsule_horizontal_bottom", "capsule_vertical_axis_interval", "offline_quaternion_orientation",
            "invalid_acquisition_clock_rejected"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stage, auxiliary, plan = [read(verify(ROOT / p, h)) for p,h in PINS.items()]
    lookup = {tuple(r["context_key"]):r for r in auxiliary["records"]}
    if len(lookup) != 16 or set(lookup) != {tuple(r["context"]) for r in stage["rows"]}:
        raise ValueError("Incomplete frozen complete16 roster")
    mount, mount_sha = mount_from_source(plan)
    source_checks = {k:sha(verify(ROOT / k, plan["source_sha256"][k]))
                     for k in ("v11/object_belief.py", "v14_coord/design.py")}
    checks = self_check(mount)
    cases = [run_case(c, lookup[tuple(c["context"])], mount) for c in stage["rows"]]
    result = dict(schema="v21_public_support_preview_archive_v1", physics_steps=0, model_instances=0,
                  contexts=len(cases), physical_seed_clusters=2, late_window_s=LAST_WINDOW_S,
                  self_checks=checks, source_sha256={**PINS, **source_checks, "foundation/assets/legacy_model.xml":mount_sha,
                      Path(__file__).name:sha(Path(__file__)), Path(preview.__file__).name:sha(Path(preview.__file__))},
                  assumption="Conditional acquisition-time rigid geometry; padded interval is engineering only. Any truth used solely for offline scoring.",
                  mount=vars(mount), template=vars(TEMPLATE), summary=summarize(cases), cases=cases)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()

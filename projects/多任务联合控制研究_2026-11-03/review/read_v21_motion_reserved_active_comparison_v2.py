"""Read one completed active run against two frozen, same-context archives.

No writes, simulator imports, or replay. Whole-run terminal/assignment gating
belongs to the caller. Per-arm contact windows use each arm's own touch clock.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from review import delivery_auxiliary_metrics_v3 as delivery
from review import read_v21_contact_aligned_force_quality_v1 as force


ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "review/delivery_auxiliary_metrics_v3.py": "0228b2526ace5f448a6f7f458ebeadb9023b79b120fe4ba16eccb3463eb5bc55",
    "review/delivery_auxiliary_contract_v3.json": "8dcda359252014d509a8b26c102570309ad893b6a4ccad31015a36ca0cb7e979",
    "review/read_v21_contact_aligned_force_quality_v1.py": "b8c39ca3e6f79237f2cd4180b3b280863ab42a02305baa4880b53ef0f60e110b",
    "review/v21_complete16_release_variation_v1.json": "a8cfe6ee01659fdbb84918c022c6e6d970b15223238f932e00eabaefb2b0d40a",
    "review/v21_stage_attribution_v1.json": "dd45c6a1fbcd70920c587b5696bf56ef1137b69e3a8f8a5d09304ae335dafc48",
}
VARIATION = "review/v21_complete16_release_variation_v1.json"
STAGES = "review/v21_stage_attribution_v1.json"
EVALUATION = "beb1befe2f27550c782d1420b841304929ebe80798796ef543aa626ca20d9b39"
PROTOCOL = "de3c6af5ebdd0abe5dcfcb47c07d11aa2beb360769c45390df883594d6b03eb0"
MATCHED_FIELDS = (
    "config", "noise", "limits", "budget", "split", "shape", "initial_position",
    "gains", "actual_ratio_task_gains", "model_sha256", "admission_sha256",
    "duration", "sensor_period", "physics_timestep", "handoff_time",
)


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _scenario(value):
    return value.get("name") if isinstance(value, dict) else value


def _context(row, seed_key="seed"):
    key = (_scenario(row["scenario"]), row["motion_mode"], row[seed_key])
    _require(isinstance(key[0], str) and key[1] in ("hold", "transport")
             and type(key[2]) is int, "Invalid scenario/motion/seed identity")
    return key


def _unique(rows, predicate, label):
    matches = [row for row in rows if predicate(row)]
    _require(len(matches) == 1, f"Missing or duplicate frozen {label}")
    return matches[0]


def _archive_pin(variation, stages, key, role, run):
    records = variation["records"] if role == "full_V21" else variation["reference_records"]
    entry = _unique(records, lambda x: x["record_key"] == [role, *key], role)
    stage = _unique(stages["rows"], lambda x: x["context"] == list(key), "contact context")
    contact = stage["metrics"][role]
    _require(Path(entry["run"]).resolve() == run and (ROOT / contact["trace"]).resolve() == run / "trace.npz",
             f"{role}: archive path does not match both frozen indexes")
    _require(entry["trace_sha256"] == contact["trace_sha256"], f"{role}: conflicting trace pins")
    return {
        "metadata.json": entry["metadata_sha256"],
        "result.json": entry["result_sha256"],
        "trace.npz": entry["trace_sha256"],
        "delivery_intervals.npz": contact["delivery_sha256"],
    }


def _load_record(run, role, archive_pin=None):
    _require(run.is_relative_to(ROOT / "output"), f"{role}: expected a run under project output")
    _require(not (run / "runtime_error.json").exists(), f"{role}: runtime error recorded")
    meta = _read(run / "metadata.json")
    result = _read(run / "result.json")
    research = result["research_metrics"]
    key = _context(meta)
    _require(_context(result) == key and _scenario(research["scenario"]) == key[0]
             and research["physical_seed"] == key[2] and type(research["noise_seed"]) is int,
             f"{role}: result context or noise seed inconsistent with metadata")
    _require(meta.get("completed") is True and result.get("record_complete") is True
             and research.get("record_complete") is True, f"{role}: incomplete record")
    _require(meta["run_id"] == result["run_id"] and meta["split"] == result["split"]
             and meta["method"] == result["method"] == research["method"] == "ratio_effort",
             f"{role}: run or method identity mismatch")
    _require((research["evaluation_source_sha256"], research["protocol_sha256"]) == (EVALUATION, PROTOCOL),
             f"{role}: different original evaluator or protocol")
    policies = (meta["task_policy"], meta["receiver_policy"])
    expected_policies = ("legacy", "blind") if role == "same_prior_pi" else ("integrated", "coordinated")
    _require(policies == expected_policies, f"{role}: wrong policy arm")
    _require(meta["duration"] == 16.0 and meta["physics_timestep"] == force.DT
             and meta["sensor_period"] == .005, f"{role}: wrong recorded clock")
    manifest = _read(run / "delivery_manifest.json")
    ledger = manifest["arrays"]
    _require(manifest.get("privileged_offline_only") is True and ledger["file"] == "delivery_intervals.npz",
             f"{role}: missing declared offline delivery ledger")
    names = ("metadata.json", "result.json", "trace.npz", "delivery_intervals.npz")
    pins = {name: _sha(run / name) for name in names}
    _require(pins["trace.npz"] == meta["trace_sha256"] == result["trace_sha256"] == research["trace_sha256"],
             f"{role}: trace hash does not agree with all recorded identities")
    _require(pins["delivery_intervals.npz"] == ledger["sha256"], f"{role}: delivery ledger hash mismatch")
    _require(archive_pin is None or archive_pin == pins, f"{role}: frozen archive bytes differ")
    return meta, result, pins


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _flag(value):
    return value if type(value) is bool else None


def _peak(section):
    values = section.get("finger_peak_N")
    if not isinstance(values, list) or len(values) != 5 or any(_number(x) is None for x in values):
        return None
    return max(values)


def _original(result, release_s):
    research = result["research_metrics"]
    release = research["release"]
    safety = research["safety"]
    stage = safety["stages"].get("mechanical_release_and_after") or {}
    _require(release["actual_command_at"] == release_s, "Original evaluation and mechanical release clocks differ")
    return {
        "task_success": _flag(result["task_success"]),
        "full_motion_mission_success": _flag(research["full_motion_mission_success"]),
        "original_failure_reason": result.get("failure_reason"),
        "original_failure_at_s": _number(result.get("failure_at")),
        "release_physical_exposure": _flag(release["physical_exposure"]),
        "release_task_qualified_exposure": _flag(release["task_qualified_exposure"]),
        "clearance_success": _flag(release["physical_success"]),
        "task_qualified_clearance_success": _flag(release["task_qualified_success"]),
        "clearance_latency_s": _number(release["physical_clearance_latency_s"]),
        "force_guard_pass": _flag(safety["guard_pass"]),
        "full_finger_peak_N": _peak(safety),
        "full_local_excess_Ns": _number(safety["local_excess_integral_Ns"]),
        "full_total_excess_Ns": _number(safety["total_excess_integral_Ns"]),
        "release_finger_peak_N": _peak(stage),
        "release_local_excess_Ns": _number(stage.get("local_excess_integral_Ns")),
        "release_total_excess_Ns": _number(stage.get("total_excess_integral_Ns")),
    }


def _window(history, touch, tag, field):
    if history is None or touch is None:
        return {"available": False, "value_N": None, "reason": "force history or touch anchor unavailable"}
    _, lo, hi = force.WINDOWS[tag]
    start, stop = touch + lo, touch + hi
    try:
        stencil = force.local_deviation(history, force.step(start), force.step(stop), force.step(.020))
        detail = force.describe(*stencil)
    except ValueError as exc:
        return {"available": False, "value_N": None, "reason": str(exc), "requested_window_s": [start, stop]}
    return {"available": True, "value_N": detail[field], "requested_window_s": [start, stop], "detail": detail}


def _arm(run, record):
    meta, result, pins = record
    release = meta.get("mechanical_release_at", meta.get("release_at"))
    original = _original(result, release)
    aux, history = None, None
    issues = []
    if release is None:
        issues.append("No actual mechanical release clock")
    else:
        try:
            with np.load(run / "delivery_intervals.npz", allow_pickle=False) as source:
                arrays = {name: source[name] for name in delivery.C["input_fields"]}
            aux = delivery.calculate(arrays, release_at_s=release, episode_end_s=meta["duration"],
                                     physical_period_s=meta["physics_timestep"], original_result=result)
        except (ValueError, KeyError, OSError) as exc:
            issues.append(f"Delivery metrics unavailable: {exc}")
    try:
        history = force.force_history({"trace": str(run / "trace.npz"), "trace_sha256": pins["trace.npz"]})
    except (ValueError, KeyError, OSError) as exc:
        issues.append(f"2 kHz force history unavailable: {exc}")
    aux = aux or {}
    provenance = aux.get("receiver_event_provenance") or {}
    contact_known = aux.get("contact_coverage", {}).get("all_valid") is True
    state_known = aux.get("coverage", {}).get("all_valid") is True
    touch = aux.get("first_receiver_contact_s")
    fresh_first_touch = contact_known and provenance.get("first_positive_onset_status") == "KNOWN_ZERO_TO_POSITIVE_TRANSITION"
    aligned_touch = touch if fresh_first_touch else None
    impact = aux.get("first_receiver_impact20ms") or {}
    receiver_impact = impact.get("contact", {}).get("receiver_top", {})
    first = aux.get("official_first_stable_receiver")
    final = aux.get("official_final_stable_suffix")
    terminal = aux.get("terminal") or {}
    return {
        "run": str(run),
        "input_sha256": pins,
        "original": original,
        "release_at_s": release,
        "delivery_status": aux.get("status"),
        "first_target_top_contact_s": touch if contact_known else None,
        "first_observed_target_top_contact_s": touch,
        "contact_provenance": provenance or None,
        "contact_coverage_known": contact_known if aux else None,
        "first_target_top_impact_start_s": impact.get("start_s"),
        "first_target_top_impact20ms_peak_N": receiver_impact.get("peak_N"),
        "first_target_top_impact20ms_impulse_Ns": receiver_impact.get("impulse_Ns"),
        "ever_target_floor_contact": aux.get("ever_floor_contact"),
        "first_low_motion_found": first is not None if state_known else None,
        "first_low_motion_confirmation_s": None if first is None else first["confirmation_s"],
        "first_low_motion_after_touch_s": None if first is None or aligned_touch is None
        else first["confirmation_s"] - aligned_touch,
        "final_low_motion_suffix_found": final is not None and not final["right_censored"] if state_known else None,
        "final_low_motion_suffix": final,
        "terminal_linear_speed_m_s": terminal.get("linear_speed_m_s"),
        "terminal_angular_speed_rad_s": terminal.get("angular_speed_rad_s"),
        "touch_0_to_200ms_p95": _window(history, aligned_touch, "touch_0_to_200ms", "finger_p95_N"),
        "touch_200_to_500ms_rms": _window(history, aligned_touch, "touch_200_to_500ms", "finger_rms_N"),
        "issues": issues,
    }


def _contrasts(candidate, reference):
    out = {}
    for metric in ("first_target_top_contact_s", "first_target_top_impact20ms_peak_N",
                   "first_target_top_impact20ms_impulse_Ns", "first_low_motion_after_touch_s",
                   "terminal_linear_speed_m_s", "terminal_angular_speed_rad_s",
                   "touch_0_to_200ms_p95", "touch_200_to_500ms_rms"):
        a, b = candidate[metric], reference[metric]
        if isinstance(a, dict):
            a, b = a["value_N"], b["value_N"]
        out[metric] = None if a is None or b is None else a - b
    for metric in ("clearance_latency_s", "release_finger_peak_N", "release_local_excess_Ns", "release_total_excess_Ns"):
        a, b = candidate["original"][metric], reference["original"][metric]
        out[metric] = None if a is None or b is None else a - b
    return out


def compare_active_run(new_run: Path, archived_v21: Path, archived_pi: Path) -> dict:
    """Return one JSON-safe paired descriptive comparison or an explicit unassessed record."""
    key = None
    try:
        for relative, digest in PINS.items():
            _require(_sha(ROOT / relative) == digest, f"Frozen comparison dependency changed: {relative}")
        variation, stages = _read(ROOT / VARIATION), _read(ROOT / STAGES)
        _require(variation.get("complete") is True and variation.get("issues") == []
                 and variation.get("assigned_count") == 16 and len(stages["rows"]) == 16,
                 "Frozen reference indexes incomplete")
        active, v21, pi = (Path(p).resolve() for p in (new_run, archived_v21, archived_pi))
        active_record = _load_record(active, "active")
        key = _context(active_record[0])
        v21_record = _load_record(v21, "frozen_v21", _archive_pin(variation, stages, key, "full_V21", v21))
        pi_record = _load_record(pi, "same_prior_pi", _archive_pin(variation, stages, key, "same_prior_legacy_ratio", pi))
        for label, record in (("frozen_v21", v21_record), ("same_prior_pi", pi_record)):
            for field in MATCHED_FIELDS:
                _require(active_record[0][field] == record[0][field], f"{label}: different {field}")
            _require(active_record[1]["research_metrics"]["noise_seed"] == record[1]["research_metrics"]["noise_seed"],
                     f"{label}: different noise seed")
        arms = {"active": _arm(active, active_record), "frozen_v21": _arm(v21, v21_record),
                "same_prior_pi": _arm(pi, pi_record)}
        return {
            "available": True, "context_key": list(key), "reason": None, "arms": arms,
            "pairs": {role: {"active_minus_reference": _contrasts(arms["active"], arms[role])}
                      for role in ("frozen_v21", "same_prior_pi")},
            "matched_fields": list(MATCHED_FIELDS) + ["physical_seed", "noise_seed"],
            "trial_count": 1,
            "scope": "Descriptive per-context contrast. PI differs in complete strategy. Post-touch metrics use each arm's own event.",
            "touch_force_definition": "Frozen 2 kHz five-finger END force residual from centered 20 ms half-span mean within each window; not raw force amplitude.",
            "impact_definition": "Offline target/top ledger first known zero-to-positive transition and following 20 ms. An observed first load may preexist release.",
        }
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {"available": False, "context_key": None if key is None else list(key),
                "reason": str(exc), "arms": None, "pairs": None, "trial_count": 1}


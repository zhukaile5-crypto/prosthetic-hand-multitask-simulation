"""Read matched V21 archives with zero plant instances or physical steps."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import research_imports
from foundation_config import FoundationConfig
from hand_plant import _model
from identification import v21_receiver_transfer_evidence_v1 as evidence

AUX = ROOT / "review/v21_complete16_auxiliary_v1.json"
AUX_SHA = "b894592f38a8ca199088fd41bda9331ab819486b995b7db08d35e898be12d933"
READ_TICKS = 100
CLOCK_TOL = 1e-8


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Frozen input changed: {path}")
    return path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def public_reading(t, process, receiver):
    raw = process.get("raw_coacquired_loads")
    detail = process["transfer"]["continuous_support_evidence"]
    prior = detail.get("prior")
    if (raw is None or prior is None or not receiver["valid"]
            or receiver["acquisition_time_s"] is None or receiver["uncertainty6"] is None):
        return None
    if not (abs(receiver["time_s"] - t) <= CLOCK_TOL
            and abs(receiver["acquisition_time_s"] - raw["acquisition_s"]) <= CLOCK_TOL
            and abs(receiver["age_s"] - evidence.DELAY_S) <= CLOCK_TOL):
        raise ValueError("Hand and receiver records lack matched public clocks")
    if (prior["role"] != "public_pre_release_motor_load_proxy_not_true_weight"
            or prior["time_s"] > process["started_s"] + CLOCK_TOL):
        raise ValueError("Unexpected pre-release demand prior")
    if prior["value_N"] <= prior["uncertainty_N"]:
        return None
    receiver_up = -float(receiver["wrench6"][2])
    if abs(receiver_up - raw["receiver_signed_N"]) > CLOCK_TOL:
        raise ValueError("Unexpected receiver axis or wrench")
    return evidence.TransferReading(t, raw["acquisition_s"], receiver_up,
                                    float(receiver["uncertainty6"][2]), raw["hand_signed_N"],
                                    raw["hand_width_N"], prior["value_N"] + prior["uncertainty_N"])


def audit_ledger(path):
    with np.load(path, allow_pickle=False) as z:
        start, stop = z["interval_start_s"], z["interval_end_s"]
        idx = np.arange(24000, 32000)
        valid = z["contact_valid"][idx]
        all_actor = z["receiver_any_actor_top_normal_N"][idx]
        target = z["target_receiver_top_normal_N"][idx]
    if (start.shape != (32000,) or not np.allclose(start, np.arange(32000) * .0005, atol=CLOCK_TOL, rtol=0.)
            or not np.array_equal(start[1:], stop[:-1]) or not valid.all()
            or not np.isfinite(all_actor).all() or not np.isfinite(target).all()):
        raise ValueError("Incomplete release-interval evidence")
    residual = all_actor - target
    return dict(valid_intervals=len(idx), target_loaded_intervals=int(np.sum(target > 0.)),
                any_actor_minus_target_max_abs_N=float(np.max(abs(residual))),
                intervals_different=int(np.sum(abs(residual) > 1e-9)),
                role="Offline corroboration only. This evidence never enters the scheduling input.")


def load_case(record):
    folder = Path(record["run"]).resolve()
    if not folder.is_relative_to((ROOT / "output").resolve()):
        raise ValueError("Unexpected archive root")
    pins = record["input_sha256"]
    m = read(verify(folder / "metadata.json", pins["metadata.json"]))
    config = {k: tuple(v) if isinstance(v, list) else v for k, v in m["config"].items()}
    model, model_sha = _model(FoundationConfig(**config))
    if model_sha != m["model_sha256"] or not np.array_equal(model.geom("support").quat, [1., 0., 0., 0.]):
        raise ValueError("Compiled geometry or sensor-axis identity mismatch")
    topology = evidence.receiver_topology(model)
    trace = verify(folder / "trace.npz", pins["trace.npz"])
    with np.load(trace, allow_pickle=False) as x:
        t = x["time"]
        if not np.array_equal(t, np.arange(3201) * evidence.PERIOD_S):
            raise ValueError("Complete 200 Hz public clock required")
        lifecycle = [json.loads(v) for v in x["receiver_release_tasks_json"][2400:]]
        k0, = [next(i for i, v in enumerate(lifecycle) if v and v["phase"] == "TRANSFER")]
        first = 2400 + k0
        receiver = x["receiver_measurement_json"][first:first + READ_TICKS]
    if len(receiver) != READ_TICKS:
        raise ValueError("A complete 500 ms readout must be available")
    scheduler = evidence.TransferEvidenceWindow()
    rows = []
    for offset, serialized in enumerate(receiver):
        now = float(t[first + offset])
        process = lifecycle[k0 + offset]
        reading = public_reading(now, process, json.loads(serialized)) if process["phase"] == "TRANSFER" else None
        state = scheduler.update(now, reading, topology)
        transfer = process["transfer"]
        rows.append(dict(time_s=now, phase=process["phase"], legacy_reaction_fraction=process["unload_reaction_fraction"],
                         first_compression_read_s=process["receiver_event_at"],
                         old_attribution_reason=transfer["unknown_reasons"], old_permission=transfer["stage_permission"],
                         reading=None if reading is None else asdict(reading), **state))
    first_sustained = next((r for r in rows if r["window_complete"]), None)
    ledger = audit_ledger(verify(folder / "delivery_intervals.npz", pins["delivery_intervals.npz"]))
    return dict(context=record["context_key"], model_sha256=model_sha,
                topology=asdict(topology), trace_sha256=pins["trace.npz"],
                delivery_sha256=pins["delivery_intervals.npz"], offline_ledger=ledger, first_transfer_s=float(t[first]),
                first_sustained_s=None if first_sustained is None else first_sustained["time_s"],
                first_sustained_fraction=None if first_sustained is None else first_sustained["sustained_proxy_fraction"],
                rows=rows)


def self_checks():
    minimal = SimpleNamespace(
        body=lambda name: SimpleNamespace(id=1), geom=lambda name: SimpleNamespace(id=0),
        nbody=2, nplugin=0, body_parentid=[0, 0], geom_bodyid=[0, 1], geom_type=[6, 6],
        geom_contype=[4, 1], geom_conaffinity=[2, 3], pair_geom1=[], pair_geom2=[])
    clean = evidence.receiver_topology(minimal)
    assert clean.robot_direct_contact_excluded
    compatible = SimpleNamespace(**{**vars(minimal), "geom_contype": [4, 2]})
    explicit = SimpleNamespace(**{**vars(minimal), "pair_geom1": [1], "pair_geom2": [0]})
    assert not evidence.receiver_topology(compatible).robot_direct_contact_excluded
    assert not evidence.receiver_topology(explicit).robot_direct_contact_excluded
    a = evidence.TransferReading(1., .990, 3., .03, -2.7, .2, 1.5)
    assert evidence.score_reading(a)[0] < .05 and evidence.score_reading(a)[1]
    scheduler = evidence.TransferEvidenceWindow()
    b = replace(a, receiver_up_N=1., hand_up_N=.4)
    def at(t):
        return replace(b, read_s=t, acquisition_s=t - evidence.DELAY_S)
    try:
        scheduler.update(1., replace(b, acquisition_s=.995), clean)
    except ValueError:
        assert scheduler.last_tick is None
    else:
        raise AssertionError("A non-coacquired packet was accepted")
    for t in (1., 1.005, 1.010, 1.015, 1.020):
        val = scheduler.update(t, at(t), clean)
        assert val["window_complete"] is (t == 1.020)
    assert scheduler.update(1.020, at(1.020), clean) == val
    assert not scheduler.update(1.030, at(1.030), clean)["window_complete"]
    assert scheduler.update(1.035, None, clean)["consecutive_positive_samples"] == 0
    return ["incompatible_masks_exclude", "dynamic_mask_pair_is_possible", "explicit_pair_overrides_masks",
            "signed_downward_hand_reduces_proxy", "bad_clock_does_not_consume_tick",
            "five_distinct_samples_and_idempotent_repeat", "gap_and_missing_sample_reset"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checks = self_checks()
    aux = read(verify(AUX, AUX_SHA))
    roster = aux["records"]
    if not aux["complete"] or len({tuple(r["context_key"]) for r in roster}) != len(roster) or len(roster) != 16:
        raise ValueError("Expected the frozen 16-case V21 roster")
    cases = [load_case(r) for r in roster]
    first = [r["rows"][0] for r in cases]
    window = [a for r in cases for a in r["rows"] if a["phase"] == "TRANSFER"]
    latencies = [r["first_sustained_s"] - r["first_transfer_s"] for r in cases if r["first_sustained_s"] is not None]
    summary = dict(
        contexts=len(cases), compiled_models=len({r["model_sha256"] for r in cases}),
        no_direct_robot_pair=sum(r["topology"]["robot_direct_contact_excluded"] for r in cases),
        eligible_receiver_candidates=sorted({r["topology"]["receiver_candidate_count"] for r in cases}),
        checked_release_intervals=sum(r["offline_ledger"]["valid_intervals"] for r in cases),
        target_loaded_intervals=sum(r["offline_ledger"]["target_loaded_intervals"] for r in cases),
        any_actor_target_mismatch_intervals=sum(r["offline_ledger"]["intervals_different"] for r in cases),
        old_possible_robot_at_first_transfer=sum("possible_hand_receiver_contact" in r["old_attribution_reason"] for r in first),
        old_ratio_one_at_first_transfer=sum(r["legacy_reaction_fraction"] == 1. for r in first),
        downward_resolved_at_first_transfer=sum(r["hand_downward_resolved"] is True for r in first),
        sustained_at_first_transfer=sum(r["window_complete"] for r in first),
        cases_sustained_within_500ms=len(latencies),
        first_sustained_latency_ms=None if not latencies else [1000. * min(latencies), 1000. * float(np.median(latencies)), 1000. * max(latencies)],
        transfer_readouts=len(window),
        complete_sustained_readouts=sum(a["window_complete"] for a in window))
    result = dict(schema="v21_receiver_transfer_evidence_archive_v1", summary=summary, checks=checks,
                  physics_steps=0, plant_instances=0, uses_compiled_model_metadata=True,
                  source_sha256={AUX.name:AUX_SHA, Path(__file__).name:sha(Path(__file__)),
                                 Path(evidence.__file__).name:sha(Path(evidence.__file__))}, cases=cases)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

"""Compare the typed online support adapter with frozen public replay."""

import argparse
from pathlib import Path
import json
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from identification import v21_motion_reserved_observation_v2 as adapter
from identification.v21_receiver_transfer_evidence_v1 import ReceiverTopology
from v15_support.receiver_sensor import ReceiverMeasurement
from v16_release_geometry.transfer_observer import HandSupportSample, PreReleaseLoadPrior
import replay_v21_support_conditioned_exit_v1 as inputs

FIELDS = ("current_proxy_fraction", "hand_downward_resolved", "consecutive_positive_samples",
          "window_complete", "sustained_proxy_fraction", "actor_identity_resolved", "scheduling_only")


def receiver_measurement(packet):
    return ReceiverMeasurement(
        packet["time_s"], packet["acquisition_time_s"], packet["age_s"], packet["valid"],
        packet["wrench6"], packet["normal_load_N"], packet["sigma6"], packet["uncertainty6"],
        packet["cop_xy_m"], packet["sampling_phase"], packet["invalid_reason"])


def hand_measurement(now, process, receiver):
    loads = process.get("raw_coacquired_loads")
    if loads is None:
        return HandSupportSample(now, receiver.acquisition_time_s, False, None, None,
                                 "frozen_no_valid_coacquired_load")
    return HandSupportSample(now, loads["acquisition_s"], True, loads["hand_signed_N"], loads["hand_width_N"])


def replay(case, record):
    folder = Path(record["run"]).resolve()
    trace = inputs.verify(folder / "trace.npz", record["input_sha256"]["trace.npz"])
    if record["input_sha256"]["trace.npz"] != case["trace_sha256"]:
        raise ValueError("Archived readout and trace are from different experiments")
    first = inputs.tick(case["first_transfer_s"])
    stop = first + len(case["rows"])
    with np.load(trace, allow_pickle=False) as z:
        clock = z["time"][first:stop]
        process = [json.loads(x) for x in z["receiver_release_tasks_json"][first:stop]]
        receivers = [receiver_measurement(json.loads(x)) for x in z["receiver_measurement_json"][first:stop]]
    raw_prior = process[0]["transfer"]["continuous_support_evidence"]["prior"]
    prior = PreReleaseLoadPrior(raw_prior["value_N"], raw_prior["uncertainty_N"], raw_prior["time_s"], raw_prior["role"])
    topology = ReceiverTopology(**case["topology"])
    if not topology.robot_direct_contact_excluded:
        raise ValueError("Expected previously validated compiled topology for this mock")
    with patch.object(adapter, "receiver_topology", return_value=topology) as refresh:
        observation = adapter.SupportObservation(object(), [0., 0., 1.], prior, process[0]["started_s"])
        matches = []
        for t, step, receiver, archived in zip(clock, process, receivers, case["rows"], strict=True):
            now = float(t)
            if now != archived["time_s"] or step["phase"] != archived["phase"]:
                raise ValueError("Mismatched frozen control time or stage")
            typed = dict(time_s=now, hand=hand_measurement(now, step, receiver), receiver=receiver)
            actual = observation.update(step["phase"], typed)
            for key in FIELDS:
                a, b = actual[key], archived[key]
                if isinstance(a, float) and isinstance(b, float):
                    if abs(a - b) > 1e-12:
                        raise ValueError(f"Archived numeric mismatch: {case['context']} {now} {key}")
                elif a != b:
                    raise ValueError(f"Archived typed mismatch: {case['context']} {now} {key}")
            if not actual["robot_direct_contact_excluded"]:
                raise ValueError("Mocked previously verified topology was not propagated")
            matches.append(actual)
        if refresh.call_count != len(matches) + 1:
            raise ValueError("Topology was not refreshed at every public tick")
    return dict(context=case["context"], public_readouts=len(matches), fields_per_readout=len(FIELDS),
                sustained=sum(r["window_complete"] for r in matches), topology_refreshes=refresh.call_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = inputs.read(inputs.verify(inputs.SUPPORT, inputs.SUPPORT_SHA))
    aux = inputs.read(inputs.verify(inputs.AUX, inputs.AUX_SHA))
    lookup = {tuple(r["context_key"]):r for r in aux["records"]}
    rows = [replay(c, lookup[tuple(c["context"])]) for c in source["cases"]]
    assert len(rows) == 16 and sum(r["public_readouts"] for r in rows) == 1600
    assert sum(r["sustained"] for r in rows) == source["summary"]["complete_sustained_readouts"]
    result = dict(schema="v21_motion_reserved_observation_replay_v2", passed=True,
                  physics_steps=0, plant_instances=0, new_sensor_samples=0,
                  public_readouts=1600, prior_compiled_topology_mocked=True,
                  source_sha256={Path(__file__).name:inputs.sha(Path(__file__)),
                                 Path(adapter.__file__).name:inputs.sha(Path(adapter.__file__)),
                                 inputs.SUPPORT.name:inputs.SUPPORT_SHA}, rows=rows)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(readouts=result["public_readouts"], contexts=len(rows), matching_fields=len(FIELDS))))


if __name__ == "__main__":
    main()

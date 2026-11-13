"""Verify V21 holding state ownership checks without constructing a model."""

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from identification import v21_fragile_fork_state_v1 as audit

TRACE = ROOT / "output/v21_small_v21_fragile_hold_v1/runs/fragile_conflict__ratio_effort__619741/trace.npz"
TRACE_SHA = "4a9b3fa471a05c6697bf2d12c9dc17026f95995f13c6e24b5d52381ba260819f"


@dataclass
class Frame:
    time: float
    normal: np.ndarray


@dataclass(frozen=True, slots=True)
class ReceiverPacket:
    time: float
    wrench: np.ndarray


def fixture(public):
    frame = Frame(public["time"], np.array(public["normal"], dtype=float))
    frame.robot_reference21 = np.array(public["robot_reference21"], dtype=float)
    frame.receiver_surface = deepcopy(public["receiver_surface"])
    packet = ReceiverPacket(public["time"] - .010, np.arange(6, dtype=float))
    shared = np.arange(8, dtype=float)
    read_only = shared[::2]
    read_only.setflags(write=False)
    return dict(frame=frame, last_receiver=packet, receiver_queue=deque((packet,), maxlen=3),
                controller_history=deque((2.5, 2.4), maxlen=5), rng=np.random.default_rng(619741),
                backing=shared, early=shared[:4], late=shared[4:], read_only=read_only,
                empty=shared[:0])


def compare(a, b, *, left_native=(), right_native=()):
    check = audit.StateComparison()
    check.compare(a, b, "fixture")
    audit.check_exported_buffers(check, audit.index_arrays(left_native), audit.index_arrays(right_native))
    return check


def reject(a, b, message, **kwargs):
    try:
        compare(a, b, **kwargs)
    except audit.ForkStateMismatch as exc:
        if message in str(exc):
            return str(exc)
        raise AssertionError(f"Rejected for an unexpected reason: {exc}") from exc
    raise AssertionError(f"Accepted the expected failure: {message}")


def negative_controls(public):
    a, b = fixture(public), fixture(public)
    b["controller_history"] = a["controller_history"]
    reports = dict(shared_mutable_history=reject(a, b, "shared across parent and child"))

    a, b = fixture(public), fixture(public)
    del b["frame"].robot_reference21
    reports["dynamic_public_field_missing"] = reject(a, b, "stored fields")

    a, b = fixture(public), fixture(public)
    b["frame"].robot_reference21[0] += .001
    reports["dynamic_public_value_changed"] = reject(a, b, "array layout, writeability or value changed")

    a, b = fixture(public), fixture(public)
    native = np.array(a["frame"].robot_reference21)
    a["frame"].robot_reference21 = native[:]
    b["frame"].robot_reference21 = native.copy()[:]
    reports["dynamic_native_view"] = reject(a, b, "aliases an exported native buffer",
                                             left_native=((native, "synthetic_native"),))

    a, b = fixture(public), fixture(public)
    b["late"] = b["backing"].copy()[4:]
    assert not np.shares_memory(a["early"], a["late"])
    reports["disjoint_sibling_detached"] = reject(a, b, "alias topology changed")

    a, b = fixture(public), fixture(public)
    b["read_only"] = b["backing"].copy()[::2]
    b["read_only"].setflags(write=False)
    reports["readonly_view_detached"] = reject(a, b, "alias topology changed")

    a, b = fixture(public), fixture(public)
    b["empty"] = b["backing"].copy()[:0]
    reports["empty_view_detached"] = reject(a, b, "alias topology changed")

    a, b = fixture(public), fixture(public)
    a["extra"] = a["backing"].copy()
    b["extra"] = b["backing"]
    reports["distinct_owners_collapsed"] = reject(a, b, "alias topology changed")

    a, b = fixture(public), fixture(public)
    b["rng"].random()
    reports["rng_advanced"] = reject(a, b, "RNG state changed")

    a, b = fixture(public), fixture(public)
    q = b["last_receiver"]
    b["receiver_queue"] = deque((ReceiverPacket(q.time, q.wrench.copy()),), maxlen=3)
    reports["receiver_packet_alias_lost"] = reject(a, b, "alias topology changed")

    a, b = fixture(public), fixture(public)
    arrays = np.zeros(3), np.ones(3)
    reports["native_buffers_swapped"] = reject(
        a, b, "native buffer shared", left_native=zip(arrays, ("x", "y")),
        right_native=zip(reversed(arrays), ("x", "y")))

    a, b = fixture(public), fixture(public)
    a["external"] = {"slot": "value"}.keys()
    b["external"] = {"slot": "value"}.keys()
    reports["unregistered_storage"] = reject(a, b, "unsupported mutable type")
    backing = np.zeros(8).tobytes()
    a, b = fixture(public), fixture(public)
    a["hard_readonly"] = np.ndarray((2, 2), dtype=float, buffer=backing)
    b["hard_readonly"] = np.ndarray((2, 2), dtype=float, buffer=backing, offset=8)
    reports["immutable_buffer_offset_changed"] = reject(a, b, "offset from immutable bytes changed")
    return reports


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha(TRACE) != TRACE_SHA:
        raise ValueError("Frozen physical input changed")
    with np.load(TRACE, allow_pickle=False) as trace:
        public = json.loads(trace["observation_json"][1855])
    assert public["time"] == 9.275 and abs(public["sample_time"] - 9.265) < 1e-9
    a, b = fixture(public), fixture(public)
    receiver_bytes = np.zeros(6).tobytes()
    a["hard_readonly"] = np.frombuffer(receiver_bytes, dtype=float).reshape((2, 3))
    b["hard_readonly"] = np.ndarray((2, 3), dtype=float, buffer=receiver_bytes)
    check = compare(a, b)
    b["backing"][0] += 10.
    assert b["read_only"][0] != a["read_only"][0]
    failures = negative_controls(public)
    result = dict(schema="v21_fragile_fork_state_software_v1", passed=True, physics_steps=0,
                  model_instances=0, sensor_samples_added=0, real_holding_copy_validated=False,
                  origin_s=9.280, last_public_read_s=public["time"],
                  last_public_acquisition_s=public["sample_time"], archive_sha256=TRACE_SHA,
                  positive=dict(checked_paths=check.paths, arrays=len(check.arrays),
                                mutation_isolated=True, shared_child_views_preserved=True,
                                immutable_receiver_bytes_reshape_supported=True),
                  rejected_controls=failures,
                  source_sha256={Path(p).name: sha(Path(p)) for p in (__file__, audit.__file__)})
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(passed=True, paths=check.paths, negatives=len(failures))))


if __name__ == "__main__":
    main()

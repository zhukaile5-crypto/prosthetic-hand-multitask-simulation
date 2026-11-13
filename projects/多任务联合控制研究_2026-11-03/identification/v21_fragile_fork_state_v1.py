"""Private start-state audit for the isolated V21 holding experiment.

Use after the existing opaque fork. This checks selected Python histories and
exported buffer ownership, without sampling, integrating or returning state to
the policy. A full unarmed continuation comparison is a separate requirement.
"""

from collections import deque
from dataclasses import is_dataclass
from enum import Enum
import inspect
import struct

import numpy as np

from v25_response_identification.forks import (
    _boundary, _native_buffers, _native_overlaps, _walk,
)

SCHEMA = "v21_fragile_fork_state_v1"
ATOMIC = (str, bytes, bool, int, type(None))
PLANT_STATE = (
    "control_steps", "mode_generation", "_joint_committed", "_joint_mode",
    "_joint_anchor_time", "_joint_anchor_reference", "joint_release_entry",
    "last_execution_receipt21", "a1_last_engine_port21", "lift_ref", "wrist_ref",
    "release_started", "phase",
)
RELEASE_MEMORY = (
    "_samples", "_order", "_prior_history", "_last_tick", "_last_time",
    "_last_signature", "hand_sample", "release_request_s", "pre_release_prior",
    "_motion_samples", "_motion_order", "_motion_last_tick", "_motion_signature", "_motion_task",
)


class ForkStateMismatch(ValueError):
    """A state or buffer ownership invariant was violated."""


def require(ok, path, reason):
    if not ok:
        raise ForkStateMismatch(f"{path}: {reason}")


def instance_fields(value):
    """Read stored fields, including attached data and populated slots."""
    result = dict(getattr(value, "__dict__", {}))
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = "_" + cls.__name__.lstrip("_") + name
            if hasattr(value, name):
                result[name] = getattr(value, name)
    return result


def pointer(array):
    return array.__array_interface__["data"][0]


def immutable_byte_backing(array):
    """Identify hard read-only storage, distinct from a read-only flag."""
    current = array
    while type(current) is np.ndarray and not current.flags.writeable:
        current = current.base
    return current if type(current) is bytes else None


class StateComparison:
    def __init__(self, *, record_types=()):
        self.record_types = record_types
        self.left = {}
        self.right = {}
        self.arrays = []
        self.paths = 0

    def bind(self, a, b, path):
        ia, ib = id(a), id(b)
        require(ia != ib and ia not in self.right and ib not in self.left,
                path, "mutable object shared across parent and child")
        if ia in self.left or ib in self.right:
            require(ia in self.left and ib in self.right and self.left[ia] is b
                    and self.right[ib] is a, path, "mutable alias topology changed")
            return False
        self.left[ia], self.right[ib] = b, a
        return True

    def compare(self, a, b, path):
        cls = type(a)
        self.paths += 1
        require(cls is type(b), path, "type changed")
        if cls in ATOMIC or isinstance(a, Enum):
            require(a == b, path, "scalar changed")
            return
        if cls is float:
            require(struct.pack("d", a) == struct.pack("d", b), path, "float bytes changed")
            return
        if isinstance(a, np.generic):
            require(a.dtype.kind in "biufc" and a.tobytes() == b.tobytes(),
                    path, "unsupported or changed NumPy scalar")
            return
        if cls in (tuple, frozenset):
            if cls is frozenset:
                require(all(type(x) in ATOMIC for x in a | b) and a == b,
                        path, "unsupported or changed immutable set")
                return
            self.sequence(a, b, path)
            return

        if not self.bind(a, b, path):
            return
        if cls is np.ndarray:
            self.array(a, b, path)
        elif cls is dict:
            self.mapping(a, b, path)
        elif cls in (list, deque):
            require(cls is list or a.maxlen == b.maxlen, path, "queue capacity changed")
            self.sequence(a, b, path)
        elif cls is set:
            require(all(type(x) in ATOMIC for x in a | b) and a == b,
                    path, "unsupported or changed set")
        elif cls is np.random.Generator:
            require(type(a.bit_generator) is type(b.bit_generator) is np.random.PCG64,
                    path, "expected the registered PCG64 generator")
            self.bind(a.bit_generator, b.bit_generator, path + ".bit_generator")
            require(a.bit_generator.state == b.bit_generator.state, path, "RNG state changed")
        elif is_dataclass(a) or cls in self.record_types:
            self.mapping(instance_fields(a), instance_fields(b), path)
        else:
            raise ForkStateMismatch(f"{path}: unsupported mutable type {cls.__module__}.{cls.__qualname__}")

    def sequence(self, a, b, path):
        require(len(a) == len(b), path, "sequence size changed")
        for i, (x, y) in enumerate(zip(a, b)):
            self.compare(x, y, f"{path}[{i}]")

    def mapping(self, a, b, path):
        ka, kb = tuple(a), tuple(b)
        allowed = (str, int, bool)
        require(all(type(k) in allowed for k in ka + kb), path, "unsupported dictionary key")
        require(tuple((type(k), k) for k in ka) == tuple((type(k), k) for k in kb),
                path, "stored fields or their order changed")
        for k in ka:
            self.compare(a[k], b[k], f"{path}.{k}")

    def array(self, a, b, path):
        require(a.dtype.kind in "biufc" and a.dtype.metadata is None and b.dtype.metadata is None,
                path, "expected a plain numeric or Boolean array")
        require(a.dtype == b.dtype and a.shape == b.shape and a.strides == b.strides
                and a.flags.writeable == b.flags.writeable and a.flags.owndata == b.flags.owndata
                and a.flags.aligned == b.flags.aligned and a.tobytes() == b.tobytes(),
                path, "array layout, writeability or value changed")
        self.arrays.append((a, b, path))
        bytes_a, bytes_b = immutable_byte_backing(a), immutable_byte_backing(b)
        if bytes_a is not None or bytes_b is not None:
            # The original receiver uses frombuffer(...).reshape(...), while
            # the frozen copy can rebuild the view directly on immutable bytes.
            # No alias in this chain can become writable.
            require(bytes_a is not None and bytes_b is not None and bytes_a == bytes_b,
                    path, "immutable byte storage changed")
            require(pointer(a) - pointer(np.frombuffer(bytes_a, dtype=np.uint8))
                    == pointer(b) - pointer(np.frombuffer(bytes_b, dtype=np.uint8)),
                    path, "array offset from immutable bytes changed")
            return

        base_a, base_b = a.base, b.base
        require(type(base_a) is type(base_b), path, "array backing type changed")
        if base_a is None:
            require(a.flags.owndata and b.flags.owndata, path, "unexpected unowned base")
            return
        require(type(base_a) is np.ndarray, path, f"unsupported array backing {type(base_a).__name__}")
        require(pointer(a) - pointer(base_a) == pointer(b) - pointer(base_b),
                path, "array offset from its backing changed")
        self.compare(base_a, base_b, path + ".base")


def index_arrays(arrays):
    """Index the bounding intervals used by the exact overlap checker."""
    entries = []
    for array, path in arrays:
        if array.size:
            low, high = np.lib.array_utils.byte_bounds(array)
            entries.append((low, high, array, path, "array"))
    entries.sort(key=lambda item: item[0])
    highs = np.maximum.accumulate([item[1] for item in entries]).tolist()
    return entries, [item[0] for item in entries], highs


def check_exported_buffers(comparison, parent, child):
    """Check cached Python views and cross-arena native and Python storage."""
    parent_python = index_arrays((a, path) for a, _, path in comparison.arrays)
    for left, right, path in comparison.arrays:
        require(not any(_native_overlaps(x, y) for x in (left, right) for y in (parent, child)),
                path, "Python state aliases an exported native buffer")
        require(immutable_byte_backing(right) is not None or not _native_overlaps(right, parent_python), path,
                "Python array memory shared across parent and child")
    for _, _, array, path, field in child[0]:
        require(not _native_overlaps(array, parent), f"{path}.{field}",
                "native buffer shared across parent and child")


def selected_holding_state(session):
    """Return explicit reviewed roots. Opaque native owners remain outside."""
    controller = session.controller
    wrapper = session.sensors
    receiver = session.receiver_sensor
    skin = session.skin_geometry
    release = session.release_adapter
    plant = session.plant
    require(receiver._force_reader is None, "receiver", "external receiver callback unsupported")
    require(plant.actor21 is plant.a1_actor, "plant.actor21", "wrong native actor binding")
    require(plant._joint_committed is plant.last_execution_receipt21.executed,
            "plant.last_execution_receipt21", "committed reference and receipt are not bound")
    require(session.preparation.plant is plant and skin.model is receiver._model is release._model is plant.model,
            "native_owner", "incorrect plant or instrument binding")
    require(type(wrapper._sample_signature) is inspect.Signature
            and tuple(wrapper._sample_signature.parameters.items())
            == tuple(inspect.signature(wrapper.inner.sample).parameters.items()),
            "hand_sensor", "sampling signature changed")
    require(session.latest_receiver_measurement is receiver._last, "receiver._last",
            "session and receiver cache do not share the last packet")
    require(session.last_obs is not None and controller.state._last_time is not None,
            "last_obs", "a populated public holding history is required")
    require(all(abs(t - (session.time - .005)) <= 1e-9 for t in (
        session.last_obs.time, wrapper._last_time, wrapper.inner.last_time, controller.state._last_time,
        receiver._last_tick * .005, release._last_time, release._motion_last_tick * .005)),
        "pre_observe", "last acquisition or state update is not from the previous control tick")
    roots = {
        "controller": {k: v for k, v in vars(controller).items()
                       if k not in ("geometry", "native", "decision_evidence")},
        "controller_decision_count": len(controller.decision_evidence),
        "hand_sensor": wrapper.inner,
        "hand_sensor_motion_domain": (wrapper._domain_history, wrapper._last_time, wrapper._last_frame),
        "receiver": {k: v for k, v in vars(receiver).items() if k not in ("_model", "_force_reader")},
        "skin": (skin.qadr, skin.lift_qadr, skin.palm, skin.samples),
        "last_obs": session.last_obs,
        "latest_receiver_measurement": session.latest_receiver_measurement,
        "release_memory": tuple(getattr(release, name) for name in RELEASE_MEMORY),
        "preparation": {k: v for k, v in vars(session.preparation).items() if k != "plant"},
        "actor": plant.actor21,
        "execution": tuple(getattr(plant, name) for name in PLANT_STATE),
    }
    classes = tuple(type(x) for x in (controller.state, controller.intent, controller.group_belief,
                                      wrapper.inner, plant.actor21) if x is not None)
    return roots, classes


def audit_holding_copy(parent, child):
    """Audit a real pair already produced by the opaque copy coordinator."""
    from identification.v21_fragile_reference_runtime_v1 import HoldingReferenceTrialSession, ORIGIN_S

    require(type(parent) is type(child) is HoldingReferenceTrialSession, "session", "wrong runtime")
    require(_boundary(parent) == _boundary(child) == ORIGIN_S, "session", "wrong paired origin")
    for session in (parent, child):
        require(session._holding_program_gate is None and session.scenario_name == "fragile_conflict"
                and session.method == "ratio_effort" and session.release_variant == "v21"
                and session.task_policy == "integrated" and session.v16_release_policy == "coordinated"
                and session.running and not session.paused and session.mode == "scheduled"
                and session.motion_mode in ("hold", "transport") and session.handoff_passed
                and session.handoff_attempted and session.plant.phase == "HOLDING"
                and session.plant.release_started is None and session.plant.joint_release_entry is None
                and session.plant._joint_committed is not None and session.plant._joint_mode == "holding"
                and not session.manual_release,
                "session", "expected an unarmed V21 fragile holding checkpoint")
    require((parent.seed, parent.motion_mode) == (child.seed, child.motion_mode), "session", "different assignment")
    left, types = selected_holding_state(parent)
    right, right_types = selected_holding_state(child)
    require(types == right_types, "session", "different concrete record types")
    comparison = StateComparison(record_types=types)
    comparison.compare(left, right, "holding")
    parent_buffers = _native_buffers(_walk(parent))
    child_buffers = _native_buffers(_walk(child))
    check_exported_buffers(comparison, parent_buffers, child_buffers)
    return dict(schema=SCHEMA, passed=True, origin_s=ORIGIN_S, checked_paths=comparison.paths,
                selected_arrays=len(comparison.arrays), selected_state_roots=list(left),
                parent_exported_buffer_count=len(parent_buffers[0]),
                child_exported_buffer_count=len(child_buffers[0]),
                physics_steps=0, sensor_samples_added=0, full_continuation_verified=False)

"""Causal evidence for scheduling optional withdrawal at the receiver.

Compiled collision topology can rule out a direct robot-to-receiver contact
in the current simulation. The reaction is still any-actor and delayed. The
dimensionless progress score is a scheduler input, not an action or safety
certificate. No target identity, weight truth, contact data or dynamics used.
"""

from collections import deque
from dataclasses import dataclass
from numbers import Real

import numpy as np
import mujoco

PERIOD_S = .005
DELAY_S = .010
TIMING_TOL = 1e-9
SUPPORT_DWELL_S = .020
REQUIRED_SAMPLES = round(SUPPORT_DWELL_S / PERIOD_S) + 1


@dataclass(frozen=True)
class ReceiverTopology:
    robot_direct_contact_excluded: bool
    robot_geom_count: int
    receiver_candidate_count: int
    reason: str


def receiver_topology(model, receiver_name="support", robot_root_name="rh_forearm"):
    """Only named robot/receiver, compiled pair masks and explicit pairs."""
    root = int(model.body(robot_root_name).id)
    receiver = int(model.geom(receiver_name).id)
    if (root <= 0 or int(model.geom_bodyid[receiver]) != 0
            or int(model.geom_type[receiver]) != int(mujoco.mjtGeom.mjGEOM_BOX)):
        raise ValueError("Expected a registered static box and a nonworld robot root")
    belongs = np.zeros(model.nbody, dtype=bool)
    belongs[root] = True
    for body in range(root + 1, model.nbody):
        parent = int(model.body_parentid[body])
        if parent < 0 or parent >= body:
            raise ValueError("Unrecognized compiled body tree")
        belongs[body] = belongs[parent]
    robot = belongs[np.asarray(model.geom_bodyid)]
    if not robot.any():
        raise ValueError("No compiled robot geometry")
    types = np.asarray(model.geom_contype)
    affinities = np.asarray(model.geom_conaffinity)
    eligible = ((types & affinities[receiver]) != 0) | ((affinities & types[receiver]) != 0)
    for a, b in zip(model.pair_geom1, model.pair_geom2):
        if a == receiver:
            eligible[b] = True
        if b == receiver:
            eligible[a] = True
    eligible[receiver] = False
    if int(model.nplugin) or mujoco.get_mjcb_contactfilter() is not None:
        return ReceiverTopology(False, int(robot.sum()), int(eligible.sum()), "external_contact_extension")
    excluded = not np.any(eligible & robot)
    return ReceiverTopology(excluded, int(robot.sum()), int(eligible.sum()),
                            "compiled_masks_and_pairs_exclude_direct_robot" if excluded else "direct_robot_pair_possible")


@dataclass(frozen=True)
class TransferReading:
    read_s: float
    acquisition_s: float
    receiver_up_N: float
    receiver_width_N: float
    hand_up_N: float
    hand_width_N: float
    demand_upper_N: float


def number(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value)


def score_reading(reading):
    """Retain signed hand force when accounting for receiver loading."""
    r = reading
    values = (r.read_s, r.acquisition_s, r.receiver_up_N, r.receiver_width_N,
              r.hand_up_N, r.hand_width_N, r.demand_upper_N)
    if (not all(number(x) for x in values) or r.acquisition_s < 0.
            or abs(r.read_s - r.acquisition_s - DELAY_S) > TIMING_TOL
            or min(r.receiver_width_N, r.hand_width_N) < 0. or r.demand_upper_N <= 0.):
        raise ValueError("Invalid or non-coacquired public transfer measurement")
    hand_lower = r.hand_up_N - r.hand_width_N
    downward = r.hand_up_N + r.hand_width_N < 0.
    effective = max(0., r.receiver_up_N - r.receiver_width_N - max(0., -hand_lower))
    return min(1., effective / r.demand_upper_N), downward


class TransferEvidenceWindow:
    """Consecutive causal readouts, without accumulating stale permission."""

    def __init__(self):
        self.window = deque(maxlen=REQUIRED_SAMPLES)
        self.last_tick = None
        self.last_reading = None
        self.last_topology = None
        self.last_result = None

    def update(self, now, reading, topology):
        if not number(now) or now < 0.:
            raise ValueError("Explicit nonnegative control time required")
        tick = round(now / PERIOD_S)
        if abs(now - tick * PERIOD_S) > TIMING_TOL:
            raise ValueError("Transfer clock is off the 200 Hz public grid")
        if self.last_tick is not None and tick <= self.last_tick:
            if tick == self.last_tick and (reading, topology) == (self.last_reading, self.last_topology):
                return self.last_result.copy()
            raise ValueError("Changed or rewound public transfer tick")
        if type(topology) is not ReceiverTopology or reading is not None and type(reading) is not TransferReading:
            raise ValueError("Typed public transfer reading and topology required")
        score = None
        downward = None
        if reading is not None and topology.robot_direct_contact_excluded:
            score, downward = score_reading(reading)
            if abs(now - reading.read_s) > TIMING_TOL:
                raise ValueError("Wrong public read time")
        if (self.last_tick is not None and tick != self.last_tick + 1) or topology != self.last_topology:
            self.window.clear()
        self.last_tick, self.last_reading, self.last_topology = tick, reading, topology
        if score is None or downward or score <= 0.:
            self.window.clear()
        else:
            self.window.append(score)
        complete = len(self.window) == REQUIRED_SAMPLES
        self.last_result = dict(current_proxy_fraction=score, hand_downward_resolved=downward,
                                consecutive_positive_samples=len(self.window), window_complete=complete,
                                sustained_proxy_fraction=min(self.window) if complete else None,
                                actor_identity_resolved=False, scheduling_only=True)
        return self.last_result.copy()

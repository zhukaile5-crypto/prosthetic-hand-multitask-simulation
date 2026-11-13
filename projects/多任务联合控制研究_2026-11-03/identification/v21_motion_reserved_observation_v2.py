"""Bind typed public load samples to the causal withdrawal evidence window."""

from collections.abc import Mapping

import numpy as np

from identification.v21_receiver_transfer_evidence_v1 import (
    DELAY_S, PERIOD_S, TIMING_TOL, TransferEvidenceWindow, TransferReading,
    number, receiver_topology,
)
from v15_support.receiver_sensor import ReceiverMeasurement
from v16_release_geometry.transfer_observer import HandSupportSample, PreReleaseLoadPrior


def _tick(value):
    if not number(value) or value < 0.:
        raise ValueError("Expected a finite nonnegative public read clock")
    tick = round(value / PERIOD_S)
    if abs(value - tick * PERIOD_S) > TIMING_TOL:
        raise ValueError("Public read clock is off the 200 Hz grid")
    return tick


class SupportObservation:
    """One release episode, using only the public hand and receiver streams.

    A known-clock miss consumes a tick without evidence. An invalid control
    clock or conflicting duplicate raises and invalidates the current window.
    The caller must treat that error as unavailable scheduling evidence.
    """

    def __init__(self, model, receiver_up_axis3, prior, release_request_s):
        self._request_tick = _tick(release_request_s)
        self._request_s = self._request_tick * PERIOD_S
        raw_up = np.asarray(receiver_up_axis3, dtype=object)
        if raw_up.shape != (3,) or not all(number(x) for x in raw_up):
            raise ValueError("Expected a finite public receiver-up axis")
        up = np.asarray(raw_up, dtype=float)
        scale = float(np.max(abs(up)))
        if scale == 0.:
            raise ValueError("Expected a nonzero public receiver-up axis")
        up = up / scale
        self._up = up / np.linalg.norm(up)

        if prior is not None and type(prior) is not PreReleaseLoadPrior:
            raise TypeError("Expected a typed public pre-release prior or None")
        self._demand_upper_N = None
        if (prior is not None and prior.reference_time_s <= self._request_s + TIMING_TOL
                and prior.value_N > prior.uncertainty_N):
            upper = prior.value_N + prior.uncertainty_N
            if np.isfinite(upper):
                self._demand_upper_N = upper

        self._model = model
        self._topology = receiver_topology(model)
        self._window = TransferEvidenceWindow()
        self._latest_tick = None

    @property
    def robot_direct_contact_excluded(self):
        """Topology admission from the constructor or most recent update."""
        return self._topology.robot_direct_contact_excluded

    def _reading(self, now, inputs):
        if self._demand_upper_N is None:
            return None
        hand, receiver = inputs.get("hand"), inputs.get("receiver")
        if (type(hand) is not HandSupportSample or type(receiver) is not ReceiverMeasurement
                or not hand.valid or not receiver.valid):
            return None
        acq = hand.acquisition_time_s
        receiver_acq = receiver.acquisition_time_s
        if (acq is None or receiver_acq is None or receiver.age_s is None
                or acq < self._request_s - TIMING_TOL
                or abs(acq - receiver_acq) > TIMING_TOL
                or abs(hand.read_time_s - now) > TIMING_TOL
                or abs(receiver.time_s - now) > TIMING_TOL
                or abs(receiver.age_s - DELAY_S) > TIMING_TOL
                or abs(now - acq - DELAY_S) > TIMING_TOL):
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            reaction = -float(receiver.wrench6[:3] @ self._up)
            width = float(receiver.uncertainty6[:3] @ abs(self._up))
        if not np.isfinite(reaction) or not np.isfinite(width):
            return None
        return TransferReading(now, acq, reaction, width, hand.signed_support_N,
                               hand.uncertainty_N, self._demand_upper_N)

    def update(self, phase, transfer_inputs):
        try:
            if not isinstance(transfer_inputs, Mapping):
                raise ValueError("Expected public transfer inputs with an explicit read clock")
            tick = _tick(transfer_inputs.get("time_s"))
            if tick < self._request_tick or (self._latest_tick is not None and tick < self._latest_tick):
                raise ValueError("Read before release or rewound public transfer clock")
            if tick == self._latest_tick and self._window.last_tick is None:
                raise ValueError("An invalidated public tick cannot restore evidence")
            now = tick * PERIOD_S
            # Recheck the compiled masks, pairs and callback admission on each
            # update. No contact result or live simulation data is inspected.
            self._topology = receiver_topology(self._model)
            reading = self._reading(now, transfer_inputs) if phase == "TRANSFER" else None
            result = self._window.update(now, reading, self._topology)
        except (ValueError, TypeError, OverflowError):
            self._window = TransferEvidenceWindow()
            raise
        self._latest_tick = tick
        return dict(result, robot_direct_contact_excluded=self.robot_direct_contact_excluded)

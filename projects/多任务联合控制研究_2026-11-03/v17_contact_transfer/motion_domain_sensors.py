"""Validity metadata for the EXISTING delayed slip instrument; no new draws.

The inner sensor remains the sole producer of every original SensorFrame
field and of its 48 Gaussian/6 uniform draws. This wrapper exposes whether
that instrument actually retained a slip sample, rather than zero padding.
It does not identify the contacted actor or certify a surface tangent plane.
"""
from collections import deque
from copy import deepcopy
import inspect
from numbers import Real
import numpy as np
import research_imports  # Frozen articulated-hand sensor module resolution.
from sensors import HandSensors, aggregate_patches

SLIP_DOMAIN_ROLE = 'VALIDITY_OF_EXISTING_INSTRUMENT_SLIP_CHANNEL'
ADDED_FIELDS = ('slip_valid5', 'slip_domain_acquisition_time_s',
                'slip_domain_time_s', 'slip_domain_role',
                'slip_domain_available', 'slip_domain_reason')


def _time(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative public sensor time required')
    return float(value)


class MotionDomainSensors:
    """Wrap one fresh or already-owned HandSensors without resetting it.

    sample has the same accepted arguments as the wrapped bound method and
    forwards them verbatim. Duplicate ticks do not call inner.sample again.
    On an already-advanced inner sensor, absent wrapper acquisition history is
    explicitly unavailable; it is never inferred from the noisy pressure.
    """
    def __init__(self, innerHandSensors):
        if not isinstance(innerHandSensors, HandSensors):
            raise TypeError('An owned HandSensors instance is required')
        self.inner = innerHandSensors
        self._sample_signature = inspect.signature(self.inner.sample)
        self._domain_history = deque(maxlen=102)
        self._last_time = None
        self._last_frame = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def reset(self):
        reset = getattr(self.inner, 'reset', None)
        if not callable(reset):
            raise RuntimeError('Inner sensor has no reset; create a new owner')
        reset()
        self._domain_history.clear()
        self._last_time = None
        self._last_frame = None

    def sample(self, raw, *args, **kwargs):
        bound = self._sample_signature.bind(raw, *args, **kwargs)
        override = bound.arguments.get('time')
        now = _time(raw['time'] if override is None else override)
        if self._last_time is not None and abs(now-self._last_time) < 1e-10:
            return deepcopy(self._last_frame)
        # The same all-external aggregate and threshold as inner._sample.
        # diagnostic target-specific fields, if any, are neither read nor
        # published. No target/actor identity is used to select patches.
        measured, _ = aggregate_patches(raw.get('patches', raw.get('contacts', [])),
            self.inner.noise.aggregation_tolerance, support_axis=raw['support_axis'])
        retained = np.asarray(measured['normal'] > .02, bool)
        frame = self.inner.sample(raw, *args, **kwargs)  # exactly one call/new tick
        self._domain_history.append((now, retained.copy()))
        acquisition = _time(frame.sample_time)
        hits = [mask for t, mask in self._domain_history if abs(t-acquisition) < 1e-9]
        available = len(hits) == 1
        valid = np.asarray(frame.valid)
        if valid.shape != (5,) or valid.dtype != np.dtype(bool):
            raise ValueError('Inner frame must retain its explicit valid5 mask')
        frame.slip_valid5 = (hits[0] & valid) if available else np.zeros(5, bool)
        frame.slip_domain_acquisition_time_s = acquisition
        frame.slip_domain_time_s = _time(frame.time)
        frame.slip_domain_role = SLIP_DOMAIN_ROLE
        frame.slip_domain_available = available
        frame.slip_domain_reason = (None if available else 'instrument_acquisition_domain_history_missing')
        self._last_time = now
        self._last_frame = deepcopy(frame)
        return frame

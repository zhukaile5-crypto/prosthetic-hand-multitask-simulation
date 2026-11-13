"""Predeclared five-arm software/identification contract, not a safety model."""
from dataclasses import dataclass
from numbers import Real
import numpy as np

CONTROL_PERIOD_S = .005
PROBE_HOLD_S = .200
AMPLITUDE_FRACTION = .5
SOURCE_NAMES = ('__init__.py', 'design.py', 'observer.py', 'runtime.py', 'forks.py', 'run_identification.py')
PROBE_ARMS = ('zero', 'lift_plus', 'lift_minus', 'wrj1_plus', 'wrj1_minus')
ORIGIN_TRIGGERS = ('pre_receiver', 'receiver_transfer', 'unload_window')
SCHEMA = 'v25_response_identification_v1'
TIMING_COLUMNS = ('control_ms', 'actor_callback_ms', 'full_control_loop_ms', 'observation_ms', 'physics_advance_ms')
DEBUG_TIMING_KEYS = ('qp', 'native_inverse', 'whole_decision')


@dataclass(frozen=True)
class ProbeSpec:
    arm: str
    origin_s: float
    hold_s: float = PROBE_HOLD_S
    amplitude_fraction: float = AMPLITUDE_FRACTION

    def __post_init__(self):
        if self.arm not in PROBE_ARMS:
            raise ValueError('Only the registered zero/lift±/WRJ1± five arms are implemented')
        for name in ('origin_s', 'hold_s', 'amplitude_fraction'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(name+' must be finite real')
            object.__setattr__(self, name, float(value))
        if self.origin_s < 0 or abs(self.origin_s / CONTROL_PERIOD_S-round(self.origin_s / CONTROL_PERIOD_S)) > 1e-8:
            raise ValueError('Probe origin must be a nonnegative5ms PRE_OBSERVE boundary')
        if self.hold_s != PROBE_HOLD_S or self.amplitude_fraction != AMPLITUDE_FRACTION:
            raise ValueError('Changed amplitude/horizon requires a new registered protocol')

    @property
    def axis(self):
        return None if self.arm == 'zero' else 18 if self.arm.startswith('lift_') else 20

    @property
    def sign(self):
        return 0 if self.arm == 'zero' else 1 if self.arm.endswith('_plus') else -1

"""Public causal interfaces for the isolated articulated-hand integration.

Contact geometry is an idealized virtual tactile measurement, not an object
state oracle. All vector geometry and forces in this contract use the palm
frame. ``force_gain`` is a positive numerical scale only; the signed, coupled
response is ``force_matrix`` and must never be replaced by abs(diag(B)).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

BENCHMARK_ID = 'shadow_hand_strategy_integration_v1'
FINGER_NAMES = ('拇指', '食指', '中指', '无名指', '小指')
SENSOR_PERIOD = .005
DECISION_PERIOD = .020
WRENCH_LENGTH = .050


@dataclass(frozen=True)
class SensorNoise:
    force_sigma: float = .0
    tangential_sigma: float = .0
    position_sigma: float = .0
    slip_sigma: float = .0
    effort_sigma: float = .0
    auxiliary_force_sigma: float | None = None
    auxiliary_moment_sigma: float | None = None
    auxiliary_dropout_probability: float | None = None
    auxiliary_dropout_windows: tuple = ()
    instrument_auxiliary: bool = True
    delay: float = .0
    dropout_probability: float = .0
    # (start_s, end_s, tuple of sensor digit indices); all use simulation time.
    dropout_windows: tuple = ()
    aggregation_tolerance: float = .10
    strict_aggregation: bool = True
    geometry_mode: str = 'distributed_wrench'

    def __post_init__(self):
        for name in ('force_sigma', 'tangential_sigma', 'position_sigma',
                     'slip_sigma', 'effort_sigma', 'delay'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0.:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0. <= self.dropout_probability <= 1.:
            raise ValueError('dropout_probability must be in [0,1]')
        for name in ('auxiliary_force_sigma', 'auxiliary_moment_sigma'):
            value=getattr(self,name)
            if value is not None and (not np.isfinite(value) or value<0.):
                raise ValueError(f'{name} must be None or finite/nonnegative')
        if self.auxiliary_dropout_probability is not None and not 0. <= self.auxiliary_dropout_probability <= 1.:
            raise ValueError('auxiliary_dropout_probability must be in [0,1]')
        for start,end in self.auxiliary_dropout_windows:
            if not (np.isfinite(start) and np.isfinite(end) and end>start):
                raise ValueError('auxiliary dropout windows require finite increasing times')
        if not 0. < self.aggregation_tolerance <= 1.:
            raise ValueError('aggregation_tolerance must be in (0,1]')
        if self.geometry_mode not in ('single_point','distributed_wrench'):
            raise ValueError('unknown contact geometry representation')
        if self.delay > .5:
            raise ValueError('delay exceeds the declared 0.5 s sensor buffer')
        for start, end, fingers in self.dropout_windows:
            if not (np.isfinite(start) and np.isfinite(end) and end > start):
                raise ValueError('dropout window must have increasing finite times')
            if any(int(i) != i or not 0 <= i < 5 for i in fingers):
                raise ValueError('dropout fingers must be indices 0..4')


@dataclass
class SensorFrame:
    time: float
    normal: np.ndarray
    tangential: np.ndarray
    contact_pos: np.ndarray
    contact_normal: np.ndarray
    tangent_basis: np.ndarray
    slip_local: np.ndarray
    joint: np.ndarray
    joint_velocity: np.ndarray
    valid: np.ndarray
    age: float
    alpha: np.ndarray
    phase: str
    lift: float
    lift_ref: float
    support_axis: np.ndarray
    raw_joint: np.ndarray
    raw_joint_velocity: np.ndarray
    force_gain: np.ndarray
    closure_permission: np.ndarray
    closure_max: np.ndarray
    closure_min: np.ndarray
    nominal_force_weights: np.ndarray
    motor_effort: float
    handload_prior: float
    load_estimate: float
    load_uncertainty: float
    load_valid: bool
    load_confidence: float
    force_tau: float
    force_matrix: np.ndarray
    model_reference_alpha: np.ndarray
    response_error_bound: np.ndarray
    aggregation_valid: np.ndarray
    aggregation_error: np.ndarray
    sample_time: float
    dt: float = SENSOR_PERIOD
    force_gain_role: str = 'positive_normalization_scale_not_signed_response'
    auxiliary_wrench: np.ndarray | None = None
    auxiliary_pressure: float = 0.
    auxiliary_valid: bool = False
    auxiliary_age: float = 0.
    auxiliary_uncertainty: float = float('inf')
    normal_wrench_map: np.ndarray | None = None
    finger_non_normal_wrench: np.ndarray | None = None
    finger_non_normal_by_finger: np.ndarray | None = None
    shear_magnitude: np.ndarray | None = None
    slip_magnitude: np.ndarray | None = None
    normal_support_coeff: np.ndarray | None = None
    tangent_support_coeff: np.ndarray | None = None
    distributed_valid: np.ndarray | None = None
    distributed_error: np.ndarray | None = None
    contact_present: np.ndarray | None = None


def tangent_basis(normal):
    n = np.asarray(normal, dtype=float)
    length = np.linalg.norm(n)
    if length < 1e-12:
        return np.array([[0., 0., 1.], [0., 1., 0.]])
    n = n / length
    anchor = np.array([0., 0., 1.])
    if abs(anchor @ n) > .9:
        anchor = np.array([0., 1., 0.])
    tangent = anchor - (anchor @ n) * n
    tangent /= np.linalg.norm(tangent)
    return np.stack((tangent, np.cross(n, tangent)))


def vector(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must have finite shape {shape}; got {result.shape}')
    return result.copy()


def public_calibration(calibration):
    """Require an explicit matrix calibration, never use five-pad defaults."""
    c = dict(calibration)
    aliases = {'reference_alpha': 'model_reference_alpha',
               'response_error_bound_N': 'response_error_bound'}
    for source, dest in aliases.items():
        if source in c and dest not in c:
            c[dest] = c[source]
    B = vector(c['force_matrix'], (5, 5), 'force_matrix')
    tau = float(c['force_tau'])
    if not np.isfinite(tau) or tau <= 0.:
        raise ValueError('force_tau must be an explicit positive public calibration')
    alpha = vector(c['model_reference_alpha'], (5,), 'model_reference_alpha')
    error = vector(c['response_error_bound'], (5,), 'response_error_bound')
    if np.any(error < 0.):
        raise ValueError('response_error_bound must be nonnegative')
    scale = vector(c.get('force_gain', np.maximum(.05, np.linalg.norm(B, axis=0))),
                   (5,), 'force_gain normalization scale')
    if np.any(scale <= 0.):
        raise ValueError('normalization scales must be positive; signed response belongs in B')
    weights = vector(c.get('nominal_force_weights', np.ones(5)), (5,),
                     'nominal_force_weights')
    if np.any(weights <= 0.):
        raise ValueError('nominal_force_weights must be positive')
    c.update(force_matrix=B, force_tau=tau, model_reference_alpha=alpha,
             response_error_bound=error, force_gain=scale,
             nominal_force_weights=weights / np.mean(weights))
    return c

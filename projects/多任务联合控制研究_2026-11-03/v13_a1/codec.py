"""Typed A1 robot references and the sole robot21 engine-port writer.

This module encodes references; it does not solve force tasks, predict contacts,
change native PD/damping/clipping, or add independent actuator actions.
"""
from dataclasses import dataclass
from numbers import Real
import numpy as np
from v11.robot_servo_effort import ACTUATORS21

REFERENCE_AXES21 = ACTUATORS21[:18] + ('lift_reference_m', 'WRJ2_reference_rad', 'WRJ1_reference_rad')

def finite_array(value, shape, name):
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in raw.flat):
        raise ValueError(name + ' requires real numeric ' + str(shape))
    result = np.asarray(value, dtype=float).copy()
    if not np.isfinite(result).all():
        raise ValueError(name + ' must be finite')
    result.setflags(write=False)
    return result

def finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(name + ' must be finite real')
    return float(value)

def _axes(value, expected, name):
    if isinstance(value, (str, bytes)) or tuple(value) != tuple(expected):
        raise ValueError(name + ' has unknown/reordered axes')
    return tuple(value)

@dataclass(frozen=True)
class Reference21:
    values: np.ndarray
    time_s: float
    axes: tuple = REFERENCE_AXES21

    def __post_init__(self):
        object.__setattr__(self, 'values', finite_array(self.values, (21,), 'Reference21'))
        object.__setattr__(self, 'time_s', finite_scalar(self.time_s, 'public_control_time_s'))
        if self.time_s < 0: raise ValueError('public_control_time_s must be nonnegative')
        object.__setattr__(self, 'axes', _axes(self.axes, REFERENCE_AXES21, 'Reference21'))

@dataclass(frozen=True)
class EnginePort21:
    values: np.ndarray
    time_s: float
    lift_bias_m: float
    axes: tuple = ACTUATORS21

    def __post_init__(self):
        object.__setattr__(self, 'values', finite_array(self.values, (21,), 'EnginePort21'))
        object.__setattr__(self, 'time_s', finite_scalar(self.time_s, 'public_control_time_s'))
        if self.time_s < 0: raise ValueError('public_control_time_s must be nonnegative')
        object.__setattr__(self, 'lift_bias_m', finite_scalar(self.lift_bias_m, 'known_robot_lift_bias_m'))
        object.__setattr__(self, 'axes', _axes(self.axes, ACTUATORS21, 'EnginePort21'))

class ReferenceActor21:
    """One robot21 writer. Out-of-range raw ports are disclosed, not reclipped.

    Original proposal clipping (including historical release behavior) is
    authoritative for A1; MuJoCo retains its original effective-control clamp.
    Adding a second clip here would change actual ctrl and old side effects.
    """
    def __init__(self, model, robot_indices21=None, lift_bias_m=None):
        aids = np.asarray([model.actuator(name).id for name in ACTUATORS21], dtype=int)
        if model.nu != 22 or len(set(aids)) != 21 or np.any(np.asarray(model.actuator_ctrlnum[aids]) != 1):
            raise ValueError('A1 requires known21 scalar robot controls plus one separate environment control')
        expected = np.asarray(model.actuator_ctrladr[aids], dtype=int)
        if robot_indices21 is not None:
            supplied = np.asarray(robot_indices21)
            if supplied.shape != (21,) or not np.array_equal(supplied, expected):
                raise ValueError('Robot control addresses do not match named ACTUATORS21')
        self.indices21 = expected.copy(); self.indices21.setflags(write=False)
        lift = int(model.actuator('lift_motor').id)
        if float(model.actuator_gainprm[lift, 0]) != 1500.:
            raise ValueError('A1 original native lift gain1500 required; no silent bias-law change')
        # Same operation order and literal denominator as the frozen writer.
        expected_bias = float(model.body_subtreemass[model.body('rh_forearm').id] * 9.81 / 1500.)
        if lift_bias_m is not None and finite_scalar(lift_bias_m, 'lift_bias_m') != expected_bias:
            raise ValueError('Caller lift bias differs from original known robot expression')
        self.lift_bias_m = expected_bias
        self.control_width = int(model.nu)
        self.ctrl_range21 = np.asarray(model.actuator_ctrlrange[expected], float).copy()
        self.ctrl_limited21 = np.asarray(model.actuator_ctrllimited[expected], bool).copy()
        self.last_engine_port21 = None
        self.last_diagnostics = None
        self.write_count = 0

    def encode(self, reference):
        if not isinstance(reference, Reference21):
            raise TypeError('Explicit Reference21 required; EnginePort21 is not an unbiased reference')
        port = reference.values.copy()
        port[18] = reference.values[18] + self.lift_bias_m
        return EnginePort21(port, reference.time_s, self.lift_bias_m)

    def decode(self, engine):
        if not isinstance(engine, EnginePort21) or engine.lift_bias_m != self.lift_bias_m:
            raise TypeError('EnginePort21 with this actor known bias required')
        reference = engine.values.copy(); reference[18] -= self.lift_bias_m
        return Reference21(reference, engine.time_s)

    def apply(self, data, reference):
        port = self.encode(reference)
        if np.shape(data.ctrl) != (self.control_width,):
            raise ValueError('Actual control buffer shape differs from bound model')
        # Sole robot21 assignment. No other input/model/physics field is read.
        data.ctrl[self.indices21] = port.values
        self.write_count += 1
        self.last_engine_port21 = port
        outside = self.ctrl_limited21 & ((port.values < self.ctrl_range21[:, 0]) | (port.values > self.ctrl_range21[:, 1]))
        self.last_diagnostics = dict(status='A1_NATIVE_REFERENCE_WRITTEN', public_control_time_s=port.time_s,
            raw_command_outside_native_ctrlrange21=outside.tolist(), native_effective_clipping_unchanged=True,
            known_lift_bias_applied_once=True, new_rate_clamp=False, no_independent_a1_actions=True,
            contact_force_prediction_available=False, physics_safety_certified=False)
        return port

"""Declared position-reference geometry, not finite servo/contact prediction.

The map distributes target-length increments in a positive robot-coordinate
metric. Force clipping does not erase that target geometry. Control clipping,
unsupported motor laws, missing geometry and rank loss remain explicit unknowns.
"""
from numbers import Real
import numpy as np


def finite_array(value, shape, name):
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in raw.flat):
        raise ValueError(name + ': explicit finite numeric shape ' + str(shape) + ' required')
    result = np.asarray(value, dtype=float).copy()
    if not np.isfinite(result).all():
        raise ValueError(name + ': nonfinite is unknown, not zero')
    result.setflags(write=False)
    return result


def _flags(value, name):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (21,) or any(not isinstance(x, (bool, np.bool_)) for x in raw):
        raise ValueError(name + ': explicit bool21 required')
    return np.asarray(value, bool)


def position_reference_map(*, transmission21x25, joint_scale25,
                           gain21x3, bias21x3, fixed_gain21, affine_bias21):
    """B=S pinv(A S), for known fixed positive position-law gain and b1=-g.

    b0/b2 are retained in native evaluation; they do not change this declared
    target-length differential. This is NOT an equal-effort inverse, an actual
    dq prediction, a force derivative, or a claim of control over the nullspace.
    """
    A = finite_array(transmission21x25, (21, 25), 'transmission')
    scale = finite_array(joint_scale25, (25,), 'positive joint metric')
    gain = finite_array(gain21x3, (21, 3), 'gain parameters')
    bias = finite_array(bias21x3, (21, 3), 'bias parameters')
    fixed = _flags(fixed_gain21, 'fixed gain mode')
    affine = _flags(affine_bias21, 'affine bias mode')
    if np.any(scale <= 0):
        raise ValueError('strictly positive robot-coordinate metric required')
    if not fixed.all() or not affine.all() or np.any(gain[:, 0] <= 0) or not np.array_equal(bias[:, 1], -gain[:, 0]):
        raise ValueError('D1 geometry requires fixed positive gain and exact affine bias_length=-gain')
    AS = A * scale[None, :]
    if not np.isfinite(AS).all() or np.linalg.matrix_rank(AS) != 21:
        raise ValueError('full row rank21 required; no epsilon-filled unknown directions')
    B = scale[:, None] * np.linalg.pinv(AS)
    if not np.isfinite(B).all():
        raise FloatingPointError('reference geometry not representable')
    residual = A @ B - np.eye(21)
    tolerance = 128 * np.finfo(float).eps * max(1., float(np.linalg.norm(A, np.inf) * np.linalg.norm(B, np.inf)))
    if not np.isfinite(tolerance) or np.max(abs(residual), initial=0.) > tolerance:
        raise ValueError('numerical right-inverse residual not certified')
    B = finite_array(B, (25, 21), 'target-length proposal map')
    return B, dict(role='POSITION_TARGET_LENGTH_GEOMETRY_ONLY', rank=21, nullity=4,
                   length_residual_max=float(np.max(abs(residual))), numerical_residual_tolerance=tolerance,
                   Dref21=[1.] * 21, force_clipping_invalidates_geometry=False,
                   finite_servo_response_prediction=False, contact_force_prediction=False,
                   native_effort_is_model_computed_not_measured=True,
                   b0_and_b2_retained_in_native_law_not_reinterpreted_as_contact_model=True)


def contact_geometry_rows(*, proposal_map25x21, current_unbiased_debt21,
                          robot_velocity25, patches):
    """Keep each old-plane normal row; no opposing-normal averaging.

    Each patch is an adapter-authenticated numeric record. n is hand-to-external
    at acquisition. J/v are current ROBOT kinematics. This mixed-time intent
    does not claim the old plane remains the external object's current plane.
    """
    B = finite_array(proposal_map25x21, (25, 21), 'proposal map')
    debt = finite_array(current_unbiased_debt21, (21,), 'unbiased reference debt')
    velocity = finite_array(robot_velocity25, (25,), 'current robot velocity')
    normal_rows, jacobian_rows, pressures, digits, stamps = [], [], [], [], []
    for p in patches:
        if not isinstance(p, dict):
            raise TypeError('authenticated patch numeric record required')
        n = finite_array(p['old_normal_world3'], (3,), 'old hand-to-external normal')
        J = finite_array(p['jacobian3x25'], (3, 25), 'current robot point Jacobian')
        if abs(float(n @ n) - 1.) > 1e-8:
            raise ValueError('unit old-plane normal required')
        pressure = p['normal_N']
        digit = p['digit']
        if isinstance(pressure, (bool, np.bool_)) or not isinstance(pressure, Real) or not np.isfinite(pressure) or pressure <= 0:
            raise ValueError('positive observed patch pressure required')
        if isinstance(digit, (bool, np.bool_)) or not isinstance(digit, (int, np.integer)) or not 0 <= digit < 5:
            raise ValueError('authenticated digit0..4 required')
        row = n @ J
        jacobian_rows.append(row)
        normal_rows.append(row @ B)
        pressures.append(float(pressure)); digits.append(int(digit))
        stamps.append(dict(patch_index=int(p['patch_index']), digit=int(digit),
                           acquisition_time_s=float(p['acquisition_time_s']),
                           robot_time_s=float(p['robot_time_s']),
                           old_normal_world3=n.tolist(), current_jacobian3x25=J.tolist()))
    count = len(pressures)
    N = finite_array(np.asarray(normal_rows).reshape(count, 21), (count, 21), 'normal reference rows')
    Jn = finite_array(np.asarray(jacobian_rows).reshape(count, 25), (count, 25), 'normal robot Jacobian')
    y = finite_array(np.asarray(pressures), (count,), 'observed patch pressures')
    vn = finite_array(Jn @ velocity, (count,), 'current robot normal velocity')
    an = finite_array(N @ debt, (count,), 'old-plane reference debt')
    d = np.asarray(digits, dtype=int); d.setflags(write=False)
    return dict(Npx21=N, normal_jacobian_Px25=Jn, y_p=y, pressure_P_N=y,
                digits_p=d, vnormal_p=vn, a_normal_p=an, patches=stamps)

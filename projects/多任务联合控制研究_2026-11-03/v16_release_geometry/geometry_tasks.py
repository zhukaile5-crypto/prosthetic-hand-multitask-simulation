"""Pure robot-old-plane clearance TASKS, never a motion/force predictor.

Only declared robot geometry, public skin observations and real reference
permissions enter this module. No simulator, object state, controller, writer
or solver is imported. The length pseudoinverse is explicitly an experimental
reference proposal; it does not identify the robot's finite servo response.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Optional
import numpy as np


def _real(value, name, *, nonnegative=False, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + ': real numeric required')
    x = float(value)
    if not np.isfinite(x) or (nonnegative and x < 0) or (positive and x <= 0):
        raise ValueError(name + ': finite value outside declared domain')
    return x


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(name + ': nonnegative integer required')
    return int(value)


def _array(value, shape, name):
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in raw.flat):
        raise ValueError(name + ': strictly numeric ' + str(shape) + ' required')
    a = np.asarray(value, dtype=float).copy()
    if not np.isfinite(a).all():
        raise ValueError(name + ': nonfinite')
    a.setflags(write=False)
    return a


def _axes(value, size, name):
    a = tuple(value)
    if len(a) != size or len(set(a)) != size or any(not isinstance(x, str) or not x for x in a):
        raise ValueError(name + ': unique declared axes required')
    return a


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ': boolean required')
    return bool(value)


@dataclass(frozen=True)
class LengthProposalMap:
    """delta q proposal = H delta reference, not a future robot state."""
    time_s: float
    robot_axes25: tuple
    reference_axes21: tuple
    H25x21: np.ndarray
    transmission21x25: np.ndarray
    reference_to_length21: np.ndarray
    joint_scale25: np.ndarray
    rank: int
    nullity: int
    length_residual21x21: np.ndarray
    declared_source: str
    role: str = field(default='EXPERIMENTAL_SCALED_LENGTH_DISTRIBUTION_PROPOSAL', init=False)
    actual_servo_response_validated: bool = field(default=False, init=False)
    pressure_prediction_available: bool = field(default=False, init=False)

    def __post_init__(self):
        object.__setattr__(self, 'time_s', _real(self.time_s, 'map time', nonnegative=True))
        object.__setattr__(self, 'robot_axes25', _axes(self.robot_axes25, 25, 'robot_axes25'))
        object.__setattr__(self, 'reference_axes21', _axes(self.reference_axes21, 21, 'reference_axes21'))
        for key, shape in (('H25x21',(25,21)), ('transmission21x25',(21,25)),
                           ('reference_to_length21',(21,)), ('joint_scale25',(25,)),
                           ('length_residual21x21',(21,21))):
            object.__setattr__(self, key, _array(getattr(self, key), shape, key))
        if np.any(self.joint_scale25 <= 0):
            raise ValueError('positive joint scales required')
        if not isinstance(self.declared_source, str) or not self.declared_source.strip():
            raise ValueError('explicit reference-to-length source/assumptions required')
        if _integer(self.rank, 'rank') > 21 or _integer(self.nullity, 'nullity') != 25-self.rank:
            raise ValueError('inconsistent rank/nullity')


def make_length_proposal_map(*, transmission21x25, reference_to_length21,
                             joint_scale25, robot_axes25, reference_axes21,
                             time_s, declared_source, known_reference21=None):
    """Compute H = Dq pinv(A Dq) Dref with explicit units/metric.

    Dref is DECLARED transmission-length change per reference change. It must
    not be silently inferred as identity from effort units or an actuator PD.
    A caller must document its assumptions/source. Unknown columns return None
    rather than zero gain. Rank deficiency remains explicit in residuals.
    """
    A = _array(transmission21x25, (21, 25), 'transmission21x25')
    D = _array(reference_to_length21, (21,), 'reference_to_length21')
    scale = _array(joint_scale25, (25,), 'joint_scale25')
    if np.any(scale <= 0):
        raise ValueError('joint_scale25: strictly positive metric scales required')
    axes = _axes(robot_axes25, 25, 'robot_axes25')
    refs = _axes(reference_axes21, 21, 'reference_axes21')
    time = _real(time_s, 'time_s', nonnegative=True)
    if not isinstance(declared_source, str) or not declared_source.strip():
        raise ValueError('reference-to-length assumptions/source must be explicit')
    if known_reference21 is not None:
        raw = np.asarray(known_reference21, dtype=object)
        if raw.shape != (21,) or any(not isinstance(x, (bool, np.bool_)) for x in raw):
            raise ValueError('known_reference21: boolean21 required')
        if not np.all(raw):
            return None
    scaled = A * scale[np.newaxis, :]
    # Default SVD cutoff is a numerical rank convention, not physical accuracy.
    inverse = np.linalg.pinv(scaled)
    H = scale[:, np.newaxis] * inverse * D[np.newaxis, :]
    residual = A @ H - np.diag(D)
    rank = int(np.linalg.matrix_rank(scaled))
    return LengthProposalMap(time, axes, refs, _array(H, (25, 21), 'H25x21'),
                             A, D, scale, rank, 25-rank,
                             _array(residual, (21, 21), 'length residual'), declared_source)


@dataclass(frozen=True)
class RobotPatchGeometry:
    """One positive public pressure patch plus current ROBOT-ONLY FK/Jacobian.

    The outer adapter, not this numeric schema, authenticates robot membership
    against the known robot definition. No external identity is accepted.
    """
    robot_body: int
    digit: int
    acquisition_time_s: float
    robot_time_s: float
    observed_normal_N: float
    acquisition_position_world_m: np.ndarray
    acquisition_normal_world: np.ndarray
    robot_position_world_m: np.ndarray
    robot_jacobian_m_per_q25: np.ndarray
    robot_axes25: tuple
    robot_point_velocity_world_m_s: Optional[np.ndarray] = None

    def __post_init__(self):
        object.__setattr__(self, 'robot_body', _integer(self.robot_body, 'robot_body'))
        digit = _integer(self.digit, 'digit')
        if digit >= 5:
            raise ValueError('only explicitly observed finger digits0..4 supported')
        object.__setattr__(self, 'digit', digit)
        for key in ('acquisition_time_s', 'robot_time_s'):
            object.__setattr__(self, key, _real(getattr(self, key), key, nonnegative=True))
        object.__setattr__(self, 'observed_normal_N', _real(self.observed_normal_N, 'observed_normal_N', positive=True))
        for key in ('acquisition_position_world_m', 'acquisition_normal_world', 'robot_position_world_m'):
            object.__setattr__(self, key, _array(getattr(self, key), (3,), key))
        n = self.acquisition_normal_world
        if abs(float(n @ n)-1) > 1e-6:
            raise ValueError('acquisition_normal_world: unit hand-to-external direction required; no implicit renormalization')
        object.__setattr__(self, 'robot_jacobian_m_per_q25', _array(self.robot_jacobian_m_per_q25, (3, 25), 'robot Jacobian'))
        object.__setattr__(self, 'robot_axes25', _axes(self.robot_axes25, 25, 'patch robot_axes25'))
        if self.robot_point_velocity_world_m_s is not None:
            object.__setattr__(self, 'robot_point_velocity_world_m_s', _array(self.robot_point_velocity_world_m_s, (3,), 'robot point velocity'))


@dataclass(frozen=True)
class GeometryTasks:
    status: str
    normal_matrix_m_per_reference: Optional[np.ndarray]
    normal_offset_m: Optional[np.ndarray]
    normal_weight: Optional[np.ndarray]
    tangent_matrix_m_per_reference: Optional[np.ndarray]
    tangent_target_m: Optional[np.ndarray]
    open_target_increment21: np.ndarray
    step_lower21: np.ndarray
    step_upper21: np.ndarray
    diagnostics: dict


def build_geometry_tasks(*, patches, mapping, current_time_s, maximum_geometry_age_s,
                         step_lower21, step_upper21, nominal_open_increment21,
                         mode_generation, robot_geometry_authenticated):
    """Generate separate noninward / low-tangent / OPEN preference tasks.

    Required future hierarchy: actual bounds -> noninward intent -> tangent
    suppression -> OPEN progress. This function does not solve or execute it.
    For each patch normal_matrix@du <= 0 means NO ADDITIONAL motion into its
    old acquisition plane. It is not a measured/current external gap constraint.
    Tangent rows are sqrt(public pressure share)*(I-nn.T)*J*H; they target zero
    proposed tangent increment, NOT cancellation of unknown dynamic velocity.
    """
    now = _real(current_time_s, 'current_time_s', nonnegative=True)
    age_limit = _real(maximum_geometry_age_s, 'maximum_geometry_age_s', positive=True)
    lo = _array(step_lower21, (21,), 'actual step lower21')
    hi = _array(step_upper21, (21,), 'actual step upper21')
    nominal = _array(nominal_open_increment21, (21,), 'nominal OPEN increment21')
    gen = _integer(mode_generation, 'mode_generation')
    authenticated = _flag(robot_geometry_authenticated, 'robot_geometry_authenticated')
    if np.any(lo > hi):
        raise ValueError('actual step bounds empty: do not replace by a model box')
    ps = tuple(patches)
    if any(not isinstance(p, RobotPatchGeometry) for p in ps):
        raise TypeError('RobotPatchGeometry sequence required; arbitrary external-state dictionaries rejected')
    diag = dict(role='ROBOT_RELATIVE_TO_ACQUISITION_PLANES_ONLY',
                normal_orientation='hand_to_external; positive means inward',
                normal_policy='no_additional_inward',
                hierarchy=('real_permissions', 'normal_noninward', 'low_tangent_increment', 'OPEN_progress'),
                mode_generation=gen, observed_patch_count=len(ps),
                robot_membership_authenticated_by_adapter=authenticated,
                real_step_bounds_unchanged=True, old_OPEN_direction_permissions_not_widened=True,
                nominal_OPEN_within_actual_bounds=bool(np.all((nominal >= lo) & (nominal <= hi))),
                unknown_reasons=[], unavailable_patches=[], patches=[],
                external_current_geometry_known=False, pressure_prediction_available=False,
                actual_servo_response_validated=False, force_safety_certified=False,
                full_hand_geometry_covered=False, delivery_certified=False,
                no_contact_not_inferred_from_empty_patches=True,
                current_robot_velocity_not_substituted_as_a_finite_servo_model=True)
    if not authenticated:
        diag['unknown_reasons'].append('known_robot_subtree_membership_not_authenticated')
    if mapping is None:
        diag['unknown_reasons'].append('reference_to_robot_geometry_proposal_map_missing')
    elif not isinstance(mapping, LengthProposalMap):
        raise TypeError('strict LengthProposalMap or None required')
    elif abs(mapping.time_s-now) > 1e-9:
        diag['unknown_reasons'].append('proposal_map_not_from_current_robot_snapshot')
    if not ps:
        diag['unknown_reasons'].append('no_observed_positive_finger_geometry_not_proof_of_clearance')
    usable = []
    for i, p in enumerate(ps):
        age = now-p.acquisition_time_s
        if age < -1e-9:
            raise ValueError('future skin acquisition rejected')
        why = []
        if abs(p.robot_time_s-now) > 1e-9:
            why.append('robot_geometry_not_current_snapshot')
        if mapping is not None and p.robot_axes25 != mapping.robot_axes25:
            why.append('robot_jacobian_axis_order_does_not_match_proposal_map')
        if age > age_limit+1e-9:
            why.append('skin_acquisition_stale')
        if why:
            diag['unavailable_patches'].append(dict(index=i, reasons=why))
        else:
            usable.append((i, p))
    if diag['unavailable_patches']:
        # Do not quietly improve the task by dropping the conflicting/stale row.
        diag['unknown_reasons'].append('partial_geometry_coverage_unavailable')
    if diag['unknown_reasons']:
        return GeometryTasks('UNAVAILABLE_GEOMETRY_TASKS', None, None, None, None, None, nominal, lo, hi, diag)
    # Rescale first to avoid summing large finite pressures into infinity.
    # A positive share that underflows is not silently replaced by epsilon.
    shares = {}
    for d in range(5):
        group = [(i, p) for i, p in usable if p.digit == d]
        if not group:
            continue
        scale = max(p.observed_normal_N for _, p in group)
        total = sum(p.observed_normal_N/scale for _, p in group)
        for i, p in group:
            share = (p.observed_normal_N/scale)/total
            if not np.isfinite(share) or share <= 0:
                diag['unknown_reasons'].append('positive_patch_share_not_numerically_representable')
            shares[i] = share
    if diag['unknown_reasons']:
        return GeometryTasks('UNAVAILABLE_GEOMETRY_TASKS', None, None, None, None, None, nominal, lo, hi, diag)
    normal, weights, tangent = [], [], []
    for i, p in usable:
        share = shares[i]
        weight = float(np.sqrt(share))
        n = p.acquisition_normal_world
        point_map = p.robot_jacobian_m_per_q25 @ mapping.H25x21
        row = n @ point_map
        T = np.eye(3)-np.outer(n, n)
        tangent.append(weight*(T @ point_map))
        normal.append(row)
        weights.append(weight)
        minimum = float(np.sum(np.where(row >= 0, row*lo, row*hi)))
        maximum = float(np.sum(np.where(row >= 0, row*hi, row*lo)))
        observed_vn = None if p.robot_point_velocity_world_m_s is None else float(n@p.robot_point_velocity_world_m_s)
        observed_vt = None if p.robot_point_velocity_world_m_s is None else float(np.linalg.norm(T@p.robot_point_velocity_world_m_s))
        diag['patches'].append(dict(index=i, robot_body=p.robot_body, digit=p.digit,
                acquisition_time_s=p.acquisition_time_s, age_s=now-p.acquisition_time_s,
                observed_normal_N=p.observed_normal_N, within_digit_pressure_share=share,
                robot_old_plane_inward_offset_m=float(n@(p.robot_position_world_m-p.acquisition_position_world_m)),
                proposed_inward_range_m=(minimum, maximum), strict_outward_proposal_direction_exists=minimum < 0,
                nominal_OPEN_inward_proposal_m=float(row@nominal),
                current_robot_normal_velocity_m_s=observed_vn,
                current_robot_tangent_speed_m_s=observed_vt))
    rows = np.asarray(normal)
    diag.update(proposal_map_role=mapping.role, proposal_map_source=mapping.declared_source,
                transmission_rank=mapping.rank, transmission_nullity=mapping.nullity,
                length_mapping_residual_max=float(np.max(np.abs(mapping.length_residual21x21))),
                pressure_share_weighting='within-digit; identical patch split/merge invariant',
                opposite_patch_normals_kept_separate=True,
                unresolved_OPEN_noninward_patch_indices=[p['index'] for p in diag['patches'] if p['nominal_OPEN_inward_proposal_m'] > 0],
                strict_outward_unavailable_patch_indices=[p['index'] for p in diag['patches'] if not p['strict_outward_proposal_direction_exists']])
    return GeometryTasks('EXPERIMENTAL_GEOMETRY_TASKS', _array(rows, (len(ps),21), 'normal rows'),
                _array(np.zeros(len(ps)), (len(ps),), 'normal offsets'),
                _array(weights, (len(ps),), 'normal weights'),
                _array(np.vstack(tangent), (3*len(ps),21), 'tangent rows'),
                _array(np.zeros(3*len(ps)), (3*len(ps),), 'tangent targets'), nominal, lo, hi, diag)


def task_loss(tasks, increment21):
    """Diagnostic soft loss only, no material force objective or solver."""
    if not isinstance(tasks, GeometryTasks) or tasks.normal_matrix_m_per_reference is None:
        return None
    u = _array(increment21, (21,), 'proposed reference increment21')
    inward = tasks.normal_matrix_m_per_reference@u + tasks.normal_offset_m
    tangent = tasks.tangent_matrix_m_per_reference@u-tasks.tangent_target_m
    violation = tasks.normal_weight*np.maximum(inward, 0.)
    return dict(normal_noninward_loss_m2=float(violation@violation),
                tangent_increment_loss_m2=float(tangent@tangent),
                inward_proposals_m=inward.tolist(),
                actual_bounds_satisfied=bool(np.all((u >= tasks.step_lower21) & (u <= tasks.step_upper21))),
                pressure_prediction_available=False, force_safety_certified=False)

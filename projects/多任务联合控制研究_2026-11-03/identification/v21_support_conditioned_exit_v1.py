"""One-snapshot support-conditioned withdrawal using the original task domain.

This has no actuator or plant. The common post-correction witness and the
whole original normal/motion domain are retained. Two numerical arms share a
per-contact no-inward corridor. A mechanical-only comparison and a
feasible-capacity arm limit extra withdrawal using the delayed evidence.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from v23_task_execution.finite_solver import Level, solve_hierarchy

AXES = 21
CONTROL_S = .005
GENTLE_RAMP_S = .020
CHECK_TOL = 1e-8


def pace(sustained_fraction):
    if sustained_fraction is None:
        sustained_fraction = 0.
    if (not isinstance(sustained_fraction, Real) or isinstance(sustained_fraction, (bool, np.bool_))
            or not np.isfinite(sustained_fraction) or not 0. <= sustained_fraction <= 1.):
        raise ValueError("Explicit support progress must lie in [0, 1]")
    floor = CONTROL_S / GENTLE_RAMP_S
    return floor + (1. - floor) * float(sustained_fraction)


@dataclass(frozen=True)
class Snapshot:
    lower: np.ndarray
    upper: np.ndarray
    hard: np.ndarray
    rhs: np.ndarray
    original_anchor: np.ndarray
    normal: np.ndarray
    work_row: np.ndarray
    levels: tuple
    motion_level: Level | None
    options: dict

    def violation(self, x):
        return max(0., float(np.max(self.lower - x)), float(np.max(x - self.upper)),
                   float(np.max(self.hard @ x - self.rhs, initial=0.)))


def snapshot(d):
    """Rebuild all intended later levels, including previously failed ones."""
    lower, upper = (np.asarray(d[k], float) for k in ("second_stage_lower", "second_stage_upper"))
    n = len(lower)
    hard = np.asarray(d["second_stage_hard_matrix"], float)
    rhs = np.asarray(d["second_stage_hard_upper"], float)
    x0 = np.asarray(d["certified_L1_extended_anchor"], float)
    units = np.asarray(d["reference_step_units21"], float)
    normals = np.asarray(d["normal_matrix_N_Px21"], float)
    if (n <= AXES or lower.shape != upper.shape or lower.shape != x0.shape or len(lower.shape) != 1
            or normals.shape != (n - AXES, AXES) or units.shape != (AXES,)
            or hard.shape != (len(rhs), n) or np.any(units <= 0.) or np.any(lower > upper)
            or not np.array_equal(lower[AXES:], upper[AXES:])):
        raise ValueError("Expected a declared post-correction domain with fixed slack")
    if not all(np.isfinite(x).all() for x in (lower, upper, hard, rhs, x0, units, normals)):
        raise ValueError("Nonfinite original task domain")
    normal_step = normals * units[None, :]
    row_scales = np.abs(normal_step).sum(axis=1)
    use = row_scales > 0.
    normal = np.zeros((int(use.sum()), n))
    normal[:, :AXES] = -normal_step[use] / row_scales[use, None]

    work = np.asarray(d["scalar_normal_work_row"], float)
    goal = float(d["scalar_normal_work_box_upper_m"])
    if work.shape != (AXES,) or not np.isfinite(work).all() or not np.isfinite(goal):
        raise ValueError("Invalid original outward task")
    scale = max(float(np.max(abs(work))), abs(goal)) or 1.
    work_level = Level("scalar_pressure_share_normal_withdrawal",
                       np.r_[work / scale, np.zeros(n - AXES)][None, :], np.array([goal / scale]))
    posture = np.zeros((AXES, n))
    posture[:, :AXES] = np.eye(AXES)
    pose_level = Level("canonical_or_explicit_path_reference", posture,
                       np.asarray(d["effective_posture_target_delta21"], float) / units)
    motion = d["motion_task"]
    motion_level = None
    if motion["available"]:
        scales = np.asarray(motion["effective_row_units_M"], float)
        matrix = np.zeros((len(scales), n))
        matrix[:, :AXES] = np.asarray(motion["matrix_Mx21"], float) * units[None, :] / scales[:, None]
        target = -float(d["dt_s"]) * np.asarray(motion["error_M_m_s"], float) / scales
        motion_level = Level("public_external_projection_low_motion", matrix, target)
    levels = (work_level, pose_level) if motion_level is None else (work_level, motion_level, pose_level)
    opts = {k:d["numerical_budget"][k] for k in ("max_iterations", "atol", "rtol")}
    return Snapshot(lower, upper, hard, rhs, x0, normal, work, levels, motion_level, opts)


def retained_domain(problem, anchor, fraction):
    """Limit outward progress on each row relative to the common witness.

    The cap uses a box envelope in normalized reference coordinates. It
    introduces no Newton-to-motion conversion. No lower pressure is assumed.
    """
    n = problem.normal
    q = n @ anchor
    retained = np.vstack((problem.hard, -n))
    right = np.r_[problem.rhs, -q]
    if fraction == 1.:
        return retained, right
    maximum = np.maximum(n, 0.) @ problem.upper + np.minimum(n, 0.) @ problem.lower
    room = np.maximum(0., maximum - q)
    return np.vstack((retained, n)), np.r_[right, q + fraction * room]


def solve_pair(problem, sustained_fraction):
    """Return matched numerical results or a declared witness failure."""
    fraction = pace(sustained_fraction)
    if problem.violation(problem.original_anchor) > CHECK_TOL:
        raise ValueError("Frozen correction witness does not satisfy the frozen domain")
    n = len(problem.lower)
    matrix = np.zeros((AXES, n))
    matrix[:, :AXES] = np.eye(AXES)
    options = dict(lower=problem.lower, upper=problem.upper, **problem.options)
    witness = solve_hierarchy(**options, levels=(Level("canonical_correction_witness", matrix, np.zeros(AXES)),),
                              initial=problem.original_anchor,
                              inequality_matrix=problem.hard, inequality_upper=problem.rhs)
    if not witness.fully_converged or problem.violation(witness.x) > CHECK_TOL:
        return dict(witness_status=witness.status, fraction=fraction, anchor=None)
    anchor = witness.x
    common_hard, common_rhs = retained_domain(problem, anchor, 1.)
    uncapped = solve_hierarchy(**options, levels=problem.levels, initial=anchor,
                               inequality_matrix=common_hard, inequality_upper=common_rhs)
    arms = {"uncapped": describe_result(problem, uncapped, anchor, common_hard, common_rhs)}
    if not uncapped.available:
        return dict(witness_status=witness.status, fraction=fraction, anchor=anchor[:AXES].tolist(),
                    patch_rows=len(problem.normal), feasible_capacity_m=None, **arms)
    capacity = arms["uncapped"]["optional_work_m"]
    if capacity < -1e-10:
        raise ValueError("Certified optional optimum fell below the common witness")
    capacity = max(0., capacity)
    row_scale = float(np.sum(abs(problem.work_row))) or 1.
    work_row = np.r_[problem.work_row / row_scale, np.zeros(n - AXES)]
    mechanical_hard, mechanical_rhs = retained_domain(problem, anchor, fraction)
    if fraction == 1.:
        arms["mechanical_only"] = arms["support_conditioned"] = arms["uncapped"]
    else:
        # The convex combination is a feasibility witness for all added rows.
        # Re-solving still lets the lower levels choose tangential motion.
        augmented_hard = np.vstack((mechanical_hard, work_row))
        augmented_rhs = np.r_[mechanical_rhs, work_row @ anchor + fraction * capacity / row_scale]
        blend = anchor + fraction * (uncapped.x - anchor)
        if (problem.violation(blend) > CHECK_TOL
                or np.max(augmented_hard @ blend - augmented_rhs, initial=0.) > CHECK_TOL):
            raise ValueError("Interpolation witness did not satisfy the new domain")
        for name, hard, rhs in (("mechanical_only", mechanical_hard, mechanical_rhs),
                                ("support_conditioned", augmented_hard, augmented_rhs)):
            solved = solve_hierarchy(**options, levels=problem.levels, initial=anchor,
                                     inequality_matrix=hard, inequality_upper=rhs)
            arms[name] = describe_result(problem, solved, anchor, hard, rhs)
    motion_probe = None
    observed = arms["support_conditioned"]["motion_residual_norm"]
    comparator = arms["uncapped"]["motion_residual_norm"]
    if observed is not None and observed > comparator + CHECK_TOL:
        # Diagnose whether reordering tasks can recover the motion value
        # without granting more outward authority. The output is never applied.
        low_motion = solve_hierarchy(**options, levels=(problem.motion_level,), initial=anchor,
                                     inequality_matrix=augmented_hard, inequality_upper=augmented_rhs)
        r = describe_result(problem, low_motion, anchor, augmented_hard, augmented_rhs)
        motion_probe = dict(certified=r["full"], minimum_residual_norm=r["motion_residual_norm"],
                            still_above_uncapped=None if not r["full"]
                            else r["motion_residual_norm"] > comparator + CHECK_TOL)
    return dict(witness_status=witness.status, fraction=fraction, anchor=anchor[:AXES].tolist(),
                patch_rows=len(problem.normal), feasible_capacity_m=capacity,
                motion_only_diagnostic=motion_probe, **arms)


def describe_result(problem, solved, anchor, hard, rhs):
    x = solved.x
    hard_error = max(problem.violation(x), float(np.max(hard @ x - rhs, initial=0.)))
    if hard_error > CHECK_TOL:
        raise ValueError("Solver output violates an original or added inequality")
    return dict(status=solved.status, prefix=solved.completed_prefix, full=solved.fully_converged,
                certified_task_names=[c["name"] for c in solved.certified_prefix],
                candidate_available=solved.available, normalized_reference=x[:AXES].tolist(),
                maximum_additional_outward=float(np.max(problem.normal @ (x - anchor), initial=0.)),
                optional_work_m=float(problem.work_row @ (x - anchor)[:AXES]),
                motion_residual_norm=None if problem.motion_level is None else float(np.linalg.norm(
                    problem.motion_level.matrix @ x - problem.motion_level.target)),
                maximum_original_and_new_domain_error=hard_error)

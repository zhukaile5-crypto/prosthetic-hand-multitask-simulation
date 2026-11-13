"""Reserve necessary motion before scheduling additional contact withdrawal.

Consumes a single current reference-domain problem. The motion optimum and
its minimal withdrawal are certified before opening an optional movement
domain. No actuator, physical model, prediction, or history update occurs.
"""

import numpy as np

from identification.v21_support_conditioned_exit_v1 import (
    AXES, CHECK_TOL, Level, pace, retained_domain, solve_hierarchy,
)


def checked(problem, hard, rhs, point, equality=None, target=None):
    error = max(problem.violation(point), float(np.max(hard @ point - rhs, initial=0.)))
    if equality is not None:
        error = max(error, float(np.max(abs(equality @ point - target), initial=0.)))
    if error > CHECK_TOL:
        raise ValueError("Returned point violated an original, retention, or motion constraint")
    return error


def solve_reserved_transfer(problem, sustained_fraction):
    """Return correction, mandatory stability, and optional withdrawal separately."""
    fraction = pace(sustained_fraction)
    if problem.motion_level is None:
        return dict(status="current_motion_unavailable", fraction=fraction)
    motion = problem.motion_level
    n = len(problem.lower)
    identity = np.zeros((AXES, n))
    identity[:, :AXES] = np.eye(AXES)
    opts = dict(lower=problem.lower, upper=problem.upper, **problem.options)
    original = dict(inequality_matrix=problem.hard, inequality_upper=problem.rhs)
    correction = solve_hierarchy(**opts, **original, initial=problem.original_anchor,
        levels=(Level("minimum_reference_correction", identity, np.zeros(AXES)),))
    if not correction.fully_converged:
        return dict(status="correction_not_certified", fraction=fraction)
    a = correction.x
    checked(problem, problem.hard, problem.rhs, a)
    retain_hard, retain_rhs = retained_domain(problem, a, 1.)
    # Minimizing this outward objective on the motion optimum face minimizes
    # necessary pressure-share geometry; choosing a reference follows last.
    work = problem.levels[0].matrix
    floor = Level("least_withdrawal_needed_for_motion", work, work @ a)
    tie = Level("canonical_stability_witness", identity, a[:AXES])
    mandatory = solve_hierarchy(**opts, initial=a, levels=(motion, floor, tie),
        inequality_matrix=retain_hard, inequality_upper=retain_rhs)
    if mandatory.completed_prefix < 2:
        return dict(status="necessary_motion_and_withdrawal_not_certified", fraction=fraction,
                    mandatory_prefix=mandatory.completed_prefix)
    b = mandatory.x
    motion_fixed = mandatory.certified_prefix[0]["attained"]
    checked(problem, retain_hard, retain_rhs, b, motion.matrix, motion_fixed)
    # A separate solver invocation is needed so the minimum *necessary* work
    # is not frozen for the optional task. The attained motion is preserved.
    frozen = dict(equality_matrix=motion.matrix, equality_target=motion_fixed)
    reserve_hard, reserve_rhs = retained_domain(problem, b, 1.)
    hard = np.vstack((retain_hard, reserve_hard[len(problem.hard):]))
    rhs = np.r_[retain_rhs, reserve_rhs[len(problem.rhs):]]
    uncapped = solve_hierarchy(**opts, **frozen, initial=b, levels=(problem.levels[0], problem.levels[-1]),
        inequality_matrix=hard, inequality_upper=rhs)
    if not uncapped.available:
        return dict(status="optional_capacity_not_certified", fraction=fraction,
                    mandatory_prefix=mandatory.completed_prefix)
    checked(problem, hard, rhs, uncapped.x, motion.matrix, motion_fixed)
    capacity = float(problem.work_row @ (uncapped.x - b)[:AXES])
    if capacity < -1e-10:
        raise ValueError("Optional optimum fell below the stability witness")
    capacity = max(0., capacity)
    step_scale = float(np.abs(problem.work_row).sum()) or 1.
    work_scaled = np.r_[problem.work_row / step_scale, np.zeros(n - AXES)]
    capped_hard, capped_rhs = hard, rhs
    if fraction < 1.:
        per_contact, per_contact_rhs = retained_domain(problem, b, fraction)
        start = len(problem.rhs) + len(problem.normal)
        capped_hard = np.vstack((hard, per_contact[start:], work_scaled))
        capped_rhs = np.r_[rhs, per_contact_rhs[start:], work_scaled @ b + fraction * capacity / step_scale]
        blend = b + fraction * (uncapped.x - b)
        checked(problem, capped_hard, capped_rhs, blend, motion.matrix, motion_fixed)
    bounded = (uncapped if fraction == 1. else solve_hierarchy(**opts, **frozen, initial=b,
        levels=(problem.levels[0], problem.levels[-1]), inequality_matrix=capped_hard,
        inequality_upper=capped_rhs))
    error = checked(problem, capped_hard, capped_rhs, bounded.x, motion.matrix, motion_fixed)
    return dict(status="solved", fraction=fraction,
                mandatory_prefix=mandatory.completed_prefix,
                canonical_stability_witness_certified=mandatory.fully_converged,
                full_optional_tasks=bounded.fully_converged,
                optional_task_prefix=bounded.completed_prefix,
                correction_reference_normalized=a[:AXES].tolist(),
                stability_reference_normalized=b[:AXES].tolist(),
                final_reference_normalized=bounded.x[:AXES].tolist(),
                uncapped_reference_normalized=uncapped.x[:AXES].tolist(),
                needed_withdrawal_proxy_m=float(problem.work_row @ (b - a)[:AXES]),
                optional_capacity_proxy_m=capacity,
                optional_withdrawal_proxy_m=float(problem.work_row @ (bounded.x - b)[:AXES]),
                total_beyond_correction_proxy_m=float(problem.work_row @ (bounded.x - a)[:AXES]),
                protected_motion_vector=motion_fixed.tolist(),
                motion_optimal_residual_norm=float(np.linalg.norm(motion_fixed - motion.target)),
                final_motion_residual_norm=float(np.linalg.norm(motion.matrix @ bounded.x - motion.target)),
                final_motion_attainment_error=float(np.max(abs(motion.matrix @ bounded.x - motion_fixed))),
                original_and_added_constraint_error=error)

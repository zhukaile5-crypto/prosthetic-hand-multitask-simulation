"""Pure current-state joint effort/normal-intent convex proposal kernel.

REQUIRED input scales: a mapping with positive finite ``residual25`` (25,),
``intentP`` (P,), ``delta21`` (21,), and scalar ``load``. Scales multiply
residuals BEFORE squaring; they are declared controller-design quantities,
never post-hoc evaluation utility weights.

Variables d21 and zP minimize exactly:
  ||Sr*(A.T@d - J.T@(z-y))||^2 + ||Sf*(z-z_intent)||^2
    + ||Sd*d||^2 + (sL*max(0, load - support_coeff@z))^2.
Here d is effort delta from the FIXED original p0, not a projected target.
Bounds ``effort_lower21/upper21`` are absolute effort bounds: d bounds are
derived by subtracting p0. The caller owns their valid native/mode mapping.
z >= 0, five digit sums <= caps5, and total z <= total_budget.

Projection is the Euclidean projection on the whole nonnegative grouped-cap
and total-budget polytope, using a shared global KKT multiplier and finite
piecewise-linear water filling, NOT sequential group-then-budget clipping.

Predeclared computational domain/budget: P <=128, monotone restarted FISTA,
400 iterations by default. A fixed global Lipschitz upper bound uses the
largest eigenvalue of a25x25 matrix, diagonal regularizers and the load hinge.
Termination checks direct box/laminar-normal-cone KKT stationarity, dual
nonnegativity, complementarity and primal violations, independently of L or
projected-step differences; default atol1e-8/rtol1e-7 is unchanged. Activity
uses exact box endpoints and local machine-relative sum roundoff for grouped
caps, never the optimization atol. Projected-gradient mappings remain diagnostic
ONLY. A zero representable step with failed direct KKT returns explicitly
UNAVAILABLE numerical_stagnation. Every budget/criterion is reported.

No native inverse/scoring, balancing or extra coupled-synergy residual rows,
runtime, codec, physics, contact gradients, future pressure, or global floating
point optimality certificate is implemented here. Unknown inputs remain
unknown. A converged QP is a force INTENT/effort proposal, not a safe action.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Optional
import numpy as np

MAX_PATCHES = 128
ALGORITHM = 'monotone_restarted_fista_with_joint_group_budget_projection'


class _InputError(ValueError):
    def __init__(self, message, unknown=False):
        super().__init__(message)
        self.unknown = unknown


def _array(value, shape, name):
    if value is None:
        raise _InputError(name + '_unknown', unknown=True)
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape:
        raise _InputError(name + '_shape_' + str(shape))
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in raw.flat):
        raise _InputError(name + '_requires_real_numbers')
    result = np.asarray(value, dtype=float).copy()
    if not np.isfinite(result).all():
        raise _InputError(name + '_nonfinite_unknown', unknown=True)
    return result


def _scalar(value, name, minimum=None, strictly_positive=False):
    if value is None:
        raise _InputError(name + '_unknown', unknown=True)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise _InputError(name + '_requires_real_number')
    result = float(value)
    if not np.isfinite(result):
        raise _InputError(name + '_nonfinite_unknown', unknown=True)
    if minimum is not None and result < minimum or strictly_positive and result <= 0.:
        raise _InputError(name + '_outside_declared_numeric_domain')
    return result


def _readonly(value):
    if value is None:
        return None
    result = np.asarray(value, dtype=float).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class IntentQPResult:
    status: str
    reason: str
    d21: Optional[np.ndarray] = None
    zP: Optional[np.ndarray] = None
    p21: Optional[np.ndarray] = None
    residual25: Optional[np.ndarray] = None
    load_shortfall_N: Optional[float] = None
    objective: Optional[float] = None
    iterations: int = 0
    diagnostics: dict = field(default_factory=dict)

    @property
    def available(self):
        return self.status == 'converged'


def _simplex_cap(v, cap):
    """Projection on x>=0,sum(x)<=cap, without modifying v."""
    if not len(v) or cap == 0.:
        return np.zeros_like(v)
    if len(v) == 1:
        # Avoid v-(v-cap) cancellation when cap is tiny relative to v.
        return np.array([min(max(float(v[0]), 0.), cap)], dtype=float)
    positive = np.maximum(v, 0.)
    if float(positive.sum()) <= cap:
        return positive
    ordered = np.sort(v)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, len(v) + 1)
    active = ordered - (cumulative - cap) / indices > 0.
    rho = int(np.flatnonzero(active)[-1])
    threshold = float((cumulative[rho] - cap) / (rho + 1))
    return np.maximum(v - threshold, 0.)


def _project_z(v, groups, caps, budget):
    """Finite shared-multiplier projection; inputs are already validated."""
    if not len(v) or budget == 0.:
        return np.zeros_like(v)
    result = np.zeros_like(v)
    total = 0.
    slope_count = 0
    events = []
    for ids, cap in zip(groups, caps):
        if not len(ids) or cap == 0.:
            continue
        values = v[ids]
        positive = values[values > 0.]
        positive_sum = float(positive.sum())
        total += min(float(cap), positive_sum)
        if positive_sum > cap:
            ordered = np.sort(values)[::-1]
            cumulative = np.cumsum(ordered)
            counts = np.arange(1, len(values) + 1)
            active = ordered - (cumulative - cap) / counts > 0.
            rho = int(np.flatnonzero(active)[-1])
            threshold = float((cumulative[rho] - cap) / (rho + 1))
            after = values[values > threshold]
            events.append((threshold, int(len(after))))
            events.extend((float(x), -1) for x in after)
        else:
            slope_count += len(positive)
            events.extend((float(x), -1) for x in positive)
    multiplier = 0.
    if total > budget:
        point = 0.
        found = False
        ordered_events = sorted(events)
        k = 0
        while k < len(ordered_events):
            next_point = ordered_events[k][0]
            next_total = total - slope_count * (next_point - point)
            if slope_count > 0 and budget >= next_total:
                multiplier = point + (total - budget) / slope_count
                found = True
                break
            total = next_total
            point = next_point
            while k < len(ordered_events) and ordered_events[k][0] == next_point:
                slope_count += ordered_events[k][1]
                k += 1
        if not found:
            # At the last endpoint all positive parts vanish. Reaching this
            # path is a roundoff endpoint, not permission to drop a constraint.
            multiplier = point if total <= budget + 64*np.finfo(float).eps*max(1., budget) else float(np.max(v))
    for ids, cap in zip(groups, caps):
        if len(ids):
            result[ids] = _simplex_cap(v[ids] - multiplier, float(cap))
    return result


def _box_normal_cone(gradient, x, lower, upper):
    """Exact endpoint activity; no small-box/step-difference convergence test."""
    fixed = lower == upper
    at_lower = x == lower
    at_upper = x == upper
    residual = gradient.copy()
    residual[at_lower] = np.minimum(gradient[at_lower], 0.)
    residual[at_upper] = np.maximum(gradient[at_upper], 0.)
    residual[fixed] = 0.
    return dict(stationarity_inf=float(np.max(abs(residual))),
                strict_interior_count=int((~fixed & ~at_lower & ~at_upper).sum()),
                fixed_coordinates=int(fixed.sum()), activity_rule='exact_float_endpoints')


def _normal_intent_cone(gradient, z, groups, caps, budget):
    """O(P) least-infinity stationarity residual of the laminar normal cone.

    For h_i=eta+lambda_i, positive z require |g+h|<=rho and zero z
    require g+h>=-rho. Their intersection is [a_i-rho,b_i+rho],
    a=-min(g_all), b=min(-g_positive), b=inf when no z is positive.
    Inactive group => h_i=eta; active group => h_i>=eta. Eta is zero
    unless the total budget is active. These interval intersections give
    rho directly. Recompute real stationarity and complementarity afterward.
    """
    machine = np.finfo(float).eps
    total = float(z.sum())
    total_slack = budget-total
    total_roundoff = float(8.*machine*max(abs(budget), float(abs(z).sum())))
    budget_active = abs(total_slack) <= total_roundoff
    records = []
    for digit, (ids, cap) in enumerate(zip(groups, caps)):
        if not len(ids):
            continue
        values, g = z[ids], gradient[ids]
        group_sum = float(values.sum())
        slack = float(cap-group_sum)
        roundoff = float(8.*machine*max(abs(float(cap)), float(abs(values).sum())))
        positive = values > 0.  # strictly positive, however small
        records.append(dict(digit=digit, ids=ids, active=bool(abs(slack)<=roundoff),
            a=float(-np.min(g)), b=float(np.min(-g[positive])) if positive.any() else float('inf'),
            slack=slack, geometric_roundoff=roundoff))
    rho = 0.
    inactive = [r for r in records if not r['active']]
    if not budget_active:
        eta = 0.
        for r in records:
            rho = max(rho, -r['b'])
            rho = max(rho, .5*(r['a']-r['b']) if r['active'] else r['a'])
    else:
        for r in records:
            rho = max(rho, -r['b'])
            if r['active']:
                rho = max(rho, .5*(r['a']-r['b']))
        if inactive:
            max_a = max(r['a'] for r in inactive)
            min_b = min(r['b'] for r in records)
            rho = max(rho, .5*(max_a-min_b))
        eta = max(0., max(r['a'] for r in inactive)-rho) if inactive else 0.
    lambdas = np.zeros(5)
    stationarity = np.zeros_like(z)
    group_complementarity = np.zeros(5)
    group_active = [False]*5
    group_slack = [None]*5
    geometric_roundoff = [None]*5
    for r in records:
        h = max(eta, r['a']-rho) if r['active'] else eta
        lambdas[r['digit']] = h-eta
        shifted = gradient[r['ids']]+h
        stationarity[r['ids']] = np.where(z[r['ids']]>0., shifted, np.minimum(shifted, 0.))
        group_complementarity[r['digit']] = abs((h-eta)*r['slack'])
        group_active[r['digit']] = r['active']
        group_slack[r['digit']] = r['slack']
        geometric_roundoff[r['digit']] = r['geometric_roundoff']
    budget_complementarity = abs(eta*total_slack)
    return dict(stationarity_inf=float(np.max(abs(stationarity))) if len(z) else 0.,
        global_multiplier=float(eta), group_multipliers=lambdas.tolist(),
        dual_nonnegativity_violation=float(max(0., -eta, -float(lambdas.min()))),
        complementarity_inf=float(max(budget_complementarity, float(group_complementarity.max()))),
        group_complementarity=group_complementarity.tolist(), budget_complementarity=float(budget_complementarity),
        group_active=group_active, budget_active=bool(budget_active),
        group_slack=group_slack, total_slack=total_slack,
        group_geometric_roundoff=geometric_roundoff, budget_geometric_roundoff=total_roundoff,
        positive_z_count=int((z>0.).sum()), interval_optimal_stationarity_bound=float(rho),
        activity_rule='positive_z_is_strict; cap_activity_is8eps_times_local_cap_or_sum_without_unit_floor')


def _domain(patches, digits, caps5, total_budget):
    raw = np.asarray(digits, dtype=object)
    if raw.shape != (patches,) or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or not 0 <= int(x) < 5 for x in raw.flat):
        raise _InputError('patch_digit_p_requires_integer0to4')
    digit = np.asarray(digits, dtype=int)
    caps = _array(caps5, (5,), 'caps5')
    if np.any(caps < 0.):
        raise _InputError('caps5_must_be_nonnegative')
    budget = _scalar(total_budget, 'total_budget', minimum=0.)
    return tuple(np.flatnonzero(digit == i) for i in range(5)), caps, budget


def project_normal_intents(v_p, patch_digit_p, caps5, total_budget):
    """Public mathematical projection helper; malformed/unknown inputs raise.

    This projects the preference, NOT the physical pressure or QP objective
    target. The solver retains its original unprojected z_intent in the loss.
    """
    raw = np.asarray(v_p, dtype=object)
    if raw.ndim != 1 or len(raw) > MAX_PATCHES:
        raise ValueError('projection_requires_vector_with_at_most128_patches')
    values = _array(v_p, (len(raw),), 'projection_vector')
    groups, caps, budget = _domain(len(values), patch_digit_p, caps5, total_budget)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        return _project_z(values, groups, caps, budget)


def solve_intent_qp(*, p0_21, A21x25, Jpx25, y_p, z_intent_p,
                    effort_lower21, effort_upper21, patch_digit_p, caps5,
                    total_budget, support_coeff_p, load_proxy_N, scales,
                    max_iterations=400, atol=1e-8, rtol=1e-7):
    """Return a proposal only when status='converged'; never an actuator action.

    Even a nonconverged result carries its feasible candidate and actual
    unretargeted residual, with available=False. Unknown/invalid input has no
    fabricated arrays. Absolute effort bounds do not project/replace p0.
    """
    budget_record = dict(max_iterations=max_iterations, atol=atol, rtol=rtol, max_patches=MAX_PATCHES)
    scope = dict(algorithm=ALGORITHM, computational_budget=budget_record,
                 convergence_revision='direct_normal_cone_v2',
                 native_mapping_verified_by_this_kernel=False,
                 future_pressure_prediction_available=False,
                 physical_safety_certified=False, global_floating_optimality_certified=False,
                 extra_balance_or_coupled_synergy_rows_implemented=False)
    try:
        p0 = _array(p0_21, (21,), 'p0_21')
        A = _array(A21x25, (21, 25), 'A21x25')
        if Jpx25 is None:
            raise _InputError('Jpx25_unknown', unknown=True)
        raw_J = np.asarray(Jpx25, dtype=object)
        if raw_J.ndim != 2 or raw_J.shape[1] != 25:
            raise _InputError('Jpx25_requires_P_by25')
        P = raw_J.shape[0]
        if P > MAX_PATCHES:
            raise _InputError('patch_count_exceeds_predeclared_computational_domain', unknown=True)
        J = _array(Jpx25, (P, 25), 'Jpx25')
        y = _array(y_p, (P,), 'y_p')
        target = _array(z_intent_p, (P,), 'z_intent_p')
        if np.any(y < 0.) or np.any(target < 0.):
            raise _InputError('normal_observations_and_intents_must_be_nonnegative')
        lo = _array(effort_lower21, (21,), 'effort_lower21') - p0
        hi = _array(effort_upper21, (21,), 'effort_upper21') - p0
        if np.any(lo > hi):
            raise _InputError('empty_effort_box')
        groups, caps, budget = _domain(P, patch_digit_p, caps5, total_budget)
        c = _array(support_coeff_p, (P,), 'support_coeff_p')
        load = _scalar(load_proxy_N, 'load_proxy_N', minimum=0.)
        if not isinstance(scales, Mapping) or set(scales) != {'residual25', 'intentP', 'delta21', 'load'}:
            raise _InputError('four_explicit_scales_required')
        sr = _array(scales['residual25'], (25,), 'scales.residual25')
        sf = _array(scales['intentP'], (P,), 'scales.intentP')
        sd = _array(scales['delta21'], (21,), 'scales.delta21')
        sl = _scalar(scales['load'], 'scales.load', strictly_positive=True)
        if np.any(sr <= 0.) or np.any(sf <= 0.) or np.any(sd <= 0.):
            raise _InputError('declared_scales_must_be_positive')
        if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral) or max_iterations < 1:
            raise _InputError('max_iterations_requires_positive_integer')
        atol = _scalar(atol, 'atol', strictly_positive=True)
        rtol = _scalar(rtol, 'rtol', minimum=0.)
    except _InputError as exc:
        return IntentQPResult('unknown' if exc.unknown else 'invalid_input', str(exc), diagnostics=scope)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            M = np.column_stack((A.T, -J.T))
            T = sr[:, None] * M
            b = J.T @ y
            diagonal = np.r_[sd*sd, sf*sf]
            # Fixed25x25 eigendecomposition, independent of patch count.
            spectrum = np.linalg.eigvalsh(T @ T.T)
            lipschitz = float(2.*(max(0., spectrum[-1]) + np.max(diagonal) + sl*sl*float(c@c)))
            if not np.isfinite(lipschitz) or lipschitz <= 0.:
                raise FloatingPointError('nonfinite_Lipschitz_bound')

            def projection(x):
                return np.r_[np.clip(x[:21], lo, hi), _project_z(x[21:], groups, caps, budget)]

            def evaluate(x):
                d, z = x[:21], x[21:]
                residual = A.T @ d - J.T @ (z - y)
                shortfall = max(0., load - float(c @ z))
                value = float((sr*residual)@(sr*residual) + (sf*(z-target))@(sf*(z-target))
                              + (sd*d)@(sd*d) + (sl*shortfall)**2)
                weighted = sr*sr*residual
                grad = np.r_[2.*(A@weighted + sd*sd*d),
                             -2.*(J@weighted) + 2.*sf*sf*(z-target) - 2.*sl*sl*shortfall*c]
                if not np.isfinite(value) or not np.isfinite(grad).all():
                    raise FloatingPointError('nonfinite_objective_or_gradient')
                return value, grad, residual, shortfall

            def diagnostics(x, gradient):
                mapping = lipschitz*(x-projection(x-gradient/lipschitz))
                pg_d = float(np.max(abs(mapping[:21])))
                pg_z = float(np.max(abs(mapping[21:]))) if P else 0.
                box = _box_normal_cone(gradient[:21], x[:21], lo, hi)
                normal = _normal_intent_cone(gradient[21:], x[21:], groups, caps, budget)
                kd, kz = box['stationarity_inf'], normal['stationarity_inf']
                tol_d = atol + rtol*max(1., float(np.max(abs(x[:21]))))
                tol_z = atol + rtol*max(1., float(np.max(abs(x[21:])))) if P else atol+rtol
                tol_comp = atol+rtol
                z = x[21:]
                violations = dict(effort_box=float(max(0., np.max(lo-x[:21]), np.max(x[:21]-hi))),
                    negative_normal=float(max(0., -np.min(z))) if P else 0.,
                    digit_caps=float(max([0.]+[float(z[ids].sum()-cap) for ids,cap in zip(groups,caps)])),
                    total_budget=float(max(0., z.sum()-budget)))
                done = (kd <= tol_d and kz <= tol_z and normal['complementarity_inf'] <= tol_comp
                        and normal['dual_nonnegativity_violation'] <= atol and max(violations.values()) <= atol)
                record = dict(box=box, normal=normal, pg_d=pg_d, pg_z=pg_z,
                              complementarity_tolerance=tol_comp)
                return done, kd, kz, tol_d, tol_z, violations, record

            x = projection(np.r_[np.zeros(21), target])
            velocity = x.copy()
            acceleration = 1.
            objective, gradient, residual, shortfall = evaluate(x)
            done, kd, kz, td, tz, violations, kkt = diagnostics(x, gradient)
            restarts = 0
            iterations = 0
            stagnated = False
            for iterations in range(1, int(max_iterations)+1):
                if done:
                    iterations -= 1
                    break
                _, gv, _, _ = evaluate(velocity)
                candidate = projection(velocity-gv/lipschitz)
                next_value, next_grad, next_residual, next_shortfall = evaluate(candidate)
                allowance = 64*np.finfo(float).eps*max(1., abs(objective), abs(next_value))
                if next_value > objective + allowance:
                    acceleration = 1.
                    velocity = x.copy()
                    candidate = projection(x-gradient/lipschitz)
                    next_value, next_grad, next_residual, next_shortfall = evaluate(candidate)
                    restarts += 1
                same_candidate = np.array_equal(candidate, x)
                new_acceleration = .5*(1.+np.sqrt(1.+4.*acceleration*acceleration))
                velocity = candidate + ((acceleration-1.)/new_acceleration)*(candidate-x)
                x = candidate
                acceleration = new_acceleration
                objective, gradient, residual, shortfall = next_value, next_grad, next_residual, next_shortfall
                done, kd, kz, td, tz, violations, kkt = diagnostics(x, gradient)
                if not done and same_candidate and np.array_equal(projection(x-gradient/lipschitz), x):
                    stagnated = True
                    break
            status = 'converged' if done else 'numerical_stagnation' if stagnated else 'nonconvergence'
            result_diagnostics = dict(**scope, lipschitz_upper_bound=lipschitz, monotone_restarts=restarts,
                fixed_original_p0_21=p0.tolist(), original_z_intent_p=target.tolist(),
                projected_gradient_inf_d21=kkt['pg_d'], projected_gradient_inf_zP=kkt['pg_z'],
                projected_gradient_is_diagnostic_only=True,
                direct_KKT_inf_d21=kd, direct_KKT_inf_zP=kz,
                direct_box_KKT=kkt['box'], direct_normal_KKT=kkt['normal'],
                complementarity_tolerance=kkt['complementarity_tolerance'],
                numerical_tolerance_d21=td, numerical_tolerance_zP=tz,
                primal_violations=violations, candidate_requires_status_check=not done,
                residual_scope='current effort-intent residual in native generalized units; not future pressure error',
                generalized_units25=['Nm']*24+['N'])
            reason = ('direct_normal_cone_KKT_and_complementarity_met' if done else
                      'zero_representable_step_with_failed_direct_KKT' if stagnated else 'predeclared_iteration_budget_exhausted')
            return IntentQPResult(status, reason,
                _readonly(x[:21]), _readonly(x[21:]), _readonly(p0+x[:21]), _readonly(residual),
                float(shortfall), float(objective), iterations, result_diagnostics)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError, ValueError) as exc:
        return IntentQPResult('numerical_failure', str(exc), diagnostics=scope)

"""Small feasible active-set lexicographic least squares, pure NumPy.

This kernel certifies ONLY its supplied finite linear task problems. Geometry
rows are virtual length-distribution proposals, never future robot/contact
predictions. Each converged level freezes its attained residual vector; lower
levels cannot trade it against a weight. No regularizer is silently added.
"""
from dataclasses import dataclass
from numbers import Real
import numpy as np


@dataclass(frozen=True)
class Level:
    name: str
    matrix: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class HierarchyResult:
    status: str
    x: np.ndarray
    levels: tuple
    diagnostics: dict

    @property
    def available(self):
        return self.status == 'CONVERGED_DECLARED_LINEAR_HIERARCHY'


def _array(value, shape, name):
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw.flat):
        raise ValueError(name + ': explicit numeric shape required')
    out = np.asarray(value, float)
    if not np.isfinite(out).all():
        raise ValueError(name + ': finite values required')
    return out.copy()


def _null(rows, n):
    if not len(rows):
        return np.eye(n)
    _, singular, vt = np.linalg.svd(rows, full_matrices=True)
    cutoff = max(rows.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0.)
    return vt[int(np.sum(singular > cutoff)):].T


def _independent_rows(rows, rhs, n):
    """Equivalent orthonormal equality rows, with dependent consistency checked."""
    if not len(rows):
        return np.empty((0, n)), np.empty(0)
    u, singular, vt = np.linalg.svd(rows, full_matrices=False)
    cutoff = max(rows.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0.)
    rank = int(np.sum(singular > cutoff))
    e = vt[:rank]
    f = (u[:, :rank].T@rhs)/singular[:rank]
    residual = rows@(e.T@f)-rhs
    if np.max(abs(residual), initial=0.) > 1e-8*max(1., np.max(abs(rhs), initial=0.)):
        raise ValueError('Inconsistent equality rows')
    return e, f


def solve_hierarchy(*, lower, upper, levels, initial=None, inequality_matrix=None,
                    inequality_upper=None, max_iterations=400, atol=1e-9, rtol=1e-8):
    """Solve levels in declared coordinates, starting from a genuine feasible x.

    Bounds and Gx<=h are hard. An infeasible start is explicitly rejected rather
    than projected through the task constraints. Physical units/scales belong
    to the caller. No equality or inequality is relaxed to gain OPEN progress.
    """
    raw = np.asarray(lower)
    if raw.ndim != 1 or not 1 <= len(raw) <= 64:
        raise ValueError('One to64 declared coordinates required')
    n = len(raw)
    lo = _array(lower, (n,), 'lower'); hi = _array(upper, (n,), 'upper')
    if np.any(lo > hi):
        raise ValueError('Empty genuine coordinate box')
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('Positive iteration budget required')
    if not np.isfinite([atol, rtol]).all() or atol <= 0 or rtol < 0:
        raise ValueError('Declared finite numerical tolerances required')
    x = np.zeros(n) if initial is None else _array(initial, (n,), 'feasible initial')
    g = np.empty((0, n)) if inequality_matrix is None else np.asarray(inequality_matrix, float)
    if g.ndim != 2 or g.shape[1] != n:
        raise ValueError('G requires M by N')
    g = _array(g, g.shape, 'G')
    h = np.empty(0) if inequality_upper is None else _array(inequality_upper, (len(g),), 'h')
    if len(h) != len(g):
        raise ValueError('Every inequality requires its declared bound')
    fixed = np.flatnonzero(lo == hi)
    equal = np.eye(n)[fixed]; rhs_equal = lo[fixed]
    box_ids = np.flatnonzero(lo != hi)
    g = np.vstack((g, np.eye(n)[box_ids], -np.eye(n)[box_ids]))
    h = np.r_[h, hi[box_ids], -lo[box_ids]]
    row_scale = np.max(abs(g), axis=1, initial=0.)
    zero = row_scale == 0
    if np.any(h[zero] < 0):
        raise ValueError('Impossible constant hard inequality')
    scaled = g[~zero]/row_scale[~zero, None]
    lengths = np.linalg.norm(scaled, axis=1)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        g = scaled/lengths[:, None]
        h = (h[~zero]/row_scale[~zero])/lengths
    if not np.isfinite(g).all() or not np.isfinite(h).all():
        raise FloatingPointError('Hard row normalization is not representable')
    tol = lambda v: atol+rtol*max(1., np.max(abs(v), initial=0.))
    violation = lambda v: max(0., np.max(g@v-h, initial=0.), np.max(abs(equal@v-rhs_equal), initial=0.))
    if violation(x) > tol(x):
        raise ValueError('Supplied initial is not feasible for actual hard domain')
    records = []; frozen = []; result_status = 'CONVERGED_DECLARED_LINEAR_HIERARCHY'
    for level in levels:
        if not isinstance(level, Level) or not level.name:
            raise TypeError('Named Level records required')
        a = np.asarray(level.matrix)
        if a.ndim != 2 or a.shape[1] != n:
            raise ValueError('Level matrix shape mismatch')
        a = _array(a, a.shape, level.name); b = _array(level.target, (len(a),), level.name+' target')
        e, f = _independent_rows(equal, rhs_equal, n)
        active = []
        def add(k):
            w = np.vstack((e, g[active])) if active else e
            z = _null(w, n)
            if np.linalg.norm(g[k]@z) <= 64*np.finfo(float).eps*max(1, n):
                return False
            active.append(k); return True
        slack = h-g@x
        for k in range(len(g)):
            rounding = 32*np.finfo(float).eps*max(1., abs(h[k]), float(abs(g[k])@abs(x)))
            if abs(slack[k]) <= rounding:
                add(k)
        status = 'ITERATION_LIMIT'; iterations = 0; stationarity = None; dual_min = None
        multipliers = np.empty(0)
        for iterations in range(1, max_iterations+1):
            w = np.vstack((e, g[active])) if active else e
            z = _null(w, n)
            p = z@np.linalg.lstsq(a@z, b-a@x, rcond=None)[0] if z.shape[1] and len(a) else np.zeros(n)
            if not np.isfinite(p).all():
                raise FloatingPointError('Unrepresentable least-squares step')
            gp = g@p; slack = h-g@x; alpha = 1.; blocker = None
            for k in range(len(g)):
                if k in active or gp[k] <= 0:
                    continue
                candidate = max(0., float(slack[k]))/float(gp[k])
                if candidate < alpha:
                    alpha = candidate; blocker = k
            x = x+alpha*p
            if not np.isfinite(x).all():
                raise FloatingPointError('Unrepresentable primal iterate')
            if violation(x) > tol(x):
                status = 'NUMERICAL_PRIMAL_FAILURE'; break
            if blocker is not None:
                if not add(blocker):
                    # A truly dependent tight row cannot block an exact
                    # nullspace step. Do not silently cross it after roundoff.
                    status = 'DEPENDENT_BLOCKER_NUMERICAL_FAILURE'; break
                continue
            gradient = 2*a.T@(a@x-b)
            if not np.isfinite(gradient).all():
                raise FloatingPointError('Unrepresentable original gradient')
            multipliers = np.linalg.lstsq(w.T, -gradient, rcond=None)[0] if len(w) else np.empty(0)
            if not np.isfinite(multipliers).all():
                raise FloatingPointError('Unrepresentable KKT multipliers')
            stationarity = float(np.max(abs(gradient+(w.T@multipliers if len(w) else 0.)), initial=0.))
            dual = multipliers[len(e):]
            dual_min = float(np.min(dual, initial=0.))
            if len(dual) and dual_min < -tol(gradient):
                active.pop(int(np.argmin(dual))); continue
            if stationarity <= tol(gradient) and dual_min >= -tol(gradient):
                status = 'KKT_CONVERGED'; break
            status = 'NUMERICAL_STATIONARITY_FAILURE'; break
        residual = a@x-b
        if not np.isfinite(residual).all() or not np.isfinite(float(residual@residual)):
            raise FloatingPointError('Unrepresentable original level loss')
        frozen_error = max([0.]+[float(np.max(abs(old_a@x-old_achieved), initial=0.)) for old_a, old_achieved in frozen])
        record = dict(name=level.name, status=status, iterations=iterations,
                      residual=residual.tolist(), squared_loss=float(residual@residual),
                      stationarity_inf=stationarity, inequality_dual_min=dual_min,
                      primal_violation=violation(x), preceding_residual_drift_inf=frozen_error,
                      active_inequalities=list(active), equality_rank=len(e),
                      numerical_atol=atol, numerical_rtol=rtol)
        records.append(record)
        if status != 'KKT_CONVERGED' or frozen_error > tol(x):
            result_status = 'UNAVAILABLE_UNCERTIFIED_HIERARCHY'; break
        achieved = a@x
        frozen.append((a.copy(), achieved.copy()))
        equal = np.vstack((equal, a)); rhs_equal = np.r_[rhs_equal, achieved]
    x.setflags(write=False)
    return HierarchyResult(result_status, x, tuple(records), dict(
        hard_constraint_violation=violation(x), levels_requested=len(levels),
        levels_completed=sum(r['status']=='KKT_CONVERGED' for r in records),
        physical_safety_certified=False, servo_response_predicted=False,
        higher_task_residuals_frozen=True, hidden_regularization=False))


def solve_geometry_release(tasks, *, reference_step_scale21):
    """Legal old-plane proposal -> OPEN progress -> low tangent -> nominal.

    Giving tangent minimization priority over OPEN would admit the trivial zero
    step as its optimum and can lock release. OPEN scalar progress therefore
    precedes tangent suppression; the attained scalar, not every axis, freezes.
    """
    scale = _array(reference_step_scale21, (21,), 'explicit positive reference units')
    if np.any(scale <= 0):
        raise ValueError('Positive per-reference scales required')
    if tasks.normal_matrix_m_per_reference is None:
        return None
    lo = tasks.step_lower21/scale; hi = tasks.step_upper21/scale
    nominal = np.minimum(hi, np.maximum(lo, tasks.open_target_increment21/scale))
    # Only finger opening is the primary progress task. Lift/wrist remain
    # available within the caller's actual authority as coordination variables.
    direction = nominal.copy(); direction[18:] = 0.
    norm = np.linalg.norm(direction)
    progress = direction[None, :]/norm if norm else np.zeros((1, 21))
    target = np.array([float((progress@nominal)[0])])
    tangent = tasks.tangent_matrix_m_per_reference*scale[None, :]
    size = float(np.max(np.sum(abs(tangent)*np.maximum(abs(lo), abs(hi))[None, :], axis=1), initial=0.))
    tangent_scale = size if size > 0 else 1.
    levels = (Level('OPEN_scalar_progress', progress, target),
              Level('minimum_tangent_at_retained_OPEN_progress', tangent/tangent_scale, tasks.tangent_target_m/tangent_scale),
              Level('closest_declared_nominal', np.eye(21), nominal))
    result = solve_hierarchy(lower=lo, upper=hi, levels=levels, initial=np.zeros(21),
        inequality_matrix=tasks.normal_matrix_m_per_reference*scale[None, :],
        inequality_upper=-tasks.normal_offset_m)
    return result

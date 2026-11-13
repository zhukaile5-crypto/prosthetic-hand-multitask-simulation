"""Finite convex LS hierarchy with current original-problem KKT witnesses.

Candidate route is fixed: certify anchor; otherwise analytic SLSQP; otherwise
ONE tight-face SVD/least-squares refinement at the original candidate tx. No
regularizer, clipping, face-threshold retry or relaxed original tolerance.
SciPy/HiGHS are real numerical dependencies, not a contact or servo model.
"""
from dataclasses import dataclass
from numbers import Integral, Real
import math, time
import numpy as np
from .numerics import ensure_numeric_environment

ensure_numeric_environment()
import scipy
from scipy.optimize import minimize, linprog

MAX_COORDINATES = 256
DUAL_LP_OPTIONS = dict(primal_feasibility_tolerance=1e-9,
    dual_feasibility_tolerance=1e-9, ipm_optimality_tolerance=1e-10)


def _array(value, shape, name):
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw.flat):
        raise ValueError(name+': explicit real numeric shape '+str(shape)+' required')
    a = np.asarray(value, float).copy()
    if not np.isfinite(a).all(): raise ValueError(name+': finite values required')
    return a


def _ro(value):
    a = np.asarray(value).copy(); a.setflags(write=False); return a


def _inf(a): return max((abs(float(v)) for v in np.asarray(a).flat), default=0.)


def _dot(A, x):
    a = np.asarray([math.fsum(float(v)*float(w) for v, w in zip(row, x)) for row in A], float)
    if not np.isfinite(a).all(): raise FloatingPointError('Unrepresentable original row evaluation')
    return a


def _rows(value, target, n, name):
    if value is None and target is None: return np.empty((0, n)), np.empty(0)
    if value is None or target is None: raise ValueError(name+': matrix/target must be co-declared')
    b = np.asarray(target, dtype=object)
    if b.ndim != 1: raise ValueError(name+': one target per row required')
    h = _array(target, b.shape, name+' target')
    if len(h) == 0 and type(value) in (list, tuple) and len(value) == 0: return np.empty((0, n)), h
    return _array(value, (len(h), n), name), h


def _normalize(A, b):
    """Positive row scaling, preserving every constant row and original index."""
    unit = np.empty_like(A); rhs = np.empty_like(b); factors = []
    for i, (row, bound) in enumerate(zip(A, b)):
        magnitude = _inf(row)
        if magnitude == 0:
            unit[i] = row; rhs[i] = bound; factors.append((0., 0.))
        else:
            scaled = row/magnitude
            length = math.sqrt(math.fsum(float(v)*float(v) for v in scaled))
            unit[i] = scaled/length; rhs[i] = (bound/magnitude)/length; factors.append((magnitude, length))
    if not np.isfinite(unit).all() or not np.isfinite(rhs).all(): raise FloatingPointError('Hard-row normalization not representable')
    return unit, rhs, factors


@dataclass(frozen=True)
class Level:
    name: str
    matrix: np.ndarray
    target: np.ndarray

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name: raise ValueError('Named objective required')
        raw = np.asarray(self.matrix, dtype=object)
        if raw.ndim != 2: raise ValueError('Objective matrix must be two dimensional')
        object.__setattr__(self, 'matrix', _ro(_array(self.matrix, raw.shape, 'objective matrix')))
        object.__setattr__(self, 'target', _ro(_array(self.target, (len(raw),), 'objective target')))


@dataclass(frozen=True)
class FiniteHierarchyResult:
    status: str
    x: np.ndarray
    levels: tuple
    diagnostics: dict
    completed_prefix: int = 0
    certified_prefix: tuple = ()
    fully_converged: bool = False
    failed_layer: object = None

    @property
    def available(self): return self.completed_prefix > 0

    @property
    def accepted_prefix(self): return self.certified_prefix


def _primal(G, h, E, f, x, frozen, atol, rtol):
    tx = atol+rtol*max(1., _inf(x))
    inequality = _dot(G, x)-h; equality = _dot(E, x)-f
    drift = max([0.]+[_inf(_dot(A, x)-y) for A, y in frozen])
    const = all(bound >= 0 or _inf(row) > 0 for row, bound in zip(G, h)) and all(bound == 0 or _inf(row) > 0 for row, bound in zip(E, f))
    ok = const and max(0., float(np.max(inequality, initial=0.))) <= tx and _inf(equality) <= tx and drift <= tx
    return dict(passed=bool(ok), coordinate_tolerance=tx, normalized_inequality_residual=_ro(inequality),
        original_equality_residual=_ro(equality), preceding_frozen_residual_drift_inf=drift,
        normalized_inequality_violation=max(0., float(np.max(inequality, initial=0.))),
        original_equality_error=_inf(equality), constant_rows_consistent=const)


def _dual_witness(G, h, E, gradient, x, tx, tg, max_iterations):
    """One HiGHS cone-dual LP, final acceptance NEVER uses LP status alone."""
    if _inf(gradient) <= tg:
        return np.zeros(len(G)), np.zeros(len(E)), dict(source='EXPLICIT_ZERO_DUAL_WITNESS_AT_ORIGINAL_TG', lp_called=False)
    slack = _dot(G, x)-h
    active = np.flatnonzero((abs(slack) <= tx) & np.any(G != 0, axis=1))
    Eu, _, efactors = _normalize(E, np.zeros(len(E)))
    eqids = np.flatnonzero(np.any(E != 0, axis=1))
    W = np.column_stack((G[active].T, Eu[eqids].T)); k = W.shape[1]
    if k == 0:
        return np.zeros(len(G)), np.zeros(len(E)), dict(source='NO_NONZERO_DUAL_CONE_GENERATORS', lp_called=False)
    gs = max(1., _inf(gradient)); g = gradient/gs
    AA = np.vstack((np.column_stack((W, -np.ones(len(x)))), np.column_stack((-W, -np.ones(len(x))))))
    bb = np.r_[-g, g]
    options = dict(DUAL_LP_OPTIONS, maxiter=int(max_iterations))
    began = time.perf_counter_ns()
    lp = linprog(np.r_[np.zeros(k), 1.], A_ub=AA, b_ub=bb,
        bounds=[(0., None)]*len(active)+[(None, None)]*len(eqids)+[(0., None)], method='highs', options=options)
    note = dict(source='HIGHS_CURRENT_POINT_CONE_DUAL_LP', lp_called=True, success=bool(lp.success),
        status=int(lp.status), message=str(lp.message), elapsed_ms=(time.perf_counter_ns()-began)/1e6,
        internal_dual_search_options=options, original_certificate_tolerances_unchanged=True,
        active_normalized_inequality_rows=active.tolist(), gradient_scale=gs)
    if lp.x is None or not np.isfinite(lp.x).all(): return None, None, note
    lam = np.zeros(len(G)); lam[active] = np.asarray(lp.x[:len(active)])*gs
    nu = np.zeros(len(E))
    for j, idx in enumerate(eqids):
        magnitude, length = efactors[idx]
        nu[idx] = ((float(lp.x[len(active)+j])*gs)/length)/magnitude
    if not np.isfinite(lam).all() or not np.isfinite(nu).all(): return None, None, dict(note, witness_unrepresentable=True)
    return lam, nu, note


def _certify(A, b, x, G, h, E, f, frozen, atol, rtol, max_iterations):
    began = time.perf_counter_ns()
    primal = _primal(G, h, E, f, x, frozen, atol, rtol)
    residual = _dot(A, x)-b; value = math.fsum(float(v)*float(v) for v in residual)
    gradient = 2.*_dot(A.T, residual)
    if not math.isfinite(value) or not np.isfinite(gradient).all(): raise FloatingPointError('Original objective/gradient unrepresentable')
    tg = atol+rtol*max(1., _inf(gradient)); tx = primal['coordinate_tolerance']; comp_tol = tg*max(1., _inf(x))
    lam = nu = None; note = dict(source='PRIMAL_REJECTED_NO_DUAL_SEARCH', lp_called=False)
    if primal['passed']: lam, nu, note = _dual_witness(G, h, E, gradient, x, tx, tg, max_iterations)
    present = lam is not None and nu is not None
    stationarity = dual_min = complementarity = tight = None
    if present:
        stationarity = _inf(gradient+_dot(G.T, lam)+_dot(E.T, nu))
        dual_min = min(0., float(np.min(lam, initial=0.)))
        slack = np.asarray(primal['normalized_inequality_residual']); complementarity = _inf(lam*slack)
        tight = _inf(slack[np.flatnonzero(lam != 0)])
    checks = dict(original_hard_primal=primal['passed'], original_previous_attainment=primal['preceding_frozen_residual_drift_inf'] <= tx,
        dual_witness_present=present, original_stationarity=present and stationarity <= tg,
        inequality_dual_nonnegative=present and dual_min >= -tg,
        complementarity=present and complementarity <= comp_tol, nonzero_dual_faces_tight=present and tight <= tx)
    return dict(checked=True, certified=all(checks.values()), checks=checks, gradient=_ro(gradient),
        coordinate_tolerance=tx, gradient_tolerance=tg, complementarity_tolerance=comp_tol,
        stationarity_inf=stationarity, inequality_dual_min=dual_min, complementarity_inf=complementarity,
        active_face_residual_inf=tight, original_hard_primal_violation=max(primal['normalized_inequality_violation'], primal['original_equality_error']),
        normalized_hard_primal_violation=max(primal['normalized_inequality_violation'], primal['original_equality_error']),
        preceding_frozen_residual_drift_inf=primal['preceding_frozen_residual_drift_inf'],
        full_normalized_inequality_dual=None if lam is None else _ro(lam), equality_raw_dual=None if nu is None else _ro(nu),
        full_equality_matrix=_ro(E), full_equality_target=_ro(f), dual_witness=note,
        residual=_ro(residual), objective=value, elapsed_ms=(time.perf_counter_ns()-began)/1e6,
        KKT_scope='Current original finite convex LS problem at supplied point; not a legacy active-set-loop status')


def _affine(E, f, anchor):
    n = len(anchor)
    if not len(E): return anchor.copy(), np.eye(n), dict(rank=0, rows=0, cutoff=0., affine_residual_inf=0.)
    Eu, fu, _ = _normalize(E, f); nonzero = np.any(Eu != 0, axis=1)
    W = Eu[nonzero]; rhs = fu[nonzero]
    if not len(W): return anchor.copy(), np.eye(n), dict(rank=0, rows=len(E), cutoff=0., affine_residual_inf=_inf(f))
    u, s, vt = np.linalg.svd(W, full_matrices=True)
    cutoff = max(W.shape)*np.finfo(float).eps*(s[0] if len(s) else 0.)
    rank = int(np.sum(s > cutoff)); Z = vt[rank:].T
    base = anchor.copy()
    if rank: base = anchor+vt[:rank].T@((u[:, :rank].T@(rhs-W@anchor))/s[:rank])
    return base, Z, dict(rank=rank, rows=len(E), cutoff=float(cutoff), singular_values=_ro(s),
        affine_residual_inf=_inf(_dot(E, base)-f), nullspace_residual_inf=_inf(E@Z))


def _slsqp_candidate(A, b, G, h, E, f, anchor, max_iterations, atol):
    base, Z, affine = _affine(E, f, anchor)
    if not Z.shape[1]: return base, dict(method='UNIQUE_AFFINE_POINT', affine=affine, iterations=0)
    R = A@Z; target = b-A@base; Q = G@Z; rhs = h-G@base
    y0 = Z.T@(anchor-base)
    constraints = [] if not len(G) else [dict(type='ineq', fun=lambda y: rhs-Q@y, jac=lambda y: -Q)]
    result = minimize(lambda y: float((R@y-target)@(R@y-target)), y0,
        jac=lambda y: 2.*R.T@(R@y-target), method='SLSQP', constraints=constraints,
        options=dict(maxiter=int(max_iterations), ftol=float(atol), disp=False))
    x = base+Z@np.asarray(result.x)
    return x, dict(method='SCIPY_SLSQP_ANALYTIC_GRADIENT_AFFINE_CANDIDATE', library_success=bool(result.success),
        library_status=int(result.status), library_message=str(result.message), iterations=int(result.nit),
        ftol_candidate_generation_only=float(atol), affine=affine, no_original_row_acceptance_relaxed=True)


def _refine_once(A, b, G, h, E, f, candidate, atol, rtol):
    tx = atol+rtol*max(1., _inf(candidate)); slack = _dot(G, candidate)-h
    tight = np.flatnonzero((abs(slack) <= tx) & np.any(G != 0, axis=1))
    W = np.vstack((E, G[tight])); rhs = np.r_[f, h[tight]]
    base, Z, affine = _affine(W, rhs, candidate)
    if affine['affine_residual_inf'] > tx: raise FloatingPointError('Once-selected tight face inconsistent at ORIGINAL tx')
    if Z.shape[1]:
        y, _, rank, singular = np.linalg.lstsq(A@Z, b-A@base, rcond=None); x = base+Z@y
    else: x = base; rank = 0; singular = np.empty(0)
    return x, dict(method='ONE_FIXED_ORIGINAL_TX_TIGHT_FACE_SVD_LSTSQ', tight_inequality_rows=tight.tolist(),
        face_selection_tolerance=tx, face_threshold_attempts=1, affine=affine,
        reduced_LS_rank=int(rank), reduced_LS_singular_values=_ro(singular), original_point=_ro(candidate))


def solve_hierarchy(*, lower, upper, levels, initial=None, inequality_matrix=None, inequality_upper=None,
    max_iterations=400, atol=1e-9, rtol=1e-8, equality_matrix=None, equality_target=None):
    """V21-compatible result/certificates, with optional explicit original E/f.

    Bounds/explicit hard rows/fixed coordinates remain unchanged. Prior layers
    freeze original attained vectors. Only current original-problem KKT can
    certify a point, regardless of any SLSQP/HiGHS status. Empty prefix retains
    the original validated anchor, never an uncommitted optimizer trial.
    """
    began = time.perf_counter_ns(); raw = np.asarray(lower, dtype=object)
    if raw.ndim != 1 or not 1 <= len(raw) <= MAX_COORDINATES: raise ValueError('One to256 finite coordinates required')
    n = len(raw); lo = _array(lower, (n,), 'lower'); hi = _array(upper, (n,), 'upper')
    if np.any(lo > hi): raise ValueError('Empty original finite box')
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral) or max_iterations < 1: raise ValueError('Positive iteration budget required')
    if isinstance(atol, (bool, np.bool_)) or isinstance(rtol, (bool, np.bool_)) or not isinstance(atol, Real) or not isinstance(rtol, Real) or not np.isfinite([atol, rtol]).all() or atol <= 0 or rtol < 0:
        raise ValueError('Original finite positive atol/nonnegative rtol required')
    atol = float(atol); rtol = float(rtol)
    x = np.zeros(n) if initial is None else _array(initial, (n,), 'original initial')
    anchor = x.copy(); Ginput, hinput = _rows(inequality_matrix, inequality_upper, n, 'original G/h')
    Eextra, fextra = _rows(equality_matrix, equality_target, n, 'original E/f')
    free = np.flatnonzero(lo != hi); fixed = np.flatnonzero(lo == hi)
    Graw = np.vstack((Ginput, np.eye(n)[free], -np.eye(n)[free])); hraw = np.r_[hinput, hi[free], -lo[free]]
    Ebase = np.vstack((np.eye(n)[fixed], Eextra)); fbase = np.r_[lo[fixed], fextra]
    if any(_inf(row) == 0 and bound < 0 for row, bound in zip(Graw, hraw)): raise ValueError('Impossible constant original inequality')
    if any(_inf(row) == 0 and bound != 0 for row, bound in zip(Ebase, fbase)): raise ValueError('Impossible constant original equality')
    G, h, normalization = _normalize(Graw, hraw)
    initial_primal = _primal(G, h, Ebase, fbase, x, (), atol, rtol)
    if not initial_primal['passed']: raise ValueError('Original initial is not feasible; no silent clipping/projection')
    levels = tuple(levels); parsed = []
    for level in levels:
        if not hasattr(level, 'name') or not isinstance(level.name, str) or not level.name: raise ValueError('Named level required')
        target = np.asarray(level.target, dtype=object)
        if target.ndim != 1: raise ValueError('One objective target per row required')
        parsed.append((level.name, _array(level.matrix, (len(target), n), 'original objective A'), _array(level.target, target.shape, 'original objective b')))
    domain = dict(lower=_ro(lo), upper=_ro(hi), original_inequality_matrix_with_box=_ro(Graw),
        original_inequality_upper_with_box=_ro(hraw), fixed_equality_matrix=_ro(Ebase), fixed_equality_target=_ro(fbase),
        normalized_inequality_matrix=_ro(G), normalized_inequality_upper=_ro(h),
        zero_rows_retained=True, explicit_extra_equality_rows=len(Eextra))
    frozen = []; certs = []; records = []; failed = None
    for index, (name, A, b) in enumerate(parsed):
        start = time.perf_counter_ns(); layer_anchor = x.copy(); attempts = []
        E = np.vstack([Ebase]+[a for a, _ in frozen]); f = np.concatenate([fbase]+[y for _, y in frozen])
        accepted = None; kkt = None
        try:
            kkt = _certify(A, b, x, G, h, E, f, frozen, atol, rtol, int(max_iterations))
            attempts.append(dict(candidate='original_feasible_layer_anchor', point=_ro(x), KKT=kkt))
            if kkt['certified']: accepted = x.copy()
            else:
                candidate, note = _slsqp_candidate(A, b, G, h, E, f, x, int(max_iterations), atol)
                if not np.isfinite(candidate).all(): raise FloatingPointError('Nonfinite SLSQP candidate')
                kkt = _certify(A, b, candidate, G, h, E, f, frozen, atol, rtol, int(max_iterations))
                attempts.append(dict(candidate='SLSQP', point=_ro(candidate), generation=note, KKT=kkt))
                if kkt['certified']: accepted = candidate
                else:
                    refined, face = _refine_once(A, b, G, h, E, f, candidate, atol, rtol)
                    if not np.isfinite(refined).all(): raise FloatingPointError('Nonfinite once-face candidate')
                    kkt = _certify(A, b, refined, G, h, E, f, frozen, atol, rtol, int(max_iterations))
                    attempts.append(dict(candidate='one_tight_face', point=_ro(refined), generation=face, KKT=kkt))
                    if kkt['certified']: accepted = refined
        except (ValueError, TypeError, FloatingPointError, OverflowError, RuntimeError, np.linalg.LinAlgError) as exc:
            attempts.append(dict(candidate='rejected_generation_or_check', error=type(exc).__name__+': '+str(exc)))
        elapsed = (time.perf_counter_ns()-start)/1e6
        record = dict(index=index, name=name, status='CURRENT_ORIGINAL_PROBLEM_KKT_CERTIFIED' if accepted is not None else 'UNAVAILABLE_NO_ORIGINAL_KKT_WITNESS',
            certified=accepted is not None, attempts=tuple(attempts), elapsed_ms=elapsed,
            point_role='certified_layer_point' if accepted is not None else 'failed_uncommitted_candidates',
            numerical_atol=atol, numerical_rtol=rtol, returned_failed_trial=False)
        records.append(record)
        if accepted is None:
            failed = dict(index=index, name=name, status=record['status'], record=record, failed_trial_returned=False)
            x = layer_anchor; break
        x = np.asarray(accepted).copy(); attained = _dot(A, x); residual = attained-b
        certificate = dict(index=index, name=name, x=_ro(x), matrix=_ro(A), target=_ro(b), attained=_ro(attained),
            residual=_ro(residual), squared_loss=math.fsum(float(v)*float(v) for v in residual), hard_domain=domain,
            previous_frozen=tuple(dict(matrix=_ro(a), attained=_ro(y)) for a, y in frozen), KKT=kkt, record=record,
            original_problem_not_retargeted=True, historical_snapshot_not_replayed=True,
            candidate_source=attempts[-1]['candidate'], backend='SCIPY_SLSQP_ONE_FACE_HIGHS_ORIGINAL_KKT')
        certs.append(certificate); frozen.append((A.copy(), attained.copy()))
    fully = bool(parsed) and failed is None and len(certs) == len(parsed)
    status = 'CONVERGED_FULL_CERTIFIED_FINITE_HIERARCHY' if fully else 'CERTIFIED_FINITE_PREFIX' if certs else 'FEASIBLE_ANCHOR_NO_TASK_CERTIFICATE'
    diagnostics = dict(original_feasible_anchor=_ro(anchor), hard_domain=domain,
        levels_requested=len(parsed), levels_attempted=len(records), levels_completed=len(certs), failed_layer=failed,
        returned_point_role='last_certified_prefix_point' if certs else 'validated_original_anchor_without_task_certificate',
        failed_trial_returned=False, available_requires_nonempty_certified_prefix=True,
        backend_dependencies=dict(numpy_version=np.__version__, scipy_version=scipy.__version__,
            candidate='scipy.optimize.minimize SLSQP analytic gradient', dual='scipy.optimize.linprog HiGHS cone-dual LP'),
        no_regularizer=True, no_original_objective_or_hard_domain_changed=True, no_clipping=True,
        original_numerical_atol=atol, original_numerical_rtol=rtol, face_threshold_attempts_at_most_one_per_layer=True,
        elapsed_ms=(time.perf_counter_ns()-began)/1e6, timing_is_single_host_walltime_not_realtime_guarantee=True,
        pressure_prediction_available=False, physics_safety_certified=False)
    return FiniteHierarchyResult(status, _ro(x), tuple(records), diagnostics, len(certs), tuple(certs), fully, failed)

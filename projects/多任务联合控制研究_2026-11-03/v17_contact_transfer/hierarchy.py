"""V17 finite degenerate-face repair around frozen5564 lexicographic LS.

This kernel certifies ONLY its supplied finite linear task problems. Geometry
rows are virtual length-distribution proposals, never future robot/contact
predictions. Each converged level freezes its attained residual vector; lower
levels cannot trade it against a weight. No regularizer is silently added.

Ordinary iterations remain floating LS. Only a tight zero-length blocker whose
normal is numerically dependent can request a proof/repair. First propagate
exact sparse coordinate constraints; only the remaining small quotient may
use float-as-exact-rational RREF. No epsilon decides exact row dependence.
Proof budgets:100000 sparse visits,20000 rational operations,8192 integer bits.
Budget/nonrepresentability failure is unavailable, never a dropped hard row.
Repair is limited to nullspace arithmetic roundoff, not a new physical margin.
All original hard rows, KKT and attained higher-level residual checks remain.
"""
from dataclasses import dataclass
from numbers import Real
from fractions import Fraction
import math,time
import numpy as np

SPARSE_VISIT_BUDGET=100000
EXACT_OPERATION_BUDGET=20000
EXACT_INTEGER_BIT_BUDGET=8192


class _FaceBudget(Exception):pass


def _rational_basis(rows,n,stats):
    basis=[]
    def accounted(q):
        stats['rational_ops']+=1
        bits=max(q.numerator.bit_length(),q.denominator.bit_length())
        stats['max_integer_bits']=max(stats['max_integer_bits'],bits)
        if stats['rational_ops']>EXACT_OPERATION_BUDGET or bits>EXACT_INTEGER_BIT_BUDGET:
            raise _FaceBudget('exact_row_span_proof_budget_exceeded')
        return q
    def eliminate(q,b,amount):
        for j,v in enumerate(b):
            if v:q[j]=accounted(q[j]-amount*v)
    for row in rows:
        q=[Fraction.from_float(float(v)) for v in row]
        for pivot,b in basis:
            if q[pivot]:eliminate(q,b,q[pivot])
        pivot=next((j for j,v in enumerate(q) if v),None)
        if pivot is None:continue
        divisor=q[pivot]
        q=[accounted(v/divisor) if v else v for v in q]
        for _,b in basis:
            if b[pivot]:eliminate(b,q,b[pivot])
        basis.append((pivot,q));basis.sort(key=lambda z:z[0])
        if len(basis)==n:break
    return basis,accounted


def _face_proof(raw_w,n):
    stats=dict(sparse_visits=0,rational_ops=0,max_integer_bits=0,rational_used=False)
    supports=[tuple(np.flatnonzero(row)) for row in raw_w]
    fixed=set();changed=True
    while changed:
        changed=False
        for support in supports:
            stats['sparse_visits']+=len(support)
            if stats['sparse_visits']>SPARSE_VISIT_BUDGET:raise _FaceBudget('sparse_face_proof_budget_exceeded')
            remaining=[j for j in support if j not in fixed]
            if len(remaining)==1:fixed.add(remaining[0]);changed=True
    free=[j for j in range(n) if j not in fixed]
    return dict(fixed=fixed,free=free,rows=raw_w[:,free],basis=None,accounted=None,stats=stats)


def _dependent_exact(proof,row):
    q=[Fraction.from_float(float(row[j])) for j in proof['free']]
    if not any(q):return True
    if proof['basis'] is None:
        proof['stats']['rational_used']=True
        proof['basis'],proof['accounted']=_rational_basis(proof['rows'],len(proof['free']),proof['stats'])
    for pivot,b in proof['basis']:
        if q[pivot]:
            amount=q[pivot]
            for j,v in enumerate(b):
                if v:q[j]=proof['accounted'](q[j]-amount*v)
    return not any(q)


def _repair_degenerate_direction(*,p,z,g,h,x,gp,active,raw_equal,raw_g,cache):
    """Return a repaired direction and only exactly certified zero derivatives.

    This does not remove any inequality. A dependent face has mathematically
    zero directional derivative in the active homogeneous nullspace. The raw
    current point and full hard domain are still checked after every candidate.
    """
    n=len(p);pnorm=float(np.max(abs(p),initial=0.))
    if pnorm==0:return p,[],None
    eps=64*np.finfo(float).eps*max(1,n,len(raw_equal)+len(active))
    slack=h-g@x;candidates=[]
    for k in range(len(g)):
        if k in active or gp[k]<=0:continue
        rounding=64*np.finfo(float).eps*(abs(h[k])+float(abs(g[k])@abs(x)))
        if (abs(slack[k])<=rounding and max(0.,float(slack[k]))/float(gp[k])<1.
                and gp[k]<=eps*float(abs(g[k]).sum())*pnorm
                and np.linalg.norm(g[k]@z)<=64*np.finfo(float).eps*max(1,n)):
            candidates.append(k)
    if not candidates:return p,[],None
    began=time.perf_counter_ns();key=tuple(active);cached=key in cache
    note=dict(candidates=candidates,cached=cached,certified_dependent_rows=[],corrected=False)
    try:
        if key not in cache:
            raw_w=np.vstack((raw_equal,raw_g[active])) if active else raw_equal
            cache[key]=_face_proof(raw_w,n)
        proof=cache[key]
        dependent=[k for k in candidates if _dependent_exact(proof,raw_g[k])]
        note['certified_dependent_rows']=dependent
        if not dependent:return p,[],note
        repaired=p.copy()
        for j in proof['fixed']:repaired[j]=0.
        if proof['basis'] is not None:
            pivots={j for j,_ in proof['basis']};remaining=[j for j in range(len(proof['free'])) if j not in pivots]
            for pivot,row in proof['basis']:
                repaired[proof['free'][pivot]]=-math.fsum(float(row[j])*float(p[proof['free'][j]]) for j in remaining if row[j])
        correction=float(np.max(abs(repaired-p),initial=0.))
        raw_w=np.vstack((raw_equal,raw_g[active])) if active else raw_equal
        scale=np.max(abs(raw_w),axis=1,initial=0.);nonzero=scale>0;scaled=raw_w[nonzero]/scale[nonzero,None]
        residual=abs(scaled@repaired);bounds=eps*pnorm*np.sum(abs(scaled),axis=1)
        note.update(correction_inf=correction,arithmetic_correction_allowance=eps*pnorm,
            raw_homogeneous_residual_inf=float(np.max(residual,initial=0.)))
        if (not np.isfinite(repaired).all() or correction>eps*pnorm or np.any(residual>bounds)):
            note['reason']='not_merely_nullspace_arithmetic_drift';return p,[],note
        newgp=g@repaired
        zero_rows=[k for k in dependent if abs(newgp[k])<=eps*float(abs(g[k]).sum())*pnorm]
        note['corrected']=True
        return repaired,zero_rows,note
    except (_FaceBudget,OverflowError,ValueError,ZeroDivisionError) as exc:
        note['reason']=str(exc);return p,[],note
    finally:
        if key in cache:note.update(cache[key]['stats'])
        note['elapsed_ms']=(time.perf_counter_ns()-began)/1e6


def _original_primal_violation(raw_g,raw_h,equal,rhs,x):
    """Independent fsum check of every original hard row, without norm overflow."""
    violation=0.
    for row,bound in zip(raw_g,raw_h):
        scale=float(np.max(abs(row),initial=0.))
        if scale==0:
            violation=max(violation,-float(bound));continue
        normalized=row/scale
        value=(math.fsum(float(a)*float(b) for a,b in zip(normalized,x))-float(bound/scale))/float(np.linalg.norm(normalized))
        if not np.isfinite(value):raise FloatingPointError('Unrepresentable original hard primal check')
        violation=max(violation,value)
    for row,bound in zip(equal,rhs):
        value=abs(math.fsum(float(a)*float(b) for a,b in zip(row,x))-float(bound))
        if not np.isfinite(value):raise FloatingPointError('Unrepresentable original equality check')
        violation=max(violation,value)
    return max(0.,float(violation))


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
    original_g=g.copy();original_h=h.copy()
    row_scale = np.max(abs(g), axis=1, initial=0.)
    zero = row_scale == 0
    if np.any(h[zero] < 0):
        raise ValueError('Impossible constant hard inequality')
    scaled = g[~zero]/row_scale[~zero, None]
    raw_g=original_g[~zero].copy()
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
    records = []; frozen = []; result_status = 'CONVERGED_DECLARED_LINEAR_HIERARCHY';all_repairs=[]
    for level in levels:
        if not isinstance(level, Level) or not level.name:
            raise TypeError('Named Level records required')
        a = np.asarray(level.matrix)
        if a.ndim != 2 or a.shape[1] != n:
            raise ValueError('Level matrix shape mismatch')
        a = _array(a, a.shape, level.name); b = _array(level.target, (len(a),), level.name+' target')
        e, f = _independent_rows(equal, rhs_equal, n)
        active = [];face_cache={};level_repairs=[]
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
            gp = g@p
            p,zero_derivatives,repair=_repair_degenerate_direction(p=p,z=z,g=g,h=h,x=x,gp=gp,
                active=active,raw_equal=equal,raw_g=raw_g,cache=face_cache)
            if repair is not None:level_repairs.append(dict(iteration=iterations,**repair))
            gp=g@p
            for row in zero_derivatives:gp[row]=0.
            slack = h-g@x; alpha = 1.; blocker = None
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
        original_primal=_original_primal_violation(original_g,original_h,equal,rhs_equal,x)
        if original_primal>tol(x):status='NUMERICAL_ORIGINAL_PRIMAL_FAILURE'
        record = dict(name=level.name, status=status, iterations=iterations,
                      residual=residual.tolist(), squared_loss=float(residual@residual),
                      stationarity_inf=stationarity, inequality_dual_min=dual_min,
                      primal_violation=violation(x), preceding_residual_drift_inf=frozen_error,
                      active_inequalities=list(active), equality_rank=len(e),
                      original_hard_primal_violation=original_primal,degenerate_face_repairs=level_repairs,
                      numerical_atol=atol, numerical_rtol=rtol)
        records.append(record)
        all_repairs.extend(level_repairs)
        if status != 'KKT_CONVERGED' or frozen_error > tol(x):
            result_status = 'UNAVAILABLE_UNCERTIFIED_HIERARCHY'; break
        achieved = a@x
        frozen.append((a.copy(), achieved.copy()))
        equal = np.vstack((equal, a)); rhs_equal = np.r_[rhs_equal, achieved]
    final_original_primal=_original_primal_violation(original_g,original_h,equal,rhs_equal,x)
    if final_original_primal>tol(x):result_status='UNAVAILABLE_UNCERTIFIED_HIERARCHY'
    x.setflags(write=False)
    return HierarchyResult(result_status, x, tuple(records), dict(
        hard_constraint_violation=violation(x), levels_requested=len(levels),
        original_hard_constraint_violation=final_original_primal,
        levels_completed=sum(r['status']=='KKT_CONVERGED' for r in records),
        physical_safety_certified=False, servo_response_predicted=False,
        higher_task_residuals_frozen=True, hidden_regularization=False,
        all_original_hard_rows_retained=True,
        degenerate_face_method='exact_sparse_or_rational_span_and_roundoff_pivot_repair',
        degenerate_face_proof_budgets=dict(sparse_visits=SPARSE_VISIT_BUDGET,rational_operations=EXACT_OPERATION_BUDGET,integer_bits=EXACT_INTEGER_BIT_BUDGET),
        degenerate_face_events=len(all_repairs),degenerate_face_elapsed_ms=sum(v['elapsed_ms'] for v in all_repairs),
        exact_rational_events=sum(bool(v.get('rational_used')) for v in all_repairs)))


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

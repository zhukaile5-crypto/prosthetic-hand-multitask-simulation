"""V21 certified-prefix bookkeeping around frozen V17 finite hierarchy.

Original active-set primal/dual iterations, tolerances and exact-repair budgets
are copied unchanged from sourcea3c9f090. The sole resource-domain extension is
one..256 coordinates instead of64 (21+128 slacks is149). This is NOT a complete
larger-dimensional accuracy, resource or real-time certificate.

Each layer is snapshotted exactly when its original KKT iteration completes
AND fresh original-primal/dual/stationarity/complementarity/frozen checks pass.
A later failure returns the last saved certified point, never the failed trial.
No prefix is re-solved to fabricate a historical certificate. With no certified
layer, x is the validated actual feasible initial anchor and available=False.
accepted_prefix is the tuple alias; completed_prefix is its integer length.
fully_converged is separate from availability. Attempt records may refer to a
failed UNCOMMITTED trial and are labelled accordingly. All arrays in snapshots
are copied/read-only, sharing no mutable working state.

Snapshots certify only their own original hard domain and problem. A caller
adding/removing rows, changing clock/mode/scales, or emitting a rounded/different
command must validate the new point/problem and preserve its provenance chain.
No original caller problem is silently retargeted to the returned anchor.

V17 copied numerical-method notes follow:

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
class PrefixHierarchyResult:
    status:str
    x:np.ndarray
    levels:tuple
    diagnostics:dict
    certified_prefix:tuple=()
    completed_prefix:int=0
    fully_converged:bool=False
    failed_layer:object=None

    @property
    def available(self):return self.completed_prefix>=1

    @property
    def accepted_prefix(self):return self.certified_prefix

# Explicit compatibility name; availability now means a nonempty certified
# PREFIX. Callers must inspect fully_converged and their required layer count.
HierarchyResult=PrefixHierarchyResult
MAX_COORDINATES=256
SOURCE_KERNEL_SHA256='a3c9f090a5cf4368d7d1ab36344b0aac4cfd76aac3c0367a298a38fdf2932457'


def _readonly(value):
    out=np.asarray(value).copy();out.setflags(write=False);return out


def _layer_certificate(*,a,b,x,status,e,f,g,h,active,w,multipliers,
                       original_g,original_h,equal,rhs_equal,frozen,atol,rtol):
    """Verify the original LS KKT record at THIS iterate, no primal re-solve.

    The active-set loop and its dual solve are unchanged. We re-evaluate the
    original gradient/adjoint and primal/frozen checks using its actual final
    working matrix and multipliers, rather than trusting a green status string.
    Complementarity has coordinate*gradient units; its allowance is the same
    original gradient tolerance times the original coordinate magnitude.
    """
    gradient=2*a.T@(a@x-b)
    tx=atol+rtol*max(1.,float(np.max(abs(x),initial=0.)))
    tg=atol+rtol*max(1.,float(np.max(abs(gradient),initial=0.)))
    correct_w=np.vstack((e,g[active])) if active else e
    shapes=(w.shape==correct_w.shape and np.array_equal(w,correct_w) and len(multipliers)==len(w))
    finite=bool(np.isfinite(gradient).all() and np.isfinite(multipliers).all())
    stationarity=None;dualmin=None;comp=None;active_tight=None;eq_error=None;full_dual=None
    if shapes and finite:
        dual=multipliers[len(e):]
        full_dual=np.zeros(len(g));full_dual[active]=dual
        stationarity=float(np.max(abs(gradient+(w.T@multipliers if len(w) else 0.)),initial=0.))
        dualmin=float(np.min(dual,initial=0.))
        slack=g@x-h
        comp=float(np.max(abs(full_dual*slack),initial=0.))
        active_tight=float(np.max(abs(slack[active]),initial=0.))
        eq_error=float(np.max(abs(e@x-f),initial=0.))
    original_primal=_original_primal_violation(original_g,original_h,equal,rhs_equal,x)
    normalized_primal=max(0.,float(np.max(g@x-h,initial=0.)),float(np.max(abs(equal@x-rhs_equal),initial=0.)))
    frozen_error=max([0.]+[float(np.max(abs(old_a@x-y),initial=0.)) for old_a,y in frozen])
    comp_limit=tg*max(1.,float(np.max(abs(x),initial=0.)))
    checks=dict(original_loop_reported_KKT=status=='KKT_CONVERGED',
        original_working_set_matches_dual_record=bool(shapes),all_KKT_values_finite=finite,
        original_hard_primal=bool(np.isfinite(original_primal) and original_primal<=tx),
        normalized_hard_primal=bool(np.isfinite(normalized_primal) and normalized_primal<=tx),
        previous_frozen_residuals=bool(np.isfinite(frozen_error) and frozen_error<=tx),
        fresh_original_stationarity=stationarity is not None and stationarity<=tg,
        inequality_dual_nonnegative=dualmin is not None and dualmin>=-tg,
        complementarity=comp is not None and comp<=comp_limit,
        working_faces_tight=active_tight is not None and active_tight<=tx,
        equality_basis_primal=eq_error is not None and eq_error<=tx)
    return dict(certified=all(checks.values()),checks=checks,
        gradient=_readonly(gradient),gradient_tolerance=tg,coordinate_tolerance=tx,
        stationarity_inf=stationarity,inequality_dual_min=dualmin,
        complementarity_inf=comp,complementarity_tolerance=comp_limit,
        active_face_residual_inf=active_tight,equality_basis_residual_inf=eq_error,
        original_hard_primal_violation=original_primal,normalized_hard_primal_violation=normalized_primal,
        preceding_frozen_residual_drift_inf=frozen_error,
        equality_basis=_readonly(e),equality_basis_target=_readonly(f),
        original_loop_working_matrix=_readonly(w),original_loop_multipliers=_readonly(multipliers),
        full_normalized_inequality_dual=None if full_dual is None else _readonly(full_dual),
        active_inequalities=tuple(active),verification_is_not_a_primal_or_prefix_replay=True,
        KKT_scope='declared finite linear LS layer at its own saved point and original numerical tolerances')

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
    levels=tuple(levels)
    raw = np.asarray(lower)
    if raw.ndim != 1 or not 1 <= len(raw) <= MAX_COORDINATES:
        raise ValueError('One to256 declared coordinates required')
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
    initial_original_primal=_original_primal_violation(original_g,original_h,equal,rhs_equal,x)
    if initial_original_primal>tol(x):
        raise ValueError('Supplied initial fails ORIGINAL actual hard domain')
    anchor=x.copy();fixed_equal=equal.copy();fixed_rhs=rhs_equal.copy()
    hard_domain=dict(lower=_readonly(lo),upper=_readonly(hi),
        original_inequality_matrix_with_box=_readonly(original_g),original_inequality_upper_with_box=_readonly(original_h),
        fixed_equality_matrix=_readonly(fixed_equal),fixed_equality_target=_readonly(fixed_rhs),
        normalized_inequality_matrix=_readonly(g),normalized_inequality_upper=_readonly(h))
    records=[];frozen=[];all_repairs=[];certified=[];failed_layer=None
    for layer_index,level in enumerate(levels):
        record=None;iterations=0;level_repairs=[]
        try:
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
            certificate=_layer_certificate(a=a,b=b,x=x,status=status,e=e,f=f,g=g,h=h,active=active,w=w,
                multipliers=multipliers,original_g=original_g,original_h=original_h,
                equal=equal,rhs_equal=rhs_equal,frozen=frozen,atol=atol,rtol=rtol)
            record.update(index=layer_index,certified=certificate['certified'],
                point_role='certified_layer_point' if certificate['certified'] else 'failed_uncommitted_trial',
                certificate_checks=dict(certificate['checks']))
            records.append(record);all_repairs.extend(level_repairs)
            if not certificate['certified']:
                failed_layer=dict(index=layer_index,name=level.name,status=status if status!='KKT_CONVERGED' else 'POST_STATUS_CERTIFICATE_FAILURE',
                    certificate=certificate,record=record,failed_trial_returned=False)
                break
            achieved=a@x
            snapshot=dict(index=layer_index,name=level.name,x=_readonly(x),matrix=_readonly(a),target=_readonly(b),
                attained=_readonly(achieved),residual=_readonly(achieved-b),squared_loss=float((achieved-b)@(achieved-b)),
                hard_domain=hard_domain,
                previous_frozen=tuple(dict(matrix=_readonly(old_a),attained=_readonly(y)) for old_a,y in frozen),
                KKT=certificate,record=dict(record),
                source_kernel_sha256=SOURCE_KERNEL_SHA256,
                original_problem_not_retargeted=True,historical_snapshot_not_replayed=True)
            certified.append(snapshot)
            frozen.append((a.copy(),achieved.copy()))
            equal=np.vstack((equal,a));rhs_equal=np.r_[rhs_equal,achieved]
        except (ValueError,TypeError,AttributeError,FloatingPointError,OverflowError,np.linalg.LinAlgError,_FaceBudget) as exc:
            name=getattr(level,'name','invalid_level')
            record=dict(index=layer_index,name=name,status='LAYER_EXCEPTION_FAILURE',reason=type(exc).__name__+': '+str(exc),
                iterations=iterations,certified=False,point_role='failed_uncommitted_trial',degenerate_face_repairs=level_repairs)
            records.append(record);all_repairs.extend(level_repairs)
            failed_layer=dict(index=layer_index,name=name,status='LAYER_EXCEPTION_FAILURE',reason=record['reason'],record=record,failed_trial_returned=False)
            break
    retracted=[]
    while True:
        returned=certified[-1]['x'].copy() if certified else anchor.copy()
        return_equal=np.vstack((fixed_equal,*[v['matrix'] for v in certified])) if certified else fixed_equal
        return_rhs=np.r_[fixed_rhs,*[v['attained'] for v in certified]] if certified else fixed_rhs
        return_primal=_original_primal_violation(original_g,original_h,return_equal,return_rhs,returned)
        return_frozen=max([0.]+[float(np.max(abs(v['matrix']@returned-v['attained']),initial=0.)) for v in certified])
        if np.isfinite(returned).all() and return_primal<=tol(returned) and return_frozen<=tol(returned):break
        if not certified:raise FloatingPointError('Validated original anchor became unrepresentable')
        removed=certified.pop();retracted.append(dict(index=removed['index'],name=removed['name']))
        failed_layer=dict(index=removed['index'],name=removed['name'],status='SAVED_POINT_RETURN_RECHECK_FAILURE',
            original_primal=return_primal,frozen_error=return_frozen,failed_trial_returned=False)
    fully=bool(len(levels)>0 and len(certified)==len(levels) and failed_layer is None)
    result_status=('CONVERGED_DECLARED_LINEAR_HIERARCHY' if fully else
        'CERTIFIED_PREFIX_WITH_FAILED_LAYER' if certified else 'FEASIBLE_ANCHOR_NO_TASK_CERTIFICATE')
    returned.setflags(write=False)
    return PrefixHierarchyResult(result_status,returned,tuple(records),dict(
        source_kernel_sha256=SOURCE_KERNEL_SHA256,algorithm='unchanged_v17_active_set_with_prefix_snapshots',
        dimension=n,maximum_resource_dimension=MAX_COORDINATES,resource_dimension_expanded_from64=True,
        hard_domain=hard_domain,original_feasible_anchor=_readonly(anchor),
        initial_original_hard_primal_violation=initial_original_primal,
        hard_constraint_violation=max(0.,float(np.max(g@returned-h,initial=0.)),float(np.max(abs(fixed_equal@returned-fixed_rhs),initial=0.))),
        original_hard_constraint_violation=return_primal,returned_certified_residual_drift_inf=return_frozen,
        levels_requested=len(levels),levels_attempted=len(records),levels_completed=len(certified),
        completed_prefix=len(certified),fully_converged=fully,
        available_requires_nonempty_certified_prefix=True,
        returned_point_role='last_certified_prefix_point' if certified else 'validated_feasible_anchor_without_task_certificate',
        failed_layer=failed_layer,failed_trial_returned=False,retracted_snapshots=retracted,
        prefix_replay_solve_count=0,initial_anchor_is_not_task_certificate=True,
        physical_safety_certified=False,servo_response_predicted=False,
        higher_task_residuals_frozen=True,hidden_regularization=False,all_original_hard_rows_retained=True,
        numerical_atol=atol,numerical_rtol=rtol,max_iterations=max_iterations,
        degenerate_face_method='exact_sparse_or_rational_span_and_roundoff_pivot_repair',
        degenerate_face_proof_budgets=dict(sparse_visits=SPARSE_VISIT_BUDGET,rational_operations=EXACT_OPERATION_BUDGET,integer_bits=EXACT_INTEGER_BIT_BUDGET),
        degenerate_face_events=len(all_repairs),degenerate_face_elapsed_ms=sum(v['elapsed_ms'] for v in all_repairs),
        exact_rational_events=sum(bool(v.get('rational_used')) for v in all_repairs)),
        tuple(certified),len(certified),fully,failed_layer)

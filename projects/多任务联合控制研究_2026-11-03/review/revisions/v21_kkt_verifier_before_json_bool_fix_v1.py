"""Independent finite convex-stage verifier. No production/model/solver import.

min ||A x-b||^2 or min c^T x; all Gx<=h and Ex=f supplied explicitly.
Dual witness uses unit-normalized inequalities (same row count including zero
rows) and ORIGINAL equality rows. No row, regularizer or bound is invented.
This is a numerical KKT check at the original tolerances, not an exact theorem.
"""
from numbers import Real
import hashlib,json,math
import numpy as np


def _vec(value,name):
    if value is None:raise ValueError(name+' unknown')
    raw=np.asarray(value,dtype=object)
    if raw.ndim!=1 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in raw):raise ValueError(name+' requires explicit numeric vector')
    out=np.asarray(value,dtype=float).copy()
    if not np.isfinite(out).all():raise ValueError(name+' nonfinite')
    return out


def _mat(value,m,n,name):
    if value is None:raise ValueError(name+' unknown')
    raw=np.asarray(value,dtype=object)
    # Only explicit zero row count plus known original n restores JSON [] shape.
    if raw.shape==(0,) and m==0:raw=np.empty((0,n),dtype=object)
    if raw.shape!=(m,n) or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in raw.flat):raise ValueError(name+' shape/type mismatch')
    out=np.asarray(raw,dtype=float).copy()
    if not np.isfinite(out).all():raise ValueError(name+' nonfinite')
    return out


def _real(value,name,positive=False):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not math.isfinite(value) or (value<=0 if positive else value<0):raise ValueError(name+' outside numeric domain')
    return float(value)


def _inf(a):return max((abs(float(v)) for v in np.asarray(a).flat),default=0.)


def _rows(A,x):
    out=np.asarray([math.fsum(float(a)*float(b) for a,b in zip(row,x)) for row in A],float)
    if not np.isfinite(out).all():raise FloatingPointError('unrepresentable row evaluation')
    return out


def canonical_problem(problem,x):
    """Return copies and positive-equivalent G normalization, retaining ALL rows.

    G/h must include all original boxes/coupled hard rows. E/f must include
    fixed coordinates/slacks and every prior frozen attained output. Explicit
    previous_frozen duplicates their independent provenance check; it does not
    silently add a missing equality to the solver problem.
    """
    if not isinstance(problem,dict):raise ValueError('explicit problem dict required')
    x=_vec(x,'x');n=len(x);h=_vec(problem['h'],'h');f=_vec(problem['f'],'f')
    G=_mat(problem['G'],len(h),n,'G');E=_mat(problem['E'],len(f),n,'E')
    kind=problem.get('kind','least_squares');A=b=c=None
    if kind=='least_squares':b=_vec(problem['b'],'b');A=_mat(problem['A'],len(b),n,'A')
    elif kind=='linear':c=_vec(problem['c'],'c');assert c.shape==(n,),'c shape mismatch'
    else:raise ValueError('kind must be least_squares or linear')
    atol=_real(problem.get('atol',1e-9),'atol',True);rtol=_real(problem.get('rtol',1e-8),'rtol')
    frozen=[]
    for item in problem.get('previous_frozen',()):
        y=_vec(item['attained'],'frozen attained');oldA=_mat(item['matrix'],len(y),n,'frozen matrix');frozen.append((oldA,y))
    unit=np.empty_like(G);uh=np.empty_like(h);norm_factor=[];constant=[]
    for i,(row,bound) in enumerate(zip(G,h)):
        scale=_inf(row)
        if scale==0:
            unit[i]=row;uh[i]=bound;norm_factor.append({'maxabs':0.,'scaled_norm':0.});constant.append(i)
        else:
            scaled=row/scale;length=math.sqrt(math.fsum(float(v)*float(v) for v in scaled));unit[i]=scaled/length;uh[i]=(bound/scale)/length;norm_factor.append({'maxabs':scale,'scaled_norm':length})
    if not np.isfinite(unit).all() or not np.isfinite(uh).all():raise FloatingPointError('original hard row normalization not representable')
    return dict(x=x,n=n,G=G,h=h,E=E,f=f,A=A,b=b,c=c,kind=kind,atol=atol,rtol=rtol,G_unit=unit,h_unit=uh,normalization=norm_factor,constant_rows=constant,frozen=frozen)


def _objective(p,x):
    if p['kind']=='linear':return math.fsum(float(a)*float(b) for a,b in zip(p['c'],x)),p['c'].copy(),None
    residual=_rows(p['A'],x)-p['b'];value=math.fsum(float(v)*float(v) for v in residual);gradient=2.*_rows(p['A'].T,residual)
    if not math.isfinite(value) or not np.isfinite(gradient).all():raise FloatingPointError('objective/gradient not representable')
    return value,gradient,residual


def _primal(p,x):
    tx=p['atol']+p['rtol']*max(1.,_inf(x));violation=_rows(p['G_unit'],x)-p['h_unit'];equality=_rows(p['E'],x)-p['f']
    drift=[_inf(_rows(a,x)-y) for a,y in p['frozen']]
    constant_ok=all(p['h'][i]>=0 for i in p['constant_rows']) and all(bound==0 or _inf(row)>0 for row,bound in zip(p['E'],p['f']))
    good=bool(constant_ok and max(0.,max(violation,default=0.))<=tx and _inf(equality)<=tx and max(drift,default=0.)<=tx)
    return dict(passed=good,coordinate_tolerance=tx,all_original_rows_checked=len(p['G']),all_original_equalities_checked=len(p['E']),constant_hard_rows_consistent=constant_ok,inequality_positive_violation_inf=max(0.,max(violation,default=0.)),equality_residual_inf=_inf(equality),previous_frozen_drift_inf=max(drift,default=0.),inequality_residual_unit=violation.tolist(),equality_residual_original=equality.tolist())


def verify_stage(problem,x,*,duals=None,comparison_x=None):
    """Check original finite convex stage, irrespective of any solver status.

    duals={'inequality_unit':lambda_m,'equality_raw':nu_k}. Unit row lambda must
    correspond to every input G row in unchanged order; nu to raw E including
    frozen outputs, not a solver's compressed equality basis. Without a dual
    witness only primal/objective diagnostics are available; KKT cannot pass.
    comparison_x is optional prior candidate in the SAME original problem.
    """
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            p=canonical_problem(problem,x);x=p['x'];primal=_primal(p,x);value,gradient,residual=_objective(p,x);tg=p['atol']+p['rtol']*max(1.,_inf(gradient));comp_tol=tg*max(1.,_inf(x))
            r=dict(available=True,kind=p['kind'],dimension=p['n'],objective=value,residual=None if residual is None else residual.tolist(),gradient=gradient.tolist(),primal=primal,dual_witness_present=duals is not None,gradient_tolerance=tg,complementarity_tolerance=comp_tol,kkt_passed=False,problem_row_normalization=p['normalization'],supplied_problem_only=True,no_original_row_dropped=True,no_regularizer_added=True)
            payload={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in p.items() if k not in ('x','frozen','normalization')};payload['frozen']=[{'matrix':a.tolist(),'attained':b.tolist()} for a,b in p['frozen']];r['original_problem_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            if duals is not None:
                lam=_vec(duals['inequality_unit'],'inequality dual');nu=_vec(duals['equality_raw'],'equality dual')
                if lam.shape!=p['h'].shape or nu.shape!=p['f'].shape:raise ValueError('dual witness original row count mismatch')
                stationarity=gradient+_rows(p['G_unit'].T,lam)+_rows(p['E'].T,nu);slack=np.asarray(primal['inequality_residual_unit']);comp=_inf(lam*slack);dual_min=min(lam,default=0.);active=np.flatnonzero(lam!=0);tight=_inf(slack[active])
                r.update(stationarity_inf=_inf(stationarity),inequality_dual_min=float(dual_min),complementarity_inf=comp,dual_nonnegative=dual_min>=-tg,nonzero_dual_faces_tight_inf=tight,dual_active_inequalities=active.tolist(),dual_provenance=duals.get('provenance','provided_current_candidate_witness_not_historical'))
                r['kkt_passed']=bool(primal['passed'] and _inf(stationarity)<=tg and dual_min>=-tg and comp<=comp_tol and tight<=primal['coordinate_tolerance'])
            r['status']='NUMERICAL_KKT_VERIFIED' if r['kkt_passed'] else 'PRIMAL_ONLY_NO_DUAL_WITNESS' if duals is None and primal['passed'] else 'KKT_OR_PRIMAL_NOT_VERIFIED'
            if comparison_x is not None:
                old=_vec(comparison_x,'comparison_x')
                if old.shape!=x.shape:raise ValueError('comparison dimension mismatch')
                old_primal=_primal(p,old);old_value,_,_=_objective(p,old);paired=primal['passed'] and old_primal['passed'];r['comparison']=dict(reference_primal=old_primal,reference_objective=old_value,same_original_problem=True,paired_feasible=paired,objective_delta=value-old_value if paired else None,strict_float_improvement=bool(paired and value<old_value),improvement_is_not_new_metric_or_safety_claim=True)
            return r
    except (ValueError,TypeError,KeyError,AssertionError,FloatingPointError,OverflowError) as exc:
        return dict(available=False,kkt_passed=False,status='INVALID_OR_UNREPRESENTABLE_VERIFICATION_INPUT',reason=type(exc).__name__+': '+str(exc),no_regularizer_added=True,no_original_row_dropped=True)

"""V18 single current native-effort / normal-intent task, pure NumPy.

Variables d21 (native effort delta from the FIXED original p0) and zP
(nonnegative normal-force INTENT only) minimize the one objective
  ||Sr*(A.T@d - (Jn.T@(z-y) + tau_extra25))||^2
  + ||Sf*(z-z_intent)||^2 + ||Sd*d||^2
  + (sL*max(0, load - support_coeff@z))^2 + ||E[d,z]-b||^2.
The parentheses above are the CANONICAL floating-point residual: first form
correction=Jn.T@(z-y)+tau_extra25, then A.T@d-correction, matching the frozen
native realizer's final residual. Sequential subtraction is mathematically
identical but is not substituted when byte consistency is claimed.

Scales multiply residuals BEFORE squaring and are strictly required positive
controller-design scales. tau_extra25 is required finite (24Nm plus liftN):
signed device-projected tangent and braking efforts are caller-owned constant
intent, never nonnegative normal columns, predicted pressure, or measured
external-state truth. None/NaN tau, J or load stays unknown, never zero.
Explicit c=0, load=0 cancels a task; it is not an observation of zero load.

The absolute effort box and nonnegative digit caps/total budget are unchanged.
Known singleton/plateau coordinates are valid fixed variables. The caller owns
native/mode/rate/clock/owner admission; this kernel does not invert a plateau,
map a command, certify contact gradients, or introduce a hard coupled domain.
Extra rows are SOFT only, M<=256, P<=128. No new tangent friction hard domain.

Numerical source: frozen v15_support/task_qp.py SHA
6d48d44ad544af323a11a33a29d73ea5757c4cb35bd183a989b6a81433c817e4
(V14 active-set1f with general soft rows). Only the mechanical rhs/objective
are extended. Original dimensionless feasible active-set LS, raw-unit direct
box/laminar KKT+complementarity and budget400/atol1e-8/rtol1e-7 are retained.
No projected-step/global-L difference certifies convergence. Failure/unknown
is explicit; no global floating-point optimum or physical safety is asserted.

Public helpers evaluate the SAME objective/gradient/KKT, accept only same-goal
feasible merit candidates, supply the exact fixed native correction, and score
the final native result with the same z and all soft terms. The frozen native
result.objective covers only mechanical+delta terms: it is not the total task
objective. This module consumes its upstream full-F067 verification; it never
runs a native model, solver outside this QP, physics, sensor or writer.
"""
from collections.abc import Mapping
from numbers import Integral
import numpy as np
from v13_joint.intent_qp import (
    IntentQPResult, MAX_PATCHES, _InputError, _array, _scalar, _readonly,
    _domain, _project_z, _box_normal_cone, _normal_intent_cone,
    project_normal_intents,
)

ALGORITHM='dimensionless_feasible_active_set_constrained_least_squares'
MAX_EXTRA_TASK_ROWS=256

def _nullspace(rows,n):
    if not len(rows):return np.eye(n),0
    _,s,vt=np.linalg.svd(np.asarray(rows,float),full_matrices=True)
    threshold=max(np.asarray(rows).shape)*np.finfo(float).eps*(s[0] if len(s) else 0.)
    rank=int(np.sum(s>threshold))
    return vt[rank:].T,rank

def solve_effort_task(*,p0_21,A21x25,Jpx25,y_p,z_intent_p,
                    effort_lower21,effort_upper21,patch_digit_p,caps5,
                    total_budget,support_coeff_p,load_proxy_N,scales,tau_extra25,
                    max_iterations=400,atol=1e-8,rtol=1e-7,warm_start=None,extra_task_rows=None):
    scope=dict(algorithm=ALGORITHM,computational_budget=dict(max_iterations=max_iterations,atol=atol,rtol=rtol,max_patches=MAX_PATCHES),
        convergence_revision='unchanged_original_unit_direct_normal_cone_v2',native_mapping_verified_by_this_kernel=False,
        future_pressure_prediction_available=False,physical_safety_certified=False,global_floating_optimality_certified=False,
        extra_balance_or_coupled_synergy_rows_implemented=False,objective_retargeted=False,
        helper_dependency='frozen v13_joint.intent_qp validation/projection/directKKT only; not its solver',
        optional_soft_task_rows_implemented=True,max_extra_task_rows=MAX_EXTRA_TASK_ROWS,extra_hard_constraints_implemented=False,
        signed_extra_is_constant_generalized_effort_not_normal_patch=True,nonzero_mechanical_rhs_implemented=True)
    try:
        p0=_array(p0_21,(21,),'p0_21');A=_array(A21x25,(21,25),'A21x25')
        tau=_array(tau_extra25,(25,),'tau_extra25')
        if Jpx25 is None:raise _InputError('Jpx25_unknown',unknown=True)
        raw=np.asarray(Jpx25,dtype=object)
        if raw.ndim!=2 or raw.shape[1]!=25:raise _InputError('Jpx25_requires_P_by25')
        P=len(raw)
        if P>MAX_PATCHES:raise _InputError('patch_count_exceeds_predeclared_computational_domain',unknown=True)
        J=_array(Jpx25,(P,25),'Jpx25');y=_array(y_p,(P,),'y_p');target=_array(z_intent_p,(P,),'z_intent_p')
        if np.any(y<0) or np.any(target<0):raise _InputError('normal_observations_and_intents_must_be_nonnegative')
        lo=_array(effort_lower21,(21,),'effort_lower21')-p0;hi=_array(effort_upper21,(21,),'effort_upper21')-p0
        if np.any(lo>hi):raise _InputError('empty_effort_box')
        groups,caps,budget=_domain(P,patch_digit_p,caps5,total_budget)
        c=_array(support_coeff_p,(P,),'support_coeff_p');load=_scalar(load_proxy_N,'load_proxy_N',minimum=0.)
        if not isinstance(scales,Mapping) or set(scales)!={'residual25','intentP','delta21','load'}:raise _InputError('four_explicit_scales_required')
        sr=_array(scales['residual25'],(25,),'scales.residual25');sf=_array(scales['intentP'],(P,),'scales.intentP');sd=_array(scales['delta21'],(21,),'scales.delta21');sl=_scalar(scales['load'],'scales.load',strictly_positive=True)
        if np.any(sr<=0) or np.any(sf<=0) or np.any(sd<=0):raise _InputError('declared_scales_must_be_positive')
        if isinstance(max_iterations,(bool,np.bool_)) or not isinstance(max_iterations,Integral) or max_iterations<1:raise _InputError('max_iterations_requires_positive_integer')
        atol=_scalar(atol,'atol',strictly_positive=True);rtol=_scalar(rtol,'rtol',minimum=0.)
        E=np.empty((0,21+P));task_target=np.empty(0)
        if extra_task_rows is not None:
            if not isinstance(extra_task_rows,Mapping) or set(extra_task_rows)!={'matrix','target'}:raise _InputError('extra_task_rows_requires_matrix_and_target_only')
            if extra_task_rows['matrix'] is None:raise _InputError('extra_task_rows.matrix_unknown',unknown=True)
            eraw=np.asarray(extra_task_rows['matrix'],dtype=object)
            # [] explicitly denotes no rows; no unknown coefficient is filled.
            if eraw.shape==(0,):eraw=np.empty((0,21+P),dtype=object)
            if eraw.ndim!=2 or eraw.shape[1]!=21+P:raise _InputError('extra_task_rows.matrix_requires_M_by_21plusP')
            count=eraw.shape[0]
            if count>MAX_EXTRA_TASK_ROWS:raise _InputError('extra_task_row_count_exceeds_predeclared_domain',unknown=True)
            E=_array(eraw,(count,21+P),'extra_task_rows.matrix')
            task_target=_array(extra_task_rows['target'],(count,),'extra_task_rows.target')
        if warm_start is not None:
            if not isinstance(warm_start,Mapping) or set(warm_start)!={'d21','zP'}:raise _InputError('warm_start_requires_d21_and_zP')
            initial=np.r_[_array(warm_start['d21'],(21,),'warm_start.d21'),_array(warm_start['zP'],(P,),'warm_start.zP')]
        else:initial=np.r_[np.zeros(21),target]
    except _InputError as exc:
        return IntentQPResult('unknown' if exc.unknown else 'invalid_input',str(exc),diagnostics=scope)
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            n=22+P;s=np.r_[sd,sf,sl]
            M=np.column_stack((A.T,-J.T,np.zeros(25)))
            R=np.vstack((sr[:,None]*M/s[None,:],np.eye(n)))
            rhs=np.r_[-sr*(J.T@y-tau),np.zeros(21),sf*target,0.]
            if len(E):
                R=np.vstack((R,np.c_[E/s[None,:-1],np.zeros(len(E))]))
                rhs=np.r_[rhs,task_target]
            original_T=sr[:,None]*M[:,:-1]
            # For comparison with the frozen V13 contract only. This bound
            # never controls the active-set step or its acceptance criterion.
            lipschitz=float(2.*(max(0.,np.linalg.eigvalsh(original_T@original_T.T)[-1])+np.max(np.r_[sd*sd,sf*sf])+sl*sl*float(c@c)))
            if len(E):lipschitz+=float(2.*np.sum(E*E))
            fixed={i:float(sd[i]*lo[i]) for i in range(21) if lo[i]==hi[i]}
            for ids,cap in zip(groups,caps):
                if cap==0 or budget==0:
                    for i in ids:fixed[21+int(i)]=0.
            equal=np.eye(n)[list(fixed)] if fixed else np.empty((0,n))
            G=[];h=[];labels=[]
            def add(row,bound,label):
                norm=float(np.linalg.norm(row))
                if not np.isfinite(norm) or norm==0:raise FloatingPointError('invalid_constraint_scaling')
                G.append(row/norm);h.append(float(bound)/norm);labels.append(label)
            for i in range(21):
                if i in fixed:continue
                row=np.eye(n)[i];add(-row,-sd[i]*lo[i],('d_lower',i));add(row,sd[i]*hi[i],('d_upper',i))
            for j in range(P):
                if 21+j not in fixed:add(-np.eye(n)[21+j],0.,('z_lower',j))
            for digit,(ids,cap) in enumerate(zip(groups,caps)):
                if len(ids) and cap>0 and budget>0:
                    row=np.zeros(n);row[21+ids]=1./sf[ids];add(row,cap,('group',digit))
            if P and budget>0:
                row=np.zeros(n);row[21:21+P]=1./sf;add(row,budget,('budget',0))
            add(-np.eye(n)[-1],0.,('t_lower',0))
            row=np.r_[np.zeros(21),-c/sf,-1./sl];add(row,-load,('hinge',0))
            G=np.asarray(G);h=np.asarray(h)

            def evaluate(x):
                return _objective_terms(A,J,y,target,sr,sf,sd,sl,c,load,E,task_target,tau,x[:21],x[21:])

            def diagnostics(x,g):
                d,z=x[:21],x[21:];box=_box_normal_cone(g[:21],d,lo,hi);normal=_normal_intent_cone(g[21:],z,groups,caps,budget)
                td=atol+rtol*max(1.,float(np.max(abs(d))));tz=atol+rtol*max(1.,float(np.max(abs(z)))) if P else atol+rtol
                violation=dict(effort_box=float(max(0.,np.max(lo-d),np.max(d-hi))),negative_normal=float(max(0.,-np.min(z))) if P else 0.,digit_caps=float(max([0.]+[float(z[ids].sum()-cap) for ids,cap in zip(groups,caps)])),total_budget=float(max(0.,z.sum()-budget)))
                done=box['stationarity_inf']<=td and normal['stationarity_inf']<=tz and normal['complementarity_inf']<=atol+rtol and normal['dual_nonnegativity_violation']<=atol and max(violation.values())<=atol
                return done,box,normal,td,tz,violation

            active=[];additions=0;removals=0;dependent_blockers=0;svd_solves=0;history=[]
            def canonical(u):
                x=u[:-1]/s[:-1]
                for i in fixed:
                    if i<21:x[i]=lo[i]
                    else:x[i]=0.
                for k in active:
                    name,i=labels[k]
                    if name=='d_lower':x[i]=lo[i]
                    elif name=='d_upper':x[i]=hi[i]
                    elif name=='z_lower':x[21+i]=0.
                x[:21]=np.minimum(hi,np.maximum(lo,x[:21]));x[21:]=_project_z(x[21:],groups,caps,budget)
                t=max(0.,float(u[-1]/sl),load-float(c@x[21:]))
                v=np.r_[s[:-1]*x,sl*t]
                for i,value in fixed.items():v[i]=value
                return v,x
            x=np.r_[np.clip(initial[:21],lo,hi),_project_z(initial[21:],groups,caps,budget)]
            u,x=canonical(np.r_[s[:-1]*x,sl*max(0.,load-float(c@x[21:]))])
            # Add only independent exactly/locally-roundoff active rows. Fixed
            # coordinates are equalities with unrestricted multipliers.
            def add_independent(index):
                nonlocal dependent_blockers
                W=np.vstack((equal,G[active])) if active else equal
                Z,rank=_nullspace(W,n)
                if np.linalg.norm(G[index]@Z)<=64*np.finfo(float).eps*max(1,n):
                    dependent_blockers+=1;return False
                active.append(index);return True
            slack=h-G@u
            for k in range(len(h)):
                roundoff=8*np.finfo(float).eps*max(abs(h[k]),float(abs(G[k])@abs(u)))
                if abs(slack[k])<=roundoff:add_independent(k)
            status='nonconvergence';reason='predeclared_iteration_budget_exhausted';iterations=0
            value,g,residual,short=evaluate(x);done,box,normal,td,tz,violations=diagnostics(x,g)
            if done:status='converged';reason='direct_normal_cone_KKT_and_complementarity_met'
            for iterations in range(1,int(max_iterations)+1):
                if done:iterations-=1;break
                W=np.vstack((equal,G[active])) if active else equal
                Z,rank=_nullspace(W,n)
                if Z.shape[1]:
                    theta=np.linalg.lstsq(R@Z,rhs-R@u,rcond=None)[0];p=Z@theta;svd_solves+=1
                else:p=np.zeros(n)
                step_limit=1.;blocking=None;gp=G@p;slack=h-G@u
                for k in range(len(h)):
                    if k in active:continue
                    if gp[k]>0:
                        limit=max(0.,float(slack[k]))/float(gp[k])
                        if limit<step_limit:step_limit=limit;blocking=k
                candidate_u,candidate=canonical(u+step_limit*p)
                next_value,next_g,next_residual,next_short=evaluate(candidate)
                next_done,next_box,next_normal,next_td,next_tz,next_violations=diagnostics(candidate,next_g)
                same=np.array_equal(candidate_u,u)
                history.append(dict(iteration=iterations,objective=next_value,step_fraction=step_limit,working_inequalities=len(active),rank=rank,step_inf=float(np.max(abs(step_limit*p))),direct_d=next_box['stationarity_inf'],direct_z=next_normal['stationarity_inf']))
                u,x=candidate_u,candidate;value,g,residual,short=next_value,next_g,next_residual,next_short
                done,box,normal,td,tz,violations=next_done,next_box,next_normal,next_td,next_tz,next_violations
                if done:status='converged';reason='direct_normal_cone_KKT_and_complementarity_met';break
                if blocking is not None:
                    if add_independent(blocking):additions+=1;continue
                # At an equality minimizer, remove a genuinely negative
                # inequality multiplier; never certify from a small step.
                at_equality_minimum=same or float(np.linalg.norm(p))<=64*np.finfo(float).eps*max(1.,float(np.linalg.norm(u))) or step_limit==1.
                if at_equality_minimum:
                    grad_u=2.*R.T@(R@u-rhs)
                    multiplier=np.linalg.lstsq(W.T,-grad_u,rcond=None)[0] if len(W) else np.empty(0)
                    inequality_multiplier=multiplier[len(equal):]
                    if len(inequality_multiplier) and np.min(inequality_multiplier)<0.:
                        remove=int(np.argmin(inequality_multiplier));active.pop(remove);removals+=1;continue
                    if same:
                        status='numerical_stagnation';reason='no_representable_active_set_step_with_failed_original_direct_KKT';break
            else:iterations=int(max_iterations)
            projected=np.r_[np.clip(x[:21]-g[:21]/lipschitz,lo,hi),_project_z(x[21:]-g[21:]/lipschitz,groups,caps,budget)]
            mapping=lipschitz*(x-projected)
            diag=dict(**scope,fixed_original_p0_21=p0.tolist(),original_z_intent_p=target.tolist(),
                fixed_tau_extra25=tau.tolist(),canonical_residual='A.T@d-(Jn.T@(z-y)+tau_extra25)',
                lipschitz_upper_bound=lipschitz,global_L_used_for_steps=False,
                projected_gradient_inf_d21=float(np.max(abs(mapping[:21]))),projected_gradient_inf_zP=float(np.max(abs(mapping[21:]))) if P else 0.,projected_gradient_is_diagnostic_only=True,
                direct_KKT_inf_d21=box['stationarity_inf'],direct_KKT_inf_zP=normal['stationarity_inf'],direct_box_KKT=box,direct_normal_KKT=normal,
                complementarity_tolerance=atol+rtol,numerical_tolerance_d21=td,numerical_tolerance_zP=tz,primal_violations=violations,
                candidate_requires_status_check=not done,residual_scope='current effort-intent residual in native generalized units; not future pressure error',generalized_units25=['Nm']*24+['N'],
                coordinate_scale=s.tolist(),least_squares_rows=R.shape[0],auxiliary_shortfall=float(u[-1]/sl),original_hinge_shortfall=short,
                active_set_additions=additions,active_set_removals=removals,dependent_blockers=dependent_blockers,least_squares_solves=svd_solves,
                active_set_history=history,expanded_least_squares_objective=float(np.square(R@u-rhs).sum()),
                extra_task_rows_count=len(E),extra_task_residual=(E@x-task_target).tolist(),extra_task_objective=float(np.square(E@x-task_target).sum()),
                numerical_linear_algebra='SVD equality nullspace and unconstrained least squares; no normal equations/global-L step',warm_start_used=warm_start is not None)
            return IntentQPResult(status,reason,_readonly(x[:21]),_readonly(x[21:]),_readonly(p0+x[:21]),_readonly(residual),float(short),float(value),iterations,diag)
    except (FloatingPointError,OverflowError,np.linalg.LinAlgError,ValueError) as exc:
        return IntentQPResult('numerical_failure',str(exc),diagnostics=scope)


def _objective_terms(A,J,y,target,sr,sf,sd,sl,c,load,E,b,tau,d,z):
    """The sole arithmetic authority for raw total objective and gradient."""
    correction=J.T@(z-y)+tau
    residual=A.T@d-correction
    short=max(0.,load-float(c@z))
    wr=sr*residual;wf=sf*(z-target);wd=sd*d
    value=float(wr@wr+wf@wf+wd@wd+(sl*short)**2)
    w=sr*sr*residual
    gradient=np.r_[2.*(A@w+sd*sd*d),
                   -2.*J@w+2.*sf*sf*(z-target)-2.*sl*sl*short*c]
    if len(E):
        extra=E@np.r_[d,z]-b
        value+=float(extra@extra)
        gradient+=2.*E.T@extra
    if not np.isfinite(value) or not np.isfinite(gradient).all() or not np.isfinite(residual).all():
        raise FloatingPointError('nonfinite_original_objective_or_gradient')
    return value,gradient,residual,short


def _objective_inputs(problem):
    """Validate copies for non-solving evaluation; never project a raw goal."""
    if not isinstance(problem,Mapping):raise _InputError('problem_requires_mapping')
    p0=_array(problem.get('p0_21'),(21,),'p0_21')
    A=_array(problem.get('A21x25'),(21,25),'A21x25')
    tau=_array(problem.get('tau_extra25'),(25,),'tau_extra25')
    raw=np.asarray(problem.get('Jpx25'),dtype=object)
    if problem.get('Jpx25') is None:raise _InputError('Jpx25_unknown',unknown=True)
    if raw.ndim!=2 or raw.shape[1]!=25:raise _InputError('Jpx25_requires_P_by25')
    P=len(raw)
    if P>MAX_PATCHES:raise _InputError('patch_count_exceeds_predeclared_computational_domain',unknown=True)
    J=_array(raw,(P,25),'Jpx25');y=_array(problem.get('y_p'),(P,),'y_p')
    target=_array(problem.get('z_intent_p'),(P,),'z_intent_p')
    if np.any(y<0) or np.any(target<0):raise _InputError('normal_observations_and_intents_must_be_nonnegative')
    lo=_array(problem.get('effort_lower21'),(21,),'effort_lower21')-p0
    hi=_array(problem.get('effort_upper21'),(21,),'effort_upper21')-p0
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():raise FloatingPointError('nonfinite_effort_delta_box')
    if np.any(lo>hi):raise _InputError('empty_effort_box')
    groups,caps,budget=_domain(P,problem.get('patch_digit_p'),problem.get('caps5'),problem.get('total_budget'))
    c=_array(problem.get('support_coeff_p'),(P,),'support_coeff_p')
    load=_scalar(problem.get('load_proxy_N'),'load_proxy_N',minimum=0.)
    scales=problem.get('scales')
    if not isinstance(scales,Mapping) or set(scales)!={'residual25','intentP','delta21','load'}:
        raise _InputError('four_explicit_scales_required')
    sr=_array(scales['residual25'],(25,),'scales.residual25')
    sf=_array(scales['intentP'],(P,),'scales.intentP')
    sd=_array(scales['delta21'],(21,),'scales.delta21')
    sl=_scalar(scales['load'],'scales.load',strictly_positive=True)
    if np.any(sr<=0) or np.any(sf<=0) or np.any(sd<=0):raise _InputError('declared_scales_must_be_positive')
    E=np.empty((0,21+P));b=np.empty(0)
    extras=problem.get('extra_task_rows')
    if extras is not None:
        if not isinstance(extras,Mapping) or set(extras)!={'matrix','target'}:
            raise _InputError('extra_task_rows_requires_matrix_and_target_only')
        if extras['matrix'] is None:raise _InputError('extra_task_rows.matrix_unknown',unknown=True)
        eraw=np.asarray(extras['matrix'],dtype=object)
        if eraw.shape==(0,):eraw=np.empty((0,21+P),dtype=object)
        if eraw.ndim!=2 or eraw.shape[1]!=21+P:raise _InputError('extra_task_rows.matrix_requires_M_by_21plusP')
        count=len(eraw)
        if count>MAX_EXTRA_TASK_ROWS:raise _InputError('extra_task_row_count_exceeds_predeclared_domain',unknown=True)
        E=_array(eraw,(count,21+P),'extra_task_rows.matrix');b=_array(extras['target'],(count,),'extra_task_rows.target')
    atol=_scalar(problem.get('atol',1e-8),'atol',strictly_positive=True)
    rtol=_scalar(problem.get('rtol',1e-7),'rtol',minimum=0.)
    return dict(p0=p0,A=A,J=J,y=y,target=target,tau=tau,lo=lo,hi=hi,
                groups=groups,caps=caps,budget=budget,c=c,load=load,
                sr=sr,sf=sf,sd=sd,sl=sl,E=E,b=b,atol=atol,rtol=rtol,P=P)


def evaluate_effort_task(problem,d21,zP):
    """Non-solving raw objective/gradient/directKKT evaluation, no projection.

    ``available`` means this arithmetic is known, NOT that a command/optimum
    is admitted. Inspect feasible/direct_KKT_met separately. Unknowns return
    explicit status and no objective/residual, never a zero replacement.
    """
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            p=_objective_inputs(problem);d=_array(d21,(21,),'d21');z=_array(zP,(p['P'],),'zP')
            value,g,residual,short=_objective_terms(p['A'],p['J'],p['y'],p['target'],p['sr'],p['sf'],p['sd'],p['sl'],p['c'],p['load'],p['E'],p['b'],p['tau'],d,z)
            box=_box_normal_cone(g[:21],d,p['lo'],p['hi'])
            normal=_normal_intent_cone(g[21:],z,p['groups'],p['caps'],p['budget'])
            td=p['atol']+p['rtol']*max(1.,float(np.max(abs(d))))
            tz=p['atol']+p['rtol']*max(1.,float(np.max(abs(z)))) if p['P'] else p['atol']+p['rtol']
            violations=dict(effort_box=float(max(0.,np.max(p['lo']-d),np.max(d-p['hi']))),
                negative_normal=float(max(0.,-np.min(z))) if p['P'] else 0.,
                digit_caps=float(max([0.]+[float(z[ids].sum()-cap) for ids,cap in zip(p['groups'],p['caps'])])),
                total_budget=float(max(0.,z.sum()-p['budget'])))
            feasible=max(violations.values())<=p['atol']
            kkt=bool(feasible and box['stationarity_inf']<=td and normal['stationarity_inf']<=tz
                and normal['complementarity_inf']<=p['atol']+p['rtol']
                and normal['dual_nonnegativity_violation']<=p['atol'])
            correction=p['J'].T@(z-p['y'])+p['tau']
            wr=p['sr']*residual;wd=p['sd']*d
            mechanical_delta=float(wr@wr+wd@wd)
            return dict(status='known',reason='same_current_effort_task_arithmetic',available=True,
                objective=value,residual25=_readonly(residual),correction25=_readonly(correction),
                gradient_d21=_readonly(g[:21]),gradient_zP=_readonly(g[21:]),load_shortfall_N=short,
                mechanical_delta_objective=mechanical_delta,
                soft_normal_load_and_extra_objective=value-mechanical_delta,
                feasible=bool(feasible),direct_KKT_met=kkt,primal_violations=violations,
                direct_box_KKT=box,direct_normal_KKT=normal,
                numerical_tolerance_d21=td,numerical_tolerance_zP=tz,
                physical_safety_certified=False,future_pressure_prediction_available=False)
    except _InputError as exc:
        return dict(status='unknown' if exc.unknown else 'invalid_input',reason=str(exc),available=False)
    except (FloatingPointError,OverflowError,np.linalg.LinAlgError,ValueError) as exc:
        return dict(status='numerical_failure',reason=str(exc),available=False)


def native_correction25(problem,zP):
    """Exact constant correction to pass to the existing native realizer.

    Raises for unknown inputs instead of returning an unknown-as-zero array.
    The caller must also pass original p0 and accepted original d unchanged.
    """
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        p=_objective_inputs(problem);z=_array(zP,(p['P'],),'zP')
        value=p['J'].T@(z-p['y'])+p['tau']
        if not np.isfinite(value).all():raise FloatingPointError('nonfinite_native_correction')
        return _readonly(value)


def check_same_problem_candidate(problem,result):
    """Original feasible anchor vs same-objective candidate; no fake KKT.

    A nonconverged but finite feasible merit candidate can be explicitly
    provisional. A claimed converged result must ALSO pass fresh direct KKT.
    Value and residual bytes must match this evaluator, not a shifted goal.
    """
    record=dict(accepted=False,optimality_certified=False,
                different_force_feedback_problem_used=False,objective_retargeted=False)
    if result.d21 is None or result.zP is None or result.objective is None:return record
    evaluated=evaluate_effort_task(problem,result.d21,result.zP)
    if not evaluated['available']:return dict(record,evaluation_status=evaluated['status'],reason=evaluated['reason'])
    try:
        p=_objective_inputs(problem)
        d0=np.minimum(p['hi'],np.maximum(p['lo'],np.zeros(21)))
        z0=_project_z(p['target'],p['groups'],p['caps'],p['budget'])
        anchor=evaluate_effort_task(problem,d0,z0)
        if not anchor['available']:return dict(record,reason='anchor_objective_unknown')
        actual=evaluated['objective'];base=anchor['objective']
        same_value=np.float64(actual).tobytes()==np.float64(result.objective).tobytes()
        same_residual=(result.residual25 is not None and
                       np.asarray(result.residual25).dtype==np.dtype(float) and
                       np.asarray(result.residual25).shape==(25,) and
                       np.asarray(result.residual25).tobytes()==evaluated['residual25'].tobytes())
        allowance=64*np.finfo(float).eps*max(1.,abs(base),abs(actual))
        claims_optimum=bool(result.available)
        honest_status=(not claims_optimum) or evaluated['direct_KKT_met']
        accepted=bool(same_value and same_residual and evaluated['feasible'] and
                      actual<=base+allowance and honest_status)
        record.update(accepted=accepted,optimality_certified=bool(accepted and claims_optimum and evaluated['direct_KKT_met']),
            classification='certified' if claims_optimum and evaluated['direct_KKT_met'] else 'feasible_nonoptimal_same_problem',
            original_objective=actual,anchor_objective=base,objective_bytes_equal=same_value,
            residual_bytes_equal=same_residual,objective_roundoff_allowance=allowance,
            primal_violations=evaluated['primal_violations'],fresh_direct_KKT_met=evaluated['direct_KKT_met'])
        return record
    except (_InputError,ValueError,TypeError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:
        return dict(record,reason=str(exc))


def score_native_realization(problem,*,original_d21,zP,native_result):
    """Total same-goal score at actual native delta, without another optimizer.

    Requires the original realizer's full-check marker and checks its reported
    p0/delta/native/canonical-residual/sub-objective bytes. This is arithmetic
    consistency, not an independent F067 reevaluation or native admission.
    Retains normal, hinge AND extra soft rows at the same z, replacing only d
    by reported native_effort-original_p0. No realized effort error is hidden.
    """
    def fail(reason):return dict(available=False,status='invalid_native_result',reason=reason,native_recomputed_here=False)
    try:
        p=_objective_inputs(problem);wanted=_array(original_d21,(21,),'original_d21')
        if not native_result.available:return fail('native_result_unavailable')
        if not native_result.diagnostics.get('prepared_value_final_full_recheck',{}).get('passed'):
            return fail('upstream_full_native_recheck_missing_or_failed')
        def exact(raw,expected):
            a=np.asarray(raw)
            return a.dtype==np.dtype(float) and a.shape==expected.shape and a.tobytes()==expected.tobytes()
        if not exact(native_result.p0_effort21,p['p0']):return fail('native_p0_not_original')
        if not exact(native_result.requested_delta_effort21,wanted):return fail('native_requested_delta_not_original')
        native=_array(native_result.native_effort21,(21,),'native_effort21')
        actual=native-p['p0']
        if not exact(native_result.actual_delta_effort21,actual):return fail('actual_delta_not_native_minus_original_p0')
        evaluated=evaluate_effort_task(problem,actual,zP)
        if not evaluated['available']:return fail('final_total_objective_unknown:'+evaluated['reason'])
        if not exact(native_result.residual25,evaluated['residual25']):return fail('native_canonical_residual_differs')
        if np.float64(native_result.objective).tobytes()!=np.float64(evaluated['mechanical_delta_objective']).tobytes():
            return fail('native_mechanical_delta_score_differs')
        return dict(available=True,status='known_final_same_task_score',objective=evaluated['objective'],
            residual25=evaluated['residual25'],correction25=evaluated['correction25'],
            actual_delta_effort21=_readonly(actual),requested_delta_effort21=_readonly(wanted),
            realization_error21=_readonly(actual-wanted),
            mechanical_delta_objective=evaluated['mechanical_delta_objective'],
            soft_normal_load_and_extra_objective=evaluated['soft_normal_load_and_extra_objective'],
            feasible=evaluated['feasible'],primal_violations=evaluated['primal_violations'],
            direct_KKT_met_at_actual_delta=evaluated['direct_KKT_met'],
            native_recomputed_here=False,upstream_full_native_marker_consumed=True,
            second_force_optimizer_used=False,physical_safety_certified=False,
            future_pressure_prediction_available=False)
    except (_InputError,ValueError,TypeError,AttributeError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:
        return fail(str(exc))

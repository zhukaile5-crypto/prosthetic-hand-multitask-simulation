"""Equivalent dimensionless active-set solution of the fixed V13 intent QP.

The original objective, fixed p0, constraints and ORIGINAL-UNIT direct KKT
acceptance are unchanged. An auxiliary t realizes the squared load hinge:
 t >= 0, t >= load-c@z, cost (sl*t)^2.
Coordinates u=(Sd*d,Sf*z,sl*t) are dimensionless. The resulting constrained
least-squares objective has identity regularization. Equality-constrained
steps use an SVD nullspace and least-squares, never normal-equation inversion
or the single worst-axis global Lipschitz step used by V13.

Frozen V13 validation/projection/normal-cone helpers are explicit read-only
dependencies. No native mapping, robot/contact dynamics, force forecast or
physical safety certificate is provided. Failure retains a feasible candidate
and the unmodified original objective/KKT. max_iterations remains400.
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

def _nullspace(rows,n):
    if not len(rows):return np.eye(n),0
    _,s,vt=np.linalg.svd(np.asarray(rows,float),full_matrices=True)
    threshold=max(np.asarray(rows).shape)*np.finfo(float).eps*(s[0] if len(s) else 0.)
    rank=int(np.sum(s>threshold))
    return vt[rank:].T,rank

def solve_intent_qp(*,p0_21,A21x25,Jpx25,y_p,z_intent_p,
                    effort_lower21,effort_upper21,patch_digit_p,caps5,
                    total_budget,support_coeff_p,load_proxy_N,scales,
                    max_iterations=400,atol=1e-8,rtol=1e-7,warm_start=None):
    scope=dict(algorithm=ALGORITHM,computational_budget=dict(max_iterations=max_iterations,atol=atol,rtol=rtol,max_patches=MAX_PATCHES),
        convergence_revision='unchanged_original_unit_direct_normal_cone_v2',native_mapping_verified_by_this_kernel=False,
        future_pressure_prediction_available=False,physical_safety_certified=False,global_floating_optimality_certified=False,
        extra_balance_or_coupled_synergy_rows_implemented=False,objective_retargeted=False,
        helper_dependency='frozen v13_joint.intent_qp validation/projection/directKKT only; not its solver')
    try:
        p0=_array(p0_21,(21,),'p0_21');A=_array(A21x25,(21,25),'A21x25')
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
            rhs=np.r_[-sr*(J.T@y),np.zeros(21),sf*target,0.]
            original_T=sr[:,None]*M[:,:-1]
            # For comparison with the frozen V13 contract only. This bound
            # never controls the active-set step or its acceptance criterion.
            lipschitz=float(2.*(max(0.,np.linalg.eigvalsh(original_T@original_T.T)[-1])+np.max(np.r_[sd*sd,sf*sf])+sl*sl*float(c@c)))
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
                d,z=x[:21],x[21:];residual=A.T@d-J.T@(z-y);short=max(0.,load-float(c@z))
                value=float((sr*residual)@(sr*residual)+(sf*(z-target))@(sf*(z-target))+(sd*d)@(sd*d)+(sl*short)**2)
                w=sr*sr*residual
                gradient=np.r_[2.*(A@w+sd*sd*d),-2.*J@w+2.*sf*sf*(z-target)-2.*sl*sl*short*c]
                if not np.isfinite(value) or not np.isfinite(gradient).all():raise FloatingPointError('nonfinite_original_objective_or_gradient')
                return value,gradient,residual,short

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
            diag=dict(**scope,fixed_original_p0_21=p0.tolist(),original_z_intent_p=target.tolist(),
                lipschitz_upper_bound=lipschitz,global_L_used_for_steps=False,
                direct_KKT_inf_d21=box['stationarity_inf'],direct_KKT_inf_zP=normal['stationarity_inf'],direct_box_KKT=box,direct_normal_KKT=normal,
                complementarity_tolerance=atol+rtol,numerical_tolerance_d21=td,numerical_tolerance_zP=tz,primal_violations=violations,
                candidate_requires_status_check=not done,residual_scope='current effort-intent residual in native generalized units; not future pressure error',generalized_units25=['Nm']*24+['N'],
                coordinate_scale=s.tolist(),least_squares_rows=R.shape[0],auxiliary_shortfall=float(u[-1]/sl),original_hinge_shortfall=short,
                active_set_additions=additions,active_set_removals=removals,dependent_blockers=dependent_blockers,least_squares_solves=svd_solves,
                active_set_history=history,expanded_least_squares_objective=float(np.square(R@u-rhs).sum()),
                numerical_linear_algebra='SVD equality nullspace and unconstrained least squares; no normal equations/global-L step',warm_start_used=warm_start is not None)
            return IntentQPResult(status,reason,_readonly(x[:21]),_readonly(x[21:]),_readonly(p0+x[:21]),_readonly(residual),float(short),float(value),iterations,diag)
    except (FloatingPointError,OverflowError,np.linalg.LinAlgError,ValueError) as exc:
        return IntentQPResult('numerical_failure',str(exc),diagnostics=scope)

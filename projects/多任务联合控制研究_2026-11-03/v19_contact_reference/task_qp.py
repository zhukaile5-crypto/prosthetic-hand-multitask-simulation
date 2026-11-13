"""V19 ONE current-branch reference/normal/progress constrained LS task.

Variables are reference INCREMENT du21 (18 angular/tendon references, lift m,
2 wrist radians), normal INTENT zP>=0 (N), and gamma in caller box[0,1].
They are NOT native effort variables. Native effort delta is d=offset+gain*du.
All inputs, offsets, bounds, OPEN/motion steps and scales remain unprojected
goals. The caller certifies the current native affine branch, q/v, ownership,
mode/rate/clock and actual reference domain; no inverse, physics or writer is
performed here. Known gain=0 leaves du as a genuine geometric variable.

The SINGLE raw objective is exactly
 ||Sr*(A.T@d-(J.T@(z-y)+tau_extra))||² + ||Sd*d||²
 +||Sref*(du-gamma*open_step-motion_step)||²
 +||Sf*(z+gamma*unload_fraction*y-y)||² + (Sgamma*(gamma-1))².
Canonical mechanical parentheses match the native final score. z remains
nonnegative normal only; signed motion efforts live in tau_extra25 (24Nm,N).
No load observation is fabricated; an unknown unload fraction is unknown.
Explicit rho0 means caller-disabled unloading, not an observation of support.

Required positive multiplier scales: residual25, delta21, reference21,
intentP, progress. Soft task scales are controller design, not score weights.
SEARCH fixes gamma=0 through its box; motion_step remains active. A caller
may provide a domain excluding du0: the merit anchor clips du0/gamma0 into
the REAL boxes and projects its unmodified normal preference. No false empty
domain or retargeted objective is created. All anchor fields are reported.

Numerics reuse the frozen V18/V15 dimensionless feasible active-set LS flow,
but x=(du,z,gamma), no auxiliary load hinge. Original-unit analytic gradients
and direct box/laminar normal-cone stationarity/complementarity govern
acceptance, never a small projected/L step. P<=128,400 iterations, default
atol1e-8/rtol1e-7 are declared computational limits, not physical thresholds.
Unknowns, nonconvergence/stagnation and nonfinite arithmetic remain explicit.
No global floating optimum, actual geometry, future force, placement, or
real-time certificate is produced. Raw KKT gradients have units objective per
reference-coordinate, N and gamma respectively; tolerances are numerical.

evaluate_task(...actual_delta21=...) scores supplied actual native effort
with the SAME du/z/gamma, all other goals unchanged. Its gradient uses the
DECLARED branch slope; any affine mismatch is reported. It does not certify
the upstream native evaluation or an optimum of opaque floating native code.
"""
from dataclasses import dataclass,field
from collections.abc import Mapping
from numbers import Integral
import numpy as np
from v13_joint.intent_qp import (_InputError,_array,_scalar,_readonly,_domain,
    _project_z,_box_normal_cone,_normal_intent_cone,MAX_PATCHES)


@dataclass(frozen=True)
class TaskQPResult:
    status:str
    reason:str
    du21:object=None
    zP:object=None
    gamma:object=None
    d21:object=None
    p21:object=None
    residual25:object=None
    objective:object=None
    iterations:int=0
    diagnostics:dict=field(default_factory=dict)
    @property
    def available(self):return self.status=='converged' and self.du21 is not None


def _nullspace(rows,n):
    if not len(rows):return np.eye(n),0
    _,s,vt=np.linalg.svd(rows,full_matrices=True)
    threshold=max(rows.shape)*np.finfo(float).eps*(s[0] if len(s) else 0.)
    rank=int(np.sum(s>threshold));return vt[rank:].T,rank


def _prepare(p):
    if not isinstance(p,Mapping):raise _InputError('problem_requires_mapping')
    data={k:_array(p.get(k),shape,k) for k,shape in (
        ('p0_21',(21,)),('A21x25',(21,25)),('gain21',(21,)),('offset21',(21,)),
        ('delta_lower21',(21,)),('delta_upper21',(21,)),('open_step21',(21,)),
        ('motion_step21',(21,)),('tau_extra25',(25,)))}
    if np.any(data['delta_lower21']>data['delta_upper21']):raise _InputError('empty_reference_increment_box')
    if p.get('Jpx25') is None:raise _InputError('Jpx25_unknown',unknown=True)
    raw=np.asarray(p['Jpx25'],dtype=object)
    if raw.ndim!=2 or raw.shape[1]!=25:raise _InputError('Jpx25_requires_P_by25')
    P=len(raw)
    if P>MAX_PATCHES:raise _InputError('patch_count_exceeds128',unknown=True)
    data['Jpx25']=_array(raw,(P,25),'Jpx25');data['y_p']=_array(p.get('y_p'),(P,),'y_p')
    if np.any(data['y_p']<0):raise _InputError('normal_observation_must_be_nonnegative')
    groups,caps,budget=_domain(P,p.get('patch_digit_p'),p.get('caps5'),p.get('total_budget'))
    rho=_scalar(p.get('unload_fraction'),'unload_fraction',minimum=0.)
    if rho>1:raise _InputError('unload_fraction_requires0to1')
    gl=_scalar(p.get('gamma_lower',0.),'gamma_lower',minimum=0.);gh=_scalar(p.get('gamma_upper',1.),'gamma_upper',minimum=0.)
    if gl>gh or gh>1:raise _InputError('gamma_box_requires0to1')
    scales=p.get('scales')
    if not isinstance(scales,Mapping) or set(scales)!={'residual25','delta21','reference21','intentP','progress'}:
        raise _InputError('five_explicit_scales_required')
    sr=_array(scales['residual25'],(25,),'residual25');sd=_array(scales['delta21'],(21,),'delta21')
    su=_array(scales['reference21'],(21,),'reference21');sf=_array(scales['intentP'],(P,),'intentP')
    sg=_scalar(scales['progress'],'progress',strictly_positive=True)
    if any(np.any(x<=0) for x in (sr,sd,su,sf)):raise _InputError('scales_require_positive_multipliers')
    atol=_scalar(p.get('atol',1e-8),'atol',strictly_positive=True);rtol=_scalar(p.get('rtol',1e-7),'rtol',minimum=0.)
    data.update(P=P,groups=groups,caps=caps,budget=budget,rho=rho,gl=gl,gh=gh,sr=sr,sd=sd,su=su,sf=sf,sg=sg,atol=atol,rtol=rtol)
    return data


def _raw(p,du,z,gamma,actual_delta21=None):
    d_affine=p['offset21']+p['gain21']*du
    d=d_affine if actual_delta21 is None else actual_delta21
    correction=p['Jpx25'].T@(z-p['y_p'])+p['tau_extra25']
    r=p['A21x25'].T@d-correction
    ur=du-gamma*p['open_step21']-p['motion_step21']
    zr=z+gamma*p['rho']*p['y_p']-p['y_p']
    wr=p['sr']*r;wd=p['sd']*d;wu=p['su']*ur;wz=p['sf']*zr;wg=p['sg']*(gamma-1.)
    value=float(wr@wr+wd@wd+wu@wu+wz@wz+wg*wg)
    a=p['sr']*p['sr']*r
    gu=2.*p['gain21']*(p['A21x25']@a+p['sd']*p['sd']*d)+2.*p['su']*p['su']*ur
    gz=-2.*p['Jpx25']@a+2.*p['sf']*p['sf']*zr
    gg=float(-2.*p['open_step21']@(p['su']*p['su']*ur)+2.*(p['rho']*p['y_p'])@(p['sf']*p['sf']*zr)+2.*p['sg']*p['sg']*(gamma-1.))
    grad=np.r_[gu,gz,gg]
    if not np.isfinite(np.r_[d,r,grad,value]).all():raise FloatingPointError('nonfinite_raw_task_objective_or_gradient')
    return value,grad,r,d,d_affine,correction


def _diagnostics(p,du,z,gamma,g):
    box=_box_normal_cone(np.r_[g[:21],g[-1]],np.r_[du,gamma],np.r_[p['delta_lower21'],p['gl']],np.r_[p['delta_upper21'],p['gh']])
    normal=_normal_intent_cone(g[21:-1],z,p['groups'],p['caps'],p['budget'])
    td=p['atol']+p['rtol']*max(1.,float(np.max(abs(du))),abs(gamma))
    tz=p['atol']+p['rtol']*max(1.,float(np.max(abs(z),initial=0.)))
    violation=dict(reference_box=float(max(0.,np.max(p['delta_lower21']-du),np.max(du-p['delta_upper21']))),
        gamma_box=float(max(0.,p['gl']-gamma,gamma-p['gh'])),negative_normal=float(max(0.,-np.min(z))) if len(z) else 0.,
        digit_caps=float(max([0.]+[float(z[ids].sum()-cap) for ids,cap in zip(p['groups'],p['caps'])])),total_budget=float(max(0.,z.sum()-p['budget'])))
    done=(box['stationarity_inf']<=td and normal['stationarity_inf']<=tz and normal['complementarity_inf']<=p['atol']+p['rtol'] and normal['dual_nonnegativity_violation']<=p['atol'] and max(violation.values())<=p['atol'])
    return bool(done),dict(direct_box_KKT=box,direct_normal_KKT=normal,primal_violations=violation,numerical_tolerance_reference_gamma=td,numerical_tolerance_z=tz,feasible=max(violation.values())<=p['atol'])


def _anchor(p):
    du=np.clip(np.zeros(21),p['delta_lower21'],p['delta_upper21'])
    gamma=float(np.clip(0.,p['gl'],p['gh']))
    raw_z=p['y_p']*(1.-gamma*p['rho']);z=_project_z(raw_z,p['groups'],p['caps'],p['budget'])
    return du,z,gamma,dict(du21=du.tolist(),gamma=gamma,zP=z.tolist(),raw_normal_preference_p=raw_z.tolist(),
        zero_reference_increment_in_domain=bool(np.all(p['delta_lower21']<=0)&np.all(p['delta_upper21']>=0)),zero_gamma_in_domain=p['gl']<=0<=p['gh'],objective_goals_not_projected=True)


def evaluate_task(problem,du,z,gamma,actual_delta21=None):
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            p=_prepare(problem);u=_array(du,(21,),'du21');z=_array(z,(p['P'],),'zP');gamma=_scalar(gamma,'gamma')
            actual=None if actual_delta21 is None else _array(actual_delta21,(21,),'actual_delta21')
            value,g,r,d,affine,correction=_raw(p,u,z,gamma,actual);done,diag=_diagnostics(p,u,z,gamma,g)
            return dict(status='known',available=True,objective=value,residual25=_readonly(r),d21=_readonly(d),affine_d21=_readonly(affine),
                correction25=_readonly(correction),gradient_du21=_readonly(g[:21]),gradient_zP=_readonly(g[21:-1]),gradient_gamma=float(g[-1]),
                gamma=gamma,du21=_readonly(u),zP=_readonly(z),direct_KKT_met=done,**diag,
                native_delta_substituted=actual is not None,affine_realization_error21=_readonly(d-affine),
                gradient_scope='declared affine branch derivative, at supplied current effort; not opaque native floating optimum certification',
                objective_scope='one unchanged normal/reference/mechanical/progress task',physical_safety_certified=False,future_pressure_prediction_available=False)
    except _InputError as exc:return dict(status='unknown' if exc.unknown else 'invalid_input',available=False,reason=str(exc))
    except (ValueError,TypeError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:return dict(status='numerical_failure',available=False,reason=str(exc))


def check_same_problem_candidate(problem,result):
    record=dict(accepted=False,optimality_certified=False,different_problem_used=False)
    if result.du21 is None or result.zP is None or result.gamma is None or result.objective is None:return record
    ev=evaluate_task(problem,result.du21,result.zP,result.gamma)
    if not ev['available']:return dict(record,reason=ev['reason'])
    try:
        p=_prepare(problem);u,z,gamma,anchor=_anchor(p);base=_raw(p,u,z,gamma)[0]
        same=np.float64(ev['objective']).tobytes()==np.float64(result.objective).tobytes()
        residual=result.residual25 is not None and np.asarray(result.residual25).shape==(25,) and np.asarray(result.residual25).dtype==np.dtype(float) and np.asarray(result.residual25).tobytes()==ev['residual25'].tobytes()
        delta=result.d21 is not None and np.asarray(result.d21).tobytes()==ev['d21'].tobytes()
        allowance=64*np.finfo(float).eps*max(1.,abs(base),abs(ev['objective']))
        honest=not result.available or ev['direct_KKT_met']
        accepted=bool(same and residual and delta and ev['feasible'] and honest and ev['objective']<=base+allowance)
        return dict(record,accepted=accepted,optimality_certified=bool(accepted and result.available and ev['direct_KKT_met']),
            classification='certified' if result.available and ev['direct_KKT_met'] else 'feasible_nonoptimal_same_problem',
            anchor=anchor,anchor_objective=base,original_objective=ev['objective'],objective_bytes_equal=same,
            residual_bytes_equal=residual,affine_delta_bytes_equal=delta,objective_roundoff_allowance=allowance,
            fresh_direct_KKT_met=ev['direct_KKT_met'],primal_violations=ev['primal_violations'])
    except (_InputError,ValueError,TypeError,FloatingPointError,OverflowError) as exc:return dict(record,reason=str(exc))


def solve_task_qp(*,p0_21,A21x25,gain21,offset21,Jpx25,y_p,tau_extra25,
    delta_lower21,delta_upper21,open_step21,motion_step21,unload_fraction,
    patch_digit_p,caps5,total_budget,scales,gamma_lower=0.,gamma_upper=1.,
    max_iterations=400,atol=1e-8,rtol=1e-7,warm_start=None):
    problem=locals().copy();scope=dict(algorithm='V19_dimensionless_feasible_active_set_LS',max_patches=MAX_PATCHES,
        max_iterations=max_iterations,atol=atol,rtol=rtol,reference_increment_is_not_native_effort=True,
        native_branch_qualification_owned_by_caller=True,physical_safety_certified=False,future_pressure_prediction_available=False,global_floating_optimum_certified=False)
    try:
        p=_prepare(problem)
        if isinstance(max_iterations,(bool,np.bool_)) or not isinstance(max_iterations,Integral) or max_iterations<1:raise _InputError('positive_integer_iteration_budget_required')
        if warm_start is not None and (not isinstance(warm_start,Mapping) or set(warm_start)!={'du21','zP','gamma'}):raise _InputError('warm_start_requires_du21_zP_gamma')
    except _InputError as exc:return TaskQPResult('unknown' if exc.unknown else 'invalid_input',str(exc),diagnostics=scope)
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            P=p['P'];n=22+P;last=n-1;s=np.r_[p['su'],p['sf'],p['sg']]
            # Build every weighted residual once, with its original nonzero rhs.
            mech=np.c_[p['A21x25'].T*p['gain21'][None,:],-p['Jpx25'].T,np.zeros(25)]
            ref=np.c_[np.eye(21),np.zeros((21,P)),-p['open_step21']]
            normal=np.c_[np.zeros((P,21)),np.eye(P),p['rho']*p['y_p']]
            effort=np.c_[np.diag(p['gain21']),np.zeros((21,P+1))]
            progress=np.zeros((1,n));progress[0,last]=1.
            E=np.vstack((p['sr'][:,None]*mech,p['su'][:,None]*ref,p['sf'][:,None]*normal,p['sd'][:,None]*effort,p['sg']*progress))
            b=np.r_[-p['sr']*(p['A21x25'].T@p['offset21']+p['Jpx25'].T@p['y_p']-p['tau_extra25']),p['su']*p['motion_step21'],p['sf']*p['y_p'],-p['sd']*p['offset21'],p['sg']]
            R=E/s[None,:]
            if not np.isfinite(np.r_[R.ravel(),b]).all():raise FloatingPointError('nonfinite_LS_system')
            fixed={i:s[i]*p['delta_lower21'][i] for i in range(21) if p['delta_lower21'][i]==p['delta_upper21'][i]}
            for ids,cap in zip(p['groups'],p['caps']):
                if cap==0 or p['budget']==0:
                    for j in ids:fixed[21+int(j)]=0.
            if p['gl']==p['gh']:fixed[last]=p['sg']*p['gl']
            equal=np.eye(n)[list(fixed)] if fixed else np.empty((0,n));G=[];h=[];labels=[]
            def add(row,bound,label):
                maximum=float(np.max(abs(row),initial=0.))
                if maximum==0 or not np.isfinite(maximum):raise FloatingPointError('invalid_constraint_row')
                norm=float(np.linalg.norm(row/maximum));G.append((row/maximum)/norm);h.append((float(bound)/maximum)/norm);labels.append(label)
            for i in range(21):
                if i not in fixed:
                    add(-np.eye(n)[i],-s[i]*p['delta_lower21'][i],('u_lower',i));add(np.eye(n)[i],s[i]*p['delta_upper21'][i],('u_upper',i))
            for j in range(P):
                if 21+j not in fixed:add(-np.eye(n)[21+j],0.,('z_lower',j))
            for digit,(ids,cap) in enumerate(zip(p['groups'],p['caps'])):
                if len(ids) and cap>0 and p['budget']>0:
                    row=np.zeros(n);row[21+ids]=1./p['sf'][ids];add(row,cap,('group',digit))
            if P and p['budget']>0:
                row=np.zeros(n);row[21:21+P]=1./p['sf'];add(row,p['budget'],('budget',0))
            if last not in fixed:
                add(-np.eye(n)[last],-p['sg']*p['gl'],('gamma_lower',0));add(np.eye(n)[last],p['sg']*p['gh'],('gamma_upper',0))
            G=np.asarray(G).reshape((-1,n));h=np.asarray(h);active=[];adds=removes=dependent=solves=0;history=[]
            def canonical(w):
                x=w/s
                for i in fixed:x[i]=(p['delta_lower21'][i] if i<21 else p['gl'] if i==last else 0.)
                for k in active:
                    name,i=labels[k]
                    if name=='u_lower':x[i]=p['delta_lower21'][i]
                    elif name=='u_upper':x[i]=p['delta_upper21'][i]
                    elif name=='z_lower':x[21+i]=0.
                    elif name=='gamma_lower':x[last]=p['gl']
                    elif name=='gamma_upper':x[last]=p['gh']
                x[:21]=np.clip(x[:21],p['delta_lower21'],p['delta_upper21']);x[21:last]=_project_z(x[21:last],p['groups'],p['caps'],p['budget']);x[last]=np.clip(x[last],p['gl'],p['gh'])
                w=s*x
                for i,value in fixed.items():w[i]=value
                return w,x
            au,az,ag,anchor=_anchor(p)
            if warm_start is None:x=np.r_[au,az,ag]
            else:x=np.r_[_array(warm_start['du21'],(21,),'warm_du21'),_array(warm_start['zP'],(P,),'warm_zP'),_scalar(warm_start['gamma'],'warm_gamma')]
            w,x=canonical(s*x)
            def add_independent(k):
                nonlocal dependent
                W=np.vstack((equal,G[active])) if active else equal;Z,_=_nullspace(W,n)
                if np.linalg.norm(G[k]@Z)<=64*np.finfo(float).eps*max(1,n):dependent+=1;return False
                active.append(k);return True
            slack=h-G@w
            for k in range(len(h)):
                rounding=8*np.finfo(float).eps*max(abs(h[k]),float(abs(G[k])@abs(w)))
                if abs(slack[k])<=rounding:add_independent(k)
            value,g,r,d,_,_=_raw(p,x[:21],x[21:last],x[last]);done,diag=_diagnostics(p,x[:21],x[21:last],x[last],g)
            status='converged' if done else 'nonconvergence';reason='direct_original_KKT_met' if done else 'iteration_budget_exhausted';iteration=0
            for iteration in range(1,int(max_iterations)+1):
                if done:iteration-=1;break
                W=np.vstack((equal,G[active])) if active else equal;Z,rank=_nullspace(W,n)
                step=Z@np.linalg.lstsq(R@Z,b-R@w,rcond=None)[0] if Z.shape[1] else np.zeros(n);solves+=bool(Z.shape[1])
                if not np.isfinite(step).all():raise FloatingPointError('nonfinite_LS_step')
                fraction=1.;block=None;gp=G@step;slack=h-G@w
                for k in range(len(h)):
                    if k not in active and gp[k]>0:
                        limit=max(0.,float(slack[k]))/float(gp[k])
                        if limit<fraction:fraction=limit;block=k
                next_w,next_x=canonical(w+fraction*step);same=np.array_equal(next_w,w)
                value,g,r,d,_,_=_raw(p,next_x[:21],next_x[21:last],next_x[last]);done,diag=_diagnostics(p,next_x[:21],next_x[21:last],next_x[last],g)
                history.append(dict(iteration=iteration,objective=value,step_fraction=fraction,active=len(active),rank=rank,direct_box=diag['direct_box_KKT']['stationarity_inf'],direct_normal=diag['direct_normal_KKT']['stationarity_inf']))
                w,x=next_w,next_x
                if done:status='converged';reason='direct_original_KKT_met';break
                if block is not None and add_independent(block):adds+=1;continue
                at_min=same or np.linalg.norm(step)<=64*np.finfo(float).eps*max(1.,float(np.linalg.norm(w))) or fraction==1.
                if at_min:
                    grad_w=2.*R.T@(R@w-b);mult=np.linalg.lstsq(W.T,-grad_w,rcond=None)[0] if len(W) else np.empty(0);dual=mult[len(equal):]
                    if len(dual) and np.min(dual)<0:active.pop(int(np.argmin(dual)));removes+=1;continue
                    if same:status='numerical_stagnation';reason='no_representable_step_with_failed_direct_KKT';break
            diagnostics=dict(scope,**diag,merit_anchor=anchor,merit_anchor_objective=_raw(p,au,az,ag)[0],
                fixed_original_p0_21=p['p0_21'].tolist(),affine_offset21=p['offset21'].tolist(),affine_gain21=p['gain21'].tolist(),
                coordinate_scale=s.tolist(),least_squares_rows=len(R),expanded_LS_objective=float(np.square(R@w-b).sum()),
                active_additions=adds,active_removals=removes,dependent_blockers=dependent,least_squares_solves=solves,history=history,
                candidate_requires_status_check=not done,numerical_method='dimensionless SVD active-set LS; raw direct cone KKT',
                raw_variable_units='du first18/wrists rad and lift m; z N; gamma dimensionless')
            return TaskQPResult(status,reason,_readonly(x[:21]),_readonly(x[21:last]),float(x[last]),_readonly(d),_readonly(p['p0_21']+d),_readonly(r),value,iteration,diagnostics)
    except _InputError as exc:return TaskQPResult('unknown' if exc.unknown else 'invalid_input',str(exc),diagnostics=scope)
    except (ValueError,TypeError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:return TaskQPResult('numerical_failure',str(exc),diagnostics=scope)

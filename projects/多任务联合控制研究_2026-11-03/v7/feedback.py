"""Reachable, coupled force/motion tasks in actual finite reference space.

There is no per-finger deadzone, free-space inverse, or post-solve clipping.
The aggregate grip request is governed and projected in the same space as
material protection, persistent contact, and public motion. Conditional noise
statistics and the unmodified calibration error have different roles.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class PlanResult:
    first_action8: np.ndarray
    plan: np.ndarray
    predicted_pressure: np.ndarray
    debug: dict


class ModelDomainUnavailable(ValueError):
    """An affine contact mode lacks a checked nonnegative pressure seed.

    This is a model availability result, never physical infeasibility.
    """


def array(value, shape, name, finite=True):
    a=np.asarray(value,float)
    if a.shape!=shape or finite and not np.isfinite(a).all():
        raise ValueError(f'{name}: expected {shape} finite numeric values')
    return a.copy()


def _qp(Q,q,A,b,seed,max_iterations=180):
    """Feasible-start primal active set; independently recheck original rows.

    All variables and constraints are scaled before this solver. Singular task
    Hessians get only a numerical proximal tie-break. An iteration limit is
    reported as such, never renamed optimal. No simulator/scipy is imported.
    """
    x=np.asarray(seed,float).copy();n=len(x)
    if np.max(A@x-b,initial=0.)>2e-7:raise ValueError('QP requires a genuinely feasible seed')
    scale=max(1.,float(np.max(abs(Q),initial=0.)),float(np.max(abs(q),initial=0.)))
    H=(Q+Q.T)*(.5/scale)+np.eye(n)*1e-10
    f=q/scale-1e-10*x
    norm=np.linalg.norm(A,axis=1);keep=norm>1e-14
    if np.any(b[~keep]<-1e-8):raise ValueError('Inconsistent zero constraint')
    AA=A[keep]/norm[keep,None];bb=b[keep]/norm[keep]
    working=[]
    def insert(index):
        if index in working:return
        rows=AA[working+[index]]
        if np.linalg.matrix_rank(rows,tol=1e-10)>len(working):working.append(int(index))
    for j in np.flatnonzero(AA@x-bb>-1e-9):insert(j)
    status='iteration_limit_feasible';iterations=0
    for iterations in range(1,max_iterations+1):
        gradient=H@x+f
        if working:
            W=AA[working]
            _,s,vt=np.linalg.svd(W,full_matrices=True)
            rank=int(np.count_nonzero(s>1e-10));Z=vt[rank:].T
        else:W=np.empty((0,n));Z=np.eye(n)
        if Z.shape[1]:
            reduced=Z.T@H@Z
            direction=-Z@np.linalg.lstsq(reduced,Z.T@gradient,rcond=1e-12)[0]
        else:direction=np.zeros(n)
        if np.linalg.norm(direction,np.inf)<2e-9:
            multipliers=np.linalg.lstsq(W.T,-gradient,rcond=1e-11)[0] if working else np.empty(0)
            if not len(multipliers) or np.min(multipliers)>=-2e-8:
                status='optimal_active_set';break
            del working[int(np.argmin(multipliers))]
            continue
        denom=AA@direction;slack=bb-AA@x
        eligible=denom>1e-12
        if working:eligible[working]=False
        ratios=np.full(len(bb),np.inf);ratios[eligible]=np.maximum(0.,slack[eligible])/denom[eligible]
        blocker=int(np.argmin(ratios)) if len(ratios) else -1
        step=min(1.,float(ratios[blocker])) if blocker>=0 else 1.
        previous=x.copy();x=x+step*direction
        if np.max(A@x-b,initial=0.)>1e-7:
            x=previous;status='numerical_step_rejected';break
        if step<1.-1e-10 and blocker>=0:insert(blocker)
    residual=max(0.,float(np.max(A@x-b,initial=0.)))
    return x,dict(status=status,iterations=iterations,residual=residual,
        optimal=status=='optimal_active_set')


class ReachableTaskPlanner:
    """One finite, horizon-wide feasible set; five explicit task levels."""
    def solve(self,*,pressure_now5,pressure_covariance5x5,pressure_map,pressure_free5H,
              calibration_error5H,observed_pressure5,pressure_valid5,limits5,budget,
              alpha5,alpha_lower5,alpha_upper5,step_lower8,step_upper8,motion_now3,
              user_targetsH3,motion_lower3,motion_upper3,tracking_tolerances3,
              persistent5,force_total_target,previous_action8=None,model_usable=True,
              forecast_covariance=None,joint_pressure_covariance=None,
              aggregate_mask5=None,observed_valid5=None,fixed_motion_planH3=None,
              aggregate_task_enabled=True,reference_fraction=1.,wrench_map=None,
              wrench_free=None,wrench_now=None,wrench_scales=None):
        user=np.asarray(user_targetsH3,float)
        if user.ndim!=2 or user.shape[1]!=3 or not len(user) or not np.isfinite(user).all():
            raise ValueError('Original finite Hx3 user targets required')
        h=len(user);nr=8*h
        p=array(pressure_now5,(5,),'pressure now',False)
        valid=np.asarray(pressure_valid5,bool)
        if valid.shape!=(5,) or not np.isfinite(p[valid]).all():
            raise ValueError('Available pressure channels must be finite')
        observed=array(observed_pressure5,(5,),'raw pressure',False)
        observed_valid=valid.copy() if observed_valid5 is None else np.asarray(observed_valid5,bool)
        if observed_valid.shape!=(5,) or not np.isfinite(observed[observed_valid]).all():raise ValueError('Valid raw pressure must be observed')
        if not valid.any() and not observed_valid.any():raise ValueError('Neither posterior nor raw pressure is available')
        covariance=array(pressure_covariance5x5,(5,5),'posterior pressure covariance')
        if np.max(abs(covariance-covariance.T))>1e-9 or np.min(np.linalg.eigvalsh(covariance))<-1e-10:
            raise ValueError('Posterior covariance must be symmetric positive semidefinite')
        P=array(pressure_map,(5*h,nr),'pressure map')
        c=array(pressure_free5H,(5*h,),'pressure free')
        error=array(calibration_error5H,(5*h,),'complete calibration error')
        caps=array(limits5,(5,),'material caps');budget=float(budget)
        a=array(alpha5,(5,),'actual alpha');al=array(alpha_lower5,(5,),'alpha lower');au=array(alpha_upper5,(5,),'alpha upper')
        lo=array(step_lower8,(8,),'step lower');hi=array(step_upper8,(8,),'step upper')
        motion=array(motion_now3,(3,),'actual motion command');ml=array(motion_lower3,(3,),'motion lower');mu=array(motion_upper3,(3,),'motion upper')
        persistent=np.asarray(persistent5,bool)&valid
        if persistent.shape!=(5,) or np.any(error<0.) or np.any(caps<0.) or not np.isfinite(budget) or budget<0.:
            raise ValueError('Finite nonnegative uncertainty/caps/budget required')
        if np.any(lo>hi) or np.any(al>au) or np.any(ml>mu) or np.any(a<al-1e-8) or np.any(a>au+1e-8) or np.any(motion<ml-1e-8) or np.any(motion>mu+1e-8):
            raise ValueError('Current state and real execution bounds are inconsistent')
        target=float(force_total_target);fraction=float(reference_fraction)
        if not np.isfinite([target,fraction]).all() or target<0. or not 0.<fraction<=1.:
            raise ValueError('Nonnegative grip request and governed fraction in(0,1] required')
        tol=tuple(tracking_tolerances3)
        if len(tol)!=3 or any(t is not None and (not np.isfinite(t) or t<0.) for t in tol):
            raise ValueError('Explicit public per-axis task tolerances required')
        # Remove fixed command coordinates BEFORE forming any inverse/optimum.
        scale=np.maximum(np.maximum(abs(lo),abs(hi)),1e-8)
        full_lo=np.tile(lo,h);full_hi=np.tile(hi,h);full_scale=np.tile(scale,h)
        fixed=abs(full_hi-full_lo)<=1e-12;offset=np.where(fixed,full_lo,0.)
        free=np.flatnonzero(~fixed);R=np.zeros((nr,len(free)))
        R[free,np.arange(len(free))]=full_scale[free]
        nu=len(free);dim=nu+14
        sf=slice(nu,nu+5);sb=nu+5;sc=slice(nu+6,nu+11);sm=slice(nu+11,nu+14)
        I=np.eye(dim);U=np.zeros((nr,dim));U[:,:nu]=R
        N=P@U;response_offset=P@offset;cp=c+response_offset
        C=np.kron(np.tril(np.ones((h,h))),np.eye(8));CU=C@U;co=C@offset
        ai=np.concatenate([np.arange(8*j,8*j+5) for j in range(h)])
        mi=np.concatenate([np.arange(8*j+5,8*j+8) for j in range(h)])
        rows=[];rhs=[]
        def add(A,b):rows.append(np.atleast_2d(A));rhs.append(np.atleast_1d(b))
        add(U,full_hi-offset);add(-U,offset-full_lo)
        add(CU[ai],np.tile(au-a,h)-co[ai]);add(-CU[ai],np.tile(a-al,h)+co[ai])
        add(CU[mi],np.tile(mu-motion,h)-co[mi]);add(-CU[mi],np.tile(motion-ml,h)+co[mi])
        if fixed_motion_planH3 is not None:
            fixed_motion=array(fixed_motion_planH3,(h,3),'explicit external finite-ramp motion plan')
            increments=np.cumsum(fixed_motion,axis=0).ravel()
            add(CU[mi],increments-co[mi]+1e-12);add(-CU[mi],co[mi]-increments+1e-12)
        add(-I[nu:],np.zeros(14))
        output_rows=np.flatnonzero(np.tile(valid,h))
        # A negative unsigned pressure cannot be spent as grip/budget credit.
        # Constrain the raw affine values rather than clipping its prediction.
        add(-N[output_rows],cp[output_rows])
        # The separate raw-anchor response allocation must not manufacture
        # negative measured-pressure credit either (even in empirical mode).
        raw_rows=np.flatnonzero(np.tile(observed_valid,h))
        add(-N[raw_rows],np.tile(np.where(observed_valid,observed,0.),h)[raw_rows]+response_offset[raw_rows])
        hard_A=np.vstack(rows);hard_b=np.concatenate(rhs)
        # The current posterior and forecast measurement statistics are NOT
        # calibration bounds and are never used as a componentwise deadzone.
        if forecast_covariance is None:fc=np.kron(np.ones((h,h)),covariance)
        else:fc=array(forecast_covariance,(5*h,5*h),'forecast pressure covariance')
        if np.min(np.linalg.eigvalsh((fc+fc.T)*.5))<-1e-8:raise ValueError('Invalid forecast covariance')
        sd=np.sqrt(np.maximum(0.,np.diag(fc)))
        guard=error+3.*sd if model_usable else np.zeros(5*h)
        for j in range(h):
            for finger in np.flatnonzero(valid|observed_valid):
                k=5*j+finger;row=N[k].copy();row[nu+finger]-=1.
                if valid[finger]:add(row,caps[finger]-cp[k]-guard[k])
                if observed_valid[finger]:
                    # Independent latest RAW-pressure intervention. This is a
                    # reactive response allocation, not replacement of the
                    # statistical posterior with an outlier or a safety proof.
                    add(row,caps[finger]-observed[finger]-response_offset[k])
                # Invalid/unknown raw channels never provide spare budget or
                # apparent zero force. Only a complete pressure vector can
                # support a complete five-finger budget constraint.
                if persistent[finger]:
                    row=-N[k].copy();row[nu+6+finger]-=1.
                    add(row,cp[k]-min(.15,max(0.,p[finger])))
            ns=N[5*j:5*j+5].sum(axis=0);ns[sb]-=1.
            if valid.all():add(ns,budget-float(cp[5*j:5*j+5].sum()+guard[5*j:5*j+5].sum()))
            if observed_valid.all():add(ns,budget-float(observed.sum()+response_offset[5*j:5*j+5].sum()))
        A=np.vstack(rows);b=np.concatenate(rhs)
        # A mechanical seed follows as much of the PUBLIC request as the real
        # robot can execute. It does not redefine that original request.
        nominal=[];last=motion.copy();alpha=a.copy()
        for j in range(h):
            da=np.minimum(hi[:5],np.maximum(lo[:5],np.zeros(5)))
            da=np.minimum(au-alpha,np.maximum(al-alpha,da));alpha+=da
            dm=np.minimum(hi[5:],np.maximum(lo[5:],user[j]-last))
            if fixed_motion_planH3 is not None:dm=fixed_motion[j].copy()
            dm=np.minimum(mu-last,np.maximum(ml-last,dm));last+=dm
            nominal.append(np.r_[da,dm])
        nominal=np.asarray(nominal).ravel();seed=np.zeros(dim)
        seed[:nu]=(nominal[free]-offset[free])/full_scale[free]
        def complete(x):
            y=cp+N@x
            for finger in range(5):
                x[nu+finger]=max(0.,float(np.max(y[finger::5]+guard[finger::5]-caps[finger]))) if valid[finger] else 0.
                if observed_valid[finger]:
                    delta=(response_offset+N@x)[finger::5]
                    x[nu+finger]=max(x[nu+finger],float(np.max(observed[finger]+delta-caps[finger])))
                x[nu+6+finger]=max(0.,min(.15,max(0.,p[finger]))-float(np.min(y[finger::5]))) if persistent[finger] else 0.
            x[sb]=max(0.,float(np.max((y+guard).reshape(h,5).sum(axis=1)-budget))) if valid.all() else 0.
            if observed_valid.all():x[sb]=max(x[sb],float(np.max((response_offset+N@x).reshape(h,5).sum(axis=1)+observed.sum()-budget)))
            return x
        seed=complete(seed)
        if np.max(A@seed-b,initial=0.)>2e-7:
            # Zero commanded increments can be a checked mode seed when the
            # public motion seed itself would leave the affine contact mode.
            zero=np.zeros(dim);zero=complete(zero)
            if np.max(A@zero-b,initial=0.)<=2e-7:seed=zero
            else:raise ModelDomainUnavailable('No checked finite nonnegative affine seed; contact model unavailable, not physical infeasibility')
        stages=[];x=seed.copy();protected=[]
        def objective(M,r):return 2.*M.T@M,-2.*M.T@r
        def solve_stage(name,M,r):
            nonlocal x,A,b
            Q,q=objective(M,r);old=x.copy();candidate,info=_qp(Q,q,A,b,x)
            old_cost=float(np.linalg.norm(M@old-r)**2);new_cost=float(np.linalg.norm(M@candidate-r)**2)
            checked=max(0.,float(np.max(A@candidate-b,initial=0.)))
            if checked>2e-7 or new_cost>old_cost+1e-7:
                candidate=old;info.update(status='checked_previous_feasible',optimal=False)
            x=candidate;info.update(stage=name,checked_primal_residual=max(0.,float(np.max(A@x-b,initial=0.))),objective=float(np.linalg.norm(M@x-r)**2))
            stages.append(info)
        def freeze_slack(indices,eps=2e-6):
            nonlocal A,b
            indices=np.asarray(indices,int);A=np.vstack((A,I[indices]));b=np.r_[b,x[indices]+eps]
            protected.extend(indices.tolist())
        safety=I[np.r_[np.arange(sf.start,sf.stop),sb]]
        solve_stage('material_pressure_and_total_budget',safety,np.zeros(6));freeze_slack(np.r_[np.arange(sf.start,sf.stop),sb])
        solve_stage('persistent_contact_retention',I[sc],np.zeros(5));freeze_slack(np.arange(sc.start,sc.stop))
        # First attain the closest real motion under the higher-priority risk
        # decisions, then preserve it for force quality. No hidden target clip.
        mscale=np.asarray([max(1e-4,float(t)) if t is not None else (1. if k==0 else .1) for k,t in enumerate(tol)])
        M=CU[mi]/np.tile(mscale,h)[:,None]
        r=(user-motion).ravel()/np.tile(mscale,h)-co[mi]/np.tile(mscale,h)
        solve_stage('public_motion_progress',M,r)
        motion_result=CU[mi]@x+co[mi]
        A=np.vstack((A,CU[mi],-CU[mi]));b=np.r_[b,motion_result-co[mi]+1e-9,co[mi]-motion_result+1e-9]
        # Aggregate grip is a governed REFERENCE SET, not five independent
        # exact pressure targets. Preserve its actual reachable projection.
        # The requested group is a task declaration, not a fresh-observation
        # denominator. A missing finger has UNKNOWN contribution, not zero.
        support=np.ones(5,bool) if aggregate_mask5 is None else np.asarray(aggregate_mask5,bool)
        if support.shape!=(5,) or not support.any():raise ValueError('An explicit nonempty grip task group is required')
        if type(aggregate_task_enabled) is not bool:raise ValueError('Explicit aggregate observability flag required')
        aggregate_available=bool(aggregate_task_enabled and np.all(valid[support]))
        sum_map=np.zeros((h,5*h))
        for j in range(h):sum_map[j,5*j:5*j+5]=support
        total_now=float(p[support].sum()) if aggregate_available else float('nan');requested=target
        governed=total_now+(1.-(1.-fraction)**np.arange(1,h+1))*(target-total_now)
        G=sum_map@N;gc=sum_map@cp
        aggregate_cov=sum_map@fc@sum_map.T
        aggregate_sd=np.sqrt(np.maximum(0.,np.diag(aggregate_cov)))
        aggregate_scale=np.maximum(aggregate_sd,1e-4)
        if aggregate_available:
            solve_stage('reachable_aggregate_grip_reference',G/aggregate_scale[:,None],(governed-gc)/aggregate_scale)
            projected=gc+G@x;allowed=np.abs(projected-governed)+aggregate_sd+2e-6
            A=np.vstack((A,G,-G));b=np.r_[b,governed+allowed-gc,gc-governed+allowed]
        else:
            projected=np.full(h,np.nan)
            stages.append(dict(stage='reachable_aggregate_grip_reference',status='group_contribution_unobserved',optimal=False,
                reason='No full-group force deficit is inferred from a missing contribution'))
        # One posterior, one total response: no E_motion point subtraction.
        # Conditional uncertainty downweights statistically weak pressure
        # channels; calibration error remains solely an independent risk bound.
        D=np.eye(5*h)
        if h>1:D[5:,:-5]-=np.eye(5*(h-1))
        dp0=D@cp;dp0[:5]-=np.where(valid,p,0.)
        use=np.tile(valid,h)
        if joint_pressure_covariance is None:
            covdiff=D@fc@D.T
            covdiff[:5,:5]+=covariance # conservative without unprovided current/future correlation
            covariance_role='full cross-future covariance plus conservative current block'
        else:
            joint_cov=array(joint_pressure_covariance,(5*(h+1),5*(h+1)),'joint current/future pressure covariance')
            difference=np.zeros((5*h,5*(h+1)))
            for j in range(h):difference[5*j:5*j+5,5*j:5*j+5]=-np.eye(5);difference[5*j:5*j+5,5*j+5:5*j+10]=np.eye(5)
            covdiff=difference@joint_cov@difference.T
            covariance_role='joint current/future conditional covariance, including cross-finger and cross-time correlation'
        subcov=covdiff[np.ix_(use,use)];values,vectors=np.linalg.eigh((subcov+subcov.T)*.5)
        if np.min(values,initial=0.)<-1e-8:raise ValueError('Invalid pressure-difference covariance')
        whitening=(vectors/np.sqrt(np.maximum(1e-8,values))).T
        Qm=whitening@(D@N)[use];qr=-whitening@dp0[use]
        # Numerically small regularization selects the least command activity
        # when a task is physically unreachable or indifferent to redistribution.
        effort=np.zeros((nr,dim));effort[:,:nu]=R/full_scale[:,None]
        previous=np.zeros(nr)
        if previous_action8 is not None:previous[:8]=array(previous_action8,(8,),'previous actual interval')/scale
        mats=[Qm,effort*1e-3];targets=[qr,np.zeros(nr)]
        if previous_action8 is not None:
            mats.append(effort[:8]*1e-3);targets.append(previous[:8]*1e-3)
        if wrench_map is not None:
            W=np.asarray(wrench_map,float);wf=np.asarray(wrench_free,float);wn=np.asarray(wrench_now,float)
            ws=np.asarray(wrench_scales,float)
            if W.shape!=(6*h,nr) or wf.shape!=(6*h,) or wn.shape!=(6,) or ws.shape!=(6,) or np.any(ws<=0.):raise ValueError('Explicit complete wrench continuity inputs required')
            # A low-priority full-skin continuity task, NEVER object equilibrium.
            mats.append((W@U)/np.tile(ws,h)[:,None]);targets.append((np.tile(wn,h)-wf-W@offset)/np.tile(ws,h))
        solve_stage('posterior_pressure_quality_and_actual_activity',np.vstack(mats),np.concatenate(targets))
        full=offset+U@x;plan=full.reshape(h,8);future=(c+P@full).reshape(h,5)
        authority=max(0.,float(np.max(hard_A@x-hard_b,initial=0.)))
        priority=max(0.,float(np.max(A@x-b,initial=0.)))
        if authority>2e-7 or priority>2e-7:raise RuntimeError('Rejected actual plan after final constraint recomputation')
        achieved=(sum_map@future.ravel()) if aggregate_available else np.full(h,np.nan)
        executed_motion=motion+np.cumsum(plan[:,5:],axis=0)
        tube=np.zeros((h,3))
        for i,t in enumerate(tol):
            if t is not None:tube[:,i]=np.maximum(0.,abs(executed_motion[:,i]-user[:,i])-t)
        debug=dict(schema='v7-reachable-task-plan-v1',stage_results=stages,
            final_authority_residual=authority,final_priority_residual=priority,
            actual_free_command_coordinates=free.tolist(),active_fixed_command_coordinates=np.flatnonzero(fixed).tolist(),
            requested_total_force_N=requested,governed_total_force_N=governed.tolist(),
            projected_total_force_N=projected.tolist(),actual_total_force_proxy_N=achieved.tolist(),
            requested_aggregate_gap=(requested-achieved).tolist(),
            governed_aggregate_gap=(governed-achieved).tolist(),reachable_projection_gap=(projected-achieved).tolist(),
            governed_projection_gap=(governed-projected).tolist(),aggregate_reference_sigma_N=aggregate_sd.tolist(),
            tube_violation=tube.tolist(),original_user_targets=user.tolist(),actual_motion_reference=executed_motion.tolist(),
            pressure_valid=valid.tolist(),raw_pressure_observed=observed_valid.tolist(),grip_support_mask=support.tolist(),
            aggregate_task_available=aggregate_available,complete_budget_observed=bool(observed_valid.all()),
            unknown_contribution_policy='suspend complete aggregate request; never make a missing finger zero or shrink its target denominator',
            raw_pressure_excess_N=np.where(observed_valid,np.maximum(0.,observed-caps),np.nan).tolist(),
            raw_guard_role='separate causal measured-pressure response allocation; not a new force posterior or safety certificate',
            material_slack_N=x[sf].tolist(),budget_slack_N=float(x[sb]),contact_slack_N=x[sc].tolist(),
            calibration_error_preserved=error.reshape(h,5).tolist(),conditional_posterior_sigma_N=np.sqrt(np.maximum(0.,np.diag(covariance))).tolist(),
            model_usable=bool(model_usable),prediction_available=bool(model_usable),
            model_based_force_safety_certified=False,global_model_admitted=False,
            pressure_prediction_role='experimental local response' if model_usable else 'uncertified empirical allocation only; no physical forecast',
            quality_covariance_role=covariance_role,no_componentwise_deadzone=True,no_inverse_then_clip=True)
        return PlanResult(plan[0].copy(),plan,future if model_usable else np.full((h,5),np.nan),debug)

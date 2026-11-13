"""Contact-conditioned force/motion predictive coordinator, research v1.

Directly plans synergy increments and continuous task progress. Pressure safety
is optimized first; force stability, complete measured hand wrench and task
progress share a second optimization over the same executable trajectory.
The common preparation, local execution domain and mechanical release are
inherited unchanged. This first version is not a certified object-dynamics MPC.
"""
from dataclasses import dataclass,asdict
from collections import deque
import numpy as np
import research_imports  # read-only foundation imports; bytecode disabled
from vendor.frozen_controllers import Gains,make_gains,solve_qp,ControlAction
from state_estimator import CoupledTactileState
from coordinator_model import ContactConditionedModel,verify_model_algebra


@dataclass
class CoordinatorSettings:
    force_anchor_weight: float=1.
    force_variation_weight: float=4.
    wrench_weight: float=.5
    contact_reserve_weight: float=3.
    progress_weight: float=8.
    action_weight: float=.02
    motion_rates: tuple=(.035,.15,.10)
    forecast_margin: float=.05
    pressure_stage_tolerance: float=.0005
    qp_passes: int=180


class JointCoordinator:
    def __init__(self, calibration, limits, budget, gains=None):
        B=np.asarray(calibration['force_matrix'],float)
        g=dict(gains or {})
        g.update(force_gain=tuple(np.maximum(.05,np.linalg.norm(B,axis=0))),
                 force_tau=float(calibration['force_tau']),release_rate=.8)
        self.g=make_gains(g);self.settings=CoordinatorSettings()
        self.state=CoupledTactileState(self.g)
        self.model=ContactConditionedModel(calibration)
        self.limits=np.asarray(limits,float);self.budget=float(budget)
        self.name='joint';self.desired=np.zeros(5);self.predicted=np.full(5,np.nan)
        self.last_decision=-np.inf;self.previous_time=None
        self.interval_action=np.zeros(5);self.interval_motion=np.zeros(3)
        self.motion_request=np.zeros(3);self.executed_motion=np.zeros(3);self.motion_reference=np.zeros(3)
        self.anchor=None;self.wrench_anchor=None;self.anchor_load=None;self.anchor_up=None
        self.previous_action=np.zeros(5);self.previous_plan=None;self.debug={};self.label='joint_initializing'
        self.numerical_failures=0;self.infeasible_ticks=0;self.control_ticks=0
        self.force_reference=np.zeros(5)

    def set_motion_request(self,target,executed_reference):
        target=np.asarray(target,float);executed=np.asarray(executed_reference,float)
        if target.shape!=(3,) or executed.shape!=(3,) or not np.isfinite(np.r_[target,executed]).all():
            raise ValueError('Finite lift and two wrist references are required')
        self.motion_request=target.copy();self.executed_motion=executed.copy()
        self.motion_reference=executed.copy()

    @staticmethod
    def _cost(Q,q,A,b,weight):
        if weight>0:
            Q+=2.*weight*(A.T@A);q+=2.*weight*(A.T@b)

    def _prediction_invalid(self,reason,raw_prediction,residual):
        """No signed tension may be presented as unilateral pressure.

        A failed local affine contact model is not a physical impossibility
        certificate. Stop using its force/wrench forecast, retain the measured
        state, and retry from the next acquisition without inventing forces.
        """
        self.interval_action=np.zeros(5);self.interval_motion=np.zeros(3)
        self.predicted=np.full(5,np.nan)
        self.desired=(self.model.snapshot.normal_now.copy() if self.anchor is None else self.anchor.copy())
        self.force_reference=self.desired.copy();self.label='joint_local_prediction_invalid'
        self.numerical_failures+=1
        self.debug=dict(controller_architecture='contact_conditioned_direct_force_motion_v2_correctness',
            prediction_available=False,prediction_invalid_reason=reason,
            rejected_raw_pressure_prediction=np.asarray(raw_prediction,float).tolist(),
            qp_primal_residual=float(residual),support_infeasibility_certified=False,
            motion_response_identified=bool(self.model.snapshot.motion_identified),
            fallback='zero executable increment; no pressure clipping or cancellation; retry next decision')

    def _solve(self):
        s=self.state;h=self.g.horizon;period=self.g.decision_period;nr=5*h
        # u[5*h], progress[h], pressure violations[5], budget violation[1],
        # local measured-shear reserve deficits[5]. All force slacks have N units.
        dim=nr+h+11;irho=slice(nr,nr+h);ie=slice(nr+h,nr+h+5)
        ib=nr+h+5;it=slice(nr+h+6,dim)
        rates=np.asarray(self.settings.motion_rates)
        preview=np.asarray([self.executed_motion+np.clip(self.motion_request-self.executed_motion,
            -rates*period*(j+1),rates*period*(j+1)) for j in range(h)])
        dm=np.diff(np.vstack((self.executed_motion,preview)),axis=0)
        P,c,M,Mu,uncertainty=self.model.trajectory(s,h,period,dm)
        N=np.zeros((nr,dim));N[:,:nr]=P;N[:,irho]=M
        Nu=N.copy();Nu[:,irho]+=Mu
        robust_c=c+uncertainty+self.settings.forecast_margin
        if self.anchor is None:
            self.anchor=np.minimum(self.model.snapshot.normal_now,np.maximum(.2,self.limits-.15))
            self.anchor_load=float(s.load);self.anchor_up=s.up.copy()
        I=np.eye(dim);rows=[];bounds=[]
        def constrain(A,b):rows.append(np.atleast_2d(A));bounds.append(np.atleast_1d(b))
        constrain(I[:nr],np.full(nr,self.g.close_rate*period))
        constrain(-I[:nr],np.full(nr,self.g.open_rate*period))
        C=np.kron(np.tril(np.ones((h,h))),np.eye(5))
        Cu=np.zeros((nr,dim));Cu[:,:nr]=C
        constrain(Cu,np.tile(s.closure_max-s.alpha,h));constrain(-Cu,np.tile(s.alpha-s.closure_min,h))
        blocked=~s.valid|~s.closure_permission
        if np.any(blocked):constrain(I[:nr][np.tile(blocked,h)],np.zeros(np.count_nonzero(blocked)*h))
        constrain(I[irho],np.ones(h));constrain(-I[irho],np.zeros(h))
        constrain(-I[ie],np.zeros(5));constrain(-I[ib],0.)
        constrain(-I[it],np.zeros(5))
        Eu=np.tile(I[ie],(h,1));Eb=np.tile(I[ib],(h,1))
        constrain(Nu-Eu,np.tile(self.limits,h)-robust_c)
        # Contact normals are unilateral pressure. Every predicted step, not
        # merely the displayed first output, must be nonnegative before it can
        # enter a total budget or a complete-hand wrench calculation.
        constrain(-N,c)
        sums=np.kron(np.eye(h),np.ones((1,5)))
        constrain(sums@Nu-Eb,np.full(h,self.budget)-sums@robust_c)
        safe_mu=s.mu*self.g.friction_margin*np.clip(s.confidence,.35,1.)
        tau=np.maximum(0.,s.shear_magnitude)
        for j in range(h):
            for i in np.flatnonzero(s.contact&s.valid):
                a=-safe_mu[i]*N[j*5+i].copy();a[it.start+i]=-1.
                constrain(a,safe_mu[i]*c[j*5+i]-tau[i])
        A,b=np.vstack(rows),np.concatenate(bounds)
        # A checked feasible seed always exists because only task/force
        # performance, never execution authority, has explicit violation states.
        seed=np.zeros(dim)
        seed[ie]=np.maximum(0.,(robust_c.reshape(h,5)-self.limits).max(axis=0))+.001
        seed[ib]=max(0.,float((sums@robust_c-self.budget).max()))+.001
        seed[it]=np.maximum(0.,tau-safe_mu*np.min(c.reshape(h,5),axis=0))+.001
        if np.max(A@seed-b)>1e-8:
            repair=solve_qp(np.eye(dim),np.zeros(dim),A,b,self.settings.qp_passes,2e-5)
            if repair.residual>1e-8:
                self._prediction_invalid('no_checked_nonnegative_local_pressure_trajectory',c,repair.residual)
                return
            seed=repair.x.copy()
        # Stage 1: smallest predicted skin-pressure violations. No support
        # capacity sign or global friction reserve is a pause certificate.
        Q1=np.eye(dim)*1e-7;q1=np.zeros(dim)
        normalizer=np.maximum(.5,self.limits)
        self._cost(Q1,q1,I[ie]/normalizer[:,None],np.zeros(5),1.)
        self._cost(Q1,q1,I[ib:ib+1]/max(1.,self.budget),np.zeros(1),1.)
        first=solve_qp(Q1,q1,A,b,self.settings.qp_passes,2e-5,feasible_initial=seed)
        safe=first.x if first.residual<=2e-5 else seed
        # Preserve the obtained safety optimum componentwise in the task stage.
        guard=I[np.r_[np.arange(ie.start,ie.stop),ib]]
        A2=np.vstack((A,guard));b2=np.r_[b,guard@safe+self.settings.pressure_stage_tolerance]
        Q=np.eye(dim)*1e-5;q=np.zeros(dim)
        scale=np.tile(1./np.maximum(.5,self.limits),h)
        self._cost(Q,q,scale[:,None]*N,scale*(c-np.tile(self.anchor,h)),self.settings.force_anchor_weight)
        D=np.kron(np.eye(h)-np.eye(h,k=-1),np.eye(5))
        previous=np.r_[self.model.snapshot.normal_now,np.zeros(nr-5)]
        self._cost(Q,q,scale[:,None]*(D@N),scale*(D@c-previous),self.settings.force_variation_weight)
        # Local measured friction demand is a task deficit, NOT a rigid-body
        # support certificate. Palm and all non-normal wrench remain explicit.
        self._cost(Q,q,I[it]/normalizer[:,None],np.zeros(5),self.settings.contact_reserve_weight)
        W=s.normal_wrench_map.copy();W[3:]/=self.g.characteristic_length
        non=s.finger_non_normal_by_finger.sum(axis=0).copy()
        aux=s.auxiliary_wrench.copy() if s.auxiliary_valid else np.zeros(6)
        extra=non+aux;extra[3:]/=self.g.characteristic_length
        current_wrench=W@self.model.snapshot.normal_now+extra
        if self.wrench_anchor is None:self.wrench_anchor=current_wrench.copy()
        target=self.wrench_anchor.copy()
        target[:3]+=s.up*(float(s.load)-self.anchor_load)
        GW=np.kron(np.eye(h),W)
        wrench_scale=1./max(1.,abs(self.anchor_load))
        auxiliary_confidence=1./(1.+min(10.,s.auxiliary_support_uncertainty)) if s.auxiliary_valid else .1
        self._cost(Q,q,wrench_scale*(GW@N),wrench_scale*(GW@c+np.tile(extra-target,h)),
            self.settings.wrench_weight*auxiliary_confidence)
        self._cost(Q,q,I[:nr]*np.tile(s.force_gain,h)[:,None],np.zeros(nr),self.settings.action_weight)
        self._cost(Q,q,I[irho],-np.ones(h),self.settings.progress_weight)
        # Measured instability changes progress continuously; it cannot create
        # the old negative-capacity / permanent Boolean stop.
        risk=min(20.,max(0.,s.slip_speed-.004)/.004)+2.*(1.-float(np.mean(s.valid)))
        self._cost(Q,q,I[irho],np.zeros(h),risk)
        second=solve_qp(Q,q,A2,b2,self.settings.qp_passes,2e-5,feasible_initial=safe)
        result=second.x if second.residual<=2e-5 else safe
        raw_prediction=N@result+c
        if np.min(raw_prediction)<-1e-8:
            self._prediction_invalid('negative_pressure_after_solver_check',raw_prediction,
                                     max(float(second.residual),float(-np.min(raw_prediction))))
            return
        if not first.converged or not second.converged:self.numerical_failures+=1
        self.previous_plan=result.copy()
        self.interval_action=result[:5].copy()
        self.interval_motion=dm[0]*result[nr]
        self.previous_action=self.interval_action.copy()
        self.predicted=raw_prediction[:5].copy()
        self.desired=self.anchor.copy();self.force_reference=self.desired.copy()
        self.label='joint_force_motion_tracking' if risk<1. else 'joint_measured_contact_recovery'
        self.debug=dict(controller_architecture='contact_conditioned_direct_force_motion_v2_correctness',
            qp_status=second.status,qp_optimal=bool(second.converged),qp_primal_residual=float(second.residual),
            pressure_priority_status=first.status,predicted_pressure_violation_N=result[ie].tolist(),
            predicted_budget_violation_N=float(result[ib]),measured_shear_reserve_deficit_N=result[it].tolist(),
            joint_progress=result[irho].tolist(),motion_request=self.motion_request.tolist(),
            interval_motion=self.interval_motion.tolist(),slow_drift_Nps=self.model.snapshot.drift_rate.tolist(),
            slow_drift_error_Nps=self.model.snapshot.drift_error_rate.tolist(),slow_fit_samples=self.model.snapshot.fit_samples.tolist(),
            pressure_now=self.model.snapshot.normal_now.tolist(),force_target=self.desired.tolist(),
            force_prediction=self.predicted.tolist(),estimated_normal=s.normal.tolist(),
            observed_complete_hand_wrench=current_wrench.tolist(),passive_auxiliary_wrench=aux.tolist(),
            auxiliary_valid=bool(s.auxiliary_valid),estimated_load=float(s.load),
            motion_response_identified=bool(self.model.snapshot.motion_identified),
            motion_bound_kind='engineering development envelope, not certified physical uncertainty',
            support_infeasibility_certified=False,negative_finger_capacity_is_not_pause=True,
            predicted_pressure_min_N=float(np.min(raw_prediction)),pressure_nonnegative_tolerance_N=1e-8,
            execution_domain_lower=s.closure_min.tolist(),execution_domain_upper=s.closure_max.tolist(),
            settings=asdict(self.settings),release_scope='inherited common mechanical path, not redesigned')

    def step(self,frame,phase=None,command='hold'):
        dt=.005 if self.previous_time is None else max(.0001,min(.05,float(frame.time)-self.previous_time))
        self.previous_time=float(frame.time);self.state.update(frame,dt);self.control_ticks+=1
        self.model.update(frame,self.state,self.executed_motion)
        if command=='release' or str(phase or frame.phase).upper() in ('RELEASE','RELEASING'):
            self.label='joint_common_release';self.desired=np.zeros(5);self.predicted=np.full(5,np.nan)
            self.motion_reference=self.executed_motion.copy()
            delta=-np.minimum(np.maximum(0.,self.state.alpha-self.state.closure_min),self.g.release_rate*dt)
            return ControlAction(delta,self.predicted.copy(),self.label,False,dict(release_authority=True,
                release_scope='common inherited mechanical execution',prediction_available=False))
        if frame.time-self.last_decision>=self.g.decision_period-1e-8:
            if np.count_nonzero(self.state.valid):self._solve()
            else:
                self.interval_action=np.zeros(5);self.interval_motion=np.zeros(3)
                self.label='joint_observation_unavailable';self.debug=dict(observation_valid=False)
            self.last_decision=float(frame.time)
        fraction=min(1.,dt/self.g.decision_period)
        delta=np.clip(self.interval_action*fraction,self.state.closure_min-self.state.alpha,
                      self.state.closure_max-self.state.alpha)
        blocked=~self.state.valid|~self.state.closure_permission
        delta[blocked]=np.minimum(0.,delta[blocked])
        self.motion_reference=self.executed_motion+self.interval_motion*fraction
        direction=self.motion_request-self.executed_motion
        self.motion_reference=np.where(direction>=0.,np.minimum(self.motion_reference,self.motion_request),
                                      np.maximum(self.motion_reference,self.motion_request))
        self.debug['executed_motion_reference']=self.motion_reference.tolist()
        return ControlAction(delta,self.predicted.copy(),self.label,False,self.debug.copy())


def create_joint(calibration,limits,budget,gains=None):
    return JointCoordinator(calibration,limits,budget,gains)


def self_test():
    from types import SimpleNamespace
    from force_model import ForceResponse
    checks=verify_model_algebra()
    B=np.eye(5)*10.;cal=dict(force_matrix=B,force_tau=.02,reference_alpha=np.zeros(5),response_error_bound=np.full(5,.02))
    c=JointCoordinator(cal,np.full(5,5.),18.)
    state=SimpleNamespace(normal=np.full(5,2.),alpha=np.zeros(5),latent_alpha=np.zeros(5),closure_min=np.full(5,-.02),closure_max=np.full(5,.02),
        valid=np.ones(5,bool),closure_permission=np.ones(5,bool),contact_model_active=np.ones(5,bool),
        contact=np.ones(5,bool),mu=np.full(5,.45),confidence=np.ones(5),shear_magnitude=np.full(5,.05),
        normal_wrench_map=np.r_[np.tile([[0.],[0.],[-1.]],(1,5)),np.zeros((3,5))],
        finger_non_normal_by_finger=np.zeros((5,6)),auxiliary_wrench=np.array([0.,0.,11.2,0.,0.,0.]),
        auxiliary_valid=True,auxiliary_support_uncertainty=.2,load=1.2,up=np.array([0.,0.,1.]),
        force_gain=np.full(5,10.),slip_speed=0.,force_model=ForceResponse(B,.02))
    from coordinator_model import PressureForecast
    c.state=state;c.model.snapshot=PressureForecast(np.full(5,2.),np.zeros(5),np.full(5,.03),np.full(5,.01),
        np.full(5,50),np.zeros((5,3)),np.tile([15.,3.,3.],(5,1)),False)
    c.set_motion_request([.09,.02,0.],[.10,0.,0.]);c._solve()
    checks['negative_finger_projection_does_not_stop_progress']=bool(c.debug['joint_progress'][0]>.9)
    checks['planned_motion_advances_actual_request']=bool(c.interval_motion[0]<0 and c.interval_motion[1]>0)
    checks['finite_checked_joint_qp']=bool(np.isfinite(c.previous_plan).all() and c.debug['qp_primal_residual']<=2e-5)
    checks['action_stays_in_real_domain']=bool(np.all(c.interval_action<=.02+1e-8) and np.all(c.interval_action>=-.02-1e-8))
    checks['not_claiming_identified_motion']=c.debug['motion_response_identified'] is False
    # Public sphere uncertainty must survive even when it exceeds 0.40 N.
    wider={**cal,'response_error_bound':np.full(5,.508)}
    wide=JointCoordinator(wider,np.full(5,5.),18.);wide.state=state
    wide.model.snapshot=c.model.snapshot
    *_,envelope=wide.model.trajectory(state,3,.02,np.zeros((3,3)))
    checks['full_public_error_not_truncated']=bool(np.min(envelope)>=.508)
    fake=np.array([-1.5,4.75,4.75,4.75,4.75])
    checks['negative_pressure_budget_counterexample_detected']=bool(
        (fake+.05).sum()<18. and np.maximum(fake,0.).sum()>18. and np.min(fake)<0.)
    positive=JointCoordinator(cal,np.full(5,5.),18.);positive.state=state
    positive.model.snapshot=PressureForecast(np.array([.01,4.75,4.75,4.75,4.75]),
        np.zeros(5),np.full(5,.03),np.full(5,.01),np.full(5,50),
        np.zeros((5,3)),np.tile([15.,3.,3.],(5,1)),False)
    positive._solve()
    checks['every_horizon_pressure_is_nonnegative']=bool(positive.debug.get('predicted_pressure_min_N',-1.)>=-1e-8)
    # A zero-authority contact model forecasting tension has no valid pressure
    # trajectory. It must be rejected, not clipped into an apparent solution.
    dead=JointCoordinator({**cal,'force_matrix':np.zeros((5,5))},np.full(5,5.),18.)
    dead.state=SimpleNamespace(**vars(state));dead.state.force_model=ForceResponse(np.zeros((5,5)),.02)
    dead.model.snapshot=PressureForecast(np.full(5,.01),np.full(5,-5.),np.full(5,.03),np.full(5,.01),
        np.full(5,50),np.zeros((5,3)),np.tile([15.,3.,3.],(5,1)),False)
    dead._solve()
    checks['unreachable_nonnegative_model_not_hidden_by_output_clip']=bool(
        np.isnan(dead.predicted).all() and dead.debug.get('prediction_available') is False and
        dead.debug.get('support_infeasibility_certified') is False)
    if not all(checks.values()):raise AssertionError(checks)
    return dict(passed=True,checks=checks,scope='pure algebra/synthetic contact state; no dynamics or real performance')


if __name__=='__main__':
    import json
    print(json.dumps(self_test(),ensure_ascii=False))

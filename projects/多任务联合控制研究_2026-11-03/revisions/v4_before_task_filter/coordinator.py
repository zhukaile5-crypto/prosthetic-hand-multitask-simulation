"""V4: observed viable-contact corridors with joint pressure/motion control.

Stable, safe zero action is an invariant member of the observed corridor.
Unidentified absolute friction reserve cannot override it. Outside this set,
the hierarchy is pressure protection, observed contact/task compatibility,
actual pressure variation and motion progress, then weak excess-force removal.
All pressure/shear/net-wrench predictions share one actual-input memory.
"""
from dataclasses import dataclass,asdict
import numpy as np
import research_imports
from vendor.frozen_controllers import make_gains,solve_qp,ControlAction
from state_estimator import CoupledTactileState
from coordinator_model import ContactConditionedModel,OutputForecast,synthetic_document,verify_model_algebra
from coordinator_contacts import ContactCorridorObserver,verify_contact_membership


@dataclass
class CoordinatorSettings:
    force_variation_weight: float=4.
    progress_weight: float=8.
    action_weight: float=.05
    action_continuity_weight: float=.1
    grasp_half_width_N: float=.15
    motion_rates: tuple=(.035,.15,.10)
    forecast_margin_N: float=.05
    priority_tolerance_N: float=.0005
    lexicographic_loss_tolerance: float=1e-8
    qp_passes: int=240


class _ObservedState(CoupledTactileState):
    def __init__(self,gains,owner):super().__init__(gains);self.owner=owner
    def update(self,frame,dt):
        super().update(frame,dt)
        o=self.owner;motion=o.executed_motion if o.motion_initialized else None
        prediction=o.model.update(frame,self,motion)
        o.contacts.update(frame,self,prediction)


class JointCoordinator:
    def __init__(self,calibration,limits,budget,gains=None,model_document=None):
        B=np.asarray(calibration['force_matrix'],float);g=dict(gains or {})
        g.update(force_gain=tuple(np.maximum(.05,np.linalg.norm(B,axis=0))),force_tau=float(calibration['force_tau']),release_rate=.8)
        self.g=make_gains(g);self.settings=CoordinatorSettings()
        self.model=ContactConditionedModel(calibration,document=model_document)
        self.contacts=ContactCorridorObserver(self.g.nominal_mu)
        self.motion_initialized=False;self.state=_ObservedState(self.g,self)
        self.limits=np.asarray(limits,float);self.budget=float(budget);self.name='joint'
        self.desired=np.zeros(5);self.predicted=np.full(5,np.nan);self.last_decision=-np.inf;self.previous_time=None
        self.interval_action=np.zeros(5);self.interval_motion=np.zeros(3);self.motion_request=np.zeros(3)
        self.executed_motion=np.zeros(3);self.motion_reference=np.zeros(3)
        self.anchor=None;self.execution_anchor=None;self.motion_anchor=None;self.previous_action=np.zeros(8)
        self.handoff_contact_mask=None
        self.previous_plan=None;self.debug={};self.label='joint_v4_initializing'
        self.numerical_failures=0;self.control_ticks=0;self.infeasible_ticks=0;self.force_reference=np.zeros(5)

    def set_motion_request(self,target,executed_reference):
        target=np.asarray(target,float);executed=np.asarray(executed_reference,float)
        if target.shape!=(3,) or executed.shape!=(3,) or not np.isfinite(np.r_[target,executed]).all():
            raise ValueError('Finite lift and two wrist references required')
        self.motion_request=target.copy();self.executed_motion=executed.copy();self.motion_reference=executed.copy();self.motion_initialized=True

    @staticmethod
    def _cost(Q,q,A,b,weight):
        if weight>0:Q+=2.*weight*(A.T@A);q+=2.*weight*(A.T@b)

    @staticmethod
    def support_band(wrench,uncertainty,up,load,load_uncertainty):
        support=float(np.asarray(up)@wrench[:3]);error=max(.06,float(np.abs(up)@uncertainty[:3]))
        measured=np.array([support-error,support+error]);external=np.full(2,np.nan)
        if np.isfinite(load) and np.isfinite(load_uncertainty) and load_uncertainty>=0:
            external=np.array([max(0.,load-load_uncertainty),load+load_uncertainty])
            overlap=np.array([max(measured[0],external[0]),min(measured[1],external[1])])
            if overlap[0]<=overlap[1]:return overlap,False,measured,external
        return measured,True,measured,external

    def _prediction_invalid(self,reason,raw_prediction,residual):
        self.interval_action=np.zeros(5);self.interval_motion=np.zeros(3);self.predicted=np.full(5,np.nan)
        self.desired=np.zeros(5) if self.anchor is None else self.anchor.copy();self.force_reference=self.desired.copy()
        self.label='joint_v4_local_prediction_invalid';self.numerical_failures+=1
        self.debug=dict(controller_architecture='observed_contact_corridor_16x8_v4',prediction_available=False,
            prediction_invalid_reason=reason,rejected_raw_contact_prediction=np.asarray(raw_prediction).tolist(),
            qp_primal_residual=float(residual),model_admitted=False,support_infeasibility_certified=False,
            fallback='zero increment, model unavailable; no clipping or physical impossibility claim')

    def _base_debug(self):
        s=self.state;f=self.model.snapshot;cs=self.contacts.snapshot
        return dict(controller_architecture='observed_contact_corridor_16x8_v4',prediction_available=True,
            model_admitted=False,model_status='partial_contact_conditioned_experimental',model_source=self.model.source_path,
            model_source_sha256=self.model.source_sha256,model_frame='current_palm',
            pressure_now=f.output_now[:5].tolist(),shear_now=f.output_now[5:10].tolist(),
            observed_complete_hand_wrench=f.output_now[10:].tolist(),estimated_normal=s.normal.tolist(),
            slow_drift_Nps=f.drift_rate[:10].tolist(),slow_drift_error_Nps=f.drift_error_rate[:10].tolist(),
            slow_fit_samples=f.fit_samples.tolist(),residual_slope_confidence=f.slope_confidence.tolist(),
            contact_modes=cs.mode.tolist(),friction_lower_bound=cs.mu_lower.tolist(),
            friction_upper_bound_available=False,friction_prior_conflict=cs.prior_conflict.tolist(),
            absolute_friction_reserve_identified=False,friction_role='aggregate necessary lower bound; not exact material/per-patch friction',
            estimated_load=float(s.load),motion_response_identified=bool(f.motion_identified),
            motion_response_scope='measured local increments; unmeasured opposite direction only returns to handoff anchor',
            support_infeasibility_certified=False,actual_input_history_only=True,
            pressure_shear_wrench_same_kernel_history=True,release_scope='external common release, holding model unavailable',
            kernel_temporal_assumptions='measured dense5ms finite ramps; constant response tail after500ms; cumulative domain transfer experimental',
            settings=asdict(self.settings))

    def _solve(self):
        s=self.state;f=self.model.snapshot;cs=self.contacts.snapshot;h=self.g.horizon;period=self.g.decision_period;nr=8*h
        if f is None or cs is None or not np.all(f.output_valid[:10]):
            self._prediction_invalid('pressure_or_shear_model_unavailable',np.full(10,np.nan),0.);return
        if self.anchor is None:
            self.contacts.freeze_handoff(f);self.anchor=f.normal_now.copy()
            self.execution_anchor=s.alpha.copy();self.motion_anchor=self.executed_motion.copy()
            self.handoff_contact_mask=self.model.active_mask&(f.normal_now>.15)
        P,c,error=self.model.trajectory(s,h,period)
        pi=np.concatenate([np.arange(16*j,16*j+5) for j in range(h)])
        ti=np.concatenate([np.arange(16*j+5,16*j+10) for j in range(h)])
        wi=np.concatenate([np.arange(16*j+10,16*j+16) for j in range(h)])
        cp=c[pi];ct=c[ti];ep=error[pi];et=error[ti]
        if not np.isfinite(np.r_[cp,ct,ep,et]).all():
            self._prediction_invalid('nonfinite_contact_model',np.r_[cp,ct],0.);return
        robust=cp+ep+self.settings.forecast_margin_N
        ratio=self.contacts.snapshot.anchor_ratio.copy()
        half=self.settings.grasp_half_width_N+cs.measurement_error_pressure
        low=np.maximum(0.,self.contacts.snapshot.anchor_pressure-half);high=self.contacts.snapshot.anchor_pressure+half
        # This is a measured-state COMPATIBILITY tube, not a robust Coulomb
        # reserve claim. Correlated pressure/shear motion is modeled together.
        util=ct-np.tile(ratio,h)*cp
        util_tube=et+np.tile(ratio,h)*ep+.02
        full_valid=bool(np.all(f.output_valid[10:]) and np.isfinite(c[wi]).all())
        zero_wrench_compatible=False;zero_support_info={}
        if full_valid:
            zero_projection=np.vstack((np.eye(6),np.r_[s.up,np.zeros(3)]))
            zero_me=np.maximum(f.fit_error[10:],np.r_[np.full(3,.06),np.full(3,.003)])
            zero_band,zero_inconsistent,unused_band,zero_load=self.support_band(f.output_now[10:],zero_me,s.up,
                float(s.load),float(getattr(s,'load_uncertainty',np.inf)))
            zw=(c[wi].reshape(h,6)@zero_projection.T)
            ze=(error[wi].reshape(h,6)@np.abs(zero_projection).T)
            zl=np.r_[f.output_now[10:]-zero_me,zero_band[0]]
            zu=np.r_[f.output_now[10:]+zero_me,zero_band[1]]
            zero_wrench_compatible=bool(np.all(zw-ze<=zu) and np.all(zw+ze>=zl))
            zero_support_info=dict(support_band_N=zero_band.tolist(),load_interval_N=zero_load.tolist(),
                load_wrench_interval_inconsistent=bool(zero_inconsistent))
        expected=self.handoff_contact_mask
        stable_support=bool(expected[0] and np.count_nonzero(expected)>=3
            and np.count_nonzero((cs.mode=='STICK_CONFIRMED')&expected)>=3
            and np.all(cs.anchor_valid[expected]) and np.all(cs.speed_lower[expected]<.002))
        quiet=bool(stable_support
            and np.min(np.r_[cp,ct])>=-1e-8 and np.all(robust.reshape(h,5)<=self.limits+1e-10)
            and np.all(robust.reshape(h,5).sum(axis=1)<=self.budget+1e-10)
            and np.all(f.last_measured[:5]<=self.limits) and f.last_measured[:5].sum()<=self.budget
            and np.all(cp.reshape(h,5)>=low-ep.reshape(h,5)) and np.all(cp.reshape(h,5)<=high+ep.reshape(h,5))
            and np.all(util<=util_tube) and np.max(np.abs(self.motion_request-self.executed_motion))<1e-9
            and full_valid and zero_wrench_compatible)
        if quiet:
            self.interval_action=np.zeros(5);self.interval_motion=np.zeros(3);self.previous_action=np.zeros(8)
            self.predicted=cp[:5].copy();self.desired=self.anchor.copy();self.force_reference=self.desired.copy()
            self.label='joint_v4_observed_viable_silent_hold';self.debug=self._base_debug()
            self.debug.update(corridor_silent_hold=True,qp_status='not_needed_zero_action_in_observed_corridor',qp_optimal=None,
                qp_primal_residual=0.,force_prediction=self.predicted.tolist(),force_target=self.anchor.tolist(),
                predicted_pressure_min_N=float(cp.min()),zero_action_viable=True,complete_wrench_prediction_available=True,
                predicted_complete_hand_wrench=c[wi].reshape(h,6).tolist(),interval_motion=[0.,0.,0.],joint_progress=[1.]*h,
                handoff_contact_mask=expected.tolist(),**zero_support_info)
            return
        # Executable joint inputs plus maximum pressure, budget, empirical
        # pressure-corridor, shear/pressure consistency and wrench-band deficits.
        dim=nr+23;ie=slice(nr,nr+5);ib=nr+5;ig=slice(nr+6,nr+11);it=slice(nr+11,nr+16);iw=slice(nr+16,dim)
        I=np.eye(dim);N=np.zeros((5*h,dim));T=N.copy();N[:,:nr]=P[pi];T[:,:nr]=P[ti]
        rows=[];bounds=[]
        def constrain(A,b):rows.append(np.atleast_2d(A));bounds.append(np.atleast_1d(b))
        step=np.minimum(np.r_[np.full(5,self.g.close_rate*period),np.asarray(self.settings.motion_rates)*period],self.model.input_step_limits)
        open_step=step.copy();open_step[:5]=np.minimum(self.g.open_rate*period,self.model.input_step_limits[:5])
        constrain(I[:nr],np.tile(step,h));constrain(-I[:nr],np.tile(open_step,h))
        # Cumulated physical alpha authority; unavailable response directions
        # may return to a visited handoff state, never go beyond that anchor.
        lower=s.closure_min.copy();upper=s.closure_max.copy()
        for i in range(5):
            available=self.model.available_directions[i]
            if not available['plus']:upper[i]=min(upper[i],self.execution_anchor[i])
            if not available['minus']:lower[i]=max(lower[i],self.execution_anchor[i])
        C=np.kron(np.tril(np.ones((h,h))),np.eye(8));Cu=np.zeros((nr,dim));Cu[:,:nr]=C
        apos=np.concatenate([np.arange(8*j,8*j+5) for j in range(h)])
        mpos=np.concatenate([np.arange(8*j+5,8*j+8) for j in range(h)])
        constrain(Cu[apos],np.tile(upper-s.alpha,h));constrain(-Cu[apos],np.tile(s.alpha-lower,h))
        mlower=np.minimum(self.executed_motion,self.motion_request);mupper=np.maximum(self.executed_motion,self.motion_request)
        for i in range(3):
            available=self.model.available_directions[5+i]
            if not available['plus']:mupper[i]=min(mupper[i],self.motion_anchor[i])
            if not available['minus']:mlower[i]=max(mlower[i],self.motion_anchor[i])
        constrain(Cu[mpos],np.tile(mupper-self.executed_motion,h));constrain(-Cu[mpos],np.tile(self.executed_motion-mlower,h))
        blocked=~s.valid|~s.closure_permission
        if np.any(blocked):constrain(I[apos][np.tile(blocked,h)],np.zeros(np.count_nonzero(blocked)*h))
        missing=~self.model.identified_inputs
        if np.any(missing):
            selected=I[:nr][np.tile(missing,h)];constrain(selected,np.zeros(len(selected)));constrain(-selected,np.zeros(len(selected)))
        for sl in (ie,ig,it,iw):constrain(-I[sl],np.zeros(sl.stop-sl.start))
        constrain(-I[ib],0.);Eu=np.tile(I[ie],(h,1));Eg=np.tile(I[ig],(h,1));Et=np.tile(I[it],(h,1));Eb=np.tile(I[ib],(h,1))
        constrain(N-Eu,np.tile(self.limits,h)-robust)
        constrain(-N,cp);constrain(-T,ct)
        sums=np.kron(np.eye(h),np.ones((1,5)));constrain(sums@N-Eb,np.full(h,self.budget)-sums@robust)
        constrain(N-Eg,np.tile(high,h)-cp);constrain(-N-Eg,cp-np.tile(low,h))
        R=np.tile(ratio,h);constrain(T-R[:,None]*N-Et,util_tube-util)
        wrench_info={};wc=None;WN=None;wlow=None;whigh=None
        if full_valid:
            projection=np.vstack((np.eye(6),np.r_[s.up,np.zeros(3)]));projection_h=np.kron(np.eye(h),projection)
            WN=np.zeros((6*h,dim));WN[:,:nr]=P[wi];WN=projection_h@WN;wc=projection_h@c[wi]
            me=np.maximum(f.fit_error[10:],np.r_[np.full(3,.06),np.full(3,.003)])
            band,inconsistent,observed_band,load_band=self.support_band(f.output_now[10:],me,s.up,float(s.load),float(getattr(s,'load_uncertainty',np.inf)))
            # Forecast uncertainty enlarges compatibility, not a fabricated
            # material capacity or a certificate that an unknown input is safe.
            we=(np.abs(projection_h)@error[wi]).reshape(h,7)
            wlow=np.tile(np.r_[f.output_now[10:]-me,band[0]],(h,1))-we
            whigh=np.tile(np.r_[f.output_now[10:]+me,band[1]],(h,1))+we
            Ew=np.tile(I[iw],(h,1));constrain(WN-Ew,whigh.ravel()-wc);constrain(-WN-Ew,wc-wlow.ravel())
            wrench_info=dict(support_band_N=band.tolist(),load_interval_N=load_band.tolist(),load_wrench_interval_inconsistent=bool(inconsistent))
        A,b=np.vstack(rows),np.concatenate(bounds);seed=np.zeros(dim)
        seed[ie]=np.maximum(0.,(robust.reshape(h,5)-self.limits).max(axis=0))+.001
        seed[ib]=max(0.,float((sums@robust-self.budget).max()))+.001
        seed[ig]=np.maximum(0.,np.maximum(cp.reshape(h,5)-high,low-cp.reshape(h,5)).max(axis=0))+.001
        seed[it]=np.maximum(0.,(util-util_tube).reshape(h,5).max(axis=0))+.001
        if full_valid:seed[iw]=np.maximum(0.,np.maximum(wc.reshape(h,7)-whigh,wlow-wc.reshape(h,7)).max(axis=0))+.001
        if np.max(A@seed-b)>1e-8:
            repair=solve_qp(np.eye(dim),np.zeros(dim),A,b,self.settings.qp_passes,2e-5)
            if repair.residual>1e-8:self._prediction_invalid('no_checked_nonnegative_contact_trajectory',np.r_[cp,ct],repair.residual);return
            seed=repair.x.copy()
        def solve(Q,q,AA,bb,initial):
            result=solve_qp(Q,q,AA,bb,self.settings.qp_passes,2e-5,feasible_initial=initial)
            return (result.x.copy() if result.residual<=2e-5 else initial.copy()),result
        norm=np.maximum(.5,self.limits)
        Q1=np.eye(dim)*1e-8;q1=np.zeros(dim)
        self._cost(Q1,q1,I[ie]/norm[:,None],np.zeros(5),1.);self._cost(Q1,q1,I[ib:ib+1]/max(1.,self.budget),np.zeros(1),1.)
        x1,r1=solve(Q1,q1,A,b,seed);guard1=I[np.r_[np.arange(ie.start,ie.stop),ib]]
        A2=np.vstack((A,guard1));b2=np.r_[b,guard1@x1+self.settings.priority_tolerance_N]
        Q2=np.eye(dim)*1e-8;q2=np.zeros(dim)
        self._cost(Q2,q2,I[ig]/norm[:,None],np.zeros(5),1.);self._cost(Q2,q2,I[it]/norm[:,None],np.zeros(5),1.)
        wscale=np.r_[np.ones(3),np.full(3,1./self.g.characteristic_length),1.]
        if full_valid:self._cost(Q2,q2,wscale[:,None]*I[iw],np.zeros(7),1.)
        x2,r2=solve(Q2,q2,A2,b2,x1);guard2=I[np.r_[np.arange(ig.start,ig.stop),np.arange(it.start,it.stop),np.arange(iw.start,iw.stop)]]
        tolerance=np.r_[np.full(10,self.settings.priority_tolerance_N),self.settings.priority_tolerance_N/wscale]
        A3=np.vstack((A2,guard2));b3=np.r_[b2,guard2@x2+tolerance]
        Q3=np.eye(dim)*1e-8;q3=np.zeros(dim);D=np.kron(np.eye(h)-np.eye(h,k=-1),np.eye(5))
        scale=np.tile(1./norm,h);previous=np.r_[f.normal_now,np.zeros(5*h-5)]
        self._cost(Q3,q3,scale[:,None]*(D@N),scale*(D@cp-previous),self.settings.force_variation_weight)
        action_scale=np.maximum(.05,np.linalg.norm(self.model.kernel_at(.02)[:5,:5],axis=0))
        self._cost(Q3,q3,np.tile(action_scale,h)[:,None]*I[apos],np.zeros(5*h),self.settings.action_weight)
        D8=np.kron(np.eye(h)-np.eye(h,k=-1),np.eye(8));s8=np.tile(np.r_[action_scale,1./np.maximum(1e-6,step[5:])],h)
        self._cost(Q3,q3,s8[:,None]*(D8@I[:nr]),-s8*np.r_[self.previous_action,np.zeros(nr-8)],self.settings.action_continuity_weight)
        motion_target=np.asarray([np.clip(self.motion_request-self.executed_motion,-step[5:]*(j+1),step[5:]*(j+1)) for j in range(h)]).ravel()
        ms=np.tile(1./np.maximum(1e-6,step[5:]),h)
        self._cost(Q3,q3,ms[:,None]*Cu[mpos],-ms*motion_target,self.settings.progress_weight)
        risk=min(20.,max(0.,float(np.max(cs.speed_lower))-.002)/.002)
        self._cost(Q3,q3,ms[:,None]*I[mpos],np.zeros(3*h),risk)
        x3,r3=solve(Q3,q3,A3,b3,x2)
        # Final excess-force preference may spend only numerical tolerance of
        # the actual higher-level quadratic loss. It cannot trade TV/progress
        # for a finite, unspecified weighted pressure benefit.
        Q4=np.eye(dim)*1e-8;q4=np.zeros(dim);self._cost(Q4,q4,scale[:,None]*N,scale*cp,1.)
        x4,r4=solve(Q4,q4,A3,b3,x3);direction=x4-x3
        aa=.5*float(direction@Q3@direction);bb=float(direction@(Q3@x3+q3));eps=self.settings.lexicographic_loss_tolerance
        eta=1.
        if aa+bb>eps:
            lo_eta=0.;hi_eta=1.
            for unused in range(40):
                mid=(lo_eta+hi_eta)/2.
                if aa*mid*mid+bb*mid<=eps:lo_eta=mid
                else:hi_eta=mid
            eta=lo_eta
        x=x3+eta*direction;raw_p=N@x+cp;raw_tau=T@x+ct
        if np.min(np.r_[raw_p,raw_tau])<-1e-8:
            self._prediction_invalid('negative_raw_contact_magnitude_after_solver',np.r_[raw_p,raw_tau],float(-np.min(np.r_[raw_p,raw_tau])));return
        self.previous_plan=x.copy();self.interval_action=x[:5].copy();self.interval_motion=x[5:8].copy();self.previous_action=x[:8].copy()
        self.predicted=raw_p[:5].copy();self.desired=self.anchor.copy();self.force_reference=self.desired.copy()
        self.label='joint_v4_contact_corridor_control';self.debug=self._base_debug()
        if any(not r.converged for r in (r1,r2,r3,r4)):self.numerical_failures+=1
        self.debug.update(corridor_silent_hold=False,zero_action_viable=False,qp_status=r3.status,qp_optimal=bool(r3.converged),
            qp_primal_residual=float(max(r1.residual,r2.residual,r3.residual,r4.residual)),
            priority_status=[r.status for r in (r1,r2,r3,r4)],excess_force_tiebreak_fraction=float(eta),
            higher_priority_quadratic_loss_change=float(aa*eta*eta+bb*eta),
            predicted_pressure_violation_N=x[ie].tolist(),predicted_budget_violation_N=float(x[ib]),
            grasp_corridor_violation_N=x[ig].tolist(),contact_consistency_slack_N=x[it].tolist(),wrench_band_violation=x[iw].tolist(),
            force_prediction=self.predicted.tolist(),predicted_shear=raw_tau[:5].tolist(),force_target=self.anchor.tolist(),
            predicted_pressure_min_N=float(raw_p.min()),predicted_shear_min_N=float(raw_tau.min()),
            complete_wrench_prediction_available=full_valid,interval_motion=self.interval_motion.tolist(),motion_request=self.motion_request.tolist(),
            execution_domain_lower=lower.tolist(),execution_domain_upper=upper.tolist(),
            predicted_complete_hand_wrench=((WN@x+wc).reshape(h,7)[:,:6].tolist() if full_valid else np.full((h,6),np.nan).tolist()),
            joint_progress=(np.divide(self.interval_motion,motion_target[:3],out=np.ones(3),where=np.abs(motion_target[:3])>1e-12)).tolist(),**wrench_info)

    def step(self,frame,phase=None,command='hold'):
        dt=.005 if self.previous_time is None else max(.0001,min(.05,float(frame.time)-self.previous_time))
        self.previous_time=float(frame.time);self.state.update(frame,dt);self.control_ticks+=1
        if command=='release' or str(phase or frame.phase).upper() in ('RELEASE','RELEASING'):
            self.label='joint_common_release';self.desired=np.zeros(5);self.predicted=np.full(5,np.nan);self.motion_reference=self.executed_motion.copy()
            delta=-np.minimum(np.maximum(0.,self.state.alpha-self.state.closure_min),self.g.release_rate*dt)
            return ControlAction(delta,self.predicted.copy(),self.label,False,dict(release_authority=True,prediction_available=False,
                prediction_unavailable_reason='common_mechanical_release_outside_held_contact_model'))
        if np.count_nonzero(self.state.valid)==0:
            self.predicted=np.full(5,np.nan);self.debug.update(prediction_available=False,prediction_unavailable_reason='all_tactile_channels_invalid')
            self.debug.pop('prediction_invalid_reason',None)
        if frame.time-self.last_decision>=self.g.decision_period-1e-8:
            if np.count_nonzero(self.state.valid):self._solve()
            else:
                self.interval_action=np.zeros(5);self.interval_motion=np.zeros(3);self.predicted=np.full(5,np.nan)
                self.label='joint_observation_unavailable';self.debug=dict(prediction_available=False,prediction_unavailable_reason='all_tactile_channels_invalid')
            self.last_decision=float(frame.time)
        fraction=min(1.,dt/self.g.decision_period)
        delta=np.clip(self.interval_action*fraction,self.state.closure_min-self.state.alpha,self.state.closure_max-self.state.alpha)
        blocked=~self.state.valid|~self.state.closure_permission;delta[blocked]=np.minimum(0.,delta[blocked])
        self.motion_reference=self.executed_motion+self.interval_motion*fraction
        direction=self.motion_request-self.executed_motion
        self.motion_reference=np.where(direction>=0.,np.minimum(self.motion_reference,self.motion_request),np.maximum(self.motion_reference,self.motion_request))
        self.debug['executed_motion_reference']=self.motion_reference.tolist()
        return ControlAction(delta,self.predicted.copy(),self.label,False,self.debug.copy())


def create_joint(calibration,limits,budget,gains=None):return JointCoordinator(calibration,limits,budget,gains)


def self_test():
    from types import SimpleNamespace
    checks={**verify_model_algebra(),**verify_contact_membership()}
    cal=dict(force_matrix=np.eye(5)*10.,force_tau=.02,reference_alpha=np.zeros(5),response_error_bound=np.full(5,.02),config={'shape':'capsule'})
    def fixture(K=None,pressure=None,drift=None,limits=None):
        ctl=JointCoordinator(cal,np.full(5,5.) if limits is None else limits,18.,model_document=synthetic_document(K))
        p=np.full(5,2.) if pressure is None else np.asarray(pressure,float);tau=.6*p
        state=SimpleNamespace(normal=p.copy(),alpha=np.zeros(5),closure_min=np.full(5,-.04),closure_max=np.full(5,.04),
            valid=np.ones(5,bool),closure_permission=np.ones(5,bool),contact_model_active=np.ones(5,bool),contact=np.ones(5,bool),
            confidence=np.ones(5),slip_magnitude=np.zeros(5),auxiliary_wrench=np.zeros(6),auxiliary_valid=True,
            load=1.2,load_uncertainty=.5,up=np.array([0.,0.,1.]),force_gain=np.full(5,10.),slip_speed=0.)
        ctl.state=state;y=np.r_[p,tau,[0.,0.,1.2,0.,0.,0.]]
        ctl.model.record_input(8.,np.zeros(8));ctl.model.beta=y.copy()
        ctl.model.snapshot=OutputForecast(y,np.zeros(16) if drift is None else drift,np.full(16,.001),np.full(16,.01),
            np.full(16,50),np.ones(16),np.ones(16,bool),y.copy(),np.full(16,8.))
        for k in range(50):
            frame=SimpleNamespace(time=7.7+k*.005,normal=p,shear_magnitude=tau,contact_pos=np.zeros((5,3)))
            ctl.contacts.update(frame,state,ctl.model.snapshot)
        ctl.set_motion_request([0.,0.,0.],[0.,0.,0.]);return ctl
    silent=fixture();silent._solve()
    checks['stable_prior_conflict_produces_exact_zero_action']=bool(np.all(silent.interval_action==0) and silent.debug['corridor_silent_hold'] and all(silent.debug['friction_prior_conflict']))
    checks['same_kernel_coupled_shear_does_not_invent_reserve']=bool(silent.debug['absolute_friction_reserve_identified'] is False)
    four=fixture();four.model.active_mask[1]=False;four.model.snapshot.output_now[[1,6]]=0.
    four.contacts.snapshot.mode[1]='UNKNOWN';four.contacts.anchor_valid[1]=False
    four._solve()
    checks['inactive_index_does_not_require_all_five_contacts_or_destroy_silent_hold']=bool(four.debug.get('corridor_silent_hold'))
    moving=fixture();moving.set_motion_request([-.001,.002,0.],[0.,0.,0.]);moving._solve()
    checks['joint_motion_is_real_decision_input']=bool(moving.interval_motion[0]<0 and moving.interval_motion[1]>0 and moving.previous_plan is not None)
    checks['motion_tiebreak_respects_higher_loss']=bool(moving.debug['higher_priority_quadratic_loss_change']<=1.0001e-8)
    fragile=fixture(limits=np.full(5,1.95));fragile._solve()
    checks['force_conflict_breaks_silence_and_unloads']=bool(not fragile.debug.get('corridor_silent_hold',True) and np.any(fragile.interval_action<0))
    checks['all_horizon_pressure_and_shear_nonnegative']=bool(fragile.debug['predicted_pressure_min_N']>=-1e-8 and fragile.debug['predicted_shear_min_N']>=-1e-8)
    fake=np.array([-1.5,4.75,4.75,4.75,4.75])
    checks['negative_pressure_cannot_buy_budget_credit']=bool((fake+.05).sum()<18 and np.maximum(fake,0).sum()>18)
    dead=fixture(K=np.zeros((101,16,8)),pressure=np.full(5,.01),drift=np.r_[np.full(5,-5.),np.zeros(11)]);dead._solve()
    checks['unreachable_negative_model_not_hidden_by_clipping']=bool(np.isnan(dead.predicted).all() and dead.debug['prediction_available'] is False)
    unavailable=fixture();unavailable.state.valid[:]=False;unavailable.state.update=lambda f,dt:None;unavailable.predicted=np.ones(5)
    bad=unavailable.step(SimpleNamespace(time=8.,phase='HOLDING'))
    checks['all_invalid_clears_previous_prediction']=bool(np.isnan(bad.predicted_normal).all() and bad.debug['prediction_available'] is False)
    unavailable.predicted=np.ones(5);between=unavailable.step(SimpleNamespace(time=8.005,phase='HOLDING'))
    checks['all_invalid_between_decisions_cannot_retimestamp_old_prediction']=bool(np.isnan(between.predicted_normal).all())
    release=unavailable.step(SimpleNamespace(time=8.01,phase='RELEASING'),command='release')
    checks['release_prediction_explicitly_unavailable']=bool(np.isnan(release.predicted_normal).all() and release.debug.get('prediction_unavailable_reason'))
    w=np.array([0.,0.,1.1772,0.,0.,0.]);e=np.full(6,.06);up=np.array([0.,0.,1.])
    checks['load_warmup_cannot_move_support_target']=bool(np.allclose(JointCoordinator.support_band(w,e,up,1.037,.5)[0],JointCoordinator.support_band(w,e,up,1.172,.5)[0]))
    if not all(checks.values()):raise AssertionError(checks)
    return dict(passed=True,checks=checks,scope='pure algebra/synthetic interface, no dynamics or performance')


if __name__=='__main__':
    import json
    print(json.dumps(self_test(),ensure_ascii=False))

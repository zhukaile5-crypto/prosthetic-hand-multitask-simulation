"""V6: force-quality task, public motion tube and contact-mode coordination.

Nominal action regulates observed force quality while following public motion.
The same observer/input history separates task response from grip feedback.
Trusted predictions coordinate force/contact/progress. Unknown contact models
retain checked empirical feedback, rather than forcing fictitious pressure or
shear corrections. No historical pressure point is tracked.
The measured local model remains experimental; passing these checks is not a
global model admission or a physical counterfactual safety certificate.
"""
from dataclasses import dataclass,asdict
from copy import deepcopy
import numpy as np
import research_imports
from vendor.frozen_controllers import make_gains,solve_qp,ControlAction
from state_estimator import CoupledTactileState
from coordinator_model import ContactConditionedModel,OutputForecast,synthetic_document,verify_model_algebra
from coordinator_contacts import ContactCorridorObserver,ForceQualityTask,verify_contact_membership
from grasp_contacts import PersistentGraspContactObserver


@dataclass
class CoordinatorSettings:
    motion_rates: tuple=(.035,.15,.10)
    forecast_margin_N: float=.05
    priority_tolerance_N: float=.0005
    progress_guard_tolerance: float=1e-6
    qp_passes: int=240


class _ObservedState(CoupledTactileState):
    def __init__(self,gains,owner):super().__init__(gains);self.owner=owner
    def update(self,frame,dt):
        super().update(frame,dt)
        o=self.owner;prediction=o.model.update(frame,self,o.executed_motion if o.motion_initialized else None)
        o.contacts.update(frame,self,prediction)
        o.grasp.update(frame,allow_enrollment=o.execution_anchor is None)
        o.latest_frame=frame
        limits,budget=o._force_limits()
        o.contacts.adopt_stable_reference(prediction,limits,budget,float(frame.time),o.last_intervention)


class JointCoordinator:
    def __init__(self,calibration,limits,budget,gains=None,model_document=None):
        B=np.asarray(calibration['force_matrix'],float);g=dict(gains or {})
        g.update(force_gain=tuple(np.maximum(.05,np.linalg.norm(B,axis=0))),force_tau=float(calibration['force_tau']),release_rate=.8)
        self.g=make_gains(g);self.settings=CoordinatorSettings()
        self.model=ContactConditionedModel(calibration,document=model_document)
        self.contacts=ContactCorridorObserver(self.g.nominal_mu)
        self.grasp=PersistentGraspContactObserver();self.force_task=ForceQualityTask(self.g);self.latest_frame=None
        self.motion_initialized=False;self.state=_ObservedState(self.g,self)
        self.limits=self._vector(limits,5,'limits');self.budget=float(budget)
        if np.any(self.limits<0.) or not np.isfinite(self.budget) or self.budget<0.:raise ValueError('Nonnegative physical force limits required')
        # Public robot actuator reference bounds, inherited unchanged from the
        # foundation Shadow wrist defaults and configured reachable lift target.
        lift=float(calibration.get('config',{}).get('lift_target',.1))
        self.motion_lower=np.array([0.,-.523599,-.698132]);self.motion_upper=np.array([lift,.174533,.488692])
        self.name='joint';self.desired=np.zeros(5);self.predicted=np.full(5,np.nan)
        self.last_decision=-np.inf;self.previous_time=None;self.last_intervention=-np.inf
        self.interval_action=np.zeros(5);self.interval_motion=np.zeros(3)
        self.motion_request=np.zeros(3);self.executed_motion=np.zeros(3);self.motion_reference=np.zeros(3)
        self.execution_anchor=None;self.motion_anchor=None;self.previous_plan=None
        self.previous_nominal=None;self.previous_correction=np.zeros(8)
        self.release_latched=False
        self.task_request={};self.debug={};self.label='joint_v6_initializing'
        self.numerical_failures=0;self.infeasible_ticks=0;self.control_ticks=0;self.force_reference=np.zeros(5)

    @staticmethod
    def _vector(value,size,name):
        result=np.asarray(value,float)
        if result.shape!=(size,) or not np.isfinite(result).all():raise ValueError(f'{name} requires {size} finite values')
        return result.copy()

    def set_motion_request(self,target,executed_reference):
        self.motion_request=self._vector(target,3,'motion_target')
        self.executed_motion=self._vector(executed_reference,3,'executed_motion_reference')
        self.motion_reference=self.executed_motion.copy();self.motion_initialized=True

    def set_task_request(self,value):
        """Consume an ordinary public task dict; no schedule or object oracle."""
        if not isinstance(value,dict):raise ValueError('Public task request must be an ordinary dict')
        target=value.get('motion_target',self.motion_request)
        executed=value.get('executed_motion_reference',self.executed_motion)
        self.set_motion_request(target,executed)
        request={}
        for key in ('schema','mode','force_scope','model_domain_transfer','priorities'):
            if key in value:request[key]=value[key].copy() if isinstance(value[key],dict) else value[key]
        for key in ('time_s','task_time_s','deadline_s','remaining_time_s'):
            if value.get(key) is not None:
                x=float(value[key])
                if not np.isfinite(x):raise ValueError(f'{key} must be finite or absent')
                request[key]=x
        if 'limits' in value:
            x=self._vector(value['limits'],5,'task limits')
            if np.any(x<0.):raise ValueError('Task force limits cannot be negative')
            request['limits']=x
        if 'budget' in value:
            x=float(value['budget'])
            if not np.isfinite(x) or x<0.:raise ValueError('Task budget must be finite and nonnegative')
            request['budget']=x
        if 'tracking_tolerances' in value:
            x=list(value['tracking_tolerances'])
            if len(x)!=3 or any(y is not None and (not np.isfinite(float(y)) or float(y)<0.) for y in x):
                raise ValueError('Each tracking tolerance is nonnegative or explicitly unknown')
            request['tracking_tolerances']=[None if y is None else float(y) for y in x]
        if value.get('motion_reference_bounds') is not None:
            bounds=value['motion_reference_bounds']
            lo=self._vector(bounds['lower'],3,'motion reference lower');hi=self._vector(bounds['upper'],3,'motion reference upper')
            if np.any(lo>hi):raise ValueError('Motion reference bounds are reversed')
            self.motion_lower=lo;self.motion_upper=hi;request['motion_bounds_source']='public_actual_robot_bounds'
        if value.get('sensor_characteristics') is not None:
            spec=dict(value['sensor_characteristics']);self.model.set_sensor_characteristics(spec)
            request['sensor_characteristics']=spec
        if value.get('motion_preview') is not None:
            preview=value['motion_preview']
            times=np.asarray(preview['requested_times_s'],float);points=np.asarray(preview['motion_targets'],float)
            if points.shape!=(len(times),3) or len(times)<1 or not np.isfinite(np.r_[times,points.ravel()]).all() or np.any(np.diff(times)<=0.):
                raise ValueError('Public motion preview requires chronological finite waypoints')
            request['motion_preview']=dict(requested_times_s=times,motion_targets=points,
                future_reference_available=bool(preview.get('future_reference_available',False)))
        self.task_request=request
        if str(request.get('mode','')).lower() in ('release','monitor'):self.release_latched=True

    def _force_limits(self):
        # Task intent may tighten material limits, never silently relax them.
        return np.minimum(self.limits,self.task_request.get('limits',self.limits)),min(self.budget,self.task_request.get('budget',self.budget))

    @staticmethod
    def _cost(Q,q,A,b,weight=1.):
        Q+=2.*weight*(A.T@A);q+=2.*weight*(A.T@b)

    @staticmethod
    def support_band(wrench,uncertainty,up,load,load_uncertainty):
        support=float(np.asarray(up)@wrench[:3]);error=max(.06,float(np.abs(up)@uncertainty[:3]))
        measured=np.array([support-error,support+error]);external=np.full(2,np.nan)
        if np.isfinite(load) and np.isfinite(load_uncertainty) and load_uncertainty>=0.:
            external=np.array([max(0.,load-load_uncertainty),load+load_uncertainty])
            overlap=np.array([max(measured[0],external[0]),min(measured[1],external[1])])
            if overlap[0]<=overlap[1]:return overlap,False,measured,external
        return measured,True,measured,external

    def _unavailable(self,reason,raw=None,residual=0.):
        self.interval_action=np.zeros(5);self.interval_motion=np.zeros(3);self.predicted=np.full(5,np.nan)
        self.desired=np.zeros(5) if self.model.snapshot is None else self.model.snapshot.normal_now.copy()
        self.force_reference=self.desired.copy();self.label='joint_v6_model_unavailable';self.numerical_failures+=1
        self.debug=dict(controller_architecture='force_quality_task_contact_motion_v6',prediction_available=False,
            prediction_invalid_reason=reason,rejected_raw_contact_prediction=[] if raw is None else np.asarray(raw).tolist(),
            qp_primal_residual=float(residual),model_admitted=False,support_infeasibility_certified=False,
            fallback='zero executable correction; local model unavailable, no clipping or physical impossibility claim')

    def _motion_preview(self,now,h,period,positive,negative):
        user=np.tile(np.clip(self.motion_request,self.motion_lower,self.motion_upper),(h,1))
        public=self.task_request.get('motion_preview')
        if public is not None and public['future_reference_available']:
            times=public['requested_times_s'];values=public['motion_targets']
            query=now+np.arange(1,h+1)*period
            user=np.column_stack([np.interp(query,times,values[:,i]) for i in range(3)])
            user=np.clip(user,self.motion_lower,self.motion_upper)
        lower=np.minimum(user,self.executed_motion);upper=np.maximum(user,self.executed_motion)
        for i,tolerance in enumerate(self.task_request.get('tracking_tolerances',[None]*3)):
            if tolerance is not None:
                lower[:,i]=user[:,i]-tolerance;upper[:,i]=user[:,i]+tolerance
        lower=np.maximum(lower,self.motion_lower);upper=np.minimum(upper,self.motion_upper)
        for i,direction in enumerate(self.model.available_directions[5:]):
            if not direction['plus']:upper[:,i]=np.minimum(upper[:,i],self.motion_anchor[i])
            if not direction['minus']:lower[:,i]=np.maximum(lower[:,i],self.motion_anchor[i])
        reachable=[];last=self.executed_motion.copy()
        for j in range(h):
            last=last+np.clip(user[j]-last,-negative[5:],positive[5:]);reachable.append(last.copy())
        return user,np.asarray(reachable)-self.executed_motion,lower,upper

    def _feedback_mode(self,nominal,positive,negative,lower,upper,mlower,mupper,reason,task_debug,assessment):
        """Checked mechanical projection; an unknown model is not a zero task.

        This is an explicitly empirical mature-feedback execution mode. It
        makes no pressure/shear/wrench forecast or model safety certification.
        """
        cumulative=np.zeros(8);out=[];s=self.state
        for j,u in enumerate(nominal):
            lo=np.maximum(-negative,np.r_[lower-s.alpha,mlower[j]-self.executed_motion]-cumulative)
            hi=np.minimum(positive,np.r_[upper-s.alpha,mupper[j]-self.executed_motion]-cumulative)
            blocked=~s.valid|~s.closure_permission;hi[:5][blocked]=np.minimum(0.,hi[:5][blocked])
            if np.any(lo>hi+1e-8):
                self._unavailable('empty_actual_execution_or_task_tube',residual=float(np.max(lo-hi)));return
            chosen=np.minimum(hi,np.maximum(lo,u));out.append(chosen);cumulative+=chosen
        out=np.asarray(out)
        if not np.isfinite(out).all():raise RuntimeError('Nonfinite empirical feedback action')
        cumulative=np.cumsum(out,axis=0)
        violations=[float(np.max(out-positive)),float(np.max(-out-negative)),
            float(np.max(cumulative[:,:5]-(upper-s.alpha))),float(np.max(lower-s.alpha-cumulative[:,:5])),
            float(np.max(cumulative[:,5:]-(mupper-self.executed_motion))),
            float(np.max(mlower-self.executed_motion-cumulative[:,5:]))]
        blocked=~s.valid|~s.closure_permission
        if blocked.any():violations.append(float(np.max(out[:,:5][:,blocked])))
        authority_residual=max(0.,max(violations))
        if authority_residual>1e-8:
            self._unavailable('unchecked_empirical_execution_projection',residual=authority_residual);return
        self.interval_action=out[0,:5].copy();self.interval_motion=out[0,5:].copy()
        self.previous_nominal=nominal[0].copy();self.previous_plan=None
        self.predicted=np.full(5,np.nan);self.label='joint_v6_observed_feedback_model_unknown'
        self.debug=dict(controller_architecture='force_quality_task_contact_motion_v6',
            prediction_available=False,prediction_invalid_reason=reason,model_assessment=assessment,
            feedback_task_preserved=bool(np.max(abs(out-nominal))<=1e-8),model_based_force_safety_certified=False,
            nominal_action_8=nominal[0].tolist(),executed_interval_action_8=out[0].tolist(),
            mechanical_projection_max_correction=float(np.max(abs(out-nominal))),
            motion_tube_lower=mlower.tolist(),motion_tube_upper=mupper.tolist(),
            final_authority_residual=authority_residual,model_admitted=False,cumulative_model_domain_certified=False,
            fallback='same-observer observed feedback task through real execution bounds; no invalid model constraints',**task_debug)

    def _step_bounds(self):
        period=self.g.decision_period
        positive=np.minimum(np.r_[np.full(5,self.g.close_rate*period),np.asarray(self.settings.motion_rates)*period],np.maximum(0.,self.model.input_upper))
        negative=np.minimum(np.r_[np.full(5,self.g.open_rate*period),np.asarray(self.settings.motion_rates)*period],np.maximum(0.,-self.model.input_lower))
        # A missing direction is not measured zero gain. Return to a visited
        # anchor may use the measured opposite linearization experimentally.
        for i,available in enumerate(self.model.available_directions):
            if not available['plus']:positive[i]=min((self.g.close_rate if i<5 else self.settings.motion_rates[i-5])*period,self.model.input_step_limits[i])
            if not available['minus']:negative[i]=min((self.g.open_rate if i<5 else self.settings.motion_rates[i-5])*period,self.model.input_step_limits[i])
        positive[~self.model.identified_inputs]=0.;negative[~self.model.identified_inputs]=0.
        return positive,negative

    def _decision(self):
        s=self.state;f=self.model.snapshot;cs=self.contacts.snapshot;h=self.g.horizon;period=self.g.decision_period;nr=8*h
        if f is None or cs is None:
            self._unavailable('no_causal_contact_observation');return
        if self.execution_anchor is None:
            self.contacts.freeze_handoff(f);self.execution_anchor=s.alpha.copy();self.motion_anchor=self.executed_motion.copy()
            self.grasp.freeze_handoff()
        limits,budget=self._force_limits();P,c,error=self.model.trajectory(s,h,period)
        pi=np.concatenate([np.arange(16*j,16*j+5) for j in range(h)])
        ti=np.concatenate([np.arange(16*j+5,16*j+10) for j in range(h)])
        wi=np.concatenate([np.arange(16*j+10,16*j+16) for j in range(h)])
        cp=c[pi];ct=c[ti];ep=error[pi];et=error[ti]
        positive,negative=self._step_bounds()
        lower=s.closure_min.copy();upper=s.closure_max.copy()
        for i,available in enumerate(self.model.available_directions):
            if i<5:
                if not available['plus']:upper[i]=min(upper[i],self.execution_anchor[i])
                if not available['minus']:lower[i]=max(lower[i],self.execution_anchor[i])
        now=float(self.model.input_history[-1][0])
        user_targets,preview,mlower,mupper=self._motion_preview(now,h,period,positive,negative)
        operating_target=user_targets[0]
        if np.any(lower>s.alpha+1e-8) or np.any(upper<s.alpha-1e-8) or np.any(mlower>mupper+1e-8):
            self._unavailable('current_executed_reference_outside_known_authority');return
        previous_actual=self.model.actual_increment(now-period,now)
        self.force_task.reconcile(previous_actual)
        hypothesis=self.grasp.snapshot
        persistent=hypothesis.support_present&hypothesis.known_support&hypothesis.valid&hypothesis.fresh
        spec=self.task_request.get('sensor_characteristics',{})
        sigma=np.full(5,float(spec.get('force_sigma_N',.01)))
        new_external=hypothesis.new_normal>np.maximum(.04,3.*sigma)
        preliminary=self.model.assess(c.reshape(h,16))
        if new_external.any():preliminary['usable']=False
        explained=self.model.input_effect(now,now,input_columns=(5,6,7))[:5] if preliminary['usable'] else None
        before_proposal=deepcopy(self.force_task)
        force_action,force_target,task_debug=self.force_task.propose(s,cs,persistent&~new_external,limits,budget,explained,sigma)
        self.desired=force_target.copy();self.force_reference=force_target.copy()
        nominal=np.zeros((h,8));nominal[0,:5]=np.clip(force_action,
            np.maximum(-negative[:5],lower-s.alpha),np.minimum(positive[:5],upper-s.alpha))
        blocked=~s.valid|~s.closure_permission;nominal[0,:5][blocked]=np.minimum(0.,nominal[0,:5][blocked])
        nominal[:,5:]=np.diff(np.vstack((np.zeros(3),preview)),axis=0);un=nominal.ravel()
        raw_nominal=(c+P@un).reshape(h,16)
        assessment=self.model.assess(raw_nominal)
        if new_external.any():
            assessment['usable']=False;assessment['reasons'].append('new_unaffiliated_external_contact')
        if not assessment['usable'] and explained is not None:
            # The candidate can invalidate an otherwise usable neutral model.
            # Discard that model's compensation too, without sampling/integrating
            # the force task twice or leaving a withdrawn model in feedback.
            self.force_task=before_proposal
            force_action,force_target,task_debug=self.force_task.propose(s,cs,persistent&~new_external,limits,budget,None,sigma)
            self.desired=force_target.copy();self.force_reference=force_target.copy()
            nominal[0,:5]=np.clip(force_action,np.maximum(-negative[:5],lower-s.alpha),np.minimum(positive[:5],upper-s.alpha))
            nominal[0,:5][blocked]=np.minimum(0.,nominal[0,:5][blocked]);un=nominal.ravel()
        task_debug.update(grasp_persistent_mask=persistent.tolist(),new_external_contact_mask=new_external.tolist(),
            grasp_contact_status=hypothesis.status.tolist(),force_quality_nominal_action=force_action.tolist())
        if not assessment['usable']:
            self._feedback_mode(nominal,positive,negative,lower,upper,mlower,mupper,
                ';'.join(assessment['reasons']),task_debug,assessment);return
        active=persistent&self.model.active_mask&(f.normal_now>.15)&f.output_valid[:5]&f.output_valid[5:10]
        contact_known=bool(active[0] and np.count_nonzero(active)>=3 and np.count_nonzero((cs.mode=='STICK_CONFIRMED')&active)>=3
            and np.all(cs.speed_lower[active]<.002))
        complete=bool(np.all(f.output_valid[10:]) and np.isfinite(c[wi]).all())
        ratio=np.divide(f.output_now[5:10],f.normal_now,out=np.zeros(5),where=f.normal_now>.15)
        # The descriptive stable reference is used only for uncertain contact
        # rows; it is never a historical force target or a pressure corridor.
        use_reference=(cs.mode!='STICK_CONFIRMED')&cs.anchor_valid
        ratio[use_reference]=cs.anchor_ratio[use_reference]
        floor=np.where(active,np.minimum(.15,f.normal_now),0.)
        tube=et+np.tile(ratio,h)*ep+.02
        robust_c=cp+ep+self.settings.forecast_margin_N
        # u[8h], pressure[5]/budget[1], contact-retention[5], shear[5], wrench[7].
        dim=nr+23;ie=slice(nr,nr+5);ib=nr+5;ir=slice(nr+6,nr+11);it=slice(nr+11,nr+16);iw=slice(nr+16,dim)
        I=np.eye(dim);N=np.zeros((5*h,dim));N[:,:nr]=P[pi];T=np.zeros_like(N);T[:,:nr]=P[ti]
        rows=[];rhs=[]
        def constraint(A,b):rows.append(np.atleast_2d(A));rhs.append(np.atleast_1d(b))
        constraint(I[:nr],np.tile(positive,h));constraint(-I[:nr],np.tile(negative,h))
        C=np.kron(np.tril(np.ones((h,h))),np.eye(8));CU=np.zeros((nr,dim));CU[:,:nr]=C
        apos=np.concatenate([np.arange(8*j,8*j+5) for j in range(h)]);mpos=np.concatenate([np.arange(8*j+5,8*j+8) for j in range(h)])
        constraint(CU[apos],np.tile(upper-s.alpha,h));constraint(-CU[apos],np.tile(s.alpha-lower,h))
        constraint(CU[mpos],(mupper-self.executed_motion).ravel());constraint(-CU[mpos],(self.executed_motion-mlower).ravel())
        blocked=~s.valid|~s.closure_permission
        if np.any(blocked):constraint(I[apos][np.tile(blocked,h)],np.zeros(np.count_nonzero(blocked)*h))
        missing=~self.model.identified_inputs
        if np.any(missing):
            Z=I[:nr][np.tile(missing,h)];constraint(Z,np.zeros(len(Z)));constraint(-Z,np.zeros(len(Z)))
        constraint(-N,cp);constraint(-T,ct)
        hard_A=np.vstack(rows);hard_b=np.concatenate(rhs)
        for sl in (ie,ir,it,iw):constraint(-I[sl],np.zeros(sl.stop-sl.start))
        constraint(-I[ib],0.)
        Eu=np.tile(I[ie],(h,1));Er=np.tile(I[ir],(h,1));Et=np.tile(I[it],(h,1));Eb=np.tile(I[ib],(h,1))
        constraint(N-Eu,np.tile(limits,h)-robust_c)
        sums=np.kron(np.eye(h),np.ones((1,5)));constraint(sums@N-Eb,np.full(h,budget)-sums@robust_c)
        constraint(-N-Er,cp-np.tile(floor,h))
        constraint(T-np.tile(ratio,h)[:,None]*N-Et,tube-(ct-np.tile(ratio,h)*cp))
        WN=None;wc=None;wl=None;wh=None;support_info={}
        if complete:
            projection=np.vstack((np.eye(6),np.r_[s.up,np.zeros(3)]));PH=np.kron(np.eye(h),projection)
            WN=np.zeros((6*h,dim));WN[:,:nr]=P[wi];WN=PH@WN;wc=PH@c[wi]
            me=np.maximum(f.fit_error[10:],np.r_[np.full(3,.06),np.full(3,.003)])
            band,inconsistent,unused,load_band=self.support_band(f.output_now[10:],me,s.up,float(s.load),float(s.load_uncertainty))
            we=(np.abs(PH)@error[wi]).reshape(h,7)
            wl=np.r_[f.output_now[10:]-me,band[0]]-we;wh=np.r_[f.output_now[10:]+me,band[1]]+we
            EW=np.tile(I[iw],(h,1));constraint(WN-EW,wh.ravel()-wc);constraint(-WN-EW,wc-wl.ravel())
            support_info=dict(support_band_N=band.tolist(),load_interval_N=load_band.tolist(),load_wrench_interval_inconsistent=bool(inconsistent))
        A=np.vstack(rows);b=np.concatenate(rhs)
        def slack_completion(u):
            x=np.zeros(dim);x[:nr]=u;p=(N@x+cp).reshape(h,5);tau=(T@x+ct).reshape(h,5)
            x[ie]=np.maximum(0.,p+ep.reshape(h,5)+self.settings.forecast_margin_N-limits).max(axis=0)
            x[ib]=max(0.,float((p+ep.reshape(h,5)+self.settings.forecast_margin_N).sum(axis=1).max()-budget))
            x[ir]=np.maximum(0.,floor-p).max(axis=0)
            x[it]=np.maximum(0.,tau-ratio*p-tube.reshape(h,5)).max(axis=0)
            if complete:
                w=(WN@x+wc).reshape(h,7);x[iw]=np.maximum(0.,np.maximum(w-wh,wl-w)).max(axis=0)
            return x
        nominal_x=slack_completion(un)
        observed_guard=bool(np.all(f.last_measured[:5]<=limits) and f.last_measured[:5].sum()<=budget)
        check=dict(execution_authority=bool(np.max(hard_A@nominal_x-hard_b)<=1e-8),
            measured_force_guard=observed_guard,predicted_force_guard=bool(np.max(nominal_x[ie])<=1e-8 and nominal_x[ib]<=1e-8),
            contact_state_available=contact_known,contact_retention=bool(np.max(nominal_x[ir])<=1e-8),
            shear_compatibility=bool(np.max(nominal_x[it])<=1e-8),complete_wrench_available=complete,
            complete_wrench_compatibility=bool(complete and np.max(nominal_x[iw])<=1e-8))
        self.previous_correction=np.zeros(8) if previous_actual is None or self.previous_nominal is None else previous_actual-self.previous_nominal
        statuses=[];residuals=[];checked_residuals=[];fallbacks=[];nominal_pass=all(check.values())
        final_A=A;final_b=b
        if nominal_pass:
            x=nominal_x;self.label='joint_v6_force_quality_task_preserved'
        else:
            # Missing contact/task evidence is not a fictitious zero wrench or
            # a material capacity certificate. Hold motion while protecting
            # observable pressure, and expose the incomplete task model.
            # Uncertain contact is reported separately. It does not turn a
            # public hold tolerance tube into a prohibition of all motion.
            seed=slack_completion(np.zeros(nr))
            seed[ie]+=.001;seed[ib]+=.001;seed[ir]+=.001;seed[it]+=.001;seed[iw]+=.001
            if np.max(A@seed-b)>1e-8:
                repair=solve_qp(np.eye(dim),np.zeros(dim),A,b,self.settings.qp_passes,2e-5)
                repair_actual=float(np.max(A@repair.x-b)) if np.isfinite(repair.x).all() else float('inf')
                if repair.residual>1e-8 or repair_actual>1e-8:
                    self._unavailable('no_checked_nonnegative_contact_trajectory',np.r_[cp,ct],max(repair.residual,repair_actual));return
                seed=repair.x.copy()
            def optimize(Q,q,AA,bb,initial):
                result=solve_qp(Q,q,AA,bb,self.settings.qp_passes,2e-5,feasible_initial=initial)
                statuses.append(result.status);residuals.append(float(result.residual))
                actual=float(np.max(AA@result.x-bb)) if np.isfinite(result.x).all() else float('inf')
                if result.residual<=2e-5 and actual<=2e-5:
                    candidate=result.x.copy();fallbacks.append(None)
                else:
                    candidate=initial.copy();fallbacks.append('checked_previous_layer')
                checked=float(np.max(AA@candidate-bb));checked_residuals.append(checked)
                if not np.isfinite(checked) or checked>2e-5:
                    raise RuntimeError('Neither solver output nor previous priority layer is checked feasible')
                if not result.converged:self.numerical_failures+=1
                return candidate
            norm=np.maximum(.5,limits);Q=np.eye(dim)*1e-8;q=np.zeros(dim)
            self._cost(Q,q,I[ie]/norm[:,None],np.zeros(5));self._cost(Q,q,I[ib:ib+1]/max(1.,budget),np.zeros(1))
            x_force=optimize(Q,q,A,b,seed)
            GF=I[np.r_[np.arange(ie.start,ie.stop),ib]]
            AF=np.vstack((A,GF));bf=np.r_[b,GF@x_force+self.settings.priority_tolerance_N]
            Q=np.eye(dim)*1e-8;q=np.zeros(dim)
            self._cost(Q,q,I[ir]/norm[:,None],np.zeros(5));self._cost(Q,q,I[it]/norm[:,None],np.zeros(5))
            ws=np.r_[np.ones(3),np.full(3,1./self.g.characteristic_length),1.]
            if complete:self._cost(Q,q,ws[:,None]*I[iw],np.zeros(7))
            x_contact=optimize(Q,q,AF,bf,x_force)
            GC=I[np.r_[np.arange(ir.start,ir.stop),np.arange(it.start,it.stop),np.arange(iw.start,iw.stop)]]
            tolerances=np.r_[np.full(10,self.settings.priority_tolerance_N),self.settings.priority_tolerance_N/ws]
            AC=np.vstack((AF,GC));bc=np.r_[bf,GC@x_contact+tolerances]
            ms=np.tile(1./np.maximum(1e-6,np.maximum(positive[5:],negative[5:])),h)
            MP=ms[:,None]*CU[mpos];mt=ms*preview.ravel()
            Q=np.eye(dim)*1e-8;q=np.zeros(dim);self._cost(Q,q,MP,-mt)
            x_progress=optimize(Q,q,AC,bc,x_contact)
            deviation=np.abs(MP@x_progress-mt)+self.settings.progress_guard_tolerance
            AP=np.vstack((AC,MP,-MP));bp=np.r_[bc,mt+deviation,-mt+deviation]
            Q=np.eye(dim)*1e-8;q=np.zeros(dim)
            scales=np.tile(1./np.maximum(1e-6,np.maximum(positive,negative)),h)
            self._cost(Q,q,scales[:,None]*I[:nr],-scales*un)
            D8=np.kron(np.eye(h)-np.eye(h,k=-1),np.eye(8))
            old=np.r_[self.previous_correction,np.zeros(nr-8)]
            self._cost(Q,q,scales[:,None]*(D8@I[:nr]),-scales*(D8@un+old))
            DP=np.kron(np.eye(h)-np.eye(h,k=-1),np.eye(5));force_scale=np.tile(1./norm,h)
            self._cost(Q,q,force_scale[:,None]*(DP@N),-force_scale*(DP@N@nominal_x))
            x=optimize(Q,q,AP,bp,x_progress);final_A=AP;final_b=bp
            self.label='joint_v6_predictive_task_coordination'
        raw_p=N@x+cp;raw_tau=T@x+ct
        if np.min(np.r_[raw_p,raw_tau])<-1e-8:
            self._unavailable('negative_raw_contact_magnitude_after_solver',np.r_[raw_p,raw_tau],float(-np.min(np.r_[raw_p,raw_tau])));return
        # The final executable plan, not the solver status alone, is checked.
        final_residual=float(np.max(A@x-b))
        hierarchy_residual=float(np.max(final_A@x-final_b));authority_residual=float(np.max(hard_A@x-hard_b))
        if final_residual>2e-5 or hierarchy_residual>2e-5 or authority_residual>1e-8:
            self._unavailable('unchecked_executable_or_priority_plan',np.r_[raw_p,raw_tau],max(final_residual,hierarchy_residual,authority_residual));return
        correction=x[:nr]-un;self.interval_action=x[:5].copy();self.interval_motion=x[5:8].copy()
        self.previous_plan=x.copy();self.previous_nominal=nominal[0].copy()
        if np.max(np.abs(correction[:8]))>1e-8:self.last_intervention=float(self.model.input_history[-1][0])
        self.predicted=raw_p[:5].copy()
        missing_directions=[]
        for i,d in enumerate(self.model.available_directions):
            if (not d['plus'] and np.any(nominal[:,i]>1e-10)) or (not d['minus'] and np.any(nominal[:,i]<-1e-10)):missing_directions.append(i)
        now=float(self.model.input_history[-1][0]);deadline=self.task_request.get('deadline_s')
        remaining=None if deadline is None else max(0.,deadline-now)
        reach_time=float(np.max(np.abs(operating_target-self.executed_motion)/np.maximum(1e-6,np.asarray(self.settings.motion_rates))))
        components=self.model.last_trajectory_components
        self.debug=dict(controller_architecture='force_quality_task_contact_motion_v6',prediction_available=True,
            nominal_task_preserved=bool(nominal_pass),nominal_operating_checks=check,
            nominal_failed_checks=[key for key,value in check.items() if not value],
            nominal_action_8=nominal[0].tolist(),nominal_plan_8=nominal.tolist(),correction_8=correction[:8].tolist(),
            actual_previous_interval_correction_8=self.previous_correction.tolist(),actual_previous_increment_available=previous_actual is not None,
            qp_status='nominal_candidate_preserved' if not statuses else statuses[-1],priority_status=statuses,
            qp_optimal=None if not statuses else bool(all(status=='optimal' for status in statuses)),
            qp_primal_residual=max([max(0.,final_residual)]+residuals),
            qp_checked_priority_residuals=checked_residuals,qp_priority_fallbacks=fallbacks,
            final_priority_residual=hierarchy_residual,final_authority_residual=authority_residual,
            predicted_pressure_violation_N=x[ie].tolist(),predicted_budget_violation_N=float(x[ib]),
            contact_retention_slack_N=x[ir].tolist(),contact_consistency_slack_N=x[it].tolist(),wrench_band_violation=x[iw].tolist(),
            pressure_now=f.normal_now.tolist(),shear_now=f.output_now[5:10].tolist(),force_prediction=self.predicted.tolist(),
            force_target=self.desired.tolist(),nominal_force_prediction=raw_nominal[:,:5].tolist(),
            predicted_extra_pressure_variation_N=(np.kron(np.eye(h)-np.eye(h,k=-1),np.eye(5))@(P[pi]@correction)).reshape(h,5).tolist(),
            predicted_shear=raw_tau[:5].tolist(),predicted_pressure_min_N=float(raw_p.min()),predicted_shear_min_N=float(raw_tau.min()),
            estimated_normal=s.normal.tolist(),observed_complete_hand_wrench=f.output_now[10:].tolist(),
            complete_wrench_prediction_available=complete,
            predicted_complete_hand_wrench=((WN@x+wc).reshape(h,7)[:,:6].tolist() if complete else np.full((h,6),np.nan).tolist()),
            motion_request=self.motion_request.tolist(),operating_motion_target=operating_target.tolist(),interval_motion=self.interval_motion.tolist(),
            task_mode=self.task_request.get('mode','legacy_motion_request'),task_deadline_s=deadline,remaining_time_s=remaining,
            deadline_infeasible_under_reference_rate=None if remaining is None else reach_time>remaining+1e-9,
            tracking_tolerances=self.task_request.get('tracking_tolerances',[None,None,None]),
            force_scope='external_all_instrumented_fingers',contact_modes=cs.mode.tolist(),
            friction_lower_bound=cs.mu_lower.tolist(),friction_upper_bound_available=False,absolute_friction_reserve_identified=False,
            contact_reference_updated_at=self.contacts.reference_updated_at,contact_reference_updates=self.contacts.reference_updates,
            contact_reference_scope='causally observed stable contact; no pressure target or model-domain expansion',
            pressure_calibration_error_N=components['calibration_error'][pi].reshape(h,5).tolist(),
            pressure_residual_fit_error_N=components['fit_error'][pi].reshape(h,5).tolist(),
            pressure_slope_error_N=components['slope_error'][pi].reshape(h,5).tolist(),
            pressure_past_input_memory_change_N=components['past_input_memory_change'][pi].reshape(h,5).tolist(),
            pressure_residual_drift_change_N=components['residual_drift_change'][pi].reshape(h,5).tolist(),
            forecast_margin_N=self.settings.forecast_margin_N,model_admitted=False,model_domain_transfer='experimental_local_model_not_admitted',
            cumulative_model_domain_certified=False,missing_measured_direction_axes=missing_directions,
            opposite_direction_use='experimental return to visited anchor only',model_source=self.model.source_path,
            model_source_sha256=self.model.source_sha256,model_frame='current_palm',future_slip_model_available=False,
            support_infeasibility_certified=False,actual_input_history_only=True,pressure_shear_wrench_same_kernel_history=True,
            release_scope='external release outside local holding model',settings=asdict(self.settings),
            motion_tube_lower=mlower.tolist(),motion_tube_upper=mupper.tolist(),
            public_preview_available=bool(self.task_request.get('motion_preview',{}).get('future_reference_available',False)),
            preview_execution='20ms uniform finite ramp to public preview endpoints; arbitrary 5ms profile not certified',
            model_assessment=assessment,**task_debug,**support_info)

    def step(self,frame,phase=None,command='hold'):
        dt=.005 if self.previous_time is None else max(.0001,min(.05,float(frame.time)-self.previous_time))
        self.previous_time=float(frame.time);self.state.update(frame,dt);self.control_ticks+=1
        if command=='release' or str(phase or frame.phase).upper() in ('RELEASE','RELEASING') or str(self.task_request.get('mode','')).lower() in ('release','monitor'):
            self.release_latched=True
        if self.release_latched:
            self.label='joint_common_release';self.desired=np.zeros(5);self.predicted=np.full(5,np.nan);self.motion_reference=self.executed_motion.copy()
            delta=-np.minimum(np.maximum(0.,self.state.alpha-self.state.closure_min),self.g.release_rate*dt)
            return ControlAction(delta,self.predicted.copy(),self.label,False,dict(release_authority=True,prediction_available=False,
                prediction_unavailable_reason='external_mechanical_release_outside_held_contact_model'))
        if np.count_nonzero(self.state.valid)==0:
            self.force_task.observation_missing()
            self.interval_action=np.zeros(5);self.interval_motion=np.zeros(3);self.predicted=np.full(5,np.nan)
            self.motion_reference=self.executed_motion.copy();self.previous_nominal=np.zeros(8)
            self.label='joint_observation_unavailable';self.debug=dict(prediction_available=False,prediction_unavailable_reason='all_tactile_channels_invalid')
            return ControlAction(np.zeros(5),self.predicted.copy(),self.label,False,self.debug.copy())
        if frame.time-self.last_decision>=self.g.decision_period-1e-8:
            self._decision();self.last_decision=float(frame.time)
        fraction=min(1.,dt/self.g.decision_period)
        delta=np.clip(self.interval_action*fraction,self.state.closure_min-self.state.alpha,self.state.closure_max-self.state.alpha)
        blocked=~self.state.valid|~self.state.closure_permission;delta[blocked]=np.minimum(0.,delta[blocked])
        self.motion_reference=self.executed_motion+self.interval_motion*fraction
        self.debug['executed_motion_reference']=self.motion_reference.tolist()
        return ControlAction(delta,self.predicted.copy(),self.label,False,self.debug.copy())


def create_joint(calibration,limits,budget,gains=None):return JointCoordinator(calibration,limits,budget,gains)


def self_test():
    from types import SimpleNamespace
    from force_model import ForceResponse
    from grasp_contacts import GraspContactHypothesis
    checks={**verify_model_algebra(),**verify_contact_membership()}
    cal=dict(force_matrix=np.eye(5)*10.,force_tau=.02,reference_alpha=np.zeros(5),response_error_bound=np.full(5,.02),config={'shape':'capsule','lift_target':.1})
    def fixture(pressure=2.,limits=5.):
        ctl=JointCoordinator(cal,np.full(5,limits),18.,model_document=synthetic_document())
        p=np.full(5,pressure);tau=.6*p
        state=SimpleNamespace(normal=p.copy(),delayed_normal=p.copy(),tangential=np.column_stack((tau,np.zeros(5))),
            alpha=np.zeros(5),closure_min=np.full(5,-.04),closure_max=np.full(5,.04),
            valid=np.ones(5,bool),closure_permission=np.ones(5,bool),contact_model_active=np.ones(5,bool),contact=np.ones(5,bool),
            confidence=np.ones(5),slip_magnitude=np.zeros(5),auxiliary_wrench=np.zeros(6),auxiliary_valid=True,
            load=1.2,load_uncertainty=.5,up=np.array([0.,0.,1.]),force_gain=np.full(5,10.),slip_speed=0.,
            force_model=ForceResponse(np.eye(5)*10.,.02),nominal_weights=np.ones(5),model_reference_alpha=np.zeros(5),time=8.)
        ctl.state=state;y=np.r_[p,tau,[0.,0.,1.2,0.,0.,0.]]
        ctl.model.record_input(7.98,np.r_[np.zeros(5),.1,0.,0.]);ctl.model.record_input(8.,np.r_[np.zeros(5),.1,0.,0.]);ctl.model.beta=y.copy()
        fit=np.r_[np.full(10,.009),np.full(3,.004),np.full(3,.0002)]
        ctl.model.snapshot=OutputForecast(y,np.zeros(16),np.full(16,.001),fit,np.full(16,50),np.ones(16),np.ones(16,bool),y.copy(),np.full(16,8.))
        for k in range(50):
            frame=SimpleNamespace(time=7.7+k*.005,normal=p,shear_magnitude=tau,contact_pos=np.zeros((5,3)))
            ctl.contacts.update(frame,state,ctl.model.snapshot)
        hyp=GraspContactHypothesis(time=8.,sample_time=7.99,persistent_normal=p,new_normal=np.zeros(5),all_normal=p,
            support_present=np.ones(5,bool),known_support=np.ones(5,bool),valid=np.ones(5,bool),fresh=np.ones(5,bool),
            geometry_available=np.ones(5,bool),confidence=np.ones(5),status=np.full(5,'PERSISTENT'),anchor_count=np.ones(5,int),frozen=True)
        ctl.grasp=SimpleNamespace(snapshot=hyp,freeze_handoff=lambda:None)
        ctl.execution_anchor=np.zeros(5);ctl.motion_anchor=np.array([.1,0.,0.])
        ctl.set_motion_request([.1,0.,0.],[.1,0.,0.]);return ctl
    quality=fixture();quality._decision()
    checks['nominal_task_contains_actual_force_regulation']=bool(np.any(quality.interval_action<0.) and quality.debug['task']=='measured_force_quality_with_friction_ratio_reference')
    checks['force_target_is_public_distribution_not_handoff_pressure']=bool(np.allclose(quality.desired,quality.g.base) and not np.allclose(quality.desired,quality.state.normal))
    moving=fixture();moving.set_task_request(dict(mode='transport',motion_target=[.1,0.,0.],executed_motion_reference=[.1,0.,0.],
        time_s=8.,deadline_s=12.,tracking_tolerances=[.01,.015,.015],
        motion_preview=dict(requested_times_s=[8.,8.02,8.04,8.06],motion_targets=[[.1,0.,0.],[.0996,.001,.0005],[.0992,.002,.001],[.0988,.003,.0015]],future_reference_available=True),
        motion_reference_bounds=dict(lower=[0.,-.4,-.6],upper=[.1,.17,.48])))
    moving._decision()
    checks['public_preview_is_consumed_as_finite_ramp_endpoint']=bool(np.allclose(moving.interval_motion,[-.0004,.001,.0005]))
    checks['hold_tolerance_is_a_real_motion_tube']=bool(np.all(np.array(moving.debug['motion_tube_lower'])[:,1:]<0.) and np.all(np.array(moving.debug['motion_tube_upper'])[:,1:]>0.))
    checks['actual_robot_reference_bounds_override_legacy_defaults']=bool(np.allclose(moving.motion_lower,[0.,-.4,-.6]))
    checks['public_error_and_raw_nonnegative_checks_remain']=bool(moving.debug['predicted_pressure_min_N']>=-1e-8 and moving.debug['predicted_shear_min_N']>=-1e-8 and not moving.debug['model_admitted'])
    impact=fixture();impact.model.snapshot.fit_error[0]=9.;impact._decision()
    checks['unexplained_impact_retains_feedback_without_fake_prediction']=bool(np.isnan(impact.predicted).all() and np.any(impact.interval_action!=0.) and impact.debug['prediction_invalid_reason'] and not impact.debug['model_based_force_safety_certified'])
    unsigned=fixture();unsigned.model.beta[5]=-.1;unsigned._decision()
    checks['negative_unsigned_shear_is_model_unknown_not_forced_grip']=bool(np.isnan(unsigned.predicted).all() and 'unsigned_shear_forecast_outside_mode' in unsigned.debug['prediction_invalid_reason'])
    fresh_contact=fixture();fresh_contact.grasp.snapshot.new_normal[0]=1.;fresh_contact._decision()
    checks['new_external_patch_never_earns_grasp_model_credit']=bool(np.isnan(fresh_contact.predicted).all() and not fresh_contact.debug['persistent_feedback_mask'][0])
    quiet=fixture(pressure=1.21);quiet.set_task_request(dict(mode='hold',motion_target=[.1,0.,0.],executed_motion_reference=[.1,0.,0.],
        sensor_characteristics=dict(force_sigma_N=.08,tangential_sigma_N=.04)))
    quiet._decision()
    checks['public_sensor_noise_is_not_force_quality_command']=bool(np.all(quiet.interval_action==0.) and np.allclose(quiet.model.noise_floor[:5],.08))
    fragile=fixture(limits=1.95);fragile._decision()
    checks['real_force_conflict_uses_four_checked_priority_layers']=bool(np.any(fragile.interval_action<0.) and len(fragile.debug['priority_status'])==4 and fragile.debug['final_priority_residual']<=2e-5 and fragile.debug['final_authority_residual']<=1e-8)
    task=ForceQualityTask(quality.g);task.target_ratio=.1
    action,_,_=task.propose(quality.state,quality.contacts.snapshot,np.ones(5,bool),np.full(5,5.),18.)
    task.reconcile(np.zeros(8));checks['blocked_actual_execution_does_not_integrate_nominal_error']=task.integral==0.
    action,_,_=task.propose(quality.state,quality.contacts.snapshot,np.ones(5,bool),np.full(5,5.),18.)
    task.reconcile(np.r_[action,np.zeros(3)]);checks['integral_commits_only_observed_realization']=bool(task.integral>0. and task.last_realization['integral_committed'])
    columns=fixture().model
    columns.record_input(8.02,np.r_[np.full(5,.003),.099,.002,.001])
    checks['motion_and_grip_response_partition_same_actual_history']=bool(np.allclose(columns.input_effect(8.06),columns.input_effect(8.06,input_columns=range(5))+columns.input_effect(8.06,input_columns=range(5,8))))
    unavailable=fixture();unavailable.state.valid[:]=False;unavailable.state.update=lambda f,dt:None;unavailable.predicted=np.ones(5)
    out=unavailable.step(SimpleNamespace(time=8.,phase='HOLDING'))
    checks['observation_missing_is_not_model_unknown_or_stale_prediction']=bool(np.isnan(out.predicted_normal).all() and np.all(out.delta_closure==0.) and 'prediction_unavailable_reason' in out.debug and 'prediction_invalid_reason' not in out.debug)
    monitor=fixture();monitor.set_task_request(dict(mode='monitor',motion_target=[.1,0.,0.],executed_motion_reference=[.1,0.,0.]));monitor.state.update=lambda f,dt:None
    first=monitor.step(SimpleNamespace(time=8.,phase='HOLDING'))
    monitor.set_task_request(dict(mode='hold',motion_target=[.1,0.,0.],executed_motion_reference=[.1,0.,0.]))
    second=monitor.step(SimpleNamespace(time=8.005,phase='HOLDING'))
    checks['release_monitor_latches_permission_without_regrip']=bool(np.isnan(first.predicted_normal).all() and np.isnan(second.predicted_normal).all() and np.all(second.delta_closure<=0.))
    import copy
    global solve_qp
    original=solve_qp;calls=[0]
    def false_residual(*args,**kwargs):
        result=original(*args,**kwargs);calls[0]+=1
        if calls[0]==4:
            result=copy.deepcopy(result);result.x=result.x.copy();result.x[0]+=100.;result.residual=0.
        return result
    try:
        solve_qp=false_residual;adversarial=fixture(limits=1.95);adversarial._decision()
    finally:solve_qp=original
    checks['solver_report_cannot_override_checked_actual_matrices']=bool(adversarial.debug['qp_priority_fallbacks'][-1]=='checked_previous_layer' and adversarial.debug['final_priority_residual']<=2e-5)
    boost=ForceQualityTask(quality.g);quality.contacts.snapshot.mode[:]='INCIPIENT'
    proposed,_,_=boost.propose(quality.state,quality.contacts.snapshot,np.ones(5,bool),np.full(5,5.),18.)
    boost.reconcile(np.zeros(8));checks['unexecuted_slip_boost_does_not_accumulate']=boost.boost==0.
    proposed,_,_=boost.propose(quality.state,quality.contacts.snapshot,np.ones(5,bool),np.full(5,5.),18.)
    boost.reconcile(np.r_[proposed,np.zeros(3)]);checks['slip_boost_requires_actual_realization']=boost.boost>0.
    gap=ForceQualityTask(quality.g);quality.contacts.snapshot.mode[:]='STICK_CONFIRMED'
    gap.propose(quality.state,quality.contacts.snapshot,np.ones(5,bool),np.full(5,5.),18.)
    gap.observation_missing();quality.state.time=9.2
    gap.propose(quality.state,quality.contacts.snapshot,np.ones(5,bool),np.full(5,5.),18.)
    checks['ratio_learning_dwell_cannot_span_unknown_observation_gap']=gap.target_ratio is None and gap.stable_since==9.2
    noise_model=fixture().model;public_before=noise_model.error_by_horizon.copy()
    noise_model.set_sensor_characteristics(dict(force_sigma_N=.02,tangential_sigma_N=.01,position_sigma_m=.001,
        auxiliary_force_sigma_N=.03,auxiliary_moment_sigma_Nm=.002))
    M=np.zeros((6,5));M[2]=1.;M[3]=.1
    geometry=SimpleNamespace(normal_wrench_map=M,tangent_basis=np.tile(np.array([[1.,0.,0.],[0.,1.,0.]]),(5,1,1)),
        contact_pos=np.tile([.1,0.,0.],(5,1)),finger_non_normal_by_finger=np.zeros((5,6)),normal=np.full(5,2.),contact_present=np.ones(5,bool))
    noise_model._wrench_noise(geometry);cov=np.array(noise_model.noise_diagnostics['covariance'])
    checks['wrench_device_noise_propagates_normal_tangent_position_and_aux_units']=bool(
        np.isclose(cov[2,2],5.*.02**2+.03**2) and cov[0,0]>.03**2 and cov[4,4]>.002**2
        and np.linalg.eigvalsh(cov).min()>-1e-12)
    checks['noise_propagation_does_not_truncate_calibration_error']=bool(np.array_equal(noise_model.error_by_horizon,public_before))
    if not all(checks.values()):raise AssertionError(checks)
    return dict(passed=True,checks=checks,scope='pure algebra/public interface; no physical stepping or performance claim')


if __name__=='__main__':
    import json
    print(json.dumps(self_test(),ensure_ascii=False))

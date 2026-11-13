"""V7: single posterior -> reachable aggregate task -> actual finite execution.

The ordinary reference is an aggregate grip task, not five permanent force
points. Mechanical limits enter before optimization. The only force posterior
is UnifiedBelief; robot/slip bookkeeping does not estimate pressure again.
Unknown motion response is never a zero physical gain: its public nominal
motion is fixed externally and pressure allocation is labelled empirical.
"""
from dataclasses import asdict
from copy import deepcopy
from types import SimpleNamespace
import hashlib,json
import numpy as np
import research_imports
from vendor.frozen_controllers import make_gains,ControlAction
from force_model import ForceResponse
from task_contract import validate_task_request
from grasp_contacts import PersistentGraspContactObserver
from .belief import UnifiedBelief
from .feedback import ReachableTaskPlanner,ModelDomainUnavailable

GAIN_KEYS=('base','ratio_floor','ratio_kp','ratio_ki','slip_boost_rate','baseline_tracking_gain','release_rate')


class RobotContext:
    """Public encoders, permissions and slip filter; NO other force estimate."""
    def __init__(self,owner):
        self.owner=owner;self.valid=np.zeros(5,bool);self.slip=np.zeros(5)
        self.slip_variance=np.zeros(5)
        self._last_sample=-np.inf;self.alpha=np.zeros(5)
        self.closure_min=np.zeros(5);self.closure_max=np.ones(5)
        self.closure_permission=np.ones(5,bool);self.nominal_weights=np.ones(5)
        self.load=np.nan;self.auxiliary_support=np.nan;self.time=0.
    @property
    def normal(self):
        b=self.owner.belief.snapshot
        return np.full(5,np.nan) if b is None else b.mean16[:5].copy()
    @property
    def force_uncertainty(self):
        b=self.owner.belief.snapshot
        return np.full(5,np.nan) if b is None else np.sqrt(np.maximum(0.,np.diag(b.covariance)[:5]))
    @property
    def contact_model_active(self):
        b=self.owner.belief.snapshot
        return np.zeros(5,bool) if b is None else b.active_contact5.copy()
    @property
    def posterior_mean_time_s(self):
        return None if self.owner.belief.snapshot is None else self.owner.belief.snapshot.mean_time
    def update(self,frame,dt):
        o=self.owner;self.time=float(frame.time);self.alpha=np.asarray(frame.alpha,float).copy()
        for key in ('closure_min','closure_max','closure_permission'):
            setattr(self,key,np.asarray(getattr(frame,key),bool if key=='closure_permission' else float).copy())
        self.nominal_weights=np.asarray(frame.nominal_force_weights,float).copy()
        self.valid=np.asarray(frame.valid,bool)&np.isfinite(frame.normal)&(np.asarray(frame.age)<=.12)
        motion=getattr(frame,'executed_motion_reference',None)
        if motion is None:
            if not o.motion_initialized:raise ValueError('Warmup requires public executed_motion_reference')
            motion=o.executed_motion
        self.executed_motion=np.asarray(motion,float).copy()
        o.executed_motion=self.executed_motion.copy();o.motion_initialized=True
        history_scope=np.ones(8,bool)
        if o.belief.input_history:
            previous=o.belief.input_history[-1]
            change=np.r_[self.alpha,self.executed_motion]-previous[1:]
            for i,d in enumerate(o.belief.response.available_directions):
                if change[i]>1e-10 and not d['plus'] or change[i]<-1e-10 and not d['minus']:history_scope[i]=False
        b=o.belief.update(frame,self.executed_motion,known_history_input8=history_scope)
        o.latest_frame=frame
        acquired=float(frame.sample_time)
        if acquired>self._last_sample+1e-10:
            gap=.005 if not np.isfinite(self._last_sample) else acquired-self._last_sample
            a=1.-np.exp(-min(.12,max(.0001,gap))/max(.001,o.g.sensor_tau))
            raw=np.asarray(frame.slip_magnitude,float)
            ok=self.valid&np.isfinite(raw)
            self.slip[ok]+=(np.maximum(0.,raw[ok])-self.slip[ok])*a
            sigma=float(getattr(frame,'sensor_characteristics',{}).get('slip_sigma_m_s',.0002))
            self.slip_variance[ok]=(1.-a)**2*self.slip_variance[ok]+a*a*sigma*sigma
            self._last_sample=acquired
        self.load=float(frame.load_estimate) if frame.load_valid else np.nan
        self.auxiliary_support=(float(np.asarray(frame.support_axis)@np.asarray(frame.auxiliary_wrench)[:3])
            if frame.auxiliary_valid else np.nan)
        o.grasp.update(frame,allow_enrollment=o.execution_anchor is None)
        return b


class AggregateGripReference:
    """One causal ratio reference with output-space execution reconciliation."""
    def __init__(self,gains):
        self.g=gains;self.scalar=gains.base;self.boost=0.;self.integral=0.
        self.target_ratio=None;self.stable_since=None;self.last_time=None
        self.samples=[];self.pending=None;self.last_confirmation=None
    def missing(self):
        self.stable_since=None;self.samples=[];self.pending=None
    def reconcile(self,actual,planned,debug):
        self.last_confirmation=None
        if self.pending is None:return
        complete=actual is not None and planned is not None
        if complete:
            error=float(np.max(abs(np.asarray(actual)-planned)))
            reachable=bool(debug.get('aggregate_task_available'))
            if reachable:
                gap=np.asarray(debug['governed_projection_gap'])
                sigma=np.asarray(debug['aggregate_reference_sigma_N'])
                reachable=bool(np.all(abs(gap)<=sigma+5e-6))
            commit=bool(error<1e-7 and reachable)
            self.last_confirmation=dict(actual_input_error=error,reference_reachable=reachable,committed=commit)
            if commit:self.integral,self.boost=self.pending
        self.pending=None
    def propose(self,now,belief,group,known_support,slip,spec,fresh_measurement,slip_variance=None):
        period=self.g.decision_period
        if self.last_time is not None and now-self.last_time>1.5*period:self.missing()
        self.last_time=now
        available=bool(group.any() and fresh_measurement and np.all(known_support[group]) and np.all(belief.valid16[:10][np.r_[group,group]]))
        sufficient=bool(known_support[0] and np.count_nonzero(known_support)>=3)
        sigma=float(spec.get('slip_sigma_m_s',.0002))
        bias=sigma/np.sqrt(2.*np.pi)
        # Statistical proxy, not an identified friction coefficient or proof.
        filtered_sd=(np.full(5,sigma) if slip_variance is None else np.sqrt(np.maximum(0.,slip_variance)))
        speed_lower=np.maximum(0.,slip-bias-3.*filtered_sd)
        speed_upper=np.maximum(0.,slip-bias)+3.*filtered_sd
        slipping=bool(np.any((speed_lower>.004)&known_support))
        locally_quiet=bool(np.all(speed_upper[known_support]<.002))
        ratio=None
        if not available:self.stable_since=None;self.samples=[]
        if available:
            shear=float(belief.mean16[5:10][group].sum());normal=float(belief.mean16[:5][group].sum())
            ratio=shear/max(.25,normal);self.samples.append(ratio);self.samples=self.samples[-5:]
            ratio=float(np.mean(self.samples))
        pending=self.integral;boost=self.boost
        if self.target_ratio is None:
            if not available or not sufficient or slipping or not locally_quiet:
                self.stable_since=None
                if slipping:boost=min(5.-self.g.base,self.boost+self.g.slip_boost_rate*period)
            else:
                self.boost*=np.exp(-period/1.5);boost=self.boost
                if self.stable_since is None:self.stable_since=now
                if now-self.stable_since>=1.-1e-9:
                    self.target_ratio=max(.025,ratio);self.scalar=self.g.base+self.boost
            scalar=self.g.base+boost
        elif available:
            error=ratio-self.target_ratio
            candidate=float(np.clip(self.integral+error*period,-.35,.75))
            request=self.scalar+self.g.ratio_kp*error+self.g.ratio_ki*candidate
            if self.g.ratio_floor<=request<=5. or request>5. and error<0. or request<self.g.ratio_floor and error>0.:
                pending=candidate
            scalar=float(np.clip(request,self.g.ratio_floor,5.))
        else:scalar=self.scalar
        self.pending=(pending,boost) if available else None
        return scalar,dict(target_ratio=self.target_ratio,ratio=ratio,ratio_observation_complete=available,
            required_group=group.tolist(),supported_group=known_support.tolist(),reference_scalar=scalar,
            integral=self.integral,boost=self.boost,actual_execution_confirmation=self.last_confirmation)


class JointTaskController:
    def __init__(self,calibration,limits,budget,gains,model_document=None):
        self.calibration=deepcopy(calibration);self.B=np.asarray(calibration['force_matrix'],float)
        configured=dict(gains);configured.update(force_gain=tuple(np.maximum(.05,np.linalg.norm(self.B,axis=0))),force_tau=float(calibration['force_tau']))
        self.g=make_gains(configured);self.limits=np.asarray(limits,float);self.budget=float(budget)
        self.belief=UnifiedBelief(calibration,model_document=model_document)
        self.grasp=PersistentGraspContactObserver();self.reference=AggregateGripReference(self.g)
        self.planner=ReachableTaskPlanner();self.state=RobotContext(self)
        self.motion_initialized=False;self.executed_motion=np.zeros(3);self.motion_reference=np.zeros(3)
        self.task_request={};self.latest_frame=None;self.last_decision=-np.inf;self.previous_time=None
        self.execution_anchor=None;self.motion_anchor=None;self.required_group=None
        self.interval_action=np.zeros(8);self.last_debug={};self.debug={};self.release_latched=False
        self.desired=np.zeros(5);self.predicted=np.full(5,np.nan);self.label='v7_initializing'
        self.numerical_failures=0;self.infeasible_ticks=0;self.last_plan_time=None
        self.nominal_feedback_contract=dict(schema='v7-explicit-reference-contract',values={k:float(gains[k]) for k in GAIN_KEYS})
    def set_task_request(self,value):
        self.task_request=validate_task_request(value)
        self.executed_motion=np.asarray(value['executed_motion_reference'],float).copy();self.motion_initialized=True
        self.motion_reference=self.executed_motion.copy()
        if value['mode'] in ('release','monitor'):self.release_latched=True
    def _bounds(self,h):
        s=self.state;r=self.belief.response;dt=self.g.decision_period
        al=s.closure_min.copy();au=s.closure_max.copy()
        limits=self.task_request.get('motion_reference_bounds')
        if limits is None:raise ValueError('V7 needs the actual public robot motion authority')
        ml=np.asarray(limits['lower'],float);mu=np.asarray(limits['upper'],float)
        motion_rates=np.asarray(getattr(self.latest_frame,'motion_reference_rate_limits',(.035,.12,.12)),float)
        if motion_rates.shape!=(3,) or not np.isfinite(motion_rates).all() or np.any(motion_rates<=0.):raise ValueError('Declared V7 motion rate envelope required')
        rates=np.r_[np.full(5,.8),motion_rates]*dt
        lower=np.maximum(-rates,np.asarray(r.input_lower));upper=np.minimum(rates,np.asarray(r.input_upper))
        for i,d in enumerate(r.available_directions):
            if not d['plus']:
                upper[i]=min(rates[i],r.input_step_limits[i])
                if i<5:au[i]=min(au[i],self.execution_anchor[i])
                else:mu[i-5]=min(mu[i-5],self.motion_anchor[i-5])
            if not d['minus']:
                lower[i]=-min(rates[i],r.input_step_limits[i])
                if i<5:al[i]=max(al[i],self.execution_anchor[i])
                else:ml[i-5]=max(ml[i-5],self.motion_anchor[i-5])
        lower[~r.identified_inputs]=0.;upper[~r.identified_inputs]=0.
        upper[:5][~s.valid|~s.closure_permission]=np.minimum(0.,upper[:5][~s.valid|~s.closure_permission])
        user=np.tile(np.asarray(self.task_request['motion_target'],float),(h,1))
        preview=self.task_request.get('motion_preview');source='current point hold assumption; future unknown'
        if preview is not None and preview['future_reference_available']:
            times=np.asarray(preview['requested_times_s']);points=np.asarray(preview['motion_targets'])
            query=float(self.latest_frame.time)+np.arange(1,h+1)*dt
            user=np.column_stack([np.interp(query,times,points[:,i]) for i in range(3)])
            source='explicit public user preview'
        nominal=[];m=self.executed_motion.copy()
        for target in user:
            dm=np.minimum(upper[5:],np.maximum(lower[5:],target-m))
            dm=np.minimum(mu-m,np.maximum(ml-m,dm));nominal.append(dm);m+=dm
        return al,au,lower,upper,ml,mu,user,np.asarray(nominal),source
    def _decision(self):
        b=self.belief.snapshot;now=float(self.latest_frame.time);h=int(self.g.horizon)
        if self.execution_anchor is None:
            self.grasp.freeze_handoff();self.execution_anchor=self.state.alpha.copy();self.motion_anchor=self.executed_motion.copy()
            self.required_group=self.grasp.snapshot.anchor_count>0
            if not self.required_group.any():self.required_group=np.ones(5,bool)
        actual=None if self.last_plan_time is None else self.belief.actual_increment(self.last_plan_time,now)
        self.reference.reconcile(actual,self.interval_action,self.last_debug)
        hyp=self.grasp.snapshot;spec=self.task_request['sensor_characteristics']
        new_contact=hyp.new_normal>max(.04,3.*float(spec['force_sigma_N']))
        known=hyp.known_support&hyp.fresh&hyp.valid&~new_contact
        self.belief.set_mode_evidence(not new_contact.any(),('new_unaffiliated_contact',) if new_contact.any() else ())
        f=self.belief.forecast(h,self.g.decision_period)
        al,au,lower,upper,ml,mu,user,nominal_motion,preview_source=self._bounds(h)
        scalar,reference_debug=self.reference.propose(now,b,self.required_group,known,self.state.slip,spec,
            bool(np.all(self.state.valid[self.required_group])),self.state.slip_variance)
        target=scalar*float(self.state.nominal_weights[self.required_group].sum())
        caps=np.minimum(self.limits,np.asarray(self.task_request['limits'],float));budget=min(self.budget,self.task_request['budget'])
        idx=np.concatenate([16*j+np.arange(5) for j in range(h)])
        joint_idx=np.concatenate([16*j+np.arange(5) for j in range(h+1)])
        known_input=np.asarray(f.usable_input_mask,bool)
        moving_unknown=bool(np.any((abs(nominal_motion)>1e-10)&~known_input[5:]))
        model=bool(f.model_usable and not moving_unknown)
        reasons=list(f.reasons)+(['future_motion_response_unknown'] if moving_unknown else [])
        p=b.mean16[:5].copy();valid=b.valid16[:5]&np.isfinite(p)&(p>=0.)
        raw_valid=b.observed_pressure_valid5.copy()
        # Missing rows are removed from allocation, not treated as zero load.
        # These numerical placeholders never enter a total-force task/budget.
        work=np.where(valid,p,0.);cov=b.covariance[:5,:5].copy()
        cov[~valid,:]=0.;cov[:,~valid]=0.
        covariance=np.nan_to_num(cov,nan=0.)
        fixed_motion=nominal_motion if not model or not known_input[5:].all() else None
        if model:
            # Identified future directions are narrower than the physical
            # return-to-visited-anchor envelope. Never call an unmeasured
            # positive return a calibrated response.
            lower=np.maximum(lower,self.belief.response.input_lower)
            upper=np.minimum(upper,self.belief.response.input_upper)
        def empirical_allocation():
            empirical=ForceResponse(self.B,float(self.calibration['force_tau']))
            grip,free=empirical.action_prediction(work,h,self.g.decision_period,np.ones(5,bool),alpha=self.state.alpha,latent=self.state.alpha)
            allocation=np.zeros((5*h,8*h))
            for j in range(h):
                for k in range(h):allocation[5*j:5*j+5,8*k:8*k+5]=grip[5*j:5*j+5,5*k:5*k+5]
            return allocation,free,np.tile(np.asarray(self.calibration['response_error_bound']),h),np.kron(np.ones((h,h)),covariance)
        if model:
            P=f.P[idx];c=f.c[idx];cal=f.calibration_error[idx]
            fc=f.covariance[np.ix_(idx,idx)];jc=f.joint_covariance[np.ix_(joint_idx,joint_idx)]
        else:
            P,c,cal,fc=empirical_allocation();jc=None
        kwargs=dict(pressure_now5=work,pressure_covariance5x5=covariance,pressure_map=P,
            pressure_free5H=c,calibration_error5H=cal,observed_pressure5=b.observed_pressure5,
            pressure_valid5=valid,observed_valid5=raw_valid,limits5=caps,budget=budget,
            alpha5=self.state.alpha,alpha_lower5=al,alpha_upper5=au,step_lower8=lower,step_upper8=upper,
            motion_now3=self.executed_motion,user_targetsH3=user,motion_lower3=ml,motion_upper3=mu,
            tracking_tolerances3=self.task_request['tracking_tolerances'],persistent5=known,
            aggregate_mask5=self.required_group,force_total_target=target,previous_action8=actual,
            aggregate_task_enabled=bool(np.all(known[self.required_group])),
            model_usable=model,forecast_covariance=fc,joint_pressure_covariance=jc,
            fixed_motion_planH3=fixed_motion,reference_fraction=self.g.baseline_tracking_gain)
        try:plan=self.planner.solve(**kwargs)
        except ModelDomainUnavailable as exc:
            # Retry the explicit empirical task in the SAME actual authority;
            # a rejected affine forecast is not a physical task impossibility.
            reasons.append(str(exc));model=False
            al,au,lower,upper,ml,mu,user,nominal_motion,preview_source=self._bounds(h)
            P,c,cal,fc=empirical_allocation()
            kwargs.update(pressure_map=P,pressure_free5H=c,calibration_error5H=cal,
                forecast_covariance=fc,joint_pressure_covariance=None,
                fixed_motion_planH3=nominal_motion,model_usable=False,step_lower8=lower,step_upper8=upper)
            plan=self.planner.solve(**kwargs)
        self.interval_action=plan.first_action8.copy();self.predicted=plan.predicted_pressure[0].copy()
        self.desired=np.where(valid,work,0.) # plotted observation reference, not five independent force targets
        self.label='v7_reachable_joint_task' if model else 'v7_empirical_reachable_feedback'
        self.debug=dict(plan.debug,controller_architecture='single_belief_reachable_aggregate_task_v7',
            prediction_invalid_reason=';'.join(reasons) if not model else '',
            belief_mean_time_s=b.mean_time,belief_current_time_s=b.current_time,
            force_posterior_source='UnifiedBelief.mean16; same value used by allocation and forecast',
            response_scope=known_input.tolist(),model_diagnostics=f.diagnostics,
            preview_source=preview_source,reference_task=reference_debug,
            nominal_feedback_contract=self.nominal_feedback_contract,
            grasp_contact_status=hyp.status.tolist(),new_external_contact=new_contact.tolist(),
            empirical_unknown_motion_is_fixed=not model,
            unmeasured_return_scope='empirical mechanical return only, no affine forecast; history records unknown sign',
            applied_input_increment_8=self.interval_action.tolist())
        self.last_debug=self.debug.copy();self.last_plan_time=now
    def step(self,frame,phase=None,command='hold'):
        now=float(frame.time);dt=.005 if self.previous_time is None else now-self.previous_time
        if not .0001<=dt<=.05:raise ValueError('V7 requires chronological actual execution clocks')
        self.previous_time=now;self.state.update(frame,dt)
        if command=='release' or str(phase or frame.phase).upper() in ('RELEASE','RELEASING') or self.task_request.get('mode') in ('release','monitor'):
            self.release_latched=True
        if self.release_latched:
            self.reference.missing();self.interval_action[:]=0.;self.predicted[:]=np.nan;self.motion_reference=self.executed_motion.copy()
            delta=-np.minimum(np.maximum(0.,self.state.alpha-self.state.closure_min),self.g.release_rate*dt)
            self.desired=np.zeros(5);self.label='v7_release_authority'
            return ControlAction(delta,self.predicted.copy(),self.label,False,dict(release_authority=True,prediction_available=False,
                prediction_unavailable_reason='release outside held-response model; native finite mechanical path'))
        b=self.belief.snapshot
        if not self.state.valid.any():
            self.reference.missing();self.interval_action[:]=0.;self.predicted[:]=np.nan;self.motion_reference=self.executed_motion.copy()
            self.label='v7_observation_unavailable';self.debug=dict(prediction_available=False,prediction_unavailable_reason='all pressure posterior channels unavailable')
            return ControlAction(np.zeros(5),self.predicted.copy(),self.label,False,self.debug.copy())
        if now-self.last_decision>=self.g.decision_period-1e-8:
            self._decision();self.last_decision=now
        fraction=min(1.,dt/self.g.decision_period);delta=self.interval_action[:5]*fraction
        # Bounds were solved before mapping. A late permission change does not
        # silently validate a modified plan; stop forbidden closing and record.
        executed=np.minimum(self.state.closure_max-self.state.alpha,np.maximum(self.state.closure_min-self.state.alpha,delta))
        blocked=~self.state.valid|~self.state.closure_permission;executed[blocked]=np.minimum(0.,executed[blocked])
        self.motion_reference=self.executed_motion+self.interval_action[5:]*fraction
        self.debug['late_permission_projection_max']=float(np.max(abs(executed-delta)))
        self.debug['actual_subinterval_delta_closure']=executed.tolist()
        if self.debug['late_permission_projection_max']>1e-10:
            self.predicted[:]=np.nan;self.debug['prediction_available']=False
            self.debug['prediction_unavailable_reason']='actual execution changed after planning; await same-history replan'
            self.debug['planned_ramp_executed']=False
        return ControlAction(executed,self.predicted.copy(),self.label,False,self.debug.copy())


def create_joint(calibration,limits,budget,gains=None):
    if not isinstance(gains,dict) or any(k not in gains for k in GAIN_KEYS):
        raise ValueError('V7 requires the complete explicit admitted reference configuration')
    values=np.array([gains[k] for k in GAIN_KEYS],float)
    if not np.isfinite(values).all() or np.any(values<0.) or not 0.<gains['ratio_floor']<=gains['base']<=5.:
        raise ValueError('Invalid reference contract or discontinuous base/floor')
    ctl=JointTaskController(calibration,limits,budget,gains)
    if any(float(getattr(ctl.g,k))!=float(gains[k]) for k in GAIN_KEYS):raise ValueError('Reference configuration was not preserved')
    ctl.nominal_feedback_contract['sha256']=hashlib.sha256(json.dumps(ctl.nominal_feedback_contract['values'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return ctl

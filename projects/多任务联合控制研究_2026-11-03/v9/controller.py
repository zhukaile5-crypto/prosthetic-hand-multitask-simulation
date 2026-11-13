"""V9: retain stabilizing feedback; project it into the multi-task set.

There is ONE decision force state: the unchanged admitted ratio controller's
coupled fixed-lag observer. No second total-pressure model runs alongside it.
The response used by the supervisor is an empirical allocation surrogate,
never a micro-force forecast, physical covariance or safety certificate.
"""
from copy import deepcopy
import numpy as np
import research_imports
from controller_adapter import create_controller
from vendor.frozen_controllers import ControlAction
from force_model import ForceResponse
from task_contract import validate_task_request
from grasp_contacts import PersistentGraspContactObserver
from .feedback import ReachableTaskPlanner

GAIN_KEYS=('base','ratio_floor','ratio_kp','ratio_ki','slip_boost_rate','baseline_tracking_gain','release_rate')
REFERENCE_MEMORY=('scalar','boost','target_ratio','samples','stable_since','integral')

class DecisionState:
    """Expose the original observer, and add robot/contact bookkeeping only."""
    def __init__(self,owner,state):
        self.owner=owner;self.force_state=state;self.slip_measurement_variance=np.zeros(5)
    def __getattr__(self,key):return getattr(self.force_state,key)
    @property
    def posterior_mean_time_s(self):return float(self.force_state.time)
    def update(self,frame,dt):
        self.force_state.update(frame,dt)
        o=self.owner;o.latest_frame=frame
        o.executed_motion=np.asarray(frame.executed_motion_reference,float).copy()
        a=1.-np.exp(-dt/max(.001,o.g.sensor_tau))
        sigma=float(frame.sensor_characteristics.get('slip_sigma_m_s',.0002))
        valid=self.force_state.valid
        self.slip_measurement_variance[valid]=(1.-a)**2*self.slip_measurement_variance[valid]+a*a*sigma*sigma
        o.grasp.update(frame,allow_enrollment=o.required_group is None)

class JointTaskController:
    def __init__(self,calibration,limits,budget,gains):
        from .runtime import PublicPreviewFollower
        self.calibration=deepcopy(calibration);self.B=np.asarray(calibration['force_matrix'],float)
        self.nominal=create_controller('ratio',calibration,limits,budget,deepcopy(gains));self.g=self.nominal.g
        self.grasp=PersistentGraspContactObserver();self.required_group=None
        self.state=DecisionState(self,self.nominal.state);self.nominal.state=self.state
        self.planner=ReachableTaskPlanner();self.follower=PublicPreviewFollower()
        self.limits=np.asarray(limits,float);self.budget=float(budget)
        self.task_request={};self.latest_frame=None;self.executed_motion=np.zeros(3);self.motion_reference=np.zeros(3)
        self.desired=np.zeros(5);self.predicted=np.full(5,np.nan);self.last_decision=-np.inf
        self.interval_action=np.zeros(8);self.use_nominal_subinterval=True
        self.interval_reference_before=None
        self.recovery_motion_hold=False;self.release_latched=False;self.debug={};self.last_debug={}
        self.label='v9_initializing';self.numerical_failures=0;self.infeasible_ticks=0
        self.nominal_feedback_contract=dict(schema='v9-retained-stabilizing-proposal',values=deepcopy(gains),
            force_state='unchanged baseline coupled fixed-lag observer; unique decision pressure state')
    def set_task_request(self,value):
        self.task_request=validate_task_request(value)
        self.executed_motion=np.asarray(value['executed_motion_reference'],float).copy()
        if value['mode'] in ('release','monitor'):self.release_latched=True
    def _known(self):
        if self.required_group is None:
            self.grasp.freeze_handoff();self.required_group=self.grasp.snapshot.anchor_count>0
            if not self.required_group.any():self.required_group=np.ones(5,bool)
        hyp=self.grasp.snapshot;spec=self.latest_frame.sensor_characteristics
        novel=hyp.new_normal>max(.04,3.*float(spec['force_sigma_N']))
        return hyp.known_support&hyp.fresh&hyp.valid&~novel,novel
    def _risk(self,known):
        f=self.latest_frame;spec=f.sensor_characteristics;s=self.state
        n=np.asarray(f.normal,float);valid=np.asarray(f.valid,bool)&np.isfinite(n)&(np.asarray(f.age)<=.12)
        caps=np.minimum(self.limits,np.asarray(self.task_request['limits'],float));budget=min(self.budget,self.task_request['budget'])
        sigma=float(spec['force_sigma_N'])
        risk=bool(np.any(valid&(n-3.*sigma>caps)))
        clear=bool(valid.all() and np.all(n+3.*sigma<caps))
        if valid.all():
            risk|=bool(n.sum()-3.*np.sqrt(5.)*sigma>budget)
            clear&=bool(n.sum()+3.*np.sqrt(5.)*sigma<budget)
        slip=np.asarray(s.slip_magnitude,float);sd=np.sqrt(np.maximum(0.,s.slip_measurement_variance))
        bias=float(spec.get('slip_sigma_m_s',.0002))/np.sqrt(2.*np.pi)
        lower=np.maximum(0.,slip-bias-3.*sd);upper=np.maximum(0.,slip-bias)+3.*sd
        slipping=bool(np.any(known&(lower>.004)));quiet=bool(known.any() and np.all(upper[known]<.002))
        if risk or slipping:self.recovery_motion_hold=True
        elif clear and quiet:self.recovery_motion_hold=False
        return caps,budget,dict(material_risk=risk,material_clear=clear,slip_risk=slipping,slip_clear=quiet)
    def _motion(self,h,paused):
        request=self.task_request;s=self.state;dt=self.g.decision_period
        bounds=request['motion_reference_bounds'];ml=np.asarray(bounds['lower']);mu=np.asarray(bounds['upper'])
        rates=np.asarray(self.latest_frame.motion_reference_rate_limits,float)
        user=np.tile(np.asarray(request['motion_target'],float),(h,1));preview=request.get('motion_preview')
        if preview and preview['future_reference_available']:
            times=np.asarray(preview['requested_times_s']);points=np.asarray(preview['motion_targets'])
            query=float(self.latest_frame.time)+np.arange(1,h+1)*dt
            user=np.column_stack([np.interp(query,times,points[:,i]) for i in range(3)])
        motion=[];m=self.executed_motion.copy()
        for target in user:
            dm=np.zeros(3) if paused else np.minimum(rates*dt,np.maximum(-rates*dt,target-m))
            dm=np.minimum(mu-m,np.maximum(ml-m,dm));motion.append(dm);m+=dm
        lo=np.r_[np.full(5,-min(.8,self.g.open_rate)*dt),-rates*dt]
        hi=np.r_[np.full(5,min(.8,self.g.close_rate)*dt),rates*dt]
        return ml,mu,user,np.asarray(motion),lo,hi
    def _project(self,known,novel,caps,budget,risk,reference_before):
        s=self.state;h=self.g.horizon;dt=self.g.decision_period
        paused=bool(self.recovery_motion_hold or self.nominal.pause_transport)
        ml,mu,user,motion,lo,hi=self._motion(h,paused)
        hi[:5][~s.valid|~s.closure_permission|~known]=np.minimum(0.,hi[:5][~s.valid|~s.closure_permission|~known])
        p=np.asarray(s.normal,float);valid=s.valid&np.isfinite(p)&(p>=0.)
        work=np.where(valid,p,0.);raw=np.asarray(self.latest_frame.normal,float)
        raw_valid=np.asarray(self.latest_frame.valid,bool)&np.isfinite(raw)&(np.asarray(self.latest_frame.age)<=.12)
        response=ForceResponse(self.B,float(self.calibration['force_tau']))
        grip,c=response.action_prediction(work,h,dt,np.ones(5,bool),alpha=s.alpha,latent=s.alpha)
        P=np.zeros((5*h,8*h))
        for j in range(h):
            for k in range(h):P[5*j:5*j+5,8*k:8*k+5]=grip[5*j:5*j+5,5*k:5*k+5]
        preferred=np.zeros((h,8));preferred[0,:5]=self.nominal.interval_action;preferred[:,5:]=motion
        minimum=self.g.ratio_floor*float(s.nominal_weights[self.required_group].sum())
        if risk['slip_risk'] and np.all(valid[self.required_group]):
            minimum=max(minimum,float(work[self.required_group].sum())+
                self.g.slip_boost_rate*dt*float(s.nominal_weights[self.required_group].sum()))
        plan=self.planner.solve(pressure_now5=work,pressure_covariance5x5=np.zeros((5,5)),
            pressure_map=P,pressure_free5H=c,calibration_error5H=np.tile(s.response_error_bound,h),
            observed_pressure5=raw,pressure_valid5=valid,observed_valid5=raw_valid,limits5=caps,budget=budget,
            alpha5=s.alpha,alpha_lower5=s.closure_min,alpha_upper5=s.closure_max,step_lower8=lo,step_upper8=hi,
            motion_now3=self.executed_motion,user_targetsH3=user,motion_lower3=ml,motion_upper3=mu,
            tracking_tolerances3=self.task_request['tracking_tolerances'],persistent5=known,
            aggregate_mask5=self.required_group,force_total_target=minimum,
            aggregate_task_enabled=bool(np.all(known[self.required_group])),model_usable=False,
            fixed_motion_planH3=motion,preferred_planH8=preferred)
        self.interval_action=plan.first_action8.copy()
        self.use_nominal_subinterval=bool(plan.debug['nominal_proposal_accepted_exact'])
        modified=bool(not self.use_nominal_subinterval or novel.any())
        self.interval_reference_before=deepcopy(reference_before)
        # The ratio observer has consumed the actual frame, not the proposal.
        # Rejected requests must not accumulate controller integral/boost.
        if modified:
            for name in ('integral','boost'):setattr(self.nominal,name,deepcopy(reference_before[name]))
        if not np.all(known[self.required_group]):
            for name in REFERENCE_MEMORY:setattr(self.nominal,name,deepcopy(reference_before[name]))
            if self.nominal.target_ratio is None:
                self.nominal.stable_since=None;self.nominal.samples.clear()
        self.debug=dict(plan.debug,controller_architecture='v9_retained_stabilizing_proposal_task_set',
            decision_force_state='unchanged baseline coupled fixed-lag observer; no second pressure estimate',
            pressure_allocation_role='uncertified empirical forward response; not a force forecast',
            conditional_measurement_covariance_available=False,
            reference_memory_rollback=modified,recovery_motion_hold=self.recovery_motion_hold,
            recovery_evidence=risk,new_unaffiliated_contact=novel.tolist(),known_grasp_support=known.tolist(),
            original_user_motion_and_deadline_retained=True,prediction_available=False,
            prediction_unavailable_reason='empirical task projection has no admitted physical force forecast',
            applied_input_increment_8=self.interval_action.tolist())
        self.last_debug=self.debug.copy();self.last_decision=float(self.latest_frame.time)
    def step(self,frame,phase=None,command='hold'):
        before={n:deepcopy(getattr(self.nominal,n)) for n in REFERENCE_MEMORY}
        if command=='release' or self.task_request.get('mode') in ('release','monitor') or str(phase or frame.phase).upper() in ('RELEASE','RELEASING'):
            self.release_latched=True
        previous_decision=self.nominal.last_decision
        action=self.nominal.step(frame,phase,'release' if self.release_latched else command)
        self.desired=self.nominal.desired.copy();self.predicted[:]=np.nan
        if self.release_latched:
            self.follower.reset();self.motion_reference=self.executed_motion.copy();self.interval_action[:]=0.
            self.label='v9_release_authority'
            return ControlAction(action.delta_closure.copy(),self.predicted.copy(),self.label,False,
                dict(release_authority=True,prediction_available=False,prediction_unavailable_reason='native release outside empirical holding allocation'))
        known,novel=self._known();caps,budget,risk=self._risk(known)
        if not self.state.valid.any():
            self.follower.reset();self.motion_reference=self.executed_motion.copy();self.label='v9_unobservable_hold'
            return ControlAction(np.zeros(5),self.predicted.copy(),self.label,True,dict(prediction_available=False,observation_unavailable=True))
        if self.nominal.last_decision!=previous_decision:self._project(known,novel,caps,budget,risk,before)
        paused=bool(self.recovery_motion_hold or self.nominal.pause_transport)
        self.motion_reference=self.follower.step(self.task_request,pause_transport=paused)
        if self.use_nominal_subinterval:delta=action.delta_closure.copy()
        else:
            delta=self.interval_action[:5]*min(1.,float(frame.dt)/self.g.decision_period)
            delta=np.minimum(self.state.closure_max-self.state.alpha,np.maximum(self.state.closure_min-self.state.alpha,delta))
        before_permission=delta.copy()
        blocked=~self.state.valid|~self.state.closure_permission|~known;delta[blocked]=np.minimum(0.,delta[blocked])
        late_override=bool(np.max(abs(delta-before_permission))>1e-12)
        if late_override and self.interval_reference_before is not None:
            for name in ('integral','boost'):setattr(self.nominal,name,deepcopy(self.interval_reference_before[name]))
            self.debug['reference_memory_rollback']=True
        self.label='v9_nominal_stabilization' if self.use_nominal_subinterval else 'v9_task_set_intervention'
        self.debug.update(recovery_motion_hold=self.recovery_motion_hold,recovery_evidence=risk,
            late_permission_override=late_override,
            actual_subinterval_delta_closure=delta.tolist(),pause_transport=paused,
            prediction_available=False,nominal_label=self.nominal.label)
        return ControlAction(delta,self.predicted.copy(),self.label,paused,self.debug.copy())

def create_joint(calibration,limits,budget,gains=None):
    if not isinstance(gains,dict) or any(k not in gains for k in GAIN_KEYS):raise ValueError('Complete admitted baseline configuration required')
    if not 0.<gains['ratio_floor']<=gains['base']<=5.:raise ValueError('Invalid public reserve contract')
    return JointTaskController(calibration,limits,budget,gains)

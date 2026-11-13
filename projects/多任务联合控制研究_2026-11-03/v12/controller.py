"""One tactile measurement, force intent and real eight-reference actor.

Desired pressure is never a prediction. J maps a virtual contact-force error
to requested robot effort; known current-state servo algebra maps that effort
to real references. Neither map identifies a contact-force response gradient.
All methods use identical sensing, bounds, contact search and execution.
"""
from copy import deepcopy
import numpy as np
import research_imports
from vendor.frozen_controllers import make_gains, ControlAction, solve_qp
from task_contract import validate_task_request
from v10.robot_contact_space import RobotContactSpace, RobotJointSample, PublicSkinPatch
from v11.object_belief import RelativeObjectBelief, DeclaredShapeTemplate
from .observation import PublicTactileState, array, get
from .intents import RatioIntent, FixedIntent, project_force, RELEASE_PHASES, ACQUISITION
from .effort_tracker import EffortTracker

METHODS=('ratio_effort','fixed_effort','fusion_effort')
GAIN_KEYS=('base','ratio_floor','ratio_kp','ratio_ki','slip_boost_rate',
           'baseline_tracking_gain','release_rate')


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


class FusionIntent:
    """Conditional object-contact load allocation, not a force trajectory model.

Grip capacity, material budget, balance, effort economy and reference continuity
share five force-intent variables. Foreign-contact rejection is represented at
patch level in the common actor, never as a mechanical closing prohibition.
"""
    def __init__(self,gains,shape):
        self.g=gains;self.fallback=RatioIntent(gains);self.previous=None
        self.belief=(RelativeObjectBelief(DeclaredShapeTemplate(shape,.028,
            .047 if shape=='capsule' else None,
            source='V12 declared nominal template, fixed before trials; no runtime physical size'))
            if shape in ('sphere','capsule') else None)
        self.snapshot=None

    def step(self,state,phase,limits,budget,frame,patches):
        g=self.g;phase=str(phase).upper()
        fallback_reason=None
        if self.belief is None:fallback_reason='declared_template_unavailable'
        else:
            self.snapshot=self.belief.update(frame)
            if not self.snapshot.grouping_unambiguous or self.snapshot.selected_index is None:
                fallback_reason='conditional_object_group_unavailable'
        if not state.valid.all():fallback_reason='partial_pressure_observation'
        if not state.load_valid:fallback_reason='public_load_proxy_unavailable'
        if phase in ACQUISITION:fallback_reason='common_acquisition'
        if fallback_reason:
            out=self.fallback.step(state,phase,limits,budget,g.decision_period)
            out['debug'].update(fusion_mode='shared_ratio_fallback',fallback_reason=fallback_reason,
                conditional_group_identity_certified=False)
            return out,None
        hypothesis=self.snapshot.hypotheses[self.snapshot.selected_index]
        inliers=set(hypothesis.inlier_indices)
        groups={i:[] for i in range(5)};foreign=np.zeros(5)
        for p in patches:
            (groups[p['digit']].append(p) if p['index'] in inliers
             else None)
            if p['index'] not in inliers:foreign[p['digit']]+=p['normal']
        active=np.array([bool(groups[i]) for i in range(5)])
        # Require an opposed grasp group, not merely many taxels on one finger.
        if not active[0] or active.sum()<3:
            out=self.fallback.step(state,phase,limits,budget,g.decision_period)
            out['debug'].update(fusion_mode='shared_ratio_fallback',fallback_reason='grip_group_insufficient')
            return out,None
        cap=np.maximum(0.,np.asarray(limits)-g.force_margin-foreign)
        cap[~active]=0.
        available_budget=max(0.,float(budget)-2*g.force_margin-float(foreign.sum()))
        observed=np.array([sum(p['normal'] for p in groups[i]) for i in range(5)])
        center=np.mean([p['patch'].reference_position for p in patches if p['index'] in inliers],axis=0)
        wrench=np.zeros((6,5));capacity=np.zeros(5)
        for i in np.flatnonzero(active):
            for p in groups[i]:
                share=p['normal']/max(1e-12,observed[i]);n=p['patch'].reference_normal
                lever=p['patch'].reference_position-center
                wrench[:,i]+=share*np.r_[n,np.cross(lever,n)/g.characteristic_length]
                mu=state.mu[i]*g.friction_margin
                # Downward normal force consumes net upward support; it cannot
                # disappear merely because friction is represented as capacity.
                capacity[i]+=share*(n[2]+mu*np.sqrt(max(0.,1.-n[2]**2)))
        floor=project_force(np.where(active,g.support_contact_floor,0.),cap,available_budget)
        # Exact linear capacity maximum in the declared intent box/budget.
        maximum=floor.copy();remaining=max(0.,available_budget-float(maximum.sum()))
        for i in sorted(range(5),key=lambda j:-capacity[j]):
            if capacity[i]<=0:continue
            inc=min(max(0.,cap[i]-maximum[i]),remaining);maximum[i]+=inc;remaining-=inc
        max_capacity=float(capacity@maximum)
        slip=max(0.,float(state.slip_speed)-.004) if np.isfinite(state.slip_speed) else 0.
        demand=float(state.load+g.load_uncertainty_weight*state.load_uncertainty
                     +g.support_buffer+g.slip_load_gain*slip)
        goal=min(demand,max_capacity)
        start=maximum.copy()
        previous=observed if self.previous is None else self.previous
        # Normal horizontal resultant and moment: friction carries vertical load.
        balance=wrench[[0,1,3,4,5]]
        Q=2*((g.force_cost+g.reference_continuity)*np.eye(5)+g.balance_weight*balance.T@balance)
        q=-2*g.reference_continuity*previous
        A=np.vstack((np.eye(5),-np.eye(5),np.ones((1,5)),-capacity[None,:]))
        b=np.r_[cap,-floor,available_budget,-goal]
        result=solve_qp(Q,q,A,b,passes=g.qp_passes,tolerance=g.qp_tolerance,feasible_initial=start)
        if not result.converged or np.max(A@result.x-b)>1e-7:
            out=self.fallback.step(state,phase,limits,budget,g.decision_period)
            out['debug'].update(fusion_mode='shared_ratio_fallback',fallback_reason='intent_optimization_unresolved',
                optimization_status=result.status,physical_infeasibility_certified=False)
            return out,None
        desired=np.asarray(result.x).copy();self.previous=desired.copy()
        return dict(desired_normal5=desired,debug=dict(label='coupled_load_contact_allocation',
            fusion_mode='conditional_object_contact_allocation',object_group_inliers=sorted(inliers),
            conditional_object_force_observed5=observed,conditional_other_pressure5=foreign,
            force_cap_after_measured_other_pressure5=cap,capacity_per_normal5=capacity,
            support_demand_N=demand,intent_support_capacity_N=float(capacity@desired),
            intent_support_deficit_N=max(0.,demand-max_capacity),optimization_status=result.status,
            desired_force_is_prediction=False,physical_support_certified=False,
            conditional_group_identity_certified=False,pressure_prediction_available=False)),inliers


class TaskController:
    def __init__(self,method,calibration,limits,budget,gains,robot_model):
        if method not in METHODS:raise ValueError('Unknown V12 method')
        if not isinstance(gains,dict) or set(gains)!=set(GAIN_KEYS):raise ValueError('Actual seven baseline task gains required')
        self.method=method;self.g=make_gains(deepcopy(gains));self.state=PublicTactileState(self.g)
        self.geometry=RobotContactSpace(robot_model);self.tracker=EffortTracker(robot_model)
        self.intent=(FusionIntent(self.g,calibration['config']['shape']) if method=='fusion_effort'
            else RatioIntent(self.g) if method=='ratio_effort' else FixedIntent(self.g))
        self.limits=array(limits,(5,),'limits',finite=True);self.budget=float(budget)
        self.desired=np.zeros(5);self.predicted=np.full(5,np.nan);self.motion_reference=np.zeros(3)
        self.task_request=None;self.last_decision=-np.inf;self.release_latched=False
        self.plan_start=None;self.plan_reference=None;self.plan_delta=np.zeros(8)
        self.plan_ratio_memory=None
        self.debug={};self.last_debug={};self.label='v12_initializing'
        self.numerical_failures=0;self.infeasible_ticks=0

    def set_task_request(self,value):
        self.task_request=validate_task_request(value)
        if value['mode'] in ('release','monitor'):self.release_latched=True

    def _patches(self,frame,sample):
        parsed=[]
        for index,raw in enumerate(get(frame,'skin_patches',[])):
            p=PublicSkinPatch.from_public(raw);i=p.digit
            if i is None or not self.state.valid[i]:continue
            force=float(raw['normal'])
            if not np.isfinite(force) or force<=0:continue
            projection=self.geometry.linearize(sample,p,maximum_age_s=.12)
            if not projection.available:continue
            parsed.append(dict(index=index,digit=i,normal=force,patch=p,
                               row=projection.normal_jacobian25))
        return parsed

    def _bounds(self,frame,period):
        alpha=array(frame.alpha,(5,),'alpha',finite=True)
        minimum=array(frame.closure_min,(5,),'actual closure min',finite=True)
        maximum=array(frame.closure_max,(5,),'actual closure max',finite=True)
        permission=np.asarray(frame.closure_permission,bool)
        if permission.shape!=(5,):raise ValueError('Five closing permissions required')
        lo=np.r_[np.maximum(-self.g.open_rate*period,minimum-alpha),-np.asarray(frame.motion_reference_rate_limits)*period]
        hi=np.r_[np.minimum(self.g.close_rate*period,maximum-alpha),np.asarray(frame.motion_reference_rate_limits)*period]
        hi[:5][~(permission&self.state.valid)]=np.minimum(0.,hi[:5][~(permission&self.state.valid)])
        return lo,hi

    def _decide(self,frame,phase,current):
        if abs(float(frame.robot_control_time_s)-float(frame.time))>1e-9:
            raise ValueError('Actual executed robot port must have the current robot sample clock')
        sample=RobotJointSample.from_public_frame(frame,self.geometry.raw_joint_names,
            lift_velocity_m_s=frame.lift_velocity_m_s)
        if sample.velocity25 is None:raise ValueError('Current robot velocity is required')
        patches=self._patches(frame,sample)
        request=self.task_request;g=self.g
        caps=np.minimum(self.limits,np.asarray(request['limits']));budget=min(self.budget,request['budget'])
        ratio=(self.intent.fallback if isinstance(self.intent,FusionIntent) else self.intent)
        memory=(ratio.checkpoint() if isinstance(ratio,RatioIntent) else None)
        if isinstance(self.intent,FusionIntent):out,inliers=self.intent.step(self.state,phase,caps,budget,frame,patches)
        else:out=self.intent.step(self.state,phase,caps,budget,g.decision_period);inliers=None
        self.desired=np.asarray(out['desired_normal5']).copy()
        correction=np.zeros(25);groups={i:[] for i in range(5)}
        for p in patches:groups[p['digit']].append(p)
        for i in range(5):
            selected=[p for p in groups[i] if inliers is None or p['index'] in inliers]
            selected_indices={p['index'] for p in selected}
            measured=sum(p['normal'] for p in selected)
            for p in groups[i]:
                target=(self.desired[i]*p['normal']/measured
                        if p['index'] in selected_indices and measured>1e-12 else 0.)
                correction+=p['row']*(target-p['normal'])
        target=np.asarray(request['motion_target'],float)
        preview=request.get('motion_preview')
        if preview and preview['future_reference_available']:
            ts=np.asarray(preview['requested_times_s']);points=np.asarray(preview['motion_targets'])
            target=np.array([np.interp(float(frame.time)+g.decision_period,ts,points[:,i]) for i in range(3)])
        lo,hi=self._bounds(frame,g.decision_period)
        # A missing skin patch has no force Jacobian. Common bounded reference
        # search is a separate kinematic task, not a fabricated zero-force map.
        search=np.zeros(8)
        present=np.asarray(frame.contact_present,bool)
        if present.shape!=(5,):raise ValueError('Explicit five-channel contact-presence observation required')
        for i in range(5):
            absent=bool(self.state.valid[i] and not present[i] and
                        self.state.observed_normal[i]<.15)
            if not groups[i] and absent and self.desired[i]>g.support_contact_floor:
                search[i]=max(0.,hi[i])
        if np.any(search):
            actual=np.asarray(frame.robot_actuator_target21,float)
            base=self.tracker.servo.evaluate(sample.q25,sample.velocity25,actual)
            shifted=self.tracker.servo.evaluate(sample.q25,sample.velocity25,self.tracker._port(current+search))
            if base.generalized_actuator_effort25 is not None and shifted.generalized_actuator_effort25 is not None:
                correction+=shifted.generalized_actuator_effort25-base.generalized_actuator_effort25
        tracked=self.tracker.solve(q25=sample.q25,v25=sample.velocity25,current_reference8=current,
            actual_target21=frame.robot_actuator_target21,desired_torque_correction25=correction,
            nominal_motion_reference3=target,step_lower8=lo,step_upper8=hi,
            motion_reference_bounds=request['motion_reference_bounds'],tracking_tolerances3=request['tracking_tolerances'])
        self.plan_start=float(frame.time);self.plan_reference=current.copy()
        self.plan_ratio_memory=memory
        if tracked.available:self.plan_delta=tracked.increment8.copy()
        else:
            # No pressure/effort inverse is substituted when this model is
            # unavailable. Continue the shared legal public motion request.
            self.plan_delta=np.clip(np.r_[np.zeros(5),target-current[5:]],lo,hi)
            bounds=request['motion_reference_bounds']
            self.plan_delta[5:]=np.clip(current[5:]+self.plan_delta[5:],bounds['lower'],bounds['upper'])-current[5:]
            self.infeasible_ticks+=1
        # Force-to-reference feedback uses the actual executed reference at
        # every decision, so an unexecuted command never accumulates. Reconcile
        # the separate ratio PI/boost when effort cannot deliver its increase.
        limited=not tracked.available
        reduced_effort_shortfall=np.zeros(5,bool)
        if tracked.available:
            observed=np.where(self.state.valid,self.state.observed_normal,0.)
            positive=self.state.valid&(self.desired>observed+1e-9)
            limited=bool(np.any(positive&(self.plan_delta[:5]>=hi[:5]-1e-9)))
            clips=tracked.debug.get('native_force_clipped21')
            if clips is None:limited=True # missing actuation-limit report is unknown
            else:
                clips=np.asarray(clips,bool)
                if clips.shape!=(21,):raise ValueError('Native effort-clipping mask must have21 entries')
                limited=limited or any(positive[i] and np.any(clips[self.tracker.rows[i]]) for i in range(5))
            known=self.tracker.servo.evaluate(sample.q25,sample.velocity25,frame.robot_actuator_target21)
            # Project only onto each real finger-reference effort direction.
            # Orthogonal underactuation residual does not itself freeze the PI.
            native_gain=self.tracker.servo.model.actuator_gainprm[self.tracker.servo.ctrladr,0]
            directions=known.transmission21x25.T@(native_gain[:,None]*self.tracker.C)
            for i in range(5):
                wanted=float(directions[:,i]@correction)
                shortage=float(directions[:,i]@tracked.residual25)
                eps=256*np.finfo(float).eps*max(1.,abs(wanted),abs(shortage))
                reduced_effort_shortfall[i]=bool(positive[i] and abs(wanted)>eps and
                    np.sign(wanted)*shortage < -eps)
            limited=limited or bool(reduced_effort_shortfall.any())
        rollback=bool(memory is not None and limited and
            (not isinstance(self.intent,FusionIntent) or inliers is None))
        if rollback:
            # Keep public sample/initialization evidence; only undo memories
            # which increase the future requested grip under actor saturation.
            ratio.integral=memory['integral'];ratio.boost=memory['boost']
        self.debug=dict(architecture='measurement_intent_known_robot_effort',method=self.method,
            force_intent=plain(out['debug']),effort_tracking=tracked.to_dict(),
            pressure_prediction_available=False,external_current_geometry_known=False,
            contact_force_response_model_used=False,legacy_pressure_model_used=False,
            virtual_force_error_torque25=correction.tolist(),common_contact_search8=search.tolist(),
            incremental_feedback_reference_source='actual executed alpha, never last unexecuted plan',
            contact_search_rule='valid explicitly absent contact and measured pressure below .15N; missing geometry alone never authorizes closing',
            ratio_effort_saturation_memory_rollback=rollback,
            finger_reference_effort_shortfall5=reduced_effort_shortfall.tolist(),
            patch_count=len(patches),nominal_motion_reference3=target.tolist(),task_request_modified=False)
        self.last_decision=float(frame.time)

    def step(self,frame,phase=None,command='hold'):
        now=float(frame.time);dt=float(frame.dt);phase=str(phase or frame.phase).upper()
        if self.state._last_time is None or self.state.time<now-1e-9:self.state.update(frame,dt)
        if self.task_request is None:raise ValueError('Explicit public task request required')
        current=np.r_[frame.alpha,self.task_request['executed_motion_reference']]
        if command=='release' or phase in RELEASE_PHASES or self.task_request['mode'] in ('release','monitor'):
            self.release_latched=True
        if self.release_latched:
            self.desired[:]=0.;self.plan_start=None;self.plan_delta[:]=0.
            self.motion_reference=current[5:].copy();self.label='v12_native_release_preempts_all_tasks'
            delta=-np.minimum(np.maximum(0.,np.asarray(frame.alpha)-np.asarray(frame.closure_min)),self.g.release_rate*dt)
            self.debug=dict(release_authority=True,force_intent_updated=False,pressure_prediction_available=False)
            return ControlAction(delta,self.predicted.copy(),self.label,False,deepcopy(self.debug))
        if now>=self.last_decision+self.g.decision_period-1e-9:
            self._decide(frame,phase,current)
        fraction=np.clip((now-self.plan_start+dt)/self.g.decision_period,0.,1.)
        proposal=self.plan_reference+fraction*self.plan_delta
        lo,hi=self._bounds(frame,dt);delta=np.clip(proposal-current,lo,hi)
        bounds=self.task_request['motion_reference_bounds']
        motion=np.clip(current[5:]+delta[5:],bounds['lower'],bounds['upper'])
        self.motion_reference=motion.copy();self.label='v12_common_finite_effort_ramp'
        execution_changed=bool(np.max(abs(np.r_[current[:5]+delta[:5],motion]-proposal))>1e-10)
        ratio=(self.intent.fallback if isinstance(self.intent,FusionIntent) else self.intent)
        late_rollback=bool(execution_changed and self.plan_ratio_memory is not None and
            (not isinstance(self.intent,FusionIntent) or self.debug['force_intent'].get('fusion_mode')=='shared_ratio_fallback'))
        if late_rollback:
            ratio.integral=self.plan_ratio_memory['integral'];ratio.boost=self.plan_ratio_memory['boost']
        self.debug.update(actual_delta_closure=delta[:5].tolist(),actual_motion_reference3=motion.tolist(),
            execution_changed_plan=execution_changed,late_execution_ratio_memory_rollback=late_rollback,
            extra_motion_freeze=False,pause_transport=False)
        self.last_debug=deepcopy(self.debug)
        return ControlAction(delta[:5],self.predicted.copy(),self.label,False,deepcopy(self.debug))

    def state_dict(self):
        intent_memory=(self.intent.checkpoint() if isinstance(self.intent,RatioIntent) else
            dict(previous=self.intent.previous,fallback=self.intent.fallback.checkpoint(),
                 belief_history=None if self.intent.belief is None else self.intent.belief.history,
                 geometry=None if self.intent.snapshot is None else self.intent.snapshot.to_dict())
            if isinstance(self.intent,FusionIntent) else None)
        return plain(dict(method=self.method,last_decision=self.last_decision,release_latched=self.release_latched,
            plan_start=self.plan_start,plan_reference=self.plan_reference,plan_delta=self.plan_delta,
            plan_ratio_memory=self.plan_ratio_memory,
            desired=self.desired,motion_reference=self.motion_reference,observer=vars(self.state),
            intent_memory=intent_memory,
            debug=self.debug))


def create_controller(method,calibration,limits,budget,gains=None,robot_model=None):
    if robot_model is None:raise ValueError('Known robot model required')
    return TaskController(method,calibration,limits,budget,gains,robot_model)

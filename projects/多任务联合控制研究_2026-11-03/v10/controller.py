"""Experimental typed contact-space coordination around a retained stabilizer.

One pressure state and one grip/slip intent producer come from frozen ratio.
Material pressure activates robot displacement tasks, never an invented force
gradient. Observation, robot authority and empirical model support are separate.
No binary risk-motion freeze or ownership-based closing prohibition is used.
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import research_imports
from controller_adapter import create_controller
from vendor.frozen_controllers import ControlAction
from task_contract import validate_task_request
from .robot_contact_space import RobotContactSpace,RobotJointSample,PublicSkinPatch
from .servo_bank import Bank
from .directional_allocator import DirectionalAllocator

GAIN_KEYS=('base','ratio_floor','ratio_kp','ratio_ki','slip_boost_rate','baseline_tracking_gain','release_rate')

class DecisionState:
    """Delegate the original pressure observer; robot telemetry adds no force estimate."""
    def __init__(self,owner,state):self.owner=owner;self.force_state=state
    def __getattr__(self,key):return getattr(self.force_state,key)
    @property
    def posterior_mean_time_s(self):return float(self.force_state.time)
    def update(self,frame,dt):
        self.force_state.update(frame,dt)
        self.owner.latest_frame=frame
        self.owner.executed_motion=np.asarray(frame.executed_motion_reference,float).copy()

class JointTaskController:
    def __init__(self,calibration,limits,budget,gains,robot_model):
        from .runtime import PublicPreviewFollower
        self.nominal=create_controller('ratio',calibration,limits,budget,deepcopy(gains));self.g=self.nominal.g
        self.state=DecisionState(self,self.nominal.state);self.nominal.state=self.state
        self.calibration=deepcopy(calibration);self.shape=calibration['config']['shape']
        self.limits=np.asarray(limits,float);self.budget=float(budget)
        self.geometry=RobotContactSpace(robot_model);self.bank=Bank(self.geometry.axes25)
        self.allocator=DirectionalAllocator();self.follower=PublicPreviewFollower()
        self.support_norms={}
        for e in self.bank.entries:
            p=Path(e['source_fingerprints']['actual_footprints'])
            data=json.loads(p.read_text(encoding='utf-8'))
            delta=np.concatenate([np.asarray(v['interval_delta'],float) for k,v in data.items() if k.startswith('mixed_')])
            weights=1./e['micro_scale8'];budget_l1=float(np.max(abs(delta)@weights))
            self.support_norms[e['context_id']]=(weights,budget_l1)
        self.latest_frame=None;self.task_request={};self.executed_motion=np.zeros(3);self.motion_reference=np.zeros(3)
        self.desired=np.zeros(5);self.predicted=np.full(5,np.nan);self.robot_predicted=None
        self.last_decision=-np.inf;self.debug={};self.last_debug={};self.label='v10_initializing'
        self.release_latched=False;self.plan_active=False;self.plan_start=None;self.plan_delta=np.zeros(8)
        self.plan_reference=None;self.reference_before=None;self.numerical_failures=0;self.infeasible_ticks=0
        self.nominal_feedback_contract=dict(schema='v10-one-stabilizer-contact-direction-task',values=deepcopy(gains),
            pressure_state='frozen baseline coupled fixed-lag observer',grip_intent='frozen ratio only; no additional slip reserve')
    def set_task_request(self,value):
        self.task_request=validate_task_request(value)
        self.executed_motion=np.asarray(value['executed_motion_reference'],float).copy()
        if value['mode'] in ('release','monitor'):self.release_latched=True
    def _risk(self):
        f=self.latest_frame;n=np.asarray(f.normal,float)
        valid=np.asarray(f.valid,bool)&np.isfinite(n)&(np.asarray(f.age)<=.12)
        caps=np.minimum(self.limits,np.asarray(self.task_request['limits'],float));budget=min(self.budget,self.task_request['budget'])
        sigma=float(f.sensor_characteristics['force_sigma_N'])
        excess=np.where(valid,np.maximum(0.,n-3.*sigma-caps),0.)
        total_excess=max(0.,float(n.sum()-3.*np.sqrt(5.)*sigma-budget)) if valid.all() else None
        urgency=excess/np.maximum(caps,max(3.*sigma,1e-12))
        if total_excess is not None and total_excess>0:
            urgency=np.maximum(urgency,np.where(valid,total_excess/max(budget,3.*np.sqrt(5.)*sigma,1e-12),0.))
        return valid,urgency,dict(raw_local_excess_evidence_N=excess.tolist(),raw_budget_excess_evidence_N=total_excess,
            raw_pressure_observed=valid.tolist(),pressure_safety_measured_resolved=bool(valid.all() and not np.any(urgency>0.)),
            pressure_prediction_available=False,pressure_safety_certified=False)
    def _motion_plan(self):
        f=self.latest_frame;r=self.task_request;dt=self.g.decision_period
        target=np.asarray(r['motion_target'],float);preview=r.get('motion_preview')
        if preview and preview['future_reference_available']:
            ts=np.asarray(preview['requested_times_s']);points=np.asarray(preview['motion_targets'])
            target=np.array([np.interp(f.time+dt,ts,points[:,i]) for i in range(3)])
        rates=np.asarray(f.motion_reference_rate_limits,float);bounds=r['motion_reference_bounds']
        lo=np.r_[np.maximum(-min(.8,self.g.open_rate)*dt,self.state.closure_min-self.state.alpha),-rates*dt]
        hi=np.r_[np.minimum(min(.8,self.g.close_rate)*dt,self.state.closure_max-self.state.alpha),rates*dt]
        blocked=~self.state.valid|~self.state.closure_permission;hi[:5][blocked]=np.minimum(0.,hi[:5][blocked])
        lo[5:]=np.maximum(lo[5:],np.asarray(bounds['lower'])-self.executed_motion)
        hi[5:]=np.minimum(hi[5:],np.asarray(bounds['upper'])-self.executed_motion)
        dm=np.minimum(hi[5:],np.maximum(lo[5:],target-self.executed_motion))
        legal_intent=np.minimum(hi,np.maximum(lo,np.r_[self.nominal.interval_action,dm]))
        return target,lo,hi,legal_intent
    def _patch_groups(self,sample,urgency,selection):
        groups={}
        for raw in getattr(self.latest_frame,'skin_patches',[]):
            digit=int(raw['digit']);pressure=float(raw['normal'])
            if not 0<=digit<5 or urgency[digit]<=0 or pressure<=0.:continue
            patch=PublicSkinPatch.from_public(raw)
            projection=self.geometry.linearize(sample,patch,maximum_age_s=.12)
            if not projection.available:continue
            groups.setdefault(digit,[]).append(dict(patch=patch,pressure=pressure,
                row25=projection.normal_jacobian25,point=projection.kinematics.point_world,
                normal=projection.acquisition_normal_world))
        rows=[];free=[];errors=[];weights=[];saved=[]
        for digit,patches in sorted(groups.items()):
            total=sum(p['pressure'] for p in patches)
            row=sum((p['pressure']/total)*p['row25'] for p in patches)
            rows.append(row@selection.response25x8)
            free.append(float(row@(selection.free25-sample.q25)))
            errors.append(float(abs(row)@selection.error25));weights.append(float(urgency[digit]))
            saved.append((digit,patches,total))
        return np.asarray(rows).reshape(-1,8),np.asarray(free),np.asarray(errors),np.asarray(weights),saved
    def _exact_geometry(self,sample,qfuture,groups):
        future=RobotJointSample(sample.time_s+.02,sample.raw_joint_names,qfuture[:24],qfuture[24])
        values=[]
        for digit,patches,total in groups:
            value=0.
            for p in patches:
                k=self.geometry.point_kinematics(future,p['patch'])
                value+=p['pressure']/total*float(p['normal']@(k.point_world-p['point']))
            values.append(value)
        return np.asarray(values)
    def _decide(self,reference_before):
        now=float(self.latest_frame.time);self.plan_active=False;self.robot_predicted=None
        valid,urgency,evidence=self._risk();target,lo,hi,nominal8=self._motion_plan()
        self.debug=dict(controller_architecture='v10_typed_contact_direction_and_stabilizer',material=evidence,
            extra_motion_freeze=False,duplicate_slip_reserve=False,ownership_changes_mechanical_permission=False,
            force_posterior_source='unique original baseline observer',pressure_prediction_available=False,
            prediction_available=False,prediction_unavailable_reason='no physical pressure-response forecast for contact-space intent',
            original_user_target3=target.tolist(),nominal_increment8=nominal8.tolist(),reference_memory_rollback=False,
            nominal_model_comparison_scope='hypothetical uniform20ms ramp of stabilizer intent; not the unknown future baseline action sequence',
            actual_baseline_future_sequence_predicted=False)
        if np.count_nonzero(self.state.valid)<3:
            self.debug['status']='BASELINE_RETAINED_GRIP_OBSERVABILITY_REQUIREMENT';return
        if not np.any(urgency>0.):
            self.debug.update(status='BASELINE_NO_MATERIAL_DIRECTION_REQUEST');return
        sample=RobotJointSample.from_public_frame(self.latest_frame,self.geometry.raw_joint_names,
            lift_velocity_m_s=getattr(self.latest_frame,'lift_velocity_m_s',None))
        current_ref=np.r_[self.state.alpha,self.executed_motion]
        selection=self.bank.select(self.shape,sample.q25,sample.velocity25,current_ref)
        self.debug.update(servo_selection=selection.debug,servo_status=selection.status,servo_reasons=list(selection.reasons))
        if selection.response25x8 is None:
            self.debug['status']='BASELINE_ROBOT_RESPONSE_UNAVAILABLE';return
        G,c,err,w,groups=self._patch_groups(sample,urgency,selection)
        if not len(G):
            self.debug['status']='BASELINE_NO_USABLE_MATERIAL_PATCH_GEOMETRY';return
        context=selection.debug['context_id'];l1weights,l1budget=self.support_norms[context]
        self.debug['nominal_future_component_check']=selection.check_future_increment(nominal8)
        kwargs=dict(nominal_increment8=nominal8,step_lower8=lo,step_upper8=hi,
            robot_free25=selection.free25,robot_response25x8=selection.response25x8,
            output_axes25=self.geometry.axes25,user_target3=target,tracking_tolerances3=self.task_request['tracking_tolerances'],
            empirical_robot_error25=selection.error25,contact_rows_mx8=G,contact_free_delta_m=c,
            contact_error_m=err,contact_urgency_m=w,model_scope_label='EXPERIMENTAL_ANCHOR_LINEARIZATION_NO_FORCE_CERTIFICATE',
            proposal_lower8=selection.proposal_lower8,proposal_upper8=selection.proposal_upper8,
            proposal_l1_weights8=l1weights,proposal_l1_budget=l1budget)
        result=self.allocator.solve(**kwargs)
        if result.predicted_robot25 is None:
            self.debug.update(status='BASELINE_DIRECTIONAL_MODEL_UNAVAILABLE',allocation=result.debug);return
        # Exact robot FK verifies the finite predicted displacement; its
        # nonlinear remainder is not covered merely by an infinitesimal J.
        exact=self._exact_geometry(sample,result.predicted_robot25,groups)
        linear=G@result.increment8+c;remainder=abs(exact-linear)
        if np.max(remainder,initial=0.)>1e-10:
            kwargs['contact_error_m']=err+remainder
            result=self.allocator.solve(**kwargs)
            exact=self._exact_geometry(sample,result.predicted_robot25,groups)
            linear=G@result.increment8+c;remainder=abs(exact-linear)
        support=selection.check_future_increment(result.increment8)
        self.debug.update(allocation=result.debug,candidate_future_component_check=support,
            empirical_l1_proposal_region=dict(weights8=l1weights.tolist(),budget=l1budget,certifies_arbitrary_mixed=False),
            exact_predicted_robot_normal_delta_m=exact.tolist(),linear_robot_normal_delta_m=linear.tolist(),
            exact_fk_linearization_remainder_m=remainder.tolist(),exact_robot_fk_is_not_actual_future_motion=True,
            contact_geometry_scope='pressure-weighted robot displacement relative to acquired skin normals; no external current gap or pressure prediction')
        if support['status']=='UNSUPPORTED_FUTURE':
            self.debug['status']='BASELINE_CANDIDATE_RESPONSE_SCOPE_UNSUPPORTED';return
        self.robot_predicted=result.predicted_robot25.copy()
        if np.max(abs(result.increment8-nominal8))<=1e-12:
            self.debug['status']='BASELINE_ALREADY_SATISFIES_DIRECTIONAL_TASK';return
        # Compare two declared finite-ramp intentions. The retained stabilizer
        # may re-clip at each future5ms tick, so this is NOT its actual unknown
        # future sequence. Only paired physical trials compare real methods.
        indices=[self.geometry.axes25.index(n) for n in ('transport_lift','rh_WRJ2','rh_WRJ1')]
        tol=np.array([np.inf if t is None else t for t in self.task_request['tracking_tolerances']],float)
        pred_nominal=selection.free25+selection.response25x8@nominal8
        nominal_scope=selection.check_future_increment(nominal8)['status']!='UNSUPPORTED_FUTURE'
        nom_tube=np.maximum(0.,abs(pred_nominal[indices]-target)+selection.error25[indices]-tol)
        new_tube=np.maximum(0.,abs(result.predicted_robot25[indices]-target)+selection.error25[indices]-tol)
        new_cost=float(w@np.maximum(0.,exact+err)**2)
        if nominal_scope:
            nominal_exact=self._exact_geometry(sample,pred_nominal,groups)
            old_cost=float(w@np.maximum(0.,nominal_exact+err)**2)
            accept=bool(np.max(new_tube-nom_tube)<=1e-8 and new_cost<old_cost-1e-16)
        else:
            old_cost=None;accept=bool(np.max(new_tube)<=1e-8 and np.max(exact+err)<=1e-10)
        self.debug.update(exact_model_geometry_cost_before=old_cost,exact_model_geometry_cost_after=new_cost,
            nominal_model_comparison_supported=nominal_scope,candidate_finite_fk_accepted=accept)
        if not accept:
            self.debug['status']='BASELINE_NO_CHECKED_FINITE_GEOMETRIC_IMPROVEMENT';return
        self.plan_active=True;self.plan_start=now;self.plan_reference=current_ref.copy();self.plan_delta=result.increment8.copy()
        self.reference_before=deepcopy(reference_before);self.debug['status']='EXPERIMENTAL_DIRECTIONAL_FINITE_RAMP'
        self.debug['applied_input_increment_8']=self.plan_delta.tolist()
    def step(self,frame,phase=None,command='hold'):
        before={n:deepcopy(getattr(self.nominal,n)) for n in ('integral','boost')}
        if command=='release' or self.task_request.get('mode') in ('release','monitor') or str(phase or frame.phase).upper() in ('RELEASE','RELEASING'):self.release_latched=True
        previous=self.nominal.last_decision
        nominal_action=self.nominal.step(frame,phase,'release' if self.release_latched else command)
        self.desired=self.nominal.desired.copy();self.predicted[:]=np.nan
        if self.release_latched:
            self.plan_active=False;self.follower.reset();self.motion_reference=self.executed_motion.copy();self.label='v10_release_authority'
            return ControlAction(nominal_action.delta_closure.copy(),self.predicted.copy(),self.label,False,
                dict(release_authority=True,prediction_available=False,prediction_unavailable_reason='native release outside held robot response'))
        if self.nominal.last_decision!=previous:
            self._decide(before);self.last_decision=float(frame.time)
        if self.plan_active:
            fraction=min(1.,max(0.,(float(frame.time)-self.plan_start+float(frame.dt))/.02))
            desired_ref=self.plan_reference+fraction*self.plan_delta
            delta=desired_ref[:5]-self.state.alpha
            actual=np.minimum(self.state.closure_max-self.state.alpha,np.maximum(self.state.closure_min-self.state.alpha,delta))
            blocked=~self.state.valid|~self.state.closure_permission;actual[blocked]=np.minimum(0.,actual[blocked])
            self.motion_reference=desired_ref[5:].copy();self.follower.reset()
            late=bool(np.max(abs(actual-delta))>1e-10)
            self.debug.update(planned_finite_ramp_realization_changed=late)
            if late:self.robot_predicted=None
            if np.max(abs(actual-nominal_action.delta_closure))>1e-12:
                for n in ('integral','boost'):setattr(self.nominal,n,deepcopy(self.reference_before[n]))
                self.debug['reference_memory_rollback']=True
            self.label='v10_contact_direction_intervention';pause=False
        else:
            actual=nominal_action.delta_closure.copy();pause=bool(nominal_action.pause_transport)
            self.motion_reference=self.follower.step(self.task_request,pause_transport=pause)
            self.label='v10_retained_stabilization'
        self.debug.update(actual_subinterval_delta_closure=actual.tolist(),pause_transport=pause,
            extra_motion_freeze=False,prediction_available=False,
            stabilizer_pause_transport=bool(nominal_action.pause_transport),
            stabilizer_pause_replaced_by_directional_plan=bool(self.plan_active and nominal_action.pause_transport),
            grip_observability_pause_inherited_at_decision=True)
        self.last_debug=deepcopy(self.debug)
        return ControlAction(actual,self.predicted.copy(),self.label,pause,deepcopy(self.debug))

def create_joint(calibration,limits,budget,gains=None,robot_model=None):
    if not isinstance(gains,dict) or any(k not in gains for k in GAIN_KEYS):raise ValueError('Complete original baseline configuration required')
    if robot_model is None:raise ValueError('Known robot model required; no object state is a controller input')
    return JointTaskController(calibration,limits,budget,gains,robot_model)

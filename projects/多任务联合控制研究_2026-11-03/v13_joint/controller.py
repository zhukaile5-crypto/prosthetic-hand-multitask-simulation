"""Public current-state task fusion -> joint intent QP -> native21 proposal.

No actual plant/data/engine write. The runtime supplies current committed21,
robot25 and delayed public skin. Force intents remain distinct from forecasts.
"""
from dataclasses import dataclass
from copy import deepcopy
import time
import numpy as np
import research_imports
from vendor.frozen_controllers import make_gains
from v12.observation import PublicTactileState,array,get
from v12.intents import FixedIntent,RatioIntent,RELEASE_PHASES
from v12_native.controller import FusionIntent,TaskController as LegacyController,plain
from v10.robot_contact_space import RobotContactSpace,RobotJointSample
from v11.object_belief import RelativeObjectBelief,DeclaredShapeTemplate
from v13_a1.codec import Reference21,EnginePort21
from task_contract import validate_task_request
from .intent_qp import solve_intent_qp
from .prepared_value_effort21 import PreparedValueEffort21
from .design import DECISION_PERIOD_S,CONTROL_PERIOD_S,reference_rates,QP_MAX_ITERATIONS,QP_ATOL,QP_RTOL

METHODS=('ratio_effort','fixed_effort','fusion_effort')
@dataclass(frozen=True)
class JointAction:
    reference21:Reference21
    mode:str
    mode_generation:int
    debug:dict

class JointController:
    def __init__(self,method,calibration,limits,budget,gains,robot_model):
        if method not in METHODS:raise ValueError('Unknown force-intent comparator')
        self.method=method;self.g=make_gains(deepcopy(gains));self.state=PublicTactileState(self.g)
        if self.g.decision_period!=DECISION_PERIOD_S:raise ValueError('One declared20ms decision clock required')
        self.geometry=RobotContactSpace(robot_model);self.native=PreparedValueEffort21(robot_model)
        self.intent=FusionIntent(self.g,calibration['config']['shape']) if method=='fusion_effort' else RatioIntent(self.g) if method=='ratio_effort' else FixedIntent(self.g)
        shape=calibration['config']['shape']
        self.group_belief=RelativeObjectBelief(DeclaredShapeTemplate(shape,.028,.047 if shape=='capsule' else None,source='fixed public nominal template')) if shape in ('capsule','sphere') else None
        self.limits=array(limits,(5,),'limits5',finite=True);self.budget=float(budget)
        self.desired=np.zeros(5);self.predicted=np.full(5,np.nan);self.motion_reference=np.zeros(3)
        self.task_request=None;self.last_decision=-np.inf;self.release_latched=False
        self.plan_start=None;self.plan_reference21=None;self.plan_delta21=None;self.plan_generation=None
        self.debug={};self.last_debug={};self.decision_evidence=[];self.plan_ratio_memory=None
        self.numerical_failures=0;self.infeasible_ticks=0;self.qp_unavailable=0

    def set_task_request(self,value):self.task_request=validate_task_request(value)
    _patches=LegacyController._patches

    def _ratio(self):return self.intent.fallback if isinstance(self.intent,FusionIntent) else self.intent

    def _prior(self,frame,phase,patches,mode,caps,budget):
        ratio=self._ratio();self.plan_ratio_memory=ratio.checkpoint() if isinstance(ratio,RatioIntent) else None
        if mode!='holding':return np.zeros(5),None,dict(label='release_cancels_all_holding_force_tasks')
        if isinstance(self.intent,FusionIntent):out,inliers=self.intent.step(self.state,phase,caps,budget,frame,patches)
        else:
            out=self.intent.step(self.state,phase,caps,budget,DECISION_PERIOD_S);inliers=None
            if self.group_belief is not None:
                hypothesis=self.group_belief.update(frame)
                if hypothesis.grouping_unambiguous and hypothesis.selected_index is not None:
                    inliers=set(hypothesis.hypotheses[hypothesis.selected_index].inlier_indices)
        return np.asarray(out['desired_normal5']).copy(),inliers,plain(out['debug'])

    def _nominal(self,frame,mode,current,patches,prior):
        nominal=current.copy();search=np.zeros(21)
        if mode!='holding':nominal=np.asarray(frame.release_nominal_reference21,float).copy()
        else:
            target=np.asarray(self.task_request['motion_target'],float);preview=self.task_request.get('motion_preview')
            if preview and preview['future_reference_available']:
                ts=np.asarray(preview['requested_times_s']);points=np.asarray(preview['motion_targets'])
                target=np.array([np.interp(float(frame.time)+DECISION_PERIOD_S,ts,points[:,i]) for i in range(3)])
            nominal[18:]=target
            present=np.asarray(frame.contact_present,bool)
            if present.shape!=(5,):raise ValueError('Explicit five-channel contact presence required')
            for i,rows in enumerate(((0,5),(5,8),(8,11),(11,14),(14,18))):
                absent=self.state.valid[i] and not present[i] and self.state.observed_normal[i]<.15
                if absent and not any(p['digit']==i for p in patches) and prior[i]>self.g.support_contact_floor:
                    from foundation_config import OPEN,CLOSED
                    search[rows[0]:rows[1]]=(np.asarray(CLOSED[i])-np.asarray(OPEN[i]))*min(.8,self.g.close_rate)*DECISION_PERIOD_S
            nominal+=search
        return nominal,search

    def _decide(self,frame,phase,mode,current,generation):
        now=float(frame.time);began=time.perf_counter_ns()
        if frame.robot_control_time_s!=now:raise ValueError('Current robot query clock mismatch')
        sample=RobotJointSample.from_public_frame(frame,self.geometry.raw_joint_names,lift_velocity_m_s=frame.lift_velocity_m_s)
        if sample.velocity25 is None:raise ValueError('Robot velocities cannot be fabricated')
        patches=self._patches(frame,sample)
        caps=np.minimum(self.limits,np.asarray(self.task_request['limits'],float));budget=min(self.budget,float(self.task_request['budget']))
        prior,inliers,prior_debug=self._prior(frame,phase,patches,mode,caps,budget)
        nominal,search=self._nominal(frame,mode,current,patches,prior)
        lower=np.asarray(frame.reference_lower21,float).copy();upper=np.asarray(frame.reference_upper21,float).copy()
        rates=reference_rates(mode)
        if mode=='holding':
            bounds=self.task_request['motion_reference_bounds'];lower[18:]=np.maximum(lower[18:],bounds['lower']);upper[18:]=np.minimum(upper[18:],bounds['upper'])
            for i,tolerance in enumerate(self.task_request['tracking_tolerances']):
                if tolerance is not None:lower[18+i]=max(lower[18+i],nominal[18+i]-float(tolerance));upper[18+i]=min(upper[18+i],nominal[18+i]+float(tolerance))
        else:lower[19:]=upper[19:]=np.asarray(frame.release_entry_reference21,float)[19:]
        ref=Reference21(current,now);target=Reference21(nominal,now)
        engine=EnginePort21(frame.robot_actuator_target21,now,self.native.actor.lift_bias_m)
        domain=self.native.build_domain(q25=sample.q25,v25=sample.velocity25,robot_axes25=sample.raw_joint_names+('transport_lift',),time_s=now,current_reference=ref,nominal_reference=target,actual_engine=engine,reference_lower21=lower,reference_upper21=upper,step_lower21=-rates*DECISION_PERIOD_S,step_upper21=rates*DECISION_PERIOD_S)
        J=np.asarray([p['row'] for p in patches],float).reshape((-1,25));y=np.asarray([p['normal'] for p in patches],float)
        digits=np.asarray([p['digit'] for p in patches],int);zprior=np.zeros(len(patches))
        selected=[p for p in patches if inliers is None or p['index'] in inliers]
        selected_indices={p['index'] for p in selected}
        for i in range(5):
            measured=sum(p['normal'] for p in selected if p['digit']==i)
            for j,p in enumerate(patches):
                if p['index'] in selected_indices and p['digit']==i and measured>1e-12:zprior[j]=prior[i]*p['normal']/measured
        supported=(mode=='holding' and inliers is not None and self.state.valid.all() and self.state.load_valid)
        active_digits={p['digit'] for p in selected}
        supported=supported and 0 in active_digits and len(active_digits)>=3 and all(p['patch'].reference_normal is not None for p in selected)
        demand=None;c=None
        if supported:
            c=np.asarray([(float(p['patch'].reference_normal[2])+self.state.mu[p['digit']]*self.g.friction_margin*np.sqrt(max(0.,1.-float(p['patch'].reference_normal[2])**2))) if p['index'] in inliers else 0. for p in patches])
            slip=max(0.,self.state.slip_speed-.004) if np.isfinite(self.state.slip_speed) else 0.
            demand=float(self.state.load+self.g.load_uncertainty_weight*self.state.load_uncertainty+self.g.support_buffer+self.g.slip_load_gain*slip)
        elif mode!='holding':
            # An inactive holding requirement is explicitly zero, not a zero load observation.
            c=np.zeros(len(patches));demand=0.
        qp=None;native=None;sr=None;sd=None;delta=np.zeros(21);z=zprior.copy()
        reason='prepared_domain_unavailable';qp_ns=0;inverse_ns=0
        if domain.available:
            force_scales=np.max(abs(self.native.servo.force_range),axis=1)
            if not np.isfinite(force_scales).all() or np.any(force_scales<=0):raise ValueError('Declared finite native effort unit scales required')
            joint_scales=abs(domain.transmission21x25.T)@force_scales
            if np.any(joint_scales<=0):raise ValueError('Undeclared joint effort normalization')
            sr=1./joint_scales;sd=1./force_scales
            if c is not None and demand is not None:
                qbegan=time.perf_counter_ns()
                qp=solve_intent_qp(p0_21=domain.p0_effort21,A21x25=domain.transmission21x25,Jpx25=J,y_p=y,z_intent_p=zprior,effort_lower21=domain.effort_lower21,effort_upper21=domain.effort_upper21,patch_digit_p=digits,caps5=np.maximum(0.,caps-self.g.force_margin),total_budget=max(0.,budget-2*self.g.force_margin),support_coeff_p=c,load_proxy_N=demand,scales=dict(residual25=sr,intentP=1./np.maximum(1.,caps[digits]),delta21=sd,load=1.),max_iterations=QP_MAX_ITERATIONS,atol=QP_ATOL,rtol=QP_RTOL)
                qp_ns=time.perf_counter_ns()-qbegan
                if qp.available:delta=qp.d21.copy();z=qp.zP.copy();reason='joint_current_state_effort_and_force_intent'
                else:self.qp_unavailable+=1;reason='joint_numerics_unavailable_direct_public_force_feedback'
            else:reason='conditional_support_unknown_direct_public_force_feedback'
            ibegan=time.perf_counter_ns()
            native=self.native.solve(domain,p0_effort21=domain.p0_effort21,delta_effort21=delta,correction25=J.T@(z-y),residual_scale25=sr,delta_scale21=sd,requested_target_effort21=None if qp is None or not qp.available else qp.p21)
            inverse_ns=time.perf_counter_ns()-ibegan
        if native is not None and native.available:end=native.reference21.values.copy()
        else:
            self.infeasible_ticks+=1;end=nominal.copy()
            # Explicit fallback uses genuine mechanical authority, not an empty
            # optimization corridor relabelled as the achieved user target.
            fallback_lower=np.maximum(frame.reference_lower21,current-rates*DECISION_PERIOD_S)
            fallback_upper=np.minimum(frame.reference_upper21,current+rates*DECISION_PERIOD_S)
            if np.any(fallback_lower>fallback_upper):raise ValueError('Empty genuine fallback reference authority')
            end=np.minimum(fallback_upper,np.maximum(fallback_lower,end))
            reason='native_unavailable_legal_nominal_reference'
        self.plan_start=now;self.plan_reference21=current.copy();self.plan_delta21=end-current;self.plan_generation=generation
        self.last_decision=now;self.desired=np.bincount(digits,weights=z,minlength=5).astype(float) if len(digits) else prior.copy()
        if isinstance(self.intent,FusionIntent) and qp is not None and qp.available:self.intent.previous=self.desired.copy()
        ratio=self._ratio()
        rollback=(native is None or not native.available or qp is not None and not qp.available)
        if isinstance(ratio,RatioIntent) and self.plan_ratio_memory is not None and rollback:
            ratio.integral=self.plan_ratio_memory['integral'];ratio.boost=self.plan_ratio_memory['boost']
        full_native=None if native is None else native.to_dict()
        evidence=dict(decision_id=len(self.decision_evidence),time_s=now,mode=mode,generation=generation,nominal_target_time_s=now+DECISION_PERIOD_S,domain=domain.to_dict(),native=full_native,qp=None if qp is None else dict(status=qp.status,reason=qp.reason,d21=plain(qp.d21),zP=plain(qp.zP),objective=qp.objective,diagnostics=plain(qp.diagnostics)),patch_indices=[p['index'] for p in patches],Jpx25=J.tolist(),observed_patch_normal=y.tolist(),force_prior5=prior.tolist(),force_intent_debug=prior_debug)
        self.decision_evidence.append(evidence)
        native_summary=None if native is None else dict(status=native.status,reason=native.reason,residual25=plain(native.residual25),objective=native.objective,actual_delta_effort21=plain(native.actual_delta_effort21))
        self.debug=dict(architecture='joint_public_current_effort_force_intent21',method=self.method,label=reason,decision_id=evidence['decision_id'],decision_time_s=now,mode_generation=generation,conditional_support_active=supported,holding_load_task_inactive=mode!='holding',support_demand_N=demand,force_prior5=prior.tolist(),desired_force5=self.desired.tolist(),public_patch_count=len(patches),qp_status=None if qp is None else qp.status,qp_diagnostics=None if qp is None else plain(qp.diagnostics),native_summary=native_summary,nominal_contact_search21=search.tolist(),pressure_prediction_available=False,actual_reference_source='observed_committed21_not_legacy_alpha',solver_timing_ms=dict(qp=qp_ns/1e6,native_inverse=inverse_ns/1e6,whole_decision=(time.perf_counter_ns()-began)/1e6),task_request_modified=False)

    def step(self,frame,phase=None,command='hold'):
        now=float(frame.time);dt=float(frame.dt);phase=str(phase or frame.phase).upper()
        if dt!=CONTROL_PERIOD_S:raise ValueError('One declared5ms public execution clock required')
        if self.state._last_time is None or self.state.time<now-1e-9:self.state.update(frame,dt)
        if self.task_request is None:raise ValueError('Explicit public task required')
        if command=='release' or phase in RELEASE_PHASES or self.task_request['mode'] in ('release','monitor'):self.release_latched=True
        mode='release' if self.release_latched else 'holding'
        current=array(frame.robot_reference21,(21,),'actual current reference21',finite=True)
        if isinstance(frame.mode_generation,(bool,np.bool_)) or not isinstance(frame.mode_generation,(int,np.integer)) or frame.mode_generation<0:raise ValueError('Explicit nonnegative mode generation required')
        generation=int(frame.mode_generation)
        if self.plan_generation!=generation:self.plan_start=None;self.last_decision=-np.inf
        if mode=='release' and self.debug.get('mode')=='holding':self.last_decision=-np.inf
        if now>=self.last_decision+DECISION_PERIOD_S-1e-9:self._decide(frame,phase,mode,current,generation)
        fraction=np.clip((now-self.plan_start+dt)/DECISION_PERIOD_S,0.,1.)
        proposal=self.plan_reference21+fraction*self.plan_delta21
        maximum=reference_rates(mode)*dt;lower=np.maximum(frame.reference_lower21,current-maximum);upper=np.minimum(frame.reference_upper21,current+maximum)
        if np.any(lower>upper):raise ValueError('Empty5ms reference execution authority')
        proposed=np.minimum(upper,np.maximum(lower,proposal))
        self.motion_reference=proposed[18:].copy();self.debug['mode']=mode
        self.debug['plan_snapshot_scope']='original20ms snapshot; intermediate_qv does_not_inherit_new_optimality_certificate'
        self.last_debug=deepcopy(self.debug)
        return JointAction(Reference21(proposed,now),mode,generation,deepcopy(self.debug))

    def reconcile(self,receipt):
        self.motion_reference=receipt.executed.values[18:].copy()
        self.debug.update(actual_reference21=receipt.executed.values.tolist(),execution_changed_plan=receipt.execution_changed,stale_plan_replaced=receipt.stale_plan_replaced)
        ratio=self._ratio()
        if receipt.execution_changed and isinstance(ratio,RatioIntent) and self.plan_ratio_memory is not None:
            ratio.integral=self.plan_ratio_memory['integral'];ratio.boost=self.plan_ratio_memory['boost']

    def state_dict(self):return plain(dict(method=self.method,last_decision=self.last_decision,desired=self.desired,motion_reference=self.motion_reference,plan_generation=self.plan_generation,decision_evidence_count=len(self.decision_evidence),debug=self.debug,observer=vars(self.state)))

def create_controller(method,calibration,limits,budget,gains=None,robot_model=None):
    if robot_model is None:raise ValueError('Known robot model required')
    return JointController(method,calibration,limits,budget,gains,robot_model)

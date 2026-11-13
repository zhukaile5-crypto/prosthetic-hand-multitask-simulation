"""Independent research kernel derived from the admitted fixed-clock kernel.
Only policy construction and explicit joint-motion interface differ.
Original session source SHA256: fd22dfa8be4bf7e16217227abda9fe3ef0cda87febf5ab5a5a007ea32edee656
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import time,json
import numpy as np
import research_imports
from research_imports import ROOT, FOUNDATION, source_identity
from session import TrialSession, HANDOFF, DURATION, SENSOR_PERIOD, event_schedule, jsonable, sha

class ResearchSession(TrialSession):
    def __init__(self,method='joint',**kwargs):
        if method not in ('ratio','unified','joint','ratio_release','joint_release'):
            raise ValueError('Unknown independent research method')
        if kwargs.get('split')=='confirmation':
            raise ValueError('Research confirmation is not frozen or admitted; development only')
        holding_method=method.removesuffix('_release')
        super().__init__(method=holding_method if holding_method in ('ratio','unified') else 'ratio',**kwargs)
        self.method=method
        self.release_driver=None
        if method.endswith('_release'):
            from lifecycle_plant import LifecyclePlant
            from lifecycle_release import LifecycleRelease
            from hand_plant import CommonPreparation
            self.plant=LifecyclePlant(self.config)
            self.preparation=CommonPreparation(self.plant)
            self.last_truth=self.plant.truth()
            self.release_driver=LifecycleRelease(self.plant)
        if holding_method=='joint':
            from coordinator import create_joint
            self.controller=create_joint(self.calibration,self.limits,self.budget)
            self.gains=asdict(self.controller.g)
        from skin_geometry import DelayedSkinGeometry
        self.skin_geometry=DelayedSkinGeometry(self.plant)
        self.research_sources=source_identity()

    def _lifecycle_step(self,obs):
        driver=self.release_driver
        driver.request(obs,self.plant.data.ctrl[driver.aids],self.plant.lift_ref,self.plant.wrist_ref)
        command=driver.step(obs);self.plant.apply_release_command(command)
        self.controller.state.update(obs,SENSOR_PERIOD)
        self.controller.desired=np.zeros(5);self.controller.predicted=np.full(5,np.nan)
        return np.zeros(5),np.full(5,np.nan),deepcopy(command.debug)

    def metadata(self):
        value=super().metadata()
        policy_rows=[r for r in self.trace if r['controller_active'] and r['command']!='release']
        value.update(schema='joint_task_research_trial_v1',
            benchmark_id='joint_task_research_development_v1',
            research_phase='development', foundation_directory=str(FOUNDATION),
            raw_joint_names=[self.plant.model.joint(int(j)).name for j in self.plant.raw_joint_ids],
            raw_joint_qpos_indices=self.plant.raw_qadr.tolist(),
            foundation_admission_sha256=self.admission['manifest_sha256'],
            research_source_sha256=self.research_sources,
            prediction_coverage=dict(policy_ticks=len(policy_rows),
                available_ticks=sum(bool(r.get('prediction_available',False)) for r in policy_rows),
                model_invalid_ticks=sum(bool(r.get('model_invalid',False)) for r in policy_rows),
                observation_unavailable_ticks=sum(bool(r.get('observation_unavailable',False)) for r in policy_rows),
                interpretation='Predictor availability is separate from actual physical record completeness and task outcomes'),
            prior_confirmation_seen_for_diagnosis=True,
            release_mode='contact_kinematic_projection' if self.release_driver is not None else 'legacy_mechanical_path',
            release_experimental=self.release_driver is not None,
            public_skin_geometry='delayed per-patch robot link/position/normal; shared finger pressure noise; no external object identity',
            controller_scope='independent research; common preparation/held domain; optional explicitly selected lifecycle mode')
        return value

    def step_one_sensor_tick(self):
        if self.paused or not self.running:return self.snapshot()
        t=self.time
        if abs(self.plant.time-t)>1e-7:raise RuntimeError('physics and session clocks diverged')
        truth=self.plant.truth();raw=self.plant.raw_sensors();obs=self.sensors.sample(raw)
        self.skin_geometry.acquire(raw,obs)
        if not all(np.isfinite(truth[k]).all() for k in ('normal','skin_normal','position','qpos','qvel')):
            raise RuntimeError('Nonfinite physical state; run stopped and retained as failure')
        self.last_truth=truth;self.last_obs=obs
        scheduled=self.mode=='scheduled'
        env,lift_ref,wrist=event_schedule(self.scenario_name,t,self.config.friction,self.config.lift_target) if scheduled else (dict(force=[0.,0.,0.],torque=[0.,0.,0.],squeeze=0.,obstacle_x=0.,friction=self.config.friction),self.config.lift_target,[0.,0.])
        if not scheduled:
            if t<self.manual_force_until:env['force'][1]=1.2
            if t<self.manual_squeeze_until:env['squeeze']=.02
        signature=(tuple(env['force']),float(env['squeeze']),float(env['obstacle_x']),float(env['friction']))
        if self.environment_signature is not None and signature!=self.environment_signature:
            if np.linalg.norm(env['force'])>0.:self.event('固定日程外载荷开始或改变。')
            elif env['obstacle_x']!=0.:self.event('有限质量障碍物开始接近；实际接触另行记录。')
            elif env['friction']!=self.config.friction:self.event('接触摩擦日程改变。')
        self.environment_signature=signature
        requested_release=(scheduled and t>=12.-1e-9) or self.manual_release
        action=np.zeros(5);prediction=np.full(5,np.nan);desired=np.zeros(5);debug={};controller_active=False
        began=time.perf_counter_ns()
        early_lifecycle_release=bool(requested_release and self.release_driver is not None
            and (t<HANDOFF-1e-9 or not self.handoff_passed))
        if early_lifecycle_release:
            if self.release_at is None:
                self.release_at=t
                physical=truth['normal']>.2
                self.release_supported=bool(np.count_nonzero(physical)>=3 and physical[0] and truth['all_hand_object']>.2)
                self.plant.begin_release()
                self.event('用户释放抢占共同准备；立即卸载，不归为成功预抓。')
            self.last_command='release'
            desired,prediction,debug=self._lifecycle_step(obs);controller_active=True
        elif t<HANDOFF-1e-9:
            # Same observed skin signal and lifecycle for both methods.
            # This is the exact public virtual-skin preparation channel that
            # was physically accepted, not object-only truth or a new filter.
            public=np.zeros(5)
            for patch in raw['patches']:
                if 0<=int(patch['digit'])<5:public[int(patch['digit'])]+=float(patch['normal'])
            desired=self.preparation.step(public)
            if t>=HANDOFF-.2-1e-9:self.controller.state.update(obs,SENSOR_PERIOD)
        else:
            if not self.handoff_attempted:self._handoff(obs,truth)
            if self.handoff_passed:
                obs.closure_min=self.local_lower.copy();obs.closure_max=self.local_upper.copy()
                obs.closure_permission=np.asarray(raw['closure_permission'],bool).copy()
                if requested_release:
                    if self.release_at is None:
                        self.release_at=t
                        physical=(truth['normal']>.2)
                        self.release_supported=bool(np.count_nonzero(physical)>=3 and physical[0] and truth['all_hand_object']>.2)
                        self.plant.begin_release();self.event('主动释放开始，公共机械张开路径执行。')
                    obs.closure_min=self.plant.closure_min.copy();obs.closure_max=self.plant.closure_max.copy()
                self.last_command='release' if requested_release else 'hold'
                if requested_release and self.release_driver is not None:
                    desired,prediction,debug=self._lifecycle_step(obs);controller_active=True
                else:
                    if hasattr(self.controller,'set_motion_request'):
                        self.controller.set_motion_request(np.r_[lift_ref,wrist],
                            np.r_[self.plant.lift_ref,self.plant.wrist_ref])
                    result=self.controller.step(obs,self.plant.phase,self.last_command)
                    action=np.asarray(result.delta_closure,float);prediction=np.asarray(result.predicted_normal,float)
                    desired=self.controller.desired.copy();debug=deepcopy(result.debug or {});controller_active=True
                    before=self.plant.alpha.copy();self.plant.execute_delta(action)
                    debug['executed_delta_closure']=(self.plant.alpha-before).tolist()
                debug['local_domain_active']=not requested_release
                debug['release_domain_override']=bool(requested_release)
                debug['local_domain_lower']=obs.closure_min.tolist();debug['local_domain_upper']=obs.closure_max.tolist()
                if not requested_release:
                    joint_motion=getattr(self.controller,'motion_reference',None)
                    if joint_motion is not None:
                        motion=np.asarray(joint_motion,float)
                        if motion.shape!=(3,) or not np.isfinite(motion).all():
                            raise RuntimeError('Invalid jointly planned motion reference')
                        self.plant.set_motion(float(motion[0]),motion[1:])
                    elif not result.pause_transport:
                        self.plant.set_motion(lift_ref,wrist)
                if not requested_release and result.pause_transport:debug['transport_paused']=True
            else:
                if self.plant.release_started is None:self.plant.begin_release()
                self.plant.execute_delta(-np.minimum(np.maximum(0.,self.plant.alpha-self.plant.closure_min),.8*SENSOR_PERIOD))
                debug={'policy_not_executed':True,'failure_reason':self.preparation_failure}
        self.control_ms=(time.perf_counter_ns()-began)/1e6
        self.last_desired=desired;self.last_prediction=prediction;self.last_action=action;self.last_debug=debug
        observation_unavailable=bool(controller_active and not np.any(obs.valid))
        model_invalid=bool(debug.get('prediction_invalid_reason'))
        prediction_available=bool(controller_active and not requested_release and not observation_unavailable
            and debug.get('prediction_available',np.isfinite(prediction).all()))
        row=dict(time=t,normal=truth['normal'],skin_normal=truth['skin_normal'],shear=truth['shear'],
            palm=truth['palm'],hand_other=truth['hand_other'],all_hand_object=truth['all_hand_object'],
            support=truth['support'],penetration=truth['penetration'],world_penetration=truth['world_penetration'],
            position=truth['position'],relative=truth['relative'],rotation=truth['rotation'],
            relative_velocity=truth['relative_velocity'],angular_velocity=truth['angular_velocity'],
             lift=truth['lift'],lift_ref=lift_ref,qpos=truth['qpos'],qvel=truth['qvel'],
             wrist_position=np.array([self.plant.data.qpos[self.plant.model.joint(n).qposadr[0]]
                 for n in ('rh_WRJ2','rh_WRJ1')]),requested_wrist_reference=np.asarray(wrist),
            ctrl_before=truth['ctrl'],ctrl=self.plant.data.ctrl.copy(),
            alpha_before=truth['alpha'],alpha=self.plant.alpha.copy(),phase=self.plant.phase,desired=desired,predicted_normal=prediction,
            requested_action=action,controller_active=controller_active,control_ms=self.control_ms,
            executed_action=self.plant.alpha-raw['alpha'],
            estimated_normal=self.controller.state.normal.copy(),
            observer_uncertainty=self.controller.state.force_uncertainty.copy(),
            contact_model_active=self.controller.state.contact_model_active.copy(),
            estimated_load=float(self.controller.state.load),
            auxiliary_support=float(self.controller.state.auxiliary_support),
             prediction_at=(float(self.controller.last_decision)+self.controller.g.decision_period
                            if prediction_available else np.nan),
             prediction_available=prediction_available,model_invalid=model_invalid,
             observation_unavailable=observation_unavailable,
            transport_paused=bool(debug.get('transport_paused',False)),command=self.last_command,
            local_domain_active=bool(controller_active and not requested_release),
            local_domain_lower=np.asarray(obs.closure_min).copy(),local_domain_upper=np.asarray(obs.closure_max).copy(),
            warning_count=truth['warning_count'],debug_json=json.dumps(jsonable(debug),ensure_ascii=False),
            raw_patches_json=json.dumps(jsonable(raw['patches']),ensure_ascii=False),
            observation_json=json.dumps(jsonable(vars(obs)),ensure_ascii=False),
            observed_normal=obs.normal,validity=obs.valid,observed_age=obs.age,
            observed_auxiliary_wrench=obs.auxiliary_wrench,observed_auxiliary_valid=obs.auxiliary_valid,
            collision=float(sum(p['normal'] for p in raw['patches'] if int(p.get('external_geom',-1))==self.obstacle_geom)),
            aggregation_error=obs.aggregation_error,distributed_error=obs.distributed_error,
            environment_json=json.dumps(jsonable(env)),actual_motion_target=np.r_[lift_ref,wrist],
            executed_motion_reference=np.r_[self.plant.lift_ref,self.plant.wrist_ref])
        self.trace.append({k:np.asarray(v).copy() if isinstance(v,np.ndarray) else v for k,v in row.items()})
        self.history.append([t,*np.asarray(truth['normal']).tolist(),*desired.tolist()])
        if t>=DURATION-1e-9:
            self.running=False;self.completed=True;self.event('本局固定时长结束。')
        else:
            self.plant.set_environment(env);self.plant.advance(SENSOR_PERIOD);self.tick+=1
            self.last_truth=self.plant.truth()
        return self.snapshot()


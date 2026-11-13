"""Independent sensor-tick carrier copied from the preserved V15 runtime.

The public observation, native execution, physics stepping and trace fields are
preserved. Only pre-policy preparation is replaced with immediate policy control
and a public contact-gated pickup task. Original module remains untouched.
"""
import json
import time
import numpy as np
from session import DURATION, SENSOR_PERIOD, event_schedule, jsonable
from task_contract import select_task_mode

PRESERVED_KERNEL_SHA256 = '1a451a34deeab6ed31ff0fb5a14ea54e6a4857826c7d859498c76026eb26e79d'


class ColdStartKernel:
    def step_one_sensor_tick(self):
            if self.paused or not self.running:return self.snapshot()
            loop_began=time.perf_counter_ns()
            callback_start_ns=self.plant.actor_callback_total_ns
            t=self.time
            if abs(self.plant.time-t)>1e-7:raise RuntimeError('physics and session clocks diverged')
            truth,raw,obs,common_sensor_json,observation_ms=self.observe_public()
            pickup_target=self._pickup_motion_request(t,obs) if t<12.-1e-9 else None
            scheduled=self.mode=='scheduled'
            env,lift_ref,wrist=event_schedule(self.scenario_name,t,self.config.friction,self.config.lift_target) if scheduled else (dict(force=[0.,0.,0.],torque=[0.,0.,0.],squeeze=0.,obstacle_x=0.,friction=self.config.friction),self.config.lift_target,[0.,0.])
            if self.motion_mode in ('hold','transport'):
                public_target=pickup_target if t<9.3-1e-9 else self.user_trajectory.position_at(t)
                lift_ref=float(public_target[0]);wrist=list(public_target[1:])
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
            executed_reference=np.r_[self.plant.lift_ref,self.plant.wrist_ref]
            release_active=bool(requested_release or self.plant.release_started is not None)
            clear_confirmed=bool(self.release_driver is not None and self.release_driver.was_clear)
            task_mode=select_task_mode('release' if release_active else 'hold',True,
                np.r_[lift_ref,wrist],executed_reference,release_active=release_active,clear_confirmed=clear_confirmed)
            release_origin=self.release_at if self.release_at is not None else self.plant.release_started
            deadline=(float(t if release_origin is None else release_origin)+3.5) if release_active else 12.
            self.task_request=self._public_task(t,task_mode,np.r_[lift_ref,wrist],executed_reference,deadline,release_active)
            action=np.zeros(5);prediction=np.full(5,np.nan);desired=np.zeros(5);debug={};controller_active=False
            began=time.perf_counter_ns()
            # This controller owns every actuator command, beginning at t=0.
            # No preparation writer, handoff gate, forced grasp or fallback servo.
            obs.closure_min=self.local_lower.copy();obs.closure_max=self.local_upper.copy()
            obs.closure_permission=np.asarray(raw['closure_permission'],bool).copy()
            if requested_release:
                if self.release_at is None:
                    self.release_at=t
                    physical=truth['normal']>.2
                    self.release_supported=bool(np.count_nonzero(physical)>=3 and physical[0] and truth['all_hand_object']>.2)
                    self.plant.begin_release()
                    self.event('12 秒释放指令：执行本策略对应的放下与张手控制。')
                obs.closure_min=self.plant.closure_min.copy();obs.closure_max=self.plant.closure_max.copy()
            self.last_command='release' if requested_release else 'hold'
            if requested_release:
                desired,prediction,debug=self._lifecycle_step(obs)
            else:
                desired,prediction,debug=self._execute_joint(obs,self.last_command)
            action=np.full(5,np.nan);controller_active=True
            debug.update(local_domain_active=False,mechanical_domain_active=not requested_release,
                         release_domain_override=bool(requested_release),
                         local_domain_lower=obs.closure_min.tolist(),local_domain_upper=obs.closure_max.tolist(),
                         cold_start_controller_active=True,common_preparation_used=False,
                         pickup_task_phase=self.pickup_task.phase)
            self.control_ms=(time.perf_counter_ns()-began)/1e6
            self.last_desired=desired;self.last_prediction=prediction;self.last_action=action;self.last_debug=debug
            observation_unavailable=bool(controller_active and not np.any(obs.valid))
            # No pressure predictor exists in V14. Unknown is not numerical failure.
            model_invalid=False;prediction_available=False
            prediction=np.full(5,np.nan);self.last_prediction=prediction.copy()
            state=self.controller.state
            estimated_load=np.nan if state.load is None else float(state.load)
            auxiliary_support=np.nan if state.auxiliary_support is None else float(state.auxiliary_support)
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
                requested_reference21=(self.plant.last_execution_receipt21.requested.values.copy() if self.plant.last_execution_receipt21 is not None else self.plant.reference_snapshot().values.copy()),
                executed_reference21=self.plant.reference_snapshot().values.copy(),
                reference_increment21=self.plant.reference_snapshot().values-obs.robot_reference21,
                mode_generation=self.plant.mode_generation,
                observation_ms=observation_ms,
                full_control_loop_ms=(time.perf_counter_ns()-loop_began)/1e6,
                executed_action=(self.plant.alpha-raw['alpha']) if self.plant._joint_committed is None else np.full(5,np.nan),
                estimated_normal=self.controller.state.normal.copy(),
                posterior_mean_time_s=np.nan,tactile_acquisition_time=np.asarray(state.normal_time).copy(),
                observer_uncertainty=self.controller.state.force_uncertainty.copy(),
                contact_model_active=np.zeros(5,bool),contact_model_available=False,
                observed_contact=state.contact.copy(),
                estimated_load=estimated_load,estimated_load_available=bool(state.load_valid),
                auxiliary_support=auxiliary_support,auxiliary_support_available=bool(state.auxiliary_valid),
                 prediction_at=np.nan,
                 prediction_available=prediction_available,model_invalid=model_invalid,
                 observation_unavailable=observation_unavailable,
                transport_paused=bool(debug.get('transport_paused',False)),command=self.last_command,
                local_domain_active=False,mechanical_domain_active=bool(controller_active and not requested_release),
                local_domain_lower=np.asarray(obs.closure_min).copy(),local_domain_upper=np.asarray(obs.closure_max).copy(),
                warning_count=truth['warning_count'],debug_json=json.dumps(jsonable(debug),ensure_ascii=False),
                raw_patches_json=json.dumps(jsonable(raw['patches']),ensure_ascii=False),
                observation_json=json.dumps(jsonable(vars(obs)),ensure_ascii=False),
                legacy_common_sensor_json=common_sensor_json,
                receiver_measurement_json=json.dumps(obs.receiver_surface,ensure_ascii=False,allow_nan=False),
                receiver_release_tasks_json=json.dumps(jsonable(obs.receiver_release_tasks),ensure_ascii=False,allow_nan=False),
                robot_actuator_target21=obs.robot_actuator_target21.copy(),robot_control_time_s=obs.robot_control_time_s,
                observed_normal=obs.normal,validity=obs.valid,observed_age=obs.age,
                observed_auxiliary_wrench=obs.auxiliary_wrench,observed_auxiliary_valid=obs.auxiliary_valid,
                collision=float(sum(p['normal'] for p in raw['patches'] if int(p.get('external_geom',-1))==self.obstacle_geom)),
                aggregation_error=obs.aggregation_error,distributed_error=obs.distributed_error,
                environment_json=json.dumps(jsonable(env)),actual_motion_target=np.r_[lift_ref,wrist],
                task_request_json=json.dumps(self.task_request,ensure_ascii=False,allow_nan=False),
                executed_motion_reference=np.r_[self.plant.lift_ref,self.plant.wrist_ref])
            row.update(pickup_task_phase=self.pickup_task.phase,
                       pickup_lift_requested=self.pickup_task.lift_started is not None)
            self.trace.append({k:np.asarray(v).copy() if isinstance(v,np.ndarray) else v for k,v in row.items()})
            self.history.append([t,*np.asarray(truth['normal']).tolist(),*desired.tolist()])
            self.trace[-1]['full_control_loop_ms']=(time.perf_counter_ns()-loop_began)/1e6
            self.trace[-1]['physics_advance_ms']=0.
            self.trace[-1]['actor_callback_ms']=(self.plant.actor_callback_total_ns-callback_start_ns)/1e6
            if t>=DURATION-1e-9:
                self.running=False;self.completed=True;self.event('本局固定时长结束。')
            else:
                self.plant.set_environment(env)
                advance_began=time.perf_counter_ns()
                self.plant.advance(SENSOR_PERIOD);self.tick+=1
                self.trace[-1]['physics_advance_ms']=(time.perf_counter_ns()-advance_began)/1e6
                self.trace[-1]['actor_callback_ms']=(self.plant.actor_callback_total_ns-callback_start_ns)/1e6
                self.last_truth=self.plant.truth()
            return self.snapshot()

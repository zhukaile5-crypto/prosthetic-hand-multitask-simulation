"""V13A1 independent sole21 runtime; fixed native decisions and sampling preserved.

Source reference: v10/runtime.py ee20-era final source, original TrialSession.
Only physical handoff, public telemetry, controller construction and honest
measurement-only logging differ. No V10/servo-bank import or old force-B gate
enters new policy decisions. This is a NEW comparator, not old full-run equality.
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
from numbers import Real
import ast,hashlib,json,sys,time,gzip
from .design import HOLDING_REFERENCE_RATE21,RELEASE_REFERENCE_RATE21,DECISION_PERIOD_S
import numpy as np
import research_imports
from research_imports import ROOT,FOUNDATION
from session import HANDOFF,DURATION,SENSOR_PERIOD,event_schedule,jsonable
from .bootstrap import JointBootstrap
from v13_a1.runtime import source_files as a1_source_files
from task_contract import build_task_request,select_task_mode,validate_task_request
from task_trajectory import constant_trajectory,public_transport_trajectory,unknown_future_point

METHODS=('ratio_effort','fixed_effort','fusion_effort')
MOTION_MODES=('scenario_default','hold','transport')
RATIO_TASK_GAIN_KEYS=('base','ratio_floor','ratio_kp','ratio_ki','slip_boost_rate','baseline_tracking_gain','release_rate')
V12Native_REFERENCE_RATE_LIMITS=(.035,.12,.12)
LEGACY_MODEL_FIELDS=('force_matrix','force_tau','force_gain','force_gain_role','model_reference_alpha','response_error_bound')


def source_files(methods=None):
    """Actual new import closure + bootstrap admission/instruments/robot assets.

    No directory scan of old research controllers or empirical servo kernels.
    Foundation old policy/model files remain only because TrialSession creates
    a temporary ratio bootstrap and admission validates its historical evidence.
    """
    if methods is not None and any(m not in METHODS for m in methods):raise ValueError('Unknown V12Native method')
    pending=list(a1_source_files(methods).values())+[ROOT/'v13_joint/runtime.py',ROOT/'v12_native/controller.py',ROOT/'research_imports.py',
        ROOT/'skin_geometry.py',ROOT/'evaluation.py',ROOT/'protocol.json']
    for name in ('__init__.py','run_development.py'):
        q=ROOT/'v13_joint'/name
        if q.is_file():pending.append(q)
    admission_path=FOUNDATION/'review/admission_manifest.json'
    admission=json.loads(admission_path.read_text(encoding='utf-8-sig'));pending.append(admission_path)
    pending.extend(FOUNDATION/n for n in admission['source_sha256'])
    pending.extend(FOUNDATION/e['path'] for e in admission['evidence'].values())
    calibration_path=FOUNDATION/admission['evidence']['calibration']['path']
    calibration=json.loads(calibration_path.read_text(encoding='utf-8-sig'))
    for item in calibration['shapes'].values():
        item={**item,**item.get('response',{})};chosen=item.get('selected_holdout',{})
        for key in ('source_path','source','summary_path'):
            if chosen.get(key):
                q=Path(chosen[key]);q=q if q.is_absolute() else FOUNDATION/q
                if not q.resolve().is_relative_to(FOUNDATION.resolve()):raise ValueError('Bootstrap evidence path escaped foundation')
                pending.append(q)
    # HandPlant actually reads this XML plus all supplied mesh buffers.
    pending.append(FOUNDATION/'assets/legacy_model.xml')
    pending.extend(q for q in (FOUNDATION/'assets/shadow/assets').iterdir() if q.is_file())
    visited=set();paths={}
    while pending:
        p=Path(pending.pop()).resolve()
        if p in visited:continue
        visited.add(p)
        if not p.is_file():raise FileNotFoundError('Required V12Native input: '+str(p))
        if p.is_relative_to(ROOT.resolve()):key=p.relative_to(ROOT).as_posix()
        elif p.is_relative_to(FOUNDATION.resolve()):key='foundation/'+p.relative_to(FOUNDATION).as_posix()
        else:raise ValueError('Source outside project/foundation')
        paths[key]=p
        if p.suffix!='.py':continue
        package_parent=p.parent
        package_root=ROOT.resolve() if p.is_relative_to(ROOT.resolve()) else FOUNDATION.resolve()
        while package_parent!=package_root and package_parent.is_relative_to(package_root):
            init=package_parent/'__init__.py'
            if init.is_file():pending.append(init)
            package_parent=package_parent.parent
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):names=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    base=p.parent
                    for _ in range(node.level-1):base=base.parent
                    stem=base.joinpath(*(node.module or '').split('.')) if node.module else base
                    opts=[stem.with_suffix('.py'),stem/'__init__.py'] if node.module else [base/'__init__.py']
                    opts += [stem/n.name for n in node.names if n.name!='*']
                    for q in opts:
                        if q.is_file():pending.append(q)
                        elif q.with_suffix('.py').is_file():pending.append(q.with_suffix('.py'))
                    continue
                names=[node.module] if node.module else []
            else:continue
            for name in names:
                relative=Path(*name.split('.'))
                for base in (ROOT,FOUNDATION):
                    q=next((q for q in (base/relative.with_suffix('.py'),base/relative/'__init__.py') if q.is_file()),None)
                    if q is not None:pending.append(q);break
    return dict(sorted(paths.items()))


def source_identity(methods=None):
    return {key:hashlib.sha256(p.read_bytes()).hexdigest() for key,p in source_files(methods).items()}

class PublicPreviewFollower:
    """Shared finite20ms ramp from actual reference to known user waypoint."""
    def __init__(self,period_s=.02,control_dt_s=.005,maximum_reference_rate=V12Native_REFERENCE_RATE_LIMITS):
        if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) for x in (period_s,control_dt_s)):
            raise ValueError('Reference clocks must be real numeric values')
        self.period=float(period_s);self.dt=float(control_dt_s)
        if not np.isfinite([self.period,self.dt]).all() or self.period<=0 or self.dt<=0 or self.dt>self.period:raise ValueError('Invalid ramp clocks')
        if isinstance(maximum_reference_rate,(str,bytes,dict)):raise ValueError('Explicit three-axis reference rate required')
        rates=tuple(maximum_reference_rate)
        if len(rates)!=3 or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) for x in rates):raise ValueError('Three numeric reference rates required')
        self.maximum_reference_rate=np.asarray(rates,float)
        if not np.isfinite(self.maximum_reference_rate).all() or np.any(self.maximum_reference_rate<=0):raise ValueError('Positive finite reference rates required')
        self.start_time=None;self.start=None;self.end=None;self.last_time=None;self.last_reason='not_started'
        self.last_diagnostics={}

    def reset(self):self.start_time=None;self.start=None;self.end=None

    def _executed_reference(self,request,actual,proposal):
        proposal=np.asarray(proposal,float)
        delta=proposal-actual;maximum=self.maximum_reference_rate*self.dt
        executed_delta=np.minimum(maximum,np.maximum(-maximum,delta))
        actual_command=actual+executed_delta
        self.last_diagnostics=dict(reference_rate_limits=self.maximum_reference_rate.tolist(),
            control_dt_s=self.dt,requested_ramp_reference=proposal.tolist(),
            executed_reference=actual_command.tolist(),requested_reference_increment=delta.tolist(),
            actual_reference_increment=executed_delta.tolist(),rate_limited_axes=(abs(delta)>maximum+1e-12).tolist(),
            ramp_tube_error=(proposal-actual_command).tolist(),
            user_target_error=(np.asarray(request['motion_target'],float)-actual_command).tolist(),
            task_request_modified=False)
        return actual_command

    def step(self,task_request,pause_transport=False):
        request=validate_task_request(task_request)
        now=float(request['time_s']);actual=np.asarray(request['executed_motion_reference'],float)
        if self.last_time is not None and now<self.last_time-1e-9:raise ValueError('Preview clock rewind')
        self.last_time=now
        if request['mode'] in ('release','monitor'):
            self.reset();self.last_reason='release_preempts_preview';self.last_diagnostics=dict(release_preempts_reference_envelope=True,task_request_modified=False);return actual.copy()
        if type(pause_transport) is not bool:raise ValueError('Explicit pause flag required')
        if pause_transport:
            self.reset();self.last_reason='traditional_force_strategy_pause';self.last_diagnostics=dict(force_strategy_pause=True,task_request_modified=False);return actual.copy()
        preview=request.get('motion_preview')
        if not preview or not preview['future_reference_available']:
            self.reset();self.last_reason='future_user_reference_unavailable';return self._executed_reference(request,actual,request['motion_target'])
        if self.start_time is None or now>=self.start_time+self.period-1e-9:
            times=np.asarray(preview['requested_times_s'],float);target_time=now+self.period
            if len(times)<2 or times[-1]<target_time-1e-9:raise ValueError('20ms public preview missing')
            targets=np.asarray(preview['motion_targets'],float)
            endpoint=np.array([np.interp(target_time,times,targets[:,i]) for i in range(3)])
            bounds=request.get('motion_reference_bounds')
            if bounds is not None and (np.any(endpoint<np.asarray(bounds['lower'])-1e-10) or np.any(endpoint>np.asarray(bounds['upper'])+1e-10)):raise ValueError('Preview exceeds actual robot authority')
            self.start_time=now;self.start=actual.copy();self.end=endpoint.copy()
        fraction=min(1.,max(0.,(now-self.start_time+self.dt)/self.period))
        self.last_reason='known_user_20ms_finite_ramp'
        return self._executed_reference(request,actual,self.start+fraction*(self.end-self.start))

def public_motion_reference_bounds(plant):
    """Robot-only command authority, including lift gravity compensation.

    The common plant clips lift references to [0, configured lift target]. Wrist
    authority is the intersection of its actual joint and actuator intervals.
    No object state or contact parameter participates in this computation.
    """
    model=plant.model
    lower=np.array([0.,-np.inf,-np.inf]);upper=np.array([plant.config.lift_target,np.inf,np.inf])
    gravity=float(model.body_subtreemass[plant.hand])*9.81/1500.
    for i,(joint,actuator) in enumerate((('transport_lift','lift_motor'),('rh_WRJ2','rh_A_WRJ2'),('rh_WRJ1','rh_A_WRJ1'))):
        jid=int(model.joint(joint).id);aid=int(model.actuator(actuator).id)
        if model.jnt_limited[jid]:
            lower[i]=max(lower[i],float(model.jnt_range[jid,0]));upper[i]=min(upper[i],float(model.jnt_range[jid,1]))
        if model.actuator_ctrllimited[aid]:
            offset=gravity if i==0 else 0.
            lower[i]=max(lower[i],float(model.actuator_ctrlrange[aid,0])-offset)
            upper[i]=min(upper[i],float(model.actuator_ctrlrange[aid,1])-offset)
    if not np.isfinite(np.r_[lower,upper]).all() or np.any(lower>upper):
        raise ValueError('Robot motion command authority is missing or empty')
    return dict(lower=lower.tolist(),upper=upper.tolist())

class JointSession(JointBootstrap):
    def __init__(self,method='fusion_effort',**kwargs):
        self.motion_mode=kwargs.pop('motion_mode','scenario_default')
        if method not in METHODS or self.motion_mode not in MOTION_MODES:raise ValueError('Unknown V12Native method/motion mode')
        if kwargs.get('split')=='confirmation':raise ValueError('V12Native is development only; no frozen confirmation')
        super().__init__(method='ratio',**kwargs)
        actual=deepcopy(self.gains)
        if set(actual)!=set(RATIO_TASK_GAIN_KEYS):raise ValueError('Full actual seven-key ratio gains required')
        self.method=method;self.actual_ratio_task_gains=actual;self.gains=deepcopy(actual)
        self.release_driver=None;self.lifecycle_enabled=False;self.release_selection=None
        self.task_request=None;self.observer_warmup_s=.2
        self.legacy_model_domain_supported=None;self.legacy_model_lower=None;self.legacy_model_upper=None
        from .controller import create_controller
        public_context=dict(config=dict(shape=self.shape),nominal_force_weights=deepcopy(self.calibration['nominal_force_weights']))
        self.controller=create_controller(method,public_context,self.limits,self.budget,actual,self.plant.model)
        from skin_geometry import DelayedSkinGeometry
        self.skin_geometry=DelayedSkinGeometry(self.plant)
        self.motion_reference_bounds=public_motion_reference_bounds(self.plant)
        transport=(self.motion_mode=='transport' or self.motion_mode=='scenario_default' and self.mode=='scheduled' and self.scenario_name=='dynamic_transport')
        self.user_trajectory=public_transport_trajectory(self.config.lift_target) if transport else constant_trajectory(self.config.lift_target)
        device=self.sensors.noise
        from contracts import WRENCH_LENGTH
        sf=device.force_sigma if device.auxiliary_force_sigma is None else device.auxiliary_force_sigma
        sm=sf*WRENCH_LENGTH if device.auxiliary_moment_sigma is None else device.auxiliary_moment_sigma
        self.sensor_characteristics=dict(force_sigma_N=float(device.force_sigma),tangential_sigma_N=float(device.tangential_sigma),
            position_sigma_m=float(device.position_sigma),slip_sigma_m_s=float(device.slip_sigma),
            effort_sigma=float(device.effort_sigma),tactile_delay_s=float(device.delay),
            auxiliary_force_sigma_N=float(sf),auxiliary_moment_sigma_Nm=float(sm))
        finger=np.concatenate(self.plant.actuator_groups)
        self.robot_actuator_ids21=np.r_[finger,self.plant.lift_actuator,
            self.plant.aid('rh_A_WRJ2'),self.plant.aid('rh_A_WRJ1')].astype(int)
        self.robot_actuator_axes21=[self.plant.model.actuator(int(i)).name for i in self.robot_actuator_ids21]
        if len(self.robot_actuator_ids21)!=21 or len(set(self.robot_actuator_ids21))!=21:raise ValueError('Actual21 robot engine target axes required')
        self.research_sources=source_identity([method])

    def _handoff(self,obs,truth):
        self.handoff_attempted=True
        self.reference_relative=np.asarray(truth['relative']).copy() # evaluation only
        touches=obs.valid & np.isfinite(obs.normal) & (obs.normal>.2)
        ready=(self.plant.phase=='HOLDING' and np.count_nonzero(touches)>=3 and touches[0]
               and abs(obs.lift-self.config.lift_target)<=.01)
        self.handoff_alpha=obs.alpha.copy()
        ref=np.asarray(self.calibration.get('reference_alpha',self.calibration.get('model_reference_alpha')),float)
        radius=self.trust_radius
        lower=np.maximum(self.plant.closure_min,np.maximum(ref-radius,obs.alpha-radius))
        upper=np.minimum(self.plant.closure_max,np.minimum(ref+radius,obs.alpha+radius))
        self.legacy_model_domain_supported=bool(np.all(lower<=obs.alpha+1e-9) and np.all(upper>=obs.alpha-1e-9))
        self.legacy_model_lower=lower;self.legacy_model_upper=upper
        self.local_lower=self.plant.closure_min.copy();self.local_upper=self.plant.closure_max.copy()
        self.handoff_passed=bool(ready) # NOT conditioned on old force-model support
        if not ready:
            self.preparation_failure='preparation_not_ready';self.failure_reason=self.preparation_failure;self.failure_at=self.time
            self.event('共同物理准备未满足接管条件，本局仍保留为失败。')
        else:self.event('共同物理准备结束，新公共观测/执行策略接管；旧模型域仅诊断。')

    def _public_task(self,t,mode,target,executed,deadline,release_active):
        preview=self.user_trajectory.preview(t) if self.mode=='scheduled' and not release_active else unknown_future_point(t,target)
        return build_task_request(t,mode,target,executed,deadline,self.limits,self.budget,
            tracking_tolerances=[.01,.015,.015],motion_preview=preview,
            motion_reference_bounds=self.motion_reference_bounds,sensor_characteristics=self.sensor_characteristics)

    def observe_public(self):
        started=time.perf_counter_ns()
        t=self.time
        truth=self.plant.truth();raw=self.plant.raw_sensors();obs=self.sensors.sample(raw)
        self.skin_geometry.acquire(raw,obs)
        common_sensor_json=json.dumps(jsonable(vars(obs)),ensure_ascii=False)
        for key in LEGACY_MODEL_FIELDS:
            vars(obs).pop(key,None)
        obs.robot_actuator_target21=self.plant.data.ctrl[self.robot_actuator_ids21].copy()
        obs.robot_control_time_s=float(self.plant.time)
        obs.robot_actuator_axes21=list(self.robot_actuator_axes21)
        obs.robot_position25=np.r_[obs.raw_joint,obs.lift].copy()
        obs.robot_velocity25=np.r_[obs.raw_joint_velocity,float(raw['lift_velocity'])].copy()
        # Current executed robot reference is public and common to every policy.
        # Never attach raw qpos, environment or object truth to the observation.
        obs.executed_motion_reference=np.r_[self.plant.lift_ref,self.plant.wrist_ref].copy()
        obs.sensor_characteristics=deepcopy(self.sensor_characteristics)
        obs.motion_reference_rate_limits=np.asarray(V12Native_REFERENCE_RATE_LIMITS,float).copy()
        obs.lift_velocity_m_s=float(raw['lift_velocity']) # public robot encoder velocity, supplied to every method
        if not all(np.isfinite(truth[k]).all() for k in ('normal','skin_normal','position','qpos','qvel')):
            raise RuntimeError('Nonfinite physical state; run stopped and retained as failure')
        self._attach_execution_state(obs)
        observation_finished=time.perf_counter_ns()
        self.last_truth=truth;self.last_obs=obs
        return truth,raw,obs,common_sensor_json,(observation_finished-started)/1e6

    def _attach_execution_state(self,obs):
        obs.robot_reference21=self.plant.reference_snapshot().values.copy()
        obs.reference_lower21,obs.reference_upper21=self.plant.reference_authority21()
        obs.mode_generation=self.plant.mode_generation
        entry=self.plant.joint_release_entry
        obs.release_entry_reference21=None if entry is None else entry.reference21.values.copy()
        obs.release_nominal_reference21=None if entry is None else self.plant.release_proposal(float(obs.time)+DECISION_PERIOD_S).values.copy()

    def _execute_joint(self,obs,command):
        self._attach_execution_state(obs)
        self.controller.set_task_request(self.task_request)
        result=self.controller.step(obs,self.plant.phase,command)
        receipt=self.plant.commit_reference21(result.reference21,mode=result.mode,expected_generation=result.mode_generation)
        self.controller.reconcile(receipt)
        return self.controller.desired.copy(),np.full(5,np.nan),deepcopy(self.controller.debug)

    def _lifecycle_step(self,obs):
        return self._execute_joint(obs,'release')

    def save(self,directory):
        meta=super().save(directory)
        path=Path(directory)/'joint_decisions.jsonl.gz'
        with gzip.open(path,'xt',encoding='utf-8') as f:
            for record in self.controller.decision_evidence:f.write(json.dumps(jsonable(record),ensure_ascii=False,allow_nan=False)+'\n')
        evidence=dict(schema='v13_joint_decision_evidence_v1',records=len(self.controller.decision_evidence),file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (Path(directory)/'joint_decision_manifest.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
        return meta

    def snapshot(self):
        value=super().snapshot()
        value['domain_label']='公共物理接管与真实机械权限；旧力模型局部域仅诊断，不作为V12Native共同门'
        value['legacy_model_domain_supported']=self.legacy_model_domain_supported
        return value

    def metadata(self):
        value=super().metadata()
        value.update(schema='v13_joint_current_effort_trial_v1',benchmark_id='v13_joint_current_effort_development',
            research_phase='development',research_method=self.method,research_source_sha256=self.research_sources,
            motion_mode=self.motion_mode,observer_warmup_s=self.observer_warmup_s,
            actual_ratio_task_gains=self.actual_ratio_task_gains,
            legacy_model_domain_supported=self.legacy_model_domain_supported,
            legacy_model_bounds_diagnostic=dict(lower=jsonable(self.legacy_model_lower),upper=jsonable(self.legacy_model_upper)),
            controller_scope='V12Native public tactile measurement + force intent + shared effort actor; no force-B domain gate/prediction',
            bootstrap_calibration_scope='historical foundation admission, temporary ratio construction and sensor instrument initialization only; not V12Native decision input',
            controller_calibration_keys=['config.shape','nominal_force_weights'],
            release_mode='actual21_entry_reference_and_public_force_unload',pressure_prediction_available=False,
            legacy_trace_columns=dict(contact_model_active='allfalse placeholder, contact_model_available=False; use observed_contact',
                posterior_mean_time_s='NaN: no model posterior; use tactile_acquisition_time5',
                estimated_load='NaN when unavailable, see estimated_load_available'),
            robot_actuator_telemetry=dict(axes21=self.robot_actuator_axes21,
                role='actual18finger+lift engine target(with gravity compensation)+2wrist; excludes obstacle actuator; current robot read clock'),
            task_contract=dict(schema='public-hand-task-request-v2',tracking_tolerances=[.01,.015,.015],
                task_deadline_s=12.,release_duration_deadline_s=3.5,
                user_motion_trajectory=self.user_trajectory.as_dict(),motion_reference_bounds=self.motion_reference_bounds,
                sensor_characteristics=self.sensor_characteristics),
            new_comparator_not_frozen_paper_reproduction=True,performance_claim_allowed=False)
        value['execution_actor']='frozen_A1_sole_robot21_writer_with_committed21_authority'
        value['joint_design']=dict(decision_period_s=.020,control_period_s=.005,holding_reference_rate21=list(HOLDING_REFERENCE_RATE21),release_reference_rate21=list(RELEASE_REFERENCE_RATE21),software_policy_not_hardware_limit=True)
        value['legacy_trace_columns'].update(alpha='legacy synergy memory only after joint takeover; not full21 reference',requested_action='NaN after joint takeover; use requested_reference21',executed_action='NaN after joint takeover; use reference_increment21')
        value['shared_execution']=dict(role='same native model and declared21 software policy for all force intents',independent_actuator_references=21)
        value['controller_scope']='public current-state joint force-intent and actuator-effort allocation; no pressure forecast'
        return value


    def step_one_sensor_tick(self):
            if self.paused or not self.running:return self.snapshot()
            loop_began=time.perf_counter_ns()
            callback_start_ns=self.plant.actor_callback_total_ns
            t=self.time
            if abs(self.plant.time-t)>1e-7:raise RuntimeError('physics and session clocks diverged')
            truth,raw,obs,common_sensor_json,observation_ms=self.observe_public()
            scheduled=self.mode=='scheduled'
            env,lift_ref,wrist=event_schedule(self.scenario_name,t,self.config.friction,self.config.lift_target) if scheduled else (dict(force=[0.,0.,0.],torque=[0.,0.,0.],squeeze=0.,obstacle_x=0.,friction=self.config.friction),self.config.lift_target,[0.,0.])
            if self.motion_mode in ('hold','transport'):
                public_target=self.user_trajectory.position_at(t)
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
            task_mode=select_task_mode('release' if release_active else 'hold',bool(self.handoff_passed),
                np.r_[lift_ref,wrist],executed_reference,release_active=release_active,clear_confirmed=clear_confirmed)
            release_origin=self.release_at if self.release_at is not None else self.plant.release_started
            deadline=(float(t if release_origin is None else release_origin)+3.5) if release_active else 12.
            self.task_request=self._public_task(t,task_mode,np.r_[lift_ref,wrist],executed_reference,deadline,release_active)
            action=np.zeros(5);prediction=np.full(5,np.nan);desired=np.zeros(5);debug={};controller_active=False
            began=time.perf_counter_ns()
            # Release intent has priority even before preparation/handoff and for
            # the common driverNone native primitive. No force strategy may resume
            # preparation merely because it lacks a specialized lifecycle driver.
            early_lifecycle_release=bool(requested_release
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
                if t>=HANDOFF-self.observer_warmup_s-1e-9:self.controller.state.update(obs,SENSOR_PERIOD)
            else:
                if not self.handoff_attempted:self._handoff(obs,truth)
                if self.handoff_passed:
                    if self.task_request['mode']=='preparation':
                        mode=select_task_mode('release' if release_active else 'hold',True,
                            np.r_[lift_ref,wrist],executed_reference,release_active=release_active,
                            clear_confirmed=clear_confirmed)
                        self.task_request=self._public_task(t,mode,np.r_[lift_ref,wrist],executed_reference,deadline,release_active)
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
                    if requested_release:
                        desired,prediction,debug=self._lifecycle_step(obs);controller_active=True
                    else:
                        if hasattr(self.controller,'set_task_request'):
                            self.controller.set_task_request(self.task_request)
                        elif hasattr(self.controller,'set_motion_request'):
                            self.controller.set_motion_request(np.r_[lift_ref,wrist],
                                np.r_[self.plant.lift_ref,self.plant.wrist_ref])
                        desired,prediction,debug=self._execute_joint(obs,self.last_command)
                        action=np.full(5,np.nan);controller_active=True
                    debug['local_domain_active']=False
                    debug['mechanical_domain_active']=not requested_release
                    debug['release_domain_override']=bool(requested_release)
                    debug['local_domain_lower']=obs.closure_min.tolist();debug['local_domain_upper']=obs.closure_max.tolist()
                else:
                    if self.plant.release_started is None:self.plant.begin_release()
                    desired,prediction,debug=self._lifecycle_step(obs)
                    action=np.full(5,np.nan)
                    debug.update(policy_not_executed=True,failure_reason=self.preparation_failure)
            self.control_ms=(time.perf_counter_ns()-began)/1e6
            self.last_desired=desired;self.last_prediction=prediction;self.last_action=action;self.last_debug=debug
            observation_unavailable=bool(controller_active and not np.any(obs.valid))
            # No pressure predictor exists in V12Native. Unknown is not numerical failure.
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
                robot_actuator_target21=obs.robot_actuator_target21.copy(),robot_control_time_s=obs.robot_control_time_s,
                observed_normal=obs.normal,validity=obs.valid,observed_age=obs.age,
                observed_auxiliary_wrench=obs.auxiliary_wrench,observed_auxiliary_valid=obs.auxiliary_valid,
                collision=float(sum(p['normal'] for p in raw['patches'] if int(p.get('external_geom',-1))==self.obstacle_geom)),
                aggregation_error=obs.aggregation_error,distributed_error=obs.distributed_error,
                environment_json=json.dumps(jsonable(env)),actual_motion_target=np.r_[lift_ref,wrist],
                task_request_json=json.dumps(self.task_request,ensure_ascii=False,allow_nan=False),
                executed_motion_reference=np.r_[self.plant.lift_ref,self.plant.wrist_ref])
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

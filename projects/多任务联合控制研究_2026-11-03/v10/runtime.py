"""Independent V10 kernel with retained preparation/sampling/execution.
Only policy construction and explicit public preview following differ.
Original session source SHA256: fd22dfa8be4bf7e16217227abda9fe3ef0cda87febf5ab5a5a007ea32edee656
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import time,json,ast,hashlib,sys
from numbers import Real
import numpy as np
import research_imports
from research_imports import ROOT, FOUNDATION
from session import TrialSession, HANDOFF, DURATION, SENSOR_PERIOD, event_schedule, jsonable, sha
from task_contract import build_task_request, select_task_mode
from task_trajectory import constant_trajectory, public_transport_trajectory, unknown_future_point
from task_contract import validate_task_request

METHODS=('ratio','passive','unified','joint','ratio_preview','passive_preview')
MOTION_MODES=('scenario_default','hold','transport')
RATIO_TASK_GAIN_KEYS=('base','ratio_floor','ratio_kp','ratio_ki','slip_boost_rate','baseline_tracking_gain','release_rate')
# A shared development planning envelope, not a new claim about certified
# hardware rate limits. Native traditional comparators keep their old decoder.
V10_REFERENCE_RATE_LIMITS=(.035,.12,.12)


def source_files(methods=None):
    """Local import closure plus exact admitted robot assets/evidence.

    Snapshot keys include foundation/ for external platform inputs. Package
    sources have their own V10 paths, never an old V6 runtime identity.
    """
    if methods is not None and any(m not in METHODS for m in methods):raise ValueError('Unknown V10 method')
    paths={}
    pending=[ROOT/'v10'/n for n in ('__init__.py','runtime.py','run_development.py','controller.py','directional_allocator.py','robot_contact_space.py','servo_bank.py')]
    pending.append(ROOT/'v9/__init__.py')
    from v10.servo_bank import Bank
    from v10.directional_allocator import ROBOT_AXES25
    bank=Bank(ROBOT_AXES25)
    if bank.load_errors or len(bank.entries)!=15:raise ValueError('All15 frozen empirical servo contexts required for this development version')
    for path,digest in bank.source_identity().items():
        p=Path(path).resolve()
        if not p.is_relative_to(ROOT.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Servo source scope/identity mismatch')
        pending.append(p)
    pending += [ROOT/n for n in ('research_imports.py','task_contract.py','task_trajectory.py','skin_geometry.py','passive_control.py','evaluation.py','protocol.json')]
    pending += [ROOT/'calibration'/n for n in ('public_model_phase1_v1.json','public_model_phase2_v1.json','public_contact_model_phase2_v1.json')]
    admission_path=FOUNDATION/'review/admission_manifest.json'
    admission=json.loads(admission_path.read_text(encoding='utf-8-sig'));pending.append(admission_path)
    pending.extend(FOUNDATION/n for n in admission['source_sha256'])
    pending.extend(FOUNDATION/e['path'] for e in admission['evidence'].values())
    calibration=json.loads((FOUNDATION/admission['evidence']['calibration']['path']).read_text(encoding='utf-8-sig'))
    for item in calibration['shapes'].values():
        item={**item,**item.get('response',{})};chosen=item.get('selected_holdout',{})
        if not chosen:
            radius=float(item.get('selected_radius',item.get('trust_radius')))
            chosen=next((e for e in item.get('holdout_by_radius',{}).values() if abs(float(e.get('radius',-1))-radius)<1e-10),{})
        for k in ('source_path','source','summary_path'):
            if chosen.get(k):
                p=Path(chosen[k]);p=p if p.is_absolute() else FOUNDATION/p
                if not p.resolve().is_relative_to(FOUNDATION.resolve()):raise ValueError('Foundation evidence escaped root')
                pending.append(p)
    visited=set()
    while pending:
        p=Path(pending.pop()).resolve()
        if p in visited:continue
        visited.add(p)
        if not p.is_file():raise FileNotFoundError('Required V10 source: '+str(p))
        if p.is_relative_to(ROOT.resolve()):key=p.relative_to(ROOT).as_posix()
        elif p.is_relative_to(FOUNDATION.resolve()):key='foundation/'+p.relative_to(FOUNDATION).as_posix()
        else:raise ValueError('Source outside research/foundation roots')
        paths[key]=p
        if p.suffix!='.py':continue
        tree=ast.parse(p.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):names=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    base=p.parent
                    for _ in range(node.level-1):base=base.parent
                    module=node.module or '';stem=base.joinpath(*module.split('.')) if module else base
                    options=[stem.with_suffix('.py'),stem/'__init__.py'] if module else [base/'__init__.py']
                    options += [stem/n.name for n in node.names if n.name!='*']
                    for candidate in options:
                        if candidate.is_file():pending.append(candidate)
                        elif candidate.with_suffix('.py').is_file():pending.append(candidate.with_suffix('.py'))
                    continue
                names=[node.module] if node.module else []
            for name in names:
                relative=Path(*name.split('.'))
                for base in (ROOT,FOUNDATION):
                    found=next((q for q in (base/relative.with_suffix('.py'),base/relative/'__init__.py') if q.is_file()),None)
                    if found is not None:pending.append(found);break
    return dict(sorted(paths.items()))


def source_identity(methods=None):
    return {key:hashlib.sha256(p.read_bytes()).hexdigest() for key,p in source_files(methods).items()}


class PublicPreviewFollower:
    """Shared finite20ms ramp from actual reference to known user waypoint."""
    def __init__(self,period_s=.02,control_dt_s=.005,maximum_reference_rate=V10_REFERENCE_RATE_LIMITS):
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

class V10Session(TrialSession):
    def __init__(self,method='joint',**kwargs):
        self.motion_mode=kwargs.pop('motion_mode','scenario_default')
        if self.motion_mode not in MOTION_MODES:raise ValueError('Unknown public motion mode')
        if method not in METHODS:raise ValueError('Unknown V10 research method')
        if kwargs.get('split')=='confirmation':
            raise ValueError('V10 confirmation is not frozen or admitted; development only')
        holding_method=method.removesuffix('_preview')
        # Foundation initialization is retained, including exact ratio task
        # gains for joint. Its temporary ratio state is never sampled/updated
        # before the V10 factory replaces it; no old joint constructor is called.
        super().__init__(method=holding_method if holding_method in ('ratio','unified') else 'ratio',**kwargs)
        self.method=method
        self.release_driver=None
        self.task_request=None
        self.lifecycle_enabled=False
        self.release_selection=None
        self.preview_follower=PublicPreviewFollower() if method.endswith('_preview') else None
        self.observer_warmup_s=.2
        if holding_method=='joint':
            from v10.controller import create_joint
            if set(self.gains)!=set(RATIO_TASK_GAIN_KEYS):
                raise ValueError('V10 joint requires the complete actual admitted ratio7-key task gains')
            actual_ratio_gains=deepcopy(self.gains)
            self.controller=create_joint(self.calibration,self.limits,self.budget,gains=actual_ratio_gains,robot_model=self.plant.model)
            self.actual_ratio_task_gains=actual_ratio_gains
            self.gains=asdict(self.controller.g)
        elif holding_method=='passive':
            from passive_control import create_passive
            self.controller=create_passive(self.calibration,self.limits,self.budget)
            self.gains=asdict(self.controller.g)
        from skin_geometry import DelayedSkinGeometry
        self.skin_geometry=DelayedSkinGeometry(self.plant)
        self.motion_reference_bounds=public_motion_reference_bounds(self.plant)
        transport=(self.motion_mode=='transport' or self.motion_mode=='scenario_default'
            and self.mode=='scheduled' and self.scenario_name=='dynamic_transport')
        self.user_trajectory=(public_transport_trajectory(self.config.lift_target)
            if transport else constant_trajectory(self.config.lift_target))
        device=self.sensors.noise
        from contracts import WRENCH_LENGTH
        auxiliary_force_sigma=(device.force_sigma if device.auxiliary_force_sigma is None else device.auxiliary_force_sigma)
        auxiliary_moment_sigma=(auxiliary_force_sigma*WRENCH_LENGTH if device.auxiliary_moment_sigma is None else device.auxiliary_moment_sigma)
        self.sensor_characteristics=dict(force_sigma_N=float(device.force_sigma),
            tangential_sigma_N=float(device.tangential_sigma),position_sigma_m=float(device.position_sigma),
            slip_sigma_m_s=float(device.slip_sigma),effort_sigma=float(device.effort_sigma),
            tactile_delay_s=float(device.delay),auxiliary_force_sigma_N=float(auxiliary_force_sigma),
            auxiliary_moment_sigma_Nm=float(auxiliary_moment_sigma))
        self.research_sources=source_identity([method])

    def _public_task(self,t,mode,target,executed,deadline,release_active):
        preview=(self.user_trajectory.preview(t) if self.mode=='scheduled' and not release_active
            else unknown_future_point(t,target))
        return build_task_request(t,mode,target,executed,deadline,self.limits,self.budget,
            tracking_tolerances=[.01,.015,.015],motion_preview=preview,
            motion_reference_bounds=self.motion_reference_bounds,sensor_characteristics=self.sensor_characteristics)

    def _lifecycle_step(self,obs):
        driver=self.release_driver
        if driver is None:
            delta=-np.minimum(np.maximum(0.,self.plant.alpha-self.plant.closure_min),.8*SENSOR_PERIOD)
            self.plant.execute_delta(delta)
            debug=dict(release_authority=True,prediction_available=False,
                prediction_unavailable_reason='public legacy release primitive outside held model',
                release_primitive='legacy_mechanical',monitoring_active=True)
        else:
            driver.request(obs,self.plant.data.ctrl[driver.aids],self.plant.lift_ref,self.plant.wrist_ref)
            command=driver.step(obs);self.plant.apply_release_command(command)
            debug=deepcopy(command.debug)
        self.controller.state.update(obs,SENSOR_PERIOD)
        self.controller.desired=np.zeros(5);self.controller.predicted=np.full(5,np.nan)
        return np.zeros(5),np.full(5,np.nan),debug

    def metadata(self):
        value=super().metadata()
        policy_rows=[r for r in self.trace if r['controller_active'] and r['command']!='release']
        value.update(schema='v10_task_research_trial_v1',motion_mode=self.motion_mode,
            public_robot_lift_velocity='same current robot encoder velocity supplied to every method; no external object state',
            benchmark_id='v10_joint_task_research_development_v1',
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
            release_mode='legacy_mechanical_path',
            release_experimental=self.lifecycle_enabled,release_primitive_selection=self.release_selection,
            public_skin_geometry='delayed per-patch robot link/position/normal; shared finger pressure noise; no external object identity',
            task_contract=dict(schema='public-hand-task-request-v2',tracking_tolerances=[.01,.015,.015],
                wrist_tolerance_role='new engineering task target for development; not a retrospective old acceptance rule',
                task_deadline_s=12.,release_duration_deadline_s=3.5,
                user_motion_trajectory=self.user_trajectory.as_dict(),motion_reference_bounds=self.motion_reference_bounds,
                sensor_characteristics=self.sensor_characteristics,
                information='explicit user motion preview, executed reference and robot/device specifications; no object truth or future disturbance schedule'),
            controller_scope='V10 development; preserved common preparation and native mechanical release; no model/domain admission',
            v10_package_version='v10_development_unadmitted',
            observer_warmup_s=self.observer_warmup_s,
            observer_warmup_scope='state update only before handoff; all methods0.2s; joint uses the single baseline force posterior',
            v10_joint_belief_owner='baseline_force_posterior_plus_robot_encoder_geometry_only' if self.method=='joint' else 'unchanged_traditional_force_policy_state',
            public_motion_follower='shared20ms_public_preview_finite_ramp' if self.preview_follower is not None else 'joint_planner' if self.method=='joint' else 'native_current_user_reference',
            v10_reference_rate_envelope=dict(values=list(V10_REFERENCE_RATE_LIMITS),units=['m/s','rad/s','rad/s'],
                role='shared V10 development planner/preview envelope; not certified original hardware authority',
                applied_to=['joint','ratio_preview','passive_preview'],native_traditional_decoder_unchanged=True),
            actual_ratio_task_gains=getattr(self,'actual_ratio_task_gains',None))
        return value

    def step_one_sensor_tick(self):
        if self.paused or not self.running:return self.snapshot()
        t=self.time
        if abs(self.plant.time-t)>1e-7:raise RuntimeError('physics and session clocks diverged')
        truth=self.plant.truth();raw=self.plant.raw_sensors();obs=self.sensors.sample(raw)
        self.skin_geometry.acquire(raw,obs)
        # Current executed robot reference is public and common to every policy.
        # Never attach raw qpos, environment or object truth to the observation.
        obs.executed_motion_reference=np.r_[self.plant.lift_ref,self.plant.wrist_ref].copy()
        obs.sensor_characteristics=deepcopy(self.sensor_characteristics)
        obs.motion_reference_rate_limits=np.asarray(V10_REFERENCE_RATE_LIMITS,float).copy()
        obs.lift_velocity_m_s=float(raw['lift_velocity']) # public robot encoder velocity, supplied to every method
        if not all(np.isfinite(truth[k]).all() for k in ('normal','skin_normal','position','qpos','qvel')):
            raise RuntimeError('Nonfinite physical state; run stopped and retained as failure')
        self.last_truth=truth;self.last_obs=obs
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
                if requested_release and self.lifecycle_enabled:
                    desired,prediction,debug=self._lifecycle_step(obs);controller_active=True
                else:
                    if hasattr(self.controller,'set_task_request'):
                        self.controller.set_task_request(self.task_request)
                    elif hasattr(self.controller,'set_motion_request'):
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
                    if self.preview_follower is not None:
                        motion=self.preview_follower.step(self.task_request,pause_transport=bool(result.pause_transport))
                        self.plant.set_motion(float(motion[0]),motion[1:])
                        debug['public_preview_follower']=self.preview_follower.last_reason
                        debug['public_preview_execution']=deepcopy(self.preview_follower.last_diagnostics)
                    elif joint_motion is not None:
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
        # An old controller can retain its last prediction internally on sensor
        # dropout. Preserve that control state, but do not publish it as a current
        # prediction. Availability and a missing prediction clock now agree.
        if not prediction_available:
            prediction=np.full(5,np.nan);self.last_prediction=prediction.copy()
        posterior_time=getattr(self.controller.state,'posterior_mean_time_s',None)
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
            posterior_mean_time_s=np.nan if posterior_time is None else float(posterior_time),
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
            task_request_json=json.dumps(self.task_request,ensure_ascii=False,allow_nan=False),
            executed_motion_reference=np.r_[self.plant.lift_ref,self.plant.wrist_ref])
        self.trace.append({k:np.asarray(v).copy() if isinstance(v,np.ndarray) else v for k,v in row.items()})
        self.history.append([t,*np.asarray(truth['normal']).tolist(),*desired.tolist()])
        if t>=DURATION-1e-9:
            self.running=False;self.completed=True;self.event('本局固定时长结束。')
        else:
            self.plant.set_environment(env);self.plant.advance(SENSOR_PERIOD);self.tick+=1
            self.last_truth=self.plant.truth()
        return self.snapshot()


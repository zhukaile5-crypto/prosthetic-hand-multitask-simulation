"""One fixed-tick run kernel shared by the browser and paired batch trials.

No current policy is constructed or executed before explicit evidence admission.
Preparation is shared and is never credited as an algorithm-specific grasp win.
"""
from __future__ import annotations

from dataclasses import dataclass,asdict,fields
from collections import deque
from pathlib import Path
from copy import deepcopy
import hashlib,json,time,uuid
import numpy as np

from contracts import SensorNoise,SENSOR_PERIOD,BENCHMARK_ID
from foundation_config import FoundationConfig
from hand_plant import HandPlant,CommonPreparation
from sensors import HandSensors

ROOT=Path(__file__).resolve().parent
APP_ID='pbl-shadow-strategy-integration-v1'
HANDOFF=7.5
DURATION=16.
SCENARIOS=[
 dict(name='nominal',title='常规持握',shape='capsule'),
 dict(name='cylinder_small',title='较小柱体与低摩擦',shape='capsule',size=.95,friction=.45),
 dict(name='box_heavy',title='较高载荷盒体',shape='box',mass=.15),
 dict(name='sphere',title='球体持握',shape='sphere'),
 dict(name='friction_drop',title='持握中摩擦突降',shape='capsule'),
 dict(name='three_pulses',title='三次横向拉动',shape='capsule'),
 dict(name='load_ramp',title='悬挂载荷逐步增加',shape='capsule'),
 dict(name='fragile_conflict',title='防滑与限力并发',shape='capsule',limits=[3.5,2.8,2.8,2.8,2.8],budget=12.),
 dict(name='local_limit',title='局部受限接触与拉动',shape='capsule',limits=[4.5,1.85,4.5,4.5,4.5],budget=16.),
 dict(name='release_slip',title='拉动中主动释放',shape='capsule'),
 dict(name='dynamic_transport',title='升降及腕部姿态扰动',shape='capsule'),
 dict(name='sensor_noise',title='触觉高噪声',shape='capsule'),
 dict(name='sensor_dropout',title='局部传感丢包',shape='capsule'),
 dict(name='collision',title='手与有限质量障碍物接触',shape='capsule'),
 dict(name='infeasible',title='极低摩擦高载荷边界',shape='capsule',friction=.18,mass=.20,boundary=True),
 dict(name='contact_robustness',title='接触参数与组合扰动',shape='capsule',solref=.012),
]
SCENE_BY_NAME={s['name']:s for s in SCENARIOS}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def jsonable(value):
    if isinstance(value,np.ndarray):return jsonable(value.tolist())
    if isinstance(value,np.generic):return jsonable(value.item())
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def validate_radius_evidence(calibration):
    """A larger local domain requires its own immutable passed holdout."""
    selected=calibration.get('selected_radius',calibration.get('trust_radius'))
    if selected is None:raise ValueError('Public hand calibration must select a validated radius')
    radius=float(selected)
    if not np.isfinite(radius) or not 0.<radius<=.040000001:
        raise ValueError('Selected local radius must be positive and at most 0.04')
    evidence=calibration.get('selected_holdout',{})
    if not evidence:
        by_radius=calibration.get('holdout_by_radius',{})
        for key in (str(radius),f'{radius:.2f}',f'{radius:.3f}'):
            if key in by_radius:evidence=by_radius[key];break
    if evidence.get('passed') is not True or abs(float(evidence.get('radius',radius))-radius)>1e-10:
        raise ValueError('Selected hand response radius has no passed independent holdout')
    source=evidence.get('source_path',evidence.get('source'))
    expected=evidence.get('source_sha256',evidence.get('sha256'))
    if not source or not expected:raise ValueError('Selected holdout must bind its source path and hash')
    path=Path(source);path=path if path.is_absolute() else ROOT/path
    if not path.is_file() or sha(path)!=expected:raise ValueError('Selected holdout source identity mismatch')
    summary=evidence.get('summary_path');summary_hash=evidence.get('summary_sha256')
    if not summary or not summary_hash:raise ValueError('Selected holdout must bind its metric summary')
    summary_path=Path(summary);summary_path=summary_path if summary_path.is_absolute() else ROOT/summary_path
    if not summary_path.is_file() or sha(summary_path)!=summary_hash:
        raise ValueError('Selected holdout metric-summary identity mismatch')
    rmse=float(evidence.get('force_rmse_N',float('inf')));p95=float(evidence.get('force_p95_N',float('inf')))
    if not (np.isfinite(rmse) and np.isfinite(p95) and rmse<=.35 and p95<=.75):
        raise ValueError('Selected holdout exceeds preregistered 0.35 N RMSE / 0.75 N p95 limits')
    return radius

def check_admission(path=None):
    """Verify immutable evidence identities, not just a UI boolean flag."""
    path=ROOT/'review/admission_manifest.json' if path is None else Path(path)
    problems=[];report={}
    if not path.is_file():
        return dict(passed=False,problems=['公共物理与独立审查尚未签发接入许可。'],manifest=str(path))
    try:
        report=load_json(path)
        if report.get('passed') is not True:problems.append('接入证书未通过。')
        if report.get('benchmark_id')!=BENCHMARK_ID:problems.append('基准版本身份不一致。')
        required=('physical','independent','protocol','calibration','experiment_plan')
        evidence=report.get('evidence',{})
        for name in required:
            item=evidence.get(name,{})
            p=ROOT/item.get('path','__missing__')
            if not p.is_file() or sha(p)!=item.get('sha256'):
                problems.append(f'{name}证据缺失或哈希不一致。');continue
            if name in ('physical','independent') and load_json(p).get('passed') is not True:
                problems.append(f'{name}证据尚未通过。')
        loaded={name:load_json(ROOT/evidence[name]['path']) for name in required
                if name in evidence and (ROOT/evidence[name].get('path','__missing__')).is_file()}
        if loaded.get('experiment_plan',{}).get('status')!='frozen':
            problems.append('实验矩阵仍为草案，尚未冻结。')
        expected_protocol=evidence.get('protocol',{}).get('sha256')
        for name in ('physical','independent','calibration'):
            if loaded.get(name,{}).get('protocol_sha256')!=expected_protocol:
                problems.append(f'{name}与当前验收协议的哈希链不一致。')
        independent=loaded.get('independent',{})
        if independent.get('physical_certificate_sha256')!=evidence.get('physical',{}).get('sha256'):
            problems.append('独立审查未绑定当前物理证书。')
        if independent.get('calibration_sha256')!=evidence.get('calibration',{}).get('sha256'):
            problems.append('独立审查未绑定当前公开校准。')
        sources=report.get('source_sha256',{})
        mandatory=('hand_plant.py','foundation_config.py','contracts.py','sensors.py',
                   'state_estimator.py','force_model.py','controller_adapter.py','policy_sources.py',
                   'session.py','evaluate_runs.py')
        for name in mandatory:
            p=ROOT/name
            if name not in sources or not p.is_file() or sha(p)!=sources[name]:
                problems.append(f'{name}未冻结或已经变化。')
        for name,expected in sources.items():
            p=ROOT/name
            if not p.is_file() or sha(p)!=expected:
                problem=f'{name}来源身份不一致。'
                if problem not in problems:problems.append(problem)
        calibration_item=evidence.get('calibration',{})
        p=ROOT/calibration_item.get('path','__missing__')
        if p.is_file():
            cal=load_json(p)
            if not all(shape in cal.get('shapes',{}) for shape in ('capsule','box','sphere')):
                problems.append('三形状公开校准不完整。')
            else:
                for shape,c in cal['shapes'].items():
                    if 'response' in c:c={**c,**c['response']}
                    if 'nominal_force_weights' not in c:problems.append(f'{shape}公共名义指力分配未明确标定。')
                    try:validate_radius_evidence(c)
                    except (ValueError,TypeError) as exc:problems.append(f'{shape}局部模型域证据无效：{exc}')
    except Exception as exc:problems.append(f'接入证据读取失败：{exc}')
    return dict(passed=not problems,problems=problems,manifest=str(path),
                manifest_sha256=sha(path),record=report)

def _evidence_path(admission,name):
    return ROOT/admission['record']['evidence'][name]['path']

def physical_parameters(scene,seed,split='demonstration',shape=None):
    s=deepcopy(SCENE_BY_NAME[scene]);s['shape']=shape or s['shape']
    values={k:s[k] for k in ('shape','size','mass','friction','solref') if k in s}
    rng=np.random.default_rng(int(seed)+511_000)
    # Fixed draw shapes; randomized parameters are shared within each pair.
    u=rng.normal(size=4)
    if split=='robustness':
        values['mass']=float(values.get('mass',.12)*np.clip(1.+.06*u[0],.85,1.15))
        values['friction']=float(values.get('friction',.60)*np.clip(1.+.08*u[1],.80,1.20))
    values['seed']=int(seed)
    return s,values

def event_schedule(name,t,base_friction,lift_target):
    env=dict(force=[0.,0.,0.],torque=[0.,0.,0.],squeeze=0.,obstacle_x=0.,friction=base_friction)
    lift=lift_target;wrist=[0.,0.]
    if name in ('friction_drop','contact_robustness') and t>=9.3:env['friction']=max(.12,.55*base_friction)
    if name in ('three_pulses','local_limit','contact_robustness'):
        for start,amount in ((9.3,.6),(10.3,.9),(11.3,1.2)):
            if start<=t<start+.30:env['force'][1]+=amount
    if name=='load_ramp' and t>=9.3:env['force'][2]=-.9*min(1.,(t-9.3)/2.)
    if name=='fragile_conflict' and 9.3<=t<10.:env['force'][1]=1.;env['squeeze']=.02
    if name=='release_slip' and 11.7<=t<12.4:env['force'][1]=1.2
    if name=='dynamic_transport' and 9.3<=t<12.:
        lift=lift_target-.008*(1.-np.cos((t-9.3)*3.))
        wrist=[.04*np.sin((t-9.3)*3.),.025*np.sin((t-9.3)*2.)]
    if name=='collision' and 9.3<=t<10.3:env['obstacle_x']=-.30
    return env,lift,wrist


class TrialSession:
    def __init__(self,method='ratio',shape=None,scenario='nominal',seed=540101,
                 mode='scheduled',split='demonstration',admission_path=None):
        if method not in ('ratio','unified'):raise ValueError('unknown admitted method')
        if scenario not in SCENE_BY_NAME:raise ValueError('unknown scenario')
        if mode not in ('scheduled','manual'):raise ValueError('unknown run mode')
        if split not in ('demonstration','development','confirmation','robustness'):raise ValueError('unknown experiment split')
        if split!='demonstration' and mode!='scheduled':raise ValueError('manual demonstrations cannot enter a formal matrix')
        self.admission=check_admission(admission_path)
        if not self.admission['passed']:raise RuntimeError(' '.join(self.admission['problems']))
        self.method=method;self.scenario_name=scenario;self.seed=int(seed);self.mode=mode;self.split=split
        plan=load_json(_evidence_path(self.admission,'experiment_plan'))
        if split=='demonstration':
            reserved={int(seed) for phase in plan.get('phases',{}).values() for seed in phase.get('seeds',[])}
            reserved|={int(job['seed']) for phase in plan.get('phases',{}).values() for job in phase.get('jobs',[])}
            if self.seed in reserved:raise ValueError('该种子保留用于开发或确认，请使用独立演示种子540101或查看已有回放。')
        scene,values=physical_parameters(scenario,seed,split,shape)
        self.scene=scene;self.shape=scene['shape']
        all_calibration=load_json(_evidence_path(self.admission,'calibration'))
        self.calibration=deepcopy(all_calibration['shapes'][self.shape])
        if 'response' in self.calibration:
            self.calibration={**self.calibration,**self.calibration['response']}
        if 'nominal_force_weights' not in self.calibration:
            raise ValueError('Production hand calibration must declare shared nominal_force_weights')
        self.trust_radius=validate_radius_evidence(self.calibration)
        bare=all_calibration.get('bare_hand',{})
        self.calibration.update({k:v for k,v in bare.items() if k in ('effort_bias_N','load_error_bound_N')})
        defaults=all_calibration.get('foundation_defaults',{})
        allowed={f.name for f in fields(FoundationConfig)}
        model_values={**self.calibration.get('config',{}),**defaults,**values}
        self.config=FoundationConfig(**{k:v for k,v in model_values.items() if k in allowed})
        self.plant=HandPlant(self.config)
        self.obstacle_geom=int(self.plant.model.geom('integration_obstacle_geom').id)
        self.preparation=CommonPreparation(self.plant)
        noise=dict(force_sigma=.01,tangential_sigma=.005,position_sigma=.0001,
                   slip_sigma=.0002,effort_sigma=.02,delay=.010)
        if scenario=='sensor_noise':noise.update(force_sigma=.08,tangential_sigma=.04,slip_sigma=.002)
        if scenario=='sensor_dropout':noise['dropout_windows']=((9.3,9.7,(1,2)),(11.3,11.4,(0,1,2,3,4)))
        self.sensors=HandSensors(self.calibration,SensorNoise(**noise),seed=self.seed)
        settings=load_json(ROOT/'vendor/frozen_settings.json')
        self.gains=deepcopy(settings.get('methods',settings.get('gains',{})).get(method,{}))
        if not self.gains:
            self.gains=deepcopy(settings.get('controllers',{}).get(method,{}))
        self.gains['release_rate']=float(self.config.release_open_rate)
        from controller_adapter import create_controller
        self.limits=np.asarray(scene.get('limits',[5.,5.,5.,5.,5.]),float)
        self.budget=float(scene.get('budget',18.))
        self.controller=create_controller(method,self.calibration,self.limits,self.budget,self.gains)
        self.run_id=uuid.uuid4().hex[:12];self.tick=0
        self.running=True;self.paused=False;self.completed=False;self.saved=False
        self.handoff_attempted=False;self.handoff_passed=False;self.handoff_alpha=None
        self.reference_relative=None;self.release_at=None;self.release_supported=False
        self.preparation_failure=None;self.failure_reason=None;self.failure_at=None
        self.events=[];self.commands=[];self.trace=[];self.history=deque(maxlen=3300)
        self.camera=dict(azimuth=150.,elevation=-15.,distance=.68)
        self.manual_force_until=0.;self.manual_squeeze_until=0.;self.manual_release=False
        self.control_ms=0.;self.last_truth=self.plant.truth();self.last_obs=None
        self.last_desired=np.zeros(5);self.last_prediction=np.full(5,np.nan)
        self.last_action=np.zeros(5);self.last_debug={};self.last_command='hold'
        self.environment_signature=None
        self.event('共同准备开始；当前方法尚未接管。')

    @property
    def time(self):return self.tick*SENSOR_PERIOD

    def event(self,message):self.events.append(dict(time=self.time,message=message))

    def command(self,action,value=None,sequence=None):
        if action=='camera':
            for key in ('azimuth','elevation','distance'):
                if isinstance(value,dict) and key in value:self.camera[key]=float(value[key])
            self.camera['elevation']=float(np.clip(self.camera['elevation'],-85.,20.))
            self.camera['distance']=float(np.clip(self.camera['distance'],.35,1.3))
        elif action=='pause':self.paused=bool(value)
        elif action in ('disturb','squeeze','release'):
            if self.mode!='manual':raise ValueError('固定日程由仿真时钟触发，请在手动演示中操作。')
            if action=='disturb':self.manual_force_until=self.time+.4
            elif action=='squeeze':self.manual_squeeze_until=self.time+.4
            else:self.manual_release=True
            self.event(dict(disturb='手动施加拉动。',squeeze='手动施加过力扰动。',release='用户请求主动释放。')[action])
        else:raise ValueError('unknown per-run command')
        record=dict(sequence=sequence,time=self.time,tick=self.tick,action=action,value=jsonable(value))
        self.commands.append(record);return record

    def _handoff(self,obs,truth):
        self.handoff_attempted=True;self.reference_relative=np.asarray(truth['relative']).copy()
        touches=obs.valid & np.isfinite(obs.normal) & (obs.normal>.2)
        ready=(self.plant.phase=='HOLDING' and np.count_nonzero(touches)>=3 and touches[0]
               and abs(obs.lift-self.config.lift_target)<=.01)
        ref=np.asarray(self.calibration.get('reference_alpha',self.calibration.get('model_reference_alpha')),float)
        radius=self.trust_radius
        self.handoff_alpha=obs.alpha.copy()
        lower=np.maximum(self.plant.closure_min,np.maximum(ref-radius,obs.alpha-radius))
        upper=np.minimum(self.plant.closure_max,np.minimum(ref+radius,obs.alpha+radius))
        domain=bool(np.all(lower<=obs.alpha+1e-9) and np.all(upper>=obs.alpha-1e-9))
        self.local_lower=lower;self.local_upper=upper
        self.handoff_passed=bool(ready and domain)
        if not self.handoff_passed:
            self.preparation_failure='preparation_not_ready' if not ready else 'handoff_outside_calibrated_local_domain'
            self.failure_reason=self.preparation_failure;self.failure_at=self.time
            self.event('共同准备未满足接管条件，本局仍保留为失败。')
        else:self.event('共同准备结束，所选策略开始接管。')

    def step_one_sensor_tick(self):
        if self.paused or not self.running:return self.snapshot()
        t=self.time
        if abs(self.plant.time-t)>1e-7:raise RuntimeError('physics and session clocks diverged')
        truth=self.plant.truth();raw=self.plant.raw_sensors();obs=self.sensors.sample(raw)
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
        if t<HANDOFF-1e-9:
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
                result=self.controller.step(obs,self.plant.phase,self.last_command)
                action=np.asarray(result.delta_closure,float);prediction=np.asarray(result.predicted_normal,float)
                desired=self.controller.desired.copy();debug=deepcopy(result.debug or {});controller_active=True
                before=self.plant.alpha.copy();self.plant.execute_delta(action)
                debug['executed_delta_closure']=(self.plant.alpha-before).tolist()
                debug['local_domain_active']=not requested_release
                debug['release_domain_override']=bool(requested_release)
                debug['local_domain_lower']=obs.closure_min.tolist();debug['local_domain_upper']=obs.closure_max.tolist()
                if not requested_release and not result.pause_transport:self.plant.set_motion(lift_ref,wrist)
                if result.pause_transport:debug['transport_paused']=True
            else:
                if self.plant.release_started is None:self.plant.begin_release()
                self.plant.execute_delta(-np.minimum(np.maximum(0.,self.plant.alpha-self.plant.closure_min),.8*SENSOR_PERIOD))
                debug={'policy_not_executed':True,'failure_reason':self.preparation_failure}
        self.control_ms=(time.perf_counter_ns()-began)/1e6
        self.last_desired=desired;self.last_prediction=prediction;self.last_action=action;self.last_debug=debug
        row=dict(time=t,normal=truth['normal'],skin_normal=truth['skin_normal'],shear=truth['shear'],
            palm=truth['palm'],hand_other=truth['hand_other'],all_hand_object=truth['all_hand_object'],
            support=truth['support'],penetration=truth['penetration'],world_penetration=truth['world_penetration'],
            position=truth['position'],relative=truth['relative'],rotation=truth['rotation'],
            relative_velocity=truth['relative_velocity'],angular_velocity=truth['angular_velocity'],
            lift=truth['lift'],lift_ref=lift_ref,qpos=truth['qpos'],qvel=truth['qvel'],
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
                           if controller_active and not requested_release else np.nan),
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

    def snapshot(self):
        s=self.last_truth
        return jsonable(dict(app=APP_ID,run_id=self.run_id,time=self.time,method=self.method,shape=self.shape,
            scenario=self.scenario_name,seed=self.seed,mode=self.mode,split=self.split,phase=self.plant.phase,
            running=self.running,paused=self.paused,completed=self.completed,recording=self.running,
            controller_active=self.handoff_passed and self.time>=HANDOFF,
            normal=np.asarray(s['normal']).tolist(),palm=float(s['palm']),
            object_lift=float(s['position'][2]-self.plant.initial_position[2]),control_ms=self.control_ms,
            history=list(self.history),events=self.events[-12:],camera=self.camera,
            message=self.failure_reason or ('当前策略已接管。' if self.handoff_passed else '共同准备；尚不计作算法抓取结果。'),
            domain_label=f'固定抓姿、公开校准参考附近±{self.trust_radius:.3f}协同坐标以内；含掌部被动接触',
            handoff_passed=self.handoff_passed))

    def render(self,renderer):
        import mujoco
        cam=mujoco.MjvCamera();cam.lookat[:]=[.045,0.,.36]
        for key,value in self.camera.items():setattr(cam,key,value)
        renderer.update_scene(self.plant.data,camera=cam)
        return renderer.render()

    def metadata(self):
        return jsonable(dict(schema='shadow_hand_trial_v1',benchmark_id=BENCHMARK_ID,run_id=self.run_id,
            method=self.method,shape=self.shape,scenario=self.scene,seed=self.seed,mode=self.mode,split=self.split,
            config=asdict(self.config),duration=DURATION,sensor_period=SENSOR_PERIOD,handoff_time=HANDOFF,
            physics_timestep=self.config.timestep,initial_position=self.plant.initial_position,
            limits=self.limits,budget=self.budget,gains=self.gains,noise=asdict(self.sensors.noise),
            calibration=self.calibration,admission_sha256=self.admission['manifest_sha256'],
            source_sha256=self.admission['record']['source_sha256'],model_sha256=self.plant.model_sha256,
            handoff_passed=self.handoff_passed,handoff_alpha=self.handoff_alpha,
            preparation_failure=self.preparation_failure,reference_relative=self.reference_relative,
            reference_kind='paired common-preparation endpoint before either policy acts; evaluation only',
            release_at=self.release_at,release_supported_at_command=self.release_supported,
            mechanical_release_at=self.plant.release_started,
            max_hidden_ctrl_clip=self.plant.max_hidden_clip,
            failure_reason=self.failure_reason,failure_at=self.failure_at,completed=self.completed,
            shared_execution=dict(release_rate=float(self.config.release_open_rate),
                role='public physically accepted mechanical rate shared by both methods'),
            commands=self.commands,events=self.events,local_trust_radius=self.trust_radius,
            controller_scope='local post-preparation policy migration; no arbitrary grasp acquisition claim'))

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        arrays={k:np.asarray([r[k] for r in self.trace]) for k in self.trace[0]} if self.trace else {}
        sub=np.asarray(self.plant.substep_records,float)
        if len(sub):
            arrays.update(substep_time=sub[:,0],substep_duration=sub[:,1],substep_normal=sub[:,2:7],
                substep_skin_normal=sub[:,7:12],substep_palm=sub[:,12],substep_hand_other=sub[:,13],
                substep_all_hand_object=sub[:,14],substep_support=sub[:,15],substep_penetration=sub[:,16],
                substep_world_penetration=sub[:,17])
        meta=self.metadata();arrays['metadata']=json.dumps(meta,ensure_ascii=False)
        np.savez_compressed(directory/'trace.npz',**arrays)
        meta['trace_sha256']=sha(directory/'trace.npz')
        (directory/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        self.saved=True
        return meta

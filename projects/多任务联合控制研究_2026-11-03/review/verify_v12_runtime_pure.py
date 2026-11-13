"""Actual runtime method bodies with fake devices; no plant/MuJoCo import."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import ast,hashlib,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
PATH=ROOT/'v12/runtime.py'


def run():
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    code=PATH.read_text(encoding='utf-8');tree=ast.parse(code)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='V12Session')
    environment=dict(np=np,time=time,json=json,deepcopy=deepcopy,HANDOFF=7.5,DURATION=16.,SENSOR_PERIOD=.005,
        V12_REFERENCE_RATE_LIMITS=(.035,.12,.12),
        LEGACY_MODEL_FIELDS=('force_matrix','force_tau','force_gain','force_gain_role','model_reference_alpha','response_error_bound'),
        TrialSession=object,jsonable=lambda value:value.tolist() if isinstance(value,np.ndarray) else
            {k:environment['jsonable'](v) for k,v in value.items()} if isinstance(value,dict) else
            [environment['jsonable'](v) for v in value] if isinstance(value,(list,tuple)) else value,
        event_schedule=lambda *a:(dict(force=[0.,0.,0.],torque=[0.,0.,0.],squeeze=0.,obstacle_x=0.,friction=.6),.1,[0.,0.]),
        select_task_mode=lambda *a,**kw:'release' if kw.get('release_active') else 'hold')
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(PATH),'exec'),environment)
    Session=environment['V12Session'];checks=[]
    def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))

    class FakePlant:
        def __init__(self,now):
            self.time=now;self.alpha=np.full(5,.55);self.closure_min=np.zeros(5);self.closure_max=np.full(5,1.4)
            self.phase='HOLDING';self.release_started=None;self.lift_ref=.1;self.wrist_ref=np.zeros(2)
            self.data=SimpleNamespace(ctrl=np.arange(22)*.01,qpos=np.zeros(24))
            self.model=SimpleNamespace(joint=lambda name:SimpleNamespace(qposadr=[0 if name=='rh_WRJ2' else 1]))
            self.executions=[];self.advances=0
        def truth(self):
            return dict(normal=np.full(5,2.),skin_normal=np.full(5,2.),shear=np.zeros(5),palm=0.,hand_other=0.,all_hand_object=10.,
                support=0.,penetration=0.,world_penetration=0.,position=np.array([0.,0.,.2]),relative=np.zeros(3),
                rotation=np.eye(3),relative_velocity=np.zeros(3),angular_velocity=np.zeros(3),lift=.1,
                qpos=np.zeros(24),qvel=np.zeros(24),ctrl=self.data.ctrl.copy(),alpha=self.alpha.copy(),warning_count=0)
        def raw_sensors(self):return dict(patches=[dict(digit=i,normal=1.,external_geom=99) for i in range(5)],alpha=self.alpha.copy(),lift_velocity=0.,closure_permission=np.ones(5,bool))
        def execute_delta(self,delta):self.executions.append(np.asarray(delta).copy());self.alpha+=delta
        def set_motion(self,lift,wrist):self.lift_ref=lift;self.wrist_ref=np.asarray(wrist).copy()
        def begin_release(self):self.release_started=self.time;self.phase='RELEASING'
        def set_environment(self,env):pass
        def advance(self,dt):self.advances+=1;self.time+=dt

    def new_session(now,ready=True):
        s=object.__new__(Session);s.time=now;s.tick=round(now/.005);s.paused=False;s.running=True
        s.plant=FakePlant(now);s.config=SimpleNamespace(friction=.6,lift_target=.1)
        if not ready:s.plant.phase='CLOSING'
        s.sensors=SimpleNamespace(sample=lambda raw:SimpleNamespace(time=now,normal=np.full(5,2.),valid=np.ones(5,bool),
            lift=.1,raw_joint=np.zeros(24),raw_joint_velocity=np.zeros(24),alpha=raw['alpha'],
            auxiliary_wrench=np.zeros(6),auxiliary_valid=False,age=.01,aggregation_error=np.zeros(5),distributed_error=np.zeros(5),
            closure_min=np.zeros(5),closure_max=np.full(5,1.4),closure_permission=np.ones(5,bool),
            force_matrix=np.eye(5)*999.,force_tau=.005,force_gain=np.ones(5)*999.,model_reference_alpha=np.zeros(5),response_error_bound=np.ones(5)))
        s.skin_geometry=SimpleNamespace(acquire=lambda *a:None);s.sensor_characteristics=dict(force_sigma_N=.01)
        s.robot_actuator_ids21=np.arange(21);s.robot_actuator_axes21=[str(i) for i in range(21)]
        s.mode='scheduled';s.motion_mode='hold';s.scenario_name='nominal';s.manual_release=False
        s.manual_force_until=s.manual_squeeze_until=0.;s.environment_signature=None;s.release_driver=None
        s.last_command='hold';s.obstacle_geom=99
        s.user_trajectory=SimpleNamespace(position_at=lambda t:np.array([.1,0.,0.]))
        s._public_task=lambda *a:dict(mode=a[1]);s.preparation=SimpleNamespace(step=lambda public:public.copy())
        s.observer_warmup_s=.2;s.handoff_attempted=False;s.handoff_passed=False;s.trust_radius=.02
        s.calibration=dict(reference_alpha=np.full(5,.8));s.release_at=None;s.release_supported=False
        s.trace=[];s.history=[];s.events=[];s.event=lambda text:s.events.append(text);s.snapshot=lambda:dict(time=s.time)
        state=SimpleNamespace(time=0.,normal=np.full(5,np.nan),normal_time=np.full(5,np.nan),valid=np.zeros(5,bool),
            force_uncertainty=np.full(5,np.nan),contact=np.zeros(5,bool),load=None,load_valid=False,
            auxiliary_support=None,auxiliary_valid=False,updates=0)
        def update(obs,dt):state.time=obs.time;state.updates+=1;return state
        state.update=update
        ctrl=SimpleNamespace(state=state,desired=np.full(5,2.),motion_reference=np.array([.1,0.,0.]),calls=0,received=None)
        ctrl.set_task_request=lambda task:None
        def step(obs,phase,command):
            ctrl.calls+=1;ctrl.received=obs;state.update(obs,.005)
            return SimpleNamespace(delta_closure=np.zeros(5),predicted_normal=np.full(5,np.nan),debug={},pause_transport=False)
        ctrl.step=step;s.controller=ctrl
        return s

    s=new_session(7.4);s.step_one_sensor_tick()
    check('preparation_raw_pressure_path_unchanged',s.controller.calls==0 and s.controller.state.updates==1 and s.plant.advances==1 and np.array_equal(s.trace[0]['desired'],np.ones(5)))
    s=new_session(7.5);s.step_one_sensor_tick();row=s.trace[0]
    check('physical_ready_passes_despite_old_model_domain_false',s.handoff_passed and s.legacy_model_domain_supported is False and s.controller.calls==1)
    check('real_mechanical_bounds_not_reference_radius',np.array_equal(s.controller.received.closure_min,np.zeros(5)) and np.array_equal(s.controller.received.closure_max,np.full(5,1.4)))
    check('legacy_force_calibration_removed_from_new_observation',all(k not in vars(s.controller.received) for k in environment['LEGACY_MODEL_FIELDS']))
    check('current_robot_21_actual_engine_targets',np.array_equal(s.controller.received.robot_actuator_target21,np.arange(21)*.01) and s.controller.received.robot_control_time_s==7.5)
    check('robot_qv25_no_object_truth',s.controller.received.robot_position25.shape==(25,) and s.controller.received.robot_velocity25.shape==(25,) and not any(k in vars(s.controller.received) for k in ('position','object_position','mass','friction','target_id')))
    check('unknown_load_aux_and_pressure_prediction_not_zero',np.isnan(row['estimated_load']) and not row['estimated_load_available'] and np.isnan(row['auxiliary_support']) and np.isnan(row['predicted_normal']).all() and not row['prediction_available'])
    check('legacy_model_mask_not_claimed_observation_contact',not row['contact_model_available'] and not row['contact_model_active'].any() and 'observed_contact' in row and np.isnan(row['posterior_mean_time_s']))
    check('raw_sampling_logged_separately', 'force_matrix' in json.loads(row['legacy_common_sensor_json']) and 'force_matrix' not in json.loads(row['observation_json']))
    s=new_session(7.5,ready=False);s.step_one_sensor_tick()
    check('actual_not_ready_preserves_early_native_release',not s.handoff_passed and s.preparation_failure=='preparation_not_ready' and s.plant.release_started==7.5 and s.controller.calls==0)
    s=new_session(12.);s.handoff_attempted=s.handoff_passed=True;s.local_lower=np.zeros(5);s.local_upper=np.full(5,1.4);s.step_one_sensor_tick()
    check('scheduled_release_excludes_holding_controller',s.controller.calls==0 and s.plant.release_started==12. and s.controller.state.updates==1 and not s.trace[0]['desired'].any())
    s=new_session(6.);s.manual_release=True;s.step_one_sensor_tick()
    check('manual_release_preempts_preparation',s.controller.calls==0 and s.plant.release_started==6. and s.trace[0]['command']=='release')

    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    env=dict(ROOT=ROOT,FOUNDATION=FOUNDATION,Path=Path,ast=ast,json=json,hashlib=hashlib,METHODS=('ratio_effort','fixed_effort','fusion_effort'))
    exec(compile(ast.Module(body=[functions['source_files']],type_ignores=[]),str(PATH),'exec'),env)
    files=env['source_files']()
    required=('v12/controller.py','v12/observation.py','v12/intents.py','v12/effort_tracker.py',
              'v11/robot_servo_effort.py','v11/object_belief.py','v10/robot_contact_space.py','v10/__init__.py')
    check('actual_new_dependency_closure',all(n in files for n in required))
    check('no_old_bank_or_empirical_kernel_closure',not any('servo_bank' in n or 'robot_servo_response_' in n or 'public_contact_model_phase2' in n for n in files))
    check('robot_xml_and_mesh_buffers_declared','foundation/assets/legacy_model.xml' in files and any(n.startswith('foundation/assets/shadow/assets/') for n in files))
    init=ast.parse((ROOT/'v10/__init__.py').read_text(encoding='utf-8'))
    check('v10_parent_init_has_no_import_or_call_sideeffects',not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.Call)) for n in ast.walk(init)))
    check('no_v10_runtime_import',not any(isinstance(n,ast.ImportFrom) and n.module in ('v10.runtime','v10.servo_bank') for n in ast.walk(tree)))
    check('bootstrap_constructor_uses_actual7_keys',"super().__init__(method='ratio',**kwargs)" in code and 'set(actual)!=set(RATIO_TASK_GAIN_KEYS)' in code)
    import mujoco
    originals={n:getattr(mujoco,n) for n in ('MjModel','MjData','mj_step','mj_step1','mj_step2')}
    guarded_calls=[]
    def reject(*args,**kwargs):guarded_calls.append('model_or_dynamics');raise AssertionError('No real model/dynamics in runtime pure QA')
    class GuardModel:
        def __init__(self,*args,**kwargs):reject()
        from_xml_string=staticmethod(reject)
        from_xml_path=staticmethod(reject)
    try:
        mujoco.MjModel=GuardModel;mujoco.MjData=reject
        for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,reject)
        import importlib
        actual=importlib.import_module('v12.runtime')
        check('actual_module_import_and_helper_default_resolve',np.array_equal(actual.PublicPreviewFollower().maximum_reference_rate,[.035,.12,.12]) and not guarded_calls)
    finally:
        for name,value in originals.items():setattr(mujoco,name,value)
    return dict(schema='v12_runtime_pure_integration_v1',passed=all(c['passed'] for c in checks),checks=checks,
        actual_source_identity={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()},
        source_sha256=hashlib.sha256(PATH.read_bytes()).hexdigest(),physics_steps=0,actual_model_constructions=0,
        limitation='Actual runtime bodies on fake hardware; pre7.5 physical equivalence requires root-run regression, not asserted by software checks alone')


if __name__=='__main__':
    result=run();out=ROOT/'review/v12_runtime_pure_integration_v2.json'
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=result['passed'],checks=len(result['checks']),failed=[c['name'] for c in result['checks'] if not c['passed']],source_count=len(result['actual_source_identity']),output=str(out))))
    raise SystemExit(0 if result['passed'] else 2)

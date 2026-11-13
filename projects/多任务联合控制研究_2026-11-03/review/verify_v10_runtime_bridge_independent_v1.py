"""AST-extracted actual runtime/control routing with pure mocks. No plant imports."""
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass,asdict
from numbers import Real
from types import SimpleNamespace,ModuleType,MethodType
import ast,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RP=ROOT/'v10/runtime.py';CP=ROOT/'v10/controller.py';VP=ROOT/'v9/runtime.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
initial={str(p.relative_to(ROOT)):sha(p) for p in [RP,CP,VP]}
rt=ast.parse(RP.read_text(encoding='utf-8-sig'));ct=ast.parse(CP.read_text(encoding='utf-8-sig'))
checks=[]
def check(name,value,detail=None):checks.append(dict(name=name,passed=bool(value),detail=detail))
@dataclass
class G:decision_period:float=.02
@dataclass
class ControlAction:
 delta_closure:np.ndarray
 predicted_normal:np.ndarray
 label:str
 pause_transport:bool
 debug:dict
class ForceState:
 def __init__(self):
  self.count=0;self.samples=[];self.time=0.;self.alpha=np.ones(5);self.normal=np.ones(5)*2;self.force_uncertainty=np.ones(5)*.05;self.contact_model_active=np.ones(5,bool);self.load=1.;self.auxiliary_support=0.;self.closure_min=np.zeros(5);self.closure_max=np.ones(5)*2;self.valid=np.ones(5,bool);self.closure_permission=np.ones(5,bool)
 def update(self,f,dt):
  self.count+=1;self.time=f.time;self.alpha=f.alpha.copy();self.samples.append(dict(time=f.time,dt=dt,lift_velocity=f.lift_velocity_m_s,reference=f.executed_motion_reference.copy(),normal=f.normal.copy(),spec=deepcopy(f.sensor_characteristics)))
class Nominal:
 def __init__(self):self.state=ForceState();self.g=G();self.integral=0.;self.boost=0.;self.last_decision=-1e6;self.desired=np.ones(5)*2;self.predicted=np.full(5,np.nan);self.interval_action=np.array([.004,-.002,.001,0.,-.001]);self.pause_transport=False;self.label='mock_original_ratio';self.calls=[]
 def step(self,f,phase,command):
  self.state.update(f,f.dt);self.calls.append((f.time,command))
  if command=='release':delta=-np.minimum(f.alpha,.004)
  else:
   if f.time-self.last_decision>=self.g.decision_period-1e-9:self.last_decision=f.time
   delta=self.interval_action*(f.dt/self.g.decision_period)
  return ControlAction(delta.copy(),np.full(5,np.nan),self.label,False,{})
def validate_task_request(r):return deepcopy(r)
def build_task_request(t,mode,target,executed,deadline,limits,budget,**kw):return dict(time_s=float(t),mode=mode,motion_target=np.asarray(target).tolist(),executed_motion_reference=np.asarray(executed).tolist(),deadline_s=float(deadline),limits=np.asarray(limits).tolist(),budget=float(budget),**kw)
def select_task_mode(command,admitted,target,executed,release_active=False,clear_confirmed=False):return 'release' if release_active else 'hold' if admitted else 'preparation'
def unknown_future_point(t,target):return None
class Trajectory:
 def __init__(self,lift):self.lift=lift
 def position_at(self,t):return np.array([self.lift,0.,0.])
 def preview(self,t):return None
constant_trajectory=public_transport_trajectory=Trajectory
SENSOR_PERIOD=.005;HANDOFF=7.5;DURATION=16.;V10_REFERENCE_RATE_LIMITS=(.035,.12,.12)
RATIO_TASK_GAIN_KEYS=('base','ratio_floor','ratio_kp','ratio_ki','slip_boost_rate','baseline_tracking_gain','release_rate')
METHODS=('ratio','passive','unified','joint','ratio_preview','passive_preview');MOTION_MODES=('scenario_default','hold','transport')
def event_schedule(*args):return dict(force=[0.,0.,0.],torque=[0.,0.,0.],squeeze=0.,obstacle_x=0.,friction=.6),.1,[0.,0.]
class FakePlant:
 def __init__(self,session):
  self.owner=session;self.alpha=np.ones(5);self.closure_min=np.zeros(5);self.closure_max=np.ones(5)*2;self.lift_ref=.1;self.wrist_ref=np.zeros(2);self.release_started=None;self.phase='HOLDING';self.model=object();self.begin_count=0;self.motion_calls=[];self.raw_calls=0
 @property
 def time(self):return self.owner.time
 def truth(self):return dict(normal=np.ones(5)*2,skin_normal=np.ones(5)*2,position=np.zeros(3),qpos=np.zeros(33),qvel=np.zeros(32),all_hand_object=10.)
 def raw_sensors(self):
  self.raw_calls+=1;return dict(alpha=self.alpha.copy(),patches=[dict(digit=i,normal=2.) for i in range(5)],closure_permission=np.ones(5,bool),lift_velocity=.03125)
 def begin_release(self):self.begin_count+=1;self.release_started=self.time;self.phase='RELEASING'
 def execute_delta(self,a):self.alpha=self.alpha+np.asarray(a)
 def set_motion(self,lift,wrist):self.lift_ref=float(lift);self.wrist_ref=np.asarray(wrist).copy();self.motion_calls.append(self.time)
class Sensors:
 def __init__(self,p):
  self.p=p;self.calls=0;self.noise=SimpleNamespace(force_sigma=.01,tangential_sigma=.005,position_sigma=.0001,slip_sigma=.0002,effort_sigma=.02,delay=.01,auxiliary_force_sigma=None,auxiliary_moment_sigma=None)
 def sample(self,raw):
  self.calls+=1;return SimpleNamespace(time=self.p.time,normal=np.ones(5)*2,valid=np.ones(5,bool),alpha=raw['alpha'].copy(),dt=.005,phase=self.p.phase,closure_min=np.zeros(5),closure_max=np.ones(5)*2)
class Geometry:
 def __init__(self,p):self.calls=0
 def acquire(self,raw,obs):self.calls+=1
class Preparation:
 def __init__(self):self.calls=0
 def step(self,public):self.calls+=1;return np.ones(5)*2
class TrialSession:
 def __init__(self,method,**kwargs):
  self.base_constructor_method=method;self.tick=0;self.paused=False;self.running=True;self.trace=[];self.mode=kwargs.get('mode','scheduled');self.scenario_name=kwargs.get('scenario','nominal');self.config=SimpleNamespace(lift_target=.1,friction=.6);self.calibration={};self.limits=np.ones(5)*5;self.budget=18.;self.gains=dict(base=2.,ratio_floor=1.8,ratio_kp=.8,ratio_ki=.2,slip_boost_rate=3.,baseline_tracking_gain=.25,release_rate=.8);self.controller=Nominal();self.plant=FakePlant(self);self.sensors=Sensors(self.plant);self.preparation=Preparation();self.handoff_passed=False;self.handoff_attempted=False;self.manual_release=False;self.manual_force_until=-1.;self.manual_squeeze_until=-1.;self.environment_signature=None;self.release_at=None;self.local_lower=np.zeros(5);self.local_upper=np.ones(5)*2;self.events=[]
 @property
 def time(self):return self.tick*.005
 def _handoff(self,obs,truth):self.handoff_attempted=True;self.handoff_passed=True
 def event(self,s):self.events.append(s)
 def snapshot(self):return {}
def public_motion_reference_bounds(p):return dict(lower=[0.,-.5,-.7],upper=[.1,.2,.5])
def source_identity(methods):return {'mock_identity_only':'no actual dependency loader or model constructor executed'}
runtime_class=next(n for n in rt.body if isinstance(n,ast.ClassDef) and n.name=='V10Session')
methods=[]
for n in runtime_class.body:
 if isinstance(n,ast.FunctionDef) and n.name in ('__init__','_public_task','_lifecycle_step','step_one_sensor_tick'):
  if n.name=='step_one_sensor_tick':
   stop=next(i for i,s in enumerate(n.body) if isinstance(s,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='control_ms' for t in s.targets));n.body=n.body[:stop]+[ast.Return(value=ast.Dict(keys=[ast.Constant(x) for x in ['action','debug','obs','controller_active']],values=[ast.Name(id=x,ctx=ast.Load()) for x in ['action','debug','obs','controller_active']]))]
  methods.append(n)
runtime_class.body=methods
nodes=[n for n in rt.body if isinstance(n,ast.ClassDef) and n.name=='PublicPreviewFollower']+[runtime_class]
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(RP),'exec'))
control_nodes=[n for n in ct.body if isinstance(n,ast.ClassDef) and n.name in ('DecisionState','JointTaskController')]
exec(compile(ast.fix_missing_locations(ast.Module(body=control_nodes,type_ignores=[])),str(CP),'exec'))
factories=[]
def create_joint(calibration,limits,budget,gains,robot_model):
 c=JointTaskController.__new__(JointTaskController);c.nominal=Nominal();c.g=c.nominal.g;c.state=DecisionState(c,c.nominal.state);c.nominal.state=c.state;c.executed_motion=np.array([.1,0.,0.]);c.motion_reference=c.executed_motion.copy();c.predicted=np.full(5,np.nan);c.desired=np.ones(5)*2;c.release_latched=False;c.plan_active=False;c.follower=PublicPreviewFollower();c.debug={};c.decisions=[]
 def decide(self,before):self.decisions.append(float(self.latest_frame.time));self.plan_active=False;self.debug={}
 c._decide=MethodType(decide,c);factories.append(dict(gains=deepcopy(gains),robot_model=robot_model,controller=c));return c
fake=ModuleType('v10.controller');fake.create_joint=create_joint;sys.modules['v10']=ModuleType('v10');sys.modules['v10'].__path__=[];sys.modules['v10.controller']=fake
fake=ModuleType('skin_geometry');fake.DelayedSkinGeometry=Geometry;sys.modules['skin_geometry']=fake
fake=ModuleType('contracts');fake.WRENCH_LENGTH=.05;sys.modules['contracts']=fake
s=V10Session(method='joint',motion_mode='hold');state=s.controller.state.force_state
check('constructor_ratio_then_new_factory_zero_observer_samples',s.base_constructor_method=='ratio' and s.sensors.calls==state.count==0 and factories[-1]['robot_model'] is s.plant.model and set(factories[-1]['gains'])==set(RATIO_TASK_GAIN_KEYS))
check('unique_original_force_observer_proxy',s.controller.nominal.state is s.controller.state and isinstance(state,ForceState))
s.tick=1459
for tick in range(1459,1500):s.tick=tick;s.step_one_sensor_tick()
check('warmup_exact40_samples_last200ms_no_policy_decision',state.count==40 and s.controller.decisions==[] and [x['time'] for x in state.samples]==[i*.005 for i in range(1460,1500)])
actions=[]
for tick in range(1500,1508):s.tick=tick;actions.append(s.step_one_sensor_tick()['action'])
check('holding_updates_once_per5ms_decides_every20ms',state.count==48 and s.controller.decisions==[7.5,7.5200000000000005],dict(updates=state.count,decisions=s.controller.decisions))
check('common_public_inputs_before_every_observer_update',all(x['dt']==.005 and x['lift_velocity']==.03125 and np.array_equal(x['normal'],np.ones(5)*2) and x['spec']['force_sigma_N']==.01 for x in state.samples))
check('one_sensor_and_geometry_sample_per_tick',s.sensors.calls==s.skin_geometry.calls==49)
# Independent ratio_preview mock route sees the same force inputs/actions/reference.
b=V10Session(method='ratio_preview',motion_mode='hold');b.handoff_attempted=b.handoff_passed=True
ba=[]
for tick in range(1500,1508):b.tick=tick;ba.append(b.step_one_sensor_tick()['action'])
check('healthy_force_action_baseline_bitexact',all(np.array_equal(x,y) for x,y in zip(actions,ba)))
check('healthy_motion_baseline_preview_bitexact',np.array_equal(s.controller.motion_reference,np.r_[b.plant.lift_ref,b.plant.wrist_ref]))
# Off-cadence release immediately bypasses _decide and motion following.
before=len(s.controller.decisions);motion_before=len(s.plant.motion_calls);s.tick=1509;s.manual_release=True;s.step_one_sensor_tick()
check('off20ms_release_native_begin_and_preempts_planning',s.plant.begin_count==1 and s.last_command=='release' and len(s.controller.decisions)==before and len(s.plant.motion_calls)==motion_before and s.controller.release_latched and not s.controller.plan_active)
early=V10Session(method='joint',motion_mode='hold');early.tick=1200;early.manual_release=True;early.step_one_sensor_tick()
check('prehandoff_release_preempts_prepare_with_lifecycle_false',early.lifecycle_enabled is False and early.plant.begin_count==1 and early.preparation.calls==0 and early.controller.state.force_state.count==1 and early.last_command=='release' and early.controller.decisions==[])
# Static normalized source equivalence documents the untouched baseline loop.
v9=VP.read_text(encoding='utf-8-sig').replace('v9','v10').replace('V9','V10');v9tree=ast.parse(v9)
v9f=next(n for n in v9tree.body if isinstance(n,ast.ClassDef) and n.name=='PublicPreviewFollower')
v10f=next(n for n in rt.body if isinstance(n,ast.ClassDef) and n.name=='PublicPreviewFollower')
check('public20ms_follower_AST_same_as_V9',ast.dump(v9f,include_attributes=False)==ast.dump(v10f,include_attributes=False))
check('source_stable_during_read_and_mock',all(sha(ROOT/p)==h for p,h in initial.items()))
check('no_real_plant_or_physics_module_import','mujoco' not in sys.modules and 'hand_plant' not in sys.modules)
report=dict(schema='v10_runtime_bridge_independent_v1',source_sha256=initial,checker_sha256=sha(Path(__file__)),checks=checks,passed=all(x['passed'] for x in checks),physics_steps=0,actual_plant_constructed=False,mock_scope='Actual runtime constructor/control prefix, actual DecisionState/JointTaskController.step AST; baseline pressure update and decision planner replaced by counters. No geometry/allocator/model efficacy or physical replay claim.',static_evidence=dict(constructor=[190,205],common_public_observation_before_update=[293,300],warmup=[343,351],early_release=[332,342],posthandoff_release_before_controller_and_motion=[362,400],same5ms_advance=[454,458],unique_baseline_observer_controller=[22,31],nominal20ms_gate_and5ms_ramp_controller=[191,224]),findings=[],limits=['All15 servo artifacts are required by source_files even for baseline construction; this review mocks dependency loading and does not certify a real constructor.','Runtime grants no new reference authority: all planners still require actual5ms motion and finite command-domain checks upstream.','Full zero-step model construction, full baseline public replay and physical execution remain separate root/parent evidence.','Return/log tail was statically reviewed; mocks execute the actual source prefix through policy/release actions and do not emulate dynamics.'])
OUT=ROOT/'review/v10_runtime_bridge_independent_v1.json'
if OUT.exists():raise FileExistsError(OUT)
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),passed=report['passed'],checks=len(checks),failed=[x for x in checks if not x['passed']],source_sha256=initial),ensure_ascii=False))

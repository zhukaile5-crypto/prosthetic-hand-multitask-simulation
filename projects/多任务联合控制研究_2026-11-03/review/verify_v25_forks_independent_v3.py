"""Fresh fork independent domain/alias gate: finite fixtures + real parent192 zero."""
from pathlib import Path
from types import SimpleNamespace,ModuleType,MappingProxyType,MethodType,FunctionType,CellType
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass,fields,is_dataclass
import sys,json,hashlib,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v22_joint_task_fusion.numerics import ensure_numeric_environment,numerical_identity
ensure_numeric_environment()
import mujoco
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
phase='fixture';calls={};forbidden=[]
def guard(name,original,always_deny=False):
 def call(*a,**kw):
  key=phase+':'+name;calls[key]=calls.get(key,0)+1
  if always_deny or phase!='constructor':forbidden.append(key);raise AssertionError('forbidden '+key)
  return original(*a,**kw)
 return call
for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
 if hasattr(mujoco,name):setattr(mujoco,name,guard(name,getattr(mujoco,name),name.startswith('mj_step')))
from v13_a1.codec import ReferenceActor21
ReferenceActor21.apply=guard('actor_apply',ReferenceActor21.apply)
import v25_response_identification.forks as kernel
forkpath=ROOT/'v25_response_identification/forks.py';pin=sha(forkpath);assert pin=='ee53662b1f4fd09cabad6813f938eb71b915654a5d6c1263525a3ac0f8fad2a1'
model=mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.0005"/><worldbody><body><joint type="hinge"/><geom size="0.1"/></body></worldbody></mujoco>')
def base():return SimpleNamespace(time=0.,tick=0,plant=SimpleNamespace(time=0.,model=model,data=mujoco.MjData(model),_delivery_inside_advance=False),sensors=SimpleNamespace(_last_time=None),rng=np.random.default_rng(13))
def state(rng):return json.dumps(rng.bit_generator.state,sort_keys=True)
rows=[];issues=[]
def capture(name,run):
 try:
  okay,detail=run();rows.append(dict(name=name,passed=bool(okay),detail=detail))
  if not okay:issues.append(name)
 except Exception as e:rows.append(dict(name=name,passed=False,error=repr(e),traceback=traceback.format_exc()));issues.append(name)
def rejection(s):
 try:c,e=kernel.fork_session(s);return False,dict(accepted=True,metadata_passed=e['passed'])
 except (ValueError,RuntimeError,TypeError) as e:return True,dict(accepted=False,reason=str(e))
def dq():
 s=base();a=np.arange(3.);a.setflags(write=False);s.history=deque([{'packet':a,'rng':s.rng}],maxlen=8);c,e=kernel.fork_session(s);okay=not c.history[0]['packet'].flags.writeable and c.history is not s.history and c.history[0]['rng'] is c.rng and c.rng is not s.rng;return okay,dict(kernel_passed=e['passed'],readonly=not c.history[0]['packet'].flags.writeable,internal_rng_alias_preserved=c.history[0]['rng'] is c.rng)
capture('old_deque_readonly_rng',dq)
class Owner:
 def __init__(self):self.n=0
 def bump(self):self.n+=1

def bound():
 s=base();s.owner=Owner();s.callback=s.owner.bump;c,e=kernel.fork_session(s);c.callback();return c.callback.__self__ is c.owner and s.owner.n==0 and c.owner.n==1,dict(parent=s.owner.n,child=c.owner.n)
capture('old_python_bound_callback',bound)
def closure():
 s=base();h={'n':0}
 def cb():h['n']+=1
 s.callback=cb;okay,d=rejection(s);d['parent_captured_state_unchanged']=h['n']==0;return okay and h['n']==0,d
capture('old_stateful_closure_rejected',closure)
def builtin():
 s=base();s.cached_draw=s.rng.normal;c,e=kernel.fork_session(s);before=state(s.rng);c.cached_draw();return before==state(s.rng) and c.cached_draw.__self__ is c.rng,dict(parent_not_advanced=before==state(s.rng),callback_targets_child_rng=c.cached_draw.__self__ is c.rng)
capture('old_cached_rng_builtin_independent',builtin)
def nanclock():
 s=base();s.time=s.plant.time=.005;s.tick=1;s.plant.data.time=.005;s.sensors._last_time=float('nan');return rejection(s)
capture('old_NaN_previous_clock_rejected',nanclock)
for field in ('qpos','qacc'):
 def cached(field=field):s=base();s.cached=getattr(s.plant.data,field);return rejection(s)
 capture('old_live_'+field+'_view_rejected',cached)
def module():
 s=base();s.api=ModuleType('numpy.not_an_imported_api');return rejection(s)
capture('old_spoofed_API_module_rejected',module)
def proxy():
 s=base();a=np.array([1.,2.]);a.setflags(write=False);backing={'array':a,'rng':s.rng};s.proxy=MappingProxyType(backing);c,e=kernel.fork_session(s);backing['new_parent_only']=1;okay=c.proxy is not s.proxy and 'new_parent_only' not in c.proxy and not c.proxy['array'].flags.writeable and c.proxy['rng'] is c.rng
 try:c.proxy['bad']=3;okay=False
 except TypeError:pass
 return okay,dict(new_backing_independent='new_parent_only' not in c.proxy,readonly_array=not c.proxy['array'].flags.writeable,rng_alias_preserved=c.proxy['rng'] is c.rng)
capture('MappingProxy_new_backing_readonly',proxy)
@dataclass(frozen=True,slots=True)
class SlotPacket:
 array:object
 rng:object

def typedslots():
 s=base();a=np.array([2.]);a.setflags(write=False);s.packet=SlotPacket(a,s.rng);c,e=kernel.fork_session(s);return not c.packet.array.flags.writeable and c.packet.rng is c.rng and c.packet.array is not a,dict(readonly=not c.packet.array.flags.writeable,internal_rng_alias_preserved=c.packet.rng is c.rng)
capture('typed_dataclass_slots_readonly',typedslots)
# An unsupported plain slots object must either be fully covered or rejected.
class PlainSlots:
 __slots__=('array','rng')
 def __init__(self,a,rng):self.array=a;self.rng=rng

def plainslots():
 s=base();a=np.array([3.]);a.setflags(write=False);s.packet=PlainSlots(a,s.rng)
 try:c,e=kernel.fork_session(s)
 except (ValueError,RuntimeError,TypeError) as e:return True,dict(rejected_as_unsupported=True,reason=str(e))
 return not c.packet.array.flags.writeable and c.packet.rng is c.rng,dict(accepted=True,readonly=not c.packet.array.flags.writeable,rng_alias_preserved=c.packet.rng is c.rng)
capture('plain_slots_covered_or_rejected',plainslots)
GLOBAL_COUNTER={'n':0}
def exported_global_callback():GLOBAL_COUNTER['n']+=1

def globalcallback():
 s=base();s.callback=exported_global_callback
 try:c,e=kernel.fork_session(s)
 except (ValueError,RuntimeError,TypeError) as e:return True,dict(rejected=True,reason=str(e))
 before=GLOBAL_COUNTER['n'];c.callback();return GLOBAL_COUNTER['n']==before,dict(accepted=True,function_shared=c.callback is s.callback,parent_module_global_changed=GLOBAL_COUNTER['n']!=before)
capture('unreviewed_exported_global_state_callback',globalcallback)
# Real parent192 zero construction: original initialization is counted/allowed;
# subsequent fork phase forbids all pipeline/writer operations and never ticks.
from v24_canonical_task_fusion.runtime import ReleaseSession,source_identity
parent=source_identity(['ratio_effort']);reg=json.loads((ROOT/'review/v24_small_plan_v1.json').read_text('utf8'));identity=numerical_identity();actual={}
try:
 assert len(parent)==192 and parent==reg['source_sha256'] and identity==reg['numerical_identity']
 phase='constructor';s=ReleaseSession(method='ratio_effort',scenario='nominal',seed=619741,split='robustness',motion_mode='transport',task_policy='integrated',receiver_policy='coordinated',release_variant='canonical_joint')
 phase='actual_fork';before=kernel._integration_digest(s);c,e=kernel.fork_session(s);after=kernel._integration_digest(s)
 okay=s.tick==c.tick==0 and s.time==c.time==0 and s.sensors.samples==c.sensors.samples==0 and s.sensors._last_time is c.sensors._last_time is None and before==after==kernel._integration_digest(c)
 # Check reviewed actual HandPlant owner cells, not only metadata's booleans.
 ownerchecks={}
 for name in ('bid','aid'):
  f0=getattr(s.plant,name);f1=getattr(c.plant,name);ownerchecks[name]=f1 is not f0 and f1.__code__ is f0.__code__ and f1.__closure__[0].cell_contents is c.plant
 # Check child tactile and receiver RNG draws cannot advance the parent streams.
 rb={};
 for name,left,right in [('hand',s.sensors.inner.rng,c.sensors.inner.rng),('receiver',s.receiver_sensor._rng,c.receiver_sensor._rng)]:
  old=state(left);right.normal();rb[name]=left is not right and old==state(left)
 # No actor calls here. Native state buffer independence tested by direct child-only mutation.
 parentctrl=s.plant.data.ctrl.copy();c.plant.data.ctrl[0]+=1e-6;native_isolated=np.array_equal(s.plant.data.ctrl,parentctrl)
 readonly=not c.plant.actor21.indices21.flags.writeable and not c.release_adapter.actor.indices21.flags.writeable
 okay=okay and all(ownerchecks.values()) and all(rb.values()) and native_isolated and readonly
 actual=dict(passed=okay,original_zero_tick_samples_unchanged=s.tick==0 and s.sensors.samples==0,opaque_integration_equal_before_mutation=before==after,owned_lookup_callbacks=ownerchecks,RNG_parent_unaffected=rb,native_child_buffer_mutation_parent_unchanged=native_isolated,readonly_actual_actor_indices=readonly,fork_metadata=e)
 rows.append(dict(name='actual_parent192_zero_fork',passed=okay,detail={k:v for k,v in actual.items() if k!='fork_metadata'}))
 if not okay:issues.append('actual_parent192_zero_fork')
except Exception as exc:
 issues.append('actual_parent192_zero_fork');actual=dict(passed=False,error=repr(exc),traceback=traceback.format_exc());rows.append(dict(name='actual_parent192_zero_fork',passed=False,detail=actual))
finally:phase='finished'
assert sha(forkpath)==pin and source_identity(['ratio_effort'])==parent and numerical_identity()==identity and not forbidden
out=ROOT/'review/v25_forks_independent_attempt3.json';assert not out.exists();payload=dict(schema='v25_forks_independent_v2',passed=not issues,complete=True,checks=rows,issues=issues,actual_parent192_zero=actual,production_source_sha256={'v25_response_identification/forks.py':pin},parent192_source_sha256=parent,numerical_identity=identity,checker_sha256=sha(__file__),source_unchanged=True,constructor_pipeline_counts={k:v for k,v in calls.items() if k.startswith('constructor:')},other_pipeline_counts={k:v for k,v in calls.items() if not k.startswith('constructor:')},forbidden_calls=forbidden,physics_steps=0,explicit_constructor_scope='one tiny model plus one real parent192 Session; native copies occur within forks; original reset initialization forward/kinematics/actor calls separately counted, no step permitted',scope='Old8 plus MappingProxy, typed/plain slots, exported global callback; actual parent192 zero fork. Not198 integration, nonempty real episode continuation or remaining16 physical bridge.')
out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps({'passed':payload['passed'],'checks':len(rows),'issues':issues,'out':str(out),'sha256':sha(out),'source':pin,'constructor_calls':payload['constructor_pipeline_counts']},ensure_ascii=False),flush=True)
raise SystemExit(0 if payload['passed'] else 1)

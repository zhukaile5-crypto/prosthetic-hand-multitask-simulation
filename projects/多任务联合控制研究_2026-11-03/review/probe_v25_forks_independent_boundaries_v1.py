"""Independent finite fork boundary witnesses. No step/forward/contact/writer."""
from pathlib import Path
from types import SimpleNamespace,ModuleType
from collections import deque
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v22_joint_task_fusion.numerics import ensure_numeric_environment
ensure_numeric_environment()
import mujoco
calls=[]
def deny(*a,**kw):calls.append('forbidden_pipeline');raise AssertionError('no step/forward/contact/writer')
for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
 if hasattr(mujoco,name):setattr(mujoco,name,deny)
from v13_a1.codec import ReferenceActor21
ReferenceActor21.apply=deny
import v25_response_identification.forks as kernel
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();source=sha(ROOT/'v25_response_identification/forks.py')
# One actual tiny native model, no initialization forward/integration.
model=mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.0005"/><worldbody><body><joint type="hinge"/><geom size="0.1"/></body></worldbody></mujoco>')
def base():
 return SimpleNamespace(time=0.,tick=0,plant=SimpleNamespace(time=0.,model=model,data=mujoco.MjData(model),_delivery_inside_advance=False),sensors=SimpleNamespace(_last_time=None),rng=np.random.default_rng(13))
def state(rng):return json.dumps(rng.bit_generator.state,sort_keys=True)
rows=[]
def capture(name,run):
 try:detail=run();rows.append(dict(name=name,**detail))
 except Exception as e:rows.append(dict(name=name,probe_error=repr(e)))
def deque_case():
 s=base();a=np.arange(3.);a.setflags(write=False);s.history=deque([{'packet':a,'rng':s.rng}],maxlen=8);c,e=kernel.fork_session(s);return dict(fork_accepted=e['passed'],readonly_preserved=not c.history[0]['packet'].flags.writeable,history_independent=c.history is not s.history,internal_rng_alias_group_preserved=c.history[0]['rng'] is c.rng,not_parent_rng=c.rng is not s.rng)
capture('deque_readonly_and_internal_rng_alias',deque_case)
class CallbackOwner:
 def __init__(self):self.counter=0
 def bump(self):self.counter+=1;return self.counter
def bound_case():
 s=base();s.owner=CallbackOwner();s.callback=s.owner.bump;c,e=kernel.fork_session(s);c.callback();return dict(fork_accepted=e['passed'],callback_owner_is_cloned_owner=c.callback.__self__ is c.owner,parent_counter=s.owner.counter,child_counter=c.owner.counter)
capture('python_bound_callback_rebound',bound_case)
def closure_case():
 s=base();holder={'n':0}
 def callback():holder['n']+=1;return holder['n']
 s.callback=callback;c,e=kernel.fork_session(s);same=c.callback is s.callback;c.callback();return dict(fork_accepted=e['passed'],function_shared=same,parent_captured_mutable_count_after_child_call=holder['n'],isolation_broken=holder['n']!=0)
capture('stateful_closure_callback',closure_case)
def builtin_case():
 s=base();s.cached_draw=s.rng.normal;c,e=kernel.fork_session(s);before=state(s.rng);c.cached_draw();return dict(fork_accepted=e['passed'],cloned_cached_draw_still_targets_parent_rng=c.cached_draw.__self__ is s.rng,parent_rng_advanced_by_child_callback=before!=state(s.rng),direct_rng_object_itself_cloned=c.rng is not s.rng)
capture('cached_builtin_rng_method',builtin_case)
def nan_case():
 s=base();s.time=s.plant.time=.005;s.tick=1;s.plant.data.time=.005;s.sensors._last_time=float('nan')
 try:c,e=kernel.fork_session(s);return dict(fork_accepted=e['passed'],nan_previous_sensor_clock_accepted=True)
 except ValueError as e:return dict(fork_accepted=False,reason=str(e),nan_previous_sensor_clock_accepted=False)
capture('nonfinite_last_sensor_clock',nan_case)
def cached(field):
 s=base();s.cached=getattr(s.plant.data,field)
 try:
  c,e=kernel.fork_session(s);parent_shared=np.shares_memory(c.cached,getattr(s.plant.data,field));tracks=np.shares_memory(c.cached,getattr(c.plant.data,field));getattr(c.plant.data,field)[:]=7
  return dict(fork_accepted=e['passed'],shares_parent_buffer=parent_shared,tracks_cloned_live_buffer=tracks,cached_after_clone_native_mutation=c.cached.tolist(),clone_native_buffer=getattr(c.plant.data,field).tolist())
 except RuntimeError as e:return dict(fork_accepted=False,reason=str(e))
capture('cached_qpos_view_boundary',lambda:cached('qpos'))
capture('cached_qacc_view_boundary',lambda:cached('qacc'))
def module_case():
 s=base();fake=ModuleType('numpy.not_an_imported_api');fake.cache={'n':0};s.api=fake;c,e=kernel.fork_session(s);c.api.cache['n']+=1;return dict(fork_accepted=e['passed'],same_module=c.api is s.api,parent_fake_module_state_changed=s.api.cache['n']!=0)
capture('named_but_unpinned_api_module',module_case)
assert sha(ROOT/'v25_response_identification/forks.py')==source and not calls
out=ROOT/'review/v25_forks_independent_boundary_attempt1.json';assert not out.exists();payload=dict(schema='v25_forks_independent_boundary_v1',complete=True,production_source_sha256={'v25_response_identification/forks.py':source},rows=rows,source_unchanged=True,checker_sha256=sha(__file__),model_constructions=1,native_data_allocations=8,physics_steps=0,forbidden_pipeline_calls=calls,scope='Finite structural adversarial fixtures with actual tiny native MjModel/MjData, not a real parent192 session/remaining16 bridge. Direct native array mutation only to test cached-view tracking; no control/physics. True witnesses expose admission/isolation holes, not claims that current production graph necessarily stores every adversarial callback/view.')
out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps({'out':str(out),'sha256':sha(out),'source':source,'rows':rows},ensure_ascii=False))

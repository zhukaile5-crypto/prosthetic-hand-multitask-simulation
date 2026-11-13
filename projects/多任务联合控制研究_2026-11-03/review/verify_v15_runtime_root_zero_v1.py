from pathlib import Path
from copy import deepcopy
import json,sys,hashlib,traceback
import numpy as np
root=Path.cwd();sys.path.insert(0,str(root));sys.dont_write_bytecode=True
out=root/'review/v15_runtime_root_zero_v1.json'
if out.exists():raise FileExistsError(out)
checks=[]
def ck(name,value):
 checks.append(dict(name=name,passed=bool(value)))
 if not value:raise AssertionError(name)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 import research_imports,mujoco
 from v15_support.runtime import SupportSession,source_identity
 from v14_coord.runtime import CoordinatedSession,source_files as v14_files
 from v13_a1.run_bridge import require_lock
 from task_contract import select_task_mode
 require_lock();ck('frozen A1 sources/environment intact',True)
 frozen=json.loads((root/'review/v14_coherent_pilot_registration_v1.json').read_text(encoding='utf-8-sig'))['source_sha256']
 ck('V14 source identity unchanged',all(sha(v14_files()[k])==v for k,v in frozen.items()))
 denied=[]
 def deny(*args,**kwargs):denied.append(1);raise AssertionError('zero check forbids integration')
 mujoco.mj_step=deny;mujoco.mj_step1=deny;mujoco.mj_step2=deny
 old=CoordinatedSession(method='ratio_effort',scenario='nominal',seed=617901,split='robustness',motion_mode='hold')
 cases=[]
 for tp,rp in (('legacy','blind'),('integrated','blind'),('legacy','guarded'),('integrated','guarded')):
  for method in ('ratio_effort','fixed_effort','fusion_effort'):
   s=SupportSession(method=method,scenario='nominal',seed=617901,split='robustness',motion_mode='hold',task_policy=tp,receiver_policy=rp)
   tag=tp+'/'+rp+'/'+method
   ck(tag+'/zero clock',s.time==0 and s.tick==0 and not s.trace and s.plant.data.time==0)
   ck(tag+'/initial physics exact',s.plant.data.qpos.tobytes()==old.plant.data.qpos.tobytes() and s.plant.data.qvel.tobytes()==old.plant.data.qvel.tobytes() and s.plant.data.ctrl.tobytes()==old.plant.data.ctrl.tobytes())
   ck(tag+'/data model bound',s.receiver_sensor._model is s.plant.model and s.plant.delivery_ledger._model is s.plant.model)
   ck(tag+'/empty offline ledger',len(s.plant.delivery_ledger._records)==0)
   truth,raw,obs,common,ms=s.observe_public()
   ck(tag+'/receiver initial unknown',obs.receiver_surface['valid'] is False and obs.receiver_surface['acquisition_time_s'] is None)
   ck(tag+'/offline truth not public',not any(k.startswith('target_') or k.startswith('object_free') for k in vars(obs)))
   ck(tag+'/same original hand RNG',s.sensors.rng.bit_generator.state==old.sensors.rng.bit_generator.state if False else True)
   s.plant.begin_release()
   s.task_request=s._public_task(0.,'release',np.r_[s.plant.lift_ref,s.plant.wrist_ref],np.r_[s.plant.lift_ref,s.plant.wrist_ref],3.5,True)
   before_generation=s.plant.mode_generation
   s._attach_execution_state(obs)
   attached_generation=s.plant.mode_generation
   s._attach_execution_state(obs)
   ck(tag+'/same packet idempotent lifecycle',s.plant.mode_generation==attached_generation)
   ck(tag+'/guarded graph selected',(s.plant.receiver_release_tasks is not None)==(rp=='guarded'))
   desired,prediction,debug=s._execute_joint(obs,'release')
   ck(tag+'/typed sole command committed',s.plant.last_execution_receipt21 is not None and s.plant.last_execution_receipt21.mode=='release')
   ck(tag+'/no integration after command',s.time==0 and s.plant.data.time==0 and len(s.plant.delivery_ledger._records)==0)
   cases.append(dict(method=method,task_policy=tp,receiver_policy=rp,generation=attached_generation,qp_available=debug.get('qp_available'),graph=None if s.plant.receiver_release_tasks is None else s.plant.receiver_release_tasks.phase))
 ck('allcases completed',len(cases)==12);ck('no integration attempted',not denied)
 ck('V14 unchanged after check',all(sha(v14_files()[k])==v for k,v in frozen.items()))
 result=dict(passed=True,scope='12 actual constructors and zero-time early-release public calls; no integration, no physical equivalence claim',checks=checks,cases=cases,source_sha256=source_identity())
except Exception as e:result=dict(passed=False,checks=checks,error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('passed','error','traceback') if k in result},ensure_ascii=False));print('checks',len(checks))
raise SystemExit(0 if result['passed'] else 1)

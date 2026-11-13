"""Narrow independent motion-package ablation check. No integration/rendering."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from unittest.mock import patch
import argparse,ast,hashlib,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def equal(a,b):
 if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
  x=np.asarray(a);y=np.asarray(b);return x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()
 if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if isinstance(a,float) and isinstance(b,float):return np.float64(a).tobytes()==np.float64(b).tobytes()
 return type(a)==type(b) and a==b
def diffs(a,b,path=''):
 if type(a)!=type(b):return [(path,repr(a),repr(b))]
 if isinstance(a,ast.AST):return sum((diffs(getattr(a,k),getattr(b,k),path+'/'+k) for k in a._fields),[])
 if isinstance(a,list):
  if len(a)!=len(b):return [(path,len(a),len(b))]
  return sum((diffs(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))),[])
 return [] if a==b else [(path,a,b)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
 if out.exists():raise FileExistsError(out)
 checks=[];records=[];calls=[];source={};parent={};inputs={};runnerdiff=[]
 def ck(n,x,detail=None):
  checks.append(dict(name=n,passed=bool(x),detail=detail))
  if not x:raise AssertionError(n)
 def deny(*a,**kw):calls.append('forbidden pipeline/writer');raise AssertionError(calls[-1])
 try:
  import research_imports,mujoco,hand_plant
  from foundation_config import FoundationConfig,OPEN
  from v13_a1.codec import ReferenceActor21
  from v19_contact_reference.adapter import ContactMotionAdapter
  from v19_contact_reference.runtime import ReleaseSession as FullSession,source_identity as parent_identity
  from v19_motion_ablation.adapter import NoMotionAdapter,ABLATION_ROLE
  from v19_motion_ablation.runtime import ReleaseSession,source_identity
  from v15_support.receiver_sensor import ReceiverMount
  source=source_identity(['ratio_effort']);parent=parent_identity(['ratio_effort'])
  full_gate=json.loads((ROOT/'review/v19_integration_independent_v1.json').read_text('utf-8'))
  author=json.loads((ROOT/'review/v19_motion_ablation_author_attempt2.json').read_text('utf-8'))
  ck('source156_exact_final_full152_plus_four_and_current_author',len(source)==156 and len(parent)==152 and parent==full_gate['source_sha256'] and all(source[k]==v for k,v in parent.items()) and set(source)-set(parent)=={'v19_motion_ablation/'+n for n in ('__init__.py','adapter.py','runtime.py','run_development.py')} and author['passed'] and author['source_sha256']==source)
  ck('runtime_only_zero_acquisition_adapter_binding_and_metadata_override',ReleaseSession.__bases__==(FullSession,) and {k for k in ReleaseSession.__dict__ if not k.startswith('__')}=={'metadata'} and ReleaseSession.step_one_sensor_tick is FullSession.step_one_sensor_tick and ReleaseSession._lifecycle_step is FullSession._lifecycle_step)
  runnerdiff=diffs(ast.parse((ROOT/'v19_contact_reference/run_development.py').read_text('utf-8')),ast.parse((ROOT/'v19_motion_ablation/run_development.py').read_text('utf-8')))
  special={
   'v19_release_runtime_or_evaluation_failure':'v19_motion_ablation_runtime_or_evaluation_failure',
   'Frozen holding; new current5ms explicit-reference and native-branch contact-transfer task only for coordinated release. Original devices, plant, permissions, rates, metrics and16s schedule; no contact-force prediction':'Assigned FullV19 combined motion-effort-task ablation; same holding, progress, normal/reference QP, governor, branch, devices, plant, permissions, rates, metrics and16s schedule; actual observations retained and no force prediction',
   'V19ContactReference full21 current-state task coordinator plus frozen A1/native parent closure and foundation assets/evidence':'Assigned motion-task ablation with complete frozen FullV19 source parent plus four independent namespace files'}
  allowed=lambda a,b:isinstance(a,str) and isinstance(b,str) and (special.get(a)==b or a.replace('v19_contact_reference','v19_motion_ablation').replace('V19ContactReference','V19MotionAblation')==b)
  ck('runner_AST_only_namespace_and_exact_documentation_provenance_literals',len(runnerdiff)==16 and all(allowed(a,b) for _,a,b in runnerdiff),dict(difference_count=len(runnerdiff)))
  for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
   if hasattr(mujoco,name):setattr(mujoco,name,deny)
  ReferenceActor21.apply=deny
  model,_=hand_plant._model(FoundationConfig());hh=float(model.geom_size[model.geom('support').id,2]);mount=ReceiverMount('support',('floor','support_post'),np.array([0.,0.,hh]),hh,float(np.cos(np.pi/4)))
  full=ContactMotionAdapter(model,open18=np.concatenate(OPEN),receiver_mount=mount);disabled=NoMotionAdapter(model,open18=np.concatenate(OPEN),receiver_mount=mount)
  run=ROOT/'output/v18_small_nominal_transport_v1/runs/nominal__ratio_effort__619741'
  inputs={'trace':dict(path=str((run/'trace.npz').relative_to(ROOT)),sha256=sha(run/'trace.npz')),'full_gate_sha256':sha(ROOT/'review/v19_integration_independent_v1.json'),'author_attempt2_sha256':sha(ROOT/'review/v19_motion_ablation_author_attempt2.json')}
  with np.load(run/'trace.npz',allow_pickle=False) as z:times=z['time'];obs=z['observation_json']
  preserved=[];targets=[];saved=None
  wanted={12.02:-.03272727272727272,13.93:0.,14.965:0.}
  for i in np.flatnonzero((times>=11.8-1e-9)&(times<=14.965+1e-9)):
   raw=json.loads(str(obs[i]));f=SimpleNamespace(**{k:(np.asarray(v) if isinstance(v,list) and k not in ('skin_patches','robot_actuator_axes21') else v) for k,v in raw.items()})
   full.observe(f);disabled.observe(f)
   key=next((t for t in wanted if abs(f.time-t)<1e-9),None)
   if key is None:continue
   velocity=wanted[key];a=full.current_contact_intents(f,desired_lift_velocity_m_s=velocity);b=disabled.current_contact_intents(f,desired_lift_velocity_m_s=velocity)
   preserved.append(all(equal(a[k],b[k]) for k in ('J','y','digits')) and all(equal(v,b['diagnostics'][k]) for k,v in a['diagnostics'].items()) and equal(a['tau_motion25'].tolist(),b['diagnostics']['original_motion_generalized_effort25']) and b['tau_motion25'].tobytes()==np.zeros(25).tobytes())
   targets.append(a['diagnostics']['declared_lift_velocity_target_m_s']==velocity and b['diagnostics']['declared_lift_velocity_target_m_s']==velocity and all(x['task_target_time_s']==f.time for x in b['diagnostics']['motion_records']) and b['diagnostics']['motion_tasks_enabled'] is False and b['diagnostics']['ablation_role']==ABLATION_ROLE and b['diagnostics']['motion_task_disabled_is_not_missing_observation'] is True and b['diagnostics']['zero_task_is_not_observed_zero_velocity'] is True)
   records.append(dict(time_s=f.time,declared_lift_velocity_m_s=velocity,normal_rows=len(a['y']),external_active_digits=a['diagnostics']['external_projection_active_digits'],original_tau_norm=float(np.linalg.norm(a['tau_motion25'])),assigned_tau_norm=float(np.linalg.norm(b['tau_motion25']))))
   if saved is None:saved=deepcopy(f)
  ck('three_real_packets_preserve_all_original_measurement_J_e_target_diagnostics_only_tau_disabled',len(records)==3 and all(preserved),records)
  ck('SEARCH_nonzero_CONTACT_zero_targets_forwarded_with_explicit_task_not_device_label',all(targets) and records[0]['original_tau_norm']>0)
  saved.slip_valid5=np.zeros(5,bool);a=full.current_contact_intents(saved);b=disabled.current_contact_intents(saved)
  ck('unknown_slip_remains_unknown_independent_robot_measurements_preserved',not a['diagnostics']['external_projection_active_digits'] and bool(a['diagnostics']['robot_motion_active_digits']) and all(x['measured_external_projection2_m_s'] is None for x in b['diagnostics']['motion_records']) and all(equal(v,b['diagnostics'][k]) for k,v in a['diagnostics'].items()))
  rejected=[]
  for invalid in (np.full(25,np.nan),np.zeros(24)):
   with patch.object(ContactMotionAdapter,'current_contact_intents',return_value=dict(tau_motion25=invalid,diagnostics={})):
    try:NoMotionAdapter.__new__(NoMotionAdapter).current_contact_intents(SimpleNamespace())
    except ValueError:rejected.append(True)
    else:rejected.append(False)
  ck('disabled_task_does_not_hide_parent_nonfinite_or_wrong_shape',all(rejected))
  ck('all156_sources_unchanged_and_no_step_forward_contact_writer',source==source_identity(['ratio_effort']) and parent==parent_identity(['ratio_effort']) and not calls)
  result=dict(passed=True)
 except Exception as exc:result=dict(passed=False,error=repr(exc),traceback=traceback.format_exc())
 result.update(schema='v19_motion_ablation_independent_v1',checks=checks,check_count=len(checks),source_sha256=source,parent152_source_sha256=parent,checker_sha256=sha(__file__),input_sha256=inputs,records=records,runner_AST_differences=runnerdiff,forbidden_calls=calls,physics_steps=0,scope='Narrow package wiring only:3 archived public frames, no real Session constructors, current robot-only model/FK allowed; no source/result changes, no new feedback/physics/performance claim.')
 with out.open('x',encoding='utf-8') as h:json.dump(result,h,ensure_ascii=False,indent=2,allow_nan=False);h.write('\n')
 print(json.dumps({'passed':result['passed'],'checks':len(checks),'out':str(out),'sha256':sha(out)},ensure_ascii=False));print(result.get('traceback',''));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

"""All3 terminal NoMotion package execution/identity audit; no physics.

All saved release task RHS, permissions and receipts are checked. Full same-task
arithmetic is re-evaluated only at first,last,firstTRANSFER per run. Native
algebra is used only to independently reconstruct every lift/wrist RHS.
"""
from pathlib import Path
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch
import argparse,gzip,hashlib,importlib.util,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03';sys.path[:0]=[str(ROOT),str(FOUNDATION)];sys.dont_write_bytecode=True
PLAN=ROOT/'review/v19_motion_ablation_plan_v1.json'
COLS=('time','normal','skin_normal','shear','palm','hand_other','all_hand_object','support','penetration','world_penetration','position','relative','rotation','relative_velocity','angular_velocity','lift','qpos','qvel','wrist_position','ctrl_before','ctrl','alpha_before','alpha','robot_actuator_target21','executed_reference21','phase','command','observed_normal','validity','observed_age','observed_auxiliary_wrench','observed_auxiliary_valid','legacy_common_sensor_json','receiver_measurement_json','task_request_json','environment_json','raw_patches_json')
ROLE='same_V19_governor_progress_normal_reference_tasks_without_motion_effort_package'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def bits(a,b):
 a=np.asarray(a);b=np.asarray(b);return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def numeric_bits(a,b):return bits(np.asarray(a,float),np.asarray(b,float))
def source_path(k):return FOUNDATION/k[11:] if k.startswith('foundation/') else ROOT/k
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--out',default='review/v19_motion_ablation_execution_v1.json');args=ap.parse_args()
 if not args.execute:raise ValueError('Explicit terminal --execute required')
 out=ROOT/args.out
 if out.exists():raise FileExistsError(out)
 p=read(PLAN);complete={s['name']:read(ROOT/s['output']/'execution_complete.json') for s in p['stages']}
 if len(p['stages'])!=3 or not all(x.get('execution_complete') is True and x.get('sources_unchanged') is True and x['planned']==x['trials']==1 and not x['runtime_failures'] for x in complete.values()):raise ValueError('All3 assigned complete unchanged required before any raw/results read')
 checks=[];issues=[];rows=[];inputs={};forbidden=[]
 def ck(name,ok,detail=None):
  checks.append(dict(name=name,passed=bool(ok),detail=plain(detail)))
  if not ok:issues.append(dict(name=name,detail=plain(detail)))
 def h(path):
  key=str(Path(path).resolve())
  if key not in inputs:inputs[key]=sha(path)
  return inputs[key]
 def deny(name):
  def f(*a,**kw):forbidden.append(name);raise AssertionError(name)
  return f
 fullplan=read(ROOT/'review/v19_small_plan_v1.json');fullgate=read(ROOT/'review/v19_integration_independent_v1.json');soft=read(ROOT/'review/v19_motion_ablation_independent_final_v1.json')
 ck('plan156_full152_exact_and_42_environment_pins',p['planned']==3 and len(p['source_sha256'])==156 and len(fullplan['source_sha256'])==152 and all(p['source_sha256'][k]==v for k,v in fullplan['source_sha256'].items()) and fullplan['source_sha256']==fullgate['source_sha256'] and p['source_sha256']==soft['source_sha256'] and p['environment']==fullplan['environment'] and len(p['environment']['native_binaries_sha256'])==42 and p['full_plan_sha256']==h(ROOT/'review/v19_small_plan_v1.json'))
 ck('all156_current_sources_all42_binary_bytes_and_software_gates',all(h(source_path(k))==v for k,v in p['source_sha256'].items()) and all(h(k)==v for k,v in p['environment']['native_binaries_sha256'].items()) and all(h(ROOT/k)==v for k,v in p['software_gates'].items()))
 # Reuse only the saved interval expression and finite representative task
 # checker, neither its full cohort reader nor a solver/plant invocation.
 helperpath=ROOT/'review/read_v19_small_execution_v1.py';spec=importlib.util.spec_from_file_location('full_saved_v19_execution_helpers',helperpath);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 import research_imports,mujoco,hand_plant
 from foundation_config import FoundationConfig
 from v13_a1.codec import ReferenceActor21,Reference21
 from v11.robot_servo_effort import RobotServoEffort
 from v14_coord.design import reference_rates
 from v19_contact_reference.task_qp import evaluate_task
 with ExitStack() as guards:
  for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_comPos','mj_collision','mj_contactForce'):
   if hasattr(mujoco,name):guards.enter_context(patch.object(mujoco,name,deny(name)))
  guards.enter_context(patch.object(hand_plant.HandPlant,'__init__',deny('HandPlant')));guards.enter_context(patch.object(ReferenceActor21,'apply',deny('actor.apply')))
  for stage in p['stages']:
   tag=stage['name'];folder=ROOT/stage['output'];assignment=stage['assignment'];row=dict(stage=tag,status='unassessed');rows.append(row)
   try:
    done=complete[tag];reg=read(ROOT/stage['registration']);manifest=read(folder/'manifest.json');snap=read(folder/'source_snapshot_manifest.json');ledger=read(folder/'assignment_ledger.json');trials=read(folder/'trials.json');fail=read(folder/'runtime_failures.json')
    ck(tag+'/assigned_registration_snapshot156_env42',reg['assignment']==assignment and reg['source_sha256']==p['source_sha256'] and reg['environment']==p['environment'] and manifest['research_source_sha256']==p['source_sha256'] and set(snap['files'])==set(p['source_sha256']) and all(snap['files'][k]['sha256']==v and h(folder/snap['files'][k]['snapshot_path'])==v for k,v in p['source_sha256'].items()) and read(folder/'environment.json')['pinned_environment']==p['environment'])
    ck(tag+'/terminal_hashes_ledger_all_attempts',len(ledger)==len(trials)==1 and ledger[0]['status']=='evaluated' and not fail and not done['runtime_failures'] and all(done[k+'_sha256']==h(folder/(k+'.json')) for k in ('trials','assignment_ledger','plan','manifest')) and manifest['source_snapshot_manifest_sha256']==h(folder/'source_snapshot_manifest.json'))
    scene,method,seed=assignment['scenarios'][0],assignment['methods'][0],assignment['seeds'][0];run=folder/'runs'/f'{scene}__{method}__{seed}';old=ROOT/stage['full_reference'];meta=read(run/'metadata.json');result=read(run/'result.json');om=read(old/'metadata.json')
    ck(tag+'/trace_result_metadata_same_declared_identity',result==trials[0] and (result['scenario'],result['method'],result['seed'])==(scene,method,seed) and meta['research_source_sha256']==p['source_sha256'] and meta['frozen_full_v19_source_sha256']==fullplan['source_sha256'] and meta['trace_sha256']==result['trace_sha256']==h(run/'trace.npz') and meta['ablation_role']==ROLE and meta['motion_tasks_enabled'] is False and om['trace_sha256']==stage['full_trace_sha256']==h(old/'trace.npz'))
    context_keys=('shape','scenario','seed','split','config','initial_position','limits','budget','noise','model_sha256','motion_mode','observer_warmup_s','actual_ratio_task_gains','task_policy','receiver_policy','release_at')
    ck(tag+'/same_physical_noise_and_public_task_inputs',all(meta[k]==om[k] for k in context_keys))
    need=set(COLS)|{'observation_json','debug_json','controller_active','requested_reference21','reference_increment21','mode_generation','robot_control_time_s','prediction_available','predicted_normal','substep_time','substep_duration'}
    with np.load(run/'trace.npz',allow_pickle=False) as z:a={k:z[k] for k in need}
    with np.load(old/'trace.npz',allow_pickle=False) as z:b={k:z[k] for k in (*COLS,'observation_json')}
    t=a['time'];pre=t<12.-1e-9;preold=b['time']<12.-1e-9;comparisons=[]
    for k in (*COLS,'observation_json'):
     x=a[k][pre];y=b[k][preold];text=x.dtype.kind in 'US' and y.dtype.kind in 'US';ok=x.shape==y.shape and (all(str(u).encode('utf8')==str(v).encode('utf8') for u,v in zip(x.flat,y.flat)) if text else bits(x,y))
     comparisons.append(dict(column=k,passed=bool(ok),exact_UTF8_if_text=text))
    ck(tag+'/pre12_all37_plus_complete_public_observation_UTF8_exact',int(pre.sum())==2400 and all(x['passed'] for x in comparisons),[x for x in comparisons if not x['passed']])
    ck(tag+'/complete16s_3201_32000_no_fake_prediction',meta['completed'] and bits(t,np.arange(3201)*.005) and len(a['substep_time'])==32000 and np.isfinite(a['substep_time']).all() and np.isfinite(a['substep_duration']).all() and np.all(a['substep_duration']>0) and np.isnan(a['predicted_normal']).all() and not np.any(a['prediction_available']))
    model,xmlsha=hand_plant._model(FoundationConfig(**meta['config']));actor=ReferenceActor21(model);servo=RobotServoEffort(model)
    enc=a['executed_reference21'].copy();enc[:,18]+=actor.lift_bias_m
    ck(tag+'/model_and_all_actual21_engine_encoding',xmlsha==meta['model_sha256'] and bits(enc,a['ctrl'][:,actor.indices21]) and bits(a['robot_actuator_target21'],a['ctrl_before'][:,actor.indices21]))
    obs=[json.loads(str(x)) for x in a['observation_json']];debug=[json.loads(str(x)) for x in a['debug_json']];active=a['controller_active'].astype(bool);actual_errors=[]
    for i in np.flatnonzero(active):
     f=obs[i];ex=a['executed_reference21'][i];cur=np.asarray(f['robot_reference21']);gen=int(a['mode_generation'][i])
     if np.any(ex<np.asarray(f['reference_lower21'])-1e-12) or np.any(ex>np.asarray(f['reference_upper21'])+1e-12) or np.max(abs(ex-cur)-reference_rates(debug[i]['mode'])*.005)>1e-12:actual_errors.append((int(i),'mechanical/rate'))
     if gen!=f['mode_generation'] or gen!=debug[i]['mode_generation'] or f['robot_control_time_s']!=t[i] or not bits(ex-cur,a['reference_increment21'][i]):actual_errors.append((int(i),'mode/time/increment'))
    ck(tag+'/every_active5ms_actual_authority_rate_generation',not actual_errors,actual_errors[:8])
    dm=read(run/'release_decision_manifest.json');decisionpath=run/dm['file'];ck(tag+'/decision_hash',h(decisionpath)==dm['sha256'])
    with gzip.open(decisionpath,'rt',encoding='utf8') as f:qs=[json.loads(x) for x in f]
    expected=np.flatnonzero([active[i] and debug[i].get('architecture')=='v19_joint_reference_contact_progress' for i in range(len(t))]);indices=np.array([round(q['time_s']/.005) for q in qs],int)
    ck(tag+'/complete_unique_release5ms_records',len(qs)==dm['records']==801 and np.array_equal(indices,expected))
    task_errors=[];rhs_errors=[];receipt_errors=[];counts=Counter();original_nonzero=0;native_motion_calls=0
    for k,q in enumerate(qs):
     i=indices[k];f=obs[i];ci=q['contact_intents'];problem=q['qp_problem'];cur=np.asarray(q['current_reference21'],float);ex=a['executed_reference21'][i];g=q['governor'];receipt=q.get('execution_receipt');counts['phase:'+q['phase']]+=1;counts['accepted' if q['reference_effort_goal_accepted'] else 'fallback']+=1;counts['qp:'+str((q['qp'] or {}).get('status'))]+=1
     if q['decision_id']!=k or q['time_s']!=t[i] or q['mode_generation']!=int(a['mode_generation'][i]) or not numeric_bits(cur,f['robot_reference21']):task_errors.append((k,'clock/ref/mode'))
     original=np.asarray(ci['original_motion_generalized_effort25'],float);assigned=np.asarray(ci['assigned_disabled_generalized_effort25'],float);original_nonzero+=bool(np.any(original!=0))
     te=np.zeros(25);tr=np.zeros(25)
     for x in ci['motion_records']:te+=np.asarray(x['external_generalized_effort25'])
     for x in ci['robot_motion_records']:tr+=np.asarray(x['generalized_effort25'])
     if ci['motion_tasks_enabled'] is not False or ci['ablation_role']!=ROLE or ci['motion_task_disabled_is_not_missing_observation'] is not True or ci['zero_task_is_not_observed_zero_velocity'] is not True or not numeric_bits(original,te+tr) or not bits(assigned,np.zeros(25)):task_errors.append((k,'disable labels/original reconstructed tau'))
     velocity=g['governed_velocity_intent_m_s'] if q['phase']=='SEARCH' else 0.
     if ci['declared_lift_velocity_target_m_s']!=velocity or ci['task_target_time_s']!=q['time_s'] or q['public_process']['holding_force_tasks_active'] or q['placement_certified'] or q['future_motion_or_pressure_prediction_available']:task_errors.append((k,'stage target/false certificate'))
     for x in ci['motion_records']:
      if x['external_projection_available']:
       if not numeric_bits(np.asarray(x['acquired_robot_projection2_m_s'])+np.asarray(x['relative_slip2_m_s']),x['measured_external_projection2_m_s']) or not numeric_bits(np.asarray(x['measured_external_projection2_m_s'])-np.asarray(x['external_projection_target2_m_s']),x['external_projection_error2_m_s']):task_errors.append((k,'measurement/target projection'))
      elif x['relative_slip2_m_s'] is not None or x['measured_external_projection2_m_s'] is not None:task_errors.append((k,'unknown filled'))
     if problem is not None:
      motion=np.asarray(q['motion_reference21']);mv=servo.evaluate(np.asarray(f['robot_position25']),np.asarray(f['robot_velocity25']),actor.encode(Reference21(motion,q['time_s'])).values,time_s=q['time_s']);native_motion_calls+=1
      delta=np.zeros(21);delta[18:]=(mv.predicted_effort21-np.asarray(problem['p0_21']))[18:];expected_rhs=np.asarray(problem['A21x25']).T@delta
      if mv.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA' or not numeric_bits(expected_rhs,problem['tau_extra25']):rhs_errors.append(k)
     L=np.maximum(f['reference_lower21'],cur-reference_rates('release')*.005);U=np.minimum(f['reference_upper21'],cur+reference_rates('release')*.005);qlift=float(f['robot_position25'][24]);vlift=float(f['robot_velocity25'][24]);task,reachable,_=helper.task_interval(g,qlift,vlift,float(L[18]),float(U[18]));debt=qlift-float(ex[18]);changed=not numeric_bits(q['proposed_reference21'],ex)
     if not numeric_bits(L,q['original_mechanical_rate_lower21']) or not numeric_bits(U,q['original_mechanical_rate_upper21']) or not numeric_bits(task,[q['actual_reference_lower21'][18],q['actual_reference_upper21'][18]]) or not task[0]<=ex[18]<=task[1] or q['governor_task_domain']['budget_band_reachable_this_tick']!=reachable:receipt_errors.append((k,'task domain'))
     if receipt is None or not numeric_bits(receipt['actual_reference21'],ex) or receipt['same_snapshot_q_lift_m']!=qlift or receipt['same_snapshot_reference_debt_m']!=debt or receipt['actual_task_interval_respected']!=(task[0]<=ex[18]<=task[1]) or receipt['execution_changed']!=changed or receipt['current_q_budget_respected']!=(-g['downward_recovery_reference_budget_m']<=debt<=g['recovery_reference_budget_m']):receipt_errors.append((k,'actual receipt'))
    ck(tag+'/all_release_disable_labels_original_tau_measurements_targets',not task_errors,task_errors[:8])
    ck(tag+'/all_QP_RHS_exactly_original_native_lift_wrist_only',not rhs_errors,dict(native_motion_evaluations=native_motion_calls,bad_indices=rhs_errors[:8]))
    ck(tag+'/all_release_TASK_interval_and_final_actual_receipt',not receipt_errors,receipt_errors[:8])
    reps=sorted({0,len(qs)-1}|{k for k,q in enumerate(qs) if q['phase']=='TRANSFER'} and {0,len(qs)-1,next((k for k,q in enumerate(qs) if q['phase']=='TRANSFER'),0)})
    re=[]
    for k in reps:
     errs,_=helper.audit_v19_task(qs[k],a['executed_reference21'][indices[k]],actor,evaluate_task);re.extend((k,e) for e in errs)
    ck(tag+'/preselected_first_last_firstTRANSFER_full_same_task_scores',not re,dict(indices=reps,errors=re[:8]))
    row.update(status='audited',run=str(run),input_sha256={name:h(run/name) for name in ('trace.npz','metadata.json','result.json','release_decisions.jsonl.gz')},prefix37_plus_observation=comparisons,full_reference=str(old),active_ticks=int(active.sum()),release_decisions=len(qs),original_nonzero_tau_records=original_nonzero,counts=dict(counts),representative_full_score_indices=reps,original_task_success=result['task_success'],original_full_motion=result['research_metrics']['full_motion_mission_success'],original_force_guard=result['research_metrics']['safety']['guard_pass'])
   except Exception as exc:
    row.update(status='reader_error',error=repr(exc));issues.append(dict(stage=tag,error=repr(exc),traceback=traceback.format_exc()))
 ck('all3_original_assigned_including_failed_labels_retained',len(rows)==3 and all(r['status']=='audited' for r in rows))
 ck('no_Data_step_forward_FK_contact_or_writer',not forbidden,forbidden)
 result=dict(schema='v19_motion_ablation_execution_independent_v1',passed=not issues,checks=checks,issues=issues,rows=rows,plan_sha256=h(PLAN),source_sha256=p['source_sha256'],parent152_source_sha256=fullplan['source_sha256'],input_sha256=inputs,reader_sha256=sha(__file__),saved_field_helper_sha256=h(helperpath),physics_steps=0,forbidden_calls=forbidden,scope=dict(assigned=3,all_release_records=True,pre12_numeric37_byte_exact=True,complete_JSON_UTF8_no_field_removal=True,extra_public_observation=True,native_algebra='Only recorded motion reference F067 RHS reconstruction, every known-branch decision; no inverse/solver/physics.',full_task_math='Only predeclared first,last,firstTRANSFER per run; full801x3 math not repeated.',model_scope='Static model compile/actor address+liftbias only; no Data/FK/forward/contacts/writer.',RNG_scope='Configuration and complete pre12 public packets exact; final RNG internal state not archived and not claimed exact.',outcomes='Original failed labels retained; package contrast not individual external-vs-robot causal decomposition or new algorithm performance.'))
 with out.open('x',encoding='utf8') as f:json.dump(plain(result),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(passed=result['passed'],checks=len(checks),issues=issues[:5],out=str(out),sha256=sha(out)),ensure_ascii=False));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

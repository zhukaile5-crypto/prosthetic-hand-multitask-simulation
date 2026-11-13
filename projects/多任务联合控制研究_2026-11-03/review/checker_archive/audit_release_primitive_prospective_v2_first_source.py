"""Independent terminal saved-data audit; no producer/physics imports.
The reviewer authored the earlier V3 measurement wrapper, not this prospective
runner. Author readout is never used as proof. New/old failed attempts persist.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
import argparse,ast,base64,hashlib,json,sys,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'output/release_primitive_prospective_v2'
OUT=ROOT/'review/release_primitive_prospective_v2_independent_v1.json'
BRIDGE=ROOT/'output/release_primitive_bridge_v2_final/regression_complete.json'
PURE=ROOT/'review/verify_release_basis_v2_independent.py'
RESOLUTION=ROOT/'review/verify_release_clearance_resolution_v1.py'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
WALL={'qp_seconds','control_ms','wall_seconds'};checks=[];issues=[]
def check(name,value,detail=None):
 row=dict(name=name,passed=bool(value),detail=detail);checks.append(row)
 if not value:issues.append(row)
def metric(name,a,b):
 same=a is b if a is None or b is None else a==b if isinstance(a,(bool,str)) or isinstance(b,(bool,str)) else bool(np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True))
 check(name,same,None if same else dict(actual=a,reported=b))
def load_helpers():
 tree=ast.parse(PURE.read_text(encoding='utf-8'))
 names={'sha','digest','encode','strip_wall','mapping','at','dwell','variation'}
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(PURE)+'::pure','exec'),globals())
 loop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='job' and isinstance(n.iter,ast.Name) and n.iter.id=='jobs')
 first=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.With) and 'np.load(trace' in ast.unparse(n));last=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.With) and 'decoder_commands.npz' in ast.unparse(n))
 body=[]
 for n in loop.body[first:last]:
  if isinstance(n,ast.Expr) and any(isinstance(v,ast.Constant) and isinstance(v.value,str) and 'summary_' in v.value for v in ast.walk(n)):continue
  if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and ast.unparse(n.target)=='(stage, mask)':continue
  body.append(deepcopy(n))
 keys=('safety','release','mission','full_motion','raw','time');values=[ast.Name(id='safety',ctx=ast.Load()),ast.Name(id='release_values',ctx=ast.Load()),ast.Name(id='mission',ctx=ast.Load()),ast.BoolOp(op=ast.And(),values=[ast.Name(id='mission',ctx=ast.Load()),ast.Compare(left=ast.Name(id='wfail',ctx=ast.Load()),ops=[ast.Is()],comparators=[ast.Constant(None)])]),ast.Name(id='x',ctx=ast.Load()),ast.Name(id='t',ctx=ast.Load())]
 body.append(ast.Return(value=ast.Dict(keys=[ast.Constant(k) for k in keys],values=values)))
 f=ast.FunctionDef(name='raw_check',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=k) for k in ('jid','d','reported','r','m','trace')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[],type_params=[])
 exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(PURE)+'::raw-arithmetic','exec'),globals())
 # Independent clearance loops; rename to avoid replacing Boolean dwell.
 tree=ast.parse(RESOLUTION.read_text(encoding='utf-8'));nodes=[]
 for n in tree.body:
  if isinstance(n,ast.FunctionDef) and n.name in {'episodes','dwell','window_violations'}:
   n=deepcopy(n)
   if n.name=='dwell':n.name='clear_dwell'
   nodes.append(n)
 globals()['EPS']=1e-9
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(RESOLUTION)+'::pure-clearance','exec'),globals())

def num(value):
 if isinstance(value,dict) and 'bytes_b64' in value:return np.frombuffer(base64.b64decode(value['bytes_b64']),dtype=value['array_dtype']).reshape(value['shape']).copy()
 if isinstance(value,dict) and 'float_hex' in value:return float.fromhex(value['float_hex'])
 return value
def clearance(t,f,release,end,step):
 a=clear_dwell(t,f,.2,.2,release,end,np.full(len(t),step))
 b=clear_dwell(t,f,.2,.2,release,min(end,release+3.5),np.full(len(t),step))
 return dict(first_any_confirmation_s=a['first_dwell_confirmation_s'],first_by_deadline_confirmation_s=b['first_dwell_confirmation_s'],terminal_clear_start_s=a['final_clear_tail_start_s'],terminal_clear_confirmation_s=a['final_clear_tail_confirmation_s'],recontact_episodes=a['recontact_after_first_confirmation'],terminal_censored=a['final_clear_tail_confirmation_s'] is None)

def read_entry(folder,row,name,sources):
 p=folder/name;signature=read(p/'entry_signature.json');side=read(p/'entry_side_state.json');local=read(p/'local_control_state.json');follower=read(p/'v10_preview_causal_state.json')
 with np.load(p/'entry_physics.npz',allow_pickle=False) as z:physics={k:z[k].item() if z[k].shape==() else z[k] for k in z.files}
 computed=dict(signature,local_control_state_sha256=digest(local),v10_preview_causal_state_sha256=digest(strip_wall(follower)))
 check(folder.name+'/'+name+'/independent_complete_checksums',computed==row['entries'][name] and signature['physics_arrays_sha256']==digest(encode(physics)) and signature['integration_state_sha256']==digest(encode(physics['integration_state'])) and signature['python_causal_side_state_sha256']==digest(strip_wall(side)))
 rng=at(side,'sensors','rng');queue=at(side,'sensors','queue');history=at(side,'controller','state','action_history');fk=at(side,'skin_geometry','fk_data')
 check(folder.name+'/'+name+'/real_rng_queue_observer_fk',rng.get('numpy_generator')=='PCG64' and len(queue['items'])>0 and len(history['items'])>0 and {'integration_state','qpos','qvel','ctrl','qacc_warmstart'}<=set(mapping(fk)))
 check(folder.name+'/'+name+'/follower_implementation_and_origin',at(follower,'preview_follower','implementation_sha256')==sources['research/v10/runtime.py'] and at(follower,'preview_follower','exact_type')=='v10.runtime.PublicPreviewFollower' and num(at(local,'probe_origin_tick'))==4)
 refs=read(p/'entry_code_references.json')
 check(folder.name+'/'+name+'/source_bound_native_API_reference',len(refs)==1 and refs[0]['origin_sha256']==sources[refs[0]['origin_source_key']] and refs[0]['namespace_not_serialized'] and all(sources[k]==h for k,h in refs[0]['covered_native_binding_sha256'].items()))
 return dict(physics=physics,side=strip_wall(side),local=local,follower=strip_wall(follower),population=dict(sensor_queue=len(queue['items']),observer_history=len(history['items']),rng='PCG64'))

def same_entries(a,b):
 return all(a[n]['physics'].keys()==b[n]['physics'].keys() and all(np.array_equal(a[n]['physics'][k],b[n]['physics'][k],equal_nan=True) for k in a[n]['physics']) and all(a[n][f]==b[n][f] for f in ['side','local','follower']) for n in a)
def same_prefix(pa,pb):
 differing=[];count=0
 with np.load(pa,allow_pickle=False) as a,np.load(pb,allow_pickle=False) as b:
  check(pa.parent.parent.name+'/paired_trace_field_set',set(a.files)==set(b.files))
  for name in a.files:
   if name in ('metadata','control_ms'):continue
   x,y=a[name],b[name]
   if x.ndim==0:continue
   clock=a['substep_time'] if name.startswith('substep_') else a['time'];mask=clock<=12.020+1e-9;x,y=x[mask],y[mask]
   if name=='debug_json':equal=all(strip_wall(json.loads(str(u)))==strip_wall(json.loads(str(v))) for u,v in zip(x,y))
   else:equal=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
   count+=1
   if not equal:differing.append(name)
 return dict(fields=count,mismatches=differing,exact=not differing,substep_mask='saved start<=12.020; offset still0 until first quarter12.025')

def main():
 summary=read(RUN/'summary.json') if (RUN/'summary.json').exists() else None
 if summary is None or summary.get('execution_complete') is not True:raise RuntimeError('NOT_READY: no branch/partial data read before complete terminal summary')
 if OUT.exists():raise FileExistsError(OUT)
 load_helpers()
 manifest=read(RUN/'manifest.json');snap=read(RUN/'source_snapshot_manifest.json');rows=read(RUN/'results.json');jobs=manifest['jobs'];ledger=read(RUN/'assignment_ledger.json');plan=read(RUN/'plan.json');terminal=read(RUN/'execution_complete.json');lock=read(ROOT/'identification/release_primitive_validation_v2/dependency_lock.json')
 expected={(s,m,n,v) for s in ('nominal','friction_drop','fragile_conflict','collision') for m in ('hold','transport') for n in (617901,617902) for v in ('zero','lift_slow')}
 key=lambda r:(r['scenario'],r['motion_mode'],r['seed'],r['variant'])
 check('exact32_assignment_result_ledger_and_all_failures',len(rows)==len(jobs)==len(ledger)==32 and {key(j) for j in jobs}=={key(r) for r in rows}==expected and Counter(r['job_id'] for r in rows)==Counter(j['job_id'] for j in jobs)==Counter(x['job']['job_id'] for x in ledger) and rows==summary['results']==read(RUN/'results_partial.json') and [r for r in rows if r['status']!='complete']==summary['failures']==read(RUN/'runtime_failures.json'))
 check('final_plan_snapshot_lock_terminal_bindings',sha(RUN/'plan.json')==manifest['plan_sha256'] and sha(RUN/'source_snapshot_manifest.json')==manifest['source_snapshot_manifest_sha256'] and manifest['sources']==lock['sources'] and manifest['dependency_lock_sha256']==sha(ROOT/'identification/release_primitive_validation_v2/dependency_lock.json')=='7b1a07e1898d06fced2368d6f5e4dc2db86a2cc0001e4553356aa26f70d0d710' and terminal['summary_sha256']==sha(RUN/'summary.json') and terminal['results_sha256']==sha(RUN/'results.json'))
 source_checks=[]
 for key,h in manifest['sources'].items():
  e=snap['files'][key];p=Path(e['original_path']);q=RUN/e['snapshot_path'];source_checks.append(dict(key=key,current_sha256=sha(p),snapshot_sha256=sha(q),expected=h,matches=sha(p)==sha(q)==e['sha256']==h and p.stat().st_size==q.stat().st_size==e['bytes']))
 check('all316_snapshot_current_bytes_and_declared_files',len(source_checks)==316 and all(e['matches'] for e in source_checks) and set(snap['files'])==set(manifest['sources'])=={p.relative_to(RUN/'sources').as_posix() for p in (RUN/'sources').rglob('*') if p.is_file()})
 bridge=read(BRIDGE);check('only_final_bridge_same_source_lock',bridge['passed'] and bridge['source_identity']==manifest['sources'] and bridge['dependency_lock_sha256']==manifest['dependency_lock_sha256'] and bridge['scientific_samples']==0 and all(sha(ROOT/p)==h for p,h in bridge['input_sha256'].items()))
 # Recompute final bridge40 physical fields and every public observation string.
 bridge_paths=[ROOT/p for p in bridge['input_sha256']];physical_node=next(n for n in ast.parse((ROOT/'identification/release_local_response_v2/run_pilot.py').read_text(encoding='utf-8')).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PHYSICAL_FIELDS' for t in n.targets));physical=ast.literal_eval(physical_node.value)
 with np.load(bridge_paths[0],allow_pickle=False) as a,np.load(bridge_paths[1],allow_pickle=False) as b:
  check('final_bridge40arrays_and_all_observations_recomputed',all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes() for k in physical) and np.array_equal(a['observation_json'],b['observation_json']))
 protocol=read(RUN/'sources/research/protocol.json');check('frozen_protocol_numeric_thresholds',protocol['thresholds']['force_numerical_tolerance_N']==.02 and protocol['thresholds']['all_hand_clearance_N']==.2 and protocol['thresholds']['release_clearance_dwell_s']==.2 and protocol['thresholds']['release_deadline_s']==3.5)
 xml=ET.parse(RUN/'sources/foundation/assets/legacy_model.xml').getroot();bounds={x.get('class'):np.fromstring(x.find('position').get('ctrlrange'),sep=' ') for x in xml.iter('default') if x.find('position') is not None and x.find('position').get('ctrlrange')};act=list(xml.find('actuator'))[2:20];lower=np.array([bounds[x.get('class')][0] for x in act]);upper=np.array([bounds[x.get('class')][1] for x in act])
 constants={}
 for n in ast.parse((RUN/'sources/foundation/foundation_config.py').read_text(encoding='utf-8')).body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('OPEN','CLOSED') for t in n.targets):constants[n.targets[0].id]=ast.literal_eval(n.value)
 records={};profiles={};entries={};metas={};paths={};byid={r['job_id']:r for r in rows}
 for job in jobs:
  jid=job['job_id'];row=byid[jid]
  if row['status']!='complete':records[jid]=dict(status=row['status'],assigned_job=job,failure=row);continue
  d=Path(job['directory']);trace=d/'record/trace.npz';m=read(d/'record/metadata.json');reported=read(d/'result.json');r=reported['research_metrics'];pilot=read(d/'pilot_metadata.json');metas[jid]=m;paths[jid]=trace
  check(jid+'/trace_and_actual_inputs_source_bindings',sha(trace)==row['trace_sha256']==m['trace_sha256']==reported['trace_sha256']==r['trace_sha256']==pilot['record_trace_sha256'] and m['research_source_sha256']==manifest['sources'] and pilot['command_record_sha256']==sha(d/'decoder_commands.npz') and pilot['robot_stream_sha256']==sha(d/'robot_physical_stream.npz') and m['config']==pilot['config']==r['physical_parameters'] and pilot['entry']==row['entries'] and r['evaluation_source_sha256']==manifest['sources']['research/evaluation.py'] and r['protocol_sha256']==manifest['sources']['research/protocol.json'])
  check(jid+'/registered_context_and_public_runtime',m['method']=='ratio_preview' and m['scenario']['name']==job['scenario'] and m['motion_mode']==job['motion_mode'] and m['seed']==job['seed'] and m['shape']=='capsule' and m['split']=='robustness' and m['config']['control_dt']==.005 and m['config']['timestep']==.0005 and m['observer_warmup_s']==.2 and m['release_at']==12.)
  record=raw_check(jid,d,reported,r,m,trace);x=record.pop('raw');t=record.pop('time');st=x['substep_time'];dur=x['substep_duration'];f=x['substep_skin_normal'];mechanical=m['mechanical_release_at'];limits=np.array(m['limits']);budget=m['budget']
  masks={'shared_preparation':st<7.5-1e-9,'policy_holding_transport' if m['handoff_passed'] else 'post_preparation_without_policy':(st>=7.5-1e-9)&(st<mechanical-1e-9),'mechanical_release_and_after':st>=mechanical-1e-9};stages={}
  for name,mask in masks.items():
   if not mask.any():continue
   values=dict(duration_s=float(dur[mask].sum()),finger_peak_N=f[mask].max(axis=0).tolist(),local_excess_integral_Ns=float(np.sum(np.maximum(f[mask]-limits,0.)*dur[mask,None])),total_excess_integral_Ns=float(np.maximum(f[mask].sum(axis=1)-budget,0.)@dur[mask]));stages[name]=values
   for k,v in values.items():metric(jid+'/stage/'+name+'/'+k,v,r['safety']['stages'][name][k])
  check(jid+'/exact_actual_mechanical_stage_partition',set(stages)==set(r['safety']['stages']))
  entries[jid]={n:read_entry(d,row,n,manifest['sources']) for n in ['handoff_boundary','scheduled_release_boundary','scheduled_probe_boundary']}
  with np.load(d/'decoder_commands.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
  profiles[jid]=p;release_tick=round(mechanical/.005);eligible=bool(m['handoff_passed'] and release_tick==2400);enabled=bool(eligible and job['variant']=='lift_slow');effective='lift_slow' if enabled else 'zero'
  check(jid+'/actual_public_entry_gate_and_early_native_fallback',m['primitive_entry_eligible']==pilot['primitive_entry_eligible']==row['primitive_entry_eligible']==eligible and m['primitive_offset_enabled']==pilot['primitive_offset_enabled']==row['primitive_offset_enabled']==enabled and m['primitive_actual_variant']==pilot['actual_variant']==effective and pilot['requested_variant']==job['variant'])
  entry=entries[jid]['scheduled_release_boundary' if release_tick==2400 else 'handoff_boundary'];plant=at(entry['side'],'plant_side');alpha=num(at(plant,'alpha'));entrylift=num(at(plant,'lift_ref'));wrists=num(at(plant,'wrist_ref'));thumb=entry['physics']['ctrl'][2:7]
  tick=p['relative_control_tick'];tau=p['control_time']-mechanical;weight=np.clip((tick-4.)/4.,0.,1.);lc=(release_tick+tick-4*weight)*.005-mechanical if enabled else tau;lift=entrylift*(1.-np.clip(lc/.6,0.,1.));unclip=entrylift*(1.-lc/.6)
  check(jid+'/all_actual_held_quarters_and_floors',len(tick)==1+11*(3200-release_tick) and np.all(tick==np.rint(tick)) and np.array_equal(p['ramp_weight'],weight) and np.array_equal(p['native_phi_s'],tau) and np.array_equal(p['actual_phi_s'],tau) and np.array_equal(p['lift_clock_s'],lc) and np.array_equal(p['actual_command21'][:,18],lift) and np.array_equal(p['unclipped_lift_reference_m'],unclip) and np.array_equal(p['lift_clipped'].astype(bool),(unclip<0.)|(unclip>entrylift)))
  # Independent closed-form native21 reference, then lift-only treatment.
  native=[]
  for elapsed in tau:
   tuck=np.array([-1.03,1.21,.13,-.02,.985]);opened=np.array([-1.,.1,0.,0.,0.])
   th=thumb+np.clip(elapsed/.8,0,1)*(tuck-thumb) if elapsed<.8 else tuck if elapsed<1.4 else tuck+np.clip((elapsed-1.4)/.8,0,1)*(opened-tuck)
   finger=[np.clip(th,lower[:5],upper[:5])]
   for i in range(1,5):finger.append(np.array(constants['OPEN'][i])+np.maximum(0.,alpha[i]-.8*elapsed)*(np.array(constants['CLOSED'][i])-constants['OPEN'][i]))
   native.append(np.r_[np.concatenate(finger),entrylift*(1.-np.clip(elapsed/.6,0.,1.)),wrists])
  native=np.asarray(native);expected=native.copy();expected[:,18]=lift
  check(jid+'/independent_native21_and_lift_only_realization',np.array_equal(native,p['native_carrier_command21']) and np.array_equal(expected,p['actual_command21']) and np.array_equal(p['actual_ctrl'][:,:2],expected[:,19:]) and np.array_equal(p['actual_ctrl'][:,2:20],expected[:,:18]) and np.max(abs((p['actual_ctrl'][:,20]-expected[:,18])-(p['native_carrier_ctrl'][:,20]-native[:,18])))<1e-15)
  if effective=='zero':check(jid+'/zero_every_full22ctrl_shadow_native_exact',np.array_equal(p['actual_ctrl'],p['native_carrier_ctrl']))
  unique=np.r_[True,np.diff(tick)!=0];steps=abs(np.diff(expected[unique],axis=0));native_steps=abs(np.diff(native[unique],axis=0));same=np.diff(tick)==0
  check(jid+'/held_native_speed_and_cumulative_21authority',np.all(np.diff(expected,axis=0)[same]==0.) and np.all(np.diff(tau[unique])<=.005+1e-12) and np.all(np.diff(lift[unique])<=1e-12) and np.all(-np.diff(lift[unique])<=entrylift/.6*.005+1e-12) and np.all(steps<=native_steps.max(axis=0)+1e-10) and np.all(steps.sum(axis=0)<=native_steps.sum(axis=0)+1e-9))
  with np.load(d/'robot_physical_stream.npz',allow_pickle=False) as z:stream={k:z[k] for k in z.files}
  with np.load(trace,allow_pickle=False) as z:robot=np.column_stack((z['qpos'][:,m['raw_joint_qpos_indices']],z['lift']));velocity=z['qvel'][:,list(m['raw_joint_qpos_indices'])+[0]];hand=z['substep_all_hand_object']
  end=st+dur;clock=stream['physical_time_s'];check(jid+'/full32001_robot_encoders_and_raw_end_query',len(clock)==32001 and np.allclose(clock[1:],end,rtol=0,atol=1e-12) and np.array_equal(stream['position25'][::10],robot) and np.array_equal(stream['velocity25'][::10],velocity) and np.array_equal(stream['external_finger_pressure5_N'][1:],f) and np.array_equal(stream['all_hand_target_pressure_N'][1:],hand))
  check(jid+'/all_target_hand_categories_and_external_scope',np.max(abs(stream['target_finger_pressure5_N'].sum(axis=1)+stream['palm_pressure_N']+stream['other_hand_pressure_N']-stream['all_hand_target_pressure_N']))<1e-10 and np.all(f>=0.) and np.all(stream['target_finger_pressure5_N']<=stream['external_finger_pressure5_N']+1e-12))
  clear200=clearance(t,x['all_hand_object'],mechanical,16.,.005);clearend=clearance(end,hand,mechanical,16.,.0005);clearsaved=clearance(st,hand,mechanical,16.,.0005);diagnostic=read(d/'clearance_diagnostics.json')
  for actual,stored,name in [(clear200,diagnostic['observed200Hz'],'200Hz'),(clearend,diagnostic['physical_end2000Hz'],'raw_end')]:
   for k in ['first_any_confirmation_s','first_by_deadline_confirmation_s','terminal_clear_start_s','terminal_clear_confirmation_s']:metric(jid+'/clearance/'+name+'/'+k,actual[k],stored[k])
  returned=r['release'];confirmation=None if returned['physical_clearance_latency_s'] is None else returned['actual_command_at']+returned['physical_clearance_latency_s'];reported_window=diagnostic['raw_inside_original200Hz_first_confirmation_window']
  if confirmation is None:check(jid+'/no_original_confirmation_no_invented_raw_window',reported_window is None)
  else:
   lo=confirmation-.2;mask=(end>=lo-1e-9)&(end<=confirmation+1e-9);interval=(end>lo+1e-9)&(end<=confirmation+1e-9);metric(jid+'/original_dwell_raw_peak',float(hand[mask].max()),reported_window['peak_target_contact_N']);metric(jid+'/original_dwell_raw_violations',int(((hand>.2)&mask).sum()),reported_window['above_threshold_physical_end_samples']);metric(jid+'/original_dwell_raw_duration',float(np.sum(dur[interval]*(hand[interval]>.2))),reported_window['above_threshold_duration_s'])
  floor=np.flatnonzero(p['actual_command21'][:,18]==0.);tol_floor=np.flatnonzero(p['actual_command21'][:,18]<=1e-12)
  pre_release_qualifiers=dict(handoff_passed=m['handoff_passed'],preparation_failure=m['preparation_failure'],initial_stability_wait_s=r['initial_stability_wait_s'],wrist_failure_before_release=r['wrist_tracking_guard']['failure_at'],failure_before_release=None if r['failure_at'] is None or r['failure_at']>12. else [r['failure_reason'],r['failure_at']])
  record.update(status='complete',job=job,actual_mechanical_release_s=mechanical,eligible=eligible,offset_enabled=enabled,effective_variant=effective,pre_release_qualifiers=pre_release_qualifiers,stage_safety=stages,clearance_200Hz=clear200,clearance_raw_start=clearsaved,clearance_raw_end=clearend,first_strict_lift_zero_s=float(p['control_time'][floor[0]]) if len(floor) else None,first_lift_within1e12m_s=float(p['control_time'][tol_floor[0]]) if len(tol_floor) else None,entry_population={n:e['population'] for n,e in entries[jid].items()},input_sha256={str(path.relative_to(RUN)):sha(path) for path in [trace,d/'result.json',d/'pilot_metadata.json',d/'record/metadata.json',d/'decoder_commands.npz',d/'robot_physical_stream.npz',d/'clearance_diagnostics.json']});records[jid]=record
 pairs=[]
 for scenario,motion,seed in sorted({(s,m,n) for s,m,n,v in expected}):
  a=f'{scenario}__{motion}__{seed}__zero';b=f'{scenario}__{motion}__{seed}__lift_slow'
  if a not in paths or b not in paths:pairs.append(dict(scenario=scenario,motion_mode=motion,seed=seed,paired=False,reason='retained incomplete assigned pair'));continue
  prefix=same_prefix(paths[a],paths[b]);exact=same_entries(entries[a],entries[b]);ma,mb=metas[a],metas[b];pa,pb=profiles[a],profiles[b]
  config_equal=all(ma[k]==mb[k] for k in ['config','noise','gains','limits','budget','initial_position','model_sha256'])
  check(a+'/all3entry_physics_RNG_observer_queue_FK_follower_exact',exact);check(a+'/all_actual_prefix_columns_exact',prefix['exact'],prefix);check(a+'/same_actual_physical_noise_gain_material',config_equal)
  check(a+'/actual_inputs_vs_matched_executed_zero',np.array_equal(pa['control_time'],pb['control_time']) and np.array_equal(pa['physical_time'],pb['physical_time']) and np.array_equal(pb['actual_phi_s']-pa['actual_phi_s'],pb['actual_phi_offset_s']) and np.array_equal(pb['actual_command21'][:,18]-pa['actual_command21'][:,18],pb['actual_lift_offset_m']) and np.array_equal(pb['native_carrier_command21'],pa['actual_command21']))
  check(a+'/pre_release_handoff_stability_tracking_qualifiers_same',records[a]['pre_release_qualifiers']==records[b]['pre_release_qualifiers'])
  pairs.append(dict(scenario=scenario,motion_mode=motion,seed=seed,paired=bool(prefix['exact'] and exact and config_equal),prefix=prefix,entry_exact=exact,offset_enabled=records[b]['offset_enabled'],whole_peak_delta_N=max(records[b]['safety']['finger_peak_N'])-max(records[a]['safety']['finger_peak_N']),local_excess_delta_Ns=records[b]['safety']['local_excess_integral_Ns']-records[a]['safety']['local_excess_integral_Ns']))
 clusters=[]
 for seed in [617901,617902]:
  mm=[m for m in metas.values() if m['seed']==seed];mass={m['config']['mass'] for m in mm};friction={m['config']['friction'] for m in mm};clusters.append(dict(seed=seed,assigned=16,masses_kg=sorted(mass),initial_frictions=sorted(friction)));check(str(seed)+'/one_actual_physical_draw_shared_across_contexts',len(mass)==len(friction)==1)
 check('no_physics_api_import','mujoco' not in sys.modules)
 check('frozen_source_originals_still_identical_after_audit',all(sha(Path(snap['files'][e['key']]['original_path']))==e['expected'] for e in source_checks))
 report=dict(schema='release_primitive_prospective_v2_independent_v1',created_utc=datetime.now(timezone.utc).isoformat(),reviewer_role='Independent prospective-run saved-data arithmetic/pair/source review; priorV3 measurement-wrapper author; this prospective runner and author readout are not executed as evidence',passed=not issues,checks=checks,issues=issues,source_checks=source_checks,records=records,pairs=pairs,physical_clusters=clusters,all_assignments=32,independent_physical_clusters=2,bridge_scientific_samples=0,final_bridge_sha256=sha(BRIDGE),invalid_old_bridge_incident_sha256=sha(ROOT/'review/release_primitive_bridge_source_revision_incident_v1.json'),input_sha256={n:sha(RUN/n) for n in ['summary.json','execution_complete.json','manifest.json','plan.json','results.json','results_partial.json','assignment_ledger.json','source_snapshot_manifest.json','seed_inventory.json']},checker_sha256=sha(Path(__file__)),arithmetic_helper_sha256=sha(PURE),clearance_helper_sha256=sha(RESOLUTION),physics_steps=0,production_or_frozen_source_modified=False,model_admitted=False,material_safety_certified=False,scope='Full assigned denominator and complete rows raw/V3/source316/pair16/entry96/native21/full22zero; raw start/end vs200Hz deadline/terminal distinct. Partial failures retained unavailable; no new-seed independent native full16 reference invented.',limitations=['32 assignments are16 paired contexts but only2 shared physical/noise clusters; no population p-value or post-hoc composite weight.', 'Clear <=.2N is a saved numerical low-force condition, not geometric gap or between-step/permanent-separation proof. Raw cached force and200Hz extra-forward queries are different.', 'RNG/queue/observer/preview state and full prefix are checked at all3 saved boundaries. This is not a new clone/restore rollout or exhaustive post-origin random-packet replay.', 'Final old-seed bridge proves software replay for that context only. Fresh-seed zeros are native shadow-control oracles, not16 additional independent native physical reruns.', 'Release-component differences do not establish grasp/controller/full-fusion benefit. Before-release task/holding values must agree within each pair.'])
 with OUT.open('x',encoding='utf-8') as out:json.dump(report,out,ensure_ascii=False,indent=2,allow_nan=False)
 with OUT.with_suffix('.md').open('x',encoding='utf-8') as out:out.write('# 前瞻释放组件 V2：独立终态核验\n\n'+f'检查{sum(c["passed"] for c in checks)}/{len(checks)}；问题{len(issues)}。32分配、16对、2个共享物理种子cluster；桥接不作科学样本。\n\n'+f'JSON SHA256 `{sha(OUT)}`。\n\n无物理重跑或冻结源修改；逐项收益和代价保持分开，不给事后权重或模型/安全认证。\n')
 print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),checks=len(checks),issues=len(issues),passed=not issues,physics_steps=0)))

if __name__=='__main__':
 parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--describe',action='store_true');mode.add_argument('--execute',action='store_true');args=parser.parse_args()
 if args.describe:print(json.dumps(dict(schema='release_primitive_prospective_v2_independent_v1',terminal_gate=True,scope=['316sources','32assignments','16pairs','96entries','all fullraw/V3 arithmetic','actual entry/quarter/zero shadow checks','raw start/end vs200Hz deadline/first/final clearance'],physics_steps=0),ensure_ascii=False))
 else:main()

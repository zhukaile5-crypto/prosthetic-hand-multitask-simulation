"""Independent saved-data calculations only; imports no project code or physics API."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import base64, hashlib, json, sys, xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/release_basis_minimal_v2'
OUT=ROOT/'review/release_basis_v2_independent_audit_v1.json'
MD=OUT.with_suffix('.md')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
def encode(v):
 if isinstance(v,np.ndarray):return dict(array_dtype=v.dtype.str,descr=v.dtype.descr if v.dtype.fields else None,shape=list(v.shape),bytes_b64=base64.b64encode(np.ascontiguousarray(v).tobytes()).decode('ascii'))
 if v is None or isinstance(v,(str,bool,int)):return v
 if isinstance(v,(float,np.floating)):return dict(float_hex=float(v).hex())
 if isinstance(v,np.integer):return int(v)
 if isinstance(v,dict):return dict(mapping_type='builtins.dict',mapping_items=sorted([[encode(k),encode(x)] for k,x in v.items()],key=lambda z:json.dumps(z[0],ensure_ascii=False,sort_keys=True)))
 raise TypeError(type(v))
WALL={'qp_seconds','control_ms','wall_seconds'}
def strip_wall(v):
 if isinstance(v,dict):
  r={k:strip_wall(x) for k,x in v.items()}
  if 'mapping_items' in r:r['mapping_items']=[p for p in r['mapping_items'] if not isinstance(p[0],str) or p[0] not in WALL]
  return r
 if isinstance(v,list):return [strip_wall(x) for x in v]
 return v
def mapping(v):return dict(v.get('state',v)['mapping_items'])
def at(v,*keys):
 for k in keys:v=mapping(v)[k]
 return v
def dwell(t,flag,start,end,seconds,exclusive=False):
 ids=np.flatnonzero((t>=start-1e-9)&(t<end-1e-9 if exclusive else t<=end+1e-9));start_i=None
 for i in ids:
  if not flag[i]:start_i=None;continue
  if start_i is None:start_i=i
  if t[i]-t[start_i]>=seconds-1e-9:return float(t[i])
 return None
checks=[];issues=[]
def check(name,ok,detail=None):
 x=dict(name=name,passed=bool(ok),detail=detail);checks.append(x)
 if not ok:issues.append(x)
def same(a,b):
 if a is None or b is None:return a is b
 if isinstance(a,(bool,str)) or isinstance(b,(bool,str)):return a==b
 return bool(np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True))
def metric(name,actual,reported):check(name,same(actual,reported),None if same(actual,reported) else dict(actual=actual,reported=reported))
def variation(t,f):
 dur=float(t[-1]-t[0]);tv=abs(np.diff(f,axis=0)).sum(axis=0);drift=abs(f[-1]-f[0]);reversal=np.maximum(0.,(tv-drift)/2)
 total=f.sum(axis=1);ttv=float(abs(np.diff(total)).sum());td=float(abs(total[-1]-total[0]))
 return dict(duration_s=dur,samples=len(t),finger_TV_N=float(tv.sum()),finger_net_drift_N=float(drift.sum()),finger_reversal_N=float(reversal.sum()),finger_TV_rate_N_s=float(tv.sum()/dur),finger_net_drift_rate_N_s=float(drift.sum()/dur),finger_reversal_rate_N_s=float(reversal.sum()/dur),total_force_TV_N=ttv,total_force_net_drift_N=td,total_force_reversal_N=max(0.,(ttv-td)/2),total_force_TV_rate_N_s=ttv/dur,finger_SD_mean_N=float(f.std(axis=0).mean()),total_force_SD_N=float(total.std()),path_identity_residual_N=float(abs(tv.sum()-drift.sum()-2*reversal.sum())))
def clearance_detail(t,f):
 ids=np.flatnonzero(t>=12-1e-9);clear=f<=.2;first=next((int(i) for i in ids if clear[i]),None)
 starts=[]
 if first is not None:
  for i in ids[ids>first]:
   if not clear[i] and clear[i-1]:starts.append(float(t[i]))
 bad=ids[~clear[ids]];last=None if len(bad) and bad[-1]==ids[-1] else float(t[bad[-1]+1]) if len(bad) else 12.
 confirmed=dwell(t,clear,12,15.5,.2)
 return dict(first_clear_s=None if first is None else float(t[first]),first_200ms_dwell_confirm_s=confirmed,final_unbroken_clear_start_s=last,recontact_episode_starts_s=starts,recontacts_after_first_dwell_confirm_s=[] if confirmed is None else [v for v in starts if v>confirmed+1e-9],final_all_hand_contact_N=float(f[-1]))

summary=read(RUN/'summary.json');manifest=read(RUN/'manifest.json')
if summary.get('execution_complete') is not True:raise RuntimeError('Terminal required')
FIRST=OUT.with_name(OUT.stem+'_first_pass.json')
FIRST_MD=MD.with_name(MD.stem+'_first_pass.md')
if OUT.exists() and (not FIRST.exists() or sha(OUT)!=sha(FIRST)):raise FileExistsError('Preserve reviewed independent evidence')
if MD.exists() and (not FIRST_MD.exists() or sha(MD)!=sha(FIRST_MD)):raise FileExistsError('Preserve reviewed independent note')
jobs=manifest['jobs'];byid={x['job_id']:x for x in summary['results']}
expected={f'{s}__{p}' for s in ['capsule','box','sphere'] for p in ['native2d','fixed_height']}
check('all_six_planned_attempts_retained',len(jobs)==len(byid)==6 and {x['job_id'] for x in jobs}==set(byid)==expected and summary['completed']==6 and summary['planned']==6 and summary['failures']==[] and all(x['status']=='complete' for x in byid.values()))
lock=read(ROOT/'identification/release_basis_v2/dependency_lock.json')
check('manifest_plan_and_lock_bindings',manifest['plan']==read(RUN/'sources/research/identification/release_basis_v2/plan.json') and manifest['sources']==lock['sources'] and all(x['lock_sha256']==sha(ROOT/'identification/release_basis_v2/dependency_lock.json') for x in jobs))
site=Path(manifest['environment']['executable']).parent.parent/'Lib/site-packages'
foundation=ROOT.parent/'五指策略接入_2026-11-03'
source_checks=[]
for key,h in manifest['sources'].items():
 if key.startswith('research/'):live=ROOT/key.removeprefix('research/')
 elif key.startswith('foundation/'):live=foundation/key.removeprefix('foundation/')
 elif key.startswith('external/mujoco/'):live=site/'mujoco'/key.removeprefix('external/mujoco/')
 elif key.startswith('external/mujoco_distribution/'):live=site/key.removeprefix('external/mujoco_distribution/')
 else:raise ValueError(key)
 ss=sha(RUN/'sources'/key);current=sha(live)
 source_checks.append(dict(key=key,expected=h,snapshot=ss,current=current,original_path=str(live),matches=h==ss==current))
check('all230_frozen_snapshot_current_sources',len(source_checks)==230 and all(x['matches'] for x in source_checks))
check('source_directory_exact_set',{p.relative_to(RUN/'sources').as_posix() for p in (RUN/'sources').rglob('*') if p.is_file()}==set(manifest['sources']))
xml=ET.parse(RUN/'sources/foundation/assets/legacy_model.xml').getroot()
actuator_names=[a.get('name') for a in xml.find('actuator')]
axes=tuple('rh_A_'+n for n in ('THJ5','THJ4','THJ3','THJ2','THJ1','FFJ4','FFJ3','FFJ0','MFJ4','MFJ3','MFJ0','RFJ4','RFJ3','RFJ0','LFJ5','LFJ4','LFJ3','LFJ0'))+('lift_reference_m','WRJ2_reference_rad','WRJ1_reference_rad')
aids=[actuator_names.index(x) for x in axes[:18]];wrists=[actuator_names.index(x) for x in ['rh_A_WRJ2','rh_A_WRJ1']];lift_id=actuator_names.index('lift_motor')
records={};entry_sides={};entry_arrays={}
for job in jobs:
 jid=job['job_id'];d=Path(job['directory']);reported=read(d/'result.json');r=reported['research_metrics'];m=read(d/'record/metadata.json');pilot=read(d/'pilot_metadata.json')
 trace=d/'record/trace.npz';thash=sha(trace)
 check(jid+':trace_result_metadata_linkage',thash==reported['trace_sha256']==r['trace_sha256']==m['trace_sha256']==byid[jid]['trace_sha256'] and pilot['command_record_sha256']==sha(d/'decoder_commands.npz') and r['evaluation_source_sha256']==manifest['sources']['research/evaluation.py'] and r['protocol_sha256']==manifest['sources']['research/protocol.json'])
 check(jid+':actual_registered_standard_config',m['shape']==job['shape'] and m['method']=='ratio' and m['seed']==610120 and m['split']=='development' and m['scenario']['name']=='nominal' and m['config']==pilot['actual_config']==r['physical_parameters'] and m['config']['mass']==.12 and m['config']['friction']==.6 and m['config']['timestep']==.0005 and m['config']['control_dt']==.005)
 with np.load(trace,allow_pickle=False) as z:
  needed=['time','normal','skin_normal','palm','support','position','relative','relative_velocity','angular_velocity','lift','lift_ref','all_hand_object','controller_active','warning_count','actual_motion_target','executed_motion_reference','wrist_position','requested_wrist_reference','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm','substep_all_hand_object']
  x={k:z[k] for k in needed}
 t=x['time'];st=x['substep_time'];dur=x['substep_duration'];f=x['substep_skin_normal'];skin=x['skin_normal'];n=x['normal'];limits=np.array(m['limits']);budget=m['budget']
 complete=len(t)==3201 and np.allclose(t,np.arange(3201)*.005,rtol=0,atol=1e-8) and len(st)==32000 and np.allclose(st,np.arange(32000)*.0005,rtol=0,atol=1e-7) and np.allclose(dur,.0005,rtol=0,atol=1e-9) and abs(dur.sum()-16)<1e-8 and all(np.isfinite(v).all() for v in x.values())
 check(jid+':full16s_truth_and_substep_coverage',complete and r['record_complete'] and reported['record_complete'])
 local=np.maximum(f-limits,0);total=np.maximum(f.sum(axis=1)-budget,0);safe=(skin<=limits+.02).all(axis=1)&(skin.sum(axis=1)<=budget+.02)
 guard=bool(complete and (f<=limits+.02).all() and (f.sum(axis=1)<=budget+.02).all() and safe[-1]);peak=np.maximum(f.max(axis=0),skin[-1])
 safety=dict(guard_pass=guard,finger_peak_N=peak.tolist(),local_excess_integral_Ns=float(np.sum(local*dur[:,None])),total_excess_integral_Ns=float(total@dur),palm_peak_N=float(max(x['substep_palm'].max(),x['palm'][-1])))
 for k,v in safety.items():metric(jid+':V3_safety.'+k,v,r['safety'][k])
 metric(jid+':canonical_forceguard',guard,reported['safety']['force_protection_pass']);metric(jid+':summary_forceguard',guard,byid[jid]['guard_pass']);metric(jid+':summary_peak',peak.tolist(),byid[jid]['finger_peak'])
 for stage,mask in [('shared_preparation',st<7.5-1e-9),('policy_holding_transport',(st>=7.5-1e-9)&(st<12-1e-9)),('mechanical_release_and_after',st>=12-1e-9)]:
  values=dict(duration_s=float(dur[mask].sum()),finger_peak_N=f[mask].max(axis=0).tolist(),local_excess_integral_Ns=float(np.sum(local[mask]*dur[mask,None])),total_excess_integral_Ns=float(total[mask]@dur[mask]))
  for k,v in values.items():metric(jid+':V3_stage.'+stage+'.'+k,v,r['safety']['stages'][stage][k])
 contact=n>.2;support=contact[:,0]&(contact.sum(axis=1)>=3);raised=x['position'][:,2]>=m['initial_position'][2]+.025
 lift_error=abs(x['lift']-x['lift_ref']);motion=(lift_error<=.01)&(x['support']<=.05);relerr=np.linalg.norm(x['relative']-m['reference_relative'],axis=1)
 established=dwell(t,support&raised&motion,0,7.5,.1);lost=dwell(t,~support|~raised|(relerr>.08),7.5,12,.1,True);trackfail=dwell(t,lift_error>.01,7.5,12,.3,True)
 exposure=bool(support[2400] and raised[2400] and x['support'][2400]<=.05 and x['all_hand_object'][2400]>.2)
 clear=dwell(t,x['all_hand_object']<=.2,12,15.5,.2);release=bool(exposure and clear is not None)
 release_values=dict(scheduled=True,actual_command_at=12.,physical_exposure=exposure,physical_success=release,task_qualified_exposure=bool(exposure and m['handoff_passed'] and lost is None and trackfail is None),task_qualified_success=bool(release and m['handoff_passed'] and lost is None and trackfail is None),physical_clearance_latency_s=clear-12 if release else None)
 for k,v in release_values.items():metric(jid+':V3_release.'+k,v,r['release'][k])
 metric(jid+':canonical_release_latency',clear-12 if release else None,reported['release_latency'])
 mission=bool(complete and m['handoff_passed'] and established is not None and lost is None and trackfail is None and release and x['warning_count'].max()==0)
 wrist_bad=np.any(abs(x['wrist_position']-x['requested_wrist_reference'])>.015,axis=1);wfail=dwell(t,wrist_bad,7.5,12,.3,True)
 metric(jid+':legacy_mission',mission,reported['task_success']);metric(jid+':V3_mission',mission,r['mission_success']);metric(jid+':V3_full_motion_mission',mission and wfail is None,r['full_motion_mission_success'])
 stable=support&raised&motion&safe&(relerr<=.005)&(np.linalg.norm(x['relative_velocity'],axis=1)<=.005)&(np.linalg.norm(x['angular_velocity'],axis=1)<=.15)
 first_stable=dwell(t,stable,7.5,8.9,.2);metric(jid+':initial_stability',1.4 if first_stable is None else first_stable-7.5,r['initial_stability_wait_s'])
 # Independent fixed V3 stage FIR; no project evaluator is imported or called.
 dt=.0005;h=50;kt=np.arange(-h,h+1)*dt;kernel=2*50*dt*np.sinc(2*50*kt)*np.hanning(101);kernel/=kernel.sum()
 grid=np.r_[0.,st+dur];physical=np.vstack((n[0],x['substep_normal']));band=np.full_like(n,np.nan)
 for lo,hi in [(0,7.5),(7.5,12),(12,16)]:
  mask=(grid>=lo-1e-9)&(grid<=hi+1e-9);sample=(t>=lo-1e-9)&(t<=hi+1e-9)
  if lo==12:sample&=t>12+1e-9
  for digit in range(5):band[sample,digit]=np.interp(t[sample],grid[mask],np.convolve(np.pad(physical[mask,digit],(h,h),mode='reflect'),kernel,mode='valid'))
 check(jid+':V3_filter_contract',r['filter_contract']['version']==3 and r['filter_contract']['phase_boundaries_s']==[0.,7.5,12.,16.] and r['filter_contract']['raw_safety_filtered'] is False)
 for name,lo,hi in [('static_quality',8.2,8.9),('dynamic_quality',9.3,11.7),('entire_holding_quality',7.5,12.)]:
  mask=(t>=lo-1e-9)&(t<=hi+1e-9);valid=bool(complete and np.all(support[mask]&raised[mask]&(x['support'][mask]<=.05)&x['controller_active'][mask]))
  metric(jid+':'+name+'.eligible',valid,r[name]['eligible'])
  if valid:
   metric(jid+':'+name+'.mean_actual_force',float(n[mask].sum(axis=1).mean()),r[name]['mean_actual_finger_force_N'])
   for signal,values in [('bandlimited',band),('raw_200Hz',n)]:
    for k,v in variation(t[mask],values[mask]).items():metric(jid+':'+name+'.'+signal+'.'+k,v,r[name][signal][k])
 for pname,lo,hi in [('holding_motion_tracking',7.5,12.),('dynamic_motion_tracking',9.3,11.7)]:
  weights=np.maximum(0,np.minimum(np.r_[t[1:],t[-1]],hi)-np.maximum(t,lo));error=np.column_stack((x['lift'],x['wrist_position']))-x['actual_motion_target'];mask=weights>0;seconds=weights.sum()
  values=dict(rmse=np.sqrt(np.sum(error**2*weights[:,None],axis=0)/seconds).tolist(),max_abs_error=abs(error[mask]).max(axis=0).tolist(),absolute_error_integral=np.sum(abs(error)*weights[:,None],axis=0).tolist(),duration_s=float(seconds))
  for k,v in values.items():metric(jid+':V3_progress.'+pname+'.'+k,v,r['progress'][pname][k])
 with np.load(d/'decoder_commands.npz',allow_pickle=False) as z:commands={k:z[k] for k in z.files}
 a=commands['actual_command21'];b=commands['native_command21'];ac=commands['actual_ctrl'];bc=commands['native_ctrl'];different=np.flatnonzero(np.any(ac!=bc,axis=0)).tolist()
 check(jid+':actual21_axes_and_decoder_complete',tuple(commands['command_axes'])==axes and len(a)==8801 and ac.shape==bc.shape==(8801,22) and np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(ac).all() and np.isfinite(bc).all())
 check(jid+':finger_wrist_exact_before_every_integration',np.array_equal(a[:,np.r_[0:18,19:21]],b[:,np.r_[0:18,19:21]]) and np.array_equal(ac[:,aids+wrists],bc[:,aids+wrists]) and np.array_equal(a[:,:18],ac[:,aids]) and np.array_equal(a[:,19:],ac[:,wrists]))
 check(jid+':registered_branch_only',np.array_equal(ac,bc) if job['path']=='native2d' else different==[lift_id] and np.ptp(a[:,18])==0.)
 decoder_check=read(d/'decoder_validation.json');check(jid+':decoder_summary_matches_raw',decoder_check==byid[jid]['decoder'] and decoder_check['applications']==len(a) and decoder_check['mismatches']==0)
 es=read(d/'entry_side_state.json');entry_sides[jid]=es
 with np.load(d/'entry_physics.npz',allow_pickle=False) as z:ea={k:z[k].item() if z[k].shape==() else z[k] for k in z.files}
 entry_arrays[jid]=ea;signature=read(d/'entry_signature.json')
 check(jid+':entry_actual_signatures_recomputed',signature==byid[jid]['entry']==pilot['entry'] and signature['integration_state_sha256']==digest(encode(ea['integration_state'])) and signature['physics_arrays_sha256']==digest(encode(ea)) and signature['python_causal_side_state_sha256']==digest(strip_wall(es)) and ea['time']<12.00000001 and ea['time']>11.99999999)
 rng=at(es,'sensors','rng');queue=at(es,'sensors','queue');hist=at(es,'controller','state','action_history');fk=at(es,'skin_geometry','fk_data');fksamples=at(es,'skin_geometry','samples')
 check(jid+':populated_rng_queue_observer_fk_retained',rng.get('numpy_generator')=='PCG64' and len(queue['items'])==102 and len(hist['items'])==140 and len(fksamples['items'])==150 and {'integration_state','qpos','qvel','ctrl','qacc_warmstart'}<=set(mapping(fk)))
 refs=read(d/'entry_code_references.json');check(jid+':mujoco_api_handle_pinned_not_namespace',len(refs)==1 and refs[0]['object_path']=='$.skin_geometry.mj' and refs[0]['origin_sha256']==manifest['sources'][refs[0]['origin_source_key']] and refs[0]['source_lock_verified'] and refs[0]['namespace_not_serialized'] and all(manifest['sources'][k]==h for k,h in refs[0]['covered_native_binding_sha256'].items()))
 event=clearance_detail(t,x['all_hand_object']);raw_event=clearance_detail(st+dur,x['substep_all_hand_object'])
 records[jid]=dict(shape=job['shape'],path=job['path'],safety=safety,release=release_values,mission_success=mission,full_motion_mission_success=mission and wfail is None,clearance_200Hz=event,clearance_2000Hz=raw_event,decoder=dict(applications=len(a),full_ctrl_native_exact=bool(np.array_equal(ac,bc)),different_ctrl_columns=different,finger_wrist_exact=True),entry_population=dict(sensor_queue=len(queue['items']),observer_action_history=len(hist['items']),fk_history=len(fksamples['items']),rng='PCG64'),input_sha256={str(p.relative_to(RUN)):sha(p) for p in [trace,d/'result.json',d/'record/metadata.json',d/'decoder_commands.npz',d/'entry_physics.npz',d/'entry_side_state.json',d/'entry_signature.json',d/'pilot_metadata.json',d/'prefix_signature.json']})

pairs=[]
for shape in ['capsule','box','sphere']:
 aid=f'{shape}__native2d';bid=f'{shape}__fixed_height';a=RUN/'runs'/aid;b=RUN/'runs'/bid
 ea,eb=entry_arrays[aid],entry_arrays[bid]
 physics={k:bool(np.array_equal(ea[k],eb[k],equal_nan=True)) for k in ea}
 sa,sb=strip_wall(entry_sides[aid]),strip_wall(entry_sides[bid])
 groups={k:at(sa,k)==at(sb,k) for k in ['controller','sensors','preparation','skin_geometry','session_causal_state','plant_side']}
 check(shape+':full_persisted_entry_pair_exact',ea.keys()==eb.keys() and all(physics.values()) and sa==sb and all(groups.values()))
 fields=[];mismatches=[];producer_prefix={}
 with np.load(a/'record/trace.npz',allow_pickle=False) as za,np.load(b/'record/trace.npz',allow_pickle=False) as zb:
  ta=za['time'];sta=za['substep_time'];control=ta<12-1e-9;sub=sta<12-1e-9
  check(shape+':prefix_same_field_set',set(za.files)==set(zb.files))
  for k in za.files:
   if k in ['control_ms','metadata']:continue
   va,vb=za[k],zb[k]
   mask=sub if k.startswith('substep_') else control
   if va.ndim==0:continue
   va,vb=va[mask],vb[mask]
   if k=='debug_json':ok=all(strip_wall(json.loads(str(u)))==strip_wall(json.loads(str(v))) for u,v in zip(va,vb))
   elif va.dtype.kind in 'biufc':ok=va.dtype==vb.dtype and va.shape==vb.shape and va.tobytes()==vb.tobytes()
   else:ok=np.array_equal(va,vb)
   fields.append(k)
   if not ok:mismatches.append(k)
   if k in read(a/'prefix_signature.json'):producer_prefix[k]=digest(encode(va))
  post=ta>=12-1e-9
  post_alpha_exact=np.array_equal(za['alpha'][post],zb['alpha'][post])
  post_ctrl_nonlift_exact=np.array_equal(za['ctrl'][post][:,[i for i in range(22) if i!=lift_id]],zb['ctrl'][post][:,[i for i in range(22) if i!=lift_id]])
  post_qpos_max=abs(za['qpos'][post]-zb['qpos'][post]).max(axis=0).tolist()
 check(shape+':all_commands_observations_truth_full_pre12_exact',not mismatches,dict(fields=len(fields),mismatches=mismatches))
 check(shape+':producer_prefix_signatures_recomputed',producer_prefix==read(a/'prefix_signature.json')==read(b/'prefix_signature.json'))
 check(shape+':post12_alpha_nonlift_reference_exact',post_alpha_exact and post_ctrl_nonlift_exact)
 with np.load(a/'decoder_commands.npz',allow_pickle=False) as za,np.load(b/'decoder_commands.npz',allow_pickle=False) as zb:
  check(shape+':all8801_paired_decoder_clocks_finger_wrist_exact',all(np.array_equal(za[k],zb[k]) for k in ['control_time','physical_time','path_time']) and np.array_equal(za['actual_command21'][:,np.r_[0:18,19:21]],zb['actual_command21'][:,np.r_[0:18,19:21]]))
 pairs.append(dict(shape=shape,entry_physics_fields_exact=physics,serialized_side_groups_exact=groups,pre12_fields=fields,pre12_mismatches=mismatches,post12_alpha_exact=bool(post_alpha_exact),post12_nonlift_ctrl_exact=bool(post_ctrl_nonlift_exact),post12_qpos_max_abs_difference=post_qpos_max,native=records[aid],fixed_height=records[bid]))

old=ROOT/'output/release_basis_minimal_v1';v1=read(old/'summary.json');v1rows=[]
for row in v1['results']:
 d=old/'runs'/row['job_id'];physics_file=d/'entry_physics.npz'
 with np.load(physics_file,allow_pickle=False) as z:entrytime=float(z['integration_state'][0])
 with np.load(d/'decoder_commands.npz',allow_pickle=False) as z:decoder_rows=len(z['control_time'])
 v1rows.append(dict(job_id=row['job_id'],status=row['status'],mappingproxy_error='mappingproxy' in row.get('error',''),partial_save_fileexists='FileExistsError' in row.get('partial_save_error',''),entry_physics_exists=True,entry_physical_time=entrytime,entry_side_exists=(d/'entry_side_state.json').exists(),full_trace_exists=(d/'trace.npz').exists() or (d/'record/trace.npz').exists(),decoder_record_exists=(d/'decoder_commands.npz').exists(),decoder_release_rows=decoder_rows))
check('v1_six_failures_separately_retained_missing_scope_verified',v1['planned']==6 and v1['completed']==0 and len(v1rows)==6 and all(x['status']=='failed' and x['mappingproxy_error'] and x['partial_save_fileexists'] and abs(x['entry_physical_time']-12)<1e-8 and not x['entry_side_exists'] and not x['full_trace_exists'] and x['decoder_release_rows']==0 for x in v1rows))
report=dict(schema='release_basis_v2_independent_audit_v1',created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,issues=issues,inputs=dict(summary_sha256=sha(RUN/'summary.json'),manifest_sha256=sha(RUN/'manifest.json'),dependency_lock_sha256=sha(ROOT/'identification/release_basis_v2/dependency_lock.json'),checker_sha256=sha(Path(__file__))),source_checks=source_checks,records=records,pairs=pairs,v1_failure_scope=dict(summary_sha256=sha(old/'summary.json'),attempts=v1rows,interpretation='Six software serialization stops before release, not six release-physics failures. Entry physics exists; Python/RNG side and full0-12 raw prefix were not retained; no reconstruction substitutes for missing evidence.'),scope=dict(full=['6 full16s traces:192000 physical substeps and19206 control endpoints, raw forceguard, stage exposure, actual release/exposure/mission, first/final clearance and recontact','independent V3 phase FIR and all numeric quality/decomposition fields in3 fixed windows per trial; actual requested-motion tracking','3 paired persisted complete integration/physics and encoded side states incl PCG64, queue102, observer140 and FK150 histories','all retained pre12 trace fields except metadata/control_ms; debug removes only declared wallclock keys','52806 decoder applications: native full22ctrl; fixed-height18finger+2wrist and lift-only divergence; actual post12 numeric command/alpha pairing','230 snapshot and current implementation/robot/asset/dependency byte identities','all6 v1 failure records and missing-side/prefix scope'],not_checked=['no new physics or solver/convergence experiment','no restart/restore rollout from persisted entry; paired runs are independent deterministic preparation with equal saved state','no generic21D pressure response/material-safety certificate or arbitrary object/domain transfer','palm/other contact recorded without invented protection caps']),physics_steps=0,production_or_original_evidence_modified=False,admission=dict(release_model_admitted=False,performance_superiority_admitted=False,scope='standard3shape,0.12kg,mu0.6,seen development seed610120; two fixed path coordinates, not a generic release policy'),interpretation='Lift-only treatment changes measured release contact force in opposite directions across shapes: capsule/sphere improve guard while box worsens. All6 physically clear. Do not aggregate as uniformly superior or join this local pilot into V9 holding-controller outcomes.')
report['audit_revision']=dict(first_pass_evidence_sha256=sha(FIRST),first_pass_source_sha256=sha(Path(__file__).with_name('verify_release_basis_v2_independent_first_pass.py')),correction='Initial independent checker assumed v1 decoder archive was absent. Actual v1 archives exist with zero rows. Corrected validation uses zero release applications; first-pass source/evidence remain preserved. No production or retained run changed.')
report['frozen_source_evidence']=[]
for relative,needle,reason in [
 ('research/identification/release_basis_v2/run_pilot.py','def make_session(', 'Real standard ratio/nominal preparation and independent per-path plant construction'),
 ('research/identification/release_basis_v2/run_pilot.py','def entry_snapshot(', 'Saved real plant integration and encoded mutable causal side state before first release'),
 ('research/identification/release_basis_v2/run_pilot.py','if s.plant.release_started is not None:raise', 'Entry is checked before the first release command'),
 ('research/identification/release_basis_v2/decoder.py','native=self.data.ctrl.copy()', 'Native decoder evaluated first before recorded candidate can be integrated'),
 ('research/identification/release_basis_v2/decoder.py','identical=np.array_equal(candidate,native)', 'A checks fullctrl; B checks the real finger/wrist authority'),
 ('research/identification/release_basis_v2/state_snapshot.py','def side_state(', 'Declared session/controller/sensors/preparation/plant side and FK MjData serialization scope'),
 ('research/identification/release_basis_v2/state_snapshot.py',"if isinstance(value,np.random.Generator)", 'Real RNG bit-generator state encoded rather than substituted by an API module handle'),
 ('research/evaluation.py','physical_released=bool(', 'Physical release uses first clearance dwell and can be followed by recontact'),
 ('research/evaluation.py','safety=dict(guard_pass=', 'Unfiltered entire-substep and final-endpoint protection gate'),
 ('research/evaluation.py','if lo==release and lo>0.', 'V3 retains the pre-action release endpoint in holding FIR')]:
 p=RUN/'sources'/relative;ls=p.read_text(encoding='utf-8-sig').splitlines();line=next(i for i,s in enumerate(ls,1) if needle in s)
 report['frozen_source_evidence'].append(dict(absolute_file=str(p),line=line,sha256=manifest['sources'][relative],reason=reason))
with OUT.open('w' if OUT.exists() else 'x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
lines=['# Release basis V2 独立只读审计','',f'检查 {sum(x["passed"] for x in checks)}/{len(checks)}；问题 {len(issues)}。完整6条，230个来源，未运行新物理。','', '|形状|原生峰值 N / 力限|固定高度峰值 N / 力限|真实释放确认延迟 原生→固定高度 s|','|---|---|---|---|']
for s in ['capsule','sphere','box']:
 a=records[s+'__native2d'];b=records[s+'__fixed_height'];lines.append(f'|{s}|{max(a["safety"]["finger_peak_N"]):.9f} / {a["safety"]["guard_pass"]}|{max(b["safety"]["finger_peak_N"]):.9f} / {b["safety"]["guard_pass"]}|{a["release"]["physical_clearance_latency_s"]:.3f} → {b["release"]["physical_clearance_latency_s"]:.3f}|')
lines+=['','3组保存的完整入口物理数组和声明的 Python 因果状态逐值相同，含传感器 PCG64 状态、102队列、140条观察器动作历史及150条几何历史/FK积分状态。全部 pre12 真值、命令与观测相同；每条8801个解码应用中 A 的22个控制量与原生完全相同，B 只有 lift 控制列20不同；18手指与2腕参考及 alpha 相同。接触变化可造成真实 qpos 分歧，不要求释放后 qpos 相同。','', '完整16s原始力限未滤波；独立重算V3分阶段 FIR、固定品质窗、请求对真实编码器的跟踪、真实释放暴露及200ms清除确认。JSON另列首清除、最终持续清除和再次接触；声明范围不等于所有皮肤均有材料阈值。sphere原生路径在12.570s完成首次200ms清除确认，却在12.610s再次接触；200Hz最后持续清除始于12.615s，2000Hz原始子步始于约12.639s。现有V3成功定义允许首次确认后的再次接触，因此不能把该延迟当成此后永不接触的时间；六条最终都为0N。','', '固定高度对 capsule/sphere 改善，却使 box 从力限通过变为失败；六条均真实释放，不可把不同方向合并宣称普遍更优。仅标准3形状、0.12kg、摩擦0.6、单个已见开发seed610120；没有通用21维响应模型或材料安全认证。','', '旧V1六条均在12s入口序列化处软件停止。保留物理入口与失败账本，decoder档存在但均0行；Python/RNG side和完整0–12s raw prefix缺失，不计为六次释放物理失败，不与V2混合。','',f'证据：`{OUT.name}`；SHA256 `{sha(OUT)}`。']
MD.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),md=str(MD),checks=len(checks),passed=sum(x['passed'] for x in checks),issues=issues),ensure_ascii=False),flush=True)

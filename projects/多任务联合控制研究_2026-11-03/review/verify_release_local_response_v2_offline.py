"""Independent arithmetic on retained seven-run data; no producer imports.

The reviewer also implemented the pilot. This is explicitly author-side
verification with a separate earlier raw-audit implementation, not a claim of
organizational independence or model admission.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
import ast,base64,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'output/release_local_response_capsule_v2';OUT=ROOT/'review/release_local_response_v2_offline_audit_v1.json';PURE=ROOT/'review/verify_release_basis_v2_independent.py'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
WALL={'qp_seconds','control_ms','wall_seconds'}
tree=ast.parse(PURE.read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'sha','digest','encode','strip_wall','mapping','at','dwell','variation','clearance_detail'}]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(PURE),'exec'))
checks=[];issues=[]
def check(name,value,detail=None):
 row=dict(name=name,passed=bool(value),detail=detail);checks.append(row)
 if not value:issues.append(row)
def metric(name,a,b):
 ok=a is b if a is None or b is None else a==b if isinstance(a,(bool,str)) or isinstance(b,(bool,str)) else bool(np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True));check(name,ok,None if ok else dict(actual=a,reported=b))
# Reuse the previous audit's raw arithmetic block only, not its top-level IO,
# application evaluator, snapshot encoder or physical execution paths.
jobloop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='job' and isinstance(n.iter,ast.Name) and n.iter.id=='jobs')
start=next(i for i,n in enumerate(jobloop.body) if isinstance(n,ast.With) and 'np.load(trace' in ast.unparse(n));stop=next(i for i,n in enumerate(jobloop.body) if isinstance(n,ast.With) and 'decoder_commands.npz' in ast.unparse(n))
body=[]
for n in jobloop.body[start:stop]:
 if isinstance(n,ast.Expr) and any(isinstance(v,ast.Constant) and isinstance(v.value,str) and 'summary_' in v.value for v in ast.walk(n)):continue
 body.append(deepcopy(n))
body.append(ast.Return(value=ast.Dict(keys=[ast.Constant(k) for k in ['safety','release','mission','full_motion','raw','time']],values=[ast.Name(id='safety',ctx=ast.Load()),ast.Name(id='release_values',ctx=ast.Load()),ast.Name(id='mission',ctx=ast.Load()),ast.BoolOp(op=ast.And(),values=[ast.Name(id='mission',ctx=ast.Load()),ast.Compare(left=ast.Name(id='wfail',ctx=ast.Load()),ops=[ast.Is()],comparators=[ast.Constant(None)])]),ast.Name(id='x',ctx=ast.Load()),ast.Name(id='t',ctx=ast.Load())])))
func=ast.FunctionDef(name='raw_check',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=k) for k in ['jid','d','reported','r','m','trace']],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[],type_params=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),str(PURE)+'::raw-arithmetic','exec'))
summary=read(RUN/'summary.json')
if summary.get('execution_complete') is not True:raise RuntimeError('Terminal required before matrix audit')
if OUT.exists():raise FileExistsError(OUT)
manifest=read(RUN/'manifest.json');snap=read(RUN/'source_snapshot_manifest.json');plan=read(RUN/'plan.json');rows=read(RUN/'results.json');partial=read(RUN/'results_partial.json');ledger=read(RUN/'assignment_ledger.json');jobs=manifest['jobs'];byv={r['variant']:r for r in rows}
variants={'zero':(0.,0.),'phi_slow':(1.,0.),'lift_slow':(0.,1.),'mixed_50_50':(.5,.5),'mixed_50_100':(.5,1.),'mixed_100_50':(1.,.5),'mixed_100_100':(1.,1.)}
check('all7_assigned_results_and_failures_retained',len(jobs)==len(rows)==len(ledger)==7 and len(byv)==7 and set(byv)==set(variants) and rows==partial==summary['results'] and all(x['status']=='complete' for x in rows) and all(x['status']=='complete' for x in ledger) and Counter(x['job_id'] for x in rows)==Counter(x['job_id'] for x in jobs)==Counter(x['job']['job_id'] for x in ledger) and summary['failures']==read(RUN/'runtime_failures.json')==[])
check('plan_roles_source_manifest_bindings',plan==manifest['plan'] and plan['training']==['zero','phi_slow','lift_slow'] and set(plan['strict_holdout'])==set(variants)-set(plan['training']) and manifest['source_snapshot_manifest_sha256']==sha(RUN/'source_snapshot_manifest.json') and manifest['plan_sha256']==sha(RUN/'plan.json') and all(tuple(plan['variants'][k])==v for k,v in variants.items()))
sources=[]
for k,h in manifest['sources'].items():
 e=snap['files'][k];p=Path(e['original_path']);q=RUN/e['snapshot_path'];a,b=sha(p),sha(q);sources.append(dict(key=k,expected=h,current=a,snapshot=b,matches=h==e['sha256']==a==b and p.stat().st_size==q.stat().st_size==e['bytes']))
check('all237_frozen_source_snapshot_current_hashes',len(sources)==237 and all(x['matches'] for x in sources) and set(snap['files'])==set(manifest['sources'])=={p.relative_to(RUN/'sources').as_posix() for p in (RUN/'sources').rglob('*') if p.is_file()})
profiles={};streams={};records={};metas={};entrydata={};pathmap={}
for job in jobs:
 v=job['variant'];d=Path(job['directory']);path=d/'record/trace.npz';pathmap[v]=path;m=read(d/'record/metadata.json');reported=read(d/'result.json');r=reported['research_metrics'];metas[v]=m
 check(v+':artifact_result_trace_binding',reported['variant']==v and reported['training_role']==job['training_role'] and sha(path)==reported['trace_sha256']==r['trace_sha256']==m['trace_sha256']==byv[v]['trace_sha256'] and m['research_source_sha256']==manifest['sources'])
 check(v+':registered_actual_context',m['method']=='ratio' and m['seed']==610120 and m['shape']=='capsule' and m['config']['mass']==.12 and m['config']['friction']==.6 and m['config']['timestep']==.0005 and m['config']['control_dt']==.005 and r['physical_parameters']==m['config'])
 record=raw_check(v,d,reported,r,m,path);x=record.pop('raw');t=record.pop('time');record['stages']=r['safety']['stages'];record['clearance_200Hz']=clearance_detail(t,x['all_hand_object'])
 with np.load(path,allow_pickle=False) as z:hand_raw=z['substep_all_hand_object']
 record['clearance_2000Hz']=clearance_detail(x['substep_time']+x['substep_duration'],hand_raw)
 entries={}
 for name in ['release_entry','probe_entry']:
  ep=d/name;side=read(ep/'entry_side_state.json');signature=read(ep/'entry_signature.json');local=read(ep/'local_control_state.json')
  with np.load(ep/'entry_physics.npz',allow_pickle=False) as z:physics={k:z[k].item() if z[k].shape==() else z[k] for k in z.files}
  computed=dict(signature,local_control_state_sha256=digest(local))
  check(v+':'+name+'_full_checksums_recomputed',computed==byv[v]['entries'][name] and signature['physics_arrays_sha256']==digest(encode(physics)) and signature['integration_state_sha256']==digest(encode(physics['integration_state'])) and signature['python_causal_side_state_sha256']==digest(strip_wall(side)))
  rng=at(side,'sensors','rng');queue=at(side,'sensors','queue');fk=at(side,'skin_geometry','fk_data')
  check(v+':'+name+'_rng_queue_fk_present',rng.get('numpy_generator')=='PCG64' and len(queue['items'])>0 and {'integration_state','qpos','qvel','ctrl','qacc_warmstart'}<=set(mapping(fk)))
  entries[name]=dict(physics=physics,side=strip_wall(side),local=local)
 entrydata[v]=entries
 with np.load(d/'decoder_commands.npz',allow_pickle=False) as z:profile={k:z[k] for k in z.files}
 with np.load(d/'robot_physical_stream.npz',allow_pickle=False) as z:stream={k:z[k] for k in z.files}
 profiles[v]=profile;streams[v]=stream;clock=stream['physical_time_s']
 check(v+':complete32001_actual_robot25_physical_end_samples',len(clock)==32001 and np.allclose(clock,np.arange(32001)*.0005,rtol=0,atol=1e-7) and stream['position25'].shape==stream['velocity25'].shape==(32001,25) and np.isfinite(stream['position25']).all() and np.isfinite(stream['velocity25']).all() and tuple(stream['axes25'])==tuple(m['raw_joint_names'])+('transport_lift',))
 check(v+':physical_end_pressure_clock_and_force_exact',np.allclose(clock[1:],x['substep_time']+x['substep_duration'],rtol=0,atol=1e-12) and np.array_equal(stream['target_finger_pressure5_N'][1:],x['substep_normal']) and np.array_equal(stream['external_finger_pressure5_N'][1:],x['substep_skin_normal']))
 with np.load(path,allow_pickle=False) as z:
  robot_control=np.column_stack((z['qpos'][:,m['raw_joint_qpos_indices']],z['lift']));velocity_control=z['qvel'][:,list(m['raw_joint_qpos_indices'])+[0]]
 check(v+':qend_robot25_agrees_shared_control_encoders',np.array_equal(robot_control,stream['position25'][::10]) and np.array_equal(velocity_control,stream['velocity25'][::10]))
 a,b=variants[v];tick=profile['relative_control_tick'];weight=np.clip((tick-4.)/4.,0.,1.);tau=profile['control_time']-12.;phi=tau if a==0 else (2400.+tick-4.*a*weight)*.005-12.;lclock=tau if b==0 else (2400.+tick-4.*b*weight)*.005-12.;lift=.1*(1.-np.clip(lclock/.6,0,1));raw_lift=.1*(1.-lclock/.6)
 check(v+':all8801_actual_quarter_clock_coordinates',len(tick)==8801 and np.all(tick==np.rint(tick)) and np.array_equal(profile['ramp_weight'],weight) and np.array_equal(profile['native_phi_s'],tau) and np.array_equal(profile['actual_phi_s'],phi) and np.array_equal(profile['lift_clock_s'],lclock) and np.array_equal(profile['actual_command21'][:,18],lift) and np.array_equal(profile['unclipped_lift_reference_m'],raw_lift))
 check(v+':clipping_explicit_after_delay_not_positive_forever',np.array_equal(profile['lift_clipped'].astype(bool),(raw_lift<0)|(raw_lift>.1)) and lift[-1]==0 and np.all((lift>=0)&(lift<=.1)))
 unique=np.r_[True,np.diff(tick)!=0];q=profile['actual_command21'][unique];ph=profile['actual_phi_s'][unique];zref=q[:,18];same=np.diff(tick)==0
 check(v+':held_substep_reference_and_progress_rate',np.all(np.diff(ph)>=-1e-12) and np.all(np.diff(ph)<=.005+1e-12) and np.all(np.diff(zref)<=1e-12) and np.all(-np.diff(zref)<=.1/.6*.005+1e-12) and np.all(np.diff(profile['actual_command21'],axis=0)[same]==0))
 check(v+':actuator_references_match_actual22ctrl',np.array_equal(profile['actual_command21'][:,:18],profile['actual_ctrl'][:,2:20]) and np.array_equal(profile['actual_command21'][:,19:],profile['actual_ctrl'][:,:2]) and np.max(abs((profile['actual_ctrl'][:,20]-profile['actual_command21'][:,18])-(profile['native_carrier_ctrl'][:,20]-profile['native_carrier_command21'][:,18])))<1e-15)
 record['input_sha256']={str(p.relative_to(RUN)):sha(p) for p in [path,d/'record/metadata.json',d/'result.json',d/'decoder_commands.npz',d/'robot_physical_stream.npz']};records[v]=record

zero=profiles['zero'];zcmd=zero['actual_command21'];ztick=zero['relative_control_tick'];unique=np.r_[True,np.diff(ztick)!=0];native_steps=np.max(abs(np.diff(zcmd[unique],axis=0)),axis=0);native_tv=abs(np.diff(zcmd[unique],axis=0)).sum(axis=0)
check('zero_all22ctrl_applications_exact_native_carrier',np.array_equal(zero['actual_ctrl'],zero['native_carrier_ctrl']) and np.array_equal(zero['actual_command21'],zero['native_carrier_command21']))
PHYSICAL=('time','normal','skin_normal','shear','palm','hand_other','all_hand_object','support','position','relative','rotation','relative_velocity','angular_velocity','lift','lift_ref','wrist_position','requested_wrist_reference','qpos','qvel','ctrl_before','ctrl','alpha_before','alpha','executed_action','warning_count','collision','actual_motion_target','executed_motion_reference','penetration','world_penetration','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm','substep_hand_other','substep_all_hand_object','substep_support','substep_penetration','substep_world_penetration')
old=ROOT/'output/release_basis_minimal_v2/runs/capsule__native2d/record/trace.npz'
with np.load(pathmap['zero'],allow_pickle=False) as a,np.load(old,allow_pickle=False) as b:
 detail={f:a[f].dtype==b[f].dtype and a[f].shape==b[f].shape and a[f].tobytes()==b[f].tobytes() for f in PHYSICAL};oldobs=np.array_equal(a['observation_json'],b['observation_json'])
check('zero_full16s40fields_and_all_public_observations_bitexact_old',all(detail.values()) and oldobs,detail)
pairs=[]
for v in variants:
 p=profiles[v];unique=np.r_[True,np.diff(p['relative_control_tick'])!=0];steps=abs(np.diff(p['actual_command21'][unique],axis=0));entry={}
 for name in ['release_entry','probe_entry']:
  a=entrydata['zero'][name];b=entrydata[v][name];entry[name]=a['physics'].keys()==b['physics'].keys() and all(np.array_equal(a['physics'][f],b['physics'][f],equal_nan=True) for f in a['physics']) and a['side']==b['side'] and a['local']==b['local']
 check(v+':both_complete_entry_side_local_state_exact_to_zero',all(entry.values()))
 check(v+':allactual_inputs_from_executed_zero_and_same_clock',np.array_equal(p['control_time'],zero['control_time']) and np.array_equal(p['physical_time'],zero['physical_time']) and np.array_equal(p['actual_phi_s']-zero['actual_phi_s'],p['actual_phi_offset_s']) and np.array_equal(p['actual_command21'][:,18]-zcmd[:,18],p['actual_lift_offset_m']) and np.array_equal(p['native_carrier_command21'],zcmd))
 check(v+':same_native21_tick_and_totalvariation_envelope',np.all(steps<=native_steps+1e-10) and np.all(steps.sum(axis=0)<=native_tv+1e-9))
 mismatch=[];fields=[]
 with np.load(pathmap['zero'],allow_pickle=False) as a,np.load(pathmap[v],allow_pickle=False) as b:
  for f in a.files:
   if f in ('metadata','control_ms'):continue
   x,y=a[f],b[f]
   if x.ndim==0:continue
   mask=a['substep_time']<12.02-1e-9 if f.startswith('substep_') else a['time']<12.02-1e-9;x,y=x[mask],y[mask]
   equal=all(strip_wall(json.loads(str(u)))==strip_wall(json.loads(str(w))) for u,w in zip(x,y)) if f=='debug_json' else bool(np.array_equal(x,y,equal_nan=True)) if x.dtype.kind in 'biufc' else bool(np.array_equal(x,y))
   fields.append(f)
   if not equal:mismatch.append(f)
 check(v+':all_stored_physical_command_observation_preprobe_fields_exact',not mismatch,dict(fields=len(fields),mismatches=mismatch));pairs.append(dict(variant=v,entry_exact=entry,prefix_fields=len(fields),prefix_mismatches=mismatch))
check('all7_actual_configs_noise_gains_identical',all(all(m[f]==metas['zero'][f] for f in ['config','noise','gains','limits','budget','initial_position','model_sha256']) for m in metas.values()))
report=dict(schema='release_local_response_v2_offline_audit_v1',reviewer_role='Pilot implementer; separate retained-data arithmetic. This is not organizationally independent authorship, model admission or a safety certificate.',created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,issues=issues,input_sha256={f:sha(RUN/f) for f in ['summary.json','manifest.json','plan.json','assignment_ledger.json','results.json','source_snapshot_manifest.json']},raw_arithmetic_source_sha256=sha(PURE),checker_sha256=sha(Path(__file__)),source_checks=sources,records=records,pairing=pairs,phase_contract=dict(origin_control_s=12.02,first_quarter_command_start_s=12.025,full_offset_command_start_s=12.04,full_offset_first_physical_end_s=12.0405,meaning='At physical end12.040 the completed dynamics used the preceding held command12.035 at75% offset. The newly applied100% command at12.040 affects first physical end12.0405. Current command and q-end must not be divided as an instantaneous response.'),scope=dict(full=['7 complete16s raw/V3 checks,224000 physical force substeps,22407 control endpoints','237 snapshot/current source hashes;7 result/config/noise/assignment/failure denominators','14 full physics/Python/RNG/queue/FK/local-decoder-entry snapshots and7 complete preprobe trace comparisons','61607 actual command applications: quarter clocks, held reference, real21 target/control, monotonic native rate and total variation, actual input minus executed zero','224007 actual physical robot25/velocity samples with original raw pressure and control-encoder phase agreement','zero full16s40physical fields+all public observations bit-exact to old native'],not_checked=['no physical rerun or frozen source edits','no pressure response model/kernel or fitted error bound in this audit','no arbitrary2D/3D action domain or material-safety admission','robot actual25 is observed; no new hard joint/tendon dynamics/rate certificate invented']),physics_steps=0,model_admitted=False,interpretation='All7 physically clear but all7 whole forceguards fail; lower full force peak in a selected direction is not a safe release policy. Only the single registered capsule entrance and one-sided slowdown profiles were tested. First/final/recontact are separate; late2.2s behavior does not validate an entry-local linear model.')
with OUT.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
lines=['# Release local response V2 离线数值复核','',f'检查{sum(x["passed"] for x in checks)}/{len(checks)}；问题{len(issues)}。实现者使用独立旧raw审计算法复算，不冒充角色独立验收。','', '|变体|全程峰值 N|局部超限 Ns|V3首次清除确认延迟 s|力限|','|---|---|---|---|---|']
for v in variants:
 r=records[v];lines.append(f'|{v}|{max(r["safety"]["finger_peak_N"]):.9f}|{r["safety"]["local_excess_integral_Ns"]:.9f}|{r["release"]["physical_clearance_latency_s"]:.3f}|{r["safety"]["guard_pass"]}|')
lines+=['','全部7条真实释放，但全16s力限均失败。14完整入口及新增decoder累积状态、全部存储的pre12.020物理/命令/观测字段配对；zero完整16s40字段与全部公共观测对旧native逐位一致。237来源未变。','', '命令相位必须保留：12.040应用满offset，q-end12.040仍来自前一个12.035的75% held输入；首个满offset物理结果在12.0405。识别需输入完整实际历史，不能以同一行新command直接除q-end或pressure。','', '当前只是一形状、一个入口和单侧profile；四mixed留出不能选优后回填训练。应先在20ms及后续分窗检查2轴叠加误差和输入兑现，之后再注册.15/.35进度/接触转换入口；不得把entry-local模型外推到2.2s。若实测局部2D任务锥冲突，再讨论TH/nonTH/lift3D，不先造缺失核或误差界。','',f'JSON SHA256 `{sha(OUT)}`。']
OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),checks=len(checks),passed=sum(x['passed'] for x in checks),issues=issues,table=[dict(variant=v,peak=max(r['safety']['finger_peak_N']),excess=r['safety']['local_excess_integral_Ns'],latency=r['release']['physical_clearance_latency_s'],guard=r['safety']['guard_pass']) for v,r in records.items()]),ensure_ascii=False),flush=True)

"""Retained data only. No controller, evaluator, plant, QP or physics API imports."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import ast,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'identification/v9_component_ablation_v1'
RUN=PACKAGE/'output/causal8_v1'
OUT=ROOT/'review/causal8_independent_audit_v1.json'
MD=OUT.with_suffix('.md')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
# Reuse only this reviewer's already independently implemented pure arithmetic.
PURE=ROOT/'review/verify_release_basis_v2_independent.py'
tree=ast.parse(PURE.read_text(encoding='utf-8'))
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'sha','dwell','variation'}]
exec(compile(ast.Module(body=functions,type_ignores=[]),str(PURE),'exec'))
checks=[];issues=[]
def check(name,ok,detail=None):
 x=dict(name=name,passed=bool(ok),detail=detail);checks.append(x)
 if not ok:issues.append(x)
def metric(name,a,b):
 ok=a is b if a is None or b is None else a==b if isinstance(a,(str,bool)) or isinstance(b,(str,bool)) else bool(np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True))
 check(name,ok,None if ok else dict(actual=a,reported=b))
finish=read(RUN/'execution_complete.json')
if finish.get('execution_complete') is not True:raise RuntimeError('Complete terminal required')
if OUT.exists() or MD.exists():raise FileExistsError('Preserve independent evidence')
manifest=read(RUN/'manifest.json');plan=read(RUN/'plan.json');registration=read(PACKAGE/'plan.json');ledger=read(RUN/'assignment_ledger.json');rows=read(RUN/'trials.json');snapshot=read(RUN/'source_snapshot_manifest.json');acceptance=read(RUN/'endpoint_execution_acceptance.json')
variants=plan['variants'];scenes=plan['environments'];key=lambda x:(x['scenario'],x['component_variant'],x['seed'])
jobs=manifest['jobs'];bykey={key(x):x for x in rows};expected={(s,v,610212) for s in ['friction_drop','collision'] for v in ['P0H0','P0H1','P1H0','P1H1']}
check('exact8_assignments_and_all_results_retained',len(jobs)==len(rows)==len(ledger)==8 and len(bykey)==8 and set(bykey)==expected and Counter(map(key,jobs))==Counter(map(key,rows))==Counter(key(x['job']) for x in ledger) and all(x['status']=='evaluated' for x in ledger))
check('terminal_and_registration_hashes',finish['planned']==finish['trials']==8 and finish['runtime_failures']==[] and finish['sources_unchanged'] is True and all(finish[k]==sha(RUN/f) for k,f in [('trials_sha256','trials.json'),('manifest_sha256','manifest.json'),('assignment_ledger_sha256','assignment_ledger.json'),('endpoint_acceptance_sha256','endpoint_execution_acceptance.json')]) and plan==registration and manifest['registration_sha256']==sha(PACKAGE/'plan.json') and manifest['source_snapshot_manifest_sha256']==sha(RUN/'source_snapshot_manifest.json'))
check('declared_factorial_packages_and_scope',variants=={'P0H0':dict(projection_package=False,additional_risk_motion_hold=False),'P0H1':dict(projection_package=False,additional_risk_motion_hold=True),'P1H0':dict(projection_package=True,additional_risk_motion_hold=False),'P1H1':dict(projection_package=True,additional_risk_motion_hold=True)} and plan['planned']==8 and plan['motion_mode']=='transport' and plan['physical_seed']==610212 and plan['split']=='robustness' and plan['performance_claim_allowed'] is False)
sources=[]
for name,h in manifest['research_source_sha256'].items():
 e=snapshot['files'][name];p=Path(e['original_path']);q=RUN/e['snapshot_path'];a,b=sha(q),sha(p)
 sources.append(dict(name=name,expected=h,snapshot=a,current=b,original_path=str(p),matches=h==a==b==e['sha256'] and p.stat().st_size==q.stat().st_size==e['bytes']))
check('all83_snapshot_and_current_source_hashes',len(sources)==83 and all(x['matches'] for x in sources))
check('frozen77_parent_and_added6_package_sources',len(plan['frozen_v9_source_sha256'])==77 and all(manifest['research_source_sha256'][k]==h for k,h in plan['frozen_v9_source_sha256'].items()) and set(snapshot['files'])==set(manifest['research_source_sha256'])=={p.relative_to(RUN/'sources').as_posix() for p in (RUN/'sources').rglob('*') if p.is_file()} and len(set(manifest['research_source_sha256'])-set(plan['frozen_v9_source_sha256']))==6)
check('registered_parent_artifacts_match',plan['parent_manifest_sha256']==sha(ROOT/'output/development_v9_transport_v1/manifest.json') and plan['parent_factorial_plan_sha256']==sha(ROOT/'review/v9_factorial_gate_plan_v1.json'))

PHYSICAL=('time','normal','skin_normal','shear','palm','hand_other','all_hand_object','support','position','relative','rotation','relative_velocity','angular_velocity','lift','lift_ref','wrist_position','requested_wrist_reference','qpos','qvel','ctrl_before','ctrl','alpha_before','alpha','executed_action','warning_count','collision','actual_motion_target','executed_motion_reference','penetration','world_penetration','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm','substep_hand_other','substep_all_hand_object','substep_support','substep_penetration','substep_world_penetration')
public_fields=['time_s','motion_axes','motion_target','tracking_tolerances','deadline_s','release_at_s','release_deadline_s','motion_preview','motion_reference_bounds','sensor_characteristics','limits','budget']
records={};meta={};selected={};paths={};endpoint=[]
for job in jobs:
 k=key(job);scene,variant,seed=k;label=scene+'/'+variant;folder=Path(job['destination']);path=folder/'trace.npz';paths[k]=path;m=read(folder/'metadata.json');reported=read(folder/'result.json');r=reported['research_metrics'];meta[k]=m
 check(label+':result_and_treatment_lineage',reported==bykey[k] and job['method']==m['method']==reported['method']=='joint' and m['component_variant']==reported['component_variant']==variant and m['component_flags']==reported['component_flags']==job['component_flags']==variants[variant] and m['motion_mode']==reported['motion_mode']=='transport' and m['seed']==610212 and m['observer_warmup_s']==.2 and m['component_registration_sha256']==manifest['registration_sha256'] and m['research_source_sha256']==manifest['research_source_sha256'])
 thash=sha(path);check(label+':raw_hash_and_evaluator_identity',thash==reported['trace_sha256']==r['trace_sha256']==m['trace_sha256'] and r['protocol_sha256']==manifest['research_source_sha256']['protocol.json'] and r['evaluation_source_sha256']==manifest['research_source_sha256']['evaluation.py'])
 u=np.random.default_rng(seed+511000).normal(size=4)
 check(label+':actual_expected_physics_RNG_and_material',m['config']==plan['expected_realized_configs'][scene]==r['physical_parameters'] and m['config']['mass']==float(.12*np.clip(1+.06*u[0],.85,1.15)) and m['config']['friction']==float(.6*np.clip(1+.08*u[1],.8,1.2)) and m['limits']==[5.]*5 and m['budget']==18.)
 with np.load(path,allow_pickle=False) as z:
  wanted=['time','normal','skin_normal','palm','support','position','relative','relative_velocity','angular_velocity','lift','lift_ref','all_hand_object','controller_active','warning_count','actual_motion_target','executed_motion_reference','wrist_position','requested_wrist_reference','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm','collision','debug_json','task_request_json','environment_json','requested_action','prediction_available','predicted_normal']
  x={f:z[f] for f in wanted}
 t=x['time'];st=x['substep_time'];dur=x['substep_duration'];f=x['substep_skin_normal'];n=x['normal'];skin=x['skin_normal'];limits=np.array(m['limits']);budget=m['budget'];hold=(t>=7.5-1e-9)&(t<12-1e-9)
 complete=len(t)==3201 and len(st)==32000 and np.allclose(t,np.arange(3201)*.005,atol=1e-8,rtol=0) and np.allclose(st,np.arange(32000)*.0005,atol=1e-7,rtol=0) and np.allclose(dur,.0005,atol=1e-9,rtol=0) and abs(dur.sum()-16)<1e-8 and all(np.isfinite(x[v]).all() for v in ['normal','skin_normal','position','relative','relative_velocity','angular_velocity','lift','lift_ref','support','substep_normal','substep_skin_normal'])
 check(label+':full16s_control_raw_coverage',complete and r['record_complete'] and reported['record_complete'])
 local=np.maximum(f-limits,0);total=np.maximum(f.sum(axis=1)-budget,0);safe=(skin<=limits+.02).all(axis=1)&(skin.sum(axis=1)<=budget+.02)
 guard=bool(complete and (f<=limits+.02).all() and (f.sum(axis=1)<=budget+.02).all() and safe[-1]);peak=np.maximum(f.max(axis=0),skin[-1])
 safety=dict(guard_pass=guard,finger_peak_N=peak.tolist(),local_excess_integral_Ns=float(np.sum(local*dur[:,None])),total_excess_integral_Ns=float(total@dur),palm_peak_N=float(max(x['substep_palm'].max(),x['palm'][-1])))
 for name,value in safety.items():metric(label+':safety.'+name,value,r['safety'][name])
 metric(label+':canonical_forceguard',guard,reported['safety']['force_protection_pass'])
 stages={}
 for stage,mask in [('shared_preparation',st<7.5-1e-9),('policy_holding_transport',(st>=7.5-1e-9)&(st<12-1e-9)),('mechanical_release_and_after',st>=12-1e-9)]:
  values=dict(duration_s=float(dur[mask].sum()),finger_peak_N=f[mask].max(axis=0).tolist(),local_excess_integral_Ns=float(np.sum(local[mask]*dur[mask,None])),total_excess_integral_Ns=float(total[mask]@dur[mask]));stages[stage]=values
  for name,value in values.items():metric(label+':stage.'+stage+'.'+name,value,r['safety']['stages'][stage][name])
 contacts=n>.2;support=contacts[:,0]&(contacts.sum(axis=1)>=3);raised=x['position'][:,2]>=m['initial_position'][2]+.025;le=abs(x['lift']-x['lift_ref']);motion=(le<=.01)&(x['support']<=.05);re=np.linalg.norm(x['relative']-m['reference_relative'],axis=1)
 established=dwell(t,support&raised&motion,0,7.5,.1);drop=dwell(t,~support|~raised|(re>.08),7.5,12,.1,True);tracking=dwell(t,le>.01,7.5,12,.3,True)
 failure=None;failure_at=None
 if not m['handoff_passed'] or established is None:failure='preparation_or_handoff_failed';failure_at=7.5
 elif drop is not None and (tracking is None or drop<=tracking):failure='support_or_object_lost';failure_at=drop
 elif tracking is not None:failure='transport_reference_unmet';failure_at=tracking
 exposure=bool(support[2400] and raised[2400] and x['support'][2400]<=.05 and x['all_hand_object'][2400]>.2);clear=dwell(t,x['all_hand_object']<=.2,12,15.5,.2);released=bool(exposure and clear is not None);qualified=bool(exposure and m['handoff_passed'] and (failure_at is None or failure_at>12))
 release=dict(scheduled=True,actual_command_at=12.,physical_exposure=exposure,physical_success=released,task_qualified_exposure=qualified,task_qualified_success=released and qualified,physical_clearance_latency_s=clear-12 if released else None)
 for name,value in release.items():metric(label+':release.'+name,value,r['release'][name])
 mission=bool(complete and m['handoff_passed'] and failure is None and released and x['warning_count'].max()==0);wb=np.any(abs(x['wrist_position']-x['requested_wrist_reference'])>.015,axis=1);wfail=dwell(t,wb,7.5,12,.3,True)
 metric(label+':legacy_task',mission,reported['task_success']);metric(label+':V3_mission',mission,r['mission_success']);metric(label+':V3_failure_reason',failure,r['failure_reason']);metric(label+':V3_failure_at',failure_at,r['failure_at']);metric(label+':wrist_failure_at',wfail,r['wrist_tracking_guard']['failure_at']);metric(label+':full_motion_mission',mission and wfail is None,r['full_motion_mission_success'])
 stable=support&raised&motion&safe&(re<=.005)&(np.linalg.norm(x['relative_velocity'],axis=1)<=.005)&(np.linalg.norm(x['angular_velocity'],axis=1)<=.15);firststable=dwell(t,stable,7.5,8.9,.2);metric(label+':initial_stability',1.4 if firststable is None else firststable-7.5,r['initial_stability_wait_s'])
 grid=np.r_[0.,st+dur];actual=np.vstack((n[0],x['substep_normal']));band=np.full_like(n,np.nan);kt=np.arange(-50,51)*.0005;kernel=.05*np.sinc(100*kt)*np.hanning(101);kernel/=kernel.sum()
 for lo,hi in [(0,7.5),(7.5,12),(12,16)]:
  physical=(grid>=lo-1e-9)&(grid<=hi+1e-9);sample=(t>=lo-1e-9)&(t<=hi+1e-9)
  if lo==12:sample&=t>12+1e-9
  for digit in range(5):band[sample,digit]=np.interp(t[sample],grid[physical],np.convolve(np.pad(actual[physical,digit],(50,50),mode='reflect'),kernel,mode='valid'))
 quality={}
 for name,lo,hi in [('static_quality',8.2,8.9),('dynamic_quality',9.3,11.7),('entire_holding_quality',7.5,12.)]:
  mask=(t>=lo-1e-9)&(t<=hi+1e-9);eligible=bool(complete and np.all(support[mask]&raised[mask]&(x['support'][mask]<=.05)&x['controller_active'][mask]));metric(label+':'+name+'.eligible',eligible,r[name]['eligible']);quality[name]=dict(eligible=eligible)
  if eligible:
   metric(label+':'+name+'.mean_force',float(n[mask].sum(axis=1).mean()),r[name]['mean_actual_finger_force_N'])
   for signal,values in [('bandlimited',band),('raw_200Hz',n)]:
    values=variation(t[mask],values[mask]);quality[name][signal]=values
    for name2,value in values.items():metric(label+':'+name+'.'+signal+'.'+name2,value,r[name][signal][name2])
 tracking_values={}
 for pname,lo,hi in [('holding_motion_tracking',7.5,12.),('dynamic_motion_tracking',9.3,11.7)]:
  weights=np.maximum(0,np.minimum(np.r_[t[1:],t[-1]],hi)-np.maximum(t,lo));error=np.column_stack((x['lift'],x['wrist_position']))-x['actual_motion_target'];mask=weights>0;seconds=weights.sum();values=dict(rmse=np.sqrt(np.sum(error**2*weights[:,None],axis=0)/seconds).tolist(),max_abs_error=abs(error[mask]).max(axis=0).tolist(),absolute_error_integral=np.sum(abs(error)*weights[:,None],axis=0).tolist(),duration_s=float(seconds));tracking_values[pname]=values
  for name,value in values.items():metric(label+':tracking.'+pname+'.'+name,value,r['progress'][pname][name])
 check(label+':no_fictitious_pressure_forecast',not x['prediction_available'].any() and np.isnan(x['predicted_normal']).all())
 debug=[json.loads(str(v)) for v in x['debug_json'][hold]];flags=[d.get('component_ablation') for d in debug];P=variants[variant]['projection_package'];H=variants[variant]['additional_risk_motion_hold']
 check(label+':actual_all900_holding_treatment_annotations',len(flags)==900 and all(d and d['variant']==variant and d['projection_package'] is P and d['additional_risk_motion_hold'] is H for d in flags))
 active=np.array([bool(d['actual_additional_risk_hold']) for d in flags]);counter=np.array([bool(d['counterfactual_risk_hold']) for d in flags]);paused=np.array([bool(d.get('pause_transport',False)) for d in debug]);rollback=np.array([bool(d.get('reference_memory_rollback',False)) for d in debug])
 check(label+':additional_hold_toggle_matches_live_execution',all(bool(d.get('recovery_motion_hold',False))==v for d,v in zip(debug,active)) and (np.array_equal(active,counter) if H else not active.any()))
 check(label+':projection_package_live_route',all('stage_results' in d for d in debug) if P else all(d.get('force_projection_applied') is False and not d.get('reference_memory_rollback',False) and d.get('force_path')=='unchanged original ratio actual action' for d in debug))
 if P:check(label+':executed_closing_mask_enforced',all(np.all(np.asarray(d['actual_subinterval_delta_closure'])[~np.asarray(d['known_grasp_support'],bool)]<=1e-10) for d in debug))
 ref=x['executed_motion_reference'];delta=np.diff(ref,axis=0,prepend=ref[:1]);moving=np.any(abs(np.diff(x['actual_motion_target'],axis=0,prepend=x['actual_motion_target'][:1]))>1e-10,axis=1)&hold;hdelta=delta[hold]
 check(label+':active_hold_actual_reference_stationary',not active.any() or np.all(abs(hdelta[active])<=1e-10))
 check(label+':shared_reference_rate_authority',np.all(abs(hdelta)<=np.array([.035,.12,.12])*.005+1e-9))
 public=hashlib.sha256()
 for encoded in x['task_request_json']:
  d=json.loads(str(encoded));public.update(json.dumps({f:d[f] for f in public_fields},sort_keys=True,separators=(',',':')).encode('utf-8')+b'\n')
 selected[k]=dict(time=t,environment=x['environment_json'],public_hash=public.hexdigest(),target=x['actual_motion_target'],reference=ref)
 hit=np.flatnonzero(x['collision']>.02);forcehold=st>=7.5-1e-9;forcehold&=st<12-1e-9
 exposure_values=dict(first_actual_fixture_contact_s=None if not len(hit) else float(t[hit[0]]),fixture_peak_200Hz_N=float(x['collision'].max()),holding_actual_peak_N=float(f[forcehold].max()),additional_risk_hold_s=float(active.sum()*.005),counterfactual_risk_hold_s=float(counter.sum()*.005),baseline_or_other_pause_s=float((paused&~active).sum()*.005),rollback_flagged_control_samples=int(rollback.sum()),projection_modified_flagged_samples=int(sum(d.get('nominal_proposal_accepted_exact') is False for d in debug)),novel_contact_flagged_samples=int(sum(np.asarray(d.get('new_unaffiliated_contact',[False]*5),bool).any() for d in debug)),user_moving_samples=int(moving.sum()),stationary_reference_while_user_moves_s=float(np.all(abs(delta[moving])<=1e-10,axis=1).sum()*.005))
 records[label]=dict(scenario=scene,variant=variant,mission_success=mission,full_motion_mission_success=mission and wfail is None,failure_reason=failure,failure_at=failure_at,wrist_failure_at=wfail,safety=safety,stages=stages,release=release,quality=quality,tracking=tracking_values,actual_exposure_and_toggle=exposure_values,public_task_sha256=public.hexdigest(),input_sha256={p.name:sha(p) for p in [path,folder/'metadata.json',folder/'result.json']})

pairs=[];prefix=[]
for scene in scenes:
 keys=[(scene,v,610212) for v in variants];base=meta[keys[0]];fields=['config','noise','limits','budget','gains','actual_ratio_task_gains','initial_position','model_sha256','observer_warmup_s','task_contract']
 equal={f:all(meta[k][f]==base[f] for k in keys) for f in fields};env=all(np.array_equal(selected[k]['environment'],selected[keys[0]]['environment']) for k in keys)
 check(scene+':all4_actual_configs_noise_material_gains_public_environment_paired',all(equal.values()) and env and len({selected[k]['public_hash'] for k in keys})==1 and all(np.array_equal(selected[k]['target'],selected[keys[0]]['target']) for k in keys))
 for variant in list(variants)[1:]:
  kk=(scene,variant,610212);differences=[]
  with np.load(paths[keys[0]],allow_pickle=False) as a,np.load(paths[kk],allow_pickle=False) as b:
   ct=a['time']<9.3-1e-9;st=a['substep_time']<9.3-1e-9
   for field in PHYSICAL+('observation_json',):
    aa=a[field][st if field.startswith('substep_') else ct];bb=b[field][st if field.startswith('substep_') else ct]
    if not np.array_equal(aa,bb,equal_nan=True) if aa.dtype.kind in 'biufc' else not np.array_equal(aa,bb):differences.append(field)
  prefix.append(dict(scenario=scene,variant=variant,window='t<9.3-1e-9',fields=len(PHYSICAL)+1,mismatches=differences))
  check(scene+':common_pre_event41_fields_'+variant,not differences,differences)
 pairs.append(dict(scenario=scene,metadata_fields_equal=equal,environment_all_variants_exact=env))
 for variant,method in [('P0H0','ratio_preview'),('P1H1','joint')]:
  spec=plan['endpoints'][scene+'/'+variant];old=Path(spec['path']);new=paths[(scene,variant,610212)];eq={}
  with np.load(new,allow_pickle=False) as a,np.load(old,allow_pickle=False) as b:
   for field in PHYSICAL:
    aa,bb=a[field],b[field];eq[field]=aa.dtype==bb.dtype and aa.shape==bb.shape and aa.tobytes()==bb.tobytes()
  check(scene+':endpoint40_actual_fields_'+variant,sha(old)==spec['sha256'] and all(eq.values()),eq)
  endpoint.append(dict(scenario=scene,variant=variant,reference_method=method,reference_trace_sha256=sha(old),fields=eq))
check('all8_same_public_user_targets_previews_deadlines',len({v['public_hash'] for v in selected.values()})==1)
check('endpoint_author_acceptance_independently_reproduced',len(endpoint)==4 and all(all(x['fields'].values()) for x in endpoint) and acceptance['passed'] and finish['endpoint_execution_acceptance_passed'])
effects={}
for scene in scenes:
 rr={v:records[scene+'/'+v] for v in variants}
 vector={v:dict(mission=float(r['mission_success']),full_motion_mission=float(r['full_motion_mission_success']),guard=float(r['safety']['guard_pass']),holding_excess_Ns=r['stages']['policy_holding_transport']['local_excess_integral_Ns'],full_excess_Ns=r['safety']['local_excess_integral_Ns'],holding_rmse=np.array(r['tracking']['holding_motion_tracking']['rmse']),dynamic_rmse=np.array(r['tracking']['dynamic_motion_tracking']['rmse']),additional_hold_s=r['actual_exposure_and_toggle']['additional_risk_hold_s']) for v,r in rr.items()}
 if all(r['quality']['dynamic_quality']['eligible'] for r in rr.values()):
  for v in variants:vector[v]['dynamic_TV_N_s']=rr[v]['quality']['dynamic_quality']['bandlimited']['finger_TV_rate_N_s']
 contrasts={}
 for name in vector['P0H0']:
  y={v:np.asarray(vector[v][name]) for v in variants};contrasts[name]={k:value.tolist() for k,value in dict(P_at_H0=y['P1H0']-y['P0H0'],P_at_H1=y['P1H1']-y['P0H1'],H_at_P0=y['P0H1']-y['P0H0'],H_at_P1=y['P1H1']-y['P1H0'],interaction=y['P1H1']-y['P1H0']-y['P0H1']+y['P0H0']).items()}
 effects[scene]=dict(contrasts=contrasts,dynamic_quality_common4_cohort=all(r['quality']['dynamic_quality']['eligible'] for r in rr.values()),interpretation='Finite differences within this fixed environment and seed only. P is the complete projection/mask/minimum-reserve/rollback package. H retains baseline pause and adds closed-loop risk hold. No per-mechanism or population identification; interactions include changed force-motion feedback.')
anchors=[]
for relative,needle,reason in [('identification/v9_component_ablation_v1/component_controller.py','if not self.risk_hold_enabled:', 'H0 retains counterfactual hysteresis evidence but disables additional hold authority'),('identification/v9_component_ablation_v1/component_controller.py','if self.projection_enabled:', 'P1 delegates entire frozen V9 step; P0 bypasses projection/mask/reserve/rollback'),('identification/v9_component_ablation_v1/component_controller.py','paused=bool(self.recovery_motion_hold or action.pause_transport)', 'Baseline pause is retained for H0; H is additional authority'),('v9/controller.py','hi[:5][~s.valid|~s.closure_permission|~known]', 'Novel/unavailable-contact closing restriction is one inseparable part of P'),('v9/controller.py',"reference_memory_rollback=modified", 'Reference adaptation rollback is part of P'),('identification/v9_component_ablation_v1/ablation_runtime.py',"if self.time!=0.", 'All variants use unchanged joint physical route and zero-step replacement')]:
 p=RUN/'sources'/relative;ls=p.read_text(encoding='utf-8-sig').splitlines();anchors.append(dict(file=str(p),line=next(i for i,s in enumerate(ls,1) if needle in s),source_sha256=manifest['research_source_sha256'][relative],reason=reason))
report=dict(schema='causal8_independent_audit_v1',created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,issues=issues,input_sha256={f:sha(RUN/f) for f in ['execution_complete.json','manifest.json','plan.json','trials.json','assignment_ledger.json','endpoint_execution_acceptance.json','source_snapshot_manifest.json']},independent_checker_sha256=sha(Path(__file__)),shared_independent_arithmetic_sha256=sha(PURE),source_checks=sources,records=records,pairing=pairs,common_pre_event_prefix=prefix,full16s_old_endpoints=endpoint,factorial_effects=effects,frozen_source_evidence=anchors,scope=dict(full=['8 complete16s records,256000 raw physical substeps and25608 control endpoints','83 snapshot/current identities, exact8 assigned/results/failure denominators and registered endpoint hashes','independent raw forceguard/stages, physical vs task-qualified release, legacy mission vs full requested wrist motion','independent V3 stage FIR, all numeric variation fields in eligible static/dynamic/holding windows, actual requested-motion tracking','7200 holding debug/treatment observations, additional-hold vs actual reference and rate authority, no claimed physical pressure forecast','4 endpoints x40 actual numeric fields over full16s bit-exact;6 within-environment pre9.3 x41physical/observation-field comparisons'],not_checked=['no new physics, QP, model fitting or response kernel','no full noise-packet reconstruction or hidden-state restart test','no separate novel-mask/reserve/rollback causal identification','no generalization outside two specified environment runs and one seen seed610212','palm/other contact has no invented material cap; native release remains unchanged']),physics_steps=0,production_or_old_evidence_modified=False,model_admitted=False,interpretation='Eight local deterministic paired development runs validate finite package contrasts and retained failures, not a population effect or a winning algorithm. Collision quality windows lacking physical support remain unavailable. Safe-looking quality cannot compensate mission/tracking/force failures; release-basis experiments are separate.')
with OUT.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
lines=['# Causal8 独立只读审计','',f'{sum(x["passed"] for x in checks)}/{len(checks)}项检查通过；问题{len(issues)}。8/8完整16s，83来源；无新物理、无冻结源修改。','', '|环境|变体|原mission / 完整运动mission|全程力限|持握局部超限 Ns|持握RMSE [m,rad,rad]|附加风险hold s|','|---|---|---|---|---|---|---|']
for scene in scenes:
 for variant in variants:
  r=records[scene+'/'+variant];lines.append(f'|{scene}|{variant}|{r["mission_success"]} / {r["full_motion_mission_success"]}|{r["safety"]["guard_pass"]}|{r["stages"]["policy_holding_transport"]["local_excess_integral_Ns"]:.9f}|{r["tracking"]["holding_motion_tracking"]["rmse"]}|{r["actual_exposure_and_toggle"]["additional_risk_hold_s"]:.3f}|')
lines+=['','P是完整projection + novel-contact mask + minimum reserve + adaptation rollback包；此矩阵不能隔离mask或rollback单独效果。H只开关额外风险暂停，原baseline暂停仍保留。处理改变真实运动后也会改变后续观测和动作，P/H交互不能当成固定外加波形的独立贡献。','', 'friction_drop的四变体全部通过原mission和力限；H1两个变体均未通过独立腕部跟踪，因此完整运动mission失败。collision全部力限失败；H0两个变体完成原mission与完整运动，H1两个变体因lift跟踪失败，腕跟踪也失败。P整包在H0和H1下对collision暴露的方向不同，不能合并为统一收益。','', '所有变体公共目标、preview、deadline和环境日程配对；公共任务未随暂停改写。原V9四个端点各40实际字段全16s逐位相同，失败端点保留。collision动态/全持握品质窗因物理支持缺失不可用，不把缺失当零或与friction不同覆盖均值混比。','', '仅两个指定环境、一个已见开发seed610212的有限因果对照；不把环境当独立总体抽样，不做总体显著性或新策略优越性声明。原native release和独立release-basis pilot未合并。','',f'JSON SHA256：`{sha(OUT)}`。']
MD.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),md=str(MD),checks=len(checks),passed=sum(x['passed'] for x in checks),issues=issues,records={k:dict(mission=x['mission_success'],full_motion=x['full_motion_mission_success'],guard=x['safety']['guard_pass'],holding_excess=x['stages']['policy_holding_transport']['local_excess_integral_Ns'],exposure=x['actual_exposure_and_toggle']) for k,x in records.items()},effects=effects),ensure_ascii=False),flush=True)

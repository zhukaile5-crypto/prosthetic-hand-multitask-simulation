"""Prespecified fixed train3 -> mixed4 diagnostics; review files only, no physics.
Pilot implementer/analysis author. Independent raw audit belongs to another reviewer.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN2=ROOT/'output/release_local_response_capsule_v2';RUN3=ROOT/'output/release_local_response_capsule_v3'
OUT=ROOT/'review/release_local_response_v3_superposition_v1.json'
AUDIT2=ROOT/'review/release_local_response_v2_offline_audit_v1.json'
PRIOR=ROOT/'review/release_local_response_v2_superposition_v1.json'
TRAIN=['zero','phi_slow','lift_slow'];HOLD=['mixed_50_50','mixed_50_100','mixed_100_50','mixed_100_100']
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
hashes={}
def sha(path):
 path=Path(path)
 if path not in hashes:hashes[path]=hashlib.sha256(path.read_bytes()).hexdigest()
 return hashes[path]
def error_stats(actual,predicted,mask,valid=None):
 residual=actual-predicted;available=np.isfinite(actual)&np.isfinite(predicted)
 if valid is not None:available&=valid
 available&=mask[:,None];count=available.sum(axis=0);rmse=[];maximum=[];mean=[]
 for j,n in enumerate(count):
  v=residual[available[:,j],j];rmse.append(None if not n else float(np.sqrt(np.mean(v**2))));maximum.append(None if not n else float(abs(v).max()));mean.append(None if not n else float(v.mean()))
 return dict(samples_per_axis=count.tolist(),expected_clock_samples=int(mask.sum()),rmse_per_axis=rmse,max_abs_per_axis=maximum,signed_mean_per_axis=mean)
def windows(origin):
 return [('cumulative_0_20ms',origin,origin+.02),('cumulative_0_40ms',origin,origin+.04),('cumulative_0_100ms',origin,origin+.1),('cumulative_0_200ms',origin,origin+.2),('later_20_40ms',origin+.02,origin+.04),('later_40_100ms',origin+.04,origin+.1),('later_100_200ms',origin+.1,origin+.2),('full_release_12_16s_DIAGNOSTIC_NOT_LOCAL_MODEL',12.,16.)]
def at(clock,time):
 i=int(np.argmin(abs(clock-time)))
 if abs(clock[i]-time)>1e-8:raise RuntimeError('Missing registered clock: '+str(time))
 return i
def describe():
 return dict(schema='release_local_response_three_stages_fixed_superposition_v1',role='pilot implementer and analysis author; not organizationally independent raw reviewer',training_per_origin=TRAIN,strict_holdout_per_origin=HOLD,origins_s=[.020,.150,.350],all_assigned_denominator=21,new_assigned_denominator=14,windows_relative_ms=[20,40,100,200],additional_disjoint_ms=[[20,40],[40,100],[100,200]],full_release_diagnostic_s=[12.,16.],formula='(y_m-y_zero) - a*(y_phi-y_zero) - b*(y_lift-y_zero); a,b=direct actual coordinate amplitudes at origin+20ms, no holdout output fit',phase='20ms endpoint is pre-full-command qend; first full-input physical end origin+20.5ms',terminal_gate='V3 summary.execution_complete true before any V3 branch data read',future_zero='OFFLINE_ORACLE_ONLY, never a deployable online free-evolution input',parameters_fitted=False,physics_steps=0,model_or_kernel_created=False,output=str(OUT))
def self_test():
 t=np.arange(401)*.0005;mask=(t>=0)&(t<=.02+1e-9);x=np.column_stack((t,2*t));zero=np.zeros_like(x)
 a=error_stats(x,x,mask);b=error_stats(x,x+np.array([.25,-.5]),mask)
 assert a['samples_per_axis']==[41,41] and a['max_abs_per_axis']==[0.,0.]
 assert np.allclose(b['max_abs_per_axis'],[.25,.5],rtol=0,atol=1e-14)
 valid=np.ones_like(x,dtype=bool);valid[:,1]=False;c=error_stats(x,zero,mask,valid)
 assert c['samples_per_axis'][1]==0 and c['rmse_per_axis'][1] is None
 assert at(t,.0205)==41 and at(t,.0200)==40
 return dict(passed=True,fixtures=['zero residual','known additive interaction','missing is not zero','first-full-input end differs from20ms boundary'],physics_steps=0)

def load_branch(run,stage,variant,origin,expected_trace):
 name='capsule__'+variant if stage=='phi_020ms_reference' else 'capsule__'+stage+'__'+variant
 folder=run/'runs'/name;raw=folder/'robot_physical_stream.npz';cmd=folder/'decoder_commands.npz';trace=folder/'record/trace.npz';result=folder/'result.json';meta=read(folder/'pilot_metadata.json')
 for path,key in [(raw,'robot_stream_sha256'),(cmd,'command_record_sha256'),(trace,'record_trace_sha256')]:
  if sha(path)!=meta[key]:raise RuntimeError('Pilot metadata input hash mismatch: '+str(path))
 if sha(trace)!=expected_trace:raise RuntimeError('Terminal trace hash mismatch')
 with np.load(raw,allow_pickle=False) as z:
  pt=z['physical_time_s'];groups=dict(robot_position25=z['position25'],robot_velocity25=z['velocity25'],physical_external_finger_pressure5_N=z['external_finger_pressure5_N'],offline_target_palm_other_pressure2_N=np.column_stack((z['palm_pressure_N'],z['other_hand_pressure_N'])));axes=z['axes25'].tolist()
 with np.load(cmd,allow_pickle=False) as z:inp={k:z[k] for k in ['control_time','physical_time','actual_phi_s','actual_command21','actual_phi_offset_s','actual_lift_offset_m','lift_clipped']}
 with np.load(trace,allow_pickle=False) as z:
  public_t=z['time'];normal=[];valid=[];aux=[];auxvalid=[];sample=[]
  for value in z['observation_json']:
   f=json.loads(str(value));normal.append(f['normal']);valid.append(f['valid']);v=f.get('auxiliary_pressure');aux.append([np.nan if v is None else v]);auxvalid.append([bool(f.get('auxiliary_valid',False)) and v is not None]);sample.append(f.get('sample_time',np.nan))
  metadata=json.loads(str(z['metadata']));forces=z['substep_skin_normal'];dur=z['substep_duration'];last=z['skin_normal'][-1]
  limits=np.asarray(metadata['limits']);budget=metadata['budget'];peak=np.maximum(forces.max(axis=0),last)
  numeric=dict(finger_peak_N=peak.tolist(),whole_peak_N=float(peak.max()),local_excess_integral_Ns=float(np.sum(np.maximum(forces-limits,0.)*dur[:,None])),total_excess_integral_Ns=float(np.maximum(forces.sum(axis=1)-budget,0.)@dur))
 metrics=read(result)['research_metrics'];safety=metrics['safety']
 if not np.allclose(peak,safety['finger_peak_N'],rtol=0,atol=1e-12) or abs(numeric['local_excess_integral_Ns']-safety['local_excess_integral_Ns'])>1e-12 or abs(numeric['total_excess_integral_Ns']-safety['total_excess_integral_Ns'])>1e-12:raise RuntimeError('Outcome table raw arithmetic disagrees with stored metric')
 numeric.update(reported_force_guard_pass=safety['guard_pass'],reported_physical_release=metrics['release']['physical_success'],reported_first_clearance_latency_s=metrics['release']['physical_clearance_latency_s'],reported_legacy_mission_success=metrics['mission_success'],reported_full_motion_mission_success=metrics['full_motion_mission_success'],release_metric_role='stored V3 first-clearance/mission outcomes; independent full metrics/late recontact audit is a separate reviewer')
 return dict(physical_time=pt,physical=groups,public_time=public_t,public=dict(public_observed_pressure5_N=np.array(normal,float),public_auxiliary_pressure1_N=np.array(aux,float)),valid=dict(public_observed_pressure5_N=np.asarray(valid,bool),public_auxiliary_pressure1_N=np.asarray(auxvalid,bool)),sample_time=np.array(sample,float),input=inp,axes25=axes,outcome=numeric,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [raw,cmd,trace,result,folder/'pilot_metadata.json']})

def stage_analysis(stage,run,origin,terminal_rows,pair_rows):
 selected=[r for r in terminal_rows if stage=='phi_020ms_reference' or r.get('stage')==stage];byvar={r['variant']:r for r in selected}
 invalid=[v for v in TRAIN+HOLD if v not in byvar or byvar[v]['status']!='complete']
 invalid_pairs=[r for r in pair_rows if (stage=='phi_020ms_reference' or r.get('stage')==stage) and not r['paired']]
 if invalid or invalid_pairs:return dict(stage=stage,status='UNAVAILABLE_INCOMPLETE_OR_UNPAIRED',assigned=7,unavailable_variants=invalid,invalid_pair_rows=invalid_pairs,training=TRAIN,strict_holdout=HOLD)
 data={v:load_branch(run,stage,v,origin,byvar[v]['trace_sha256']) for v in TRAIN+HOLD};base=data['zero'];zero=base['input'];physical=base['physical_time'];public=base['public_time'];full=origin+.02
 for v in TRAIN+HOLD:
  d=data[v]
  for field in ['physical_time','public_time']:
   if not np.array_equal(d[field],base[field]):raise RuntimeError('Output clock mismatch: '+stage+'/'+v)
  for field in ['control_time','physical_time']:
   if not np.array_equal(d['input'][field],zero[field]):raise RuntimeError('Actual input clock mismatch')
 pphi=data['phi_slow']['input']['actual_phi_s']-zero['actual_phi_s'];plift=data['lift_slow']['input']['actual_command21'][:,18]-zero['actual_command21'][:,18];formation=at(zero['control_time'],full)
 if abs(pphi[formation])<1e-12 or abs(plift[formation])<1e-12:raise RuntimeError('Missing actual finite input denominator')
 phase=dict(origin_s=origin,probe_snapshot_phase='before origin tick sensor/control; previous carrier is origin-minus5ms',quarter_start_commands_s=[origin+i*.005 for i in (1,2,3,4)],full_input_control_time_s=full,full_input_first_application_physical_time_s=float(zero['physical_time'][formation]),first_full_input_physical_end_s=full+.0005,warning='20ms endpoint precedes physical effect of the full command applied at that boundary; do not divide by that newly applied command as instantaneous gradient')
 results=[];costs=[]
 for v in TRAIN+HOLD:
  row=dict(variant=v,training_role='train' if v in TRAIN else 'heldout_mixed',**data[v]['outcome'])
  for key in ['whole_peak_N','local_excess_integral_Ns','reported_first_clearance_latency_s']:row[key+'_difference_vs_stage_zero']=None if row[key] is None or base['outcome'][key] is None else row[key]-base['outcome'][key]
  costs.append(row)
 for variant in HOLD:
  inp=data[variant]['input'];aphi=inp['actual_phi_s']-zero['actual_phi_s'];alift=inp['actual_command21'][:,18]-zero['actual_command21'][:,18]
  if not np.array_equal(aphi,inp['actual_phi_offset_s']) or not np.array_equal(alift,inp['actual_lift_offset_m']):raise RuntimeError('Actual-minus-zero input mismatch')
  a=float(aphi[formation]/pphi[formation]);b=float(alift[formation]/plift[formation]);res=np.column_stack((aphi-a*pphi,alift-b*plift));apptime=zero['physical_time'];local=(apptime>=origin-1e-9)&(apptime<=origin+.2+1e-9);allrelease=(apptime>=12-1e-9)&(apptime<=16+1e-9)
  divergent=np.flatnonzero(abs(res[:,1])>1e-12);floor=np.flatnonzero(inp['actual_command21'][:,18]==0.);tol_floor=np.flatnonzero(inp['actual_command21'][:,18]<=1e-12)
  record=dict(variant=variant,actual_input_amplitude_ratios=dict(phi=a,lift=b),actual_formation_amplitudes=dict(phi_s=float(aphi[formation]),lift_m=float(alift[formation])),input_profile_scaling_residual_local200ms_maxabs_s_m=abs(res[local]).max(axis=0).tolist(),input_profile_scaling_residual_fullrelease_maxabs_s_m=abs(res[allrelease]).max(axis=0).tolist(),first_lift_profile_nonproportional_control_time_s=None if not len(divergent) else float(zero['control_time'][divergent[0]]),first_lift_strict_bitzero_control_time_s=float(inp['control_time'][floor[0]]),first_lift_within1e12m_control_time_s=float(inp['control_time'][tol_floor[0]]),public_sample_clock_equal_to_zero=bool(np.array_equal(data[variant]['sample_time'],base['sample_time'],equal_nan=True)),windows={},points={})
  for kind,clock,groups in [('physical',physical,data[variant]['physical']),('public',public,data[variant]['public'])]:
   for group,y in groups.items():
    y0=base[kind][group];actual=y-y0;predicted=a*(data['phi_slow'][kind][group]-y0)+b*(data['lift_slow'][kind][group]-y0)
    valid=None if kind=='physical' else data[variant]['valid'][group]&base['valid'][group]&data['phi_slow']['valid'][group]&data['lift_slow']['valid'][group]
    for name,lo,hi in windows(origin):
     mask=(clock>=lo-1e-9)&(clock<=hi+1e-9);record['windows'].setdefault(name,dict(window_s=[lo,hi],outputs={}));record['windows'][name]['outputs'][group]=error_stats(actual,predicted,mask,valid)
    if kind=='physical':
     for name,point in [('20ms_pre_full_input_qend',full),('first_full_input_physical_end',full+.0005),('40ms',origin+.04),('100ms',origin+.1),('200ms',origin+.2)]:
      i=at(clock,point);record['points'].setdefault(name,dict(physical_end_s=float(clock[i]),outputs={}));record['points'][name]['outputs'][group]=dict(actual_delta=actual[i].tolist(),axis_superposition_delta=predicted[i].tolist(),residual=(actual[i]-predicted[i]).tolist())
  results.append(record)
 return dict(stage=stage,status='DESCRIPTIVE_COMPLETE_PAIRED',assigned=7,origin_s=origin,training=TRAIN,strict_holdout=HOLD,phase_contract=phase,axis_actual_full_formation_denominators=dict(phi_s=float(pphi[formation]),lift_reference_m=float(plift[formation])),axes25=base['axes25'],input_sha256={v:data[v]['input_sha256'] for v in data},outcomes=costs,holdout_results=results)

def execute(raw_audit=None):
 if OUT.exists():raise FileExistsError(OUT)
 # Gate before reading a single V3 branch/partial result. No waiting/polling inside.
 summary3=read(RUN3/'summary.json') if (RUN3/'summary.json').exists() else None
 if summary3 is None or summary3.get('execution_complete') is not True:raise RuntimeError('NOT_READY: complete terminal V3 summary required; no V3 branch data read')
 if summary3.get('planned')!=14 or len(summary3['results'])!=14:raise RuntimeError('Terminal assignment denominator is not14')
 summary2=read(RUN2/'summary.json');audit=read(AUDIT2)
 if not summary2['execution_complete'] or not summary2['passed'] or audit['issues'] or not all(c['passed'] for c in audit['checks']) or sha(RUN2/'summary.json')!=audit['input_sha256']['summary.json']:raise RuntimeError('Frozen reference audit/identity unavailable')
 plan3=read(RUN3/'plan.json');ref=plan3['frozen_reference']
 if ref['summary_sha256']!=sha(RUN2/'summary.json'):raise RuntimeError('Run V3 reference does not bind these old7')
 spec=[('phi_020ms_reference',RUN2,12.02,summary2),('phi_150ms',RUN3,12.15,summary3),('phi_350ms',RUN3,12.35,summary3)]
 stages=[stage_analysis(name,run,origin,s['results'],s['pairs']) for name,run,origin,s in spec]
 prior=read(PRIOR);old=stages[0]
 # Regression against the frozen earlier diagnostic, not additional independent proof.
 reference_reproduced=all(a['windows']==b['windows'] and a['points']==b['points'] for a,b in zip(old['holdout_results'],prior['holdout_results']))
 if not reference_reproduced:raise RuntimeError('Frozen .020 diagnostic changed under common analyzer')
 report=describe();report.update(created_utc=datetime.now(timezone.utc).isoformat(),status='descriptive_complete' if all(s['status']=='DESCRIPTIVE_COMPLETE_PAIRED' for s in stages) else 'contains_unavailable_stages_all_attempts_retained',checker_sha256=sha(Path(__file__)),frozen_020_diagnostic_reproduced_exact=reference_reproduced,terminal_completed_new=summary3['completed'],terminal_failures_new=summary3['failures'],terminal_sources_unchanged_reported=summary3['sources_unchanged'],source_audit_scope='analysis binds terminal manifest/snapshot hashes and each input; full244 source/current/prefix/raw audit belongs to independent reviewer',input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [RUN2/'summary.json',RUN2/'manifest.json',AUDIT2,PRIOR,RUN3/'summary.json',RUN3/'manifest.json',RUN3/'source_snapshot_manifest.json',RUN3/'plan.json']},independent_raw_audit=None if raw_audit is None else dict(path=str(raw_audit),sha256=sha(raw_audit)),stages=stages,limitations=['Future zero force/q is an offline paired oracle used solely to define response residual. An online model must obtain free evolution from then-public history/current phase; this artifact supplies no such forecaster.','No mixed output is fitted, no preferred action is selected, and no error/pressure clipping rescues the benchmark. All4 heldouts remain heldout at every origin.','Full12..16 windows combine changing contacts, path saturation and bottom-boundary input nonproportionality. They are not a fixed-gradient or2.2s local model.','All three zeros repeat one same native trajectory. Three origins/21 branches do not constitute21 independent physical subjects or population replications.','Robot q25/velocity25 are robot-only telemetry; current external motion/gap/contact identity remains unknown online. Physical palm/other are target-object-only offline categories; public aux retains its own validity/sample clock.','Raw whole-force maxima/integrals are recomputed for the benefit/cost table. Stored guard, release/mission numbers are clearly marked reported; independent complete V3/FIR/clearance/recontact acceptance is separate.','No predeclared numerical local-model acceptance threshold exists. Residuals and tradeoffs are descriptive; no kernel, safety certificate, online policy, force MPC or global model admission is produced.'])
 with OUT.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
 lines=['# 三个释放阶段的固定轴响应叠加诊断','','作者角色：pilot实施者及本分析器作者；完整raw/source/paired独审由另一reviewer负责。本结果使用未来zero作为离线对照oracle，不是在线free-evolution预测器。','','|origin|共同累计窗|4留出最大逐指压力误差范围 N|24角最大误差范围 rad|lift最大误差范围 m|','|---|---|---|---|---|']
 for s in stages:
  if s['status']!='DESCRIPTIVE_COMPLETE_PAIRED':continue
  for name in ['cumulative_0_20ms','cumulative_0_40ms','cumulative_0_100ms','cumulative_0_200ms','full_release_12_16s_DIAGNOSTIC_NOT_LOCAL_MODEL']:
   pressure=[];angular=[];lift=[]
   for h in s['holdout_results']:
    o=h['windows'][name]['outputs'];pressure.append(max(o['physical_external_finger_pressure5_N']['max_abs_per_axis']));q=o['robot_position25']['max_abs_per_axis'];angular.append(max(q[:24]));lift.append(q[24])
   rng=lambda x:f'{min(x):.9g}..{max(x):.9g}'
   lines.append(f'|{s["origin_s"]-12.:.3f}|{name}|{rng(pressure)}|{rng(angular)}|{rng(lift)}|')
 lines+=['','|origin|branch|raw peak N|Δpeak vs zero N|local excess N·s|Δexcess N·s|reported first-clear latency s|reported guard|','|---|---|---|---|---|---|---|---|']
 for s in stages:
  for row in s.get('outcomes',[]):lines.append(f'|{s["origin_s"]-12.:.3f}|{row["variant"]}|{row["whole_peak_N"]:.9g}|{row["whole_peak_N_difference_vs_stage_zero"]:.9g}|{row["local_excess_integral_Ns"]:.9g}|{row["local_excess_integral_Ns_difference_vs_stage_zero"]:.9g}|{row["reported_first_clearance_latency_s"]}|{row["reported_force_guard_pass"]}|')
 lines+=['','20ms端点仍是新满输入生效前的q-end；首次满输入physical-end分别12.0405、12.1705、12.3705。JSON另列点值、20–40/40–100/100–200ms分窗、全部逐轴误差、公开观测有效率/延迟钟和actual输入夹界后的比例残差。','','完整释放误差混合了接触改变和lowering floor，不能作为同一固定gradient。不能挑峰值最好的branch回填训练或宣称所有减速都优。三个zero是同一native参考的复现，不是三个独立population样本。','','JSON SHA256 `'+sha(OUT)+'`。']
 with OUT.with_suffix('.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
 print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),status=report['status'],stages=len(stages),parameters_fitted=False,physics_steps=0)))

if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--describe',action='store_true');g.add_argument('--self-test',action='store_true');g.add_argument('--execute',action='store_true');p.add_argument('--raw-audit',type=Path);a=p.parse_args()
 if a.describe:print(json.dumps(describe(),ensure_ascii=False,indent=2))
 elif a.self_test:print(json.dumps(self_test(),ensure_ascii=False))
 else:execute(a.raw_audit)

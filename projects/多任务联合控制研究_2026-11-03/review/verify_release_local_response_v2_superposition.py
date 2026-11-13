"""Fixed train3 axis-difference superposition against four untouched holdouts.
No parameter fitting, simulator imports, model artifact or kernel admission.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'output/release_local_response_capsule_v2';AUDIT=ROOT/'review/release_local_response_v2_offline_audit_v1.json';OUT=ROOT/'review/release_local_response_v2_superposition_v1.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if OUT.exists():raise FileExistsError(OUT)
audit=read(AUDIT)
if audit['issues'] or not all(c['passed'] for c in audit['checks']):raise RuntimeError('Full raw audit must pass first')
if sha(RUN/'summary.json')!=audit['input_sha256']['summary.json']:raise RuntimeError('Raw audit no longer binds this run')
plan=read(RUN/'plan.json');train=['zero','phi_slow','lift_slow'];hold=plan['strict_holdout'];data={};inputs={};files={}
for variant in train+hold:
 folder=RUN/'runs'/('capsule__'+variant);raw=folder/'robot_physical_stream.npz';commands=folder/'decoder_commands.npz';trace=folder/'record/trace.npz'
 for p in [raw,commands,trace]:
  rel=p.relative_to(RUN).as_posix()
  if sha(p)!=audit['records'][variant]['input_sha256'][rel.replace('/', '\\') if '\\' in next(iter(audit['records'][variant]['input_sha256'])) else rel]:raise RuntimeError('Retained audit input changed: '+str(p))
 with np.load(raw,allow_pickle=False) as z:
  physical_t=z['physical_time_s'];groups=dict(robot_position25=z['position25'],robot_velocity25=z['velocity25'],physical_external_finger_pressure5_N=z['external_finger_pressure5_N'],offline_target_palm_other_pressure2_N=np.column_stack((z['palm_pressure_N'],z['other_hand_pressure_N'])));axes25=z['axes25'].tolist()
 with np.load(commands,allow_pickle=False) as z:inp={k:z[k] for k in ['control_time','physical_time','actual_phi_s','actual_command21']}
 with np.load(trace,allow_pickle=False) as z:
  public_t=z['time'];encoded=z['observation_json'];normal=[];valid=[];aux=[];auxvalid=[];sample=[]
  for value in encoded:
   f=json.loads(str(value));normal.append(f['normal']);valid.append(f['valid']);a=f.get('auxiliary_pressure');aux.append([np.nan if a is None else a]);auxvalid.append([bool(f.get('auxiliary_valid',False)) and a is not None]);sample.append(f.get('sample_time',np.nan))
 public=dict(public_observed_pressure5_N=np.array(normal,float),public_auxiliary_pressure1_N=np.array(aux,float));masks=dict(public_observed_pressure5_N=np.asarray(valid,bool),public_auxiliary_pressure1_N=np.asarray(auxvalid,bool))
 data[variant]=dict(physical_time=physical_t,physical=groups,public_time=public_t,public=public,valid=masks,sample_time=np.array(sample,float));inputs[variant]=inp;files[variant]={str(p.relative_to(RUN)):sha(p) for p in [raw,commands,trace]}
for variant in train+hold:
 if not np.array_equal(data[variant]['physical_time'],data['zero']['physical_time']) or not np.array_equal(data[variant]['public_time'],data['zero']['public_time']):raise RuntimeError('Common clock mismatch')
 if not np.array_equal(inputs[variant]['control_time'],inputs['zero']['control_time']) or not np.array_equal(inputs[variant]['physical_time'],inputs['zero']['physical_time']):raise RuntimeError('Applied-input clock mismatch')
times=data['zero']['physical_time'];public_times=data['zero']['public_time'];origin=12.02;quarter_full=12.04
zero=inputs['zero'];profile_phi=inputs['phi_slow']['actual_phi_s']-zero['actual_phi_s'];profile_lift=inputs['lift_slow']['actual_command21'][:,18]-zero['actual_command21'][:,18]
formation=np.flatnonzero(abs(zero['control_time']-quarter_full)<1e-9)[0]
if abs(profile_phi[formation])<1e-12 or abs(profile_lift[formation])<1e-12:raise RuntimeError('Actual axis amplitudes are unavailable')

def error_stats(actual,predicted,mask,valid=None):
 residual=actual-predicted;available=np.isfinite(actual)&np.isfinite(predicted)
 if valid is not None:available&=valid
 available&=mask[:,None];count=available.sum(axis=0);rmse=[];maximum=[];mean=[]
 for j,n in enumerate(count):
  values=residual[available[:,j],j];rmse.append(None if not n else float(np.sqrt(np.mean(values**2))));maximum.append(None if not n else float(abs(values).max()));mean.append(None if not n else float(values.mean()))
 result=dict(samples_per_axis=count.tolist(),expected_clock_samples=int(mask.sum()),rmse_per_axis=rmse,max_abs_per_axis=maximum,signed_mean_per_axis=mean)
 return result

windows=[('cumulative_0_20ms',origin,origin+.02),('cumulative_0_40ms',origin,origin+.04),('cumulative_0_100ms',origin,origin+.1),('cumulative_0_200ms',origin,origin+.2),('later_20_40ms',origin+.02,origin+.04),('later_40_100ms',origin+.04,origin+.1),('later_100_200ms',origin+.1,origin+.2),('full_release_12_16s_DIAGNOSTIC_NOT_LOCAL_MODEL',12.,16.)]
points=[('20ms_pre_full_input_qend',origin+.02),('first_full_input_physical_end',12.0405),('40ms',origin+.04),('100ms',origin+.1),('200ms',origin+.2)]
results=[]
for variant in hold:
 inp=inputs[variant];actual_phi=inp['actual_phi_s']-zero['actual_phi_s'];actual_lift=inp['actual_command21'][:,18]-zero['actual_command21'][:,18]
 # Direct ratio of measured actuator-coordinate amplitudes, not a fitted
 # parameter. No holdout OUTPUT participates in this coefficient calculation.
 a=float(actual_phi[formation]/profile_phi[formation]);b=float(actual_lift[formation]/profile_lift[formation]);input_residual=np.column_stack((actual_phi-a*profile_phi,actual_lift-b*profile_lift));application_t=zero['physical_time'];local=(application_t>=origin-1e-9)&(application_t<=origin+.2+1e-9);full=(application_t>=12-1e-9)&(application_t<=16+1e-9)
 record=dict(variant=variant,actual_input_amplitude_ratios=dict(phi=a,lift=b),actual_formation_amplitudes=dict(phi_s=float(actual_phi[formation]),lift_m=float(actual_lift[formation])),input_profile_scaling_residual_local200ms_maxabs_s_m=abs(input_residual[local]).max(axis=0).tolist(),input_profile_scaling_residual_fullrelease_maxabs_s_m=abs(input_residual[full]).max(axis=0).tolist(),windows={},points={})
 for group_kind,clock,groups in [('physical',times,data[variant]['physical']),('public',public_times,data[variant]['public'])]:
  for group,actual_output in groups.items():
   y0=data['zero'][group_kind][group];p=data['phi_slow'][group_kind][group]-y0;l=data['lift_slow'][group_kind][group]-y0;actual=actual_output-y0;predicted=a*p+b*l
   common=None if group_kind=='physical' else data[variant]['valid'][group]&data['zero']['valid'][group]&data['phi_slow']['valid'][group]&data['lift_slow']['valid'][group]
   for name,lo,hi in windows:
    mask=(clock>=lo-1e-9)&(clock<=hi+1e-9);record['windows'].setdefault(name,dict(window_s=[lo,hi],outputs={}));record['windows'][name]['outputs'][group]=error_stats(actual,predicted,mask,common)
   if group_kind=='physical':
    for name,point in points:
     i=int(np.argmin(abs(clock-point)));record['points'].setdefault(name,dict(physical_end_s=float(clock[i]),outputs={}));record['points'][name]['outputs'][group]=dict(actual_delta=actual[i].tolist(),axis_superposition_delta=predicted[i].tolist(),residual=(actual[i]-predicted[i]).tolist())
 results.append(record)
public_clock_checks={v:bool(np.array_equal(data[v]['sample_time'],data['zero']['sample_time'],equal_nan=True)) for v in hold}
report=dict(schema='release_local_response_v2_fixed_superposition_v1',created_utc=datetime.now(timezone.utc).isoformat(),role='offline fixed-axis superposition diagnostic after full1077 raw audit; pilot implementer, not organizationally independent authorship',audit_sha256=sha(AUDIT),manifest_sha256=sha(RUN/'manifest.json'),plan_sha256=sha(RUN/'plan.json'),checker_sha256=sha(Path(__file__)),input_sha256=files,training=train,strict_holdout=hold,parameters_fitted=False,formula='heldout measured differential output y_m-y_0 compared with a*(y_phi-y_0)+b*(y_lift-y_0); a,b are direct actual input amplitude ratios at12.040, not estimated from holdout outputs',future_zero_role='Paired zero trajectory is used offline only to define differential response. Its future force/q is not an online input; this is not a deployable complete-response forecaster.',axis_units=dict(inputs=['native_path_seconds','lift_reference_m'],robot_position25=['rad']*24+['m'],robot_velocity25=['rad/s']*24+['m/s'],physical_external_finger_pressure5_N=['N']*5,public_observed_pressure5_N=['N']*5,public_auxiliary_pressure1_N=['N'],offline_target_palm_other_pressure2_N=['N','N']),axes25=axes25,public_sample_clock_equal=public_clock_checks,phase_contract=dict(origin_s=12.02,quarter_start_commands_s=[12.025,12.03,12.035,12.04],first_full_input_physical_end_s=12.0405,warning='20ms point12.040 is q/pressure after earlier held inputs, before the full command applied at that same boundary can affect dynamics; do not divide this output by that new input as an instantaneous gradient.'),holdout_results=results,limitations=['All four mixed branches stay excluded from any coefficient fitting, model selection or gain tuning. No parameters are optimized here.','Physical external finger pressure is five-channel raw truth output. Palm/other physical stream fields are TARGET-object-only offline labels, not complete external auxiliary skin pressure. Public auxiliary pressure has its own measured validity/delay clock.','Each cumulative/disjoint window is reported separately. Full12..16 residuals are nonstationary/contact-transition/boundary-clipping diagnostics, not an admitted entry-local fixed gradient or2.2s kernel.','Input lift profiles stop being globally scaled copies when their independently delayed lowering clocks hit the bottom boundary; full-release residual combines this with changed contacts.','Actual q25/velocity25 is robot-only telemetry. Current external-body motion, relative gap and target identity are not available online; signs of lift/progress/encoder change do not prove actual separation or pressure safety.','One capsule context, one seen physical/noise seed and one-sided prescribed profiles cannot identify separate individual finger/tendon gains or validate another progress/contact state.','No local numeric error acceptance threshold was predeclared; observed residuals are descriptive, not hard error bounds. No pressure clipping/deadzone is applied to rescue the linear benchmark.'],physics_steps=0,model_or_kernel_created=False,global_model_admitted=False,material_safety_certified=False)
with OUT.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
table=[]
for r in results:
 for name in ['cumulative_0_20ms','cumulative_0_40ms','cumulative_0_100ms','cumulative_0_200ms','full_release_12_16s_DIAGNOSTIC_NOT_LOCAL_MODEL']:
  o=r['windows'][name]['outputs'];q=o['robot_position25'];p=o['physical_external_finger_pressure5_N'];table.append(dict(variant=r['variant'],window=name,max_finger_pressure_abs_error_N=max(p['max_abs_per_axis']),max_angular_q_abs_error_rad=max(q['max_abs_per_axis'][:24]),lift_q_abs_error_m=q['max_abs_per_axis'][24]))
lines=['# Release V2 固定轴响应叠加诊断','', '先通过完整1077项raw复核，再用zero/phi_slow/lift_slow三个训练分支定义已测差分响应。四mixed全部留出。系数只取实际已执行phi/lift在12.040的幅值比例；不使用留出输出求系数，不拟合参数，不建立kernel或误差界。','', '|留出|共同累计窗|5指压力最大绝对叠加误差 N|24角最大误差 rad|lift误差 m|','|---|---|---|---|---|']
for x in table:lines.append(f'|{x["variant"]}|{x["window"]}|{x["max_finger_pressure_abs_error_N"]:.9g}|{x["max_angular_q_abs_error_rad"]:.9g}|{x["lift_q_abs_error_m"]:.9g}|')
lines+=['', 'JSON另列20ms点、12.0405首次满输入physical-end、40/100/200ms点以及20–40、40–100、100–200ms分窗；公开观测压力/aux有效率与延迟钟独立保留。20ms点的q-end不能与同一时刻新施加满输入当瞬时gain。','', '完整释放窗仅诊断：后续contact mode与lowering边界会改变，lift actual input不再是全轴分支的固定比例；终端峰值差不能当同一固定gradient。未来zero只在离线paired差分定义中使用，不能作在线预测输入。','', 'q25只代表机器人编码器。当前外物运动、真实相对gap和接触存续未知；lift/progress的符号不等于物体清离方向。physical palm/other字段是目标物离线输出，不能冒称全部外部aux皮肤压力。没有参数调优、物理重跑或控制器接入。','', f'JSON SHA256 `{sha(OUT)}`。']
OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),parameters_fitted=False,table=table),ensure_ascii=False),flush=True)

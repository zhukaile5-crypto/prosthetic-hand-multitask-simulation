"""Explain unexcited pressure windows and immutable pre-intervention peaks."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PRIMARY=ROOT/'review/release_local_response_v3_superposition_v1.json'
OUT=ROOT/'review/release_local_response_v3_exposure_addendum_v1.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if OUT.exists():raise FileExistsError(OUT)
d=json.loads(PRIMARY.read_text(encoding='utf-8'));records=[];files={}
for stage in d['stages']:
 if stage['status']!='DESCRIPTIVE_COMPLETE_PAIRED':raise RuntimeError('Complete paired stages required')
 origin=stage['origin_s']
 for outcome in stage['outcomes']:
  variant=outcome['variant'];matches=[(p,h) for p,h in stage['input_sha256'][variant].items() if p.endswith('robot_physical_stream.npz')]
  if len(matches)!=1:raise RuntimeError('Exactly one bound robot stream required')
  relative,expected=matches[0];p=ROOT/relative;actual=sha(p)
  if actual!=expected:raise RuntimeError('Input changed after primary analysis')
  files[relative]=actual
  with np.load(p,allow_pickle=False) as z:t=z['physical_time_s'];f=z['external_finger_pressure5_N']
  m=(t>=origin-1e-9)&(t<=origin+.02+1e-9);pre=t<=origin+1e-9;post=t>origin+1e-9
  index=np.unravel_index(np.argmax(f),f.shape);whole=float(f[index]);premax=float(f[pre].max());postmax=float(f[post].max())
  if whole!=outcome['whole_peak_N']:raise RuntimeError('Primary peak disagrees with bound raw physical stream')
  records.append(dict(stage=stage['stage'],variant=variant,origin_s=origin,window_first20ms_s=[origin,origin+.02],first20ms_physical_samples=int(m.sum()),first20ms_external_finger_peak_N=float(f[m].max()),first20ms_all_external_finger_pressure_exact_zero=bool(np.array_equal(f[m],np.zeros_like(f[m]))),first20ms_frames_any_finger_over_1e_minus10_N=int(np.count_nonzero(np.any(f[m]>1e-10,axis=1))),whole_peak_N=whole,whole_peak_first_physical_end_s=float(t[index[0]]),whole_peak_digit_index=int(index[1]),shared_prefix_peak_through_origin_N=premax,postorigin_peak_N=postmax,whole_peak_occurred_before_origin=bool(t[index[0]]<origin-1e-9)))
summary=[]
for name in ['phi_020ms_reference','phi_150ms','phi_350ms']:
 rows=[r for r in records if r['stage']==name];zero=next(r for r in rows if r['variant']=='zero')
 for r in rows:r['postorigin_peak_difference_vs_stage_zero_N']=r['postorigin_peak_N']-zero['postorigin_peak_N']
 summary.append(dict(stage=name,branches=len(rows),all_first20ms_external_pressure_zero=all(r['first20ms_all_external_finger_pressure_exact_zero'] for r in rows),shared_prefix_peak_N=zero['shared_prefix_peak_through_origin_N'],shared_prefix_peak_equal_across7=len({r['shared_prefix_peak_through_origin_N'] for r in rows})==1,postorigin_zero_peak_N=zero['postorigin_peak_N']))
result=dict(schema='release_local_response_v3_exposure_addendum_v1',created_utc=datetime.now(timezone.utc).isoformat(),role=d['role'],primary_analysis_sha256=sha(PRIMARY),checker_sha256=sha(Path(__file__)),input_sha256=files,summary=summary,records=records,physics_steps=0,parameters_fitted=False,interpretation=['At origin.350 all7 branches have all5 external finger pressure channels exactly zero at every41 physical-end samples in the first20ms. A zero superposition pressure residual there is an unexcited/free-finger-contact window, not validation of a force response gain. It does not assert every palm/aux surface is clear.','The native7.2350799517N peak occurs at12.334, before origin12.350. Every late branch inherits it exactly. The whole-window maximum cannot be reduced below this already-observed prefix by a later intervention; postorigin peak and whole exposure are distinct descriptive endpoints.','Postorigin is defined here by physical-end time strictly greater than origin, before inspecting outcomes. This is not a replacement for the full16s material guard or clearance/recontact metrics.','All21 raw streams are bound to the primary report. No branch selection, parameter fitting, new response kernel or online future-zero oracle is introduced.'])
with OUT.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
lines=['# 释放分阶段叠加：接触激励与已发生峰值补充','','.350阶段的7个分支，在首20ms的41个physical-end采样中五指外部压力逐值全为0。因此该窗0叠加误差不能验证受压响应模型，也不证明全部palm/aux都清离。','','7.2350799517N共同峰在12.334s已发生，先于12.350s干预起点；较晚动作不能抹去共同前缀峰。完整16s guard仍保留，另列post-origin峰仅作解释。','','|stage|branch|共同前缀峰 N|post-origin峰 N|相对本stage zero变化 N|','|---|---|---|---|---|']
for r in records:lines.append(f'|{r["stage"]}|{r["variant"]}|{r["shared_prefix_peak_through_origin_N"]:.9g}|{r["postorigin_peak_N"]:.9g}|{r["postorigin_peak_difference_vs_stage_zero_N"]:.9g}|')
lines+=['','本补充不选择最好动作，不修改主分析或冻结来源。post-origin窗口按physical-end严格晚于origin预先定义，不能替代完整安全/释放指标。','','JSON SHA256 `'+sha(OUT)+'`。']
with OUT.with_suffix('.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),summary=summary,physics_steps=0)))

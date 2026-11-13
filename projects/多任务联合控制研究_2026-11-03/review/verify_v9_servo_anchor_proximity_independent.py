"""Public robot observations/reference footprints only; no response evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SERVO=ROOT/'output/robot_servo_response_all15_v2';OUT=ROOT/'review/v9_servo_anchor_proximity_independent_v1.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if OUT.exists():raise FileExistsError(OUT)
anchors=[]
for path in sorted(SERVO.glob('*/checkpoint_*/model.json')):
 m=read(path);fp=read(path.with_name('actual_input_footprints.json'))
 mins=np.min(np.asarray([x['reference_min'] for x in fp.values()]),axis=0);maxs=np.max(np.asarray([x['reference_max'] for x in fp.values()]),axis=0)
 anchors.append(dict(id=m['context_id'],declared_shape=m['context_id'].split('/')[0],axes25=m['output_axes25'],q=np.array(m['robot_anchor25']),velocity=np.array(m['robot_velocity_anchor25']),base=np.array(m['base_reference8']),known_plus=np.array(m['known_plus8']),known_minus=np.array(m['known_minus8']),single_lower=np.array(m['measured_single_ramp_increment_lower8']),single_upper=np.array(m['measured_single_ramp_increment_upper8']),footprint_min=mins,footprint_max=maxs,model_sha256=sha(path),footprints_sha256=sha(path.with_name('actual_input_footprints.json'))))
assert len(anchors)==15
records=[];checks=[]
for mode in ['hold','transport']:
 for scene in ['nominal','friction_drop','fragile_conflict','collision']:
  folder=ROOT/f'output/development_v9_{mode}_v1/runs/{scene}__ratio_preview__610212';meta=read(folder/'metadata.json')
  with np.load(folder/'trace.npz',allow_pickle=False) as z:
   time=z['time'];mask=(time>=7.5-1e-9)&(time<12-1e-9);ids=np.flatnonzero(mask);rows=z['observation_json'];observations=[json.loads(str(rows[i])) for i in ids]
   q=np.array([x['raw_joint']+[x['lift']] for x in observations]);v24=np.array([x['raw_joint_velocity'] for x in observations]);ref=np.array([x['alpha']+x['executed_motion_reference'] for x in observations])
   # Robot-only encoder subset; no object columns or physical object parameters enter selection.
   qcheck=np.column_stack((z['qpos'][mask][:,meta['raw_joint_qpos_indices']],z['lift'][mask]));vcheck=z['qvel'][mask][:,meta['raw_joint_qpos_indices']]
  names=meta['raw_joint_names']+['transport_lift'];assert all(a['axes25']==names for a in anchors)
  consistent=bool(np.array_equal(q,qcheck) and np.array_equal(v24,vcheck));checks.append(dict(mode=mode,scenario=scene,public_robot_q_and_velocity24_match_recorded_robot_only_subset=consistent))
  delta20=ref[4:]-ref[:-4];summaries=[]
  for a in anchors:
   dq=abs(q-a['q']);dr=ref-a['base'];outside=np.maximum(a['footprint_min']-ref,0)+np.maximum(ref-a['footprint_max'],0)
   unknown=((delta20>1e-12)&~a['known_plus'])|((delta20< -1e-12)&~a['known_minus']);outside_single=(delta20<a['single_lower']-1e-12)|(delta20>a['single_upper']+1e-12)
   summaries.append(dict(context_id=a['id'],public_shape_match=meta['shape']==a['declared_shape'],initial_q25_delta=(q[0]-a['q']).tolist(),initial_public_angular_velocity24_delta=(v24[0]-a['velocity'][:24]).tolist(),initial_lift_velocity_comparison='unavailable_in_public_frame_do_not_substitute_zero',anchor_lift_velocity_m_s=float(a['velocity'][24]),initial_reference8_delta=dr[0].tolist(),q25_abs_distance_p50=np.median(dq,axis=0).tolist(),q25_abs_distance_max=np.max(dq,axis=0).tolist(),reference8_abs_distance_max=np.max(abs(dr),axis=0).tolist(),coordinatewise_recorded_absolute_reference_footprint_outside_fraction=(outside>1e-12).mean(axis=0).tolist(),coordinatewise_recorded_absolute_reference_footprint_max_excess=outside.max(axis=0).tolist(),all8_inside_coordinatewise_footprint_fraction=float(np.all(outside<=1e-12,axis=1).mean()),observed20ms_increment_unknown_sign_fraction=unknown.mean(axis=0).tolist(),observed20ms_increment_outside_single_ramp_range_fraction=outside_single.mean(axis=0).tolist()))
  examples=[]
  compatible=[a for a in anchors if a['declared_shape']==meta['shape']]
  for clock in [7.5,8.5,9.5,10.,11.,11.995]:
   i=int(np.argmin(abs(time[mask]-clock)));angular=min(compatible,key=lambda a:float(abs(q[i,:24]-a['q'][:24]).max()));lift=min(compatible,key=lambda a:float(abs(q[i,24]-a['q'][24])));alpha=min(compatible,key=lambda a:float(abs(ref[i,:5]-a['base'][:5]).max()))
   examples.append(dict(time_s=float(time[mask][i]),selection_scope='only declared public-shape-matching anchors; each criterion separately, no mixed-unit total score',angular_posture=dict(anchor=angular['id'],max24_rad=float(abs(q[i,:24]-angular['q'][:24]).max()),lift_separate_m=float(abs(q[i,24]-angular['q'][24])),velocity24_max_difference_rad_s=float(abs(v24[i]-angular['velocity'][:24]).max())),lift_only=dict(anchor=lift['id'],abs_m=float(abs(q[i,24]-lift['q'][24]))),alpha_reference_only=dict(anchor=alpha['id'],max5_normalized_closure=float(abs(ref[i,:5]-alpha['base'][:5]).max()),lift_reference_difference_m=float(abs(ref[i,5]-alpha['base'][5])),wrist_reference_max_difference_rad=float(abs(ref[i,6:]-alpha['base'][6:]).max()))))
  records.append(dict(mode=mode,scenario=scene,method='ratio_preview',seed=610212,public_shape=meta['shape'],window_s=[7.5,12.],samples=900,observation_clock='current public robot encoders/reference before action; tactile delay is unrelated to q25',metadata_sha256=sha(folder/'metadata.json'),trace_sha256=meta['trace_sha256'],anchors=summaries,transparent_examples=examples))
report=dict(schema='v9_servo_anchor_proximity_independent_v1',created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,passed=all(x['public_robot_q_and_velocity24_match_recorded_robot_only_subset'] for x in checks),all15_model_and_footprint_inputs=[{k:a[k] for k in ['id','model_sha256','footprints_sha256']} for a in anchors],axes25=anchors[0]['axes25'],input_axes8=['alpha_0','alpha_1','alpha_2','alpha_3','alpha_4','lift_reference_m','WRJ2_reference_rad','WRJ1_reference_rad'],units=dict(q25=['rad']*24+['m'],velocity24='rad/s',reference8=['normalized_closure']*5+['m','rad','rad']),records=records,scope=dict(full='8 strongestbaseline holding traces x900public frames x15 anchors: coordinatewise actual posture and reference footprints, observed20ms direction/range diagnostics; initial public24 angular velocity separate',examples='six fixed clocks per trace with separate same-public-shape angle/lift/alpha criteria',not_checked=['no servo prediction/rolling replay; parent owns it','no objectmass/friction/pose/targetID used in selection','no currentFrame lift velocity, so no fake full25 velocity match','no pressure/physics/control/model changes']),limits=['Coordinatewise union of recorded footprints is only a necessary descriptive comparison; it includes holes/unmeasured mixed sequences and certifies no joint input domain.','Small nearest-anchor distance is experimental proximity, not model admission, contact-history equivalence, measured-error validity or interpolation permission.','20ms input direction and amplitude do not cover long prefix/cumulative history; initial pose, velocity, references and direction remain separate.','Reference alpha is normalized closure, not joint radians; all comparisons preserve units, with no mixed-unit composite score.'],physics_steps=0,frozen_sources_modified=False,model_admitted=False)
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
lines=['# V9 strongest baseline 与15个servo锚点的独立距离摘要','', '8条ratio_preview、每条持握900个公开帧，逐坐标比较全部15个锚点。关节角rad、升降m、alpha归一化参考、腕参考rad、初始角速度rad/s分别保留；当前公开帧没有lift velocity，因此不补零、不宣称完整25速度接近。','', 'JSON逐anchor列出起始差、姿态p50/max、参考footprint超出比例/幅值、20ms实际增量方向/单ramp范围，并列7.5/8.5/9.5/10/11/11.995s透明例子。例子分别按同公开shape中的最大角差、lift差、alpha差选择，未合成跨单位总分。','', '所有接近程度均仅experimental。实际footprint的逐坐标并集不是mixed域；nearestar锚点不证明接触历史、速度、组合未来动作或误差界适用。没有读取物体mass/pose来选择，也没有servo预测、物理试验或旧源修改。','',f'JSON SHA256 `{sha(OUT)}`。']
OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),passed=report['passed'],traces=len(records),anchors=len(anchors),initial_examples=[dict(mode=x['mode'],scenario=x['scenario'],example=x['transparent_examples'][0]) for x in records]),ensure_ascii=False))

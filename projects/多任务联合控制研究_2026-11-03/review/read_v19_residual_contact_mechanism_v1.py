"""Saved V19 finger/auxiliary contact and motion evidence; no physical calls.
Only offline raw patch class reads is_target. It is never an online parameter.
Current raw200Hz packets and delayed public acquisitions remain separate.
"""
from pathlib import Path
import gzip,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v

def summary(patches):
    target=[p for p in patches if p['is_target']];other=[p for p in patches if not p['is_target']]
    aux=[p for p in patches if p.get('digit') is None or p['digit']<0]
    return dict(patches=len(patches),target_patches=len(target),non_target_patches=len(other),auxiliary_patches=len(aux),
        normal_by_digit_all_external_N=[sum(p['normal'] for p in patches if p['digit']==i) for i in range(5)],
        normal_by_digit_target_N=[sum(p['normal'] for p in target if p['digit']==i) for i in range(5)],
        auxiliary_normal_all_external_N=sum(p['normal'] for p in aux),auxiliary_target_normal_N=sum(p['normal'] for p in aux if p['is_target']),
        target_hand_normal_sum_N=sum(p['normal'] for p in target),non_target_hand_normal_sum_N=sum(p['normal'] for p in other),
        target_force_vector_palm_N=np.sum(np.asarray([p['force'] for p in target]),axis=0).tolist() if target else [0.,0.,0.],
        scope='offline is_target classification; vectors are palm-frame contact force convention, no online target identity')

def main():
    plan=read(ROOT/'review/v19_small_plan_v1.json');rows=[];inputs={};checks=[]
    for stage in plan['stages']:
        if not read(ROOT/stage['output']/'execution_complete.json')['execution_complete']:raise ValueError('pending')
        if stage['assignment']['receiver_policy']=='blind':continue
        a=stage['assignment'];run=ROOT/stage['output']/'runs'/f"{a['scenarios'][0]}__{a['methods'][0]}__{a['seeds'][0]}";dm=read(run/'release_decision_manifest.json')
        with gzip.open(run/dm['file'],'rt',encoding='utf-8') as f:records=[json.loads(x) for x in f]
        for p in (run/'trace.npz',run/'metadata.json',run/dm['file']):inputs[str(p)]=sha(p)
        keys=('time','normal','skin_normal','palm','hand_other','all_hand_object','support','wrist_position','lift','lift_ref','observation_json','raw_patches_json','receiver_measurement_json')
        with np.load(run/'trace.npz',allow_pickle=False) as z:data={k:z[k] for k in keys}
        t=data['time'];idx=np.flatnonzero((t>=12.5)&(t<=16.));public={int(i):json.loads(str(data['observation_json'][i])) for i in idx};raw={int(i):json.loads(str(data['raw_patches_json'][i])) for i in idx}
        entry=np.asarray(records[0]['current_reference21']);bytime={q['time_s']:q for q in records};shots=[]
        for when in (15.5,16.):
            i=round(when/.005);q=bytime[when];f=public[i];g=q['governor'];acq=f['sample_time'];ai=int(round(acq/.005))
            raw_acq=raw.get(ai)
            if raw_acq is None:raw_acq=json.loads(str(data['raw_patches_json'][ai]))
            ex=np.asarray(q['execution_receipt']['actual_reference21']);current=np.asarray(q['current_reference21']);lo=np.asarray(q['actual_reference_lower21']);hi=np.asarray(q['actual_reference_upper21'])
            derived=float(np.maximum(np.asarray(f['normal']),0).sum()+max(0.,f['auxiliary_pressure']))
            shots.append(dict(time_s=when,phase=q['phase'],gamma=q['qp']['gamma'],gamma_domain=q['gamma_domain'],
                public=dict(hand_acquisition_s=acq,hand_age_s=f['age'],finger_normal_all_external_N=f['normal'],finger_valid=f['valid'],auxiliary_pressure_N=f['auxiliary_pressure'],auxiliary_wrench6=f['auxiliary_wrench'],auxiliary_valid=f['auxiliary_valid'],auxiliary_age_s=f['auxiliary_age'],
                    derived_hand_low_pressure_N=derived,logged_hand_low=q['public_process']['hand_low_evidence'],skin_patches=f['skin_patches'],public_auxiliary_patch_count=sum(p.get('digit') is None for p in f['skin_patches']),
                    R_H=q['public_process']['raw_coacquired_loads'],receiver_surface=f['receiver_surface']),
                current_200Hz_target_truth=dict(time_s=float(t[i]),normal5_N=data['normal'][i],palm_N=data['palm'][i],hand_other_N=data['hand_other'][i],all_hand_target_normal_N=data['all_hand_object'][i],all_external_digit_normal5_N=data['skin_normal'][i],target_world_support_normal_N=data['support'][i],raw_patch_summary=summary(raw[i]),raw_patches=raw[i]),
                acquisition_200Hz_raw_truth=dict(time_s=float(t[ai]),raw_patch_summary=summary(raw_acq),scope='sample-time raw class coexistence only; no forced one-to-one public/raw patch correspondence'),
                lift=dict(q_encoder_m=g['q_m'],actual_velocity_m_s=g['actual_velocity_m_s'],current_reference_m=current[18],actual_committed_reference_m=ex[18],requested_velocity_m_s=g['requested_velocity_m_s'],governed_velocity_intent_m_s=g['governed_velocity_intent_m_s'],equivalent_velocity_at_current_q_m_s=g['equivalent_velocity_at_current_q_m_s'],braking_intent=g['braking_intent_active'],reason=g['reason'],TASK_interval_m=[lo[18],hi[18]],motion_reference_m=q['motion_reference21'][18]),
                wrist=dict(entry_reference_rad=entry[19:],current_reference_rad=current[19:],actual_reference_rad=ex[19:],actual_reference_minus_entry_rad=ex[19:]-entry[19:],actual_joint_position_rad=data['wrist_position'][i],actual_joint_velocity_rad_s=np.asarray(f['robot_velocity25'])[:2],original_mechanical_rate_interval_rad=np.c_[q['original_mechanical_rate_lower21'][19:],q['original_mechanical_rate_upper21'][19:]],actual_TASK_interval_rad=np.c_[lo[19:],hi[19:]],permission_width_rad=hi[19:]-lo[19:],branch_interval_rad=np.c_[q['branch']['reference_lower21'][19:],q['branch']['reference_upper21'][19:]],soft_motion_reference_rad=q['motion_reference21'][19:]),
                contact_objective=dict(J_patch_indices=q['contact_intents']['patch_indices'],omitted=q['contact_intents']['omitted'],known_digits=q['contact_intents']['normal_geometry_known_digits'],declared_lift_velocity_target_m_s=q['contact_intents'].get('declared_lift_velocity_target_m_s'))))
            checks.append(dict(stage=stage['name'],time_s=when,hand_low_is_sum_of_public_digits_plus_aux=bool(np.float64(derived).tobytes()==np.float64(q['public_process']['hand_low_evidence']['pressure_N']).tobytes())))
        window=dict(samples=len(idx),public_auxiliary_patch_frames=sum(any(p.get('digit') is None for p in f['skin_patches']) for f in public.values()),raw_auxiliary_patch_frames=sum(any(p.get('digit') is None or p['digit']<0 for p in ps) for ps in raw.values()),
            public_auxiliary_pressure_max_N=max(f['auxiliary_pressure'] for f in public.values()),raw_auxiliary_pressure_max_N=max(summary(ps)['auxiliary_normal_all_external_N'] for ps in raw.values()),
            phase_counts={phase:sum(q['phase']==phase for q in records if q['time_s']>=12.5) for phase in sorted({q['phase'] for q in records if q['time_s']>=12.5})})
        rows.append(dict(stage=stage['name'],snapshots=shots,last3p5_window=window))
    source_keys=('v19_contact_reference/adapter.py','v19_contact_reference/supervisor.py','v18_effort_transfer/supervisor.py','v17_contact_transfer/supervisor.py','foundation/hand_plant.py')
    out=ROOT/'review/v19_residual_contact_mechanism_v1.json'
    if out.exists():raise FileExistsError(out)
    report=dict(schema='v19_residual_contact_and_motion_mechanism_v1',passed=all(c['hand_low_is_sum_of_public_digits_plus_aux'] for c in checks),checks=checks,rows=rows,source_sha256={k:plan['source_sha256'][k] for k in source_keys},input_sha256=inputs,reader_sha256=sha(__file__),
        scope=dict(terminal_runs=3,selected_times_s=[15.5,16.],public_and_raw_auxiliary_presence_window_s=[12.5,16.],frames_each_window=701,models_or_physics=0),
        limitations=['Public pressure/patches are delayed acquisition-labelled all-external measurements; raw target truth is offline, separately labelled at acquisition/read times.','Nonzero noisy auxiliary wrench does not imply nonzero auxiliary pressure or an actual auxiliary contact.','Normal sum is contact compression, not hand gravity support; R/H are signed public coacquired proxies, not true target weight.','Persistent finger contact plus no positive retreat is evidence of the selected closed-loop state, not proof that a particular alternative retreat would clear safely.'])
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# V19末段接触与退让机制证据','', '|条目|时刻|阶段|公开TH/FF N|公开aux压力N|离线目标手压力N|请求/等效/实测升降速度m/s|腕权限宽度rad|','|---|---:|---|---|---:|---:|---|---|']
    for row in rows:
        for s in row['snapshots']:
            f=s['public'];g=s['lift'];w=s['wrist'];lines.append(f"|{row['stage']}|{s['time_s']}|{s['phase']}|{f['finger_normal_all_external_N'][:2]}|{f['auxiliary_pressure_N']:.8g}|{s['current_200Hz_target_truth']['all_hand_target_normal_N']:.8g}|{g['requested_velocity_m_s']:.8g}/{g['equivalent_velocity_at_current_q_m_s']:.8g}/{g['actual_velocity_m_s']:.8g}|{np.asarray(w['permission_width_rad']).tolist()}|")
    lines+=['','两条失败在15.5/16的残压需按表与JSON逐指识别；接触身份仅离线。低手压包含aux的代码事实不能代替本次aux是否实际接触的证据。','', '源码因果边界：手低压确认才转SEPARATE；UNSUPPORTED_EXIT中原OPENγ固定1，而默认升降请求仍0、到deadline governor以当前q作制动参考。观测到未清离且指接触持续时，该路径未主动请求正向退让；这说明任务图可能缺少在持续接触下寻找清离运动的任务，但没有反事实证明某方向即可安全清离。','', '腕权限区间有宽度是允许微动；motion目标仍entry腕，实际参考可因软目标偏移。它既非完全锁腕，也不是主动清离腕目标。','', '无新模型、FK、物理或参数调整；原任务/安全/交付标签不变。']
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],out=str(out),sha256=sha(out),rows=[dict(stage=r['stage'],window=r['last3p5_window'],snapshots=[dict(t=s['time_s'],phase=s['phase'],finger=s['public']['finger_normal_all_external_N'],aux=s['public']['auxiliary_pressure_N'],raw_target=s['current_200Hz_target_truth']['all_hand_target_normal_N'],lift=s['lift'],wrist_ref_delta_entry=s['wrist']['actual_reference_minus_entry_rad']) for s in r['snapshots']]) for r in rows]),ensure_ascii=False,default=lambda x:plain(x)));return 0
if __name__=='__main__':raise SystemExit(main())

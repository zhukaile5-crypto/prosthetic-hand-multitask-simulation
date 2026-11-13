"""Offline first-development release diagnosis; never imports a simulator.

Physical force peaks come only from saved 0.5ms substeps. Saved 5ms patches
and kinematics are adjacent context, never reconstructed peak-time forces.
Target flags are used for evaluation attribution only, never policy inputs.
"""
from pathlib import Path
import hashlib,json
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'output/development_v1'
OUT=ROOT/'review/release_diagnosis_v1.json'
MD=ROOT/'review/release_diagnosis_v1.md'
DIGITS=['thumb','index','middle','ring','little']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def intervals(time,duration,mask):
    indices=np.flatnonzero(mask)
    if not len(indices):return []
    splits=np.flatnonzero(np.diff(indices)>1)+1
    return [dict(start_s=float(time[group[0]]),end_s=float(time[group[-1]]+duration[group[-1]]),
                 duration_s=float(duration[group].sum()),samples=len(group)) for group in np.split(indices,splits)]


def first_clear(time,force,start,dwell=.2):
    since=None
    for k in np.flatnonzero(time>=start-1e-9):
        if force[k]<=.2:
            if since is None:since=float(time[k])
            if time[k]-since>=dwell-1e-9:return dict(first_clear_s=since,confirmation_s=float(time[k]),latency_s=float(time[k]-start))
        else:since=None
    return None


def near_packet(saved,times,index,digit):
    patches=json.loads(str(saved['raw_patches_json'][index]))
    selected=[]
    for original in patches:
        if original['digit']!=digit:continue
        p=dict(original);normal=np.asarray(p['normal_dir']);velocity=np.asarray(p['slip_velocity'])
        p['normal_relative_speed_mps']=float(normal@velocity)
        p['tangential_relative_speed_mps']=float(np.linalg.norm(velocity-normal*(normal@velocity)))
        selected.append(p)
    return dict(time_s=float(times[index]),patches=selected)


def analyze(path):
    # Cache each compressed array once: repeated NpzFile string indexing would
    # decompress the complete contact packet series on every inspected tick.
    needed={'metadata','time','substep_time','substep_duration','substep_skin_normal','substep_normal',
        'substep_support','substep_all_hand_object','position','lift','skin_normal','normal','support',
        'all_hand_object','raw_patches_json','qpos','alpha','executed_motion_reference',
        'substep_penetration','substep_world_penetration','ctrl_before','ctrl'}
    with np.load(path,allow_pickle=False) as saved:
        z={name:saved[name] for name in needed}
        meta=json.loads(str(z['metadata']));start=float(meta['mechanical_release_at'])
        t=z['time'];st=z['substep_time'];duration=z['substep_duration']
        skin=z['substep_skin_normal'];target=z['substep_normal'];support=z['substep_support']
        all_hand=z['substep_all_hand_object'];mask=st>=start-1e-9
        indices=np.flatnonzero(mask);flat=int(np.argmax(skin[indices]));k=int(indices[flat//5]);digit=flat%5
        peak_time=float(st[k]);peak_force=float(skin[k,digit]);nearest=int(np.argmin(abs(t-(peak_time+duration[k]/2))))
        before=np.flatnonzero((t>=peak_time-.05-1e-9)&(t<=peak_time+1e-9))
        velocities=np.gradient(z['position'],t,axis=0)
        lift_velocity=np.gradient(z['lift'],t)
        first_support=np.flatnonzero(mask&(support>.05))
        limits=np.asarray(meta['limits']);budget=float(meta['budget'])
        terminal_safe=bool(np.all(z['skin_normal'][-1]<=limits+.02) and z['skin_normal'][-1].sum()<=budget+.02)
        guard=lambda select:bool(np.all(skin[select]<=limits+.02) and np.all(skin[select].sum(axis=1)<=budget+.02))
        macro_mask=t>=start-1e-9
        near_indices=np.flatnonzero(abs(t-peak_time)<=.012)
        near=[near_packet(z,t,j,digit) for j in near_indices]
        packet_hits=[p for packet in near for p in packet['patches']]
        all_little=[]
        for j in np.flatnonzero((t>=start-1e-9)&(t<=start+1.5)):
            for patch in json.loads(str(z['raw_patches_json'][j])):
                if patch['digit']==4 and patch['normal']>.02:
                    all_little.append(dict(time_s=float(t[j]),hand_geom=int(patch['hand_geom']),
                        external_geom=int(patch['external_geom']),is_target=bool(patch['is_target']),
                        normal_N=float(patch['normal'])))
        qnames=meta['raw_joint_names'];qindices=meta['raw_joint_qpos_indices'];qpos=z['qpos']
        named={name:np.asarray(qpos[:,index]) for name,index in zip(qnames,qindices)}
        joint_names=[name for name in qnames if name.startswith(('rh_TH','rh_LF'))]
        event_times=[start,start+.1,start+.2,start+.3,start+.4,start+.6,start+.8,start+1.4,start+2.2]
        checkpoints=[]
        for time in event_times:
            j=int(np.argmin(abs(t-time)))
            checkpoints.append(dict(time_s=float(t[j]),elapsed_s=float(t[j]-start),
                object_position_m=z['position'][j].tolist(),object_velocity_mps=velocities[j].tolist(),
                lift_m=float(z['lift'][j]),lift_velocity_mps=float(lift_velocity[j]),
                executed_motion_reference=z['executed_motion_reference'][j].tolist(),
                alpha=z['alpha'][j].tolist(),object_world_support_N=float(z['support'][j]),
                saved_actuator_control_before=z['ctrl_before'][j].tolist(),saved_actuator_control_after=z['ctrl'][j].tolist(),
                hand_object_force_N=float(z['all_hand_object'][j]),
                finger_object_force_N=z['normal'][j].tolist(),joints_rad={name:float(named[name][j]) for name in joint_names}))
        little_episodes=intervals(st,duration,mask&(target[:,4]>.2))
        maximum_episode=max((e['duration_s'] for e in intervals(st,duration,mask&(skin[:,digit]>limits[digit]+.02))),default=0.)
        timing=dict(release_start_s=start,first_world_support_s=float(st[first_support[0]]) if len(first_support) else None,
            peak_after_first_world_support_s=float(peak_time-st[first_support[0]]) if len(first_support) else None,
            complete_hand_clear=first_clear(st,all_hand,start),
            thumb_clear=first_clear(st,target[:,0],start),
            last_hand_object_contact_s=float(st[np.flatnonzero(mask&(all_hand>.2))[-1]]) if np.any(mask&(all_hand>.2)) else None)
        return dict(scenario=meta['scenario']['name'],shape=meta['shape'],method=meta['method'],seed=meta['seed'],
            trace_path=str(path.relative_to(ROOT)),limits_N=limits.tolist(),budget_N=budget,
            source_revision='development_v1; seen diagnostic only',task_failure=meta.get('failure_reason'),
            mechanics=dict(common_release_rate=meta['config']['release_open_rate'],
                release_lift_reference_m=float(z['executed_motion_reference'][int(np.argmin(abs(t-start))),0]),
                lower_duration_s=.6,thumb_tuck_duration_s=.8,thumb_tuck_hold_until_s=1.4,thumb_extension_end_s=2.2),
            release_force_guard=bool(guard(mask) and terminal_safe),holding_force_guard=guard((st>=7.5-1e-9)&(st<start-1e-9)),
            whole_force_guard=bool(guard(np.ones(len(st),bool)) and terminal_safe),
            release_peak=dict(force_N=peak_force,digit=digit,digit_name=DIGITS[digit],
                substep_interval_s=[peak_time,float(peak_time+duration[k])],elapsed_s=peak_time-start,
                object_same_digit_N=float(target[k,digit]),non_target_same_digit_N=float(skin[k,digit]-target[k,digit]),
                exact_all_hand_object_N=float(all_hand[k]),exact_object_world_support_N=float(support[k]),
                exact_hand_object_penetration_m=float(z['substep_penetration'][k]),
                exact_object_world_penetration_m=float(z['substep_world_penetration'][k]),
                nearest_saved_time_s=float(t[nearest]),nearest_saved_skin_N=float(z['skin_normal'][nearest,digit]),
                sampled_release_digit_peak_N=float(z['skin_normal'][macro_mask,digit].max()),
                nearest_object_position_m=z['position'][nearest].tolist(),
                nearest_object_velocity_mps=velocities[nearest].tolist(),nearest_lift_m=float(z['lift'][nearest]),
                nearest_lift_velocity_mps=float(lift_velocity[nearest]),nearest_alpha=z['alpha'][nearest].tolist(),
                object_speed_50ms_before_peak_max_mps=float(np.linalg.norm(velocities[before],axis=1).max()) if len(before) else None,
                thumb_and_little_joints_nearest_rad={name:float(named[name][nearest]) for name in joint_names},
                exact_peak_patch_available=False,adjacent_200Hz_patch_packets=near,
                adjacent_matching_patch_count=len(packet_hits),longest_limit_excess_episode_s=maximum_episode),
            timing=timing,little_finger_contact_episodes=little_episodes,
            saved_little_patch_summary=dict(count=len(all_little),geom_counts=dict(Counter(str(p['hand_geom']) for p in all_little)),
                target_patch_count=sum(p['is_target'] for p in all_little)),
            checkpoints=checkpoints,
            confidence='Peak force/target attribution/table support exact saved substep; all pose/joint/patch context at 200Hz, not exact peak microcontact reconstruction')


def main():
    if OUT.exists() or MD.exists():raise FileExistsError('Keep previous release diagnoses')
    paths=sorted((INPUT/'runs').glob('*/trace.npz'))
    if len(paths)!=18:raise ValueError('Expected complete first 18-run development matrix')
    core=[ROOT/name for name in ['coordinator.py','coordinator_model.py','runtime.py','evaluation.py']]
    before={str(p.relative_to(ROOT)):sha(p) for p in [*paths,*core]}
    records=[analyze(path) for path in paths]
    after={str(p.relative_to(ROOT)):sha(p) for p in [*paths,*core]}
    if before!=after:raise RuntimeError('A release input or frozen core source changed')
    summary=[]
    for method in ['ratio','unified','joint']:
        rows=[r for r in records if r['method']==method]
        summary.append(dict(method=method,trials=len(rows),release_guards=sum(r['release_force_guard'] for r in rows),
            whole_guards=sum(r['whole_force_guard'] for r in rows),
            largest_release_peak_N=max(r['release_peak']['force_N'] for r in rows),
            little_finger_peak_trials=sum(r['release_peak']['digit']==4 for r in rows),
            peak_packet_missing_trials=sum(not r['release_peak']['adjacent_matching_patch_count'] for r in rows)))
    result=dict(schema='first_development_offline_release_diagnosis_v1',passed=True,
        scope='Existing first 18 trials only; no new dynamics, controller/runner/evaluator/protocol edit or force reconstruction',
        summary=summary,records=records,input_sha256_before=before,input_sha256_after=after,
        inputs_and_core_unchanged=True,script_sha256=sha(Path(__file__)),
        attribution='Saved object/target tags only for offline diagnosis, not proposed control inputs',
        timing_note='Substep time labels integration intervals; 5ms patches can entirely miss a sub-ms contact.')
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 首18条开发轨迹：主动释放阶段离线诊断','',
        '只读既有数据，没有新仿真。原始0.5ms力峰和5ms关节/位姿/接触包分开保留；不把恢复几何后的mj_forward受力当作原始峰。',
        '', '|场景|方法|释放峰 N|指|释放后 s|同指物体力 N|物体世界支撑 N|邻近5ms接触包数|释放保护|',
        '|---|---|---:|---|---:|---:|---:|---:|---|']
    for r in records:
        x=r['release_peak']
        lines.append(f"|{r['scenario']}|{r['method']}|{x['force_N']:.4f}|{x['digit_name']}|{x['elapsed_s']:.4f}|{x['object_same_digit_N']:.4f}|{x['exact_object_world_support_N']:.4f}|{x['adjacent_matching_patch_count']}|{'通过' if r['release_force_guard'] else '未通过'}|")
    lines += ['', '## 证据指向的生命周期问题','',
        '非robustness的15条轨迹，最大释放指压都在小指，峰发生时同指object-only力等于全皮肤力，且物体已与世界支撑接触。故这些峰属于物体—小指短促接触，不是单独小指撞台。很多接触在相邻5ms包中完全消失，不能从宏观图上未见接触断言没有冲击。',
        '胶囊体unified/joint约0.316s的峰伴落台冲击/较大物体世界接触压入；ratio胶囊体约0.58s及部分其他方法较晚的峰发生在对象已下落、手继续降低且手指尚未完全张开时。精确首落台与峰间隔、前50ms物速、实际TH/LF关节和alpha均在JSON逐局保留，用于区分首次落台附近冲击与已落台后的清离再碰。',
        '最大峰不是拇指直接施压；拇指轨迹对物体姿态的间接作用没有被这些观察性记录独立分离，不能据此断言改一个拇指角即可解决。',
        '公共路径同时开始卸力和0.6s手架下降，四指按0.8/s打开，拇指0.8s内收、保持到1.4s、2.2s完成伸展。下降结束早于多指实际清离，路径没有用实测压力或分离速度进行事件协调；固定时间表无法防止自然下落物体与正在撤离的指节再接触。',
        '', '## 后续整体闭环清离设计','',
        '释放请求立即取得最高权限并卸载，不能继续持握或先放置来替代任务。采用“卸载—受约束清离—防再接触确认”的统一混合任务：以全部皮肤压力、局部相对速度、已知手部运动学和被估计物体占据域/不确定度，联合选择实际可用的闭合、升降和腕部运动；接触分离和安全余量决定阶段切换，不按0.6/0.8秒硬切。',
        '平台目前只有升降和两腕轴，不能假设已有任意横向退手。若需要新的指节清离原语，应在新执行适配中公开权限并共同物理验收，不能在控制器偷偷覆盖关节。卸力期间仍允许物体自然脱离，不以承重放回台面过关。',
        '由于部分峰小于5ms，接触后的低频限力反馈不足以消除首次冲击；需要清离前的几何/相对速度预测和压力变化约束。完整力守卫继续覆盖释放及后续16秒，不滤峰、不略去落台或完成后再接触；所有手部接触清除驻留在再接触后重计。',
        '本诊断仅为已见开发首轮机制证据，不是新闭环方案的性能验证。核心v2.1及已有18条记录前后哈希未变。']
    MD.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,records=len(records),summary=summary,json=str(OUT),markdown=str(MD)),ensure_ascii=False))


if __name__=='__main__':main()

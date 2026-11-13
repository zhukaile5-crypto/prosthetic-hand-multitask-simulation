"""V20 saved exit/motion mechanism evidence. NO solver/model/FK/physics calls.
All3 already-terminal candidates retained. Not an independent proof of the
task_solver authored by this reviewer. Offline object/contact truth is never
used to construct an online parameter. Only saved matrices are factorized.
"""
from pathlib import Path
from collections import Counter
import gzip,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def stats(a):
    a=np.asarray(a,float)
    return None if not a.size else dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()))
def matrix(value,cols):
    a=np.asarray(value,float)
    if a.size==0 and a.shape==(0,):return a.reshape(0,cols)
    if a.ndim!=2 or a.shape[1]!=cols:raise ValueError('declared matrix shape lost/malformed')
    return a

def nullspace(a,n):
    if n==0:return 0,np.empty((0,0))
    if not len(a):return 0,np.eye(n)
    u,s,vt=np.linalg.svd(a,full_matrices=True);tol=max(a.shape)*np.finfo(float).eps*(s[0] if len(s) else 0.)
    rank=int(sum(s>tol));return rank,vt[rank:].T

def positive_episodes(mask,starts,ends,force,dt):
    ids=np.flatnonzero(mask);groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []
    return [dict(start_s=float(starts[g[0]]),end_s=float(ends[g[-1]]),samples=len(g),peak_N=float(force[g].max()),impulse_Ns=float(force[g]@dt[g])) for g in groups]

def physics_point(d,wanted):
    i=min(len(d['interval_end_s'])-1,int(np.searchsorted(d['interval_end_s'],wanted)))
    v=d['object_free_qvel6_end'][i];q=d['object_free_qpos7_end'][i]
    return dict(requested_time_s=float(wanted),interval_start_s=float(d['interval_start_s'][i]),interval_end_s=float(d['interval_end_s'][i]),
        cached_target_hand_N=float(d['target_hand_normal_N'][i]),cached_target_receiver_top_N=float(d['target_receiver_top_normal_N'][i]),cached_receiver_any_actor_top_N=float(d['receiver_any_actor_top_normal_N'][i]),
        cached_target_floor_N=float(d['target_floor_normal_N'][i]),object_position_END_m=q[:3].tolist(),object_quaternion_END=q[3:].tolist(),
        object_linear_velocity_END_world_m_s=v[:3].tolist(),object_speed_END_m_s=float(np.linalg.norm(v[:3])),object_horizontal_speed_END_m_s=float(np.linalg.norm(v[:2])),object_angular_speed_END_rad_s=float(np.linalg.norm(v[3:])))

def physics_window(d,start,end):
    mask=(d['interval_end_s']>start)&(d['interval_start_s']<end);ids=np.flatnonzero(mask)
    if not len(ids):return None
    q=d['object_free_qpos7_end'][ids];v=d['object_free_qvel6_end'][ids];first=q[0,3:]/np.linalg.norm(q[0,3:]);last=q[-1,3:]/np.linalg.norm(q[-1,3:]);ang=2*np.arccos(np.clip(abs(first@last),0.,1.))
    return dict(requested_window_s=[start,end],samples=len(ids),actual_endpoints_s=[float(d['interval_end_s'][ids[0]]),float(d['interval_end_s'][ids[-1]])],
        speed_m_s=stats(np.linalg.norm(v[:,:3],axis=1)),angular_speed_rad_s=stats(np.linalg.norm(v[:,3:],axis=1)),horizontal_displacement_m=(q[-1,:2]-q[0,:2]).tolist(),quaternion_net_rotation_rad=float(ang),
        target_hand_positive_samples=int(sum(d['target_hand_positive_contact_count'][ids]>0)),target_hand_peak_N=float(max(d['target_hand_normal_N'][ids])),target_receiver_top_positive_fraction=float(np.mean(d['target_receiver_top_positive_contact_count'][ids]>0)),
        receiver_top_force_N=stats(d['target_receiver_top_normal_N'][ids]),target_floor_positive_samples=int(sum(d['target_floor_positive_contact_count'][ids]>0)))

def public_point(q,obs):
    p=q['public_process'];tasks=q.get('contact_tasks');h=q.get('hierarchy');motion=None if tasks is None else tasks['diagnostics'].get('motion')
    surf=obs['receiver_surface'];cop=None
    if surf['valid'] and -surf['wrench6'][2]>surf['uncertainty6'][2]:
        w=surf['wrench6'];cop=[-w[4]/w[2],w[3]/w[2]]
    return dict(decision_time_s=q['time_s'],phase=q['phase'],public_hand_acquisition_s=obs['sample_time'],public_hand_age_s=obs['age'],public_finger_normal_N=obs['normal'],public_aux_pressure_N=obs['auxiliary_pressure'],
        current_robot_normal_velocity_P_m_s=None if h is None else h['diagnostics']['normal_current_velocity_P_m_s'],
        delayed_external_motion=motion,measured_error_M_m_s=None if tasks is None else tasks.get('motion_error_M_m_s'),
        receiver_measurement=surf,derived_public_equivalent_CoP_xy_m=cop,derived_CoP_role='net-wrench equivalent point in receiver frame; engineering positive-load gate; not target center/CoM',
        logged_observer_past_window_CoP=None if p.get('transfer',{}).get('continuous_support_evidence') is None else p['transfer']['continuous_support_evidence'].get('cop'),
        logged_observer_past_window_CoP_trend=None if p.get('transfer',{}).get('continuous_support_evidence') is None else p['transfer']['continuous_support_evidence'].get('cop_trend'))

def main():
    out=ROOT/'review/v20_exit_motion_mechanism_v1.json'
    if out.exists():raise FileExistsError(out)
    planpath=ROOT/'review/v20_small_plan_v1.json';auxpath=ROOT/'review/v20_small_auxiliary_v1.json';fclass=ROOT/'review/v20_small_fallback_classification_v1.json'
    plan=read(planpath);aux=read(auxpath)
    if not aux['complete'] or aux['issues']:raise ValueError('terminal auxiliary contract not complete')
    if not all(read(ROOT/s['output']/'execution_complete.json')['execution_complete'] for s in plan['stages']):raise ValueError('PENDING')
    inputs={str(p):sha(p) for p in (planpath,auxpath,fclass)};rows=[];checks=[];total=Counter()
    for stage in plan['stages']:
        if stage['assignment']['receiver_policy']=='blind':continue
        a=stage['assignment'];run=ROOT/stage['output']/'runs'/f"{a['scenarios'][0]}__{a['methods'][0]}__{a['seeds'][0]}";dm=read(run/'release_decision_manifest.json');delivery_manifest=read(run/'delivery_manifest.json')
        with gzip.open(run/dm['file'],'rt',encoding='utf-8') as f:qs=[json.loads(x) for x in f]
        with np.load(run/'delivery_intervals.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
        keys=('time','observation_json','executed_reference21','substep_time','substep_duration','substep_normal','substep_palm','substep_hand_other','raw_patches_json')
        with np.load(run/'trace.npz',allow_pickle=False) as z:t={k:z[k] for k in keys}
        for p in (run/dm['file'],run/'delivery_intervals.npz',run/'trace.npz',run/'metadata.json',run/'result.json'):inputs[str(p)]=sha(p)
        meta=read(run/'metadata.json');result=read(run/'result.json');ar=next(v for v in aux['records'] if v['stage']==stage['name']);frozen_aux=ar['auxiliary']
        obs={int(i):json.loads(str(t['observation_json'][i])) for i in np.flatnonzero(t['time']>=12.)};qmap={int(round(q['time_s']/.005)):q for q in qs};axisnames=meta['robot_actuator_telemetry']['axes21']
        counts=Counter(all_records=len(qs));normal_records=[];fallback=[]
        for q in qs:
            h=q.get('hierarchy');counts['phase:'+q['phase']]+=1
            if h is None:counts['direct']+=1;continue
            diag=h['diagnostics'];P=len(diag.get('pressure_P_N',[]));counts['positive_pressure_attempt' if P else 'known_P0_attempt']+=1
            accepted=q['geometry_proposal_accepted'];counts['accepted' if accepted else 'fallback']+=1
            if not accepted:
                last=h['levels'][-1];fallback.append(dict(time_s=q['time_s'],phase=q['phase'],P=P,status=h['status'],failed_layer=last['name'],failed_status=last['status'],iterations=last['iterations'],stationarity_inf=last['stationarity_inf'],inequality_dual_min=last['inequality_dual_min'],original_hard_primal_violation=last['original_hard_primal_violation'],preceding_residual_drift_inf=last['preceding_residual_drift_inf'],levels_completed=diag['levels_completed'],levels_requested=diag['levels_requested'],degenerate_face_events=diag['degenerate_face_events'],exact_rational_events=diag['exact_rational_events'],fallback=q['fallback'],actual_reference21=q['execution_receipt']['actual_reference21'],failed_trial_was_not_applied=True))
            if not P:continue
            mats={v['name']:v for v in diag['task_matrices_and_targets']};levs={v['name']:v for v in h['levels']};nname='pressure_share_current_contact_exit';mname='public_external_projection_low_motion'
            A=matrix(mats[nname]['matrix'],21);target=np.asarray(mats[nname]['target']);units=np.asarray(diag['reference_step_units21']);lo=np.asarray(diag['actual_step_lower21'])/units;hi=np.asarray(diag['actual_step_upper21'])/units;free=np.flatnonzero(lo<hi)
            rank,Z=nullspace(A[:,free],len(free));motion=mats.get(mname);M=np.empty((0,21)) if motion is None else matrix(motion['matrix'],21);mr,mz=nullspace(M[:,free]@Z,Z.shape[1]);joint_rank,_=nullspace(np.vstack((A[:,free],M[:,free])),len(free))
            ex=np.asarray(q['execution_receipt']['actual_reference21']);x=(ex-np.asarray(q['current_reference21']))/units
            ne=np.asarray(q.get('emitted_hierarchy_score',{}).get('normalized_coordinate_roundoff_bound21',np.zeros(21))) if q.get('emitted_hierarchy_score') else np.zeros(21)
            tol=ne+128*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(x),np.maximum(abs(lo),abs(hi))))
            bound=(abs(x-lo)<=tol)|(abs(x-hi)<=tol);active=nactive=levs[nname]['active_inequalities'];eye=np.eye(len(free));af=[]
            for k in active:
                if not 0<=k<2*len(free):raise ValueError('unexpected nonbox hard-row indexing')
                af.append(eye[k%len(free)])
            active_rank,_=nullspace(np.vstack((A[:,free],np.asarray(af).reshape(-1,len(free)))) if len(free) else np.empty((0,0)),len(free))
            N=matrix(diag['normal_matrix_N_Px21'],21);mind=np.asarray(diag['normal_box_min_increment_P_m']);ach=N@(ex-np.asarray(q['current_reference21']));unit=np.asarray(diag['normal_reference_row_units_P_m']);near=128*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(mind),np.sum(abs(N)*abs(ex-np.asarray(q['current_reference21']))[None,:],axis=1)))
            l2active=set() if motion is None else set(levs[mname]['active_inequalities']);released=set(active)-l2active if motion is not None else set()
            normal_records.append(dict(time_s=q['time_s'],phase=q['phase'],accepted=accepted,P=P,
                geometry_public_read_time_s=q['contact_tasks']['diagnostics']['current_time_s'],geometry_patch_acquisition_times_s=[p['acquisition_time_s'] for p in q['contact_tasks']['diagnostics']['patches']],motion_observation=q['contact_tasks']['diagnostics'].get('motion'),
                positive_width_axes=len(free),positive_width_axis_names=[axisnames[i] for i in free],L1_numeric_rank_on_positive_width_axes=rank,L1_affine_nullity=len(free)-rank,L2_available=motion is not None,L2_rows=len(M),L2_rank_in_L1_affine_nullspace=mr,L1_L2_combined_affine_nullity=len(free)-joint_rank,
                L1_recorded_working_box_faces=len(active),L1_affine_plus_recorded_working_faces_nullity=len(free)-active_rank,L2_released_L1_working_faces=len(released),
                emitted_axes_at_some_bound=int(sum(bound[free])),emitted_bound_axis_names=[axisnames[i] for i in free if bound[i]],
                normal_rows_at_individual_outward_box_min=int(sum(abs(ach-mind)<=near)),normal_rows_with_outward_box_capacity=int(sum(mind<0)),
                normalized_box_width_fraction=stats((hi-lo)[free]),L1_logged_squared_loss=levs[nname]['squared_loss'],L1_at_actual_emission_squared_loss=float(np.square(A@x-target).sum()),
                L2_logged_squared_loss=None if motion is None else levs[mname]['squared_loss'],L2_at_actual_emission_squared_loss=None if motion is None else float(np.square(M@x-np.asarray(motion['target'])).sum()),
                upper_residuals_preserved_in_solver=None if not accepted else max(v['preceding_residual_drift_inf'] for v in h['levels']),
                emitted_reference_scope='accepted_hierarchy_command' if accepted else 'fallback_recovery_anchor_not_the_failed_trial',
                intermediate_layer_x_logged=False,box_cone_feasible_dimension_not_established=True))
        total.update(counts)
        start=d['interval_start_s'];end=d['interval_end_s'];dur=d['interval_duration_s'];release=start>=12.-1e-9;hand=d['target_hand_normal_N'];episodes=positive_episodes(release&(d['target_hand_positive_contact_count']>0),start,end,hand,dur);last=episodes[-1]
        lastend=last['end_s'];firstpos=np.flatnonzero(release&(d['target_receiver_top_positive_contact_count']>0));firstR=float(start[firstpos[0]]) if len(firstpos) else None
        event_times=[lastend+offset for offset in (-.05,-.01,0.,.01,.05)]+[15.,16.];event_points=[physics_point(d,v) for v in event_times]
        pi=int(np.searchsorted(t['time'],lastend));ii=np.flatnonzero((t['time']>=lastend)&(t['time']<=16.));refs=t['executed_reference21'][ii];qraw=np.array([obs[int(i)]['robot_position25'] for i in ii]);refTV=abs(np.diff(refs,axis=0)).sum(axis=0);qTV=abs(np.diff(qraw,axis=0)).sum(axis=0)
        public_samples=[]
        for wanted in ([firstR] if firstR is not None else [])+[lastend,15.,16.]:
            i=min(3200,int(np.searchsorted(t['time'],wanted)));public_samples.append(public_point(qmap[i],obs[i]))
        summary=dict(positive_pressure_attempts=len(normal_records),accepted_with_pressure=sum(x['accepted'] for x in normal_records),fallback_with_pressure=sum(not x['accepted'] for x in normal_records),
            L1_nullity=stats([x['L1_affine_nullity'] for x in normal_records]),L2_projected_rank=stats([x['L2_rank_in_L1_affine_nullspace'] for x in normal_records if x['L2_available']]),
            L1_working_face_nullity=stats([x['L1_affine_plus_recorded_working_faces_nullity'] for x in normal_records]),L2_releases_L1_working_faces_records=sum(x['L2_released_L1_working_faces']>0 for x in normal_records),
            accepted_normal_records_with_all_positive_width_axes_at_bound=sum(x['emitted_axes_at_some_bound']==x['positive_width_axes'] for x in normal_records if x['accepted']),
            L1_loss=stats([x['L1_logged_squared_loss'] for x in normal_records]),L2_loss=stats([x['L2_logged_squared_loss'] for x in normal_records if x['L2_available']]))
        nominal=None
        if stage['name']=='nominal_transport':
            wi=np.flatnonzero((t['time']>=14.42-1e-9)&(t['time']<=14.72+1e-9));window=[]
            for i in wi:
                q=qmap[int(i)];f=obs[int(i)];h=q.get('hierarchy');tasks=q.get('contact_tasks');window.append(dict(time_s=float(t['time'][i]),phase=q['phase'],pressure_rows=None if tasks is None else len(tasks.get('pressure_P_N') or []),
                    task_priority=[] if h is None else h['diagnostics']['task_priority'],current_reference21=q['current_reference21'],preferred_reference21=q['preferred_reference21'],proposed_reference21=q['proposed_reference21'],actual_reference21=t['executed_reference21'][i].tolist(),
                    robot_q25=f['robot_position25'],robot_v25=f['robot_velocity25'],public_hand_acquisition_s=f['sample_time'],public_aux_pressure_N=f['auxiliary_pressure'],public_normal5_N=f['normal'],raw_200Hz_patches=json.loads(str(t['raw_patches_json'][i]))))
            stable=frozen_aux['observed_stable_candidates']['qualifying_episodes'][0];breaktime=stable['end_s'];bm=(t['substep_time']>=breaktime-.025)&(t['substep_time']<=breaktime+.025);bi=np.flatnonzero(bm)
            nominal=dict(window_s=[14.42,14.72],samples=window,first_low_motion_episode_unchanged=stable,
                micro_cached_force_around_loss=[dict(start_s=float(t['substep_time'][i]),duration_s=float(t['substep_duration'][i]),target_finger_normal5_N=t['substep_normal'][i].tolist(),target_palm_N=float(t['substep_palm'][i]),target_other_hand_N=float(t['substep_hand_other'][i])) for i in bi],
                physical_window_around_loss=physics_window(d,breaktime-.025,breaktime+.025),physical_points_around_loss=[physics_point(d,breaktime+o) for o in (-.01,0.,.0035,.01,.025)],
                direct_identity_evidence='Raw14.710 patch part=palm,digit=-1,is_target=True; 2kHz recontact is entirely substep_palm, no finger normal or otherhand normal.')
        checks.append(dict(stage=stage['name'],release_records_801=len(qs)==801,trace3201=len(t['time'])==3201,ledger32000=len(end)==32000,all_ledger_valid=bool(d['contact_valid'].all() and d['object_state_valid'].all()),archive_hash=sha(run/dm['file'])==dm['sha256'],ledger_hash=sha(run/'delivery_intervals.npz')==delivery_manifest['arrays']['sha256']))
        rows.append(dict(stage=stage['name'],run=str(run),original_metrics=ar['metrics'],assignment=a,counts=dict(counts),hierarchy_summary=summary,normal_decisions=normal_records,unavailable_task_details=fallback,
            receiver_first_cached_contact_start_s=firstR,last_true_target_hand_episode=last,all_target_hand_episodes_after_request=episodes,
            physical_event_points=event_points,post_last_true_hand_contact=physics_window(d,lastend,16.0001),late_motion15_16=physics_window(d,15.,16.0001),late_motion15p5_16=physics_window(d,15.5,16.0001),public_event_samples=public_samples,
            robot_after_last_target_hand_contact=dict(first_trace_time_s=float(t['time'][ii[0]]),last_trace_time_s=float(t['time'][ii[-1]]),reference_TV_finger18_rad=refTV[:18].tolist(),reference_TV_lift_m=float(refTV[18]),reference_TV_wrist2_rad=refTV[19:].tolist(),reference_net21=(refs[-1]-refs[0]).tolist(),robot_encoder_TV_first24_rad=qTV[:24].tolist(),robot_encoder_TV_lift_m=float(qTV[24]),robot_encoder_net25=(qraw[-1]-qraw[0]).tolist(),scope='axis units kept separate; no new FK/point path and no inference from joint sign to Cartesian separation'),nominal_recontact_focus=nominal))
    passed=all(all(v for k,v in c.items() if k!='stage') for c in checks)
    selected_sources=('v20_contact_exit/task_solver.py','v20_contact_exit/supervisor.py','v20_contact_exit/adapter.py','v17_contact_transfer/hierarchy.py','skin_geometry.py')
    report=dict(schema='v20_exit_motion_mechanism_v1',passed_read_integrity=passed,plan_sha256=sha(planpath),reader_sha256=sha(__file__),source_sha256={k:plan['source_sha256'][k] for k in selected_sources},input_sha256=inputs,checks=checks,counts_all3=dict(total),rows=rows,
        scope=dict(terminal_candidate_runs=3,release_decisions=total['all_records'],saved_positive_pressure_hierarchies=sum(len(row['normal_decisions']) for row in rows),solver_calls=0,FK=0,models=0,physics=0,scans_of_new13=0),
        role='task_solver author execution-mechanism diagnosis; not own-kernel independent acceptance',
        limitations=['Numeric nullity is calculated from logged scaled matrices after exact fixed-box coordinates are removed. It is affine algebraic freedom, NOT feasible-box tangent-cone dimension or physical motor authority.','Intermediate layer x/nullspace was not logged; recorded working faces can later be released. Never infer all motion locked from residual size or active-face counts.','Contact force is cached solver-stage data labelled interval start/end; q/v are integration END. Substep timing association has up to one0.5ms solve/integration phase uncertainty.','Public e is acquisition-labelled delayed external projection and current J is a geometric proposal; object q/v are offline truth and cannot be moved into controller inputs.','Velocity norms/quaternion change show persistent motion, not a certified rolling/sliding/rocking mode.','No parameters, constraints, score definitions, source160 or original results changed.'])
    out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# V20清离—运动机制诊断','', '完整三例保留。仅读取已终态4条计划内的3候选；不读取补齐13条。无重解、FK、模型或物理。','', '|条目|有压层级/accepted/fallback|L1后仿射nullity范围|L2在该空间rank范围|末线速m/s|末角速rad/s|','|---|---|---|---|---:|---:|']
    for row in rows:
        h=row['hierarchy_summary'];terminal=row['late_motion15p5_16'];lines.append(f"|{row['stage']}|{h['positive_pressure_attempts']}/{h['accepted_with_pressure']}/{h['fallback_with_pressure']}|{h['L1_nullity']}|{h['L2_projected_rank']}|{terminal['speed_m_s']['median']:.9g}|{terminal['angular_speed_rad_s']['median']:.9g}|")
    lines+=['','**数值失败另列：**25次fallback分别10/7/8，全部是在L1、L2已完成后，L3 `canonical_reference_intent_and_minimal_wrist_action` 出现`DEPENDENT_BLOCKER_NUMERICAL_FAILURE`。不是adapter未知、native失败或400步耗尽；当前实现将整次层级判不可用，实际执行合法恢复anchor，并未把未通过的试算当已兑现。原始硬域违例很小不代表L3 KKT已通过，负对偶/退化阻塞日志保持。','', '**权限解释：**完整matrix允许重算L1后的仿射空间及L2是否仍有作用方向；但未保存各层中间x和完整可行cone。L1工作集触边不等于所有下层自由度消失，部分L2确实释放L1工作面。大motion loss也不能单独归因全部锁死。','', '**物理机制：**三例最后目标手接触之后均仍运动到16s，且15–16s速度近恒定、receiver top保持接触而floor无接触。后续机器人虽继续执行参考/编码器运动，账本在末次手接触后无再次直接目标手接触；不能把无接触期间的持续运动全部归于后续腕/lift命令，也不能仅凭速度范数称已证明滚动。','', '**名义例短再触：**14.4225–14.708有低运动episode；14.7080–14.7115约3.5ms目标—palm接触，峰0.42113N，指压力全0。同期对象线速约0.0001486→0.0089976m/s，角速0.00531→0.32172rad/s，此后近恒定。14.42–14.72公开几何P0使层级只有L3：THJ5继续回OPEN，SEPARATE14.635后lift退让，腕参考恒定而实际关节仍有滞后运动。OPEN返回、lift和腕实际运动的独立因果贡献未拆分；不得只挑其中一个定因。','', '**延迟与记忆：**14.710公开手压仍来自14.700，尚未看到刚发生的palm再触，14.720才反映aux压力。10ms延迟反馈无法被描述成能阻止本次3.5ms事件。P0不再保留已有清离路径/旧接触任务，且当前局部几何仅五指；这为下一整体方案提出路径连续性与全外皮肤清离约束的需求。','', '**下一步证据要求：**将最后接触前的低外部运动、持续接收支持和退出动作放到同一任务偏序中；显式保留清离后不回扫的路径/接触记忆，不能仅换gain或把OPEN当几何清离证书。数值层需区分已通过的高层任务和失败的末级偏好，避免无标签地丢弃整次可用高层证据；本轮不修改策略或使用未通过候选。每一结构改动仍需同条件因果验证，不能由3例宣布通用安全。']
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed_read_integrity=passed,out=str(out),sha256=sha(out),counts=dict(total),summaries=[dict(stage=r['stage'],hierarchy=r['hierarchy_summary'],last_hand=r['last_true_target_hand_episode'],late=r['late_motion15p5_16']) for r in rows]),ensure_ascii=False));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())

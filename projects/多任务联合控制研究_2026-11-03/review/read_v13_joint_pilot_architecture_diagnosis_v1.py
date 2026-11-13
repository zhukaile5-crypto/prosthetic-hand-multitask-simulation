"""Architecture evidence reader, not a full metrics/physics audit or policy run."""
from pathlib import Path
import collections,gzip,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'output/development_v13_joint_pilot_v1'
OLD=ROOT/'output/development_v12_native_pilot_v1'
OUT=ROOT/'review/v13_joint_pilot_architecture_diagnosis_v1.json'
MD=ROOT/'review/v13_joint_pilot_architecture_diagnosis_v1.md'
METHODS=('ratio_effort','fixed_effort','fusion_effort')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

def main():
    terminal=load(NEW/'execution_complete.json');assert terminal['execution_complete'] and terminal['trials']==terminal['planned']==3 and terminal['sources_unchanged'] and not terminal['runtime_failures']
    assert sha(NEW/'trials.json')==terminal['trials_sha256']
    assert load(OLD/'execution_complete.json')['execution_complete']
    source_names=('v13_joint/controller.py','v13_joint/intent_qp.py','v13_joint/lifecycle.py','v13_joint/plant.py','v13_joint/runtime.py','v13_joint/prepared_value_effort21.py','v13_joint/native_effort21.py','v12_native/controller.py','v13_joint/design.py')
    sources={n:sha(NEW/'sources'/n) for n in source_names};manifest=load(NEW/'manifest.json')
    assert all(manifest['research_source_sha256'][n]==v for n,v in sources.items())
    source=lambda n,line:dict(file=str(NEW/'sources'/n),line=line)
    provenance={};rows={}
    for method in METHODS:
        run=NEW/'runs'/f'nominal__{method}__617901';old=OLD/'runs'/run.name
        r=load(run/'result.json');ro=load(old/'result.json');meta=load(run/'metadata.json');oldmeta=load(old/'metadata.json')
        assert meta['shape']=='capsule' and meta['seed']==617901 and meta['motion_mode']=='hold' and oldmeta['motion_mode']=='hold'
        for name in ('result.json','metadata.json','joint_decisions.jsonl.gz','joint_decision_manifest.json'):
            provenance[str((run/name).relative_to(ROOT))]=sha(run/name)
        provenance[str((old/'result.json').relative_to(ROOT))]=sha(old/'result.json')
        assert sha(run/'joint_decisions.jsonl.gz')==load(run/'joint_decision_manifest.json')['sha256']
        with gzip.open(run/'joint_decisions.jsonl.gz','rt',encoding='utf-8') as stream:decisions=[json.loads(line) for line in stream]
        with np.load(run/'trace.npz',allow_pickle=False) as z:
            t=z['time'];f=z['normal'];refs=z['executed_reference21'];inc=z['reference_increment21'];debug=[json.loads(str(v)) for v in z['debug_json']]
            observations=z['observation_json'] # compressed string column materialized once
            def ix(time):return int(np.argmin(abs(t-time)))
            def episode(d):
                i=ix(d['time_s']);q=d['qp'];dg=q['diagnostics'];ob=json.loads(str(observations[i]));digits=[int(ob['skin_patches'][j]['digit']) for j in d['patch_indices']]
                candidate=np.bincount(digits,weights=q['zP'],minlength=5) if q.get('zP') is not None else None
                actual=np.asarray(debug[i]['desired_force5']);prior=np.asarray(d['force_prior5'])
                return dict(decision_id=d['decision_id'],time_s=d['time_s'],mode=d['mode'],status=q['status'],reason=q['reason'],patch_count=len(d['patch_indices']),
                    prior5_N=prior,qp_candidate_force5_N=candidate,actually_selected_force5_N=actual,
                    selected_minus_candidate5_N=None if candidate is None else actual-candidate,
                    kkt_d=dg['direct_KKT_inf_d21'],tol_d=dg['numerical_tolerance_d21'],kkt_z=dg['direct_KKT_inf_zP'],tol_z=dg['numerical_tolerance_zP'],L=dg['lipschitz_upper_bound'],
                    primal_violations=dg['primal_violations'],reference_increment_finger18_rad=inc[i,:18],reference_increment_lift_m=float(inc[i,18]),reference_increment_wrists_rad=inc[i,19:],
                    native_status=d['native']['status'],native_full_recheck=d['native']['diagnostics']['prepared_value_final_full_recheck'])
            counts=collections.Counter((d['mode'],d['qp']['status']) for d in decisions)
            holding=[d for d in decisions if d['mode']=='holding'];release=[d for d in decisions if d['mode']=='release']
            selected={}
            for mode,group in (('holding',holding),('release',release)):
                fail=next(d for d in group if d['qp']['status']!='converged');j=decisions.index(fail)
                selected[mode+'_first_failure_neighbors']=[episode(d) for d in decisions[max(0,j-1):j+2]]
            static=[d for d in holding if 8.2-1e-9<=d['time_s']<=8.9+1e-9]
            transitions=[dict(time_s=b['time_s'],from_status=a['qp']['status'],to_status=b['qp']['status']) for a,b in zip(holding,holding[1:]) if a['qp']['status']!=b['qp']['status']]
            def metric(rr):
                m=rr['research_metrics'];q=m['static_quality'];s=m['safety']
                return dict(task_success=rr['task_success'],full_motion_success=m['full_motion_mission_success'],guard_pass=s['guard_pass'],static_eligible=q['eligible'],static_mean_N=q['mean_actual_finger_force_N'],static_raw200_TV_N_s=q['raw_200Hz']['finger_TV_rate_N_s'],static_band_TV_N_s=q['bandlimited']['finger_TV_rate_N_s'],static_total_SD_N=q['raw_200Hz']['total_force_SD_N'],full_peak_N=max(s['finger_peak_N']),full_local_excess_Ns=s['local_excess_integral_Ns'],holding_local_excess_Ns=s['stages']['policy_holding_transport']['local_excess_integral_Ns'],release_latency_coarse_s=m['release']['physical_clearance_latency_s'])
            current=metric(r);before=metric(ro)
            row=dict(logged_decision_counts={f'{a}/{b}':v for (a,b),v in counts.items()},logged_native_available_count=sum(d['native']['reference21'] is not None for d in decisions),logged_full_recheck_pass_count=sum(d['native']['diagnostics']['prepared_value_final_full_recheck']['passed'] for d in decisions),
                holding_support_active_count=sum(debug[ix(d['time_s'])]['conditional_support_active'] for d in holding),holding_decisions=len(holding),release_decisions=len(release),
                holding_status_switch_count=len(transitions),holding_status_switches=transitions,static_status_counts=dict(collections.Counter(d['qp']['status'] for d in static)),selected_episodes=selected,
                stored_V3_metrics_scope='Selected stored V3 result values for context only; not independently recalculated V3/FIR/guard.',new_stored_metrics=current,old_stored_metrics=before,
                static_raw200_TV_relative_change=current['static_raw200_TV_N_s']/before['static_raw200_TV_N_s']-1.,
                declared_trace_sha256=r['trace_sha256'],raw_field_read_scope='All3 200Hz debug/reference columns for architecture episodes; only fusion raw2k release peak segment is additionally queried.')
            if method=='fusion_effort':
                st=z['substep_time'];sn=z['substep_normal'];dt=z['substep_duration'];ri=np.flatnonzero(st>=12.)[np.argmax(sn[st>=12.].max(axis=1))]
                samples=[]
                for qt in (12.,12.16,12.18,12.2,12.22,12.225,12.23,12.235,12.24):
                    i=ix(qt);ob=json.loads(str(observations[i]));age=np.asarray(ob['age'])
                    samples.append(dict(time_s=float(t[i]),LF_200Hz_true_N=float(f[i,4]),LF_observed_N=float(ob['normal'][4]),public_age_s=float(age if age.ndim==0 else age[4]),public_patch_count=len(ob['skin_patches']),LF_reference4=refs[i,14:18],LF_increment4=inc[i,14:18],lift_reference_m=float(refs[i,18])))
                near=[]
                for d in release:
                    if 12.14-1e-9<=d['time_s']<=12.28+1e-9:
                        near.append(dict(decision_id=d['decision_id'],time_s=d['time_s'],patch_count=len(d['patch_indices']),y_patch_N=d['observed_patch_normal'],qp=d['qp']['status'],LF_actual_effort_delta4=d['native']['actual_delta_effort21'][14:18],full_recheck=d['native']['diagnostics']['prepared_value_final_full_recheck']['passed']))
                row['release_peak_episode']=dict(peak_N=float(sn[ri].max()),digit=int(sn[ri].argmax()),raw_start_s=float(st[ri]),raw_end_s=float(st[ri]+dt[ri]),raw_duration_s=float(dt[ri]),selected_samples=samples,nearby_decisions=near,
                    interpretation='Raw pulse is visible offline at2kHz but not as its true magnitude in sampled200Hz public pressure; no claim of exact same-patch correspondence across acquisition clocks.')
            rows[method]=plain(row)
    findings=[
      dict(id='F1',level='source_fact_and_actual_control_branch',claim='KKT-unavailable selects a different force allocation rather than only an unverified solve of the same output: z becomes prior while QP candidate z is discarded; native inverse still executes.',refs=[source('v13_joint/controller.py',121),source('v13_joint/controller.py',133),source('v13_joint/controller.py',137)],limit='This proves a target/command-branch discontinuity mechanism, not its isolated causal contribution to force TV.'),
      dict(id='F2',level='source_fact',claim='Every QP cold-starts at projected[d=0,z=intent] with one fixed global Lipschitz step and400 iterations. Logged failures are genuine direct-KKT budget exhaustion, not unknown load or native arithmetic failure.',refs=[source('v13_joint/intent_qp.py',350),source('v13_joint/intent_qp.py',397),source('v13_joint/intent_qp.py',405)],limit='Ill-conditioning is a candidate explanation; this reader has not rebuilt exact Hessians/condition numbers or solved alternative QPs.'),
      dict(id='F3',level='source_fact',claim='FusionIntent allocates five force goals with balance/continuity, then common outer QP reallocates patch forces under effort/load costs. Outer kernel has no additional balance or soft-synergy rows. Fixed/ratio are soft priors, not their former final desired-force laws.',refs=[source('v12_native/controller.py',104),source('v13_joint/controller.py',90),source('v13_joint/controller.py',131),source('v13_joint/controller.py',151)],limit='Composition changes the package and can alter balance/continuity priorities; extra21 freedom itself is not proven harmful.'),
      dict(id='F4',level='source_fact',claim='Effort regularizer is ||scale*d||^2 with d=p-p0; p0 is raw nominal native effort. During release nominal is the moving tuck/open/lift path, so this is closeness to a moving nominal effort, not a previous-executed-effort temporal smoothness term.',refs=[source('v13_joint/controller.py',66),source('v13_joint/native_effort21.py',142),source('v13_joint/intent_qp.py',364),source('v13_joint/lifecycle.py',25)],limit='Reference rates and nonthumb monotonicity are respected but do not guarantee patch separation or pressure bounds.'),
      dict(id='F5',level='source_fact_and_timed_episode',claim='Release immediately removes holding capacity/force prior. At fusion12.22 current accepted patch set is empty, QP/native pass and LF effort correction is0, while geometric release reference still moves. Offline LF peak follows at raw end12.229 and is missed in true magnitude by200Hz public pressure.',refs=[source('v13_joint/controller.py',54),source('v13_joint/controller.py',118),source('v13_joint/lifecycle.py',25)],limit='This event is not explained by a simultaneous QP failure, stale/lateclip, or false native value. It does not identify which finger/lift interaction caused recontact.'),
      dict(id='F6',level='scope_fact',claim='Old/new paired runs estimate the local total package change, including holding posture/release-entry and8→21 execution changes. They do not isolate release policy or numerical branch effects.',refs=[source('v13_joint/controller.py',36),source('v13_joint/plant.py',61)],limit='One nominal/hold/capsule/physical-noise seed617901 seen development context; methods are not independent populations.')]
    directions=[
      dict(id='N1',title='One mathematical problem, stable numerical path',proposal='Exploit admitted disjoint actuator supports to eliminate each d_i exactly for fixed z: d_i=clip((A_i @ (sr^2 * (J.T@(z-y))))/(sum(sr^2*A_i^2)+sd_i^2),lo_i,hi_i). Solve reduced convex piecewise-quadratic z problem with original objective/weights/scales/caps/p0, then verify full original KKT/complementarity. Correct variable transforms require corresponding metric-aware projection. No loosened tolerance or hidden goal retarget.',test='Archived exact public-input fixtures: same objective and active constraints, fixedp0 and intent bytes; huge-L/tiny-box/zero-gain/saturation/disjoint-support oracles; alternative solver may honestly return unavailable. Only then matched closed-loop test holding TV.',limitation='Disjoint supports are mandatory; cannot apply conditional scalar elimination to overlapping A rows. Finite native inverse/recheck remains required.'),
      dict(id='N2',title='Single allocation plus explicit continuity/soft synergies',proposal='Keep oneforce intent/patch-allocation problem instead of two implicit allocations. Add declared balance/previous-executed-effort/reference-progress/soft5synergy tasks with explicit units and priorities; preserve21 solewriter and real bounds. Existing8 actor remains a permissions-matched ablation.',test='Healthy nominal identity and stable target perturbation tests; equal21 rights across fixed/ratio/fusion; same sensing and gains; do not call inherited force law unchanged.',limitation='This is a new controller objective, requiring preregistration, not an algebraic rewrite of the current QP.'),
      dict(id='N3',title='Release contact-transition coordinator, without delaying the release request',proposal='Cancel holding immediately. Coordinate finger-path progress and lift descent from actual21/qv plus public tactile contact history, with monotone progress and declared rate/curvature limits. Empty current patches mean unknown clearance, not proof of clearance. Last patch robot geometry may restrict robot motion directions but cannot invent external-object motion or future pressure.',test='First exact originalboth recorded-input bridge; matched-entry finger/lift2x2 replay identifies interaction, not policy performance. Then separate closed-loop numeric-only × release-coordination factorial with all failures/latency/force metrics kept.',limitation='No placing/tabletruth inference or pressure certificate. A pause in path progress must not restore holding support or postpone release activation.'),
      dict(id='N4',title='Make short-contact observability explicit and fair',proposal='If using per-interval peak/impulse/contact-transition telemetry, define it as a new public all-skin device measurement with real acquisition clocks/delay and give it equally to all baselines. Existing offline objecttrue raw peaks cannot enter control.',test='Show which events are sampled/missed under200Hz and peak-hold frontends; fixed-input dt refinement and first-impact limits remain separate from feedback-policy efficacy.',limitation='Observing a pulse after impact does not show it could have been prevented before detection.')]
    report=plain(dict(schema='v13_joint_pilot_architecture_diagnosis_v1',role='independent_source_and_selected_temporal_architecture_diagnosis',terminal_gate=True,context=dict(scenario='nominal',motion='hold',shape='capsule',seed=617901,methods=list(METHODS),development_only=True),
        inputs=dict(terminal_sha256=sha(NEW/'execution_complete.json'),trials_sha256=terminal['trials_sha256'],source_snapshot_sha256=sources,selected_input_sha256=provenance),rows=rows,findings=findings,next_directions=directions,
        scope=dict(all_decision_records_read=3*426,all_V3_metrics_recomputed=False,all_physics_audited=False,selected_fusion_raw2k_peak_only=True,no_new_model_or_controller_imports=True,physics_steps=0,production_edits=0),
        causal_limits=['Logged KKT failure deterministically changes the selected allocation branch; pressure TV causality is not identified without same-input alternative solves and matched closed-loop ablation.',
                      'No frequency-resonance or timestep-artifact attribution is made.',
                      'Cross-version release starts differ after different holding trajectories; same-seed pairing does not equate release entry states.',
                      'No new utility weights, success-loss tolerance, passing force guard or global model admission is invented.']))
    assert all(sha(NEW/'sources'/n)==v for n,v in sources.items())
    report['reader_sha256']=sha(Path(__file__))
    assert not OUT.exists() and not MD.exists();OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    table=[]
    for m in METHODS:
        v=rows[m];new=v['new_stored_metrics'];old=v['old_stored_metrics'];c=v['logged_decision_counts']
        table.append(f"| {m} | {c.get('holding/nonconvergence',0)}/225 | {c.get('release/nonconvergence',0)}/201 | {old['static_raw200_TV_N_s']:.6f} → {new['static_raw200_TV_N_s']:.6f} | {old['full_peak_N']:.6f} → {new['full_peak_N']:.6f} |")
    fusion=rows['fusion_effort'];ep=fusion['release_peak_episode'];first=rows['fixed_effort']['selected_episodes']['holding_first_failure_neighbors'][1]
    md=f"""# V13 Joint pilot 架构诊断（只读）

结论：现有证据支持两条独立问题线。持握时，数值状态会切换实际力分配目标；释放时，已知原生努力映射和连续参考不能替代接触转换约束。当前三法任务/完整运动都通过、全部过力来自释放；这不等于整体协调已改善。不能把失败归为模型未知、原生算式错误或执行裁剪，也不能把所有压力波动归给QP。

范围：单个已见开发条件 nominal / hold / capsule / seed617901。读取全部1278份决策记录来定位分支，读取200Hz参考/观测片段；仅抽核fusion的2kHz释放峰片段。表内V3指标来自已存result，未重复父审/基础代理的全raw指标独审。冻结执行源未改、没有控制器/模型导入或新物理。

| 方法 | 持握QP未收敛 | 释放QP未收敛 | 静态raw200Hz TV率 旧→新 N/s | 全程峰 旧→新 N |
|---|---:|---:|---:|---:|
"""+'\n'.join(table)+f"""

## 源码确定的机制

1. **数值失败会换任务。** [controller.py:121]({NEW.as_posix()}/sources/v13_joint/controller.py:121)先设z=prior；只有QP可用才采用QP.z。未收敛后原生逆映射仍执行。fixed与ratio首个持握失败在8.04s，d的真实KKT残差{first['kkt_d']:.9g}高于{first['tol_d']:.9g}，其原始可行性违反仍全0；这是真实未满足停止合同，不应签成收敛。但切换到prior会放弃候选力分配，形成与“继续求同一目标”不同的控制分支。

2. **数值实现与闭环机制耦合。** [intent_qp.py:350]({NEW.as_posix()}/sources/v13_joint/intent_qp.py:350)采用固定全局L、[397]({NEW.as_posix()}/sources/v13_joint/intent_qp.py:397)每次从投影的d=0/z=intent冷启动、400步预算。三法持握225次条件承载都可用；fusion有123次未收敛。数据证明预算耗尽引发分支切换，尚未证明具体矩阵条件数是唯一原因；不能靠改权重/放宽KKT或加迭代掩盖。

3. **新增QP会重分配原有intent。** 原FusionIntent已做5指平衡/连续性分配，外层又优化patch z与承载slack；外层并未实现额外balance或soft-synergy行。因此fixed/ratio只是软先验，fusion也不是原分配目标原样兑现。完整21自由度、分配层和holding姿态同时变了，不能把退步孤立归给“21自由度”或某一个分组模块。

4. **所谓effort continuity实际锚在移动nominal。** [QP:364]({NEW.as_posix()}/sources/v13_joint/intent_qp.py:364)的正则是d=p−p0；[native domain:142]({NEW.as_posix()}/sources/v13_joint/native_effort21.py:142)的p0来自原始nominal。释放nominal是[固定tuck/open/lift路径]({NEW.as_posix()}/sources/v13_joint/lifecycle.py:25)。这限制偏离移动nominal的努力，并非上一执行努力的时间连续性。真实参考率合法、非拇指单向张开，均不蕴含每个触觉斑块向外离开。

## 释放峰的关键时序

fusion LF峰{ep['peak_N']:.9f}N位于原始子步起点{ep['raw_start_s']:.12f}s/终点{ep['raw_end_s']:.12f}s。12.225与12.230s的200Hz真实LF力都是0；12.22s公共LF观测0、12.23s约0.0103N，触觉年龄10ms。12.22s决策的accepted patch为空、QP已收敛、native/full复核通过、LF努力修正0；LF和lift仍继续名义释放路径。**该峰当刻并不是QP失败分支，也未以7.49N的幅度进入200Hz公共压力流。**

这是“当前无已知斑块，但短时重新接触仍可能发生”的证据。它支持检查路径/升降耦合及可观测性；尚不能判定哪一关节/升降单独制造峰，也不能把参考单调误当接触力单调。释放从12s立即取消holding目标，未发现holding支持迟迟不撤的机制。

## 下一版应先验证什么

* **保持同一数学问题，修求解结构。** 原生A的已准入supports互不重叠时，可对固定z解析消去每个d_i：`clip((A_i @ (sr² * (Jᵀ(z−y)))) / (sum(sr²*A_i²)+sd_i²), lo_i, hi_i)`。再解较小的凸分段二次z问题，以原始完整KKT/互补验证。保持p0、目标、caps、scales不变；重参数化必须带正确投影度量。先归档公共输入纯数值对照，再闭环验证TV，不能把离线数值成功当压力收益。
* **明确唯一分配与连续性任务。** 平衡、上一真实执行努力/参考、5synergy软结构和用户运动采用明确、预注册的任务，不由两层allocation隐含覆盖。21唯一writer保留，8执行限制作为同权限消融。
* **释放仍立即激活，但路径进度与升降共同协调。** 用actual21/qv、公共触觉历史与可声明的机器人几何方向定义progress约束；不重新启用holding、不以对象/桌面truth延迟release。当前patch为空只能令清离未知，不能认证清离。
* **先固定输入归因，再测新闭环。** 原both全物理/传感/21命令桥接exact后，matched-entry finger×lift2×2重放辨别交互；这是输入反事实，不是新策略性能。数值结构与release协调再做分开的闭环消融，所有失败/NULL/延迟成本保留，不事后加综合权重。
* **短脉冲观测升级必须公平。** 若增加每间隔all-skin峰值/冲量，需作为有真实时钟、延迟的公共设备信息给全部baseline；离线对象真值峰不得在线使用。事后看到峰仍不能保证检测前峰受控。

旧/新同seed只支持该条件下整体package变化的比较。不同holding轨迹已改变release entry；要识别release机制必须同完整入口，而非只同qpos或只同seed。频谱相近仅是描述，本报告不宣称共振、单一因果、力安全证书或独立总体优势。

结构化证据：[JSON](v13_joint_pilot_architecture_diagnosis_v1.json)。
"""
    MD.write_text(md,encoding='utf-8')
    print(json.dumps(dict(json=str(OUT),json_sha256=sha(OUT),markdown=str(MD),markdown_sha256=sha(MD),decisions_read=1278,physics_steps=0),ensure_ascii=False))

if __name__=='__main__':main()

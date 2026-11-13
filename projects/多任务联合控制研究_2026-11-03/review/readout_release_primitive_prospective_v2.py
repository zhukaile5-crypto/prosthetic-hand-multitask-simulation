"""Author readout of the complete fixed release-component experiment.

Not the independent audit. No physics import/constructor/step. The terminal
marker is required before reading any assigned branch result or trace.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from evaluation import force_variation, force_band
REPORT_SCALARS=('local_excess_Ns','force_peak_N','release_excess_Ns',
    'original_first_clear200Hz_s','first_by_deadline_raw_end_s','terminal_low_force_raw_end_s',
    'release_target_raw200Hz_TV_rate_N_s','release_skin_raw2000Hz_TV_rate_N_s',
    'release_target_FIR50Hz_TV_rate_N_s')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def scalar_summary(values):
    allvalues=list(values);good=[float(v) for v in allvalues if v is not None and np.isfinite(v)]
    return dict(assigned=len(allvalues),available=len(good),missing=len(allvalues)-len(good),
        mean=float(np.mean(good)) if good else None,median=float(np.median(good)) if good else None,
        minimum=min(good) if good else None,maximum=max(good) if good else None,
        sum=float(np.sum(good)) if good else None)


def paired_difference(zero,slow):
    # Missing times are not zero benefits; binary success remains full ITT.
    return None if zero is None or slow is None else float(slow-zero)


def binary_pairs(rows,key):
    counts=dict(both_true=0,zero_only=0,slow_only=0,both_false=0,unavailable=0)
    for p in rows:
        a,b=p['zero_values'].get(key),p['slow_values'].get(key)
        if a is None or b is None:counts['unavailable']+=1
        elif a and b:counts['both_true']+=1
        elif a:counts['zero_only']+=1
        elif b:counts['slow_only']+=1
        else:counts['both_false']+=1
    return counts


def phase_cost(t,duration,force,limits,budget,lo,hi):
    end=t+duration
    # Stored force is post-mj_step; interval costs use actual overlap duration.
    overlap=np.maximum(0.,np.minimum(end,hi)-np.maximum(t,lo))
    mask=overlap>1e-10
    if not mask.any():return dict(available=False,window_s=[lo,hi],duration_s=0.,peak_N=None,local_excess_Ns=None)
    return dict(available=True,window_s=[lo,hi],duration_s=float(overlap.sum()),
        peak_N=float(force[mask].max()),per_finger_peak_N=force[mask].max(axis=0).tolist(),
        local_excess_Ns=float(np.sum(np.maximum(force-limits,0)*overlap[:,None])),
        total_excess_Ns=float(np.maximum(force.sum(axis=1)-budget,0)@overlap),
        mean_total_force_N=float(np.sum(force.sum(axis=1)*overlap)/(overlap.sum())),
        query='raw post-step cached force, interval overlap over saved start+duration; not filtered')


def window_variation(t,force,lo,hi):
    mask=(t>=lo-1e-8)&(t<=hi+1e-8)
    if mask.sum()<2 or abs(t[mask][0]-lo)>1e-7 or abs(t[mask][-1]-hi)>1e-7:
        return dict(available=False,window_s=[lo,hi],value=None)
    value=force_variation(t[mask],force[mask])
    return dict(available=value is not None,window_s=[lo,hi],value=value,
        interpretation='Transition path variation includes intentional unloading/contact changes; reversal is not automatically oscillation')


def values(row):
    if row['status']!='complete':return {}
    m=row['metrics'];c=row['clearance'];raw=c.get('physical_end2000Hz',{});coarse=c.get('observed200Hz',{})
    window=c.get('raw_inside_original200Hz_first_confirmation_window') or {}
    release=row['phase_costs']['scheduled_release_12_16']
    tv=row['force_variation']['release_12_16']
    return dict(mission=m['mission_success'],full_motion_mission=m['full_motion_mission_success'],
        guard=m['safety']['guard_pass'],preparation=m['preparation_passed'],
        physical_release200Hz=m['release']['physical_success'],task_qualified_release=m['release']['task_qualified_success'],
        force_peak_N=max(m['safety']['finger_peak_N']),local_excess_Ns=m['safety']['local_excess_integral_Ns'],
        total_excess_Ns=m['safety']['total_excess_integral_Ns'],release_excess_Ns=release.get('local_excess_Ns'),
        post_origin_excess_Ns=row['phase_costs']['post_origin_12_020_16'].get('local_excess_Ns'),
        initial_stability_s=m['initial_stability_wait_s'],
        original_first_clear200Hz_s=m['release']['physical_clearance_latency_s'],
        first_any_frame_s=coarse.get('first_clear_confirmation_latency_s'),
        first_by_deadline_frame_s=coarse.get('first_by_deadline_confirmation_latency_s'),
        first_any_raw_end_s=raw.get('first_clear_confirmation_latency_s'),
        first_by_deadline_raw_end_s=raw.get('first_by_deadline_confirmation_latency_s'),
        terminal_low_force_raw_end_s=raw.get('terminal_clear_confirmation_latency_s'),
        raw_recontact_episodes=raw.get('recontact_count'),
        raw_touch_duration_inside_original_confirmation_s=window.get('above_threshold_duration_s'),
        raw_peak_inside_original_confirmation_N=window.get('peak_target_contact_N'),
        release_target_raw200Hz_TV_rate_N_s=tv['target_frame200Hz']['value']['finger_TV_rate_N_s'] if tv['target_frame200Hz']['available'] else None,
        release_skin_raw2000Hz_TV_rate_N_s=tv['skin_cached2000Hz']['value']['finger_TV_rate_N_s'] if tv['skin_cached2000Hz']['available'] else None,
        release_target_FIR50Hz_TV_rate_N_s=tv['target_FIR50Hz_frame_grid']['value']['finger_TV_rate_N_s'] if tv['target_FIR50Hz_frame_grid']['available'] else None)


def self_test():
    checks=[]
    def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
    check('missing_clearance_not_zero',paired_difference(None,.8) is None and paired_difference(.8,None) is None)
    check('paired_latency_sign_is_candidate_minus_zero',abs(paired_difference(.8,.9)-.1)<1e-12)
    x=scalar_summary([None,0.,2.]);check('zero_is_real_missing_is_not_zero',x['mean']==1. and x['available']==2 and x['assigned']==3)
    fake=[dict(zero_values={'x':True},slow_values={'x':False}),dict(zero_values={'x':False},slow_values={'x':True}),dict(zero_values={},slow_values={})]
    b=binary_pairs(fake,'x');check('all_binary_outcomes_kept',b['zero_only']==1 and b['slow_only']==1 and b['unavailable']==1)
    force=np.ones((4,5))*2;cost=phase_cost(np.arange(4)*.5,np.ones(4)*.5,force,np.ones(5),9.,.25,1.25)
    check('phase_overlap_not_start_label_approximation',abs(cost['local_excess_Ns']-5)<1e-12 and abs(cost['total_excess_Ns']-1)<1e-12)
    t=np.linspace(0,1,5);f=np.tile([0,1,0,1,0],(5,1)).T.astype(float);v=window_variation(t,f,0,1)['value']
    check('TV_drift_retrace_identity',v['finger_TV_N']==20 and v['finger_net_drift_N']==0 and v['finger_reversal_N']==10)
    check('missing_window_is_unavailable',not window_variation(t,f,0,1.2)['available'])
    return dict(passed=all(c['passed'] for c in checks),checks=checks,physics_steps=0)


def execute(run,out):
    run=run.resolve();out=out.resolve();allowed=(ROOT/'output').resolve()
    if not run.is_relative_to(allowed):raise ValueError('Actual research/output required')
    # Absolutely no branch/partial read above this complete terminal gate.
    marker=run/'execution_complete.json'
    if not marker.is_file():raise RuntimeError('NOT_READY: no terminal marker; no branch result read')
    terminal=read(marker)
    if terminal.get('execution_complete') is not True or terminal.get('planned')!=32:
        raise RuntimeError('NOT_READY: terminal32 assignment closure required; no branch result read')
    if out.exists() or not out.is_relative_to((ROOT/'report').resolve()):
        raise ValueError('Fresh research/report output required; old report never overwritten')
    manifest=read(run/'manifest.json');plan=read(run/'plan.json');summary=read(run/'summary.json')
    jobs=manifest['jobs'];terminal_rows={r['job_id']:r for r in summary['results']}
    checks=[];issues=[];input_hashes={}
    def check(name,ok,detail=None):
        c=dict(name=name,passed=bool(ok),detail=detail);checks.append(c)
        if not ok:issues.append(c)
    for p in (marker,run/'manifest.json',run/'plan.json',run/'summary.json',run/'results.json',run/'assignment_ledger.json',run/'source_snapshot_manifest.json'):
        input_hashes[str(p.relative_to(ROOT))]=sha(p)
    check('exact32unique_assignments',len(jobs)==32 and len({j['job_id'] for j in jobs})==32)
    check('all_terminal_records_retained',set(terminal_rows)=={j['job_id'] for j in jobs})
    check('recorded_sources_unchanged',terminal.get('sources_unchanged') is True and summary.get('sources_unchanged') is True)
    expected={(s,m,seed,v) for s in plan['scenarios'] for m in plan['motion_modes'] for seed in plan['seeds'] for v in plan['variants']}
    actual={(j['scenario'],j['motion_mode'],j['seed'],j['variant']) for j in jobs}
    check('predeclared_full_factorial_exact',expected==actual)
    rows=[];protocol=read(ROOT/'protocol.json')
    for job in jobs:
        folder=Path(job['directory']);r=terminal_rows[job['job_id']]
        row=dict(job=job,status=r['status'],terminal_result=r)
        if r['status']!='complete':
            row['unavailable_reason']='assigned runtime failure; no zero-filled metrics';rows.append(row);continue
        result=read(folder/'result.json');m=result['research_metrics'];c=read(folder/'clearance_diagnostics.json');meta=read(folder/'pilot_metadata.json')
        trace_path=folder/'record/trace.npz';cmd_path=folder/'decoder_commands.npz';physical_path=folder/'robot_physical_stream.npz'
        for p in (folder/'result.json',folder/'pilot_metadata.json',folder/'clearance_diagnostics.json',trace_path,cmd_path,physical_path):input_hashes[str(p.relative_to(ROOT))]=sha(p)
        check(job['job_id']+'/trace_identity',sha(trace_path)==r['trace_sha256']==meta['record_trace_sha256'])
        check(job['job_id']+'/clearance_identity',sha(folder/'clearance_diagnostics.json')==meta['clearance_diagnostics_sha256'])
        with np.load(trace_path,allow_pickle=False) as z:
            names=['time','normal','skin_normal','support','controller_active','substep_time','substep_duration',
                   'substep_normal','substep_skin_normal','substep_all_hand_object','all_hand_object']
            trace={k:z[k] for k in names};md=json.loads(str(z['metadata']))
            t=trace['time'];st=trace['substep_time'];dur=trace['substep_duration'];f=trace['substep_skin_normal']
            limits=np.asarray(md['limits']);budget=float(md['budget'])
            coverage=bool(len(t)==3201 and len(st)==32000 and np.allclose(t,np.arange(3201)*.005,atol=1e-8,rtol=0)
                and np.allclose(st,np.arange(32000)*.0005,atol=1e-7,rtol=0) and np.allclose(dur,.0005,atol=1e-9,rtol=0))
            check(job['job_id']+'/full_force_clock_coverage',coverage)
            peak=np.maximum(f.max(axis=0),trace['skin_normal'][-1]);excess=float(np.sum(np.maximum(f-limits,0)*dur[:,None]));total=float(np.maximum(f.sum(axis=1)-budget,0)@dur)
            tol=protocol['thresholds']['force_numerical_tolerance_N']
            guard=bool(coverage and (f<=limits+tol).all() and (f.sum(axis=1)<=budget+tol).all()
                and (trace['skin_normal'][-1]<=limits+tol).all() and trace['skin_normal'][-1].sum()<=budget+tol)
            check(job['job_id']+'/raw_safety_recomputed',np.allclose(peak,m['safety']['finger_peak_N'],rtol=0,atol=1e-12)
                and abs(excess-m['safety']['local_excess_integral_Ns'])<1e-12 and abs(total-m['safety']['total_excess_integral_Ns'])<1e-12
                and guard==m['safety']['guard_pass'])
            band,band_contract=force_band(trace,md,protocol)
            windows={'release_12_16':(12.,16.),'holding_7_5_12':(7.5,12.),'release_prefix_12_020':(12.,12.02),'post_origin_12_020_16':(12.02,16.)}
            variation={}
            cached_clock=np.r_[0.,st+dur]
            for key,(lo,hi) in windows.items():
                variation[key]={
                    'target_frame200Hz':window_variation(t,trace['normal'],lo,hi),
                    'skin_frame200Hz':window_variation(t,trace['skin_normal'],lo,hi),
                    'target_cached2000Hz':window_variation(cached_clock,np.vstack([trace['normal'][0],trace['substep_normal']]),lo,hi),
                    'skin_cached2000Hz':window_variation(cached_clock,np.vstack([trace['skin_normal'][0],trace['substep_skin_normal']]),lo,hi),
                    'target_FIR50Hz_frame_grid':window_variation(t,band,lo,hi) if band is not None else dict(available=False,value=None)}
            costs={key:phase_cost(st,dur,f,limits,budget,lo,hi) for key,(lo,hi) in {
                'whole_0_16':(0.,16.),'shared_preparation_0_7_5':(0.,7.5),'scheduled_holding_7_5_12':(7.5,12.),
                'scheduled_release_12_16':(12.,16.),'release_prefix_12_020':(12.,12.02),
                'post_origin_12_020_16':(12.02,16.),'first200ms_after_origin':(12.02,12.22)}.items()}
        with np.load(physical_path,allow_pickle=False) as z:
            check(job['job_id']+'/robot_physical_coverage',len(z['physical_time_s'])==32001)
        with np.load(cmd_path,allow_pickle=False) as z:
            commands=dict(applications=len(z['control_time']),
                maximum_actual_phi_offset_s=float(abs(z['actual_phi_offset_s']).max()),
                maximum_actual_lift_offset_m=float(z['actual_lift_offset_m'].max()),
                actual_reference_start_s=float(z['control_time'][0]),
                first_nonzero_lift_offset_control_s=float(z['control_time'][np.flatnonzero(z['actual_lift_offset_m']>1e-12)[0]]) if np.any(z['actual_lift_offset_m']>1e-12) else None)
        row.update(metrics=m,clearance=c,metadata=md,actual_config=meta['config'],
            entry_eligible=meta['primitive_entry_eligible'],offset_enabled=meta['primitive_offset_enabled'],
            actual_mechanical_release_s=meta['actual_mechanical_release_s'],
            commands=commands,force_variation=variation,phase_costs=costs,force_filter_contract=band_contract)
        row['values']=values(row);rows.append(row)
    bykey={(r['job']['scenario'],r['job']['motion_mode'],r['job']['seed'],r['job']['variant']):r for r in rows}
    pairs=[]
    for seed in plan['seeds']:
        for scene in plan['scenarios']:
            for motion in plan['motion_modes']:
                z=bykey[(scene,motion,seed,'zero')];s=bykey[(scene,motion,seed,'lift_slow')]
                a=z.get('values',{});b=s.get('values',{})
                p=dict(scenario=scene,motion_mode=motion,seed=seed,zero_status=z['status'],slow_status=s['status'],
                    zero_values=a,slow_values=b,difference_slow_minus_zero={},component_exposed=bool(s.get('offset_enabled',False)))
                if a and b:
                    p['difference_slow_minus_zero']={k:paired_difference(a[k],b[k]) for k in a if not isinstance(a[k],bool)}
                    p['config_exact']=z['actual_config']==s['actual_config']
                    check(f'{scene}/{motion}/{seed}/matched_config',p['config_exact'])
                    p['retention_and_preparation_outcomes_same']=all(z['metrics'][k]==s['metrics'][k] for k in ('preparation_passed','initial_stability_wait_s','failure_reason','failure_at'))
                    p['retention_check_note']='failure_reason may additionally include release failure; compare raw prefix/holding metrics before attributing any difference'
                recorded=next((x for x in summary['pairs'] if x['scenario']==scene and x['motion_mode']==motion and x['seed']==seed),None)
                p['execution_recorded_pairing']=recorded
                p['causal_fixed_primitive_comparison_available']=bool(recorded and recorded['paired'] and a and b and p.get('config_exact'))
                pairs.append(p)
    metrics=[k for r in rows if r.get('values') for k,v in r['values'].items() if not isinstance(v,bool)]
    metric_names=sorted(set(metrics)|set(REPORT_SCALARS))
    groups={}
    for variant in plan['variants']:
        rr=[r for r in rows if r['job']['variant']==variant]
        groups[variant]=dict(assigned=len(rr),complete=sum(r['status']=='complete' for r in rr),
            entry_eligible=sum(bool(r.get('entry_eligible')) for r in rr),offset_enabled=sum(bool(r.get('offset_enabled')) for r in rr),
            binary={key:dict(true=sum(r.get('values',{}).get(key) is True for r in rr),assigned=len(rr),
                unavailable=sum(key not in r.get('values',{}) for r in rr)) for key in (
                    'preparation','mission','full_motion_mission','guard','physical_release200Hz','task_qualified_release')},
            scalar={key:scalar_summary(r.get('values',{}).get(key) for r in rr) for key in metric_names})
    # Quality averages only use the identical common eligible pairs per window.
    quality={}
    for name in ('static_quality','dynamic_quality','entire_holding_quality'):
        qp=[]
        for p in pairs:
            z=bykey[(p['scenario'],p['motion_mode'],p['seed'],'zero')]
            s=bykey[(p['scenario'],p['motion_mode'],p['seed'],'lift_slow')]
            common=bool(z.get('metrics',{}).get(name,{}).get('eligible') and s.get('metrics',{}).get(name,{}).get('eligible'))
            qp.append(dict(scenario=p['scenario'],motion_mode=p['motion_mode'],seed=p['seed'],common_eligible=common,
                zero=z.get('metrics',{}).get(name),lift_slow=s.get('metrics',{}).get(name)))
        quality[name]=dict(assigned_pairs=16,common_eligible=sum(p['common_eligible'] for p in qp),pairs=qp,
            zero_common_TV_mean=scalar_summary(p['zero']['bandlimited']['finger_TV_rate_N_s'] for p in qp if p['common_eligible']),
            slow_common_TV_mean=scalar_summary(p['lift_slow']['bandlimited']['finger_TV_rate_N_s'] for p in qp if p['common_eligible']))
    # Pairwise timing summaries explicitly state conditional availability and do
    # not substitute censored or failed releases with perfect/zero latency.
    differences={k:scalar_summary(p['difference_slow_minus_zero'].get(k) for p in pairs) for k in metric_names}
    by_seed={str(seed):dict(pairs=8,
        differences={k:scalar_summary(p['difference_slow_minus_zero'].get(k) for p in pairs if p['seed']==seed) for k in metric_names},
        binary={k:binary_pairs([p for p in pairs if p['seed']==seed],k) for k in ('full_motion_mission','guard','physical_release200Hz')}) for seed in plan['seeds']}
    out.mkdir(parents=True)
    report=dict(schema='prospective_release_primitive_v2_author_readout_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        role='Runner and readout author; separate independent audit is not replaced',run=str(run),
        source=dict(checker_sha256=sha(Path(__file__)),input_sha256=input_hashes,
            frozen_execution_sources=manifest['sources'],bridge_regression=summary.get('bridge_regression')),
        scope=dict(assigned_trials=32,paired_contexts=16,physical_seed_clusters=len(plan['seeds']),
            seeds=plan['seeds'],shape='capsule',phase='prospective_development_not_confirmation',
            candidate_selected_after_seen_training_data=True,complete_fusion_algorithm=False),
        checks=checks,issues=issues,rows=rows,pairs=pairs,groups=groups,paired_differences=differences,by_physical_seed=by_seed,
        common_quality=quality,binary_pair_tables={k:binary_pairs(pairs,k) for k in ('mission','full_motion_mission','guard','physical_release200Hz','task_qualified_release')},
        clocks=dict(frame_s=.005,raw_start_plus_duration_end_s=.0005,
            query_semantics='Frame contact force is recomputed by extra mj_forward; raw cached integration force is not merely a subsample of that channel',
            force_FIR='fixed50Hz cutoff/50ms span V3 lifecycle reflection; release-prefix filter can use later post-origin release values, never causal pre-origin evidence'),
        limitations=['32 assigned trials are16 paired contexts with only2 reused physical/noise seed clusters; no population inference or CI.',
            'Whole-task reliability/guard retain all assigned attempts. Missing/censored scalar times remain unavailable, with counts.',
            'No weighted score, new threshold, success-only cohort, Pareto-dominance requirement, or posthoc numerical utility is introduced.',
            'A small reliability cost may be weighed by the user against force/time quality, but this release-only component cannot establish a grasp-success tradeoff or full fusion advantage.',
            'Holding and pre-origin raw/observation prefixes must remain shared; any equal pre-release quality is inherited, not a new primitive benefit.',
            'Low-force dwell is not geometric gap/permanent separation, raw cached observations are not mathematical continuous-contact safety.',
            'TV includes useful release unloading and contact transitions. Net drift/retracing are separate; peak/excess decrease is not automatically TV improvement.',
            'Source/pairing execution pass is distinct from forceguard, measured empirical effect and global model/safety admission.'],
        new_physics_steps=0,performance_claim_allowed=False)
    write(out/'readout.json',report)
    fmt=lambda x:'NULL' if x is None else f'{x:.7g}'
    lines=['# 固定释放原语：32次前瞻开发配对读数','',
        '作者角色：runner/报告作者；独立审计另列。32全分母、16配对情境，但只有2个复用的物理/噪声种子cluster。无新仿真、评分权重或阈值。',
        '', '|分支|完整/16|准备通过/16|完整任务/16|全16s力限/16|200Hz物理释放/16|实际offset暴露/16|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for v,g in groups.items():
        b=g['binary'];lines.append(f"|{v}|{g['complete']}|{b['preparation']['true']}|{b['full_motion_mission']['true']}|{b['guard']['true']}|{b['physical_release200Hz']['true']}|{g['offset_enabled']}|")
    lines+=['','差值为lift_slow−zero；缺失不是0。时间按共同可用配对另列，不能把成功样本时间替代全任务成功率。',
        '', '|维度|zero可用/16|zero均值|slow可用/16|slow均值|配对差可用/16|配对差均值|', '|---|---:|---:|---:|---:|---:|---:|']
    for k in REPORT_SCALARS:
        a=groups['zero']['scalar'][k];b=groups['lift_slow']['scalar'][k];d=differences[k]
        lines.append(f"|{k}|{a['available']}|{fmt(a['mean'])}|{b['available']}|{fmt(b['mean'])}|{d['available']}|{fmt(d['mean'])}|")
    lines+=['','|seed|环境|任务|实际offset|Δ超限N·s|Δ峰N|Δ原200Hz首次s|Δraw期限内首次s|Δraw末段s|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for p in pairs:
        d=p['difference_slow_minus_zero'];lines.append(f"|{p['seed']}|{p['scenario']}|{p['motion_mode']}|{int(p['component_exposed'])}|{fmt(d.get('local_excess_Ns'))}|{fmt(d.get('force_peak_N'))}|{fmt(d.get('original_first_clear200Hz_s'))}|{fmt(d.get('first_by_deadline_raw_end_s'))}|{fmt(d.get('terminal_low_force_raw_end_s'))}|")
    lines+=['','静态/动态/整持握原V3品质保持NULL规则，JSON另列16配对中的共同可用品质窗口；释放干预不能解释为抓握成功或前段稳定品质改进。',
        '200Hz额外forward与2kHz缓存积分力查询语义不同，不能仅称采样分辨率。完整峰/超限积分、首次、期限内、末段持续低力、再接触及原确认窗中的raw触点全部分列。',
        '用户可以评估某项可靠性代价换取力/时间等收益；本报告不临时发明权重、不要求全指标支配，也不据局部原语宣称完整fusion优势。',
        '',f"完整JSON SHA256 `{sha(out/'readout.json')}`。"]
    with (out/'readout.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(out),checks=len(checks),issues=len(issues),report_sha256=sha(out/'readout.json'),new_physics_steps=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--self-test',action='store_true');g.add_argument('--execute',action='store_true')
    p.add_argument('--run',type=Path,default=ROOT/'output/release_primitive_prospective_v2')
    p.add_argument('--out',type=Path,default=ROOT/'report/release_primitive_prospective_v2_author_v1')
    a=p.parse_args()
    if a.self_test:
        r=self_test();print(json.dumps(r));raise SystemExit(0 if r['passed'] else 2)
    execute(a.run,a.out)

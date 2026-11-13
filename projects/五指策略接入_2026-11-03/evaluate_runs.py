"""Independent full-hand task, force, stability and paired-result evaluation."""
from pathlib import Path
import argparse,json,hashlib,itertools
import numpy as np

from vendor.frozen_evaluation import force_quality_trace
from session import ROOT,HANDOFF,DURATION,jsonable,sha,load_json


def first_dwell(t,mask,start,end,dwell=.2):
    since=None
    for i in np.flatnonzero((t>=start-1e-9)&(t<=end+1e-9)):
        if mask[i]:
            if since is None:since=float(t[i])
            if t[i]-since>=dwell-1e-9:return float(t[i])
        else:since=None
    return None

def evaluate_trace(path):
    path=Path(path)
    needed=('time','normal','skin_normal','palm','hand_other','all_hand_object','support','position','relative',
        'relative_velocity','angular_velocity','lift','lift_ref','warning_count','transport_paused',
        'substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm',
        'substep_hand_other','substep_all_hand_object','substep_support','collision',
        'alpha','local_domain_active','local_domain_lower','local_domain_upper','requested_action','executed_action')
    with np.load(path,allow_pickle=False) as saved:
        meta=json.loads(str(saved['metadata']))
        trace={k:saved[k].copy() for k in needed if k in saved.files}
    t=np.asarray(trace.get('time',[]),float);planned=meta.get('mode')=='scheduled'
    base=dict(run_id=meta.get('run_id'),method=meta['method'],scenario=meta['scenario']['name'],
        shape=meta['shape'],seed=meta['seed'],split=meta['split'],mode=meta['mode'],
        boundary=bool(meta['scenario'].get('boundary',False)),trace_path=str(path.resolve()),
        trace_sha256=sha(path),admission_sha256=meta.get('admission_sha256'),
        task_success=False,record_complete=False,A_S=1.4,first_stable_after_handoff=None,
        whole_task_first_stable=None,static_eligible=False,mean_force=None,finger_tv_rate=None,
        release_planned=planned,release_exposed=False,release_success=False,release_latency=None,
        failure_reason=meta.get('failure_reason'))
    if len(t)<2:return {**base,'failure_reason':'missing_truth_samples'}
    optional=('collision','alpha','local_domain_active','local_domain_lower','local_domain_upper','requested_action','executed_action')
    required=[k for k in needed if not k.startswith('substep_') and k not in optional]
    complete=(all(k in trace for k in required) and abs(t[0])<1e-8 and abs(t[-1]-DURATION)<1e-8
              and len(t)==round(DURATION/.005)+1 and np.allclose(np.diff(t),.005,atol=1e-8,rtol=0))
    for k in required:
        if k in trace:complete &= bool(np.isfinite(trace[k]).all())
    ref=meta.get('reference_relative')
    normal=np.asarray(trace['normal']);skin=np.asarray(trace['skin_normal']);limits=np.asarray(meta['limits'])
    contact=(normal>.2);support=contact[:,0]&(np.sum(contact,axis=1)>=3)
    lift_error=np.abs(trace['lift']-trace['lift_ref']);motion=(lift_error<=.01)&(trace['support']<=.05)
    initial=np.asarray(meta.get('initial_position',trace['position'][0]),float)
    raised=trace['position'][:,2]>=initial[2]+.025
    relative_error=np.full(len(t),np.inf) if ref is None else np.linalg.norm(trace['relative']-np.asarray(ref),axis=1)
    safe=np.all(skin<=limits[None,:]+.02,axis=1)&(skin.sum(axis=1)<=float(meta['budget'])+.02)
    stable=(support&motion&raised&safe&(relative_error<=.005)&
            (np.linalg.norm(trace['relative_velocity'],axis=1)<=.005)&
            (np.linalg.norm(trace['angular_velocity'],axis=1)<=.15))
    post=first_dwell(t,stable,HANDOFF,8.9)
    whole=first_dwell(t,stable,0.,DURATION)
    release_at=meta.get('release_at');hold_end=12. if planned else (release_at if release_at is not None else DURATION)
    established=first_dwell(t,support&motion&raised,0.,HANDOFF,dwell=.1)
    failure_at=None;failure=meta.get('preparation_failure')
    if not meta.get('handoff_passed') or established is None:
        failure=failure or 'preparation_or_establishment_failed';failure_at=HANDOFF
    else:
        contact_loss=motion_loss=None
        for i in np.flatnonzero((t>=HANDOFF-1e-9)&(t<hold_end-1e-9)):
            lost=not support[i] or not raised[i] or relative_error[i]>.08
            if lost:
                if contact_loss is None:contact_loss=float(t[i])
                if t[i]-contact_loss>=.1-1e-9:
                    failure_at=float(t[i]);failure='support_or_object_lost';break
            else:contact_loss=None
            if lift_error[i]>.01:
                if motion_loss is None:motion_loss=float(t[i])
                if t[i]-motion_loss>=.3-1e-9:
                    failure_at=float(t[i]);failure='transport_reference_unmet';break
            else:motion_loss=None
    quality=force_quality_trace(trace,meta,dict(require_full_substeps=True,require_skin_force=True,
        evaluation_sample_interval=.005,force_bandwidth_hz=50.,force_filter_span=.05))
    complete &= bool(quality['contract']['valid'] and quality['contract']['full_substep_coverage'])
    if not complete:failure=failure or 'incomplete_truth_or_force_record'
    window=(t>=8.2-1e-9)&(t<=8.9+1e-9)
    eligible=bool(complete and established is not None and meta.get('handoff_passed') and
                  (failure_at is None or failure_at>8.9) and np.all(support[window]) and np.count_nonzero(window)>=141)
    metrics={}
    if eligible:
        actual=normal[window];filtered=np.asarray(quality['normal'])[window];tw=t[window]
        metrics=dict(mean_force=float(np.mean(actual.sum(axis=1))),
            finger_tv_rate=float(np.abs(np.diff(filtered,axis=0)).sum()/(tw[-1]-tw[0])),
            finger_sd_mean=float(np.std(actual,axis=0).mean()),total_sd=float(np.std(actual.sum(axis=1))),
            raw_finger_tv_rate=float(np.abs(np.diff(actual,axis=0)).sum()/(tw[-1]-tw[0])))
    short=(t>=8.3-1e-9)&(t<=8.8+1e-9)
    short_eligible=bool(complete and established is not None and meta.get('handoff_passed') and
        (failure_at is None or failure_at>8.8) and np.all(support[short]) and np.count_nonzero(short)>=101)
    short_metrics={}
    if short_eligible:
        shorter=np.asarray(quality['normal'])[short];ts=t[short]
        short_metrics=dict(short_mean_force=float(np.mean(normal[short].sum(axis=1))),
            short_finger_tv_rate=float(np.abs(np.diff(shorter,axis=0)).sum()/(ts[-1]-ts[0])))
    st=np.asarray(trace.get('substep_time',[]));dur=np.asarray(trace.get('substep_duration',[]))
    physical=np.asarray(trace.get('substep_skin_normal',np.zeros((0,5))));safety={}
    if len(st) and physical.shape==(len(st),5) and dur.shape==st.shape:
        local=np.maximum(physical-limits,0.);total=np.maximum(physical.sum(axis=1)-float(meta['budget']),0.)
        terminal=np.any(skin[-1]>limits+.02) or skin[-1].sum()>float(meta['budget'])+.02
        safety=dict(force_protection_pass=bool(complete and not terminal and np.all(physical<=limits+.02) and np.all(physical.sum(axis=1)<=float(meta['budget'])+.02)),
            local_excess_integral=float(np.sum(local*dur[:,None])),total_excess_integral=float(total@dur),
            finger_peak=np.maximum(physical.max(axis=0),skin[-1]).tolist(),
            palm_peak=float(max(np.max(trace.get('substep_palm',[0.])),trace['palm'][-1])),
            protected_scope='all instrumented finger external contacts; palm/other recorded without material-damage threshold')
        controlled=st>=HANDOFF-1e-9
        safety['post_handoff_local_excess_integral']=float(np.sum(local[controlled]*dur[controlled,None]))
        safety['post_handoff_total_excess_integral']=float(total[controlled]@dur[controlled])
        mechanical_release=meta.get('mechanical_release_at',meta.get('release_at'))
        epochs={'preparation':st<HANDOFF-1e-9}
        if meta.get('handoff_passed'):
            epochs['policy_holding']=(st>=HANDOFF-1e-9)&(st<(float(mechanical_release) if mechanical_release is not None else DURATION)-1e-9)
        else:epochs['post_preparation_without_policy']=(st>=HANDOFF-1e-9)&(st<(float(mechanical_release) if mechanical_release is not None else DURATION)-1e-9)
        if mechanical_release is not None:epochs['mechanical_release_and_after']=st>=float(mechanical_release)-1e-9
        safety['stages']={name:dict(duration_s=float(np.sum(dur[mask])),
            finger_peak_N=physical[mask].max(axis=0).tolist(),
            total_finger_peak_N=float(physical[mask].sum(axis=1).max()),
            local_excess_integral_Ns=float(np.sum(local[mask]*dur[mask,None])),
            total_excess_integral_Ns=float(total[mask]@dur[mask]))
            for name,mask in epochs.items() if np.any(mask)}
        safety['stage_scope']='actual common preparation / admitted policy hold / mechanical release; failed preparation does not acquire a policy epoch'
    release_planned=planned or release_at is not None
    exposure=False;released=False;release_time=None
    if release_at is not None and complete:
        index=int(np.argmin(abs(t-float(release_at))))
        exposure=bool(established is not None and (failure_at is None or failure_at>release_at) and
                      support[index] and trace['all_hand_object'][index]>.2 and meta.get('release_supported_at_command'))
        cleared=trace['all_hand_object']<=.2
        end=first_dwell(t,cleared,float(release_at),min(DURATION,float(release_at)+3.5),.2)
        released=bool(exposure and end is not None)
        release_time=float(end-release_at) if released else None
    task_success=bool(complete and meta.get('handoff_passed') and failure is None and
                      (not release_planned or released) and np.max(trace['warning_count'])==0)
    if release_planned and not released and failure is None:failure='release_failed_or_not_exposed'
    hold=(t>=HANDOFF-1e-9)&(t<hold_end-1e-9)
    collision={}
    if 'collision' in trace:
        observed=np.asarray(trace['collision'],float);touch=np.flatnonzero(observed>.02)
        collision=dict(actual_collision_exposed=bool(len(touch)),collision_peak_200Hz_N=float(np.max(observed)),
            collision_first_200Hz_time=float(t[touch[0]]) if len(touch) else None,
            collision_scope='actual external fixture contact sampled at 200Hz; complete all-skin substeps separately guard force peaks')
    domain={}
    if all(k in trace for k in ('alpha','local_domain_active','local_domain_lower','local_domain_upper')):
        active=np.asarray(trace['local_domain_active'],bool)
        at_boundary=np.any((trace['alpha']-trace['local_domain_lower']<=1e-6)|
                           (trace['local_domain_upper']-trace['alpha']<=1e-6),axis=1)
        domain=dict(local_domain_samples=int(np.sum(active)),
            local_domain_boundary_fraction=float(np.mean(at_boundary[active])) if np.any(active) else None,
            static_window_domain_boundary_fraction=float(np.mean(at_boundary[window&active])) if np.any(window&active) else None,
            local_domain_metric_scope='commanded five-synergy coordinates, actual admitted hold only; not unconstrained policy preference or all joint saturation')
        if 'requested_action' in trace and 'executed_action' in trace:
            mismatch=np.any(abs(trace['requested_action']-trace['executed_action'])>1e-6,axis=1)
            domain['execution_request_mismatch_fraction']=float(np.mean(mismatch[active])) if np.any(active) else None
    return jsonable({**base,**metrics,**short_metrics,'task_success':task_success,'record_complete':bool(complete),
        **collision,**domain,
        'preparation_passed':bool(meta.get('handoff_passed')),'established':established,
        'A_S':float(post-HANDOFF) if post is not None else 1.4,
        'first_stable_after_handoff':post,'whole_task_first_stable':whole,
        'stable_hold_fraction':float(np.mean(stable[hold])) if np.any(hold) else 0.,
        'static_eligible':eligible,'safety':safety,'failure_reason':failure,'failure_at':failure_at,
        'short_window_eligible':short_eligible,
        'release_planned':release_planned,'release_exposed':exposure,'release_success':released,
        'release_latency':release_time,'force_quality_contract':quality['contract'],
        'task_reference_kind':meta.get('reference_kind')})


def summarize(records,bootstrap=5000):
    rows=[];comparisons=[]
    for split in sorted({r['split'] for r in records}):
        selected=[r for r in records if r['split']==split and r.get('mode')=='scheduled']
        for domain in ('main','boundary'):
            group=[r for r in selected if bool(r.get('boundary'))==(domain=='boundary')]
            for method in ('ratio','unified'):
                m=[r for r in group if r['method']==method]
                if not m:continue
                quality=[r for r in m if r.get('static_eligible')]
                rows.append(dict(split=split,domain=domain,method=method,
                    method_title='摩擦比基线' if method=='ratio' else '统一协调策略',
                    trials=len(m),successes=sum(r['task_success'] for r in m),
                    preparation_passes=sum(r.get('preparation_passed',False) for r in m),
                    mean_force=float(np.mean([r['mean_force'] for r in quality])) if quality else None,
                    tv=float(np.mean([r['finger_tv_rate'] for r in quality])) if quality else None,
                    quality_windows=len(quality),A_S=float(np.mean([r['A_S'] for r in m])),
                    force_passes=sum(r.get('safety',{}).get('force_protection_pass',False) for r in m),
                    total_excess_integral=float(np.mean([r.get('safety',{}).get('total_excess_integral',np.nan) for r in m])),
                    local_excess_integral=float(np.mean([r.get('safety',{}).get('local_excess_integral',np.nan) for r in m])),
                    releases=sum(r.get('release_success',False) for r in m),
                    planned_releases=sum(r.get('release_planned',False) for r in m),
                    local_domain_boundary_fraction=float(np.mean([r['local_domain_boundary_fraction'] for r in m if r.get('local_domain_boundary_fraction') is not None])) if any(r.get('local_domain_boundary_fraction') is not None for r in m) else None,
                    static_window_domain_boundary_fraction=float(np.mean([r['static_window_domain_boundary_fraction'] for r in m if r.get('static_window_domain_boundary_fraction') is not None])) if any(r.get('static_window_domain_boundary_fraction') is not None for r in m) else None,
                    actual_collision_exposures=sum(r.get('actual_collision_exposed',False) for r in m),
                    failures={reason:sum(r.get('failure_reason')==reason for r in m)
                              for reason in sorted({r.get('failure_reason') for r in m if r.get('failure_reason')})}))
            baseline={(r['scenario'],r['shape'],r['seed']):r for r in group if r['method']=='ratio'}
            candidate={(r['scenario'],r['shape'],r['seed']):r for r in group if r['method']=='unified'}
            keys=sorted(baseline.keys()&candidate.keys());rng=np.random.default_rng(20261103)
            for endpoint in ('task_success','A_S','finger_tv_rate','mean_force'):
                pairs=[(k,baseline[k],candidate[k]) for k in keys if endpoint not in ('finger_tv_rate','mean_force') or
                       (baseline[k].get('static_eligible') and candidate[k].get('static_eligible'))]
                if not pairs:continue
                seeds=sorted({k[2] for k,_,_ in pairs})
                seed_rows={seed:[(float(c[endpoint]),float(b[endpoint])) for k,b,c in pairs if k[2]==seed] for seed in seeds}
                draws=[]
                for _ in range(bootstrap):
                    values=[v for seed in rng.choice(seeds,len(seeds),replace=True) for v in seed_rows[int(seed)]]
                    draws.append(float(np.mean([a-b for a,b in values])))
                differences=[float(c[endpoint])-float(b[endpoint]) for _,b,c in pairs]
                comparison=dict(split=split,domain=domain,endpoint=endpoint,pairs=len(pairs),clusters=len(seeds),
                    baseline_mean=float(np.mean([b[endpoint] for _,b,c in pairs])),
                    candidate_mean=float(np.mean([c[endpoint] for _,b,c in pairs])),difference=float(np.mean(differences)),
                    ci95=np.quantile(draws,[.025,.975]).tolist(),bootstrap_replicates=bootstrap)
                if endpoint in ('A_S','finger_tv_rate') and len(seeds)<=16:
                    observed=abs(float(np.mean(differences)));extreme=0;total=0
                    for signs in itertools.product((-1.,1.),repeat=len(seeds)):
                        perm=[sign*(a-b) for seed,sign in zip(seeds,signs) for a,b in seed_rows[seed]]
                        extreme+=int(abs(float(np.mean(perm)))>=observed-1e-12);total+=1
                    comparison['p']=extreme/total
                comparisons.append(comparison)
            primary=[r for r in comparisons if r['split']==split and r['domain']==domain and r['endpoint'] in ('A_S','finger_tv_rate') and 'p' in r]
            ordered=sorted(primary,key=lambda r:r['p']);last=0.
            for i,r in enumerate(ordered):last=max(last,min(1.,(len(ordered)-i)*r['p']));r['p_holm']=last
            for cohort in ('completed_window','short_window'):
                for endpoint in ('finger_tv_rate','mean_force'):
                    field='short_'+endpoint if cohort=='short_window' else endpoint
                    paired=[(k,baseline[k],candidate[k]) for k in keys if
                        (baseline[k].get('short_window_eligible') and candidate[k].get('short_window_eligible') if cohort=='short_window' else
                         baseline[k].get('static_eligible') and candidate[k].get('static_eligible') and
                         baseline[k].get('task_success') and candidate[k].get('task_success'))]
                    if paired:
                        comparisons.append(dict(split=split,domain=domain,endpoint=endpoint,cohort=cohort,
                            role='predeclared_descriptive_sensitivity',pairs=len(paired),
                            baseline_mean=float(np.mean([b[field] for _,b,c in paired])),
                            candidate_mean=float(np.mean([c[field] for _,b,c in paired])),
                            difference=float(np.mean([c[field]-b[field] for _,b,c in paired]))))
    return jsonable(dict(rows=rows,comparisons=comparisons,trial_count=len(records),
        note='共同准备不计作算法优势；主品质使用双方共同有效窗口，失败与窗口覆盖单列。五指为局部固定抓姿迁移，掌部另列。',
        statistical_unit='paired whole seed across scenarios',records=records))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--bootstrap',type=int,default=5000)
    a=p.parse_args();folder=Path(a.root)
    metadata_files=sorted(folder.glob('runs/*/metadata.json'));records=[]
    for path in metadata_files:
        trace=path.parent/'trace.npz'
        if trace.exists():records.append(evaluate_trace(trace))
        else:
            m=load_json(path)
            records.append(dict(method=m['method'],shape=m['shape'],scenario=m['scenario']['name'],seed=m['seed'],
                split=m['split'],mode='scheduled',boundary=m['scenario'].get('boundary',False),
                task_success=False,A_S=1.4,static_eligible=False,record_complete=False,
                release_planned=True,release_exposed=False,release_success=False,failure_reason='runtime_failure'))
    result=summarize(records,a.bootstrap)
    out=folder/'analysis';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 完整五指同条件对照','',result['note'],'',
        '|域|分组|方法|完成|准备成功|有效窗|首次稳定等待 s|平均总指力 N|逐指TV N/s|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['rows']:
        mean='—' if r['mean_force'] is None else f"{r['mean_force']:.3f}"
        tv='—' if r['tv'] is None else f"{r['tv']:.3f}"
        lines.append(f"|{r['split']}|{r['domain']}|{r['method_title']}|{r['successes']}/{r['trials']}|{r['preparation_passes']}|{r['quality_windows']}|{r['A_S']:.3f}|{mean}|{tv}|")
    lines+=['','差异、95%种子簇区间和两主端点Holm检验见summary.json。','A_S为7.5 s接管后重新累计200 ms稳定的等待，未达成记1.4 s；全任务首次稳定另列。','平均力与TV是有效窗口条件值，不能补偿任务失败。']
    (out/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'trials':len(records),'summary':str(out/'summary.json')},ensure_ascii=False))

if __name__=='__main__':main()

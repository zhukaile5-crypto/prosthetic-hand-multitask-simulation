"""Read-only review of the complete saved confirmation matrix; no dynamics.

This reviewer imports only stdlib/numpy plus the earlier independent arithmetic
functions. It never imports production controllers, runtime or evaluator code.
No partial matrix is accepted. Run only when root reports confirmation ready.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import io
import itertools
import json
import runpy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
independent=runpy.run_path(str(ROOT/'review/independent_development_review.py'))
audit_rng=independent['audit_rng']
dwell=independent['dwell']
close=independent['close']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def key(r):
    return r['scenario'],r['shape'],int(r['seed']),r['method']


def first_bad_dwell(t,bad,lo,hi,length):
    since=None
    for i in np.flatnonzero((t>=lo-1e-9)&(t<hi-1e-9)):
        if bad[i]:
            since=float(t[i]) if since is None else since
            if t[i]-since>=length-1e-9:
                return float(t[i])
        else:
            since=None
    return None


def derive(z,m):
    t=z['time'];n=z['normal'];skin=z['skin_normal'];limits=np.asarray(m['limits'])
    support=(n[:,0]>.2)&((n>.2).sum(axis=1)>=3)
    raised=z['position'][:,2]>=np.asarray(m['initial_position'])[2]+.025
    lift_error=abs(z['lift']-z['lift_ref'])
    motion=(lift_error<=.01)&(z['support']<=.05)
    ref=m.get('reference_relative')
    error=np.full(len(t),np.inf) if ref is None else np.linalg.norm(z['relative']-ref,axis=1)
    safe=(skin<=limits+.02).all(axis=1)&(skin.sum(axis=1)<=m['budget']+.02)
    stable=support&raised&motion&safe&(error<=.005)&(np.linalg.norm(z['relative_velocity'],axis=1)<=.005)&(np.linalg.norm(z['angular_velocity'],axis=1)<=.15)
    physical_dt=float(m['physics_timestep']);st=z['substep_time'];duration=z['substep_duration']
    clock_complete=(len(t)==3201 and close(t,np.arange(3201)*.005,1e-8) and len(st)==round(16./physical_dt) and
        close(st,np.arange(len(st))*physical_dt,1e-7) and close(duration,physical_dt,1e-9) and abs(duration.sum()-16.)<1e-8)
    finite_fields=('normal','skin_normal','position','relative','relative_velocity','angular_velocity','lift','lift_ref',
        'warning_count','substep_normal','substep_skin_normal','substep_palm','substep_hand_other','substep_all_hand_object','substep_support')
    complete=bool(clock_complete and all(np.isfinite(z[k]).all() for k in finite_fields))
    established=dwell(t,support&motion&raised,0.,7.5,.1)
    failure=m.get('preparation_failure');failure_at=None
    if not m.get('handoff_passed') or established is None:
        failure=failure or 'preparation_or_establishment_failed';failure_at=7.5
    else:
        lost=first_bad_dwell(t,~support|~raised|(error>.08),7.5,12.,.1)
        missed=first_bad_dwell(t,lift_error>.01,7.5,12.,.3)
        if lost is not None and (missed is None or lost<=missed):
            failure='support_or_object_lost';failure_at=lost
        elif missed is not None:
            failure='transport_reference_unmet';failure_at=missed
    if not complete:
        failure=failure or 'incomplete_truth_or_force_record'
    mask=(t>=8.2-1e-9)&(t<=8.9+1e-9)
    eligible=bool(complete and established is not None and m.get('handoff_passed') and (failure_at is None or failure_at>8.9) and support[mask].all() and mask.sum()>=141)
    post=dwell(t,stable,7.5,8.9,.2)
    release_at=m.get('release_at');exposed=False;released=False;latency=None
    if release_at is not None and complete:
        idx=np.argmin(abs(t-release_at))
        exposed=bool(established is not None and (failure_at is None or failure_at>release_at) and support[idx] and z['all_hand_object'][idx]>.2 and m.get('release_supported_at_command'))
        end=dwell(t,z['all_hand_object']<=.2,float(release_at),min(16.,float(release_at)+3.5),.2)
        released=bool(exposed and end is not None)
        latency=end-release_at if released else None
    task=bool(complete and m.get('handoff_passed') and failure is None and released and np.max(z['warning_count'])==0)
    if not released and failure is None:
        failure='release_failed_or_not_exposed'
    result=dict(record_complete=complete,task_success=task,preparation_passed=bool(m.get('handoff_passed')),
        established=established,A_S=post-7.5 if post is not None else 1.4,first_stable_after_handoff=post,
        whole_task_first_stable=dwell(t,stable,0.,16.,.2),static_eligible=eligible,
        failure_reason=failure,failure_at=failure_at,release_planned=True,release_exposed=exposed,
        release_success=released,release_latency=latency,
        stable_hold_fraction=float(stable[(t>=7.5-1e-9)&(t<12.-1e-9)].mean()))
    if complete:
        # Independent full-task symmetric filter, followed by grid alignment.
        half=round(.05/(2*physical_dt));kt=np.arange(-half,half+1)*physical_dt
        kernel=2*50*physical_dt*np.sinc(2*50*kt)*np.hanning(2*half+1);kernel/=kernel.sum()
        x=np.vstack([z['substep_normal'],n[-1]])
        source_t=np.r_[st,st[-1]+duration[-1]]
        filtered=np.column_stack([np.interp(t,source_t,np.convolve(np.pad(x[:,i],(half,half),mode='reflect'),kernel,mode='valid')) for i in range(5)])
        if eligible:
            result.update(mean_force=float(n[mask].sum(axis=1).mean()),
                finger_tv_rate=float(np.abs(np.diff(filtered[mask],axis=0)).sum()/(t[mask][-1]-t[mask][0])),
                raw_finger_tv_rate=float(np.abs(np.diff(n[mask],axis=0)).sum()/(t[mask][-1]-t[mask][0])),
                finger_sd_mean=float(np.std(n[mask],axis=0).mean()),total_sd=float(np.std(n[mask].sum(axis=1))))
        short=(t>=8.3-1e-9)&(t<=8.8+1e-9)
        short_ok=bool(established is not None and m.get('handoff_passed') and (failure_at is None or failure_at>8.8) and support[short].all() and short.sum()>=101)
        result['short_window_eligible']=short_ok
        if short_ok:
            result.update(short_mean_force=float(n[short].sum(axis=1).mean()),
                short_finger_tv_rate=float(np.abs(np.diff(filtered[short],axis=0)).sum()/(t[short][-1]-t[short][0])))
        physical=z['substep_skin_normal'];excess=np.maximum(physical-limits,0.);total=np.maximum(physical.sum(axis=1)-m['budget'],0.)
        safety=dict(force_protection_pass=bool(np.all(physical<=limits+.02) and np.all(physical.sum(axis=1)<=m['budget']+.02) and np.all(skin[-1]<=limits+.02) and skin[-1].sum()<=m['budget']+.02),
            local_excess_integral=float(np.sum(excess*duration[:,None])),total_excess_integral=float(total@duration),
            finger_peak=np.maximum(physical.max(axis=0),skin[-1]).tolist(),
            palm_peak=float(max(z['substep_palm'].max(),z['palm'][-1])))
        stage_masks={'preparation':st<7.5-1e-9}
        mechanical=m.get('mechanical_release_at',release_at)
        stage_masks['policy_holding' if m.get('handoff_passed') else 'post_preparation_without_policy']=(st>=7.5-1e-9)&(st<(mechanical if mechanical is not None else 16.)-1e-9)
        if mechanical is not None:
            stage_masks['mechanical_release_and_after']=st>=mechanical-1e-9
        safety['stages']={name:dict(duration_s=float(duration[s].sum()),finger_peak_N=physical[s].max(axis=0).tolist(),
            total_finger_peak_N=float(physical[s].sum(axis=1).max()),local_excess_integral_Ns=float(np.sum(excess[s]*duration[s,None])),
            total_excess_integral_Ns=float(total[s]@duration[s])) for name,s in stage_masks.items() if s.any()}
        result['safety']=safety
    active=z['local_domain_active'].astype(bool)
    boundary=np.any((z['alpha']-z['local_domain_lower']<=1e-6)|(z['local_domain_upper']-z['alpha']<=1e-6),axis=1)
    mismatch=np.any(abs(z['requested_action']-z['executed_action'])>1e-6,axis=1)
    result.update(local_domain_boundary_fraction=float(boundary[active].mean()) if active.any() else None,
        static_window_domain_boundary_fraction=float(boundary[mask&active].mean()) if (mask&active).any() else None,
        execution_request_mismatch_fraction=float(mismatch[active].mean()) if active.any() else None)
    return result


def compare_value(a,b):
    if a is None or b is None:
        return a is b
    if isinstance(a,str) or isinstance(b,str):
        return a==b
    return close(a,b,1e-8)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--canonical',help='Optional canonical analysis/summary.json for independent cluster-statistic audit')
    args=parser.parse_args()
    folder=(ROOT/args.root).resolve()
    if not folder.is_relative_to(ROOT):raise ValueError('Independent directory only')
    batch=read(folder/'manifest.json');finished=read(folder/'execution_complete.json');trials=read(folder/'trials.json')
    if batch.get('phase')!='confirmation' or not finished.get('execution_complete') or len(trials)!=256:
        raise ValueError('Wait until all 256 confirmation assignments have finished')
    admission_path=ROOT/f"review/admission_history/admission_{batch['admission_sha256']}.json"
    admission=read(admission_path)
    plan_path=ROOT/admission['evidence']['experiment_plan']['path'];plan=read(plan_path)
    expected_scenes={s['name']:s for s in plan['scenarios']}
    confirm=plan['phases']['confirmation']
    expected={(s,expected_scenes[s]['shape'],seed,method) for s,seed,method in itertools.product(confirm['scenarios'],confirm['seeds'],plan['methods'])}
    checks=dict(exact_256_cartesian_assignments=len(expected)==256 and {key(r) for r in trials}==expected and len(trials)==256,
        exact_planned_cartesian={key(r) for r in batch['planned_jobs']}==expected and len(batch['planned_jobs'])==256,
        execution_manifest_hash=sha(folder/'manifest.json')==finished['manifest_sha256'],
        execution_trials_hash=sha(folder/'trials.json')==finished['trials_sha256'],
        archive_identity=sha(admission_path)==batch['admission_sha256'],
        plan_identity=sha(plan_path)==batch['experiment_plan_sha256']==admission['evidence']['experiment_plan']['sha256'],
        batch_admission_sources=batch['source_sha256']==admission['source_sha256'],
        current_frozen_sources=all(sha(ROOT/k)==v for k,v in batch['source_sha256'].items()))
    indexed={key(r):r for r in trials};run_evidence=[];pair_evidence=[];snapshot={};handoff_by_key={}
    for source in (folder/'manifest.json',folder/'execution_complete.json',folder/'trials.json',admission_path,plan_path):snapshot[str(source.relative_to(ROOT))]=sha(source)
    for number,(scene,shape,seed) in enumerate(sorted({k[:3] for k in expected})):
        pair={}
        for method in ('ratio','unified'):
            identity=(scene,shape,seed,method);row=indexed[identity]
            run=folder/'runs'/f'{scene}__{shape}__{method}__{seed}'
            m=read(run/'metadata.json');result=read(run/'result.json');trace=run/'trace.npz'
            handoff_by_key[identity]=bool(m.get('handoff_passed',False))
            local_checks=dict(exact_result_dictionary=result==row,
                scheduled_confirmation=row['split']=='confirmation' and row['mode']=='scheduled',
                release_full_denominator=row['release_planned'] is True,
                scene_boundary=bool(row.get('boundary'))==bool(expected_scenes[scene].get('boundary',False)))
            snapshot[str((run/'metadata.json').relative_to(ROOT))]=sha(run/'metadata.json')
            snapshot[str((run/'result.json').relative_to(ROOT))]=sha(run/'result.json')
            if not trace.exists():
                local_checks['missing_trace_retained_failure']=row.get('failure_reason')=='runtime_failure' and row['task_success'] is False and row['record_complete'] is False
                run_evidence.append(dict(key=identity,checks=local_checks,missing_trace=True))
                checks.update({str(identity)+'/'+k:bool(v) for k,v in local_checks.items()})
                continue
            # Hash and decode the exact same in-memory bytes. This avoids a
            # second multi-GB read and eliminates a hash-vs-decode race.
            before=trace.stat();blob=trace.read_bytes();h=hashlib.sha256(blob).hexdigest()
            with np.load(io.BytesIO(blob),allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
            del blob
            after=trace.stat()
            local_checks['same_raw_snapshot']=before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
            local_checks['trace_provenance']=h==m.get('trace_sha256')==row.get('trace_sha256')
            local_checks['runtime_identity']=m.get('source_sha256')==batch['source_sha256'] and m.get('admission_sha256')==batch['admission_sha256']
            local_checks['embedded_meta_identity']=json.loads(str(z['metadata']))=={k:v for k,v in m.items() if k!='trace_sha256'}
            obs=[json.loads(x) for x in z['observation_json']]
            arithmetic=derive(z,m)
            if row.get('failure_reason')!='runtime_failure':
                local_checks['independent_all_primary_truth']=all(compare_value(value,row.get(k)) for k,value in arithmetic.items() if k!='safety')
                if 'safety' in arithmetic:
                    local_checks['independent_raw_force_safety']=all(compare_value(v,row['safety'].get(k)) for k,v in arithmetic['safety'].items() if k!='stages')
                    local_checks['independent_stage_raw_force']=arithmetic['safety']['stages']==row['safety']['stages']
            else:
                local_checks['runtime_failure_retained']=not row['task_success'] and not row['record_complete']
            noise=audit_rng(z,m,obs)
            local_checks['actual_sensor_noise_reconstruction']=noise['passed']
            active=z['local_domain_active'].astype(bool);release=z['controller_active']&(z['time']>=12.-1e-9)
            local_checks['actual_action_matches_alpha']=close(z['executed_action'],z['alpha']-z['alpha_before'])
            local_checks['actual_domain_respected']=bool(np.all(z['alpha'][active]>=z['local_domain_lower'][active]-1e-9) and np.all(z['alpha'][active]<=z['local_domain_upper'][active]+1e-9))
            local_checks['request_execution_semantics']=close(z['executed_action'][active],z['requested_action'][active])
            local_checks['held_force_predictions_finite']=np.isfinite(z['predicted_normal'][active]).all()
            local_checks['release_force_forecast_unavailable']=np.isnan(z['predicted_normal'][release]).all() and np.isnan(z['prediction_at'][release]).all()
            local_checks['no_hidden_actuator_clipping']=m.get('max_hidden_ctrl_clip')==0.
            run_evidence.append(dict(key=identity,checks=local_checks,trace_sha256=h,
                independent_metrics=arithmetic,random_packets=noise,controller_active_samples=int(active.sum()),
                nonzero_actual_policy_actions=int(np.any(abs(z['executed_action'][active])>1e-9,axis=1).sum())))
            checks.update({str(identity)+'/'+k:bool(v) for k,v in local_checks.items()})
            pair[method]=(z,m,obs,noise)
        if len(pair)==2:
            a,am,ao,an=pair['ratio'];b,bm,bo,bn=pair['unified']
            count=min(np.searchsorted(a['time'],7.5+1e-9,side='right'),np.searchsorted(b['time'],7.5+1e-9,side='right'))
            pc=dict(common_parameters=am['config']==bm['config'] and am['calibration']==bm['calibration'] and am['noise']==bm['noise'],
                exact_preparation_observations=ao[:count]==bo[:count],
                exact_preparation_state=all(np.array_equal(a[k][:count],b[k][:count],equal_nan=True) for k in ('qpos','qvel','ctrl_before','alpha_before','normal','skin_normal')),
                identical_noise_draw_sequence_or_validated_common_prefix=(an['reconstructed_draw_stream_sha256']==bn['reconstructed_draw_stream_sha256'] if len(a['time'])==len(b['time']) else an['passed'] and bn['passed']))
            checks.update({str((scene,shape,seed))+'/paired/'+k:bool(v) for k,v in pc.items()})
            pair_evidence.append(dict(key=(scene,shape,seed),checks=pc,compared_preparation_samples=int(count),
                same_length=bool(len(a['time'])==len(b['time']))))
        pair.clear();gc.collect()
        if (number+1)%4==0:print(json.dumps(dict(pairs_reviewed=number+1,of=128)),flush=True)
    if args.canonical:
        canonical_path=(ROOT/args.canonical).resolve();canonical=read(canonical_path)
        checks['canonical_256_identity']=len(canonical['records'])==256 and {key(r) for r in canonical['records']}==expected
        checks['canonical_matches_saved_endpoints']=all(all(indexed[key(r)].get(k)==r.get(k) for k in ('task_success','A_S','finger_tv_rate','static_eligible','mean_force','safety','release_success')) for r in canonical['records'])
        official={(r['domain'],r['endpoint']):r for r in canonical['comparisons'] if 'cohort' not in r}
        for domain in ('main','boundary'):
            rng=np.random.default_rng(20261103);family=[]
            for endpoint in ('task_success','A_S','finger_tv_rate','mean_force'):
                pairs=[]
                for scene,shape,seed in sorted({k[:3] for k in expected}):
                    if bool(expected_scenes[scene].get('boundary',False))!=(domain=='boundary'):continue
                    a=indexed[(scene,shape,seed,'ratio')];b=indexed[(scene,shape,seed,'unified')]
                    if endpoint in ('finger_tv_rate','mean_force') and not(a.get('static_eligible') and b.get('static_eligible')):continue
                    pairs.append((seed,float(a[endpoint]),float(b[endpoint])))
                if not pairs:
                    checks[f'canonical/{domain}/{endpoint}/no_empty_inference']=(domain,endpoint) not in official
                    continue
                summary=official[(domain,endpoint)];seeds=sorted({s for s,a,b in pairs})
                differences=np.asarray([b-a for s,a,b in pairs])
                seed_values=[np.asarray([b-a for s,a,b in pairs if s==seed]) for seed in seeds]
                counts=np.asarray([len(x) for x in seed_values]);sums=np.asarray([x.sum() for x in seed_values])
                selections=rng.integers(0,len(seeds),size=(5000,len(seeds)))
                draws=sums[selections].sum(axis=1)/counts[selections].sum(axis=1)
                stats=dict(pairs=len(pairs),clusters=len(seeds),baseline_mean=float(np.mean([a for s,a,b in pairs])),candidate_mean=float(np.mean([b for s,a,b in pairs])),difference=float(differences.mean()))
                checks[f'canonical/{domain}/{endpoint}/point_pair_counts']=all(compare_value(v,summary[k]) for k,v in stats.items())
                checks[f'canonical/{domain}/{endpoint}/whole_seed_bootstrap_CI']=close(np.quantile(draws,[.025,.975]),summary['ci95'],1e-8)
                if endpoint in ('A_S','finger_tv_rate'):
                    observed=abs(float(differences.mean()));extreme=0;total=0
                    for signs in itertools.product((-1.,1.),repeat=len(seeds)):
                        permuted=np.concatenate([sign*values for sign,values in zip(signs,seed_values)])
                        extreme+=int(abs(float(permuted.mean()))>=observed-1e-12);total+=1
                    p=extreme/total;family.append((p,endpoint))
                    checks[f'canonical/{domain}/{endpoint}/exact_seed_sign_p']=compare_value(p,summary.get('p'))
            previous=0.
            for rank,(p,endpoint) in enumerate(sorted(family)):
                previous=max(previous,min(1.,(len(family)-rank)*p))
                checks[f'canonical/{domain}/{endpoint}/Holm']=compare_value(previous,official[(domain,endpoint)].get('p_holm'))
        snapshot[str(canonical_path.relative_to(ROOT))]=sha(canonical_path)
    checks['all_small_inputs_unchanged']=all(sha(ROOT/k)==v for k,v in snapshot.items())
    checks['frozen_sources_unchanged_after']=all(sha(ROOT/k)==v for k,v in batch['source_sha256'].items())
    gap_cases=[dict(key=identity,A_S=indexed[identity]['A_S'],failure_reason=indexed[identity].get('failure_reason'))
        for identity in sorted(expected) if not handoff_by_key[identity] and float(indexed[identity]['A_S'])<1.4-1e-9]
    sensitivity=[]
    for domain in ('all','main','boundary'):
        values=[]
        for scene,shape,seed in sorted({k[:3] for k in expected}):
            if domain!='all' and bool(expected_scenes[scene].get('boundary',False))!=(domain=='boundary'):continue
            identities=[(scene,shape,seed,m) for m in ('ratio','unified')]
            original=[float(indexed[k]['A_S']) for k in identities]
            gated=[original[i] if handoff_by_key[k] else 1.4 for i,k in enumerate(identities)]
            values.append((original,gated))
        sensitivity.append(dict(domain=domain,pairs=len(values),
            primary_baseline_mean=float(np.mean([a[0] for a,b in values])),primary_candidate_mean=float(np.mean([a[1] for a,b in values])),
            admission_gated_baseline_mean=float(np.mean([b[0] for a,b in values])),admission_gated_candidate_mean=float(np.mean([b[1] for a,b in values])),
            admission_gated_difference=float(np.mean([b[1]-b[0] for a,b in values])),
            role='Descriptive contract sensitivity: retain preregistered primary unchanged; replace unadmitted-run A_S with 1.4 only in this separate table.'))
    certificate=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
        created_utc=datetime.now(timezone.utc).isoformat(),assigned_trials=256,paired_assignments=128,
        admission_sha256=batch['admission_sha256'],plan_sha256=batch['experiment_plan_sha256'],
        trials_sha256=sha(folder/'trials.json'),source_sha256=batch['source_sha256'],
        own_script_sha256=sha(__file__),independent_arithmetic_script_sha256=sha(ROOT/'review/independent_development_review.py'),
        runs=run_evidence,paired=pair_evidence,input_sha256=snapshot,
        A_S_admission_diagnostic=dict(unadmitted_with_A_S_below_horizon=len(gap_cases),cases=gap_cases,
            unadmitted_total=sum(not v for v in handoff_by_key.values()),sensitivity=sensitivity,
            explanation='Frozen A_S derives from physical stability after attempted handoff, not an explicit handoff_passed gate. This diagnostic does not change the primary result.'),
        scope='Pure saved-data and fixed random-stream review; no production policy, runtime, evaluator or dynamics imported.')
    out=ROOT/('review/independent_confirmation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.write_text(json.dumps(certificate,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=certificate['passed'],checks=len(checks),failed=certificate['failed'],path=str(out),sha256=sha(out)),ensure_ascii=False))
    if not certificate['passed']:raise SystemExit(1)


if __name__=='__main__':main()

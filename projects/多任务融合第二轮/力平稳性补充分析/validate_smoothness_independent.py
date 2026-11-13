"""Independent audit of saved force-smoothness analysis, with no simulation.

Twenty scene/seed groups are fixed in this file before its first execution.
Raw feature calculations use moments and cumulative-window sums rather than
the production reducer. All pooled effects are rebuilt; all 24 inferential
family CIs, sign tests and Holm values are independently reproduced.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUTPUT=HERE/'output'
DT=.005
SAMPLE_GROUPS=(
    *[("test",name,8001) for name in (
        'nominal','cylinder_small','box_heavy','sphere','friction_drop',
        'three_pulses','local_limit','release_slip','sensor_dropout',
        'collision','infeasible','contact_robustness')],
    *[("robustness",name,9001) for name in (
        'nominal','fragile_conflict','local_limit','release_slip',
        'sensor_noise','sensor_dropout','collision','infeasible')],
)
METRICS=('force_total_mean_N','force_total_sd_N','force_finger_sd_rms_N','force_total_cv',
    'force_total_tv_Nps','force_finger_tv_Nps','force_total_increment_rms_N',
    'force_total_highpass_rms_N','force_finger_highpass_rms_N',
    'force_total_sd_budget_fraction','force_finger_sd_limit_rms_fraction',
    'command_total_mean_N','command_total_sd_N','command_finger_tv_Nps',
    'retained_window_max_slip_mm','window_peak_finger_N','window_force_excess_Ns','window_budget_excess_Ns')
CONFLICT=('fragile_conflict','local_limit','release_slip','collision')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def equal(a,b,label):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),(label,'keys')
        for k in a:equal(a[k],b[k],f'{label}.{k}')
    elif isinstance(a,(tuple,list)):
        assert isinstance(b,(tuple,list)) and len(a)==len(b),(label,'length')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,f'{label}[{i}]')
    elif a is None or isinstance(a,(str,bool)):
        assert a==b,(label,a,b)
    elif isinstance(a,(int,float,np.number)):
        assert b is not None and math.isfinite(float(a)) and math.isfinite(float(b)),(label,a,b)
        assert np.isclose(float(a),float(b),atol=2e-9,rtol=2e-10),(label,a,b)
    else:assert a==b,(label,a,b)

def variance_sd(x,axis=None):
    # Two-pass centered moments avoid cancellation in near-constant traces.
    center=np.mean(x,axis=axis,keepdims=True)
    return np.sqrt(np.mean((x-center)**2,axis=axis))

def independent_features(trace,columns,start,end,limits,budget):
    columns=list(columns);loc={k:i for i,k in enumerate(columns)}
    chosen=np.flatnonzero((trace[:,loc['time']]>=start-1e-10)&
                         (trace[:,loc['time']]<end-1e-10))
    if len(chosen)<100:return None
    t=trace[chosen,loc['time']]
    assert np.allclose(np.diff(t),DT,atol=1e-9,rtol=0)
    normal=trace[np.ix_(chosen,[loc[f'normal_{i}'] for i in range(5)])]
    request=trace[np.ix_(chosen,[loc[f'desired_{i}'] for i in range(5)])]
    assert np.isfinite(normal).all() and np.isfinite(request).all()
    assert normal.min()>=-1e-10
    total=np.add.reduce(normal,axis=1)
    mean=float(np.mean(total));sd=float(variance_sd(total))
    fsd=variance_sd(normal,axis=0)
    # Independent centered 51-point box mean using running sums, not convolution.
    prefix=np.vstack([np.zeros((1,5)),np.cumsum(normal,axis=0)])
    means=(prefix[51:]-prefix[:-51])/51.
    residual=normal[25:len(normal)-25]-means
    span=float(t[-1]-t[0]);gap=np.subtract(total[1:],total[:-1])
    force_excess=np.maximum(0.,normal-np.asarray(limits)).sum(axis=1)
    budget_excess=np.maximum(0.,total-budget)
    # Also independently verify the saved engineering-excess channels.
    equal(force_excess.tolist(),trace[chosen,loc['force_excess_N']].tolist(),'force excess channel')
    equal(budget_excess.tolist(),trace[chosen,loc['budget_excess_N']].tolist(),'budget excess channel')
    return dict(samples=len(chosen),start_s=float(t[0]),last_s=float(t[-1]),duration_s=span,
        force_total_mean_N=mean,force_total_sd_N=sd,force_finger_sd_N=fsd.tolist(),
        force_finger_mean_N=np.mean(normal,axis=0).tolist(),
        force_finger_sd_rms_N=float(np.sqrt(np.mean(fsd**2))),
        force_total_cv=sd/mean if mean>=.2 else None,
        force_total_tv_Nps=float(np.abs(gap).sum()/span),
        force_finger_tv_Nps=float(np.abs(np.subtract(normal[1:],normal[:-1])).sum()/span),
        force_total_increment_rms_N=float(np.sqrt(np.dot(gap,gap)/len(gap))),
        force_total_highpass_rms_N=float(np.sqrt(np.mean(np.add.reduce(residual,axis=1)**2))),
        force_finger_highpass_rms_N=float(np.sqrt(np.mean(residual**2))),
        force_total_sd_budget_fraction=sd/budget,
        force_finger_sd_limit_rms_fraction=float(np.sqrt(np.mean((fsd/limits)**2))),
        command_total_mean_N=float(np.mean(request.sum(axis=1))),
        command_total_sd_N=float(variance_sd(request.sum(axis=1))),
        command_finger_tv_Nps=float(np.abs(request[1:]-request[:-1]).sum()/span),
        retained_window_max_slip_mm=float(np.max(trace[chosen,loc['position_error_m']])*1000),
        window_peak_finger_N=float(normal.max()),
        window_force_excess_Ns=float(force_excess.sum()*DT),
        window_budget_excess_Ns=float(budget_excess.sum()*DT))

def bound(trial,end):
    return end if trial.get('drop_time_s') is None else min(end,trial['drop_time_s']-.05)

def contact_diagnostic(trace,columns,start,end):
    time_column=columns.index('time')
    chosen=(trace[:,time_column]>=start-1e-10)&(trace[:,time_column]<end-1e-10)
    normal=trace[chosen][:,[columns.index(f'normal_{i}') for i in range(5)]]
    n=len(normal)
    weak_total=int(np.count_nonzero(normal.sum(axis=1)<.2))
    few_fingers=int(np.count_nonzero(np.count_nonzero(normal>.2,axis=1)<2))
    return dict(samples=n,normal_sum_below_0p2_count=weak_total,
        normal_sum_below_0p2_fraction=weak_total/n if n else None,
        fewer_than_two_fingers_above_0p2_count=few_fingers,
        fewer_than_two_fingers_above_0p2_fraction=few_fingers/n if n else None)

def pure_feature_audit():
    spec=importlib.util.spec_from_file_location('smoothness_production_for_pure_tests',HERE/'analyze_smoothness.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cols=['time']+[f'normal_{i}' for i in range(5)]+[f'desired_{i}' for i in range(5)]+['position_error_m','force_excess_N','budget_excess_N']
    cases=[]
    for name in ('constant','linear_ramp','opposed_fingers','below_cv_guard','above_cv_guard','zero'):
        trace=np.zeros((200,len(cols)));trace[:,0]=np.arange(200)*DT
        trace[:,1:6]=np.array([1.,2.,3.,4.,5.]);trace[:,6:11]=1.
        if name=='linear_ramp':trace[:,1]=1.+2*np.arange(200)*DT
        if name=='opposed_fingers':
            alternating=.1*(-1.)**np.arange(200)
            trace[:,1]=1+alternating;trace[:,2]=2-alternating
        if name in ('below_cv_guard','above_cv_guard','zero'):
            trace[:,1:6]=0.
            if name!='zero':trace[:,1]=(.199 if name=='below_cv_guard' else .201)+.01*(-1.)**np.arange(200)
        limits=np.ones(5)*100.;budget=1000.
        a=independent_features(trace,cols,0,1,limits,budget)
        equal(a,module.features(trace,cols,0,1,limits,budget),name)
        if name=='linear_ramp':
            equal(a['force_total_tv_Nps'],2.,name)
            equal(a['force_total_increment_rms_N'],.01,name)
            equal(a['force_total_sd_N'],2*DT*np.sqrt((200**2-1)/12),name)
        if name=='opposed_fingers':
            assert a['force_total_sd_N']<1e-12
            equal(a['force_finger_tv_Nps'],80.,name)
            equal(a['force_finger_sd_rms_N'],.1*np.sqrt(2/5),name)
        if name in ('below_cv_guard','zero'):assert a['force_total_cv'] is None
        if name=='above_cv_guard':equal(a['force_total_cv'],.01/.201,name)
        cases.append(name)
    return cases

def seed_key(*labels):
    digest=hashlib.sha256('\0'.join(str(v) for v in labels).encode()).digest()
    return (20261014+int.from_bytes(digest[:8],byteorder='little'))%(2**64)

def independent_ci(sums,counts,seed,reps):
    total=int(sum(counts))
    if not total:return dict(estimate=None,ci95=[None,None],valid_bootstrap_reps=0)
    rng=np.random.default_rng(seed);n=len(sums)
    sample=rng.integers(0,n,size=(reps,n))
    # Convert sampled indices to multiplicity vectors before weighted reduction.
    offset=np.arange(reps)[:,None]*n
    multiplicities=np.bincount((sample+offset).ravel(),minlength=reps*n).reshape(reps,n)
    numerator=multiplicities@np.asarray(sums)
    denominator=multiplicities@np.asarray(counts)
    values=np.sort(numerator[denominator>0]/denominator[denominator>0])
    def q(p):
        rank=(len(values)-1)*p;left=int(np.floor(rank));right=int(np.ceil(rank))
        return float(values[left]*(right-rank)+values[right]*(rank-left)) if left!=right else float(values[left])
    return dict(estimate=float(sum(sums)/total),ci95=[q(.025),q(.975)],valid_bootstrap_reps=len(values))

def independent_sign_p(sums,seed):
    values=np.asarray([x for x in sums if abs(x)>1e-12],float);n=len(values)
    if not n:return dict(p_value=1.,p_method='exact_cluster_sign_flip',permutations=1,nonzero_seed_clusters=0)
    exact=n<=16;reps=2**n if exact else 100000
    if exact:
        codes=np.arange(reps,dtype=np.uint64)
        signs=np.column_stack([2*((codes>>j)&1).astype(np.int8)-1 for j in range(n)])
    else:
        signs=np.random.default_rng(seed).integers(0,2,size=(reps,n),dtype=np.int8)*2-1
    stats=np.abs(signs@values)
    exceed=int(np.count_nonzero(stats>=abs(float(sum(values)))-1e-10))
    return dict(p_value=exceed/reps if exact else (exceed+1)/(reps+1),
        p_method='exact_cluster_sign_flip' if exact else 'monte_carlo_cluster_sign_flip',
        permutations=reps,nonzero_seed_clusters=n)

def main():
    started=time.perf_counter();plan=read(HERE/'分析计划.json');metadata=read(OUTPUT/'metadata.json')
    tracked={Path(path):h for path,h in metadata['sources'].items()}
    tracked.update({ROOT/name:h for name,h in plan['frozen_sources'].items()})
    tracked.update({OUTPUT/name:h for name,h in metadata['output_sha256'].items()})
    for path,h in tracked.items():assert sha(path)==h,('source hash',str(path))
    assert metadata['plan_sha256']==sha(HERE/'分析计划.json')
    assert metadata['script_sha256']==sha(HERE/'analyze_smoothness.py')
    assert plan['bootstrap']['unit']=='whole seed cluster' and plan['bootstrap']['repetitions']==10000
    assert metadata['bootstrap_reps']==10000
    assert plan['exploratory_test_family']['comparisons']==24
    pure=pure_feature_audit()
    individual=read(OUTPUT/'trial_metrics.json');pairs=read(OUTPUT/'paired_windows.json');effects=read(OUTPUT/'paired_effects.json')
    assert len(individual)==2880 and len(effects)==3456
    ind={(r['suite'],r['scenario'],r['seed'],r['method']):r for r in individual}
    assert len(ind)==len(individual)
    manifests={};trial_index={};definitions={};suite_dirs={};seeds={};domains={}
    for suite,relative in plan['suites'].items():
        directory=ROOT/relative;suite_dirs[suite]=directory
        m=read(directory/'manifest.json');manifests[suite]=m;seeds[suite]=sorted(m['seeds'])
        definitions[suite]={d['name']:d for d in m['scenario_definitions']}
        for t in read(directory/'trials.json'):trial_index[(suite,t['scenario'],t['seed'],t['method'])]=t
        domains[suite]=dict(main=[d['name'] for d in m['scenario_definitions'] if not d['boundary']],
            quiet=plan['quiet_scenarios'],conflict=[n for n in m['scenarios'] if n in CONFLICT],
            boundary=[d['name'] for d in m['scenario_definitions'] if d['boundary']])
    expected_pairs={};pair_map={}
    for suite,m in manifests.items():
        for scene in m['scenarios']:
            end=min(17.,definitions[suite][scene].get('release_at') or 17.)
            for seed in seeds[suite]:
                for focal,comparison in plan['pairs']:
                    a=trial_index[(suite,scene,seed,focal)];b=trial_index[(suite,scene,seed,comparison)]
                    ia=ind[(suite,scene,seed,focal)];ib=ind[(suite,scene,seed,comparison)]
                    for condition in ('pre_hold','all_hold_diagnostic','paired_retained','jointly_completed'):
                        target_end=end;allowed=True
                        if condition=='pre_hold':allowed=ia['pre_valid'] and ib['pre_valid'];target_end=8.9
                        elif condition=='paired_retained':
                            target_end=min(bound(a,end),bound(b,end));allowed=a['setup_ok'] and b['setup_ok'] and target_end-8.>=.5
                        elif condition=='jointly_completed':allowed=a['completed'] and b['completed']
                        if allowed:expected_pairs[(suite,scene,seed,condition,focal,comparison)]=target_end
    for p in pairs:
        key=(p['suite'],p['scenario'],p['seed'],p['condition'],p['focal'],p['comparator'])
        assert key not in pair_map;pair_map[key]=p
        assert key in expected_pairs
        equal(p['window_end_s'],expected_pairs[key],('pair end',key))
        equal(p['focal_metrics']['samples'],p['comparator_metrics']['samples'],('same sample count',key))
        equal(p['focal_metrics']['duration_s'],p['comparator_metrics']['duration_s'],('same duration',key))
    assert set(pair_map)==set(expected_pairs),'Pair eligibility/window rows differ'
    covered={str(Path(r['path']).resolve()):r['sha256'] for r in metadata['trace_hashes']}
    coverage=read(ROOT/'review'/'trace_coverage_audit.json')
    previous={r['path']:r['npz_sha256'] for s in coverage['suites'] for r in s['records']}
    assert covered==previous and len(covered)==2880
    raw_checked=0;pair_raw_checked=0;sample_paths=[];contact_diagnostics=[]
    for suite,scene,seed in SAMPLE_GROUPS:
        d=definitions[suite][scene];end=min(17.,d.get('release_at') or 17.);loaded={}
        for method in plan['methods']:
            key=(suite,scene,seed,method);trial=trial_index[key]
            path=suite_dirs[suite]/trial['trace_file'];h=covered[str(path.resolve())]
            assert sha(path)==h
            with np.load(path,allow_pickle=False) as npz:trace=npz['trace'];columns=npz['columns'].tolist()
            assert len(trace)==3400
            equal(trace[:,columns.index('time')].tolist(),(np.arange(3400)*DT).tolist(),'canonical times')
            row=ind[key];held=independent_features(trace,columns,8.,end,d['limit'],d['budget'])
            early=independent_features(trace,columns,8.2,8.9,d['limit'],d['budget'])
            mask=(trace[:,columns.index('time')]>=8.2-1e-10)&(trace[:,columns.index('time')]<8.9-1e-10)
            valid=bool(trial['setup_ok'] and bound(trial,end)>=8.9-1e-10 and np.all(trace[mask,columns.index('phase')]==2))
            assert row['pre_valid']==valid
            equal(row['pre_hold'],early if valid else None,('pre',key))
            equal(row['all_hold_diagnostic'],held,('all hold',key))
            equal(row['completed_only'],held if trial['completed'] else None,('completed only',key))
            for condition,start,stop in (('pre_hold',8.2,8.9),('all_hold_diagnostic',8.,end)):
                contact_diagnostics.append(dict(suite=suite,scenario=scene,seed=seed,method=method,
                    condition=condition,pre_valid=valid,completed=trial['completed'],
                    **contact_diagnostic(trace,columns,start,stop)))
            loaded[method]=(trace,columns);raw_checked+=1;sample_paths.append(dict(path=str(path),sha256=h))
        for focal,comparison in plan['pairs']:
            for condition in ('pre_hold','all_hold_diagnostic','paired_retained','jointly_completed'):
                key=(suite,scene,seed,condition,focal,comparison)
                if key not in pair_map:continue
                p=pair_map[key];begin=8.2 if condition=='pre_hold' else 8.;stop=expected_pairs[key]
                for method,field in ((focal,'focal_metrics'),(comparison,'comparator_metrics')):
                    trace,columns=loaded[method]
                    equal(p[field],independent_features(trace,columns,begin,stop,d['limit'],d['budget']),('paired raw',key,field))
                pair_raw_checked+=1
    pool=defaultdict(list)
    for p in pairs:pool[(p['suite'],p['condition'],p['focal'],p['comparator'])].append(p)
    family=[];rebuilt=0
    for e in effects:
        eligible=[p for p in pool[(e['suite'],e['condition'],e['focal'],e['comparator'])]
                  if p['scenario'] in domains[e['suite']][e['domain']] and
                  p['focal_metrics'][e['metric']] is not None and p['comparator_metrics'][e['metric']] is not None]
        sums=[];counts=[]
        for seed in seeds[e['suite']]:
            subset=[p for p in eligible if p['seed']==seed]
            sums.append(sum(p['focal_metrics'][e['metric']]-p['comparator_metrics'][e['metric']] for p in subset));counts.append(len(subset))
        a=[p['focal_metrics'][e['metric']] for p in eligible];b=[p['comparator_metrics'][e['metric']] for p in eligible]
        equal(e['seed_difference_sums'],sums,'cluster sums');equal(e['seed_eligible_counts'],counts,'cluster counts')
        equal(e['n_pairs'],len(eligible),'eligible pairs')
        equal(e['n_total_pairs'],len(domains[e['suite']][e['domain']])*len(seeds[e['suite']]),'all pairs')
        equal(e['estimate'],sum(sums)/sum(counts) if sum(counts) else None,'pooled estimate')
        equal(e['focal_mean'],float(np.mean(a)) if a else None,'focal mean')
        equal(e['comparator_mean'],float(np.mean(b)) if b else None,'comparator mean')
        equal(e['percent_change'],100*(float(np.mean(a))/float(np.mean(b))-1) if b and abs(float(np.mean(b)))>1e-12 else None,'percent change')
        equal(e['duration_mean_s'],float(np.mean([p['focal_metrics']['duration_s'] for p in eligible])) if eligible else None,'duration mean')
        planned=(e['comparator']=='ratio' and e['focal'] in plan['exploratory_test_family']['focals'] and
            e['metric'] in plan['exploratory_test_family']['metrics'] and
            f"{e['domain']}.{e['condition']}" in plan['exploratory_test_family']['strata'])
        assert bool(e['holm_family'])==planned
        assert e['exploratory'] is True
        if planned:
            seed=seed_key(e['suite'],e['domain'],e['condition'],e['focal'],e['comparator'],e['metric'])
            ci=independent_ci(sums,counts,seed,10000)
            for field,value in ci.items():equal(e[field],value,('CI',field))
            ptest=independent_sign_p(sums,seed+1)
            for field,value in ptest.items():equal(e[field],value,('sign flip',field))
            family.append(e)
        rebuilt+=1
    assert len(family)==24
    ranked=sorted(enumerate(family),key=lambda x:x[1]['p_value']);previous_adjustment=0.
    for rank,(i,e) in enumerate(ranked):
        adjusted=min(1.,max(previous_adjustment,(24-rank)*e['p_value']));previous_adjustment=adjusted
        equal(e['p_value_holm'],adjusted,'Holm')
    for path,h in tracked.items():assert sha(path)==h,('source changed during audit',str(path))
    for row in sample_paths:assert sha(row['path'])==row['sha256']
    report=dict(passed=True,no_simulation_executed=True,no_frozen_files_changed=True,
        sample_groups=[list(x) for x in SAMPLE_GROUPS],raw_traces_recomputed=raw_checked,
        paired_raw_windows_recomputed=pair_raw_checked,total_pair_eligibility_checked=len(pairs),
        pooled_effect_rows_rebuilt=rebuilt,bootstrap_and_sign_test_family_rows_recomputed=len(family),
        pure_feature_cases=pure,source_hashes={str(p):h for p,h in tracked.items()},
        source_coverage_hash_maps_match=True,sample_trace_hashes=sample_paths,
        sampled_contact_window_diagnostics=contact_diagnostics,
        contact_diagnostics_policy='Descriptive only; no new exclusions, advantages, or whole-dataset waveform scan',
        summary_semantics='completed_only summaries use per-method completed sets; jointly_completed paired effects use the same pair cohort',
        time_duration_semantics='duration_s is sampled span (n-1)*0.005; SD is ddof0 on n samples',
        limitations=['200Hz saved full five-finger normal force only; no substep full-vector smoothness claimed',
            'Post-hoc exploratory family; marginal CIs and Holm tests do not replace original endpoints',
            'No new timestep waveform convergence claim'],wall_seconds=time.perf_counter()-started)
    (HERE/'独立核查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_hashes','sample_trace_hashes','sampled_contact_window_diagnostics')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()

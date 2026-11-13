"""Independent read-only audit of frozen complete confirmation.

Does not import controllers, scenarios, evaluation, or analyze calculation helpers.
All mutations are confined to this review's new output files.
"""
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,json,hashlib,itertools,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DOMAIN_FOLDERS={'test':ROOT/'output/confirmation_test','robustness':ROOT/'output/confirmation_robustness'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def same(a,b,tol=1e-9):
    if a is None or b is None:return a is None and b is None
    if isinstance(a,(dict,list)) or isinstance(b,(dict,list)):return a==b
    return abs(float(a)-float(b))<=tol*max(1.,abs(float(a)),abs(float(b)))
def mean(v):
    x=[float(z) for z in v if z is not None and np.isfinite(z)]
    return float(np.mean(x)) if x else None
def metric(r,e):
    if e.startswith('static.'):return (r['static_window']['metrics'] or {}).get(e[7:])
    if e.startswith('sensitivity.'):return (r['static_window_sensitivity']['metrics'] or {}).get(e[12:])
    if e.startswith('safety.'):return r['force_safety'].get(e[7:])
    if e=='static_coverage':return r['static_window']['eligible']
    return r.get(e)

def cohort_ok(a,b,cohort):
    window=a['static_window']['eligible'] and b['static_window']['eligible']
    completed=a['task_success'] and b['task_success']
    if cohort=='window':return window
    if cohort=='completed':return completed
    if cohort=='completed_window':return window and completed
    if cohort=='sensitivity_window':return a['static_window_sensitivity']['eligible'] and b['static_window_sensitivity']['eligible']
    if cohort=='release':return a['release_required'] and b['release_required']
    return True

ENDPOINTS=[('restricted_unstable_time','all'),('static.finger_tv_rate','window'),('task_success','all'),
 ('safety.force_protection_pass','all'),('release_success','release'),('static_coverage','all'),
 ('static.mean_force','window'),('static.total_sd','window'),('static.finger_sd_mean','window'),
 ('static.raw200_finger_tv_rate','window'),('static.substep_finger_tv_rate','window'),('first_stable_time','all'),
 ('safety.total_excess_integral','all'),('safety.local_excess_integral','all'),('stable_hold_fraction','all'),
 ('static.finger_tv_rate','completed_window'),('static.mean_force','completed_window'),
 ('sensitivity.finger_tv_rate','sensitivity_window'),('sensitivity.mean_force','sensitivity_window')]

def independent_estimate(pairs,e,cohort,rng,replicates):
    values=[];av=[];bv=[]
    for a,b in pairs:
        if not cohort_ok(a,b,cohort):continue
        x,y=metric(a,e),metric(b,e)
        if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):continue
        values.append((a['seed'],float(x)-float(y)));av.append(float(x));bv.append(float(y))
    if not values:return dict(pairs=0,clusters=0,candidate_mean=None,baseline_mean=None,difference=None,ci_low=None,ci_high=None,p=None,relative_percent=None)
    grouped=defaultdict(list)
    for seed,difference in values:grouped[seed].append(difference)
    seeds=sorted(grouped);sums=np.array([sum(grouped[s]) for s in seeds]);counts=np.array([len(grouped[s]) for s in seeds])
    difference=float(sums.sum()/counts.sum())
    indexes=rng.integers(0,len(seeds),(replicates,len(seeds)))
    boot=(sums[indexes].sum(axis=1)/counts[indexes].sum(axis=1))
    low,high=np.quantile(boot,[.025,.975])
    signs=np.array(list(itertools.product((-1.,1.),repeat=len(seeds))))
    null=signs@sums/counts.sum();p=float(np.mean(np.abs(null)>=abs(difference)-1e-12))
    am,bm=float(np.mean(av)),float(np.mean(bv))
    return dict(pairs=len(values),clusters=len(seeds),candidate_mean=am,baseline_mean=bm,difference=difference,
        ci_low=float(low),ci_high=float(high),p=p,relative_percent=100*difference/bm if abs(bm)>1e-12 else None)

def descriptive(rows):
    eligible=[r for r in rows if r['static_window']['eligible']];released=[r for r in rows if r['release_required']]
    events=[e for r in rows for e in r['events']]
    result=dict(trials=len(rows),seeds=len({r['seed'] for r in rows}),successes=sum(r['task_success'] for r in rows),
        completion_rate=mean([r['task_success'] for r in rows]),establishment_failures=sum(r['establishment_failure'] for r in rows),
        post_establishment_losses=sum(r['post_establishment_loss'] for r in rows),
        failure_types=dict(Counter(r['end_reason'] for r in rows if not r['task_success'])),
        force_protection_passes=sum(r['force_safety']['force_protection_pass'] for r in rows),
        force_protection_rate=mean([r['force_safety']['force_protection_pass'] for r in rows]),
        complete_truth_records=sum(r['truth_record_complete'] for r in rows),
        complete_physical_force_records=sum(r['force_safety']['full_substeps'] for r in rows),
        valid_force_quality_records=sum(r['force_quality_contract']['valid'] for r in rows),
        complete_evidence_records=sum(r.get('complete_evidence',r['truth_record_complete']) for r in rows),
        total_peak_max=max((metric(r,'safety.total_peak') or 0) for r in rows),
        total_excess_integral_mean=mean([metric(r,'safety.total_excess_integral') for r in rows]),
        A_S=mean([r['restricted_unstable_time'] for r in rows]),
        A_S_normalized=mean([r['restricted_unstable_fraction'] for r in rows]),
        stable_attained=sum(r['first_stability_attained'] for r in rows),
        stable_attainment_rate=mean([r['first_stability_attained'] for r in rows]),
        first_stable_conditional_mean=mean([r['first_stable_time'] for r in rows]),
        first_stable_conditional_median=float(np.median([r['first_stable_time'] for r in rows if r['first_stable_time'] is not None])) if any(r['first_stable_time'] is not None for r in rows) else None,
        stable_hold_fraction_mean=mean([r['stable_hold_fraction'] for r in rows]),static_eligible=len(eligible),
        static_coverage_rate=len(eligible)/len(rows),static_low_contact_fraction=mean([r['static_window']['low_contact_fraction'] for r in rows]),
        release_trials=len(released),release_successes=sum(r['release_success'] for r in released),
        release_completion_rate=mean([r['release_success'] for r in released]),
        release_exposed_trials=sum(r.get('release_exposed_support',False) for r in released),
        release_physical_clearance_trials=sum(r.get('release_physical_clearance_success',False) for r in released),
        release_outcomes=dict(Counter(r.get('release_outcome') for r in released)),
        release_success_conditional_exposed_rate=mean([r['release_success'] for r in released if r.get('release_exposed_support')]),
        release_time_conditional_mean=mean([r['release_time'] for r in released]),
        scheduled_events=len(events),event_outcomes=dict(Counter(e['outcome'] for e in events)),
        recovery_conditional_mean=mean([e['recovery_time'] for e in events]))
    for output,key in [('finger_tv_eligible_mean','finger_tv_rate'),('finger_sd_eligible_mean','finger_sd_mean'),
        ('total_sd_eligible_mean','total_sd'),('raw200_finger_tv_eligible_mean','raw200_finger_tv_rate'),
        ('substep_finger_tv_eligible_mean','substep_finger_tv_rate'),('mean_force_eligible_mean','mean_force'),
        ('relative_drift_eligible_mean','relative_drift_max')]:result[output]=mean([metric(r,'static.'+key) for r in eligible])
    return result

def integrity(plan):
    failures=[];count=0
    for group in ['code_sha256','supporting_sha256','evidence_sha256']:
        for name,expected in plan[group].items():
            count+=1
            if sha(ROOT/name)!=expected:failures.append(group+':'+name)
    snapshots=0
    for group in ['code_sha256','supporting_sha256']:
        for name,expected in plan[group].items():
            snapshots+=1
            if sha(ROOT/'review/frozen_sources'/name)!=expected:failures.append('snapshot:'+name)
    previous=read(ROOT/'review/previous_before.json');old=0
    for name,expected in previous.items():
        old+=1
        if sha(name)!=expected:failures.append('previous:'+name)
    physical=read(ROOT/'review/physical_acceptance.json');phys=0
    for name,item in physical['evidence_files'].items():
        phys+=1
        if sha(ROOT/name)!=item['sha256'] or (ROOT/name).stat().st_size!=item['bytes']:failures.append('physical:'+name)
    return dict(passed=not failures,failures=failures,frozen_live_files=count,frozen_snapshot_files=snapshots,
        previous_project_files=old,physical_certificate_files=phys,physical_certificate_physics_matches=physical['physics_sha256']==plan['code_sha256']['physics.py'],
        historical_v1_calibration_status=read(ROOT/'review/development_record_integrity.json')['original_v1_artifacts_status'])

def inspect_trace(job):
    row,folder,plan,configuration=job;filename=folder/row['trace_file'];bad=[]
    if sha(filename)!=row['trace_sha256']:bad.append('trace_hash')
    with np.load(filename) as z:
        meta=json.loads(str(z['metadata']));t=z['time'];skin=z['skin_normal'];observed=z['observed_normal'];valid=z['validity']
        if meta['model_metadata']!=plan['model_metadata'] or meta['benchmark_id']!=plan['benchmark_id']:bad.append('model_meta')
        for key in ['method','seed','split','parameters']:
            if meta[key]!=row[key]:bad.append('meta_'+key)
        if meta['scenario']['name']!=row['scenario']:bad.append('meta_scenario')
        if meta['evaluation_config']!=configuration:bad.append('evaluation_config')
        if meta['physics_timestep']!=plan['physics_timestep']:bad.append('physics_dt')
        expected_count=round(meta['scenario']['duration']/plan['execution_timestep'])+1
        if len(t)!=expected_count or abs(t[0])>1e-8 or np.max(abs(np.diff(t)-plan['execution_timestep']))>1e-8:bad.append('truth_clock')
        # All fixed draw shapes imply 41 Gaussian values per sensor tick, even
        # when no skin contact or an invalid sample is present.
        sigma=float(meta['parameters']['force_noise']);rng=np.random.default_rng(int(row['seed'])+910000)
        noise=rng.standard_normal((len(t),41))[:,:5]*sigma
        expected=np.maximum(0.,skin+noise);expected_valid=np.ones_like(valid,bool)
        event=meta['scenario']['event']
        if event=='noise':expected[(t>=11.)&(t<11.03),1]+=2.
        if event=='dropout':
            expected_valid[(t>=9.3)&(t<9.7),1:3]=False;expected_valid[(t>=12.)&(t<12.15),:]=False
        expected[~expected_valid]=np.nan
        lag=round(float(meta['parameters']['delay'])/plan['execution_timestep']);indices=np.maximum(0,np.arange(len(t))-lag)
        if not np.allclose(expected[indices],observed,atol=1e-10,rtol=0,equal_nan=True):bad.append('causal_noise_sequence')
        if not np.array_equal(expected_valid[indices],valid):bad.append('dropout_sequence')
    return dict(key=[row['split'],row['scenario'],row['method'],row['seed']],passed=not bad,failures=bad)

def force_sample(row,folder,cfg):
    with np.load(folder/row['trace_file']) as z:
        t=z['time'];st=z['substep_time'];dur=z['substep_duration'];force=z['substep_normal'];skin=z['substep_skin_normal']
        normal=z['normal'];terminal=z['skin_normal'][-1];limits=np.asarray(row['limit']) if 'limit' in row else np.asarray(json.loads(str(z['metadata']))['scenario']['limit'])
        meta=json.loads(str(z['metadata']));budget=meta['scenario']['budget'];dt=float(np.median(dur));end=meta['scenario']['duration']
        peak=np.maximum(skin.max(axis=0),terminal);total=skin.sum(axis=1)
        local_int=float(np.sum(np.maximum(skin-limits,0)*dur[:,None]));total_int=float(np.sum(np.maximum(total-budget,0)*dur))
        cutoff=cfg['force_bandwidth_hz'];half=round(cfg['force_filter_span']/(2*dt));grid=np.arange(-half,half+1)*dt
        kernel=(2*cutoff*dt*np.sinc(2*cutoff*grid))*np.hanning(2*half+1);kernel/=kernel.sum()
        physical=np.vstack([force,normal[-1]]);source_t=np.r_[st,st[-1]+dur[-1]]
        aligned=[]
        for digit in range(5):
            filtered=np.convolve(np.pad(physical[:,digit],(half,half),mode='reflect'),kernel,mode='valid')
            aligned.append(np.interp(t,source_t,filtered))
        aligned=np.column_stack(aligned);diagnostics={}
        bad=[]
        compare=lambda name,a,b:bad.append(name) if not np.allclose(a,b,atol=1e-8,rtol=1e-9) else None
        compare('finger_peak',peak,row['force_safety']['finger_peak'])
        compare('total_peak',max(total.max(),terminal.sum()),row['force_safety']['total_peak'])
        compare('local_exposure',local_int,row['force_safety']['local_excess_integral'])
        compare('total_exposure',total_int,row['force_safety']['total_excess_integral'])
        compare('local_violation_duration',float(np.sum(np.any(skin>limits+cfg['force_tolerance'],axis=1)*dur)),row['force_safety']['local_violation_duration'])
        compare('total_violation_duration',float(np.sum((total>budget+cfg['force_tolerance'])*dur)),row['force_safety']['total_violation_duration'])
        for name,window_key,interval in [('main','static_window',cfg['static_window']),('short','static_window_sensitivity',cfg['static_window_sensitivity'])]:
            selected=(t>=interval[0]-1e-8)&(t<=interval[1]+1e-8);tw=t[selected];f=normal[selected];span=float(tw[-1]-tw[0])
            band_tv=float(abs(np.diff(aligned[selected],axis=0)).sum()/span);raw_tv=float(abs(np.diff(f,axis=0)).sum()/span)
            metrics=row[window_key]['metrics'];compare(name+'_bandTV',band_tv,metrics['finger_tv_rate']);compare(name+'_rawTV',raw_tv,metrics['raw200_finger_tv_rate'])
            compare(name+'_mean',f.sum(axis=1).mean(),metrics['mean_force']);compare(name+'_SD',np.std(f.sum(axis=1)),metrics['total_sd'])
            diagnostics[name]=dict(filtered_TV=band_tv,raw_TV=raw_tv)
        if row['scenario']=='box_heavy':
            selected=(t>=8.2-1e-8)&(t<=8.9+1e-8);span=float(t[selected][-1]-t[selected][0])
            diagnostics['box']=dict(reference_TV=float(abs(np.diff(z['desired'][selected],axis=0)).sum()/span),
                force_TV=diagnostics['main']['raw_TV'],observer_RMSE=float(np.sqrt(np.mean((z['estimated_normal'][selected]-normal[selected])**2))),
                contact_switches=int(np.count_nonzero(np.diff(z['contact_model_active'][selected].astype(int),axis=0))),
                pause_ticks=int(z['transport_paused'][selected].sum()),
                inner_status=dict(zip(*[x.tolist() for x in np.unique(z['qp_status'][selected],return_counts=True)])),
                outer_status=dict(zip(*[x.tolist() for x in np.unique(z['trajectory_qp_status'][selected],return_counts=True)])))
        if abs(st[0])>1e-8 or abs(st[-1]+dur[-1]-end)>1e-8 or np.max(abs(st[:-1]+dur[:-1]-st[1:]))>1e-8:bad.append('interval_coverage')
        return dict(key=[row['split'],row['scenario'],row['method'],row['seed']],passed=not bad,failures=bad,diagnostics=diagnostics)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--summary',type=Path,required=True);args=parser.parse_args()
    plan_path=ROOT/'review/confirmation_plan.json';plan=read(plan_path);initial_plan_hash=sha(plan_path);settings=read(ROOT/'settings.json')
    records=[];matrix=[];paired_failures=[];inputs={};started=time.perf_counter()
    for split,folder in DOMAIN_FOLDERS.items():
        manifest=read(folder/'manifest.json');complete=read(folder/'execution_complete.json');rows=read(folder/'trials.json')
        expected={(split,s,m,seed) for s in plan['scenarios'] for m in plan['methods'] for seed in plan[split+'_seeds']}
        actual=[(r['split'],r['scenario'],r['method'],r['seed']) for r in rows]
        assert set(actual)==expected and len(actual)==len(expected)==len(set(actual)), 'confirmation matrix differs'
        assert complete['source_unchanged'] and complete['trials']==len(expected), 'execution not complete'
        assert manifest['code_sha256']==plan['code_sha256'] and manifest['settings']==settings and manifest['benchmark_id']==plan['benchmark_id'], 'frozen manifest differs'
        grouped=defaultdict(list)
        for r in rows:grouped[(r['scenario'],r['seed'])].append(r)
        for group,rr in grouped.items():
            if any(r['parameters']!=rr[0]['parameters'] or r['reference']!=rr[0]['reference'] or r['model_metadata']!=plan['model_metadata'] for r in rr):paired_failures.append([split,*group])
        inputs[split]={name:sha(folder/name) for name in ['manifest.json','trials.json','execution_complete.json']}
        matrix.append(dict(split=split,trials=len(rows),expected=len(expected),passed=True));records.extend(rows)
    final=read(args.summary);assert final['trial_count']==len(records)==2240
    checks=dict(integrity=integrity(plan),matrix=matrix,paired_parameter_failures=paired_failures)
    jobs=[(r,DOMAIN_FOLDERS[r['split']],plan,settings['evaluation']) for r in records];trace_failures=[];checked=0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(inspect_trace,job) for job in jobs]
        for future in as_completed(futures):
            out=future.result();checked+=1
            if not out['passed']:trace_failures.append(out)
            if checked%400==0:print(json.dumps(dict(stage='trace_hash_metadata_noise',checked=checked,total=len(jobs)),ensure_ascii=False),flush=True)
    checks['trace_integrity']=dict(checked=checked,passed=not trace_failures,failures=trace_failures)
    descriptive_failures=[];descriptive_checks=0;own_summaries={}
    for split in sorted(DOMAIN_FOLDERS):
        subset=[r for r in records if r['split']==split];own_summaries[split]={}
        for domain in ['main','boundary','all']:
            own_summaries[split][domain]={}
            for method in plan['methods']:
                rows=[r for r in subset if r['method']==method and (domain=='all' or bool(r['boundary'])==(domain=='boundary'))]
                independent=descriptive(rows);own_summaries[split][domain][method]=independent
                if domain=='all':continue
                supplied=final['summaries'][split][domain][method]
                for key,value in independent.items():
                    descriptive_checks+=1
                    if not same(value,supplied.get(key)):descriptive_failures.append([split,domain,method,key,value,supplied.get(key)])
        for scenario in plan['scenarios']:
            for method in plan['methods']:
                independent=descriptive([r for r in subset if r['scenario']==scenario and r['method']==method]);supplied=final['scenarios'][split][scenario][method]
                for key,value in independent.items():
                    descriptive_checks+=1
                    if not same(value,supplied.get(key)):descriptive_failures.append([split,scenario,method,key,value,supplied.get(key)])
    checks['descriptive']=dict(passed=not descriptive_failures,values_checked=descriptive_checks,failures=descriptive_failures)
    rng=np.random.default_rng(2651901);statistics=[];stat_failures=[]
    for split in sorted(DOMAIN_FOLDERS):
        for boundary in [False,True]:
            domain='boundary' if boundary else 'main';subset=[r for r in records if r['split']==split and bool(r['boundary'])==boundary]
            lookup={(r['scenario'],r['seed'],r['method']):r for r in subset}
            for method in sorted(plan['methods']):
                if method=='ratio':continue
                pairs=[(r,lookup[(r['scenario'],r['seed'],'ratio')]) for r in subset if r['method']==method]
                for e,cohort in ENDPOINTS:
                    computed=independent_estimate(pairs,e,cohort,rng,plan['bootstrap_replicates'])
                    computed.update(split=split,domain=domain,method=method,endpoint=e,cohort=cohort)
                    supplied=next(r for r in final['comparisons'] if all(r[k]==computed[k] for k in ['split','domain','method','endpoint','cohort']))
                    for key in ['pairs','clusters','candidate_mean','baseline_mean','difference','ci_low','ci_high','p','relative_percent']:
                        if not same(computed[key],supplied[key],1e-8):stat_failures.append([split,domain,method,e,cohort,key,computed[key],supplied[key]])
                    if 'engineering_margin' in supplied and computed['pairs']:
                        direction='lower' if e.endswith('excess_integral') else 'higher';margin=supplied['engineering_margin']
                        expected_margin=.02 if direction=='lower' else -.05
                        decision=('supported' if computed['ci_high']<=margin else 'exceeds_margin' if computed['ci_low']>margin else 'inconclusive') if direction=='lower' else ('supported' if computed['ci_low']>=margin else 'exceeds_margin' if computed['ci_high']<margin else 'inconclusive')
                        if margin!=expected_margin or supplied['gate_judgement']!=decision:stat_failures.append(['gate_contract',split,domain,method,e])
                        computed['gate_judgement']=decision
                    statistics.append(computed)
    for split in sorted(DOMAIN_FOLDERS):
        rows=[r for r in statistics if r['split']==split and r['domain']=='main' and r['method']=='unified' and (r['endpoint'],r['cohort']) in ENDPOINTS[:2]]
        ordered=sorted(rows,key=lambda r:r['p']);running=0.
        for rank,row in enumerate(ordered):
            running=max(running,min(1.,row['p']*(2-rank)));row['p_holm']=running
            supplied=next(r for r in final['comparisons'] if all(r[k]==row[k] for k in ['split','domain','method','endpoint','cohort']))
            if not same(supplied.get('p_holm'),running):stat_failures.append(['holm',split,row['endpoint']])
    checks['statistics']=dict(passed=not stat_failures,comparisons=len(statistics),failures=stat_failures,formula_scope='paired complete seed-cluster bootstrap, exact sign permutation, independent Holm and gates')
    sample=[]
    for split in ['test','robustness']:
        seed=plan[split+'_seeds'][0]
        for scenario in plan['scenarios']:
            for method in ['ratio','unified']:
                row=next(r for r in records if r['split']==split and r['seed']==seed and r['scenario']==scenario and r['method']==method)
                sample.append(force_sample(row,DOMAIN_FOLDERS[split],settings['evaluation']))
        print(json.dumps(dict(stage='raw_force_sample',domain=split,samples=len(sample)),ensure_ascii=False),flush=True)
    checks['raw_force_sample']=dict(passed=all(r['passed'] for r in sample),traces=len(sample),failures=[r for r in sample if not r['passed']],scope='one prespecified seed per domain, all16scenarios, ratio+unified; not all force waveforms')
    assert sha(plan_path)==initial_plan_hash,'plan changed during audit'
    checks['frozen_unchanged_after']=integrity(plan)
    out=dict(passed=all([checks['integrity']['passed'],not paired_failures,checks['trace_integrity']['passed'],checks['descriptive']['passed'],checks['statistics']['passed'],checks['raw_force_sample']['passed'],checks['frozen_unchanged_after']['passed']]),
        scope='independent final analysis/evidence audit; no runtime mutations or simulation reruns',formal_records_read=2240,
        runtime_sha256=plan['code_sha256'],confirmation_plan_sha256=initial_plan_hash,input_sha256=inputs,summary_sha256=sha(args.summary),
        checks=checks,independent_summaries=own_summaries,independent_comparisons=statistics,raw_sample_details=sample,wall_seconds=time.perf_counter()-started)
    output=ROOT/'review/final_result_audit.json';output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=out['passed'],output=str(output),wall_seconds=out['wall_seconds'],failures={k:v.get('failures') for k,v in checks.items() if isinstance(v,dict) and v.get('failures')}),ensure_ascii=False),flush=True)

if __name__=='__main__':main()

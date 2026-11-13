"""Post-hoc force smoothness analysis; never changes experiments or their endpoints."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import analyze_results as original

DT = .005
METHODS = ('event', 'ratio', 'fusion', 'coordinated',
           'coordinated_no_guard', 'coordinated_no_redistribution')
PAIRS = (('coordinated', 'ratio'), ('fusion', 'ratio'),
         ('coordinated', 'event'), ('fusion', 'event'),
         ('coordinated', 'coordinated_no_guard'),
         ('coordinated', 'coordinated_no_redistribution'))
QUIET = ('nominal', 'cylinder_small', 'box_heavy', 'sphere',
         'sensor_noise', 'sensor_dropout')
FAMILY_METRICS = ('force_total_sd_N', 'force_finger_tv_Nps', 'force_total_highpass_rms_N')
CONDITIONS = ('pre_hold', 'all_hold_diagnostic', 'paired_retained', 'jointly_completed')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def frozen_sources():
    plan = read(ROOT/'review/formal_plan.json')
    expected = {**plan['code_sha256'], **plan['supporting_sha256']}
    assert all(sha(ROOT/name) == value for name, value in expected.items())
    return expected


def make_plan():
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='post-hoc exploratory; defined after success results, before smoothness calculation',
        no_controller_or_parameter_changes=True, original_endpoints_retained=True,
        frozen_sources=frozen_sources(),
        suites={'test': 'output/test_holdout_v2', 'robustness': 'output/robustness_holdout_v2'},
        methods=list(METHODS), pairs=[list(p) for p in PAIRS],
        raw_sampling=dict(control_s=.005, physics_s=.0005,
            limitation='Only 200 Hz full five-finger raw force available; substeps store max finger force, not all finger forces or their sum'),
        window_definitions={
            'pre_hold': '[8.2,8.9) seconds; setup established, no confirmed-loss onset before end; all phase codes HOLDING',
            'all_hold_diagnostic': '[8,min(release_at,17)) seconds including failed setup/drop; NEVER evidence that reduced variability is beneficial',
            'paired_retained': 'both setups valid; same [8,min(task hold end,drop_time_a-0.05,drop_time_b-0.05)) window; >=0.5 s; excludes confirmed-loss interval conservatively',
            'jointly_completed': 'same full holding window only when both completed; outcome-conditional, selection bias'},
        quiet_scenarios=list(QUIET),
        metric_definitions={
            'force_total_mean_N': 'mean of scalar normal-force sum; not vector net force',
            'force_total_sd_N': 'population temporal SD (ddof=0) of sum of five raw normal magnitudes',
            'force_finger_sd_rms_N': 'sqrt(mean_i(var_t(normal_i))); individual SDs also retained',
            'force_total_cv': 'SD/mean only if mean>=0.2 N; otherwise missing, not imputed',
            'force_total_tv_Nps': 'sum(abs(diff(sum_i normal_i)))/((n-1)*0.005)',
            'force_finger_tv_Nps': 'sum_i sum(abs(diff(normal_i)))/((n-1)*0.005)',
            'force_total_increment_rms_N': 'sqrt(mean(diff(sum_i normal_i)^2))',
            'force_total_highpass_rms_N': 'RMS residual from centered 51-sample (0.255s) box mean, valid interior only; descriptive fast-change proxy, not a frequency-band power claim',
            'force_finger_highpass_rms_N': 'RMS across time and fingers of same centered residual',
            'force_total_sd_budget_fraction': 'total SD/budget',
            'force_finger_sd_limit_rms_fraction': 'RMS across fingers of SD_i/limit_i',
            'command_total_sd_N': 'SD of sum_i desired_i; command smoothness separate from actual force',
            'command_finger_tv_Nps': 'sum_i TV(desired_i)/((n-1)*DT)'},
        bootstrap=dict(repetitions=10000, unit='whole seed cluster', ci='95% marginal percentile', seed=20261014),
        exploratory_test_family=dict(metrics=list(FAMILY_METRICS),
            focals=['coordinated', 'fusion'], comparator='ratio',
            strata=['main.pre_hold', 'quiet.jointly_completed'], suites=['test', 'robustness'],
            comparisons=24, test='two-sided whole-seed label exchange; Holm across 24 exploratory comparisons'),
        caveats=['All windows/metrics post-hoc; new independent evaluation needed for confirmation',
                 'Lower force, drop, release, slow disturbance response can look smooth',
                 'Conditional samples do not establish global superiority',
                 'Time samples are not independent statistical observations',
                 'Contact model penetration and timestep sensitivity remain unresolved'])


def features(trace, columns, start, end, limits, budget):
    idx = {name:i for i,name in enumerate(columns)}
    t = trace[:,idx['time']]
    mask = (t >= start-1e-10) & (t < end-1e-10)
    x = trace[mask]
    if len(x) < 100:
        return None
    times = x[:,idx['time']]
    assert np.allclose(np.diff(times), DT, rtol=0, atol=1e-9)
    force = x[:,[idx[f'normal_{i}'] for i in range(5)]]
    command = x[:,[idx[f'desired_{i}'] for i in range(5)]]
    assert np.isfinite(force).all() and np.isfinite(command).all()
    assert (force >= -1e-10).all()
    total = force.sum(axis=1)
    dtotal = np.diff(total)
    dfinger = np.diff(force, axis=0)
    mean, sd = float(total.mean()), float(total.std(ddof=0))
    finger_sd = force.std(axis=0, ddof=0)
    duration = float(times[-1]-times[0])
    smooth = np.stack([np.convolve(force[:,i], np.ones(51)/51, mode='valid') for i in range(5)], axis=1)
    residual = force[25:-25]-smooth
    answer = dict(samples=len(x), start_s=float(times[0]), last_s=float(times[-1]),
        duration_s=duration, force_total_mean_N=mean, force_total_sd_N=sd,
        force_finger_sd_N=finger_sd.tolist(),
        force_finger_mean_N=force.mean(axis=0).tolist(),
        force_finger_sd_rms_N=float(np.sqrt(np.mean(finger_sd**2))),
        force_total_cv=sd/mean if mean>=.2 else None,
        force_total_tv_Nps=float(np.abs(dtotal).sum()/duration),
        force_finger_tv_Nps=float(np.abs(dfinger).sum()/duration),
        force_total_increment_rms_N=float(np.sqrt(np.mean(dtotal**2))),
        force_total_highpass_rms_N=float(np.sqrt(np.mean(residual.sum(axis=1)**2))),
        force_finger_highpass_rms_N=float(np.sqrt(np.mean(residual**2))),
        force_total_sd_budget_fraction=sd/budget,
        force_finger_sd_limit_rms_fraction=float(np.sqrt(np.mean((finger_sd/limits)**2))),
        command_total_mean_N=float(command.sum(axis=1).mean()),
        command_total_sd_N=float(command.sum(axis=1).std(ddof=0)),
        command_finger_tv_Nps=float(np.abs(np.diff(command,axis=0)).sum()/duration),
        retained_window_max_slip_mm=float(x[:,idx['position_error_m']].max()*1000),
        window_peak_finger_N=float(force.max()),
        window_force_excess_Ns=float(x[:,idx['force_excess_N']].sum()*DT),
        window_budget_excess_Ns=float(x[:,idx['budget_excess_N']].sum()*DT))
    return answer


def pure_checks():
    columns=['time']+[f'normal_{i}' for i in range(5)]+[f'desired_{i}' for i in range(5)]+['position_error_m','force_excess_N','budget_excess_N']
    a=np.zeros((200,len(columns))); a[:,0]=np.arange(200)*DT; a[:,1:11]=1.
    f=features(a,columns,0.,1.,np.ones(5)*5,22.)
    assert f['force_total_sd_N']==0. and f['force_finger_tv_Nps']==0.
    assert f['force_total_highpass_rms_N']<1e-12
    a[:,1]=np.arange(200)*DT
    f=features(a,columns,0.,1.,np.ones(5)*5,22.)
    assert np.isclose(f['force_total_tv_Nps'],1.)
    assert np.isclose(f['force_total_increment_rms_N'],DT)
    assert f['force_total_highpass_rms_N']<1e-12
    a[:,1]=1.+.1*(-1.)**np.arange(200); a[:,2]=1.-.1*(-1.)**np.arange(200)
    f=features(a,columns,0.,1.,np.ones(5)*5,22.)
    assert f['force_total_sd_N']<1e-12 and f['force_finger_tv_Nps']>70.
    a[:,1:11]=0.
    assert features(a,columns,0.,1.,np.ones(5)*5,22.)['force_total_cv'] is None
    return dict(passed=True,cases=['constant','linear_ramp','opposing_finger_oscillation','near_zero_cv'])


def loss_bound(trial, end):
    drop=trial.get('drop_time_s')
    return min(end, float(drop)-.05) if drop is not None else end


def group_calculation(job):
    suite_name,directory,definition,seed,trials,coverage=job
    scene=definition['name']; end=min(17.,definition.get('release_at') or 17.)
    limits=np.asarray(definition['limit'],float); budget=float(definition['budget'])
    loaded={}; individual=[]; pair_rows=[]; hashes=[]
    for trial in trials:
        path=directory/trial['trace_file']; before=sha(path)
        assert before==coverage[str(path.resolve())]
        with np.load(path,allow_pickle=False) as data:
            trace=data['trace']; columns=data['columns'].tolist()
        assert len(trace)==3400
        assert np.allclose(trace[:,columns.index('time')], np.arange(3400)*DT, atol=1e-10,rtol=0)
        assert sha(path)==before
        hashes.append(dict(path=str(path),sha256=before))
        method=trial['method']; loaded[method]=(trace,columns,trial)
        hold=features(trace,columns,8.,end,limits,budget)
        pre=features(trace,columns,8.2,8.9,limits,budget)
        mask=(trace[:,columns.index('time')]>=8.2-1e-10)&(trace[:,columns.index('time')]<8.9-1e-10)
        pre_valid=bool(trial['setup_ok'] and loss_bound(trial,end)>=8.9-1e-10 and np.all(trace[mask,columns.index('phase')]==2))
        individual.append(dict(suite=suite_name,scenario=scene,method=method,seed=seed,
            completed=trial['completed'],safe_completed=trial['safe_completed'],dropped=trial['dropped'],setup_ok=trial['setup_ok'],
            physics_peak_force_N=trial['physics_peak_force_N'],physics_peak_penetration_mm=trial['physics_peak_penetration_mm'],
            pre_valid=pre_valid,pre_hold=pre if pre_valid else None,all_hold_diagnostic=hold,
            completed_only=hold if trial['completed'] else None))
    by_method={r['method']:r for r in individual}
    for focal,comparator in PAIRS:
        ta,ca,ra=loaded[focal]; tb,cb,rb=loaded[comparator]
        for condition in CONDITIONS:
            shared_end=end
            eligible=True
            if condition=='pre_hold':
                eligible=by_method[focal]['pre_valid'] and by_method[comparator]['pre_valid']
                fa,fb=by_method[focal]['pre_hold'],by_method[comparator]['pre_hold']
            elif condition=='paired_retained':
                shared_end=min(loss_bound(ra,end),loss_bound(rb,end))
                eligible=ra['setup_ok'] and rb['setup_ok'] and shared_end-8.>=.5
                fa=features(ta,ca,8.,shared_end,limits,budget) if eligible else None
                fb=features(tb,cb,8.,shared_end,limits,budget) if eligible else None
            else:
                eligible=condition!='jointly_completed' or (ra['completed'] and rb['completed'])
                fa,fb=by_method[focal]['all_hold_diagnostic'],by_method[comparator]['all_hold_diagnostic']
            if not eligible or fa is None or fb is None:
                continue
            assert fa['samples']==fb['samples'] and abs(fa['duration_s']-fb['duration_s'])<1e-10
            pair_rows.append(dict(suite=suite_name,scenario=scene,seed=seed,focal=focal,comparator=comparator,
                condition=condition,window_end_s=shared_end if condition!='pre_hold' else 8.9,
                focal_metrics=fa,comparator_metrics=fb))
    return individual,pair_rows,hashes


METRICS=('force_total_mean_N','force_total_sd_N','force_finger_sd_rms_N','force_total_cv',
         'force_total_tv_Nps','force_finger_tv_Nps','force_total_increment_rms_N',
         'force_total_highpass_rms_N','force_finger_highpass_rms_N',
         'force_total_sd_budget_fraction','force_finger_sd_limit_rms_fraction',
         'command_total_mean_N','command_total_sd_N','command_finger_tv_Nps',
         'retained_window_max_slip_mm','window_peak_finger_N','window_force_excess_Ns','window_budget_excess_Ns')


def aggregate(individual,pairs,suites):
    summaries=[]; effects=[]
    for suite in suites:
        domains={'main':suite.domain_names('main'),'quiet':list(QUIET),
                 'conflict':suite.domain_names('conflict'),'boundary':suite.domain_names('boundary')}
        for domain,names in domains.items():
            for condition in ('pre_hold','all_hold_diagnostic','completed_only'):
                for method in METHODS:
                    rows=[r for r in individual if r['suite']==suite.name and r['scenario'] in names and r['method']==method]
                    valid=[r[condition] for r in rows if r[condition] is not None]
                    stats={}
                    for metric in METRICS:
                        values=[r[metric] for r in valid if r[metric] is not None]
                        stats[metric]=dict(n=len(values),mean=float(np.mean(values)) if values else None,
                            median=float(np.median(values)) if values else None)
                    summaries.append(dict(suite=suite.name,domain=domain,condition=condition,method=method,
                        n_total=len(rows),n_valid=len(valid),metrics=stats,
                        completed=sum(r['completed'] for r in rows),safe_completed=sum(r['safe_completed'] for r in rows),
                        dropped=sum(r['dropped'] for r in rows),setup_failure=sum(not r['setup_ok'] for r in rows)))
            for condition in CONDITIONS:
                for focal,comparator in PAIRS:
                    rows=[r for r in pairs if r['suite']==suite.name and r['scenario'] in names and
                          r['condition']==condition and r['focal']==focal and r['comparator']==comparator]
                    for metric in METRICS:
                        valid=[r for r in rows if r['focal_metrics'][metric] is not None and r['comparator_metrics'][metric] is not None]
                        sums=np.array([sum(r['focal_metrics'][metric]-r['comparator_metrics'][metric]
                                           for r in valid if r['seed']==seed) for seed in suite.seeds])
                        counts=np.array([sum(r['seed']==seed for r in valid) for seed in suite.seeds])
                        seed=original.stable_seed(20261014,suite.name,domain,condition,focal,comparator,metric)
                        effect=original.bootstrap_cluster_mean(sums,counts,seed)
                        a=[r['focal_metrics'][metric] for r in valid]; b=[r['comparator_metrics'][metric] for r in valid]
                        family=(comparator=='ratio' and focal in ('coordinated','fusion') and metric in FAMILY_METRICS and
                            ((domain=='main' and condition=='pre_hold') or (domain=='quiet' and condition=='jointly_completed')))
                        effect.update(suite=suite.name,domain=domain,condition=condition,focal=focal,comparator=comparator,
                            metric=metric,n_pairs=len(valid),n_total_pairs=len(names)*len(suite.seeds),
                            focal_mean=float(np.mean(a)) if a else None,comparator_mean=float(np.mean(b)) if b else None,
                            percent_change=(100*(float(np.mean(a))/float(np.mean(b))-1)) if b and abs(float(np.mean(b)))>1e-12 else None,
                            duration_mean_s=float(np.mean([r['focal_metrics']['duration_s'] for r in valid])) if valid else None,
                            exploratory=True,holm_family='24 post-hoc smoothness comparisons' if family else None,
                            p_value_holm=None,seed_difference_sums=sums.tolist(),seed_eligible_counts=counts.tolist())
                        if family:
                            effect.update(original.cluster_sign_flip(sums,seed+1))
                        effects.append(effect)
    family=[r for r in effects if r['holm_family'] is not None]
    assert len(family)==24
    adjusted=original.holm_adjust([r['p_value'] for r in family])
    for row,p in zip(family,adjusted): row['p_value_holm']=p
    return summaries,effects


def csv_output(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--plan-only',action='store_true'); args=parser.parse_args()
    plan_path=HERE/'分析计划.json'
    if args.plan_only:
        assert not plan_path.exists(), 'Plan already exists; do not overwrite after seeing smoothness results'
        write(plan_path,make_plan()); print(str(plan_path)); return
    assert plan_path.is_file()
    plan=read(plan_path); assert frozen_sources()==plan['frozen_sources']
    script_hash=sha(__file__); plan_hash=sha(plan_path)
    checks=pure_checks()
    suites=[original.load_suite(ROOT/'output/test_holdout_v2','test'),
            original.load_suite(ROOT/'output/robustness_holdout_v2','robustness')]
    original.validate_suite_pair(*suites)
    coverage_path=ROOT/'review/trace_coverage_audit.json'; coverage=read(coverage_path)
    assert coverage['passed']
    coverage_hashes={r['path']:r['npz_sha256'] for s in coverage['suites'] for r in s['records']}
    sources={str(coverage_path):sha(coverage_path),str(plan_path):plan_hash,str(Path(__file__)):script_hash}
    jobs=[]
    for suite in suites:
        audit=suite.directory/'audit.json'; assert read(audit)['passed']
        for name in ('manifest.json','trials.json','audit.json'):
            path=suite.directory/name; sources[str(path)]=sha(path)
        for scenario in suite.manifest['scenarios']:
            for seed in suite.seeds:
                trials=[suite.indexed[(scenario,method,seed)] for method in METHODS]
                jobs.append((suite.name,suite.directory,suite.definitions[scenario],seed,trials,coverage_hashes))
    individual=[]; pairs=[]; trace_hashes=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(group_calculation,job) for job in jobs]
        for count,future in enumerate(as_completed(futures),1):
            a,b,c=future.result(); individual.extend(a); pairs.extend(b); trace_hashes.extend(c)
            if count%40==0 or count==len(jobs): print(f'Traces {count*6}/2880',flush=True)
    assert len(individual)==2880 and len(trace_hashes)==2880
    individual.sort(key=lambda r:(r['suite'],r['scenario'],r['seed'],r['method']))
    pairs.sort(key=lambda r:(r['suite'],r['scenario'],r['seed'],r['condition'],r['focal'],r['comparator']))
    summaries,effects=aggregate(individual,pairs,suites)
    assert frozen_sources()==plan['frozen_sources'] and sha(__file__)==script_hash
    assert all(sha(path)==h for path,h in sources.items())
    output=HERE/'output'; output.mkdir(exist_ok=True)
    write(output/'trial_metrics.json',individual); write(output/'paired_windows.json',pairs)
    write(output/'summary.json',summaries); write(output/'paired_effects.json',effects)
    csv_output(output/'paired_effects.csv',effects)
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,exploratory=True,
        trials=2880,paired_windows=len(pairs),effect_rows=len(effects),sources=sources,
        trace_hashes=trace_hashes,pure_metric_checks=checks,bootstrap_reps=10000,
        frozen_sources_unchanged=True,plan_sha256=plan_hash,script_sha256=script_hash,
        summary_condition_note='completed_only uses each method own successful cohort; only paired effects/windows jointly_completed use the same paired cohort',
        output_sha256={p.name:sha(p) for p in output.glob('*') if p.name!='metadata.json'})
    write(output/'metadata.json',metadata)
    print(json.dumps(dict(passed=True,trials=2880,effects=len(effects),output=str(output)),ensure_ascii=False))


if __name__=='__main__': main()

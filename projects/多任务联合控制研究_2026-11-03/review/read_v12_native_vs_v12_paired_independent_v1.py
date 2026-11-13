"""Same-method native-minus-old paired readout after BOTH full48 reports exist.

No cohort execution by this file until four native batches and four old batches
are complete. Stored V3 vectors retain NULL; runtime columns are cached once.
"""
from pathlib import Path
from itertools import product
from collections import Counter
import argparse,hashlib,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def definitions():
    spec=importlib.util.spec_from_file_location('v12_pure_pair_definitions',ROOT/'review/read_v12_multiscene_comparison_independent_v1.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def compare_method(old_rows,new_rows,pure):
    old={(r['scenario'],r['seed'],r['motion_mode']):r for r in old_rows}
    new={(r['scenario'],r['seed'],r['motion_mode']):r for r in new_rows}
    expected=set(product(pure.SCENES,pure.SEEDS,pure.MODES))
    assert set(old)==set(new)==expected and len(old_rows)==len(new_rows)==16
    metrics={}
    for spec in pure.SPECS:
        pairs=[dict(context=dict(scenario=s,physical_seed=seed,motion_mode=mode),
            **pure.comparison(new[(s,seed,mode)]['metrics'],old[(s,seed,mode)]['metrics'],spec)) for s,seed,mode in sorted(expected)]
        metrics[spec['name']]=dict(specification=spec,pooled_descriptive=pure.summarize_pairs(pairs),
            by_physical_seed={str(seed):pure.summarize_pairs([p for p in pairs if p['context']['physical_seed']==seed]) for seed in pure.SEEDS},
            by_motion={mode:pure.summarize_pairs([p for p in pairs if p['context']['motion_mode']==mode]) for mode in pure.MODES},pairs=pairs)
    return metrics

def runtime(trace):
    with np.load(trace,allow_pickle=False) as archive:t=archive['time'];ms=archive['control_ms']
    grid=(t>=7.5-1e-9)&(t<12.-1e-9)&(np.abs((t-7.5)/.02-np.rint((t-7.5)/.02))<1e-7)
    a=ms[grid]
    return dict(count=int(len(a)),median_ms=float(np.median(a)),mean_ms=float(np.mean(a)),p95_ms=float(np.percentile(a,95)),
                max_ms=float(np.max(a)),above20ms_count=int(np.sum(a>20.)))

def self_test(pure):
    old=[];new=[]
    for index,(scene,seed,mode) in enumerate(product(pure.SCENES,pure.SEEDS,pure.MODES)):
        o=dict(mission_success=True,static_quality=dict(eligible=index!=0,mean_actual_finger_force_N=2.))
        n=dict(mission_success=index!=1,static_quality=dict(eligible=True,mean_actual_finger_force_N=1.))
        common=dict(scenario=scene,seed=seed,motion_mode=mode)
        old.append(dict(**common,metrics=o));new.append(dict(**common,metrics=n))
    result=compare_method(old,new,pure);checks=[]
    def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
    q=result['static_quality.mean_total_force']['pooled_descriptive'];mission=result['mission_success']['pooled_descriptive']
    check('fixed16_method_pairs',q['planned_context_pairs']==16)
    check('NULL_not_filled_zero',q['not_evaluable']==1 and q['better']==15)
    check('failed_mission_still_in_joint_eligible_quality',q['evaluable_absolute_differences']==15)
    check('mission_loss_preserved',mission['worse']==1 and mission['same']==15)
    check('two_seed_groups_eight_each',all(r['planned_context_pairs']==8 for r in result['mission_success']['by_physical_seed'].values()))
    check('mean_delta_and_relative',q['median_absolute_delta']==-1. and q['median_relative_change_percent']==-50.)
    check('unavailable_forceguard_not_false_or_passed',result['whole_force_guard']['pooled_descriptive']['not_evaluable']==16)
    check('same83metric_definitions',len(result)==83)
    path=ROOT/'review/v12_native_vs_v12_paired_readout_pure_v1.json';assert not path.exists()
    record=dict(schema='v12_native_vs_v12_paired_readout_pure_v1',checks=checks,passed=sum(x['passed'] for x in checks),count=len(checks),
        source_sha256=sha(Path(__file__)),actual_experiment_results_read=False,actual_full48_comparison_executed=False,physics_steps=0)
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(path),sha256=sha(path),passed=record['passed'],count=record['count']),ensure_ascii=False))
    return 0 if record['passed']==record['count'] else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--old-json',type=Path,default=ROOT/'review/v12_multiscene_comparison_independent_v1.json')
    parser.add_argument('--native-json',type=Path,default=ROOT/'review/v12_native_multiscene_comparison_independent_v1.json')
    parser.add_argument('--out',type=Path,default=ROOT/'review/v12_native_vs_v12_paired_independent_v1.json');args=parser.parse_args();pure=definitions()
    if args.self_test:return self_test(pure)
    if not args.old_json.is_file() or not args.native_json.is_file():
        print(json.dumps(dict(status='waiting_for_both_full48_completed_readouts',branch_results_read=False)));return 3
    old=load(args.old_json);new=load(args.native_json)
    for report in (old,new):
        assert report['status']=='all_four_batches_terminal_complete' and report['planned_trials']==48 and report['assigned_per_method']==16
        for batch in report['batches']:
            directory=Path(batch['directory']);t=load(directory/'execution_complete.json')
            assert t['execution_complete'] and t['sources_unchanged'] and not t['runtime_failures']
            key=str((directory/'execution_complete.json').relative_to(ROOT))
            assert sha(directory/'execution_complete.json')==report['input_file_sha256'][key]
    assert len(old['source_identity79'])==79 and len(new['source_identity80'])==80
    assert not args.out.exists() and not args.out.with_suffix('.md').exists()
    results={};pair_context_checks=[]
    for method in pure.METHODS:
        oo=[r for r in old['all_assigned_rows'] if r['method']==method];nn=[r for r in new['all_assigned_rows'] if r['method']==method]
        metrics=compare_method(oo,nn,pure);lookup={tuple(r['key']):r for r in oo};runtime_pairs=[]
        for n in nn:
            o=lookup[tuple(n['key'])]
            assert n['metrics']['physical_parameters']==o['metrics']['physical_parameters']
            om=load(Path(o['result_file']).parent/'metadata.json');nm=load(Path(n['result_file']).parent/'metadata.json')
            assert om['config']==nm['config'] and om['noise']==nm['noise']
            pair_context_checks.append(dict(key=n['key'],physical_config_and_noise_exact=True))
            oldtime=runtime(Path(o['trace_file']));newtime=runtime(Path(n['trace_file']))
            runtime_pairs.append(dict(context=n['key'],old=oldtime,native=newtime,
                delta_native_minus_old_ms={k:newtime[k]-oldtime[k] for k in ('median_ms','mean_ms','p95_ms','max_ms')}))
        numerical={}
        for revision,rows in [('old',oo),('native',nn)]:
            numerical[revision]=dict(assigned_trials=16,holding_ticks=sum(r['actual_policy_diagnostics']['holding_ticks_denominator'] for r in rows),
                decision_grid_denominator=sum(r['actual_policy_diagnostics']['planned_20ms_decisions'] for r in rows),
                tick_counts=dict(sum((Counter(r['actual_policy_diagnostics']['tick_counts']) for r in rows),Counter())),
                decision_counts=dict(sum((Counter(r['actual_policy_diagnostics']['decision_grid_counts']) for r in rows),Counter())))
        results[method]=dict(paired_metrics=metrics,actual_numerical_and_fallback_counts=numerical,control_ms_pairs=runtime_pairs)
    report=dict(schema='v12_native_vs_v12_same_method_paired_v1',role='independent_revision_comparative_readout_author',
        planned_pairs=48,per_method_pairs=16,physical_seed_streams=2,source_identities_deliberately_different=dict(old_count=79,native_count=80),
        input_readout_sha256=dict(old=sha(args.old_json),native=sha(args.native_json)),reader_source_sha256=sha(Path(__file__)),
        pair_context_checks=pair_context_checks,same_method_native_minus_old=results,physics_steps=0,performance_superiority_proven=False,
        limitations=['Matched realized config/noise and unchanged intent contract are necessary; parent source/prefix audit provides actual pre-action mechanical equivalence.',
            'Only two physical/noise streams, not48 independent population draws; no significance/confirmation/utility weights.',
            'All16 assigned outcomes per method remain; quality uses old/native common eligibility without selecting whole-task success.',
            'Native scoring may alter actual post-handoff commands even with unchanged nominal intent. Numerical recovery is not pressure response identification or safety certification.',
            'Control_ms is whole sensor-tick branch wall time on the public20ms grid with concurrent workers; timing differences are descriptive, not isolated real-time guarantees.',
            'Stored V3 vectors are compared; independent fullraw and raw-END clearance audits remain separate.'])
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['V12 native versus original V12: same-method paired development readout','',
        'Each method has16 matched scene/motion/seed contexts; two physical streams. All outcomes and ineligible quality remain.','',
        '|Method|Metric|Better|Worse|Same|Unavailable|Median absolute Δ|Median relative %|','|---|---|---:|---:|---:|---:|---:|---:|']
    visible=('mission_success','full_motion_mission_success','whole_force_guard','whole_local_excess','whole_max_finger_peak',
        'static_quality.raw_200Hz.finger_TV_rate_N_s','dynamic_quality.raw_200Hz.finger_TV_rate_N_s','entire_holding_quality.raw_200Hz.total_force_SD_N')
    for method,value in results.items():
        for name in visible:
            x=value['paired_metrics'][name]['pooled_descriptive'];lines.append('|'+ '|'.join(map(str,(method,name,x['better'],x['worse'],x['same'],x['not_evaluable'],x['median_absolute_delta'],x['median_relative_change_percent'])))+'|')
    args.out.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(args.out),sha256=sha(args.out),paired_context_method_rows=48,metrics_per_method=83),ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())

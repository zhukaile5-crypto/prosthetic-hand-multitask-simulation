"""Terminal-gated descriptive V12 comparison, with every assigned denominator.

Consumes stored unchanged V3 evaluation vectors and actual holding debug/short
static arrays. This is an independently written comparative readout, not a new
full-raw evaluator or a replacement for the parent's source/prefix/raw audit.
No plant import, physics, threshold change or weighted utility is performed.
"""
from pathlib import Path
from itertools import product
from collections import Counter,defaultdict
import argparse,hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=('ratio_effort','fixed_effort','fusion_effort')
SCENES=('nominal','friction_drop','fragile_conflict','collision')
SEEDS=(617901,617902)
MODES=('hold','transport')
DIRECTORIES=('development_v12_pilot_v1','development_v12_hold_seed617901_v1',
             'development_v12_hold_seed617902_v1','development_v12_transport_v1')
EXPECTED_BATCH_COUNTS=(3,9,12,24)
CONTRASTS=(('fusion_effort','ratio_effort'),('fusion_effort','fixed_effort'),('ratio_effort','fixed_effort'))
# Numerical reporting guards only: neither an efficacy margin nor a force cap.
RELATIVE_ZERO_ATOL=1e-9
EQUALITY_ATOL=1e-12
EQUALITY_RTOL=1e-10

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def finite(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number)):return None
    value=float(value)
    return value if math.isfinite(value) else None
def at(value,*keys):
    for key in keys:
        try:value=value[key]
        except (KeyError,IndexError,TypeError):return None
    return value

def terminal_gate(directories,reader=load):
    """Only these four completion files may be opened before all are ready."""
    states=[]
    for directory in directories:
        path=Path(directory)/'execution_complete.json'
        try:record=reader(path)
        except FileNotFoundError:
            states.append(dict(directory=str(directory),state='pending',reason='completion_file_not_present'));continue
        if record.get('execution_complete') is True:
            state='complete'
        elif record.get('ended_utc'):
            state='terminal_execution_not_complete'
        else:state='pending'
        states.append(dict(directory=str(directory),state=state,completion=record))
    return dict(ready=all(s['state']=='complete' for s in states),states=states,
                pending_is_failure=False,branch_results_read=False)

def metric_specs():
    s=[]
    def add(name,kind,unit,path,quality=None,release_condition=False):
        s.append(dict(name=name,kind=kind,unit=unit,path=list(path),quality=quality,release_condition=release_condition))
    for key in ('record_complete','preparation_passed','policy_exposed','mission_success','full_motion_mission_success'):
        add(key,'higher','boolean',(key,))
    add('whole_force_guard','higher','boolean',('safety','guard_pass'))
    for key in ('physical_exposure','physical_success','task_qualified_exposure','task_qualified_success'):
        add('release.'+key,'higher','boolean',('release',key))
    add('initial_stability_wait','lower','s',('initial_stability_wait_s',))
    add('release_coarse200Hz_latency','lower','s',('release','physical_clearance_latency_s'),release_condition=True)
    add('whole_local_excess','lower','Ns',('safety','local_excess_integral_Ns'))
    add('whole_total_excess','lower','Ns',('safety','total_excess_integral_Ns'))
    add('whole_max_finger_peak','lower','N',('safety','finger_peak_N','max'))
    for i,digit in enumerate(('TH','FF','MF','RF','LF')):
        add('whole_peak.'+digit,'lower','N',('safety','finger_peak_N',i))
    for stage in ('shared_preparation','policy_holding_transport','post_preparation_without_policy','mechanical_release_and_after'):
        add('stage.'+stage+'.local_excess','lower','Ns',('safety','stages',stage,'local_excess_integral_Ns'))
        add('stage.'+stage+'.total_excess','lower','Ns',('safety','stages',stage,'total_excess_integral_Ns'))
        add('stage.'+stage+'.max_finger_peak','lower','N',('safety','stages',stage,'finger_peak_N','max'))
    for window in ('static_quality','dynamic_quality','entire_holding_quality'):
        add(window+'.mean_total_force','lower','N',(window,'mean_actual_finger_force_N'),window)
        for signal in ('raw_200Hz','bandlimited'):
            for key,unit in (('finger_TV_rate_N_s','N/s'),('finger_net_drift_rate_N_s','N/s'),
                             ('finger_reversal_rate_N_s','N/s'),('finger_SD_mean_N','N'),('total_force_SD_N','N')):
                add(window+'.'+signal+'.'+key,'lower',unit,(window,signal,key),window)
    for window in ('holding_motion_tracking','dynamic_motion_tracking'):
        for statistic in ('rmse','max_abs_error','absolute_error_integral'):
            for i,axis in enumerate(('lift','WRJ2','WRJ1')):
                unit=('m' if i==0 else 'rad')+('*s' if statistic=='absolute_error_integral' else '')
                add(window+'.'+statistic+'.'+axis,'lower',unit,('progress',window,statistic,i))
    return s

SPECS=metric_specs()

def value_for(record,spec):
    if record is None:return None,'result_unavailable'
    if spec['quality'] and at(record,spec['quality'],'eligible') is not True:
        return None,'quality_window_ineligible_or_unknown'
    if spec['release_condition'] and not (at(record,'release','physical_exposure') is True and at(record,'release','physical_success') is True):
        return None,'conditional_release_latency_not_exposed_or_not_successful'
    path=spec['path'];value=at(record,*path[:-1]) if path[-1]=='max' else at(record,*path)
    if path[-1]=='max':
        try:value=max(value) if all(finite(x) is not None for x in value) and len(value) else None
        except (TypeError,ValueError):value=None
    if spec['unit']=='boolean':
        return (int(value),None) if type(value) is bool else (None,'boolean_metric_unavailable')
    value=finite(value)
    return (value,None) if value is not None else (None,'numeric_metric_unavailable')

def comparison(candidate,baseline,spec):
    c,cr=value_for(candidate,spec);b,br=value_for(baseline,spec)
    row=dict(candidate_value=c,baseline_value=b,absolute_delta=None,relative_change_percent=None,classification='not_evaluable',
             unavailable_reason=dict(candidate=cr,baseline=br))
    if c is None or b is None:return row
    delta=c-b;row['absolute_delta']=delta
    same=abs(delta)<=EQUALITY_ATOL+EQUALITY_RTOL*max(abs(c),abs(b))
    row['classification']='same' if same else ('better' if (delta<0)==(spec['kind']=='lower') else 'worse')
    row['unavailable_reason']=None
    if spec['unit']=='boolean':row['relative_unavailable_reason']='binary_effect_reported_as_absolute_success_difference'
    elif abs(b)<=RELATIVE_ZERO_ATOL:row['relative_unavailable_reason']='baseline_numerically_near_zero_in_native_units'
    else:
        row['relative_change_percent']=100*delta/abs(b)
        row['relative_unavailable_reason']=None
    return row

def summarize_pairs(rows):
    counts=Counter(r['classification'] for r in rows)
    delta=[r['absolute_delta'] for r in rows if r['absolute_delta'] is not None]
    percent=[r['relative_change_percent'] for r in rows if r['relative_change_percent'] is not None]
    return dict(planned_context_pairs=len(rows),better=counts['better'],worse=counts['worse'],same=counts['same'],not_evaluable=counts['not_evaluable'],
        evaluable_absolute_differences=len(delta),median_absolute_delta=None if not delta else float(np.median(delta)),
        relative_change_evaluable_pairs=len(percent),relative_change_unavailable_pairs=len(rows)-len(percent),
        median_relative_change_percent=None if not percent else float(np.median(percent)))

def diagnostics(trace_path):
    """Read each compressed column once; classify actual holding ticks/decisions."""
    with np.load(trace_path,allow_pickle=False) as z:
        times=z['time'];active=z['controller_active'];debug=z['debug_json'];normal=z['normal']
    hold=(times>=7.5-1e-9)&(times<12.-1e-9);ids=np.flatnonzero(hold)
    counters={kind:Counter() for kind in ('intent_modes','fallback_reasons','effort_status','effort_unavailable_reason')}
    counts=Counter();decision_counts=Counter();sample_count=int(hold.sum())
    for i in ids:
        d=json.loads(str(debug[i]));intent=d.get('force_intent');effort=d.get('effort_tracking')
        decision=abs((float(times[i])-7.5)/.02-round((float(times[i])-7.5)/.02))<1e-7
        flags={'actual_policy_active':bool(active[i]),'policy_not_executed':bool(d.get('policy_not_executed',False)),
               'ratio_memory_rollback':bool(d.get('ratio_effort_saturation_memory_rollback',False)),
               'late_execution_memory_rollback':bool(d.get('late_execution_ratio_memory_rollback',False))}
        if isinstance(intent,dict):
            mode=intent.get('fusion_mode',intent.get('label','intent_mode_unknown'));counters['intent_modes'][str(mode)]+=1
            fallback=intent.get('fusion_mode')=='shared_ratio_fallback';flags['fusion_ratio_fallback']=fallback
            flags['fusion_conditional_allocation']=intent.get('fusion_mode')=='conditional_object_contact_allocation'
            if fallback:counters['fallback_reasons'][str(intent.get('fallback_reason','missing_reason'))]+=1
        else:flags['intent_report_missing']=True
        if isinstance(effort,dict):
            status=str(effort.get('status','status_unknown'));counters['effort_status'][status]+=1
            unavailable=effort.get('increment8') is None
            flags['effort_unavailable']=unavailable;flags['effort_numerical_failure']=status=='NUMERICAL_FAILURE'
            if unavailable:
                why=at(effort,'debug','reason') or at(effort,'debug','reasons') or status
                counters['effort_unavailable_reason'][str(why)]+=1
        else:flags['effort_report_missing']=True
        for key,yes in flags.items():
            if yes:counts[key]+=1
            if yes and decision:decision_counts[key]+=1
    static=(times>=8.2-1e-9)&(times<=8.9+1e-9)
    contiguous=np.ascontiguousarray(normal[static])
    return dict(window_s=[7.5,12.],holding_ticks_denominator=sample_count,planned_20ms_decisions=225,
        tick_counts=dict(counts),decision_grid_counts=dict(decision_counts),categories={k:dict(v) for k,v in counters.items()},
        expected_grid_only='decision counts refer to public 7.5+20ms grid; late execution overrides separately counted at 5ms',
        numerical_failure_is_not_a_physical_infeasibility_certificate=True,
        static_force_array_sha256=hashlib.sha256(contiguous.tobytes()).hexdigest(),static_force_samples=int(static.sum()),
        no_raw2000_metrics_recalculated=True)

def read_completed(directories,gate):
    """Called only after the outer four-batch terminal gate has succeeded."""
    assert gate['ready'];all_rows=[];sources=None;inputs={};batch_evidence=[]
    for directory,wanted,state in zip(directories,EXPECTED_BATCH_COUNTS,gate['states']):
        directory=Path(directory);complete=state['completion']
        assert complete['planned']==complete['trials']==wanted and complete['sources_unchanged'] is True and not complete['runtime_failures']
        plan=load(directory/'plan.json');manifest=load(directory/'manifest.json');ledger=load(directory/'assignment_ledger.json');trials=load(directory/'trials.json')
        for name in ('execution_complete.json','plan.json','manifest.json','assignment_ledger.json','trials.json'):
            inputs[str((directory/name).relative_to(ROOT))]=sha(directory/name)
        for name,key in (('plan.json','plan_sha256'),('manifest.json','manifest_sha256'),('trials.json','trials_sha256'),('assignment_ledger.json','assignment_ledger_sha256')):
            assert sha(directory/name)==complete[key],(directory,name,'terminal identity mismatch')
        identity=manifest['research_source_sha256'];assert len(identity)==79
        if sources is None:sources=identity
        else:assert sources==identity,'Different execution source identities cannot be pooled'
        assert len(plan['jobs'])==len(trials)==len(ledger)==wanted
        by_trial={(r['method'],r['scenario'],int(r['seed']),r['motion_mode']):r for r in trials}
        assert len(by_trial)==wanted
        for job in plan['jobs']:
            key=(job['method'],job['scenario'],int(job['seed']),job['motion_mode'])
            destination=Path(job['destination']).resolve();assert destination.is_relative_to((directory/'runs').resolve())
            result=load(destination/'result.json');metadata=load(destination/'metadata.json')
            assert result==by_trial[key],'Stored result differs from aggregate trial dictionary'
            assert key==(result['method'],result['scenario'],int(result['seed']),result['motion_mode'])
            assert metadata['research_source_sha256']==sources
            assert result['trace_sha256']==metadata['trace_sha256']
            for path in (destination/'result.json',destination/'metadata.json'):
                inputs[str(path.relative_to(ROOT))]=sha(path)
            all_rows.append(dict(key=list(key),method=key[0],scenario=key[1],seed=key[2],motion_mode=key[3],
                source_batch=directory.name,result_file=str(destination/'result.json'),trace_file=str(destination/'trace.npz'),
                trace_declared_sha256=result['trace_sha256'],actual_NPZ_bytes_hashed=False,
                metrics=result['research_metrics'],legacy_model_domain_supported=metadata.get('legacy_model_domain_supported'),
                failure_reason=result.get('failure_reason'),preparation_failure=metadata.get('preparation_failure'),
                actual_policy_diagnostics=diagnostics(destination/'trace.npz')))
        batch_evidence.append(dict(directory=str(directory),planned=wanted,execution_complete=True,declared_source_count=79))
    expected=set(product(METHODS,SCENES,SEEDS,MODES));actual=[tuple(row['key']) for row in all_rows]
    assert len(actual)==len(set(actual))==48 and set(actual)==expected,'Exact preregistered factorial assignment required'
    return all_rows,inputs,batch_evidence,sources

def build_report(rows,inputs,batches,sources):
    by={(r['method'],r['scenario'],r['seed'],r['motion_mode']):r for r in rows}
    comparisons={}
    contexts=list(product(SCENES,SEEDS,MODES))
    for candidate,baseline in CONTRASTS:
        metrics={}
        for spec in SPECS:
            pairs=[]
            for scenario,seed,mode in contexts:
                c=by[(candidate,scenario,seed,mode)]['metrics'];b=by[(baseline,scenario,seed,mode)]['metrics']
                pairs.append(dict(context=dict(scenario=scenario,physical_seed=seed,motion_mode=mode),**comparison(c,b,spec)))
            metrics[spec['name']]=dict(specification=spec,pooled_descriptive=summarize_pairs(pairs),
                by_physical_seed={str(seed):summarize_pairs([p for p in pairs if p['context']['physical_seed']==seed]) for seed in SEEDS},
                by_motion={mode:summarize_pairs([p for p in pairs if p['context']['motion_mode']==mode]) for mode in MODES},pairs=pairs)
        comparisons[candidate+'_minus_'+baseline]=metrics
    method_summary={}
    for method in METHODS:
        selected=[r for r in rows if r['method']==method];summary={}
        assert len(selected)==16
        for spec in SPECS:
            values=[value_for(r['metrics'],spec)[0] for r in selected];known=[v for v in values if v is not None]
            record=dict(assigned_denominator=16,evaluable=len(known),unavailable=16-len(known),median=None if not known else float(np.median(known)))
            if spec['unit']=='boolean':record.update(known_true=sum(v==1 for v in known),known_false=sum(v==0 for v in known),
                assigned_success_fraction=sum(v==1 for v in known)/16.,unknowns_not_filled_false=True)
            summary[spec['name']]=record
        method_summary[method]=dict(assigned=16,metrics=summary,
            by_physical_seed={str(seed):dict(assigned=8,rows=[r['key'] for r in selected if r['seed']==seed]) for seed in SEEDS})
    duplicate=defaultdict(list)
    for r in rows:
        duplicate[(r['method'],r['seed'],r['actual_policy_diagnostics']['static_force_array_sha256'])].append(dict(scenario=r['scenario'],motion_mode=r['motion_mode']))
    groups=[dict(method=k[0],physical_seed=k[1],static_normal_array_sha256=k[2],contexts=v,context_count=len(v)) for k,v in duplicate.items() if len(v)>1]
    return dict(schema='v12_multiscene_independent_comparison_v1',role='independent_comparative_readout_author_not_fullraw_auditor',
        status='all_four_batches_terminal_complete',planned_trials=48,planned_contexts=16,assigned_per_method=16,
        independent_physical_seed_streams=2,methods=list(METHODS),batches=batches,source_identity79=sources,
        input_file_sha256=inputs,reader_source_sha256=sha(Path(__file__)),metric_specifications=SPECS,
        numerical_reporting_guards=dict(relative_denominator_zero_atol_native_units=RELATIVE_ZERO_ATOL,equality_atol=EQUALITY_ATOL,equality_rtol=EQUALITY_RTOL,
            interpretation='roundoff/undefined-percent guards only; no reliability-loss threshold, effect margin or utility weight'),
        method_summaries=method_summary,paired_comparisons=comparisons,all_assigned_rows=rows,exact_duplicate_static_force_groups=groups,
        performance_superiority_proven=False,utility_weights=None,success_loss_acceptance_margin=None,
        limitations=[
            'Only two realized physical/noise seed streams; 16 scenario-motion contexts and within-run time samples are not 16 independent population draws. No inference/CI/significance claim.',
            'Descriptive win/loss counts refer to predeclared scene-motion contexts, including duplicated pre-event windows; exact static-force duplicates explicitly listed.',
            'All 48 assignments and failures retained; each method outcome denominator is 16. Paired force quality requires both corresponding fixed windows eligible, independently of whole-task success.',
            'Individual-method quality medians can have different coverage; comparison conclusions must use paired common-eligible rows and their unavailable counts.',
            'Lower mean force/TV or effort residual does not override mission/guard/tracking cost; no scalar utility or approved success-loss tradeoff threshold exists.',
            'Phase excess/peaks, policy absence, fusion fallback and actor NUMERICAL_FAILURE remain alongside outcomes. Optimization failure is not proven physical infeasibility.',
            'Existing V3 metric vectors are consumed without calling/modifying evaluator. Parent independently audits full raw/source/prefix; root supplies raw2k diagnostics separately.',
            'Trace SHA is declared in result/metadata and crosschecked between them; this reader hashes selected static arrays but does not claim full NPZ bytes/source files were independently rehashed.',
            'Release latency here is the stored conditional 200Hz force-clearance dwell only; raw-END clocks/geometric clearance are not substituted or inferred.'
        ])

def markdown(report):
    lines=['V12 independent multiscene descriptive comparison','',
        '48 assigned trials; 16 contexts per method; two physical/noise seed streams. All four batches terminal and source identities equal.',
        '', 'Each metric retains better/worse/same/not-evaluable counts. Quality comparisons use common eligible fixed windows; failures are retained. No utility weights or superiority claim.','',
        '|Contrast|Metric|Better|Worse|Same|Unavailable|Median absolute Δ|Median relative %|','|---|---|---:|---:|---:|---:|---:|---:|']
    visible={'mission_success','full_motion_mission_success','whole_force_guard','whole_local_excess','whole_max_finger_peak','initial_stability_wait',
             'static_quality.mean_total_force','dynamic_quality.mean_total_force','entire_holding_quality.mean_total_force',
             'dynamic_quality.raw_200Hz.finger_TV_rate_N_s','dynamic_quality.bandlimited.finger_TV_rate_N_s',
             'entire_holding_quality.raw_200Hz.total_force_SD_N'}
    for contrast,metrics in report['paired_comparisons'].items():
        for name,metric in metrics.items():
            if name not in visible:continue
            s=metric['pooled_descriptive'];a=s['median_absolute_delta'];p=s['median_relative_change_percent']
            lines.append('|'+ '|'.join(map(str,(contrast,name,s['better'],s['worse'],s['same'],s['not_evaluable'],None if a is None else round(a,9),None if p is None else round(p,6))))+'|')
    lines+=['','Full per-seed summaries, all paired values, stage/force/fallback diagnostics and every assigned row are in the adjacent JSON.','',
            'Lower variation can include intentional force redistribution; full holding SD and tracking costs must remain visible. Coarse release latency does not establish raw geometric clearance.']
    return '\n'.join(lines)+'\n'

def self_test():
    tests=[]
    def check(name,yes):tests.append(dict(name=name,passed=bool(yes)))
    low=dict(name='x',kind='lower',unit='N',path=['x'],quality=None,release_condition=False)
    high={**low,'kind':'higher'}
    check('lower_direction',comparison(dict(x=1.),dict(x=2.),low)['classification']=='better')
    check('higher_direction',comparison(dict(x=1.),dict(x=2.),high)['classification']=='worse')
    check('roundoff_same',comparison(dict(x=2.+1e-12),dict(x=2.),low)['classification']=='same')
    check('missing_kept_unavailable',comparison(dict(x=None),dict(x=2.),low)['classification']=='not_evaluable')
    check('near_zero_no_percentage',comparison(dict(x=1.),dict(x=0.),low)['relative_change_percent'] is None)
    check('nonzero_relative_sign',comparison(dict(x=1.),dict(x=2.),low)['relative_change_percent']==-50.)
    q={**low,'path':['static_quality','x'],'quality':'static_quality'}
    check('quality_null_if_either_ineligible',comparison(dict(static_quality=dict(eligible=True,x=1.)),dict(static_quality=dict(eligible=False,x=2.)),q)['classification']=='not_evaluable')
    check('failed_mission_does_not_remove_supported_quality',comparison(dict(mission_success=False,static_quality=dict(eligible=True,x=1.)),dict(mission_success=True,static_quality=dict(eligible=True,x=2.)),q)['classification']=='better')
    b={**low,'unit':'boolean'}
    check('boolean_unknown_not_false',comparison(dict(x=None),dict(x=True),b)['classification']=='not_evaluable')
    check('binary_no_relative',comparison(dict(x=False),dict(x=True),b)['relative_change_percent'] is None)
    release={**low,'release_condition':True}
    check('unexposed_release_latency_unavailable',comparison(dict(x=0.,release=dict(physical_exposure=False,physical_success=False)),dict(x=1.,release=dict(physical_exposure=True,physical_success=True)),release)['classification']=='not_evaluable')
    pairs=[comparison(dict(x=1.),dict(x=2.),low),comparison(dict(x=3.),dict(x=2.),low),comparison(dict(x=2.),dict(x=2.),low),comparison(dict(x=None),dict(x=2.),low)]
    summary=summarize_pairs(pairs)
    check('all_count_categories_and_denominator',summary['planned_context_pairs']==4 and all(summary[k]==1 for k in ('better','worse','same','not_evaluable')))
    check('medians_do_not_zero_fill_missing',summary['evaluable_absolute_differences']==3 and summary['median_absolute_delta']==0.)
    opened=[]
    def pending_reader(path):
        opened.append(path.name)
        if str(path.parent)=='d':raise FileNotFoundError(path)
        return dict(execution_complete=True)
    pending=terminal_gate(['a','b','c','d'],pending_reader)
    check('pending_not_failure_or_partial_read',not pending['ready'] and not pending['pending_is_failure'] and opened==['execution_complete.json']*4 and pending['states'][-1]['state']=='pending')
    ready=terminal_gate(['a','b','c','d'],lambda path:dict(execution_complete=True))
    check('only_all_four_complete_opens_gate',ready['ready'])
    check('exact_factorial_denominator',len(set(product(METHODS,SCENES,SEEDS,MODES)))==48)
    check('all_specs_unique',len(SPECS)==len({s['name'] for s in SPECS}))
    check('null_peak_not_zero',value_for(dict(safety=dict(finger_peak_N=None)),next(s for s in SPECS if s['name']=='whole_max_finger_peak'))[0] is None)
    check('no_weight_or_acceptance_threshold',RELATIVE_ZERO_ATOL==1e-9 and not any('utility' in s['name'] for s in SPECS))
    path=ROOT/'review/v12_multiscene_comparison_readout_pure_v1.json';assert not path.exists()
    record=dict(schema='v12_multiscene_readout_pure_v1',author_role='independent_readout_author',source_sha256=sha(Path(__file__)),
        tests=tests,passed=sum(t['passed'] for t in tests),count=len(tests),physics_steps=0,actual_experiment_results_read=False,
        actual_comparison_executed=False,metric_count=len(SPECS),ready_to_wait_for_root_terminal_notice=True)
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(path),sha256=sha(path),passed=record['passed'],count=len(tests),metric_count=len(SPECS)),ensure_ascii=False))
    return 0 if record['passed']==len(tests) else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--directories',nargs=4,type=Path);parser.add_argument('--out',type=Path,default=ROOT/'review/v12_multiscene_comparison_independent_v1.json')
    args=parser.parse_args()
    if args.self_test:return self_test()
    directories=args.directories or [ROOT/'output'/name for name in DIRECTORIES]
    gate=terminal_gate(directories)
    if not gate['ready']:
        print(json.dumps(dict(status='waiting_for_all_four_terminal_complete',terminal_gate=gate,no_branch_results_read=True),ensure_ascii=False));return 3
    assert not args.out.exists() and not args.out.with_suffix('.md').exists(),'Fresh output required; prior evidence is immutable'
    rows,inputs,batches,sources=read_completed(directories,gate);report=build_report(rows,inputs,batches,sources)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    args.out.with_suffix('.md').write_text(markdown(report),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],path=str(args.out),sha256=sha(args.out),rows=len(rows),metric_count=len(SPECS),physical_seed_streams=2),ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())

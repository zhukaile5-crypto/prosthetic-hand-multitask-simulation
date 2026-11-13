"""Independent same-context pilot3 revision readout; no plant or physics.

Stored unchanged V3 vectors plus cached actual trace columns. This is a local
development comparison, separate from the parent's source/prefix/rate gate.
"""
from pathlib import Path
from collections import Counter
import importlib.util,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/v12_native_pilot_comparison_independent_v1.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def stats(values):
    return dict(count=len(values),median_ms=float(np.median(values)),mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)),
        max_ms=float(np.max(values)),above20ms_count=int(np.sum(values>20.)),above5ms_count=int(np.sum(values>5.)))

def main():
    spec=importlib.util.spec_from_file_location('pure_comparison_definitions',ROOT/'review/read_v12_multiscene_comparison_independent_v1.py')
    pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
    directories={'old':ROOT/'output/development_v12_pilot_v1','native':ROOT/'output/development_v12_native_pilot_v1'}
    # Both terminal gates are checked before any branch record is read.
    finish={k:read(d/'execution_complete.json') for k,d in directories.items()}
    assert all(t['execution_complete'] and t['trials']==t['planned']==3 and not t['runtime_failures'] and t['sources_unchanged'] for t in finish.values())
    rows={};inputs={};checks=[]
    for revision,directory in directories.items():
        manifest=read(directory/'manifest.json');expected=79 if revision=='old' else 80
        assert len(manifest['research_source_sha256'])==expected
        inputs[str((directory/'execution_complete.json').relative_to(ROOT))]=sha(directory/'execution_complete.json')
        for path in sorted(directory.glob('runs/*/result.json')):
            result=read(path);m=read(path.parent/'metadata.json');r=result['research_metrics'];method=result['method']
            assert result['scenario']=='nominal' and result['seed']==617901 and result['motion_mode']=='hold'
            assert m['research_source_sha256']==manifest['research_source_sha256']
            for p in (path,path.parent/'metadata.json'):inputs[str(p.relative_to(ROOT))]=sha(p)
            # Each compressed column is loaded exactly once in this reader.
            with np.load(path.parent/'trace.npz',allow_pickle=False) as z:
                time=z['time'];wall=z['control_ms'];debug=z['debug_json'];normal=z['normal'];skin=z['substep_skin_normal'];dur=z['substep_duration']
            holding=(time>=7.5-1e-9)&(time<12.-1e-9)
            decisions=holding&(np.abs((time-7.5)/.02-np.rint((time-7.5)/.02))<1e-7)
            idle=holding&~decisions;decision_counts=Counter();tick_counts=Counter();reasons=Counter()
            for i in np.flatnonzero(holding):
                value=json.loads(str(debug[i]));intent=value.get('force_intent',{});effort=value.get('effort_tracking',{});status=effort.get('status')
                flags=dict(effort_numerical_failure=status=='NUMERICAL_FAILURE',effort_unavailable=effort.get('increment8') is None,
                    conditional_allocation=intent.get('fusion_mode')=='conditional_object_contact_allocation',ratio_fallback=intent.get('fusion_mode')=='shared_ratio_fallback')
                for key,yes in flags.items():
                    if yes:tick_counts[key]+=1
                    if yes and decisions[i]:decision_counts[key]+=1
                if flags['effort_unavailable']:reasons[str(effort.get('debug',{}).get('reason',status))]+=1
            static=(time>=8.2-1e-9)&(time<=8.9+1e-9);wholehold=(time>=7.5-1e-9)&(time<=12.+1e-9)
            actual_static_force=float(normal[static].sum(1).mean())
            actual_static_TV=float(np.abs(np.diff(normal[static],axis=0)).sum(axis=0).sum()/(time[static][-1]-time[static][0]))
            actual_hold_SD=float(normal[wholehold].sum(1).std())
            limits=np.asarray(m['limits']);Ns=float((np.maximum(skin-limits,0)*dur[:,None]).sum())
            checks.extend([dict(revision=revision,method=method,metric=name,passed=bool(np.isclose(actual,expected_value,atol=1e-9,rtol=1e-10)))
                for name,actual,expected_value in [('static_force',actual_static_force,r['static_quality']['mean_actual_finger_force_N']),
                    ('static_raw200_TV',actual_static_TV,r['static_quality']['raw_200Hz']['finger_TV_rate_N_s']),
                    ('wholehold_raw200_totalSD',actual_hold_SD,r['entire_holding_quality']['raw_200Hz']['total_force_SD_N']),
                    ('full_raw_local_excess_Ns',Ns,r['safety']['local_excess_integral_Ns'])]])
            rows[(revision,method)]=dict(revision=revision,method=method,context=dict(scenario='nominal',seed=617901,motion_mode='hold'),
                metrics=r,diagnostics=dict(holding_ticks=int(holding.sum()),decision_grid_points=int(decisions.sum()),
                    numerical_counts_20ms_grid=dict(decision_counts),numerical_counts_5ms=dict(tick_counts),unavailable_reasons_5ms=dict(reasons),
                    control_ms_decision_grid=stats(wall[decisions]),control_ms_other_holding_ticks=stats(wall[idle]),
                    timing_scope='Measured wall duration of whole common sensor-tick control branch; three concurrent workers, not isolated actor timing or real-time guarantee'),
                actual_NPZ_bytes_hashed=False,declared_trace_sha256=result['trace_sha256'],declared_source_count=expected,
                config=m['config'],noise=m['noise'])
    comparisons={}
    for method in pure.METHODS:
        old=rows[('old',method)];new=rows[('native',method)]
        assert old['config']==new['config'] and old['noise']==new['noise']
        comparisons[method]=dict(metrics={s['name']:dict(specification=s,**pure.comparison(new['metrics'],old['metrics'],s)) for s in pure.SPECS},
            numerical_old=old['diagnostics'],numerical_native=new['diagnostics'],physical_config_and_noise_exact=True)
    assert len(rows)==6 and all(c['passed'] for c in checks)
    report=dict(schema='v12_native_same_context_pilot3_independent_comparison_v1',role='independent_local_comparative_reader',
        old_terminal=finish['old'],native_terminal=finish['native'],inputs_sha256=inputs,reader_source_sha256=sha(Path(__file__)),physics_steps=0,
        metrics_checked_against_actual_trace=checks,all_checked_metrics_match=True,rows=list(rows.values()),native_minus_old_same_method=comparisons,
        assignment_denominator_per_revision_method=1,independent_physical_contexts=1,global_48_comparison_executed=False,
        limitations=['One already seen nominal/hold/617901 context only; no inference for the full registered48 or population.',
            'Performance values are descriptive, not the source/prefix/software acceptance gate; parent performs that independent gate.',
            'No runtime exception is different from no effort NUMERICAL_FAILURE; both reported separately.',
            'V3 pressure/release outcomes do not certify a pressure predictor, contact model, or general material safety.',
            'Wall control_ms comparisons share nominal worker count but are not hardware-isolated timing benchmarks.'])
    assert not OUT.exists();OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    names=['mission_success','full_motion_mission_success','whole_force_guard','static_quality.mean_total_force','static_quality.raw_200Hz.finger_TV_rate_N_s',
        'entire_holding_quality.raw_200Hz.total_force_SD_N','whole_local_excess','whole_max_finger_peak','release_coarse200Hz_latency']
    lines=['V12 native pilot3 independent local comparison','',
        'Same nominal/hold/617901 context; old and native each contain the three assigned methods. This is descriptive, not the software gate or full48 conclusion.','',
        '|Method|Metric|Old|Native|Native−old|','|---|---|---:|---:|---:|']
    for method,record in comparisons.items():
        for name in names:
            x=record['metrics'][name];lines.append(f"|{method}|{name}|{x['baseline_value']}|{x['candidate_value']}|{x['absolute_delta']}|")
    OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),actual_metric_checks=len(checks),passed=True,
        decisions={m:dict(old=comparisons[m]['numerical_old']['numerical_counts_20ms_grid'],native=comparisons[m]['numerical_native']['numerical_counts_20ms_grid']) for m in pure.METHODS},
        wall_ms={m:dict(old=comparisons[m]['numerical_old']['control_ms_decision_grid'],native=comparisons[m]['numerical_native']['control_ms_decision_grid']) for m in pure.METHODS}),ensure_ascii=False))

if __name__=='__main__':main()

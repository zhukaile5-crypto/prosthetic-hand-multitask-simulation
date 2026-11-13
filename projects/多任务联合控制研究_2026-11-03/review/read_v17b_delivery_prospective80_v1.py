"""All-assigned V17b auxiliary delivery reader; never control or physics.

Readiness inspection reads completion/assignment metadata only. No outcome or
ledger is read until the registered complete80 sequence is terminal. A stopped
incomplete sequence may be archived explicitly, without cohort comparisons.
"""
from pathlib import Path
from itertools import product,combinations
from collections import Counter
from datetime import datetime,timezone
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).parent))
import delivery_auxiliary_metrics_v3 as aux

PLAN=ROOT/'review/v17b_prospective80_plan_v1.json'
PLAN_SHA='b08fe0d57b2c43a8abcf58e9ff869d1fdc01ae0f2ed5cfe1546028e7268a6a06'
PROGRESS=ROOT/'review/v17b_prospective80_progress_v1.json'
CALC_SHA='0228b2526ace5f448a6f7f458ebeadb9023b79b120fe4ba16eccb3463eb5bc55'
CONTRACT_SHA='8dcda359252014d509a8b26c102570309ad893b6a4ccad31015a36ca0cb7e979'
ROLES=('full_V17','holding_A','same_prior_legacy_ratio','baseline_fixed','legacy_fusion')
CONTRASTS=tuple(('full_V17',reference) for reference in ROLES[1:])+tuple(('holding_A',reference) for reference in ROLES[2:])
BASELINE_ROLES={'ratio_effort':'same_prior_legacy_ratio','fixed_effort':'baseline_fixed','fusion_effort':'legacy_fusion'}
BOOL_METRICS=('first_stable_found','stable_by_original_deadline','final_stable_suffix_found','ever_floor',
              'terminal_receiver_top','terminal_moving_receiver')
NUMERIC_METRICS=('first_stable_latency_s','terminal_speed_m_s','terminal_angular_speed_rad_s')+tuple(
    w+'.'+c+'.'+q for w in ('first_receiver_impact20ms','first_receiver_impact100ms','first_floor_impact100ms')
    for c in ('receiver_top','hand','floor') for q in ('peak_N','impulse_Ns'))
METRICS=BOOL_METRICS+NUMERIC_METRICS

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def compact(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))
def record_key(role,scenario,motion,seed):return [role,scenario,motion,int(seed)]

def roster(plan,root=ROOT):
    rows=[]
    for stage in plan['stages']:
        stage_count=0
        for seed,scenario,method in product(stage['seeds'],stage['scenarios'],stage['methods']):
            role=BASELINE_ROLES[method] if stage['arm']=='baselines' else stage['arm']
            if role not in ROLES:raise ValueError('Unexpected assigned role')
            key=record_key(role,scenario,stage['motion_mode'],seed)
            run=(Path(root)/stage['output']/'runs'/f'{scenario}__{method}__{seed}').resolve()
            rows.append(dict(record_key=key,record_id=compact(key),role=role,scenario=scenario,
                motion_mode=stage['motion_mode'],physical_seed=int(seed),method=method,stage=stage['name'],
                task_policy=stage['task_policy'],receiver_policy=stage['receiver_policy'],run=str(run)))
            stage_count+=1
        if stage_count!=stage['planned']:raise ValueError('Stage planned/product mismatch')
    if len(rows)!=plan['planned'] or len(rows)!=80 or len({r['record_id'] for r in rows})!=80 or len({r['run'] for r in rows})!=80:
        raise ValueError('Exactly80 unique assignments/paths required')
    counts=Counter(r['role'] for r in rows);contexts={tuple(r['record_key'][1:]) for r in rows}
    if counts!=Counter({role:16 for role in ROLES}) or len(contexts)!=16 or len(plan['seeds'])!=2:
        raise ValueError('Five16-record roles and16 contexts/two seed clusters required')
    expected_contexts=set(product(plan['scenarios'],('hold','transport'),plan['seeds']))
    if contexts!=expected_contexts:raise ValueError('Registered context coverage mismatch')
    return rows

def readiness(plan,progress,completion):
    """Outcome-free. A successful task flag is never a readiness condition."""
    if progress is None:return dict(status='pending',ready=False,reason='sequence_progress_missing')
    if progress.get('master_plan_sha256')!=PLAN_SHA:raise ValueError('Sequence belongs to another master plan')
    if progress.get('physics_active') is True:return dict(status='pending',ready=False,reason='physics_active')
    terminal=[]
    for stage in plan['stages']:
        e=completion.get(stage['name'])
        terminal.append(dict(stage=stage['name'],file_present=e is not None,
            execution_complete=None if e is None else e.get('execution_complete'),
            trials=None if e is None else e.get('trials'),planned=stage['planned']))
    if progress.get('complete') is not True:
        stopped=bool(progress.get('ended_utc') and (progress.get('stopped_on_runtime_failure') or progress.get('stopped_on_prefix_contract_failure')))
        return dict(status='terminal_incomplete' if stopped else 'pending',ready=False,
                    reason='registered_sequence_incomplete',stages=terminal)
    good=all(e is not None and e.get('execution_complete') is True and e.get('sources_unchanged') is True
        and e.get('trials')==s['planned'] and e.get('planned')==s['planned']
        for s in plan['stages'] for e in (completion.get(s['name']),))
    if not good:raise ValueError('Sequence complete conflicts with stage completion/source/count')
    return dict(status='ready_complete',ready=True,stages=terminal)

def pinned_plan(path=PLAN):
    if sha(path)!=PLAN_SHA:raise ValueError('Master plan identity changed')
    p=read(path);pins=p['auxiliary_metrics']
    if (sha(Path(aux.__file__))!=CALC_SHA or sha(aux.CONTRACT_PATH)!=CONTRACT_SHA or
            pins['contract_sha256']!=CONTRACT_SHA or pins['reader_sha256']!=CALC_SHA):
        raise ValueError('Frozen v3 calculator/contract identity changed')
    return p

def inspect(plan,root=ROOT,progress_path=PROGRESS):
    progress=read(progress_path) if Path(progress_path).exists() else None
    completions={s['name']:(read(Path(root)/s['output']/'execution_complete.json')
        if (Path(root)/s['output']/'execution_complete.json').exists() else None) for s in plan['stages']}
    return readiness(plan,progress,completions),progress,completions

def _job_key(job):return (job['method'],job['scenario'],int(job['seed']),job['motion_mode'],job['task_policy'],job['receiver_policy'])
def stage_assignments(plan,rows,stage,root=ROOT):
    folder=Path(root)/stage['output'];expected=[r for r in rows if r['stage']==stage['name']]
    ledger=read(folder/'assignment_ledger.json');seen={};issues=[]
    for item in ledger:
        key=_job_key(item['job'])
        if key in seen:raise ValueError('Duplicate ledger assignment')
        seen[key]=item
    expected_keys={(r['method'],r['scenario'],r['physical_seed'],r['motion_mode'],r['task_policy'],r['receiver_policy']) for r in expected}
    if set(seen)!=expected_keys:raise ValueError('Assignment ledger differs from registered roster')
    manifest=read(folder/'manifest.json') if (folder/'manifest.json').exists() else None
    if manifest is not None:
        if manifest.get('research_source_sha256')!=plan['source_sha256']:issues.append('stage_source_identity')
        if {_job_key(j) for j in manifest['jobs']}!=expected_keys:issues.append('manifest_assignment')
    trials=read(folder/'trials.json') if (folder/'trials.json').exists() else []
    indexed={}
    for trial in trials:
        key=(trial['method'],trial['scenario'],int(trial['seed']))
        if key in indexed:raise ValueError('Duplicate trials result')
        indexed[key]=trial
    return seen,indexed,issues

def load_assigned(item,ledger_item,trial,plan,stage_issues):
    row=dict(item,assignment_status=ledger_item['status'],status='not_evaluated',auxiliary=None,
             original_result=None,input_sha256={},issues=list(stage_issues))
    run=Path(item['run']);job=ledger_item['job']
    if Path(job['destination']).resolve()!=run:row['issues'].append('ledger_destination')
    if ledger_item['status']!='evaluated':return row
    try:
        meta=read(run/'metadata.json');original=read(run/'result.json')
        identity=(meta['method'],meta['scenario'],int(meta['seed']),meta['motion_mode'],meta['task_policy'],meta['receiver_policy'])
        expected=(item['method'],item['scenario'],item['physical_seed'],item['motion_mode'],item['task_policy'],item['receiver_policy'])
        if identity!=expected:row['issues'].append('metadata_identity')
        row['execution_source_identity_equal']=meta.get('research_source_sha256')==plan['source_sha256']
        if not row['execution_source_identity_equal']:row['issues'].append('record_source_identity')
        if trial is None or trial!=original:row['issues'].append('trial_result_identity')
        row.update(original_result=original,physical_config=meta.get('config'),noise=meta.get('noise'),
            limits=meta.get('limits'),budget=meta.get('budget'),handoff_passed=meta.get('handoff_passed'))
        release=meta.get('mechanical_release_at',meta.get('release_at'))
        if release is None:
            row['status']='no_actual_release_clock';return row
        if row['issues']:
            row['status']='identity_untrusted';return row
        with np.load(run/'delivery_intervals.npz',allow_pickle=False) as z:
            arrays={k:z[k] for k in aux.C['input_fields']}
        result=aux.calculate(arrays,release_at_s=release,episode_end_s=meta['duration'],
            physical_period_s=meta['physics_timestep'],original_result=original)
        row.update(status=result['status'],auxiliary=result,input_sha256={name:sha(run/name)
            for name in ('metadata.json','result.json','delivery_intervals.npz')})
    except Exception as exc:
        row.update(status='invalid_or_missing_record',error=type(exc).__name__+': '+str(exc))
    return row

def scalar_values(row):
    m=row.get('auxiliary');out={key:None for key in METRICS}
    if m is None:return out
    complete=m['coverage']['all_valid'];first=m['official_first_stable_receiver'];suffix=m['official_final_stable_suffix'];term=m['terminal'] or {}
    out.update(first_stable_found=bool(first) if complete else None,
        first_stable_latency_s=None if first is None else first['latency_s'],
        stable_by_original_deadline=bool(m['official_first_stable_by_release_deadline']) if complete else None,
        final_stable_suffix_found=bool(suffix and not suffix['right_censored']) if complete else None,
        ever_floor=m['ever_floor_contact'],terminal_receiver_top=term.get('target_receiver_top_contact'),
        terminal_moving_receiver=term.get('moving_receiver_contact'),terminal_speed_m_s=term.get('linear_speed_m_s'),
        terminal_angular_speed_rad_s=term.get('angular_speed_rad_s'))
    for window in ('first_receiver_impact20ms','first_receiver_impact100ms','first_floor_impact100ms'):
        w=m[window]
        for cat in ('receiver_top','hand','floor'):
            for quantity in ('peak_N','impulse_Ns'):
                out[window+'.'+cat+'.'+quantity]=None if w is None else w['contact'][cat][quantity]
    return out

def summarize(rows):
    """All-assigned denominators. Latency is conditional and never zero-imputed."""
    values=[scalar_values(r) for r in rows];metrics={}
    for key in METRICS:
        all_values=[v.get(key) for v in values];known=[v for v in all_values if v is not None]
        if key in BOOL_METRICS:
            metrics[key]=dict(assigned=len(rows),known=len(known),unknown=len(rows)-len(known),true=sum(known),false=len(known)-sum(known))
        else:
            finite=[float(v) for v in known if type(v) is not bool and np.isfinite(v)]
            metrics[key]=dict(assigned=len(rows),available=len(finite),unavailable=len(rows)-len(finite),
                min=None if not finite else min(finite),median=None if not finite else float(np.median(finite)),max=None if not finite else max(finite),
                values_are_conditional_on_measurement_availability_not_success=True,no_zero_imputation=True)
    return dict(assigned=len(rows),status_counts=dict(Counter(r['status'] for r in rows)),
        assignment_status_counts=dict(Counter(r['assignment_status'] for r in rows)),metrics=metrics,
        latency_interpretation='Read first_stable_found true/false/unknown alongside conditional latency; no stability never counts as0 seconds',
        no_success_subset_selection=True)

def paired_records(rows):
    by={tuple(r['record_key']):r for r in rows};contexts=sorted({tuple(r['record_key'][1:]) for r in rows});pairs=[]
    for context in contexts:
        for role,reference in CONTRASTS:
            a=by[(role,*context)];b=by[(reference,*context)]
            equal=all(a.get(k) is not None and a.get(k)==b.get(k) for k in ('physical_config','noise','limits','budget'))
            av=scalar_values(a);bv=scalar_values(b);difference={}
            for k in sorted(set(av)|set(bv)):
                x=av.get(k);y=bv.get(k);known=x is not None and y is not None and equal
                difference[k]=dict(candidate=x,reference=y,paired_available=known,
                    delta=(int(x)-int(y) if type(x) is bool and type(y) is bool else float(x)-float(y)) if known else None)
            pairs.append(dict(context_key=list(context),candidate_record_id=a['record_id'],reference_record_id=b['record_id'],
                candidate_role=role,reference_role=reference,actual_physical_noise_limits_equal=equal,
                primary_same_prior_contrast=role=='full_V17' and reference=='same_prior_legacy_ratio',
                component_contrast=role=='full_V17' and reference=='holding_A',differences=difference,
                paired_missing_or_not_reached_not_zero=True))
    return pairs

def grouped(rows):
    out={'all_by_role':{role:summarize([r for r in rows if r['role']==role]) for role in ROLES}}
    for field in ('physical_seed','motion_mode','scenario'):
        out['by_'+field]={str(value):{role:summarize([r for r in rows if r[field]==value and r['role']==role])
            for role in ROLES} for value in sorted({r[field] for r in rows})}
    return out

def collect(plan,rows,*,root=ROOT):
    output=[];source_issues=[]
    for stage in plan['stages']:
        expected=[r for r in rows if r['stage']==stage['name']]
        try:
            ledger,trials,issues=stage_assignments(plan,rows,stage,root)
            source_issues.extend(dict(stage=stage['name'],issue=x) for x in issues)
            for item in expected:
                key=(item['method'],item['scenario'],item['physical_seed'],item['motion_mode'],item['task_policy'],item['receiver_policy'])
                output.append(load_assigned(item,ledger[key],trials.get(key[:3]),plan,issues))
        except Exception as exc:
            source_issues.append(dict(stage=stage['name'],issue=type(exc).__name__+': '+str(exc)))
            output.extend(dict(item,assignment_status='untrusted_stage_manifest_or_ledger',status='stage_unavailable',
                               auxiliary=None,original_result=None,input_sha256={},issues=[str(exc)]) for item in expected)
    if len(output)!=80 or len({r['record_id'] for r in output})!=80:raise RuntimeError('All-assigned record invariant failed')
    return output,source_issues

def main():
    p=argparse.ArgumentParser();p.add_argument('--inspect',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--terminal-incomplete',action='store_true');p.add_argument('--out');args=p.parse_args()
    if args.inspect==args.execute:raise ValueError('Choose exactly --inspect or --execute')
    plan=pinned_plan();assigned=roster(plan);state,progress,completions=inspect(plan)
    report=dict(schema='v17b_delivery_auxiliary_prospective80_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        status=state['status'],complete=False,readiness=state,plan_sha256=PLAN_SHA,calculator_sha256=CALC_SHA,contract_sha256=CONTRACT_SHA,
        execution_source_sha256=plan['source_sha256'],
        reader_sha256=sha(Path(__file__)),planned=80,role_counts=dict(Counter(r['role'] for r in assigned)),
        physics_steps=0,production_source_modified=False,metrics_read=False,
        scope=dict(physical_seed_clusters=2,contexts=16,independent_n80_claim=False,
            prospective_definitions=True,prospective_development_not_global_confirmation=True,
            all_assigned=True,weighted_total_score=False,incomplete_subset_selection=False))
    if args.execute and (state['ready'] or args.terminal_incomplete and state['status']=='terminal_incomplete'):
        records,issues=collect(plan,assigned);report.update(records=records,metrics_read=True,source_issues=issues,
            complete=state['ready'],status='computed_complete80' if state['ready'] else 'terminated_incomplete_all_assigned_archive')
        if state['ready']:
            report.update(summaries=grouped(records),pairs=paired_records(records),
                source_identity_passed=not issues and not any(r['issues'] for r in records),
                record_status_counts=dict(Counter(r['status'] for r in records)),
                record_error_count=sum('error' in r for r in records))
    if args.out:
        out=ROOT/args.out
        if out.exists():raise FileExistsError(out)
        out.write_text(json.dumps(aux._plain(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(dict(path=str(out),status=report['status'],records=len(report.get('records',[])),metrics_read=report['metrics_read'],sha256=sha(out))))
    else:print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if args.inspect or report['status']=='computed_complete80' else 3
if __name__=='__main__':raise SystemExit(main())

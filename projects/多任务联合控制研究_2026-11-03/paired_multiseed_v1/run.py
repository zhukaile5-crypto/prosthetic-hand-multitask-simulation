"""Ten new seeds, four scenes, two frozen controllers; statistics only."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'output/cold_start_take_pilot_v1_20261112'
SCENES = ('nominal','friction_drop','fragile_conflict','collision')
ARMS = ('baseline','macro')

STAT_SPEC = {
    'primary_success': 'registered full_task_completed AND every physical interval in (15.8,16.0] on receiver top >0.05N, hand <=0.2N, floor <=0.02N',
    'historical_success_also_reported': 'unchanged full_task_completed endpoint, reported separately',
    'force_quality': 'unchanged 50Hz FIR total-force reversal rate N/s and SD N; full and lifecycle stages',
    'force_reporting_populations': ['all assigned trials with valid measurements','paired functional subset: both methods pickup_completed and retained_to_release'],
    'stable_time': 'unchanged pickup_stable_time_s; missing values are failures, never zero; 7.5s-capped time additionally reported',
    'release_time': 'unchanged release_latency_s and steady_placement_time_s; report number attaining each endpoint',
    'safety': 'unchanged raw all-external finger limits plus total budget, tolerance 0.02N; pass rate, peak force, exceedance duration and impulse',
    'primary_denominators': '10 trials per method per scene; 40 per method overall',
    'inference': 'paired differences macro minus baseline; resample seed clusters for pooled CIs, keeping all four conditions together',
    'bootstrap_replicates': 20000,
    'bootstrap_seed': 2026111207,
    'binary_scene_intervals': '95% Wilson intervals',
    'missing_or_runtime_failures': 'retain every assignment; distinguish runtime/physical failures from task failures; never replace seeds or discard bad outcomes',
    'selection': '10 unique system-random seeds sampled before outcomes; reject previously used/planned project seeds',
    'multiple_metrics': 'descriptive paired estimates and 95% intervals; no unadjusted multi-metric significance claims',
    'plots_or_videos': False,
}

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(p)
def now(): return datetime.now(timezone.utc).isoformat()

def history_seeds():
    found={619741}; files=[]
    def collect(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k=='seed' and isinstance(v,int): found.add(v)
                elif k in ('seeds','development_seeds','confirmation_seeds','robustness_seeds') and isinstance(v,list):
                    found.update(n for n in v if isinstance(n,int))
                elif isinstance(v,(dict,list)): collect(v)
        elif isinstance(value,list):
            for v in value: collect(v)
    for base in (ROOT/'output',ROOT.parent/'五指策略接入_2026-11-03/output'):
        if not base.is_dir(): continue
        for name in ('metadata.json','plan.json'):
            for p in base.rglob(name):
                collect(read(p));files.append(str(p))
    return sorted(found), files

def evaluate_terminal_and_checks(directory,result):
    import numpy as np
    directory=Path(directory);meta=read(directory/'metadata.json')
    with np.load(directory/'trace.npz',allow_pickle=False) as a:
        t=a['time'];n=a['normal'];contact=n>.2
        stable=(contact.sum(axis=1)>=3)&contact[:,0]&(a['position'][:,2]-meta['initial_position'][2]>=.025)&(a['support']<=.05)
        stable &= abs(a['lift']-meta['config']['lift_target'])<=.01
        stable &= np.linalg.norm(a['relative_velocity'],axis=1)<=.005
        stable &= np.linalg.norm(a['angular_velocity'],axis=1)<=.15
        stable &= t<=7.5+1e-9
        count=41
        idx=np.flatnonzero(np.convolve(stable.astype(int),np.ones(count,int),mode='valid')==count)
        pickup=None if len(idx)==0 else float(t[idx[0]+count-1])
        assert pickup==result['pickup_stable_time_s']
        raw=a['substep_skin_normal'];endpoints=a['skin_normal'];limits=np.array(meta['limits'])
        over=(np.any(raw>limits+.02,axis=1))|(raw.sum(axis=1)>meta['budget']+.02)
        passed=not (over.any() or np.any(endpoints>limits+.02) or np.any(endpoints.sum(axis=1)>meta['budget']+.02))
        assert bool(passed)==result['force_limit_pass']
        exceed=float(np.sum(a['substep_duration'][over]))
        assert abs(exceed-result['force_exceedance_duration_s'])<1e-9
        raw_force_mean=float(a['substep_normal'].sum(axis=1).mean())
        assert abs(raw_force_mean-result['total_force']['full']['raw_total_mean_N'])<1e-10
    with np.load(directory/'delivery_intervals.npz',allow_pickle=False) as d:
        end=d['interval_end_s'];mask=(end>15.8+1e-8)&(end<=16.+1e-8)
        assert mask.sum()==400
        valid=bool(np.all(d['contact_valid']) and np.all(d['object_state_valid']))
        on_top=d['target_receiver_top_normal_N']>.05
        clear=d['target_hand_normal_N']<=.2
        floor_free=d['target_floor_normal_N']<=.02
        terminal=bool(valid and np.all((on_top&clear&floor_free)[mask]))
        floor_idx=np.flatnonzero((end>12.+1e-9)&~floor_free)
    extra=dict(schema='prospective_terminal_success_v1',
        terminal_on_receiver_and_clear=terminal,
        terminal_qualified_completion=bool(result['full_task_completed'] and terminal),
        safe_terminal_completion=bool(result['full_task_completed'] and terminal and result['force_limit_pass']),
        first_post_release_floor_contact_s=None if not len(floor_idx) else float(end[floor_idx[0]]),
        pickup_independently_recomputed=True,limit_pass_independently_recomputed=True,
        exceedance_duration_independently_recomputed=True,raw_mean_force_independently_recomputed=True)
    write(directory/'repeated_evaluation.json',extra)
    return extra

def worker(job,plan,output):
    # Reuse the unchanged trial loop, saved-record evaluator, and numeric carrier.
    from cold_start_take_v1.run import worker as original_worker
    response=original_worker(job,plan,output)
    if response['status']=='evaluated':
        try:
            response['repeat_metrics']=evaluate_terminal_and_checks(Path(output)/'runs'/job['id'],response['result'])
        except Exception as exc:
            response['status']='postcheck_failure'
            response['postcheck_error']=f'{type(exc).__name__}: {exc}'
            response['postcheck_traceback']=traceback.format_exc()
    write(Path(output)/'runs'/job['id']/'assignment_record.json',response)
    return response

def progress(out,plan,results,began,status):
    complete={r['job']['id'] for r in results};counts={s:0 for s in plan['seeds']}
    for r in results: counts[r['job']['seed']]+=1
    active=[]
    for p in (out/'progress').glob('*.json'):
        if p.stem in complete: continue
        try:
            r=read(p);active.append(dict(job=p.stem,time_s=r['simulated_time_s'],phase=r['phase']))
        except (ValueError,OSError): pass
    elapsed=time.perf_counter()-began
    value=dict(status=status,updated_utc=now(),elapsed_s=elapsed,total=80,completed=len(results),
        evaluated=sum(r['status']=='evaluated' for r in results),
        infrastructure_or_postcheck_errors=sum(r['status']!='evaluated' for r in results),
        completed_seed_groups=sum(n==8 for n in counts.values()),seed_counts=counts,active=active,workers=plan['workers'])
    write(out/'progress_summary.json',value)
    lines=[f'状态：{status}',f'已完成：{len(results)}/80 次，完整种子组：{value["completed_seed_groups"]}/10',
        f'并行数：{plan["workers"]}，已运行：{elapsed/60:.1f} 分钟',
        f'技术异常：{value["infrastructure_or_postcheck_errors"]}',
        '当前策略、参数及评价规则固定；不绘图。',f'更新时间（UTC）：{value["updated_utc"]}']
    (out/'当前进度.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    print(json.dumps({k:value[k] for k in ('status','completed','completed_seed_groups','infrastructure_or_postcheck_errors','elapsed_s')},ensure_ascii=False),flush=True)

def main():
    from v22_joint_task_fusion.numerics import ensure_numeric_environment,numerical_identity
    ensure_numeric_environment()
    from cold_start_take_v1.runtime import identity,source_files
    from cold_start_take_v1.metrics import METRIC_SPEC
    from cold_start_take_v1.run import pair_audit
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();out=Path(args.out).resolve()
    assert 1<=args.workers<=8
    assert out.is_relative_to(ROOT/'output') and not out.exists()
    old=read(OLD/'plan.json');sources=identity();numerics=numerical_identity()
    assert sources==old['sources_sha256'],'Controller/model/evaluator source differs from the delivered pilot.'
    assert numerics==old['numerics'],'Numeric environment differs from the delivered pilot.'
    used,history_files=history_seeds()
    rng=secrets.SystemRandom();seeds=[]
    while len(seeds)<10:
        seed=rng.randrange(1_000_000,2_000_000_000)
        if seed not in used and seed not in seeds: seeds.append(seed)
    jobs=[dict(id=f'{arm}__{scene}__{seed}',arm=arm,scene=scene,seed=seed)
          for seed in seeds for scene in SCENES for arm in ARMS]
    preserved={str(p):sha(p) for p in OLD.rglob('*') if p.is_file()}
    report=ROOT.parent/'中期汇报_宏观策略_2026-11-11'
    preserved.update({str(p):sha(p) for p in report.rglob('*') if p.is_file()})
    harness={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    plan=dict(schema='fixed_policy_multiseed_plan_v1',status='frozen_before_execution',created_utc=now(),
        seeds=seeds,jobs=jobs,workers=args.workers,sources_sha256=sources,numerics=numerics,
        prior_pilot=str(OLD),prior_plan_sha256=sha(OLD/'plan.json'),duration_s=16.,control_period_s=.005,physics_period_s=.0005,
        original_metric_spec=METRIC_SPEC,statistics_spec=STAT_SPEC,
        no_policy_or_parameter_changes=True,no_parameter_search=True,no_seed_replacements=True,no_automatic_trial_retries=True,
        preserve_failures=True,no_plotting=True,seed_history_files_checked=len(history_files),historical_seeds=used,
        prospective_seed_selection=True,previous_development_seed_excluded=619741 not in seeds,
        harness_sha256=harness,preserved_outputs_sha256=preserved,
        metadata_labels='Original runtime metadata calls itself development/pilot; outer plan defines this new 10-seed repeat batch.')
    out.mkdir();write(out/'plan.json',plan)
    write(out/'seed_manifest.json',dict(seeds=seeds,history_files_checked=len(history_files),history_files=history_files,previous_seeds=used))
    for key,path in source_files().items():
        if path.suffix in ('.py','.json'):
            target=out/'source_snapshot'/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for path in Path(__file__).parent.glob('*.py'):
        target=out/'evaluation_harness'/path.name;target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
    began=time.perf_counter();results=[]
    print(json.dumps(dict(status='frozen_and_started',seeds=seeds,jobs=len(jobs),workers=args.workers,plan_sha256=sha(out/'plan.json')),ensure_ascii=False),flush=True)
    progress(out,plan,results,began,'running')
    with ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=1) as executor:
        pending={executor.submit(worker,job,plan,str(out)):job for job in jobs}
        while pending:
            done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                job=pending.pop(future)
                try: response=future.result()
                except Exception as exc:
                    response=dict(status='worker_failure',job=job,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
                    write(out/'runs'/job['id']/'assignment_record.json',response)
                results.append(response)
                write(out/'interim_jobs.json',results)
                print(json.dumps(dict(job=job,status=response['status'],wall_seconds=response.get('wall_seconds')),ensure_ascii=False),flush=True)
            progress(out,plan,results,began,'running')
    progress(out,plan,results,began,'checking_pairs')
    pairs={f'{scene}__{seed}':pair_audit(out,scene,seed) for seed in seeds for scene in SCENES}
    audit=dict(sources_unchanged=identity()==sources,numerics_unchanged=numerical_identity()==numerics,
        prior_outputs_unchanged=all(sha(p)==h for p,h in preserved.items()),
        expected_jobs=80,retained_jobs=len(results),evaluated_jobs=sum(r['status']=='evaluated' for r in results),
        expected_pairs=40,pairs=pairs,
        physical_audits_passed=all(all(r['result']['physical_audit'].values()) for r in results if r['status']=='evaluated'),
        harness_unchanged=all(sha(Path(__file__).parent/name)==h for name,h in harness.items()))
    audit['passed']=bool(audit['sources_unchanged'] and audit['numerics_unchanged'] and audit['prior_outputs_unchanged']
        and audit['retained_jobs']==audit['evaluated_jobs']==80 and all(v['passed'] for v in pairs.values())
        and audit['physical_audits_passed'] and audit['harness_unchanged'])
    write(out/'audit.json',audit)
    write(out/'summary.json',dict(plan_sha256=sha(out/'plan.json'),finished_utc=now(),wall_seconds=time.perf_counter()-began,jobs=results,audit=audit))
    progress(out,plan,results,began,'simulation_complete' if audit['passed'] else 'simulation_complete_with_audit_findings')

if __name__=='__main__':main()

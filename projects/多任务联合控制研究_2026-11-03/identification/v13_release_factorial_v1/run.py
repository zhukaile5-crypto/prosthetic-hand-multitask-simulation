"""ROOT-only -m dispatcher: zero preflight -> pin -> exact both -> other3."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time,traceback
from .inputs import ROOT,VARIANTS,load,read,sha
from .pinning import identity,input_identity,pin,require,snapshot,LOCK
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def preflight(out):
    from .program import preflight as verify
    r=load();before=identity()
    try:value=verify(r)
    except Exception as exc:value=dict(passed=False,error=repr(exc),traceback=traceback.format_exc())
    value.update(schema='v13_release_factorial_zero_preflight_v1',sources=before,input=input_identity(r),sources_unchanged=before==identity(),physics_steps=0)
    value['passed']=value['passed'] and value['sources_unchanged'];write(out,value);return value
def worker(variant,target):
    from .replay import run_job
    r=load();require(r);return run_job(r,variant,target)
def execute(args):
    r=load();lock=require(r);before=identity()
    if args.phase=='bridge':variants=['both'];bridge=None
    else:
        bridge=read(Path(args.bridge)/'execution_complete.json')
        if not (bridge.get('passed') and bridge.get('execution_complete') and bridge.get('sources_unchanged') and bridge.get('identical_noise_rng_final') and bridge.get('both_bridge_all_recomputed_exact') and bridge['source_lock_sha256']==sha(LOCK)):
            raise ValueError('Same-lock exact both bridge mandatory before other3')
        variants=['lift_only','fingers_only','neither']
    target=Path(args.out).resolve()
    if not target.is_relative_to((ROOT/'output').resolve()) or target==(ROOT/'output').resolve():raise ValueError('Fresh ROOT/output descendant required')
    target.mkdir(parents=True,exist_ok=False)
    jobs=[dict(variant=v,destination=str(target/v)) for v in variants]
    write(target/'manifest.json',dict(schema='v13_release_factorial_jobs_v1',phase=args.phase,jobs=jobs,source_lock_sha256=sha(LOCK),input=lock['input'],
        role='fixed_input_contact_chain_diagnostic_no_new_feedback_or_strategy_success',all_four_scientific_cells=['both','lift_only','fingers_only','neither']))
    del r;began=time.perf_counter();results=[];failures=[];stage='snapshot'
    try:
        snapshot(target,lock);stage='dispatch'
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fs={}
            # Keep already-submitted jobs observable if a later submit fails.
            for job in jobs:
                try:fs[pool.submit(worker,job['variant'],job['destination'])]=job
                except Exception as exc:
                    failures.append(dict(stage='dispatch',job=job,error=repr(exc),traceback=traceback.format_exc()));break
            for f in as_completed(fs):
                job=fs[f]
                try:result=f.result()
                except Exception as exc:result=dict(variant=job['variant'],complete=False,error=repr(exc),traceback=traceback.format_exc())
                results.append(result);print(json.dumps(dict(variant=job['variant'],complete=result['complete']),ensure_ascii=False),flush=True)
    except Exception as exc:
        failures.append(dict(stage=stage,error=repr(exc),traceback=traceback.format_exc()))
    finally:
        results.sort(key=lambda x:variants.index(x['variant']))
        complete=len(results)==len(jobs) and all(x['complete'] for x in results) and not failures
        rng=[x.get('noise_rng_final') for x in results]
        if bridge is not None:rng.append(bridge['results'][0].get('noise_rng_final'))
        same_rng=bool(rng and None not in rng and len(set(rng))==1)
        try:same_sources=before==identity()
        except Exception as exc:
            same_sources=False;failures.append(dict(stage='final_source_check',error=repr(exc),traceback=traceback.format_exc()));complete=False
        both=bool(args.phase=='bridge' and complete and results[0]['bridge_all_recomputed_exact'])
        assignment=[]
        for job in jobs:
            result=next((x for x in results if x['variant']==job['variant']),None)
            if result is None:assignment.append(dict(job=job,status='not_executed_due_to_snapshot_or_dispatch_failure',stage=stage))
            else:assignment.append(dict(job=job,status='completed' if result['complete'] else 'runtime_failed',stage='runtime',error=result.get('error')))
        outcome=dict(schema='v13_release_factorial_completion_v1',phase=args.phase,execution_complete=complete,results=results,failures=failures,
            sources_unchanged=same_sources,source_lock_sha256=sha(LOCK),identical_noise_rng_final=same_rng,both_bridge_all_recomputed_exact=both,
            rng_scope='Source final RNG was not saved: source3201 public packets must match; all new factor branches must share the same final RNG state.',
            wall_seconds=time.perf_counter()-began,new_feedback_decisions=0,new_strategy_success_rate=None,
            passed=complete and same_sources and same_rng and (both if args.phase=='bridge' else True))
        write(target/'assignment_ledger.json',assignment);write(target/'execution_complete.json',outcome)
    return outcome
def main():
    p=argparse.ArgumentParser(description=__doc__);kind=p.add_mutually_exclusive_group(required=True)
    kind.add_argument('--preflight',action='store_true');kind.add_argument('--pin-sources',action='store_true');kind.add_argument('--execute',action='store_true')
    p.add_argument('--out');p.add_argument('--preflight-result');p.add_argument('--phase',choices=('bridge','factorial'));p.add_argument('--bridge');p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    if args.preflight:
        if not args.out:p.error('fresh --out required')
        value=preflight(args.out);print(json.dumps(dict(passed=value['passed'],physics_steps=0)));return 0 if value['passed'] else 1
    if args.pin_sources:
        if not args.preflight_result:p.error('--preflight-result required')
        check=read(args.preflight_result);check['_file_sha256']=sha(args.preflight_result);value=pin(load(),check)
        print(json.dumps(dict(source_count=len(value['sources']),native_binaries=len(value['environment']['native_package_binaries_sha256']),lock_sha256=sha(LOCK))));return 0
    if not args.out or not args.phase or not 1<=args.workers<=3:p.error('fresh --out --phase and1..3workers required')
    if args.phase=='factorial' and not args.bridge:p.error('complete --bridge required')
    value=execute(args);print(json.dumps(dict(passed=value['passed'],complete=value['execution_complete'])));return 0 if value['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

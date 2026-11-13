"""Review frozen report arithmetic with synthetic fixtures; no formal results."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import ast,copy,hashlib,json,runpy,tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    path=ROOT/'analysis_report.py';source=path.read_text(encoding='utf-8');tree=ast.parse(source)
    namespace=runpy.run_path(str(path))
    checks=dict(report_frozen_identity=sha(path)=='8d6ab4ca8d1eba001924bc297a75f776ade0cdc57924b26a857f7cfa35c5e8a9',
        no_production_import=not any(isinstance(n,ast.ImportFrom) and n.module and n.module.split('.')[0] in
            ('session','hand_plant','controller_adapter','policy_sources','evaluate_runs','evaluate_recovery') for n in ast.walk(tree)),
        author_self_tests=namespace['self_test']()['passed'])
    paired=namespace['paired'];bootstrap=namespace['bootstrap_pairs']
    rows=[dict(method=m,scenario='a',shape='capsule',seed=s,static_eligible=(m=='ratio' or s==1),
        mean_force=1. if m=='ratio' else 3.,release_success=(m=='ratio' or s==1),release_latency=.7) for s in (1,2) for m in ('ratio','unified')]
    checks['joint_window_only']=len(paired(rows,'mean_force',True))==1
    checks['explicit_both_successful_release_only']=len(paired(rows,'release_latency'))==1
    checks['empty_has_null_not_zero']=bootstrap([])['difference'] is None
    pairs=[((f's{j}','capsule',i),0.,float(i)) for i in range(8) for j in range(i+1)]
    out=bootstrap(pairs,200);rng=np.random.default_rng(20261103);draws=[]
    for _ in range(200):
        chosen=rng.integers(0,8,size=8)
        draws.append(np.mean([b-a for s in chosen for identity,a,b in pairs if identity[2]==s]))
    checks['whole_seed_unbalanced_bootstrap']=np.allclose(out['ci95'],np.quantile(draws,[.025,.975]),atol=1e-12,rtol=0.) and out['clusters']==8
    plan=json.loads((ROOT/'review/recovery_plan_v1.json').read_text(encoding='utf-8'))
    primary=[];records=[];summary=[]
    for method in ('ratio','unified'):
        p=dict(method=method,shape='capsule',scenario='three_pulses',seed=999999,boundary=False,trace_sha256=None)
        primary.append(p)
        events=[dict(**spec,status='invalid_record',stable_before=False,exposed=False,first_instability=None,
            latency_from_event=None,latency_from_instability=None,recovered=False,censored=False) for spec in plan['events']['three_pulses']]
        # Deliberately omit boundary from secondary invalid records; report must
        # use canonical primary identity instead of crashing or mislabelling.
        records.append({k:v for k,v in p.items() if k!='boundary'}|dict(record_complete=False,events=events))
        summary.append(dict(method=method,trials=1,trials_with_events=1,scheduled_events=3,status_counts={'invalid_record':3},
            eligible_events=0,unstable_eligible_events=0,recovered_events=0,recovery_fraction=None,recovered_latency_median=None))
    result=dict(schema='secondary_event_recovery_results_v1',role='descriptive_secondary_only',
        recovery_plan_sha256=sha(ROOT/'review/recovery_plan_v1.json'),analyzer_sha256=sha(ROOT/'evaluate_recovery.py'),records=records,rows=summary)
    loader=namespace['load_recovery']
    def rejects(p):
        try:loader(p,primary,dict(input_sha256_before={}))
        except ValueError:return True
        return False
    with tempfile.TemporaryDirectory(prefix='independent_report_fixture_',dir=ROOT/'review') as temporary:
        directory=Path(temporary);good=directory/'good.json';good.write_text(json.dumps(result),encoding='utf-8')
        accepted=loader(good,primary,dict(input_sha256_before={}))
        checks['invalid_records_full_denominator_and_boundary']=len(accepted['flattened_events'])==6 and accepted['missing_primary_records']==0 and all(e['boundary'] is False for e in accepted['flattened_events'])
        bad=copy.deepcopy(result);bad['records']=bad['records'][:1]
        missing=directory/'missing_trial.json';missing.write_text(json.dumps(bad),encoding='utf-8')
        checks['reject_dropped_trial']=rejects(missing)
        bad=copy.deepcopy(result);bad['records'][0]['events']=bad['records'][0]['events'][:2]
        bad['rows'][0]['scheduled_events']=2;bad['rows'][0]['status_counts']={'invalid_record':2}
        missing=directory/'missing_event.json';missing.write_text(json.dumps(bad),encoding='utf-8')
        checks['reject_dropped_event_even_adjusted_aggregate']=rejects(missing)
    # Read-only source checks establish that the exact registered Cartesian
    # product and complete per-run result equality are enforced by the loader.
    loader_tree=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_confirmation')
    code=ast.unparse(loader_tree)
    checks['registered_cartesian_enforced']='registered != expected' in code and "sorted(phase['seeds'])" in code
    checks['per_run_result_equals_matrix']='saved_result != original[key(row)]' in code
    checks['raw_trace_provenance']='actual_sha != meta.get' in code and "meta.get('source_sha256') != manifest['source_sha256']" in code
    checks['stage_guard_limits_and_budget']='analysis_stage_substep_protection' in code and 'limits + 0.02' in code and "float(budget) + 0.02" in code
    certificate=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
        created_utc=datetime.now(timezone.utc).isoformat(),analysis_report_sha256=sha(path),reviewer_sha256=sha(__file__),
        recovery_plan_sha256=sha(ROOT/'review/recovery_plan_v1.json'),
        scope='Frozen report tool only: static import/source audit and synthetic pairing/bootstrap/recovery denominator fixtures. No confirmation performance accessed; no runtime, controller, evaluator or physics imported.')
    target=ROOT/('review/independent_report_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    target.write_text(json.dumps(certificate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=certificate['passed'],checks=len(checks),failed=certificate['failed'],path=str(target),sha256=sha(target)),ensure_ascii=False))


if __name__=='__main__':main()

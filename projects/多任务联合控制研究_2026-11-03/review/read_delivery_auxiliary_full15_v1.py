"""Posthoc regression on exactly3contexts x5historicalroles; no new physics."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).parent))
import delivery_auxiliary_metrics_v1 as aux

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main(out):
    roster=ROOT/'review/v17_release_mechanisms_full15_v1.json';prior=read(roster);rows=prior['records']
    if len(rows)!=15 or len({r['run'] for r in rows})!=15:raise ValueError('Exactly15unique historical records required')
    author=read(ROOT/'review/delivery_auxiliary_metrics_author_v1.json')
    if not author['passed'] or author['calculator_sha256']!=sha(Path(aux.__file__)) or author['contract_sha256']!=sha(aux.CONTRACT_PATH):raise ValueError('Calculator/contract author gate identity changed')
    results=[];failures=[];group=defaultdict(list)
    for row in rows:
        p=Path(row['run']);identity=dict(role=row['role'],context=row['context'],run=str(p))
        try:
            meta=read(p/'metadata.json');original=read(p/'result.json')
            with np.load(p/'delivery_intervals.npz',allow_pickle=False) as z:
                arrays={key:z[key] for key in aux.C['input_fields']}
            release=meta.get('mechanical_release_at',meta.get('release_at'))
            if release is None:raise ValueError('Actual public release clock missing')
            metrics=aux.calculate(arrays,release_at_s=release,episode_end_s=meta['duration'],physical_period_s=meta['physics_timestep'],original_result=original)
            result=dict(**identity,identity=dict(scenario=meta['scenario'],motion=meta['motion_mode'],seed=meta['seed'],shape=meta['shape'],method=meta['method']),
                original_release_exposed=(original.get('research_metrics',{}).get('release',{}).get('physical_exposure')),
                handoff_passed=meta['handoff_passed'],metrics=metrics,
                input_sha256={name:sha(p/name) for name in ('metadata.json','result.json','delivery_intervals.npz')})
        except Exception as exc:
            result=dict(**identity,status='invalid_record',error=repr(exc),metrics=None)
            failures.append(dict(**identity,error=repr(exc)))
        results.append(result);group[row['role']].append(result)
    summaries={}
    for role,rr in group.items():
        valid=[r['metrics'] for r in rr if r.get('metrics') is not None]
        summaries[role]=dict(assigned=len(rr),computed=len(valid),complete_coverage=sum(m['coverage']['all_valid'] for m in valid),
            original_task_success=sum(m['original_result'].get('task_success') is True for m in valid),
            original_release_success=sum(m['original_result'].get('release_success') is True for m in valid),
            original_force_guard_pass=sum(m['original_result'].get('safety',{}).get('force_protection_pass') is True for m in valid),
            first_sampled_stable_receiver_found=sum(m['official_first_stable_receiver'] is not None for m in valid),
            stable_by_original_release_deadline=sum(m['official_first_stable_by_release_deadline'] is not None for m in valid),
            final_200ms_sampled_stable_receiver_suffix=sum(m['official_final_stable_suffix'] is not None and not m['official_final_stable_suffix']['right_censored'] for m in valid),
            ever_floor_true=sum(m['ever_floor_contact'] is True for m in valid),ever_floor_unknown=sum(m['ever_floor_contact'] is None for m in valid),
            terminal_moving_receiver_contact=sum(m['terminal'] is not None and m['terminal']['moving_receiver_contact'] is True for m in valid),
            no_weighted_total_score=True)
    report=dict(schema='delivery_auxiliary_full15_posthoc_v1',created_utc=datetime.now(timezone.utc).isoformat(),complete=len(results)==15,
        runtime_failures=failures,records=results,all_assigned_by_historical_role=summaries,
        contract_sha256=sha(aux.CONTRACT_PATH),calculator_sha256=sha(Path(aux.__file__)),author_gate_sha256=sha(ROOT/'review/delivery_auxiliary_metrics_author_v1.json'),
        cohort_roster_sha256=sha(roster),regression_reader_sha256=sha(Path(__file__)),
        scope=dict(contexts=3,historical_roles=5,physical_seed_clusters=1,not15_independent_samples=True,
            posthoc_all_historical=True,only_future_registered_fresh_cohort_prospective=True,
            production_sources_modified=False,physics_steps=0,model_constructions=0,trace_or_contact_recomputation=False))
    out=ROOT/out
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(aux._plain(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(out),records=len(results),failures=failures,summaries=summaries,sha256=sha(out)),ensure_ascii=False))
    return 0 if not failures else 1
if __name__=='__main__':raise SystemExit(main(sys.argv[1]))

"""Lightweight archived JSON/gzip decision inventory. NEVER reads NPZ."""
from pathlib import Path
from collections import Counter
import argparse,gzip,hashlib,json,struct

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def packed(xs):return struct.pack('<'+'d'*len(xs),*xs)
def count(rows,key):return dict(Counter(str(key(d)) for d in rows))
def compact(d):
    q=d.get('qp') or {};n=d.get('native') or {};a=d.get('proposal_acceptance') or {};f=d.get('force_intent_debug') or {};m=f.get('task_metadata') or {}
    return dict(time_s=d['time_s'],mode=d['mode'],qp_status=q.get('status'),accepted=a.get('accepted'),optimality_certified=a.get('optimality_certified'),
        native_status=n.get('status'),full_native_check=(n.get('diagnostics') or {}).get('prepared_value_final_full_recheck'),
        force_task_label=f.get('label'),fallback_reason=f.get('reason'),conditional_load=d.get('task_activation',{}).get('conditional_load'),
        public_patch_count=len(d.get('observed_patch_normal',[])),observed_public_patch_normal_sum_N=sum(d.get('observed_patch_normal',[])),
        selected_patch_count=m.get('selected_patch_count'),native_history_status=(d.get('native_tasks') or {}).get('history_task_status'))
def summarize(rows):
    checks=dict(certified_accepted=0,provisional_accepted=0,rejected=0,no_qp_or_acceptance=0,native_absent=0,full_native_recheck_passed=0,full_native_recheck_failed=0,full_native_recheck_missing=0,
        native_reference_bounds_false=0,native_engine_bounds_false=0,accepted_delta_bytes_mismatch=0,p0_bytes_mismatch=0,objective_bytes_false=0)
    anomalies=[];maxerror=0.
    for d in rows:
        a=d.get('proposal_acceptance');q=d.get('qp');n=d.get('native')
        if a is None or q is None:checks['no_qp_or_acceptance']+=1
        elif a.get('accepted'):checks['certified_accepted' if a.get('optimality_certified') else 'provisional_accepted']+=1
        else:checks['rejected']+=1;anomalies.append(compact(d))
        if a is not None and a.get('objective_bytes_equal') is False:checks['objective_bytes_false']+=1
        if n is None:checks['native_absent']+=1;continue
        di=n.get('diagnostics') or {};r=di.get('prepared_value_final_full_recheck')
        if r is None:checks['full_native_recheck_missing']+=1;anomalies.append(compact(d))
        elif r.get('passed') is True and r.get('native_bytes_equal') is True and r.get('transmission_bytes_equal') is True:checks['full_native_recheck_passed']+=1
        else:checks['full_native_recheck_failed']+=1;anomalies.append(compact(d))
        for field,key in [('actual_reference_bounds_ok','native_reference_bounds_false'),('actual_engine_bounds_ok','native_engine_bounds_false')]:
            if di.get(field) is False:checks[key]+=1;anomalies.append(compact(d))
        if q and packed(n.get('requested_delta_effort21',[]))!=packed(q.get('d21',[])):checks['accepted_delta_bytes_mismatch']+=1;anomalies.append(compact(d))
        if packed(n.get('p0_effort21',[]))!=packed((d.get('domain') or {}).get('p0_effort21',[])):checks['p0_bytes_mismatch']+=1;anomalies.append(compact(d))
        err=di.get('scaled_realization_linf')
        if isinstance(err,(int,float)):maxerror=max(maxerror,err)
    return dict(decisions=len(rows),first_time_s=None if not rows else rows[0]['time_s'],last_time_s=None if not rows else rows[-1]['time_s'],counts=checks,
        qp_status=count(rows,lambda d:(d.get('qp') or {}).get('status')),domain_status=count(rows,lambda d:(d.get('domain') or {}).get('status')),
        native_status=count(rows,lambda d:(d.get('native') or {}).get('status')),force_task_labels=count(rows,lambda d:(d.get('force_intent_debug') or {}).get('label')),
        force_fallback_reasons=count(rows,lambda d:(d.get('force_intent_debug') or {}).get('reason')),
        conditional_load=count(rows,lambda d:d.get('task_activation',{}).get('conditional_load')),
        conditional_load_reasons=count(rows,lambda d:d.get('task_activation',{}).get('conditional_load_reason')),
        native_history_status=count(rows,lambda d:(d.get('native_tasks') or {}).get('history_task_status')),
        max_scaled_native_realization_linf=maxerror,anomalies=anomalies)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='output/v15_holding_task_ablation_v1');ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    base=ROOT/a.input;end=load(base/'execution_complete.json')
    if end.get('execution_complete') is not True:raise RuntimeError('Group not terminal')
    records=[];checks=[]
    for run in sorted((base/'runs').iterdir()):
        if not run.is_dir():continue
        meta=load(run/'metadata.json');result=load(run/'result.json');file=run/'joint_decisions.jsonl.gz'
        with gzip.open(file,'rt',encoding='utf-8') as f:rows=[json.loads(line) for line in f if line.strip()]
        failure=result.get('failure_at');pre=[d for d in rows if failure is not None and d['time_s']<=failure]
        modes={mode:summarize([d for d in rows if d['mode']==mode]) for mode in sorted({d['mode'] for d in rows})}
        transitions=[];last=None
        for d in rows:
            f=d.get('force_intent_debug') or {};key=(d['mode'],f.get('label'),f.get('reason'),d.get('task_activation',{}).get('conditional_load'))
            if key!=last:transitions.append(compact(d));last=key
        checks.append(dict(run=run.name,decisions426=len(rows)==426,holding225=len([d for d in rows if d['mode']=='holding'])==225,release201=len([d for d in rows if d['mode']=='release'])==201,
            policy_registered=meta.get('task_policy')=='integrated' and meta.get('receiver_policy')=='blind',ids_unique=len({d['decision_id'] for d in rows})==len(rows)))
        records.append(dict(run=run.name,method=meta.get('research_method',meta.get('method')),original_result={k:result.get(k) for k in ('task_success','failure_reason','failure_at','release_exposed','release_success','release_latency')},
            runtime_metadata_failure=dict(reason=meta.get('failure_reason'),time_s=meta.get('failure_at')),
            by_mode=modes,all_decisions=summarize(rows),pre_original_failure=summarize(pre) if failure is not None else None,
            last_pre_failure_decisions=[compact(d) for d in pre[-6:]],force_task_transitions=transitions,
            source_sha256_declared=meta.get('research_source_sha256'),
            input_sha256={p.name:sha(p) for p in (run/'metadata.json',run/'result.json',file)}))
    report=dict(schema='v15_holding_ablation_decision_status_readout_v1',input=str(base),checks=checks,records=records,
        execution_complete_declared=end,input_completion_sha256=sha(base/'execution_complete.json'),npz_files_read=0,models_constructed=0,physics_steps=0,
        scope='All three complete JSON/gzip decision streams. Counts and saved native/optimality certificates inspected; no solver rerun, raw force/trajectory recomputation or control timing. QPz is an intent, not measured pressure. Evaluation failure and runtime failure are distinct.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([dict(run=r['run'],original_result=r['original_result'],by_mode={k:dict(counts=v['counts'],qp=v['qp_status'],native=v['native_status'],labels=v['force_task_labels'],conditional=v['conditional_load']) for k,v in r['by_mode'].items()},pre_failure=None if r['pre_original_failure'] is None else r['pre_original_failure']['counts']) for r in records],ensure_ascii=False))
if __name__=='__main__':main()

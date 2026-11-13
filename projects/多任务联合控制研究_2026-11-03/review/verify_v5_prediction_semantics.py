"""Supplement an immutable audit with the passive ablation's declared semantics."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'output/development_v5_pilot'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--base-audit',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();base=read(a.base_audit);manifest=read(FOLDER/'manifest.json');rows=read(FOLDER/'trials.json')
    checks={};records=[]
    checks['same_original_trials_manifest_execution_identity']=all(sha(FOLDER/n)==base[k] for n,k in [('trials.json','trials_sha256'),('manifest.json','manifest_sha256'),('execution_complete.json','execution_complete_sha256')])
    expected={str((r['scenario'],r['method'],r['seed']))+'/unavailable_force_predictions_not_falsely_timestamped' for r in rows if r['method']=='passive'}
    checks['exact_original_failures_are_only_this_semantic_assumption']=set(base['failed'])==expected and len(expected)==6
    checks['other853_original_checks_passed']=sum(base['checks'].values())==853 and len(base['checks'])==859
    source=FOLDER/'sources/passive_control.py';checks['frozen_passive_source_identity']=sha(source)==manifest['research_source_sha256']['passive_control.py']
    for row in rows:
        path=Path(row['trace_path'])
        with np.load(path,allow_pickle=False) as z:
            available=z['prediction_available'];pred=z['predicted_normal'];stamp=z['prediction_at'];debug=z['debug_json'];time=z['time']
        absent=~available;retained=absent&np.isfinite(pred).all(1);item=dict(identity=[row['scenario'],row['method'],row['seed']],retained_nonforecast_samples=int(retained.sum()),trace_identity=sha(path)==row['trace_sha256'])
        item['unavailable_forecast_never_has_prediction_timestamp']=bool(np.isnan(stamp[absent]).all())
        item['available_forecasts_finite']=bool(np.isfinite(pred[available]).all())
        if row['method']=='passive':
            item['nonforecast_placeholder_declared']=bool(retained.sum()==900 and all(json.loads(str(debug[i])).get('passive_hold') is True and json.loads(str(debug[i])).get('prediction_available') is False and 'no autonomous force prediction' in json.loads(str(debug[i])).get('prediction_unavailable_reason','') for i in np.flatnonzero(retained)))
            item['no_passive_predictor_is_evaluated']=not available.any()
        else:item['nonforecast_placeholders_absent_from_autonomous_methods']=bool(np.isnan(pred[absent]).all())
        checks.update({str(tuple(item['identity']))+'/'+k:bool(v) for k,v in item.items() if isinstance(v,bool)});records.append(item)
    result=dict(schema='independent_v5_passive_prediction_semantics_supplement_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        semantic_recheck_passed=all(checks.values()),saved_data_integrity_and_metric_audit_passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
        original_audit_path=str(a.base_audit.resolve()),original_audit_sha256=sha(a.base_audit),original_audit_passed_flag_preserved=base['passed'],original_audit_modified=False,
        original_source_archive='review/checker_archive/independent_development_v5_review_'+base['reviewer_sha256']+'.py',trials_sha256=sha(FOLDER/'trials.json'),manifest_sha256=sha(FOLDER/'manifest.json'),
        passive_source_sha256=sha(source),reviewer_sha256=sha(__file__),records=records,
        interpretation='The passive ablation retains the last observed reference in a legacy predicted field, explicitly marks prediction_available=false and prediction_at=NaN. It has no autonomous forecast; those retained values must not enter prediction-error comparisons. Other autonomous methods still require NaN when unavailable. This corrects a checker assumption, not controller performance.',
        production_dependencies_modified=False,physics_steps_executed=0,confirmation_admitted=False)
    out=a.out.resolve()
    if not out.is_relative_to(ROOT/'review') or out.exists():raise ValueError('Fresh own review output required')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['semantic_recheck_passed'],checks=len(checks),failed=result['failed'],path=str(out),sha256=sha(out)),ensure_ascii=False))
if __name__=='__main__':main()

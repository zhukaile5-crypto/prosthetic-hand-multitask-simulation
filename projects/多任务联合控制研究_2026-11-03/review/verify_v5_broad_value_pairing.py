"""Supplement only: compare saved preparation values independently of Unicode padding.

The primary raw audit is retained, including its strict optional-forecast flags.
No plant, controller, producer, evaluator, protocol or saved run is changed.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'output/development_v5_physical_all16'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def value_digest(a):
    a=np.asarray(a)
    # NPZ Unicode field width is selected from the entire 16s run. Later
    # different trajectories can change padding without changing preparation.
    if a.dtype.kind in 'US':
        body=json.dumps(a.tolist(),ensure_ascii=False,separators=(',',':')).encode('utf-8')
        dtype='text_values'
    else:
        body=np.ascontiguousarray(a).tobytes();dtype=str(a.dtype)
    return hashlib.sha256(dtype.encode()+str(a.shape).encode()+body).hexdigest()

def main():
    source=sorted((ROOT/'review').glob('independent_development_v5_broad_*.json'))[-1]
    original=read(source);rows=read(FOLDER/'trials.json');manifest=read(FOLDER/'manifest.json')
    checks={};groups=defaultdict(dict);exceptions=[];storage=[]
    def check(k,v):checks[k]=bool(v)
    check('original_exact_trials_identity',original['trials_sha256']==sha(FOLDER/'trials.json')=='f3f3126e065e1edd9aed1e0d9c017a09d11a34ba38a3a96caf2b9f35d4d22e46')
    check('all_original_160_raw_records_retained',len(original['records'])==len(rows)==len(manifest['jobs']))
    allowed=('/all1501_common_preparation_values_equal','/autonomous_unavailable_force_predictions_are_NaN','/passive_retained_reference_explicitly_not_forecast')
    check('no_other_primary_audit_failed_checks',all(any(k.endswith(x) for x in allowed) for k in original['failed']))
    fields=['qpos','qvel','ctrl_before','alpha_before','normal','skin_normal','observation_json','raw_patches_json','task_request_json']
    for row in rows:
        key=(row['scenario'],row['seed']);path=Path(row['trace_path'])
        values={};widths={}
        with np.load(path,allow_pickle=False) as a:
            time=a['time'];prefix=time<=7.5+1e-9
            for field in fields:
                value=a[field];values[field]=value_digest(value[prefix]);widths[field]=str(value.dtype)
            # Explicitly unavailable legacy cache is never a timestamped forecast.
            available=a['prediction_available'];prediction=a['predicted_normal'];at=a['prediction_at'];obs=a['observation_unavailable'];debug=a['debug_json']
            finite=(~available)&np.isfinite(prediction).any(1)
            if row['method']=='passive':
                exceptional=finite&np.array([json.loads(str(v)).get('passive_hold') is not True for v in debug])
                check(str((key,row['method']))+'/passive_normal_reference_not_forecast',all(json.loads(str(debug[i])).get('prediction_available') is False for i in np.flatnonzero(finite&~exceptional)))
            else:exceptional=finite
            ids=np.flatnonzero(exceptional)
            check(str((key,row['method']))+'/unavailable_no_false_timestamp',np.isnan(at[~available]).all())
            check(str((key,row['method']))+'/exception_only_public_observation_unavailable_cache',not len(ids) or row['method'] in ['ratio','ratio_release','passive'] and obs[ids].all() and all(i>0 and np.array_equal(prediction[i],prediction[i-1]) for i in ids))
            if len(ids):exceptions.append(dict(scenario=row['scenario'],seed=row['seed'],method=row['method'],samples=len(ids),first=float(time[ids[0]]),last=float(time[ids[-1]]),interpretation='Unavailable finite legacy cached value; no prediction timestamp. Exclude from forecast-error analyses. This is not a model-infeasibility or physical-failure certificate.'))
        groups[key][row['method']]=values;storage.append(dict(scenario=row['scenario'],seed=row['seed'],method=row['method'],storage_dtypes=widths))
    for key,methods in groups.items():
        ref=methods['ratio']
        for method,values in methods.items():
            for field,value in values.items():check(str((key,method))+'/preparation_value_equal/'+field,value==ref[field])
        print(key,'preparation_values_verified',flush=True)
    foundation=Path(manifest['foundation']);admission=read(foundation/'review/admission_manifest.json')
    check('actual_foundation_admission_identity',sha(foundation/'review/admission_manifest.json')==manifest['foundation_admission_sha256'])
    check('physical_draw_implementation_identity_in_admission',sha(foundation/'session.py')==admission['source_sha256']['session.py'])
    result=dict(schema='independent_v5_broad_preparation_value_and_optional_prediction_supplement_v1',passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],created_utc=datetime.now(timezone.utc).isoformat(),original_audit=str(source),original_audit_sha256=sha(source),original_audit_passed=original['passed'],original_failed=original['failed'],trials_sha256=sha(FOLDER/'trials.json'),reviewer_sha256=sha(__file__),legacy_unavailable_cache_exceptions=exceptions,storage_dtypes=storage,physics_steps_executed=0,interpretation='All original raw/metrics/source/RNG checks remain preserved. This supplement compares actual preparation values rather than entire-run Unicode padding; it does not erase strict optional-prediction logging findings, change metrics, admit a model or establish algorithm superiority.')
    out=ROOT/'review'/('independent_v5_broad_value_pairing_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],path=str(out),sha256=sha(out),cache_exceptions=exceptions),ensure_ascii=False),flush=True)

if __name__=='__main__':main()

"""No-physics final lineage/numeric equality/prefix-API checks over saved fits."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('causal_api',HERE/'causal_forecast.py')
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(root,old_root=None):
    ledger=json.loads((root/'execution_complete.json').read_text(encoding='utf-8'));assert ledger['complete'] and len(ledger['contexts'])==15
    checks=[];rows=[]
    checks.append('15_contexts_complete')
    assert sha(HERE/'identify.py')==json.loads((root/'analysis_definition.json').read_text(encoding='utf-8'))['execution_source_sha256']==sha(root/'identify_source.py')
    checks.append('execution_source_exact_snapshot')
    for entry in ledger['contexts']:
        model_path=Path(entry['model_path']);assert sha(model_path)==entry['model_sha256']
        m=json.loads(model_path.read_text(encoding='utf-8'));assert m['global_model_admitted'] is False
        assert sha(m['dense_kernel_npz_path'])==m['dense_kernel_npz_sha256']
        assert sha(m['source_manifest_path'])==m['source_manifest_sha256']
        with np.load(m['dense_kernel_npz_path'],allow_pickle=False) as f:a={k:f[k] for k in f.files}
        assert np.isfinite(a['response25x8']).all();assert a['response25x8'].shape==(101,25,8)
        assert np.array_equal(a['response25x8'][0],np.zeros((25,8)))
        equal=None
        if old_root:
            old_model=old_root/m['context_id']/'model.json';om=json.loads(old_model.read_text(encoding='utf-8'))
            assert m['validation']==om['validation']
            with np.load(om['dense_kernel_npz_path'],allow_pickle=False) as f:
                assert all(np.array_equal(a[k],f[k]) for k in f.files)
            equal=True
        v=m['validation'];rows.append(dict(context_id=m['context_id'],known_plus8=m['known_plus8'],known_minus8=m['known_minus8'],
            zero_replay_exact=entry['zero_replay_exact'],executed_branches=entry['executed_branches'],unavailable_branches=entry['unavailable_branches'],
            numerical_arrays_and_all_error_statistics_equal_prior=equal,
            causal={h:dict(angular_rad=z['angular_rad'],lift_m=z['lift_m']) for h,z in v['causal_by_horizon'].items()},
            paired_zero_differential=dict(angular_rad=v['paired_zero_differential']['angular_rad'],lift_m=v['paired_zero_differential']['lift_m'])))
    checks.extend(['all_model_npz_manifest_hashes','all_25x8_finite_origin_zero','all_global_admitted_false'])
    if old_root:checks.append('all15_denseK_diagnostic_arrays_and_validation_statistics_bit_identical_to_v1')
    # Exercise production-prefix API on all origins/horizons of all8 mixed cp00
    # sequences. This API has no zero future argument and receives q prefix only.
    case=root/'capsule/checkpoint_00';m=json.loads((case/'model.json').read_text(encoding='utf-8'))
    with np.load(m['dense_kernel_npz_path'],allow_pickle=False) as f:a={k:f[k] for k in f.files}
    total=0;max_diff=0.
    for sequence in range(8):
        name=f'mixed_{sequence:02d}';q=a[name+'_actual_q25'];u=a[name+'_actual_reference8'];t=np.arange(101)*.005
        for h in (.02,.04,.06,.1):
            prefix=name+'_h'+str(round(h*1000));origins=a[prefix+'_origin_times_s']
            for j,origin in enumerate(origins):
                k=round(origin/.005);i=k//4
                forecast=api.forecast_from_prefix(a['response25x8'],t[:k+1],q[:k+1],t[::4][:i+1],u[::4][:i+1],t[::4][i:],u[::4][i:],horizons_s=[h])
                d=float(np.max(abs(forecast['predicted_robot_q25'][0]-a[prefix+'_prediction25'][j])))
                max_diff=max(max_diff,d);assert d<1e-12;total+=1
    checks.append('causal_prefix_API_matches_all_cp00_mixed_saved_forecasts')
    return dict(passed=True,new_physics=False,checks=checks,prefix_forecast_comparisons=total,prefix_API_max_abs_difference=max_diff,
        actual_branches=sum(x['executed_branches'] for x in rows),unavailable_branches=sum(len(x['unavailable_branches']) for x in rows),
        context_rows=rows,global_model_admitted=False,
        script_sha256=sha(__file__),identify_sha256=sha(HERE/'identify.py'),causal_api_sha256=sha(HERE/'causal_forecast.py'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old-root',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=verify(args.root,args.old_root);text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    with args.out.open('x',encoding='utf-8') as f:f.write(text)
    print(json.dumps({k:v for k,v in result.items() if k!='context_rows'},indent=2))

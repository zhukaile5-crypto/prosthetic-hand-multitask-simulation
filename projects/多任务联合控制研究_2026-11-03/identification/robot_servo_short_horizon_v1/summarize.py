"""Pool saved per-axis errors with exact availability masks; no model choice."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('servo_summary_utils',ROOT/'identification/robot_servo_response_v1/identify.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def summarize(root):
    ledger=m.read_json(root/'execution_complete.json')
    if not ledger['complete'] or len(ledger['contexts'])!=15 or not ledger['dependencies_unchanged']:raise ValueError('Complete stable15 contexts required')
    pool={str(h):{label:[] for label in ('A','B','full_reference','common_A','common_B','common_full_reference')} for h in (.02,.04,.06,.1)}
    context_rows=[];source_models=[]
    for row in ledger['contexts']:
        path=Path(row['path'])
        if m.sha(path)!=row['sha256']:raise ValueError('Context source mismatch')
        result=m.read_json(path)
        if m.sha(result['predictions_npz_path'])!=result['predictions_npz_sha256']:raise ValueError('Prediction NPZ mismatch')
        source_models.append(dict(context_id=row['context_id'],path=str(path),sha256=row['sha256']))
        context_rows.append(dict(context_id=row['context_id'],metrics_by_horizon=result['metrics_by_horizon'],unavailable=result['unavailable']))
        with np.load(result['predictions_npz_path'],allow_pickle=False) as f:a={key:f[key] for key in f.files}
        for sequence in range(8):
            name=f'mixed_{sequence:02d}'
            for h in (.02,.04,.06,.1):
                p=name+'_h'+str(round(h*1000));target=a[p+'_actual25'];valid=a[p+'_B_available']
                ea=target-a[p+'_A_prediction25'];eb=target[valid]-a[p+'_B_prediction25'][valid];er=target-a[p+'_full_reference_prediction25']
                for label,value in dict(A=ea,B=eb,full_reference=er,common_A=ea[valid],common_B=eb,common_full_reference=er[valid]).items():pool[str(h)][label].extend(value)
    aggregate={h:{label:m.score(errors) for label,errors in group.items()} for h,group in pool.items()}
    worst={h:{} for h in pool}
    for h in pool:
        for label in pool[h]:
            worst[h][label]={}
            for unit,stat in (('angular_rad','max_axis_rmse'),('angular_rad','max_abs'),('lift_m','rmse'),('lift_m','max_abs')):
                top=max(context_rows,key=lambda r:r['metrics_by_horizon'][h][label][unit][stat])
                worst[h][label][unit+'_'+stat]=dict(context_id=top['context_id'],value=top['metrics_by_horizon'][h][label][unit][stat])
    return dict(schema='robot_servo_short_horizon_summary_v1',complete=True,new_physics=False,global_model_admitted=False,
        source_ledger_path=str(root/'execution_complete.json'),source_ledger_sha256=m.sha(root/'execution_complete.json'),context_sources=source_models,
        model_selection_data_previously_seen=True,aggregate_by_horizon=aggregate,worst_contexts_by_horizon=worst,context_rows=context_rows,
        denominator_disclosure='A/reference20ms=3000; B/common20ms=2880 because eachof120branches lacks2pointOLS at firstorigin. Allfirstorigin failures retained; compare common subsets separately.')


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=summarize(args.root);m.write_json(args.out,result)
    concise={h:{label:dict(n=v['samples'],angular_rad=v['angular_rad'],lift_m=v['lift_m']) for label,v in row.items()} for h,row in result['aggregate_by_horizon'].items()}
    print(json.dumps(concise,ensure_ascii=False,indent=2))

"""Cross-check final aggregates directly against the 80 saved trial results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def mean(values):
    return math.fsum(values)/len(values) if values else None


def close(a,b):
    return a==b if a is None or b is None else math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out);stats=out/'statistics';plan=read(out/'plan.json')
    computed=read(stats/'statistics_summary.json');checks=[]
    raw={}
    for job in plan['jobs']:
        directory=out/'runs'/job['id'];r=read(directory/'result.json');extra=read(directory/'repeated_evaluation.json')
        assert r['seed']==job['seed']
        assert r['arm']==job['arm']
        row=dict(terminal_success=extra['terminal_qualified_completion'],stage_success=r['full_task_completed'],
            pickup_success=r['pickup_completed'],retained_to_release=r['retained_to_release'],
            force_limit_pass=r['force_limit_pass'],safe_terminal_success=extra['safe_terminal_completion'],
            steady_placement_attained=r['steady_placement_time_s'] is not None)
        for key in ('pickup_stable_time_s','release_latency_s','force_exceedance_duration_s','maximum_force_excess_N',
                    'force_excess_impulse_N_s','peak_total_external_finger_force_N'):
            row[key]=r[key]
        row['pickup_deadline_capped_s']=r['pickup_stable_time_s'] if r['pickup_stable_time_s'] is not None else 7.5
        row['release_deadline_capped_s']=r['release_latency_s'] if r['release_latency_s'] is not None else 3.5
        row['steady_release_latency_s']=None if r['steady_placement_time_s'] is None else r['steady_placement_time_s']-12
        row['steady_release_deadline_capped_s']=3.5 if r['steady_placement_time_s'] is None else r['steady_placement_time_s']-12
        for stage, values in r['total_force'].items():
            for key in ('total_force_reversal_rate_N_s','total_force_SD_N','raw_total_mean_N','raw_total_peak_N'):
                row[f'{stage}__{key}']=values[key]
        raw[(job['seed'],job['scene'],job['arm'])]=row
    for scene,group in computed['groups'].items():
        selected={(seed,s,a):v for (seed,s,a),v in raw.items() if scene=='overall' or s==scene}
        for arm,arm_data in group['arms'].items():
            for key,entry in arm_data['metrics'].items():
                values=[r[key] for (seed,s,a),r in selected.items() if a==arm and r[key] is not None]
                assert len(values)==entry['measured_n']
                assert close(mean(values),entry['mean']), (scene,arm,key)
                checks.append(f'{scene}/{arm}/{key}')
        for key,entry in group['pairs'].items():
            pairs=[(r[key],selected[(seed,s,'macro')][key]) for (seed,s,a),r in selected.items() if a=='baseline']
            pairs=[(a,b) for a,b in pairs if a is not None and b is not None]
            assert len(pairs)==entry['paired_n']
            assert close(mean([b-a for a,b in pairs]),entry['macro_minus_baseline']), (scene,key)
            assert close(mean([a for a,b in pairs]),entry['baseline_mean'])
            assert close(mean([b for a,b in pairs]),entry['macro_mean'])
            checks.append(f'{scene}/paired/{key}')
    with (stats/'per_trial_statistics.csv').open(encoding='utf-8-sig',newline='') as stream:
        csv_rows=list(csv.DictReader(stream))
    assert len(csv_rows)==80 and len({r['id'] for r in csv_rows})==80
    assert len(list((out/'runs').glob('*/trace.npz')))==80
    assert not any(p.suffix.lower() in ('.png','.jpg','.svg','.mp4','.gif','.pdf') for p in out.rglob('*') if p.is_file())
    result=dict(passed=True,independent_mean_and_pair_checks=len(checks),csv_rows=80,trace_count=80,
        no_plots_or_videos=True,summary_sha256=hashlib.sha256((stats/'statistics_summary.json').read_bytes()).hexdigest())
    (stats/'independent_statistics_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(result)


if __name__=='__main__':main()

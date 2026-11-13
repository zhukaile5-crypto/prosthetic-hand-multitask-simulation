"""Secondary dimensionless fluctuation vectors; original evaluations unchanged."""
from pathlib import Path
import argparse,json,math,hashlib
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('directories',nargs='+');args=parser.parse_args()
    out=Path(args.out);assert not out.exists()
    records=[];keys=set()
    for directory in args.directories:
        for p in sorted(Path(directory).glob('runs/*/result.json')):
            result=json.loads(p.read_text(encoding='utf-8'));r=result['research_metrics']
            key=(result['scenario'],result['seed'],result['motion_mode'],result['method'])
            assert key not in keys;keys.add(key)
            row=dict(scenario=key[0],seed=key[1],motion_mode=key[2],method=key[3],
                     result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),windows={})
            for window in ('static_quality','dynamic_quality','entire_holding_quality'):
                q=r[window];mean=q.get('mean_actual_finger_force_N');raw=q.get('raw_200Hz')
                valid=bool(q['eligible'] and raw is not None and mean is not None and math.isfinite(mean) and mean>0)
                row['windows'][window]=dict(eligible=valid,mean_N=mean,
                    total_force_CV=float(raw['total_force_SD_N']/mean) if valid else None,
                    normalized_finger_TV_rate_per_s=float(raw['finger_TV_rate_N_s']/mean) if valid else None)
            records.append(row)
    pairs=[]
    bykey={(r['scenario'],r['seed'],r['motion_mode'],r['method']):r for r in records}
    contexts=sorted({k[:3] for k in bykey})
    for base in ('ratio_effort','fixed_effort'):
        for window in ('static_quality','dynamic_quality','entire_holding_quality'):
            for metric in ('total_force_CV','normalized_finger_TV_rate_per_s'):
                deltas=[];relative=[];better=worse=same=0
                for ctx in contexts:
                    a=bykey.get(ctx+('fusion_effort',));b=bykey.get(ctx+(base,))
                    if a is None or b is None:continue
                    a=a['windows'][window];b=b['windows'][window]
                    if not a['eligible'] or not b['eligible']:continue
                    change=a[metric]-b[metric];deltas.append(change)
                    if b[metric]>1e-12:relative.append(100*change/b[metric])
                    if change < -1e-12:better+=1
                    elif change > 1e-12:worse+=1
                    else:same+=1
                pairs.append(dict(fusion_vs=base,window=window,metric=metric,planned=len(contexts),
                    common_eligible=len(deltas),unavailable=len(contexts)-len(deltas),better=better,worse=worse,same=same,
                    absolute_difference_median=float(np.median(deltas)) if deltas else None,
                    relative_difference_percent_median=float(np.median(relative)) if relative else None))
    report=dict(scope='secondary descriptive diagnostics, not replacement of original metrics or independent confirmation',
        directories=args.directories,records=records,pairs=pairs,original_metrics_unchanged=True,utility_weights=None)
    out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(pairs,ensure_ascii=False))


if __name__=='__main__':main()

"""Post-hoc matched-event description only; no inferential test or rerun."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'output/confirmation_integration_v2/recovery_secondary_v1.json'
data=json.loads(path.read_text(encoding='utf-8'))
maps={m:{} for m in ('ratio','unified')}
for row in data['records']:
    for event in row['events']:
        key=(row['scenario'],row['shape'],row['seed'],event['event_id'])
        assert key not in maps[row['method']]
        maps[row['method']][key]=event
assert set(maps['ratio'])==set(maps['unified'])
planned=sorted(maps['ratio'])
def eligible(e):return bool(e['stable_before'] and e['exposed'] and e['status'] not in ('failed_before','invalid_record'))
joint_eligible=[k for k in planned if all(eligible(maps[m][k]) for m in maps)]
joint_unstable=[k for k in joint_eligible if all(maps[m][k]['first_instability'] is not None for m in maps)]
joint_recovered=[k for k in joint_unstable if all(maps[m][k]['recovered'] for m in maps)]
rows=[dict(key=k,baseline_latency_from_instability=maps['ratio'][k]['latency_from_instability'],
    candidate_latency_from_instability=maps['unified'][k]['latency_from_instability'],
    difference=maps['unified'][k]['latency_from_instability']-maps['ratio'][k]['latency_from_instability']) for k in joint_recovered]
result=dict(role='Post-hoc descriptive matched-event diagnostic; no significance test and no replacement for registered secondary status denominators.',
    created_utc=datetime.now(timezone.utc).isoformat(),planned_event_pairs=len(planned),joint_eligible_pairs=len(joint_eligible),
    joint_eligible_unstable_pairs=len(joint_unstable),joint_recovered_pairs=len(rows),
    joint_recovered_seed_clusters=len({k[2] for k in joint_recovered}),
    baseline_conditional_latency_median=float(np.median([r['baseline_latency_from_instability'] for r in rows])) if rows else None,
    candidate_conditional_latency_median=float(np.median([r['candidate_latency_from_instability'] for r in rows])) if rows else None,
    paired_latency_difference_median=float(np.median([r['difference'] for r in rows])) if rows else None,
    paired_latency_difference_mean=float(np.mean([r['difference'] for r in rows])) if rows else None,
    rows=rows,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    limitations=['Conditioning on both methods recovering selects an outcome-defined subset.',
        'Own-method recovered medians are not a matched performance contrast.',
        'Repeated events share seeds and trajectories; counts are not independent experiments.',
        'All planned, ineligible, never-unstable and censored event states remain in the registered full-denominator secondary result.'])
out=ROOT/'review/independent_recovery_paired_posthoc.json'
assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','limitations')},ensure_ascii=False))
print('certificate_sha256',hashlib.sha256(out.read_bytes()).hexdigest())
preserved=ROOT/'review/independent_preservation_final.json'
print('preservation_sha256',hashlib.sha256(preserved.read_bytes()).hexdigest())

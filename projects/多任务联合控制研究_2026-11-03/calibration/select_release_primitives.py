"""Choose among publicly probed primitives by identical declared gates.

This is development selection at one nominal setup, not global safety proof.
Every available morphology and rejected primitive remains in the output.
"""
from pathlib import Path
import json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'calibration/release_physics_v4_matched'
destination=ROOT/'lifecycle_library.json'
if destination.exists():raise FileExistsError('Primitive selection is immutable; issue a new library revision')
summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
if not summary['execution_complete'] or not summary['sources_unchanged']:
    raise RuntimeError('Matched public probe is incomplete or its executed source changed')
report=dict(schema='nominal_public_release_library_v1',global_safety_certified=False,
    selection_rule='Require complete task, raw pressure guard, release <=3.5s and max peak step sensitivity <=15%; among passing primitives minimize mean clearance latency.',
    context='declared public morphology; nominal 0.12kg prepared grasp; development seed610120',
    limitations=['Single nominal setup; new mass/contact/force-limit conditions need empirical validation.',
        'No hard safety guarantee outside the measured primitive domain.',
        'Morphology is declared task context, not online shape recognition.'],
    source_summary_sha256=hashlib.sha256((source/'summary.json').read_bytes()).hexdigest(),contexts={})
for shape in ('capsule','box','sphere'):
    candidates=[]
    for method in ('ratio','ratio_release'):
        rows=sorted([r for r in summary['results'] if r['shape']==shape and r['method']==method],key=lambda r:r['probe_dt'])
        if len(rows)!=2:raise RuntimeError('Both numerical steps required')
        vectors=[]
        for r in rows:
            with np.load(r['trace_path'],allow_pickle=False) as a:
                raw=a['substep_skin_normal'];final=a['skin_normal'][-1]
                vectors.append(np.r_[np.maximum(raw.max(axis=0),final),max(raw.sum(axis=1).max(),final.sum())])
        sensitivity=float(np.max(abs(vectors[0]-vectors[1])/np.maximum(1e-8,np.maximum(abs(vectors[0]),abs(vectors[1])))))
        passed=all(r['record_complete'] and r['task_success'] and r['safety']['force_protection_pass']
            and r['release_success'] and r['release_latency']<=3.5 for r in rows) and sensitivity<=.15
        candidates.append(dict(primitive='legacy_mechanical' if method=='ratio' else 'contact_clearance',
            nominal_gate_passed=passed,numerical_peak_sensitivity=sensitivity,
            mean_clearance_s=float(np.mean([r['release_latency'] for r in rows])),
            raw_finger_peak_N=max(max(r['safety']['finger_peak']) for r in rows),
            trials=[dict(path=r['trace_path'],sha256=r['trace_sha256'],dt=r['probe_dt']) for r in rows]))
    allowed=[x for x in candidates if x['nominal_gate_passed']]
    chosen=min(allowed,key=lambda x:x['mean_clearance_s']) if allowed else None
    report['contexts'][shape]=dict(selected=chosen['primitive'] if chosen else None,candidates=candidates)
if any(x['selected'] is None for x in report['contexts'].values()):raise RuntimeError('No nominal admitted primitive for a declared morphology')
destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({s:x['selected'] for s,x in report['contexts'].items()}))

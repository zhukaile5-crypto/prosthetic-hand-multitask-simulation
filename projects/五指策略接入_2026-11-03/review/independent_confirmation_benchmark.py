"""Benchmark independent review arithmetic on six development records only."""
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone
import hashlib,io,json,runpy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
r=runpy.run_path(str(ROOT/'review/independent_confirmation_review.py'))
rows=json.loads((ROOT/'output/development_integration_v1/trials.json').read_text(encoding='utf-8'))
timings=[]
for row in rows:
    started=perf_counter();path=Path(row['trace_path']);blob=path.read_bytes()
    with np.load(io.BytesIO(blob),allow_pickle=False) as a:z={k:a[k] for k in a.files}
    m=json.loads(path.with_name('metadata.json').read_text(encoding='utf-8'))
    obs=[json.loads(x) for x in z['observation_json']]
    actual=r['derive'](z,m);noise=r['audit_rng'](z,m,obs)
    timings.append(dict(method=row['method'],shape=row['shape'],seconds=perf_counter()-started,
        passed=noise['passed'],snapshot_size_bytes=len(blob)))
result=dict(passed=all(t['passed'] for t in timings),created_utc=datetime.now(timezone.utc).isoformat(),
    six_development_seconds=sum(t['seconds'] for t in timings),
    extrapolated_256_seconds=sum(t['seconds'] for t in timings)*256/6,
    rows=timings,scope='Development only; identical independent arithmetic/noise kernels, no dynamics or confirmation data.')
out=ROOT/('review/independent_confirmation_benchmark_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))

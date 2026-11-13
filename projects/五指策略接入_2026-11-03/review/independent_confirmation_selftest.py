"""Test independent saved-data arithmetic on development only, never dynamics."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,runpy
import numpy as np

root=Path(__file__).resolve().parents[1]
reviewer=runpy.run_path(str(root/'review/independent_confirmation_review.py'))
rows=json.loads((root/'output/development_integration_v1/trials.json').read_text(encoding='utf-8'))
checks={}
for row in rows:
    path=Path(row['trace_path'])
    m=json.loads(path.with_name('metadata.json').read_text(encoding='utf-8'))
    with np.load(path,allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
    values=reviewer['derive'](z,m)
    name=f"{row['shape']}/{row['method']}"
    checks[name+'/truth']=all(reviewer['compare_value'](v,row.get(k)) for k,v in values.items() if k!='safety')
    checks[name+'/safety']=all(reviewer['compare_value'](v,row['safety'][k]) for k,v in values['safety'].items() if k!='stages')
    checks[name+'/stages']=values['safety']['stages']==row['safety']['stages']
seeds=[530101,530102,530103]
generator=np.random.default_rng(20261103)
reference=np.asarray([generator.choice(seeds,len(seeds),replace=True) for _ in range(5000)])
vectorized=np.random.default_rng(20261103).integers(0,len(seeds),size=(5000,len(seeds)))
checks['cluster_draws_match_sequential_choice']=np.array_equal(reference,np.asarray(seeds)[vectorized])
values=[np.asarray([.1,.3]),np.asarray([-.2]),np.asarray([.4,.1,.2])]
sums=np.asarray([x.sum() for x in values]);counts=np.asarray([len(x) for x in values])
fast=sums[vectorized].sum(axis=1)/counts[vectorized].sum(axis=1)
slow=np.asarray([np.concatenate([values[i] for i in row]).mean() for row in vectorized])
checks['unbalanced_cluster_bootstrap_mean']=np.allclose(fast,slow,atol=1e-14,rtol=0.)
result=dict(passed=all(checks.values()),checks=checks,checks_count=len(checks),
    failed=[k for k,v in checks.items() if not v],created_utc=datetime.now(timezone.utc).isoformat(),
    reviewer_sha256=hashlib.sha256((root/'review/independent_confirmation_review.py').read_bytes()).hexdigest(),
    scope='All six development traces only. No confirmation data accessed or strategy/physics/evaluator imported.')
out=root/('review/independent_confirmation_reviewer_selftest_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))

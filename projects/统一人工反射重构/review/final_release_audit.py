"""Raw-trace release exposure audit, independent of evaluation helpers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1]

def confirm(t,mask,start,end,dwell):
    began=None
    for time,ok in zip(t,mask):
        if time<start-1e-10 or time>end+1e-10 or not ok:began=None;continue
        if began is None:began=time
        if time-began>=dwell-1e-10:return float(time)
    return None

def inspect(job):
    r,folder,cfg=job;bad=[]
    with np.load(folder/r['trace_file']) as z:
        meta=json.loads(str(z['metadata']));t=z['time'];normal=z['normal'];force=z['all_hand_object'];s=meta['scenario']
        support=(normal[:,0]>cfg['contact_min'])&((normal>cfg['contact_min']).sum(axis=1)>=cfg['minimum_contacts'])
        reference=meta['reference'];motion=(abs(z['qpos'][:,0]-.160)<=cfg['hand_motion_tolerance'])&(abs(z['qpos'][:,1])<=cfg['wrist_motion_tolerance'])
        attained=motion&(z['position'][:,2]>=reference['absolute_z']-cfg['lift_tolerance'])&(z['support']<=cfg['table_support_max'])
        established=confirm(t,support&attained,0.,cfg['establishment_deadline'],cfg['establishment_dwell'])
        if established is not None and r['registered_failure_at'] is not None and established>=r['registered_failure_at']:established=None
        point=float(s['release_at']);index=int(np.flatnonzero(t>=point-1e-8)[0])
        exposed=bool(established is not None and established<=point and support[index] and force[index]>cfg['release_force_max'])
        clearance=confirm(t,np.isfinite(force)&(force<=cfg['release_force_max']),point,min(t[-1],point+cfg['release_deadline']),cfg['release_dwell'])
        outcome='already_contact_free' if force[index]<=cfg['release_force_max'] else 'unsupported_at_command' if not exposed else 'released' if clearance is not None else 'deadline_failed'
        if exposed!=r['release_exposed_support']:bad.append('exposure')
        if outcome!=r['release_outcome']:bad.append('outcome')
        if (clearance is not None)!=r['release_physical_clearance_success']:bad.append('clearance')
        if (outcome=='released')!=r['release_success']:bad.append('release_success')
        if clearance is not None and abs(clearance-point-r['release_physical_clearance_time'])>1e-9:bad.append('clearance_time')
        if established is None or r['established_at'] is None:
            if established!=r['established_at']:bad.append('establishment')
        elif abs(established-r['established_at'])>1e-9:bad.append('establishment_time')
        return dict(key=[r['split'],r['method'],r['seed']],passed=not bad,failures=bad,exposed=exposed,
            prior_task_failure=r['failure_at'] is not None and r['failure_at']<point,
            outcome=outcome,clearance_delay=clearance-point if clearance is not None else None)

if __name__=='__main__':
    cfg=json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))['evaluation'];jobs=[]
    for folder in [ROOT/'output/confirmation_test',ROOT/'output/confirmation_robustness']:
        records=json.loads((folder/'trials.json').read_text(encoding='utf-8'))
        jobs.extend((r,folder,cfg) for r in records if r['release_required'])
    with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(inspect,jobs))
    out=dict(passed=all(r['passed'] for r in results),traces=len(results),
        prior_task_failures=sum(r['prior_task_failure'] for r in results),
        prior_failure_and_exposed=sum(r['prior_task_failure'] and r['exposed'] for r in results),
        outcomes=dict(Counter(r['outcome'] for r in results)),checks=results,
        scope='all scheduled release traces; raw target-hand force and support at command, no helper or simulation rerun')
    (ROOT/'review/final_release_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='checks'},ensure_ascii=False))

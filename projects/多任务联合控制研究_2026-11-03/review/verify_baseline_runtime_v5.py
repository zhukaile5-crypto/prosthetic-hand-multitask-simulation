"""Exact regression against seen frozen full-16s baseline records.

This repeats old seeds for compatibility only. It is not new performance or
confirmation evidence and cannot be pooled with development/holdout trials.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from research_imports import FOUNDATION,source_identity
from runtime import ResearchSession


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=(ROOT/a.out).resolve()
    if not out.is_relative_to(ROOT) or out.exists():raise ValueError('Fresh research-only output required')
    identity=source_identity(['ratio','unified']);out.mkdir(parents=True)
    checks={};details=[];began=time.perf_counter()
    fields=('time','qpos','qvel','normal','skin_normal','desired','requested_action','executed_action',
        'observed_normal','estimated_normal','executed_motion_reference','transport_paused',
        'substep_time','substep_normal','substep_skin_normal','substep_all_hand_object')
    for scene in ('nominal','dynamic_transport'):
        for method in ('ratio','unified'):
            original=FOUNDATION/'output/confirmation_integration_v2/runs'/f'{scene}__capsule__{method}__530101'/'trace.npz'
            old_hash=hashlib.sha256(original.read_bytes()).hexdigest()
            s=ResearchSession(method=method,scenario=scene,seed=530101,split='development')
            while s.running:s.step_one_sensor_tick()
            run=out/f'{scene}__{method}';s.save(run)
            with np.load(original,allow_pickle=False) as x,np.load(run/'trace.npz',allow_pickle=False) as y:
                for f in fields:checks[f'{scene}/{method}/{f}']=bool(np.array_equal(x[f],y[f],equal_nan=True))
                checks[f'{scene}/{method}/public_contract_every_tick']='task_request_json' in y and len(y['task_request_json'])==3201
            checks[f'{scene}/{method}/original_unchanged']=hashlib.sha256(original.read_bytes()).hexdigest()==old_hash
            details.append(dict(scene=scene,method=method,seed=530101,old_trace_sha256=old_hash,
                purpose='seen-seed exact regression only',new_trace=str(run/'trace.npz')))
            print(json.dumps(dict(completed=len(details),total=4,scene=scene,method=method,
                equal=all(v for k,v in checks.items() if k.startswith(f'{scene}/{method}/')))),flush=True)
    checks['executed_sources_unchanged']=identity==source_identity(['ratio','unified'])
    result=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
        source_sha256=identity,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trials=details,wall_seconds=time.perf_counter()-began,
        scope='Four full-duration repetitions of old seen baseline records; no novel policy/physics performance claim')
    (out/'certificate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],path=str(out/'certificate.json')),ensure_ascii=False),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()

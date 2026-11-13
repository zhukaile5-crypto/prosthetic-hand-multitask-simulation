"""Numerical regression of inherited baseline behavior, no new policy claims."""
import argparse,json,time
from pathlib import Path
import numpy as np
import research_imports
from research_imports import ROOT
from runtime import ResearchSession
from session import TrialSession, check_admission


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    output=(ROOT/a.out).resolve()
    if not output.is_relative_to(ROOT) or output.exists():raise ValueError('Fresh independent evidence required')
    checks={};began=time.perf_counter()
    checks['foundation_admitted']=check_admission()['passed']
    for method in ('ratio','unified'):
        original=TrialSession(method=method,scenario='nominal',seed=610199,split='development')
        migrated=ResearchSession(method=method,scenario='nominal',seed=610199,split='development')
        for _ in range(1601):
            original.step_one_sensor_tick();migrated.step_one_sensor_tick()
        fields=('qpos','qvel','normal','skin_normal','requested_action','executed_action',
            'observed_normal','estimated_normal','executed_motion_reference','transport_paused')
        for field in fields:
            x=np.asarray([r[field] for r in original.trace]);y=np.asarray([r[field] for r in migrated.trace])
            checks[f'{method}_{field}_exact']=bool(np.array_equal(x,y,equal_nan=True))
        checks[f'{method}_handoff']=bool(original.handoff_passed and migrated.handoff_passed)
    result=dict(passed=all(checks.values()),checks=checks,wall_seconds=time.perf_counter()-began,
        scope='Two frozen baselines at one nominal seed, preparation and first 0.5s policy; no full-domain claim')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()

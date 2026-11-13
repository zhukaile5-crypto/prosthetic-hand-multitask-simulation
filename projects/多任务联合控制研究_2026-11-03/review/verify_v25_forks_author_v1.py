"""Actual zero-step fork author gate; no remaining-episode bridge claim."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(out):
    if out.exists():raise FileExistsError(out)
    fork_path=ROOT/'v25_response_identification/forks.py';fork_pin=sha(fork_path);checker_pin=sha(__file__)
    from v24_canonical_task_fusion.runtime import ReleaseSession,source_identity
    from v22_joint_task_fusion.numerics import numerical_identity
    from v25_response_identification.forks import fork_session
    from v13_a1.codec import Reference21
    parent=json.loads((ROOT/'review/v24_small_plan_v1.json').read_text(encoding='utf-8'))
    pins=source_identity(['ratio_effort']);num=numerical_identity();assert pins==parent['source_sha256'] and num==parent['numerical_identity']
    s=ReleaseSession(method='ratio_effort',scenario='nominal',seed=619741,split='robustness',motion_mode='transport',task_policy='integrated',receiver_policy='coordinated',release_variant='canonical_joint')
    c,evidence=fork_session(s);checks=list(evidence['checks']);issues=[]
    def ck(name,passed):
        checks.append(dict(name=name,passed=bool(passed)))
        if not passed:issues.append(name)
    before=s.plant.data.ctrl.copy();writes=s.plant.actor21.write_count
    reference=c.plant.reference_snapshot();c.plant.actor21.apply(c.plant.data,Reference21(reference.values,reference.time_s))
    ck('sole_actor_child_only',s.plant.data.ctrl.tobytes()==before.tobytes() and s.plant.actor21.write_count==writes and c.plant.actor21.write_count==writes+1)
    flags=[]
    for array in (c.plant.actor21.indices21,c.release_adapter.actor.indices21):flags.append(not array.flags.writeable)
    ck('declared_readonly_preserved',all(flags))
    c.tick=1
    try:fork_session(c);ck('wrong_tick_rejected',False)
    except ValueError:ck('wrong_tick_rejected',True)
    c.tick=0;c.sensors._last_time=0.
    try:fork_session(c);ck('postobserve_boundary_rejected',False)
    except ValueError:ck('postobserve_boundary_rejected',True)
    ck('original_zero_unobserved_unchanged',s.tick==0 and s.time==0 and s.sensors.samples==0 and s.sensors._last_time is None)
    ck('parent192_numerical_unchanged',source_identity(['ratio_effort'])==pins and numerical_identity()==num)
    ck('final_fork_and_checker_unchanged_during_gate',sha(fork_path)==fork_pin and sha(__file__)==checker_pin)
    result=dict(schema='v25_forks_author_zero_v1',passed=not issues,checks=checks,issues=issues,
        fork_evidence=evidence,production_source_sha256={'v25_response_identification/forks.py':fork_pin},
        parent192_source_sha256=pins,numerical_identity=num,checker_sha256=sha(__file__),physics_steps=0,
        scope='Actual parent192 session zero-step production fork/alias/RNG/readonly/actor/clock capability only; not198 runtime gate or full16 continuation bridge')
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(out=str(out),passed=result['passed'],checks=len(checks),issues=issues),ensure_ascii=False))
    return 0 if result['passed'] else 1

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();raise SystemExit(run(ROOT/args.out))

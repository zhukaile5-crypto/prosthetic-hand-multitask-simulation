"""Compare the actual common run kernel with/without interleaved GL rendering.

This verification uses isolated demonstration seeds, never confirmation seeds,
and never starts an HTTP service. Rendering and camera/cache reads must leave
the physical, sensor and controller state unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np

from session import TrialSession,ROOT,sha


def state_signature(session):
    data=session.plant.data
    physical=[getattr(data,k).copy() for k in ('qpos','qvel','qacc','act','ctrl','qfrc_applied','xfrc_applied')]
    value=(float(data.time),physical,session.plant.alpha.copy(),session.tick,
           session.sensors.__dict__,session.controller.__dict__)
    return hashlib.sha256(pickle.dumps(value,protocol=5)).hexdigest()


def run(method,rendered,out,seed):
    s=TrialSession(method=method,shape='capsule',scenario='nominal',seed=seed,
                   mode='scheduled',split='demonstration')
    renderer=None;render_checks=[];frames=0;started=time.perf_counter()
    if rendered:
        import mujoco
        renderer=mujoco.Renderer(s.plant.model,height=240,width=320)
    try:
        while s.running:
            s.step_one_sensor_tick()
            # 200Hz integer ticks approximating 12Hz display cadence. No skipped
            # control/physics samples, and camera actions do not enter a policy.
            if renderer is not None and (s.tick%17==0 or not s.running):
                s.camera.update(azimuth=130.+(frames%5)*8.,elevation=-12.-frames%3,distance=.68)
                before=state_signature(s);snap=s.snapshot();pixels=s.render(renderer);snap2=s.snapshot()
                render_checks.append(before==state_signature(s) and snap['time']==snap2['time']==s.time and pixels.shape==(240,320,3))
                if frames in (0,88,141):
                    from PIL import Image
                    Image.fromarray(pixels.copy()).save(out/f'{method}_frame_{frames:03d}.png')
                frames+=1
        folder=out/(method+'__'+('rendered' if rendered else 'headless'))
        meta=s.save(folder)
    finally:
        if renderer is not None:renderer.close()
    return dict(path=str(folder/'trace.npz'),frames=frames,render_state_checks=render_checks,
                wall_s=time.perf_counter()-started,trace_sha256=meta['trace_sha256'])


def compare(a,b):
    checks=[]
    excluded={'metadata','control_ms'}
    with np.load(a['path'],allow_pickle=False) as x,np.load(b['path'],allow_pickle=False) as y:
        for key in x.files:
            if key in excluded:continue
            u,v=x[key],y[key]
            if key=='debug_json':
                # The optimizer reports elapsed wall-clock time, just like the
                # separately excluded control_ms. It never enters its result.
                def normalize(text):
                    value=json.loads(str(text));value.pop('qp_seconds',None);return value
                equal=all(normalize(a)==normalize(b) for a,b in zip(u,v))
            else:
                equal=np.array_equal(u,v,equal_nan=True) if u.dtype.kind not in 'US' else np.array_equal(u,v)
            checks.append(dict(field=key,passed=bool(equal),shape=list(u.shape)))
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--recheck');p.add_argument('--seed',type=int,default=540101);args=p.parse_args()
    if args.recheck:
        original=Path(args.recheck);prior=json.loads(original.read_text(encoding='utf-8'))
        target=original.with_name('determinism_recheck.json')
        if target.exists():raise FileExistsError(target)
        rows=[]
        for method in prior['methods']:
            checks=compare(method['headless'],method['rendered'])
            trace_identity=all(sha(method[k]['path'])==method[k]['trace_sha256'] for k in ('headless','rendered'))
            rows.append(dict(method=method['method'],comparisons=checks,trace_identity=trace_identity,
                render_state_checks=sum(method['rendered']['render_state_checks']),
                total_render_checks=len(method['rendered']['render_state_checks']),
                passed=trace_identity and all(c['passed'] for c in checks) and all(method['rendered']['render_state_checks'])))
        core={k:v for k,v in prior['source_sha256'].items() if k!='verify_render_determinism.py'}
        unchanged=all(sha(ROOT/k)==v for k,v in core.items())
        report=dict(schema='render_headless_determinism_review_v1',passed=unchanged and all(x['passed'] for x in rows),
            original_report_path=str(original.resolve()),original_report_sha256=sha(original),
            script_sha256=sha(__file__),original_runs_source_sha256=prior['source_sha256'],
            core_source_unchanged=unchanged,methods=rows,
            exclusions=['run UUID/metadata','control_ms wall-clock duration','debug_json.qp_seconds wall-clock duration','UI camera state'],
            reason='Initial strict raw debug-string comparison differed only in reported optimizer elapsed time. Original report and all four complete actual runs are retained; this is offline rechecking, with no new dynamics.')
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(dict(passed=report['passed'],report=str(target)),ensure_ascii=False))
        if not report['passed']:raise SystemExit(1)
        return
    if not args.out:p.error('--out or --recheck is required')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    source_before={k:sha(ROOT/k) for k in ('session.py','server.py','controller_adapter.py','sensors.py','hand_plant.py','verify_render_determinism.py')}
    rows=[]
    for method in ('ratio','unified'):
        headless=run(method,False,out,args.seed);print(method,'headless completed',flush=True)
        rendered=run(method,True,out,args.seed);print(method,'rendered completed',flush=True)
        comparisons=compare(headless,rendered)
        rows.append(dict(method=method,headless=headless,rendered=rendered,comparisons=comparisons,
                         passed=all(c['passed'] for c in comparisons) and all(rendered['render_state_checks'])))
        print(method,'deterministic',rows[-1]['passed'],flush=True)
    report=dict(schema='render_headless_determinism_v1',created_utc=datetime.now(timezone.utc).isoformat(),
                scope='same TrialSession used by GUI, with/without interleaved renderer/camera/snapshot calls; no HTTP service',
                seed=args.seed,shape='capsule',scenario='nominal',duration_s=16,methods=rows,
                source_sha256=source_before,source_unchanged=all(sha(ROOT/k)==v for k,v in source_before.items()),
                exclusions=['run UUID and metadata encapsulating it','control_ms measured wall time','debug_json.qp_seconds measured wall time','UI camera state'],
                passed=all(r['passed'] for r in rows) and all(sha(ROOT/k)==v for k,v in source_before.items()))
    (out/'determinism.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],report=str(out/'determinism.json')),ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()

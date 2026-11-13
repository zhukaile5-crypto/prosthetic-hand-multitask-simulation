"""Saved-input algebra only: never imports/constructs a plant or simulator."""
import argparse,json
import numpy as np
from .inputs import ROOT,VARIANTS,load,reference_program,exact,sha,unchanged
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    r=load();a=r['arrays'];checks=[]
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:raise AssertionError(name)
    programs={v:reference_program(r,v) for v in VARIANTS};k=r['release_index'];entry=r['entry_reference21']
    for v,(lift,fingers) in VARIANTS.items():
        q=programs[v];check(v+'/complete_finite_3201x21',q.shape==(3201,21) and np.isfinite(q).all())
        check(v+'/prefix_before12_exact',exact(q[:k],a['executed_reference21'][:k]))
        check(v+'/wrist_always_source',exact(q[:,19:],a['executed_reference21'][:,19:]))
        check(v+'/lift_factor',exact(q[k:,18],a['executed_reference21'][k:,18]) if lift else np.array_equal(q[k:,18],np.full(3201-k,entry[18])))
        check(v+'/finger_factor',exact(q[k:,:18],a['executed_reference21'][k:,:18]) if fingers else np.array_equal(q[k:,:18],np.tile(entry[:18],(3201-k,1))))
    check('both_whole_source21_exact',exact(programs['both'],a['executed_reference21']))
    check('entry_uses_incoming_not_postrelease_quarter',not exact(entry,a['executed_reference21'][k]))
    try:reference_program(r,'unregistered_best_action')
    except ValueError:check('unknown_variant_refused',True)
    else:check('unknown_variant_refused',False)
    unchanged(r)
    value=dict(schema='v13_release_factorial_pure_input_v1',passed=True,checks=checks,source_trace_sha256=r['trace_sha256'],physics_steps=0,
        scope='Factor masks and source-entry clock only; no physical or full-input bridge proof',checker_sha256=sha(__file__))
    from .run import write
    write(out,value);print(json.dumps(dict(passed=True,checks=len(checks),path=str(out),sha256=sha(out))))
if __name__=='__main__':main()

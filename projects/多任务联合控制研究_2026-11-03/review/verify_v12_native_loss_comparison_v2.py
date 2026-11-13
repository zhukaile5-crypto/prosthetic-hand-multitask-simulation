"""Pure regression for root's actual-packet numeric comparison counterexample."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v12.effort_tracker import _native_loss_not_worse

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError('Fresh evidence required')
    checks=[]
    def check(name,flag):
        if not flag:raise AssertionError(name)
        checks.append(name)
    actual=.01111028309292718;enumerated=.011110283092928679;tol=1.26309e-15
    check('old_absolute_difference_rejects_real_improvement',abs(actual-enumerated)>tol)
    check('native_better_real_packet_passes',_native_loss_not_worse(actual,enumerated,tol))
    check('identical_loss_passes',_native_loss_not_worse(enumerated,enumerated,tol))
    check('better_by_large_amount_passes',_native_loss_not_worse(enumerated/2,enumerated,tol))
    check('worse_within_roundoff_passes',_native_loss_not_worse(enumerated+.5*tol,enumerated,tol))
    check('worse_beyond_roundoff_rejected',not _native_loss_not_worse(enumerated+2*tol,enumerated,tol))
    check('nonfinite_native_loss_rejected',not _native_loss_not_worse(np.nan,enumerated,tol))
    check('invalid_negative_loss_rejected',not _native_loss_not_worse(-1.,enumerated,tol))
    original=ROOT/'review/frozen_candidates/v12_effort_tracker_author_v1.py'
    check('original88bb_source_exact_preserved',sha(original)=='88bb0cb9d44f0f8d3e75994fb7eed6c5b7537229651752ecff0a1c1983f4b577')
    result=dict(schema='v12_native_loss_comparison_regression_v2',passed=True,checks=checks,
        actual_packet_counterexample=dict(native_actual_loss=actual,enumerated_min_loss=enumerated,
            absolute_difference=abs(actual-enumerated),native_worsening=actual-enumerated,tolerance=tol,
            provenance='root actual saved7.5s public packet/controller zero-physics integration; literal numeric regression, not an independently regenerated physical sample'),
        no_goal_model_constraint_or_tolerance_change=True,new_physics=False,
        source_sha256={str(x):sha(x) for x in [ROOT/'v12/effort_tracker.py',original,Path(__file__)]})
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=True,checks=len(checks),out=str(out),sha256=sha(out))))
if __name__=='__main__':main()

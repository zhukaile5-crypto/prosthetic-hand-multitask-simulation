"""Actual unknown-return scope checks; pure public algebra, no plant/physics."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from review.verify_v7_belief import build_constant_belief,frame_fixture,verify


def check_return_scope():
    original=verify()  # Original 22 checks unchanged; original evidence untouched.
    checks={};b=build_constant_belief()
    before=np.asarray(b.input_history).copy()
    known=np.ones(8,bool);known[1]=False
    alpha=np.zeros(5);alpha[1]=.001
    frame=frame_fixture(.405,alpha=alpha)
    s=b.update(frame,[.1,0.,0.],known_history_input8=known)
    f=b.forecast()
    checks['future_grip_scope_not_replaced_by_actual_return_scope']=bool(f.known_input8[1] and not b.response.input_scope[-1][1])
    checks['scope_argument_cannot_claim_unknown_motion_columns']=bool(not b.response.input_scope[-1][5:].any())
    checks['unknown_actual_return_marks_current_posterior_acquisition_time']=bool(s.mean_time==frame.sample_time and s.mean_time<s.current_time)
    checks['unknown_actual_return_disables_predictive_safety']=bool(not f.model_usable and 'unidentified_input_changed_since_acquisition' in f.reasons)
    events=s.diagnostics['unknown_executed_increments']
    checks['actual_return_event_identifies_axis_and_signed_increment']=bool(len(events)==1 and events[0]['axes']==[1] and events[0]['actual_delta'][1]==.001)
    checks['unknown_return_not_point_subtracted_with_future_known_kernel']=bool(np.array_equal(b.response.input_effect(.405,.405),np.zeros(16)))
    checks['prior_actual_history_unchanged']=bool(np.array_equal(before,np.asarray(b.input_history)[:-1]))
    # A supported reverse step cancels the actual total delta. That cancellation
    # cannot retrospectively identify the unsupported positive return.
    frame=frame_fixture(.410,alpha=np.zeros(5),sample_time=.4)
    s=b.update(frame,[.1,0.,0.],known_history_input8=np.ones(8,bool))
    checks['net_zero_actual_delta_does_not_hide_unknown_increment']=bool(np.array_equal(b.actual_increment(.4,.41),np.zeros(8)) and s.diagnostics['unidentified_input_changed_since_acquisition'])
    checks['known_reverse_does_not_relabel_previous_return']=bool(not b.response.input_scope[-2][1] and b.response.input_scope[-1][1])
    checks['mixed_interval_response_stays_unclaimed_after_cancellation']=bool(np.array_equal(b.response.input_effect(.410,.410),np.zeros(16)))
    # Merely withholding scope on a stationary axis is not an actual unknown input.
    still=build_constant_belief();ss=still.update(frame_fixture(.405),[.1,0.,0.],known_history_input8=known)
    checks['unknown_scope_without_increment_does_not_invent_event']=bool(not ss.diagnostics['unknown_executed_increments'] and not ss.diagnostics['unidentified_input_changed_since_acquisition'])
    same=build_constant_belief();same.update(frame_fixture(.405,alpha=alpha),[.1,0.,0.],known_history_input8=known)
    try:same.update(frame_fixture(.405,alpha=alpha),[.1,0.,0.],known_history_input8=np.ones(8,bool))
    except ValueError:checks['same_time_cannot_rewrite_historical_scope']=True
    else:checks['same_time_cannot_rewrite_historical_scope']=False
    for name,bad in [('shape',np.ones(7,bool)),('non_boolean',[1]*8)]:
        clean=build_constant_belief()
        try:clean.update(frame_fixture(.405),[.1,0.,0.],known_history_input8=bad)
        except ValueError:checks['invalid_'+name+'_scope_rejected']=True
        else:checks['invalid_'+name+'_scope_rejected']=False
    if not all(checks.values()):raise AssertionError(checks)
    return dict(schema='v7_belief_actual_return_scope_validation_v1',passed=True,
        original_author_checks=len(original['checks']),original_author_passed=original['passed'],checks=checks,
        source_sha256=hashlib.sha256((ROOT/'v7/belief.py').read_bytes()).hexdigest(),
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Pure actual-history and delayed-posterior algebra; no physics, no global admission. Future calibrated input subset is distinct from already executed signed direction scope.')


if __name__=='__main__':
    target=ROOT/'review/v7_belief_return_scope_validation_v1.json'
    if target.exists():raise FileExistsError(target)
    report=check_return_scope()
    with target.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report,ensure_ascii=False,indent=2))

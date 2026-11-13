"""Pure release-request acquisition boundary checks; no model or physics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from v13_a1.codec import Reference21
from v13_joint.lifecycle import ReleaseEntry
from v15_support.receiver_sensor import ReceiverMeasurement
from v15_support.release_tasks import ReceiverReleaseTasks


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    checks=[]
    def check(name,value):checks.append(dict(name=name,passed=bool(value)))
    start=np.r_[np.zeros(18),.1,0.,0.]
    opened=np.zeros(18)
    def graph():return ReceiverReleaseTasks(ReleaseEntry(12.,Reference21(start,12.),1),opened)
    def update(g,k,load=1.,age=.01):
        now=(2400+k)*.005
        sigma=np.r_[np.full(3,.01),np.full(3,.0005)]
        measurement=ReceiverMeasurement(now,now-.01,.01,True,
            np.array([0.,0.,-load,0.,0.,0.]),load,sigma,3*sigma,None,
            'cached_after_step_end_label',None)
        state=g.update(time_s=now,current_ref21=Reference21(start,now),robot_lift_m=.1,
            normal5=np.zeros(5),valid5=np.ones(5,dtype=bool),age5=np.full(5,age),
            aux_pressure=0.,aux_valid=True,aux_age=age,receiver=measurement)
        return state,measurement
    g=graph()
    first,m0=update(g,0)
    second,m1=update(g,1)
    check('typed_pre_request_receiver_remains_valid',m0.valid and m1.valid)
    check('typed_pre_request_load_values_unmodified',m0.normal_load_N==1. and m1.normal_load_N==1.)
    check('pre_request_receiver_does_not_latch',not second['receiver_contact_latched'])
    check('pre_request_receiver_streak_stays_zero',second['receiver']['consecutive_fresh_acquisitions']==0)
    check('pre_request_receiver_task_reason_explicit',second['receiver']['reason']=='not_yet_postrequest')
    check('pre_request_receiver_valid_separate_from_eligibility',second['receiver']['measurement_valid'] and not second['receiver']['release_evidence_eligible'])
    check('pre_request_receiver_load_retained_not_zero',second['receiver']['load_N']==1.)
    check('pre_request_receiver_holds_descent',second['phase']=='UNKNOWN_HOLD')
    check('pre_request_hand_measurement_known_but_task_ineligible',second['hand']['measurement_available'] and not second['hand']['release_evidence_eligible'])
    check('pre_request_hand_never_starts_dwell',second['hand_low_since_acquisition_s'] is None and not second['hand_low_confirmed'])
    boundary,_=update(g,2)
    check('acquisition_equal_request_is_eligible',boundary['receiver']['release_evidence_eligible'])
    check('first_post_request_contact_streak_exactly_one',boundary['receiver']['consecutive_fresh_acquisitions']==1 and not boundary['receiver_contact_latched'])
    check('hand_dwell_origin_is_request_clock',boundary['hand_low_since_acquisition_s']==12.)
    after,_=update(g,3)
    check('second_post_request_contact_latches',after['receiver_contact_latched'] and after['phase']=='GUARDED_CLEAR')
    for k in range(4,42):last,_=update(g,k)
    check('pre_request_low_history_cannot_make_195ms_confirm',not last['hand_low_confirmed'])
    done,_=update(g,42)
    check('exact_post_request_200ms_low_dwell_confirms',done['hand_low_confirmed'] and abs(done['hand']['dwell_s']-.2)<1e-12)
    g=graph()
    for k in range(0,4):last,_=update(g,k,load=0.)
    check('post_request_valid_absence_can_resume_approach',last['phase']=='APPROACH' and not last['receiver_contact_latched'])
    check('absence_approach_starts_at_first_eligible_packet',last['descent_start_s']==12.01)
    g=graph()
    # Some fingers may have a post-request acquisition, but an older auxiliary
    # channel still excludes the complete low-force proxy until all are eligible.
    mixed,_=update(g,3,age=.02)
    check('oldest_hand_channel_controls_request_eligibility',not mixed['hand']['release_evidence_eligible'] and mixed['hand_low_since_acquisition_s'] is None)
    equal,_=update(g,4,age=.02)
    check('oldest_hand_equal_request_accepts_without_early_origin',equal['hand']['release_evidence_eligible'] and equal['hand_low_since_acquisition_s']>=12.)
    files=[ROOT/'v15_support/release_tasks.py',Path(__file__)]
    report=dict(schema='v15-release-postrequest-eligibility-author-v1',
        passed=all(x['passed'] for x in checks),checks=checks,check_count=len(checks),
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        model_constructions=0,physics_steps=0,
        scope='typed pure measurements; only task eligibility changes, not receiver values or physical thresholds')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[x['name'] for x in checks if not x['passed']],out=str(out)),ensure_ascii=False))
    raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__':main()

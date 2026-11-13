"""Independent finite scenario oracles for the pure release task graph.

No model/plant/solver or physics. Source is explicitly SHA-bound; archived
versions can be rerun without touching production.
"""
from pathlib import Path
from copy import deepcopy
import argparse, hashlib, importlib.util, json, sys, traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v13_a1.codec import Reference21
from v13_joint.lifecycle import ReleaseEntry, release_nominal
from v15_support.receiver_sensor import ReceiverMeasurement
START=np.r_[[-.3,.7,.1,.15,.8],np.linspace(.2,.5,13),.1,.03,-.04]
OPEN=np.r_[[.4,1.1,.2,.3,1.2],np.linspace(-.1,.1,13)]
LO=np.full(21,-10.);HI=np.full(21,10.)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def recv(t,*,load=0.,valid=True,sigma=.01,shift=0.):
    delivered=t+shift;s=np.r_[np.full(3,sigma),np.full(3,sigma*.05)]
    return ReceiverMeasurement(delivered,delivered-.01 if valid else None,.01 if valid else None,valid,
        np.array([0.,0.,-load,0.,0.,0.]) if valid else None,load if valid else None,s,3*s,None,
        'cached_after_step_end_label' if valid else None,None if valid else 'acquisition_unavailable')
def run(cls):
    checks=[]
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def close(name,a,b):check(name,np.allclose(a,b,atol=3e-13,rtol=3e-13),dict(actual=np.asarray(a).tolist(),expected=np.asarray(b).tolist()))
    def reject(name,fn):
        try:fn()
        except (ValueError,TypeError,RuntimeError):check(name,True)
        else:check(name,False)
    def graph():return cls(ReleaseEntry(12.,Reference21(START,12.),7),OPEN)
    def update(g,t,*,current=None,receiver=None,low=False,age=None,aux_valid=True,encoder=None):
        x=START.copy() if current is None else np.array(current,float)
        return g.update(time_s=t,current_ref21=Reference21(x,t),robot_lift_m=x[18] if encoder is None else encoder,
            normal5=np.zeros(5) if low else np.full(5,.3),valid5=np.ones(5,dtype=bool),
            age5=np.full(5,.01) if age is None else age,aux_pressure=0.,aux_valid=aux_valid,
            aux_age=.01 if age is None else float(np.max(age)),receiver=recv(t) if receiver is None else receiver)
    # Independent request boundary: delivery freshness alone is insufficient.
    g=graph();m0=recv(12.,load=1.);m1=recv(12.005,load=1.)
    a=update(g,12.,receiver=m0,low=True);b=update(g,12.005,receiver=m1,low=True)
    check('pre_request_receiver_acquisitions_cannot_latch_release_evidence',not b['receiver_contact_latched'] and m0.valid and m1.valid and m1.normal_load_N==1.
        and b['receiver'].get('measurement_valid') is True and b['receiver'].get('release_evidence_eligible') is False and b['receiver']['consecutive_fresh_acquisitions']==0,
        dict(first=a['receiver'],second=b['receiver'],phase=b['phase'],original_measurement_valid=m1.valid,original_load_N=m1.normal_load_N))
    check('hand_low_dwell_cannot_start_before_request',b['hand_low_since_acquisition_s'] is None and b['hand'].get('measurement_available') is True
        and b['hand'].get('release_evidence_eligible') is False and b['hand']['dwell_s'] is None,dict(low_since=b['hand_low_since_acquisition_s'],evidence=b['hand']))
    # Fingers and wrists are frozen-original even when receiver/hand are unknown.
    u=graph();update(u,12.,receiver=recv(12.,valid=False),aux_valid=False)
    times=(12.,12.4,12.8,13.4,14.2,15.)
    exact=all(np.array_equal(u.proposal(t,LO,HI).values[:18],release_nominal(u.entry,t,OPEN,LO,HI).values[:18]) for t in times)
    check('unknown_never_blocks_or_retards_frozen_finger_path',exact)
    check('unknown_never_changes_entry_wrists',all(np.array_equal(u.proposal(t,LO,HI).values[19:],START[19:]) for t in times))
    check('terminal18_is_frozen_arithmetic_endpoint_not_declared_OPEN',np.array_equal(u.release_terminal18,release_nominal(u.entry,15.,OPEN,LO,HI).values[:18]) and not np.array_equal(u.release_terminal18[:5],OPEN[:5]))
    # Two post-request acquisitions above a strict engineering gate, any actor.
    l=graph();s1=update(l,12.010,receiver=recv(12.010,load=1.));v1=l.version
    s2=update(l,12.015,receiver=recv(12.015,load=1.))
    check('one_post_request_receiver_acquisition_does_not_latch',not s1['receiver_contact_latched'] and s1['receiver']['available']
        and s1['receiver']['consecutive_fresh_acquisitions']==1 and abs(s1['receiver']['acquisition_time_s']-12.)<1e-12,dict(evidence=s1['receiver'],boundary_role='acquisition_equals_request_is_eligible'))
    check('two_adjacent_fresh_post_request_acquisitions_latch',s2['receiver_contact_latched'] and s2['phase']=='GUARDED_CLEAR')
    check('receiver_gate_has_no_target_identity_or_placement_certificate',s2['semantics']['any_actor_receiver_contact_not_target_identity'] and s2['semantics']['receiver_contact_not_placed_proof'])
    eq=graph();update(eq,12.010,receiver=recv(12.010,load=.03));se=update(eq,12.015,receiver=recv(12.015,load=.03))
    check('engineering_gate_strictly_greater_not_equal',not se['receiver_contact_latched'])
    # Approach, interruption, anchor braking, and resumption are actual-clock based.
    p=graph();update(p,12.010);v=p.version
    x=START.copy();x[18]=.1-.005*(.1/.6);update(p,12.015,current=x)
    check('normal_5ms_reference_progress_does_not_increment_version',p.version==v)
    x[18]=.07;paused=update(p,12.020,current=x,encoder=.075,receiver=recv(12.020,valid=False))
    check('unknown_receiver_brakes_from_current_ref_and_encoder',paused['phase']=='UNKNOWN_HOLD' and paused['hold_anchor_m']==.075)
    close('unknown_hold_proposal_does_not_descend_with_wall_clock',p.proposal(12.220,LO,HI).values[18],.075)
    close('unknown_hold_lift_authority_forbids_down',p.lift_interval(Reference21(x,12.020)),[.07,.075])
    hi=HI.copy();hi[18]=.05;projected=p.proposal(12.220,LO,hi);e=p.state_dict()['last_proposal']
    check('global_projection_and_mode_authority_are_honestly_separate',projected.values[18]==.05 and not e['mode_bounds_applied'] and not e['rate_bounds_applied'] and e['global_projection_changed'][18] and e['raw_graph_nominal21'][18]==.075)
    x[18]=.075;resumed=update(p,12.225,current=x)
    close('resume_from_current_anchor_no_paused_wallclock_catchup',p.proposal(12.245,LO,HI).values[18],.075-.020*(.1/.6))
    check('resume_phase_is_approach_not_fabricated_receiver_contact',resumed['phase']=='APPROACH' and not resumed['receiver_contact_latched'])
    # Low proxy uses the oldest acquisition; duplicates and missing updates reset.
    d=graph()
    for k in range(2,23):
        t=(2400+k)*.005;update(d,t,low=True,age=np.full(5,t-12.))
    ds=d.state_dict();check('repeated_oldest_packet_does_not_accumulate_low_dwell',not ds['hand_low_confirmed'] and abs(ds['hand']['dwell_s'])<1e-10)
    fresh=graph()
    for k in range(2,43):update(fresh,(2400+k)*.005,low=True)
    check('fresh_oldest_acquisition_accumulates_declared_dwell',fresh.state_dict()['hand_low_confirmed'])
    aftergap=update(fresh,12.220,low=True)
    check('missing_5ms_update_resets_low_dwell',not aftergap['hand_low_confirmed'])
    # Even after 2.2s, declared OPEN is insufficient: actual frozen terminal is required.
    w=graph();terminal=w.release_terminal18.copy()
    for k in range(2,441):
        t=(2400+k)*.005;x=START.copy();x[:18]=OPEN if k==440 else release_nominal(w.entry,t,OPEN,LO,HI).values[:18];x[18]=.02
        last=update(w,t,current=x,low=True,receiver=recv(t,valid=False))
    check('elapsed_2p2_and_low_dwell_without_actual_terminal_cannot_withdraw',last['hand_low_confirmed'] and not last['terminal_reference_matched'] and last['phase']!='WITHDRAW')
    x[:18]=terminal;started=update(w,14.205,current=x,low=True,receiver=recv(14.205,valid=False))
    check('actual_terminal_plus_dwell_starts_withdraw_without_fabricated_placement',started['phase']=='WITHDRAW' and started['withdraw_start_lift_m']==.02 and started['withdraw_goal_m']==.1)
    close('withdraw_nominal_is_upward_from_actual_reference',w.proposal(14.505,LO,HI).values[18],.06)
    close('withdraw_lift_authority_never_allows_down',w.lift_interval(Reference21(x,14.205)),[.02,.1])
    interrupted=deepcopy(w);ui=update(interrupted,14.210,current=x,low=True,aux_valid=False,receiver=recv(14.210,valid=False))
    check('hand_unknown_revokes_withdraw_and_low_confirmation',ui['phase']=='GUARDED_CLEAR' and not ui['hand_low_confirmed'])
    late=update(w,14.805,current=x,low=True);check('withdraw_wallclock_done_without_actual_goal_cannot_monitor',late['phase']=='WITHDRAW')
    x[18]=.1;done=update(w,14.810,current=x,low=True);check('actual_withdraw_goal_allows_monitor',done['phase']=='MONITOR')
    contact=update(w,14.815,current=x,low=False);check('hand_recontact_revokes_monitor_without_reclosing_fingers',contact['phase']=='GUARDED_CLEAR' and np.array_equal(w.proposal(15.,LO,HI).values[:18],terminal))
    # Exact public clock discipline and idempotency, separate from unknown sensors.
    c=graph();state=update(c,12.010);same=update(c,12.010)
    check('same_tick_identical_payload_idempotent',state==same)
    reject('same_tick_changed_payload_rejected',lambda:update(c,12.010,low=True))
    reject('public_clock_rewind_rejected',lambda:update(c,12.005))
    reject('proposal_before_latest_evidence_rejected',lambda:c.proposal(12.005,LO,HI))
    reject('authority_with_wrong_actual_clock_rejected',lambda:c.lift_interval(Reference21(START,12.005)))
    bad=graph();f=update(bad,12.010,receiver=recv(12.010,load=1.,shift=.005))
    check('future_receiver_delivery_not_consumed_as_current_evidence',not f['receiver']['available'] and not f['receiver_contact_latched'])
    return dict(passed=all(x['passed'] for x in checks),checks=checks,
        scope='Independent finite pure task-graph sequences, no physical model, optimizer, actor/writer, real sensing or safety/placement validation.',
        limitations=['Receiver is any-actor top-face measurement; latch is not target placement.','Low hand force is an oldest-acquisition sampled proxy, not geometric clearance.',
            'Global proposal projection is separate from mode/rate feasibility; future caller must intersect and actually execute authority.'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--expected-sha',required=True);ap.add_argument('--source',default='v15_support/release_tasks.py');ap.add_argument('--out',required=True);args=ap.parse_args()
    source=ROOT/args.source;out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    before=sha(source)
    if before!=args.expected_sha:raise RuntimeError('Source SHA mismatch')
    spec=importlib.util.spec_from_file_location('v15_support.release_graph_under_independent_review',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    try:spec.loader.exec_module(module);r=run(module.ReceiverReleaseTasks)
    except Exception as exc:r=dict(passed=False,error=repr(exc),traceback=traceback.format_exc())
    r.update(schema='v15_release_tasks_independent_v1',source_path=str(source),source_sha256=before,source_unchanged=sha(source)==before,
        checker_sha256=sha(__file__),physics_steps=0,model_constructions=0,mujoco_imported='mujoco' in sys.modules)
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=r['passed'],checks=len(r.get('checks',[])),failed=[c['name'] for c in r.get('checks',[]) if not c['passed']],sha256=sha(out),path=str(out)),ensure_ascii=False))
    raise SystemExit(0 if r['passed'] else 1)

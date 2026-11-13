"""Author pure task-graph checks. Never constructs a plant/model or integrates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v13_a1.codec import Reference21, EnginePort21
from v13_joint.lifecycle import ReleaseEntry, release_nominal
from v15_support.receiver_sensor import ReceiverMeasurement
from v15_support.release_tasks import ReceiverReleaseTasks, PERIOD_S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
    def reject(name, call):
        try:
            call()
        except (ValueError, TypeError, RuntimeError):
            check(name, True)
        else:
            check(name, False)

    import mujoco
    forbidden_counts = {}
    originals = {}
    for name in ('MjModel', 'MjData', 'mj_step', 'mj_step1', 'mj_step2',
            'mj_forward', 'mj_fwdPosition', 'mj_fwdVelocity', 'mj_fwdActuation',
            'mj_kinematics', 'mj_transmission', 'mj_contactForce'):
        originals[name] = getattr(mujoco, name)
        forbidden_counts[name] = 0
        def forbid(*a, _name=name, **kw):
            forbidden_counts[_name] += 1
            raise AssertionError('Forbidden model/dynamics pipeline: ' + _name)
        setattr(mujoco, name, forbid)
    try:
        start = np.r_[[-.2, .9, .1, .1, .8], np.full(13, .4), .1, .02, -.03]
        opened = np.r_[[.2, 1.15, .1, .2, 1.15], np.zeros(9), [.1, 0., 0., 0.]]
        lower = np.full(21, -10.)
        upper = np.full(21, 10.)
        def entry(lift=.1):
            values = start.copy(); values[18] = lift
            return ReleaseEntry(12., Reference21(values, 12.), 4)
        def graph(lift=.1):
            return ReceiverReleaseTasks(entry(lift), opened)
        def receiver(k, load=0., sigma=.01, valid=True, delivery_shift=0):
            t = (2400+k+delivery_shift)*PERIOD_S
            sig = np.r_[np.full(3, sigma), np.full(3, sigma*.05)]
            return ReceiverMeasurement(t, t-.01 if valid else None,
                .01 if valid else None, valid,
                np.array([0., 0., -load, 0., 0., 0.]) if valid else None,
                load if valid else None, sig, 3.*sig, None,
                'boundary_after_forward' if valid else None,
                None if valid else 'acquisition_unavailable')
        def update(g, k, values=None, *, load=0., sigma=.01, receiver_valid=True,
                normal=None, valid=None, age=None, aux=0., aux_valid=True,
                aux_age=.01, encoder=None, recv=None):
            t = (2400+k)*PERIOD_S
            cur = start.copy() if values is None else np.array(values, float)
            return g.update(time_s=t, current_ref21=Reference21(cur,t),
                robot_lift_m=cur[18] if encoder is None else encoder,
                normal5=np.full(5,.3) if normal is None else normal,
                valid5=np.ones(5,bool) if valid is None else valid,
                age5=np.full(5,.01) if age is None else age,
                aux_pressure=aux, aux_valid=aux_valid, aux_age=aux_age,
                receiver=receiver(k,load,sigma,receiver_valid) if recv is None else recv)

        g = graph()
        reject('proposal_before_observation_unavailable', lambda:g.proposal(12.,lower,upper))
        update(g,0)
        check('pre_request_valid_absence_is_ineligible_and_holds',g.phase=='UNKNOWN_HOLD')
        check('initial_mode_version_changes',g.version==5)
        check('initial_declared_entry_preserved',np.array_equal(g.entry.reference21.values,start))
        for k in (0,1,20,80,159,160,200,279,280,300,400,440,441,600):
            t=(2400+k)*PERIOD_S
            actual=g.proposal(t,lower,upper).values
            native=release_nominal(entry(),t,opened,lower,upper).values
            check('frozen_finger_path_exact_tick_'+str(k),np.array_equal(actual[:18],native[:18]))
            check('entry_wrists_exact_tick_'+str(k),np.array_equal(actual[19:],start[19:]))
            check('pre_request_ineligible_receiver_holds_lift_tick_'+str(k),
                actual[18]==start[18])
        check('declared_open_thumb_differs_from_release_terminal',
            not np.array_equal(g.declared_open18[:5],g.release_terminal18[:5]))
        check('frozen_thumb_terminal_correct',np.allclose(g.release_terminal18[:5],[-1.,.1,0.,0.,0.],rtol=0.,atol=64*np.finfo(float).eps))
        saved=json.dumps(g.state_dict(),sort_keys=True,allow_nan=False)
        update(g,0)
        check('same_tick_idempotent_including_version_and_diagnostics',saved==json.dumps(g.state_dict(),sort_keys=True,allow_nan=False))
        reject('same_tick_changed_packet_rejected',lambda:update(g,0,load=.2))
        update(g,1)
        reject('public_clock_rewind_rejected',lambda:update(g,0))
        reject('future_executed_reference_clock_rejected',lambda:g.update(time_s=12.01,
            current_ref21=Reference21(start,12.015),robot_lift_m=.1,
            normal5=np.zeros(5),valid5=np.ones(5,bool),age5=np.zeros(5),
            aux_pressure=0.,aux_valid=True,aux_age=0.,receiver=receiver(2)))
        reject('engineport_not_unbiased_reference',lambda:g.update(time_s=12.01,
            current_ref21=EnginePort21(start,12.01,0.),robot_lift_m=.1,
            normal5=np.zeros(5),valid5=np.ones(5,bool),age5=np.zeros(5),
            aux_pressure=0.,aux_valid=True,aux_age=0.,receiver=receiver(2)))
        reject('negative_entry_lift_rejected',lambda:graph(-.01))
        reject('nongrid_entry_rejected',lambda:ReceiverReleaseTasks(
            ReleaseEntry(12.001,Reference21(start,12.001),1),opened))

        g=graph(); update(g,2); update(g,3,load=.031)
        check('one_fresh_receiver_contact_not_latched',not g.state_dict()['receiver_contact_latched'])
        version=g.version
        update(g,3,load=.031)
        check('duplicate_receiver_does_not_count_twice',not g.state_dict()['receiver_contact_latched'] and g.version==version)
        cur=start.copy();cur[18]=.083
        update(g,4,cur,load=.031,encoder=.087)
        check('two_fresh_compressions_latch',g.state_dict()['receiver_contact_latched'] and g.phase=='GUARDED_CLEAR')
        check('brake_anchor_max_reference_encoder',g.state_dict()['hold_anchor_m']==.087)
        check('guard_never_allows_further_descent',g.lift_interval(Reference21(cur,12.02))==(.083,.087))
        check('guard_proposal_holds_captured_anchor',g.proposal(12.1,lower,upper).values[18]==.087)
        v=g.version
        update(g,5,cur,load=0.,encoder=.095)
        check('receiver_latch_irreversible_when_load_disappears',g.phase=='GUARDED_CLEAR' and g.version==v)
        check('hold_anchor_not_chasing_encoder_noise',g.state_dict()['hold_anchor_m']==.087)
        update(g,6,cur,receiver_valid=False)
        check('latched_receiver_unknown_does_not_reenable_descent',g.phase=='GUARDED_CLEAR')
        g=graph();update(g,0);update(g,1,load=.03);update(g,2,load=.03)
        check('three_sigma_equality_not_contact',not g.state_dict()['receiver_contact_latched'])
        g=graph();update(g,2);update(g,3,load=.04);update(g,5,load=.04)
        check('missing_receiver_tick_breaks_two_acquisition_streak',not g.state_dict()['receiver_contact_latched'])
        update(g,6,load=.04)
        check('two_after_gap_can_latch',g.state_dict()['receiver_contact_latched'])

        g=graph();cur=start.copy();cur[18]=.08
        update(g,2);update(g,3,cur,receiver_valid=False,encoder=.085)
        check('receiver_unknown_pauses_descent',g.phase=='UNKNOWN_HOLD')
        check('unknown_never_invented_zero_load',g.state_dict()['receiver']['load_N'] is None)
        check('unknown_hold_anchor_uses_encoder',g.proposal(12.15,lower,upper).values[18]==.085)
        cur[18]=.085
        for k in range(4,40):update(g,k,cur,receiver_valid=False,encoder=.09)
        update(g,40,cur)
        check('fresh_absence_resumes_from_actual_reference',g.phase=='APPROACH' and g.state_dict()['descent_start_lift_m']==.085)
        check('resume_does_not_catch_up_wall_clock',abs(g.proposal(12.205,lower,upper).values[18]-(.085-.1/.6*.005))<1e-14)
        check('resume_uses_original_declared_speed',g.state_dict()['descent_speed_m_s']==.1/.6)
        g=graph();update(g,0,recv=receiver(0,delivery_shift=-1))
        check('receiver_delivery_clock_mismatch_unknown',g.phase=='UNKNOWN_HOLD' and g.state_dict()['receiver']['reason']=='delivery_clock_mismatch')

        g=graph()
        for k in range(2,42):update(g,k,normal=np.full(5,.02),aux=.1)
        check('dwell_195ms_not_200ms',not g.state_dict()['hand_low_confirmed'])
        update(g,42,normal=np.full(5,.02),aux=.1)
        check('dwell_exact_200ms_confirms',g.state_dict()['hand_low_confirmed'])
        check('early_low_does_not_skip_finger_trajectory',g.phase=='APPROACH')
        check('low_pressure_threshold_includes_auxiliary',g.state_dict()['hand']['load_proxy_N']==.2)
        update(g,43,normal=np.full(5,.02),aux=.10001)
        check('auxiliary_recontact_resets_low_evidence',not g.state_dict()['hand_low_confirmed'] and g.state_dict()['hand_low_since_acquisition_s'] is None)
        update(g,44,normal=np.full(5,-.1),aux=-.1)
        check('negative_noisy_pressures_only_in_proxy_clipped',g.state_dict()['hand']['load_proxy_N']==0.)
        update(g,45,normal=np.array([0.,0.,np.nan,0.,0.]))
        check('nonfinite_hand_pressure_unknown_not_zero',not g.state_dict()['hand']['available'] and g.state_dict()['hand']['load_proxy_N'] is None)
        update(g,46,valid=np.array([True,True,True,False,True]))
        check('partial_validity_does_not_confirm_clearance',not g.state_dict()['hand']['available'])
        update(g,47,age=np.full(5,.1201))
        check('stale_finger_breaks_evidence',not g.state_dict()['hand']['available'])
        update(g,48,aux_valid=False)
        check('unknown_auxiliary_breaks_evidence',not g.state_dict()['hand']['available'])
        update(g,49,aux_age=.1201)
        check('stale_auxiliary_breaks_evidence',not g.state_dict()['hand']['available'])
        g=graph()
        for k in range(2,22):
            update(g,k,normal=np.zeros(5),age=np.full(5,k*.005),aux_age=k*.005)
        check('duplicate_oldest_acquisition_never_accumulates_dwell',not g.state_dict()['hand_low_confirmed'] and abs(g.state_dict()['hand']['dwell_s'])<1e-9)
        g=graph()
        for k in range(30):update(g,k,normal=np.zeros(5))
        update(g,31,normal=np.zeros(5))
        check('missing_public_tick_resets_hand_dwell',not g.state_dict()['hand_low_confirmed'] and g.state_dict()['hand']['dwell_s']==0.)
        update(g,32,normal=np.zeros(5),age=np.full(5,.04),aux_age=.04)
        check('acquisition_rewind_unknown_resets',not g.state_dict()['hand']['available'] and not g.state_dict()['hand_low_confirmed'])

        g=graph()
        terminal=g.release_terminal18.copy()
        cur=start.copy();cur[:18]=opened;cur[18]=.015
        for k in range(400,442):update(g,k,cur,normal=np.zeros(5),load=.1)
        check('declared_open_pose_cannot_fake_release_terminal',g.phase=='GUARDED_CLEAR' and not g.state_dict()['terminal_reference_matched'])
        cur[:18]=terminal
        update(g,442,cur,normal=np.zeros(5),load=.1)
        check('terminal_and_low_enable_upward_withdraw',g.phase=='WITHDRAW')
        check('receiver_load_does_not_prevent_withdraw',g.state_dict()['receiver']['compressive_contact'])
        check('withdraw_target_entry_or_higher',g.state_dict()['withdraw_goal_m']==.1)
        check('withdraw_lift_interval_no_down',g.lift_interval(Reference21(cur,14.21))==(.015,.1))
        proposal=g.proposal(14.51,lower,upper)
        check('withdraw_half_segment_upward',abs(proposal.values[18]-.0575)<1e-14)
        check('withdraw_hand_terminal_unchanged',np.array_equal(proposal.values[:18],terminal))
        update(g,443,cur,normal=np.zeros(5),receiver_valid=False)
        check('unknown_receiver_not_unknown_hand_does_not_block_upward',g.phase=='WITHDRAW')
        cur[18]=.1
        update(g,562,cur,normal=np.zeros(5))
        check('withdraw_completed_elapsed_and_executed_goal_enters_monitor',g.phase=='MONITOR')
        update(g,563,cur,normal=np.full(5,.1))
        check('monitor_recontact_returns_guard',g.phase=='GUARDED_CLEAR' and not g.state_dict()['hand_low_confirmed'])
        check('monitor_recontact_still_always_open',np.array_equal(g.proposal(14.82,lower,upper).values[:18],terminal))
        for k in range(564,605):update(g,k,cur,normal=np.zeros(5),receiver_valid=False)
        check('fresh_low_can_restart_withdraw_without_receiver_target_proof',g.phase=='WITHDRAW')
        update(g,605,cur,normal=np.zeros(5),aux_valid=False)
        check('withdraw_unknown_hand_returns_guard',g.phase=='GUARDED_CLEAR')
        check('withdraw_unknown_resets_low_since',g.state_dict()['hand_low_since_acquisition_s'] is None)

        g=graph();cur=start.copy();cur[:18]=g.release_terminal18;cur[18]=.025
        for k in range(398,440):update(g,k,cur,normal=np.zeros(5),load=.1)
        check('fullopen_and_low_before_2point2_still_no_withdraw',g.phase=='GUARDED_CLEAR')
        update(g,440,cur,normal=np.zeros(5),load=.1)
        check('integer_tick_2point2_gate_has_no_float_late_tick',g.phase=='WITHDRAW')
        update(g,560,cur,normal=np.zeros(5),load=.1)
        check('elapsed_withdraw_without_actual_executed_goal_not_monitor',g.phase=='WITHDRAW')
        update(g,561,cur,normal=np.zeros(5),valid=np.array([True,True,False,True,True]))
        check('withdraw_partial_unknown_hand_resets_to_guard',g.phase=='GUARDED_CLEAR')
        check('post_withdraw_guard_never_resumes_blind_descent',g.state_dict()['withdraw_started_once'])
        cur[18]=.13
        for k in range(562,603):update(g,k,cur,normal=np.zeros(5),load=0.)
        check('retreat_above_entry_never_targets_down',g.phase=='WITHDRAW' and g.state_dict()['withdraw_goal_m']==.13)
        check('no_distance_withdraw_interval_is_point',g.lift_interval(Reference21(cur,(2400+602)*.005))==(.13,.13))

        g=graph();update(g,0,normal=np.full(5,np.nan),aux=None,aux_valid=False,aux_age=None)
        same=g.state_dict()
        update(g,0,normal=np.full(5,np.nan),aux=None,aux_valid=False,aux_age=None)
        check('same_unknown_payload_is_idempotent',same==g.state_dict())
        check('unknown_hand_cannot_be_confirmed_by_available_receiver',not g.state_dict()['hand_low_confirmed'])
        g=graph();update(g,0,receiver_valid=False)
        check('unknown_receiver_does_not_stop_initial_finger_opening',
            np.array_equal(g.proposal(12.1,lower,upper).values[:18],
                release_nominal(entry(),12.1,opened,lower,upper).values[:18]))
        update(g,1,receiver_valid=False)
        check('hold_phase_repeated_tick_no_version_churn',g.version==5)
        check('phase_state_and_proposal_json_are_finite_or_null',
            isinstance(json.dumps(g.state_dict(),allow_nan=False),str))

        g=graph();update(g,0)
        projected_lo=lower.copy();projected_hi=upper.copy();projected_hi[0]=-.5
        p=g.proposal(12.,projected_lo,projected_hi)
        check('only_true_global_box_projects_nominal',p.values[0]==-.5)
        diag=g.state_dict()['last_proposal']
        check('raw_nominal_saved_before_projection',diag['raw_graph_nominal21'][0]==start[0])
        check('global_conflict_explicit',diag['global_projection_changed'][0] and diag['global_projection_max_abs']>0.)
        check('mode_and_rate_bounds_not_silently_combined',not diag['mode_bounds_applied'] and not diag['rate_bounds_applied'])
        check('original_descent_to_zero_target_explicit',diag['original_descent_terminal_m']==0.)
        state=g.state_dict();state['entry_reference21'][0]=999.
        check('state_dict_cannot_mutate_internal_entry',g.state_dict()['entry_reference21'][0]!=999.)
        check('state_json_strict_finite_or_none',isinstance(json.dumps(g.state_dict(),allow_nan=False),str))
        reject('invalid_global_box_rejected',lambda:g.proposal(12.,upper,lower))
        reject('past_intent_query_rejected',lambda:g.proposal(11.995,lower,upper))
        reject('stale_authority_reference_clock_rejected',lambda:g.lift_interval(Reference21(start,11.995)))
        reject('boolean_hand_pressure_not_measurement',lambda:update(g,1,normal=np.ones(5,bool)))
        reject('numeric_validity_not_boolmask',lambda:update(g,1,valid=np.ones(5)))
        check('no_model_data_or_pipeline_calls',sum(forbidden_counts.values())==0)
    finally:
        for name, original in originals.items():setattr(mujoco,name,original)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        ROOT/'v15_support/release_tasks.py',ROOT/'v15_support/receiver_sensor.py',
        ROOT/'v13_joint/lifecycle.py',ROOT/'v13_joint/design.py',ROOT/'v13_a1/codec.py',Path(__file__))}
    report=dict(schema='v15-release-tasks-author-pure-v1',passed=all(c['passed'] for c in checks),
        checks=checks,check_count=len(checks),failures=[c['name'] for c in checks if not c['passed']],
        sources=sources,forbidden_model_or_pipeline_calls=forbidden_counts,
        scope='pure task graph; typed class imports allowed; no model/plant construction, no integration, no performance evidence')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],check_count=len(checks),failures=report['failures'],output=str(out)),ensure_ascii=False))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()

"""One current-state native-effort lifecycle for contact transfer.

No physical state truth or force forecast enters. SEARCH, CAPTURE, TRANSFER
and SEPARATE change task meaning, not the sole actor or the effort evaluator.
Every 5ms action is solved at its own public q/v and actual reference domain.
"""
from copy import deepcopy
import json
import time
import numpy as np
from v13_a1.codec import Reference21, EnginePort21, finite_array
from v14_coord.effort_domain import FastEffortGoal21
from v14_coord.design import reference_rates, QP_MAX_ITERATIONS, QP_ATOL, QP_RTOL
from v15_support.receiver_sensor import ReceiverMeasurement, DELAY_S
from v17_contact_transfer.supervisor import ReleaseSupervisor as HandEvidenceOwner, _clock, CLOCK_TOL
from v12_native.controller import plain
from .design import CONTROL_PERIOD_S, RELEASE_DEADLINE_S
from .governor import LiftProgressGovernor
from .effort_task import (solve_effort_task, check_same_problem_candidate,
                          native_correction25, score_native_realization)


def governor_task_interval(governed, *, q_m, v_m_s, actual_lower_m, actual_upper_m):
    """A declared TASK restriction inside the unchanged mechanical/rate domain.

    If its current-q recovery band is not reachable on this tick, select the
    legal nearest recovery endpoint. That is task debt, not mechanical failure.
    Neither this interval nor reaching q is a future robot-stop certificate.
    """
    g=governed.diagnostics;L=float(actual_lower_m);U=float(actual_upper_m)
    if L>U:raise ValueError('empty original mechanical/rate interval')
    band_lo=q_m-g['recovery_reference_budget_m']
    band_hi=q_m+g['downward_recovery_reference_budget_m']
    a=max(L,band_lo);b=min(U,band_hi)
    reachable=a<=b
    if not reachable:
        a=b=float(np.clip(q_m,L,U))
    elif g['braking_intent_active']:
        brake=float(np.clip(q_m,a,b))
        if v_m_s<0:a=brake
        elif v_m_s>0:b=brake
        else:a=b=brake
    return a,b,dict(role='declared_reference_recovery_task_not_hardware_bound',
        original_mechanical_and_rate_interval_m=[L,U],current_q_budget_band_m=[band_lo,band_hi],
        task_reference_interval_m=[a,b],budget_band_reachable_this_tick=reachable,
        no_intersection_action='nearest_legal_current_q_recovery_endpoint' if not reachable else None,
        debt_scope='same_current_q_command_budget_not_future_stopping',physical_stop_certified=False)


class ReleaseSupervisor(HandEvidenceOwner):
    def __init__(self, entry, *, open18, observer, robot_model, limits5, budget, force_margin):
        super().__init__(entry,open18=open18,observer=observer)
        self.phase='SEARCH'; self.native=FastEffortGoal21(robot_model)
        self.governor=LiftProgressGovernor(robot_model)
        self.limits=finite_array(limits5,(5,),'declared material limits5')
        self.budget=float(budget);self.margin=float(force_margin)
        if not np.isfinite([self.budget,self.margin]).all() or self.budget<0 or self.margin<0:
            raise ValueError('finite nonnegative release intent limits required')
        self.receiver_current=None;self.unload_fraction=None;self.raw_loads=None
        self._observe_signature=None

    def observe(self, *, frame, transfer_inputs, current21):
        tick,now=_clock(frame.time)
        if now<self.entry.started_s-CLOCK_TOL:raise ValueError('release before request')
        signature=json.dumps(plain(dict(current21=current21,
            hand=vars(transfer_inputs['hand']) if hasattr(transfer_inputs['hand'],'__dict__') else
                {k:getattr(transfer_inputs['hand'],k) for k in ('read_time_s','acquisition_time_s','valid','signed_support_N','uncertainty_N')},
            receiver={k:getattr(transfer_inputs['receiver'],k) for k in
                ('time_s','acquisition_time_s','age_s','valid','wrench6','uncertainty6')},
            pressure={k:getattr(frame,k,None) for k in ('sample_time','age','normal','valid',
                'auxiliary_pressure','auxiliary_valid','auxiliary_age')},
            robot={k:getattr(frame,k,None) for k in ('robot_position25','robot_velocity25','robot_control_time_s')})),
            sort_keys=True,allow_nan=True)
        if self.last_tick is not None and tick==self.last_tick:
            if signature!=self._observe_signature:raise ValueError('consumed public release payload changed on duplicate tick')
            return deepcopy(self.public_state)
        if self.last_tick is not None and tick<self.last_tick:raise ValueError('public clock rewind')
        evidence=self.observer.update(**transfer_inputs)
        current=finite_array(current21,(21,),'authoritative reference21')
        gap=self.last_tick is not None and tick!=self.last_tick+1
        receiver=transfer_inputs['receiver'];hand=transfer_inputs['hand']
        if type(receiver) is not ReceiverMeasurement:raise TypeError('already sampled receiver required')
        eligible=bool(receiver.valid and receiver.acquisition_time_s is not None and
            abs(receiver.time_s-now)<=CLOCK_TOL and receiver.age_s is not None and
            abs(receiver.age_s-DELAY_S)<=CLOCK_TOL and
            receiver.acquisition_time_s>=self.entry.started_s-CLOCK_TOL)
        acq=receiver.acquisition_time_s if eligible else None
        fresh=eligible and (self.last_acquisition is None or acq>self.last_acquisition+CLOCK_TOL)
        adjacent=eligible and self.last_acquisition is not None and abs(acq-self.last_acquisition-CONTROL_PERIOD_S)<=CLOCK_TOL and not gap
        compressed=bool(eligible and receiver.normal_load_N>receiver.uncertainty6[2])
        if not compressed or gap:self.receiver_streak=0
        if compressed and fresh:self.receiver_streak=self.receiver_streak+1 if adjacent else 1
        if eligible and fresh:self.last_acquisition=acq
        if compressed and not self.receiver_event_seen:
            self.receiver_event_seen=True;self.receiver_event_at=now
            self.receiver_lift_reference_at_event=float(current[18])
            if self.phase=='SEARCH':self._phase('CAPTURE')
        if self.phase=='CAPTURE' and self.receiver_streak>=2:self._phase('TRANSFER')
        self.receiver_current=receiver if eligible else None
        self.unload_fraction=None;self.raw_loads=None
        if eligible and hand.valid and hand.acquisition_time_s is not None and abs(hand.acquisition_time_s-acq)<=CLOCK_TOL:
            H=float(hand.signed_support_N)
            R=float(-np.asarray(receiver.wrench6[:3])@self.observer.receiver_up)
            h=max(0.,H);r=max(0.,R)
            self.unload_fraction=(r/(h+r)) if h+r>0 and compressed else None
            self.raw_loads=dict(hand_signed_N=H,receiver_signed_N=R,
                hand_width_N=float(hand.uncertainty_N),receiver_compression_N=float(receiver.normal_load_N),
                acquisition_s=acq,load_sum_N=H+R,
                role='reaction progress only; any-actor loads, not object weight or placement')
        low=self._hand_low(frame,now,gap)
        if low['confirmed']:self._phase('SEPARATE')
        deadline=now-self.entry.started_s>=RELEASE_DEADLINE_S-CLOCK_TOL
        if deadline and not self.was_clear:self._phase('UNSUPPORTED_EXIT')
        self._current21=current.copy();self.last_transfer=evidence.to_public()
        self.public_state=dict(schema='v18_native_effort_contact_transfer_process_v1',time_s=now,
            phase=self.phase,version=self.version,started_s=self.entry.started_s,
            holding_preempted=True,holding_force_tasks_active=False,
            first_receiver_event_latched=self.receiver_event_seen,receiver_event_at=self.receiver_event_at,
            receiver_current_compression_resolved=compressed,receiver_fresh_streak=self.receiver_streak,
            receiver_event_is_any_actor_candidate_not_placement=True,
            low_hand_pressure_N=low['pressure_N'],low_hand_pressure_available=low['available'],
            low_hand_pressure_confirmed=low['confirmed'],hand_low_evidence=low,
            sampled_hand_low_was_observed=self.was_clear,recontact_episode_count=self.recontact_episode_count,
            raw_coacquired_loads=self.raw_loads,unload_reaction_fraction=self.unload_fraction,
            transfer=evidence.to_public(),unsupported_clearance=self.phase=='UNSUPPORTED_EXIT',
            unconfirmed_receiver_support=not self.receiver_event_seen,
            coordinated_retreat_started=self.phase=='SEPARATE',deadline_reached=deadline,
            deadline_without_sampled_hand_clearance=deadline and not self.was_clear,
            delivery_confirmed=False,placed=False,placement_certified=False,
            geometric_full_hand_clearance_certified=False,no_regrip_after_release=True,
            release_request_delayed=False)
        self.last_time=now;self.last_tick=tick
        self._observe_signature=signature
        return deepcopy(self.public_state)

    def nominal(self, query_time_s, lower21, upper21):
        """Only the stale-plan fallback; ordinary actions use actual q/v below."""
        lo=finite_array(lower21,(21,),'fallback lower');hi=finite_array(upper21,(21,),'fallback upper')
        value=self._current21.copy()
        if self.phase in ('TRANSFER','SEPARATE','UNSUPPORTED_EXIT'):value[:18]=self.open18
        # A fallback cannot invent a fresh robot snapshot or jump to q.
        return Reference21(np.minimum(hi,np.maximum(lo,value)),query_time_s)

    def action(self, *, frame, adapter, current21, lower21, upper21, generation):
        _,now=_clock(frame.time)
        if self.last_time is None or abs(now-self.last_time)>CLOCK_TOL:raise ValueError('observe same public tick first')
        if float(frame.dt)!=CONTROL_PERIOD_S:raise ValueError('one genuine5ms action required')
        if isinstance(generation,(bool,np.bool_)) or not isinstance(generation,(int,np.integer)) or generation<0:
            raise ValueError('explicit mechanical generation required')
        began=time.perf_counter_ns()
        current=finite_array(current21,(21,),'current21');self._current21=current.copy()
        lower=finite_array(lower21,(21,),'actual lower21');upper=finite_array(upper21,(21,),'actual upper21')
        rates=reference_rates('release');lo=np.maximum(lower,current-rates*CONTROL_PERIOD_S);hi=np.minimum(upper,current+rates*CONTROL_PERIOD_S)
        if np.any(lo>hi):raise ValueError('empty actual5ms domain')
        q=finite_array(frame.robot_position25,(25,),'robot q25');v=finite_array(frame.robot_velocity25,(25,),'robot v25')
        if (float(frame.robot_control_time_s)!=now or
            finite_array(frame.robot_reference21,(21,),'public reference').tobytes()!=current.tobytes()):
            raise ValueError('non-authoritative robot command snapshot')
        desired_velocity=-float(rates[18]) if self.phase=='SEARCH' else 0.
        if self.phase=='SEPARATE':desired_velocity=float(rates[18])
        elif self.phase in ('CAPTURE','TRANSFER') and self.raw_loads is not None:
            prior=self.observer.prior
            if prior is not None:
                ceiling=prior.value_N+prior.uncertainty_N
                R=self.raw_loads['receiver_compression_N']
                if R>ceiling and R>0:desired_velocity=float(rates[18])*(R-ceiling)/R
        governed=self.governor.step(time_s=now,q_m=float(q[24]),v_m_s=float(v[24]),
            current_reference_m=float(current[18]),reference_lower_m=float(lower[18]),
            reference_upper_m=float(upper[18]),reference_rate_lower_m_s=-float(rates[18]),
            reference_rate_upper_m_s=float(rates[18]),dt_s=CONTROL_PERIOD_S,
            requested_velocity_m_s=desired_velocity,
            brake_requested=self.phase in ('CAPTURE','TRANSFER','UNSUPPORTED_EXIT') and desired_velocity==0.,
            deadline_s=self.entry.started_s+RELEASE_DEADLINE_S)
        if governed.reference_m is None:
            raise ValueError('known lift governor unavailable: '+governed.status)
        mechanical_rate_lo=lo.copy();mechanical_rate_hi=hi.copy()
        lo=lo.copy();hi=hi.copy()
        lo[18],hi[18],governor_domain=governor_task_interval(governed,
            q_m=float(q[24]),v_m_s=float(v[24]),actual_lower_m=float(lo[18]),actual_upper_m=float(hi[18]))
        posture=current.copy();posture[18]=governed.reference_m
        if self.phase in ('TRANSFER','SEPARATE','UNSUPPORTED_EXIT'):
            posture[:18]=self.open18
        posture=np.minimum(hi,np.maximum(lo,posture))
        posture[19:]=np.minimum(hi[19:],np.maximum(lo[19:],self.entry.reference21.values[19:]))
        ref=Reference21(current,now)
        engine=EnginePort21(frame.robot_actuator_target21,now,self.native.actor.lift_bias_m)
        domain=self.native.build_domain(q25=q,v25=v,robot_axes25=self.native.robot_axes25,time_s=now,
            current_reference=ref,nominal_reference=ref,actual_engine=engine,
            reference_lower21=lo,reference_upper21=hi)
        intent=adapter.current_contact_intents(frame)
        J=intent['J'];y=intent['y'];digits=intent['digits'];P=len(y)
        fraction=self.unload_fraction if self.phase in ('CAPTURE','TRANSFER') else None
        if self.phase in ('SEPARATE','UNSUPPORTED_EXIT'):fraction=1.
        # Unknown transfer progress preserves the current normal intent. It is
        # explicitly not a zero support observation or a placement decision.
        zprior=y.copy() if fraction is None else y*(1.-fraction)
        caps=np.maximum(0.,self.limits-self.margin);budget=max(0.,self.budget-2*self.margin)
        qp=native=acceptance=problem=score=None;error=None;end=posture.copy()
        if domain.available:
            force_scale=np.max(abs(self.native.servo.force_range),axis=1)
            joint_scale=abs(domain.transmission21x25.T)@force_scale
            if np.any(force_scale<=0) or np.any(joint_scale<=0):raise ValueError('explicit effort units required')
            sr=1./joint_scale;sd=1./force_scale
            ptarget=self.native.servo.evaluate(q,v,self.native.actor.encode(Reference21(posture,now)).values,time_s=now)
            if ptarget.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA':raise ValueError('posture native value unknown')
            posture_delta=ptarget.predicted_effort21-domain.p0_effort21
            motion_delta=np.zeros(21);motion_delta[18:]=posture_delta[18:]
            tau_extra=intent['tau_tangent25']+domain.transmission21x25.T@motion_delta
            E=np.zeros((21,21+P));E[:,:21]=np.diag(sd)
            digit_pressure=np.bincount(digits,weights=y,minlength=5)
            shares=y/digit_pressure[digits]
            # One digit's normal-intent cost is invariant to splitting an
            # identical observed patch: sum(delta_z_i^2/share_i)/cap^2.
            sf=1./(np.maximum(1.,caps[digits])*np.sqrt(shares))
            problem=dict(p0_21=domain.p0_effort21,A21x25=domain.transmission21x25,
                Jpx25=J,y_p=y,z_intent_p=zprior,tau_extra25=tau_extra,
                effort_lower21=domain.effort_lower21,effort_upper21=domain.effort_upper21,
                patch_digit_p=digits,caps5=caps,total_budget=budget,
                support_coeff_p=np.zeros(P),load_proxy_N=0.,
                scales=dict(residual25=sr,intentP=sf,delta21=sd,load=1.),
                extra_task_rows=dict(matrix=E,target=sd*posture_delta),
                max_iterations=QP_MAX_ITERATIONS,atol=QP_ATOL,rtol=QP_RTOL)
            try:
                qp=solve_effort_task(**problem)
                acceptance=check_same_problem_candidate(problem,qp)
                if acceptance['accepted']:
                    native=self.native.solve(domain,p0_effort21=domain.p0_effort21,
                        delta_effort21=qp.d21,correction25=native_correction25(problem,qp.zP),
                        residual_scale25=sr,delta_scale21=sd,requested_target_effort21=qp.p21)
                    if native.available:
                        score=score_native_realization(problem,original_d21=qp.d21,zP=qp.zP,native_result=native)
                        if score.get('available') and score.get('feasible'):
                            end=native.reference21.values.copy()
                        else:error='final_same_task_score_or_primal_check_failed'
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                error=type(exc).__name__+': '+str(exc)
        accepted=bool(native is not None and native.available and score is not None and
                      score.get('available') and score.get('feasible'))
        if np.any(end<lo) or np.any(end>hi) or not np.isfinite(end).all():
            raise ValueError('emitted reference outside original actual5ms domain')
        ev=dict(decision_id=len(self.decision_evidence),time_s=now,phase=self.phase,
            mode_generation=int(generation),process_version=self.version,
            current_reference21=current.tolist(),actual_reference_lower21=lo.tolist(),actual_reference_upper21=hi.tolist(),
            original_mechanical_rate_lower21=mechanical_rate_lo.tolist(),original_mechanical_rate_upper21=mechanical_rate_hi.tolist(),
            governor=plain(governed.diagnostics),governor_task_domain=governor_domain,posture_reference21=posture.tolist(),
            contact_intents=plain(intent['diagnostics']),unload_fraction=fraction,
            normal_intent_weighting='per-digit pressure-share normalized, invariant to identical patch splitting',
            unknown_progress_preserves_current_normal_intent=fraction is None,
            qp_problem=plain(problem),proposal_acceptance=acceptance,
            domain=domain.to_dict(),native=None if native is None else native.to_dict(),
            native_full_task_score=plain(score),
            qp=None if qp is None else dict(status=qp.status,reason=qp.reason,d21=plain(qp.d21),zP=plain(qp.zP),objective=qp.objective,diagnostics=plain(qp.diagnostics)),
            proposed_reference21=end.tolist(),effort_goal_realized=accepted,numerical_error=error,
            fallback=None if accepted else 'legal_governed_posture_without_joint_task_certificate',
            public_process=deepcopy(self.public_state),solver_ms=(time.perf_counter_ns()-began)/1e6,
            current_qv_scope_only=True,future_pressure_prediction_available=False,
            actual_stopping_or_contact_safety_certified=False,placement_certified=False)
        self.decision_evidence.append(ev);self.plan_generation=int(generation)
        self._receipt_governor_context=dict(q=float(q[24]),lower=float(lo[18]),upper=float(hi[18]),
            budget_up=governed.diagnostics['recovery_reference_budget_m'],
            budget_down=governed.diagnostics['downward_recovery_reference_budget_m'])
        self.debug=dict(architecture='v18_single_current_effort_contact_transfer',mode='release',
            decision_id=ev['decision_id'],decision_time_s=now,mode_generation=int(generation),
            phase=self.phase,effort_goal_realized=accepted,fallback=ev['fallback'],
            process=deepcopy(self.public_state),governor=plain(governed.diagnostics),
            holding_force_tasks_active=False,pressure_prediction_available=False,
            solver_timing_ms=dict(whole_decision=ev['solver_ms']))
        return Reference21(end,now),deepcopy(self.debug)

    def reconcile(self, receipt):
        c=self._receipt_governor_context;u=float(receipt.executed.values[18]);debt=c['q']-u
        actual=dict(actual_reference21=receipt.executed.values.tolist(),
            same_snapshot_q_lift_m=c['q'],same_snapshot_reference_debt_m=debt,
            current_q_budget_respected=bool(-c['budget_down']<=debt<=c['budget_up']),
            actual_task_interval_respected=bool(c['lower']<=u<=c['upper']),
            execution_changed=bool(receipt.execution_changed),stale_plan_replaced=bool(receipt.stale_plan_replaced),
            reference_budget_not_actual_motion_stop=True)
        self.decision_evidence[-1]['execution_receipt']=actual
        self.debug.update(actual_reference21=receipt.executed.values.tolist(),
            execution_changed_plan=receipt.execution_changed,stale_plan_replaced=receipt.stale_plan_replaced,
            governor_execution_receipt=actual)

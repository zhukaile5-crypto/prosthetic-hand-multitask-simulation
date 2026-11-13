"""One explicit reference/effort/progress problem for the release lifecycle."""
from copy import deepcopy
import time
import numpy as np
from v13_a1.codec import Reference21,EnginePort21,finite_array
from v14_coord.design import reference_rates,QP_MAX_ITERATIONS,QP_ATOL,QP_RTOL
from v17_contact_transfer.supervisor import _clock,CLOCK_TOL
from v18_effort_transfer.supervisor import ReleaseSupervisor as PublicLifecycle
from v18_effort_transfer.supervisor import governor_task_interval
from v12_native.controller import plain
from .design import CONTROL_PERIOD_S,RELEASE_DEADLINE_S,PROGRESS_SCALE
from .native_branch import NativeBranchBuilder21
from .task_qp import solve_task_qp,evaluate_task,check_same_problem_candidate


class ReleaseSupervisor(PublicLifecycle):
    def __init__(self,*args,robot_model,**kwargs):
        super().__init__(*args,robot_model=robot_model,**kwargs)
        self.branch_builder=NativeBranchBuilder21(robot_model)

    def observe(self,*,frame,transfer_inputs,current21):
        super().observe(frame=frame,transfer_inputs=transfer_inputs,current21=current21)
        rho=None;width=None;reason='coacquired_support_unknown'
        if self.raw_loads is not None and self.receiver_current is not None:
            H=self.raw_loads['hand_signed_N'];R=self.raw_loads['receiver_signed_N']
            width=float(abs(self.observer.receiver_up)@self.receiver_current.uncertainty6[:3])
            receiver_lower=max(0.,R-width)
            denominator=receiver_lower+abs(H)+self.raw_loads['hand_width_N']
            if denominator>0:
                rho=float(receiver_lower/denominator)
                reason='engineering_reaction_share_not_true_transferred_weight'
        self.unload_fraction=rho
        self.public_state.update(schema='v19_explicit_reference_contact_transfer_process_v1',
            unload_reaction_fraction=rho,v19_reaction_share=rho,
            receiver_support_engineering_width_N=width,reaction_share_role=reason,
            signed_negative_hand_load_is_not_automatic_full_transfer=True,
            clearance_reference_is_explicit=True)
        return deepcopy(self.public_state)

    def action(self,*,frame,adapter,current21,lower21,upper21,generation):
        _,now=_clock(frame.time)
        if self.last_time is None or abs(now-self.last_time)>CLOCK_TOL:raise ValueError('observe same public tick first')
        if float(frame.dt)!=CONTROL_PERIOD_S:raise ValueError('one actual5ms tick required')
        if isinstance(generation,(bool,np.bool_)) or not isinstance(generation,(int,np.integer)) or generation<0:
            raise ValueError('explicit nonnegative generation required')
        began=time.perf_counter_ns()
        current=finite_array(current21,(21,),'actual current21');self._current21=current.copy()
        lower=finite_array(lower21,(21,),'actual lower21');upper=finite_array(upper21,(21,),'actual upper21')
        q=finite_array(frame.robot_position25,(25,),'public q25');v=finite_array(frame.robot_velocity25,(25,),'public v25')
        if (float(frame.robot_control_time_s)!=now or
            finite_array(frame.robot_reference21,(21,),'public reference').tobytes()!=current.tobytes()):
            raise ValueError('non-authoritative public command snapshot')
        rates=reference_rates('release')
        lo=np.maximum(lower,current-rates*CONTROL_PERIOD_S);hi=np.minimum(upper,current+rates*CONTROL_PERIOD_S)
        if np.any(lo>hi):raise ValueError('empty original mechanical/rate interval')
        mechanical_lo=lo.copy();mechanical_hi=hi.copy()
        requested_velocity=-float(rates[18]) if self.phase=='SEARCH' else 0.
        if self.phase=='SEPARATE':requested_velocity=float(rates[18])
        elif self.phase in ('CAPTURE','TRANSFER') and self.raw_loads is not None and self.observer.prior is not None:
            R=self.raw_loads['receiver_compression_N'];ceiling=self.observer.prior.value_N+self.observer.prior.uncertainty_N
            if R>ceiling and R>0:requested_velocity=float(rates[18])*(R-ceiling)/R
        governed=self.governor.step(time_s=now,q_m=float(q[24]),v_m_s=float(v[24]),
            current_reference_m=float(current[18]),reference_lower_m=float(lower[18]),reference_upper_m=float(upper[18]),
            reference_rate_lower_m_s=-float(rates[18]),reference_rate_upper_m_s=float(rates[18]),
            requested_velocity_m_s=requested_velocity,dt_s=CONTROL_PERIOD_S,
            brake_requested=self.phase in ('CAPTURE','TRANSFER','UNSUPPORTED_EXIT') and requested_velocity==0.,
            deadline_s=self.entry.started_s+RELEASE_DEADLINE_S)
        if governed.reference_m is None:raise ValueError('robot governor unavailable: '+governed.status)
        lo=lo.copy();hi=hi.copy()
        lo[18],hi[18],task_domain=governor_task_interval(governed,q_m=float(q[24]),v_m_s=float(v[24]),
            actual_lower_m=float(lo[18]),actual_upper_m=float(hi[18]))
        if self.phase in ('SEARCH','CAPTURE'):
            gamma_lo=gamma_hi=0.;unload=0.;progress_role='support_search_or_capture_no_clearance_progress'
        elif self.phase=='TRANSFER':
            gamma_lo=0.;gamma_hi=0. if self.unload_fraction is None else self.unload_fraction
            unload=1.;progress_role='joint_unload_clearance_progress_bounded_by_reaction_share'
        elif self.phase=='UNSUPPORTED_EXIT':
            gamma_lo=gamma_hi=1.;unload=1.;progress_role='original_deadline_bounded_exit_not_placement'
        else:
            gamma_lo=0.;gamma_hi=1.;unload=1.;progress_role='sampled_low_hand_pressure_clearance_not_stationarity'
        motion=current.copy();motion[18]=governed.reference_m;motion[19:]=self.entry.reference21.values[19:]
        motion=np.minimum(hi,np.maximum(lo,motion))
        opened=motion.copy();opened[:18]=np.minimum(hi[:18],np.maximum(lo[:18],self.open18))
        motion_step=motion-current;open_step=opened-motion
        preferred=np.minimum(hi,np.maximum(lo,motion+gamma_hi*open_step))
        typed_current=Reference21(current,now)
        engine=EnginePort21(frame.robot_actuator_target21,now,self.branch_builder.native.actor.lift_bias_m)
        branch=self.branch_builder.build(q25=q,v25=v,robot_axes25=self.branch_builder.native.robot_axes25,
            time_s=now,current_reference=typed_current,actual_engine=engine,
            preferred_reference=Reference21(preferred,now),reference_lower21=lo,reference_upper21=hi)
        contact_velocity_target=(governed.diagnostics['governed_velocity_intent_m_s'] if self.phase=='SEARCH' else 0.)
        intent=adapter.current_contact_intents(frame,desired_lift_velocity_m_s=contact_velocity_target)
        J=intent['J'];y=intent['y'];digits=intent['digits'];P=len(y)
        caps=np.maximum(0.,self.limits-self.margin);budget=max(0.,self.budget-2*self.margin)
        qp=problem=acceptance=actual=score=None;error=None;accepted=False
        end=preferred.copy();rounding_changed=False;rounding_projection21=None
        if branch.available:
            native=self.branch_builder.native
            force_units=np.max(abs(native.servo.force_range),axis=1)
            joint_units=abs(branch.transmission21x25.T)@force_units
            if np.any(force_units<=0) or np.any(joint_units<=0):raise ValueError('finite positive effort units required')
            pressure=np.bincount(digits,weights=y,minlength=5)
            shares=y/pressure[digits]
            sf=1./(np.maximum(1.,caps[digits])*np.sqrt(shares))
            motion_value=native.servo.evaluate(q,v,native.actor.encode(Reference21(motion,now)).values,time_s=now)
            if motion_value.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA':raise ValueError('native motion effort unknown')
            motion_delta=np.zeros(21);motion_delta[18:]=(motion_value.predicted_effort21-branch.p0_effort21)[18:]
            tau_extra=intent['tau_motion25']+branch.transmission21x25.T@motion_delta
            problem=dict(p0_21=branch.p0_effort21,A21x25=branch.transmission21x25,
                gain21=branch.gain21,offset21=branch.offset21,Jpx25=J,y_p=y,tau_extra25=tau_extra,
                delta_lower21=branch.delta_lower21,delta_upper21=branch.delta_upper21,
                open_step21=open_step,motion_step21=motion_step,unload_fraction=unload,
                patch_digit_p=digits,caps5=caps,total_budget=budget,
                scales=dict(residual25=1./joint_units,intentP=sf,reference21=1./(rates*CONTROL_PERIOD_S),
                    delta21=1./force_units,progress=PROGRESS_SCALE),
                gamma_lower=gamma_lo,gamma_upper=gamma_hi,max_iterations=QP_MAX_ITERATIONS,atol=QP_ATOL,rtol=QP_RTOL)
            try:
                qp=solve_task_qp(**problem);acceptance=check_same_problem_candidate(problem,qp)
                if acceptance['accepted']:
                    raw=current+qp.du21
                    # Only representational roundoff is returned to the same
                    # certified branch. This is rescored below, not re-labelled
                    # as the original exact affine optimum.
                    proposal=np.minimum(branch.absolute_reference_upper21,np.maximum(branch.absolute_reference_lower21,raw))
                    rounding_projection21=proposal-raw
                    allowed_roundoff=64*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(raw),abs(proposal)))
                    if np.any(abs(rounding_projection21)>allowed_roundoff):
                        raise ValueError('reference_branch_projection_exceeds_declared_roundoff')
                    rounding_changed=proposal.tobytes()!=raw.tobytes()
                    actual=self.branch_builder.evaluate_reference(branch,Reference21(proposal,now))
                    if actual['available']:
                        actual_du=proposal-current
                        score=evaluate_task(problem,actual_du,qp.zP,qp.gamma,
                            actual_delta21=actual['actual_delta_effort21'])
                        if score.get('available') and score.get('feasible'):
                            value=float(score['objective']);anchor=float(acceptance['anchor_objective'])
                            allowance=64*np.finfo(float).eps*max(1.,abs(value),abs(anchor))
                            if value<=anchor+allowance:
                                end=proposal;accepted=True
                            else:error='actual_full_same_task_merit_worse_than_declared_anchor'
                        else:error='actual_full_same_task_score_or_primal_invalid'
                    else:error='full_native_reference_recheck_failed'
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                error=type(exc).__name__+': '+str(exc)
        if not np.isfinite(end).all() or np.any(end<lo) or np.any(end>hi):
            raise ValueError('reference left original5ms mechanical/task domain')
        record=dict(decision_id=len(self.decision_evidence),time_s=now,phase=self.phase,
            mode_generation=int(generation),process_version=self.version,
            current_reference21=current.tolist(),original_mechanical_rate_lower21=mechanical_lo.tolist(),
            original_mechanical_rate_upper21=mechanical_hi.tolist(),actual_reference_lower21=lo.tolist(),actual_reference_upper21=hi.tolist(),
            governor=plain(governed.diagnostics),governor_task_domain=task_domain,
            motion_reference21=motion.tolist(),open_reference21=opened.tolist(),preferred_reference21=preferred.tolist(),
            progress_role=progress_role,gamma_domain=[gamma_lo,gamma_hi],reaction_share=self.unload_fraction,
            contact_intents=plain(intent['diagnostics']),branch=branch.to_dict(),
            qp_problem=plain(problem),proposal_acceptance=acceptance,
            qp=None if qp is None else dict(status=qp.status,reason=qp.reason,du21=plain(qp.du21),
                zP=plain(qp.zP),gamma=qp.gamma,d21=plain(qp.d21),p21=plain(qp.p21),objective=qp.objective,diagnostics=plain(qp.diagnostics)),
            native_actual=plain(actual),actual_full_task_score=plain(score),rounding_projection_changed=rounding_changed,
            rounding_projection21=plain(rounding_projection21),
            proposed_reference21=end.tolist(),reference_effort_goal_accepted=accepted,numerical_error=error,
            fallback=None if accepted else 'legal_governed_reference_progress_without_joint_task_certificate',
            public_process=deepcopy(self.public_state),solver_ms=(time.perf_counter_ns()-began)/1e6,
            future_motion_or_pressure_prediction_available=False,placement_certified=False)
        self.decision_evidence.append(record);self.plan_generation=int(generation)
        self._receipt_governor_context=dict(q=float(q[24]),lower=float(lo[18]),upper=float(hi[18]),
            budget_up=governed.diagnostics['recovery_reference_budget_m'],
            budget_down=governed.diagnostics['downward_recovery_reference_budget_m'])
        self.debug=dict(architecture='v19_joint_reference_contact_progress',mode='release',phase=self.phase,
            decision_id=record['decision_id'],decision_time_s=now,mode_generation=int(generation),
            reference_effort_goal_accepted=accepted,gamma=None if qp is None else qp.gamma,
            fallback=record['fallback'],process=deepcopy(self.public_state),
            pressure_prediction_available=False,solver_timing_ms=dict(whole_decision=record['solver_ms']))
        return Reference21(end,now),deepcopy(self.debug)

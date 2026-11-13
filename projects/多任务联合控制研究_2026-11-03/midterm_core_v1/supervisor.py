"""Independent core release: approach, contact coordination, and one native commit.
No exit-path memory, optional-withdrawal budgeting, or predictive task compensation.
Numerical validation and mechanical protections remain shared.
"""
from copy import deepcopy
import time
import numpy as np
from v13_a1.codec import Reference21, finite_array
from v14_coord.design import reference_rates
from v17_contact_transfer.supervisor import _clock, CLOCK_TOL
from v18_effort_transfer.supervisor import ReleaseSupervisor as PublicLifecycle
from v18_effort_transfer.supervisor import governor_task_interval
from v12_native.controller import plain
from v21_stable_handoff.design import CONTROL_PERIOD_S, RELEASE_DEADLINE_S
from v21_stable_handoff.task_solver import solve_contact_exit


from v21_stable_handoff.prefix_supervisor import contact_exit_lift_interval, score_emitted_reference
from .approach import SupportPreview, TEMPLATE, approach_request

class CoreSupervisor(PublicLifecycle):
    def __init__(self, *args, approach_mount, anticipation=True, coordinated=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.anticipation = bool(anticipation)
        self.coordinated = bool(coordinated)
        self.preview = SupportPreview(approach_mount, TEMPLATE)
        self.preview_packet = None

    def observe(self, *, frame, transfer_inputs, current21):
        super().observe(frame=frame, transfer_inputs=transfer_inputs, current21=current21)
        active = self.phase in ('TRANSFER', 'SEPARATE', 'UNSUPPORTED_EXIT')
        self.public_state.update(schema='midterm_core_release_process_v1',
            coordinated_retreat_started=active,
            contact_exit_task_active=active,
            reaction_share_is_not_exit_permission=True,
            low_hand_pressure_is_posterior_evidence_not_exit_start_gate=True,
            terminal_OPEN_is_secondary_preference=True,
            known_empty_does_not_restore_OPEN_sweep=True,
            default_upward_retreat_removed=True)
        self.preview_packet = None
        if self.phase == "SEARCH" and self.anticipation:
            packet = {k:getattr(frame,k,None) for k in
                ("time","sample_time","skin_patches","sensor_characteristics","lift_velocity_m_s")}
            try:
                self.preview_packet = self.preview.read(packet)
            except (ValueError,TypeError,KeyError,FloatingPointError,np.linalg.LinAlgError) as exc:
                self.preview_packet = dict(available=False, reason=str(exc), read_s=float(frame.time))
        return deepcopy(self.public_state)

    def nominal(self, query_time_s, lower21, upper21):
        # A stale command has no new q/v or task certificate. Preserve the
        # authoritative committed reference rather than inventing an OPEN path.
        lo = finite_array(lower21, (21,), 'stale actual lower')
        hi = finite_array(upper21, (21,), 'stale actual upper')
        return Reference21(np.minimum(hi, np.maximum(lo, self._current21)), query_time_s)

    def action(self, *, frame, adapter, current21, lower21, upper21, generation):
        _, now = _clock(frame.time)
        if self.last_time is None or abs(now-self.last_time) > CLOCK_TOL:
            raise ValueError('observe same public tick first')
        if float(frame.dt) != CONTROL_PERIOD_S:
            raise ValueError('one actual5ms action required')
        if isinstance(generation, (bool, np.bool_)) or not isinstance(generation, (int, np.integer)) or generation < 0:
            raise ValueError('explicit nonnegative generation required')
        began = time.perf_counter_ns()
        current = finite_array(current21, (21,), 'authoritative current21')
        self._current21 = current.copy()
        lower = finite_array(lower21, (21,), 'real lower21')
        upper = finite_array(upper21, (21,), 'real upper21')
        q = finite_array(frame.robot_position25, (25,), 'public q25')
        v = finite_array(frame.robot_velocity25, (25,), 'public v25')
        if (float(frame.robot_control_time_s) != now or
            finite_array(frame.robot_reference21, (21,), 'public reference').tobytes() != current.tobytes()):
            raise ValueError('non-authoritative command snapshot')
        rates = reference_rates('release')
        scale = rates*CONTROL_PERIOD_S
        lo = np.maximum(lower, current-scale)
        hi = np.minimum(upper, current+scale)
        if np.any(lo > hi):
            raise ValueError('empty original mechanical/rate interval')
        mechanical_lo, mechanical_hi = lo.copy(), hi.copy()
        exit_phase = self.phase in ('TRANSFER', 'SEPARATE', 'UNSUPPORTED_EXIT')
        tasks = None
        if exit_phase:
            tasks = adapter.current_contact_tasks(frame, current21=current,
                reference_lower21=lo, reference_upper21=hi)
        requested = -float(rates[18]) if self.phase == 'SEARCH' else 0.
        approach = None
        if self.phase == 'SEARCH':
            envelope = float(rates[18])*self.governor.recovery_budget_s/self.governor.tau_s
            requested, approach = approach_request(requested, envelope, now,
                                                   self.preview_packet, self.anticipation)
        governed = self.governor.step(time_s=now, q_m=float(q[24]), v_m_s=float(v[24]),
            current_reference_m=float(current[18]), reference_lower_m=float(lower[18]),
            reference_upper_m=float(upper[18]), reference_rate_lower_m_s=-float(rates[18]),
            reference_rate_upper_m_s=float(rates[18]), requested_velocity_m_s=requested,
            dt_s=CONTROL_PERIOD_S, brake_requested=self.phase == 'CAPTURE',
            deadline_s=self.entry.started_s+RELEASE_DEADLINE_S)
        if governed.reference_m is None:
            raise ValueError('robot governor unavailable: '+governed.status)
        if exit_phase:
            lo[18], hi[18], task_domain = contact_exit_lift_interval(governed,
                q_m=float(q[24]), v_m_s=float(v[24]), tau_s=self.governor.tau_s,
                actual_lower_m=float(lo[18]), actual_upper_m=float(hi[18]))
        else:
            lo[18], hi[18], task_domain = governor_task_interval(governed,
                q_m=float(q[24]), v_m_s=float(v[24]),
                actual_lower_m=float(lo[18]), actual_upper_m=float(hi[18]))
        # Current wrists remain a minimum-change preference, with genuine
        # motion authority for the higher-priority contact task.
        preferred = current.copy()
        if exit_phase and tasks is not None and tasks.get("available") and len(tasks["pressure_P_N"]):
            preferred[:18] = self.open18
        # With no currently observed finger contact, hold the current reference.
        # No remembered exit pose or additional lift retreat is generated.
        if not exit_phase:
            preferred[18] = governed.reference_m
        target_delta = preferred-current
        anchor = np.minimum(hi, np.maximum(lo, current))
        end = anchor.copy()
        solved = native_current = native_proposed = emitted_score = None
        rounding_projection = None
        error = None
        accepted = False
        if not exit_phase or not self.coordinated:
            end = np.minimum(hi, np.maximum(lo, preferred))
            role = 'common_approach_governor' if not exit_phase else 'ablation_direct_bounded_open_preference'
            accepted = True
        else:
            role = 'contact_exit_hierarchy_current_public_snapshot'
            try:
                if tasks['available']:
                    solved = solve_contact_exit(time_s=now, dt_s=CONTROL_PERIOD_S,
                        current_reference21=current,
                        encoder_reference21=tasks['encoder_reference21'],
                        step_lower21=lo-current, step_upper21=hi-current,
                        reference_rates21=rates,
                        normal_jacobian_Px25=tasks['normal_jacobian_Px25'],
                        proposal_map25x21=tasks['proposal_map25x21'],
                        robot_velocity25=v, pressure_P_N=tasks['pressure_P_N'],
                        reference_target_delta21=target_delta,
                        motion_matrix_Mx21=tasks.get('motion_matrix_Mx21'),
                        motion_error_M_m_s=tasks.get('motion_error_M_m_s'),
                        motion_time_s=tasks.get('motion_time_s'),
                        max_iterations=400, atol=1e-9, rtol=1e-8)
                    if solved.available:
                        raw = current+finite_array(solved.x, (21,), 'normalized reference solution')*scale
                        proposed = np.minimum(hi, np.maximum(lo, raw))
                        rounding_projection = proposed-raw
                        rounding = 64*np.finfo(float).eps*np.maximum(1., np.maximum(abs(raw), abs(proposed)))
                        if np.any(abs(proposed-raw) > rounding):
                            raise ValueError('solution_requires_more_than_roundoff_domain_projection')
                        emitted_score = score_emitted_reference(solved, current21=current,
                            proposed21=proposed, scale21=scale)
                        if not emitted_score['available']:
                            raise ValueError('emitted_reference_does_not_preserve_declared_hierarchy_attainment')
                        # This is a same-current-q/v native effort VALUE check,
                        # not a forecast of contact pressure or finite motion.
                        native_current = self.native.servo.evaluate(q, v,
                            frame.robot_actuator_target21, time_s=now)
                        native_proposed = self.native.servo.evaluate(q, v,
                            self.native.actor.encode(Reference21(proposed, now)).values, time_s=now)
                        if (native_current.status != 'KNOWN_ROBOT_ACTUATION_ALGEBRA' or
                            native_proposed.status != 'KNOWN_ROBOT_ACTUATION_ALGEBRA'):
                            raise ValueError('current native actuation value unknown')
                        end = proposed
                        accepted = True
            except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                error = type(exc).__name__+': '+str(exc)
        if not np.isfinite(end).all() or np.any(end < lo) or np.any(end > hi):
            raise ValueError('reference outside actual5ms mechanical/task domain')
        fallback = None if accepted or not exit_phase else 'legal_current_q_recovery_anchor_without_exit_certificate'
        record = dict(decision_id=len(self.decision_evidence), time_s=now, phase=self.phase,
            mode_generation=int(generation), process_version=self.version, task_role=role,
            current_reference21=current.tolist(), original_mechanical_rate_lower21=mechanical_lo.tolist(),
            original_mechanical_rate_upper21=mechanical_hi.tolist(),
            actual_reference_lower21=lo.tolist(), actual_reference_upper21=hi.tolist(),
            governor=plain(governed.diagnostics), governor_task_domain=task_domain,
            preferred_reference21=preferred.tolist(), reference_target_delta21=target_delta.tolist(),
            core_mechanisms=dict(anticipation=self.anticipation, coordinated_handoff=self.coordinated),
            approach_rule=plain(approach), exit_path_memory_active=False,
            geometry_probe_original_rate_lower21=mechanical_lo.tolist(),
            geometry_probe_original_rate_upper21=mechanical_hi.tolist(),
            feasible_recovery_anchor21=anchor.tolist(), contact_tasks=plain(tasks),
            hierarchy=None if solved is None else dict(status=solved.status, x=plain(solved.x),
                levels=plain(solved.levels), diagnostics=plain(solved.diagnostics),
                completed_prefix=solved.completed_prefix, fully_converged=solved.fully_converged,
                failed_layer=plain(solved.failed_layer), certified_prefix=plain(solved.certified_prefix)),
            native_current=None if native_current is None else plain(vars(native_current)),
            native_proposed=None if native_proposed is None else plain(vars(native_proposed)),
            emitted_hierarchy_score=plain(emitted_score), rounding_projection21=plain(rounding_projection),
            proposed_reference21=end.tolist(), geometry_proposal_accepted=accepted,
            numerical_error=error, fallback=fallback, public_process=deepcopy(self.public_state),
            solver_ms=(time.perf_counter_ns()-began)/1e6,
            future_motion_or_pressure_prediction_available=False, placement_certified=False)
        self.decision_evidence.append(record)
        self.plan_generation = int(generation)
        self._receipt_governor_context = dict(q=float(q[24]), lower=float(lo[18]), upper=float(hi[18]),
            budget_up=governed.diagnostics['recovery_reference_budget_m'],
            budget_down=governed.diagnostics['downward_recovery_reference_budget_m'])
        self.debug = dict(architecture='midterm_macro_core_v1', mode='release',
            phase=self.phase, decision_id=record['decision_id'], decision_time_s=now,
            mode_generation=int(generation), geometry_proposal_accepted=accepted,
            fallback=fallback, process=deepcopy(self.public_state),
            pressure_prediction_available=False, solver_timing_ms=dict(whole_decision=record['solver_ms']))
        return Reference21(end, now), deepcopy(self.debug)

    def reconcile(self, receipt):
        super().reconcile(receipt)

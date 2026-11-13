"""One contact-conditioned21 reference hierarchy and one original actor.

SEARCH/CAPTURE retain the previous low-impact command governor. During exit,
real finger, wrist and lift permissions are shared by pressure-conditioned
withdrawal, measured external-motion intent and terminal posture preference.
No reference geometry is certified as future servo or object response.
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
from .design import CONTROL_PERIOD_S, RELEASE_DEADLINE_S
from .prefix_task import solve_contact_exit


def contact_exit_lift_interval(governed, *, q_m, v_m_s, tau_s,
                               actual_lower_m, actual_upper_m):
    """Same current-q recovery budget, available to the contact task.

    Zero nominal lift velocity is not a request to remove all lift authority.
    Actual overspeed still restricts the command toward current-q recovery.
    This is a declared command task box, not a stopping or safety certificate.
    """
    g = governed.diagnostics
    L, U = float(actual_lower_m), float(actual_upper_m)
    if L > U or not tau_s > 0:
        raise ValueError('ordered real domain and known positive robot time scale required')
    up = float(g['recovery_reference_budget_m'])
    down = float(g['downward_recovery_reference_budget_m'])
    band_lo, band_hi = q_m-up, q_m+down
    a, b = max(L, band_lo), min(U, band_hi)
    reachable = a <= b
    direction = None
    if not reachable:
        a = b = float(np.clip(q_m, L, U))
    else:
        # Arithmetic equality allowance only; the final interval is never
        # expanded beyond the original mechanical/rate/recovery intersection.
        eps = 32*np.finfo(float).eps*max(1., abs(v_m_s), up/tau_s, down/tau_s)
        brake = float(np.clip(q_m, a, b))
        if v_m_s < -up/tau_s-eps:
            a = brake
            direction = 'measured_descent_requires_current_q_recovery'
        elif v_m_s > down/tau_s+eps:
            b = brake
            direction = 'measured_ascent_requires_current_q_recovery'
    return a, b, dict(role='contact_exit_current_q_recovery_task_not_hardware_bound',
        original_mechanical_and_rate_interval_m=[L, U],
        current_q_budget_band_m=[band_lo, band_hi], task_reference_interval_m=[a, b],
        budget_band_reachable_this_tick=reachable, overspeed_direction_restriction=direction,
        no_intersection_action=None if reachable else 'nearest_legal_current_q_recovery_endpoint',
        zero_nominal_velocity_does_not_remove_exit_authority=True,
        debt_scope='same_current_q_command_budget_not_future_stopping', physical_stop_certified=False)


def score_emitted_reference(solved, *, current21, proposed21, scale21):
    """Recheck only actual certified tasks at the emitted reference.

    For extended one-sided tasks, the original certified slack is retained.
    A failed lower-layer iterate never supplies either x or the slack vector.
    """
    x = (proposed21-current21)/scale21
    raw = np.asarray(solved.diagnostics.get('extended_normalized_x', solved.x), float)
    extended = raw.copy()
    extended[:21] = x
    coordinate_roundoff = np.zeros(len(raw))
    coordinate_roundoff[:21] = 128*np.finfo(float).eps*np.maximum(
        1., np.maximum(abs(current21), abs(proposed21)))/scale21
    rows, domains = [], []
    for task in solved.certified_prefix:
        matrix = np.asarray(task['matrix'], float)
        target = np.asarray(task['target'], float)
        original = np.asarray(task['attained'], float)
        returned_attained = matrix@raw
        attained = matrix@extended
        residual = attained-target
        tolerance = abs(matrix)@coordinate_roundoff + 128*np.finfo(float).eps*np.maximum(
            1., abs(matrix)@abs(extended))
        kernel_tolerance = float(task['KKT']['coordinate_tolerance'])
        kernel_preserved = bool(np.all(abs(returned_attained-original) <= kernel_tolerance))
        preserved = bool(np.isfinite(residual).all() and kernel_preserved
            and np.all(abs(attained-returned_attained) <= tolerance))
        rows.append(dict(name=task['name'], attained=attained.tolist(), residual=residual.tolist(),
            squared_loss=float(residual@residual),
            certified_original_attainment=original.tolist(),
            returned_attainment=returned_attained.tolist(),
            kernel_original_tolerance=kernel_tolerance,
            certified_attainment_and_emission_roundoff_respected=preserved))
        domain = task['hard_domain']
        G = np.asarray(domain['normalized_inequality_matrix'], float)
        h = np.asarray(domain['normalized_inequality_upper'], float)
        E = np.asarray(domain['fixed_equality_matrix'], float)
        f = np.asarray(domain['fixed_equality_target'], float)
        if E.size == 0:
            E = np.empty((0, len(raw)))
        inequality = G@extended-h
        equality = E@extended-f
        allowance_g = kernel_tolerance+abs(G)@coordinate_roundoff
        allowance_e = kernel_tolerance+abs(E)@coordinate_roundoff
        ok = bool(np.all(inequality <= allowance_g) and np.all(abs(equality) <= allowance_e))
        domains.append(dict(layer=task['name'], available=ok,
            normalized_inequality_violation=float(np.max(inequality, initial=0.)),
            fixed_equality_error=float(np.max(abs(equality), initial=0.))))
    # A new motion budget is constructed at the certified L1 point. If its
    # first optimization fails, its feasibility still must hold at emission.
    if 'second_stage_hard_matrix' in solved.diagnostics:
        G = np.asarray(solved.diagnostics['second_stage_hard_matrix'], float)
        h = np.asarray(solved.diagnostics['second_stage_hard_upper'], float)
        if G.size == 0:
            G = np.empty((0, len(raw)))
        magnitude = np.linalg.norm(G, axis=1)
        unit = np.where(magnitude > 0, magnitude, 1.)
        G = G/unit[:, None]
        h = h/unit
        tol = max([float(c['KKT']['coordinate_tolerance']) for c in solved.certified_prefix], default=0.)
        violation = G@extended-h
        domains.append(dict(layer='witnessed_motion_and_normal_task_domain',
            available=bool(np.all(violation <= tol+abs(G)@coordinate_roundoff)),
            normalized_inequality_violation=float(np.max(violation, initial=0.))))
    return dict(available=bool(rows) and all(r['certified_attainment_and_emission_roundoff_respected']
        and np.isfinite(r['squared_loss']) for r in rows) and all(d['available'] for d in domains),
        normalized_emitted_delta21=x.tolist(), extended_emitted_x=extended.tolist(),
        normalized_coordinate_roundoff_bound=coordinate_roundoff.tolist(), levels=rows,
        hard_domains=domains, completed_prefix=solved.completed_prefix,
        fully_converged=solved.fully_converged,
        scope='same_declared_reference_task_not_actual_motion_or_contact_prediction')


class ReleaseSupervisor(PublicLifecycle):
    def observe(self, *, frame, transfer_inputs, current21):
        super().observe(frame=frame, transfer_inputs=transfer_inputs, current21=current21)
        active = self.phase in ('TRANSFER', 'SEPARATE', 'UNSUPPORTED_EXIT')
        self.public_state.update(schema='v21_prefix_only_v20_process_v1',
            coordinated_retreat_started=active,
            contact_exit_task_active=active,
            reaction_share_is_not_exit_permission=True,
            low_hand_pressure_is_posterior_evidence_not_exit_start_gate=True,
            terminal_OPEN_is_secondary_preference=True)
        return deepcopy(self.public_state)

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
        requested = -float(rates[18]) if self.phase == 'SEARCH' else 0.
        if self.phase == 'SEPARATE':
            requested = float(rates[18])
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
        if exit_phase:
            preferred[:18] = self.open18
        if not exit_phase or self.phase == 'SEPARATE':
            preferred[18] = governed.reference_m
        target_delta = preferred-current
        anchor = np.minimum(hi, np.maximum(lo, current))
        end = anchor.copy()
        tasks = solved = native_current = native_proposed = emitted_score = None
        rounding_projection = None
        error = None
        accepted = False
        if not exit_phase:
            end = np.minimum(hi, np.maximum(lo, preferred))
            role = 'direct_frozen_search_capture_governor'
        else:
            role = 'contact_exit_hierarchy_current_public_snapshot'
            try:
                tasks = adapter.current_contact_tasks(frame, current21=current,
                    reference_lower21=lo, reference_upper21=hi)
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
        self.debug = dict(architecture='v21_prefix_only_v20_contact_exit', mode='release',
            phase=self.phase, decision_id=record['decision_id'], decision_time_s=now,
            mode_generation=int(generation), geometry_proposal_accepted=accepted,
            fallback=fallback, process=deepcopy(self.public_state),
            pressure_prediction_available=False, solver_timing_ms=dict(whole_decision=record['solver_ms']))
        return Reference21(end, now), deepcopy(self.debug)

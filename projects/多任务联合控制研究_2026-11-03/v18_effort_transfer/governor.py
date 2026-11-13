"""Current (q, v, u) lift progress, with a finite reference-recovery budget.

This is a robot velocity-task proposal, not a future response, contact-force,
stopping-time or stopping-distance certificate.  References are unbiased A1
coordinates.  The sole actor still adds the known robot gravity bias once.
No plant data, object state, contact reader, writer or dynamics call is used.
"""
from dataclasses import dataclass
from numbers import Real
import hashlib
import numpy as np
from .design import CONTROL_PERIOD_S, DECISION_PERIOD_S


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(name + ' requires a finite real number')
    return float(value)


@dataclass(frozen=True)
class LiftProgressResult:
    available: bool
    reference_m: object
    increment_m: object
    status: str
    diagnostics: dict


class LiftProgressGovernor:
    """Use existing servo constants and a declared recovery-command horizon.

    tau=(kv+passive_damping)/kp.  u_velocity=q+tau*v_requested is the
    corresponding steady robot velocity intention.  The SEARCH debt q-u is
    limited by B_up=upward_reference_rate*recovery_budget_s, with the analogous
    downward recovery budget for an upward intention.  Existing excessive
    debt OR measured velocity prevents further progress in the same direction;
    the opposite unloading intention remains allowed.  Every reference obeys the *current* actual
    absolute and dt-rate intersection.  q is not propagated into the future.

    The velocity envelopes B/tau are source-derived engineering governor scales,
    not a validated bound on actual velocity.  The supervisor owns phase,
    receiver interpretation, the unchanged task deadline and unknown exit.
    This scalar function has no clock history or writer: the supervisor must
    validate same-age q/v/reference and reject duplicate/rewound actor ticks.
    """
    def __init__(self, robot_model, *, recovery_budget_s=DECISION_PERIOD_S):
        self.recovery_budget_s = _real(recovery_budget_s, 'recovery budget')
        if self.recovery_budget_s <= 0:
            raise ValueError('positive recovery-command budget required')
        self.model = robot_model
        self._reasons = []
        try:
            import mujoco
            m = robot_model
            self._aid = int(m.actuator('lift_motor').id)
            self._jid = int(m.joint('transport_lift').id)
            self._dof = int(m.jnt_dofadr[self._jid])
            a, j = self._aid, self._jid
            self._ctrladr = int(m.actuator_ctrladr[a])
            self._outadr = int(m.actuator_outadr[a])
            if (m.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE or
                m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT or
                int(m.actuator_trnid[a, 0]) != j or
                not np.array_equal(np.asarray(m.actuator_gear[a]), np.array([1., 0., 0., 0., 0., 0.]))):
                raise ValueError('known unit linear lift transmission required')
            if (m.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE or
                int(m.actuator_actnum[a]) != 0 or m.actuator_delay[a] != 0 or
                int(m.actuator_ctrlnum[a]) != 1 or int(m.actuator_outnum[a]) != 1 or
                int(m.actuator_plugin[a]) >= 0 or
                m.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED or
                m.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE):
                raise ValueError('known stateless fixed-gain affine lift law required')
            gain = np.asarray(m.actuator_gainprm[a, :3], float)
            bias = np.asarray(m.actuator_biasprm[a, :3], float)
            self.kp = _real(gain[0], 'lift kp')
            self.kv = -_real(bias[2], 'lift velocity coefficient')
            self.passive_damping = _real(m.dof_damping[self._dof], 'lift passive damping')
            self.bias0 = _real(bias[0], 'lift affine bias0')
            if (self.kp <= 0 or self.kv < 0 or self.passive_damping < 0 or
                gain[1] != 0 or gain[2] != 0 or bias[1] != -self.kp):
                raise ValueError('known positive position servo with nonnegative damping required')
            self.tau_s = (self.kv + self.passive_damping) / self.kp
            if not self.tau_s > 0:
                raise ValueError('velocity-task scale unavailable for zero total damping')
            # Exact original actor expression.  This module never applies it.
            self.lift_bias_m = float(m.body_subtreemass[m.body('rh_forearm').id] * 9.81 / 1500.)
            if self.kp != 1500.:
                raise ValueError('A1 lift bias expression requires original kp1500')
            if int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION):
                raise ValueError('actuation is disabled')
            if int(m.opt.disableactuator) & (1 << int(m.actuator_group[a])):
                raise ValueError('lift actuator group is disabled')
            if (int(m.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP) or
                m.jnt_actfrclimited[j] or m.jnt_actgravcomp[j]):
                raise ValueError('unsupported sleep or joint total actuation mode')
            self._bound_signature = self._signature()
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self._reasons.append(str(exc))

    def _signature(self):
        m, a, j = self.model, self._aid, self._jid
        h = hashlib.sha256()
        for value in (m.actuator_gainprm[a], m.actuator_biasprm[a], m.actuator_gear[a],
                      m.actuator_trnid[a], m.actuator_trntype[a], m.actuator_gaintype[a],
                      m.actuator_biastype[a], m.actuator_dyntype[a], m.actuator_actnum[a],
                      m.actuator_delay[a], m.actuator_plugin[a], m.actuator_group[a],
                      m.actuator_ctrlnum[a], m.actuator_outnum[a], m.actuator_ctrladr[a], m.actuator_outadr[a],
                      m.actuator_ctrllimited[self._ctrladr], m.actuator_ctrlrange[self._ctrladr],
                      m.actuator_forcelimited[self._outadr], m.actuator_forcerange[self._outadr],
                      m.dof_damping[self._dof], m.jnt_type[j], m.jnt_dofadr[j],
                      m.jnt_actfrclimited[j], m.jnt_actgravcomp[j],
                      m.opt.disableflags, m.opt.disableactuator, m.opt.enableflags,
                      m.body_subtreemass[m.body('rh_forearm').id]):
            h.update(np.asarray(value).tobytes())
        return h.hexdigest()

    def _native_effort(self, reference, q, v):
        m, a = self.model, self._aid
        engine = reference + self.lift_bias_m
        effective = engine
        import mujoco
        if (not int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
                and m.actuator_ctrllimited[self._ctrladr]):
            effective = float(np.clip(engine, *m.actuator_ctrlrange[self._ctrladr]))
        # Keep the native arithmetic order; no beta/precombined intercept.
        raw = self.kp * effective + self.bias0 + (-self.kp) * q + (-self.kv) * v
        effort = float(np.clip(raw, *m.actuator_forcerange[self._outadr])) if m.actuator_forcelimited[self._outadr] else raw
        return dict(engine_target_m=engine, effective_engine_target_m=effective,
                    raw_effort_N=raw, effort_N=effort,
                    control_clipped=bool(effective != engine), force_clipped=bool(effort != raw))

    def step(self, *, time_s, q_m, v_m_s, current_reference_m,
             reference_lower_m, reference_upper_m,
             reference_rate_lower_m_s, reference_rate_upper_m_s,
             requested_velocity_m_s, dt_s=CONTROL_PERIOD_S, brake_requested=False,
             deadline_s=None):
        scope = dict(contact_force_prediction_available=False, physics_safety_certified=False,
                     future_robot_response_predicted=False, stopping_time_or_distance_certified=False,
                     actor_gravity_bias_applied_here=False,
                     recovery_budget_scope='reference return to CURRENT encoder q, not future stopping',
                     model_uses_external_object_parameters=False)
        if self._reasons:
            return LiftProgressResult(False, None, None, 'UNAVAILABLE_ROBOT_SERVO', dict(scope, reasons=list(self._reasons)))
        if self._signature() != self._bound_signature:
            return LiftProgressResult(False, None, None, 'UNAVAILABLE_MODEL_CHANGED', scope)
        try:
            t = _real(time_s, 'time'); q = _real(q_m, 'lift encoder')
            v = _real(v_m_s, 'lift encoder velocity'); u = _real(current_reference_m, 'committed lift reference')
            lower = _real(reference_lower_m, 'actual lower reference')
            upper = _real(reference_upper_m, 'actual upper reference')
            rlo = _real(reference_rate_lower_m_s, 'negative reference rate')
            rhi = _real(reference_rate_upper_m_s, 'positive reference rate')
            dt = _real(dt_s, 'execution interval'); requested = _real(requested_velocity_m_s, 'requested search velocity')
            if not isinstance(brake_requested, (bool, np.bool_)):
                raise ValueError('brake_requested requires bool')
            deadline = None if deadline_s is None else _real(deadline_s, 'task deadline')
            if t < 0 or dt != CONTROL_PERIOD_S or lower > upper or rlo > 0 or rhi < 0:
                raise ValueError('ordered bounds, signed rates, genuine5ms interval and nonnegative time required')
            L, U = max(lower, u + rlo * dt), min(upper, u + rhi * dt)
            if L > U:
                return LiftProgressResult(False, None, None, 'UNAVAILABLE_EMPTY_ACTUAL_AUTHORITY',
                    dict(scope, current_reference_m=u, actual_interval_m=[L, U]))
            B = rhi * self.recovery_budget_s
            B_down = -rlo * self.recovery_budget_s
            velocity_envelope = B / self.tau_s
            upward_velocity_envelope = B_down / self.tau_s
            debt = q - u
            # Tolerance only disambiguates representational equality, never
            # enlarges emitted mechanical or rate intervals.
            eps = 32 * np.finfo(float).eps * max(1., abs(q), abs(u), abs(B), abs(v))
            debt_excess = debt > B + eps
            upper_debt_excess = -debt > B_down + eps
            actual_overspeed = -v > velocity_envelope + eps
            upward_overspeed = v > upward_velocity_envelope + eps
            timed_out = deadline is not None and t >= deadline
            governed_velocity = float(np.clip(requested, -velocity_envelope, upward_velocity_envelope))
            if debt_excess or actual_overspeed:
                governed_velocity = max(0., governed_velocity)
            if upper_debt_excess or upward_overspeed:
                governed_velocity = min(0., governed_velocity)
            if brake_requested or timed_out:
                governed_velocity = 0.
            braking = bool(brake_requested or debt_excess or actual_overspeed or upper_debt_excess or
                           upward_overspeed or requested == 0 or timed_out)
            desired = q + self.tau_s * governed_velocity
            reference = float(np.clip(desired, L, U))
            remaining_brake_debt = q - reference
            return_rate = rhi if remaining_brake_debt >= 0 else -rlo
            return_time = (abs(remaining_brake_debt)/return_rate if return_rate > 0 else
                           (0. if remaining_brake_debt == 0 else None))
            reason = ('deadline_requires_supervisor_exit' if timed_out else
                      'explicit_brake' if brake_requested else
                      'existing_reference_debt' if debt_excess else
                      'measured_descent_outside_command_recovery_envelope' if actual_overspeed else
                      'existing_upward_reference_debt' if upper_debt_excess else
                      'measured_ascent_outside_command_recovery_envelope' if upward_overspeed else
                      'zero_velocity_request' if requested == 0 else 'bounded_search')
            effort = self._native_effort(reference, q, v)
            diagnostic = dict(scope, time_s=t, q_m=q, actual_velocity_m_s=v,
                committed_reference_m=u, actual_reference_lag_m=debt,
                kp_N_m=self.kp, kv_N_s_m=self.kv, passive_damping_N_s_m=self.passive_damping,
                servo_velocity_scale_s=self.tau_s, recovery_budget_s=self.recovery_budget_s,
                recovery_reference_budget_m=B, source_derived_velocity_envelope_m_s=velocity_envelope,
                downward_recovery_reference_budget_m=B_down,
                source_derived_upward_velocity_envelope_m_s=upward_velocity_envelope,
                requested_velocity_m_s=requested, governed_velocity_intent_m_s=governed_velocity,
                actual_authority_interval_m=[L, U], unprojected_velocity_reference_m=desired,
                emitted_reference_m=reference, reference_projection_residual_m=desired-reference,
                current_encoder_brake_reference_m=q, remaining_brake_reference_debt_m=remaining_brake_debt,
                minimum_reference_return_time_at_current_q_s=return_time,
                current_q_brake_reference_inside_global_authority=bool(lower <= q <= upper),
                current_q_brake_reference_reachable_within_budget=bool(
                    max(lower, u+rlo*self.recovery_budget_s) <= q <= min(upper, u+rhi*self.recovery_budget_s)),
                current_debt_exceeds_budget=bool(debt_excess), measured_descent_exceeds_envelope=bool(actual_overspeed),
                current_upward_debt_exceeds_budget=bool(upper_debt_excess),
                measured_ascent_exceeds_envelope=bool(upward_overspeed),
                braking_intent_active=braking, reason=reason,
                equivalent_velocity_at_current_q_m_s=(reference-q)/self.tau_s,
                actual_velocity_not_inferred_from_reference=True,
                current_native_effort_of_emitted_reference=effort,
                deadline_exceeded=timed_out, supervisor_exit_required=timed_out,
                brake_reference_reached_is_not_actual_stop=True)
            return LiftProgressResult(True, reference, reference-u,
                                      'BRAKE_RECOVER' if braking else 'BOUNDED_SEARCH', diagnostic)
        except (ValueError, TypeError) as exc:
            return LiftProgressResult(False, None, None, 'UNAVAILABLE_PUBLIC_INPUT', dict(scope, reasons=[str(exc)]))

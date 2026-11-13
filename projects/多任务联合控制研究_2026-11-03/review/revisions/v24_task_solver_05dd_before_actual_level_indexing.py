"""Same-domain backend control and joint soft-task contact exit.

All matrices describe CURRENT reference geometry intents, never actual future
velocity/pressure or physical safety. The normal minimum witnesses a motion
TASK budget. Backend-only keeps its certified slack frozen, exactly as V21;
joint releases that optimum and coordinates correction with observed motion.
No model inversion, object identity, integration, engine write or sensor exists.
"""
from dataclasses import dataclass
from numbers import Integral, Real
import numpy as np

from v21_stable_handoff.design import CONTROL_PERIOD_S, COMMAND_DEBT_INTENT_PERIOD_S
from v23_task_execution.finite_solver import Level, solve_hierarchy
from v22_joint_task_fusion.task_bundle import assemble

MAX_PATCHES = 128
MAX_MOTION_ROWS = 384
NORMAL_WITNESS_SELECTOR = "NORMAL_OPTIMUM_THEN_MINIMUM_RATE_DT_NORMALIZED_REFERENCE"
CANONICAL_WITNESS_LEVEL = "canonical_minimum_normalized_reference_witness"


@dataclass(frozen=True)
class ExitTaskResult:
    status: str
    x: np.ndarray
    levels: tuple
    diagnostics: dict
    completed_prefix: int = 0
    certified_prefix: tuple = ()
    fully_converged: bool = False
    failed_layer: object = None

    @property
    def available(self):
        return self.completed_prefix > 0 and self.x.shape == (21,) and np.isfinite(self.x).all()

    @property
    def accepted_prefix(self):
        return self.certified_prefix


class _Unknown(ValueError): pass


def _array(value, shape, name):
    if value is None: raise _Unknown(name+'_unknown')
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw.flat):
        raise ValueError(name+'_requires_explicit_numeric_'+str(shape))
    out = np.asarray(value, float).copy()
    if not np.isfinite(out).all(): raise ValueError(name+'_nonfinite')
    return out


def _scalar(value, name, positive=False):
    if value is None: raise _Unknown(name+'_unknown')
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(name+'_outside_real_domain')
    return float(value)


def _ro(value):
    result = np.asarray(value).copy(); result.setflags(write=False); return result


def _scope():
    return dict(role='CURRENT_REFERENCE_GEOMETRY_WITH_CANONICAL_WITNESSED_MOTION_TASK_BUDGET',
        canonical_selector_role='BUDGET_WITNESS_ONLY_NOT_EXECUTION_POSTURE_TASK',
        canonical_reference_norm_is_not_physical_energy=True,
        pressure_prediction_available=False, future_robot_response_predicted=False,
        actual_external_motion_budget_guaranteed=False, physical_safety_certified=False,
        placement_certified=False, no_pressure_threshold_or_score_change=True,
        normal_slack_not_normal_displacement_frozen=True, software_return_budget_s=COMMAND_DEBT_INTENT_PERIOD_S,
        software_return_budget_is_not_servo_time_constant=True,
        force_saturation_is_not_target_geometry_loss=True, current_reference_origin_preserved=True)


def _failure(status, reason):
    return ExitTaskResult(status, _ro(np.full(21, np.nan)), (), dict(_scope(), reason=str(reason), actual_increment_available=False))


def _certificates(result):
    return tuple(result.certified_prefix)


def solve_contact_exit(*, time_s, dt_s, current_reference21, encoder_reference21,
    step_lower21, step_upper21, reference_rates21, normal_jacobian_Px25,
    proposal_map25x21, robot_velocity25, pressure_P_N, reference_target_delta21,
    motion_matrix_Mx21=None, motion_error_M_m_s=None, motion_row_scale_M=None,
    motion_time_s=None, path_reference_target_delta21=None,
    max_iterations=400, atol=1e-9, rtol=1e-8, formulation='joint'):
    """V20-compatible arguments, normalized x21 output, no writer or prediction.

    Internal variables [x21,etaP] are dimensionless; physical slack r=s*eta.
    A positive-pressure witness requires normal optimum THEN minimum ||x21||^2
    certified. The second certificate is witness-only, never an execution task.
    Backend-only fixes eta AFTER this witness is complete. Joint keeps
    eta's original derived bounds and submits only the actual simultaneous
    objective as its prefix. Both witness motion-budget rows by their current
    normal anchor x_A, never reset/nominal zero. A later failure keeps only
    genuinely certified tasks of the selected formulation.
    Optional path target changes ONLY the last posture preference; the original
    canonical target remains logged. It is not an extra higher-priority task.
    """
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            if formulation not in ('backend','joint'):
                raise ValueError('explicit backend or joint task formulation required')
            now = _scalar(time_s, 'time_s'); dt = _scalar(dt_s, 'dt_s', True)
            if now < 0 or dt != CONTROL_PERIOD_S: raise ValueError('nonnegative_current_time_and_exact5ms_required')
            if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral) or max_iterations < 1:
                raise ValueError('positive_integer_iteration_budget_required')
            atol = _scalar(atol, 'atol', True); rtol = _scalar(rtol, 'rtol')
            if rtol < 0: raise ValueError('nonnegative_rtol_required')
            current = _array(current_reference21, (21,), 'current_reference')
            encoder = _array(encoder_reference21, (21,), 'encoder_reference')
            lo = _array(step_lower21, (21,), 'actual_step_lower'); hi = _array(step_upper21, (21,), 'actual_step_upper')
            if np.any(lo > hi): raise ValueError('empty_real_reference_domain')
            rates = _array(reference_rates21, (21,), 'reference_rates')
            if np.any(rates <= 0): raise ValueError('positive_reference_units_required')
            units = rates*dt
            if np.any(units == 0): raise FloatingPointError('positive_reference_units_underflow')
            B = _array(proposal_map25x21, (25, 21), 'declared_reference_geometry')
            v = _array(robot_velocity25, (25,), 'current_full_robot_velocity')
            if normal_jacobian_Px25 is None: raise _Unknown('normal_geometry_unknown_not_knownP0')
            raw = np.asarray(normal_jacobian_Px25, dtype=object)
            if raw.ndim != 2 or raw.shape[1] != 25 or len(raw) > MAX_PATCHES: raise ValueError('normal_rows_require_P_by25_at_most128')
            P = len(raw); J = _array(raw, (P, 25), 'normal_robot_J'); pressure = _array(pressure_P_N, (P,), 'positive_patch_pressure')
            if np.any(pressure <= 0): raise ValueError('positive_observed_patch_pressure_required')
            original_target = _array(reference_target_delta21, (21,), 'canonical_reference_target')
            chosen_target = original_target if path_reference_target_delta21 is None else _array(path_reference_target_delta21, (21,), 'explicit_path_reference_target')
            N = J@B; Nunit = N*units[None, :]; debt = current-encoder
            vn = J@v; an = N@debt
            b = -dt*(np.maximum(vn, 0.)+np.maximum(an/COMMAND_DEBT_INTENT_PERIOD_S, 0.))
            row_units = np.sum(abs(Nunit), axis=1); zero_normal = row_units == 0
            if np.any(zero_normal & np.any(N != 0, axis=1)): raise FloatingPointError('normal_units_underflow_not_zero_authority')
            s = np.where(zero_normal, 1., row_units)
            shares = np.empty(0)
            if P:
                relative = pressure/pressure.max()
                if np.any(relative == 0): raise FloatingPointError('positive_pressure_share_underflow')
                shares = relative/relative.sum()
                if np.any(shares == 0): raise FloatingPointError('positive_pressure_share_underflow')
            rowmax = np.sum(np.where(N >= 0, N*hi[None, :], N*lo[None, :]), axis=1)
            r_upper = np.maximum(0., rowmax-b)
            eta_upper = r_upper/s
            lower = np.r_[lo/units, np.zeros(P)]; upper = np.r_[hi/units, eta_upper]
            x0 = np.clip(np.zeros(21), lower[:21], upper[:21])
            # Max in raw units before division keeps slack definition explicit.
            eta0 = np.maximum(Nunit@x0-b, 0.)/s
            if np.any(eta0 > eta_upper):
                excess = eta0-eta_upper
                if np.max(excess, initial=0.) > 128*np.finfo(float).eps*max(1., float(np.max(eta_upper, initial=0.))):
                    raise FloatingPointError('anchor_slack_exceeds_derived_box_upper')
                # Bound endpoint reconstruction only, never a larger slack box.
                eta0 = np.minimum(eta0, eta_upper)
            anchor0 = np.r_[x0, eta0]
            normal_A = np.column_stack((Nunit/s[:, None], -np.eye(P)))
            normal_upper = b/s
            unfrozen_lower = lower.copy(); unfrozen_upper = upper.copy()
            first = None; first_certs = (); levels = (); xA = anchor0.copy()
            witness_certs = (); witness_levels = ()
            if P:
                cost = np.zeros((P, 21+P)); cost[:, 21:] = np.diag(np.sqrt(shares))
                canonical_cost = np.zeros((21, 21+P)); canonical_cost[:, :21] = np.eye(21)
                first = solve_hierarchy(lower=lower, upper=upper,
                    levels=(Level('normal_inward_correction_slack', cost, np.zeros(P)),
                            Level(CANONICAL_WITNESS_LEVEL, canonical_cost, np.zeros(21))),
                    initial=anchor0, inequality_matrix=normal_A, inequality_upper=normal_upper,
                    max_iterations=int(max_iterations), atol=atol, rtol=rtol)
                witness_levels = tuple(first.levels)
                witness_certs = tuple(dict(c, witness_index=i, solver_local_index=c['index'],
                    task_stage='canonical_normal_witness',
                    certificate_role='NORMAL_OPTIMUM_WITNESS' if i == 0 else 'CANONICAL_SELECTOR_WITNESS_ONLY_NEVER_SUBMITTED')
                    for i, c in enumerate(_certificates(first)))
                canonical_ready = bool(first.fully_converged and first.completed_prefix == 2
                    and len(witness_certs) == len(witness_levels) == 2
                    and all(c.get('KKT', {}).get('certified') is True for c in witness_certs))
                if not canonical_ready:
                    diagnostics = dict(_scope(), time_s=now, first_stage_diagnostics=first.diagnostics,
                        first_stage_status=first.status, original_lower=_ro(lower), original_upper=_ro(upper),
                        actual_increment_available=False, canonical_witness_required=True,
                        canonical_witness_available=False, canonical_witness_certified=False,
                        normal_witness_selector=NORMAL_WITNESS_SELECTOR,
                        normal_witness_budget_defined=False,
                        normal_witness_certified_prefix=witness_certs, normal_witness_levels=witness_levels,
                        witness_failed_layer=first.failed_layer, witness_completed_prefix=first.completed_prefix,
                        actual_submitted_certificate_count=0, normal_witness_is_submitted_prefix=False,
                        canonical_witness_certificate_submitted=False,
                        motion_task=dict(available=False, reason='canonical_witness_unavailable_budget_not_defined'),
                        reason='canonical_normal_witness_two_certificates_required')
                    # A witness failure is NOT an attempted actual-task layer.
                    # In particular its local index1 must not become actual1.
                    return ExitTaskResult('CANONICAL_NORMAL_WITNESS_UNAVAILABLE', _ro(first.x[:21]), (), diagnostics,
                        completed_prefix=0, certified_prefix=(), fully_converged=False, failed_layer=None)
                # Only the ORIGINAL normal optimum may be an actual backend
                # prefix. NEVER freeze [I21,0] from the canonical selector.
                first_certs = (dict(witness_certs[0], index=0, solver_local_index=0,
                    task_stage='normal_slack', certificate_role='ORIGINAL_NORMAL_PREFIX_ONLY'),)
                levels = (first.levels[0],)
                xA = np.asarray(first.x).copy()
                if xA.shape != (21+P,) or not np.isfinite(xA).all(): raise FloatingPointError('invalid_certified_canonical_normal_anchor')
                if formulation == 'backend':
                    lower[21:] = xA[21:]; upper[21:] = xA[21:]
            fixed_slack = xA[21:]*s
            M = e = munit = meffective = motion_A = motion_upper = anchor_motion = None
            motion = dict(available=False, reason='unknown_public_motion_task_omitted_not_zero', witness_reference_delta21=_ro(xA[:21]*units))
            present = (motion_matrix_Mx21 is not None, motion_error_M_m_s is not None)
            if any(present) and not all(present): raise _Unknown('motion_matrix_and_error_must_be_co_declared')
            if all(present):
                stamp = _scalar(motion_time_s, 'motion_time_s')
                if abs(stamp-now) > 1e-9: raise ValueError('motion_current_snapshot_clock_mismatch')
                rawM = np.asarray(motion_matrix_Mx21, dtype=object)
                if rawM.ndim != 2 or rawM.shape[1] != 21 or not 1 <= len(rawM) <= MAX_MOTION_ROWS: raise ValueError('motion_matrix_requires_M_by21')
                M = _array(rawM, rawM.shape, 'public_motion_matrix'); e = _array(motion_error_M_m_s, (len(M),), 'acquired_motion_projection')
                munit = np.sum(abs(M)*units[None, :], axis=1); zero = munit == 0
                if np.any(zero & np.any(M != 0, axis=1)): raise FloatingPointError('motion_units_underflow_not_zero_authority')
                if motion_row_scale_M is None: meffective = np.where(zero, 1., munit)
                else:
                    meffective = _array(motion_row_scale_M, (len(M),), 'explicit_motion_row_scale')
                    if np.any(meffective <= 0): raise ValueError('positive_motion_row_units_required')
                m = (M*units[None, :])/meffective[:, None]
                offset = dt*e/meffective
                anchor_motion = m@xA[:21]+offset
                bound = abs(anchor_motion)
                motion_A = np.zeros((2*len(M), 21+P)); motion_A[:len(M), :21] = m; motion_A[len(M):, :21] = -m
                motion_upper = np.r_[bound-offset, bound+offset]
                motion = dict(available=True, construction_time_s=stamp, matrix_Mx21=_ro(M), error_M_m_s=_ro(e),
                    row_units_M=_ro(munit), effective_row_units_M=_ro(meffective),
                    witness_normalized_x21=_ro(xA[:21]), witness_reference_delta21=_ro(xA[:21]*units),
                    witness_normalized_residual_M=_ro(anchor_motion), normalized_absolute_budget_M=_ro(bound),
                    budget_inequality_matrix=_ro(motion_A), budget_inequality_upper=_ro(motion_upper),
                    zero_local_authority_rows=np.flatnonzero(zero).tolist(), actual_external_velocity_guaranteed=False)
            elif motion_row_scale_M is not None or motion_time_s is not None:
                raise ValueError('motion_metadata_without_motion_task')
            hard_A = normal_A if motion_A is None else np.vstack((normal_A, motion_A))
            hard_upper = normal_upper if motion_upper is None else np.r_[normal_upper, motion_upper]
            # Scalar pressure-share withdrawal, bounded target above all box
            # values; LS to this bound maximizes the scalar in the TASK domain.
            normal_work = -shares@Nunit if P else np.zeros(21)
            work_box_upper = float(np.sum(np.where(normal_work >= 0, normal_work*upper[:21], normal_work*lower[:21])))
            work_scale = max(float(np.max(abs(normal_work), initial=0.)), abs(work_box_upper))
            if work_scale == 0: work_scale = 1.
            row = np.zeros((1, 21+P)); row[0, :21] = normal_work/work_scale
            joint_level = None
            if formulation == 'joint' and (P or M is not None):
                # The historical correction optimum is NOT submitted as an
                # accepted prefix for this different simultaneous objective.
                if P and M is not None:
                    domain = first_certs[0]['hard_domain']
                    projection = np.zeros((len(M),21+P))
                    projection[:,:21] = (M*units[None,:])/meffective[:,None]
                    bundle = assemble(lower=unfrozen_lower,upper=unfrozen_upper,
                        normal_cost_A=cost,normal_target_b=np.zeros(P),
                        projection_A=projection,projection_target_b=-dt*e/meffective,
                        original_base_G=domain['original_inequality_matrix_with_box'],
                        original_base_h=domain['original_inequality_upper_with_box'],
                        original_fixed_E=domain['fixed_equality_matrix'],
                        original_fixed_f=domain['fixed_equality_target'],
                        witnessed_motion_G=motion_A,witnessed_motion_h=motion_upper)
                    joint_level=Level('joint_normal_and_observed_projection',bundle.A,bundle.b)
                elif P:
                    # Motion UNKNOWN is omitted explicitly, never a zero
                    # observed velocity. Only the known correction task exists.
                    joint_level=Level('normal_correction_motion_unknown',cost,np.zeros(P))
                else:
                    projection=np.zeros((len(M),21))
                    projection[:,:21]=(M*units[None,:])/meffective[:,None]
                    factor=1/np.sqrt(len(M))
                    joint_level=Level('observed_projection_without_positive_normal_rows',
                        factor*projection,-factor*dt*e/meffective)
            later_levels = [] if joint_level is None else [joint_level]
            later_levels.append(Level('scalar_pressure_share_normal_withdrawal', row, np.array([work_box_upper/work_scale])))
            if M is not None and formulation == 'backend':
                matrix = np.zeros((len(M), 21+P)); matrix[:, :21] = (M*units[None, :])/meffective[:, None]
                later_levels.append(Level('public_external_projection_low_motion', matrix, -dt*e/meffective))
            posture = np.zeros((21, 21+P)); posture[:, :21] = np.eye(21)
            later_levels.append(Level('canonical_or_explicit_path_reference', posture, chosen_target/units))
            second = solve_hierarchy(lower=lower, upper=upper, levels=tuple(later_levels), initial=xA,
                inequality_matrix=hard_A, inequality_upper=hard_upper,
                max_iterations=int(max_iterations), atol=atol, rtol=rtol)
            submitted_first_certs = first_certs if formulation == 'backend' else ()
            later_certs = tuple(dict(c, index=i+len(submitted_first_certs), solver_local_index=c['index'],
                                    task_stage='joint_motion_budget_exit' if formulation=='joint' else 'motion_budget_exit')
                                for i, c in enumerate(_certificates(second)))
            certs = submitted_first_certs+later_certs
            completed = len(certs); extended = np.asarray(second.x)
            # A failed first later layer still retains the certified L1 anchor.
            if not second.available and first_certs: extended = xA.copy()
            failed = second.failed_layer
            if failed is not None:
                failed = dict(failed, index=int(failed['index'])+len(submitted_first_certs))
            final_levels = (levels if formulation=='backend' else ())+tuple(second.levels)
            fully = (first is None or first.fully_converged) and second.fully_converged
            status = 'CONVERGED_FULL_CONTACT_EXIT_HIERARCHY' if fully else 'CERTIFIED_CONTACT_EXIT_PREFIX' if completed else second.status
            delta = extended[:21]*units
            final_eta = extended[21:]; final_r = final_eta*s
            reported_slack = fixed_slack if formulation=='backend' else final_r
            reported_eta = xA[21:] if formulation=='backend' else final_eta
            diagnostic = dict(_scope(), time_s=now, dt_s=dt, actual_increment_available=completed > 0,
                formulation=formulation,normal_witness_is_submitted_prefix=False,
                normal_first_certificate_is_submitted_prefix=bool(P and formulation=='backend'),
                canonical_witness_required=bool(P), canonical_witness_available=bool(P),
                canonical_witness_not_required_knownP0=not bool(P),
                canonical_witness_certified=bool(P), canonical_witness_certificate_submitted=False,
                normal_witness_selector=NORMAL_WITNESS_SELECTOR if P else 'NOT_REQUIRED_KNOWN_P0',
                normal_witness_budget_defined=M is not None,
                witness_failed_layer=None, witness_completed_prefix=len(witness_certs),
                actual_submitted_certificate_count=completed,
                normal_optimal_slack_frozen_before_motion=formulation=='backend',
                normal_witness_certified_prefix=witness_certs,
                normal_witness_levels=witness_levels,
                normal_witness_final_extended_anchor=_ro(xA),
                normal_witness_optimal_eta_P=_ro(xA[21:]),
                joint_normal_cost_rows=P if formulation=='joint' else None,
                joint_observed_motion_rows=0 if M is None else len(M),
                joint_motion_mean_square_factor=None if M is None else 1/len(M),
                motion_unknown_not_zero=M is None,
                completed_prefix=completed, fully_converged=bool(fully), failed_layer=failed,
                reference_step_units21=_ro(units), current_reference21=_ro(current), encoder_reference21=_ro(encoder),
                reference_debt21=_ro(debt), actual_step_lower21=_ro(lo), actual_step_upper21=_ro(hi),
                normal_jacobian_Px25=_ro(J), proposal_map25x21=_ro(B), normal_matrix_N_Px21=_ro(N),
                pressure_P_N=_ro(pressure), pressure_share_P=_ro(shares), normal_current_velocity_P_m_s=_ro(vn),
                normal_reference_debt_P_m=_ro(an), normal_correction_target_P_m=_ro(b),
                normal_row_units_P_m=_ro(row_units), normal_effective_report_units_P_m=_ro(s),
                normal_slack_upper_P_m=_ro(r_upper), certified_slack_P_m=_ro(reported_slack),
                certified_slack_eta_P=_ro(reported_eta), extended_normalized_x=_ro(extended),
                final_slack_P_m=_ro(final_r), final_normal_residual_P_m=_ro(N@delta-b),
                normal_one_sided_bound_P_m=_ro(b+reported_slack),
                zero_local_normal_authority_rows=np.flatnonzero(zero_normal).tolist(),
                initial_normalized_anchor=_ro(anchor0), certified_L1_extended_anchor=_ro(xA),
                current_reference_outside_task_box=not bool(np.all(lo <= 0)&np.all(hi >= 0)),
                second_stage_lower=_ro(lower), second_stage_upper=_ro(upper),
                second_stage_hard_matrix=_ro(hard_A), second_stage_hard_upper=_ro(hard_upper),
                motion_task=motion, scalar_normal_work_row=_ro(normal_work), scalar_normal_work_box_upper_m=work_box_upper,
                scalar_normal_work_attained_m=float(normal_work@extended[:21]),
                delta_reference21=_ro(delta), reference21=_ro(current+delta),
                original_canonical_target_delta21=_ro(original_target), effective_posture_target_delta21=_ro(chosen_target),
                explicit_path_target_active=path_reference_target_delta21 is not None,
                first_stage_diagnostics=None if first is None else first.diagnostics,
                second_stage_diagnostics=second.diagnostics,
                task_matrices_and_targets=tuple(dict(name=c['name'], matrix=c['matrix'], target=c['target'],
                    attained=c['attained'], residual=c['residual'], squared_loss=c['squared_loss']) for c in certs),
                numerical_budget=dict(max_iterations=int(max_iterations), atol=atol, rtol=rtol),
                upper_layer_failure_not_convergence=True, certified_prefix_not_full_optimality=not fully,
                one_m_zero_row_unit_is_reporting_only=True)
            return ExitTaskResult(status, _ro(extended[:21]), final_levels, diagnostic,
                completed_prefix=completed, certified_prefix=certs, fully_converged=bool(fully), failed_layer=failed)
    except _Unknown as exc: return _failure('UNKNOWN_CONTACT_EXIT_INPUT', exc)
    except (ValueError, TypeError, AttributeError, KeyError) as exc: return _failure('INVALID_CONTACT_EXIT_INPUT', exc)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc: return _failure('UNAVAILABLE_NUMERICAL_CONTACT_EXIT', exc)

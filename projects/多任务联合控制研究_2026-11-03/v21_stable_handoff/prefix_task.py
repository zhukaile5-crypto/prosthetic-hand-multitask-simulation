"""V20-task prefix-only reference hierarchy; pure numerical task construction.

The sole variable is x = delta_reference21 / (reference_rates21 * .005).
Hard constraints are ONLY the caller's genuine current5ms reference box. It
may exclude delta0. The initial anchor is clip0 in that actual box. No native
model cell, force-B domain, pressure threshold, rho gate or noninward hard
inequality is invented here. There is no target identity or object state.

Jn: P by25 normal robot Jacobian, hand->external/inward positive, at current
robot q. H:25by21 DECLARED kinematic reference-to-length distribution. N=Jn H
is a reference proposal, NOT servo response or predicted point displacement.
Full current25 encoder velocity is retained, including uncontrolled tendon
null-space motion. The25 mechanical coordinates are not21 controls.

For every positive-pressure patch:
  a = current_reference21 - encoder_reference21
  v = Jn @ current_robot_velocity25
  d = N @ a
  c = max(0, -min_actual_box(N du)) / dt
  b = -dt * [c + max(v,0) + max(d/.020,0)]
The .020 is the existing software reference-return budget, not a servo time
constant. Outward measured velocity and outward virtual reference debt NEVER
cancel the new withdrawal request while positive contact is observed.
encoder_reference21 must be in the same known length/reference coordinates;
the lift actor's gravity bias is excluded by the caller, not subtracted twice.

L1: min ||sqrt(p/sum(p)) * (N du-b)/normal_row_unit||^2.
normal_row_unit=sum(abs(N*reference_rates*dt)), a fixed declared reference
unit, not box capacity. A known zero row uses an explicitly declared1m unit
only to retain its unavoidable constant residual and is labelled zero local
authority. There is no epsilon floor and no unknown substituted by zero.
L2, if supplied: min ||(M du + dt*e)/motion_row_unit||^2. M/e are caller-built
public external-projection TASKS, not an object-motion response model. Unknown
motion is omitted with an explicit diagnostic, not a fabricated e=0. Default
units use sum(abs(M*reference_rates*dt)); exact zero rows retain constants in
an explicit1m unit. Positive caller units may instead be declared.
L3: min ||du/(rates*dt)-reference_target_delta21/(rates*dt)||^2. The raw
reference target is NOT projected into the box. Caller supplies OPEN finger
intent, actual governor lift intent, and wrist preferred=current (delta0).
Minimal wrist action is therefore an explicit part of L3. There is no L4.

The frozen V17 active-set solver freezes every attained upper-layer residual.
Default400 iterations/layer, atol1e-9/rtol1e-8 are declared numerical budgets.
Original box and all frozen residuals are reported. Failure is unavailable;
unknown x is NaN, never a made-up stationary action. There are no physical
safety, actual stopping, low pressure, clearance, placement or global claims.
"""
from numbers import Real,Integral
from dataclasses import replace
import numpy as np
from .prefix_hierarchy import Level,PrefixHierarchyResult as HierarchyResult,solve_hierarchy

from .design import CONTROL_PERIOD_S,COMMAND_DEBT_INTENT_PERIOD_S
REFERENCE_RETURN_BUDGET_S=COMMAND_DEBT_INTENT_PERIOD_S
MAX_PATCHES=128
MAX_MOTION_ROWS=384
ZERO_ROW_REPORT_UNIT_M=1.0

class _Unknown(ValueError):pass

def _scalar(value,name,*,positive=False,nonnegative=False):
    if value is None:raise _Unknown(name+'_unknown')
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real):raise ValueError(name+'_requires_real')
    value=float(value)
    if not np.isfinite(value) or positive and value<=0 or nonnegative and value<0:raise ValueError(name+'_outside_finite_domain')
    return value

def _array(value,shape,name):
    if value is None:raise _Unknown(name+'_unknown')
    raw=np.asarray(value,dtype=object)
    if raw.shape!=shape or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) for x in raw.flat):raise ValueError(name+'_requires_real_shape_'+str(shape))
    out=np.asarray(value,float).copy()
    if not np.isfinite(out).all():raise ValueError(name+'_nonfinite')
    return out

def _ro(value):
    out=np.asarray(value).copy();out.setflags(write=False);return out

def _scope():
    return dict(task_role='CURRENT_CONTACT_REFERENCE_EXIT_HIERARCHY',
        reference_map_is_declared_proposal_not_response=True,
        pressure_prediction_available=False,future_robot_response_predicted=False,
        physical_safety_certified=False,placement_certified=False,
        actual_stopping_certified=False,holding_support_requested=False,
        no_reaction_share_gate=True,no_hard_contact_noninward_constraint=True,
        current_reference_origin_preserved=True,full_robot25_velocity_retained=True,
        passive_tendon_nullspace_is_not_an_extra_control=True,
        CONTROL_PERIOD_S=CONTROL_PERIOD_S,reference_return_budget_s=REFERENCE_RETURN_BUDGET_S,
        reference_return_budget_is_not_servo_time_constant=True,
        L3_includes_minimal_wrist_preference=True,implicit_L4=False)

def _failure(status,reason):
    return HierarchyResult(status,_ro(np.full(21,np.nan)),(),dict(_scope(),reason=str(reason),
        actual_increment_available=False,reference21=None,delta_reference21=None))

def solve_contact_exit(*,time_s,dt_s,current_reference21,encoder_reference21,
    step_lower21,step_upper21,reference_rates21,normal_jacobian_Px25,
    proposal_map25x21,robot_velocity25,pressure_P_N,reference_target_delta21,
    motion_matrix_Mx21=None,motion_error_M_m_s=None,motion_row_scale_M=None,
    motion_time_s=None,max_iterations=400,atol=1e-9,rtol=1e-8):
    """Return frozen HierarchyResult; x is NORMALIZED reference delta21.

    Stateful clock/generation/observation provenance, q/u age, encoder length
    axes/units, native encoding/rates/receipt and actual mode qualification are
    owned by the caller. This stateless task kernel accepts repeated offline
    same-input queries. Native force clipping does not turn H into a response.
    """
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            now=_scalar(time_s,'time_s',nonnegative=True);dt=_scalar(dt_s,'dt_s',positive=True)
            if dt!=CONTROL_PERIOD_S:raise ValueError('exact_current5ms_control_period_required')
            if isinstance(max_iterations,(bool,np.bool_)) or not isinstance(max_iterations,Integral) or max_iterations<1:raise ValueError('positive_integer_iteration_budget_required')
            max_iterations=int(max_iterations);atol=_scalar(atol,'atol',positive=True);rtol=_scalar(rtol,'rtol',nonnegative=True)
            current=_array(current_reference21,(21,),'current_reference21');encoder=_array(encoder_reference21,(21,),'encoder_reference21')
            lo=_array(step_lower21,(21,),'actual_step_lower21');hi=_array(step_upper21,(21,),'actual_step_upper21')
            if np.any(lo>hi):raise ValueError('empty_actual_reference_box')
            rates=_array(reference_rates21,(21,),'reference_rates21')
            if np.any(rates<=0):raise ValueError('positive_reference_units_required')
            units=rates*dt
            if np.any(units==0):raise FloatingPointError('positive_reference_units_underflow')
            h=_array(proposal_map25x21,(25,21),'declared_proposal_map25x21');v25=_array(robot_velocity25,(25,),'robot_velocity25')
            if normal_jacobian_Px25 is None:raise _Unknown('normal_geometry_unknown_not_no_contact')
            raw=np.asarray(normal_jacobian_Px25,dtype=object)
            if raw.ndim!=2 or raw.shape[1]!=25 or len(raw)>MAX_PATCHES:raise ValueError('normal_geometry_requires_P_by25_with_P_at_most128')
            P=len(raw);jn=_array(normal_jacobian_Px25,(P,25),'normal_jacobian_Px25');pressure=_array(pressure_P_N,(P,),'pressure_P_N')
            if np.any(pressure<=0):raise ValueError('explicit_positive_pressure_rows_required')
            reference_target=_array(reference_target_delta21,(21,),'raw_reference_target_delta21')
            scaled_lo=lo/units;scaled_hi=hi/units;anchor=np.clip(np.zeros(21),scaled_lo,scaled_hi)
            N=jn@h;debt21=current-encoder;vnormal=jn@v25;normal_debt=N@debt21
            normal_box_min=np.sum(np.where(N>=0,N*lo[None,:],N*hi[None,:]),axis=1)
            capacity=np.maximum(0.,-normal_box_min)/dt
            target=-dt*(capacity+np.maximum(vnormal,0.)+np.maximum(normal_debt/REFERENCE_RETURN_BUDGET_S,0.))
            normal_units=np.sum(abs(N)*units[None,:],axis=1);normal_zero=normal_units==0
            if np.any(normal_zero & np.any(N!=0,axis=1)):raise FloatingPointError('normal_reference_row_units_underflow_not_zero_authority')
            report_units=np.where(normal_zero,ZERO_ROW_REPORT_UNIT_M,normal_units)
            if P:
                relative=pressure/float(np.max(pressure))
                if np.any(relative==0):raise FloatingPointError('positive_pressure_share_underflow')
                shares=relative/float(relative.sum())
                if np.any(shares==0):raise FloatingPointError('positive_pressure_share_underflow')
                weights=np.sqrt(shares)
            else:shares=weights=np.empty(0)
            A1=weights[:,None]*(N*units[None,:])/report_units[:,None];b1=weights*target/report_units
            levels=[]
            if P:levels.append(Level('pressure_share_current_contact_exit',A1,b1))
            motion=dict(available=False,reason='public_external_projection_task_unknown_omitted_not_zero',
                matrix_Mx21=None,error_M_m_s=None,row_units_M=None,effective_row_units_M=None,
                zero_local_authority_rows=[],construction_time_s=None)
            present=(motion_matrix_Mx21 is not None,motion_error_M_m_s is not None)
            if any(present) and not all(present):raise _Unknown('motion_matrix_and_error_must_be_co_declared')
            if all(present):
                stamp=_scalar(motion_time_s,'motion_task_time_s',nonnegative=True)
                if abs(stamp-now)>1e-9:raise ValueError('motion_task_current_snapshot_clock_mismatch')
                mr=np.asarray(motion_matrix_Mx21,dtype=object)
                if mr.ndim!=2 or mr.shape[1]!=21 or not 1<=len(mr)<=MAX_MOTION_ROWS:raise ValueError('motion_matrix_requires_M_by21')
                M=_array(motion_matrix_Mx21,mr.shape,'motion_matrix_Mx21');e=_array(motion_error_M_m_s,(len(M),),'motion_error_M_m_s')
                mu=np.sum(abs(M)*units[None,:],axis=1);zero_motion=mu==0
                if np.any(zero_motion & np.any(M!=0,axis=1)):raise FloatingPointError('motion_reference_row_units_underflow_not_zero_authority')
                if motion_row_scale_M is None:effective=np.where(zero_motion,ZERO_ROW_REPORT_UNIT_M,mu);unit_source='declared_reference_rate_step_row_units_with_explicit1m_for_known_zero_rows'
                else:
                    effective=_array(motion_row_scale_M,(len(M),),'motion_row_scale_M')
                    if np.any(effective<=0):raise ValueError('motion_row_scale_requires_explicit_positive_units')
                    unit_source='caller_declared_positive_row_units'
                levels.append(Level('public_external_projection_low_motion',(M*units[None,:])/effective[:,None],-dt*e/effective))
                motion=dict(available=True,reason=None,matrix_Mx21=_ro(M),error_M_m_s=_ro(e),
                    row_units_M=_ro(mu),effective_row_units_M=_ro(effective),unit_source=unit_source,
                    zero_local_authority_rows=np.flatnonzero(zero_motion).tolist(),construction_time_s=stamp,
                    zero_rows_keep_constant_residual=True,not_external_response_model=True)
            elif motion_row_scale_M is not None or motion_time_s is not None:raise ValueError('motion_metadata_without_motion_task')
            levels.append(Level('canonical_reference_intent_and_minimal_wrist_action',np.eye(21),reference_target/units))
            for level in levels:
                if not np.isfinite(level.matrix).all() or not np.isfinite(level.target).all():raise FloatingPointError('nonfinite_declared_task')
            raw_result=solve_hierarchy(lower=scaled_lo,upper=scaled_hi,levels=tuple(levels),initial=anchor,
                max_iterations=max_iterations,atol=atol,rtol=rtol)
            delta=raw_result.x*units;reference=current+delta
            if not np.isfinite(np.r_[delta,reference]).all():raise FloatingPointError('unrepresentable_reference_result')
            recorded=[]
            for level in levels:
                attained=level.matrix@raw_result.x;residual=attained-level.target
                loss=float(residual@residual)
                if not np.isfinite(loss):raise FloatingPointError('unrepresentable_task_loss')
                recorded.append(dict(name=level.name,matrix=_ro(level.matrix),target=_ro(level.target),attained=_ro(attained),
                    residual=_ro(residual),residual_norm=float(np.linalg.norm(residual)),squared_loss=loss))
            bounds_exact=bool(np.all(delta>=lo) and np.all(delta<=hi))
            diag=dict(raw_result.diagnostics,**_scope(),time_s=now,dt_s=dt,actual_increment_available=raw_result.available,
                current_reference21=_ro(current),encoder_reference21=_ro(encoder),reference_debt21=_ro(debt21),
                reference_step_units21=_ro(units),actual_step_lower21=_ro(lo),actual_step_upper21=_ro(hi),
                normalized_anchor21=_ro(anchor),anchor_delta21=_ro(anchor*units),anchor_reference21=_ro(current+anchor*units),
                zero_increment_in_actual_box=bool(np.all(lo<=0)&np.all(hi>=0)),
                delta_reference21=_ro(delta),reference21=_ro(reference),exact_original_step_box_satisfied=bounds_exact,
                normal_jacobian_Px25=_ro(jn),declared_proposal_map25x21=_ro(h),current_robot_velocity25=_ro(v25),
                normal_matrix_N_Px21=_ro(N),pressure_P_N=_ro(pressure),pressure_share_P=_ro(shares),normal_weight_P=_ro(weights),
                normal_current_velocity_P_m_s=_ro(vnormal),normal_reference_debt_P_m=_ro(normal_debt),
                normal_box_min_increment_P_m=_ro(normal_box_min),normal_outward_box_capacity_P_m_s=_ro(capacity),
                normal_target_b_P_m=_ro(target),normal_reference_row_units_P_m=_ro(normal_units),
                normal_effective_report_units_P_m=_ro(report_units),zero_local_normal_authority_rows=np.flatnonzero(normal_zero).tolist(),
                normal_geometry_role='KNOWN_NO_ACTIVE_POSITIVE_PRESSURE_ROWS' if P==0 else 'CURRENT_ROBOT_OLD_PLANE_REFERENCE_INTENT',
                zero_normal_rows_keep_unavoidable_constant_residual=True,
                one_m_zero_row_unit_is_not_epsilon_or_physical_threshold=True,
                outward_velocity_and_outward_debt_not_credited_as_new_exit=True,
                normal_attained_increment_P_m=_ro(N@delta),normal_task_residual_P_m=_ro(N@delta-target),
                motion_task=motion,raw_reference_target_delta21=_ro(reference_target),
                raw_reference_goal_projected=False,task_matrices_and_targets=tuple(recorded),
                task_priority=tuple(level.name for level in levels),no_hidden_regularizer=True,
                numerical_budget=dict(max_iterations=max_iterations,atol=atol,rtol=rtol),
                solver_status_is_not_physical_clearance_or_infeasibility=True)
            return replace(raw_result,diagnostics=diag)
    except _Unknown as exc:return _failure('UNKNOWN_CONTACT_EXIT_INPUT',exc)
    except (ValueError,TypeError,AttributeError) as exc:return _failure('INVALID_CONTACT_EXIT_INPUT',exc)
    except (FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:return _failure('UNAVAILABLE_NUMERICAL_CONTACT_EXIT',exc)

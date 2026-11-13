"""Experimental robot-relative directional allocation, never a force certificate.

Priority: original user tube -> old-plane inward geometry -> least change of
the stabilizing proposal. All optimization happens inside finite action bounds;
fixed coordinates are removed first. Missing models return legal nominal input.
"""
from __future__ import annotations
from dataclasses import dataclass
from numbers import Real
import numpy as np
from v9.feedback import _qp

ROBOT_AXES25 = ("rh_WRJ2", "rh_WRJ1", "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1", "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1", "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1", "transport_lift")


@dataclass(frozen=True)
class Result:
    increment8: np.ndarray
    predicted_robot25: np.ndarray | None
    debug: dict


def _array(value, shape, name):
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw.flat):
        raise ValueError(name + ": real numeric values required, no boolean or numeric strings")
    result = np.asarray(value, float).copy()
    if not np.isfinite(result).all():
        raise ValueError(name + ": finite real values required")
    return result


def _tolerances(value):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (3,):raise ValueError("tracking_tolerances3 requires three explicit values or None entries")
    valid = np.array([v is not None for v in raw], bool)
    result = np.zeros(3)
    for i in np.flatnonzero(valid):
        result[i] = _array([raw[i]], (1,), "tracking tolerance")[0]
    if np.any(result < 0):raise ValueError("Negative public tracking tolerance")
    return result, valid


class DirectionalAllocator:
    def solve(self, *, nominal_increment8, step_lower8, step_upper8,
              robot_free25, robot_response25x8, output_axes25,
              user_target3, tracking_tolerances3, empirical_robot_error25=None,
              contact_rows_mx8=None, contact_free_delta_m=None,
              contact_error_m=None, contact_urgency_m=None,
              model_scope_label="EXPERIMENTAL_LOCAL", proposal_lower8=None,
              proposal_upper8=None, proposal_l1_weights8=None,
              proposal_l1_budget=None):
        nominal = _array(nominal_increment8, (8,), "nominal_increment8")
        actual_lo = _array(step_lower8, (8,), "step_lower8")
        actual_hi = _array(step_upper8, (8,), "step_upper8")
        if np.any(actual_lo > actual_hi):raise ValueError("Empty actual mechanical/rate/domain interval")
        target = _array(user_target3, (3,), "original user_target3")
        tolerance, tube_known = _tolerances(tracking_tolerances3)
        if not isinstance(model_scope_label, str) or not model_scope_label.strip():
            raise ValueError("Explicit nonempty model scope label required")
        if (proposal_lower8 is None) != (proposal_upper8 is None):
            raise ValueError("Both model proposal bounds or neither must be supplied")
        pl = None if proposal_lower8 is None else _array(proposal_lower8, (8,), "proposal_lower8")
        pu = None if proposal_upper8 is None else _array(proposal_upper8, (8,), "proposal_upper8")
        if pl is not None and np.any(pl > pu):raise ValueError("Internally inconsistent model proposal interval")
        if (proposal_l1_weights8 is None) != (proposal_l1_budget is None):
            raise ValueError("Both empirical proposal L1 weights and budget or neither required")
        lw = None if proposal_l1_weights8 is None else _array(proposal_l1_weights8, (8,), "proposal_l1_weights8")
        lb = None if proposal_l1_budget is None else float(_array([proposal_l1_budget], (1,), "proposal_l1_budget")[0])
        if lw is not None and (np.any(lw < 0) or lb < 0):raise ValueError("Nonnegative empirical proposal L1 weights/budget required")
        if (robot_free25 is None) != (robot_response25x8 is None):
            raise ValueError("Robot free state and response must be present together or both unavailable")
        model_available = robot_free25 is not None
        axes = None if output_axes25 is None else tuple(output_axes25)
        if axes is not None and (len(axes) != 25 or set(axes) != set(ROBOT_AXES25)):
            raise ValueError("Explicit known robot25 axes required; object coordinates are not inputs")
        if model_available and axes is None:raise ValueError("Known model requires named robot25 outputs")
        free = None if not model_available else _array(robot_free25, (25,), "robot_free25")
        response = None if not model_available else _array(robot_response25x8, (25, 8), "robot_response25x8")
        robot_error = None if empirical_robot_error25 is None else _array(empirical_robot_error25, (25,), "empirical_robot_error25")
        if robot_error is not None and np.any(robot_error < 0):raise ValueError("Negative empirical robot error")
        if contact_rows_mx8 is None:
            if any(v is not None for v in (contact_free_delta_m, contact_error_m, contact_urgency_m)):
                raise ValueError("Unavailable contact rows cannot carry invented residual arrays")
            G = np.empty((0, 8)); c = error = urgency = np.empty(0)
            geometry_status = "UNKNOWN_ROWS_NOT_SUPPLIED"
        else:
            raw = np.asarray(contact_rows_mx8, dtype=object)
            if raw.ndim != 2 or raw.shape[1] != 8:raise ValueError("contact_rows_mx8 requires m by8 geometry rows")
            m = len(raw); G = _array(contact_rows_mx8, (m, 8), "contact_rows_mx8")
            if m == 0:
                c = _array([] if contact_free_delta_m is None else contact_free_delta_m, (0,), "contact_free_delta_m")
                error = _array([] if contact_error_m is None else contact_error_m, (0,), "contact_error_m")
                urgency = _array([] if contact_urgency_m is None else contact_urgency_m, (0,), "contact_urgency_m")
                geometry_status = "KNOWN_EMPTY_RISK_SET"
            else:
                c = _array(contact_free_delta_m, (m,), "contact_free_delta_m")
                urgency = _array(contact_urgency_m, (m,), "contact_urgency_m")
                error = None if contact_error_m is None else _array(contact_error_m, (m,), "contact_error_m")
                if np.any(urgency < 0) or error is not None and np.any(error < 0):raise ValueError("Nonnegative geometry error and measured urgency required")
                geometry_status = "MEASURED_ROWS_WITH_EMPIRICAL_ERROR" if error is not None else "GEOMETRY_ERROR_UNKNOWN"
        legal_nominal = np.minimum(actual_hi, np.maximum(actual_lo, nominal))
        active = np.flatnonzero(urgency > 0) if error is not None else np.empty(0, int)
        motion_indices = None if axes is None else [axes.index(n) for n in ("transport_lift", "rh_WRJ2", "rh_WRJ1")]
        motion_margin = None if robot_error is None or motion_indices is None else robot_error[motion_indices]
        tube_scale = np.maximum(tolerance, 1e-6)
        stages = []; search_lo = actual_lo.copy(); search_hi = actual_hi.copy()
        debug_base = dict(model_scope_label=model_scope_label, geometry_status=geometry_status,
            user_target3=target.tolist(), tracking_tolerances3=list(tracking_tolerances3),
            known_tracking_tolerance3=tube_known.tolist(), tube_empirical_error_known=motion_margin is not None,
            empirical_robot_error25=None if robot_error is None else robot_error.tolist(),
            actual_step_lower8=actual_lo.tolist(), actual_step_upper8=actual_hi.tolist(),
            proposal_lower8=None if pl is None else pl.tolist(), proposal_upper8=None if pu is None else pu.tolist(),
            proposal_l1_weights8=None if lw is None else lw.tolist(), proposal_l1_budget=lb,
            proposal_trust_region_scope="empirical model proposal search only; not mechanical authority or mixed-sequence certification",
            zero_geometry_rows=np.flatnonzero(np.all(G == 0, axis=1)).tolist(),
            active_geometry_rows=active.tolist(), pressure_prediction_available=False,
            pressure_safety_certified=False, global_model_admitted=False,
            force_safety_slack_N=None, geometry_scope="robot-relative-to-old-acquisition-plane intent, not pressure or current external gap")

        def diagnostics(u, status, **extra):
            prediction = None if not model_available else free + response @ u
            if prediction is not None and not np.isfinite(prediction).all():raise ValueError("Robot prediction arithmetic overflow")
            motion_error = None if prediction is None else prediction[motion_indices] - target
            point_violation = None if motion_error is None else np.where(tube_known, np.maximum(0., abs(motion_error) - tolerance), np.nan)
            empirical_violation = None if motion_error is None or motion_margin is None else np.where(tube_known, np.maximum(0., abs(motion_error) + motion_margin - tolerance), np.nan)
            geometry = G @ u + c
            guarded_geometry = None if error is None else geometry + error
            info = dict(debug_base, status=status, stages=stages,
                unchanged_nominal=bool(np.array_equal(u, nominal)), nominal_real_bound_adjustment=bool(not np.array_equal(legal_nominal, nominal)),
                model_prediction_available=model_available, model_proposal_search_lower8=search_lo.tolist(), model_proposal_search_upper8=search_hi.tolist(),
                tube_actual_signed_error3=None if motion_error is None else motion_error.tolist(),
                tube_point_violation3=None if point_violation is None else [None if not tube_known[i] else float(point_violation[i]) for i in range(3)],
                tube_empirical_violation3=None if empirical_violation is None else [None if not tube_known[i] else float(empirical_violation[i]) for i in range(3)],
                geometric_point_residual_m=geometry.tolist(), geometric_empirical_residual_m=None if guarded_geometry is None else guarded_geometry.tolist(),
                unresolved_geometry_m=None if guarded_geometry is None else np.maximum(0., guarded_geometry).tolist(),
                proposal_l1_value=None if lw is None else float(lw @ abs(u)),
                real_bound_primal_residual=max(0., float(np.max(actual_lo - u)), float(np.max(u - actual_hi))), **extra)
            return Result(u.copy(), None if prediction is None else prediction.copy(), info)

        if not model_available:return diagnostics(legal_nominal, "UNAVAILABLE_ROBOT_MODEL_NOMINAL")
        if not len(active):return diagnostics(legal_nominal, "NOMINAL_NO_VALID_RISK_ROWS")
        signed_nominal = (free + response @ legal_nominal)[motion_indices] - target
        margin = np.zeros(3) if motion_margin is None else motion_margin
        nominal_tube = np.where(tube_known, np.maximum(0., abs(signed_nominal) + margin - tolerance), 0.)
        nominal_geometry = G[active] @ legal_nominal + c[active] + error[active]
        if np.max(nominal_tube, initial=0.) <= 1e-12 and np.max(nominal_geometry, initial=0.) <= 1e-12:
            return diagnostics(legal_nominal, "NOMINAL_ALREADY_IN_TASK_SET")
        if pl is not None:
            search_lo = np.maximum(actual_lo, pl); search_hi = np.minimum(actual_hi, pu)
            if np.any(search_lo > search_hi):
                return diagnostics(legal_nominal, "UNAVAILABLE_MODEL_PROPOSAL_DOMAIN_NOMINAL", physical_infeasibility_certified=False)
        closest_zero = np.minimum(search_hi, np.maximum(search_lo, np.zeros(8)))
        if lw is not None and lw @ abs(closest_zero) > lb:
            return diagnostics(legal_nominal, "UNAVAILABLE_MODEL_PROPOSAL_L1_DOMAIN_NOMINAL", physical_infeasibility_certified=False)
        # Centered normalized action coordinates retain one-sided and fixed
        # mechanical domains without inverse-then-clip. No tiny-span division.
        center = search_lo * .5 + search_hi * .5
        half = search_hi * .5 - search_lo * .5
        fixed = search_lo == search_hi; variables = np.flatnonzero(~fixed); n = len(variables)
        center[fixed] = search_lo[fixed]
        if np.any(~np.isfinite(center)) or np.any(~np.isfinite(half)) or np.any(half[~fixed] <= 0):
            raise ValueError("Action normalization is not finite and resolvable")
        R = np.zeros((8, n)); R[variables, np.arange(n)] = half[variables]
        nt = int(tube_known.sum()); ng = len(active); dim = n + nt + ng + (8 if lw is not None else 0)
        T = np.zeros((8, dim)); T[:, :n] = R
        tube_axes = np.flatnonzero(tube_known); tube_slots = np.arange(n, n + nt)
        geo_slots = np.arange(n + nt, n + nt + ng)
        abs_slots = np.arange(n + nt + ng, dim)
        abs_scale = np.maximum(abs(search_lo), abs(search_hi)); abs_scale[abs_scale == 0] = 1.
        motion_map = response[motion_indices] @ T
        motion_offset = (free + response @ center)[motion_indices] - target
        geom_map = G[active] @ T; geom_offset = G[active] @ center + c[active] + error[active]
        geom_scale = max(1e-6, float(np.max(abs(geom_offset), initial=0.)), float(np.max(np.sum(abs(G[active] @ R), axis=1), initial=0.)))
        if not all(np.isfinite(a).all() for a in (motion_map, motion_offset, geom_map, geom_offset)):
            raise ValueError("Finite model/action inputs overflowed constraint arithmetic")
        rows = []; bounds = []
        def add(row, bound):rows.append(np.asarray(row, float)); bounds.append(float(bound))
        for j in range(n):
            row = np.zeros(dim); row[j] = 1.; add(row, 1.); add(-row, 1.)
        for slot in np.r_[tube_slots, geo_slots, abs_slots]:
            row = np.zeros(dim); row[slot] = -1.; add(row, 0.)
        for axis, slot in zip(tube_axes, tube_slots):
            for sign in (1., -1.):
                row = sign * motion_map[axis] / tube_scale[axis]; row[slot] -= 1.
                add(row, (tolerance[axis] - margin[axis] - sign * motion_offset[axis]) / tube_scale[axis])
        for i, slot in enumerate(geo_slots):
            row = geom_map[i] / geom_scale; row[slot] -= 1.; add(row, -geom_offset[i] / geom_scale)
        if lw is not None:
            for i, slot in enumerate(abs_slots):
                row = T[i] / abs_scale[i]; row[slot] -= 1.; add(row, -center[i] / abs_scale[i])
                row = -T[i] / abs_scale[i]; row[slot] -= 1.; add(row, center[i] / abs_scale[i])
            row = np.zeros(dim); row[abs_slots] = lw * abs_scale
            if not np.isfinite(row).all():raise ValueError("Proposal L1 normalization overflow")
            add(row, lb)
        A = np.array(rows).reshape(-1, dim); b = np.array(bounds)
        seed = np.zeros(dim); seed_u = np.minimum(search_hi, np.maximum(search_lo, legal_nominal))
        if lw is not None and lw @ abs(seed_u) > lb:
            least = float(lw @ abs(closest_zero)); requested = float(lw @ abs(seed_u))
            blend = max(0., (lb - least) / (requested - least))
            seed_u = closest_zero + blend * (seed_u - closest_zero)
        if n:seed[:n] = (seed_u[variables] - center[variables]) / half[variables]
        def complete(x):
            u = center + T @ x
            me = (free + response @ u)[motion_indices] - target
            x[tube_slots] = np.maximum(0., (abs(me[tube_axes]) + margin[tube_axes] - tolerance[tube_axes]) / tube_scale[tube_axes])
            x[geo_slots] = np.maximum(0., (G[active] @ u + c[active] + error[active]) / geom_scale)
            if lw is not None:x[abs_slots] = abs(u) / abs_scale
            return x
        seed = complete(seed); x = seed.copy(); stage_failed = False; stage_rejected = False
        def actual_slack(z):
            u = center + T @ z
            me = (free + response @ u)[motion_indices] - target
            ts = np.maximum(0., (abs(me[tube_axes]) + margin[tube_axes] - tolerance[tube_axes]) / tube_scale[tube_axes])
            gs = np.maximum(0., (G[active] @ u + c[active] + error[active]) / geom_scale)
            return ts, gs
        def stage(name, M, wanted, protect_slots):
            nonlocal x, A, b, stage_failed, stage_rejected
            previous = x.copy()
            try:
                candidate, info = _qp(2 * M.T @ M, -2 * M.T @ wanted, A, b, x)
            except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
                stages.append(dict(name=name, solver_exception=type(exc).__name__ + ": " + str(exc), independently_checked=False))
                stage_failed = stage_rejected = True; return False
            checked = bool(np.isfinite(candidate).all() and np.max(A @ candidate - b, initial=0.) <= 2e-7)
            physical = center + T @ candidate if checked else None
            if checked:
                # Re-evaluate original output equations, never trust supplied
                # solver.residual or its auxiliary slack values alone.
                ts, gs = actual_slack(candidate)
                checked &= bool(np.max(ts - candidate[tube_slots], initial=0.) <= 2e-7 and np.max(gs - candidate[geo_slots], initial=0.) <= 2e-7)
                checked &= bool(np.all(physical >= search_lo) and np.all(physical <= search_hi))
                if lw is not None:checked &= bool(lw @ abs(physical) <= lb + 1e-10 * max(1., lb))
            stages.append(dict(name=name, solver=dict(info), independently_checked=checked,
                original_normalized_primal_residual=None if not checked else float(max(0., np.max(A @ candidate - b, initial=0.)))))
            if not checked:
                x = previous; stage_failed = stage_rejected = True; return False
            x = complete(candidate)
            if not info.get("optimal", False):stage_failed = True; return False
            # Freeze actual attained slack components. Squared nonnegative
            # slack objective has a unique optimal slack vector; later stages
            # may improve it but may not trade one component for another.
            if len(protect_slots):
                protection = np.zeros((len(protect_slots), dim)); protection[np.arange(len(protect_slots)), protect_slots] = 1.
                A = np.vstack((A, protection)); b = np.r_[b, x[protect_slots] + 2e-9]
            return True
        if nt:
            M = np.zeros((nt, dim)); M[np.arange(nt), tube_slots] = 1.
            proceed = stage("original_user_tube", M, np.zeros(nt), tube_slots)
        else:proceed = True
        if proceed:
            # A common positive factor changes no minimizer and prevents
            # arbitrary numerical magnitude of the measured urgency ratios.
            urgency_scale = float(np.max(urgency[active]))
            M = np.zeros((ng, dim)); M[np.arange(ng), geo_slots] = np.sqrt(urgency[active] / urgency_scale)
            proceed = stage("old_plane_inward_geometry", M, np.zeros(ng), geo_slots)
        if proceed:
            actual_half = actual_hi * .5 - actual_lo * .5
            activity_scale = np.where(actual_half > 0, actual_half, 1.)
            M = T / activity_scale[:, None]; wanted = (nominal - center) / activity_scale
            stage("least_nominal_modification", M, wanted, np.empty(0, int))
        u = center + T @ x
        # Numerical candidates are accepted only after real equations above;
        # no post-solve inverse/clip correction is made to a reported optimum.
        if not np.isfinite(u).all() or np.any(u < actual_lo) or np.any(u > actual_hi):
            return diagnostics(legal_nominal, "NUMERICAL_RESULT_REJECTED_NOMINAL", physical_infeasibility_certified=False)
        ts, gs = actual_slack(x)
        status = "NUMERICAL_STAGE_REJECTED_CHECKED_PREVIOUS" if stage_rejected else "CHECKED_NONOPTIMAL_STAGE" if stage_failed else "EXPERIMENTAL_DIRECTIONAL_ALLOCATION"
        return diagnostics(u, status,
            independently_checked=True, lexicographic_optimal=not stage_failed,
            fixed_input_coordinates=np.flatnonzero(fixed).tolist(), action_normalization_halfspan8=half.tolist(),
            tube_scales3=tube_scale.tolist(), geometry_scale_m=geom_scale,
            protected_tube_slack3=[None if not tube_known[i] else float(ts[list(tube_axes).index(i)] * tube_scale[i]) for i in range(3)],
            physical_infeasibility_certified=False, geometry_empirical_error_is_hard_bound=False)


def solve(**kwargs):
    """Stateless convenience entry; same keyword contract as the allocator."""
    return DirectionalAllocator().solve(**kwargs)

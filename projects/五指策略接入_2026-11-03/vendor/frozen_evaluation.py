"""Independent truth-only evaluation for the unified artificial-reflex experiment.

The controller must never import or receive evaluate_state's output.  All force
quality endpoints use measured physical contact force, not filtered observation.
The frozen metadata/config provide shared task references and tolerances.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
import numpy as np


DEFAULT_CONFIG = {
    "stable_dwell": .200,
    "establishment_dwell": .100,
    "release_dwell": .200,
    "linear_speed_max": .005,
    "angular_speed_max": .15,
    "relative_error_max": .005,
    "orientation_error_max": float(np.deg2rad(5)),
    "contact_min": .20,
    "minimum_contacts": 2,
    "required_fingers": [0, 1],
    "table_support_max": .05,
    "lift_tolerance": .005,
    "force_tolerance": .02,
    "release_force_max": .20,
    "static_window": [8.2, 8.9],
    "static_window_sensitivity": [8.3, 8.8],
    "static_contact_fraction_min": 1.0,
    "static_coverage_min": .999,
    "first_stability_horizon": 8.9,
    "establishment_deadline": 9.0,
    "release_deadline": 1.5,
    "cv_mean_min": .20,
    "evaluation_sample_interval": .005,
    "clock_tolerance": 1e-8,
    "force_bandwidth_hz": 50.,
    "force_filter_span": .05,
    "require_full_substeps": False,
    "require_skin_force": False,
    "require_all_hand_release": False,
    "support_loss_dwell": .1,
    "transport_loss_dwell": .3,
}


def _dict(value):
    return asdict(value) if is_dataclass(value) else dict(value or {})


def _json(value):
    """Return standards-compliant JSON values; unavailable estimates stay null."""
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def evaluation_config(metadata=None, config=None):
    result = copy.deepcopy(DEFAULT_CONFIG)
    result.update(_dict((metadata or {}).get("evaluation_config")))
    result.update(_dict(config))
    return result


def _series(trace, key, count, shape=(), default=np.nan):
    value = np.asarray(trace.get(key, default), dtype=float)
    if value.shape == shape:
        value = np.broadcast_to(value, (count,) + shape)
    if value.shape != (count,) + shape:
        raise ValueError(f"{key}: expected {(count,) + shape}, got {value.shape}")
    return value


def _reference(trace, metadata, key, count, shape, default=np.nan):
    # Time-dependent references are supplied by the common task generator.
    reference = _dict(metadata.get("reference"))
    value = trace.get("reference_" + key, reference.get(key, default))
    return _series({key: value}, key, count, shape)


def _time(trace):
    t = np.asarray(trace["time"], dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)):
        raise ValueError("time must contain at least two finite synchronized samples")
    if not np.all(np.diff(t) > 0):
        raise ValueError("time must be strictly increasing")
    return t


def _confirm_times(t, mask, dwell, start=-np.inf, end=np.inf):
    """Confirmation endpoints of continuous runs; no backdating or sample count shortcut."""
    result = []
    run_start = None
    confirmed = False
    maximum_gap = float(np.median(np.diff(t))) * 1.51
    for i in range(len(t)):
        if i and t[i] - t[i - 1] > maximum_gap:
            run_start, confirmed = None, False
        if t[i] < start - 1e-10 or t[i] > end + 1e-10 or not mask[i]:
            run_start, confirmed = None, False
            continue
        if run_start is None:
            run_start = t[i]
        if not confirmed and t[i] - run_start >= dwell - 1e-10:
            result.append(float(t[i]))
            confirmed = True
    return result


def _first_confirmation(t, mask, dwell, start=-np.inf, end=np.inf):
    values = _confirm_times(t, mask, dwell, start, end)
    return values[0] if values else None


def _integral(t, y):
    return float(np.trapezoid(y, t)) if len(t) > 1 else 0.0


def _orientation_errors(rotation, reference, mode, axis):
    count = len(rotation)
    if mode in ("none", "free"):
        return np.zeros(count)
    finite = np.all(np.isfinite(rotation), axis=(1, 2)) & np.all(
        np.isfinite(reference), axis=(1, 2))
    errors = np.full(count, np.nan)
    if mode == "axis":
        dot = np.sum(rotation[:, :, axis] * reference[:, :, axis], axis=1)
        errors[finite] = np.arccos(np.clip(dot[finite], -1, 1))
    elif mode == "full":
        relative = np.einsum("nji,njk->nik", reference, rotation)
        cosine = (np.trace(relative, axis1=1, axis2=2) - 1) / 2
        errors[finite] = np.arccos(np.clip(cosine[finite], -1, 1))
    else:
        raise ValueError("orientation_mode must be free, none, axis, or full")
    return errors


def evaluate_state(trace: dict, metadata: dict, config: dict | None = None) -> dict:
    """Compute S(t) from physical truth and a common predeclared reference."""
    cfg = evaluation_config(metadata, config)
    scenario = _dict(metadata.get("scenario"))
    reference = _dict(metadata.get("reference"))
    t = _time(trace)
    count = len(t)
    normal = _series(trace, "normal", count, (5,))
    position = _series(trace, "position", count, (3,))
    relative = _series(trace, "relative", count, (3,))
    velocity = _series(trace, "relative_velocity", count, (3,))
    rotation = _series(trace, "rotation", count, (3, 3))
    angular = _series(trace, "angular_velocity", count, (3,))
    table = _series(trace, "support", count)
    mode = reference.get("orientation_mode", "full")
    target_relative = _reference(trace, metadata, "relative", count, (3,))
    target_rotation = _reference(trace, metadata, "rotation", count, (3, 3))
    relative_error = np.linalg.norm(relative - target_relative, axis=1)
    angle_error = _orientation_errors(rotation, target_rotation, mode,
                                       int(reference.get("axis", 2)))
    # A sphere has no functional angular state. For an axial object, exclude
    # angular velocity around its functional axis from the motion predicate.
    if mode == "none":
        angular_speed = np.zeros(count)
    elif mode == "axis":
        axis_vector = rotation[:, :, int(reference.get("axis", 2))]
        angular_speed = np.linalg.norm(angular - np.sum(
            angular * axis_vector, axis=1)[:, None] * axis_vector, axis=1)
    else:
        angular_speed = np.linalg.norm(angular, axis=1)
    required = np.asarray(cfg["required_fingers"], dtype=int)
    if len(required) == 0 or np.any(required < 0) or np.any(required > 4):
        raise ValueError("required_fingers must identify a predeclared grasp")
    contacts = normal > cfg["contact_min"]
    effective_support = np.all(contacts[:, required], axis=1) & (
        np.sum(contacts, axis=1) >= cfg["minimum_contacts"])
    if "effective_support" in trace:
        # Optional richer geometry predicate is truth-only and shared by all methods.
        effective_support &= np.asarray(trace["effective_support"], dtype=bool)
    lift_target = _reference(trace, metadata, "absolute_z", count, (),
                             reference.get("lift_height", np.nan))
    motion_attained = (position[:, 2] >= lift_target - cfg["lift_tolerance"])
    motion_attained &= table <= cfg["table_support_max"]
    if "motion_attained" in trace:
        motion_attained &= np.asarray(trace["motion_attained"], dtype=bool)
    limits = _series({"limits": trace.get("force_limit", scenario.get("limit", [np.inf] * 5))},
                     "limits", count, (5,))
    budget = _series({"budget": trace.get("budget", scenario.get("budget", np.inf))},
                     "budget", count)
    protected_normal = _series(trace, "skin_normal", count, (5,)) if "skin_normal" in trace else normal
    safe = np.all(protected_normal <= limits + cfg["force_tolerance"], axis=1)
    safe &= np.sum(protected_normal, axis=1) <= budget + cfg["force_tolerance"]
    palm = _series(trace, "palm", count, default=np.nan if "palm_limit" in cfg else 0.)
    if "palm_limit" in cfg:
        safe &= palm <= cfg["palm_limit"] + cfg["force_tolerance"]
    observed_valid = np.all(np.isfinite(normal), axis=1) & np.all(
        np.isfinite(position), axis=1) & np.all(np.isfinite(relative), axis=1)
    observed_valid &= np.all(np.isfinite(velocity), axis=1) & np.isfinite(table)
    observed_valid &= np.all(np.isfinite(rotation), axis=(1, 2)) & np.all(np.isfinite(angular), axis=1)
    observed_valid &= np.all(np.isfinite(protected_normal), axis=1)
    if cfg["require_skin_force"] and "skin_normal" not in trace:
        observed_valid[:] = False
    if cfg["require_all_hand_release"] and scenario.get("release_at", metadata.get("release_at")) is not None:
        if "all_hand_object" not in trace:
            observed_valid[:] = False
        else:
            observed_valid &= np.isfinite(_series(trace, "all_hand_object", count))
    if mode != "none":
        observed_valid &= np.isfinite(angle_error) & np.isfinite(angular_speed)
    if "palm_limit" in cfg:
        observed_valid &= np.isfinite(palm)
    stable = observed_valid & motion_attained & effective_support & safe
    stable &= relative_error <= cfg["relative_error_max"]
    stable &= angle_error <= cfg["orientation_error_max"]
    stable &= np.linalg.norm(velocity, axis=1) <= cfg["linear_speed_max"]
    stable &= angular_speed <= cfg["angular_speed_max"]
    return dict(time=t, stable=stable, record_valid=observed_valid,
                contacts=contacts, effective_support=effective_support,
                motion_attained=motion_attained, force_safe=safe,
                relative_error=relative_error, orientation_error=angle_error,
                linear_speed=np.linalg.norm(velocity, axis=1),
                angular_speed=angular_speed, limits=limits, budget=budget)


def _truth_coverage(t, state, metadata, cfg):
    scenario = _dict(metadata.get("scenario"))
    start = float(metadata.get("command_at", 0.))
    end = float(scenario.get("duration", metadata.get("duration", t[-1])))
    dt = float(cfg["evaluation_sample_interval"])
    tolerance = float(cfg["clock_tolerance"])
    expected_count = round((end - start) / dt) + 1
    start_ok = abs(t[0] - start) <= tolerance
    end_ok = abs(t[-1] - end) <= tolerance
    grid = start + np.arange(len(t)) * dt
    max_error = float(np.max(np.abs(t - grid)))
    gaps = int(np.sum(np.abs(np.diff(t) - dt) > tolerance))
    finite = bool(np.all(state["record_valid"]))
    complete = start_ok and end_ok and gaps == 0 and len(t) == expected_count and finite
    return dict(complete=bool(complete), start=start, end=end, expected_samples=expected_count,
                actual_samples=len(t), start_matches=start_ok, end_matches=end_ok,
                grid_gap_or_interval_errors=gaps, maximum_grid_error=max_error,
                all_required_truth_finite=finite, interval=dt)


def _substep_coverage(trace, metadata, cfg):
    scenario = _dict(metadata.get("scenario"))
    start = float(metadata.get("command_at", 0.))
    end = float(scenario.get("duration", metadata.get("duration", np.asarray(trace["time"])[-1])))
    tolerance = float(cfg["clock_tolerance"])
    available = "substep_time" in trace and "substep_normal" in trace
    result = dict(available=available, complete=False, start=start, end=end,
                  interval_durations_available="substep_duration" in trace)
    if not available:
        result["reason"] = "missing chronological full force vectors"
        return result
    t = np.asarray(trace["substep_time"], dtype=float)
    force = np.asarray(trace["substep_normal"], dtype=float)
    duration = np.asarray(trace.get("substep_duration", []), dtype=float)
    shape_ok = t.ndim == 1 and len(t) > 0 and force.shape == (len(t), 5)
    if not shape_ok:
        result["reason"] = "invalid substep vector dimensions"
        return result
    finite = bool(np.all(np.isfinite(t)) and np.all(np.isfinite(force)))
    if duration.shape != (len(t),) or not np.all(np.isfinite(duration)) or np.any(duration <= 0):
        result["reason"] = "missing or invalid physical interval duration"
        return result
    start_ok = abs(t[0] - start) <= tolerance
    end_ok = abs(t[-1] + duration[-1] - end) <= tolerance
    gaps = int(np.sum(np.abs(t[:-1] + duration[:-1] - t[1:]) > tolerance))
    total_duration = float(np.sum(duration))
    declared_dt = metadata.get("physics_timestep", cfg.get("physical_substep_interval"))
    interval_matches = (declared_dt is None or bool(np.allclose(duration, float(declared_dt),
                                                               atol=tolerance, rtol=0)))
    skin_available = "substep_skin_normal" in trace
    skin_ok = True
    if skin_available:
        skin = np.asarray(trace["substep_skin_normal"], dtype=float)
        skin_ok = skin.shape == force.shape and bool(np.all(np.isfinite(skin)))
    elif cfg["require_skin_force"] or "skin_normal" in trace:
        skin_ok = False
    complete = (finite and start_ok and end_ok and gaps == 0 and skin_ok and interval_matches and
                abs(total_duration - (end - start)) <= tolerance and
                np.all(np.diff(t) > 0))
    result.update(complete=bool(complete), samples=len(t), start_matches=start_ok,
                  end_matches=end_ok, gap_or_overlap_count=gaps, covered_duration=total_duration,
                  all_force_finite=finite, skin_force_record_valid=skin_ok,
                  declared_physics_interval=declared_dt,
                  intervals_match_declared_physics=interval_matches,
                  force_signal_kind="object_specific_grasp_contact",
                  protected_signal_kind="all_instrumented_finger_external_contacts" if skin_available
                                         else "object_specific_contact_fallback")
    if not complete:
        result["reason"] = "substep records do not completely cover the declared task"
    return result


def force_quality_trace(trace, metadata, config=None):
    """Offline fixed-band actual-force waveform; never sensor or controller filtering.

    The symmetric Hann-windowed sinc FIR is applied to the complete physical
    record before any static-window selection.  It has no phase lag and is
    interpolated onto the frozen common evaluation clock.  Terminal truth is
    appended only as an instantaneous endpoint, not a positive-time force step.
    """
    cfg = evaluation_config(metadata, config)
    coverage = _substep_coverage(trace, metadata, cfg)
    target_t = _time(trace)
    contract = dict(valid=False, full_substep_coverage=coverage["complete"],
                    force_signal_kind="object_specific_actual_grasp_force",
                    filter_kind="offline symmetric Hann-windowed sinc FIR",
                    cutoff_hz=float(cfg["force_bandwidth_hz"]),
                    kernel_span_seconds=float(cfg["force_filter_span"]),
                    evaluation_interval=float(cfg["evaluation_sample_interval"]),
                    filter_before_window=True, sensor_filter=False, stability_filter=False,
                    peak_or_exposure_filter=False)
    if not coverage["complete"]:
        if cfg["require_full_substeps"]:
            contract["reason"] = coverage.get("reason", "incomplete full-force coverage")
            return dict(time=target_t, normal=None, contract=contract)
        contract.update(valid=True, filter_kind="unfiltered raw evaluation sampling fallback",
                        reason="synthetic/development fallback without complete physical intervals")
        return dict(time=target_t, normal=np.asarray(trace["normal"], dtype=float), contract=contract)
    t = np.asarray(trace["substep_time"], dtype=float)
    duration = np.asarray(trace["substep_duration"], dtype=float)
    dt = float(np.median(duration))
    if not np.allclose(duration, dt, atol=cfg["clock_tolerance"], rtol=0):
        contract["reason"] = "fixed FIR requires uniformly timed physical intervals"
        return dict(time=target_t, normal=None, contract=contract)
    cutoff = float(cfg["force_bandwidth_hz"])
    if cutoff <= 0 or cutoff >= .5 / dt:
        raise ValueError("force_bandwidth_hz must lie below physical-record Nyquist")
    half = max(1, int(round(float(cfg["force_filter_span"]) / (2 * dt))))
    kernel_time = np.arange(-half, half + 1) * dt
    kernel = 2 * cutoff * dt * np.sinc(2 * cutoff * kernel_time) * np.hanning(2 * half + 1)
    kernel /= np.sum(kernel)
    force = np.asarray(trace["substep_normal"], dtype=float)
    # Adding the terminal evaluation endpoint gives both clocks the same
    # declared end. Padding reflects actual waveform at the task boundaries.
    endpoint = np.asarray(trace["normal"], dtype=float)[-1]
    force = np.vstack([force, endpoint])
    source_time = np.r_[t, t[-1] + duration[-1]]
    if not np.all(np.isfinite(force)):
        contract["reason"] = "nonfinite actual-force waveform or terminal endpoint"
        return dict(time=target_t, normal=None, contract=contract)
    filtered = np.empty_like(force)
    for finger in range(5):
        padded = np.pad(force[:, finger], (half, half), mode="reflect")
        filtered[:, finger] = np.convolve(padded, kernel, mode="valid")
    aligned = np.column_stack([np.interp(target_t, source_time, filtered[:, i]) for i in range(5)])
    contract.update(valid=True, physical_interval=dt, taps=len(kernel),
                    effective_kernel_span=float(2 * half * dt), group_delay_seconds=0.,
                    alignment="physical solver timestamps to shared 200 Hz truth clock",
                    boundary_padding="reflect entire task waveform at endpoints")
    return dict(time=target_t, normal=aligned, contract=contract)


def _force_metrics(t, force, relative=None, angle=None):
    if len(t) < 2 or not np.all(np.isfinite(force)):
        return None
    dt = float(t[-1] - t[0])
    total = np.sum(force, axis=1)
    mean = float(np.mean(total))
    result = {
        "mean_force": mean,
        "force_integral": _integral(t, total),
        "finger_sd": np.std(force, axis=0, ddof=0),
        "finger_sd_mean": float(np.mean(np.std(force, axis=0, ddof=0))),
        "total_sd": float(np.std(total, ddof=0)),
        "finger_tv_rate": float(np.sum(np.abs(np.diff(force, axis=0))) / dt),
        "total_tv_rate": float(np.sum(np.abs(np.diff(total))) / dt),
        "increment_rms": float(np.sqrt(np.mean(np.diff(force, axis=0) ** 2))),
        "duration": dt,
        "samples": len(t),
    }
    if relative is not None and np.all(np.isfinite(relative)):
        result["relative_drift_max"] = float(np.max(
            np.linalg.norm(relative - relative[0], axis=1)))
    if angle is not None and np.all(np.isfinite(angle)):
        result["orientation_error_max"] = float(np.max(angle))
    return result


def _window(trace, state, cfg, start, end, established, failure, quality, truth_complete):
    t = state["time"]
    mask = (t >= start - 1e-8) & (t <= end + 1e-8)
    selected = np.flatnonzero(mask)
    median_dt = float(np.median(np.diff(t)))
    coverage = 0.0
    record_complete = False
    if len(selected) >= 2:
        tw = t[selected]
        coverage = min(1., max(0., (tw[-1] - tw[0]) / (end - start)))
        record_complete = (tw[0] <= start + median_dt * .51 and
                           tw[-1] >= end - median_dt * .51 and
                           np.max(np.diff(tw)) <= median_dt * 1.51 and
                           np.all(state["record_valid"][selected]))
    fraction = (float(np.mean(state["effective_support"][selected]))
                if len(selected) else 0.)
    eligible = bool(established is not None and established <= start + 1e-8 and
                    (failure is None or failure > end + 1e-8) and record_complete and
                    coverage >= cfg["static_coverage_min"] and
                    fraction >= cfg["static_contact_fraction_min"] and truth_complete and
                    quality["contract"]["valid"])
    normal = np.asarray(trace["normal"], dtype=float)[mask]
    metrics = _force_metrics(t[mask], normal,
                            np.asarray(trace["relative"])[mask],
                            state["orientation_error"][mask])
    if metrics is not None:
        rotation = np.asarray(trace["rotation"])[mask]
        if len(rotation) and np.all(np.isfinite(rotation)):
            metrics["body_rotation_drift_max"] = float(np.max(_orientation_errors(
                rotation, np.tile(rotation[0], (len(rotation), 1, 1)), "full", 2)))
        for key in ("finger_tv_rate", "total_tv_rate", "finger_sd", "finger_sd_mean", "total_sd", "increment_rms"):
            metrics["raw200_" + key] = metrics[key]
        if quality["normal"] is not None:
            band = _force_metrics(t[mask], quality["normal"][mask])
            if band:
                for key in ("finger_tv_rate", "total_tv_rate", "finger_sd", "finger_sd_mean", "total_sd", "increment_rms"):
                    metrics["bandlimited_" + key] = band[key]
                metrics["finger_tv_rate"] = band["finger_tv_rate"]
                metrics["total_tv_rate"] = band["total_tv_rate"]
        elif cfg["require_full_substeps"]:
            metrics["finger_tv_rate"], metrics["total_tv_rate"] = None, None
        if "substep_normal" in trace and "substep_time" in trace:
            sub_t = np.asarray(trace["substep_time"], dtype=float)
            sub_mask = (sub_t >= start - cfg["clock_tolerance"]) & (sub_t <= end + cfg["clock_tolerance"])
            raw_sub = _force_metrics(sub_t[sub_mask], np.asarray(trace["substep_normal"])[sub_mask])
            if raw_sub:
                for key in ("finger_tv_rate", "total_tv_rate", "finger_sd", "finger_sd_mean", "total_sd", "increment_rms"):
                    metrics["substep_" + key] = raw_sub[key]
                metrics["substep_sampling_hz"] = 1. / float(np.median(np.diff(sub_t)))
    # Raw descriptive values are retained for inspection but never called a
    # primary smoothness gain unless the cohort is eligible in both methods.
    return dict(start=start, end=end, eligible=eligible, coverage=coverage,
                record_complete=record_complete, support_fraction=fraction,
                full_trial_truth_complete=truth_complete,
                force_quality_valid=quality["contract"]["valid"],
                low_contact_fraction=1. - fraction, metrics=metrics)


def _force_safety(trace, state, cfg, metadata=None):
    metadata = metadata or {}
    coverage = _substep_coverage(trace, metadata, cfg)
    t = np.asarray(trace.get("substep_time", state["time"]), dtype=float)
    if "substep_normal" in trace:
        object_n = np.asarray(trace["substep_normal"], dtype=float)
        n = np.asarray(trace.get("substep_skin_normal", object_n), dtype=float)
        skin_available = "substep_skin_normal" in trace
    else:
        object_n = np.asarray(trace["normal"], dtype=float)
        n = np.asarray(trace.get("skin_normal", object_n), dtype=float)
        skin_available = "skin_normal" in trace
    force_signal_kind = ("all_instrumented_finger_external_contacts" if skin_available
                         else "object_specific_contact_fallback")
    if n.shape != (len(t), 5) or np.any(np.diff(t) <= 0):
        raise ValueError("substep_normal must be synchronized N x 5 raw contact forces")
    indices = np.clip(np.searchsorted(state["time"], t, side="right") - 1,
                      0, len(state["time"]) - 1)
    limits = state["limits"][indices]
    budget = state["budget"][indices]
    valid = bool(np.all(np.isfinite(n)))
    if not valid:
        return dict(valid=False, full_substeps=coverage["complete"], coverage=coverage,
                    force_protection_pass=False)
    excess = np.maximum(n - limits, 0.)
    total = np.sum(n, axis=1)
    total_excess = np.maximum(total - budget, 0.)
    local_violation = np.any(n > limits + cfg["force_tolerance"], axis=1)
    total_violation = total > budget + cfg["force_tolerance"]
    palm = np.asarray(trace.get("substep_palm", np.zeros(len(t))), dtype=float)
    if "substep_palm" not in trace and "palm" in trace:
        palm = np.asarray(trace["palm"])[indices]
    if "palm_limit" in cfg and (("substep_palm" not in trace and "palm" not in trace) or
                               not np.all(np.isfinite(palm))):
        return dict(valid=False, full_substeps=coverage["complete"], coverage=coverage,
                    force_protection_pass=False,
                    invalid_reason="missing or nonfinite protected palm contact truth")
    palm_violation = np.zeros(len(t), dtype=bool)
    if "palm_limit" in cfg:
        palm_violation = palm > cfg["palm_limit"] + cfg["force_tolerance"]
    durations = trace.get("substep_duration")
    if durations is not None:
        durations = np.asarray(durations, dtype=float)
        if durations.shape != (len(t),) or not np.all(np.isfinite(durations)) or np.any(durations <= 0):
            raise ValueError("substep_duration must contain each physical solver step's positive duration")
        integrate = lambda values: np.sum(values * durations.reshape((-1,) + (1,) * (values.ndim - 1)), axis=0)
        integration_method = "solver_interval_sum"
        covered_duration = float(np.sum(durations))
    else:
        integrate = lambda values: np.trapezoid(values, t, axis=0)
        integration_method = "trapezoidal_fallback_without_interval_duration"
        covered_duration = float(t[-1] - t[0])
    local_integral = integrate(excess)
    force_record_valid = (not cfg["require_full_substeps"] or coverage["complete"])
    force_record_valid &= not cfg["require_skin_force"] or skin_available
    if "palm_limit" in cfg and cfg["require_full_substeps"] and "substep_palm" not in trace:
        force_record_valid = False
    terminal_n = np.asarray(trace.get("skin_normal", trace["normal"]), dtype=float)[-1]
    terminal_object = np.asarray(trace["normal"], dtype=float)[-1]
    terminal_palm = float(np.asarray(trace.get("palm", [0.]))[-1])
    terminal_valid = bool(np.all(np.isfinite(terminal_n)))
    terminal_violation = (not terminal_valid or np.any(
        terminal_n > state["limits"][-1] + cfg["force_tolerance"]) or
        np.sum(terminal_n) > state["budget"][-1] + cfg["force_tolerance"])
    if "palm_limit" in cfg:
        terminal_violation |= not np.isfinite(terminal_palm) or terminal_palm > cfg["palm_limit"] + cfg["force_tolerance"]
    force_record_valid &= terminal_valid
    result = dict(valid=bool(force_record_valid), full_substeps=coverage["complete"],
                  coverage=coverage, force_signal_kind=force_signal_kind,
                  protected_scope="instrumented fingers" + (" and palm" if "palm_limit" in cfg else " only"),
                  other_contact_loads="palm and fixture collision monitored; no material-damage threshold unless configured",
                  integration_method=integration_method,
                  covered_duration=covered_duration,
                  finger_peak=np.maximum(np.max(n, axis=0), terminal_n),
                  total_peak=float(max(np.max(total), np.sum(terminal_n))),
                  object_specific_finger_peak=np.maximum(np.max(object_n, axis=0), terminal_object),
                  object_specific_total_peak=float(max(np.max(np.sum(object_n, axis=1)), np.sum(terminal_object))),
                  palm_peak=float(max(np.max(palm), terminal_palm)),
                  terminal_endpoint_violation=bool(terminal_violation),
                  finger_excess_integral=local_integral,
                  local_excess_integral=float(np.sum(local_integral)),
                  total_excess_integral=float(integrate(total_excess)),
                  local_violation_duration=float(integrate(local_violation.astype(float))),
                  total_violation_duration=float(integrate(total_violation.astype(float))),
                  palm_violation_duration=float(integrate(palm_violation.astype(float))),
                  force_protection_pass=bool(force_record_valid) and not bool(terminal_violation) and not bool(
                      np.any(local_violation | total_violation | palm_violation)))
    if "palm_limit" in cfg:
        result["palm_excess_integral"] = float(integrate(np.maximum(palm - cfg["palm_limit"], 0.)))
    if "substep_collision" in trace:
        collision = np.asarray(trace["substep_collision"], dtype=float)
        if collision.shape == (len(t),) and np.all(np.isfinite(collision)):
            result["collision_peak"] = float(np.max(collision))
            result["collision_force_integral"] = float(integrate(collision))
    if "substep_all_hand_object" in trace:
        all_hand = np.asarray(trace["substep_all_hand_object"], dtype=float)
        if all_hand.shape == (len(t),) and np.all(np.isfinite(all_hand)):
            endpoint_hand = float(np.asarray(trace.get("all_hand_object", [0.]))[-1])
            result["all_hand_object_peak"] = float(max(np.max(all_hand), endpoint_hand))
            result["all_hand_object_force_integral"] = float(integrate(all_hand))
    if not force_record_valid:
        result["invalid_reason"] = "required complete physical force record or protected contact signal is missing"
        for key in ("finger_excess_integral", "local_excess_integral", "total_excess_integral",
                    "local_violation_duration", "total_violation_duration", "palm_violation_duration"):
            result["partial_" + key] = result[key]
            result[key] = None
    return result


def _recovery_events(trace, metadata, state, cfg, failure):
    t = state["time"]
    stable = state["stable"]
    results = []
    scenario = _dict(metadata.get("scenario"))
    release_at = scenario.get("release_at", metadata.get("release_at"))
    for j, ev in enumerate(metadata.get("events", [])):
        ev = _dict(ev)
        onset = float(ev["onset"])
        end = float(ev.get("end", onset))
        permanent = ev.get("kind") == "permanent" or bool(ev.get("permanent", False))
        origin = onset if permanent else end
        deadline = float(ev.get("deadline", origin + 2.))
        preempted = release_at is not None and float(release_at) < deadline
        if preempted:
            deadline = float(release_at)
        result = dict(id=ev.get("id", str(j)), onset=onset, end=end,
                      origin=origin, deadline=deadline, permanent=permanent,
                      outcome=None, recovery_time=None, confirmation_at=None)
        if release_at is not None and float(release_at) <= origin + cfg["stable_dwell"]:
            result["outcome"] = "preempted_by_release"
        elif failure is not None and failure <= onset + 1e-10:
            result["outcome"] = "not_exposed_failure"
        else:
            previous = np.flatnonzero(t < onset - 1e-10)
            before = None
            if len(previous):
                last_before = float(t[previous[-1]])
                before_window = (t >= last_before - cfg["stable_dwell"] - 1e-10) & (t <= last_before)
                if np.all(stable[before_window]):
                    before = _first_confirmation(t, stable, cfg["stable_dwell"],
                                                 last_before - cfg["stable_dwell"] - 1e-10,
                                                 last_before)
            if before is None:
                result["outcome"] = "not_stable_before_event"
            else:
                # For a transient disturbance, never-unstable describes the
                # disturbance itself, not later unrelated evolution in its horizon.
                exposure_end = deadline if permanent else end
                event_mask = (t >= onset - 1e-10) & (t <= exposure_end + 1e-10)
                if failure is not None:
                    event_mask &= t < failure - 1e-10
                event_indices = np.flatnonzero(event_mask)
                invalid = [i for i in event_indices if not stable[i]]
                if not invalid and (failure is None or failure > exposure_end):
                    result["outcome"] = "never_unstable"
                    after = (t > exposure_end) & (t <= deadline)
                    result["subsequent_instability"] = bool(np.any(~stable[after]))
                    result["subsequent_task_failure"] = bool(failure is not None and failure > exposure_end)
                elif invalid:
                    lost_at = float(t[invalid[0]])
                    result["unstable_at"] = lost_at
                    if not permanent:
                        within = _first_confirmation(t, stable, cfg["stable_dwell"],
                                                     lost_at, end - 1e-10)
                    else:
                        within = None
                    # A recovery during the disturbance is its own category only
                    # when it remains stable until the disturbance ends.
                    if within is not None and np.all(stable[(t >= within) & (t <= end)]):
                        result.update(outcome="recovered_during_event", confirmation_at=within)
                    else:
                        limit = min(deadline, failure) if failure is not None else deadline
                        confirm = _first_confirmation(t, stable, cfg["stable_dwell"],
                                                      origin, limit - (1e-10 if failure else 0.))
                        if confirm is not None:
                            result.update(outcome="recovered_after_event", confirmation_at=confirm,
                                          recovery_time=confirm - origin)
                        elif failure is not None and failure <= deadline:
                            result["outcome"] = "failed_before_recovery"
                        else:
                            result["outcome"] = "preempted_by_release" if preempted else "unrecovered"
                else:
                    result["outcome"] = "failed_before_recovery"
                window = (t >= onset) & (t <= deadline)
                if np.any(window):
                    result["maximum_relative_error"] = float(np.nanmax(
                        state["relative_error"][window]))
                    result["maximum_speed"] = float(np.nanmax(state["linear_speed"][window]))
        if result["confirmation_at"] is not None:
            result["subsequent_task_failure"] = bool(failure is not None and
                                                      failure >= result["confirmation_at"])
        results.append(result)
    return results


def evaluate_trial(trace: dict, metadata: dict, config: dict | None = None) -> dict:
    """One complete trial -> JSON-safe endpoint record with explicit denominators."""
    cfg = evaluation_config(metadata, config)
    scenario = _dict(metadata.get("scenario"))
    state = evaluate_state(trace, metadata, cfg)
    t = state["time"]
    command = float(metadata.get("command_at", t[0]))
    release_at = scenario.get("release_at", metadata.get("release_at"))
    declared_end = float(scenario.get("duration", metadata.get("duration", t[-1])))
    hold_end = min(declared_end, float(release_at)) if release_at is not None else declared_end
    failure = metadata.get("failure_at")
    failure = float(failure) if failure is not None else None
    registered_failure = failure
    failure_reason = metadata.get("failure_reason")
    deadline = min(command + cfg["establishment_deadline"], hold_end)
    established = _first_confirmation(t, state["motion_attained"] &
                                      state["effective_support"] & state["record_valid"],
                                      cfg["establishment_dwell"], command, deadline)
    # Never count a truth mask after the registered terminal loss as establishment.
    if established is not None and failure is not None and established >= failure:
        established = None
    # Independently verify continuous support loss from physical truth. This is
    # a task-support criterion, not an S failure due to speed/pose/force limits.
    # Unknown/missing truth cannot be relabelled as an observed contact loss.
    support_loss_at = None
    transport_loss_at = None
    if established is not None:
        unsupported = ~state["effective_support"] & state["record_valid"]
        support_loss_at = _first_confirmation(t, unsupported, cfg["support_loss_dwell"],
                                              established, hold_end - 1e-10)
        if support_loss_at is not None and (failure is None or support_loss_at < failure - 1e-10):
            failure, failure_reason = support_loss_at, "support_contact_lost"
        if "motion_attained" in trace:
            motion = np.asarray(trace["motion_attained"], dtype=float)
            if motion.shape != t.shape:
                raise ValueError("motion_attained must follow the common task evaluation clock")
            transport_missed = (motion == 0.) & np.isfinite(motion) & state["record_valid"]
            transport_loss_at = _first_confirmation(t, transport_missed, cfg["transport_loss_dwell"],
                                                    established, hold_end - 1e-10)
            if transport_loss_at is not None and (failure is None or transport_loss_at < failure - 1e-10):
                failure, failure_reason = transport_loss_at, "transport_reference_lost"
    horizon = min(float(cfg["first_stability_horizon"]), hold_end - command)
    first_end = min(command + horizon, failure) if failure is not None else command + horizon
    first = _first_confirmation(t, state["stable"], cfg["stable_dwell"], command,
                                first_end - (1e-10 if failure is not None else 0.))
    first_delay = first - command if first is not None else None
    restricted = min(first_delay, horizon) if first_delay is not None else horizon
    mechanical = metadata.get("mechanical_arrival_at")
    if mechanical is None:
        index = np.flatnonzero(state["motion_attained"] & (t >= command))
        mechanical = float(t[index[0]]) if len(index) else None
    truth_coverage = _truth_coverage(t, state, metadata, cfg)
    complete_record = truth_coverage["complete"]
    release_confirm = None
    release_record_available = "all_hand_object" in trace
    release_scope = "all hand-geometry contacts with target object"
    release_exposed = None
    release_outcome = "not_required"
    release_endpoint = None
    if release_at is not None:
        if release_record_available:
            all_hand_object = _series(trace, "all_hand_object", len(t))
            contact_free = np.isfinite(all_hand_object) & (all_hand_object <= cfg["release_force_max"])
        elif cfg["require_all_hand_release"]:
            all_hand_object = np.full(len(t), np.nan)
            contact_free = np.zeros(len(t), dtype=bool)
        else:
            all_hand_object = np.sum(np.asarray(trace["normal"]), axis=1) + _series(trace, "palm", len(t), default=np.nan)
            contact_free = np.isfinite(all_hand_object) & (all_hand_object <= cfg["release_force_max"])
            release_scope = "development fallback: five fingers plus palm target-object contact"
        # The public task executes at the first common clock endpoint at/after
        # the scheduled command. Its truth snapshot precedes the release action.
        dt = float(cfg["evaluation_sample_interval"])
        release_endpoint = command + np.ceil((float(release_at) - command) / dt - 1e-10) * dt
        release_index = int(np.searchsorted(t, release_endpoint - cfg["clock_tolerance"]))
        endpoint_recorded = (release_index < len(t) and
                             abs(t[release_index] - release_endpoint) <= cfg["clock_tolerance"])
        measurement_valid = complete_record and endpoint_recorded and np.all(np.isfinite(all_hand_object))
        release_exposed = bool(measurement_valid and established is not None and
                               established <= release_endpoint and state["effective_support"][release_index] and
                               all_hand_object[release_index] > cfg["release_force_max"])
        release_confirm = _first_confirmation(t, contact_free, cfg["release_dwell"],
                                              float(release_at), min(float(t[-1]),
                                              float(release_at) + cfg["release_deadline"]))
        if not measurement_valid:
            release_outcome = "missing_record"
        elif all_hand_object[release_index] <= cfg["release_force_max"]:
            release_outcome = "already_contact_free"
        elif not release_exposed:
            release_outcome = "unsupported_at_command"
        elif release_confirm is not None:
            release_outcome = "released"
        else:
            release_outcome = "deadline_failed"
    release_success = release_at is None or release_outcome == "released"
    force_safety = _force_safety(trace, state, cfg, metadata)
    complete_evidence = complete_record and (not cfg["require_full_substeps"] or force_safety["valid"])
    task_success = (established is not None and failure is None and release_success
                    and complete_evidence)
    end_reason = ((failure_reason or "post_establishment_loss") if failure is not None
                  else "establishment_failure" if established is None
                  else "release_failure" if not release_success
                  else "incomplete_truth_record" if not complete_record
                  else "incomplete_force_record" if not complete_evidence else "completed")
    hold_mask = (t >= (established if established is not None else command)) & (t <= hold_end)
    if failure is not None:
        # Failures cannot contribute post-drop zeros as stable duration.
        stable_hold = state["stable"] & (t < failure)
    else:
        stable_hold = state["stable"]
    hold_duration = hold_end - (established if established is not None else command)
    stable_fraction = (_integral(t[hold_mask], stable_hold[hold_mask].astype(float)) / hold_duration
                       if hold_duration > 0 and np.any(hold_mask) else 0.)
    if not complete_record:
        stable_fraction = None
    start, end = map(float, cfg["static_window"])
    quality = force_quality_trace(trace, metadata, cfg)
    window = _window(trace, state, cfg, start, end, established, failure, quality, complete_record)
    short_start, short_end = map(float, cfg["static_window_sensitivity"])
    short_window = _window(trace, state, cfg, short_start, short_end, established,
                           failure, quality, complete_record)
    result = {
        "scenario": scenario.get("name", metadata.get("scenario_name", "unknown")),
        "title": scenario.get("title", metadata.get("scenario_name", "unknown")),
        "method": metadata.get("method", "unknown"), "seed": metadata.get("seed"),
        "split": metadata.get("split", "unknown"), "boundary": scenario.get("boundary", False),
        "command_at": command, "horizon": horizon,
        "established_at": established, "first_stable_at": first,
        "first_stable_time": first_delay, "first_stability_attained": first is not None,
        "restricted_unstable_time": restricted,
        "restricted_unstable_fraction": restricted / horizon if horizon > 0 else None,
        "mechanical_arrival_at": mechanical,
        "extra_stabilization_time": first - mechanical if first is not None and mechanical is not None else None,
        "failure_at": failure, "task_success": task_success, "end_reason": end_reason,
        "failure_reason": failure_reason if failure is not None else None,
        "registered_failure_at": registered_failure,
        "derived_support_failure_at": support_loss_at,
        "derived_transport_failure_at": transport_loss_at,
        "force_tracking_comparison": "unavailable: desired has method-dependent target/prediction semantics; raw diagnostics only",
        "establishment_failure": established is None,
        "post_establishment_loss": established is not None and failure is not None,
        "stable_hold_fraction": stable_fraction,
        "release_required": release_at is not None, "release_success": release_success,
        "release_contact_scope": release_scope,
        "release_record_available": release_record_available,
        "release_command_endpoint_at": release_endpoint,
        "release_exposed_support": release_exposed,
        "release_physical_clearance_success": release_confirm is not None if release_at is not None else None,
        "release_outcome": release_outcome,
        "release_physical_clearance_time": release_confirm - float(release_at) if release_confirm is not None else None,
        "release_time": release_confirm - float(release_at) if release_confirm is not None and release_success else None,
        "static_window": window, "force_safety": force_safety,
        "static_window_sensitivity": short_window,
        "force_quality_contract": quality["contract"], "truth_coverage": truth_coverage,
        "events": _recovery_events(trace, metadata, state, cfg, failure),
        "truth_record_fraction": float(np.mean(state["record_valid"])),
        "truth_record_complete": complete_record,
        "complete_evidence": complete_evidence,
        "observed_valid_fraction": float(np.mean(trace["validity"])) if "validity" in trace else None,
    }
    if window["metrics"]:
        mean = window["metrics"]["mean_force"]
        result["static_window"]["metrics"]["total_cv"] = (
            window["metrics"]["total_sd"] / mean if mean >= cfg["cv_mean_min"] else None)
    return _json(result)

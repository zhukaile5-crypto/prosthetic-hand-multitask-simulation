"""Fixed-window offline wrist alignment, retention, recovery and force readout."""

import json
from numbers import Real
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


TASK_START_S = 9.30
RAMP_END_S = 10.30
TASK_END_S = 12.00
PULL_WINDOW_S = (9.85, 10.15)
CONTROL_DT_S = .005
PHYSICS_DT_S = .0005
CLOCK_TOL_S = 1e-8
ALIGNMENT_TOL_RAD = .015
DWELL_S = .2
CONTACT_MIN_N = .20
STABLE_DRIFT_M = .005
STABLE_SPEED_M_S = .005
GUARD_TOL_N = .02
FIELDS = (
    "time", "metadata", "wrist_position", "requested_wrist_reference",
    "relative", "relative_velocity", "normal", "skin_normal",
    "substep_time", "substep_duration", "substep_skin_normal",
)


def _number(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value)


def _window(clock, start, end, period):
    """Select a complete half-open fixed grid, independently of data elsewhere."""
    if clock is None or clock.ndim != 1 or clock.dtype.kind not in "fiu" or not np.isfinite(clock).all():
        return None
    start_index, end_index = round(start / period), round(end / period)
    if (start_index >= end_index or abs(start_index * period - start) > CLOCK_TOL_S
            or abs(end_index * period - end) > CLOCK_TOL_S):
        return None
    indices = np.flatnonzero((clock >= start - CLOCK_TOL_S) & (clock < end - CLOCK_TOL_S))
    expected = np.arange(start_index, end_index) * period
    if len(indices) != len(expected) or not np.allclose(clock[indices], expected, atol=CLOCK_TOL_S, rtol=0.):
        return None
    return indices


def _values(raw, field, indices, width, *, nonnegative=False):
    values = raw.get(field)
    clock = raw.get("substep_time" if field.startswith("substep_") else "time")
    if (indices is None or values is None or clock is None
            or values.shape != (len(clock), width) or values.dtype.kind not in "fiu"):
        return None
    data = values[indices].astype(float, copy=False)
    if not np.isfinite(data).all() or (nonnegative and np.any(data < 0.)):
        return None
    return data


def _dwell(time, condition, *, from_s):
    start = None
    for now, good in zip(time, condition, strict=True):
        if now < from_s - CLOCK_TOL_S:
            continue
        if good:
            if start is None:
                start = float(now)
            if now - start >= DWELL_S - CLOCK_TOL_S:
                return start, float(now)
        else:
            start = None
    return None, None


def _tracking(raw, indices, preempted):
    report = dict(available=False, axes=["WRJ2", "WRJ1"], samples_expected=540,
                  samples_observed=None if indices is None else len(indices),
                  rmse_rad=None, max_abs_error_rad=None, terminal_error_rad=None,
                  terminal_abs_error_rad=None, terminal_time_s=None,
                  target_held_after_ramp=None, target_delta_from_onset_rad=None,
                  alignment_confirmed=None, alignment_dwell_start_s=None,
                  alignment_confirmation_s=None, alignment_latency_from_ramp_end_s=None)
    measured = _values(raw, "wrist_position", indices, 2)
    requested = _values(raw, "requested_wrist_reference", indices, 2)
    if preempted or measured is None or requested is None:
        report["reason"] = "task_preempted_by_early_release" if preempted else "incomplete_clock_or_invalid_wrist_fields"
        return report
    time = raw["time"][indices]
    error = measured - requested
    settled = requested[time >= RAMP_END_S - CLOCK_TOL_S]
    held = bool(np.allclose(settled, settled[0], atol=1e-12, rtol=0.))
    aligned = (abs(error) <= ALIGNMENT_TOL_RAD).all(axis=1)
    start, confirmation = _dwell(time, aligned, from_s=RAMP_END_S) if held else (None, None)
    report.update(available=True, reason=None, rmse_rad=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                  max_abs_error_rad=np.max(abs(error), axis=0).tolist(), terminal_error_rad=error[-1].tolist(),
                  terminal_abs_error_rad=abs(error[-1]).tolist(), terminal_time_s=float(time[-1]),
                  terminal_requested_rad=requested[-1].tolist(), terminal_actual_rad=measured[-1].tolist(),
                  target_held_after_ramp=held, target_delta_from_onset_rad=(settled[0]-requested[0]).tolist(),
                  alignment_confirmed=(confirmation is not None) if held else None,
                  alignment_dwell_start_s=start, alignment_confirmation_s=confirmation,
                  alignment_latency_from_ramp_end_s=None if confirmation is None else confirmation-RAMP_END_S)
    return report


def _retention_and_recovery(raw, indices, pull, preempted):
    retention = dict(available=False, max_relative_drift_m=None, terminal_relative_drift_m=None,
                     max_relative_speed_m_s=None, terminal_relative_speed_m_s=None,
                     support_loss_samples=None, support_loss_duration_s=None, support_loss_episodes=None,
                     first_support_loss_s=None, stable_fraction=None)
    recovery = dict(available=False, status="no_pull" if pull is None else "unavailable",
                    pull_window_s=None if pull is None else list(pull), stable_before_pull=None,
                    first_instability_s=None, stable_dwell_start_s=None, stable_confirmation_s=None,
                    latency_from_pull_start_s=None, latency_from_instability_s=None)
    contact = _values(raw, "normal", indices, 5, nonnegative=True)
    relative = _values(raw, "relative", indices, 3)
    velocity = _values(raw, "relative_velocity", indices, 3)
    if preempted or contact is None or relative is None or velocity is None:
        retention["reason"] = "task_preempted_by_early_release" if preempted else "incomplete_clock_or_invalid_retention_fields"
        if pull is not None:
            recovery["reason"] = retention["reason"]
        return retention, recovery
    time = raw["time"][indices]
    drift = np.linalg.norm(relative-relative[0], axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    touch = contact > CONTACT_MIN_N
    support = touch[:, 0] & (touch.sum(axis=1) >= 3)
    stable = support & (drift <= STABLE_DRIFT_M) & (speed <= STABLE_SPEED_M_S)
    lost = ~support
    retention.update(available=True, reason=None, max_relative_drift_m=float(drift.max()),
                     terminal_relative_drift_m=float(drift[-1]), max_relative_speed_m_s=float(speed.max()),
                     terminal_relative_speed_m_s=float(speed[-1]), support_loss_samples=int(lost.sum()),
                     support_loss_duration_s=float(lost.sum()*CONTROL_DT_S),
                     support_loss_episodes=int((lost & ~np.r_[False, lost[:-1]]).sum()),
                     first_support_loss_s=float(time[lost][0]) if lost.any() else None,
                     stable_fraction=float(stable.mean()))
    if pull is None:
        return retention, recovery
    start, stop = pull
    before = (time >= start-DWELL_S-CLOCK_TOL_S) & (time < start-CLOCK_TOL_S)
    stable_before = bool(before.sum() == round(DWELL_S/CONTROL_DT_S) and stable[before].all())
    bad = (time >= start-CLOCK_TOL_S) & ~stable
    onset = float(time[bad][0]) if bad.any() else None
    dwell, confirmation = (None, None) if onset is None else _dwell(time, stable, from_s=onset)
    status = ("not_stable_before_pull" if not stable_before else
              "no_disruption" if onset is None else "unrecovered_by_release" if confirmation is None else
              "recovered_during_pull" if confirmation < stop-CLOCK_TOL_S else "recovered_after_pull")
    recovery.update(available=True, reason=None, status=status, stable_before_pull=stable_before,
                    first_instability_s=onset, stable_dwell_start_s=dwell, stable_confirmation_s=confirmation,
                    latency_from_pull_start_s=confirmation-start if stable_before and confirmation is not None else None,
                    latency_from_instability_s=confirmation-onset if stable_before and confirmation is not None else None)
    return retention, recovery


def _force(raw, start, end, limits, budget):
    report = dict(window_s=[start, end], available=False, guard_pass=None, substep_guard_pass=None,
                  pre_action_endpoint_guard_pass=None, substeps=None, finger_peak_N=None,
                  total_peak_N=None, finger_excess_Ns=None, local_excess_integral_Ns=None,
                  total_excess_integral_Ns=None, first_substep_guard_breach_interval_s=None)
    if limits is None or budget is None:
        report["reason"] = "missing_or_invalid_force_limits"
        return report
    clock = raw.get("substep_time")
    idx = _window(clock, start, end, PHYSICS_DT_S)
    normal = _values(raw, "substep_skin_normal", idx, 5, nonnegative=True)
    duration = raw.get("substep_duration")
    if (normal is None or duration is None or duration.shape != clock.shape
            or duration.dtype.kind not in "fiu" or not np.all(duration[idx] == PHYSICS_DT_S)):
        report["reason"] = "incomplete_clock_or_invalid_2kHz_external_skin_force"
        return report
    weight = duration[idx]
    combined = normal.sum(axis=1)
    exceeds = (normal > limits + GUARD_TOL_N).any(axis=1) | (combined > budget + GUARD_TOL_N)
    excess_by_finger = (np.maximum(normal-limits, 0.)*weight[:, None]).sum(axis=0)
    contact_boundary = raw.get("time")
    endpoint = None
    if contact_boundary is not None and contact_boundary.ndim == 1 and contact_boundary.dtype.kind in "fiu":
        hits = np.flatnonzero(abs(contact_boundary-end) <= CLOCK_TOL_S)
        if len(hits) == 1:
            rows = _values(raw, "skin_normal", hits, 5, nonnegative=True)
            if rows is not None:
                endpoint = rows[0]
    endpoint_pass = None if endpoint is None else bool(
        (endpoint <= limits+GUARD_TOL_N).all() and endpoint.sum() <= budget+GUARD_TOL_N)
    raw_pass = not bool(exceeds.any())
    first = None
    if not raw_pass:
        i = idx[int(np.argmax(exceeds))]
        first = [float(clock[i]), float(clock[i]+duration[i])]
    peaks = normal.max(axis=0)
    total_peak = float(combined.max())
    report.update(available=True, reason=None, substeps=len(idx), substep_guard_pass=raw_pass,
                  pre_action_endpoint_guard_pass=endpoint_pass,
                  pre_action_endpoint_finger_N=None if endpoint is None else endpoint.tolist(),
                  guard_pass=raw_pass, finger_peak_N=peaks.tolist(),
                  total_peak_N=total_peak, finger_excess_Ns=excess_by_finger.tolist(),
                  local_excess_integral_Ns=float(excess_by_finger.sum()),
                  total_excess_integral_Ns=float(np.maximum(combined-budget, 0.) @ weight),
                  first_substep_guard_breach_interval_s=first)
    return report


def evaluate(path: Path, *, pull_start_s, pull_end_s) -> dict:
    """Read one assigned run, preserving unavailable and failed cases separately."""
    declared = (pull_start_s, pull_end_s)
    if declared == (None, None):
        pull = None
    elif all(_number(value) for value in declared) and declared == PULL_WINDOW_S:
        pull = PULL_WINDOW_S
    else:
        raise ValueError("Declare both pull times as 9.85/10.15 s, or both as None")
    raw, metadata, issue = {}, {}, None
    try:
        with np.load(Path(path), allow_pickle=False) as trace:
            raw = {key: trace[key] for key in FIELDS if key in trace.files}
        if "metadata" in raw and raw["metadata"].shape == ():
            meta = json.loads(str(raw["metadata"].item()))
            if isinstance(meta, dict):
                metadata = meta
        if not metadata:
            issue = "embedded metadata missing or invalid"
    except (OSError, ValueError, KeyError, BadZipFile) as exc:
        issue = f"{type(exc).__name__}: {exc}"
    release = metadata.get("mechanical_release_at", metadata.get("release_at"))
    release = float(release) if _number(release) and release > 0. else None
    preempted = release is not None and release < TASK_END_S - CLOCK_TOL_S
    indices = _window(raw.get("time"), TASK_START_S, TASK_END_S, CONTROL_DT_S)
    tracking = _tracking(raw, indices, preempted)
    retention, recovery = _retention_and_recovery(raw, indices, pull, preempted)
    limits_raw, budget = metadata.get("limits"), metadata.get("budget")
    limits = (np.asarray(limits_raw, dtype=float) if isinstance(limits_raw, list)
              and len(limits_raw) == 5 and all(_number(x) and x >= 0. for x in limits_raw) else None)
    budget = float(budget) if _number(budget) and budget >= 0. else None
    pre_release = _force(raw, 0., release if preempted else TASK_END_S, limits, budget)
    task_force = _force(raw, TASK_START_S, TASK_END_S, limits, budget)
    task_force["task_preempted_by_early_release"] = preempted
    return dict(
        schema="wrist_alignment_v1", trace=str(Path(path)), assigned_trials=1, read_issue=issue,
        task_window_s=[TASK_START_S, TASK_END_S], ramp_end_s=RAMP_END_S, actual_mechanical_release_s=release,
        task_preempted_by_early_release=preempted, record_completed=metadata.get("completed"),
        handoff_passed=metadata.get("handoff_passed"), tracking=tracking, retention=retention, recovery=recovery,
        force=dict(finger_limits_N=None if limits is None else limits.tolist(), total_budget_N=budget,
                   guard_tolerance_N=GUARD_TOL_N, pre_release=pre_release, task_window=task_force),
        definitions=dict(
            tracking="Actual wrist encoders minus recorded user request; fixed 200 Hz left-sample intervals. Terminal is the final sample strictly before release. Dwell requires both errors at most 0.015 rad for 0.2 s after the declared ramp and a constant target.",
            retention="Target contact requires thumb and at least three fingers strictly above 0.20 N. Relative displacement is evaluated from the 9.30 s same-run relative position. Recovery stability also requires drift at most 5 mm and relative speed at most 5 mm/s.",
            recovery="Report first observed post-onset instability and first subsequent confirmed 0.2 s stable dwell. Pull-attributed latency requires 0.2 s stable before onset. No pull or no disruption has no recovery latency.",
            force="Raw all-external five-finger instrumented skin force, not target-only force. Half-open physical interval starts include the cached interval ending at the stop, never intervals starting at release. An extra pre-action endpoint sample is optional and reported separately. The guard, peaks and doses use only 2 kHz intervals. Task force remains descriptive if preempted.",
        ),
    )


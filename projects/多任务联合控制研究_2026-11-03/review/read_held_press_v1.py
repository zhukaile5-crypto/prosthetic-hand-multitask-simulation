"""Read held-object press performance from offline trajectory and contact evidence."""

import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from review.read_wrist_alignment_v1 import (
    CONTACT_MIN_N, CONTROL_DT_S, CLOCK_TOL_S, PHYSICS_DT_S, _force, _number, _values, _window,
)


TARGET_FRACTION = .10
TARGET_DWELL_S = .20
ATTRIBUTION_TOL_N = 1e-8
TRACE_FIELDS = (
    "metadata", "time", "relative", "relative_velocity", "normal", "skin_normal",
    "substep_time", "substep_duration", "substep_skin_normal",
)
DELIVERY_FIELDS = (
    "interval_start_s", "interval_end_s", "contact_valid", "target_receiver_top_normal_N",
    "receiver_any_actor_top_normal_N", "target_floor_normal_N", "target_receiver_otherface_normal_N",
    "target_receiver_top_positive_contact_count", "target_floor_positive_contact_count",
    "target_receiver_otherface_positive_contact_count", "target_hand_positive_contact_count",
)


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return data


def _arrays(path, fields):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in fields if key in data.files}


def _inputs(run):
    """Check sidecar claims when present. No missing hash is reported verified."""
    trace_path, ledger_path = run / "trace.npz", run / "delivery_intervals.npz"
    meta_path, manifest_path = run / "metadata.json", run / "delivery_manifest.json"
    trace, ledger, metadata = {}, {}, {}
    audit = dict(trace_sha256_verified=None, delivery_sha256_verified=None, issues=[])
    errors = (OSError, ValueError, KeyError, TypeError, BadZipFile)
    try:
        metadata = _json(meta_path)
        declared = metadata.get("trace_sha256")
        if declared is not None:
            audit["trace_sha256_verified"] = declared == _sha(trace_path)
            if not audit["trace_sha256_verified"]:
                raise ValueError("Trace digest differs from metadata")
        trace = _arrays(trace_path, TRACE_FIELDS)
        if "metadata" in trace:
            embedded = json.loads(str(trace["metadata"].item()))
            if not isinstance(embedded, dict) or any(
                metadata[k] != embedded[k] for k in
                ("duration", "sensor_period", "physics_timestep", "limits", "budget", "run_id", "method")
                if k in metadata and k in embedded
            ):
                raise ValueError("Embedded and sidecar metadata disagree")
    except errors as exc:
        audit["issues"].append(f"trace: {type(exc).__name__}: {exc}")
        trace, metadata = {}, {}
    try:
        if manifest_path.is_file():
            declaration = _json(manifest_path)
            entry = declaration["arrays"]
            audit["delivery_sha256_verified"] = (declaration.get("privileged_offline_only") is True
                                                and entry["file"] == ledger_path.name
                                                and entry["sha256"] == _sha(ledger_path))
            if not audit["delivery_sha256_verified"]:
                raise ValueError("Delivery manifest does not bind this offline ledger")
        ledger = _arrays(ledger_path, DELIVERY_FIELDS)
    except errors as exc:
        audit["issues"].append(f"delivery: {type(exc).__name__}: {exc}")
    return trace, ledger, metadata, audit


def _contact_grid(ledger, start, stop, field="target_receiver_top_normal_N"):
    """Map validated ledger intervals onto the fixed 2 kHz grid, retaining gaps."""
    lo, hi = round(start / PHYSICS_DT_S), round(stop / PHYSICS_DT_S)
    grid = np.arange(lo, hi) * PHYSICS_DT_S
    values = np.full(len(grid), np.nan)
    starts = ledger.get("interval_start_s")
    ends = ledger.get("interval_end_s")
    valid = ledger.get("contact_valid")
    force = ledger.get(field)
    if (any(x is None for x in (starts, ends, valid, force)) or starts.ndim != 1
            or any(x.shape != starts.shape for x in (ends, valid, force)) or valid.dtype.kind != "b"
            or any(x.dtype.kind not in "fiu" for x in (starts, ends, force))):
        return grid, values, dict(covered_intervals=0, invalid_grid_rows=None)
    selected = np.flatnonzero((starts >= start-CLOCK_TOL_S) & (starts < stop-CLOCK_TOL_S))
    ticks = np.rint(starts[selected] / PHYSICS_DT_S).astype(np.int64)
    unique, counts = np.unique(ticks, return_counts=True)
    nominal = ticks * PHYSICS_DT_S
    matches = ((ticks >= lo) & (ticks < hi) & (abs(starts[selected]-nominal) <= CLOCK_TOL_S)
               & (abs(ends[selected]-nominal-PHYSICS_DT_S) <= CLOCK_TOL_S)
               & np.isin(ticks, unique[counts == 1]))
    usable = matches & valid[selected] & np.isfinite(force[selected]) & (force[selected] >= 0.)
    if field.endswith("_count"):
        usable &= force[selected] == np.rint(force[selected])
    values[ticks[usable]-lo] = force[selected[usable]]
    return grid, values, dict(covered_intervals=int(matches.sum()), invalid_grid_rows=int((~matches).sum()))


def _coverage(values, grid_status):
    valid = int(np.isfinite(values).sum())
    return dict(expected_intervals=len(values), **grid_status, valid_intervals=valid,
                complete_valid=valid == len(values) and grid_status["invalid_grid_rows"] == 0)


def _interval(grid, i):
    return None if i is None else [float(grid[i]), float(grid[i]+PHYSICS_DT_S)]


def _first(mask):
    ids = np.flatnonzero(mask)
    return int(ids[0]) if len(ids) else None


def _in_target_band(force, target):
    return (force >= target*(1.-TARGET_FRACTION)) & (force <= target*(1.+TARGET_FRACTION))


def _press(ledger, grid, f, covered, target, start):
    coverage = _coverage(f, covered)
    complete = coverage["complete_valid"]
    i = _first(f > 0.)
    first_known = i is not None and covered["invalid_grid_rows"] == 0 and np.isfinite(f[:i]).all()
    previous = np.nan
    if start >= PHYSICS_DT_S:
        _, preceding, prior_status = _contact_grid(ledger, start-PHYSICS_DT_S, start)
        if _coverage(preceding, prior_status)["complete_valid"]:
            previous = preceding[0]
    status = ("known_no_contact" if complete else "unknown") if i is None else "observed_positive_onset_unresolved"
    onset = None
    if first_known:
        if i > 0 or previous == 0.:
            status, onset = "known_zero_to_positive", _interval(grid, i)
        elif previous > 0.:
            status = "preexisting_at_press_start"
    band = _in_target_band(f, target)
    arrival = _first(band) if complete else None
    dwell_start, confirmation = None, None
    if complete:
        required = round(TARGET_DWELL_S / PHYSICS_DT_S)
        edges = np.flatnonzero(np.diff(np.r_[False, band, False]))
        for begin, end in edges.reshape((-1, 2)):
            if end-begin >= required:
                dwell_start = float(grid[begin])
                confirmation = float(grid[begin+required-1] + PHYSICS_DT_S)
                break
    peak = float(f.max()) if complete else None
    return dict(available=complete, coverage=coverage,
                contact_occurred=True if i is not None else False if complete else None,
                observed_positive_interval_count=int((f > 0.).sum()),
                known_zero_to_positive_transition_count=int(((f > 0.) & (np.r_[previous, f[:-1]] == 0.)).sum()),
                first_observed_positive_interval_s=_interval(grid, i),
                first_positive_interval_s=_interval(grid, i) if first_known else None,
                onset_status=status, new_contact_onset_interval_s=onset,
                peak_receiver_force_N=peak, overshoot_N=None if peak is None else max(0., peak-target),
                target_band_reached=(arrival is not None) if complete else None,
                first_target_band_interval_s=_interval(grid, arrival),
                target_dwell_confirmed=(confirmation is not None) if complete else None,
                target_dwell_start_s=dwell_start, target_dwell_confirmation_s=confirmation)


def _hold(f, covered, target):
    coverage = _coverage(f, covered)
    complete = coverage["complete_valid"]
    if not complete:
        return dict(available=False, coverage=coverage, mean_force_N=None, signed_mean_error_N=None,
                    mean_abs_error_N=None, rmse_N=None, mean_centered_std_N=None, peak_to_peak_N=None,
                    in_target_band_fraction=None, contact_fraction=None)
    e = f-target
    return dict(available=True, coverage=coverage, mean_force_N=float(f.mean()),
                signed_mean_error_N=float(e.mean()), mean_abs_error_N=float(abs(e).mean()),
                rmse_N=float(np.sqrt(np.mean(e**2))), mean_centered_std_N=float(f.std()),
                peak_to_peak_N=float(np.ptp(f)), in_target_band_fraction=float(_in_target_band(f, target).mean()),
                contact_fraction=float((f > 0.).mean()))


def _attribution(ledger, start, stop, top, top_status):
    _, any_actor, any_status = _contact_grid(ledger, start, stop, "receiver_any_actor_top_normal_N")
    valid = _coverage(top, top_status)["complete_valid"] and _coverage(any_actor, any_status)["complete_valid"]
    difference = float(np.max(abs(top-any_actor))) if valid else None
    out = dict(available=valid, target_equals_any_actor_top=None if difference is None else difference <= ATTRIBUTION_TOL_N,
               top_comparison_tolerance_N=ATTRIBUTION_TOL_N, max_abs_top_load_difference_N=difference,
               any_actor_top_coverage=_coverage(any_actor, any_status),
               comparison_limitation="target_attribution_unavailable" if difference is None else
               "receiver_any_actor_load_differs_from_target_load" if difference > ATTRIBUTION_TOL_N else None)
    for name in ("target_floor", "target_receiver_otherface"):
        grid, values, status = _contact_grid(ledger, start, stop, name+"_normal_N")
        coverage = _coverage(values, status)
        positive = values > 0.
        observed = bool(positive.any())
        out[name] = dict(coverage=coverage,
            contact_occurred=True if observed else False if coverage["complete_valid"] else None,
            first_observed_positive_interval_s=_interval(grid, _first(positive)),
            peak_N=float(values.max()) if coverage["complete_valid"] else None)
    counts = {}
    for name in ("target_receiver_top", "target_floor", "target_receiver_otherface", "target_hand"):
        _, values, status = _contact_grid(ledger, start, start+PHYSICS_DT_S, name+"_positive_contact_count")
        counts[name] = int(values[0]) if _coverage(values, status)["complete_valid"] else None
    out["press_start"] = dict(interval_s=[start, start+PHYSICS_DT_S], positive_contact_counts=counts,
        target_receiver_top_normal_N=float(top[0]) if np.isfinite(top[0]) else None,
        receiver_any_actor_top_normal_N=float(any_actor[0]) if np.isfinite(any_actor[0]) else None)
    return out


def _retention(trace, start, stop):
    indices = _window(trace.get("time"), start, stop, CONTROL_DT_S)
    relative = _values(trace, "relative", indices, 3)
    velocity = _values(trace, "relative_velocity", indices, 3)
    contact = _values(trace, "normal", indices, 5, nonnegative=True)
    out = dict(available=False, samples=None if indices is None else len(indices),
               max_relative_displacement_m=None, final_relative_displacement_m=None,
               max_relative_speed_m_s=None, final_relative_speed_m_s=None, support_loss_samples=None,
               support_loss_duration_s=None, support_loss_episodes=None, first_support_loss_s=None)
    if relative is None or velocity is None or contact is None:
        return out
    displacement = np.linalg.norm(relative-relative[0], axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    touched = contact > CONTACT_MIN_N
    loss = ~(touched[:, 0] & (touched.sum(axis=1) >= 3))
    time = trace["time"][indices]
    out.update(available=True, max_relative_displacement_m=float(displacement.max()),
               final_relative_displacement_m=float(displacement[-1]), max_relative_speed_m_s=float(speed.max()),
               final_relative_speed_m_s=float(speed[-1]), support_loss_samples=int(loss.sum()),
               support_loss_duration_s=float(loss.sum()*CONTROL_DT_S),
               support_loss_episodes=int((loss & ~np.r_[False, loss[:-1]]).sum()),
               first_support_loss_s=float(time[loss][0]) if loss.any() else None)
    return out


def evaluate(run: Path, *, target_N, press_start_s, hold_start_s, hold_end_s) -> dict:
    """Return JSON-safe, independent outcomes for one assigned press run."""
    parameters = (target_N, press_start_s, hold_start_s, hold_end_s)
    if (not all(_number(x) for x in parameters) or target_N <= 0.
            or not 0. <= press_start_s < hold_start_s < hold_end_s
            or any(abs(round(t/CONTROL_DT_S)*CONTROL_DT_S - t) > CLOCK_TOL_S for t in parameters[1:])):
        raise ValueError("Finite positive force and ordered press/hold times on the 5 ms grid required")
    target, press_start, hold_start, hold_end = map(float, parameters)
    run = Path(run)
    trace, ledger, metadata, audit = _inputs(run)
    grid, force, covered = _contact_grid(ledger, press_start, hold_end)
    _, held, held_covered = _contact_grid(ledger, hold_start, hold_end)
    limit_value, budget = metadata.get("limits"), metadata.get("budget")
    limits = (np.array(limit_value, dtype=float) if isinstance(limit_value, list) and len(limit_value) == 5
              and all(_number(x) and x >= 0. for x in limit_value) else None)
    budget = float(budget) if _number(budget) and budget >= 0. else None
    duration = metadata.get("duration")
    full = (_force(trace, 0., float(duration), limits, budget)
            if _number(duration) and duration > 0. else
            dict(available=False, guard_pass=None, reason="episode_duration_unavailable"))
    return dict(
        schema="held_press_v1", run=str(run), assigned_trials=1, target_N=target,
        press_window_s=[press_start, hold_end], hold_window_s=[hold_start, hold_end],
        record_completed=metadata.get("completed"), handoff_passed=metadata.get("handoff_passed"),
        integrity=audit, press=_press(ledger, grid, force, covered, target, press_start),
        hold=_hold(held, held_covered, target), retention=_retention(trace, press_start, hold_end),
        attribution=_attribution(ledger, press_start, hold_end, force, covered),
        force=dict(full=full, task_window=_force(trace, press_start, hold_end, limits, budget),
                   finger_limits_N=None if limits is None else limits.tolist(), total_budget_N=budget),
        definitions=dict(
            receiver_force="Offline target-only compressive normal force on the declared receiver top. Valid cached 2 kHz intervals are equal-duration grid samples. Unknown intervals receive no contact credit.",
            contact="Report observed positive load separately from a new onset. A new onset requires known preceding zero load. Overshoot is max(0, peak minus target) over the fixed press window.",
            target_band_fraction=TARGET_FRACTION, target_dwell_s=TARGET_DWELL_S,
            tracking="Hold error is measured force minus target over the entire fixed hold window. Mean-centered SD and peak-to-peak are separate from target error and contact.",
            retention="200 Hz hand-frame displacement from press start, relative speed and target contact loss. Support requires the thumb and at least three fingers strictly above 0.20 N.",
            skin_force="Independent 2 kHz guards and raw excess integrals use all external instrumented finger loads, not receiver load. No criterion is converted to a single winner or fed back to the controller.",
        ),
    )


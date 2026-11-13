"""Predeclared cold-start endpoints; archived pregrasp success is inapplicable."""
import hashlib
import json
from pathlib import Path
import numpy as np

METRIC_SPEC = dict(
    stages_s=dict(pickup=[0., 7.5], holding_transport=[7.5, 12.], release=[12., 16.], full=[0., 16.]),
    pickup=dict(deadline_s=7.5, dwell_s=.2, contact_min_N=.2, minimum_contacts=3,
        thumb_required=True, object_rise_min_m=.025, table_support_max_N=.05,
        lift_target_tolerance_m=.01, relative_speed_max_m_s=.005, angular_speed_max_rad_s=.15),
    retention=dict(object_rise_min_m=.025, relative_drift_max_m=.08, failure_dwell_s=.1),
    motion=dict(lift_tolerance_m=.01, wrist_tolerance_rad=.015, failure_dwell_s=.3,
                window_s=[9.3, 12.], request_not_executed_reference=True),
    release=dict(deadline_s=15.5, dwell_s=.2, object_hand_clearance_N=.2,
        object_receiver_top_min_N=.05, object_floor_max_N=.02,
        steady_linear_speed_max_m_s=.02, steady_angular_speed_max_rad_s=.15),
    safety=dict(signal="raw all-external instrumented finger forces at every 0.5ms step plus endpoints",
                tolerance_N=.02, limits="unchanged per-scene finger limits and total budget"),
    quality=dict(signal="sum of five target-object normal forces", filter="existing V3 50Hz FIR, 50ms span",
                 main="total force reversal rate in N/s; total force SD also reported",
                 failed_pickup="all raw metrics retained; unavailable matched functional comparison"),
    slip=dict(signal="delayed public per-finger slip_magnitude on valid >0.2N contacts",
              threshold_m_s=.004, window_s=[7.5, 12.],
              note="contact-conditioned tactile proxy, not hidden ground-truth sliding distance"),
    conclusion="one already-seen development seed, four scenarios; descriptive pilot, no statistical superiority claim")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dwell_time(t, mask, duration):
    since = None
    for now, ok in zip(t, mask):
        if ok:
            if since is None:
                since = float(now)
            if float(now)-since >= duration-1e-9:
                return float(now)
        else:
            since = None
    return None


def interval_dwell(end_times, durations, mask, required):
    elapsed = 0.
    for t, dt, ok in zip(end_times, durations, mask):
        elapsed = elapsed+float(dt) if ok else 0.
        if elapsed >= required-1e-9:
            return float(t)
    return None


def evaluate(directory):
    from evaluation import force_band, force_variation, motion_tracking
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    with np.load(directory / "trace.npz", allow_pickle=False) as z:
        a = {k: z[k] for k in z.files if k != "metadata"}
    with np.load(directory / "delivery_intervals.npz", allow_pickle=False) as z:
        ledger = {k: z[k] for k in z.files}
    t = a["time"]
    normal = a["normal"]
    contacts = normal > .2
    rise = a["position"][:, 2]-float(meta["initial_position"][2])
    support = (contacts.sum(axis=1) >= 3) & contacts[:, 0] & (rise >= .025) & (a["support"] <= .05)
    stable = support & (abs(a["lift"]-meta["config"]["lift_target"]) <= .01)
    stable &= (np.linalg.norm(a["relative_velocity"], axis=1) <= .005)
    stable &= (np.linalg.norm(a["angular_velocity"], axis=1) <= .15)
    pickup = dwell_time(t, stable & (t <= 7.5+1e-9), .2)
    retained, retention_failure = False, None
    if pickup is not None:
        i = np.searchsorted(t, pickup)
        bad = (rise < .025) | (np.linalg.norm(a["relative"]-a["relative"][i], axis=1) > .08)
        window = (t >= pickup-1e-9) & (t <= 12.+1e-9)
        retention_failure = dwell_time(t, bad & window, .1)
        retained = retention_failure is None
    actual = np.column_stack((a["lift"], a["wrist_position"]))
    error = np.abs(actual-a["actual_motion_target"])
    tracking_bad = (error[:, 0] > .01) | np.any(error[:, 1:] > .015, axis=1)
    tracking_failure = dwell_time(t, tracking_bad & (t >= 9.3-1e-9) & (t < 12.-1e-9), .3)
    release_index = np.searchsorted(t, 12.)
    exposed = bool(support[release_index] and a["all_hand_object"][release_index] > .2)
    end, dt = ledger["interval_end_s"], ledger["interval_duration_s"]
    valid_ledger = bool(np.all(ledger["contact_valid"]) and np.all(ledger["object_state_valid"]))
    placed = (ledger["target_receiver_top_normal_N"] > .05) & (ledger["target_hand_normal_N"] <= .2)
    placed &= ledger["target_floor_normal_N"] <= .02
    placed &= (end > 12.+1e-9) & (end <= 15.5+1e-9)
    clear = interval_dwell(end, dt, placed, .2) if exposed and valid_ledger else None
    velocity = ledger["object_free_qvel6_end"]
    steady = placed & (np.linalg.norm(velocity[:, :3], axis=1) <= .02)
    steady &= np.linalg.norm(velocity[:, 3:], axis=1) <= .15
    steady_time = interval_dwell(end, dt, steady, .2) if exposed and valid_ledger else None
    raw = a["substep_skin_normal"]
    raw_total = raw.sum(axis=1)
    limits = np.asarray(meta["limits"])
    excess = np.maximum(np.max(raw-limits, axis=1), raw_total-meta["budget"])
    endpoint = a["skin_normal"]
    endpoint_excess = np.maximum(np.max(endpoint-limits, axis=1), endpoint.sum(axis=1)-meta["budget"])
    exceed_s = float(np.sum(a["substep_duration"] * (excess > .02)))
    limit_pass = bool(np.max(excess) <= .02 and np.max(endpoint_excess) <= .02)
    protocol = json.loads((Path(__file__).resolve().parents[1]/"protocol.json").read_text(encoding="utf-8"))
    band, filter_evidence = force_band(a, meta, protocol)
    quality = {}
    for name, (lo, hi) in METRIC_SPEC["stages_s"].items():
        mask = (t >= lo-1e-9) & (t <= hi+1e-9)
        physical_end = a["substep_time"]+a["substep_duration"]
        pmask = (physical_end > lo+1e-9) & (physical_end <= hi+1e-9)
        raw_sum = a["substep_normal"][pmask].sum(axis=1)
        stats = force_variation(t[mask], band[mask]) if filter_evidence["valid"] else None
        quality[name] = dict(raw_total_peak_N=float(np.max(raw_sum)),
            raw_total_mean_N=float(np.mean(raw_sum)),
            total_force_SD_N=None if stats is None else stats["total_force_SD_N"],
            total_force_reversal_rate_N_s=None if stats is None else stats["total_force_reversal_N"] / stats["duration_s"],
            total_force_TV_rate_N_s=None if stats is None else stats["total_force_TV_rate_N_s"],
            metrics_available=True, functional_comparison_eligible=bool(pickup is not None and retained))
    slip_values, slip_weights = [], []
    observed_seconds, slipping_seconds = 0., 0.
    for i in np.flatnonzero((t >= 7.5-1e-9) & (t < 12.-1e-9)):
        obs = json.loads(str(a["observation_json"][i]))
        v = obs.get("slip_magnitude")
        if v is None:
            continue
        v = np.asarray([np.nan if x is None else x for x in v], float)
        n = np.asarray([np.nan if x is None else x for x in obs["normal"]], float)
        valid = np.asarray(obs["valid"], bool) & (n > .2) & np.isfinite(v)
        if valid.any():
            speed = float(np.average(v[valid], weights=n[valid]))
            slip_values.append(speed)
            slip_weights.append(.005)
            observed_seconds += .005
            slipping_seconds += .005 * (speed > .004)
    coverage = len(t) == 3201 and len(raw) == 32000 and abs(t[-1]-16.) < 1e-8
    warning_free = int(np.max(a["warning_count"])) == 0
    reasons = []
    if pickup is None: reasons.append("pickup_not_stable_by_7_5s")
    if pickup is not None and not retained: reasons.append("object_not_retained_until_release")
    if tracking_failure is not None: reasons.append("transport_tracking_outside_tolerance")
    if not exposed: reasons.append("no_valid_raised_grasp_at_release_command")
    if clear is None: reasons.append("placed_and_clear_not_confirmed_by_15_5s")
    if not coverage or not warning_free or not valid_ledger: reasons.append("physical_record_invalid")
    result = dict(schema="cold_start_take_result_v1", arm=meta["experiment_arm"],
        scenario=meta["scenario"]["name"], seed=meta["seed"],
        full_task_completed=not reasons, reasons=reasons,
        pickup_completed=pickup is not None, pickup_stable_time_s=pickup,
        lift_request_start_s=meta["pickup_task"]["actual_lift_request_start_s"],
        maximum_object_rise_m=float(np.max(rise[t <= 7.5+1e-9])),
        retained_to_release=retained, retention_failure_time_s=retention_failure,
        transport_tracking_failure_time_s=tracking_failure,
        release_physically_exposed=exposed,
        placed_and_clear_time_s=clear, release_latency_s=None if clear is None else clear-12.,
        steady_placement_time_s=steady_time,
        force_limit_pass=limit_pass, force_exceedance_duration_s=exceed_s,
        peak_total_external_finger_force_N=float(max(np.max(raw_total), np.max(endpoint.sum(axis=1)))),
        peak_finger_force_N=np.maximum(raw.max(axis=0),endpoint.max(axis=0)).tolist(),
        maximum_force_excess_N=float(max(0., float(np.max(excess)), float(np.max(endpoint_excess)))),
        force_excess_impulse_N_s=float(np.sum(a["substep_duration"] * np.maximum(excess, 0.))),
        total_force=quality, force_filter=filter_evidence,
        slip=dict(observed_contact_seconds=observed_seconds, above_4mm_s_seconds=slipping_seconds,
            mean_speed_mm_s=None if not slip_values else float(np.average(slip_values,weights=slip_weights)*1000.),
            functional_comparison_eligible=bool(pickup is not None and retained)),
        transport_motion=motion_tracking(a,9.3,12.),
        physical_audit=dict(complete_16s_record=bool(coverage), warning_free=warning_free,
            valid_delivery_ledger=valid_ledger, controller_active_every_tick=bool(np.all(a["controller_active"])),
            selected_equals_executed=bool(np.array_equal(a["requested_reference21"],a["executed_reference21"])),
            no_hidden_clipping=meta["max_hidden_ctrl_clip"] <= 1e-10),
        artifacts_sha256={name:sha(directory/name) for name in ("trace.npz","metadata.json","delivery_intervals.npz")})
    (directory/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return result

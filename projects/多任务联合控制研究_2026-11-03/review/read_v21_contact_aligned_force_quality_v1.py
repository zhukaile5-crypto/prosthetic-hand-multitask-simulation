"""Exploratory 2 kHz force and settling readout from the matched V21 archives."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "review/v21_stage_attribution_v1.json": "dd45c6a1fbcd70920c587b5696bf56ef1137b69e3a8f8a5d09304ae335dafc48",
    "review/v21_complete16_auxiliary_v1.json": "b894592f38a8ca199088fd41bda9331ab819486b995b7db08d35e898be12d933",
    "protocol.json": "de3c6af5ebdd0abe5dcfcb47c07d11aa2beb360769c45390df883594d6b03eb0",
}
ROLES = ("full_V21", "full_V20", "same_prior_legacy_ratio")
DT = .0005
CONTACT_N = .20
HALF_SPANS_S = (.020, .005, .050)
WINDOWS = {
    "static_holding": ("absolute", 8.2, 8.9),
    "dynamic_holding": ("absolute", 9.3, 11.7),
    "full_release": ("absolute", 12.0, 16.0),
    "touch_0_to_200ms": ("touch", 0., .2),
    "touch_200_to_500ms": ("touch", .2, .5),
}
SETTLING_HORIZON_S = 1.5
MIN_MATCHED_CONTACT_SAMPLES = 40
NUMERIC_TIE_N = 1e-9


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_json(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Frozen source changed: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def step(t):
    index = round(t / DT)
    if abs(index * DT - t) > 1e-8:
        raise ValueError(f"Off-grid physics clock: {t}")
    return index


def force_history(record):
    path = (ROOT / record["trace"]).resolve()
    if not path.is_relative_to((ROOT / "output").resolve()) or sha(path) != record["trace_sha256"]:
        raise ValueError(f"Trace identity mismatch: {path}")
    with np.load(path, allow_pickle=False) as raw:
        st = raw["substep_time"]
        duration = raw["substep_duration"]
        force = np.vstack((raw["normal"][0], raw["substep_normal"]))
    if (st.shape != (32000,) or duration.shape != st.shape or force.shape != (32001, 5)
            or not np.isfinite(force).all() or np.any(force < 0.)
            or not np.allclose(st, np.arange(32000) * DT, atol=1e-8, rtol=0.)
            or not np.array_equal(duration, np.full(32000, DT))):
        raise ValueError(f"Invalid complete 2 kHz force history: {path}")
    # Grid index 0 is the initial sample. Index j>0 is the end of interval j-1.
    return force


def local_deviation(force, start, stop, half_span):
    width = 2 * half_span + 1
    centers = np.arange(start + half_span, stop - half_span + 1)
    if not len(centers) or start < 0 or stop >= len(force):
        raise ValueError("Window must contain the full centered stencil")
    values = force[start:stop + 1]
    cumulative = np.vstack((np.zeros((1, 5)), np.cumsum(values, axis=0)))
    mean = (cumulative[width:] - cumulative[:-width]) / width
    deviation = force[centers] - mean
    contact = np.vstack((np.zeros((1, 5), dtype=int), np.cumsum(values > CONTACT_N, axis=0)))
    contact_count = contact[width:] - contact[:-width]
    return deviation, contact_count == width, (contact_count > 0) & (contact_count < width)


def rms(value):
    return float(np.sqrt(np.mean(value ** 2)))


def describe(deviation, continuous, intermittent):
    squares = deviation ** 2
    total = float(squares.sum())
    return dict(finger_rms_N=rms(deviation), finger_p95_N=float(np.quantile(abs(deviation), .95)),
                sum_force_rms_N=rms(deviation.sum(axis=1)),
                finger_rms_by_digit_N=np.sqrt(squares.mean(axis=0)).tolist(),
                continuous_contact_fraction=float(continuous.mean()),
                mean_square_contributions_N2=dict(
                    continuous=float(squares[continuous].sum() / squares.size),
                    intermittent=float(squares[intermittent].sum() / squares.size),
                    always_below_threshold=float(squares[~(continuous | intermittent)].sum() / squares.size)),
                intermittent_energy_fraction=None if total <= 1e-20 else float(squares[intermittent].sum() / total),
                center_samples=len(deviation))


def window_record(histories, metrics, kind, lo, hi, half_span):
    signals = {}
    for role in ROLES:
        anchor = metrics[role]["first_receiver_contact_s"] if kind == "touch" else 0.
        signals[role] = local_deviation(histories[role], step(anchor + lo), step(anchor + hi), step(half_span))
    description = {role: describe(*signals[role]) for role in ROLES}
    a, active_a, _ = signals[ROLES[0]]
    pairs = {}
    for role in ROLES[1:]:
        b, active_b, _ = signals[role]
        common = active_a & active_b
        n = int(common.sum())
        pairs[role] = dict(shared_continuous_finger_samples=n,
                           shared_continuous_fraction=float(common.mean()),
                           shared_rms_v21_N=rms(a[common]) if n >= MIN_MATCHED_CONTACT_SAMPLES else None,
                           shared_rms_reference_N=rms(b[common]) if n >= MIN_MATCHED_CONTACT_SAMPLES else None)
    return dict(roles=description, pairs=pairs)


def settling_record(metrics, archived):
    result = {}
    for role in ROLES:
        touch = metrics[role]["first_receiver_contact_s"]
        release = metrics[role]["release_s"]
        found = archived[role]["first_low_motion_found"]
        latency = archived[role]["first_low_motion_latency_s"]
        if found is not (latency is not None) or touch < release or touch + SETTLING_HORIZON_S > 16. + 1e-9:
            raise ValueError("Inconsistent archived low-motion time or follow-up window")
        waiting = None if not found else release + latency - touch
        if waiting is not None and (waiting < 0. or touch + waiting > 16. + 1e-8):
            raise ValueError("Archived low-motion confirmation outside the event-follow-up")
        within = waiting is not None and waiting <= SETTLING_HORIZON_S
        result[role] = dict(touch_s=touch, first_confirmation_after_touch_s=waiting, within_horizon=within,
                            restricted_wait_s=waiting if within else SETTLING_HORIZON_S)
    return result


def contrast(values, tolerance=NUMERIC_TIE_N):
    if not values:
        return dict(n=0, lower=0, ties=0, higher=0, median_delta=None, mean_delta=None, range_delta=None)
    x = np.asarray(values)
    return dict(n=len(x), lower=int((x < -tolerance).sum()), ties=int((abs(x) <= tolerance).sum()),
                higher=int((x > tolerance).sum()), median_delta=float(np.median(x)),
                mean_delta=float(np.mean(x)), range_delta=[float(x.min()), float(x.max())])


def summary(rows):
    result = {}
    for tag in WINDOWS:
        result[tag] = {}
        for span in HALF_SPANS_S:
            key = str(round(span * 1000))
            windows = [r["windows"][tag][key] for r in rows]
            by_reference = {}
            for role in ROLES[1:]:
                active = [w["pairs"][role] for w in windows]
                delta = {name: contrast([w["roles"][ROLES[0]][name] - w["roles"][role][name] for w in windows])
                         for name in ("finger_rms_N", "finger_p95_N", "sum_force_rms_N")}
                by_reference[role] = dict(**delta, mean_square_contributions_N2={
                    contact_class: contrast([
                        w["roles"][ROLES[0]]["mean_square_contributions_N2"][contact_class]
                        - w["roles"][role]["mean_square_contributions_N2"][contact_class] for w in windows
                    ], tolerance=1e-12) for contact_class in ("continuous", "intermittent", "always_below_threshold")
                }, shared_continuous_rms_N=contrast([
                    a["shared_rms_v21_N"] - a["shared_rms_reference_N"] for a in active if a["shared_rms_v21_N"] is not None]),
                    median_shared_continuous_fraction=float(np.median([a["shared_continuous_fraction"] for a in active])))
            result[tag][key] = dict(pairs=by_reference, roles={
                role: dict(median_finger_rms_N=float(np.median([w["roles"][role]["finger_rms_N"] for w in windows])),
                           median_contact_fraction=float(np.median([w["roles"][role]["continuous_contact_fraction"] for w in windows])))
                for role in ROLES})
    return result


def archived_settling(aux):
    rows = {tuple(r["context_key"]): {ROLES[0]: r["metrics"]} for r in aux["records"]}
    for pair in aux["pairs"]:
        role = pair["reference_role"]
        if role not in ROLES[1:]:
            continue
        row = rows[tuple(pair["context_key"])]
        if role in row or pair["same_physical_noise_limits"] is not True:
            raise ValueError("Duplicate or mismatched archived comparison")
        row[role] = pair["reference_metrics"]
    if len(rows) != 16 or any(set(row) != set(ROLES) for row in rows.values()):
        raise ValueError("Incomplete original low-motion roster")
    return rows


def self_check():
    grid = np.arange(400, dtype=float)[:, None] * np.ones((1, 5))
    linear, _, _ = local_deviation(2. * grid + 3., 0, 399, 40)
    oscillation, active, _ = local_deviation(2. + .3 * np.cos(2 * np.pi * 50. * grid * DT), 0, 399, 40)
    disrupted = np.full_like(grid, 2.)
    disrupted[200, 0] = 0.
    _, mask, intermittent = local_deviation(disrupted, 0, 399, 40)
    assert not np.any(linear) and rms(oscillation) > .15 and active.all()
    assert (~mask[:, 0]).sum() == 81 and mask[:, 1:].all() and np.array_equal(~mask, intermittent)
    return ["exact_linear_ramp_cancelled", "50Hz_oscillation_detected", "whole_stencil_contact_interruption_detected"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checks = self_check()
    stage, aux, protocol = (verified_json(ROOT / p, h) for p, h in FILES.items())
    assert protocol["thresholds"]["finger_contact_min_N"] == CONTACT_N
    assert protocol["endpoints"]["static_quality"]["window_s"] == [8.2, 8.9]
    assert protocol["endpoints"]["dynamic_quality"]["window_s"] == [9.3, 11.7]
    settling = archived_settling(aux)
    rows = []
    for item in stage["rows"]:
        ctx = tuple(item["context"])
        metrics = item["metrics"]
        if set(metrics) != set(ROLES) or any(x["release_s"] != 12. for x in metrics.values()):
            raise ValueError(f"Unexpected comparison assignment: {ctx}")
        histories = {r: force_history(metrics[r]) for r in ROLES}
        windows = {tag: {str(round(h * 1000)): window_record(histories, metrics, *window, h)
                         for h in HALF_SPANS_S} for tag, window in WINDOWS.items()}
        rows.append(dict(context=ctx, force_sha256={r: metrics[r]["trace_sha256"] for r in ROLES},
                         windows=windows, settling=settling_record(metrics, settling[ctx])))
    assert len(rows) == len(set(tuple(r["context"]) for r in rows)) == 16
    assert len({s for r in rows for s in r["force_sha256"].values()}) == 48
    stage_summary = summary(rows)
    wait = {role: dict(within=sum(r["settling"][role]["within_horizon"] for r in rows),
                      observed_through_episode=sum(r["settling"][role]["first_confirmation_after_touch_s"] is not None for r in rows),
                      mean_restricted_wait_s=float(np.mean([r["settling"][role]["restricted_wait_s"] for r in rows])))
            for role in ROLES}
    report = dict(schema="v21_contact_aligned_force_quality_v1", contexts=16, archives=48,
                  independent_physical_seed_clusters=2, physics_steps=0, source_sha256={
                      **FILES, Path(__file__).name: sha(Path(__file__))}, self_checks=checks,
                  windows=WINDOWS, half_spans_s=HALF_SPANS_S, primary_half_span_s=.020,
                  description="Unfiltered target contact end-sample minus centered mean within each fixed window, all times retained. Touch-aligned windows use each run's frozen contact interval start.",
                  continuous_contact_threshold_N=CONTACT_N, minimum_common_finger_samples=MIN_MATCHED_CONTACT_SAMPLES,
                  contact_conditioning="Jointly continuous per finger across the full centered stencil. All-data and conditional results are separate.",
                  numeric_tie_N=NUMERIC_TIE_N, settling_horizon_s=SETTLING_HORIZON_S,
                  settling=wait, summaries=stage_summary,
                  by_seed={str(seed): summary([r for r in rows if r["context"][2] == seed])
                           for seed in sorted({r["context"][2] for r in rows})},
                  rows=rows)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(contexts=len(rows), archives=48, settling=wait, output=str(args.out))))


if __name__ == "__main__":
    main()

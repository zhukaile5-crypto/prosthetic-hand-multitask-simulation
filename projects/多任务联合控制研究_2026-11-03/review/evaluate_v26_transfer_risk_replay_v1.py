"""Offline risk replay on frozen V21 and V24 traces. Does not instantiate a plant."""
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V21 = ROOT / 'review/v21_complete16_auxiliary_v1.json'
V24 = ROOT / 'review/v24_small_readout_v1.json'
DT, SUBSTEP = .005, .0005
TOL = 1e-9
SEEDS = (619741, 619742)
PRESSURE = .20
HISTORY, REPLAY_STEPS = 4, 40
HORIZON, RECOVERY = 40, 10
RIDGE, ALARM_QUANTILE = .025, .90
FEATURES = ('pressure_N', 'last_change_N', 'hand_mean_N', 'hand_abs_step_N',
            'receiver_load_N', 'receiver_last_change_N', 'receiver_abs_step_N')
MODELS = {'pressure_and_slope': 2, 'hand_history': 4, 'hand_and_receiver': 7}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


@dataclass(frozen=True)
class Opportunity:
    context: str
    features: tuple
    issue_substep: int
    events: tuple


def records():
    full = read(V21)
    short = read(V24)
    assert full['complete'] is True and short['complete'] is True
    assert len(full['records']) == 16 and len(short['records']) == 9
    cohorts = {'V21': full['records'],
               'V24': [r for r in short['records'] if r['variant'] == 'canonical_joint']}
    assert len(cohorts['V24']) == 3
    assert {tuple(r['context_key']) for r in cohorts['V21']} == {
        (s, m, k) for s in ('nominal', 'collision', 'friction_drop', 'fragile_conflict')
        for m in ('hold', 'transport') for k in SEEDS
    }
    return cohorts


def short_loss_onsets(forces):
    """Start of a recorded zero-load pulse with load >=0.2 N before and within 5 ms."""
    zeros = forces <= 1e-12
    starts = np.flatnonzero(zeros[1:] & (forces[:-1] >= PRESSURE)) + 1
    return np.array([s for s in starts if s + RECOVERY < len(forces)
                     and np.any(forces[s+1:s+RECOVERY+1] >= PRESSURE)], dtype=int)


def receiver_valid(row, time):
    return (row.get('valid') is True and row.get('normal_load_N') is not None
            and np.isfinite(row['normal_load_N']) and row['normal_load_N'] >= 0.
            and abs(row['time_s']-time) <= TOL
            and abs(row['acquisition_time_s']-(time-.010)) <= TOL
            and abs(row['age_s']-.010) <= TOL)


def transfer(row):
    return isinstance(row, dict) and row.get('phase') == 'TRANSFER' and row.get('receiver_fresh_streak', 0) >= 2


def collect(version, record):
    scene, motion, seed = record['context_key']
    ctx = f'{version}/{scene}/{motion}/{seed}'
    trace = (Path(record['run']) / 'trace.npz').resolve()
    assert trace.is_relative_to((ROOT/'output').resolve())
    digest = sha(trace)
    assert digest == record['input_sha256']['trace.npz'], ctx
    with np.load(trace, allow_pickle=False) as a:
        t, sub, duration = (a[key] for key in ('time', 'substep_time', 'substep_duration'))
        force = a['substep_skin_normal']
        normal, valid, contact_mirror = (a[key] for key in ('observed_normal', 'validity', 'observed_contact'))
        assert t.shape == (3201,) and sub.shape == duration.shape == (32000,), ctx
        assert force.shape == (32000, 5) and normal.shape == valid.shape == contact_mirror.shape == (3201, 5), ctx
        assert valid.dtype == contact_mirror.dtype == bool and np.isfinite(force).all() and np.all(force >= 0.), ctx
        assert np.allclose(t, np.arange(len(t))*DT, atol=TOL, rtol=0.), ctx
        assert np.allclose(sub, np.arange(len(sub))*SUBSTEP, atol=TOL, rtol=0.), ctx
        assert np.allclose(duration, SUBSTEP, atol=TOL, rtol=0.), ctx
        phases = [json.loads(str(x)) for x in a['receiver_release_tasks_json']]
        rr = a['receiver_measurement_json']
        observations = a['observation_json']
        trigger = next((k for k, p in enumerate(phases) if transfer(p)), None)
        events = [short_loss_onsets(force[:, j]) for j in range(5)]
        stat = dict(version=version, context=[scene, motion, seed], trace=trace.relative_to(ROOT).as_posix(),
                    trace_sha256=digest, origin_s=None, assigned_ticks=REPLAY_STEPS, possible_rows=REPLAY_STEPS*5,
                    phase_ticks=0, receiver_ticks=0, eligible_rows=0, positive_rows=0, observable_events=0,
                    excluded_unloaded_or_no_contact=0, excluded_hand_clock_or_valid=0,
                    excluded_label_tail=0, excluded_receiver_ticks=0, contact_mirror_differing_rows=0)
        if trigger is None:
            return [], stat
        origin = trigger + 1
        stat['origin_s'] = float(t[origin])
        rows = []
        for k in range(origin, min(origin+REPLAY_STEPS, len(t))):
            phase = phases[k-1]
            if not transfer(phase):
                continue
            assert abs(phase['time_s']-t[k-1]) <= TOL, ctx
            stat['phase_ticks'] += 1
            ts = slice(k-HISTORY+1, k+1)
            receiver = [json.loads(str(x)) for x in rr[ts]]
            if len(receiver) != HISTORY or not all(receiver_valid(r, v) for r, v in zip(receiver, t[ts])):
                stat['excluded_receiver_ticks'] += 1
                continue
            stat['receiver_ticks'] += 1
            load = np.array([r['normal_load_N'] for r in receiver])
            hand = [json.loads(str(x)) for x in observations[ts]]
            hand_clocks = all(abs(o['time']-v) <= TOL and o['sample_time'] is not None
                              and abs(o['sample_time']-(v-.010)) <= TOL
                              for o, v in zip(hand, t[ts]))
            for h, o in enumerate(hand, start=k-HISTORY+1):
                assert np.array_equal(o['normal'], normal[h]) and np.array_equal(o['valid'], valid[h]), ctx
            current_contact = hand[-1]['contact_present']
            assert len(current_contact) == 5 and all(type(v) is bool for v in current_contact), ctx
            stat['contact_mirror_differing_rows'] += int(np.sum(current_contact != contact_mirror[k]))
            issue = k * round(DT / SUBSTEP)
            if issue+HORIZON+RECOVERY >= len(sub):
                stat['excluded_label_tail'] += 5
                continue
            for j in range(5):
                p = normal[ts,j]
                if not hand_clocks or p.shape != (HISTORY,) or not valid[ts,j].all() or not np.isfinite(p).all():
                    stat['excluded_hand_clock_or_valid'] += 1
                    continue
                if not current_contact[j] or p[-1] < PRESSURE:
                    stat['excluded_unloaded_or_no_contact'] += 1
                    continue
                xx = (p[-1], p[-1]-p[-2], np.mean(p), np.mean(np.abs(np.diff(p))),
                      load[-1], load[-1]-load[-2], np.mean(np.abs(np.diff(load))))
                e = events[j]
                hit = tuple((ctx, j, int(s)) for s in e[(issue < e) & (e <= issue+HORIZON)])
                rows.append(Opportunity(ctx, tuple(float(x) for x in xx), issue, hit))
    stat.update(eligible_rows=len(rows), positive_rows=sum(bool(r.events) for r in rows),
                observable_events=len({e for row in rows for e in row.events}))
    return rows, stat


def logistic(train_x, train_y, test_x, weights):
    """Small ridge logistic fit using weighted Newton steps and decreasing loss."""
    center = np.average(train_x, axis=0, weights=weights)
    scale = np.sqrt(np.average((train_x-center)**2, axis=0, weights=weights))
    scale[scale <= 1e-12] = 1.
    def design(x):
        return np.column_stack((np.ones(len(x)), (x-center)/scale))
    x = design(train_x)
    prior = np.average(train_y, weights=weights)
    b = np.zeros(x.shape[1])
    b[0] = np.log(prior / (1-prior))
    penalty = RIDGE * np.r_[0., np.ones(x.shape[1]-1)]
    def loss(c):
        z = x @ c
        return np.sum(weights * (np.logaddexp(0., z)-train_y*z))+.5*np.sum(penalty*c*c)
    for _ in range(40):
        pred = 1./(1.+np.exp(-np.clip(x @ b, -100., 100.)))
        gradient = x.T @ (weights*(pred-train_y))+penalty*b
        if np.max(np.abs(gradient)) < 1e-9:
            break
        hess = (x.T * (weights*pred*(1.-pred))) @ x + np.diag(penalty)
        step = np.linalg.solve(hess, gradient)
        old = loss(b)
        for depth in range(20):
            trial = b-step/(2**depth)
            if loss(trial) <= old:
                b = trial
                break
        else:
            raise RuntimeError('Ridge fit failed line search')
    else:
        raise RuntimeError('Ridge fit failed to converge')
    def scores(z):
        return 1./(1.+np.exp(-np.clip(design(z) @ b, -100., 100.)))
    return scores(train_x), scores(test_x)


def average_precision(y, scores):
    if not y.any():
        return None
    order = np.argsort(-scores, kind='stable')
    s = scores[order]
    boundary = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s)-1]
    tp = np.cumsum(y[order])[boundary]
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / (boundary+1))/np.sum(y))


def metrics(rows, y, scores, threshold=None):
    base = dict(average_precision=average_precision(y, scores), brier=float(np.mean((scores-y)**2)))
    if threshold is None:
        return base
    warning = scores >= threshold
    captured = {}
    for alarm, row in zip(warning, rows):
        if alarm:
            for e in row.events:
                captured[e] = max(captured.get(e, 0), e[2]-row.issue_substep)
    all_events = {e for row in rows for e in row.events}
    return dict(**base, threshold=float(threshold), warning_rows=int(warning.sum()),
                false_warning_rows=int(np.sum(warning & ~y)),
                eligible_events=len(all_events), detected_events=len(captured),
                lead_ms_median=None if not captured else float(np.median(list(captured.values())) * SUBSTEP * 1000.),
                contexts_with_warnings=len({r.context for w,r in zip(warning,rows) if w}))


def eval_fold(train, test, name):
    y = np.array([bool(r.events) for r in train], dtype=bool)
    yt = np.array([bool(r.events) for r in test], dtype=bool)
    stats = dict(fold=name, train_rows=len(train), test_rows=len(test), train_positive=int(y.sum()),
                 test_positive=int(yt.sum()), test_events=len({e for r in test for e in r.events}),
                 train_contexts=sorted({r.context for r in train}), test_contexts=sorted({r.context for r in test}),
                 test_positive_contexts=sorted({r.context for r in test if r.events}))
    if not len(yt) or not y.any() or y.all():
        return dict(**stats, status='not_evaluable_without_train_classes_or_test')
    counts = {ctx: sum(r.context == ctx for r in train) for ctx in stats['train_contexts']}
    weights = np.array([1./counts[r.context] for r in train])
    weights /= weights.sum()
    x = np.array([r.features for r in train])
    xt = np.array([r.features for r in test])
    model = {'training_prior': metrics(test, yt, np.full(len(yt), np.average(y, weights=weights)))}
    for label, width in MODELS.items():
        fitted, scores = logistic(x[:,:width], y, xt[:,:width], weights)
        threshold = np.quantile(fitted, ALARM_QUANTILE, method='higher')
        model[label] = metrics(test, yt, scores, threshold)
    return dict(**stats, status='evaluated', model=model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    data, cases = [], []
    for version, group in records().items():
        for record in group:
            rows, stat = collect(version, record)
            data.extend(rows)
            cases.append(stat)
    results = []
    for seed in SEEDS:
        train = [r for r in data if r.context.startswith('V21/') and not r.context.endswith('/'+str(seed))]
        for version in ('V21', 'V24'):
            test = [r for r in data if r.context.startswith(version+'/') and r.context.endswith('/'+str(seed))]
            results.append(eval_fold(train, test, f'V21_other_seed_to_{version}_{seed}'))
    artifact = dict(schema='v26_public_transfer_transient_risk_replay_v1', physics_steps=0,
                    prior_record_sha256={p.name:sha(p) for p in (V21,V24)}, code_sha256=sha(Path(__file__)),
                    method=dict(issue_window_s=REPLAY_STEPS*DT, trigger='next tick after TRANSFER/fresh streak 2',
                                features=FEATURES, pressure_eligibility_N=PRESSURE, acquisition_delay_s=.010,
                                history_frames=HISTORY, pulse_zero_N=1e-12, recovery_within_s=RECOVERY*SUBSTEP,
                                horizon_s=HORIZON*SUBSTEP, ridge=RIDGE, training_alarm_quantile=ALARM_QUANTILE,
                                label='skin normal falls from >=0.20N to zero and returns to >=0.20N within 5ms',
                                eligible_label_times='strictly after issue through issue+20ms',
                                threshold_source='training scores only', row_clock_source='current or earlier public reads only',
                                independent_cluster_count=2, role='observational replay, not action-conditional identification'),
                    contexts=cases, folds=results)
    with out.open('x', encoding='utf-8') as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(path=str(out), trace_count=len(cases), eligible_rows=len(data), folds=len(results),
                          sha256=sha(out)), ensure_ascii=False))


if __name__ == '__main__':
    main()

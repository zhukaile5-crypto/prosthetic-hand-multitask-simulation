"""Apply the V21 public risk score to the V26 intervention branches without refitting on V26."""
import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_v26_transfer_risk_replay_v1 as base


RUN = base.ROOT / 'output/v26_five_handover_v1'
PREVIOUS = base.ROOT / 'review/v26_transfer_risk_replay_v1.json'
ARMS = ('zero', 'zero_repeat', 'lift_plus', 'finger_single', 'mixed_holdout')
ORIGIN = 13.940
TRAIN_SEED = 619742
LOCKED_EXECUTION = '06d316177f238f2c0ffa62428c63a24edfc24fefdc6cfa6c4af30d2290973875'
LOCKED_PREVIOUS = '5a9b4a56e50d503f1ef7c37bbcb528bb02c24efd4d75e8bd69f0f1aa5e2e566f'


def collect(arm):
    trace = RUN / 'branches' / arm / 'trace.npz'
    with np.load(trace, allow_pickle=False) as a:
        t, t_sub, force, duration = (a[key] for key in ('time','substep_time','substep_skin_normal','substep_duration'))
        assert np.allclose(t, np.arange(len(t))*base.DT, atol=base.TOL, rtol=0)
        assert np.allclose(t_sub, np.arange(len(t_sub))*base.SUBSTEP, atol=base.TOL, rtol=0)
        assert np.allclose(duration, base.SUBSTEP, atol=base.TOL, rtol=0)
        assert np.isfinite(force).all() and np.all(force >= 0) and force.shape == (len(t_sub),5)
        origin_index = round(ORIGIN / base.DT)
        origin_substep = round(ORIGIN / base.SUBSTEP)
        j = origin_index-base.HISTORY+1
        sources = slice(j, origin_index+base.REPLAY_STEPS)
        hand = [json.loads(str(v)) for v in a['observation_json'][sources]]
        receiver = [json.loads(str(v)) for v in a['receiver_measurement_json'][sources]]
        phases = [json.loads(str(v)) for v in a['receiver_release_tasks_json'][sources]]
        assert base.transfer(phases[origin_index-j-1])
    onsets = [base.short_loss_onsets(force[:,d]) for d in range(5)]
    stats = dict(trace_sha256=base.sha(trace), phase_ticks=0, clock_or_valid_excluded=0, right_censored_ticks=0,
                 unloaded_or_no_contact=0,
                 little_onsets_ms=[round((x-origin_substep)*base.SUBSTEP*1000,5)
                                   for x in onsets[4] if origin_substep < x < origin_substep+400])
    rows = {}
    for k in range(origin_index, origin_index+base.REPLAY_STEPS):
        if not base.transfer(phases[k-j-1]):
            continue
        stats['phase_ticks'] += 1
        issue = k*round(base.DT/base.SUBSTEP)
        if issue+base.HORIZON+base.RECOVERY >= len(t_sub):
            stats['right_censored_ticks'] += 1
            continue
        indexes = range(k-base.HISTORY+1,k+1)
        readings = [hand[i-j] for i in indexes]
        receives = [receiver[i-j] for i in indexes]
        if not all(base.receiver_valid(r,t[i]) and abs(o['time']-t[i]) <= base.TOL
                   and o['sample_time'] is not None and abs(o['sample_time']-(t[i]-.010)) <= base.TOL
                   for i,o,r in zip(indexes,readings,receives)):
            stats['clock_or_valid_excluded'] += 5
            continue
        load = np.array([r['normal_load_N'] for r in receives])
        for d in range(5):
            normal = np.array([o['normal'][d] for o in readings])
            if not np.isfinite(normal).all() or not all(o['valid'][d] is True for o in readings):
                stats['clock_or_valid_excluded'] += 1
                continue
            if not readings[-1]['contact_present'][d] or normal[-1] < base.PRESSURE:
                stats['unloaded_or_no_contact'] += 1
                continue
            xx = (normal[-1], normal[-1]-normal[-2], np.mean(normal), np.mean(np.abs(np.diff(normal))),
                  load[-1], load[-1]-load[-2], np.mean(np.abs(np.diff(load))))
            events = tuple((arm,d,int(s)) for s in onsets[d] if issue < s <= issue+base.HORIZON)
            rows[(k-origin_index,d)] = base.Opportunity(arm, tuple(float(x) for x in xx), issue, events)
    return rows, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    out = p.parse_args().out
    complete = RUN/'execution_complete.json'
    assert base.sha(complete) == LOCKED_EXECUTION
    assert base.sha(PREVIOUS) == LOCKED_PREVIOUS
    prior = base.read(PREVIOUS)
    assert prior['code_sha256'] == base.sha(Path(base.__file__))
    assert base.read(complete)['passed'] and base.read(RUN/'zero_reproduction.json')['passed']
    train = [row for record in base.records()['V21'] if record['context_key'][2] == TRAIN_SEED
             for row in base.collect('V21',record)[0]]
    original = next(f for f in prior['folds'] if f['fold'] == 'V21_other_seed_to_V21_619741')
    y = np.array([bool(r.events) for r in train])
    assert len(train) == original['train_rows'] and int(y.sum()) == original['train_positive']
    x = np.array([r.features for r in train])
    counts = {ctx:sum(r.context==ctx for r in train) for ctx in {r.context for r in train}}
    weight = np.array([1./counts[r.context] for r in train])
    weight /= weight.sum()
    arm_data, arm_stats = {}, {}
    for arm in ARMS:
        arm_data[arm], arm_stats[arm] = collect(arm)
    initial = (0,4)
    assert all(initial in d for d in arm_data.values())
    assert arm_data['zero'].keys() == arm_data['zero_repeat'].keys()
    assert all(tuple(arm_data['zero'][k].features)==tuple(arm_data['zero_repeat'][k].features) and
               tuple(e[1:] for e in arm_data['zero'][k].events)==tuple(e[1:] for e in arm_data['zero_repeat'][k].events)
               for k in arm_data['zero'])
    for arm in ARMS:
        keys, rows = list(arm_data[arm]), list(arm_data[arm].values())
        truth = np.array([bool(r.events) for r in rows])
        xx = np.array([r.features for r in rows])
        values = {}
        for name,width in base.MODELS.items():
            fit,scores = base.logistic(x[:,:width],y,xx[:,:width],weight)
            threshold = float(np.quantile(fit, base.ALARM_QUANTILE, method='higher'))
            assert abs(threshold-original['model'][name]['threshold']) <= 1e-12
            score_little = [dict(issue_ms=k[0]*5, score=float(s), alarm=bool(s>=threshold),
                                 events_ms=[round((e[2]*base.SUBSTEP-ORIGIN)*1000,5) for e in row.events])
                            for k,row,s in zip(keys,rows,scores) if k[1]==4 and k[0]<=8]
            values[name] = dict(**base.metrics(rows,truth,scores,threshold), little_first_40ms=score_little)
        left, right = arm_data['zero'], arm_data[arm]
        first = next((k[0]*5 for k in sorted(left.keys() | right.keys())
                      if k not in left or k not in right or left[k].features != right[k].features), None)
        arm_stats[arm].update(eligible_rows=len(rows),positive_rows=int(truth.sum()),
                              first_feature_difference_from_zero_ms=first,models=values)
    assert len({arm_data[a][initial].features for a in ARMS}) == 1
    artifact = dict(schema='v26_risk_intervention_replay_v1',physics_steps=0,origin_s=ORIGIN,
                    training_seed=TRAIN_SEED,test_seed=619741,training_rows=len(train),
                    physical_test_checkpoints=1,shared_zero_duplicate=True,action_is_model_input=False,
                    same_little_features_at_origin=True,arm_stats=arm_stats,
                    hashes=dict(execution=LOCKED_EXECUTION,previous_replay=LOCKED_PREVIOUS,
                                previous_code=prior['code_sha256'],code=base.sha(Path(__file__))))
    with out.open('x',encoding='utf-8') as f:
        json.dump(artifact,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(output=str(out),sha256=base.sha(out)),ensure_ascii=False))


if __name__=='__main__':
    main()

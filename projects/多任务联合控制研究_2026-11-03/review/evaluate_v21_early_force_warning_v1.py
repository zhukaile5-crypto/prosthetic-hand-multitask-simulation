"""Check whether the archived public force readings could warn before an early limit breach."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION = ROOT/'review/v21_stage_attribution_v1.json'
PROTOCOL = ROOT/'protocol.json'
TOL, DT = 1e-9, .005
EVAL_START, EVAL_END = 8.9, 11.
LOOKBACK_S, MIN_LEAD_S = .050, .005
DIGITS = ('thumb', 'index', 'middle', 'ring', 'little')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def first(t, mask):
    rows = np.flatnonzero(mask)
    return None if not len(rows) else float(t[rows[0]])


def violates(f, limits, budget, numerical_tol):
    return np.any(f>limits+numerical_tol,axis=-1) | (np.sum(f,axis=-1)>budget+numerical_tol)


def read_row(text, time):
    o = json.loads(str(text))
    assert abs(o['time']-time)<=TOL and o['sample_time']<=time+TOL
    assert abs(o['sample_time']-(time-.010))<=TOL and abs(o['age']-.010)<=TOL
    f=np.asarray(o['normal'],float)
    assert f.shape==(5,) and len(o['valid'])==5 and np.all(f>=0)
    return f,o['sample_time'],bool(all(o['valid']) and np.isfinite(f).all())


def analyze(row, numerical_tol):
    record=row['metrics']['full_V21']
    trace=ROOT/record['trace']
    assert sha(trace)==record['trace_sha256']
    with np.load(trace,allow_pickle=False) as a:
        t,st=a['time'],a['substep_time']
        f=a['substep_skin_normal']
        target=a['substep_normal']
        duration=a['substep_duration']
        obs=a['observation_json']
        collision=a['collision']
        available=a['prediction_available']
        meta=json.loads(str(a['metadata'].item()))
    assert np.array_equal(t,np.arange(3201)*DT)
    limits=np.asarray(meta['limits'])
    budget=meta['budget']
    raw_breach=violates(f,limits,budget,numerical_tol)
    earliest=first(st,raw_breach)
    assert earliest == record['first_guard_breach_s']
    if earliest is not None:
        assert EVAL_START < earliest < EVAL_END
    span = (t>=EVAL_START-TOL) & (t<EVAL_END-TOL)
    use=np.flatnonzero(span)
    times=t[use]
    observed=[read_row(obs[i],t[i]) for i in range(use[0]-1,use[-1]+1)]
    f_now=np.asarray([r[0] for r in observed[1:]])
    ages=times-np.array([r[1] for r in observed[1:]])
    validity=np.asarray([r[2] for r in observed])
    growth=np.maximum((f_now-np.asarray([r[0] for r in observed[:-1]]))/DT,0.)
    extrapolated=f_now+(ages[:,None]+DT)*growth
    forecasts={'read_pressure':(f_now,validity[1:]),'positive_slope_15ms':(extrapolated,validity[1:]&validity[:-1])}
    stats={}
    for name,(pred,valid) in forecasts.items():
        alarms=valid & violates(pred,limits,budget,numerical_tol)
        detail=dict(eligible=int(valid.sum()), alarms=int(alarms.sum()), first_alarm_s=first(times,alarms))
        if earliest is not None:
            window=(times>=earliest-LOOKBACK_S-TOL)&(times<earliest-TOL)
            actionable=window&(times<=earliest-MIN_LEAD_S+TOL)
            first_after=first(times,alarms&(times>=earliest-TOL))
            before=first(times,alarms&window)
            detail.update(warnings_in_prior_50ms=int(np.sum(alarms&window)),
                          warning_at_least_5ms_early_s=first(times,alarms&actionable),
                          first_prior_50ms_s=before, first_at_or_after_breach_s=first_after,
                          first_after_delay_ms=None if first_after is None else (first_after-earliest)*1000.)
        stats[name]=detail
    first_collision=first(t,collision>.02)
    event=None
    if earliest is not None:
        k=np.flatnonzero(raw_breach)[0]
        j=int(np.argmax(f[k]-limits))
        last=use[t[use]<earliest-TOL][-1]
        p,acq,valid=read_row(obs[last],t[last])
        gaps=np.flatnonzero(~raw_breach[k:])
        stop=len(f) if not len(gaps) else k+gaps[0]
        window=(st>=earliest-TOL)&(st<earliest+LOOKBACK_S-TOL)
        event=dict(time_s=earliest,digit=DIGITS[j],first_breach_skin_N=f[k].tolist(),
                   target_only_N=target[k].tolist(),combined_budget_exceeded=bool(f[k].sum()>budget+numerical_tol),
                   first_contiguous_duration_ms=float(np.sum(duration[k:stop])*1000),
                   peak_skin_first_50ms_N=float(np.max(f[window])), actual_violating_substeps_first_50ms=int(raw_breach[window].sum()),
                   last_read_s=float(t[last]),last_sample_s=acq,last_read_valid=valid,last_observed_pressure_N=p.tolist())
    return dict(context=row['context'],trace=record['trace'],trace_sha256=record['trace_sha256'],
                limits_N=limits.tolist(),budget_N=budget,issue_count=len(use),
                archived_model_prediction_available=int(np.sum(available[use])),first_physical_fixture_readout_s=first_collision,
                first_force_event=event,alarms=stats)


def prefix_pair(attack,nominal):
    ta=ROOT/attack['trace']
    tb=ROOT/nominal['trace']
    with np.load(ta,allow_pickle=False) as a,np.load(tb,allow_pickle=False) as b:
        t=a['time']
        assert np.array_equal(t,b['time'])
        oa=a['observation_json']
        ob=b['observation_json']
        mt=[json.loads(str(x['metadata'].item())) for x in (a,b)]
        assert all(mt[0][field]==mt[1][field] for field in ('limits','budget','noise'))
        idx=np.flatnonzero(oa!=ob)
        if not len(idx):
            return dict(first_public_difference_s=None,keys=[],same_limits_budget_noise=True)
        k=idx[0]
        pa,pb=json.loads(str(oa[k])),json.loads(str(ob[k]))
        keys=[key for key in sorted(pa.keys()|pb.keys())
              if json.dumps(pa.get(key),sort_keys=True) != json.dumps(pb.get(key),sort_keys=True)]
        return dict(first_public_difference_s=float(t[k]),keys=keys,same_limits_budget_noise=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    archive=read(ATTRIBUTION)
    assert archive['contexts']==16 and archive['source_sha256']['protocol.json']==sha(PROTOCOL)
    tol=read(PROTOCOL)['thresholds']['force_numerical_tolerance_N']
    rows=[analyze(row,tol) for row in archive['rows']]
    records={tuple(row['context']):row for row in rows}
    pairs=[]
    for (scene,mode,seed),row in records.items():
        if scene=='collision':
            paired=prefix_pair(row,records['nominal',mode,seed])
            pairs.append(dict(context=row['context'],breach_s=row['first_force_event']['time_s'],**paired))
    artifact=dict(schema='v21_early_force_public_warning_replay_v1',physics_steps=0,rows=rows,collision_nominal_pairs=pairs,
                  timing=dict(evaluation_window_s=[EVAL_START,EVAL_END],lookback_s=LOOKBACK_S,min_lead_s=MIN_LEAD_S,
                              forecast='latest normal plus nonnegative slope over device age plus 5 ms',
                              alarm_threshold='same fixed local limits and total budget plus 0.020N tolerance'),
                  sources={x.name:sha(x) for x in (ATTRIBUTION,PROTOCOL,Path(__file__))})
    payload=json.dumps(artifact,indent=2,ensure_ascii=False,allow_nan=False)
    with args.out.open('x',encoding='utf-8') as f:
        f.write(payload+'\n')
    print(json.dumps(dict(output=str(args.out),sha256=sha(args.out)),ensure_ascii=False))


if __name__=='__main__':
    main()

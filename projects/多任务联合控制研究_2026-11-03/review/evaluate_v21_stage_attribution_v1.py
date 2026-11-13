"""Offline phase attribution for the archived V21/V20/PI same-condition matrix."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRIAD = ROOT / 'review/v21_complete16_release_variation_v1.json'
V21 = ROOT / 'review/v21_complete16_auxiliary_v1.json'
PLAN = ROOT / 'review/v21_complete16_plan_v1.json'
ROLES = ('full_V21', 'full_V20', 'same_prior_legacy_ratio')
CLOCK_TOL = 1e-8
FIELDS = ('executed_reference21', 'ctrl', 'ctrl_before', 'qpos', 'qvel',
          'normal', 'skin_normal', 'position', 'rotation')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def index(records, mode):
    result = {}
    for record in records:
        if mode != record['record_key'][0]:
            continue
        key = tuple(record['context_key'] if 'context_key' in record else record['record_key'][1:])
        if key in result:
            raise ValueError(f'Duplicate archived record {mode} {key}')
        result[key] = record
    return result


def first(time, mask):
    hits = np.flatnonzero(mask)
    return None if not len(hits) else float(time[hits[0]])


def load_archive(record, aux, tolerance):
    trace = (Path(record['run'])/'trace.npz').resolve()
    assert trace.is_relative_to((ROOT/'output').resolve()) and sha(trace) == record['trace_sha256']
    assert trace.parent == Path(aux['run']).resolve()
    auxiliary_trace_hash = aux['input_sha256'].get('trace.npz')
    if auxiliary_trace_hash is not None:
        assert auxiliary_trace_hash == record['trace_sha256']
    delivery = trace.parent/'delivery_intervals.npz'
    assert sha(delivery) == aux['input_sha256']['delivery_intervals.npz']
    with np.load(trace, allow_pickle=False) as raw:
        t = raw['time']
        st = raw['substep_time']
        f = raw['substep_skin_normal']
        numeric = {k:raw[k] for k in FIELDS}
        phase = [json.loads(str(x)) for x in raw['receiver_release_tasks_json']]
        meta = json.loads(str(raw['metadata'].item()))
    assert np.array_equal(t, np.arange(3201)*.005)
    assert st.shape == (32000,) and f.shape == (32000,5)
    assert np.allclose(st, np.arange(32000)*.0005, atol=CLOCK_TOL, rtol=0.)
    assert np.isfinite(f).all() and np.all(f>=0) and all(a.shape[0]==len(t) for a in numeric.values())
    limits, budget = np.array(meta['limits']), float(meta['budget'])
    assert np.array_equal(limits, aux['limits']) and budget == aux['budget']
    augmented_force = np.vstack((f,numeric['skin_normal'][-1]))
    force_time = np.r_[st,t[-1]]
    breach = np.any(augmented_force>limits+tolerance,axis=1) | (augmented_force.sum(axis=1)>budget+tolerance)
    peak_row, peak_finger = np.unravel_index(np.argmax(augmented_force), augmented_force.shape)
    release = float(meta['mechanical_release_at'] if meta.get('mechanical_release_at') is not None else meta['release_at'])
    result = record['original_result']['research_metrics']
    assert np.array_equal(np.max(augmented_force,axis=0),result['safety']['finger_peak_N'])
    assert bool(not breach.any()) is result['safety']['guard_pass']
    impact = aux['auxiliary']['first_receiver_impact20ms']
    start = impact['start_s']
    end = impact['end_requested_s']
    assert impact['available'] and impact['coverage']['complete'] and impact['coverage']['all_valid']
    with np.load(delivery,allow_pickle=False) as raw:
        ts,te = raw['interval_start_s'],raw['interval_end_s']
        support = raw['target_receiver_top_normal_N']
        contact_valid = raw['contact_valid']
        take = (te > start+CLOCK_TOL) & (ts < end-CLOCK_TOL)
        ids = np.flatnonzero(take)
        assert len(ids) and np.all(contact_valid[ids])
        pick = ids[np.argmax(support[ids])]
        assert float(support[pick]) == impact['contact']['receiver_top']['peak_N']
        peak_impact_time = float(ts[pick])
    fields = dict(role=record['record_key'][0],trace=trace.relative_to(ROOT).as_posix(),
                  trace_sha256=record['trace_sha256'],delivery_sha256=aux['input_sha256']['delivery_intervals.npz'],
                  release_s=release,first_transfer_s=next((float(p['time_s']) for p in phase if p and p['phase']=='TRANSFER'),None),
                  guard_pass=not breach.any(), first_guard_breach_s=first(force_time,breach),
                  first_guard_breach_after_release_s=first(force_time,breach&(force_time>=release-CLOCK_TOL)),
                  peak_finger_N=float(augmented_force[peak_row,peak_finger]), peak_finger_s=float(force_time[peak_row]),
                  first_receiver_contact_s=float(start), receiver_peak_s=peak_impact_time,
                  receiver_peak_N=float(support[pick]), floor_event=aux['auxiliary']['ever_floor_contact'],
                  final_low_motion=aux['metrics'].get('final_low_motion_suffix_found'))
    return fields, t, st, numeric, f


def compare(left, right):
    a,t,st,x,fx = left
    b,tb,stb,y,fy = right
    assert np.array_equal(t,tb) and np.array_equal(st,stb)
    changes={}
    for k in FIELDS:
        assert x[k].shape == y[k].shape
        equal=(x[k]==y[k]) | (np.isnan(x[k])&np.isnan(y[k]))
        changes[k]=first(t, ~equal.reshape(len(t),-1).all(axis=1))
    execution = changes['executed_reference21']
    return dict(first_200Hz_difference_s=changes, first_2kHz_skin_difference_s=first(st,np.any(fx!=fy,axis=1)),
                V21_guard_breach_before_first_command=(a['first_guard_breach_s'] is not None and
                    (execution is None or a['first_guard_breach_s']<execution-CLOCK_TOL)),
                V21_receiver_peak_before_first_command=(execution is None or a['receiver_peak_s']<execution-CLOCK_TOL),
                reference_receiver_peak_before_first_command=(execution is None or b['receiver_peak_s']<execution-CLOCK_TOL),
                V21_peak_finger_before_first_command=(execution is None or a['peak_finger_s']<execution-CLOCK_TOL),
                reference_peak_finger_before_first_command=(execution is None or b['peak_finger_s']<execution-CLOCK_TOL))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    output=parser.parse_args().out
    plan=read(PLAN)
    protocol=ROOT/'protocol.json'
    assert sha(protocol)==plan['source_sha256']['protocol.json']
    tolerance=read(protocol)['thresholds']['force_numerical_tolerance_N']
    roster=read(TRIAD)
    matrix={r:index(roster['records']+roster['reference_records'],r) for r in ROLES}
    refs={r:ROOT / plan['evidence_pins'][key]['path'] for r,key in
          ((ROLES[1],'historicalV20_complete16_auxiliary'),(ROLES[2],'historical80_auxiliary'))}
    for r,key in ((ROLES[1],'historicalV20_complete16_auxiliary'),(ROLES[2],'historical80_auxiliary')):
        assert sha(refs[r])==plan['evidence_pins'][key]['sha256']
    refs[ROLES[0]]=V21
    auxiliary={role:index(read(path)['records'],role) for role,path in refs.items()}
    universe=set(matrix[ROLES[0]])
    assert len(universe)==16 and all(set(d)==universe for d in list(matrix.values())+list(auxiliary.values()))
    pairing=read(V21)['pairs']
    assert all(len([r for r in pairing if r['reference_role']==role and r['same_physical_noise_limits']])==16
               for role in ROLES[1:])
    for row in pairing:
        role = row['reference_role']
        if role in ROLES[1:]:
            auxiliary[role][tuple(row['context_key'])].setdefault('metrics', row['reference_metrics'])
    rows=[]
    for ctx in sorted(universe):
        archive={role:load_archive(matrix[role][ctx],auxiliary[role][ctx],tolerance) for role in ROLES}
        rows.append(dict(context=list(ctx),metrics={role:archive[role][0] for role in ROLES},
                         contrasts={role:compare(archive[ROLES[0]],archive[role]) for role in ROLES[1:]}))
    result=dict(schema='v21_stage_attribution_v1',physics_steps=0,contexts=len(rows),physical_seed_clusters=2,
                raw_traces_checked=48, raw_delivery_ledgers_checked=48, guard_tolerance_N=tolerance,
                source_sha256={p.name:sha(p) for p in (TRIAD,PLAN,protocol,*refs.values(),Path(__file__))},
                source_boundary='ordered temporal attribution in matched archived traces, not individual mechanism ablation',
                rows=rows)
    with output.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(output=str(output),sha256=sha(output)),ensure_ascii=False))


if __name__=='__main__':
    main()

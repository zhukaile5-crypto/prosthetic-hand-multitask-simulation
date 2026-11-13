"""Independent recovery classifications from small saved truth arrays only."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,runpy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'output/confirmation_integration_v2'
helpers=runpy.run_path(str(ROOT/'review/independent_confirmation_review.py'))
dwell=helpers['dwell'];bad_dwell=helpers['first_bad_dwell']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def key(r):return r['scenario'],r['shape'],r['seed'],r['method']


def equal(a,b):
    if a is None or b is None:return a is b
    if isinstance(a,str) or isinstance(b,str):return a==b
    return bool(np.allclose(a,b,atol=1e-8,rtol=0.))


def classify(t,stable,spec,cutoff,failure,collision,reason):
    commanded=float(spec['start']);requires=bool(spec.get('requires_actual_contact'))
    hits=np.flatnonzero((t>=commanded-1e-9)&(t<min(cutoff,failure if failure is not None else cutoff)-1e-9)&(collision>.02)) if requires else []
    exposure=float(t[hits[0]]) if len(hits) else None
    exposed=not requires or exposure is not None
    start=max(commanded,exposure) if requires and exposure is not None else commanded
    if failure is not None and start<failure<cutoff:cutoff=failure;reason='task_failure'
    out=dict(observation_end=cutoff,cutoff_reason=reason,exposed=exposed,actual_exposure_at=exposure,
        analysis_start=start,exposure_latency=None if exposure is None else exposure-commanded,
        stable_before=False,first_instability=None,stable_confirmation=None,latency_from_event=None,
        latency_from_exposure=None,latency_from_instability=None,recovered=False,censored=False)
    if failure is not None and failure<=start+1e-9:out['status']='failed_before';return out
    if not exposed:out['status']='not_exposed';return out
    before=(t>=start-.2-1e-9)&(t<start-1e-9)
    out['stable_before']=bool(before.sum()==40 and stable[before].all())
    if not out['stable_before']:out['status']='not_stable_before';return out
    event_window=(t>=start-1e-9)&(t<cutoff-1e-9)
    bad=np.flatnonzero(event_window&~stable)
    if not len(bad):out['status']='never_unstable';return out
    instability=float(t[bad[0]]);out['first_instability']=instability
    # Independently find consecutive stable index runs and use their first
    # 200ms confirmation. The right boundary is excluded, including release.
    indexes=np.flatnonzero(event_window&(t>=instability-1e-9)&stable)
    groups=np.split(indexes,np.flatnonzero(np.diff(indexes)>1)+1)
    confirmation=None
    for group in groups:
        if len(group) and t[group[-1]]-t[group[0]]>=.2-1e-9:
            i=group[0]+int(round(.2/.005));confirmation=float(t[i]);break
    if confirmation is None:
        out.update(status='release_preempt' if reason=='release' else 'unrecovered',censored=True)
    else:
        out.update(status='recovered_during' if spec['end'] is None or confirmation<spec['end']-1e-9 else 'recovered_after',
            stable_confirmation=confirmation,recovered=True,latency_from_event=confirmation-commanded,
            latency_from_exposure=None if exposure is None else confirmation-exposure,
            latency_from_instability=confirmation-instability)
    return out


def main():
    path=FOLDER/'recovery_secondary_v1.json';result=json.loads(path.read_text(encoding='utf-8'))
    plan_path=ROOT/'review/recovery_plan_v1.json';plan=json.loads(plan_path.read_text(encoding='utf-8'))
    trials=json.loads((FOLDER/'trials.json').read_text(encoding='utf-8'));primary={key(r):r for r in trials}
    checks=dict(full_256_assignments=len(result['records'])==256 and {key(r) for r in result['records']}==set(primary),
        secondary_hash=sha(path)=='5c8d869b5c1163ea4848285925c43fdcd487ae1ea86b2f40c98081ccf0c121ab',
        source_plan_binding=result['recovery_plan_sha256']==sha(plan_path) and result['analyzer_sha256']==sha(ROOT/'evaluate_recovery.py'))
    independent_events=[]
    names=('time','normal','skin_normal','position','lift','lift_ref','support','relative','relative_velocity','angular_velocity','collision','metadata')
    for row in result['records']:
        identity=key(row);p=primary[identity];trace=Path(p['trace_path'])
        with np.load(trace,allow_pickle=False) as a:z={k:a[k] for k in names}
        m=json.loads(str(z.pop('metadata')));t=z['time'];n=z['normal'];skin=z['skin_normal']
        support=(n[:,0]>.2)&((n>.2).sum(axis=1)>=3)
        raised=z['position'][:,2]>=np.asarray(m['initial_position'])[2]+.025
        lift_error=abs(z['lift']-z['lift_ref']);motion=(lift_error<=.01)&(z['support']<=.05)
        error=np.full(len(t),np.inf) if m.get('reference_relative') is None else np.linalg.norm(z['relative']-m['reference_relative'],axis=1)
        safe=(skin<=np.asarray(m['limits'])+.02).all(axis=1)&(skin.sum(axis=1)<=m['budget']+.02)
        stable=support&raised&motion&safe&(error<=.005)&(np.linalg.norm(z['relative_velocity'],axis=1)<=.005)&(np.linalg.norm(z['angular_velocity'],axis=1)<=.15)
        established=dwell(t,support&raised&motion,0.,7.5,.1)
        failure=7.5 if not m.get('handoff_passed') or established is None else None
        if failure is None:
            contact=bad_dwell(t,~support|~raised|(error>.08),7.5,12.,.1)
            motionfail=bad_dwell(t,lift_error>.01,7.5,12.,.3)
            candidates=[v for v in (contact,motionfail) if v is not None]
            failure=min(candidates) if candidates else None
        checks[str(identity)+'/primary_failure_boundary']=equal(failure,p.get('failure_at'))
        checks[str(identity)+'/trace_identity']=row.get('trace_sha256')==p.get('trace_sha256')
        specs=plan['events'][row['scenario']]
        events={e['event_id']:e for e in row['events']}
        checks[str(identity)+'/exact_planned_events']=len(events)==len(row['events'])==len(specs) and set(events)=={s['event_id'] for s in specs}
        for index,spec in enumerate(specs):
            next_start=specs[index+1]['start'] if index+1<len(specs) else 12.
            reason='next_event' if next_start<12.-1e-9 else 'release'
            expected=classify(t,stable,spec,min(next_start,12.),failure,z['collision'],reason)
            actual=events[spec['event_id']]
            checks[str(identity)+'/'+spec['event_id']+'/all_classification_fields']=all(equal(value,actual.get(k)) for k,value in expected.items())
            independent_events.append(dict(method=row['method'],**expected))
    for row in result['rows']:
        selected=[e for e in independent_events if e['method']==row['method']]
        eligible=[e for e in selected if e['stable_before'] and e['exposed'] and e['status'] not in ('failed_before','invalid_record')]
        unstable=[e for e in eligible if e['first_instability'] is not None]
        recovered=[e for e in unstable if e['recovered']]
        summary=dict(scheduled_events=len(selected),status_counts=dict(Counter(e['status'] for e in selected)),
            eligible_events=len(eligible),unstable_eligible_events=len(unstable),recovered_events=len(recovered),
            recovery_fraction=len(recovered)/len(unstable) if unstable else None,
            recovered_latency_median=float(np.median([e['latency_from_instability'] for e in recovered])) if recovered else None)
        checks[row['method']+'/all_summary_denominators']=all(row[k]==v for k,v in summary.items())
    certificate=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
        created_utc=datetime.now(timezone.utc).isoformat(),trials=256,events=len(independent_events),
        secondary_sha256=sha(path),plan_sha256=sha(plan_path),primary_trials_sha256=sha(FOLDER/'trials.json'),
        independent_script_sha256=sha(__file__),
        scope='Independent stability, failure boundaries and event-state recomputation from saved truth arrays. Raw full-file hashes/noise/actions are covered by concurrent independent confirmation audit. No simulator/controller/evaluator imports.')
    out=ROOT/('review/independent_recovery_data_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.write_text(json.dumps(certificate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=certificate['passed'],checks=len(checks),failed=certificate['failed'],path=str(out),sha256=sha(out)),ensure_ascii=False))


if __name__=='__main__':main()

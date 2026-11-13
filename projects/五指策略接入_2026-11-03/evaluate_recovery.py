"""Secondary, offline event recovery analysis; no simulation or policy changes.

Written after the six integration development trials were visible and before
confirmation. This is descriptive supplementary analysis, not an unseen-data
primary endpoint. The primary evaluator, plan and admission remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from evaluate_runs import evaluate_trace, first_dwell
from session import ROOT, SCENARIOS, sha

DWELL = .2
PERIOD = .005
RELEASE = 12.


def event(name, start, end, kind, **extra):
    return dict(event_id=name, start=float(start), end=None if end is None else float(end),
                kind=kind, **extra)


PULSES = [event(f'pull_{i+1}', s, s+.3, 'external_pull')
          for i, s in enumerate((9.3, 10.3, 11.3))]
EVENTS = {s['name']: [] for s in SCENARIOS}
EVENTS.update({
    'friction_drop': [event('friction_step', 9.3, None, 'persistent_friction_change')],
    'three_pulses': PULSES,
    'local_limit': PULSES,
    'load_ramp': [event('load_ramp', 9.3, None, 'persistent_added_load', ramp_end=11.3)],
    'fragile_conflict': [event('pull_and_squeeze', 9.3, 10., 'external_pull_and_squeeze')],
    'release_slip': [event('pull_across_release', 11.7, 12.4, 'external_pull')],
    'dynamic_transport': [event('lift_and_wrist_motion', 9.3, 12., 'motion_reference_change')],
    'sensor_dropout': [event('local_dropout', 9.3, 9.7, 'sensor_dropout'),
                       event('all_dropout', 11.3, 11.4, 'sensor_dropout')],
    'collision': [event('obstacle_approach', 9.3, 10.3, 'obstacle_command',
                        requires_actual_contact=True)],
    'contact_robustness': [event('friction_and_pull_1', 9.3, None,
                                'persistent_friction_change_and_pull', pull_end=9.6),
                           PULSES[1], PULSES[2]],
})


def stable_mask(trace, meta):
    """Exact frozen evaluate_runs.py stable predicate, without new thresholds."""
    normal=np.asarray(trace['normal']); skin=np.asarray(trace['skin_normal'])
    limits=np.asarray(meta['limits']); contact=normal>.2
    support=contact[:,0] & (contact.sum(axis=1)>=3)
    motion=(np.abs(trace['lift']-trace['lift_ref'])<=.01) & (trace['support']<=.05)
    initial=np.asarray(meta.get('initial_position',trace['position'][0]),float)
    raised=trace['position'][:,2]>=initial[2]+.025
    ref=meta.get('reference_relative')
    error=np.full(len(normal),np.inf) if ref is None else np.linalg.norm(trace['relative']-np.asarray(ref),axis=1)
    safe=np.all(skin<=limits[None,:]+.02,axis=1) & (skin.sum(axis=1)<=float(meta['budget'])+.02)
    return (support & motion & raised & safe & (error<=.005) &
            (np.linalg.norm(trace['relative_velocity'],axis=1)<=.005) &
            (np.linalg.norm(trace['angular_velocity'],axis=1)<=.15))


def classify_event(t, stable, spec, observation_end, *, failure_at=None,
                   record_complete=True, exposed=True, actual_exposure_at=None,
                   cutoff_reason='release'):
    """Exclusive right boundary prevents the next event/release from counting."""
    t=np.asarray(t,float); stable=np.asarray(stable,bool); planned_start=spec['start']
    start=max(planned_start,float(actual_exposure_at)) if spec.get('requires_actual_contact') and actual_exposure_at is not None else planned_start
    end=float(observation_end)
    if failure_at is not None and start<failure_at<end:
        end=float(failure_at); cutoff_reason='task_failure'
    row={**spec, 'observation_end':end, 'cutoff_reason':cutoff_reason,
         'exposed':bool(exposed), 'actual_exposure_at':actual_exposure_at,
         'analysis_start':start, 'exposure_latency':None if actual_exposure_at is None else float(actual_exposure_at-planned_start),
         'stable_before':False, 'first_instability':None, 'stable_confirmation':None,
         'latency_from_event':None, 'latency_from_exposure':None, 'latency_from_instability':None,
         'recovered':False, 'censored':False}
    if not record_complete:
        row['status']='invalid_record'; return row
    if failure_at is not None and failure_at<=start+1e-9:
        row['status']='failed_before'; return row
    if not exposed:
        row['status']='not_exposed'; return row
    before=(t>=start-DWELL-1e-9)&(t<start-1e-9)
    # Forty acquired 5ms samples spanning [start-.2,start), not one stable point.
    row['stable_before']=bool(np.count_nonzero(before)==round(DWELL/PERIOD) and np.all(stable[before]))
    if not row['stable_before']:
        row['status']='not_stable_before'; return row
    during=(t>=start-1e-9)&(t<end-1e-9)
    if not np.any(during):
        row.update(status='invalid_record',censored=True); return row
    bad=np.flatnonzero(during & ~stable)
    if not len(bad):
        row['status']='never_unstable'; return row
    onset=float(t[bad[0]]);row['first_instability']=onset
    confirmation=first_dwell(t,stable,onset,end-PERIOD,DWELL)
    if confirmation is not None:
        row.update(status='recovered_during' if spec['end'] is None or confirmation<spec['end']-1e-9 else 'recovered_after',
                   recovered=True,stable_confirmation=confirmation,
                   latency_from_event=float(confirmation-planned_start),
                   latency_from_exposure=None if actual_exposure_at is None else float(confirmation-actual_exposure_at),
                   latency_from_instability=float(confirmation-onset))
    else:
        row.update(status='release_preempt' if cutoff_reason=='release' else 'unrecovered',
                   censored=True)
    return row


def evaluate_saved_trace(path, plan):
    path=Path(path); primary=evaluate_trace(path)
    with np.load(path,allow_pickle=False) as saved:
        meta=json.loads(str(saved['metadata'])); trace={k:saved[k] for k in saved.files if k!='metadata'}
    specs=plan['events'][meta['scenario']['name']]
    result={k:primary.get(k) for k in ('run_id','method','scenario','shape','seed','split','mode','boundary','task_success')}
    result.update(trace_path=str(path.resolve()),trace_sha256=sha(path),
                  record_complete=primary['record_complete'],events=[],
                  recovery_applicable=bool(specs), no_event_reason=None if specs else
                  'No scheduled post-handoff event; constant geometry/load/noise conditions are not recovery onsets.')
    if meta['mode']!='scheduled':
        result.update(recovery_applicable=False,no_event_reason='Manual demonstrations are outside scheduled recovery statistics.');return result
    stable=stable_mask(trace,meta);t=np.asarray(trace['time'])
    for i,spec in enumerate(specs):
        next_start=specs[i+1]['start'] if i+1<len(specs) else RELEASE
        cutoff=min(next_start,RELEASE); reason='next_event' if next_start<RELEASE-1e-9 else 'release'
        exposure=None
        if spec.get('requires_actual_contact'):
            exposure_cutoff=min(cutoff,primary['failure_at']) if primary.get('failure_at') is not None else cutoff
            hits=np.flatnonzero((t>=spec['start']-1e-9)&(t<exposure_cutoff-1e-9)&(np.asarray(trace['collision'])>.02))
            exposure=float(t[hits[0]]) if len(hits) else None
        row=classify_event(t,stable,spec,cutoff,failure_at=primary.get('failure_at'),
             record_complete=primary['record_complete'],exposed=not spec.get('requires_actual_contact') or exposure is not None,
             actual_exposure_at=exposure,cutoff_reason=reason)
        result['events'].append(row)
    return result


def invalid_record(job, path, plan, reason):
    """A planned run with missing/corrupt truth retains every planned event."""
    scenario=job['scenario']['name'] if isinstance(job['scenario'],dict) else job['scenario']
    specs=plan['events'][scenario]
    return dict(method=job['method'],scenario=scenario,shape=job['shape'],seed=job['seed'],
        split=job.get('split','unknown'),mode=job.get('mode','scheduled'),
        task_success=False,record_complete=False,trace_path=str(path),trace_sha256=None,
        recovery_applicable=bool(specs),no_event_reason=None if specs else 'No scheduled post-handoff event.',
        invalid_reason=reason,events=[classify_event([],[],spec,min(specs[i+1]['start'] if i+1<len(specs) else RELEASE,RELEASE),
                  record_complete=False) for i,spec in enumerate(specs)])


def evaluate_folder(folder,plan):
    folder=Path(folder); manifest=folder/'manifest.json'
    if manifest.exists():
        jobs=json.loads(manifest.read_text(encoding='utf-8'))['planned_jobs']
        entries=[(j,folder/'runs'/f"{j['scenario']}__{j['shape']}__{j['method']}__{j['seed']}"/'trace.npz') for j in jobs]
    else:
        entries=[]
        for metadata in sorted(folder.glob('runs/*/metadata.json')):
            entries.append((json.loads(metadata.read_text(encoding='utf-8')),metadata.with_name('trace.npz')))
        if not entries:raise ValueError('No planned-job manifest or per-run metadata; full denominator cannot be established')
    records=[]
    for job,path in entries:
        try:
            if not path.exists():raise FileNotFoundError('Planned trial has no saved truth trace')
            records.append(evaluate_saved_trace(path,plan))
        except Exception as exc:
            records.append(invalid_record(job,path,plan,type(exc).__name__+': '+str(exc)))
    return records


def summarize(records):
    rows=[]
    for method in ('ratio','unified'):
        trials=[r for r in records if r['method']==method and r['mode']=='scheduled']
        events=[e for r in trials for e in r['events']]
        eligible=[e for e in events if e['stable_before'] and e['exposed'] and e['status'] not in ('failed_before','invalid_record')]
        unstable=[e for e in eligible if e['first_instability'] is not None]
        recovered=[e for e in unstable if e['recovered']]
        rows.append(dict(method=method,trials=len(trials),trials_with_events=sum(bool(r['events']) for r in trials),
             scheduled_events=len(events),status_counts=dict(Counter(e['status'] for e in events)),
             eligible_events=len(eligible),unstable_eligible_events=len(unstable),recovered_events=len(recovered),
             recovery_fraction=float(len(recovered)/len(unstable)) if unstable else None,
             recovered_latency_median=float(np.median([e['latency_from_instability'] for e in recovered])) if recovered else None,
             latency_scope='conditional on recovered; censored/never_unstable/missing are never zero latency'))
    return dict(schema='secondary_event_recovery_results_v1',role='descriptive_secondary_only',
                development_was_visible=True,confirmation_was_unseen_when_defined=True,
                rows=rows,records=records)


def make_plan():
    return dict(schema='secondary_recovery_plan_v1',status='fixed_before_confirmation',
        created_utc=datetime.now(timezone.utc).isoformat(),
        disclosure=dict(development_seen=True,development='output/development_integration_v1: six integration smoke trials',
                        confirmation_run=False,primary_endpoints_unchanged=True,not_an_unseen_development_endpoint=True),
        source_sha256={f:sha(ROOT/f) for f in ('session.py','evaluate_runs.py','evaluate_recovery.py','experiment_plan.json')},
        development_trials_sha256=sha(ROOT/'output/development_integration_v1/trials.json'),
        events=EVENTS, definitions=dict(sample_period_s=PERIOD,dwell_s=DWELL,release_s=RELEASE,
          stable='Exact evaluate_runs.py predicate: thumb plus >=3 finger contacts, raised object, lift/reference/table, relative pose and 3D velocities, same finger limits/budget.',
          precondition='All 40 samples of [analysis_start-.2,analysis_start) stable; analysis_start=planned onset except collision uses first actual contact; failure-at-or-before-analysis-start takes precedence.',
          horizon='[analysis_start,min(next_event_start,12s,failure_at)); no next event/release sample can complete recovery.',
          instability='First unstable 5ms sample in this exclusive event horizon.',
          recovery='First continuous .2s stable confirmation after first instability using frozen first_dwell.',
          during_after_boundary='Finite event end is exclusive: confirmation exactly at end is recovered_after, matching schedule t<end.',
          persistent='end=null means perturbation persists; recovery is during the changed condition, never after.',
          statuses=['invalid_record','failed_before','not_exposed','not_stable_before','never_unstable','recovered_during','recovered_after','unrecovered','release_preempt'],
          latency='Both event-to-confirmation and instability-to-confirmation; null for never_unstable, ineligible or censored.',
          collision='Requires measured collision>0.02N at 200Hz strictly before failure/cutoff. Analysis and pre-stability begin at actual contact, not command; planned onset and exposure latency remain explicit.',
          aggregation='Enumerate batch planned_jobs, including missing/corrupt traces as invalid_record for every planned event. Full event/status denominators; conditional latency and recovery coverage only descriptive. No extra p-values or change to primary A_S/TV/quality windows.'),
        limitations=['No causal proof that an instability was caused by the registered event.',
                     'Constant noisy/low-friction/heavy conditions with no post-handoff onset have no event-recovery row.',
                     'Persistent or overlapping changes are evaluated only until the next registered event; right censoring remains explicit.'])


def self_test():
    t=np.arange(0,16+PERIOD/2,PERIOD);good=np.ones(len(t),bool)
    spec=event('test',9.3,9.6,'synthetic');checks=[]
    def run(stable=good,**kw):return classify_event(t,stable,spec,kw.pop('observation_end',12.),**kw)
    checks.append(run()['status']=='never_unstable' and run()['latency_from_event'] is None)
    a=good.copy();a[(t>=9.3)&(t<9.35)]=False
    checks.append(run(a)['status']=='recovered_during')
    b=good.copy();b[(t>=9.3)&(t<9.7)]=False
    checks.append(run(b)['status']=='recovered_after')
    checks.append(abs(run(b)['latency_from_instability']-.6)<1e-9)
    c=good.copy();c[(t>=9.1)&(t<9.3)]=False
    checks.append(run(c)['status']=='not_stable_before')
    checks.append(run(failure_at=9.3)['status']=='failed_before')
    checks.append(run(exposed=False)['status']=='not_exposed')
    checks.append(run(record_complete=False)['status']=='invalid_record')
    d=good.copy();d[t>=9.3]=False
    checks.append(run(d)['status']=='release_preempt' and run(d)['censored'])
    checks.append(run(d,observation_end=10.3,cutoff_reason='next_event')['status']=='unrecovered')
    checks.append(run(d,failure_at=10.)['cutoff_reason']=='task_failure')
    e=good.copy();e[(t>=9.3)&(t<11.8)]=False
    checks.append(run(e)['status']=='release_preempt')
    persistent=classify_event(t,b,event('persist',9.3,None,'synthetic'),12.)
    checks.append(persistent['status']=='recovered_during')
    checks.append(set(EVENTS)=={s['name'] for s in SCENARIOS})
    collision_spec=event('collision',9.3,10.3,'synthetic',requires_actual_contact=True)
    early=good.copy();early[(t>=9.3)&(t<9.35)]=False
    collision_row=classify_event(t,early,collision_spec,12.,actual_exposure_at=9.8)
    checks.append(collision_row['status']=='never_unstable' and collision_row['stable_confirmation'] is None)
    checks.append(abs(collision_row['analysis_start']-9.8)<1e-12 and abs(collision_row['exposure_latency']-.5)<1e-12)
    missing=invalid_record(dict(method='ratio',shape='capsule',scenario='three_pulses',seed=1),Path('missing'),dict(events=EVENTS),'missing')
    checks.append(len(missing['events'])==3 and all(e['status']=='invalid_record' for e in missing['events']))
    boundary=good.copy();boundary[(t>=9.3)&(t<9.4)]=False
    checks.append(run(boundary)['status']=='recovered_after' and abs(run(boundary)['stable_confirmation']-9.6)<1e-9)
    if not all(checks):raise AssertionError(f'Recovery self-test failed: {checks}')
    return dict(passed=True,checks=len(checks))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--out')
    p.add_argument('--plan',default=str(ROOT/'review/recovery_plan_v1.json'))
    p.add_argument('--write-plan',action='store_true');p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    plan_path=Path(args.plan)
    if args.write_plan:
        if plan_path.exists():raise FileExistsError(plan_path)
        plan_path.write_text(json.dumps(make_plan(),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(plan_path);return
    if not args.root or not args.out:p.error('--root and --out are required for offline evaluation')
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if any(sha(ROOT/k)!=v for k,v in plan['source_sha256'].items()):raise RuntimeError('Supplementary plan source identity changed')
    target=Path(args.out)
    if target.exists():raise FileExistsError(target)
    records=evaluate_folder(args.root,plan)
    result=summarize(records);result.update(recovery_plan_sha256=sha(plan_path),analyzer_sha256=sha(__file__))
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(target),trials=len(records),events=sum(len(r['events']) for r in records))))


if __name__=='__main__':main()

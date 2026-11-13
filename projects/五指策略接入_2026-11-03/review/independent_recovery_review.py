"""AST-isolated synthetic review; never imports simulator/session/policy."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dwell(t, stable, start, end, duration):
    since=None
    for k in np.flatnonzero((t>=start-1e-9)&(t<=end+1e-9)):
        if stable[k]:
            since=float(t[k]) if since is None else since
            if t[k]-since>=duration-1e-9:
                return float(t[k])
        else:
            since=None
    return None


def main():
    source=ROOT/'evaluate_recovery.py'
    module=ast.parse(source.read_text(encoding='utf-8'))
    names=('classify_event','invalid_record','evaluate_folder')
    functions=[node for node in module.body if isinstance(node,ast.FunctionDef) and node.name in names]
    namespace=dict(np=np,DWELL=.2,PERIOD=.005,RELEASE=12.,first_dwell=dwell,Path=Path,json=json)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
    classify=namespace['classify_event']
    t=np.arange(3201)*.005
    good=np.ones(len(t),bool)
    spec=dict(event_id='isolated',start=9.3,end=9.6,kind='synthetic')
    def call(stable=None,**kw):
        return classify(t,good if stable is None else stable,spec,kw.pop('observation_end',12.),**kw)
    checks={}
    seen={}
    seen['never']=call()
    checks['never_unstable_not_zero_latency']=seen['never']['status']=='never_unstable' and seen['never']['latency_from_event'] is None and seen['never']['latency_from_instability'] is None
    unstable=good.copy();unstable[(t>=9.3-1e-9)&(t<9.35-1e-9)]=False
    seen['during']=call(unstable)
    checks['transient_recovery_during']=seen['during']['status']=='recovered_during' and seen['during']['recovered']
    unstable2=good.copy();unstable2[(t>=9.3-1e-9)&(t<9.7-1e-9)]=False
    seen['after']=call(unstable2)
    checks['transient_recovery_after']=seen['after']['status']=='recovered_after' and seen['after']['recovered']
    persistent=dict(spec,end=None)
    seen['persistent']=classify(t,unstable2,persistent,12.)
    checks['persistent_recovery_never_after']=seen['persistent']['status']=='recovered_during'
    prebad=good.copy();prebad[(t>=9.1-1e-9)&(t<9.3-1e-9)]=False
    checks['unstable_before_is_ineligible']=call(prebad)['status']=='not_stable_before'
    checks['failure_precedence']=call(failure_at=9.3,exposed=False)['status']=='failed_before'
    checks['invalid_record_precedence']=call(record_complete=False,failure_at=9.3)['status']=='invalid_record'
    checks['not_exposed_complete_status']=call(exposed=False)['status']=='not_exposed'
    neverback=good.copy();neverback[t>=9.3-1e-9]=False
    seen['release']=call(neverback)
    checks['release_right_censoring']=seen['release']['status']=='release_preempt' and seen['release']['censored'] and seen['release']['latency_from_event'] is None
    checks['next_event_right_censoring']=call(neverback,observation_end=10.3,cutoff_reason='next_event')['status']=='unrecovered'
    seen['task_failure']=call(neverback,failure_at=9.8)
    checks['failure_within_window_clipped']=seen['task_failure']['observation_end']==9.8 and seen['task_failure']['cutoff_reason']=='task_failure'
    releasetie=good.copy();releasetie[(t>=9.3-1e-9)&(t<11.8-1e-9)]=False
    checks['release_exact_confirmation_excluded']=call(releasetie)['status']=='release_preempt'
    nexttie=good.copy();nexttie[(t>=9.3-1e-9)&(t<10.1-1e-9)]=False
    checks['next_event_exact_confirmation_excluded']=call(nexttie,observation_end=10.3,cutoff_reason='next_event')['status']=='unrecovered'
    # This regression is deliberately expected to fail on the first candidate:
    # a command-time transient has already recovered before physical contact.
    collision=dict(spec,requires_actual_contact=True,kind='obstacle_command')
    seen['contact_after_recovery']=classify(t,unstable,collision,12.,exposed=True,actual_exposure_at=9.8)
    checks['cannot_recover_before_actual_contact']=not seen['contact_after_recovery']['recovered'] or seen['contact_after_recovery']['stable_confirmation']>=9.8
    # Static structure check: scanning only trace files loses metadata-only
    # runtime failures and hence all of their predeclared event rows.
    main_node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    trace_glob_only=any(isinstance(n,ast.Constant) and n.value=='runs/*/trace.npz' for n in ast.walk(main_node))
    checks['planned_denominators_not_trace_only']=not trace_glob_only
    protocol=json.loads((ROOT/'review/recovery_plan_v1.json').read_text(encoding='utf-8'))
    # Enumerate a planned missing run and a present-but-corrupt trace using
    # isolated fixtures. No real experiment directory is changed or evaluated.
    if 'evaluate_folder' in namespace:
        jobs=[dict(method=m,scenario='three_pulses',shape='capsule',seed=999999,split='synthetic') for m in ('ratio','unified')]
        namespace['evaluate_saved_trace']=lambda *a: (_ for _ in ()).throw(ValueError('synthetic corrupt trace'))
        with tempfile.TemporaryDirectory(prefix='independent_recovery_fixture_',dir=ROOT/'review') as temporary:
            d=Path(temporary)
            (d/'manifest.json').write_text(json.dumps(dict(planned_jobs=jobs)),encoding='utf-8')
            present=d/'runs/three_pulses__capsule__unified__999999';present.mkdir(parents=True)
            (present/'trace.npz').write_bytes(b'corrupt synthetic fixture')
            records=namespace['evaluate_folder'](d,protocol)
            checks['missing_and_corrupt_runs_preserved']=len(records)==2 and all(not r['record_complete'] and len(r['events'])==3 for r in records)
            checks['every_missing_run_event_invalid']=all(e['status']=='invalid_record' and e['latency_from_event'] is None for r in records for e in r['events'])
    else:
        checks['missing_and_corrupt_runs_preserved']=False
    # Event schedules are active on [start,end), so the end tick is already
    # after a transient and must not be labelled under-load recovery.
    endtie=good.copy();endtie[(t>=9.3-1e-9)&(t<9.4-1e-9)]=False
    seen['confirmation_exact_event_end']=call(endtie)
    checks['event_end_confirmation_is_after']=seen['confirmation_exact_event_end']['status']=='recovered_after'
    checks['discloses_development_seen']=protocol['disclosure']['development_seen'] and protocol['disclosure']['not_an_unseen_development_endpoint']
    checks['no_primary_source_change']=all(sha(ROOT/k)==v for k,v in protocol['source_sha256'].items())
    certificate=dict(passed=all(checks.values()),created_utc=datetime.now(timezone.utc).isoformat(),
        checks=checks,failed=[k for k,v in checks.items() if not v],examples=seen,
        analyzer_sha256=sha(source),plan_sha256=sha(ROOT/'review/recovery_plan_v1.json'),
        own_script_sha256=sha(__file__),scope='Synthetic classifier and static enumeration only; no dynamics, no confirmation result, no primary metric or admission changes.')
    out=ROOT/('review/independent_recovery_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.write_text(json.dumps(certificate,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=certificate['passed'],failed=certificate['failed'],output=str(out),sha256=sha(out)),ensure_ascii=False))


if __name__=='__main__':
    main()

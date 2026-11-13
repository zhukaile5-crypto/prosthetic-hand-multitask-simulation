"""Read-only all-column nine-run physical bridge audit. Never starts a run."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
EXCLUDED={'metadata','control_ms'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def resolved(p):
    p=Path(p);return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
def source_path(key):return FOUNDATION/key.removeprefix('foundation/') if key.startswith('foundation/') else ROOT/key
def costs(values,active):
    t=np.arange(3201)*.005;mask=active&(t>=7.5-1e-9)&(t<12.-1e-9)&(abs((t-7.5)/.02-np.rint((t-7.5)/.02))<1e-7)
    a=values[mask];valid=np.isfinite(a)&(a>=0)
    return dict(samples=len(a),finite_nonnegative=int(valid.sum()),median_ms=float(np.median(a[valid])) if valid.any() else None,
        p95_ms=float(np.percentile(a[valid],95)) if valid.any() else None,max_ms=float(a[valid].max()) if valid.any() else None,
        scope='20ms decision-grid control-branch wallclock; excludes outer sensor/geometry/task construction and physical integration')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    run=args.run.resolve();out=args.out.resolve();marker=run/'execution_complete.json'
    if not marker.is_file():print(json.dumps(dict(status='pending',new_physics_steps=0)));return 3
    if out.exists():raise FileExistsError(out)
    end=read(marker);manifest=read(run/'manifest.json');lock=read(run/'source_lock.json');checks=[];issues=[];rows=[]
    def check(name,ok,detail=None):
        r=dict(name=name,passed=bool(ok),detail=detail);checks.append(r)
        if not ok:issues.append(r)
    locksha=sha(run/'source_lock.json')
    check('nine_full_terminal_records',end['execution_complete'] is True and len(end['results'])==9 and all(r['complete'] for r in end['results']))
    check('source87_and_same_lock',end['sources_unchanged'] is True and len(lock['source_sha256'])==87 and
        locksha==end['source_lock_sha256']==manifest['source_lock_sha256']==sha(ROOT/'v13_a1/source_lock.json'))
    check('plan_pinned',sha(ROOT/'review/v13_a1_behavior_bridge_plan_v1.json')==lock['plan_sha256'])
    gate=Path(lock['software_gate']);check('software_gate_pinned',sha(gate)==lock['software_gate_sha256'] and read(gate)['passed'] is True)
    for key,value in lock['source_sha256'].items():check('source/'+key,sha(source_path(key))==sha(run/'sources'/key)==value)
    def key(j):return (j['scenario'],j['motion_mode'],int(j['seed']),j['method'])
    planned={key(j) for j in lock['jobs']};actual={key(j) for j in manifest['jobs']}
    check('unique_predeclared_nine_matrix',len(planned)==9 and planned==actual and len(manifest['jobs'])==9)
    for job in manifest['jobs']:
        name=str(key(job));folder=resolved(job['destination']);original=resolved(job['original_trace']);actual=folder/'trace.npz'
        result=read(folder/'result.json');meta=read(folder/'metadata.json');oldmeta=read(original.with_name('metadata.json'))
        terminal=next(r for r in end['results'] if key(r['job'])==key(job))
        check(name+'/terminal_result',result==terminal and result['sources_and_input_unchanged'] is True)
        check(name+'/frozen_input_and_new_trace',sha(original)==job['original_trace_sha256']==oldmeta['trace_sha256'] and
            sha(actual)==result['trace_sha256']==meta['trace_sha256'])
        check(name+'/same_physical_config_and_actual_source87',meta['config']==oldmeta['config'] and
            meta['model_sha256']==oldmeta['model_sha256'] and meta['research_source_sha256']==lock['source_sha256'])
        behavior_meta=('limits','budget','noise','gains','actual_ratio_task_gains','handoff_passed','handoff_alpha',
            'legacy_model_domain_supported','preparation_failure','reference_relative','release_at',
            'release_supported_at_command','mechanical_release_at','max_hidden_ctrl_clip','failure_reason','failure_at','completed')
        meta_differences=[key for key in behavior_meta if key not in meta or key not in oldmeta or meta[key]!=oldmeta[key]]
        check(name+'/material_handoff_lifecycle_metadata_unchanged',not meta_differences,meta_differences)
        comparison={};different=[];stored=read(folder/'complete_native_equivalence.json')
        control=None;oldcontrol=None;active=None
        with np.load(actual,allow_pickle=False) as a,np.load(original,allow_pickle=False) as b:
            check(name+'/identical_all_column_keys',set(a.files)==set(b.files))
            for column in sorted(set(a.files)|set(b.files)):
                if column=='metadata':continue
                if column not in a.files or column not in b.files:
                    check(name+'/column/'+column,False,'missing');different.append(column);continue
                x,y=a[column],b[column]  # Read each compressed column once.
                if column=='control_ms':control=x;oldcontrol=y;continue
                same=exact(x,y);check(name+'/column/'+column,same)
                comparison[column]=same
                if not same:different.append(column)
                if column=='controller_active':active=x.astype(bool)
                if column=='time':check(name+'/3201_exact_public_clock',exact(x,np.arange(3201,dtype=float)*.005))
                if column=='substep_time':check(name+'/32000_microrecords',len(x)==32000)
                if column=='substep_duration':check(name+'/half_ms_full16s',len(x)==32000 and np.all(x==.0005) and abs(float(x.sum())-16.)<1e-9)
                del x,y
        check(name+'/author_comparison_agrees',stored['passed'] is True and stored['same_column_keys'] is True and
            set(stored['excluded_columns'])==EXCLUDED and set(stored['fields'])==set(comparison) and
            all(stored['fields'][k]['exact']==v for k,v in comparison.items()))
        check(name+'/actual_actor_used',result['actor_write_count']>0 and result['physical_steps']==32000)
        rows.append(dict(job=job,all_nonexcluded_columns_exact=not different,compared_columns=len(comparison),
            differences=different,trace_sha256=result['trace_sha256'],actor_write_count=result['actor_write_count'],
            behavior_metadata_differences=meta_differences,
            old_control_ms=costs(oldcontrol,active),A1_control_ms=costs(control,active)))
        print(json.dumps(dict(records_checked=len(rows),planned=9,issues=len(issues))),flush=True)
    check('all9_flag_agrees_with_full_independent_fields',end['all9_exact'] is True and all(r['all_nonexcluded_columns_exact'] for r in rows))
    report=dict(schema='v13_a1_physical_bridge_independent_readonly_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        passed=not issues,checks=checks,issues=issues,rows=rows,source_lock_sha256=locksha,excluded_columns=sorted(EXCLUDED),
        input_sha256={str(p):sha(p) for p in (marker,run/'manifest.json',run/'source_lock.json')},
        checker_sha256=sha(Path(__file__)),new_physics_steps_by_reviewer=0,
        scope='A1 actual full-run behavior equivalence at the nine predeclared contexts; not a strategy superiority or global pressure-safety certificate',
        IO='Each source/new NPZ column read once perpair; at most one column pair plus smallcost arrays retained')
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(passed=not issues,checks=len(checks),issues=len(issues),output=str(out),sha256=sha(out))))
    return 0 if not issues else 2


if __name__=='__main__':raise SystemExit(main())

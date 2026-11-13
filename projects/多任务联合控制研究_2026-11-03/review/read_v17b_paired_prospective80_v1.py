"""Complete-cohort V17b five-arm descriptive reader, offline only.

Reuses frozen V15A pair/metric formulas and continuous_diagnostics05e86. No
production imports, evaluator changes, partial winner summaries or simulation.
The separate v3 delivery artifact is referenced, never recomputed here.
"""
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import argparse,gzip,hashlib,importlib.util,itertools,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PLAN_SHA='b08fe0d57b2c43a8abcf58e9ff869d1fdc01ae0f2ed5cfe1546028e7268a6a06'
OLD=ROOT/'review/read_v15A_paired_prospective_v1.py'
OLD_SHA='58e425e8713c5d83cb4477ac287143d135eeb3009651106f6177652e51b4d7fc'
ROLES=('full_V17','holding_A','same_prior_legacy_ratio','baseline_fixed','legacy_fusion')
CONTRASTS=(('full_V17','same_prior_legacy_ratio'),('full_V17','holding_A'),('full_V17','baseline_fixed'),('full_V17','legacy_fusion'),('holding_A','same_prior_legacy_ratio'),('holding_A','baseline_fixed'),('holding_A','legacy_fusion'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def load_old():
    if sha(OLD)!=OLD_SHA:raise RuntimeError('Frozen paired formulas changed')
    s=importlib.util.spec_from_file_location('frozen_pair_formulas58e',OLD);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def role(stage,method):
    if stage['arm'] in ('full_V17','holding_A'):
        if method!='ratio_effort':raise ValueError('Wrong integrated force preference')
        return stage['arm']
    if stage['arm']!='baselines':raise ValueError('Unregistered arm')
    return dict(ratio_effort='same_prior_legacy_ratio',fixed_effort='baseline_fixed',fusion_effort='legacy_fusion')[method]
def record_id(key):return json.dumps(key,ensure_ascii=False,separators=(',',':'))

class ArrayCache:
    """Real decompression happens at most once per requested column per run."""
    def __init__(self):self.loader=np.load;self.handles={};self.values={}
    def load(self,path,**kw):
        key=str(Path(path).resolve())
        if key not in self.handles:self.handles[key]=self.loader(path,**kw);self.values[key]={}
        parent=self
        class View:
            files=parent.handles[key].files
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def __getitem__(self,k):
                if k not in parent.values[key]:parent.values[key][k]=parent.handles[key][k]
                return parent.values[key][k]
        return View()
    def close(self):
        for h in self.handles.values():h.close()

def complete_timing(folder,b,old,cache):
    """V17 release decisions have their own log; do not dilute into5ms ticks."""
    path=folder/'trace.npz'
    if not path.is_file():return b
    with cache.load(path,allow_pickle=False) as z:a={k:z[k] for k in ('time',*old.TIMING) if k in z.files}
    t=a.get('time',np.empty(0));holding=[];release=[];unaligned=[];orig_release=[]
    for name in ('joint_decisions.jsonl.gz','release_decisions.jsonl.gz'):
        p=folder/name
        if not p.is_file():continue
        with gzip.open(p,'rt',encoding='utf-8') as f:ds=[json.loads(line) for line in f]
        for d in ds:
            time=float(d['time_s']);j=int(np.argmin(abs(t-time))) if len(t) else -1
            if j<0 or abs(t[j]-time)>1e-9:unaligned.append(dict(file=name,time_s=time));continue
            if name.startswith('release_'):release.append(j)
            elif d.get('mode')=='holding':holding.append(j)
            elif d.get('mode')=='release':orig_release.append(j)
    chosen=release if release else orig_release
    timing=b.setdefault('timing',{})
    timing.update(holding_decision=old.timing_stats(a,holding),release_decision=old.timing_stats(a,chosen),
        actual_holding_decision_count=len(holding),actual_release_decision_count=len(chosen),
        release_decision_log='release_decisions.jsonl.gz' if release else 'joint_decisions.jsonl.gz' if orig_release else None,
        phase_decision_unaligned=unaligned,duplicate_decision_sensor_ticks=dict(holding=len(holding)-len(set(holding)),release=len(chosen)-len(set(chosen))),
        release_missing_decision_telemetry_not_zero_cost=not bool(chosen))
    return b

def get_bundle(folder,old,aux,cache_dir):
    cache=ArrayCache()
    try:
        with patch.object(np,'load',cache.load):b=old.checked_bundle(folder,aux,cache_dir)
        b=complete_timing(folder,b,old,cache)
        b['additional_release_log_sha256']=sha(folder/'release_decisions.jsonl.gz') if (folder/'release_decisions.jsonl.gz').is_file() else None
        return b
    finally:cache.close()

def roster(plan):
    out=[]
    for s in plan['stages']:
        for seed,scene,method in itertools.product(s['seeds'],s['scenarios'],s['methods']):
            rr=role(s,method);key=[rr,scene,s['motion_mode'],int(seed)]
            out.append(dict(record_key=key,record_id=record_id(key),role=rr,method=method,context_key=key[1:],stage=s['name'],
                task_policy=s['task_policy'],receiver_policy=s['receiver_policy'],run=ROOT/s['output']/'runs'/f'{scene}__{method}__{seed}'))
    if len(out)!=80 or len({x['record_id'] for x in out})!=80 or Counter(x['role'] for x in out)!=Counter({k:16 for k in ROLES}):raise ValueError('Expected exactly80 unique five-arm assignments')
    return out

def summary(records,pairs,keys,old):
    def one(rr,pp):
        return dict(arms={role:{k:old.descriptive([r['metrics'].get(k) for r in rr if r['role']==role]) for k in keys} for role in ROLES},
            contrasts={a+'__minus__'+b:{k:dict(delta=old.descriptive([p['differences'][k]['delta'] for p in pp if p['contrast']==[a,b]]),relative_delta=old.descriptive([p['differences'][k]['relative_delta'] for p in pp if p['contrast']==[a,b]])) for k in keys} for a,b in CONTRASTS})
    seeds=sorted({r['context_key'][2] for r in records})
    return dict(all_assigned_descriptive=one(records,pairs),per_physical_seed={str(s):one([r for r in records if r['context_key'][2]==s],[p for p in pairs if p['context_key'][2]==s]) for s in seeds},
        per_motion={m:one([r for r in records if r['context_key'][1]==m],[p for p in pairs if p['context_key'][1]==m]) for m in ('hold','transport')},
        statistical_unit='Descriptive repeated context summaries, two physical seed clusters; no CI, p value, independent n16/n80, weighted score or winner filtering')

def build(plan_path,cache_dir,delivery_path=None):
    if sha(plan_path)!=PLAN_SHA:raise RuntimeError('Registered cohort identity changed')
    plan=read(plan_path);assign=roster(plan);stages=[]
    # This gate precedes every new NPZ/result read: never summarize partial80.
    for s in plan['stages']:
        f=ROOT/s['output']/'execution_complete.json';done=read(f) if f.is_file() else None
        stages.append(dict(stage=s['name'],planned=s['planned'],terminal=done,sha256=sha(f) if f.is_file() else None))
        if not done or done.get('execution_complete') is not True or done.get('sources_unchanged') is not True:raise RuntimeError('WAIT_ALL80_TERMINAL_NO_PARTIAL_READ:'+s['name'])
    old=load_old();aux=old.load_auxiliary();records=[];issues=[]
    for item in assign:
        folder=item['run']
        try:b=get_bundle(folder,old,aux,cache_dir)
        except Exception as e:b=dict(status='retained_reader_error',error=repr(e),run_directory=str(folder),diagnostics=None,timing=None)
        diag=b.get('diagnostics') or {};identity=diag.get('identity') or {};expected=dict(method=item['method'],seed=item['context_key'][2],motion_mode=item['context_key'][1],task_policy=item['task_policy'],receiver_policy=item['receiver_policy'])
        mismatch=[k for k,v in expected.items() if identity.get(k)!=v]
        if (identity.get('scenario') or {}).get('name')!=item['context_key'][0]:mismatch.append('scenario')
        ready=old.bundle_ready(b) and not mismatch
        if not ready:issues.append(dict(record_id=item['record_id'],status=b.get('status'),identity_mismatch=mismatch))
        records.append(dict(**{k:v for k,v in item.items() if k!='run'},run=str(folder),assigned=True,measurements_ready=ready,
            original_status=b.get('status'),identity_mismatch=mismatch,metrics=old.metric_values(b),bundle=b))
    keys=sorted(set().union(*(r['metrics'] for r in records)));pairs=[];by={r['record_id']:r for r in records}
    for context in sorted({tuple(r['context_key']) for r in records}):
        for a,b in CONTRASTS:
            A=by[record_id([a,*context])];B=by[record_id([b,*context])]
            ready=A['measurements_ready'] and B['measurements_ready'];same=A['bundle'].get('physical_context')==B['bundle'].get('physical_context') if ready else None
            if ready and not same:issues.append(dict(context=context,contrast=[a,b],reason='actual_config_noise_or_initial_condition_mismatch'))
            pairs.append(dict(context_key=list(context),contrast=[a,b],candidate_record_id=A['record_id'],reference_record_id=B['record_id'],
                actual_physical_context_equal=same,differences={k:old.difference(A['metrics'].get(k),B['metrics'].get(k)) for k in keys}))
    complete=len(records)==80 and len(pairs)==112 and not issues
    delivery=None
    if delivery_path is not None:
        artifact=read(delivery_path)
        required={r['record_id'] for r in records};aux_records=artifact.get('records',[])
        actual=[record_id(x['record_key']) for x in aux_records]
        if (artifact.get('complete') is not True or len(actual)!=80 or len(set(actual))!=80 or set(actual)!=required or
            artifact.get('plan_sha256')!=PLAN_SHA or artifact.get('execution_source_sha256')!=plan['source_sha256'] or
            artifact.get('contract_sha256')!=plan['auxiliary_metrics']['contract_sha256'] or
            artifact.get('calculator_sha256')!=plan['auxiliary_metrics']['reader_sha256'] or
            any(x.get('record_id')!=record_id(x['record_key']) or x.get('execution_source_identity_equal') is not True for x in aux_records)):
            raise RuntimeError('Separate delivery artifact is incomplete or wrong cohort/calculator/source identity')
        delivery=dict(path=str(delivery_path),sha256=sha(delivery_path),schema=artifact.get('schema'),status='separate_artifact_not_recomputed',
            exact80_record_key_join=True,contract_sha256=artifact['contract_sha256'],calculator_sha256=artifact['calculator_sha256'],
            instruction='Exact four-part record_key/record_id join validated; metrics remain in separate artifact and are never recomputed or substituted')
    return old.plain(dict(schema='v17b_five_arm_prospective80_paired_descriptive_v1',status='complete_descriptive_cohort' if complete else 'withheld_comparison_retained_assigned_issues',complete=complete,assigned=80,records=records,pairs=pairs,
        issues=issues,metric_registry=keys,summaries=summary(records,pairs,keys,old) if complete else None,delivery_auxiliary_reference=delivery,
        provenance=dict(plan=str(plan_path),plan_sha256=sha(plan_path),reader_sha256=sha(__file__),frozen_pair_formulas_sha256=OLD_SHA,continuous_diagnostics_sha256=old.AUX_SHA,stages=stages),
        comparison_order=['full_V17 vs same_prior_legacy_ratio','full_V17 vs holding_A (release mechanism)','full_V17 vs baseline_fixed / legacy_fusion','holding_A vs all3legacy (holding mechanism)'],
        units_and_eligibility='Original eligible NULL never replaced by unselected raw0; 200Hz holding endpoints include pre-release12s. Full16s and12–16s safety are original raw integrals. Timing control_ms excludes outer sensing; full_control_loop excludes physics.',
        inference=dict(total_trajectories=80,contexts=16,physical_seed_clusters=2,method_contrasts=112,CI=False,weighted_score=False,posthoc_selection=False),physics_steps=0))

def regression(cache_dir):
    """One old immutable run against its published non-timing V15A64 values."""
    old=load_old();aux=old.load_auxiliary();published=read(ROOT/'report/v15A_prospective_v2/paired_readout_v1.json')
    target=next(r for r in published['records'] if r['role']=='candidate_A' and r['context_key']==['fragile_conflict','hold',619732])
    b=get_bundle(Path(target['bundle']['run_directory']),old,aux,cache_dir);now=old.metric_values(b)
    keys=[k for k in target['metrics'] if not k.startswith('timing.')];diff=[k for k in keys if target['metrics'][k]!=now.get(k)]
    return dict(schema='v17b_paired_reader_old_archive_regression_v1',passed=not diff and old.bundle_ready(b),compared_non_timing_metric_count=len(keys),differences=diff,
        input_run=b['run_directory'],input_sha256=b['inputs'],reader_sha256=sha(__file__),frozen_pair_formulas_sha256=OLD_SHA,
        continuous_diagnostics_sha256=old.AUX_SHA,original_published_readout_sha256=sha(ROOT/'report/v15A_prospective_v2/paired_readout_v1.json'),
        scope='One previously seen archive only, not new80 data/performance; actual holding/release decision clocks are separately extracted',physics_steps=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',default='review/v17b_prospective80_plan_v1.json');p.add_argument('--cache',default='review/v17b_paired_metric_cache_v1');p.add_argument('--out',required=True);p.add_argument('--delivery');p.add_argument('--regression',action='store_true');args=p.parse_args()
    out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    report=regression(ROOT/args.cache) if args.regression else build(ROOT/args.plan,ROOT/args.cache,None if args.delivery is None else ROOT/args.delivery)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(out),passed=report.get('passed'),complete=report.get('complete'),sha256=sha(out))))
    return 0 if report.get('passed',report.get('complete')) else 1
if __name__=='__main__':raise SystemExit(main())

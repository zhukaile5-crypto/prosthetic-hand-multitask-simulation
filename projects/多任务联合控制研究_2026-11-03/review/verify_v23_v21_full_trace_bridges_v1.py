"""Read-only exact complete-run bridge; only the frozen walltime whitelist."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(out):
    if out.exists():raise FileExistsError(out)
    planpath=ROOT/'review/v23_small_plan_v1.json';plan=read(planpath)
    source=ROOT/'review/v18_passive_trace_comparison_v3.py'
    assert sha(source)==plan['dependencies']['review/v18_passive_trace_comparison_v3.py'],'Frozen bridge helper changed.'
    assert sha(__file__)==plan['dependencies']['review/verify_v23_v21_full_trace_bridges_v1.py'],'Registered bridge checker changed.'
    spec=importlib.util.spec_from_file_location('v23_frozen_complete_bridge',source);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    originals={tuple(r['context_key']):r for r in plan['original_v21_bridges']}
    stages=[s for s in plan['stages'] if s['variant']=='v21'];assert len(stages)==3
    expected={('nominal','transport',619741),('fragile_conflict','hold',619741),('friction_drop','transport',619742)}
    assert len({tuple(s['context_key']) for s in stages})==3 and {tuple(s['context_key']) for s in stages}==expected
    assert len(plan['source_sha256'])==186 and len(plan['parent179_source_sha256'])==179
    # Complete all3 before opening a trace; numerical change/failure is explicit.
    for s in stages:
        folder=ROOT/s['output'];m=read(folder/'execution_complete.json')
        if not (m['execution_complete'] and m['sources_unchanged'] and m['numerical_unchanged'] and not m['runtime_failures'] and m['trials']==m['planned']==1):raise ValueError('All3 actual v21 bridges terminal first.')
        assert sha(ROOT/s['registration'])==s['registration_sha256']
        reg=read(ROOT/s['registration'])
        assert reg['assignment']==s['assignment']
        assert reg['source_sha256']==plan['source_sha256'] and reg['numerical_identity']==plan['numerical_identity'] and reg['environment']==plan['environment']
        for name,field in [('plan.json','plan_sha256'),('manifest.json','manifest_sha256'),('assignment_ledger.json','assignment_ledger_sha256'),('trials.json','trials_sha256')]:
            assert sha(folder/name)==m[field]
        manifest=read(folder/'manifest.json')
        assert manifest['research_source_sha256']==plan['source_sha256'] and manifest['numerical_identity']==plan['numerical_identity']
        assert read(folder/'environment.json')['pinned_environment']==plan['environment']
    rows=[]
    for s in stages:
        key=tuple(s['context_key']);old=originals[key];original=Path(old['run'])
        for n,h in old['input_sha256'].items():assert sha(original/n)==h,'Old V21 evidence changed.'
        actual=ROOT/s['output']/'runs'/f'{key[0]}__ratio_effort__{key[2]}'
        meta=read(actual/'metadata.json');oldmeta=read(original/'metadata.json');scene=meta['scenario'];scene=scene['name'] if isinstance(scene,dict) else scene
        assert (scene,meta['motion_mode'],meta['seed'])==key
        assert (meta['method'],meta['task_policy'],meta['receiver_policy'],meta['release_variant'])==('ratio_effort','integrated','coordinated','v21')
        assert meta['research_source_sha256']==plan['source_sha256'] and meta['numerical_identity']==plan['numerical_identity']
        assert meta['frozen_v22_source_sha256']==plan['parent179_source_sha256'] and meta['frozen_v21_source_sha256']==plan['original170_source_sha256']
        assert all(meta[k]==oldmeta[k] for k in ('config','noise','limits','budget'))
        assert sha(actual/'trace.npz')==meta['trace_sha256']
        rows.append(dict(context_key=list(key),**helper.compare(actual,original)))
    result=dict(schema='v23_original_v21_complete_bridges_v1',passed=all(r['passed'] for r in rows),
        assigned=3,assessed=len(rows),rows=rows,plan_sha256=sha(planpath),helper_sha256=sha(source),
        reader_sha256=sha(__file__),physics_steps=0,no_numerical_tolerance_widened=True,
        scope='Entire16s all array fields except original provenance metadata and original explicit walltime whitelist.')
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'out':str(out),'passed':result['passed'],'checks_each':[len(r['checks']) for r in rows]},ensure_ascii=False))
    return 0 if result['passed'] else 1
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);args=a.parse_args();raise SystemExit(run(ROOT/args.out))

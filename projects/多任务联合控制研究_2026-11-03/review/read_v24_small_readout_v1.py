"""Terminal-only canonical-contract comparison on unchanged physical metrics."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'review/v24_small_plan_v1.json'
VARIANTS=('v23_joint','canonical_backend','canonical_joint')
HISTORICAL_VARIANTS=('v21','backend','joint')
CONTEXTS=(('nominal','transport',619741),('fragile_conflict','hold',619741),('friction_drop','transport',619742))

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def eq(a,b,label):
    if a!=b:raise ValueError(label+' differs')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def context(meta,key,variant):
    scene=meta.get('scenario');scene=scene.get('name') if isinstance(scene,dict) else scene
    eq((scene,meta.get('motion_mode'),meta.get('seed')),tuple(key),'actual context')
    eq((meta.get('method'),meta.get('task_policy'),meta.get('receiver_policy'),meta.get('release_variant')),
       ('ratio_effort','integrated','coordinated',variant),'actual policy and role')

def barrier(plan,markers):
    keys=[(s['variant'],*s['context_key']) for s in plan['stages']]
    eq((len(keys),len(set(keys)),set(keys)),(9,9,{(v,*c) for v in VARIANTS for c in CONTEXTS}),'exact3x3 roster')
    pending=[]
    for s in plan['stages']:
        a=s['assignment']
        eq((s['variant'],*s['context_key']),(a['release_variant'],a['scenarios'][0],a['motion_mode'],a['seeds'][0]),'stage assignment')
        eq((a['methods'],a['task_policy'],a['receiver_policy'],a['physical_randomization']),(['ratio_effort'],'integrated','coordinated',True),'common policy and physical draw')
        m=markers.get(s['name'])
        if not (m and m.get('execution_complete') is True and m.get('sources_unchanged') is True and m.get('numerical_unchanged') is True and m.get('runtime_failures')==[] and m.get('planned')==m.get('trials')==1):pending.append(s['name'])
    return dict(ready=not pending,pending=pending,assigned=9)

def difference(a,b):
    result={}
    for k in sorted(set(a)|set(b)):
        x,y=a.get(k),b.get(k);known=type(x) in (int,float,bool) and type(y) in (int,float,bool)
        result[k]=dict(candidate=x,reference=y,available=known,delta=float(x)-float(y) if known else None,
            relative_delta=(float(x)-float(y))/float(y) if known and type(x) is not bool and type(y) is not bool and y!=0 else None)
    return result

def collect(plan):
    markers={s['name']:read(ROOT/s['output']/'execution_complete.json') for s in plan['stages'] if (ROOT/s['output']/'execution_complete.json').exists()}
    if not barrier(plan,markers)['ready']:raise ValueError('All nine terminal before result/ledger/NPZ read')
    for name,digest in plan['dependencies'].items():eq(sha(ROOT/name),digest,'frozen definition or reference '+name)
    eq(sha(__file__),plan['dependencies']['review/read_v24_small_readout_v1.py'],'registered reader')
    calc=module('v24_fixed_auxv3',ROOT/'review/delivery_auxiliary_metrics_v3.py')
    projection=module('v24_fixed_projection',ROOT/'review/read_v18_small_auxiliary_v1.py')
    tv=module('v24_fixed_TV',ROOT/'review/read_v21_release_variation_v1.py')
    historical={};historical_meta={}
    for version in ('v22','v23'):
        report=read(ROOT/f'review/{version}_small_readout_v1.json')
        eq((report['complete'],report['issues'],report['assigned']),(True,[],9),version+' all9 report')
        pins={(r['variant'],*r['context_key']):r for r in plan['historical'][version]}
        keys=[(r['variant'],*r['context_key']) for r in report['records']]
        eq((len(keys),len(set(keys)),set(keys)),(9,9,{(v,*c) for v in HISTORICAL_VARIANTS for c in CONTEXTS}),version+' roster')
        for row in report['records']:
            key=(row['variant'],*row['context_key']);run=Path(row['run'])
            eq(row['input_sha256'],pins[key]['input_sha256'],version+' original inputs')
            for name,digest in row['input_sha256'].items():eq(sha(run/name),digest,version+' input '+name)
            meta=read(run/'metadata.json');context(meta,row['context_key'],row['variant'])
            eq(meta['research_source_sha256'],plan['reference_sources'][version],version+' source identity')
            historical[(version,*key)]=row;historical_meta[(version,*key)]=meta
    original21={tuple(r['context_key']):r for r in read(ROOT/'review/v21_complete16_auxiliary_v1.json')['records']}
    prior=read(ROOT/'review/v17b_delivery_auxiliary_readout_v1.json')
    refs={(r['role'],r['scenario'],r['motion_mode'],r['physical_seed']):r for r in prior['records']}
    records=[];metas={};issues=[];pairs=[];history=[];bridges=[]
    for stage in plan['stages']:
        folder=ROOT/stage['output'];marker=markers[stage['name']];key=tuple(stage['context_key']);variant=stage['variant']
        eq(sha(ROOT/stage['registration']),stage['registration_sha256'],'registration bytes')
        reg=read(ROOT/stage['registration'])
        for k in ('source_sha256','environment','numerical_identity'):eq(reg[k],plan[k],'registration '+k)
        eq(reg['assignment'],stage['assignment'],'assigned job')
        for name,field in [('trials.json','trials_sha256'),('assignment_ledger.json','assignment_ledger_sha256'),('plan.json','plan_sha256'),('manifest.json','manifest_sha256')]:eq(sha(folder/name),marker[field],'terminal '+name)
        ledger=read(folder/'assignment_ledger.json');trials=read(folder/'trials.json')
        eq((len(ledger),len(trials),ledger[0]['status']),(1,1,'evaluated'),'actual assigned evaluation')
        run=folder/'runs'/f'{key[0]}__ratio_effort__{key[2]}'
        eq(Path(ledger[0]['job']['destination']).resolve(),run.resolve(),'actual destination')
        meta=read(run/'metadata.json');result=read(run/'result.json');context(meta,key,variant);eq(result,trials[0],'actual result')
        eq(meta['research_source_sha256'],plan['source_sha256'],'actual source192')
        for v,label in [('v23','frozen_v23_source_sha256'),('v22','frozen_v22_source_sha256'),('v21','frozen_v21_source_sha256')]:eq(meta[label],plan['reference_sources'][v],'actual frozen '+v)
        eq(meta['numerical_identity'],plan['numerical_identity'],'actual numerical identity')
        eq(meta['release_role'],variant,'declared release role')
        eq(meta['task_contract'],'frozen_V23' if variant=='v23_joint' else 'canonical_V24','task contract')
        eq(read(folder/'environment.json')['pinned_environment'],plan['environment'],'original42 environment')
        manifest=read(folder/'manifest.json');eq(manifest['research_source_sha256'],plan['source_sha256'],'manifest source192');eq(manifest['numerical_identity'],plan['numerical_identity'],'manifest numerics')
        job=ledger[0]['job']
        eq((job['method'],job['scenario'],job['seed'],job['motion_mode'],job['task_policy'],job['receiver_policy'],job['release_variant']),('ratio_effort',key[0],key[2],key[1],'integrated','coordinated',variant),'actual selected job')
        eq(sha(run/'trace.npz'),meta['trace_sha256'],'actual trace')
        release=meta.get('mechanical_release_at',meta.get('release_at'));aux=metrics=None
        if release is None:issues.append(stage['name']+'/no release exposure')
        else:
            with np.load(run/'delivery_intervals.npz',allow_pickle=False) as z:arrays={k:z[k] for k in calc.C['input_fields']}
            aux=calc.calculate(arrays,release_at_s=release,episode_end_s=meta['duration'],physical_period_s=meta['physics_timestep'],original_result=result)
            metrics=projection.metrics(aux)
        with np.load(run/'trace.npz',allow_pickle=False) as z:
            times=z['time'];variation=tv.calculate(times,z['normal'],z['skin_normal']);wall=np.asarray(z['control_ms'])[times>=12.-1e-9]
            known=bool(len(wall) and np.isfinite(wall).all())
            timing=dict(available=known,samples=len(wall),median_ms=float(np.median(wall)) if known else None,
                p95_ms=float(np.percentile(wall,95)) if known else None,maximum_ms=float(np.max(wall)) if known else None,
                above_5ms_count=int(np.sum(wall>5.)) if known else None,
                scope='Saved release control branch walltimes; direct phases included; computation does not advance fixed simulation clock; not hardware realtime.')
        row=dict(stage=stage['name'],variant=variant,context_key=list(key),run=str(run),original_result=result,
            auxiliary=aux,metrics=metrics,release_TV=variation,observed_control_walltime=timing,
            input_sha256={n:sha(run/n) for n in ('metadata.json','result.json','delivery_intervals.npz','trace.npz')})
        records.append(row);metas[(variant,*key)]=meta
        if variant=='v23_joint':
            old=historical[('v23','joint',*key)]
            exact=aux is not None and projection.same_non_original(aux,old['auxiliary'])
            bridges.append(dict(context_key=list(key),all_non_original_aux_fields_exact=exact))
            if not exact:issues.append(stage['name']+'/frozenV23 auxiliary mismatch')
        for version in ('v22','v23'):
            for oldrole in HISTORICAL_VARIANTS:
                ref=historical[(version,oldrole,*key)];rm=ref['metrics'];rmmeta=historical_meta[(version,oldrole,*key)]
                paired=all(meta[k]==rmmeta[k] for k in ('config','noise','limits','budget'))
                history.append(dict(variant=variant,context_key=list(key),reference_role=version+'_'+oldrole,paired=paired,
                    candidate_metrics=metrics,reference_metrics=rm,differences=difference(metrics or {},rm),
                    release_TV_differences=difference(variation,ref['release_TV']),reference_input_sha256=ref['input_sha256']))
                if not paired:issues.append(stage['name']+'/'+version+'/'+oldrole+'/unpaired')
        r21=original21[key];m21=projection.metrics(r21['auxiliary'])
        paired=all(meta[k]==r21[r] for k,r in [('config','physical_config'),('noise','noise'),('limits','limits'),('budget','budget')])
        history.append(dict(variant=variant,context_key=list(key),reference_role='original_V21_complete16',paired=paired,candidate_metrics=metrics,reference_metrics=m21,differences=difference(metrics or {},m21)))
        if not paired:issues.append(stage['name']+'/originalV21/unpaired')
        for role in ('same_prior_legacy_ratio','full_V17','holding_A'):
            ref=refs[(role,*key)];rm=projection.metrics(ref['auxiliary'])
            paired=all(meta[k]==ref[r] for k,r in [('config','physical_config'),('noise','noise'),('limits','limits'),('budget','budget')])
            history.append(dict(variant=variant,context_key=list(key),reference_role=role,paired=paired,candidate_metrics=metrics,reference_metrics=rm,differences=difference(metrics or {},rm)))
            if not paired:issues.append(stage['name']+'/'+role+'/unpaired')
    index={(r['variant'],*r['context_key']):r for r in records}
    for key in CONTEXTS:
        for candidate,reference in [('canonical_backend','v23_joint'),('canonical_joint','v23_joint'),('canonical_joint','canonical_backend')]:
            ar=index[(candidate,*key)];br=index[(reference,*key)]
            paired=all(metas[(candidate,*key)][k]==metas[(reference,*key)][k] for k in ('config','noise','limits','budget'))
            pairs.append(dict(candidate_variant=candidate,reference_variant=reference,context_key=list(key),paired=paired,
                differences=difference(ar['metrics'] or {},br['metrics'] or {}),release_TV_differences=difference(ar['release_TV'],br['release_TV'])))
            if not paired:issues.append(candidate+'/'+reference+'/unpaired')
    eq((len(records),len(pairs),len(history),len(bridges)),(9,9,90,3),'complete comparisons')
    return dict(schema='v24_canonical_small_frozen_metric_readout_v1',complete=not issues,assigned=9,records=records,
        within_pairs=pairs,historical_pairs=history,bridges=bridges,issues=issues,
        plan_sha256=sha(PLAN),reader_sha256=sha(__file__),
        completion_marker_sha256={s['name']:sha(ROOT/s['output']/'execution_complete.json') for s in plan['stages']},
        created_utc=datetime.now(timezone.utc).isoformat(),definitions_changed=False,physics_steps=0,
        scope='Three seen contexts/two seed clusters; all failures and NULL retained; no weighted score, unseen superiority or realtime claim.')

def selftest():
    stages=[dict(name=f'{v}{i}',variant=v,context_key=list(c),assignment=dict(release_variant=v,scenarios=[c[0]],motion_mode=c[1],seeds=[c[2]],methods=['ratio_effort'],task_policy='integrated',receiver_policy='coordinated',physical_randomization=True)) for v in VARIANTS for i,c in enumerate(CONTEXTS)]
    plan=dict(stages=stages);good=dict(execution_complete=True,sources_unchanged=True,numerical_unchanged=True,runtime_failures=[],planned=1,trials=1)
    marks={s['name']:dict(good) for s in stages};checks=[barrier(plan,marks)['ready']]
    for field,value in [('execution_complete',False),('sources_unchanged',False),('numerical_unchanged',False),('runtime_failures',[{'error':'retained'}]),('trials',0)]:
        broken={k:dict(v) for k,v in marks.items()};broken[stages[-1]['name']][field]=value;checks.append(not barrier(plan,broken)['ready'])
    missing=dict(marks);missing.pop(stages[-1]['name']);checks.append(not barrier(plan,missing)['ready'])
    try:barrier(dict(stages=stages[:-1]+[stages[0]]),marks);checks.append(False)
    except ValueError:checks.append(True)
    diff=difference(dict(task_success=False,latency_s=None),dict(task_success=True,latency_s=.8))
    checks.append(diff['task_success']['candidate'] is False and diff['latency_s']['delta'] is None)
    meta=dict(scenario={'name':'nominal'},motion_mode='transport',seed=619741,method='ratio_effort',task_policy='integrated',receiver_policy='coordinated',release_variant='canonical_joint')
    context(meta,CONTEXTS[0],'canonical_joint')
    try:context(dict(meta,release_variant='joint'),CONTEXTS[0],'canonical_joint');checks.append(False)
    except ValueError:checks.append(True)
    return dict(passed=all(checks),checks=len(checks),physics_steps=0,new_results_read=False,reader_sha256=sha(__file__))

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--self-test',action='store_true');g.add_argument('--execute',action='store_true');ap.add_argument('--out',required=True)
    args=ap.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    result=selftest() if args.self_test else collect(read(PLAN))
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(out=str(out),complete=result.get('complete'),passed=result.get('passed')),ensure_ascii=False))

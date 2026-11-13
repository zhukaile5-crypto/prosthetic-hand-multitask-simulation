"""All-nine V22 terminal readout, using frozen auxiliaryv3 and raw release TV."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'review/v22_small_plan_v1.json'
VARIANTS=('v21','backend','joint')
CONTEXTS=(('nominal','transport',619741),('fragile_conflict','hold',619741),('friction_drop','transport',619742))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def eq(a,b,why):
    if a!=b:raise ValueError(why+' differs')
def barrier(plan,markers):
    keys=[(s['variant'],*s['context_key']) for s in plan['stages']]
    if len(keys)!=9 or len(set(keys))!=9 or set(keys)!={(v,*c) for v in VARIANTS for c in CONTEXTS}:raise ValueError('Exact assigned3x3 roster required.')
    pending=[]
    for s in plan['stages']:
        a=s['assignment'];eq((s['variant'],*s['context_key']),(a['release_variant'],a['scenarios'][0],a['motion_mode'],a['seeds'][0]),'stage context')
        m=markers.get(s['name'])
        if not m or not (m.get('execution_complete') is True and m.get('sources_unchanged') is True and m.get('numerical_unchanged') is True and m.get('runtime_failures')==[] and m.get('planned')==m.get('trials')==1):pending.append(s['name'])
    return dict(ready=not pending,pending=pending,assigned=9)
def difference(a,b):
    result={}
    for k in sorted(set(a)|set(b)):
        x,y=a.get(k),b.get(k);known=type(x) in (int,float,bool) and type(y) in (int,float,bool)
        result[k]=dict(candidate=x,reference=y,available=known,delta=float(x)-float(y) if known else None,
            relative_delta=(float(x)-float(y))/float(y) if known and type(x) is not bool and type(y) is not bool and y!=0 else None)
    return result
def collect(plan):
    marks={s['name']:read(ROOT/s['output']/'execution_complete.json') for s in plan['stages'] if (ROOT/s['output']/'execution_complete.json').exists()}
    if not barrier(plan,marks)['ready']:raise ValueError('All nine terminal before any result/ledger read.')
    for n,h in plan['dependencies'].items():eq(sha(ROOT/n),h,'frozen definition/reference '+n)
    calc=module('v22_unchanged_auxv3',ROOT/'review/delivery_auxiliary_metrics_v3.py')
    projection=module('v22_unchanged_projection',ROOT/'review/read_v18_small_auxiliary_v1.py')
    tv=module('v22_unchanged_TV',ROOT/'review/read_v21_release_variation_v1.py')
    old=read(ROOT/'review/v17b_delivery_auxiliary_readout_v1.json')
    refs={(r['role'],r['scenario'],r['motion_mode'],r['physical_seed']):r for r in old['records']}
    old21={tuple(r['context_key']):r for r in read(ROOT/'review/v21_complete16_auxiliary_v1.json')['records']}
    records=[];metas={};issues=[];history=[];bridges=[]
    for s in plan['stages']:
        folder=ROOT/s['output'];mark=marks[s['name']];reg=read(ROOT/s['registration']);a=s['assignment'];key=tuple(s['context_key'])
        eq(sha(ROOT/s['registration']),s['registration_sha256'],'original registration')
        for k in ('source_sha256','environment','numerical_identity'):eq(reg[k],plan[k],'registration '+k)
        eq(reg['assignment'],a,'assignment')
        for name,field in [('trials.json','trials_sha256'),('assignment_ledger.json','assignment_ledger_sha256'),('plan.json','plan_sha256'),('manifest.json','manifest_sha256')]:eq(sha(folder/name),mark[field],'terminal '+name)
        ledger=read(folder/'assignment_ledger.json');trials=read(folder/'trials.json')
        eq(len(ledger),1,'one assigned');eq(len(trials),1,'one result');eq(ledger[0]['status'],'evaluated','actual evaluated')
        run=folder/'runs'/f'{key[0]}__ratio_effort__{key[2]}'
        eq(Path(ledger[0]['job']['destination']).resolve(),run.resolve(),'executed destination')
        meta=read(run/'metadata.json');result=read(run/'result.json');eq(result,trials[0],'original result')
        scene=meta.get('scenario');scene=scene.get('name') if isinstance(scene,dict) else scene
        eq((scene,meta.get('motion_mode'),meta.get('seed')),key,'actual metadata context')
        eq((meta.get('method'),meta.get('task_policy'),meta.get('receiver_policy')),
            ('ratio_effort','integrated','coordinated'),'actual metadata policies')
        eq(meta['research_source_sha256'],plan['source_sha256'],'actual source179')
        eq(meta['numerical_identity'],plan['numerical_identity'],'actual numerical identity')
        eq(meta['release_variant'],s['variant'],'actual selected algorithm')
        eq(read(folder/'environment.json')['pinned_environment'],plan['environment'],'original42 environment')
        manifest=read(folder/'manifest.json')
        eq(manifest['research_source_sha256'],plan['source_sha256'],'manifest source179')
        eq(manifest['numerical_identity'],plan['numerical_identity'],'manifest numerical identity')
        eq(sha(run/'trace.npz'),meta['trace_sha256'],'raw trace')
        release=meta.get('mechanical_release_at',meta.get('release_at'))
        aux=metrics=None
        if release is None:issues.append(s['name']+'/no release exposure')
        else:
            with np.load(run/'delivery_intervals.npz',allow_pickle=False) as z:arrays={k:z[k] for k in calc.C['input_fields']}
            aux=calc.calculate(arrays,release_at_s=release,episode_end_s=meta['duration'],physical_period_s=meta['physics_timestep'],original_result=result)
            metrics=projection.metrics(aux)
        with np.load(run/'trace.npz',allow_pickle=False) as z:
            variation=tv.calculate(z['time'],z['normal'],z['skin_normal'])
            wall=np.asarray(z['control_ms'],float)[np.asarray(z['time'])>=12.-1e-9]
            timing=dict(available=bool(len(wall) and np.isfinite(wall).all()),samples=len(wall),
                median_ms=float(np.median(wall)) if len(wall) else None,
                p95_ms=float(np.percentile(wall,95)) if len(wall) else None,
                maximum_ms=float(np.max(wall)) if len(wall) else None,
                above_5ms_count=int(np.sum(wall>5.)),
                scope='Observed single-host batch walltime; computation does not advance the simulated sensor clock, not a hardware realtime guarantee.')
        row=dict(stage=s['name'],variant=s['variant'],context_key=list(key),run=str(run),original_result=result,
            auxiliary=aux,metrics=metrics,release_TV=variation,observed_control_walltime=timing,
            input_sha256={n:sha(run/n) for n in ['metadata.json','result.json','delivery_intervals.npz','trace.npz']})
        records.append(row);metas[(s['variant'],*key)]=meta
        if s['variant']=='v21':
            exact=aux is not None and projection.same_non_original(aux,old21[key]['auxiliary'])
            bridges.append(dict(context_key=list(key),all_non_original_aux_fields_exact=exact))
            if not exact:issues.append(s['name']+'/old V21 aux mismatch')
        for role in ('full_V17','holding_A','same_prior_legacy_ratio'):
            ref=refs[(role,*key)];paired=all(meta[k]==ref[r] for k,r in [('config','physical_config'),('noise','noise'),('limits','limits'),('budget','budget')])
            rm=projection.metrics(ref['auxiliary'])
            history.append(dict(variant=s['variant'],context_key=list(key),reference_role=role,paired=paired,
                candidate_metrics=metrics,reference_metrics=rm,differences=difference(metrics or {},rm)))
            if not paired:issues.append(s['name']+'/'+role+'/unpaired')
    index={(r['variant'],*r['context_key']):r for r in records};pairs=[]
    for key in CONTEXTS:
        for candidate,reference in [('backend','v21'),('joint','v21'),('joint','backend')]:
            ar=index[(candidate,*key)];br=index[(reference,*key)]
            paired=all(metas[(candidate,*key)][k]==metas[(reference,*key)][k] for k in ('config','noise','limits','budget'))
            pairs.append(dict(candidate_variant=candidate,reference_variant=reference,context_key=list(key),paired=paired,
                differences=difference(ar['metrics'] or {},br['metrics'] or {}),release_TV_differences=difference(ar['release_TV'],br['release_TV'])))
            if not paired:issues.append(candidate+'/'+reference+'/unpaired')
    eq(len(records),9,'all records');eq(len(pairs),9,'all within pairs');eq(len(history),27,'all historical pairs')
    return dict(schema='v22_small_frozen_metric_readout_v1',complete=not issues,assigned=9,records=records,
        within_pairs=pairs,historical_pairs=history,bridges=bridges,issues=issues,plan_sha256=sha(PLAN),reader_sha256=sha(__file__),
        created_utc=datetime.now(timezone.utc).isoformat(),definitions_changed=False,physics_steps=0,
        scope='Three seen contexts and two physical seed clusters; all failures/NULL retained; no weighted total or unseen superiority claim.')
def selftest():
    stages=[dict(name=f'{v}{i}',variant=v,context_key=list(c),assignment=dict(release_variant=v,scenarios=[c[0]],motion_mode=c[1],seeds=[c[2]])) for v in VARIANTS for i,c in enumerate(CONTEXTS)]
    p=dict(stages=stages);good=dict(execution_complete=True,sources_unchanged=True,numerical_unchanged=True,runtime_failures=[],planned=1,trials=1)
    ms={s['name']:dict(good) for s in stages};tests=[barrier(p,ms)['ready']]
    for field,value in [('execution_complete',False),('numerical_unchanged',False),('runtime_failures',[{'error':'retained'}])]:
        changed={k:dict(v) for k,v in ms.items()};changed[stages[-1]['name']][field]=value;tests.append(not barrier(p,changed)['ready'])
    missing=dict(ms);missing.pop(stages[-1]['name']);tests.append(not barrier(p,missing)['ready'])
    return dict(passed=all(tests),checks=len(tests),physics_steps=0,new_results_read=False,reader_sha256=sha(__file__))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');ap.add_argument('--execute',action='store_true');ap.add_argument('--out',required=True)
    a=ap.parse_args();out=ROOT/a.out
    if out.exists():raise FileExistsError(out)
    result=selftest() if a.self_test else collect(read(PLAN)) if a.execute else None
    if result is None:raise ValueError('explicit self-test or execute required')
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'out':str(out),'complete':result.get('complete'),'passed':result.get('passed')},ensure_ascii=False))

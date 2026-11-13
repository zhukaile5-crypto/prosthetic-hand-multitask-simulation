"""Read-only A3 bridge to historical V23 joint; original walltime whitelist."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'review/v24_small_plan_v1.json'
SELF='review/verify_v24_v23_full_trace_bridges_v1.py'
HELPER='review/v18_passive_trace_comparison_v3.py'
CONTEXTS=(('nominal','transport',619741),('fragile_conflict','hold',619741),('friction_drop','transport',619742))
ROLES=('v23_joint','canonical_backend','canonical_joint')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b,name):
    if a!=b:raise ValueError(name+' differs')
def module(p):
    s=importlib.util.spec_from_file_location('v24_unchanged_full_bridge_helper',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def roster(plan):
    keys=[(s['variant'],*s['context_key']) for s in plan['stages']]
    eq((len(keys),len(set(keys)),set(keys)),(9,9,{(r,*c) for r in ROLES for c in CONTEXTS}),'registered9 role/context roster')
    selected=[s for s in plan['stages'] if s['variant']=='v23_joint']
    refs=plan['original_v23_joint_bridges'];rk=[tuple(r['context_key']) for r in refs]
    eq((len(rk),len(set(rk)),set(rk)),(3,3,set(CONTEXTS)),'three original V23 joint references')
    for s in selected:
        a=s['assignment'];key=tuple(s['context_key'])
        eq((a['methods'],a['scenarios'],a['seeds'],a['motion_mode'],a['task_policy'],a['receiver_policy'],a['release_variant'],a['physical_randomization']),
            (['ratio_effort'],[key[0]],[key[2]],key[1],'integrated','coordinated','v23_joint',True),'actual A assignment')
    return selected,{tuple(r['context_key']):r for r in refs}
def barrier(stages,markers):
    return all((m:=markers.get(s['name'])) and m.get('execution_complete') is True and m.get('sources_unchanged') is True
        and m.get('numerical_unchanged') is True and m.get('runtime_failures')==[] and m.get('planned')==m.get('trials')==1 for s in stages)
def context(meta,key,variant):
    scene=meta.get('scenario');scene=scene.get('name') if isinstance(scene,dict) else scene
    eq((scene,meta.get('motion_mode'),meta.get('seed')),key,'actual metadata context')
    eq((meta.get('method'),meta.get('task_policy'),meta.get('receiver_policy'),meta.get('release_variant')),
        ('ratio_effort','integrated','coordinated',variant),'actual policy/role')
def run(out):
    if out.exists():raise FileExistsError(out)
    plan=read(PLAN)
    for name in (HELPER,SELF):eq(sha(ROOT/name),plan['dependencies'][name],'registered helper/checker '+name)
    eq(len(plan['source_sha256']),192,'actual source192')
    eq([len(plan['reference_sources'][v]) for v in ('v21','v22','v23')],[170,179,186],'frozen source subsets')
    stages,refs=roster(plan)
    markers={s['name']:read(ROOT/s['output']/'execution_complete.json') for s in stages
        if (ROOT/s['output']/'execution_complete.json').is_file()}
    if not barrier(stages,markers):raise ValueError('All3 A bridges terminal before any trace read')
    inputs=[]
    # Validate ALL3 source/input/role/env gates before opening the first NPZ.
    for s in stages:
        key=tuple(s['context_key']);folder=ROOT/s['output'];m=markers[s['name']]
        eq(sha(ROOT/s['registration']),s['registration_sha256'],'registration bytes')
        reg=read(ROOT/s['registration']);eq(reg['assignment'],s['assignment'],'registered assignment')
        for field in ('source_sha256','numerical_identity','environment'):eq(reg[field],plan[field],'registration '+field)
        for name,field in [('plan.json','plan_sha256'),('manifest.json','manifest_sha256'),('assignment_ledger.json','assignment_ledger_sha256'),('trials.json','trials_sha256')]:eq(sha(folder/name),m[field],'terminal '+name)
        manifest=read(folder/'manifest.json');eq(manifest['research_source_sha256'],plan['source_sha256'],'manifest192')
        eq(manifest['numerical_identity'],plan['numerical_identity'],'manifest numerical identity')
        env=read(folder/'environment.json');eq(env['pinned_environment'],plan['environment'],'original42 environment')
        eq(env['numerical_identity'],plan['numerical_identity'],'original numerical dependency identity')
        old=refs[key];original=Path(old['run'])
        for name,digest in old['input_sha256'].items():eq(sha(original/name),digest,'historical V23 joint '+name)
        actual=folder/'runs'/f'{key[0]}__ratio_effort__{key[2]}'
        meta=read(actual/'metadata.json');oldmeta=read(original/'metadata.json')
        context(meta,key,'v23_joint');context(oldmeta,key,'joint')
        eq(meta['research_source_sha256'],plan['source_sha256'],'actual source192')
        eq(oldmeta['research_source_sha256'],plan['reference_sources']['v23'],'historical source186')
        for version,label in [('v21','frozen_v21_source_sha256'),('v22','frozen_v22_source_sha256'),('v23','frozen_v23_source_sha256')]:eq(meta[label],plan['reference_sources'][version],'actual parent '+version)
        eq(meta['numerical_identity'],plan['numerical_identity'],'actual numerics')
        eq(oldmeta['numerical_identity'],plan['numerical_identity'],'historical numerics')
        eq((meta['release_role'],meta['task_contract']),('v23_joint','frozen_V23'),'frozen task contract role')
        for field in ('config','noise','limits','budget'):eq(meta[field],oldmeta[field],'matched '+field)
        eq(sha(actual/'trace.npz'),meta['trace_sha256'],'actual trace identity')
        ledger=read(folder/'assignment_ledger.json');eq((len(ledger),ledger[0]['status']),(1,'evaluated'),'actual single evaluation')
        job=ledger[0]['job'];eq(Path(job['destination']).resolve(),actual.resolve(),'actual assigned destination')
        eq((job['method'],job['scenario'],job['seed'],job['motion_mode'],job['task_policy'],job['receiver_policy'],job['release_variant']),
            ('ratio_effort',key[0],key[2],key[1],'integrated','coordinated','v23_joint'),'actual selected job')
        inputs.append((key,actual,original))
    helper=module(ROOT/HELPER)
    rows=[dict(context_key=list(key),reference_role='historical_V23_joint',**helper.compare(actual,original)) for key,actual,original in inputs]
    result=dict(schema='v24_original_v23_joint_complete_bridges_v1',passed=all(r['passed'] for r in rows),assigned=3,assessed=3,rows=rows,
        plan_sha256=sha(PLAN),helper_sha256=sha(ROOT/HELPER),reader_sha256=sha(__file__),production_source_sha256=plan['source_sha256'],numerical_identity=plan['numerical_identity'],
        completion_marker_sha256={s['name']:sha(ROOT/s['output']/'execution_complete.json') for s in stages},
        timing_columns=list(helper.TIMING_COLUMNS),debug_timing_paths=['solver_timing_ms.'+s for s in helper.DEBUG_TIMING_KEYS],
        excluded_provenance_column='metadata',physics_steps=0,no_numerical_tolerance_widened=True,
        scope='Entire16s recomputed trace arrays; only frozen explicit walltime and provenance exclusions. This reader does not simulate or rescore policies.')
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(out=str(out),passed=result['passed'],checks_each=[len(r['checks']) for r in rows],sha256=sha(out)),ensure_ascii=False))
    return 0 if result['passed'] else 1
def selftest(out):
    if out.exists():raise FileExistsError(out)
    stages=[dict(name=f'{r}_{i}',variant=r,context_key=list(c),assignment=dict(methods=['ratio_effort'],scenarios=[c[0]],seeds=[c[2]],motion_mode=c[1],task_policy='integrated',receiver_policy='coordinated',release_variant=r,physical_randomization=True)) for r in ROLES for i,c in enumerate(CONTEXTS)]
    plan=dict(stages=stages,original_v23_joint_bridges=[dict(context_key=list(c)) for c in CONTEXTS]);selected,_=roster(plan)
    good=dict(execution_complete=True,sources_unchanged=True,numerical_unchanged=True,runtime_failures=[],planned=1,trials=1)
    markers={s['name']:dict(good) for s in selected};tests=[barrier(selected,markers)]
    for field,value in [('execution_complete',False),('sources_unchanged',False),('numerical_unchanged',False),('runtime_failures',[dict(error='retained')])]:
        bad={k:dict(v) for k,v in markers.items()};bad[selected[-1]['name']][field]=value;tests.append(not barrier(selected,bad))
    try:roster(dict(plan,stages=stages[:-1]+[stages[0]]));tests.append(False)
    except ValueError:tests.append(True)
    helper=module(ROOT/HELPER);a=dict(time=np.array([0.,.005]),normal=np.ones((2,5)),control_ms=np.array([1.,2.]),metadata=np.array('old'))
    b={k:v.copy() for k,v in a.items()};b['control_ms']+=5.;b['metadata']=np.array('new');tests.append(helper.compare_arrays(a,b)['passed'])
    b['normal'][0,0]+=1e-12;tests.append(not helper.compare_arrays(a,b)['passed'])
    meta=dict(scenario={'name':'nominal'},motion_mode='transport',seed=619741,method='ratio_effort',task_policy='integrated',receiver_policy='coordinated',release_variant='v23_joint')
    context(meta,CONTEXTS[0],'v23_joint')
    try:context(dict(meta,release_variant='joint'),CONTEXTS[0],'v23_joint');tests.append(False)
    except ValueError:tests.append(True)
    result=dict(schema='v24_bridge_reader_software_v1',passed=all(tests),checks=len(tests),reader_sha256=sha(__file__),helper_sha256=sha(ROOT/HELPER),physics_steps=0,new_results_read=False,plan_read=False)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=result['passed'],checks=len(tests),out=str(out),sha256=sha(out))))
    return 0 if result['passed'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--self-test',action='store_true');m.add_argument('--execute',action='store_true');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();raise SystemExit(selftest(args.out) if args.self_test else run(args.out))

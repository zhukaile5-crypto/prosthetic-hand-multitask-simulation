"""Terminal-gated complete16 V21 raw release-TV diagnostic. No physics.
Calls the frozen original9 calculate function unchanged. Preparation and pure
software gates open no new13 results. Core170, original metrics and v3 remain
untouched. Per-channel NULLs and every assigned failed trial remain visible.
"""
from pathlib import Path
from collections import Counter
from copy import deepcopy
from unittest.mock import patch
import argparse,hashlib,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/'review/v21_complete16_plan_v1.json'
SCOPE=ROOT/'review/v21_complete16_release_variation_contract_v1.json'
SCOPE_SHA='b9ad7a31bf86595c269aaadc3d2b710dc197a147ff25b7dc025f880945fe0507'
CALCULATOR=ROOT/'review/read_v21_release_variation_v1.py'
CALCULATOR_SHA='40b67f7ba77862656e22ed6e4dd9f85a924732281f52ed3b85867d0ec0ebb7d2'
NINE_CONTRACT=ROOT/'review/v21_release_variation_contract_v1.json'
NINE_CONTRACT_SHA='7d2b915c7b4e97150b4ef32604d6c687a8433179523698610866ee1815ddc231'
V20_AUX=ROOT/'review/v20_complete16_auxiliary_v1.json'
V20_AUX_SHA='e45d2881818c9bbcd3ae7ca83f2d167446d75429150a3ab8140a2130aa786d10'
PI_PLAN=ROOT/'review/v17b_prospective80_plan_v1.json'
PI_PLAN_SHA='b08fe0d57b2c43a8abcf58e9ff869d1fdc01ae0f2ed5cfe1546028e7268a6a06'
FIELDS=('target_normal_TV_N','all_actor_skin_TV_N')
SCENES=('nominal','friction_drop','fragile_conflict','collision');MOTIONS=('hold','transport');SEEDS=(619741,619742)
CONTEXTS={(s,m,se) for s in SCENES for m in MOTIONS for se in SEEDS}
ORIGINS={'retained_initial3':3,'new_remaining13':13}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(test,why):
    if not test:raise ValueError(why)
def safe(p):
    p=(ROOT/Path(p)).resolve();require(p==ROOT or ROOT in p.parents,'run/reference path outside research root');return p
def load_calculator():
    require(sha(CALCULATOR)==CALCULATOR_SHA,'frozen9 calculate source changed');spec=importlib.util.spec_from_file_location('frozen_v21_release_TV_calculate',CALCULATOR);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.calculate

def frozen_pins():
    for p,s in ((SCOPE,SCOPE_SHA),(CALCULATOR,CALCULATOR_SHA),(NINE_CONTRACT,NINE_CONTRACT_SHA),(V20_AUX,V20_AUX_SHA),(PI_PLAN,PI_PLAN_SHA)):require(sha(p)==s,'frozen input pin changed: '+str(p))

def context(meta):
    scene=meta.get('scenario');scene=scene.get('name') if isinstance(scene,dict) else scene
    return scene,meta.get('motion_mode'),meta.get('seed')
def metadata_identity(meta,key,*,source,variant=None):
    require(context(meta)==tuple(key),'actual metadata context differs')
    require(meta.get('method')=='ratio_effort','actual method differs')
    require(meta.get('research_source_sha256')==source,'actual own-source dictionary differs')
    if variant is not None:
        require(meta.get('release_variant')==variant and meta.get('task_policy')=='integrated' and meta.get('receiver_policy')=='coordinated','actual fullV21 variant/policy differs')

def validate_master(plan):
    require((plan.get('planned'),plan.get('new_records'),plan.get('retained_count'),plan.get('combined_count'))==(13,13,3,16),'13new+3retained master counts required')
    require(len(plan.get('source_sha256',{}))==170 and len(plan.get('environment',{}).get('native_binaries_sha256',{}))==42,'full170/42 identity declaration required')
    require(plan.get('release_variant')=='v21','fullV21 variant required')
    pin=dict(path='review/v21_complete16_release_variation_contract_v1.json',sha256=SCOPE_SHA)
    require(plan.get('release_variation_scope_contract')==pin and plan.get('evidence_pins',{}).get('release_variation_scope_contract')==pin,'master complete16 scope pin absent/different')
    require(plan.get('release_variation_contract_sha256')==NINE_CONTRACT_SHA and plan.get('release_variation_reader_sha256')==CALCULATOR_SHA,'original9 definition/source pins differ')
    rows=plan.get('roster',[]);require(len(rows)==16,'all16 roster required');keys=[tuple(x['record_key']) for x in rows]
    require(len(set(keys))==16 and set(keys)=={('full_V21',*k) for k in CONTEXTS},'duplicate/missing/wrong fullV21 context')
    require(Counter(x['origin'] for x in rows)==Counter(ORIGINS),'retained/new origins differ')
    require(all(tuple(x['context_key'])==tuple(x['record_key'][1:]) for x in rows),'record/context keys differ')
    retained=plan.get('retained',[]);require(len(retained)==3 and {tuple(r['record_key']) for r in retained}=={tuple(r['record_key']) for r in rows if r['origin']=='retained_initial3'},'retained3 provenance roster differs')
    stages=plan.get('stages',[]);require(len(stages)==4 and len({s['name'] for s in stages})==4 and sum(s['planned'] for s in stages)==13,'all4 new13 completion stages required')
    newkeys=set()
    for s in stages:
        a=s['assignment'];require(a['methods']==['ratio_effort'] and len(a['seeds'])==1 and a['task_policy']=='integrated' and a['receiver_policy']=='coordinated' and a['release_variant']=='v21' and a['physical_randomization'] is True,'actual registered fullV21 assignment required')
        expected={(scene,a['motion_mode'],a['seeds'][0]) for scene in a['scenarios']};require(len(expected)==s['planned'],'stage assigned count differs')
        own={tuple(r['context_key']) for r in rows if r['origin']=='new_remaining13' and r.get('stage')==s['name']};require(own==expected,'new stage-to-roster mapping differs');newkeys|=own
        for r in rows:
            if r['origin']=='new_remaining13' and r.get('stage')==s['name']:
                scene,motion,seed=r['context_key'];require(r['run']==s['output']+'/runs/'+scene+'__ratio_effort__'+str(seed),'new actual run path differs')
    require(len(newkeys)==13,'new13 coverage mismatch');return rows

def terminal_barrier(plan,marker_read=read,marker_sha=sha,exists=lambda p:p.is_file()):
    validate_master(plan);states=[]
    entries=[dict(name=s['name'],path=ROOT/s['output']/'execution_complete.json',planned=s['planned'],pin=None,origin='new_remaining13') for s in plan['stages']]
    entries += [dict(name=r['stage'],path=ROOT/r['completion_path'],planned=1,pin=r['completion_sha256'],origin='retained_initial3') for r in plan['retained']]
    for e in entries:
        p=e['path'];m=marker_read(p) if exists(p) else None
        valid=bool(m is not None and m.get('execution_complete') is True and m.get('sources_unchanged') is True and m.get('runtime_failures')==[] and m.get('planned')==m.get('trials')==e['planned'])
        samepin=bool(e['pin'] is None or m is not None and marker_sha(p)==e['pin'])
        states.append(dict(name=e['name'],origin=e['origin'],planned=e['planned'],complete=valid and samepin,completion_sha256=marker_sha(p) if m is not None else None,marker=m))
    return dict(ready_to_read_outcomes=all(x['complete'] for x in states),planned=16,new13=13,retained3=3,states=states,pending=[s['name'] for s in states if not s['complete']],outcomes_read=False,pending_is_numeric_failure=False)

def parameter_equal(a,b):return all(a[k]==b[k] for k in ('config','noise','limits','budget'))
def differences(candidate,reference,paired):
    result={}
    for f in FIELDS:
        a,b=candidate.get(f),reference.get(f);known=bool(paired and type(a) in (float,int) and type(b) in (float,int) and np.isfinite(a) and np.isfinite(b))
        delta=float(a)-float(b) if known else None
        result[f]=dict(candidate=a,reference=b,paired_available=known,absolute_delta_N=delta,relative_delta=delta/float(b) if known and b>0 else None,comparison='unknown_or_unpaired' if not known else 'lower' if a<b else 'higher' if a>b else 'equal',zero_baseline_relative_NULL=bool(known and b==0))
    return result

def summaries(pairs):
    out=[]
    for role in ('full_V20','same_prior_legacy_ratio'):
        for seed in (None,*SEEDS):
            ps=[p for p in pairs if p['reference_role']==role and (seed is None or p['context_key'][2]==seed)]
            for f in FIELDS:
                ds=[p['differences'][f] for p in ps];absolute=[d['absolute_delta_N'] for d in ds if d['paired_available']];relative=[d['relative_delta'] for d in ds if d['relative_delta'] is not None];counts=Counter(d['comparison'] for d in ds)
                out.append(dict(reference_role=role,physical_seed=seed,metric=f,assigned_pairs=len(ps),known_pairs=len(absolute),unknown_or_unpaired=counts['unknown_or_unpaired'],lower=counts['lower'],equal=counts['equal'],higher=counts['higher'],absolute_delta_median_N=None if not absolute else float(np.median(absolute)),relative_change_median=None if not relative else float(np.median(relative)),relative_known_pairs=len(relative),zero_baseline_relative_NULL=sum(d['zero_baseline_relative_NULL'] for d in ds)))
    return out

def collect(plan):
    # Must remain first: no result/metadata/NPZ, including retained-run values,
    # before all13 and all3 retained completion/provenance claims are terminal.
    barrier=terminal_barrier(plan)
    require(barrier['ready_to_read_outcomes'],'all13 stages and retained3 terminal before any complete16 outcome read')
    frozen_pins();calculate=load_calculator();old20=read(V20_AUX);oldpi=read(PI_PLAN)
    require(old20.get('complete') is True and old20.get('issues')==[] and len(old20['records'])==16,'complete originalV20 all16 reference required')
    v20index={tuple(r['context_key']):r for r in old20['records']};require(set(v20index)==CONTEXTS and len(v20index)==16,'unique fullV20 reference contexts required')
    inputs={str(p):sha(p) for p in (MASTER,SCOPE,CALCULATOR,NINE_CONTRACT,V20_AUX,PI_PLAN)};cache={};stage_cache={};issues=[]
    def stage_data(folder,source,environment,registration=None,assignment=None,marker=None):
        folder=safe(folder);name=str(folder)
        if name in stage_cache:return stage_cache[name]
        manifest=read(folder/'manifest.json');require(manifest['research_source_sha256']==source,'own manifest source identity differs');require(read(folder/'environment.json')['pinned_environment']==environment,'own environment identity differs')
        if registration is not None:
            reg=read(safe(registration));require(reg['assignment']==assignment and reg['source_sha256']==source and reg['environment']==environment,'own registration assignment/source/environment differs')
            require(reg.get('release_variation_scope_contract')==plan['release_variation_scope_contract'],'stage scope contract differs')
        trials=read(folder/'trials.json')
        if marker is not None:
            for filename,key in (('trials.json','trials_sha256'),('assignment_ledger.json','assignment_ledger_sha256'),('plan.json','plan_sha256'),('manifest.json','manifest_sha256')):require(sha(folder/filename)==marker[key],'terminal pinned '+filename+' changed')
            require(len(trials)==marker['trials'],'terminal result count differs')
        stage_cache[name]=trials;return trials
    def row(run,key,role,source,expected_trace=None,old_digest=None,variant=None):
        run=safe(run);cachekey=str(run)
        if cachekey in cache:return cache[cachekey]
        meta=read(run/'metadata.json');result=read(run/'result.json');metadata_identity(meta,key,source=source,variant=variant)
        if old_digest:
            for f in ('metadata.json','result.json','trace.npz'):
                if f in old_digest:require(sha(run/f)==old_digest[f],'retained/comparator pinned input changed '+f)
        digest=sha(run/'trace.npz');require(digest==meta['trace_sha256']==result['trace_sha256'],'actual trace identity differs')
        if expected_trace is not None:require(digest==expected_trace,'master trace pin differs')
        require(meta.get('completed') is True and meta.get('duration')==16.,'full16 completed record required')
        with np.load(run/'trace.npz',allow_pickle=False) as z:value=calculate(z['time'],z['normal'],z['skin_normal'])
        value=dict(value);nonfinite=[]
        for field in FIELDS:
            if value.get(field) is not None and not np.isfinite(value[field]):value[field]=None;nonfinite.append(field)
        if nonfinite:value['status']='unknown_nonfinite_calculated_TV';value['nonfinite_fields']=nonfinite
        if value['status']!='complete_fixed_200Hz_release_window':issues.append(role+'/'+str(tuple(key))+'/'+value['status'])
        params={k:meta[k] for k in ('config','noise','limits','budget')}
        record=dict(record_key=[role,*key],context_key=list(key),run=str(run),metrics=value,trace_sha256=digest,metadata_sha256=sha(run/'metadata.json'),result_sha256=sha(run/'result.json'),original_result=result,parameters=params,own_source_sha256=source,metadata_identity=dict(scenario=context(meta)[0],motion_mode=meta['motion_mode'],seed=meta['seed'],method=meta['method'],release_variant=meta.get('release_variant'),task_policy=meta['task_policy'],receiver_policy=meta['receiver_policy']))
        for f in ('metadata.json','result.json','trace.npz'):inputs[str(run/f)]=sha(run/f)
        cache[cachekey]=record;return record
    candidates=[];refs=[];pairs=[]
    for entry in validate_master(plan):
        key=tuple(entry['context_key']);run=safe(entry['run']);origin=entry['origin']
        if origin=='retained_initial3':
            retained=next(r for r in plan['retained'] if tuple(r['record_key'])==tuple(entry['record_key']));digest={name:retained['files_sha256'][(run/name).relative_to(ROOT).as_posix()] for name in ('metadata.json','result.json','trace.npz')}
            candidate=row(run,key,'full_V21',plan['source_sha256'],old_digest=digest,variant='v21')
            folder=ROOT/retained['original_stage_output'];trials=stage_data(folder,plan['source_sha256'],plan['environment'])
        else:
            stage=next(s for s in plan['stages'] if s['name']==entry['stage']);mark=next(s['marker'] for s in barrier['states'] if s['name']==stage['name']);candidate=row(run,key,'full_V21',plan['source_sha256'],variant='v21')
            trials=stage_data(ROOT/stage['output'],plan['source_sha256'],plan['environment'],stage['registration'],stage['assignment'],mark)
        require(sum(v==candidate['original_result'] for v in trials)==1,'actual assigned result not exactly present once in own trials')
        candidate=dict(candidate,origin=origin);candidates.append(candidate)
        old=v20index[key];require(tuple(old['record_key'])==('full_V20',*key),'reference role identity differs');reference=row(old['run'],key,'full_V20',old20['source_sha256'],old_digest=old['input_sha256'])
        require(reference['parameters']==dict(config=old['physical_config'],noise=old['noise'],limits=old['limits'],budget=old['budget']),'originalV20 JSON/meta parameters differ')
        piv=next(s for s in oldpi['stages'] if s['name']=='baselines_'+key[1]);pifolder=ROOT/piv['output'];pi=row(pifolder/'runs'/f'{key[0]}__ratio_effort__{key[2]}',key,'same_prior_legacy_ratio',oldpi['source_sha256'])
        pitrials=stage_data(pifolder,oldpi['source_sha256'],oldpi['environment']);require(sum(v==pi['original_result'] for v in pitrials)==1,'PI actual reference result not present once')
        require(pi['metadata_identity']['task_policy']=='legacy' and pi['metadata_identity']['receiver_policy']=='blind','original ratio PI policy identity differs')
        for ref in (reference,pi):
            refs.append(ref);paired=parameter_equal(candidate['parameters'],ref['parameters'])
            if not paired:issues.append(str(key)+'/'+ref['record_key'][0]+'/unpaired_parameters')
            pairs.append(dict(candidate_record_key=candidate['record_key'],reference_record_key=ref['record_key'],reference_role=ref['record_key'][0],context_key=list(key),same_physical_noise_material=paired,candidate_metrics=candidate['metrics'],reference_metrics=ref['metrics'],differences=differences(candidate['metrics'],ref['metrics'],paired),candidate_original_task_success=candidate['original_result']['task_success'],reference_original_task_success=ref['original_result']['task_success']))
    require(len(candidates)==16 and len(pairs)==32 and len({tuple(c['record_key']) for c in candidates})==16,'all16/32 pairs required')
    return dict(schema='v21_complete16_release_variation_v1',complete=not issues,assigned_count=16,all_assigned_read=True,records=candidates,reference_records=refs,pair_count=32,pairs=pairs,summaries=summaries(pairs),issues=issues,plan_sha256=sha(MASTER),scope_contract_sha256=SCOPE_SHA,original9_contract_sha256=NINE_CONTRACT_SHA,calculate_source_sha256=CALCULATOR_SHA,reader_sha256=sha(__file__),source_sha256=plan['source_sha256'],environment=plan['environment'],completion_barrier=barrier,input_sha256=inputs,unique_trace_load_count=len(cache),original_metrics_changed=False,definition_changed=False,physics_steps=0,performance_scope='All16 repeated contexts,2 physical seed clusters;16 V20 and16 oldPI references,not32 new baseline experiments. Raw releaseTV includes intended unloading/motion,not a weighted score or delivery certificate.')

def markdown(d,out):
    lines=['# 完整16条件释放TV诊断','', '固定原12–16秒801个200Hz样本，无滤波。全部16候选/32旧同条件配对保留，不筛成功。两个物理seed复用，不作独立n16或总体优越结论。','', '|对照|字段|全部配对|可评|降低/相同/增加/未知|绝对变化中位数N|相对变化中位数|','|---|---|---:|---:|---|---:|---:|']
    fmt=lambda x:'不可评' if x is None else f'{x:.8g}'
    for s in d['summaries']:
        if s['physical_seed'] is not None:continue
        lines.append(f"|{s['reference_role']}|{s['metric']}|{s['assigned_pairs']}|{s['known_pairs']}|{s['lower']}/{s['equal']}/{s['higher']}/{s['unknown_or_unpaired']}|{fmt(s['absolute_delta_median_N'])}|{fmt(s['relative_change_median'])}|")
    lines+=['','JSON另保留逐对值、原任务失败/NULL、按seed统计和所有输入SHA。相对变化在对照0时为NULL，绝对0仍合法。skin指标沿旧公式仅五指全部外接触，不含palm/aux；TV含刻意卸力与运动，较低不自动代表更稳定/安全交付。','',f'完整数据：{out.name}。没有改原v3成功定义、控制源或数据。']
    return '\n'.join(lines)+'\n'

def self_test():
    frozen_pins();calc=load_calculator();checks=[]
    def ck(name,condition):checks.append(dict(name=name,passed=bool(condition)))
    t=12.+np.arange(801)*.005;zero=np.zeros((801,5));step=zero.copy();step[500:,3]=3.;zig=zero.copy();zig[1::2,1]=1.
    ck('original_calculate_reused_0_step_and_zig_rawunits',calc(t,zero,zero)['target_normal_TV_N']==0. and calc(t,step,zig)['target_normal_TV_N']==3. and calc(t,step,zig)['all_actor_skin_TV_N']==800.)
    missing=calc(t[:-1],zero[:-1],zero[:-1]);nan=step.copy();nan[10,0]=np.nan;v=calc(t,nan,zero)
    ck('window_missing_and_channel_nonfinite_preserve_NULL',missing['target_normal_TV_N'] is None and v['target_normal_TV_N'] is None and v['all_actor_skin_TV_N']==0.)
    d=differences(dict(target_normal_TV_N=0.,all_actor_skin_TV_N=None),dict(target_normal_TV_N=0.,all_actor_skin_TV_N=0.),True)
    ck('zero_baseline_relative_and_NULL_not_fabricated',d[FIELDS[0]]['relative_delta'] is None and d[FIELDS[0]]['comparison']=='equal' and d[FIELDS[1]]['comparison']=='unknown_or_unpaired')
    ck('unpaired_values_not_compared',not differences({f:1. for f in FIELDS},{f:2. for f in FIELDS},False)[FIELDS[0]]['paired_available'])
    ps=[]
    for role in ('full_V20','same_prior_legacy_ratio'):
        for i,key in enumerate(sorted(CONTEXTS)):
            a={f:float(i%3) for f in FIELDS};b={f:1. for f in FIELDS}
            if i==15:a[FIELDS[1]]=None
            ps.append(dict(reference_role=role,context_key=list(key),differences=differences(a,b,True),candidate_original_task_success=False))
    ss=summaries(ps);all16=[s for s in ss if s['physical_seed'] is None]
    ck('all16_per_role_no_failed_trial_selection',len(ps)==32 and all(s['assigned_pairs']==16 and s['lower']+s['equal']+s['higher']+s['unknown_or_unpaired']==16 for s in all16))
    ck('two_seed_denominators_and_metricwise_NULL',all(s['assigned_pairs']==8 for s in ss if s['physical_seed'] is not None) and all(s['known_pairs']==15 for s in all16 if s['metric']==FIELDS[1]))
    # Roster/barrier exercise uses only virtual registrations and callbacks.
    scopepin=dict(path='review/v21_complete16_release_variation_contract_v1.json',sha256=SCOPE_SHA);source={str(i):'x' for i in range(170)};env={'native_binaries_sha256':{str(i):'x' for i in range(42)}}
    retained_contexts={('nominal','transport',619741),('fragile_conflict','hold',619741),('friction_drop','transport',619742)};ret=[];roster=[];stages=[]
    for i,key in enumerate(sorted(retained_contexts)):
        stage='ret'+str(i);run='virtual/'+stage;ret.append(dict(record_key=['full_V21',*key],stage=stage,completion_path='virtual/'+stage+'/execution_complete.json',completion_sha256='marker'))
        roster.append(dict(record_key=['full_V21',*key],context_key=list(key),origin='retained_initial3',run=run))
    for motion in MOTIONS:
        for seed in SEEDS:
            scenes=[s for s in SCENES if (s,motion,seed) not in retained_contexts];name=motion+str(seed);output='virtual/'+name;ass=dict(methods=['ratio_effort'],scenarios=scenes,seeds=[seed],motion_mode=motion,task_policy='integrated',receiver_policy='coordinated',release_variant='v21',physical_randomization=True);stages.append(dict(name=name,planned=len(scenes),output=output,assignment=ass))
            for scene in scenes:roster.append(dict(record_key=['full_V21',scene,motion,seed],context_key=[scene,motion,seed],origin='new_remaining13',stage=name,run=output+'/runs/'+scene+'__ratio_effort__'+str(seed)))
    p=dict(planned=13,new_records=13,retained_count=3,combined_count=16,source_sha256=source,environment=env,release_variant='v21',roster=roster,retained=ret,stages=stages,evidence_pins={'release_variation_scope_contract':scopepin},release_variation_scope_contract=scopepin,release_variation_contract_sha256=NINE_CONTRACT_SHA,release_variation_reader_sha256=CALCULATOR_SHA)
    validate_master(p);ck('full16_roster_3retained13new4stages',len(roster)==16 and sum(s['planned'] for s in stages)==13)
    bad=deepcopy(p);bad['roster'][-1]=deepcopy(bad['roster'][0])
    try:validate_master(bad);reject=False
    except ValueError:reject=True
    ck('duplicate_or_wrong_assignment_roster_rejected',reject)
    markers={str(ROOT/s['output']/'execution_complete.json'):dict(execution_complete=True,sources_unchanged=True,runtime_failures=[],planned=s['planned'],trials=s['planned']) for s in stages}
    markers.update({str(ROOT/r['completion_path']):dict(execution_complete=True,sources_unchanged=True,runtime_failures=[],planned=1,trials=1) for r in ret})
    ok=terminal_barrier(p,lambda x:markers[str(x)],lambda x:'marker',lambda x:str(x) in markers);del markers[str(ROOT/stages[-1]['output']/'execution_complete.json')];pending=terminal_barrier(p,lambda x:markers[str(x)],lambda x:'marker',lambda x:str(x) in markers)
    ck('all13_and_retained3_terminal_barrier_pending_not_failure',ok['ready_to_read_outcomes'] and not pending['ready_to_read_outcomes'] and pending['outcomes_read'] is False)
    called=[]
    with patch.dict(collect.__globals__,terminal_barrier=lambda plan:dict(ready_to_read_outcomes=False)),patch.object(np,'load',lambda *a,**k:called.append('NPZ')):
        try:collect(p);rejected=False
        except ValueError:rejected=True
    ck('inner_collect_refuses_outcome_access_before_terminal',rejected and not called)
    goodmeta=dict(scenario={'name':'nominal'},motion_mode='hold',seed=619741,method='ratio_effort',research_source_sha256=source,release_variant='v21',task_policy='integrated',receiver_policy='coordinated');metadata_identity(goodmeta,('nominal','hold',619741),source=source,variant='v21')
    try:metadata_identity(dict(goodmeta,release_variant='prefix'),('nominal','hold',619741),source=source,variant='v21');reject=False
    except ValueError:reject=True
    ck('metadata_dict_context_own_source_and_release_variant_binding',reject)
    a=dict(config={},noise={},limits=[1]*5,budget=5);b=deepcopy(a);b['budget']=6
    ck('physical_noise_material_pairing_gate',parameter_equal(a,a) and not parameter_equal(a,b))
    return dict(schema='v21_complete16_release_variation_software_v1',passed=all(c['passed'] for c in checks),checks=checks,reader_sha256=sha(__file__),scope_contract_sha256=SCOPE_SHA,calculate_source_sha256=CALCULATOR_SHA,nine_contract_sha256=NINE_CONTRACT_SHA,new13_results_read=False,physics_steps=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--execute',action='store_true');ap.add_argument('--out',required=True);ap.add_argument('--md');args=ap.parse_args();out=ROOT/args.out
    require(not out.exists() and (not args.md or not (ROOT/args.md).exists()),'fresh readout paths required')
    if args.self_test:result=self_test()
    elif args.prepare:
        frozen_pins();result=dict(schema='v21_complete16_release_variation_readiness_v1',registered_plan_present=MASTER.is_file(),outcomes_read=False,physics_steps=0,reader_sha256=sha(__file__),scope_contract_sha256=SCOPE_SHA)
        if MASTER.is_file():result.update(terminal_barrier(read(MASTER)),plan_sha256=sha(MASTER))
    elif args.execute:result=collect(read(MASTER))
    else:raise ValueError('explicit --prepare/--self-test/terminal --execute required')
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    if args.execute and args.md:
        with (ROOT/args.md).open('x',encoding='utf-8') as f:f.write(markdown(result,out))
    print(json.dumps(dict(path=str(out),sha256=sha(out),passed=result.get('passed'),complete=result.get('complete'),outcomes_read=result.get('outcomes_read'),physics_steps=0),ensure_ascii=False));return 1 if result.get('passed') is False or result.get('complete') is False else 0
if __name__=='__main__':raise SystemExit(main())

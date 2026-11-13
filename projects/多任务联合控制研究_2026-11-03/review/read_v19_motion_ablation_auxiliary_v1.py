"""All-three-terminal fixed-v3 NoMotion package readout, no model or control.

Primary same-context reference=FullV19; V18/V17 are historical descriptive
references. Never read partial outcomes, select a new algorithm, retarget a
metric, fill unknown with0, or replace original task/qualification labels.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,gzip,hashlib,importlib.util,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'review/v19_motion_ablation_plan_v1.json'
FULL_PLAN=ROOT/'review/v19_small_plan_v1.json'
FULL_AUX=ROOT/'review/v19_small_auxiliary_v1.json'
FULL_PLAN_SHA='b7061760a7e2e124dab4c02535540b7a00b7cf146a75752fed2371520c6141d5'
FULL_AUX_SHA='272936e294d9690ddee2bb2f91f591fefd72aafa1f599beb2286c0ecda268c49'
FULL_READER_SHA='ed870f5bd2a4bd201f5e40df751b84c4756b3a065c2313267a52ecc85f69e96d'
ABLATION_ROLE='same_V19_governor_progress_normal_reference_tasks_without_motion_effort_package'
ROLES=('full_V19','full_V18','full_V17')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def eq(a,b,name):
    if a!=b:raise ValueError(name+' differs')

def terminal_barrier(plan,markers):
    if plan.get('planned')!=3 or len(plan.get('stages',[]))!=3:raise ValueError('exact three assigned stages required')
    names=[s['name'] for s in plan['stages']]
    if len(set(names))!=3:raise ValueError('duplicate assigned stage')
    pending=[]
    for stage in plan['stages']:
        m=markers.get(stage['name'])
        if (m is None or m.get('execution_complete') is not True or m.get('sources_unchanged') is not True or
            m.get('planned')!=1 or m.get('trials')!=1 or m.get('runtime_failures')!=[]):pending.append(stage['name'])
    return dict(ready_to_read_metrics=not pending,pending=pending,metrics_read=False,assigned=3)

def readiness():
    if not PLAN.is_file():return dict(ready_to_read_metrics=False,pending=['registered_plan_missing'],metrics_read=False),None,{}
    plan=read(PLAN);markers={}
    for stage in plan['stages']:
        p=ROOT/stage['output']/'execution_complete.json'
        if p.is_file():markers[stage['name']]=read(p)
    return terminal_barrier(plan,markers),plan,markers

def helpers():
    path=ROOT/'review/read_v19_small_auxiliary_v1.py';eq(sha(path),FULL_READER_SHA,'frozen FullV19 reader')
    full=load(path,'frozen_full_v19_aux_for_ablation')
    paths,pins=full.frozen_dependencies()
    calc=load(paths['calculator'],'frozen_v3_for_v19_ablation')
    projection=load(paths['v18_reader'],'frozen_v18_metrics_for_v19_ablation')
    return full,calc,projection,paths,pins

def metric_projection(aux,projection):
    # Original metric projection plus literal frozen terminal fields only.
    m=projection.metrics(aux);t=aux.get('terminal') or {}
    m.update(terminal_target_hand_clear=t.get('target_hand_clear'),
        terminal_low_absolute_motion=t.get('low_absolute_motion'),
        terminal_target_floor_contact=t.get('target_floor_contact'),
        terminal_target_receiver_top_contact=t.get('target_receiver_top_contact'))
    return m

def disabled_task_tags(path):
    """Read only assigned-task labels; independent execution audit owns QP replay."""
    records=0;bad=[];original_nonzero=0
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            d=json.loads(line);ci=d.get('contact_intents') or {};records+=1
            original=np.asarray(ci.get('original_motion_generalized_effort25'),dtype=object)
            disabled=np.asarray(ci.get('assigned_disabled_generalized_effort25'),dtype=object)
            valid=(ci.get('motion_tasks_enabled') is False and ci.get('ablation_role')==ABLATION_ROLE and
                ci.get('motion_task_disabled_is_not_missing_observation') is True and
                ci.get('zero_task_is_not_observed_zero_velocity') is True and original.shape==(25,) and disabled.shape==(25,))
            if valid:
                try:
                    a=np.asarray(original,float);z=np.asarray(disabled,float)
                    valid=bool(np.isfinite(a).all() and np.isfinite(z).all() and np.all(z==0.))
                    original_nonzero+=int(np.any(a!=0.))
                except (ValueError,TypeError):valid=False
            if not valid:bad.append(d.get('decision_id'))
    return dict(passed=records>0 and not bad,records=records,invalid_decision_ids=bad,
        original_nonzero_task_diagnostic_records=original_nonzero,
        scope='assigned disable tags only; not proof of unavailable observations or zero measured speed; actual RHS execution audited separately')

def references(full,projection,paths,plan):
    eq(sha(FULL_PLAN),FULL_PLAN_SHA,'original FullV19 plan');eq(sha(FULL_AUX),FULL_AUX_SHA,'complete FullV19 auxiliary')
    p=read(FULL_PLAN);a=read(FULL_AUX)
    if a.get('complete') is not True or a.get('issues')!=[] or len(a.get('records',[]))!=4:raise ValueError('complete original FullV19 four required')
    eq(a['plan_sha256'],FULL_PLAN_SHA,'FullV19 auxiliary plan chain');eq(plan['full_plan_sha256'],FULL_PLAN_SHA,'ablation parent plan pin')
    eq(a['source_sha256'],p['source_sha256'],'FullV19 auxiliary source')
    if len(plan.get('source_sha256',{}))!=156 or len(p['source_sha256'])!=152 or any(plan['source_sha256'].get(k)!=v for k,v in p['source_sha256'].items()):
        raise ValueError('unchanged complete Full152 plus new4 required')
    added=set(plan['source_sha256'])-set(p['source_sha256'])
    if added!={'v19_motion_ablation/'+name for name in ('__init__.py','adapter.py','runtime.py','run_development.py')}:raise ValueError('unexpected ablation source additions')
    eq(plan['environment'],p['environment'],'native environment unchanged')
    index=full.reference_index(paths,p,projection)
    full_rows={tuple(r['context_key']):r for r in a['records'] if r['stage']!='passive_ratio_transport'}
    for stage in plan['stages']:
        key=full.context(stage['assignment']);row=full_rows[key];run=Path(row['run'])
        eq(run.resolve(),(ROOT/stage['full_reference']).resolve(),'registered same-context FullV19 reference')
        for name,digest in row['input_sha256'].items():eq(sha(run/name),digest,'preserved FullV19 '+name)
        meta=read(run/'metadata.json');result=read(run/'result.json')
        eq(sha(run/'trace.npz'),stage['full_trace_sha256'],'registered FullV19 trace')
        eq(meta['trace_sha256'],stage['full_trace_sha256'],'FullV19 metadata trace pin')
        eq(result,row['original_result'],'FullV19 original result preserved')
        eq(meta['research_source_sha256'],p['source_sha256'],'FullV19 record source')
        eq((projection.scenario(meta['scenario']),meta['motion_mode'],meta['seed']),key,'FullV19 metadata context')
        index[('full_V19',*key)]=dict(auxiliary=row['auxiliary'],physical_config=meta['config'],noise=meta['noise'],
            limits=meta['limits'],budget=meta['budget'],gains=meta['gains'],record_id='FullV19/'+row['stage'])
    return index,p,a

def collect(plan,markers):
    full,calc,projection,paths,pins=helpers();refs,fullplan,fullaux=references(full,projection,paths,plan)
    rows=[];pairs=[];issues=[];marker_pins={}
    for stage in plan['stages']:
        dest=ROOT/stage['output'];assignment=stage['assignment'];key=full.context(assignment);marker=markers[stage['name']]
        if assignment['task_policy']!='integrated' or assignment['receiver_policy']!='coordinated':raise ValueError('assigned integrated coordinated ablation required')
        marker_pins[stage['name']]=sha(dest/'execution_complete.json')
        for file,field in (('trials.json','trials_sha256'),('assignment_ledger.json','assignment_ledger_sha256'),('plan.json','plan_sha256'),('manifest.json','manifest_sha256')):
            eq(sha(dest/file),marker[field],'completion '+file)
        ledger=read(dest/'assignment_ledger.json');trials=read(dest/'trials.json');manifest=read(dest/'manifest.json');reg=read(ROOT/stage['registration'])
        if len(ledger)!=1 or len(trials)!=1 or ledger[0]['status']!='evaluated':raise ValueError('full unique assigned record required')
        eq(manifest['research_source_sha256'],plan['source_sha256'],'executed source156')
        eq(reg['source_sha256'],plan['source_sha256'],'registration source');eq(reg['assignment'],assignment,'registration assignment')
        eq(reg['environment'],plan['environment'],'registration environment');eq(read(dest/'environment.json')['pinned_environment'],plan['environment'],'executed native environment')
        run=Path(ledger[0]['job']['destination']);eq(run.resolve(),(dest/'runs'/f'{key[0]}__ratio_effort__{key[2]}').resolve(),'assigned output directory')
        meta=read(run/'metadata.json');result=read(run/'result.json')
        eq((projection.scenario(meta['scenario']),meta['motion_mode'],meta['seed']),key,'actual metadata context')
        eq((meta['method'],meta['task_policy'],meta['receiver_policy']),('ratio_effort','integrated','coordinated'),'actual assigned policies')
        eq(meta['research_source_sha256'],plan['source_sha256'],'actual record source')
        eq(meta['frozen_full_v19_source_sha256'],fullplan['source_sha256'],'actual complete FullV19 parent source')
        if meta.get('ablation_role')!=ABLATION_ROLE or meta.get('motion_tasks_enabled') is not False:raise ValueError('actual metadata ablation role/disable label missing')
        eq(result,trials[0],'original evaluator record');tags=disabled_task_tags(run/'release_decisions.jsonl.gz')
        if not tags['passed']:issues.append(stage['name']+'/disabled_task_tags_invalid')
        original_release=meta.get('mechanical_release_at',meta.get('release_at'));aux=metric=None;errors=[]
        if original_release is None:status='no_actual_release_clock'
        else:
            try:
                with np.load(run/'delivery_intervals.npz',allow_pickle=False) as z:arrays={k:z[k] for k in calc.C['input_fields']}
                aux=calc.calculate(arrays,release_at_s=original_release,episode_end_s=meta['duration'],physical_period_s=meta['physics_timestep'],original_result=result)
                metric=metric_projection(aux,projection);status=aux['status'];del arrays
            except (ValueError,TypeError,KeyError) as exc:
                status='diagnostic_unavailable';errors.append(type(exc).__name__+': '+str(exc));issues.append(stage['name']+'/diagnostic_unavailable')
        row=dict(stage=stage['name'],context_key=list(key),run=str(run),handoff_passed=meta['handoff_passed'],original_result=result,
            status=status,issues=errors,metrics=metric,auxiliary=aux,original_release_cost=full.original_release_cost(result),assigned_task_tags=tags,
            input_sha256={name:sha(run/name) for name in ('metadata.json','result.json','delivery_intervals.npz','release_decisions.jsonl.gz')})
        rows.append(row)
        for role in ROLES:
            ref=refs[(role,*key)];rm=metric_projection(ref['auxiliary'],projection)
            paired=all(meta[k]==ref[rk] for k,rk in (('config','physical_config'),('noise','noise'),('limits','limits'),('budget','budget')))
            gain_equal=(meta['gains']==ref['gains']) if role=='full_V19' else None
            if not paired or gain_equal is False:issues.append(stage['name']+'/'+role+'/unpaired_parameters')
            pairs.append(dict(stage=stage['name'],context_key=list(key),reference_role=role,reference_record_id=ref['record_id'],
                same_physical_noise_limits=paired,same_full_v19_gains=gain_equal,candidate_metrics=metric,reference_metrics=rm,
                candidate_original_release_cost=row['original_release_cost'],reference_original_release_cost=full.original_release_cost(ref['auxiliary']['original_result']),
                differences=full.differences(metric,rm,paired and gain_equal is not False),
                causal_scope='assigned motion-effort package only' if role=='full_V19' else 'historical descriptive comparison, multiple architecture differences'))
    if len(rows)!=3 or len(pairs)!=9:raise ValueError('exact3 attempts and9 references required')
    return dict(schema='v19_motion_ablation_auxiliary_v1',complete=not issues,status='computed_all_registered_three' if not issues else 'computed_with_issues',
        created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(PLAN),source_sha256=plan['source_sha256'],environment=plan['environment'],
        original_full_v19_plan_sha256=FULL_PLAN_SHA,original_full_v19_auxiliary_sha256=FULL_AUX_SHA,
        frozen_full_reader_sha256=FULL_READER_SHA,calculator_sha256=pins['calculator'],contract_sha256=pins['contract'],
        original_v18_auxiliary_sha256=pins['v18_aux'],original_v18_plan_sha256=pins['v18_plan'],original_v17_complete80_sha256=pins['old80'],
        reader_sha256=sha(__file__),completion_marker_sha256=marker_pins,planned=3,records=rows,pairs=pairs,comparison_pairs=9,issues=issues,
        passive_bridge_is_original_full_v19_integration_only=fullaux['passive_auxiliary_equivalence'],
        definition_changed=False,original_metrics_changed=False,physics_steps=0,
        performance_scope='Three seen same-context package ablations, two physical seed clusters; no independent n3, superiority selection or weighted total; all failures and NULL retained')

def report(payload,out):
    names={'full_V19':'完整V19','full_V18':'原V18','full_V17':'原V17'}
    fields=('original_task_success','original_200Hz_clearance_latency_s','terminal_target_hand_clear','first_low_motion_found','final_low_motion_suffix_found',
        'ever_floor','end_linear_speed_m_s','end_angular_speed_rad_s','original_full_guard_pass','original_full_max_finger_peak_N','original_full_local_excess_Ns')
    def fmt(x):
        if x is None:return '不可评/未达到'
        if type(x) is bool:return '是' if x else '否'
        return f'{x:.6g}' if type(x) in (float,int) else str(x)
    lines=['# V19运动任务包消融：固定辅助v3完整读数','',
        '全部三阶段终态后才读结果。同FullV19的参考／normal／gamma／governor／原生分支／物理和设备完全相同，明确禁用组合运动努力任务。此为机制对照，不能称选择了一个新赢家；V18/V17仅历史描述性参考。',
        '', '|条件/版本|原任务|原清离s|末态手clear|首次驻留|末段驻留|曾floor|末线速m/s|末角速rad/s|全16s力限|全指峰N|全局部超力Ns|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in payload['records']:
        matched=[p for p in payload['pairs'] if p['stage']==row['stage']]
        for name,m in [('NoMotion',row['metrics'])]+[(names[x['reference_role']],x['reference_metrics']) for x in matched]:
            lines.append('|'+row['stage']+'/'+name+'|'+'|'.join(fmt((m or {}).get(k)) for k in fields)+'|')
    lines+=['','原task/qualified/physical暴露、frame与raw清离、首次/末段驻留与deadline、触台20ms峰/冲量、完整终态及释放分段原成本全存JSON。NULL不作0，不删失败或仅挑成功清离；原辅助阈值/窗口/成功标签均未修改。',
        '',f'完整结果`{out}`，SHA `{sha(out)}`。issues={len(payload["issues"])}，9组配对各自显示参数匹配与代价；不以低速度但未清手称稳定交付。']
    return '\n'.join(lines)+'\n'

def self_test():
    p=dict(planned=3,stages=[dict(name=str(i)) for i in range(3)]);good=dict(execution_complete=True,sources_unchanged=True,planned=1,trials=1,runtime_failures=[])
    markers={str(i):dict(good) for i in range(3)};checks=[terminal_barrier(p,markers)['ready_to_read_metrics']]
    for key,value in (('execution_complete',False),('sources_unchanged',False),('runtime_failures',[{'error':'x'}]),('trials',0)):
        bad={k:dict(v) for k,v in markers.items()};bad['1'][key]=value;checks.append(not terminal_barrier(p,bad)['ready_to_read_metrics'])
    checks.append(not terminal_barrier(p,{k:v for k,v in markers.items() if k!='2'})['ready_to_read_metrics'])
    full,_,projection,_,_=helpers();eq(sha(FULL_AUX),FULL_AUX_SHA,'complete frozen reference');a=read(FULL_AUX)
    checks.append(all(all(metric_projection(r['auxiliary'],projection)[k]==v for k,v in r['metrics'].items()) for r in a['records']))
    d=full.differences({'failure':False,'latency':None,'dose':0.},{'failure':True,'latency':1.,'dose':0.},True)
    checks.append(d['latency']['delta'] is None and d['dose']['relative_delta'] is None and d['failure']['delta']==-1.)
    return dict(passed=all(checks),checks=len(checks),new_ablation_outcomes_read=False,physics_steps=0,
        regression='all4 completed FullV19 metric dictionaries exact common fields; terminal fields copied from frozen auxiliary')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--execute',action='store_true')
    ap.add_argument('--out',required=True);ap.add_argument('--md');args=ap.parse_args();out=ROOT/args.out
    if out.exists() or (args.md and (ROOT/args.md).exists()):raise FileExistsError('fresh output path required')
    if args.self_test:payload=dict(schema='v19_ablation_auxiliary_reader_software_v1',**self_test(),reader_sha256=sha(__file__))
    elif args.prepare or not args.execute:
        payload=dict(schema='v19_ablation_auxiliary_readiness_v1',prepared=True,reader_sha256=sha(__file__),plan_present=PLAN.is_file(),
            plan_sha256=sha(PLAN) if PLAN.is_file() else None,completion_markers_not_inspected=True,new_ablation_outcomes_read=False,physics_steps=0)
    else:
        barrier,plan,markers=readiness()
        if not barrier['ready_to_read_metrics']:raise ValueError('all3 complete before result/NPZ access: '+repr(barrier['pending']))
        payload=collect(plan,markers)
    with out.open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    if args.execute and args.md and payload.get('complete'):
        with (ROOT/args.md).open('x',encoding='utf-8') as f:f.write(report(payload,out))
    print(json.dumps(dict(path=str(out),sha256=sha(out),passed=payload.get('passed'),complete=payload.get('complete'),status=payload.get('status'),physics_steps=0),ensure_ascii=False))
    return 1 if payload.get('passed') is False or (args.execute and payload.get('complete') is not True) else 0
if __name__=='__main__':raise SystemExit(main())

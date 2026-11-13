"""Frozen-v3 four-stage V19 readout; no result/ledger read before all terminal.

One passive bridge plus three seen candidate contexts. References are the same
V17/A/ratio records and the original, plan-bound V18 three records. Definitions,
original labels, failed assignments and NULL are retained without new scores.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'review/v19_small_plan_v1.json'
CALC_SHA='0228b2526ace5f448a6f7f458ebeadb9023b79b120fe4ba16eccb3463eb5bc55'
CONTRACT_SHA='8dcda359252014d509a8b26c102570309ad893b6a4ccad31015a36ca0cb7e979'
V18_READER_SHA='962f5b8c346bdfe7f044878c93246e3681b71201ee39abfe4294ade3a5d7d49e'
V18_AUX_SHA='de5d913d056e4ad72303aa7085e1c4c02e11d53b008a4728e6ea1cb4c346a681'
V18_PLAN_SHA='d7a78553acf3c58e5093770c0f16b477691933adc03ceb05c58426b166db8c8f'
OLD80_SHA='8a5e9a3cdfcffe4f484ded0c183ec72da718d7c29f30264fbea65beea459e116'
ROLES=('full_V17','holding_A','same_prior_legacy_ratio','full_V18')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def assert_equal(a,b,name):
    if a!=b:raise ValueError(name+' differs')
def context(assignment):
    if (assignment.get('methods')!=['ratio_effort'] or len(assignment.get('scenarios',[]))!=1 or
        len(assignment.get('seeds',[]))!=1 or assignment.get('physical_randomization') is not True or
        assignment.get('motion_mode') not in ('hold','transport')):raise ValueError('explicit one-job paired assignment required')
    return (assignment['scenarios'][0],assignment['motion_mode'],assignment['seeds'][0])

def terminal_barrier(plan,markers):
    """Only assignment descriptors and completion receipts, never outcomes."""
    if plan.get('planned')!=4 or len(plan.get('stages',[]))!=4:raise ValueError('exact registered four stages required')
    names=[s['name'] for s in plan['stages']]
    if len(set(names))!=4:raise ValueError('duplicate stage name')
    pending=[]
    for stage in plan['stages']:
        context(stage['assignment']);m=markers.get(stage['name'])
        if (m is None or m.get('execution_complete') is not True or m.get('sources_unchanged') is not True or
            m.get('runtime_failures')!=[] or m.get('trials')!=1 or m.get('planned')!=1):pending.append(stage['name'])
    return dict(ready_to_read_metrics=not pending,pending=pending,assigned=4,metrics_read=False)

def readiness():
    if not PLAN.is_file():return dict(ready_to_read_metrics=False,pending=['registered_v19_small_plan_not_present'],metrics_read=False),None,{}
    plan=read(PLAN);markers={}
    for s in plan['stages']:
        file=ROOT/s['output']/'execution_complete.json'
        if file.is_file():markers[s['name']]=read(file)
    return terminal_barrier(plan,markers),plan,markers

def frozen_dependencies():
    paths=dict(calculator=ROOT/'review/delivery_auxiliary_metrics_v3.py',contract=ROOT/'review/delivery_auxiliary_contract_v3.json',
        v18_reader=ROOT/'review/read_v18_small_auxiliary_v1.py',v18_aux=ROOT/'review/v18_small_auxiliary_v1.json',
        v18_plan=ROOT/'review/v18_small_plan_v1.json',old80=ROOT/'review/v17b_delivery_auxiliary_readout_v1.json')
    expected=dict(calculator=CALC_SHA,contract=CONTRACT_SHA,v18_reader=V18_READER_SHA,v18_aux=V18_AUX_SHA,v18_plan=V18_PLAN_SHA,old80=OLD80_SHA)
    for k,p in paths.items():assert_equal(sha(p),expected[k],'frozen '+k+' identity')
    return paths,expected

def original_release_cost(result):
    # Direct original evaluator fields, no threshold/window recomputation.
    x=result.get('safety',{}).get('stages',{}).get('mechanical_release_and_after')
    return None if x is None else dict(finger_peak_N=x.get('finger_peak_N'),
        local_excess_integral_Ns=x.get('local_excess_integral_Ns'),total_excess_integral_Ns=x.get('total_excess_integral_Ns'),
        duration_s=x.get('duration_s'),scope='original mechanical_release_and_after, original metric definition')

def differences(candidate,reference,paired):
    keys=set(candidate or {})|set(reference or {});answer={}
    for k in sorted(keys):
        a=(candidate or {}).get(k);b=(reference or {}).get(k)
        known=bool(paired and a is not None and b is not None and type(a) in (bool,int,float) and type(b) in (bool,int,float))
        answer[k]=dict(candidate=a,reference=b,paired_available=known,delta=float(a)-float(b) if known else None,
            relative_delta=((float(a)-float(b))/float(b) if known and type(a) in (int,float) and type(b) in (int,float) and b!=0 else None))
    return answer

def reference_index(paths,plan,helper):
    old=read(paths['old80']);v18=read(paths['v18_aux']);oldplan=read(paths['v18_plan'])
    if v18.get('complete') is not True or v18.get('issues')!=[] or len(v18['records'])!=4:raise ValueError('original V18 complete four required')
    assert_equal(v18['plan_sha256'],sha(paths['v18_plan']),'V18 auxiliary plan chain')
    assert_equal(v18['source_sha256'],oldplan['source_sha256'],'V18 auxiliary source chain')
    if len(oldplan['source_sha256'])!=144 or any(plan['source_sha256'].get(k)!=v for k,v in oldplan['source_sha256'].items()):
        raise ValueError('parent executed V18 source144 changed')
    refs={tuple(r['record_key']):dict(auxiliary=r['auxiliary'],physical_config=r['physical_config'],noise=r['noise'],
        limits=r['limits'],budget=r['budget'],record_id=r['record_id']) for r in old['records']}
    for stage in oldplan['stages']:
        assignment=stage['assignment']
        if assignment['task_policy']!='integrated' or assignment['receiver_policy']!='coordinated':continue
        key=context(assignment);row=next(r for r in v18['records'] if r['stage']==stage['name'])
        assert_equal(tuple(row['context_key']),key,'V18 row context')
        run=Path(row['run']);expected=(ROOT/stage['output']/'runs'/f'{key[0]}__ratio_effort__{key[2]}').resolve()
        assert_equal(run.resolve(),expected,'V18 registered run directory')
        for name,value in row['input_sha256'].items():assert_equal(sha(run/name),value,'V18 preserved '+name)
        meta=read(run/'metadata.json');result=read(run/'result.json')
        assert_equal(result,row['original_result'],'V18 original result preserved')
        assert_equal((helper.scenario(meta['scenario']),meta['motion_mode'],meta['seed']),key,'V18 real metadata identity')
        refs[('full_V18',*key)]=dict(auxiliary=row['auxiliary'],physical_config=meta['config'],noise=meta['noise'],
            limits=meta['limits'],budget=meta['budget'],record_id='plan_bound_V18/'+stage['name'])
    return refs

def collect(plan,markers):
    paths,pins=frozen_dependencies();calc=load_module(paths['calculator'],'frozen_v3_aux_for_v19')
    helper=load_module(paths['v18_reader'],'frozen_v18_aux_projection_for_v19')
    if len(plan.get('source_sha256',{}))!=152:raise ValueError('registered FullV19 source152 required')
    refs=reference_index(paths,plan,helper);rows=[];pairs=[];issues=[];passive=[];candidate_keys=[]
    marker_pins={}
    for s in plan['stages']:
        dest=ROOT/s['output'];mark=markers[s['name']];ass=s['assignment'];key=context(ass)
        marker_pins[s['name']]=sha(dest/'execution_complete.json')
        ledger=read(dest/'assignment_ledger.json');trials=read(dest/'trials.json');manifest=read(dest/'manifest.json');reg=read(ROOT/s['registration'])
        assert_equal(len(ledger),1,'single assigned ledger');assert_equal(len(trials),1,'single original result')
        assert_equal(ledger[0]['status'],'evaluated','assigned job evaluated')
        for file,field in (('trials.json','trials_sha256'),('assignment_ledger.json','assignment_ledger_sha256'),('plan.json','plan_sha256'),('manifest.json','manifest_sha256')):
            assert_equal(sha(dest/file),mark[field],'completion '+file)
        assert_equal(manifest['research_source_sha256'],plan['source_sha256'],'manifest source')
        assert_equal(reg['source_sha256'],plan['source_sha256'],'registration source');assert_equal(reg['environment'],plan['environment'],'registration environment')
        assert_equal(reg['assignment'],ass,'registration assignment')
        assert_equal(read(dest/'environment.json')['pinned_environment'],plan['environment'],'executed native environment')
        run=Path(ledger[0]['job']['destination']);expected=(dest/'runs'/f'{key[0]}__ratio_effort__{key[2]}').resolve()
        assert_equal(run.resolve(),expected,'actual assigned run directory')
        meta=read(run/'metadata.json');result=read(run/'result.json')
        assert_equal((helper.scenario(meta['scenario']),meta['motion_mode'],meta['seed']),key,'real scenario dict/seed/motion identity')
        assert_equal((meta['method'],meta['task_policy'],meta['receiver_policy']),('ratio_effort',ass['task_policy'],ass['receiver_policy']),'actual policy assignment')
        assert_equal(meta['research_source_sha256'],plan['source_sha256'],'record source');assert_equal(result,trials[0],'record original evaluator output')
        release=meta.get('mechanical_release_at',meta.get('release_at'));a=None;m=None;errors=[]
        if release is None:status='no_actual_release_clock'
        else:
            try:
                with np.load(run/'delivery_intervals.npz',allow_pickle=False) as z:arrays={k:z[k] for k in calc.C['input_fields']}
                a=calc.calculate(arrays,release_at_s=release,episode_end_s=meta['duration'],physical_period_s=meta['physics_timestep'],original_result=result)
                m=helper.metrics(a);status=a['status'];del arrays
            except (ValueError,TypeError,KeyError) as exc:
                status='diagnostic_unavailable';errors.append(type(exc).__name__+': '+str(exc))
        row=dict(stage=s['name'],context_key=list(key),run=str(run),handoff_passed=meta['handoff_passed'],original_result=result,
            original_release_cost=original_release_cost(result),issues=errors,status=status,auxiliary=a,metrics=m,
            input_sha256={name:sha(run/name) for name in ('metadata.json','result.json','delivery_intervals.npz')})
        rows.append(row)
        if errors:issues.append(s['name']+'/diagnostic_unavailable')
        if ass['task_policy']=='legacy' and ass['receiver_policy']=='blind':
            ref=refs[('same_prior_legacy_ratio',*key)]
            exact=a is not None and helper.same_non_original(a,ref['auxiliary'])
            passive.append(dict(stage=s['name'],context_key=list(key),reference_role='same_prior_legacy_ratio',
                all_non_original_auxiliary_fields_exact=bool(exact)))
            if not exact:issues.append(s['name']+'/passive_auxiliary_non_original_fields_differ')
        elif ass['task_policy']=='integrated' and ass['receiver_policy']=='coordinated':
            candidate_keys.append(key)
            for role in ROLES:
                ref=refs[(role,*key)];rm=helper.metrics(ref['auxiliary'])
                paired=all(meta[k]==ref[rk] for k,rk in (('config','physical_config'),('noise','noise'),('limits','limits'),('budget','budget')))
                pairs.append(dict(stage=s['name'],context_key=list(key),reference_role=role,reference_record_id=ref['record_id'],
                    same_physical_noise_limits=paired,candidate_metrics=m,reference_metrics=rm,
                    candidate_original_release_cost=row['original_release_cost'],reference_original_release_cost=original_release_cost(ref['auxiliary']['original_result']),
                    differences=differences(m,rm,paired)))
                if not paired:issues.append(s['name']+'/'+role+'/unpaired_parameters')
        else:raise ValueError('unexpected task/receiver role in small registered plan')
    if len(passive)!=1 or len(candidate_keys)!=3 or len(set(candidate_keys))!=3 or len(pairs)!=12:raise ValueError('exact1 passive +3 candidate x4 references required')
    payload=dict(schema='v19_small_auxiliary_v1',complete=not issues,status='computed_all_registered_four' if not issues else 'computed_with_issues',
        created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(PLAN),source_sha256=plan['source_sha256'],environment=plan['environment'],
        calculator_sha256=CALC_SHA,contract_sha256=CONTRACT_SHA,metric_projection_from_frozen_v18_reader_sha256=V18_READER_SHA,
        reference_complete80_sha256=OLD80_SHA,reference_v18_aux_sha256=V18_AUX_SHA,reference_v18_plan_sha256=V18_PLAN_SHA,
        reader_sha256=sha(__file__),completion_marker_sha256=marker_pins,planned=4,records=rows,candidate_contexts=3,
        comparison_pairs=12,pairs=pairs,passive_auxiliary_equivalence=passive[0],issues=issues,
        definition_changed=False,original_metrics_changed=False,physics_steps=0,
        performance_scope='three seen development candidate contexts, two physical seed clusters; passive bridge is integration only; no independent confirmation or weighted total; all failed and NULL retained')
    return payload

def text_report(payload,out):
    label={'full_V17':'原V17','holding_A':'原持握A','same_prior_legacy_ratio':'原力比PI','full_V18':'原V18'}
    lines=['# V19四条完整开发记录：原v3辅助读数','',
        '四阶段全部完成后才读取。1条被动桥、3个已见候选条件；每个条件与原V17/A/力比PI/V18同物理、噪声和材料限配对。不同版本不构成新独立随机样本。原任务、力限、清离和交付读数分开，不调阈值或重算成功标签。','',
        f"被动辅助字段精确复现（只排原结果身份）：{payload['passive_auxiliary_equivalence']['all_non_original_auxiliary_fields_exact']}。",'',
        '|条件/版本|原任务|原清离s|首次低运动|末段低运动|曾floor|末线速m/s|末角速rad/s|触台20ms峰N|冲量Ns|全16s力限|全指峰N|全局部超力Ns|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):
        if x is None:return '不可评/未达到'
        if type(x) is bool:return '是' if x else '否'
        return f'{x:.6g}' if type(x) in (int,float) else str(x)
    for row in payload['records']:
        matched=[p for p in payload['pairs'] if p['stage']==row['stage']]
        if not matched:continue
        bundle=[('新V19',row['metrics'])]+[(label[p['reference_role']],p['reference_metrics']) for p in matched]
        for role,m in bundle:
            m=m or {};fields=('original_task_success','original_200Hz_clearance_latency_s','first_low_motion_found','final_low_motion_suffix_found','ever_floor',
                'end_linear_speed_m_s','end_angular_speed_rad_s','receiver_first20ms_peak_N','receiver_first20ms_impulse_Ns','original_full_guard_pass','original_full_max_finger_peak_N','original_full_local_excess_Ns')
            lines.append('|'+row['stage']+'/'+role+'|'+'|'.join(fmt(m.get(k)) for k in fields)+'|')
    lines+=['','JSON保留原qualified/exposed、raw首次清离、首次/末段驻留及deadline、接收事件既有/新生/未知、末态与原释放分段成本。NULL不作零；unknown、实际任务失败与未达到驻留不相互替换。首次触台冲击只在冻结v3判据可评时显示，不把已有支撑伪装新撞击。',
        '', '这里只是固定定义的辅助读数；低手压/接收面载荷不等于稳定交付或长期放置。不能据3个已见条件宣称全局优势、独立n3或新成功率。',
        '',f'完整JSON：`{out}`，SHA `{sha(out)}`。']
    return '\n'.join(lines)+'\n'

def self_test():
    ass=dict(methods=['ratio_effort'],scenarios=['nominal'],seeds=[619741],motion_mode='transport',physical_randomization=True)
    plan=dict(planned=4,stages=[dict(name=str(i),assignment=ass) for i in range(4)])
    good=dict(execution_complete=True,sources_unchanged=True,runtime_failures=[],trials=1,planned=1)
    markers={str(i):dict(good) for i in range(4)};checks=[]
    checks.append(terminal_barrier(plan,markers)['ready_to_read_metrics'])
    missing=dict(markers);missing.pop('3');checks.append(not terminal_barrier(plan,missing)['ready_to_read_metrics'])
    for key,value in (('execution_complete',False),('sources_unchanged',False),('runtime_failures',[{'error':'x'}]),('trials',0)):
        bad={k:dict(v) for k,v in markers.items()};bad['2'][key]=value;checks.append(not terminal_barrier(plan,bad)['ready_to_read_metrics'])
    d=differences({'force':0.,'stable':None},{'force':0.,'stable':False},True)
    checks.append(d['force']['delta']==0. and d['force']['relative_delta'] is None and d['stable']['delta'] is None)
    paths,_=frozen_dependencies();helper=load_module(paths['v18_reader'],'fixed_metric_projection_selftest');old=read(paths['v18_aux'])
    checks.append(all(helper.metrics(r['auxiliary'])==r['metrics'] for r in old['records']))
    return dict(passed=all(checks),checks=len(checks),physics_steps=0,new_v19_outcomes_read=False,original_metrics_regression='all4 old V18 metric dictionaries exact')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--execute',action='store_true');ap.add_argument('--out',required=True);ap.add_argument('--md');args=ap.parse_args()
    out=ROOT/args.out
    if out.exists() or (args.md and (ROOT/args.md).exists()):raise FileExistsError('fresh output paths required')
    if args.self_test:
        payload=dict(schema='v19_small_auxiliary_reader_software_preparation_v1',**self_test(),reader_sha256=sha(__file__))
    else:
        barrier,plan,markers=readiness()
        if args.prepare or not args.execute:
            payload=dict(schema='v19_small_auxiliary_readiness_v1',**barrier,reader_sha256=sha(__file__),physics_steps=0)
        else:
            if not barrier['ready_to_read_metrics']:raise ValueError('all four terminal before result/NPZ access: '+repr(barrier['pending']))
            payload=collect(plan,markers)
    with out.open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    if args.execute and args.md and payload.get('complete'):
        with (ROOT/args.md).open('x',encoding='utf-8') as f:f.write(text_report(payload,out))
    print(json.dumps(dict(path=str(out),sha256=sha(out),complete=payload.get('complete'),passed=payload.get('passed'),
        status=payload.get('status'),ready_to_read_metrics=payload.get('ready_to_read_metrics'),physics_steps=0),ensure_ascii=False))
    return 1 if payload.get('passed') is False or (args.execute and payload.get('complete') is not True) else 0
if __name__=='__main__':raise SystemExit(main())

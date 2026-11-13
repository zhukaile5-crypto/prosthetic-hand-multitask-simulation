"""Frozen-v3 offline auxiliary readout of the registered four V18 attempts.

No model/control imports, no outcomes until all four markers are terminal.
Three seen candidate contexts are paired with their V17/A/ratio records from
the already published complete80 auxiliary artifact; no posthoc new metric.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CALC_SHA='0228b2526ace5f448a6f7f458ebeadb9023b79b120fe4ba16eccb3463eb5bc55'
CONTRACT_SHA='8dcda359252014d509a8b26c102570309ad893b6a4ccad31015a36ca0cb7e979'
OLD_SHA='8a5e9a3cdfcffe4f484ded0c183ec72da718d7c29f30264fbea65beea459e116'
ROLES=('full_V17','holding_A','same_prior_legacy_ratio')
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scenario(x):
    if type(x) is str and x:return x
    if type(x) is dict and type(x.get('name')) is str and x['name']:return x['name']
    raise ValueError('explicit scenario name required')
def same_non_original(a,b):
    a={k:v for k,v in a.items() if k!='original_result'};b={k:v for k,v in b.items() if k!='original_result'}
    return json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(b,sort_keys=True,separators=(',',':'),allow_nan=False)
def metrics(a):
    o=a['original_result'];rm=o['research_metrics'];term=a.get('terminal') or {};known=a['coverage']['all_valid'];first=a['official_first_stable_receiver'];final=a['official_final_stable_suffix'];impact=a['first_receiver_impact20ms'];ic=impact.get('contact',{}).get('receiver_top',{})
    return dict(original_task_success=o['task_success'],original_full_motion_success=rm['full_motion_mission_success'],original_physical_release_exposure=a['original_release_physical_exposure'],original_task_qualified_release_exposure=rm['release']['task_qualified_exposure'],original_physical_clearance_success=rm['release']['physical_success'],original_200Hz_clearance_latency_s=rm['release']['physical_clearance_latency_s'],raw_first_clearance_latency_s=None if a['observed_raw_hand_clearance']['first'] is None else a['observed_raw_hand_clearance']['first']['latency_s'],first_low_motion_found=(first is not None) if known else None,first_low_motion_latency_s=None if first is None else first['latency_s'],low_motion_by_original_deadline=(a['official_first_stable_by_release_deadline'] is not None) if known else None,final_low_motion_suffix_found=(final is not None and not final['right_censored']) if known else None,final_low_motion_duration_s=None if final is None else final['duration_s'],ever_floor=a['ever_floor_contact'],stable_after_previous_floor_event=a['stable_after_previous_floor_event'],receiver_preexisting_at_request=a['receiver_event_provenance']['contact_at_command'],receiver_onset=a['receiver_event_provenance']['first_positive_onset_status'],end_linear_speed_m_s=term.get('linear_speed_m_s'),end_angular_speed_rad_s=term.get('angular_speed_rad_s'),end_receiver_top=term.get('target_receiver_top_contact'),end_moving_receiver=term.get('moving_receiver_contact'),receiver_first20ms_available=impact['available'],receiver_first20ms_peak_N=ic.get('peak_N') if impact['available'] else None,receiver_first20ms_impulse_Ns=ic.get('impulse_Ns') if impact['available'] else None,original_full_guard_pass=rm['safety']['guard_pass'],original_full_max_finger_peak_N=max(rm['safety']['finger_peak_N']),original_full_local_excess_Ns=rm['safety']['local_excess_integral_Ns'],original_full_total_excess_Ns=rm['safety']['total_excess_integral_Ns'])
def fmt(x,n=4):
    if x is None:return '不可评/未达到'
    if type(x) is bool:return '是' if x else '否'
    return f'{x:.{n}f}' if type(x) in (int,float) else str(x)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--out',required=True);ap.add_argument('--md',required=True);args=ap.parse_args();out=ROOT/args.out;md=ROOT/args.md
    if out.exists() or md.exists():raise FileExistsError('fresh readout paths required')
    pp=ROOT/'review/v18_small_plan_v1.json';plan=read(pp);stages=plan['stages'];markers={};pending=[]
    for s in stages:
        p=ROOT/s['output']/'execution_complete.json'
        if not p.exists():pending.append(s['name']);continue
        markers[s['name']]=read(p)
        if not markers[s['name']].get('execution_complete'):pending.append(s['name'])
    if pending or not args.execute:
        d=dict(schema='v18_small_auxiliary_v1',complete=False,status='pending_all_four_terminal_or_execute',pending=pending,metrics_read=False,physics_steps=0)
        with out.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
        print(json.dumps(d));return 0
    calcpath=ROOT/'review/delivery_auxiliary_metrics_v3.py';contract=ROOT/'review/delivery_auxiliary_contract_v3.json';oldpath=ROOT/'review/v17b_delivery_auxiliary_readout_v1.json'
    assert sha(calcpath)==CALC_SHA and sha(contract)==CONTRACT_SHA and sha(oldpath)==OLD_SHA
    spec=importlib.util.spec_from_file_location('frozen_v3_aux_for_v18',calcpath);calc=importlib.util.module_from_spec(spec);spec.loader.exec_module(calc)
    old=read(oldpath);index={tuple(r['record_key']):r for r in old['records']};rows=[];issues=[];comparisons=[];passive=None
    for s in stages:
        dest=ROOT/s['output'];mark=markers[s['name']];ledger=read(dest/'assignment_ledger.json');trials=read(dest/'trials.json');manifest=read(dest/'manifest.json');reg=read(ROOT/s['registration'])
        assert mark['sources_unchanged'] and not mark['runtime_failures'] and mark['trials']==1
        assert len(ledger)==len(trials)==1 and ledger[0]['status']=='evaluated'
        assert sha(dest/'trials.json')==mark['trials_sha256'] and sha(dest/'assignment_ledger.json')==mark['assignment_ledger_sha256']
        assert manifest['research_source_sha256']==plan['source_sha256']==reg['source_sha256'] and reg['environment']==plan['environment']
        job=ledger[0]['job'];run=Path(job['destination']);meta=read(run/'metadata.json');result=read(run/'result.json');ass=s['assignment']
        key=(scenario(meta['scenario']),meta['motion_mode'],int(meta['seed']))
        assert key==(ass['scenarios'][0],ass['motion_mode'],ass['seeds'][0]) and meta['method']==ass['methods'][0] and meta['task_policy']==ass['task_policy'] and meta['receiver_policy']==ass['receiver_policy']
        assert meta['research_source_sha256']==plan['source_sha256'] and result==trials[0]
        release=meta.get('mechanical_release_at',meta.get('release_at'));record=dict(stage=s['name'],context_key=list(key),run=str(run),handoff_passed=meta['handoff_passed'],original_result=result,issues=[])
        if release is None:record.update(status='no_actual_release_clock',auxiliary=None,metrics=None)
        else:
            with np.load(run/'delivery_intervals.npz',allow_pickle=False) as z:arrays={k:z[k] for k in calc.C['input_fields']}
            a=calc.calculate(arrays,release_at_s=release,episode_end_s=meta['duration'],physical_period_s=meta['physics_timestep'],original_result=result)
            record.update(status=a['status'],auxiliary=a,metrics=metrics(a));del arrays
        record['input_sha256']={n:sha(run/n) for n in ('metadata.json','result.json','delivery_intervals.npz')};rows.append(record)
        if s['name']=='passive_ratio_transport':
            reference=index[('same_prior_legacy_ratio',*key)];passive=dict(stage=s['name'],reference_record_key=reference['record_key'],all_non_original_auxiliary_fields_exact=same_non_original(record['auxiliary'],reference['auxiliary']))
            if not passive['all_non_original_auxiliary_fields_exact']:issues.append('passive_auxiliary_non_original_fields_differ')
        else:
            for role in ROLES:
                ref=index[(role,*key)];equal=meta['config']==ref['physical_config'] and meta['noise']==ref['noise'] and meta['limits']==ref['limits'] and meta['budget']==ref['budget'];m=metrics(ref['auxiliary']);delta={}
                for k,v in record['metrics'].items():
                    rv=m.get(k);avail=equal and v is not None and rv is not None and type(v) in (bool,int,float) and type(rv) in (bool,int,float)
                    delta[k]=dict(candidate=v,reference=rv,paired_available=avail,delta=float(v)-float(rv) if avail else None)
                comparisons.append(dict(stage=s['name'],context_key=list(key),reference_role=role,reference_record_id=ref['record_id'],same_physical_noise_limits=equal,candidate_metrics=record['metrics'],reference_metrics=m,differences=delta))
                if not equal:issues.append(s['name']+'/'+role+'/unpaired_parameters')
    d=dict(schema='v18_small_auxiliary_v1',complete=True,status='computed_all_registered_four',created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(pp),calculator_sha256=CALC_SHA,contract_sha256=CONTRACT_SHA,reference_complete80_sha256=OLD_SHA,reader_sha256=sha(__file__),source_sha256=plan['source_sha256'],planned=4,records=rows,candidate_contexts=3,comparison_pairs=9,pairs=comparisons,passive_auxiliary_equivalence=passive,issues=issues,physics_steps=0,definition_changed=False,original_metrics_changed=False,performance_scope='Three seen development candidate contexts;2 physical seed clusters,not independent confirmation; one passive bridge is integration only; all original failed/NULL preserved. Auxiliary low-motion dwell not placement/safety certificate.')
    with out.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    names={'full_V17':'原V17','holding_A':'原仅持握A','same_prior_legacy_ratio':'原力比PI'}
    lines=['# V18四条开发记录的固定v3辅助读数','', '使用原辅助v3计算器与合同，未改阈值或成功定义。四条全部终态后一次读取；1条被动桥、3个已见候选条件，比较只针对同条件原V17/A/力比PI。不得把3条称泛化证据或3个独立随机物理簇。','',f"被动桥除original_result身份外的全部辅助字段与原力比记录精确相同：{passive['all_non_original_auxiliary_fields_exact']}。原始结果、失败和NULL保留。",'', '|条件/角色|原任务|原物理清离(s)|首次驻留|末段驻留|曾落地|末速(mm/s)|末角速(rad/s)|首次触台20ms峰(N)|冲量(N·s)|全guard|全最大指峰(N)|局部超力(N·s)|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['stage']=='passive_ratio_transport':continue
        bundle=[('新V18',row['metrics'])]+[(names[r],next(p['reference_metrics'] for p in comparisons if p['stage']==row['stage'] and p['reference_role']==r)) for r in ROLES]
        for role,m in bundle:
            speed=m['end_linear_speed_m_s'];cells=[row['stage']+'/'+role,fmt(m['original_task_success']),fmt(m['original_200Hz_clearance_latency_s']),fmt(m['first_low_motion_found']),fmt(m['final_low_motion_suffix_found']),fmt(m['ever_floor']),fmt(None if speed is None else speed*1000),fmt(m['end_angular_speed_rad_s']),fmt(m['receiver_first20ms_peak_N']),fmt(m['receiver_first20ms_impulse_Ns'],6),fmt(m['original_full_guard_pass']),fmt(m['original_full_max_finger_peak_N']),fmt(m['original_full_local_excess_Ns'],6)];lines.append('|'+ '|'.join(cells)+'|')
    lines+=['','各条件全部保留，驻留延迟未达到时为NULL，不作0秒。原200Hz清离和原始积分段手低力驻留并列存于JSON，时钟/求力相位不同，不互相替换。首次触台峰与冲量仅在可评新接触事件时给值，既有支撑和未知起点不造冲击；资格暴露、原任务失败、覆盖与所有原辅助原件均在JSON。','', '任务、清离、力限、未落地、末态速度与低运动驻留是不同维度，不相互代替。接收冲击与手指材料限力也不同，不使用加权总分或仅成功子集。','',f'完整结果：[v18_small_auxiliary_v1.json]({out.name})，SHA `{sha(out)}`。原完整80参考：[v17b_delivery_auxiliary_readout_v1.json](v17b_delivery_auxiliary_readout_v1.json)。']
    with md.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(path=str(out),sha256=sha(out),records=len(rows),pairs=len(comparisons),issues=issues,passive=passive,candidate_metrics=[dict(stage=r['stage'],metrics=r['metrics']) for r in rows if r['stage']!='passive_ratio_transport']),ensure_ascii=False))
    return 0 if not issues else 1

if __name__=='__main__':raise SystemExit(main())

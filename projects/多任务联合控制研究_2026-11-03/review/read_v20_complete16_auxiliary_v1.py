"""All13-terminal, retained3+new13 fixed-v3 readout. No simulator imports."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from copy import deepcopy
import argparse,hashlib,importlib.util,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'review/v20_complete16_plan_v1.json'
SMALL_READER=ROOT/'review/read_v20_small_auxiliary_v1.py';SMALL_READER_SHA='ce9f08bef02473e548cd859a8a75f094e4fc525b8db4c080f877cb2e7e6099ba'
MAIN_ROLES=('full_V17','holding_A','same_prior_legacy_ratio');SECONDARY=('full_V18','full_V19')
SCENES=('nominal','friction_drop','fragile_conflict','collision');SEEDS=(619741,619742);MOTIONS=('hold','transport')
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def equal(a,b,name):
 if a!=b:raise ValueError(name+' mismatch')
def validate_roster(plan):
 if (plan.get('schema')!='v20_complete16_development_plan_v1' or plan.get('new_records')!=13 or plan.get('retained_count')!=3 or plan.get('combined_count')!=16 or len(plan.get('stages',[]))!=4 or len(plan.get('retained',[]))!=3 or len(plan.get('roster',[]))!=16):raise ValueError('exact registered13+3 cohort required')
 universe={('full_V20',s,m,se) for s in SCENES for m in MOTIONS for se in SEEDS};actual=[tuple(x['record_key']) for x in plan['roster']]
 if len(set(actual))!=16 or set(actual)!=universe or len({s['name'] for s in plan['stages']})!=4 or sum(s['planned'] for s in plan['stages'])!=13:raise ValueError('complete unique16 roster/four batch identities required')
 retained={tuple(x['record_key']) for x in plan['retained']};new=[]
 for s in plan['stages']:
  a=s['assignment']
  if a.get('methods')!=['ratio_effort'] or a.get('task_policy')!='integrated' or a.get('receiver_policy')!='coordinated' or a.get('physical_randomization') is not True or len(a.get('seeds',[]))!=1 or s['planned']!=len(a['scenarios']):raise ValueError('exact candidate batch assignment required')
  new.extend(('full_V20',scene,a['motion_mode'],a['seeds'][0]) for scene in a['scenarios'])
 if len(new)!=len(set(new))!=13 or len(new)!=13 or retained & set(new) or retained | set(new)!=universe:raise ValueError('retained3 must never be rerun')
 return universe
def terminal_barrier(plan,markers):
 validate_roster(plan);pending=[]
 for s in plan['stages']:
  m=markers.get(s['name'])
  if m is None or m.get('execution_complete') is not True or m.get('sources_unchanged') is not True or m.get('runtime_failures')!=[] or m.get('planned')!=s['planned'] or m.get('trials')!=s['planned']:pending.append(s['name'])
 return dict(ready_to_read_metrics=not pending,pending=pending,new_assigned=13,retained=3,combined=16,new_metrics_read=False)
def readiness():
 if not PLAN.is_file():return dict(ready_to_read_metrics=False,pending=['master_not_registered'],new_metrics_read=False),None,{}
 p=read(PLAN);markers={}
 for s in p['stages']:
  f=ROOT/s['output']/'execution_complete.json'
  if f.is_file():markers[s['name']]=read(f)
 return terminal_barrier(p,markers),p,markers
def labels(result):
 rm=result.get('research_metrics') or {};rel=rm.get('release') or {};safe=rm.get('safety') or {}
 return dict(task_success=result.get('task_success'),full_motion_success=rm.get('full_motion_mission_success'),physical_clearance_success=rel.get('physical_success'),physical_release_exposure=rel.get('physical_exposure'),task_qualified_exposure=rel.get('task_qualified_exposure'),full_guard_pass=safe.get('guard_pass'))
def tri(values):return dict(assigned=len(values),true=sum(x is True for x in values),false=sum(x is False for x in values),unknown=sum(x is not True and x is not False for x in values))
def summarize(rows):
 keys=('first_low_motion_found','final_low_motion_suffix_found','ever_floor','receiver_preexisting_at_request','original_physical_release_exposure','original_task_qualified_release_exposure')
 stats={k:tri([(r.get('metrics') or {}).get(k) for r in rows]) for k in keys}
 stats['original_labels']={k:tri([r.get('original_labels',{}).get(k) for r in rows]) for k in ('task_success','full_motion_success','physical_clearance_success','physical_release_exposure','task_qualified_exposure','full_guard_pass')}
 for k in ('first_low_motion_latency_s','final_low_motion_duration_s','end_linear_speed_m_s','end_angular_speed_rad_s','receiver_first20ms_peak_N','receiver_first20ms_impulse_Ns','original_full_max_finger_peak_N','original_full_local_excess_Ns'):
  vals=[(r.get('metrics') or {}).get(k) for r in rows];known=[float(x) for x in vals if type(x) in (int,float) and np.isfinite(x)]
  stats[k]=dict(assigned=len(rows),available=len(known),missing=len(rows)-len(known),median=None if not known else float(np.median(known)),minimum=None if not known else min(known),maximum=None if not known else max(known),scope='Available observed values only; absent/unreached staysNULL,not0 or failed latency benefit')
 stats['handoff_passed']=tri([r.get('handoff_passed') for r in rows])
 stats['first_low_motion_with_physical_exposure']=sum((r.get('metrics') or {}).get('first_low_motion_found') is True and r['original_labels'].get('physical_release_exposure') is True for r in rows)
 stats['first_low_motion_with_task_qualified_exposure']=sum((r.get('metrics') or {}).get('first_low_motion_found') is True and r['original_labels'].get('task_qualified_exposure') is True for r in rows)
 return dict(assigned=len(rows),metrics=stats)
def collect(plan,markers):
 # Internal callers must obey the same barrier; never bypass it by importing.
 if not terminal_barrier(plan,markers)['ready_to_read_metrics']:raise ValueError('all13 terminal before any result/ledger read')
 if sha(SMALL_READER)!=SMALL_READER_SHA:raise ValueError('fixed small reader changed')
 small=load(SMALL_READER,'fixed_v20_small_helpers_for_complete16');paths,pins=small.frozen_dependencies();calc=load(paths['calculator'],'fixed_v3_for_v20_complete16');helper=load(paths['v18_reader'],'fixed_projection_for_v20_complete16')
 for pin in plan['evidence_pins'].values():equal(sha(ROOT/pin['path']),pin['sha256'],'retained evidence pin')
 smallplan=read(ROOT/plan['evidence_pins']['small_plan']['path']);oldsmall=read(ROOT/plan['evidence_pins']['small_auxiliary']['path']);gate=read(ROOT/plan['evidence_pins']['execution_gate']['path'])
 if not gate.get('passed') or not gate.get('all_assigned_checked') or gate['source_sha256']!=plan['source_sha256'] or oldsmall.get('complete') is not True or oldsmall.get('issues')!=[]:raise ValueError('passed initial4 and complete fixed small readout required')
 equal(plan['source_sha256'],smallplan['source_sha256'],'frozen160');equal(plan['environment'],smallplan['environment'],'same42environment');equal(calc.C and sha(paths['calculator']),plan['auxiliary_calculator_sha256'],'fixed v3 calculator');equal(sha(paths['contract']),plan['auxiliary_contract_sha256'],'fixed v3 contract')
 refs=small.reference_index(paths,plan,helper);rows=[];issues=[];inputs={};npz_reads=0
 def h(p):
  k=str(Path(p).resolve())
  if k not in inputs:inputs[k]=sha(p)
  return inputs[k]
 def row_from(run,item,retained=None):
  nonlocal npz_reads
  meta=read(run/'metadata.json');result=read(run/'result.json');scene,motion,seed=item['context_key'];ctx=(helper.scenario(meta['scenario']),meta['motion_mode'],meta['seed'])
  equal(ctx,(scene,motion,seed),'actual scene/name/motion/seed');equal((meta['method'],meta['task_policy'],meta['receiver_policy'],meta['split']),('ratio_effort','integrated','coordinated','robustness'),'actual fullV20 role');equal(meta['research_source_sha256'],plan['source_sha256'],'record160');equal(meta['trace_sha256'],result['trace_sha256'],'trace declared hash');equal(h(run/'trace.npz'),meta['trace_sha256'],'trace bytes')
  errors=[]
  if retained is not None:
   equal(result,retained['original_result'],'retained result unchanged');aux=deepcopy(retained['auxiliary']);m=deepcopy(retained['metrics']);status=retained['status']
  else:
   release=meta.get('mechanical_release_at',meta.get('release_at'));aux=m=None
   if release is None:status='no_actual_release_clock_auxiliary_unknown'
   else:
    try:
     with np.load(run/'delivery_intervals.npz',allow_pickle=False) as z:arrays={k:z[k] for k in calc.C['input_fields']};npz_reads+=1
     aux=calc.calculate(arrays,release_at_s=release,episode_end_s=meta['duration'],physical_period_s=meta['physics_timestep'],original_result=result);m=helper.metrics(aux);status=aux['status'];del arrays
    except (ValueError,TypeError,KeyError) as exc:status='auxiliary_unavailable';errors.append(type(exc).__name__+': '+str(exc))
  return dict(record_key=item['record_key'],record_id=item['record_id'],context_key=item['context_key'],origin=item['origin'],run=str(run),handoff_passed=meta['handoff_passed'],status=status,issues=errors,original_result=result,original_labels=labels(result),original_release_cost=small.original_release_cost(result),auxiliary=aux,metrics=m,physical_config=meta['config'],noise=meta['noise'],limits=meta['limits'],budget=meta['budget'],input_sha256={n:h(run/n) for n in ('metadata.json','result.json','trace.npz','delivery_intervals.npz')})
 roster={tuple(r['record_key']):r for r in plan['roster']}
 for retained in plan['retained']:
  for path,v in retained['files_sha256'].items():equal(h(ROOT/path),v,'retained3 file byte identity')
  equal(h(ROOT/retained['completion_path']),retained['completion_sha256'],'retained completion');equal(h(ROOT/retained['source_snapshot_manifest_path']),retained['source_snapshot_manifest_sha256'],'retained source snapshot')
  original=next(r for r in oldsmall['records'] if r['stage']==retained['original_auxiliary_stage']);equal(tuple(original['context_key']),tuple(retained['context_key']),'retained auxiliary context');rows.append(row_from(ROOT/retained['run'],roster[tuple(retained['record_key'])],original))
 for s in plan['stages']:
  dest=ROOT/s['output'];mark=markers[s['name']];ledger=read(dest/'assignment_ledger.json');trials=read(dest/'trials.json');manifest=read(dest/'manifest.json');reg=read(ROOT/s['registration']);a=s['assignment']
  equal(len(ledger),s['planned'],'complete assigned ledger');equal(len(trials),s['planned'],'complete trial results')
  for file,field in [('trials.json','trials_sha256'),('assignment_ledger.json','assignment_ledger_sha256'),('plan.json','plan_sha256'),('manifest.json','manifest_sha256')]:equal(h(dest/file),mark[field],'terminal '+file)
  equal(manifest['research_source_sha256'],plan['source_sha256'],'manifest160');equal(reg['source_sha256'],plan['source_sha256'],'registration160');equal(reg['assignment'],a,'registration assignment');equal(reg['environment'],plan['environment'],'registration42environment');equal(read(dest/'environment.json')['pinned_environment'],plan['environment'],'executed42environment')
  expected={(scene,a['motion_mode'],a['seeds'][0]) for scene in a['scenarios']};seen=set()
  for entry in ledger:
   equal(entry['status'],'evaluated','retained failed tasks still evaluated');j=entry['job'];context=(j['scenario'],j['motion_mode'],j['seed'])
   if context not in expected or context in seen:raise ValueError('missing/duplicate/wrong batch context')
   seen.add(context);rid=('full_V20',*context);item=roster[rid];run=Path(j['destination']).resolve();equal(run,(ROOT/item['run']).resolve(),'planned run directory');row=row_from(run,item);result=next(t for t in trials if (t['scenario'],t['method'],t['seed'])==(context[0],'ratio_effort',context[2]));equal(row['original_result'],result,'original result/trials exact');rows.append(row)
  equal(seen,expected,'all assigned stage rows')
 rows.sort(key=lambda r:tuple(r['record_key']));equal({tuple(r['record_key']) for r in rows},set(roster),'all16 unique')
 pairs=[];secondary=[]
 for row in rows:
  ctx=tuple(row['context_key']);row['secondary_reference_availability']={}
  for role in (*MAIN_ROLES,*SECONDARY):
   ref=refs.get((role,*ctx))
   if ref is None:
    if role in MAIN_ROLES:raise ValueError('missing historical primary context')
    row['secondary_reference_availability'][role]=dict(available=False,reason='not_executed_in_historical_three_context_cohort');continue
   rm=helper.metrics(ref['auxiliary']);paired=all(row[k]==ref[rk] for k,rk in [('physical_config','physical_config'),('noise','noise'),('limits','limits'),('budget','budget')]);pair=dict(record_key=row['record_key'],record_id=row['record_id'],context_key=row['context_key'],reference_role=role,reference_record_id=ref['record_id'],same_physical_noise_limits=paired,candidate_metrics=row['metrics'],reference_metrics=rm,original_label_differences=small.differences(row['original_labels'],labels(ref['auxiliary']['original_result']),paired),differences=small.differences(row['metrics'],rm,paired))
   (pairs if role in MAIN_ROLES else secondary).append(pair)
   if role in SECONDARY:row['secondary_reference_availability'][role]=dict(available=True,record_id=ref['record_id'])
   if not paired:issues.append(row['record_id']+'/'+role+'/unpaired_parameters')
  if row['issues']:issues.append(row['record_id']+'/auxiliary_unavailable')
 if len(rows)!=16 or len(pairs)!=48 or len(secondary)!=6:raise ValueError('exact16x3 primary and3x2 secondary required')
 strata={}
 for kind,idx in [('seed',2),('motion',1),('scene',0)]:strata[kind]={str(v):summarize([r for r in rows if r['context_key'][idx]==v]) for v in sorted({r['context_key'][idx] for r in rows})}
 paired_summary={}
 for role in MAIN_ROLES:
  ps=[p for p in pairs if p['reference_role']==role];metrics={}
  for k in sorted({k for p in ps for k in p['differences']}):
   vals=[p['differences'].get(k,{}) for p in ps];ds=[v['delta'] for v in vals if v.get('paired_available') and v.get('delta') is not None];rs=[v['relative_delta'] for v in vals if v.get('paired_available') and v.get('relative_delta') is not None]
   metrics[k]=dict(assigned_pairs=16,delta_available=len(ds),delta_median=None if not ds else float(np.median(ds)),relative_available=len(rs),relative_median=None if not rs else float(np.median(rs)),scope='Descriptive paired available values; failure/NULL denominator shown, no success-subset or weighted total')
  paired_summary[role]=metrics
 return dict(schema='v20_complete16_fixed_v3_auxiliary_v1',complete=not issues,status='all16_computed' if not issues else 'all16_retained_with_issues',created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(PLAN),source_sha256=plan['source_sha256'],environment=plan['environment'],calculator_sha256=pins['calculator'],contract_sha256=pins['contract'],frozen_small_reader_sha256=SMALL_READER_SHA,reader_sha256=sha(__file__),retained_count=3,new_count=13,combined_count=16,records=rows,primary_comparison_pairs=48,pairs=pairs,secondary_comparison_pairs=6,secondary_pairs=secondary,summary_all16=summarize(rows),summaries_by=strata,paired_summaries=paired_summary,issues=issues,input_sha256=inputs,completion_marker_sha256={s['name']:sha(ROOT/s['output']/'execution_complete.json') for s in plan['stages']},new_ledger_npz_open_count=npz_reads,retained3_auxiliary_reused_exact=True,physics_steps=0,definition_changed=False,original_metrics_changed=False,performance_scope='Registered development13 appended to seen initial3. Only2 physical seed clusters,16 contexts not independent iid repetitions. No independent confirmation, no global/safety/placement certificate, all original failure/unknown/NULL preserved.')
def text_report(x):
 lines=['# V20补齐13并合并16：固定辅助v3完整读数','', '新增四批全部终态后读取。旧3条原字节保留且不重跑，复用其冻结辅助字段；新13按同v3计算。两物理种子簇的16条件，不能当16个独立物理重复或未见确认。','', '|场景|运动|种子|来源|原任务|清离|首次驻留|末段驻留|曾floor|末线速m/s|末角速rad/s|20ms峰N|冲量Ns|全guard|','|---|---|---:|---|---|---|---|---|---|---:|---:|---:|---:|---|']
 def fmt(v):return '不可评/未达到' if v is None else '是' if v is True else '否' if v is False else f'{v:.6g}' if type(v) in (int,float) else str(v)
 for r in x['records']:
  m=r['metrics'] or {};c=r['context_key'];values=[*c,r['origin'],r['original_labels'].get('task_success'),r['original_labels'].get('physical_clearance_success')]+[m.get(k) for k in ['first_low_motion_found','final_low_motion_suffix_found','ever_floor','end_linear_speed_m_s','end_angular_speed_rad_s','receiver_first20ms_peak_N','receiver_first20ms_impulse_Ns']]+[r['original_labels'].get('full_guard_pass')];lines.append('|'+'|'.join(fmt(v) for v in values)+'|')
 lines+=['','主表全部16×V17/A/力比PI共48对存于JSON；V18/V19只在原3条件各有参考，共6辅助对，未执行部分明确不可用。固定first/final、暴露/既有支撑、floor、末速和冲击分别报告，不以无floor但未清手称放置。NULL不填0，未驻留不作0秒，条件可用延迟中位数不能替代全16分母。分seed/运动/场景和完整原始任务结果见JSON。没有新增成功阈值、权重或总分。']
 return '\n'.join(lines)+'\n'
def selftest():
 # Importing the preparer does not execute registration or simulator imports.
 prep=load(ROOT/'review/prepare_v20_complete16_v1.py','v20_complete16_roster_for_reader_test');stages=prep.descriptors();roster=[dict(record_key=['full_V20',s,m,se]) for s in SCENES for m in MOTIONS for se in SEEDS];p=dict(schema='v20_complete16_development_plan_v1',new_records=13,retained_count=3,combined_count=16,stages=stages,retained=[dict(record_key=['full_V20',*k]) for k in prep.RETAINED],roster=roster);markers={s['name']:dict(execution_complete=True,sources_unchanged=True,runtime_failures=[],planned=s['planned'],trials=s['planned']) for s in stages};tests=[terminal_barrier(p,markers)['ready_to_read_metrics']]
 for field,v in [('execution_complete',False),('sources_unchanged',False),('runtime_failures',[{'error':'x'}]),('trials',0)]:
  bad=deepcopy(markers);bad[stages[-1]['name']][field]=v;tests.append(not terminal_barrier(p,bad)['ready_to_read_metrics'])
 bad=deepcopy(markers);bad.pop(stages[0]['name']);tests.append(not terminal_barrier(p,bad)['ready_to_read_metrics'])
 try:collect(p,bad)
 except ValueError:tests.append(True)
 else:tests.append(False)
 rows=[dict(metrics=dict(first_low_motion_found=False,first_low_motion_latency_s=None,ever_floor=False),original_labels=dict(task_success=False),handoff_passed=False),dict(metrics=None,original_labels=dict(task_success=True),handoff_passed=True)];s=summarize(rows);tests.append(s['metrics']['first_low_motion_found']==dict(assigned=2,true=0,false=1,unknown=1) and s['metrics']['first_low_motion_latency_s']['median'] is None and s['metrics']['original_labels']['task_success']['false']==1)
 return dict(passed=all(tests),checks=len(tests),new_results_read=False,physics_steps=0,reader_sha256=sha(__file__))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--execute',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--out',required=True);ap.add_argument('--md');a=ap.parse_args();out=ROOT/a.out
 if out.exists() or a.md and (ROOT/a.md).exists():raise FileExistsError('fresh outputs required')
 if a.self_test:x=dict(schema='v20_complete16_auxiliary_software_v1',**selftest())
 else:
  barrier,p,markers=readiness()
  if a.prepare or not a.execute:x=dict(schema='v20_complete16_auxiliary_readiness_v1',**barrier,reader_sha256=sha(__file__),physics_steps=0)
  else:
   if not barrier['ready_to_read_metrics']:raise ValueError('all13 terminal before metrics:'+repr(barrier['pending']))
   x=collect(p,markers)
 with out.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 if a.execute and a.md:
  with (ROOT/a.md).open('x',encoding='utf8') as f:f.write(text_report(x))
 print(json.dumps(dict(path=str(out),sha256=sha(out),passed=x.get('passed'),complete=x.get('complete'),ready=x.get('ready_to_read_metrics'),physics_steps=0),ensure_ascii=False));return 1 if x.get('passed') is False or a.execute and x.get('complete') is not True else 0
if __name__=='__main__':raise SystemExit(main())

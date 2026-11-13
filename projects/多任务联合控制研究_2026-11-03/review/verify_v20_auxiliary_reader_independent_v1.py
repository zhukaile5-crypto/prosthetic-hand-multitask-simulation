"""Finite helper/old-data check only. Never read new V20 outcomes."""
from pathlib import Path
from unittest.mock import patch
from copy import deepcopy
import argparse,hashlib,importlib.util,json,sys,traceback
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
 if out.exists():raise FileExistsError(out)
 path=ROOT/'review/read_v20_small_auxiliary_v1.py';m=load(path,'v20_aux_narrow_independent');checks=[];reads=[];raw_reads=[]
 def ck(n,x,d=None):
  checks.append(dict(name=n,passed=bool(x),detail=d))
  if not x:raise AssertionError(n)
 def reject(fn):
  try:fn()
  except ValueError:return True
  return False
 try:
  ck('reviewed_reader_exact_author_source',sha(path)=='ce9f08bef02473e548cd859a8a75f094e4fc525b8db4c080f877cb2e7e6099ba')
  ass=dict(methods=['ratio_effort'],scenarios=['nominal'],seeds=[619741],motion_mode='transport',physical_randomization=True)
  fake=dict(planned=4,stages=[dict(name='test'+str(i),assignment=deepcopy(ass)) for i in range(4)]);good=dict(execution_complete=True,sources_unchanged=True,runtime_failures=[],trials=1,planned=1);markers={s['name']:deepcopy(good) for s in fake['stages']}
  negatives=[]
  for field,value in [('execution_complete',False),('sources_unchanged',False),('runtime_failures',[{'error':'saved_failure'}]),('trials',0),('planned',2)]:
   bad=deepcopy(markers);bad['test3'][field]=value;negatives.append(not m.terminal_barrier(fake,bad)['ready_to_read_metrics'])
  bad=deepcopy(markers);del bad['test3'];negatives.append(not m.terminal_barrier(fake,bad)['ready_to_read_metrics'])
  ck('all4_required_no_active_missing_failed_or_changed_marker',m.terminal_barrier(fake,markers)['ready_to_read_metrics'] and all(negatives))
  def no_collect(*a,**kw):reads.append('collect');raise AssertionError('pending must not collect')
  def no_npz(*a,**kw):raw_reads.append(str(a[0]));raise AssertionError('no raw/new results')
  with patch.object(m,'readiness',return_value=(dict(ready_to_read_metrics=False,pending=['test3']),fake,markers)),patch.object(m,'collect',side_effect=no_collect),patch.object(m.np,'load',side_effect=no_npz),patch.object(sys,'argv',['helper','--execute','--out','review/NEVER_CREATED_v20_aux_pending_test.json']):
   refused=reject(m.main)
  ck('actual_CLI_pending_path_refuses_before_collect_or_NPZ',refused and not reads and not raw_reads and not (ROOT/'review/NEVER_CREATED_v20_aux_pending_test.json').exists())
  paths,pins=m.frozen_dependencies();helper=m.load_module(paths['v18_reader'],'frozen_v18_projection_independent_v20');plan=json.loads((ROOT/'review/v20_small_plan_v1.json').read_text('utf8'));original_read=m.read
  def bounded_read(p):
   q=Path(p).resolve()
   if q.is_relative_to(ROOT/'output') and '/v20_' in q.as_posix():raise AssertionError('new V20 output read forbidden')
   reads.append(str(q));return original_read(q)
  with patch.object(m,'read',side_effect=bounded_read),patch.object(m.np,'load',side_effect=no_npz):refs=m.reference_index(paths,plan,helper)
  contexts={m.context(s['assignment']) for s in plan['stages'] if s['assignment']['task_policy']=='integrated'};keys=[(role,*key) for key in contexts for role in m.ROLES]
  ck('all3contexts_x5_distinct_historical_roles_actual_identity',len(contexts)==3 and len(keys)==len(set(keys))==15 and all(k in refs for k in keys) and m.ROLES==('full_V17','holding_A','same_prior_legacy_ratio','full_V18','full_V19'),dict(contexts=sorted(contexts),reference_count=15))
  v19=original_read(ROOT/'review/v19_small_auxiliary_v1.json');pinned=m.V19_AUX_SHA=='272936e294d9690ddee2bb2f91f591fefd72aafa1f599beb2286c0ecda268c49' and sha(ROOT/'review/v19_small_auxiliary_v1.json')==m.V19_AUX_SHA
  realsha=m.sha
  def wrongsha(p):return '0'*64 if Path(p).resolve()==(ROOT/'review/v19_small_auxiliary_v1.json').resolve() else realsha(p)
  with patch.object(m,'sha',side_effect=wrongsha),patch.object(m,'read',side_effect=bounded_read):refused=reject(lambda:m.reference_index(paths,plan,helper))
  ck('V19_fifth_reference_actual_frozen_aux_source_and_bad_SHA_rejected',pinned and refused and all(plan['source_sha256'][k]==v for k,v in v19['source_sha256'].items()))
  ck('oldV19_all4_metric_projection_exact_no_new_definition',all(helper.metrics(r['auxiliary'])==r['metrics'] for r in v19['records']) and m.CALC_SHA==pins['calculator'] and m.CONTRACT_SHA==pins['contract'])
  d=m.differences({'null':None,'false':False,'zero':0.},{'null':False,'false':True,'zero':0.},True);unpaired=m.differences({'x':1.},{'x':2.},False)
  ck('NULL_failed_boolean_true_zero_and_unpaired_not_collapsed',d['null']['paired_available'] is False and d['null']['delta'] is None and d['false']['candidate'] is False and d['false']['delta']==-1. and d['zero']['delta']==0. and d['zero']['relative_delta'] is None and unpaired['x']['delta'] is None)
  mm=dict(original_task_success=False,original_200Hz_clearance_latency_s=None,first_low_motion_found=False,final_low_motion_suffix_found=False,ever_floor=True,end_linear_speed_m_s=0.,end_angular_speed_rad_s=None,receiver_first20ms_peak_N=0.,receiver_first20ms_impulse_Ns=None,original_full_guard_pass=False,original_full_max_finger_peak_N=0.,original_full_local_excess_Ns=0.)
  payload=dict(passive_auxiliary_equivalence=dict(all_non_original_auxiliary_fields_exact=True),records=[dict(stage='synthetic',metrics=mm)],pairs=[dict(stage='synthetic',reference_role=role,reference_metrics=deepcopy(mm)) for role in m.ROLES]);text=m.text_report(payload,ROOT/'review/v19_small_auxiliary_v1.json')
  ck('table_keeps_candidate_and_all5_roles_failure_NULL_zero',len([x for x in text.splitlines() if x.startswith('|synthetic/')])==6 and '|synthetic/新V20|否|不可评/未达到|否|否|是|0|不可评/未达到|0|不可评/未达到|否|0|0|' in text)
  ck('newV20_outcomes_and_raw_never_read_reader_unchanged',not raw_reads and sha(path)=='ce9f08bef02473e548cd859a8a75f094e4fc525b8db4c080f877cb2e7e6099ba')
  result=dict(passed=True)
 except Exception as exc:result=dict(passed=False,error=repr(exc),traceback=traceback.format_exc())
 result.update(schema='v20_auxiliary_reader_independent_v1',checks=checks,reader_sha256=sha(path),checker_sha256=sha(__file__),old_reference_json_reads=reads,npz_reads=raw_reads,new_V20_results_read=False,physics_steps=0,scope='Only finite pure terminal/NULL/table negatives and actual frozen historical reference JSON/hash checks. Does not audit future V20 execution or recompute auxiliary definitions.')
 with out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(passed=result['passed'],checks=len(checks),out=str(out),sha256=sha(out)),ensure_ascii=False));print(result.get('traceback',''));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

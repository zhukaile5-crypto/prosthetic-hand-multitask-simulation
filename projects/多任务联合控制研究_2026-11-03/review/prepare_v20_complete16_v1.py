"""Register remaining13/retain initial3; never dispatch physics."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
SCENES=('nominal','friction_drop','fragile_conflict','collision');SEEDS=(619741,619742);MOTIONS=('hold','transport')
MAIN=ROOT/'review/v20_complete16_plan_v1.json';SMALL=ROOT/'review/v20_small_plan_v1.json';AUX=ROOT/'review/v20_small_auxiliary_v1.json'
RETAINED=(('nominal','transport',619741),('fragile_conflict','hold',619741),('friction_drop','transport',619742))
BATCHES=(('hold_seed619741','hold',619741,('nominal','friction_drop','collision')),('transport_seed619741','transport',619741,('friction_drop','fragile_conflict','collision')),('hold_seed619742','hold',619742,SCENES),('transport_seed619742','transport',619742,('nominal','fragile_conflict','collision')))
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def key(scene,motion,seed):return ['full_V20',scene,motion,int(seed)]
def descriptor(name,motion,seed,scenes):
 return dict(name=name,output='output/v20_complete16_'+name+'_v1',registration='review/v20_complete16_'+name+'_registration_v1.json',planned=len(scenes),assignment=dict(methods=['ratio_effort'],scenarios=list(scenes),seeds=[seed],motion_mode=motion,physical_randomization=True,task_policy='integrated',receiver_policy='coordinated'))
def descriptors():return [descriptor(*b) for b in BATCHES]
def check_execution_gate(gate,identity,small_plan_sha):
 if gate.get('passed') is not True or gate.get('all_assigned_checked') is not True or gate.get('assigned_count')!=4 or len(gate.get('rows',[]))!=4 or gate.get('issues')!=[] or gate.get('source_sha256')!=identity or gate.get('plan_sha256')!=small_plan_sha:
  raise ValueError('Actual passed all4 execution audit with exact160/initialplan required')
def command(stage):
 a=stage['assignment'];return 'python -B -m v20_contact_exit.run_development --out '+stage['output']+' --registration '+stage['registration']+' --task-policy integrated --receiver-policy coordinated --motion-mode '+a['motion_mode']+' --methods ratio_effort --scenarios '+' '.join(a['scenarios'])+' --seeds '+str(a['seeds'][0])+' --physical-randomization --workers 2'
def selftest():
 stages=descriptors();new=[(s,m,se) for _,m,se,ss in BATCHES for s in ss];universe={(s,m,se) for s in SCENES for m in MOTIONS for se in SEEDS};tests=[]
 tests.append(len(new)==len(set(new))==13 and set(new).isdisjoint(RETAINED) and set(new)|set(RETAINED)==universe)
 tests.append([s['planned'] for s in stages]==[3,3,4,3] and all('--workers 2' in command(s) for s in stages))
 gate=dict(passed=True,all_assigned_checked=True,assigned_count=4,rows=[{}]*4,issues=[],source_sha256={'test':'sha'},plan_sha256='plan')
 check_execution_gate(gate,{'test':'sha'},'plan');tests.append(True)
 for k,v in [('passed',False),('all_assigned_checked',False),('source_sha256',{'test':'stale'}),('plan_sha256','stale'),('rows',[{}]*3),('issues',[{'error':'retained'}])]:
  bad=dict(gate);bad[k]=v
  try:check_execution_gate(bad,{'test':'sha'},'plan')
  except ValueError:tests.append(True)
  else:tests.append(False)
 return dict(passed=all(tests),checks=len(tests),physics_dispatched=False,actual_results_read=False,descriptors=stages,commands=[command(s) for s in stages],script_sha256=sha(__file__))
def prepare(execution_gate):
 from v20_contact_exit.runtime import source_identity
 from v13_a1.run_bridge import environment
 from research_imports import FOUNDATION
 small=read(SMALL);aux=read(AUX);identity=source_identity(['ratio_effort']);env=environment();gatepath=(ROOT/execution_gate).resolve();gate=read(gatepath)
 check_execution_gate(gate,identity,sha(SMALL))
 if len(identity)!=160 or identity!=small['source_sha256'] or env!=small['environment'] or len(env['native_binaries_sha256'])!=42:raise ValueError('Unchanged executed160 and same42 native environment required')
 if aux.get('complete') is not True or aux.get('issues')!=[] or aux.get('source_sha256')!=identity or aux['plan_sha256']!=sha(SMALL) or len(aux['records'])!=4:raise ValueError('Original complete four-row fixed-v3 readout required')
 parent=read(ROOT/'review/v19_small_plan_v1.json')['source_sha256'];nomotion=read(ROOT/'review/v19_motion_ablation_plan_v1.json')['source_sha256']
 if len(parent)!=152 or len(nomotion)!=156 or any(identity.get(k)!=v for k,v in parent.items()):raise ValueError('Parent152 identity changed')
 for k,v in nomotion.items():
  path=FOUNDATION/k[11:] if k.startswith('foundation/') else ROOT/k
  if sha(path)!=v:raise ValueError('Frozen NoMotion156 changed:'+k)
 retained=[]
 for row in aux['records']:
  if row['stage']=='passive_ratio_transport':continue
  context=tuple(row['context_key'])
  if context not in RETAINED or row['metrics']['original_physical_clearance_success'] is not True:raise ValueError('Retained3 must be exact initial completed clearance-restored contexts')
  stage=next(s for s in small['stages'] if s['name']==row['stage']);dest=ROOT/stage['output'];done=read(dest/'execution_complete.json');run=Path(row['run']).resolve()
  if not done['execution_complete'] or not done['sources_unchanged'] or not done['planned']==done['trials']==1 or done['runtime_failures']:raise ValueError('Retained stage not complete')
  if read(run/'metadata.json')['research_source_sha256']!=identity:raise ValueError('Retained source identity')
  for n,v in row['input_sha256'].items():
   if sha(run/n)!=v:raise ValueError('Retained input changed:'+n)
  files={f.relative_to(ROOT).as_posix():sha(f) for f in sorted(run.rglob('*')) if f.is_file()}
  retained.append(dict(record_key=key(*context),record_id=json.dumps(key(*context),ensure_ascii=False,separators=(',',':')),context_key=list(context),stage=stage['name'],run=run.relative_to(ROOT).as_posix(),original_stage_output=stage['output'],source_count=160,retained_before_expansion=True,seen_development=True,rerun=False,files_sha256=files,completion_path=(dest/'execution_complete.json').relative_to(ROOT).as_posix(),completion_sha256=sha(dest/'execution_complete.json'),source_snapshot_manifest_path=(dest/'source_snapshot_manifest.json').relative_to(ROOT).as_posix(),source_snapshot_manifest_sha256=sha(dest/'source_snapshot_manifest.json'),original_auxiliary_stage=row['stage']))
 if {tuple(r['context_key']) for r in retained}!=set(RETAINED):raise ValueError('Exactly initial3 retained')
 stages=descriptors();roster=[]
 for r in retained:roster.append(dict(record_key=r['record_key'],record_id=r['record_id'],origin='retained_initial3',run=r['run'],context_key=r['context_key']))
 for s in stages:
  a=s['assignment']
  for scene in a['scenarios']:
   context=(scene,a['motion_mode'],a['seeds'][0]);rid=key(*context);roster.append(dict(record_key=rid,record_id=json.dumps(rid,ensure_ascii=False,separators=(',',':')),origin='new_remaining13',stage=s['name'],run=s['output']+'/runs/'+scene+'__ratio_effort__'+str(a['seeds'][0]),context_key=list(context)))
 if len(roster)!=16 or len({tuple(x['record_key']) for x in roster})!=16:raise ValueError('Complete unique16 roster required')
 targets=[ROOT/s['registration'] for s in stages]+[MAIN]
 if any(t.exists() for t in targets) or any((ROOT/s['output']).exists() for s in stages):raise FileExistsError('Fresh4 outputs/registrations/master required; retain all earlier attempts')
 now=datetime.now(timezone.utc).isoformat();pins=dict(execution_gate=dict(path=gatepath.relative_to(ROOT).as_posix(),sha256=sha(gatepath)),small_plan=dict(path=SMALL.relative_to(ROOT).as_posix(),sha256=sha(SMALL)),small_auxiliary=dict(path=AUX.relative_to(ROOT).as_posix(),sha256=sha(AUX)))
 contract=ROOT/'review/V20整体接触退出实施合同_v1.md';regs=[]
 for s in stages:regs.append(dict(schema='v20_complete16_remaining13_registration_v1',created_utc=now,source_sha256=identity,environment=env,assignment=s['assignment'],software_gate_passed=True,execution_gate=pins['execution_gate'],implementation_contract_sha256=sha(contract),scope='Fixed-version development expansion after seen initial3; no independent confirmation, no new score, no parameter selection',retained3_not_rerun=True,workers_root_dispatch=2))
 plan=dict(schema='v20_complete16_development_plan_v1',created_utc=now,source_sha256=identity,environment=env,planned=13,new_records=13,retained_count=3,combined_count=16,source_count=160,parent152_unchanged=True,noMotion156_unchanged=True,stages=stages,retained=retained,roster=sorted(roster,key=lambda x:tuple(x['record_key'])),evidence_pins=pins,physical_seed_clusters=list(SEEDS),scenes=list(SCENES),motion_modes=list(MOTIONS),main_reference_roles=['full_V17','holding_A','same_prior_legacy_ratio'],secondary_reference_roles=['full_V18','full_V19'],secondary_reference_scope='Only initial3 same contexts; unavailable elsewhere, never zero-filled',auxiliary_calculator_sha256=aux['calculator_sha256'],auxiliary_contract_sha256=aux['contract_sha256'],frozen_small_reader_sha256=sha(ROOT/'review/read_v20_small_auxiliary_v1.py'),implementation_contract_sha256=sha(contract),expansion_basis='Initial3 restored basic clearance and local force/impact benefits under frozen contract; final residence0/3 retained. Remaining13 not observed before registration.',initial3_are_not_prospective_unseen=True,remaining13_are_registered_development_not_confirmation=True,no_parameter_sweep=True,no_new_score_or_success_threshold=True,performance_claim_allowed=False,superiority_proven=False,physics_dispatched_by_preparer=False,workers_root_dispatch=2,commands=[command(s) for s in stages])
 written=[]
 try:
  for s,reg in zip(stages,regs):write(ROOT/s['registration'],reg);written.append(s['registration'])
  write(MAIN,plan);written.append(MAIN.relative_to(ROOT).as_posix())
 except Exception as exc:
  incident=ROOT/('review/v20_complete16_registration_incident_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');write(incident,dict(complete=False,physics_dispatched=False,written=written,error=repr(exc),traceback=traceback.format_exc()));raise
 return dict(registered_new=13,retained=3,combined=16,master=MAIN.relative_to(ROOT).as_posix(),master_sha256=sha(MAIN),physics_dispatched=False,stages=stages,commands=plan['commands'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--describe',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--execution-gate');ap.add_argument('--out');a=ap.parse_args()
 if a.prepare:
  if not a.execution_gate:raise ValueError('--execution-gate actual passed receipt required')
  x=prepare(a.execution_gate)
 elif a.self_test:x=selftest()
 else:x=dict(stages=descriptors(),commands=[command(s) for s in descriptors()],physics_dispatched=False,results_read=False)
 if a.out:write(ROOT/a.out,x)
 print(json.dumps(x,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())

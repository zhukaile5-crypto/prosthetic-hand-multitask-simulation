"""Fourteen independent full16s stage trials. Root alone schedules --execute."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,os,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from identification.release_basis_v2 import run_pilot as old
from identification.release_local_response_v2 import run_pilot as v2
from identification.release_local_response_v3.decoder import ORIGINS,VARIANTS,TRAIN,make_plant_class
from identification.release_local_response_v3.provenance import PACKAGE,identity,require_lock,snapshot,sha

PHYSICAL_FIELDS=v2.PHYSICAL_FIELDS
write=v2.write
save_diagnostics=v2.save_diagnostics
def validate_references():
 plan=json.loads((PACKAGE/'plan.json').read_text(encoding='utf-8'));r=plan['frozen_reference']
 files={ROOT/r['v2_output']/'summary.json':r['summary_sha256'],ROOT/'identification/release_local_response_v2/dependency_lock.json':r['v2_dependency_lock_sha256'],ROOT/r['v2_output']/'runs/capsule__zero/record/trace.npz':r['v2_zero_trace_sha256'],ROOT/'output/release_basis_minimal_v2/runs/capsule__native2d/record/trace.npz':r['old_native_trace_sha256']}
 if any(not p.is_file() or sha(p)!=h for p,h in files.items()):raise RuntimeError('Frozen reference identity changed; no execution')
 return {str(p):h for p,h in files.items()}

def make_session(stage,variant,source_identity):
 if stage not in ORIGINS or variant not in VARIANTS:raise ValueError('Registered stage and direction required')
 from runtime import ResearchSession
 from hand_plant import CommonPreparation
 from skin_geometry import DelayedSkinGeometry
 origin_tick=ORIGINS[stage];origin_time=(2400+origin_tick)*.005
 class LocalSession(ResearchSession):
  def metadata(self):
   value=super().metadata();value.update(release_local_response_revision='v3',release_local_response_stage=stage,release_local_response_variant=variant,release_local_response_role='train_axis' if variant in TRAIN else 'heldout_mixed',release_local_response_source_identity=source_identity,release_local_response_probe_origin_tick=origin_tick,release_local_response_probe_origin_s=origin_time,release_local_response_probe_entry_phase='before origin sensor/control action; previous held carrier phi is origin-minus5ms',release_local_response_full_input_control_s=origin_time+.02,release_local_response_first_full_input_physical_end_s=origin_time+.0205,release_local_response_clock='original control_steps5ms-held carrier; physical500us applications logged',release_response_model_available=False,release_force_safety_certified=False)
   return value
 session=LocalSession(method='ratio',scenario='nominal',shape='capsule',seed=610120,split='development')
 if session.config.mass!=.12 or session.config.friction!=.6 or session.config.timestep!=.0005 or session.config.control_dt!=.005 or session.release_driver is not None:raise ValueError('Registered native standard context changed')
 session.plant=make_plant_class(origin_tick)(session.config,variant)
 session.preparation=CommonPreparation(session.plant);session.last_truth=session.plant.truth();session.skin_geometry=DelayedSkinGeometry(session.plant);session.research_sources=dict(source_identity)
 return session

def prefix_checks(zero,branch,origin_s):
 differences=[]
 with np.load(zero,allow_pickle=False) as x,np.load(branch,allow_pickle=False) as y:
  for field in PHYSICAL_FIELDS+('observation_json',):
   clock=x['substep_time'] if field.startswith('substep_') else x['time'];mask=clock<=origin_s+1e-9
   xx,yy=x[field][mask],y[field][mask]
   equal=np.array_equal(xx,yy,equal_nan=True) if xx.dtype.kind in 'biufc' else np.array_equal(xx,yy)
   if not equal:differences.append(field)
 return dict(through_inclusive_s=origin_s,fields=len(PHYSICAL_FIELDS)+1,mismatches=differences,exact=not differences)

def compare_full_native(new,reference):
 detail={}
 with np.load(new,allow_pickle=False) as x,np.load(reference,allow_pickle=False) as y:
  for field in PHYSICAL_FIELDS:
   a,b=x[field],y[field];detail[field]=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
  observed=np.array_equal(x['observation_json'],y['observation_json'])
 return dict(reference=str(reference),reference_sha256=sha(reference),actual40fields_bitexact=detail,public_observations_exact=bool(observed),passed=all(detail.values()) and bool(observed))

def worker(job,source):
 target=Path(job['directory']);target.mkdir(parents=True,exist_ok=False);write(target/'job.json',job);s=None;entries={};began=time.perf_counter()
 try:
  if identity()!=source:raise RuntimeError('Source changed before construction')
  s=make_session(job['stage'],job['variant'],source)
  while s.running:
   for tick,name in [(2400,'release_entry'),(2400+ORIGINS[job['stage']],'probe_entry')]:
    if s.plant.control_steps==tick and name not in entries:
     directory=target/name;directory.mkdir();entries[name]=old.entry_snapshot(s,directory)
     p=s.plant;state=dict(probe_origin_tick=p.probe_origin_tick,release_tick=p.probe_release_tick,last_applied_tick=p.probe_last_tick,last_phi_s=p.probe_last_phi,last_lift_reference_m=p.probe_last_lift,last_command21=p.probe_last_command,total_variation21=p.probe_total_variation,native_reference_envelope=p.probe_envelope,release_decoder_state=None if p.probe_decoder is None else vars(p.probe_decoder))
     frozen=old.freeze(state);write(directory/'local_control_state.json',frozen);entries[name]['local_control_state_sha256']=old.digest(frozen)
   s.step_one_sensor_tick()
  if s.plant.time!=16. or len(entries)!=2:raise ValueError('Full16s and both stage entry snapshots required')
  s.save(target/'record')
  from evaluate_runs import evaluate_trace
  from evaluation import evaluate_research
  result=evaluate_trace(target/'record/trace.npz');result['research_metrics']=evaluate_research(target/'record/trace.npz');result.update(stage=job['stage'],variant=job['variant'],training_role=job['training_role'],force_response_model_available=False)
  write(target/'result.json',result);decoder=save_diagnostics(s,target)
  write(target/'pilot_metadata.json',dict(source_identity=source,stage=job['stage'],origin_relative_tick=ORIGINS[job['stage']],origin_control_time_s=job['origin_control_time_s'],entry=entries,config=vars(s.config),command_record_sha256=sha(target/'decoder_commands.npz'),robot_stream_sha256=sha(target/'robot_physical_stream.npz'),record_trace_sha256=sha(target/'record/trace.npz'),physical_execution_owned_by='root',no_hydration_used=True))
  if identity()!=source:raise RuntimeError('Source changed during execution')
  return dict(job_id=job['job_id'],stage=job['stage'],variant=job['variant'],status='complete',entries=entries,trace_sha256=sha(target/'record/trace.npz'),decoder=decoder,task_success=result['task_success'],guard_pass=result['research_metrics']['safety']['guard_pass'],release=result['research_metrics']['release'],elapsed_s=time.perf_counter()-began)
 except Exception as exc:
  fail=dict(job_id=job['job_id'],stage=job['stage'],variant=job['variant'],status='failed',error=repr(exc),traceback=traceback.format_exc(),entries=entries,elapsed_s=time.perf_counter()-began)
  if s is not None:
   fail['stopped_at_s']=s.plant.time
   try:
    partial=target/'partial_record';s.save(partial);fail['partial_record']=str(partial)
   except Exception as save_exc:fail['partial_save_error']=repr(save_exc)
   try:save_diagnostics(s,target)
   except Exception as save_exc:fail['diagnostic_save_error']=repr(save_exc)
  write(target/'failure.json',fail);return fail

def planned_jobs(out):
 return [dict(job_id='capsule__'+stage+'__'+variant,stage=stage,variant=variant,origin_relative_tick=tick,origin_control_time_s=(2400+tick)*.005,training_role='train' if variant in TRAIN else 'heldout_mixed',directory=str(out/'runs'/('capsule__'+stage+'__'+variant))) for stage,tick in ORIGINS.items() for variant in VARIANTS]

def preflight():
 import mujoco
 original={k:getattr(mujoco,k) for k in ('mj_step','mj_step1','mj_step2')};calls={k:0 for k in original}
 def block(name):
  def reject(*a,**kw):calls[name]+=1;raise RuntimeError('No dynamics in preflight: '+name)
  return reject
 for name in original:setattr(mujoco,name,block(name))
 try:
  references=validate_references();source=identity();reports=[]
  for job in planned_jobs(Path('not_executed')):
   s=make_session(job['stage'],job['variant'],source);p=s.plant;p.set_closure(np.full(5,.75));p.set_motion(.1,[0.,0.]);p.control_steps=2400;p.begin_release()
   for tick in range(2400,3201):p.control_steps=tick;p._apply_ctrl()
   if p.data.time!=0. or p.probe_mismatches or p.lift_ref!=0.:raise AssertionError('Zero-step actual-model decoder authority failed')
   reports.append(dict(stage=job['stage'],variant=job['variant'],applications=len(p.probe_rows),physical_time=float(p.data.time),source_count=len(source),final_lift_reference_m=p.lift_ref,final_phi_s=p.probe_last_phi))
  if identity()!=source:raise RuntimeError('Source changed during preflight')
  return dict(scope='actual model/plant constructors and command algebra only, no controller/sensor tick or simulation advance',passed=True,physics_steps=0,dynamics_guard_calls=calls,source_identity=source,frozen_reference_sha256=references,reports=reports)
 finally:
  for name,value in original.items():setattr(mujoco,name,value)

def summarize(out,results,source):
 bykey={(x['stage'],x['variant']):x for x in results};pairs=[];zeros={}
 for stage,tick in ORIGINS.items():
  zero=bykey.get((stage,'zero'));origin=(2400+tick)*.005
  if zero is None or zero['status']!='complete':
   for variant in VARIANTS:pairs.append(dict(stage=stage,variant=variant,paired=False,reason='stage zero incomplete; assigned attempts retained'))
   continue
  zp=out/'runs'/('capsule__'+stage+'__zero')
  comparisons={label:compare_full_native(zp/'record/trace.npz',reference) for label,reference in [('old_native',ROOT/'output/release_basis_minimal_v2/runs/capsule__native2d/record/trace.npz'),('frozen_v2_zero',ROOT/'output/release_local_response_capsule_v2/runs/capsule__zero/record/trace.npz')]}
  zeros[stage]=comparisons
  with np.load(zp/'decoder_commands.npz',allow_pickle=False) as z:carrier={k:z[k] for k in ['control_time','physical_time','actual_phi_s','actual_command21']}
  for variant in VARIANTS:
   row=bykey.get((stage,variant))
   if row is None or row['status']!='complete':pairs.append(dict(stage=stage,variant=variant,paired=False,reason='incomplete retained branch'));continue
   p=out/'runs'/('capsule__'+stage+'__'+variant);pref=prefix_checks(zp/'record/trace.npz',p/'record/trace.npz',origin);entry_equal=row['entries']==zero['entries']
   with np.load(p/'decoder_commands.npz',allow_pickle=False) as z:
    clocks=all(np.array_equal(z[k],carrier[k]) for k in ['control_time','physical_time']);phi=z['actual_phi_s']-carrier['actual_phi_s'];lift=z['actual_command21'][:,18]-carrier['actual_command21'][:,18]
    realized=np.array_equal(phi,z['actual_phi_offset_s']) and np.array_equal(lift,z['actual_lift_offset_m'])
    native_carrier=np.array_equal(z['native_carrier_command21'],carrier['actual_command21'])
   pairs.append(dict(stage=stage,variant=variant,paired=bool(pref['exact'] and entry_equal and clocks and realized and native_carrier),release_and_stage_entries_exact=entry_equal,prefix=pref,decoder_clocks_exact=clocks,actual_inputs_against_executed_zero=bool(realized),native_carrier_matches_executed_zero=native_carrier))
 cross=None
 if len(zeros)==2:
  a,b=ORIGINS.keys();cross=compare_full_native(out/'runs'/('capsule__'+a+'__zero')/'record/trace.npz',out/'runs'/('capsule__'+b+'__zero')/'record/trace.npz')
 failures=[x for x in results if x['status']!='complete'];unchanged=identity()==source
 passed=bool(len(results)==14 and not failures and unchanged and len(pairs)==14 and all(x['paired'] for x in pairs) and len(zeros)==2 and all(c['passed'] for v in zeros.values() for c in v.values()) and cross and cross['passed'])
 return dict(execution_complete=len(results)==14,completed=sum(x['status']=='complete' for x in results),planned=14,failures=failures,results=results,pairs=pairs,zero_full16s_comparisons=zeros,cross_stage_zero_full16s=cross,sources_unchanged=unchanged,passed=passed,passed_scope='protocol/paired identities only; force guard may fail and is retained',force_kernel_available=False,model_admitted=False,performance_claim_allowed=False)

def main():
 parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--pin-sources',action='store_true');group.add_argument('--preflight',action='store_true');group.add_argument('--execute',action='store_true');parser.add_argument('--out');parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
 references=validate_references()
 if args.pin_sources:
  data=dict(schema='release_local_response_v3_source_lock',sources=identity(),frozen_reference_sha256=references,environment=old.environment(),physics_started=False);write(PACKAGE/'dependency_lock.json',data);print(json.dumps(dict(pinned=len(data['sources']),lock_sha256=sha(PACKAGE/'dependency_lock.json'))));return 0
 if args.preflight:
  report=preflight();path=PACKAGE/f'preflight_{time.time_ns()}.json';write(path,report);print(json.dumps(dict(output=str(path),passed=True,physics_steps=0)));return 0
 if not args.out:parser.error('--execute requires fresh --out inside research/output')
 out=(ROOT/args.out).resolve();allowed=(ROOT/'output').resolve()
 if not out.is_relative_to(allowed) or out==allowed or out.exists():raise ValueError('Fresh strict descendant of research/output required')
 if not 1<=args.workers<=14:raise ValueError('workers1..14')
 lock=require_lock();out.mkdir(parents=True);snapshot(out/'sources',lock);jobs=planned_jobs(out)
 plan=json.loads((PACKAGE/'plan.json').read_text(encoding='utf-8'));environment=old.environment();write(out/'plan.json',plan);write(out/'environment.json',environment)
 write(out/'manifest.json',dict(schema='release_local_response_v3_fourteen_manifest',plan=plan,sources=lock['sources'],frozen_reference_sha256=references,environment=environment,jobs=jobs,lock_sha256=sha(PACKAGE/'dependency_lock.json'),source_snapshot_manifest_sha256=sha(out/'source_snapshot_manifest.json'),plan_sha256=sha(out/'plan.json'),physics_started=True,performance_claim_allowed=False))
 ledger=[dict(job=j,status='assigned') for j in jobs];write(out/'assignment_ledger.json',ledger);write(out/'results_partial.json',[]);write(out/'runtime_failures.json',[])
 results=[];os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';began=time.perf_counter()
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  futures={pool.submit(worker,j,lock['sources']):j for j in jobs}
  for future in as_completed(futures):
   job=futures[future]
   try:result=future.result()
   except Exception as exc:result=dict(job_id=job['job_id'],stage=job['stage'],variant=job['variant'],status='worker_exception',error=repr(exc))
   results.append(result);next(x for x in ledger if x['job']==job)['status']=result['status'];print(json.dumps(dict(completed=len(results),planned=14,stage=job['stage'],variant=job['variant'],status=result['status'])),flush=True)
   for name,value in [('assignment_ledger.json',ledger),('results_partial.json',results),('runtime_failures.json',[x for x in results if x['status']!='complete'])]:(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 final=summarize(out,results,lock['sources']);final['wall_seconds']=time.perf_counter()-began
 write(out/'results.json',results);write(out/'summary.json',final);print(json.dumps(dict(execution_complete=final['execution_complete'],passed=final['passed'],output=str(out)),ensure_ascii=False),flush=True);return 0 if final['passed'] else 2
if __name__=='__main__':raise SystemExit(main())

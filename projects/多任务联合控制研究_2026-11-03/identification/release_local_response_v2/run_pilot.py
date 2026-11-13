"""Seven full16s independent-prefix trials. Only root schedules --execute."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,os,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
# Importing the old runner establishes its original bare-module resolution;
# no old command path executes at import. Its audited full snapshot codec is reused.
from identification.release_basis_v2 import run_pilot as old
from identification.release_local_response_v2.decoder import VARIANTS,TRAIN,make_plant_class,coordinates,COMMAND_AXES
from identification.release_local_response_v2.provenance import PACKAGE,identity,require_lock,snapshot,sha

PHYSICAL_FIELDS=('time','normal','skin_normal','shear','palm','hand_other','all_hand_object','support','position','relative','rotation','relative_velocity','angular_velocity','lift','lift_ref','wrist_position','requested_wrist_reference','qpos','qvel','ctrl_before','ctrl','alpha_before','alpha','executed_action','warning_count','collision','actual_motion_target','executed_motion_reference','penetration','world_penetration','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm','substep_hand_other','substep_all_hand_object','substep_support','substep_penetration','substep_world_penetration')
def write(path,value):
 path=Path(path)
 with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def make_session(variant,source_identity):
 from runtime import ResearchSession
 from hand_plant import CommonPreparation
 from skin_geometry import DelayedSkinGeometry
 class LocalSession(ResearchSession):
  def metadata(self):
   value=super().metadata();value.update(release_local_response_variant=variant,release_local_response_role='train_axis' if variant in TRAIN else 'heldout_mixed',release_local_response_source_identity=source_identity,release_local_response_probe_origin_s=12.02,release_local_response_clock='original control_steps5ms-held carrier; physical500us applications logged',release_response_model_available=False,release_force_safety_certified=False)
   return value
 session=LocalSession(method='ratio',scenario='nominal',shape='capsule',seed=610120,split='development')
 if session.config.mass!=.12 or session.config.friction!=.6 or session.config.timestep!=.0005 or session.config.control_dt!=.005 or session.release_driver is not None:raise ValueError('Registered native standard context changed')
 session.plant=make_plant_class()(session.config,variant);session.preparation=CommonPreparation(session.plant);session.last_truth=session.plant.truth();session.skin_geometry=DelayedSkinGeometry(session.plant);session.research_sources=dict(source_identity)
 return session
def save_diagnostics(session,target):
 p=session.plant;nu=p.model.nu;rows=np.asarray(p.probe_rows,float)
 if not rows.size:rows=np.empty((0,55+2*nu))
 scalars=['control_time','physical_time','relative_control_tick','native_phi_s','actual_phi_s','lift_clock_s','ramp_weight','requested_phi_offset_s','requested_lift_offset_m','actual_phi_offset_s','actual_lift_offset_m','unclipped_lift_reference_m','lift_clipped']
 data={name:rows[:,i] for i,name in enumerate(scalars)}
 data.update(actual_command21=rows[:,13:34],native_carrier_command21=rows[:,34:55],actual_ctrl=rows[:,55:55+nu],native_carrier_ctrl=rows[:,55+nu:55+2*nu],command_axes=np.asarray(COMMAND_AXES))
 np.savez_compressed(target/'decoder_commands.npz',**data)
 robot=np.asarray(p.probe_robot_rows,float)
 if not robot.size:robot=np.empty((0,65))
 axes=[p.model.joint(int(j)).name for j in p.raw_joint_ids]+['transport_lift']
 np.savez_compressed(target/'robot_physical_stream.npz',physical_time_s=robot[:,0],position25=robot[:,1:26],velocity25=robot[:,26:51],target_finger_pressure5_N=robot[:,51:56],external_finger_pressure5_N=robot[:,56:61],palm_pressure_N=robot[:,61],other_hand_pressure_N=robot[:,62],all_hand_target_pressure_N=robot[:,63],table_support_N=robot[:,64],axes25=np.asarray(axes),units25=np.asarray(['rad']*24+['m']))
 report=dict(applications=len(rows),robot_physical_samples=len(robot),authority_mismatches=p.probe_mismatches,variant=p.probe_variant,requested_fraction=VARIANTS[p.probe_variant],complete_native_reference_envelope=None if p.probe_envelope is None else {k:v.tolist() for k,v in p.probe_envelope.items()},zero_full_ctrl_native_exact=bool(np.array_equal(data['actual_ctrl'],data['native_carrier_ctrl'])) if p.probe_variant=='zero' else None,force_kernel_available=False,force_safety_certified=False)
 write(target/'decoder_checks.json',report);return report
def prefix_checks(a,b,until=12.02):
 differences=[]
 with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
  for field in PHYSICAL_FIELDS+('observation_json',):
   clock=x['substep_time'] if field.startswith('substep_') else x['time'];mask=clock<until-1e-9
   xx,yy=x[field][mask],y[field][mask]
   equal=np.array_equal(xx,yy,equal_nan=True) if xx.dtype.kind in 'biufc' else np.array_equal(xx,yy)
   if not equal:differences.append(field)
 return dict(until_exclusive_s=until,fields=len(PHYSICAL_FIELDS)+1,mismatches=differences,exact=not differences)
def full_zero_comparison(new):
 old_path=ROOT/'output/release_basis_minimal_v2/runs/capsule__native2d/record/trace.npz';detail={}
 with np.load(new,allow_pickle=False) as x,np.load(old_path,allow_pickle=False) as y:
  for field in PHYSICAL_FIELDS:
   a,b=x[field],y[field];detail[field]=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
  obs_exact=np.array_equal(x['observation_json'],y['observation_json'])
 return dict(reference=str(old_path),reference_sha256=sha(old_path),actual40fields_bitexact=detail,public_observations_exact=bool(obs_exact),passed=all(detail.values()) and obs_exact)
def worker(job,source):
 target=Path(job['directory']);target.mkdir(parents=True,exist_ok=False);write(target/'job.json',job);s=None;entries={};began=time.perf_counter()
 try:
  if identity()!=source:raise RuntimeError('Source changed before construction')
  s=make_session(job['variant'],source)
  while s.running:
   for tick,name in [(2400,'release_entry'),(2404,'probe_entry')]:
    if s.plant.control_steps==tick and name not in entries:
     d=target/name;d.mkdir();entries[name]=old.entry_snapshot(s,d)
     p=s.plant;control_state=dict(release_tick=p.probe_release_tick,last_applied_tick=p.probe_last_tick,last_phi_s=p.probe_last_phi,last_lift_reference_m=p.probe_last_lift,last_command21=p.probe_last_command,total_variation21=p.probe_total_variation,native_reference_envelope=p.probe_envelope,release_decoder_state=None if p.probe_decoder is None else vars(p.probe_decoder))
     frozen=old.freeze(control_state);write(d/'local_control_state.json',frozen);entries[name]['local_control_state_sha256']=old.digest(frozen)
   s.step_one_sensor_tick()
  if s.plant.time!=16. or len(entries)!=2:raise ValueError('Full16s and both actual source snapshots required')
  s.save(target/'record');from evaluate_runs import evaluate_trace
  from evaluation import evaluate_research
  result=evaluate_trace(target/'record/trace.npz');result['research_metrics']=evaluate_research(target/'record/trace.npz');result.update(variant=job['variant'],training_role=job['training_role'],force_response_model_available=False)
  write(target/'result.json',result);decoder=save_diagnostics(s,target)
  write(target/'pilot_metadata.json',dict(source_identity=source,entry=entries,config=vars(s.config),command_record_sha256=sha(target/'decoder_commands.npz'),robot_stream_sha256=sha(target/'robot_physical_stream.npz'),record_trace_sha256=sha(target/'record/trace.npz'),physical_execution_owned_by='root',no_hydration_used=True))
  if identity()!=source:raise RuntimeError('Source changed during execution')
  return dict(job_id=job['job_id'],variant=job['variant'],status='complete',entries=entries,trace_sha256=sha(target/'record/trace.npz'),decoder=decoder,task_success=result['task_success'],guard_pass=result['research_metrics']['safety']['guard_pass'],release=result['research_metrics']['release'],elapsed_s=time.perf_counter()-began)
 except Exception as exc:
  fail=dict(job_id=job['job_id'],variant=job['variant'],status='failed',error=repr(exc),traceback=traceback.format_exc(),entries=entries,elapsed_s=time.perf_counter()-began)
  if s is not None:
   fail['stopped_at_s']=s.plant.time
   try:
    partial=target/'partial_record';s.save(partial);fail['partial_record']=str(partial)
   except Exception as save_exc:fail['partial_save_error']=repr(save_exc)
   try:save_diagnostics(s,target)
   except Exception as save_exc:fail['diagnostic_save_error']=repr(save_exc)
  write(target/'failure.json',fail);return fail
def preflight():
 source=identity();reports=[]
 for variant in VARIANTS:
  s=make_session(variant,source);p=s.plant;p.set_closure(np.full(5,.75));p.set_motion(.1,[0.,0.]);p.control_steps=2400;p.begin_release()
  for tick in range(2400,3201):p.control_steps=tick;p._apply_ctrl()
  if p.data.time!=0. or p.probe_mismatches:raise AssertionError('Zero-step real-model decoder authority failed')
  reports.append(dict(variant=variant,applications=len(p.probe_rows),physical_time=float(p.data.time),source_count=len(source)))
 return dict(scope='real robot/model constructor and actual control algebra only; no sensor sample/controller action/dynamics',passed=True,physics_steps=0,reports=reports)
def main():
 parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--pin-sources',action='store_true');group.add_argument('--preflight',action='store_true');group.add_argument('--execute',action='store_true');parser.add_argument('--out');parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
 if args.pin_sources:
  data=dict(schema='release_local_response_v2_source_lock',sources=identity(),environment=old.environment(),physics_started=False);write(PACKAGE/'dependency_lock.json',data);print(json.dumps(dict(pinned=len(data['sources']),lock_sha256=sha(PACKAGE/'dependency_lock.json'))));return 0
 if args.preflight:
  report=preflight();path=PACKAGE/f'preflight_{time.time_ns()}.json';write(path,report);print(json.dumps(dict(output=str(path),passed=True,physics_steps=0)));return 0
 if not args.out:parser.error('--execute requires fresh --out inside research/output')
 out=(ROOT/args.out).resolve();allowed=(ROOT/'output').resolve()
 if not out.is_relative_to(allowed) or out==allowed or out.exists():raise ValueError('Fresh strict descendant of research/output required')
 if not 1<=args.workers<=7:raise ValueError('workers1..7')
 lock=require_lock();out.mkdir(parents=True);snapshot(out/'sources',lock)
 jobs=[dict(job_id='capsule__'+v,variant=v,training_role='train' if v in TRAIN else 'heldout_mixed',directory=str(out/'runs'/('capsule__'+v))) for v in VARIANTS]
 plan=json.loads((PACKAGE/'plan.json').read_text(encoding='utf-8'));environment=old.environment();write(out/'plan.json',plan);write(out/'environment.json',environment)
 write(out/'manifest.json',dict(schema='release_local_response_v2_seven_manifest',plan=plan,sources=lock['sources'],environment=environment,jobs=jobs,lock_sha256=sha(PACKAGE/'dependency_lock.json'),source_snapshot_manifest_sha256=sha(out/'source_snapshot_manifest.json'),plan_sha256=sha(out/'plan.json'),physics_started=True,performance_claim_allowed=False));ledger=[dict(job=j,status='assigned') for j in jobs]
 write(out/'assignment_ledger.json',ledger);write(out/'results_partial.json',[]);write(out/'runtime_failures.json',[])
 results=[];os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';began=time.perf_counter()
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  futures={pool.submit(worker,j,lock['sources']):j for j in jobs}
  for future in as_completed(futures):
   job=futures[future]
   try:result=future.result()
   except Exception as exc:result=dict(job_id=job['job_id'],variant=job['variant'],status='worker_exception',error=repr(exc))
   results.append(result);next(x for x in ledger if x['job']==job)['status']=result['status'];write_progress=dict(completed=len(results),planned=7,variant=job['variant'],status=result['status']);print(json.dumps(write_progress),flush=True)
   for name,value in [('assignment_ledger.json',ledger),('results_partial.json',results),('runtime_failures.json',[x for x in results if x['status']!='complete'])]:
    (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 byvariant={x['variant']:x for x in results};pairs=[];zero=byvariant.get('zero');zero_full=None
 if zero and zero['status']=='complete':
  zp=out/'runs/capsule__zero';zero_full=full_zero_comparison(zp/'record/trace.npz')
  with np.load(zp/'decoder_commands.npz',allow_pickle=False) as z:zero_commands={k:z[k] for k in ['control_time','physical_time','actual_phi_s','actual_command21']}
  for variant in VARIANTS:
   row=byvariant[variant]
   if row['status']!='complete':pairs.append(dict(variant=variant,paired=False,reason='incomplete retained branch'));continue
   p=out/'runs'/('capsule__'+variant);pref=prefix_checks(zp/'record/trace.npz',p/'record/trace.npz');entry_equal=row['entries']==zero['entries']
   with np.load(p/'decoder_commands.npz',allow_pickle=False) as z:
    clock_equal=all(np.array_equal(z[k],zero_commands[k]) for k in ['control_time','physical_time']);real_phi=z['actual_phi_s']-zero_commands['actual_phi_s'];real_lift=z['actual_command21'][:,18]-zero_commands['actual_command21'][:,18]
    actual_inputs_match=np.array_equal(real_phi,z['actual_phi_offset_s']) and np.array_equal(real_lift,z['actual_lift_offset_m'])
   pairs.append(dict(variant=variant,paired=bool(pref['exact'] and entry_equal and clock_equal and actual_inputs_match),full_release_and_probe_entries_exact=entry_equal,prefix=pref,decoder_clocks_exact=clock_equal,actual_inputs_against_executed_zero=bool(actual_inputs_match)))
 unchanged=identity()==lock['sources'];failures=[x for x in results if x['status']!='complete']
 final=dict(execution_complete=len(results)==7,completed=sum(x['status']=='complete' for x in results),planned=7,failures=failures,results=results,pairs=pairs,zero_full16s_old_native_comparison=zero_full,sources_unchanged=unchanged,passed=bool(not failures and unchanged and zero_full and zero_full['passed'] and len(pairs)==7 and all(x['paired'] for x in pairs)),wall_seconds=time.perf_counter()-began,force_kernel_available=False,model_admitted=False,performance_claim_allowed=False)
 write(out/'results.json',results);write(out/'summary.json',final);print(json.dumps(dict(execution_complete=final['execution_complete'],passed=final['passed'],output=str(out)),ensure_ascii=False),flush=True);return 0 if final['passed'] else 2
if __name__=='__main__':raise SystemExit(main())

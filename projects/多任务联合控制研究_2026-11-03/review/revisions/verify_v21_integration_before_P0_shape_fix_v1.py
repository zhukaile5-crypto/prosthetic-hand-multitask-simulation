"""Finite source170/three-variant archived-public integration; no physics."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import fields
from copy import deepcopy
from unittest.mock import patch
import argparse,hashlib,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return plain(x.tolist())
 if isinstance(x,np.generic):return plain(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def bits(a,b):return np.asarray(a,float).shape==np.asarray(b,float).shape and np.asarray(a,float).tobytes()==np.asarray(b,float).tobytes()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=ROOT/a.out
 if out.exists():raise FileExistsError(out)
 checks=[];records=[];constructors=[];calls=[];truth=[];reset={};source={};parent={};inputs={};path_cases=[]
 def ck(n,x,detail=None):
  checks.append(dict(name=n,passed=bool(x),detail=plain(detail)))
  if not x:raise AssertionError(n+' '+str(detail))
 class Frame(SimpleNamespace):
  def __getattribute__(self,k):
   if k in {'object_position','object_velocity','object_pose','object_mass','object_friction','target_id','is_target','target_contact_id','true_qpos','raw_qpos'}:truth.append(k);raise AssertionError('truth:'+k)
   return super().__getattribute__(k)
 def deny(*a,**kw):calls.append('forbidden integration/forward/contact/writer');raise AssertionError(calls[-1])
 try:
  import research_imports,mujoco,hand_plant
  from foundation_config import FoundationConfig,OPEN
  from v13_a1.codec import ReferenceActor21,Reference21
  from v13_joint.lifecycle import ReleaseEntry
  from v15_support.receiver_sensor import ReceiverMount,ReceiverMeasurement
  from v16_release_geometry.transfer_observer import TransferObserver
  from v16b_release_goal.goal import release_clearance_goal
  from v14_coord.design import reference_rates
  from v20_contact_exit.adapter import ContactExitAdapter
  from v20_contact_exit.supervisor import ReleaseSupervisor as OldSupervisor
  from v20_contact_exit.runtime import ReleaseSession as OldSession
  from v21_stable_handoff.runtime import ReleaseSession,source_identity
  from v21_stable_handoff.supervisor import ReleaseSupervisor as FullSupervisor
  from v21_stable_handoff.prefix_supervisor import ReleaseSupervisor as PrefixSupervisor
  from v21_stable_handoff.path_memory import ExitPathMemory
  source=source_identity(['ratio_effort']);parent=read(ROOT/'review/v20_complete16_plan_v1.json')['source_sha256'];cross=read(ROOT/'review/v21_task_cross_independent_v2.json');zero=read(ROOT/'review/v21_constructor_author_v1.json');no=read(ROOT/'review/v19_motion_ablation_plan_v1.json')['source_sha256'];foundation=ROOT.parent/'五指策略接入_2026-11-03'
  sourcepath=lambda k:foundation/k[11:] if k.startswith('foundation/') else ROOT/k
  ck('source170_exact_parent160_NoMotion156_and_delivered_math',len(source)==170 and len(parent)==160 and all(source.get(k)==v for k,v in parent.items()) and all(sha(sourcepath(k))==v for k,v in no.items()) and zero['passed'] and zero['source_sha256']==source and cross['passed'] and all(source[k]==v for k,v in cross['source_sha256'].items()))
  inputs.update(contract=sha(ROOT/'review/V21稳定交接实施合同_v1.md'),math_gate=sha(ROOT/'review/v21_task_cross_independent_v2.json'),constructor_author=sha(ROOT/'review/v21_constructor_author_v1.json'))
  for n in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,n,deny)
  origf=mujoco.mj_forward;origw=ReferenceActor21.apply
  def fwd(*a,**kw):reset['forward']=reset.get('forward',0)+1;return origf(*a,**kw)
  def writer(*a,**kw):reset['actor.apply']=reset.get('actor.apply',0)+1;return origw(*a,**kw)
  mujoco.mj_forward=fwd;ReferenceActor21.apply=writer
  roles={'v20':'unchanged_original_V20','prefix':'original_V20_tasks_with_certified_prefix_commit','v21':'one_sided_normal_motion_budget_exit_path_certified_prefix'}
  for variant in roles:
   before=dict(reset);s=ReleaseSession(method='ratio_effort',release_variant=variant,scenario='nominal',seed=619741,split='robustness',motion_mode='transport',task_policy='integrated',receiver_policy='coordinated');meta=s.metadata()
   ck('zero_constructor_role/'+variant,s.tick==s.time==s.sensors.samples==0 and not s.trace and meta['release_variant']==variant and meta['decision_algorithm']==roles[variant] and meta['research_source_sha256']==source and meta['frozen_v20_source_sha256']==parent)
   constructors.append(dict(variant=variant,role=meta['decision_algorithm'],reset_calls={k:v-before.get(k,0) for k,v in reset.items()},time_s=s.time,tick=s.tick,samples=s.sensors.samples,execution_actor=meta['execution_actor']))
  del s
  for n in ('mj_forward','mj_forwardSkip','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
   if hasattr(mujoco,n):setattr(mujoco,n,deny)
  ReferenceActor21.apply=deny
  token=object()
  dispatch_fixture=ReleaseSession.__new__(ReleaseSession);dispatch_fixture.release_variant='v20'
  with patch.object(OldSession,'_lifecycle_step',return_value=token) as delegated:result=ReleaseSession._lifecycle_step(dispatch_fixture,SimpleNamespace())
  ck('v20_runtime_literally_delegates_old_lifecycle',result is token and delegated.call_count==1)
  model,_=hand_plant._model(FoundationConfig());actor=ReferenceActor21(model);lo=actor.ctrl_range21[:,0].copy();hi=actor.ctrl_range21[:,1].copy();lo[18]-=actor.lift_bias_m;hi[18]-=actor.lift_bias_m;lo[18]=max(0.,lo[18]);hi[18]=min(.1,hi[18])
  for j,n in ((19,'rh_WRJ2'),(20,'rh_WRJ1')):
   k=model.joint(n).id
   if model.jnt_limited[k]:lo[j]=max(lo[j],model.jnt_range[k,0]);hi[j]=min(hi[j],model.jnt_range[k,1])
  h=float(model.geom_size[model.geom('support').id,2]);mount=ReceiverMount('support',('floor','support_post'),np.array([0.,0.,h]),h,float(np.cos(np.pi/4)))
  run=ROOT/'output/v20_small_nominal_transport_v1/runs/nominal__ratio_effort__619741';rm=read(run/'metadata.json');inputs.update({str((run/n).relative_to(ROOT)):sha(run/n) for n in ('trace.npz','metadata.json')})
  with np.load(run/'trace.npz',allow_pickle=False) as z:times=z['time'];encoded=z['observation_json'];recv=z['receiver_measurement_json']
  rf={f.name for f in fields(ReceiverMeasurement)};packets={}
  for i in np.flatnonzero((times>=11.8-1e-9)&(times<=14.965+1e-9)):
   raw=json.loads(str(encoded[i]));f=Frame(**{k:np.asarray(v) if isinstance(v,list) and k not in ('skin_patches','robot_actuator_axes21') else v for k,v in raw.items()});rr=json.loads(str(recv[i]));packets[int(i)]=(f,ReceiverMeasurement(**{k:v for k,v in rr.items() if k in rf}))
  empty=next(i for i,(f,r) in packets.items() if f.time>=13.935-1e-9 and np.all(f.valid&f.distributed_valid) and not np.any(f.contact_present) and not f.skin_patches)
  positive=max(i for i,(f,r) in packets.items() if i<empty and np.any(f.contact_present) and f.skin_patches)
  selected={round(t/.005) for t in (12.02,13.93,13.935,13.96,14.965)}|{positive,empty}
  ad=ContactExitAdapter(model,open18=np.concatenate(OPEN),receiver_mount=mount);graphs={};classes={'v20':OldSupervisor,'prefix':PrefixSupervisor,'v21':FullSupervisor};firstempty=None
  for i,(f,receiver) in packets.items():
   ad.observe(f)
   if f.time<12.-1e-9:continue
   current=np.asarray(f.robot_reference21,float)
   if not graphs:
    entry=ReleaseEntry(12.,Reference21(current,12.),1);goal=release_clearance_goal(entry,preparation_open18=np.concatenate(OPEN),mechanical_lower21=lo,mechanical_upper21=hi);prior=ad.begin_release(12.)
    for variant,cls in classes.items():graphs[variant]=cls(entry,open18=goal.reference18,observer=TransferObserver(request_time_s=12.,receiver_half_size_xy_m=ad.receiver_half_size_xy_m,receiver_up_axis3=ad.receiver_up_axis3,prior=prior),robot_model=model,limits5=rm['limits'],budget=rm['budget'],force_margin=.02)
   transfer=ad.transfer_inputs(receiver)
   for g in graphs.values():g.observe(frame=f,transfer_inputs=transfer,current21=current)
   if i not in selected:continue
   outputs={}
   for variant,g in graphs.items():
    ref,debug=g.action(frame=f,adapter=ad,current21=current,lower21=lo,upper21=hi,generation=int(f.mode_generation));ev=g.decision_evidence[-1];outputs[variant]=ref.values.copy();rates=reference_rates('release');scale=rates*.005
    ck(f'{variant}/{f.time}/actual_domain_clock_bias',np.all(ref.values>=ev['actual_reference_lower21']) and np.all(ref.values<=ev['actual_reference_upper21']) and np.all(ref.values>=np.maximum(lo,current-scale)) and np.all(ref.values<=np.minimum(hi,current+scale)) and bits(actor.encode(Reference21(current,f.time)).values,f.robot_actuator_target21) and ev['mode_generation']==int(f.mode_generation))
    hr=ev['hierarchy'];accepted=ev['geometry_proposal_accepted'];extdim=None;prefix=None;fully=None
    if hr is not None and accepted:
     score=ev['emitted_hierarchy_score'];ck(f'{variant}/{f.time}/emitted_score_available',score['available'])
     if variant!='v20':
      prefix=hr['completed_prefix'];fully=hr['fully_converged'];certs=hr['certified_prefix'];ext=np.asarray(hr['diagnostics'].get('extended_normalized_x',hr['x']),float).copy();ext[:21]=(ref.values-current)/scale;extdim=len(ext);coord=np.zeros(extdim);coord[:21]=128*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(current),abs(ref.values)))/scale;okay=[]
      for cert,row in zip(certs,score['levels']):
       A=np.asarray(cert['matrix']);b=np.asarray(cert['target']);att=A@ext;res=att-b;dom=cert['hard_domain'];G=np.asarray(dom['normalized_inequality_matrix']);hh=np.asarray(dom['normalized_inequality_upper']);E=np.asarray(dom['fixed_equality_matrix']).reshape((-1,extdim));ff=np.asarray(dom['fixed_equality_target']);tol=cert['KKT']['coordinate_tolerance'];okay.append(bits(att,row['attained']) and bits(res,row['residual']) and bits(float(res@res),row['squared_loss']) and np.all(G@ext-hh<=tol+abs(G)@coord) and np.all(abs(E@ext-ff)<=tol+abs(E)@coord))
      ck(f'{variant}/{f.time}/independent_extended_prefix_attainment_domain',prefix==len(certs)>=1 and len(score['levels'])==prefix and all(okay) and bits(score['extended_emitted_x'],ext) and bits(score['normalized_coordinate_roundoff_bound'],coord))
      if 'second_stage_hard_matrix' in hr['diagnostics']:
       G=np.asarray(hr['diagnostics']['second_stage_hard_matrix']);hh=np.asarray(hr['diagnostics']['second_stage_hard_upper']);mag=np.max(abs(G),axis=1,initial=0.);nz=mag>0;Gn=G[nz]/mag[nz,None];un=np.linalg.norm(Gn,axis=1);Gn=Gn/un[:,None];hn=(hh[nz]/mag[nz])/un;tol=max(c['KKT']['coordinate_tolerance'] for c in certs)
       ck(f'{variant}/{f.time}/actual_eta_and_new_motion_domain',np.all(hh[~nz]>=0) and np.all(Gn@ext-hn<=tol+abs(Gn)@coord))
     native=g.native.servo.evaluate(np.asarray(f.robot_position25),np.asarray(f.robot_velocity25),actor.encode(ref).values,time_s=f.time);ck(f'{variant}/{f.time}/native_actual_value',native.status=='KNOWN_ROBOT_ACTUATION_ALGEBRA' and bits(native.predicted_effort21,ev['native_proposed']['predicted_effort21']))
    elif hr is not None:ck(f'{variant}/{f.time}/unavailable_honest_anchor',ev['fallback']=='legal_current_q_recovery_anchor_without_exit_certificate' and bits(ref.values,ev['feasible_recovery_anchor21']))
    g.reconcile(SimpleNamespace(executed=ref,execution_changed=False,stale_plan_replaced=False,mode_generation=int(f.mode_generation)));rc=g.decision_evidence[-1]['execution_receipt'];ck(f'{variant}/{f.time}/nonwriter_receipt_debt',bits(rc['actual_reference21'],ref.values) and rc['same_snapshot_reference_debt_m']==float(f.robot_position25[24])-ref.values[18] and rc['actual_task_interval_respected'])
    records.append(dict(variant=variant,time_s=f.time,phase=g.phase,accepted=accepted,status=None if hr is None else hr['status'],completed_prefix=prefix,fully_converged=fully,extended_dimension=extdim,error=ev['numerical_error']))
    if variant=='v21' and i==empty:
     path=ev['exit_path_memory'];capture=path['capture'];firstempty=f.time
     ck('real_firstP0_captures_incoming_committed_not_OPEN',path['capture_created'] and path['fresh_acquisition'] and bits(capture['actual_committed_reference21'],current) and bits(np.asarray(hr['diagnostics']['effective_posture_target_delta21'])[:18],np.zeros(18)) and not bits(current[:18],goal.reference18))
     ck('real_P0_direction_is_oldplane_plan_not_up_or_stop_claim',capture['lift_direction']==-1 and path['lift_plan_advance'] and capture['lift_endpoint_m']==lo[18] and path['reference_hold_is_not_actual_stopping'] and capture['source_plane_age_at_creation_s']<=.12)
     ck('reference_hold_keeps_actual_encoder_debt',bits(hr['diagnostics']['reference_debt21'],current-np.asarray(ev['contact_tasks']['encoder_reference21'])) and np.max(abs(np.asarray(f.robot_velocity25)))>0)
   if graphs['v20'].phase in ('SEARCH','CAPTURE'):ck('preexit_identical_reference/'+str(f.time),bits(outputs['v20'],outputs['prefix']) and bits(outputs['v20'],outputs['v21']))
  ck('three_variants_include_real_exit_and_prefix_semantics',all(any(r['variant']==v and r['accepted'] for r in records) for v in classes) and firstempty is not None)
  # Pure public path state sequence, independent numbers/no FK or solver.
  memory=ExitPathMemory();u=np.zeros(21);u[18]=.06;lower=np.full(21,-1.);upper=np.ones(21);lower[18]=0.;upper[18]=.1
  def pf(t,positive=False,aux=0.,auxvalid=True,ref=None):return Frame(time=t,sample_time=t-.01,age=.01,robot_control_time_s=t,robot_reference21=u.copy() if ref is None else ref.copy(),valid=np.ones(5,bool),distributed_valid=np.ones(5,bool),contact_present=np.array([positive,False,False,False,False]),auxiliary_pressure=aux,auxiliary_valid=auxvalid,auxiliary_age=.01)
  def tasks(t,pos=True,sign=1.):
   N=np.zeros((1 if pos else 0,21));N[:,18]=sign
   return dict(available=True,pressure_P_N=np.array([.7]) if pos else np.empty(0),Npx21=N,diagnostics=dict(partial_geometry=False,patches=[dict(acquisition_time_s=t-.01)] if pos else []))
  memory.update(pf(1.,True),tasks=tasks(1.),current21=u,lower21=lower,upper21=upper);v=u.copy();v[0]=.3;r=memory.update(pf(1.005,ref=v),tasks=tasks(1.005,False),current21=v,lower21=lower,upper21=upper)
  ck('pure_freshP0_actual_pose_episode_and_consensus',r['capture_created'] and r['capture']['episode']==1 and bits(r['path_reference21'],v) and r['requested_lift_direction']==-1)
  high=memory.update(pf(1.01,aux=.3,ref=v),tasks=tasks(1.01,False),current21=v,lower21=lower,upper21=upper);bad=memory.update(pf(1.015,auxvalid=False,ref=v),tasks=tasks(1.015,False),current21=v,lower21=lower,upper21=upper);ck('aux_high_and_invalid_pause_without_new_pose',not high['lift_plan_advance'] and not bad['lift_plan_advance'] and not high['capture_created'] and bits(bad['path_reference21'],v))
  late=memory.update(pf(1.16,ref=v),tasks=tasks(1.16,False),current21=v,lower21=lower,upper21=upper);ck('expired_source_plane_is_historical_plan_not_current_geometry',late['lift_plan_advance'] and late['capture']['source_plane_current_geometry_valid'] is False and late['capture']['source_plane_age_at_creation_s']<=.12)
  positive=memory.update(pf(1.165,True,ref=v),tasks=tasks(1.165,True,-1.),current21=v,lower21=lower,upper21=upper);new=v.copy();new[0]=.4;r=memory.update(pf(1.17,ref=new),tasks=tasks(1.17,False),current21=new,lower21=lower,upper21=upper);ck('recontact_pauses_and_nextP0_rebases_actual_not_oldfreeze',not positive['lift_plan_advance'] and positive['path_reference21'] is None and r['capture_created'] and r['capture']['episode']==2 and bits(r['path_reference21'],new) and r['requested_lift_direction']==1)
  unknown=ExitPathMemory();r=unknown.update(pf(2.),tasks=dict(available=False),current21=u,lower21=lower,upper21=upper);ck('unknown_not_known_empty_capture',not r['capture_created'] and r['capture'] is None and not r['lift_plan_advance'])
  expired=ExitPathMemory();expired.update(pf(3.,True),tasks=tasks(3.),current21=u,lower21=lower,upper21=upper);r=expired.update(pf(3.2),tasks=tasks(3.2,False),current21=u,lower21=lower,upper21=upper);ck('expired_plane_cannot_create_direction',r['capture_created'] and r['capture']['lift_direction'] is None and not r['lift_plan_advance'])
  ck('source170_after_unchanged_no_action_forbidden_calls',source==source_identity(['ratio_effort']) and not calls and not truth)
  result=dict(passed=True)
 except Exception as exc:result=dict(passed=False,error=repr(exc),traceback=traceback.format_exc())
 result.update(schema='v21_integration_independent_v1',checks=checks,records=records,constructors=constructors,constructor_original_reset_calls=reset,source_sha256=source,parent160_source_sha256=parent,input_sha256=inputs,checker_sha256=sha(__file__),forbidden_calls=calls,forbidden_truth_reads=truth,physics_steps=0,scope='Three variant zero constructors count original reset forward/write. Afterwards only archived V20 public frames/current robot FK/native fixed-joint algebra, no step/forward/contact/writer. Independent finite emitted21+slack prefix checks and public path fixtures; not closed-loop performance, placement/safety or realtime proof.')
 with out.open('x',encoding='utf8') as f:json.dump(plain(result),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(passed=result['passed'],checks=len(checks),records=len(records),out=str(out),sha256=sha(out)),ensure_ascii=False));print(result.get('traceback',''));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

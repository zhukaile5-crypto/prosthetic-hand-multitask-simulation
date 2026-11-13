"""Few real public packets through final V19; no physical integration.

Real Session constructors keep original reset forward/write and count them.
After constructors all global forward/contact/actor-write calls are forbidden.
Current robot-only FK and frozen native algebra are permitted, never mj_step.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import fields
from copy import deepcopy,copy
from unittest.mock import patch
import argparse,hashlib,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
EXPECTED={'supervisor.py':'b8400b9d6f5e3d44fe6ff7a6b8565b4e1ba8bc74e5265194dd2d6695874aff06','adapter.py':'963aca1e2ea5c4869c7d19586010175e30cbbc45192b05ef3b385ee15bf60fa0','native_branch.py':'10ed80e816d0d5c10d6bba4b814db26bcb8c3942aece1e64bd39b59974528cb0','task_qp.py':'197d565bc639fbf1a69a47d60f490143ca1e72c35358d9c81e39b7f2c6c4a978'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
 if out.exists():raise FileExistsError(out)
 checks=[];records=[];fixtures=[];forbidden=[];truth=[];constructors=[];initcalls={};sources={};inputs={}
 def ck(name,ok,detail=None):
  checks.append(dict(name=name,passed=bool(ok),detail=plain(detail)))
  if not ok:raise AssertionError(name+' '+str(detail))
 def rejected(fn):
  try:fn()
  except (ValueError,TypeError):return True
  return False
 class PublicFrame(SimpleNamespace):
  def __getattribute__(self,n):
   if n in {'object_position','object_velocity','object_pose','object_mass','object_friction','target_id','target_contact_id','is_target','raw_qpos','true_qpos'}:truth.append(n);raise AssertionError('external truth '+n)
   return super().__getattribute__(n)
 try:
  import research_imports,mujoco,hand_plant
  from foundation_config import FoundationConfig,OPEN
  from v19_contact_reference.runtime import ReleaseSession,source_identity
  from v18_effort_transfer.runtime import source_identity as parent_identity
  import v19_contact_reference.supervisor as sm
  from v19_contact_reference.supervisor import ReleaseSupervisor
  from v19_contact_reference.adapter import ContactMotionAdapter
  from v19_contact_reference.native_branch import NativeBranchBuilder21
  from v19_contact_reference.task_qp import TaskQPResult,solve_task_qp,evaluate_task
  from v13_a1.codec import ReferenceActor21,Reference21
  from v13_joint.lifecycle import ReleaseEntry
  from v16b_release_goal.goal import release_clearance_goal
  from v16_release_geometry.transfer_observer import TransferObserver
  from v15_support.receiver_sensor import ReceiverMeasurement,ReceiverMount
  from v14_coord.design import reference_rates
  from v11.robot_servo_effort import ACTUATORS21
  sources=source_identity(['ratio_effort']);parent=parent_identity(['ratio_effort']);old= json.loads((ROOT/'review/v18_small_plan_v1.json').read_text('utf-8'))['source_sha256']
  ck('final152_and_original144',len(sources)==152 and len(parent)==144 and parent==old and all(sources[k]==v for k,v in old.items()))
  ck('final_four_component_identities',all(sources['v19_contact_reference/'+k]==v for k,v in EXPECTED.items()))
  for n in ('mj_step','mj_step1','mj_step2'):
   def deny(*a,_n=n,**kw):forbidden.append(_n);raise AssertionError(_n)
   setattr(mujoco,n,deny)
  originals={n:getattr(mujoco,n) for n in ('mj_forward','mj_contactForce')};original_apply=ReferenceActor21.apply
  for n,fn in originals.items():
   def counted(*a,_n=n,_fn=fn,**kw):initcalls[_n]=initcalls.get(_n,0)+1;return _fn(*a,**kw)
   setattr(mujoco,n,counted)
  def counted_apply(*a,**kw):initcalls['actor.apply']=initcalls.get('actor.apply',0)+1;return original_apply(*a,**kw)
  ReferenceActor21.apply=counted_apply
  for method in ('ratio_effort','fixed_effort','fusion_effort'):
   before=dict(initcalls);s=ReleaseSession(method=method,scenario='nominal',seed=619741,split='robustness',motion_mode='hold',task_policy='integrated',receiver_policy='coordinated')
   ck('zero_constructor/'+method,s.time==0 and s.tick==0 and s.sensors.samples==0 and not s.trace and type(s.release_adapter) is ContactMotionAdapter)
   constructors.append(dict(method=method,time_s=s.time,tick=s.tick,sensor_samples=s.sensors.samples,original_reset_calls={k:v-before.get(k,0) for k,v in initcalls.items()}))
  del s
  for n in ('mj_forward','mj_forwardSkip','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
   if hasattr(mujoco,n):
    def deny(*a,_n=n,**kw):forbidden.append(_n);raise AssertionError(_n)
    setattr(mujoco,n,deny)
  def denywrite(*a,**kw):forbidden.append('actor.apply');raise AssertionError('writer')
  ReferenceActor21.apply=denywrite
  model,_=hand_plant._model(FoundationConfig());actor=ReferenceActor21(model);lo=actor.ctrl_range21[:,0].copy();hi=actor.ctrl_range21[:,1].copy();lo[18]-=actor.lift_bias_m;hi[18]-=actor.lift_bias_m;lo[18]=max(0.,lo[18]);hi[18]=min(.1,hi[18])
  for j,n in ((19,'rh_WRJ2'),(20,'rh_WRJ1')):
   k=model.joint(n).id
   if model.jnt_limited[k]:lo[j]=max(lo[j],model.jnt_range[k,0]);hi[j]=min(hi[j],model.jnt_range[k,1])
  hh=float(model.geom_size[model.geom('support').id,2]);mount=ReceiverMount('support',('floor','support_post'),np.array([0.,0.,hh]),hh,float(np.cos(np.pi/4)))
  run=ROOT/'output/v18_small_nominal_transport_v1/runs/nominal__ratio_effort__619741';meta=json.loads((run/'metadata.json').read_text('utf-8'));inputs={str((run/n).relative_to(ROOT)):sha(run/n) for n in ('trace.npz','metadata.json')}
  with np.load(run/'trace.npz',allow_pickle=False) as z:times=z['time'];obs=z['observation_json'];recv=z['receiver_measurement_json']
  rf={f.name for f in fields(ReceiverMeasurement)}
  def packet(i):
   f=PublicFrame(**json.loads(str(obs[i])))
   for k,v in list(vars(f).items()):
    if isinstance(v,list) and k not in ('skin_patches','robot_actuator_axes21'):setattr(f,k,np.asarray(v))
   r=json.loads(str(recv[i]));return f,ReceiverMeasurement(**{k:v for k,v in r.items() if k in rf})
  adapter=ContactMotionAdapter(model,open18=np.concatenate(OPEN),receiver_mount=mount);graph=None;saved={};target=(12.02,13.93,14.965);engines=[]
  for i in np.flatnonzero((times>=11.8-1e-9)&(times<=14.965+1e-9)):
   f,r=packet(int(i));adapter.observe(f)
   if f.time<12.-1e-9:continue
   current=np.asarray(f.robot_reference21,float);engines.append(actor.encode(Reference21(current,f.time)).values.tobytes()==np.asarray(f.robot_actuator_target21).tobytes())
   if graph is None:
    entry=ReleaseEntry(12.,Reference21(current,12.),1);goal=release_clearance_goal(entry,preparation_open18=np.concatenate(OPEN),mechanical_lower21=lo,mechanical_upper21=hi);prior=adapter.begin_release(12.)
    observer=TransferObserver(request_time_s=12.,receiver_half_size_xy_m=adapter.receiver_half_size_xy_m,receiver_up_axis3=adapter.receiver_up_axis3,prior=prior)
    graph=ReleaseSupervisor(entry,open18=goal.reference18,observer=observer,robot_model=model,limits5=meta['limits'],budget=meta['budget'],force_margin=.02)
   transfer=adapter.transfer_inputs(r);public=graph.observe(frame=f,transfer_inputs=transfer,current21=current)
   if not any(abs(f.time-t)<1e-9 for t in target):continue
   ref,debug=graph.action(frame=f,adapter=adapter,current21=current,lower21=lo,upper21=hi,generation=int(f.mode_generation));ev=graph.decision_evidence[-1];p=ev['qp_problem'];br=ev['branch'];a=ev['native_actual'];qpres=ev['qp'];rate=reference_rates('release')*.005
   ck('accepted_real_reference_effort_goal/'+str(f.time),ev['reference_effort_goal_accepted'],dict(error=ev['numerical_error'],qp=None if qpres is None else qpres['status']))
   ck('real5ms_and_task_domain/'+str(f.time),np.all(ref.values>=np.maximum(lo,current-rate)) and np.all(ref.values<=np.minimum(hi,current+rate)) and np.all(ref.values>=ev['actual_reference_lower21']) and np.all(ref.values<=ev['actual_reference_upper21']))
   ck('gamma_original_domain/'+str(f.time),ev['gamma_domain'][0]-1e-14<=qpres['gamma']<=ev['gamma_domain'][1]+1e-14)
   native=graph.branch_builder.servo.evaluate_full(f.robot_position25,f.robot_velocity25,actor.encode(ref).values,time_s=f.time);p0=graph.branch_builder.servo.evaluate_full(f.robot_position25,f.robot_velocity25,np.asarray(f.robot_actuator_target21),time_s=f.time)
   ck('fullF067_p0_and_final_effort/'+str(f.time),p0.predicted_effort21.tobytes()==np.asarray(p['p0_21']).tobytes() and native.predicted_effort21.tobytes()==np.asarray(a['native_effort21']).tobytes())
   du=ref.values-current;d=native.predicted_effort21-p0.predicted_effort21;z=np.asarray(qpres['zP']);g=qpres['gamma'];J=np.asarray(p['Jpx25']).reshape((-1,25));A=np.asarray(p['A21x25']);y=np.asarray(p['y_p']);sc={k:np.asarray(v) for k,v in p['scales'].items()};res=A.T@d-(J.T@(z-y)+np.asarray(p['tau_extra25']));rr=du-g*np.asarray(p['open_step21'])-np.asarray(p['motion_step21']);zz=z+g*p['unload_fraction']*y-y
   cost=float((sc['residual25']*res)@(sc['residual25']*res)+(sc['delta21']*d)@(sc['delta21']*d)+(sc['reference21']*rr)@(sc['reference21']*rr)+(sc['intentP']*zz)@(sc['intentP']*zz)+(sc['progress']*(g-1.))**2)
   ck('independent_complete_cost_and_canonical_residual/'+str(f.time),np.float64(cost).tobytes()==np.float64(ev['actual_full_task_score']['objective']).tobytes() and res.tobytes()==np.asarray(ev['actual_full_task_score']['residual25']).tobytes())
   ck('reference_and_offset_not_effort_variable/'+str(f.time),np.allclose(np.asarray(p['offset21'])+np.asarray(p['gain21'])*np.asarray(qpres['du21']),qpres['d21'],atol=0,rtol=0) and np.array_equal(br['current_reference21'],current))
   ck('roundoff_only_projection/'+str(f.time),np.all(abs(np.asarray(ev['rounding_projection21']))<=64*np.finfo(float).eps*np.maximum(1.,abs(ref.values))))
   graph.reconcile(SimpleNamespace(executed=ref,execution_changed=False,stale_plan_replaced=False));rc=graph.decision_evidence[-1]['execution_receipt'];ck('sameq_nonwriter_receipt/'+str(f.time),rc['same_snapshot_reference_debt_m']==float(f.robot_position25[24])-float(ref.values[18]) and rc['actual_task_interval_respected'])
   ci=ev['contact_intents'];expected_velocity=ev['governor']['governed_velocity_intent_m_s'] if graph.phase=='SEARCH' else 0.
   ck('stage_motion_target_not_zeroed_measurement/'+str(f.time),ci['declared_lift_velocity_target_m_s']==expected_velocity and ci['task_target_time_s']==f.time and (graph.phase!='SEARCH' or expected_velocity<0) and all(np.array_equal(np.asarray(x['measured_external_projection2_m_s'])-np.asarray(x['external_projection_target2_m_s']),x['external_projection_error2_m_s']) for x in ci['motion_records'] if x['external_projection_available']))
   records.append(dict(time_s=f.time,phase=graph.phase,gamma=g,qp=qpres['status'],accepted=True,patches=len(y),cost=cost,declared_lift_velocity_m_s=expected_velocity,reference_max_projection=float(np.max(abs(np.asarray(ev['rounding_projection21']))))))
   if abs(f.time-12.02)<1e-9:
    saved=dict(frame=deepcopy(f),decision=deepcopy(ev),reference=ref,current=current.copy())
    ck('duplicate_same_public_payload_idempotent',graph.observe(frame=f,transfer_inputs=transfer,current21=current)==public)
    bad=deepcopy(f);bad.robot_control_time_s-=.005;ck('wrong_engine_clock_rejected',rejected(lambda:graph.action(frame=bad,adapter=adapter,current21=current,lower21=lo,upper21=hi,generation=1)))
    with patch.object(sm,'solve_task_qp',return_value=TaskQPResult('unknown','independent_injection')):
     fallback,fd=graph.action(frame=f,adapter=adapter,current21=current,lower21=lo,upper21=hi,generation=1)
    ck('unresolved_solver_honest_legal_fallback',not fd['reference_effort_goal_accepted'] and fd['fallback']=='legal_governed_reference_progress_without_joint_task_certificate' and np.all(fallback.values>=np.maximum(lo,current-rate)) and np.all(fallback.values<=np.minimum(hi,current+rate)))
  ck('three_real_stages_and_public_engine_history',len(records)==3 and {r['phase'] for r in records}=={'SEARCH','TRANSFER','SEPARATE'} and all(engines))
  # Same real device geometry, with causal history replayed only once for this fixture.
  f=saved['frame'];ad=ContactMotionAdapter(model,open18=np.concatenate(OPEN),receiver_mount=mount)
  for i in np.flatnonzero((times>=11.8-1e-9)&(times<=f.time+1e-9)):ad.observe(packet(int(i))[0])
  base=ad.current_contact_intents(f);mr=base['diagnostics']['motion_records'];ck('real_coacquired_absolute_motion_present',bool(mr) and all(x['acquisition_s']==f.sample_time for x in mr))
  descend=ad.current_contact_intents(f,desired_lift_velocity_m_s=-.02)['diagnostics']
  ck('search_descend_vs_contact_zero_same_measured_external',all(np.array_equal(x['measured_external_projection2_m_s'],z['measured_external_projection2_m_s']) and np.array_equal(z['external_projection_target2_m_s'],[0.,0.]) and np.array_equal(np.asarray(x['measured_external_projection2_m_s'])-np.asarray(x['external_projection_target2_m_s']),x['external_projection_error2_m_s']) for x,z in zip(descend['motion_records'],mr) if x['external_projection_available']) and any(np.linalg.norm(x['external_projection_target2_m_s'])>0 for x in descend['motion_records']) and descend['task_target_time_s']==f.time)
  ck('absolute_projection_same_age_sum',all(np.array_equal(np.asarray(x['acquired_robot_projection2_m_s'])+np.asarray(x['relative_slip2_m_s']),x['measured_external_projection2_m_s']) for x in mr if x['external_projection_available']))
  cancel=deepcopy(f);cancel.slip_local=np.asarray(f.slip_local,float).copy()
  for x in mr:cancel.slip_local[x['digit']]=-np.asarray(x['acquired_robot_projection2_m_s'])
  cm=ad.current_contact_intents(cancel)['diagnostics'];ck('absolute_projection_cancellation_exact_zero',all(np.array_equal(x['measured_external_projection2_m_s'],[0.,0.]) for x in cm['motion_records'] if x['external_projection_available']))
  miss=deepcopy(f);miss.slip_valid5=np.zeros(5,bool);mm=ad.current_contact_intents(miss)
  ck('unknown_external_keeps_independent_robot_feedback',not mm['diagnostics']['external_projection_active_digits'] and bool(mm['diagnostics']['robot_motion_active_digits']) and all(x['measured_external_projection2_m_s'] is None for x in mm['diagnostics']['motion_records']) and np.array_equal(mm['J'],base['J']))
  split=deepcopy(f);idx=base['diagnostics']['patch_indices'][0];half=deepcopy(split.skin_patches[idx]);half['normal']/=2;split.skin_patches[idx]=half;split.skin_patches.insert(idx+1,deepcopy(half));sp=ad.current_contact_intents(split)
  ck('split_normal_and_both_motion_intents_invariant',np.allclose(sp['J'].T@sp['y'],base['J'].T@base['y'],rtol=2e-14,atol=2e-14) and np.allclose(sp['tau_motion25'],base['tau_motion25'],rtol=2e-14,atol=2e-14))
  # Small real-native adapter->QP->full-native tests, not another component sweep.
  ax=ACTUATORS21.index('rh_A_FFJ0');q=np.asarray(f.robot_position25);v=np.asarray(f.robot_velocity25)
  for label,cur,goal,lower,upper in [('nonflat',0.,.004,0.,.005),('flat',.25,.254,.25,.255),('offset_current_outside',0.,.25,.2,.3)]:
   m=copy(model);aid=m.actuator('rh_A_FFJ0').id;m.actuator_gainprm[aid,:]=0;m.actuator_gainprm[aid,0]=4.;m.actuator_biasprm[aid,:]=0;m.actuator_gaintype[aid]=mujoco.mjtGain.mjGAIN_FIXED;m.actuator_biastype[aid]=mujoco.mjtBias.mjBIAS_AFFINE;m.actuator_ctrlrange[aid]=[-.75,.75];m.actuator_ctrllimited[aid]=True;m.actuator_forcerange[aid]=[-.5,.5];m.actuator_forcelimited[aid]=True
   nb=NativeBranchBuilder21(m);u=saved['current'].copy();u[ax]=cur;pref=u.copy();pref[ax]=goal;l=u.copy();h=u.copy();l[ax]=lower;h[ax]=upper;br=nb.build(q25=q,v25=v,robot_axes25=nb.robot_axes25,time_s=f.time,current_reference=Reference21(u,f.time),actual_engine=nb.actor.encode(Reference21(u,f.time)),preferred_reference=Reference21(pref,f.time),reference_lower21=l,reference_upper21=h)
   ck('fixture_branch_available/'+label,br.available,br.reason)
   p=dict(p0_21=br.p0_effort21,A21x25=br.transmission21x25,gain21=br.gain21,offset21=br.offset21,Jpx25=np.empty((0,25)),y_p=np.empty(0),tau_extra25=np.zeros(25),delta_lower21=br.delta_lower21,delta_upper21=br.delta_upper21,open_step21=pref-u,motion_step21=np.zeros(21),unload_fraction=1.,patch_digit_p=np.empty(0,int),caps5=np.ones(5)*5,total_budget=18.,scales=dict(residual25=np.ones(25),delta21=np.ones(21),reference21=np.ones(21)*1000,intentP=np.empty(0),progress=1.),gamma_lower=1.,gamma_upper=1.)
   sol=solve_task_qp(**p);ac=sm.check_same_problem_candidate(p,sol);ck('fixture_same_goal_accepted/'+label,ac['accepted']);rr=Reference21(u+sol.du21,f.time);actual=nb.evaluate_reference(br,rr);ev=evaluate_task(p,rr.values-u,sol.zP,sol.gamma,actual_delta21=actual['actual_delta_effort21']);ck('fixture_reference_progress_and_final_score/'+label,actual['available'] and ev['available'] and ev['feasible'] and sol.du21[ax]>0 and abs(rr.values[ax]-goal)<2e-6)
   if label!='nonflat':ck('plateau_or_offset_preserves_raw_p0/'+label,br.gain21[ax]==0. and (label!='offset_current_outside' or (br.offset21[ax]==.5 and br.p0_effort21[ax]==0.)))
   fixtures.append(dict(label=label,gain=br.gain21[ax],offset=br.offset21[ax],p0=br.p0_effort21[ax],reference=rr.values[ax],target=goal,actual_delta=actual['actual_delta_effort21'][ax]))
  ck('no_forbidden_calls_or_truth_reads',not forbidden and not truth)
  ck('current152_sources_unchanged',sources==source_identity(['ratio_effort']))
  result=dict(passed=True)
 except Exception as exc:result=dict(passed=False,error=repr(exc),traceback=traceback.format_exc())
 result.update(schema='v19_integration_independent_v1',checks=checks,records=records,native_fixtures=fixtures,constructors=constructors,constructor_original_reset_calls=initcalls,source_sha256=sources,checker_sha256=sha(__file__),input_sha256=inputs,forbidden_calls=forbidden,forbidden_truth_reads=truth,physics_steps=0,scope='Three zero-tick real Sessions allow/count frozen reset initialization forward/write; afterwards global forward/contact/writer and all integration forbidden. One completed V18 public stream,3 off-policy release actions plus finite synthetic native/measurement cases. No closed-loop performance or force/placement certificate.')
 with out.open('x',encoding='utf-8') as h:json.dump(plain(result),h,ensure_ascii=False,indent=2,allow_nan=False);h.write('\n')
 print(json.dumps(dict(passed=result['passed'],checks=len(checks),records=len(records),path=str(out)),ensure_ascii=False));print(result.get('traceback',''));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

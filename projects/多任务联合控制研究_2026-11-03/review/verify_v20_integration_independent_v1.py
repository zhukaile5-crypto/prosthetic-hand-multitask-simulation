"""Finite independent V20 integration check; never integrate a plant.

Real zero-tick constructors retain/count original reset forward/write.
After construction: no forward/contact/writer/step, only public robot FK and
frozen native fixed-joint/tendon algebra. Forced phases are off-policy tests.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import fields
from copy import copy,deepcopy
from unittest.mock import patch
import argparse,hashlib,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
EXPECTED={'geometry.py':'f62a2b633f574d4997a724ee3017546a88b5896e6e12cfc832830eb19ae44b5f','adapter.py':'c31bd161b29940ac2741a7a7dc20f5dfa8739b54cb5d184f3177c72828606a3b','task_solver.py':'d395f6ab67695db7d4ac20650dc4b32e2cd0c8826eafed0baffd58708734be9f','supervisor.py':'99ad172af33b33e158d87123f5d15f98fd8b0c055667fb5d32bb5cbd0c31da67','runtime.py':'c5567dc6ce1a95eee4cbb3d7f40d83f66274b2baf31cb9ad7b25b2af47bc172b'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def bits(a,b):return np.asarray(a,float).shape==np.asarray(b,float).shape and np.asarray(a,float).tobytes()==np.asarray(b,float).tobytes()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=ROOT/args.out
 if out.exists():raise FileExistsError(out)
 checks=[];records=[];constructors=[];forbidden=[];truth=[];initcalls={};sources={};inputs={};parent={};no_motion={}
 def ck(n,x,detail=None):
  checks.append(dict(name=n,passed=bool(x),detail=plain(detail)))
  if not x:raise AssertionError(n+' '+str(detail))
 def reject(fn):
  try:fn()
  except (ValueError,TypeError):return True
  return False
 class Frame(SimpleNamespace):
  def __getattribute__(self,k):
   if k in {'object_position','object_velocity','object_pose','object_mass','object_friction','target_id','target_contact_id','is_target','true_qpos','raw_qpos'}:truth.append(k);raise AssertionError('externaltruth:'+k)
   return super().__getattribute__(k)
 try:
  import research_imports,mujoco,hand_plant
  from foundation_config import FoundationConfig,OPEN
  from v13_a1.codec import Reference21,ReferenceActor21
  from v13_joint.lifecycle import ReleaseEntry
  from v16b_release_goal.goal import release_clearance_goal
  from v16_release_geometry.transfer_observer import TransferObserver
  from v15_support.receiver_sensor import ReceiverMeasurement,ReceiverMount
  from v14_coord.design import reference_rates
  from v20_contact_exit.runtime import ReleaseSession,source_identity
  from v20_contact_exit.adapter import ContactExitAdapter
  from v20_contact_exit.supervisor import ReleaseSupervisor,contact_exit_lift_interval
  from v20_contact_exit.task_solver import solve_contact_exit
  from v17_contact_transfer.hierarchy import HierarchyResult
  import v20_contact_exit.supervisor as sm
  sources=source_identity(['ratio_effort']);parent=read(ROOT/'review/v19_small_plan_v1.json')['source_sha256'];no_motion=read(ROOT/'review/v19_motion_ablation_plan_v1.json')['source_sha256'];foundation=ROOT.parent/'五指策略接入_2026-11-03'
  def sourcepath(k):return foundation/k[11:] if k.startswith('foundation/') else ROOT/k
  ck('complete160_preserves_executed152_and_all_NoMotion156',len(sources)==160 and len(parent)==152 and len(no_motion)==156 and all(sources[k]==v for k,v in parent.items()) and all(sha(sourcepath(k))==v for k,v in no_motion.items()))
  ck('final_delivered_component_and_author_gate_pins',all(sources['v20_contact_exit/'+k]==v for k,v in EXPECTED.items()) and read(ROOT/'review/v20_geometry_adapter_author_v1.json')['passed'] and read(ROOT/'review/v20_task_solver_author_v1.json')['passed'])
  inputs['implementation_contract_sha256']=sha(ROOT/'review/V20整体接触退出实施合同_v1.md')
  inputs['geometry_adapter_author_gate_sha256']=sha(ROOT/'review/v20_geometry_adapter_author_v1.json');inputs['task_solver_author_gate_sha256']=sha(ROOT/'review/v20_task_solver_author_v1.json')
  for name in ('mj_step','mj_step1','mj_step2'):
   def deny(*a,_n=name,**kw):forbidden.append(_n);raise AssertionError(_n)
   setattr(mujoco,name,deny)
  original_forward=mujoco.mj_forward;original_apply=ReferenceActor21.apply
  def initforward(*a,**kw):initcalls['mj_forward']=initcalls.get('mj_forward',0)+1;return original_forward(*a,**kw)
  def initapply(*a,**kw):initcalls['actor.apply']=initcalls.get('actor.apply',0)+1;return original_apply(*a,**kw)
  mujoco.mj_forward=initforward;ReferenceActor21.apply=initapply
  for method in ('ratio_effort','fixed_effort','fusion_effort'):
   before=dict(initcalls);s=ReleaseSession(method=method,scenario='nominal',seed=619741,split='robustness',motion_mode='transport',task_policy='integrated',receiver_policy='coordinated');meta=s.metadata()
   ck('real_zero_constructor/'+method,s.time==s.tick==s.sensors.samples==0 and not s.trace and type(s.release_adapter) is ContactExitAdapter and meta['research_source_sha256']==sources and meta['frozen_v19_source_sha256']==parent and meta['v20_release_design']['placement_certified'] is False)
   constructors.append(dict(method=method,reset_calls={k:v-before.get(k,0) for k,v in initcalls.items()},ticks=s.tick,public_samples=s.sensors.samples))
  del s
  for name in ('mj_forward','mj_forwardSkip','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
   if hasattr(mujoco,name):
    def deny(*a,_n=name,**kw):forbidden.append(_n);raise AssertionError(_n)
    setattr(mujoco,name,deny)
  def no_write(*a,**kw):forbidden.append('actor.apply');raise AssertionError('writer')
  ReferenceActor21.apply=no_write
  model,_=hand_plant._model(FoundationConfig());actor=ReferenceActor21(model);lo=actor.ctrl_range21[:,0].copy();hi=actor.ctrl_range21[:,1].copy();lo[18]-=actor.lift_bias_m;hi[18]-=actor.lift_bias_m;lo[18]=max(0.,lo[18]);hi[18]=min(.1,hi[18])
  for j,n in ((19,'rh_WRJ2'),(20,'rh_WRJ1')):
   idx=model.joint(n).id
   if model.jnt_limited[idx]:lo[j]=max(lo[j],model.jnt_range[idx,0]);hi[j]=min(hi[j],model.jnt_range[idx,1])
  h=float(model.geom_size[model.geom('support').id,2]);mount=ReceiverMount('support',('floor','support_post'),np.array([0.,0.,h]),h,float(np.cos(np.pi/4)))
  run=ROOT/'output/v18_small_nominal_transport_v1/runs/nominal__ratio_effort__619741';meta=read(run/'metadata.json');inputs.update({str((run/n).relative_to(ROOT)):sha(run/n) for n in ('trace.npz','metadata.json')})
  with np.load(run/'trace.npz',allow_pickle=False) as z:times=z['time'];obs=z['observation_json'];rec=z['receiver_measurement_json']
  receiver_fields={f.name for f in fields(ReceiverMeasurement)}
  def packet(i):
   f=Frame(**json.loads(str(obs[i])))
   for k,v in list(vars(f).items()):
    if isinstance(v,list) and k not in ('skin_patches','robot_actuator_axes21'):setattr(f,k,np.asarray(v))
   r=json.loads(str(rec[i]));return f,ReceiverMeasurement(**{k:v for k,v in r.items() if k in receiver_fields})
  ad=ContactExitAdapter(model,open18=np.concatenate(OPEN),receiver_mount=mount);graph=None;target=(12.02,13.925,13.93,14.965)
  def action_check(g,f,label,forced=False):
   current=np.asarray(f.robot_reference21,float);ref,debug=g.action(frame=f,adapter=ad,current21=current,lower21=lo,upper21=hi,generation=int(f.mode_generation));e=g.decision_evidence[-1];L=np.asarray(e['actual_reference_lower21']);U=np.asarray(e['actual_reference_upper21']);rates=reference_rates('release')
   ck(label+'/actual5ms_authority_actor_bias',np.all(ref.values>=L) and np.all(ref.values<=U) and np.all(ref.values>=np.maximum(lo,current-rates*.005)) and np.all(ref.values<=np.minimum(hi,current+rates*.005)) and bits(actor.encode(Reference21(current,f.time)).values,f.robot_actuator_target21))
   exit_=g.phase in ('TRANSFER','SEPARATE','UNSUPPORTED_EXIT')
   if exit_:
    tasks=e['contact_tasks'];hr=e['hierarchy'];accepted=e['geometry_proposal_accepted']
    ck(label+'/known_geometry_gets_same_clock_exit_attempt',tasks is not None and tasks['available'] and hr is not None,dict(error=e['numerical_error'],status=None if hr is None else hr['status']))
    if accepted:
     d=hr['diagnostics'];x=(ref.values-current)/(rates*.005);rows=e['emitted_hierarchy_score']['levels'];independent=[]
     for level,row in zip(d['task_matrices_and_targets'],rows):
      A=np.asarray(level['matrix']);b=np.asarray(level['target']);att=A@x;res=att-b
      independent.append(bits(att,row['attained']) and bits(res,row['residual']) and bits(float(res@res),row['squared_loss']))
     state=g.native.servo.evaluate(np.asarray(f.robot_position25),np.asarray(f.robot_velocity25),actor.encode(ref).values,time_s=f.time)
     ck(label+'/every_actual_layer_rescored_and_full_native_value',all(independent) and state.status=='KNOWN_ROBOT_ACTUATION_ALGEBRA' and bits(state.predicted_effort21,e['native_proposed']['predicted_effort21']) and e['emitted_hierarchy_score']['available'])
     J=np.asarray(tasks['normal_jacobian_Px25']).reshape((-1,25));B=np.asarray(tasks['proposal_map25x21']);N=J@B;vl=J@np.asarray(f.robot_velocity25);debt=N@(current-np.asarray(tasks['encoder_reference21']));lower=L-current;upper=U-current;cap=np.maximum(0.,-np.sum(np.where(N>=0,N*lower,N*upper),axis=1))/.005;b=-.005*(cap+np.maximum(vl,0.)+np.maximum(debt/.02,0.))
     ck(label+'/independent_normal_target_preserves_current_origin',np.allclose(b,d['normal_target_b_P_m'],rtol=1e-13,atol=1e-15) and bits(d['current_reference21'],current) and bits(np.asarray(d['anchor_delta21']),np.clip(np.zeros(21),lower/(rates*.005),upper/(rates*.005))*(rates*.005)))
    else:ck(label+'/honest_failure_legal_anchor',e['fallback']=='legal_current_q_recovery_anchor_without_exit_certificate' and bits(ref.values,e['feasible_recovery_anchor21']))
   else:ck(label+'/frozen_search_capture_direct_governor',e['task_role']=='direct_frozen_search_capture_governor' and e['contact_tasks'] is None and e['hierarchy'] is None and e['fallback'] is None)
   g.reconcile(SimpleNamespace(executed=ref,execution_changed=False,stale_plan_replaced=False));rc=g.decision_evidence[-1]['execution_receipt'];ck(label+'/nonwriter_same_snapshot_receipt',bits(rc['actual_reference21'],ref.values) and rc['same_snapshot_reference_debt_m']==float(f.robot_position25[24])-ref.values[18] and rc['actual_task_interval_respected'])
   records.append(dict(label=label,time_s=f.time,phase=g.phase,forced_off_policy_phase=forced,pressure_rows=0 if e['contact_tasks'] is None else len(e['contact_tasks']['pressure_P_N']),accepted=e['geometry_proposal_accepted'],status=None if e['hierarchy'] is None else e['hierarchy']['status'],error=e['numerical_error']))
   return ref,e
  for i in np.flatnonzero((times>=11.8-1e-9)&(times<=14.965+1e-9)):
   f,r=packet(int(i));ad.observe(f)
   if f.time<12.-1e-9:continue
   current=np.asarray(f.robot_reference21,float)
   if graph is None:
    entry=ReleaseEntry(12.,Reference21(current,12.),1);goal=release_clearance_goal(entry,preparation_open18=np.concatenate(OPEN),mechanical_lower21=lo,mechanical_upper21=hi);prior=ad.begin_release(12.);observer=TransferObserver(request_time_s=12.,receiver_half_size_xy_m=ad.receiver_half_size_xy_m,receiver_up_axis3=ad.receiver_up_axis3,prior=prior);graph=ReleaseSupervisor(entry,open18=goal.reference18,observer=observer,robot_model=model,limits5=meta['limits'],budget=meta['budget'],force_margin=.02)
   transfer=ad.transfer_inputs(r);graph.observe(frame=f,transfer_inputs=transfer,current21=current)
   if not any(abs(f.time-t)<1e-9 for t in target):continue
   ref,e=action_check(graph,f,'real/'+str(f.time))
   if graph.phase=='TRANSFER':
    ck('TRANSFER_exit_starts_before_sampled_low_hand_clearance',not graph.was_clear and e['public_process']['contact_exit_task_active'] and len(e['contact_tasks']['pressure_P_N'])>0)
    ck('all_wrists_and_lift_real_permissions_not_entry_freeze',e['actual_reference_upper21'][18]>e['actual_reference_lower21'][18] and np.all(np.asarray(e['actual_reference_upper21'])[19:]>np.asarray(e['actual_reference_lower21'])[19:]) and bits(np.asarray(e['preferred_reference21'])[19:],current[19:]))
    forced=copy(graph);forced.decision_evidence=[];forced.public_state=deepcopy(graph.public_state);forced.phase='UNSUPPORTED_EXIT';forced.public_state['phase']='UNSUPPORTED_EXIT';action_check(forced,f,'forced_persistent_contact_exit',True)
    missing=deepcopy(f);missing.slip_valid5=np.zeros(5,bool);mt=ad.current_contact_tasks(missing,current21=current,reference_lower21=np.asarray(e['actual_reference_lower21']),reference_upper21=np.asarray(e['actual_reference_upper21']))
    ck('unknown_motion_omitted_without_erasing_known_normal_task',mt['available'] and len(mt['pressure_P_N'])>0 and mt['motion_matrix_Mx21'] is mt['motion_error_M_m_s'] is mt['motion_time_s'] is None)
    absent=deepcopy(f);absent.valid=np.zeros(5,bool);absent.distributed_valid=np.zeros(5,bool);ut=ad.current_contact_tasks(absent,current21=current,reference_lower21=lo,reference_upper21=hi);ck('unknown_all_finger_geometry_not_known_empty',not ut['available'] and ut['normal_jacobian_Px25'] is None)
    bad=deepcopy(f);bad.robot_control_time_s-=.005;ck('stale_command_clock_rejected',reject(lambda:graph.action(frame=bad,adapter=ad,current21=current,lower21=lo,upper21=hi,generation=int(f.mode_generation))))
    with patch.object(sm,'solve_contact_exit',return_value=HierarchyResult('unknown',np.full(21,np.nan),(),{})):
     fallback,db=graph.action(frame=f,adapter=ad,current21=current,lower21=lo,upper21=hi,generation=int(f.mode_generation))
    ck('unresolved_solver_honestly_returns_legal_recovery_not_certificate',not db['geometry_proposal_accepted'] and db['fallback']=='legal_current_q_recovery_anchor_without_exit_certificate' and bits(fallback.values,graph.decision_evidence[-1]['feasible_recovery_anchor21']))
  ck('four_real_phases_plus_forced_persistent_exit',len(records)==5 and {r['phase'] for r in records if not r['forced_off_policy_phase']}=={'SEARCH','CAPTURE','TRANSFER','SEPARATE'} and any(r['accepted'] and r['pressure_rows']>0 for r in records))
  # Independent finite oracle: original du0 illegal; normal first coordinate
  # must stay at legal minimum, motion uses orthogonal coordinate, canonical
  # goal cannot buy back either earlier attainment.
  H=np.zeros((25,21));H[:21]=np.eye(21);J=np.zeros((1,25));J[0,0]=1.;v=np.zeros(25);v[0]=.01;sl=np.full(21,-.005);su=np.full(21,.005);sl[0]=.001;su[0]=.002;M=np.zeros((1,21));M[0,1]=1.;goal=np.full(21,.003);goal[0]=-.004;goal[1]=-.004
  sol=solve_contact_exit(time_s=1.,dt_s=.005,current_reference21=np.zeros(21),encoder_reference21=np.zeros(21),step_lower21=sl,step_upper21=su,reference_rates21=np.ones(21),normal_jacobian_Px25=J,proposal_map25x21=H,robot_velocity25=v,pressure_P_N=np.ones(1),reference_target_delta21=goal,motion_matrix_Mx21=M,motion_error_M_m_s=np.array([-.02]),motion_time_s=1.)
  du=sol.x*.005;ck('nonzero_anchor_true_hierarchy_analytic_oracle',sol.available and not sol.diagnostics['zero_increment_in_actual_box'] and abs(du[0]-.001)<1e-12 and abs(du[1]-.0001)<1e-12 and np.allclose(du[2:],goal[2:],rtol=0,atol=1e-12),dict(status=sol.status,du=du[:3]))
  g=SimpleNamespace(diagnostics=dict(recovery_reference_budget_m=.001,downward_recovery_reference_budget_m=.001))
  low,high,d=contact_exit_lift_interval(g,q_m=.05,v_m_s=0.,tau_s=.02,actual_lower_m=.0495,actual_upper_m=.0505);ck('zero_velocity_exit_lift_box_does_not_collapse',low==.0495 and high==.0505 and d['physical_stop_certified'] is False)
  a,b,d=contact_exit_lift_interval(g,q_m=.05,v_m_s=-.1,tau_s=.02,actual_lower_m=.0495,actual_upper_m=.0505);aa,bb,_=contact_exit_lift_interval(g,q_m=.05,v_m_s=.1,tau_s=.02,actual_lower_m=.0495,actual_upper_m=.0505);ck('actual_overspeed_recovery_half_domains_both_directions',a==.05 and b==.0505 and aa==.0495 and bb==.05)
  a,b,d=contact_exit_lift_interval(g,q_m=.05,v_m_s=0.,tau_s=.02,actual_lower_m=.06,actual_upper_m=.061);ck('unreachable_recovery_is_legal_endpoint_not_false_mechanical_failure',a==b==.06 and not d['budget_band_reachable_this_tick'] and d['no_intersection_action']=='nearest_legal_current_q_recovery_endpoint')
  ck('sources_unchanged_and_no_action_pipeline_or_truth_calls',sources==source_identity(['ratio_effort']) and all(sha(sourcepath(k))==v for k,v in no_motion.items()) and not forbidden and not truth)
  result=dict(passed=True)
 except Exception as exc:result=dict(passed=False,error=repr(exc),traceback=traceback.format_exc())
 result.update(schema='v20_integration_independent_v1',checks=checks,records=records,constructors=constructors,constructor_original_reset_calls=initcalls,source_sha256=sources,parent152_source_sha256=parent,noMotion156_source_sha256=no_motion,input_sha256=inputs,checker_sha256=sha(__file__),forbidden_calls=forbidden,forbidden_truth_reads=truth,physics_steps=0,scope='Three real zero-tick Sessions allow/count original reset forward/write; later no step/forward/contact/writer. Four archived off-policy phases+one forced persistent-contact exit, finite hierarchy/interval oracles; no physical performance/safety/placement/stop certificate.')
 with out.open('x',encoding='utf8') as f:json.dump(plain(result),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(passed=result['passed'],checks=len(checks),records=len(records),out=str(out),sha256=sha(out)),ensure_ascii=False));print(result.get('traceback',''));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

from pathlib import Path
from copy import copy
from dataclasses import replace
import sys,json,hashlib,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
checks=[];cases={};forbidden=[]
def check(name,ok,detail=None):
 checks.append(dict(name=name,passed=bool(ok),detail=detail))
 if not ok:raise AssertionError(name+' '+str(detail))
def run():
 import research_imports,mujoco,hand_plant
 from foundation_config import FoundationConfig
 from v13_a1.codec import Reference21,ReferenceActor21
 from v14_coord.effort_realization import EffortGoalRealizer21
 for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_fwdPosition','mj_fwdVelocity','mj_kinematics','mj_collision','mj_contactForce'):
  if hasattr(mujoco,name):
   def deny(*a,_name=name,**k):forbidden.append(_name);raise AssertionError('Forbidden '+_name)
   setattr(mujoco,name,deny)
 ReferenceActor21.apply=lambda *a,**k:(_ for _ in ()).throw(AssertionError('writer forbidden'))
 model,_=hand_plant._model(FoundationConfig());axis=5
 proto=EffortGoalRealizer21(model);q=model.qpos0[proto.servo.qadr].copy();v=np.zeros(25);r=np.zeros(21);lo=r.copy();hi=r.copy();lo[axis]=-.2;hi[axis]=.2
 def domain(core=proto,current=None,nominal=None,**kwargs):
  cur=Reference21(r if current is None else current,1.);nom=cur if nominal is None else Reference21(nominal,1.)
  args=dict(q25=q,v25=v,robot_axes25=core.robot_axes25,time_s=1.,current_reference=cur,nominal_reference=nom,actual_engine=core.actor.encode(cur),reference_lower21=lo,reference_upper21=hi)
  args.update(kwargs);return core.build_domain(**args)
 def solve(d,core=proto,delta=None,correction=None,**kwargs):
  a=dict(p0_effort21=d.p0_effort21,delta_effort21=np.zeros(21) if delta is None else delta,correction25=np.zeros(25) if correction is None else correction,residual_scale25=np.ones(25),delta_scale21=np.ones(21));a.update(kwargs);return core.solve(d,**a)
 d=domain();check('known_actual_model_domain',d.available,d.reason)
 z=solve(d);check('zero_goal_holds_actual_reference',z.available and z.reference21.values.tobytes()==r.tobytes())
 delta=np.zeros(21);delta[axis]=.05
 t=time.perf_counter();x=solve(d,delta=delta,correction=np.ones(25)*100.);elapsed=(time.perf_counter()-t)*1000
 check('accepted_goal_not_second_force_optimization',x.available and abs(x.actual_delta_effort21[axis]-.05)<1e-13,x.reason)
 check('original_goal_preserved_bytes',x.requested_delta_effort21.tobytes()==delta.tobytes() and x.p0_effort21.tobytes()==d.p0_effort21.tobytes())
 full=proto.servo.evaluate_full(q,v,x.engine_port21.values,time_s=1.)
 check('combined_effort_original_full_bytes',full.predicted_effort21.tobytes()==x.native_effort21.tobytes())
 expected_residual=d.transmission21x25.T@(full.predicted_effort21-d.p0_effort21)-100.
 check('reported_residual_original_full_bytes',expected_residual.tobytes()==x.residual25.tobytes())
 check('declares_finite_family_and_no_safety',x.diagnostics['finite_family_only'] and not x.diagnostics['physical_safety_certified'] and not x.diagnostics['all_float_global_optimum_certified'])
 check('fixed_coordinates_hold',x.reference21.values[np.arange(21)!=axis].tobytes()==r[np.arange(21)!=axis].tobytes())
 n=r.copy();n[axis]=.3;shifted=domain(nominal=n);raw=proto.servo.evaluate(q,v,proto.actor.encode(Reference21(n,1.)).values)
 check('raw_nominal_p0_not_projected',shifted.available and shifted.p0_effort21.tobytes()==raw.predicted_effort21.tobytes())
 check('retargeted_p0_rejected',not solve(d,p0_effort21=np.nextafter(d.p0_effort21,np.inf)).available)
 check('foreign_domain_rejected',not solve(d,core=EffortGoalRealizer21(model)).available)
 check('tampered_domain_rejected',not solve(replace(d,reference_lower21=d.reference_lower21-.01)).available)
 check('invalid_scale_rejected',not solve(d,delta_scale21=np.zeros(21)).available)
 check('coupled_domain_unknown',not domain(coupled_reference_constraints={'A':np.ones((1,21)),'b':[0]}).available)
 for label in ('negative_gain','flat','force_platform','swallowed_delta'):
  m=copy(model);aid=int(proto.servo.aids[axis]);oid=int(proto.servo.outadr[axis]);cur=r.copy();nom=None;request=np.zeros(21)
  if label=='negative_gain':m.actuator_gainprm[aid,:3]=[-1.,0.,0.];request[axis]=.07
  if label=='flat':m.actuator_gainprm[aid,:3]=0.;m.actuator_biasprm[aid,:3]=[.7,0.,0.];request[axis]=1.
  if label=='force_platform':m.actuator_forcerange[oid]=[-.01,.01];cur[axis]=.15;nom=cur.copy();nom[axis]=.19
  if label=='swallowed_delta':m.actuator_gainprm[aid,:3]=[1.,0.,0.];m.actuator_biasprm[aid,:3]=[1e16,0.,0.];m.actuator_forcelimited[oid]=False;request[axis]=1.
  core=EffortGoalRealizer21(m);dom=domain(core,current=cur,nominal=nom);result=solve(dom,core,delta=request)
  check(label+'/known_value_and_full_combination',dom.available and result.available and result.diagnostics['prepared_value_final_full_recheck']['passed'],result.reason)
  if label=='negative_gain':check(label+'/goal_realized',abs(result.actual_delta_effort21[axis]-.07)<1e-13)
  if label=='flat':check(label+'/unreachable_goal_explicit',result.actual_delta_effort21[axis]==0. and result.diagnostics['actual_effort_minus_requested_delta21'][axis]==-1.)
  if label=='force_platform':check(label+'/equal_effort_closest_current_not_nominal',result.reference21.values.tobytes()==cur.tobytes())
  if label=='swallowed_delta':check(label+'/unrounded_goal_not_faked',result.requested_delta_effort21[axis]==1. and result.diagnostics['absolute_addition_lost_delta21'][axis]==1. and result.diagnostics['actual_effort_minus_requested_delta21'][axis]==-1.)
  cases[label]=result.to_dict()
 m=copy(model);guard=EffortGoalRealizer21(m);gd=domain(guard);m.actuator_group[int(guard.servo.aids[axis])]=5
 check('post_binding_group_mutation_rejected',not solve(gd,guard).available)
 check('zero_runtime_and_writer_calls',not forbidden and proto.actor.write_count==0)
 cases['accepted_goal']=x.to_dict()
 return dict(passed=True,checks=checks,cases=cases,sample_solve_ms=elapsed,physics_steps=0,forbidden_calls=forbidden,scope='Actual model plus isolated negative/flat/clip/rounding fixtures; goal-realization contract intentionally differs from V13 inverse force reoptimization')
if __name__=='__main__':
 out=ROOT/'review/v14_effort_goal_author_attempt1.json';assert not out.exists()
 try:report=run()
 except Exception as e:report=dict(passed=False,checks=checks,error=repr(e),traceback=traceback.format_exc(),physics_steps=0)
 report['source_sha256']=hashlib.sha256((ROOT/'v14_coord/effort_realization.py').read_bytes()).hexdigest()
 with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({k:report[k] for k in ('passed','source_sha256')},ensure_ascii=False));print(report.get('traceback',len(checks)))
 raise SystemExit(0 if report['passed'] else 1)

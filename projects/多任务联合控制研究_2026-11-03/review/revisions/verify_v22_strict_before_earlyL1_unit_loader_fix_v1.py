"""Independent V22 finite/task numerical gate; no engine construction/physics."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from unittest.mock import patch
import sys,json,hashlib,time,traceback,argparse
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'review'))
from v22_joint_task_fusion.numerics import ensure_numeric_environment,numerical_identity
ensure_numeric_environment()
import mujoco
forbidden=[]
def deny(*a,**kw):forbidden.append('engine_model_step_forward_contact_writer');raise AssertionError(forbidden[-1])
class NoModel:
 def __new__(cls,*a,**kw):return deny()
 from_xml_path=staticmethod(deny);from_xml_string=staticmethod(deny);from_binary_path=staticmethod(deny)
mujoco.MjModel=NoModel;mujoco.MjData=deny
for k in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
 if hasattr(mujoco,k):setattr(mujoco,k,deny)
from v13_a1.codec import ReferenceActor21
ReferenceActor21.apply=deny
from v21_original_matrix_kkt_verifier_v1 import verify_stage
from v22_joint_task_fusion import finite_solver as finite,task_solver as task
from v21_stable_handoff import task_solver as oldtask,prefix_hierarchy as oldkernel
from v21_stable_handoff.prefix_supervisor import score_emitted_reference
from v22_joint_task_fusion.runtime import source_identity
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if hasattr(x,'__dataclass_fields__'):return plain({k:getattr(x,k) for k in x.__dataclass_fields__})
 if isinstance(x,np.ndarray):return plain(x.tolist())
 if isinstance(x,np.generic):return plain(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [plain(v) for v in x]
 return x

def bits(a,b):return np.shape(a)==np.shape(b) and np.asarray(a,float).tobytes()==np.asarray(b,float).tobytes()
checks=[];rows=[];issues=[];synthetic={};source={};identity={};subset={}
def ck(name,okay,detail=None):
 checks.append(dict(name=name,passed=bool(okay),detail=plain(detail)))
 if not okay:issues.append(name)

def check_certificate(cert,problem):
 k=cert['KKT'];duals=dict(inequality_unit=k['full_normalized_inequality_dual'],equality_raw=k['equality_raw_dual'],provenance='new_finite_solver_witness_checked_independently_original_rows');v=verify_stage(problem,cert['x'],duals=duals,comparison_x=problem['initial_x']);hd=cert['hard_domain']
 exact=bits(cert['matrix'],problem['A']) and bits(cert['target'],problem['b']) and bits(hd['original_inequality_matrix_with_box'],problem['G']) and bits(hd['original_inequality_upper_with_box'],problem['h']) and bits(hd['fixed_equality_matrix'],problem['E']) and bits(hd['fixed_equality_target'],problem['f'])
 return exact,v

parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args();out=ROOT/args.out;assert not out.exists()
try:
 source=source_identity(['ratio_effort']);parent=json.loads((ROOT/'review/v21_small_plan_v1.json').read_text('utf8'))['source_sha256'];ck('source179_and_original170_exact',len(source)==179 and len(parent)==170 and all(source.get(k)==v for k,v in parent.items()))
 for name in ('numerics.py','finite_solver.py','task_solver.py','task_bundle.py'):subset['v22_joint_task_fusion/'+name]=sha(ROOT/'v22_joint_task_fusion'/name)
 identity=numerical_identity();ck('legacy42_and_new_dependency_identity',identity['original_legacy42_unchanged'] and len(identity['legacy_pinned_environment']['native_binaries_sha256'])==42 and len(identity['scipy_native_binaries_sha256'])==99)
 cp=ROOT/'review/v21_numerical_corpus_v1.json';corpus=json.loads(cp.read_text('utf8'));assert len(corpus['samples'])==65
 for sample in corpus['samples']:
  began=time.perf_counter();p=sample['failed_stage_problem'];lo=np.asarray(p['lower']);hi=np.asarray(p['upper']);n=len(lo);nf=int(np.sum(lo==hi));E=np.asarray(p['E']).reshape((-1,n));f=np.asarray(p['f']);ck('fixed_E_prefix/'+sample['sample_id'],bits(E[:nf],np.eye(n)[np.flatnonzero(lo==hi)]) and bits(f[:nf],lo[lo==hi]))
  result=finite.solve_hierarchy(lower=lo,upper=hi,levels=[finite.Level(p['layer_name'],np.asarray(p['A']),np.asarray(p['b']))],initial=np.asarray(p['initial_x']),inequality_matrix=np.asarray(p['input_inequality_matrix']),inequality_upper=np.asarray(p['input_inequality_upper']),equality_matrix=E[nf:],equality_target=f[nf:],max_iterations=p['max_iterations'],atol=p['atol'],rtol=p['rtol'])
  if result.available:
   cert=result.certified_prefix[-1];exact,v=check_certificate(cert,p);good=exact and v['kkt_passed'] and bits(result.x,cert['x']) and result.completed_prefix==1;ck('accepted_original_KKT/'+sample['sample_id'],good)
   ck('accepted_strict_original_box/'+sample['sample_id'],np.all(result.x>=lo) and np.all(result.x<=hi))
   original=sample['original_record'];d=original['hierarchy']['diagnostics'];current=np.asarray(original['current_reference21']);scale=np.asarray(d['reference_step_units21']);actual_lo=np.asarray(original['actual_reference_lower21']);actual_hi=np.asarray(original['actual_reference_upper21']);raw=current+result.x[:21]*scale;proposed=np.minimum(actual_hi,np.maximum(actual_lo,raw));rounding=64*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(raw),abs(proposed)))
   ck('accepted_real_reference_requires_roundoff_only/'+sample['sample_id'],np.all(abs(proposed-raw)<=rounding))
   wrapper=SimpleNamespace(x=result.x[:21],diagnostics={'extended_normalized_x':result.x},completed_prefix=result.completed_prefix,fully_converged=result.fully_converged,certified_prefix=result.certified_prefix)
   actual_score=score_emitted_reference(wrapper,current21=current,proposed21=proposed,scale21=scale)
   ck('accepted_real_emitted_original_goal_score/'+sample['sample_id'],actual_score['available'])
   v['real_reference_issue_check']=dict(strict_normalized_box=bool(np.all(result.x>=lo) and np.all(result.x<=hi)),raw_projection_max_abs=float(np.max(abs(proposed-raw),initial=0)),roundoff_only=bool(np.all(abs(proposed-raw)<=rounding)),emitted_score=actual_score)
  else:
   v=verify_stage(p,result.x);exact=True;good=result.completed_prefix==0 and not result.certified_prefix and bits(result.x,p['initial_x']) and v.get('primal',{}).get('passed') is True;ck('unavailable_retains_original_anchor/'+sample['sample_id'],good)
  rows.append(dict(sample_id=sample['sample_id'],context_key=sample['context_key'],selection_kind=sample['selection_kind'],layer_name=p['layer_name'],status=result.status,available=result.available,fully_converged=result.fully_converged,source='saved65_original_matrices',exact_original_domain=exact,independent_verification=v,candidate_sources=[c['candidate'] for level in result.levels for c in level['attempts']],elapsed_s=time.perf_counter()-began))
 print('CORPUS65_COMPLETE '+str(dict(Counter((r['layer_name'],r['available']) for r in rows))),flush=True)
 # Analytic finite hierarchy failure must retain its genuinely certified prefix.
 with patch.object(finite,'_gradient_segment_candidate',side_effect=FloatingPointError('injected_all_primal_candidate_generation_failure')), patch.object(finite,'_slsqp_candidate',side_effect=FloatingPointError('injected_all_primal_candidate_generation_failure')):
  r=finite.solve_hierarchy(lower=[-1.,-1.],upper=[1.,1.],initial=[0.,0.],levels=[finite.Level('keep_first',np.array([[1.,0.]]),np.array([0.])),finite.Level('missing_second',np.array([[0.,1.]]),np.array([1.]))])
 ck('late_generation_failure_keeps_only_first_certified_prefix',r.available and r.completed_prefix==1 and not r.fully_converged and r.failed_layer['index']==1 and bits(r.x,[0.,0.]))
 rates=np.ones(21);units=rates*.005;B=np.zeros((25,21));B[:21,:]=np.eye(21);J=np.zeros((1,25));J[0,0]=1.;M=np.zeros((1,21));M[0,0]=1.
 fixture=dict(time_s=12.5,dt_s=.005,current_reference21=np.zeros(21),encoder_reference21=np.zeros(21),step_lower21=-units,step_upper21=units,reference_rates21=rates,normal_jacobian_Px25=J,proposal_map25x21=B,robot_velocity25=np.zeros(25),pressure_P_N=np.array([1.]),reference_target_delta21=np.zeros(21),motion_matrix_Mx21=M,motion_error_M_m_s=np.array([-1.]),motion_time_s=12.5)
 a=task.solve_contact_exit(**fixture,formulation='backend');b=task.solve_contact_exit(**fixture,formulation='joint');synthetic.update(backend=a,joint=b)
 ck('analytic_backend_normal_satisfied_set',a.available and abs(a.x[0])<=1.1e-8 and a.diagnostics['normal_optimal_slack_frozen_before_motion'])
 ck('analytic_joint_free_eta_and_new_prefix',b.available and abs(b.x[0]-.5)<=1.1e-8 and abs(b.diagnostics['final_slack_P_m'][0]-.0025)<=1e-10 and b.certified_prefix[0]['name']=='joint_normal_and_observed_projection' and not b.diagnostics['normal_witness_is_submitted_prefix'] and not b.diagnostics['normal_optimal_slack_frozen_before_motion'])
 for name,result in [('backend',a),('joint',b)]:
  proposed=result.x*units;score=score_emitted_reference(result,current21=np.zeros(21),proposed21=proposed,scale21=units);changed=proposed.copy();changed[0]+=.2*units[0];bad=score_emitted_reference(result,current21=np.zeros(21),proposed21=changed,scale21=units);ck('actual_emitted_score_and_changed_goal_rejected/'+name,score['available'] and not bad['available']);synthetic[name+'_emitted_score']=score
 # Compare assembly with the SAME supplied normal witness/backend, not distinct optima.
 captures={'old':[],'new':[]}
 def capture(which):
  def call(**kw):
   captures[which].append(dict(lower=np.asarray(kw['lower']).copy(),upper=np.asarray(kw['upper']).copy(),initial=np.asarray(kw['initial']).copy(),G=np.asarray(kw['inequality_matrix']).copy(),h=np.asarray(kw['inequality_upper']).copy(),levels=[(z.name,np.asarray(z.matrix).copy(),np.asarray(z.target).copy()) for z in kw['levels']]))
   k=dict(kw);k['levels']=tuple(oldkernel.Level(z.name,z.matrix,z.target) for z in kw['levels']);return oldkernel.solve_hierarchy(**k)
  return call
 with patch.object(oldtask,'solve_hierarchy',side_effect=capture('old')):old_result=oldtask.solve_contact_exit(**fixture)
 with patch.object(task,'solve_hierarchy',side_effect=capture('new')):new_result=task.solve_contact_exit(**fixture,formulation='backend')
 exact=len(captures['old'])==len(captures['new'])==2
 for x,y in zip(captures['old'],captures['new']):
  exact=exact and all(bits(x[k],y[k]) for k in ('lower','upper','initial','G','h')) and len(x['levels'])==len(y['levels']) and all(nx==ny and bits(ax,ay) and bits(bx,by) for (nx,ax,bx),(ny,ay,by) in zip(x['levels'],y['levels']))
 ck('backend_same_witness_all_original_task_matrices_exact',exact and bits(old_result.x,new_result.x))
 no_motion=dict(fixture,motion_matrix_Mx21=None,motion_error_M_m_s=None,motion_time_s=None);u=task.solve_contact_exit(**no_motion,formulation='joint');ck('unknown_motion_omitted_not_zero',u.available and u.diagnostics['motion_unknown_not_zero'] and u.certified_prefix[0]['name']=='normal_correction_motion_unknown');synthetic['unknown_motion']=u
 p0=dict(no_motion,normal_jacobian_Px25=np.empty((0,25)),pressure_P_N=np.empty(0),reference_target_delta21=np.r_[.002,np.zeros(20)],path_reference_target_delta21=np.r_[.001,np.zeros(20)]);z=task.solve_contact_exit(**p0,formulation='joint');ck('knownP0_21dim_explicit_path_retained',z.available and len(z.diagnostics['extended_normalized_x'])==21 and abs(z.x[0]-.2)<=1.1e-8 and z.diagnostics['explicit_path_target_active']);synthetic['knownP0']=z
 for name,change in [('unknown_normal',dict(normal_jacobian_Px25=None)),('partial_motion',dict(motion_error_M_m_s=None)),('old_motion_clock',dict(motion_time_s=12.495)),('boolean_pressure',dict(pressure_P_N=[True]))]:
  result=task.solve_contact_exit(**dict(fixture,**change),formulation='joint');ck('invalid_unknown_is_unavailable/'+name,not result.available and result.completed_prefix==0 and not result.certified_prefix);synthetic[name]=dict(status=result.status,available=result.available,reason=result.diagnostics.get('reason'))
 ck('source179_and_numerical_identity_unchanged',source==source_identity(['ratio_effort']) and identity==numerical_identity())
 ck('zero_engine_forward_contact_writer_calls',not forbidden)
 status='completed_all65_and_finite_interfaces'
except Exception as exc:issues.append('UNEXPECTED:'+repr(exc));status='independent_attempt_error';synthetic['unexpected']=dict(error=repr(exc),traceback=traceback.format_exc())
summary=dict(assigned65=65,assessed65=len(rows),accepted65=sum(r['available'] for r in rows),unavailable65=sum(not r['available'] for r in rows),accepted_independent_KKT=sum(r['available'] and r['independent_verification'].get('kkt_passed') is True for r in rows),all_accepted_domains_exact=all(r['exact_original_domain'] for r in rows),by_layer={name:dict(assigned=sum(r['layer_name']==name for r in rows),accepted=sum(r['layer_name']==name and r['available'] for r in rows)) for name in sorted({r['layer_name'] for r in rows})})
payload=dict(schema='v22_numerical_strict_box_independent_v1',passed=not issues and len(rows)==65,status=status,checks=checks,issues=issues,summary=summary,rows=rows,synthetic=synthetic,production_source_sha256=subset,source_sha256=source,numerical_identity=identity,corpus_sha256=sha(ROOT/'review/v21_numerical_corpus_v1.json'),checker_sha256=sha(__file__),independent_verifier_sha256=sha(ROOT/'review/v21_original_matrix_kkt_verifier_v1.py'),forbidden_calls=forbidden,physics_steps=0,model_constructions=0,scope='All65 original saved single-stage problems once; every accepted certificate independently checked, honest unavailable retained. Strict normalized boxes and actual saved-current/scale physical emission gate plus samegoal score for all accepted65; limited task/unknown/P0/same-witness fixtures. No new physical performance, no all-conditions numerical completeness or realtime claim.')
with out.open('x',encoding='utf8') as f:json.dump(plain(payload),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(passed=payload['passed'],checks=len(checks),issues=issues,summary=summary,out=str(out),sha256=sha(out)),ensure_ascii=False),flush=True)
raise SystemExit(0 if payload['passed'] else 1)

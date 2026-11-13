"""Fixed three states: lexicographic minimum-normal then minimum normalized-reference witness."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'review'))
from v22_joint_task_fusion.numerics import ensure_numeric_environment,numerical_identity
ensure_numeric_environment()
import mujoco
forbidden=[]
def deny(*a,**kw):forbidden.append('model_step_forward_contact_writer');raise AssertionError(forbidden[-1])
class NoModel:
 def __new__(cls,*a,**kw):return deny()
 from_xml_path=staticmethod(deny);from_xml_string=staticmethod(deny);from_binary_path=staticmethod(deny)
mujoco.MjModel=NoModel;mujoco.MjData=deny
for k in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
 if hasattr(mujoco,k):setattr(mujoco,k,deny)
from v13_a1.codec import ReferenceActor21
ReferenceActor21.apply=deny
from v22_joint_task_fusion import finite_solver as old
from v23_task_execution import finite_solver as new
from v23_task_execution.runtime import source_identity
from v21_original_matrix_kkt_verifier_v1 import verify_stage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return plain(x.tolist())
 if isinstance(x,np.generic):return plain(x.item())
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [plain(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def bits(a,b):return np.shape(a)==np.shape(b) and np.asarray(a,float).tobytes()==np.asarray(b,float).tobytes()
def check(p,point,cert):
 k=cert['KKT'];return verify_stage(p,point,duals={'inequality_unit':k['full_normalized_inequality_dual'],'equality_raw':k['equality_raw_dual'],'provenance':'backend dual separately checked against all original rows and own frozen first-layer attained vector'},comparison_x=p['initial_x'])
cp=ROOT/'review/v21_numerical_corpus_v1.json';c=json.loads(cp.read_text('utf8'));basepath=ROOT/'review/v23_normal_witness_motion_budget_diagnostic_v1.json';baseline=json.loads(basepath.read_text('utf8'));contexts=[['collision','hold',619741],['collision','hold',619742],['collision','transport',619741]];identity=numerical_identity();source=source_identity(['ratio_effort']);assert len(source)==186
rows=[];issues=[]
for context in contexts:
 sample=next(z for z in c['samples'] if z['context_key']==context and z['selection_kind']=='first_failed_layer' and z['failed_stage_problem']['layer_name']=='public_external_projection_low_motion');p=sample['stage_problems'][0];A1=np.asarray(p['A']);b1=np.asarray(p['b']);lo=np.asarray(p['lower']);hi=np.asarray(p['upper']);n=len(lo);P=n-21;E=np.asarray(p['E']).reshape((-1,n));f=np.asarray(p['f']);fixed=np.flatnonzero(lo==hi);assert bits(E[:len(fixed)],np.eye(n)[fixed]);A2=np.column_stack((np.eye(21),np.zeros((21,P))));b2=np.zeros(21);weights=np.diag(A1[:,21:]);structure=bits(A1[:,:21],np.zeros((P,21))) and bits(A1[:,21:],np.diag(weights)) and bool(np.all(weights>0)) and bits(b1,np.zeros(P));assert structure
 d=sample['original_record']['hierarchy']['diagnostics'];motion=d['motion_task'];units=np.asarray(d['reference_step_units21']);M=np.asarray(motion['matrix_Mx21']);e=np.asarray(motion['error_M_m_s']);m=M*units[None,:]/np.asarray(motion['effective_row_units_M'])[:,None];offset=d['dt_s']*e/np.asarray(motion['effective_row_units_M']);solutions={}
 for label,module in [('frozen_V22_FW_first',old),('V23_SLSQP_first',new)]:
  print('CASE_START '+str(context)+' '+label,flush=True);start=time.perf_counter();r=module.solve_hierarchy(lower=lo,upper=hi,levels=[module.Level(p['layer_name'],A1,b1),module.Level('unique_minimum_normalized_reference_witness',A2,b2)],initial=np.asarray(p['initial_x']),inequality_matrix=np.asarray(p['input_inequality_matrix']),inequality_upper=np.asarray(p['input_inequality_upper']),equality_matrix=E[len(fixed):],equality_target=f[len(fixed):],max_iterations=p['max_iterations'],atol=p['atol'],rtol=p['rtol']);elapsed=time.perf_counter()-start
  item={'status':r.status,'fully_converged':r.fully_converged,'completed_prefix':r.completed_prefix,'elapsed_s':elapsed,'attempt_sources':[[a['candidate'] for a in lev['attempts']] for lev in r.levels],'all_attempts_retained_in_original_result_not_logged_here':True,'failed_layer':None if r.failed_layer is None else {'index':r.failed_layer['index'],'name':r.failed_layer['name'],'status':r.failed_layer['status']}}
  if not r.certified_prefix:issues.append(str(context)+' '+label+' no normal certificate');solutions[label]=item;continue
  cert1=r.certified_prefix[0];v1=check(p,cert1['x'],cert1);hd=cert1['hard_domain'];domainexact=bits(hd['original_inequality_matrix_with_box'],p['G']) and bits(hd['original_inequality_upper_with_box'],p['h']) and bits(hd['fixed_equality_matrix'],E) and bits(hd['fixed_equality_target'],f);normal_final=check(p,r.x,cert1);item.update(normal_certificate=v1,normal_at_final_witness_verification=normal_final,all_original_domain_rows_exact=domainexact,normal_attained_first=cert1['attained'],normal_attained_final=A1@r.x,normal_attained_preservation_inf=float(np.max(abs(A1@r.x-np.asarray(cert1['attained'])),initial=0)),x21=r.x[:21],etaP=r.x[21:],strictbox=bool(np.all(r.x>=lo) and np.all(r.x<=hi)),normalized_motion_budget=abs(m@r.x[:21]+offset),normal_objective_final=normal_final.get('objective'))
  if r.completed_prefix==2 and r.fully_converged:
   cert2=r.certified_prefix[1];p2=dict(p,A=A2,b=b2,E=np.vstack((E,A1)),f=np.r_[f,cert1['attained']],initial_x=cert1['x'],previous_frozen=[{'matrix':A1,'attained':cert1['attained']}]);v2=check(p2,r.x,cert2);pf=cert2['previous_frozen'];exact=bits(cert2['matrix'],A2) and bits(cert2['target'],b2) and len(pf)==1 and bits(pf[0]['matrix'],A1) and bits(pf[0]['attained'],cert1['attained']);item.update(canonical_certificate=v2,original_normal_frozen_not_retargeted=exact,canonical_cost=v2.get('objective'),canonical_available=v2['kkt_passed'] and normal_final['kkt_passed'] and domainexact and exact and item['strictbox'])
  else:item.update(canonical_available=False)
  if not item['canonical_available']:issues.append(str(context)+' '+label+' canonical incomplete/original KKT/strictbox failed')
  solutions[label]=item;print('CASE_DONE '+str(context)+' '+label+' '+str(item['canonical_available'])+' '+str(elapsed),flush=True)
 a=solutions['frozen_V22_FW_first'];b=solutions['V23_SLSQP_first'];before=next(z for z in baseline['rows'] if z['context_key']==context)['difference'];difference={'before_x21_difference_inf':before['x21_difference_inf'],'before_budget_difference_inf':before['normalized_budget_difference_inf']}
 if a.get('canonical_available') and b.get('canonical_available'):
  db=np.asarray(b['normalized_motion_budget'])-np.asarray(a['normalized_motion_budget']);dx=np.asarray(b['x21'])-np.asarray(a['x21']);difference.update(after_x21_difference_inf=float(np.max(abs(dx),initial=0)),after_eta_difference_inf=float(np.max(abs(np.asarray(b['etaP'])-np.asarray(a['etaP'])),initial=0)),after_budget_difference_inf=float(np.max(abs(db),initial=0)),after_budget_difference=db,canonical_cost_delta=b['canonical_cost']-a['canonical_cost'],normal_attained_across_backends_difference_inf=float(np.max(abs(np.asarray(b['normal_attained_first'])-np.asarray(a['normal_attained_first'])),initial=0)),physical_step_difference21=dx*units,physical_step_difference_inf=float(np.max(abs(dx*units),initial=0)),same_second_problem_exact_attained=bits(a['normal_attained_first'],b['normal_attained_first']))
 rows.append({'context_key':context,'sample_id':sample['sample_id'],'time_s':sample['original_record']['time_s'],'unique_witness_assumptions':{'normal_A_only_eta_positive_diagonal':structure,'eta_weights':weights,'cost2_only_x21_identity':True,'nonempty_original_compact_polytope':True},'solutions':solutions,'difference':difference})
assert source_identity(['ratio_effort'])==source and numerical_identity()==identity and not forbidden
out=ROOT/'review/v23_canonical_normal_witness_probe_v1.json';assert not out.exists();payload={'schema':'v23_canonical_normal_witness_probe_v1','complete':True,'passed':not issues and len(rows)==3,'assigned_states':3,'assigned_two_layer_calls':6,'assessed_states':len(rows),'canonical_certified_calls':sum(z.get('canonical_available',False) for row in rows for z in row['solutions'].values()),'issues':issues,'rows':rows,'source_sha256':source,'production_source_sha256':{p:sha(ROOT/p) for p in ('v22_joint_task_fusion/finite_solver.py','v23_task_execution/finite_solver.py','v22_joint_task_fusion/numerics.py')},'numerical_identity':identity,'corpus_sha256':sha(cp),'baseline_probe_sha256':sha(basepath),'checker_sha256':sha(__file__),'forbidden_calls':forbidden,'physics_steps':0,'model_constructions':0,'scope':'Review-only prospective task-contract alternative, NOT current 186 runtime. Exact mathematical lexicographic uniqueness follows positive definite eta objective then strictly convex x21 objective; finite computed L1 attained vectors may differ by rounding so two frozen numeric L2 problems need not be byte-identical. No physical energy, servo response,6D motion, safety,placement or benchmark claim.'}
with out.open('x',encoding='utf8') as f:json.dump(plain(payload),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'complete':True,'passed':payload['passed'],'issues':issues,'canonical_certified_calls':payload['canonical_certified_calls'],'differences':plain([r['difference'] for r in rows]),'out':str(out),'sha256':sha(out)},ensure_ascii=False),flush=True)

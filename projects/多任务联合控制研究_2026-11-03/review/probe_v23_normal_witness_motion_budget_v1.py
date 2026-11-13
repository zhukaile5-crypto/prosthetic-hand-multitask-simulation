"""Fixed three-state witness propagation: original domain/cost, no new physical data."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'review'))
from v22_joint_task_fusion.numerics import ensure_numeric_environment,numerical_identity
ensure_numeric_environment()
import mujoco
forbidden=[]
def deny(*a,**kw):forbidden.append('engine_call');raise AssertionError('no engine/model/step/forward/contact/writer')
class NoModel:
 def __new__(cls,*a,**kw):return deny()
 from_xml_path=staticmethod(deny);from_xml_string=staticmethod(deny);from_binary_path=staticmethod(deny)
mujoco.MjModel=NoModel;mujoco.MjData=deny
for k in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_collision','mj_contactForce'):
 if hasattr(mujoco,k):setattr(mujoco,k,deny)
from v13_a1.codec import ReferenceActor21
ReferenceActor21.apply=deny
from v22_joint_task_fusion import finite_solver as old
from v23_task_execution import finite_solver as new
from v21_original_matrix_kkt_verifier_v1 import verify_stage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cp=ROOT/'review/v21_numerical_corpus_v1.json';c=json.loads(cp.read_text('utf8'))
contexts=[['collision','hold',619741],['collision','hold',619742],['collision','transport',619741]]
rows=[];identity=numerical_identity();issues=[]
for context in contexts:
 s=next(s for s in c['samples'] if s['context_key']==context and s['selection_kind']=='first_failed_layer' and s['failed_stage_problem']['layer_name']=='public_external_projection_low_motion');p=s['stage_problems'][0];n=len(p['lower']);lo=np.asarray(p['lower']);hi=np.asarray(p['upper']);E=np.asarray(p['E']).reshape((-1,n));f=np.asarray(p['f']);fixed=np.flatnonzero(lo==hi);assert np.array_equal(E[:len(fixed)],np.eye(n)[fixed]);d=s['original_record']['hierarchy']['diagnostics'];motion=d['motion_task'];units=np.asarray(d['reference_step_units21']);M=np.asarray(motion['matrix_Mx21']);e=np.asarray(motion['error_M_m_s']);scale=np.asarray(motion['effective_row_units_M']);m=M*units[None,:]/scale[:,None];offset=float(d['dt_s'])*e/scale;solutions={}
 for label,module in [('frozen_V22_FW_first',old),('V23_SLSQP_first',new)]:
  began=time.perf_counter();r=module.solve_hierarchy(lower=lo,upper=hi,levels=[module.Level(p['layer_name'],np.asarray(p['A']),np.asarray(p['b']))],initial=np.asarray(p['initial_x']),inequality_matrix=np.asarray(p['input_inequality_matrix']),inequality_upper=np.asarray(p['input_inequality_upper']),equality_matrix=E[len(fixed):],equality_target=f[len(fixed):],max_iterations=p['max_iterations'],atol=p['atol'],rtol=p['rtol']);elapsed=time.perf_counter()-began
  if not r.available:issues.append(str(context)+' '+label+' unavailable');solutions[label]={'available':False};continue
  cert=r.certified_prefix[0];k=cert['KKT'];v=verify_stage(p,r.x,duals={'inequality_unit':k['full_normalized_inequality_dual'],'equality_raw':k['equality_raw_dual'],'provenance':'backend dual independently checked in original normal problem'},comparison_x=p['initial_x']);strict=bool(np.all(r.x>=lo) and np.all(r.x<=hi));budget=abs(m@r.x[:21]+offset)
  if not v['kkt_passed'] or not strict:issues.append(str(context)+' '+label+' KKT/box')
  solutions[label]={'available':True,'independent_original_normal_KKT':v['kkt_passed'],'strictbox':strict,'original_problem_sha256':v['original_problem_sha256'],'normal_objective':v['objective'],'x21':r.x[:21].tolist(),'etaP':r.x[21:].tolist(),'normal_attained':(np.asarray(p['A'])@r.x).tolist(),'normalized_motion_budget':budget.tolist(),'solver_elapsed_s':elapsed,'candidate_sources':[a['candidate'] for lev in r.levels for a in lev['attempts']]}
 a=solutions['frozen_V22_FW_first'];b=solutions['V23_SLSQP_first'];diff={}
 if a['available'] and b['available']:
  ba=np.asarray(a['normalized_motion_budget']);bb=np.asarray(b['normalized_motion_budget']);dx=np.asarray(b['x21'])-np.asarray(a['x21']);de=np.asarray(b['etaP'])-np.asarray(a['etaP']);db=bb-ba
  diff={'x21_difference_inf':float(np.max(abs(dx),initial=0)),'eta_difference_inf':float(np.max(abs(de),initial=0)),'normal_objective_delta_V23_minus_V22':b['normal_objective']-a['normal_objective'],'normalized_budget_difference':db.tolist(),'normalized_budget_difference_inf':float(np.max(abs(db),initial=0)),'budget_rows_lower':int(np.count_nonzero(db<0)),'budget_rows_equal':int(np.count_nonzero(db==0)),'budget_rows_higher':int(np.count_nonzero(db>0)),'same_original_normal_problem':a['original_problem_sha256']==b['original_problem_sha256'],'later_motion_G_same_but_h_changes_with_witness':True}
 rows.append({'context_key':context,'sample_id':s['sample_id'],'time_s':s['source'].get('time_s',s['original_record']['time_s']),'solutions':solutions,'difference':diff})
assert numerical_identity()==identity
out=ROOT/'review/v23_normal_witness_motion_budget_diagnostic_v1.json';assert not out.exists();payload={'schema':'v23_normal_witness_motion_budget_diagnostic_v1','complete':True,'passed':not issues and len(rows)==3,'assigned':3,'rows':rows,'issues':issues,'corpus_sha256':sha(cp),'finite_sources_sha256':{'V22':sha(ROOT/'v22_joint_task_fusion/finite_solver.py'),'V23':sha(ROOT/'v23_task_execution/finite_solver.py')},'numerical_identity':identity,'checker_sha256':sha(__file__),'forbidden_calls':forbidden,'model_constructions':0,'physics_steps':0,'scope':'Preselected first three original L3-failure contexts in corpus. BOTH backends solve the SAME original normal LS/domain; different valid normal witnesses can change subsequently defined componentwise motion budget abs(M*units*xA+dt*e)/row_scale. This is endogenous task-definition propagation, NOT same later problem, future-motion guarantee, physical causality, benchmark, or independent trial evidence.'};out.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps({'passed':payload['passed'],'out':str(out),'sha256':sha(out),'issues':issues,'differences':[r['difference'] for r in rows]},ensure_ascii=False))

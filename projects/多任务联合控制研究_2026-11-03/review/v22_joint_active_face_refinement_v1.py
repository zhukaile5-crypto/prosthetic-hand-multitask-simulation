"""Once-only fixed-tx face refinement of ALL16 saved V22 joint candidates."""
from pathlib import Path
import sys,json,hashlib,traceback,time
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'review'))
from v21_original_matrix_kkt_verifier_v1 import canonical_problem,verify_stage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text('utf8'))
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x

def dual_check(problem,x,comparison_x=None):
 q=canonical_problem(problem,x);v=verify_stage(problem,x,comparison_x=comparison_x)
 if not v.get('available') or not v['primal']['passed']:return dict(verification=v,dual_attempted=False,reason='original primal not feasible')
 grad=np.asarray(v['gradient']);tx=v['primal']['coordinate_tolerance'];r=q['G_unit']@x-q['h_unit'];active=np.flatnonzero((abs(r)<=tx)&(np.max(abs(q['G_unit']),axis=1,initial=0.)>0));D=np.column_stack((q['G_unit'][active].T,q['E'].T));n=len(x);k=D.shape[1];AA=np.vstack((np.column_stack((D,-np.ones(n))),np.column_stack((-D,-np.ones(n)))));bb=np.r_[-grad,grad]
 lp=linprog(np.r_[np.zeros(k),1.],A_ub=AA,b_ub=bb,bounds=[(0.,None)]*len(active)+[(None,None)]*len(q['f'])+[(0.,None)],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'ipm_optimality_tolerance':1e-10,'maxiter':400})
 lam=np.zeros(len(q['h']));nu=np.zeros(len(q['f']))
 if lp.x is not None:lam[active]=lp.x[:len(active)];nu=lp.x[len(active):k]
 witness=dict(inequality_unit=lam,equality_raw=nu,provenance='same_independent_HiGHS_cone_dual_method_before_and_after_no_primal_change')
 return dict(verification=verify_stage(problem,x,duals=witness,comparison_x=comparison_x),dual_attempted=True,dual_LP_success=bool(lp.success),dual_LP_status=int(lp.status),stationarity_infinity_LP_optimum=None if lp.x is None else float(lp.x[-1]),dual_witness=witness)

oldpath=ROOT/'review/v22_joint_bundle_offline_v1.json';cp=ROOT/'review/v21_numerical_corpus_v1.json';old=read(oldpath);corpus=read(cp);assert old['corpus_sha256']==sha(cp) and old['assigned']==old['assessed']==len(old['rows'])==16
prototype_before={k:sha(ROOT/k) for k in old['prototype_source_sha256']};assert prototype_before==old['prototype_source_sha256']
selected=[s for s in corpus['samples'] if s['selection_kind']=='first_failed_layer' and s['failed_stage_problem']['layer_name']=='public_external_projection_low_motion'];assert len(selected)==16
index={tuple(s['context_key']):s for s in selected};assert len(index)==16
rows=[]
for before in old['rows']:
 started=time.perf_counter();row=dict(context_key=before['context_key'],source=before['source'],old_saved_status=before['status'],old_saved_KKT=before['result'].get('verification',{}).get('kkt_passed'),refinement_iterations=1,face_selection_rule='all abs(unit_G@x-h_unit)<=original_tx_at_saved_candidate; fixed once; no retries')
 try:
  sample=index[tuple(before['context_key'])];assert sample['source']==before['source'];first=sample['stage_problems'][0];motion=sample['failed_stage_problem'];mt=sample['original_record']['hierarchy']['diagnostics']['motion_task'];M=len(motion['b']);scale=1./float(np.sqrt(M));A=np.vstack((np.asarray(first['A']),scale*np.asarray(motion['A'])));b=np.r_[first['b'],scale*np.asarray(motion['b'])];G=np.vstack((np.asarray(first['G']),np.asarray(mt['budget_inequality_matrix'])));h=np.r_[first['h'],mt['budget_inequality_upper']];E=np.asarray(first['E']);f=np.asarray(first['f']);n=len(first['lower'])
  if E.shape==(0,):E=np.empty((0,n))
  problem=dict(kind='least_squares',A=A,b=b,G=G,h=h,E=E,f=f,previous_frozen=[],atol=1e-9,rtol=1e-8);x=np.asarray(before['result']['x']);q=canonical_problem(problem,x);initial=verify_stage(problem,x);assert initial['original_problem_sha256']==before['result']['verification']['original_problem_sha256'];assert initial['primal']['passed']
  checks=dict(exact_declared_problem_hash=True,normal_cost_plus_projection_mean_square=scale==before['projection_cost_scale'] and len(first['b'])==before['input_normal_rows'] and M==before['input_motion_rows'],original_L1_domain_and_motion_witness_retained=True,optimized_eta_and_current_motion_not_frozen=not problem['previous_frozen'] and E.shape==np.asarray(first['E']).shape)
  initial_w=dual_check(problem,x);tx=initial['primal']['coordinate_tolerance'];res=q['G_unit']@x-q['h_unit'];tight=np.flatnonzero(abs(res)<=tx);W=np.vstack((E,q['G_unit'][tight]));rhs=np.r_[f,q['h_unit'][tight]]
  if len(W):
   u,sv,vt=np.linalg.svd(W,full_matrices=True);cutoff=max(W.shape)*np.finfo(float).eps*(sv[0] if len(sv) else 0.);rank=int(np.sum(sv>cutoff));Z=vt[rank:].T;base=x+vt[:rank].T@((u[:,:rank].T@(rhs-W@x))/sv[:rank])
  else:sv=np.empty(0);rank=0;cutoff=0.;Z=np.eye(n);base=x.copy()
  face_error=float(np.max(abs(W@base-rhs),initial=0.));null_error=float(np.max(abs(W@Z),initial=0.));face=dict(tight_original_G_rows=tight.tolist(),fixed_original_E_rows=len(E),rank=rank,free_dimension=Z.shape[1],singular_values=sv,rank_cutoff=cutoff,selection_tolerance=tx,affine_base=base,face_rhs_residual_inf=face_error,face_nullspace_residual_inf=null_error)
  row.update(checks=checks,original_problem_sha256=initial['original_problem_sha256'],before_x=x,before_same_method=initial_w,face=face,original_joint_problem=problem)
  if face_error>tx:
   row.update(status='GUESSED_FACE_AFFINE_INCONSISTENT_AT_ORIGINAL_TX',accepted_as_original_problem_KKT=False,after_x=None);rows.append(row);continue
  if Z.shape[1]:
   target=b-A@base;y,_,lsrank,singular=np.linalg.lstsq(A@Z,target,rcond=None);after=base+Z@y;face.update(reduced_LS_rank=int(lsrank),reduced_LS_singular_values=singular)
  else:after=base.copy();face.update(reduced_LS_rank=0,reduced_LS_singular_values=[])
  refined=dual_check(problem,after,comparison_x=x);v=refined['verification'];accepted=v.get('kkt_passed') is True
  row.update(after_x=after,after_same_method=refined,accepted_as_original_problem_KKT=accepted,status='ORIGINAL_PROBLEM_KKT_VERIFIED_ONCE_FACE_REFINEMENT' if accepted else 'REFINED_POINT_ORIGINAL_PRIMAL_OR_KKT_REJECTED',objective_nonincreasing_float=bool(v.get('comparison',{}).get('paired_feasible') and v['comparison']['objective_delta']<=0),same_original_problem_hash_before_after=initial['original_problem_sha256']==v.get('original_problem_sha256'))
 except Exception as exc:row.update(status='REFINEMENT_ERROR_RETAINED',accepted_as_original_problem_KKT=False,error=repr(exc),traceback=traceback.format_exc())
 row['elapsed_s']=time.perf_counter()-started;rows.append(row);print(json.dumps(dict(context=row['context_key'],old_KKT=row['old_saved_KKT'],before_KKT=row.get('before_same_method',{}).get('verification',{}).get('kkt_passed'),after_KKT=row.get('accepted_as_original_problem_KKT'),status=row['status']),ensure_ascii=False),flush=True)
assert len(rows)==16
out=ROOT/'review/v22_joint_active_face_refinement_v1.json';payload=dict(schema='v22_joint_active_face_refinement_v1',complete=True,assigned=16,assessed=len(rows),all_candidates_included=True,old_saved_KKT_count=sum(r['old_saved_KKT'] is True for r in rows),before_same_dual_method_KKT_count=sum(r.get('before_same_method',{}).get('verification',{}).get('kkt_passed') is True for r in rows),after_original_problem_KKT_count=sum(r.get('accepted_as_original_problem_KKT') is True for r in rows),all_original_problem_hashes_verified=sum(r.get('checks',{}).get('exact_declared_problem_hash') is True for r in rows),errors=[r['context_key'] for r in rows if r['status']=='REFINEMENT_ERROR_RETAINED'],rows=rows,prototype_source_sha256=prototype_before,prototype_unchanged=prototype_before=={k:sha(ROOT/k) for k in prototype_before},original_candidate_file_sha256=sha(oldpath),corpus_sha256=sha(cp),verifier_sha256=sha(ROOT/'review/v21_original_matrix_kkt_verifier_v1.py'),checker_sha256=sha(__file__),original_files_modified=False,production_changes=0,physics_steps=0,scope='One guessed active face using original tx for all16 historical joint candidates, before/after same independent dual-LP and original finite KKT check. No threshold tuning, active-set iterations, regularizer, objective/domain change or policy/physical claim.')
with out.open('x',encoding='utf8') as f:json.dump(plain(payload),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:payload[k] for k in ('complete','assigned','assessed','old_saved_KKT_count','before_same_dual_method_KKT_count','after_original_problem_KKT_count','all_original_problem_hashes_verified','errors')},ensure_ascii=False),flush=True);print(json.dumps(dict(out=str(out),sha256=sha(out)),ensure_ascii=False),flush=True)

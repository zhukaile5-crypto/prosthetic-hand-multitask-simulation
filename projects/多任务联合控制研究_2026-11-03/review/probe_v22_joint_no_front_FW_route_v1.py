"""Fixed16 experimental no-front-FW route; original f304/private math unchanged."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'review'))
from v22_joint_task_fusion.numerics import ensure_numeric_environment,numerical_identity
ensure_numeric_environment()
from v22_joint_task_fusion import finite_solver as fs
from v21_original_matrix_kkt_verifier_v1 import canonical_problem,verify_stage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text('utf8'))
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
EXPECTED='f30419b471c614b6ef81101ac52ce6679013fb2ed0caef58225b5699dc6c5c3b';assert sha(fs.__file__)==EXPECTED
jointpath=ROOT/'review/v22_joint_active_face_refinement_v1.json';corpuspath=ROOT/'review/v21_numerical_corpus_v1.json';priorpath=ROOT/'review/v22_joint_level_resource_probe_v1.json';joint=read(jointpath);corpus=read(corpuspath);prior=read(priorpath);index={tuple(s['context_key']):s for s in corpus['samples'] if s['selection_kind']=='first_failed_layer' and s['failed_stage_problem']['layer_name']=='public_external_projection_low_motion'};assert len(joint['rows'])==len(prior['rows'])==len(index)==16
baseline={tuple(r['context_key']):r for r in prior['rows']};identity=numerical_identity();rows=[];out=ROOT/'review/v22_joint_no_front_FW_route_v1.json';assert not out.exists()
for old in joint['rows']:
 ctx=tuple(old['context_key']);sample=index[ctx];p=old['original_joint_problem'];initial=np.asarray(sample['failed_stage_problem']['initial_x']);q=canonical_problem(p,initial);lo=np.asarray(sample['stage_problems'][0]['lower']);hi=np.asarray(sample['stage_problems'][0]['upper']);A,b,G,h,E,f=q['A'],q['b'],q['G_unit'],q['h_unit'],q['E'],q['f'];initialcheck=verify_stage(p,initial);assert initialcheck['original_problem_sha256']==baseline[ctx]['independent_original_bundle_verification']['original_problem_sha256'];assert sample['source']==old['source']
 attempts=[];accepted=None;point=None;began=time.perf_counter()
 def check(kind,x,note):
  cert=fs._certify(A,b,x,G,h,E,f,(),p['atol'],p['rtol'],400,lower=lo,upper=hi);strict=bool(np.all(x>=lo) and np.all(x<=hi));duals=None
  if cert['full_normalized_inequality_dual'] is not None and cert['equality_raw_dual'] is not None:duals=dict(inequality_unit=cert['full_normalized_inequality_dual'],equality_raw=cert['equality_raw_dual'],provenance='unchanged_f304_current_candidate_original_domain_witness')
  v=verify_stage(p,x,duals=duals,comparison_x=initial);okay=bool(cert['certified'] and strict and v['kkt_passed']);metric={k:v.get(k) for k in ('available','kkt_passed','objective','stationarity_inf','gradient_tolerance','complementarity_inf','status','original_problem_sha256')};metric['primal']=v.get('primal');metric['comparison']=v.get('comparison')
  attempts.append(dict(candidate=kind,point=x,strictbox=strict,core_current_KKT=cert['certified'],independent_original_verification=metric,accepted=okay,generation=note,dual_LP_called=cert['dual_witness'].get('lp_called',False)))
  return okay
 try:
  point,note=fs._slsqp_candidate(A,b,G,h,E,f,initial,400,p['atol'],lower=lo,upper=hi)
  if check('f304_analytic_SLSQP_first_no_FW',point,note):accepted=point.copy()
  else:
   refined,face=fs._refine_once(A,b,G,h,E,f,point,p['atol'],p['rtol'],lower=lo,upper=hi)
   if check('same_once_exact_coordinate_face',refined,face):accepted=refined.copy()
   elif np.any(refined<lo) or np.any(refined>hi):
    box=np.minimum(hi,np.maximum(lo,refined));boxnote=dict(method='explicit_NEW_literal_original_box_candidate_then_all_original_KKT_recheck',change_inf=float(np.max(abs(box-refined),initial=0)),previous_uncertified_point=refined)
    if check('explicit_new_box_candidate_not_post_certificate_clip',box,boxnote):accepted=box.copy()
 except Exception as exc:attempts.append(dict(candidate='rejected_generation_or_check',error=repr(exc),traceback=traceback.format_exc(),accepted=False))
 elapsed=time.perf_counter()-began;last=next((a for a in reversed(attempts) if 'independent_original_verification' in a),None);row=dict(context_key=list(ctx),original_problem_sha256=initialcheck['original_problem_sha256'],input_initial_source='unchanged original old-joint runner oldL3 initial_x',available=accepted is not None,strictbox=False if accepted is None else bool(np.all(accepted>=lo) and np.all(accepted<=hi)),status='EXPERIMENTAL_ROUTE_ORIGINAL_KKT_ACCEPTED' if accepted is not None else 'EXPERIMENTAL_ROUTE_NO_ORIGINAL_KKT_CERTIFICATE',attempts=attempts,attempt_count=len(attempts),FW_direction_count=0,solver_elapsed_s=elapsed,accepted_point=accepted,prior_f304=dict(available=baseline[ctx]['available'],strictbox=baseline[ctx]['strictbox'],solver_elapsed_s=baseline[ctx]['solver_elapsed_s'],FW_direction_count=baseline[ctx]['FW_direction_count'],attempt_count=baseline[ctx]['attempt_count']),last_objective=None if last is None else last['independent_original_verification']['objective'],cost_change_vs_initial=None if last is None else last['independent_original_verification'].get('comparison',{}).get('objective_delta'))
 rows.append(row);print(json.dumps(dict(context=ctx,accepted=row['available'],attempts=len(attempts),elapsed_s=elapsed),ensure_ascii=False),flush=True)
summary=dict(assigned=16,assessed=len(rows),accepted=sum(r['available'] for r in rows),strictbox=sum(r['strictbox'] for r in rows),FW_directions=0,total_attempts=sum(r['attempt_count'] for r in rows),accepted_candidate_sources=dict(Counter(next(a['candidate'] for a in r['attempts'] if a.get('accepted')) for r in rows if r['available'])))
for metric in ('solver_elapsed_s','attempt_count'):
 v=[r[metric] for r in rows];summary[metric]=dict(minimum=min(v),median=float(np.median(v)),p95=float(np.percentile(v,95)),maximum=max(v))
payload=dict(schema='v22_experimental_no_front_FW_route_v1',complete=True,summary=summary,rows=rows,production_source_sha256={'v22_joint_task_fusion/finite_solver.py':sha(fs.__file__)},production_source_unchanged=sha(fs.__file__)==EXPECTED,numerical_identity=identity,numerical_identity_unchanged=numerical_identity()==identity,input_joint_sha256=sha(jointpath),corpus_sha256=sha(corpuspath),prior_full_f304_probe_sha256=sha(priorpath),checker_sha256=sha(__file__),route_predeclared='ALL16: analytic-gradient f304 SLSQP first -> same one coordinate-face refinement -> explicit new original box candidate only if refined point outside; every accepted candidate full original KKT+strictbox; no front FW, no threshold/goal/domain sweep',candidate_organization_only=True,current_runtime_changed=False,production_changes=0,physics_steps=0,model_constructions=0,current_live_B_C_partial_data_read=False,scope='Historical16 single-level offline numerical/resource comparison under current parallel host load, not benchmark superiority, all65 or closed-loop performance. Old f304 15/16 with400-path evidence retained.')
with out.open('x',encoding='utf8') as f:json.dump(plain(payload),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(complete=True,summary=summary,out=str(out),sha256=sha(out)),ensure_ascii=False),flush=True)

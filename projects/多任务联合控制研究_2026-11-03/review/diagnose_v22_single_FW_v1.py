"""Fixed10 original-polytope FW iterations for sole residual sample015."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v22_joint_task_fusion import finite_solver as fs
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
out=ROOT/'review/v22_single_FW_review_v1.json'
if out.exists():raise FileExistsError(out)
src=ROOT/'v22_joint_task_fusion/finite_solver.py';before=sha(src)
c=json.loads((ROOT/'review/v21_numerical_corpus_v1.json').read_text(encoding='utf-8'))
s=next(s for s in c['samples'] if s['sample_id']=='v21_numeric_015');p=s['failed_stage_problem'];n=p['variable_count']
A=np.asarray(p['A']);b=np.asarray(p['b']);G,h,_=fs._normalize(np.asarray(p['G']),np.asarray(p['h']));E=np.asarray(p['E']);f=np.asarray(p['f']);x=np.asarray(p['initial_x']);lo=np.asarray(p['lower']);hi=np.asarray(p['upper'])
frozen=[(np.asarray(a['matrix']),np.asarray(a['attained'])) for a in p['previous_frozen']];rows=[]
for i in range(10):
 start=time.perf_counter_ns();old=x.copy();x,note=fs._gradient_segment_candidate(A,b,G,h,E,f,x,p['max_iterations'])
 k=fs._certify(A,b,x,G,h,E,f,frozen,p['atol'],p['rtol'],p['max_iterations'],lower=lo,upper=hi)
 face,fnote=fs._refine_once(A,b,G,h,E,f,x,p['atol'],p['rtol'],lower=lo,upper=hi)
 fk=fs._certify(A,b,face,G,h,E,f,frozen,p['atol'],p['rtol'],p['max_iterations'],lower=lo,upper=hi)
 gap=float((2*A.T@(A@old-b))@(old-np.asarray(note['endpoint'])))
 row=dict(iteration=i+1,LP=note,FW_gap=gap,x=x,KKT=k,face=face,face_KKT=fk,face_note=fnote,elapsed_ms=(time.perf_counter_ns()-start)/1e6);rows.append(row)
 print(i+1,note['segment_alpha'],gap,k['stationarity_inf'],k['certified'],fk['stationarity_inf'],fk['certified'])
 if k['certified'] or fk['certified']:break
report=dict(schema='v22_single_FW_review_v1',sample_id=s['sample_id'],fixed_iteration_budget=10,rows=rows,
 source_sha256=before,source_unchanged=sha(src)==before,original_problem_unchanged=True,physics_steps=0,
 face_refinements_are_independent_diagnostic_endpoints_not_recursive_feedback=True)
out.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(sha(out))

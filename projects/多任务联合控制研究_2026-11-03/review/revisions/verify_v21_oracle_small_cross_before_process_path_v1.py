"""Three fixed small convex cases through ROOT's real offline oracle."""
from pathlib import Path
import sys,json,hashlib,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'review'))
print('SCIPY_ORACLE_IMPORT_BEGIN',flush=True)
from v21_same_problem_oracle_v1 import solve_problem,plain
print('SCIPY_ORACLE_IMPORT_COMPLETE',flush=True)
log=ROOT/'review/v21_oracle_small_cross_attempt1.jsonl';out=ROOT/'review/v21_oracle_small_cross_attempt1.json';assert not log.exists() and not out.exists()
I=np.eye(2);G=np.vstack((I,-I));h=np.array([1.,1.,0.,0.]);cases=[
 ('redundant_E_original_anchor_already_KKT',dict(kind='least_squares',A=I,b=[2.,2.],G=G,h=h,E=[[1.,1.],[2.,2.]],f=[1.,2.],initial_x=[.5,.5],atol=1e-9,rtol=1e-8),False,[.5,.5]),
 ('coupled_LS_nonzero_target',dict(kind='least_squares',A=I,b=[1.,.4],G=np.vstack((G,[1.,1.])),h=np.r_[h,1.],E=[],f=[],initial_x=[0.,0.],atol=1e-9,rtol=1e-8),False,[.8,.2]),
 ('withdraw_LP_fixed_coordinate_original_LS_KKT',dict(kind='least_squares',A=[[1.,.5,0.]],b=[1.5],G=np.vstack((np.eye(3),-np.eye(3),[1.,1.,0.])),h=[1.,1.,0.,0.,0.,0.,1.],E=[[0.,0.,1.]],f=[0.],lower=[0.,0.,0.],upper=[1.,1.,0.],initial_x=[0.,0.,0.],atol=1e-9,rtol=1e-8),True,[1.,0.,0.])]
rows=[]
for name,p,lp,expected in cases:
 print('CASE_BEGIN '+name,flush=True)
 try:
  result=solve_problem(p,scalar_withdrawal=lp);verified=result.get('verification',{});tx=verified.get('primal',{}).get('coordinate_tolerance',0.);got=np.asarray(result.get('x',[]));passed=bool(verified.get('kkt_passed') and got.shape==np.shape(expected) and np.max(abs(got-expected),initial=0.)<=tx)
  if name.startswith('redundant_E'):passed=passed and result.get('original_anchor_independent_KKT',{}).get('kkt_passed') is True
  row=dict(name=name,passed=passed,expected=expected,result=result)
 except Exception as e:row=dict(name=name,passed=False,error=repr(e),traceback=traceback.format_exc())
 rows.append(row)
 with log.open('a',encoding='utf8') as f:json.dump(plain(row),f,ensure_ascii=False,allow_nan=False);f.write('\n')
 print('CASE_END '+name+' passed='+str(row['passed']),flush=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=dict(schema='v21_same_problem_oracle_small_cross_v1',passed=all(r['passed'] for r in rows),checks=len(rows),rows=rows,oracle_sha256=sha(ROOT/'review/v21_same_problem_oracle_v1.py'),verifier_sha256=sha(ROOT/'review/v21_original_matrix_kkt_verifier_v1.py'),checker_sha256=sha(Path(__file__)),production_changes=0,physics_steps=0,solver_scope='three analytic cases only, original rows and numerical tolerances; not saved1377 tasks or physical performance')
with out.open('x',encoding='utf8') as f:json.dump(plain(r),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(passed=r['passed'],checks=len(rows),out=str(out),sha256=sha(out)),ensure_ascii=False),flush=True)
raise SystemExit(0 if r['passed'] else 1)

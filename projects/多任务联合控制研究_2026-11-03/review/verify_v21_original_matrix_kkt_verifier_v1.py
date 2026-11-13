"""Finite analytic oracles for the new review-only numerical KKT verifier."""
from pathlib import Path
import hashlib,importlib.util,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'review/v21_original_matrix_kkt_verifier_v1.py'
s=importlib.util.spec_from_file_location('independent_kkt',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rows=[]
def case(name,passed,r=None):
 rows.append(dict(name=name,passed=bool(passed),result=r));assert passed,name

def p(A,b,G,h,E=(),f=(),**extra):return dict(kind='least_squares',A=A,b=b,G=G,h=h,E=E,f=f,atol=1e-9,rtol=1e-8,**extra)
def d(lam,nu=()):return dict(inequality_unit=lam,equality_raw=nu,provenance='independent_analytic_current_candidate_not_historical')
x=p([[1.]],[2.],[[1.],[-1.]],[1.,0.]);r=m.verify_stage(x,[1.],duals=d([2.,0.]),comparison_x=[0.]);case('LS_box_projection_KKT_and_objective_improvement',r['kkt_passed'] and r['objective']==1. and r['comparison']['objective_delta']==-3.,r)
r=m.verify_stage(dict(x,solver_success=True),[.5],duals=d([0.,0.]));case('success_flag_cannot_hide_nonstationary_candidate',not r['kkt_passed'] and r['primal']['passed'],r)
r=m.verify_stage(p([[1.]],[0.],[[1.],[-1.]],[1.,0.]),[1.],duals=d([-2.,0.]));case('negative_dual_rejects_stationary_wrong_optimum',not r['kkt_passed'] and r['stationarity_inf']==0. and not r['dual_nonnegative'],r)
r=m.verify_stage(p([[1.]],[1.5],[[1.],[-1.]],[1.,0.]),[.5],duals=d([2.,0.]));case('inactive_nonzero_dual_rejected_complementarity_and_face',not r['kkt_passed'] and r['stationarity_inf']==0. and r['complementarity_inf']==1.,r)
q=p([[1.,0.]],[.25],[[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]],[1.,1.,0.,0.],[[1.,1.]],[1.],previous_frozen=[dict(matrix=[[1.,1.]],attained=[1.])]);r=m.verify_stage(q,[.25,.75],duals=d([0.]*4,[0.]));bad=m.verify_stage(q,[.25,.7],duals=d([0.]*4,[0.]));case('raw_equality_and_frozen_output_not_silently_eliminated',r['kkt_passed'] and not bad['kkt_passed'] and bad['primal']['previous_frozen_drift_inf']>.049,r)
r=m.verify_stage(p([[1.]],[1.],[],[],[[1.]],[.3]),[.3],duals=d([],[1.4]));case('all_coordinates_fixed_raw_equality_dual',r['kkt_passed'] and r['primal']['all_original_equalities_checked']==1,r)
base=dict(kind='linear',c=[-1.],G=[[1.],[1.],[-1.]],h=[.5,1.,0.],E=[],f=[],atol=1e-9,rtol=1e-8);lp=m.verify_stage(base,[.5],duals=d([1.,0.,0.]),comparison_x=[0.]);ls=m.verify_stage(p([[1.]],[1.],base['G'],base['h']),[.5],duals=d([1.,0.,0.]));case('scalar_withdraw_LP_and_original_monotone_LS',lp['kkt_passed'] and ls['kkt_passed'] and lp['comparison']['objective_delta']==-.5 and ls['objective']==.25,dict(lp=lp,ls=ls))
G=[[1.,1.],[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]];h=[1.,1.,1.,0.,0.];v=[[0.,0.],[1.,0.],[0.,1.]];b=dict(kind='linear',c=[-1.,-.3],G=G,h=h,E=[],f=[]);r=m.verify_stage(b,[1.,0.],duals=d([math.sqrt(2)*.3,.7,0.,0.,0.]));case('2D_vertex_oracle_with_coupled_hard_row',r['kkt_passed'] and r['objective']==min(np.dot(b['c'],z) for z in v),r)
r=m.verify_stage(p([[1.,0.],[0.,1.]],[1.,.4],G,h),[.8,.2],duals=d([math.sqrt(2)*.4,0.,0.,0.,0.]));case('coupled_convex_LS_face_interior_oracle',r['kkt_passed'] and abs(r['objective']-.08)<1e-15,r)
r=m.verify_stage(p([[0.]],[0.],[[0.]],[-1e-300]),[0.],duals=d([0.]));case('impossible_constant_hard_row_never_dropped',not r['kkt_passed'] and not r['primal']['constant_hard_rows_consistent'],r)
r=m.verify_stage(p([[0.,0.]],[0.],[[1e200,-1e200]],[0.]),[1.,1.],duals=d([0.]));case('extreme_finite_row_normalization_does_not_overflow_norm',r['kkt_passed'] and r['primal']['all_original_rows_checked']==1,r)
r=m.verify_stage(p([[0.]],[0.],[[1e-300]],[1e300]),[0.],duals=d([0.]));case('unrepresentable_scaling_is_unavailable_not_zero_gain',r['available'] is False and r['kkt_passed'] is False,r)
unknown=m.verify_stage(dict(x,G=None),[1.],duals=d([2.,0.]));badnum=m.verify_stage(x,['1.'],duals=d([2.,0.]));r=m.verify_stage(x,[1.]);case('unknown_strict_numeric_and_no_dual_not_certified',not unknown['available'] and not badnum['available'] and r['primal']['passed'] and not r['kkt_passed'],r)
a=m.verify_stage(x,[1.],duals=d([2.,0.]));b=m.verify_stage(x,[.5],comparison_x=[2.]);case('same_problem_hash_excludes_candidate_and_infeasible_comparison_has_no_delta',a['original_problem_sha256']==b['original_problem_sha256'] and b['comparison']['objective_delta'] is None,dict(a=a,b=b))
r=m.verify_stage(p(np.eye(21),np.zeros(21),[],[]),np.zeros(21),duals=d([]));case('explicit_known_empty_rows_at21_coordinates',r['kkt_passed'] and r['dimension']==21 and r['objective']==0.,r)
source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('v21_stable_handoff/prefix_hierarchy.py','v21_stable_handoff/task_solver.py')}
out=ROOT/'review/v21_original_matrix_kkt_verifier_author_attempt2.json';result=dict(schema='v21_review_only_original_matrix_kkt_verifier_author_v1',passed=all(z['passed'] for z in rows),checks=len(rows),cases=rows,checker_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frozen_mathematical_source_sha256=source,production_imports=[],solver_calls=0,model_constructions=0,physics_steps=0,scope='Finite analytic small cases only; no saved V21 mathematical problem or physical performance evaluated.')
with out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(passed=result['passed'],checks=len(rows),out=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),checker_sha256=result['checker_sha256']),ensure_ascii=False))

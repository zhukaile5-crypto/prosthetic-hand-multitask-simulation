"""Independent affine-coordinate candidate experiment on frozen V21 tasks.

Only original equalities are eliminated. Every original inequality remains in
the candidate problem, and all original rows/objectives/KKT are checked later.
No physical code, original certificate, objective, bound or regularizer changes.
"""
from pathlib import Path
from collections import Counter
import json, sys, time, traceback
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v21_same_problem_oracle_v1 import canonical_problem, verify_stage, recover_dual, solve_problem, plain, sha

ROOT=Path(__file__).resolve().parents[1]

def solve_quotient(problem):
    x0=np.asarray(problem['initial_x'],float)
    p=canonical_problem(problem,x0)
    old=verify_stage(problem,x0)
    if not old.get('available') or not old['primal']['passed']:
        return dict(status='ORIGINAL_ANCHOR_NOT_FEASIBLE',original_anchor=old)
    started=time.perf_counter()
    E,f=p['E'],p['f'];n=p['n']
    if len(E):
        u,s,vt=np.linalg.svd(E,full_matrices=True)
        threshold=max(E.shape)*np.finfo(float).eps*(s[0] if len(s) else 0.)
        rank=int(np.sum(s>threshold));Z=vt[rank:].T
        base=x0+vt[:rank].T@((u[:,:rank].T@(f-E@x0))/s[:rank])
    else:
        rank=0;Z=np.eye(n);base=x0.copy();threshold=0.
    G=p['G_unit']@Z;h=p['h_unit']-p['G_unit']@base
    A=p['A']@Z;b=p['b']-p['A']@base
    if Z.shape[1]:
        fun=lambda y:float((A@y-b)@(A@y-b))
        jac=lambda y:2*A.T@(A@y-b)
        constraints=[dict(type='ineq',fun=lambda y:h-G@y,jac=lambda y:-G)] if len(G) else []
        result=minimize(fun,np.zeros(Z.shape[1]),jac=jac,method='SLSQP',constraints=constraints,
            options={'ftol':1e-12,'maxiter':400,'disp':False})
        x=base+Z@result.x
        library=dict(success=bool(result.success),status=int(result.status),message=str(result.message),iterations=int(result.nit))
    else:
        x=base.copy();library=dict(success=None,status=None,message='Original affine domain has zero free coordinates; candidate is its base.',iterations=0)
    elapsed=time.perf_counter()-started
    primal=verify_stage(problem,x,comparison_x=x0)
    report=dict(method='original_equality_affine_quotient_SLSQP',x=x,library=library,
        primal_solver_elapsed_s=elapsed,original_anchor=old,primal_before_dual=primal,
        representation=dict(original_dimension=n,original_equality_rows=len(E),equality_rank=rank,
            free_dimension=Z.shape[1],svd_rank_threshold=threshold,
            original_inequality_rows=len(p['G']),transformed_inequality_rows=len(G),
            all_inequality_bounds_unchanged_under_declared_substitution=True,
            base_original_equality_residual_inf=float(np.max(abs(E@base-f),initial=0.)),
            nullspace_original_equality_residual_inf=float(np.max(abs(E@Z),initial=0.))),
        physics_steps=0,no_regularizer_added=True,original_problem_modified=False)
    if not primal.get('available') or not primal['primal']['passed']:
        return dict(report,status='INDEPENDENT_CANDIDATE_NOT_ORIGINAL_PRIMAL_FEASIBLE')
    dual,note=recover_dual(problem,x)
    verified=verify_stage(problem,x,duals=dual,comparison_x=x0)
    return dict(report,dual_witness=dual,dual_recovery=note,verification=verified,
        status='INDEPENDENT_ORIGINAL_PROBLEM_KKT_VERIFIED' if verified['kkt_passed'] else 'FEASIBLE_BUT_KKT_NOT_VERIFIED')

def main():
    corpus_path=ROOT/'review/v21_numerical_corpus_v1.json'
    prior_path=ROOT/'review/v21_same_problem_oracle_results_v1.json'
    out=ROOT/'review/v21_affine_quotient_oracle_results_v1.json'
    assert not out.exists(),'Preserve prior outcomes; use a fresh version.'
    corpus=json.loads(corpus_path.read_text(encoding='utf-8'))
    prior=json.loads(prior_path.read_text(encoding='utf-8'))
    assert prior['oracle_sha256']==sha(Path(__file__).with_name('v21_same_problem_oracle_v1.py'))
    rows=[]
    for sample,old in zip(corpus['samples'],prior['rows']):
        p=sample['failed_stage_problem']
        assert sample['context_key']==old['context_key']
        name=p['layer_name']
        try:
            row=solve_problem(p,scalar_withdrawal=True) if name=='scalar_pressure_share_normal_withdrawal' else solve_quotient(p)
        except Exception as exc:
            row=dict(status='ORACLE_ERROR',error=repr(exc),traceback=traceback.format_exc())
        row.update(context_key=sample['context_key'],layer_name=name,sample_role=sample['selection_kind'],
            source=sample['source'],original_coordinate_oracle_status=old['status'])
        rows.append(row)
        print(json.dumps({k:row[k] for k in ('context_key','layer_name','sample_role','status')},ensure_ascii=False),flush=True)
    payload=dict(schema='v21_affine_quotient_offline_oracle_v1',assigned=65,assessed=len(rows),rows=rows,
        counts=dict(Counter(row['status'] for row in rows)),corpus_sha256=sha(corpus_path),
        prior_oracle_result_sha256=sha(prior_path),quotient_source_sha256=sha(__file__),
        verifier_sha256=sha(Path(__file__).with_name('v21_original_matrix_kkt_verifier_v1.py')),
        physics_steps=0,production_changes=0,
        scope='Fixed49 first failures and16 first full-success controls; representation experiment, not original1377 or a physical performance result.')
    out.write_text(json.dumps(plain(payload),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(out),'counts':payload['counts']},ensure_ascii=False))

if __name__=='__main__':main()

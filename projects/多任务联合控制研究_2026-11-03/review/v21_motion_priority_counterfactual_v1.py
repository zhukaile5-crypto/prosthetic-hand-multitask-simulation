"""Falsifiable priority experiment, NOT a same-problem solver comparison.

For each of16 preselected L3 failures, remove ONLY the prior scalar task's
attained equality. Keep all physical/TASK G/h and fixed coordinates/eta. Find
a motion descent direction using a certified LP; evaluate its actual LS loss
along the feasible segment. No rollout or physical velocity prediction.
"""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v21_same_problem_oracle_v1 import canonical_problem,verify_stage,plain,sha

ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'review/v21_numerical_corpus_v1.json'
out=ROOT/'review/v21_motion_priority_counterfactual_v1.json'
assert not out.exists(),'Preserve all previous outcomes.'
corpus=json.loads(src.read_text(encoding='utf-8'))
samples=[s for s in corpus['samples'] if s['selection_kind']=='first_failed_layer' and s['failed_stage_problem']['layer_name']=='public_external_projection_low_motion']
assert len(samples)==16
rows=[]
for sample in samples:
    old=sample['failed_stage_problem'];n=old['variable_count']
    priors=old['previous_frozen']
    assert len(priors)==1 and priors[0]['name']=='scalar_pressure_share_normal_withdrawal'
    scalar=np.asarray(priors[0]['matrix'],float);attained=np.asarray(priors[0]['attained'],float)
    assert scalar.shape==(1,n)
    E=np.asarray(old['E'],float).reshape((-1,n));f=np.asarray(old['f'],float)
    assert np.array_equal(E[-1:],scalar) and np.array_equal(f[-1:],attained)
    p=dict(old,E=E[:-1].tolist(),f=f[:-1].tolist(),previous_frozen=[])
    x0=np.asarray(p['initial_x'],float);cp=canonical_problem(p,x0)
    r0=cp['A']@x0-cp['b'];gradient=2*cp['A'].T@r0
    linear=dict(p,kind='linear',c=gradient.tolist())
    lp=linprog(gradient,A_ub=cp['G_unit'],b_ub=cp['h_unit'],
        A_eq=cp['E'] if len(cp['E']) else None,b_eq=cp['f'] if len(cp['E']) else None,
        bounds=[(None,None)]*n,method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'maxiter':400})
    row=dict(context_key=sample['context_key'],source=sample['source'],library_success=bool(lp.success),
        library_message=str(lp.message),all_G_h_identical=True,only_prior_task_scalar_equality_removed=True,
        original_fixed_eta_equalities_retained=len(cp['E']),new_problem_is_explicit_priority_counterfactual=True)
    if lp.x is not None:
        witness=dict(inequality_unit=-lp.ineqlin.marginals,equality_raw=-lp.eqlin.marginals,
            provenance='HiGHS first-order direction LP marginals; current counterfactual only')
        cert=verify_stage(linear,lp.x,duals=witness,comparison_x=x0)
        d=lp.x-x0;ad=cp['A']@d;denom=float(ad@ad)
        alpha=float(np.clip(-float(r0@ad)/denom,0.,1.)) if denom else 0.
        x=x0+alpha*d
        point=verify_stage(p,x,comparison_x=x0)
        old_work=float((scalar@x0).item());new_work=float((scalar@x).item())
        row.update(direction_LP_KKT=cert,segment_alpha=alpha,segment_point=x,
            segment_original_hard_primal=point,original_motion_loss=float(r0@r0),
            counterfactual_motion_loss=point.get('objective'),
            scalar_attained_original=old_work,scalar_attained_counterfactual=new_work,
            scalar_task_tradeoff=new_work-old_work,
            interpretation='Feasible reference-intent tradeoff, not actual object motion or physical performance.')
    rows.append(row)
payload=dict(schema='v21_motion_priority_counterfactual_v1',assigned=16,assessed=len(rows),rows=rows,
    corpus_sha256=sha(src),source_sha256=sha(__file__),physics_steps=0,production_changes=0,
    scope='Every one of the16 fixed first-L3-failure contexts. Does not alter frozen original results or count a counterfactual as an observed success.')
out.write_text(json.dumps(plain(payload),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'out':str(out),'rows':len(rows),
    'direction_LP_KKT':sum(r.get('direction_LP_KKT',{}).get('kkt_passed',False) for r in rows),
    'feasible_segment':sum(r.get('segment_original_hard_primal',{}).get('primal',{}).get('passed',False) for r in rows),
    'losses':[[r['context_key'],r.get('original_motion_loss'),r.get('counterfactual_motion_loss'),r.get('scalar_task_tradeoff')] for r in rows]},ensure_ascii=False))

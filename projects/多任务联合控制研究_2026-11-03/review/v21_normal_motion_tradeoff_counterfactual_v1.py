"""Explicit TASK tradeoff diagnostic, never a safety or physical rollout claim.

Keep original21 mechanical/rate bounds, every normal inequality and the exact
motion budget. Release only the L1 optimum-slack freeze plus L2 achievement;
restore eta's ORIGINAL derived L1 bounds. Quantify achievable motion descent
and its original normal-correction task cost. No force limit changes.
"""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v21_same_problem_oracle_v1 import canonical_problem,verify_stage,plain,sha
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'review/v21_numerical_corpus_v1.json'
out=ROOT/'review/v21_normal_motion_tradeoff_counterfactual_v1.json'
assert not out.exists(),'Preserve previous outcomes.'
corpus=json.loads(src.read_text(encoding='utf-8'))
samples=[s for s in corpus['samples'] if s['selection_kind']=='first_failed_layer' and s['failed_stage_problem']['layer_name']=='public_external_projection_low_motion']
assert len(samples)==16
rows=[]
for s in samples:
    old=s['failed_stage_problem'];n=old['variable_count'];first=s['stage_problems'][0]
    assert first['layer_name']=='normal_inward_correction_slack'
    assert np.array_equal(np.asarray(old['lower'])[:21],np.asarray(first['lower'])[:21])
    assert np.array_equal(np.asarray(old['upper'])[:21],np.asarray(first['upper'])[:21])
    G=np.vstack((np.asarray(old['G'],float).reshape((-1,n)),np.asarray(first['G'],float).reshape((-1,n))))
    h=np.r_[old['h'],first['h']]
    p=dict(old,G=G.tolist(),h=h.tolist(),E=first['E'],f=first['f'],previous_frozen=[],
        lower=first['lower'],upper=first['upper'])
    x0=np.asarray(old['initial_x'],float);cp=canonical_problem(p,x0)
    r0=cp['A']@x0-cp['b'];gradient=2*cp['A'].T@r0
    linear=dict(p,kind='linear',c=gradient.tolist())
    lp=linprog(gradient,A_ub=cp['G_unit'],b_ub=cp['h_unit'],
        A_eq=cp['E'] if len(cp['E']) else None,b_eq=cp['f'] if len(cp['E']) else None,
        bounds=[(None,None)]*n,method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'maxiter':400})
    row=dict(context_key=s['context_key'],source=s['source'],library_success=bool(lp.success),library_message=str(lp.message),
        original21_mechanical_rate_bounds_unchanged=True,original_normal_inequalities_retained=True,
        original_motion_budget_unchanged=True,eta_original_L1_bounds_restored=True,
        original_fixed_eta_optimum_and_scalar_task_freezes_removed=True,
        unknown_future_force_or_motion_not_certified=True)
    if lp.x is not None:
        cert=verify_stage(linear,lp.x,duals=dict(inequality_unit=-lp.ineqlin.marginals,equality_raw=-lp.eqlin.marginals,
            provenance='HiGHS marginals for explicit normal/motion task tradeoff counterfactual'),comparison_x=x0)
        d=lp.x-x0;ad=cp['A']@d;denom=float(ad@ad)
        alpha=float(np.clip(-float(r0@ad)/denom,0.,1.)) if denom else 0.
        x=x0+alpha*d;point=verify_stage(p,x,comparison_x=x0)
        a1=np.asarray(first['A'],float);b1=np.asarray(first['b'],float)
        normal0=a1@x0-b1;normal1=a1@x-b1
        row.update(direction_LP_KKT=cert,segment_alpha=alpha,segment_point=x,segment_primal=point,
            original_motion_loss=float(r0@r0),counterfactual_motion_loss=point.get('objective'),
            normal_correction_loss_original=float(normal0@normal0),normal_correction_loss_counterfactual=float(normal1@normal1))
    rows.append(row)
payload=dict(schema='v21_normal_motion_tradeoff_counterfactual_v1',assigned=16,assessed=len(rows),rows=rows,
    corpus_sha256=sha(src),source_sha256=sha(__file__),physics_steps=0,production_changes=0,
    scope='Fixed16 first L3 failures; explicit task-achievement tradeoff with original mechanical constraints, not a new observed outcome or force-safety guarantee.')
out.write_text(json.dumps(plain(payload),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'out':str(out),'rows':len(rows),'direction_KKT':sum(r.get('direction_LP_KKT',{}).get('kkt_passed',False) for r in rows),
    'segment_feasible':sum(r.get('segment_primal',{}).get('primal',{}).get('passed',False) for r in rows),
    'tradeoffs':[[r['context_key'],r.get('original_motion_loss'),r.get('counterfactual_motion_loss'),r.get('normal_correction_loss_original'),r.get('normal_correction_loss_counterfactual')] for r in rows]},ensure_ascii=False))

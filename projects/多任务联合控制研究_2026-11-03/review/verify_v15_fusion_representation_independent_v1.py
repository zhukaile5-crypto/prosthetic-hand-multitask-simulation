"""Small independent split/merge/pullback oracles, run only at an explicit SHA.

No solver, robot model, public packet replay or physics. The representation map
is proportional, same-digit/group, same-geometry subdivision, not arbitrary
contact merging or a claim of physical invariance under a changed contact set.
"""
from pathlib import Path
import argparse, hashlib, importlib, json, sys, traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(build):
    checks=[]
    def check(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:raise AssertionError(name+' '+str(detail))
    def close(name,a,b):
        a=np.asarray(a,float);b=np.asarray(b,float)
        gap=float(np.max(np.abs(a-b))) if a.size else 0.
        check(name,np.allclose(a,b,atol=3e-13,rtol=3e-13),dict(max_absolute_difference=gap))
    y=np.array([1.,3.,2.,4.,1.,.6]);z=np.array([1.4,2.2,1.1,3.5,.7,.9])
    d=np.array([0,0,1,2,3,0]);selected=np.array([True]*5+[False])
    points=[np.array(v,float) for v in [[.02,.03,.01],[-.01,.02,.04],[.04,-.02,.02],[-.02,-.01,.03],[.01,.04,-.01]]]+[None]
    normals=[np.array(v,float) for v in [[0,0,1],[.6,0,.8],[0,.8,.6],[-.8,0,.6],[0,-.6,.8]]]+[None]
    base=dict(patch_digit_p=d,normal_p=y,reference_point_p3=points,reference_normal_p3=normals,selected_p=selected,
        previous_conditional_force5=np.array([4.,1.5,3.,.8,0.]),caps5=np.array([5.,3.,4.,4.,2.]),
        force_cost=.07,reference_continuity=.4,balance_weight=.13,support_contact_floor=.4,characteristic_length=.05)
    old=build(**base);check('base_available',old.available,old.to_dict() if not old.available else None)
    # Split one selected and one foreign patch independently; retain other rows.
    parents=np.array([0,0,1,2,3,4,5,5,5]);weights=np.array([.23,.77,1,1,1,1,.2,.3,.5])
    T=np.zeros((len(parents),len(y)));T[np.arange(len(parents)),parents]=weights
    changed={**base,'patch_digit_p':d[parents],'normal_p':T@y,'selected_p':selected[parents],
        'reference_point_p3':[points[i] for i in parents],'reference_normal_p3':[normals[i] for i in parents]}
    new=build(**changed);check('selected_and_foreign_split_available',new.available)
    zn=T@z
    cost=lambda s,v:float(np.sum((s.intentP*(v-s.base_prior_p))**2))
    grad=lambda s,v:2*s.intentP**2*(v-s.base_prior_p)
    close('raw_prior_proportional_pullforward',new.base_prior_p,T@old.base_prior_p)
    close('economy_cost_split_equal',cost(new,zn),cost(old,z))
    close('economy_gradient_pullback',T.T@grad(new,zn),grad(old,z))
    close('economy_hessian_pullback',T.T@np.diag(2*new.intentP**2)@T,np.diag(2*old.intentP**2))
    close('pressure_weighted_public_anchor_invariant',new.metadata['center_public_pressure_weighted_anchor'],old.metadata['center_public_pressure_weighted_anchor'])
    Bo=np.array(old.metadata['conditional_digit_sum_map']);Bn=np.array(new.metadata['conditional_digit_sum_map'])
    Wo=np.array(old.metadata['unweighted_balance_map']);Wn=np.array(new.metadata['unweighted_balance_map'])
    close('conditional_digit_sum_map_pullback',Bn@T,Bo)
    close('balance_map_pullback',Wn@T,Wo)
    L=np.zeros((21+len(parents),21+len(y)));L[:21,:21]=np.eye(21);L[21:,21:]=T
    close('continuity_and_balance_row_pullback',new.extra_matrix@L,old.extra_matrix)
    close('continuity_and_balance_targets_equal',new.extra_target,old.extra_target)
    x=np.r_[np.linspace(-.2,.1,21),z]
    close('all_extra_task_residuals_equal',new.extra_matrix@(L@x)-new.extra_target,old.extra_matrix@x-old.extra_target)
    # J is independent synthetic public robot geometry; duplicate a parent's row.
    J=np.arange(6*25,dtype=float).reshape(6,25)/137.-.4;Jn=J[parents]
    close('jt_normal_response_under_subdivision',Jn.T@((T@z)-(T@y)),J.T@(z-y))
    # Merge is equivalent only for proportional values and identical geometry.
    recovered=np.zeros_like(z)
    np.add.at(recovered,parents,zn)
    close('reverse_proportional_merge_restores_intent',recovered,z)
    close('reverse_proportional_merge_cost_equal',cost(old,recovered),cost(new,zn))
    # Deforming child allocations is a different force measure, not an invariance.
    nonproportional=zn.copy();nonproportional[0]+=.1;nonproportional[1]-=.1
    check('nonproportional_subpatch_reallocation_not_falsely_identical_cost',cost(new,nonproportional)>cost(new,zn))
    # No selected economy dilution by splitting a foreign row; missing foreign
    # geometry is not an inferred physical zero and must retain explicit status.
    close('selected_multiplier_after_foreign_only_change',new.intentP[2:6],old.intentP[1:5])
    check('foreign_geometry_none_is_inactive_unknown_not_known_zero',
        all(new.metadata['selected_reference_point_status'][i]=='unknown_not_used' and new.metadata['selected_reference_normal_status'][i]=='unknown_not_used' for i in (6,7,8))
        and np.array_equal(Bn[:,6:],np.zeros((5,3))) and np.array_equal(Wn[:,6:],np.zeros((5,3))))
    missing_selected=points.copy();missing_selected[0]=None
    unknown=build(**{**base,'reference_point_p3':missing_selected})
    check('selected_geometry_unknown_rejected',not unknown.available and unknown.status=='UNAVAILABLE_UNKNOWN',dict(status=unknown.status,reason=unknown.reason))
    yy=y.copy();yy[-1]=0.
    zero=build(**{**base,'normal_p':yy})
    check('zero_foreign_share_not_epsilon_repaired',not zero.available,dict(status=zero.status,reason=zero.reason))
    tiny=y.copy();tiny[0]=np.nextafter(0.,1.);tiny[1]=1e308
    under=build(**{**base,'normal_p':tiny})
    check('unrepresentable_positive_share_not_clamped',not under.available,dict(status=under.status,reason=under.reason))
    representable=y.copy();representable[0]=1e-100
    small=build(**{**base,'normal_p':representable})
    check('positive_representable_share_retained_without_arbitrary_floor',small.available and small.metadata['observed_pressure_share_p'][0]>0 and np.isfinite(small.intentP).all())
    rawfloor=build(**{**base,'support_contact_floor':20.})
    check('raw_prior_above_caps_not_retargeted',rawfloor.available and rawfloor.metadata['floor_was_cap_or_budget_retargeted'] is False and np.allclose(Bo@rawfloor.base_prior_p,[20,20,20,20,0]))
    return dict(passed=True,checks=checks,transform=T.tolist(),base_normal=y.tolist(),base_intent=z.tolist(),
        economy=dict(original=cost(old,z),split=cost(new,zn)),
        scope='Small synthetic pure task-row construction. Proportional subdivision/merge only; no solver, native effort mapping, hardware, physics, controller integration or safety certification.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--expected-sha',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    source=ROOT/'v15_support/fusion_tasks.py';out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    before=sha(source)
    if before!=args.expected_sha:raise RuntimeError('Source version mismatch; refusing to import changing source')
    sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
    assert 'mujoco' not in sys.modules
    try:r=run(importlib.import_module('v15_support.fusion_tasks').build_fusion_tasks)
    except Exception as e:r=dict(passed=False,error=repr(e),traceback=traceback.format_exc())
    r.update(schema='v15_fusion_representation_independent_v1',source_path=str(source),source_sha256=before,source_unchanged=sha(source)==before,
             checker_sha256=sha(__file__),physics_steps=0,model_imports=0,mujoco_imported='mujoco' in sys.modules,
             independent_scope=True)
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=r['passed'],checks=len(r.get('checks',[])),sha256=sha(out),path=str(out)),ensure_ascii=False))
    raise SystemExit(0 if r['passed'] else 1)

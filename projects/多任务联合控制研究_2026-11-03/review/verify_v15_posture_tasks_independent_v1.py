"""Independent closed-form holding task checks; NumPy only, no model/solver."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, traceback, warnings
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(build):
    checks=[]
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def close(name,a,b):
        a=np.asarray(a,float);b=np.asarray(b,float)
        check(name,np.allclose(a,b,rtol=2e-13,atol=2e-13),dict(max_difference=float(np.max(np.abs(a-b))) if a.size else 0.))
    def reject(name,fn):
        try:fn()
        except (ValueError,TypeError):check(name,True)
        else:check(name,False)
    span=np.linspace(1.,5.,21);g=np.linspace(2.,6.,21);g[[0,2,7,12,16]]*=-1
    opened=np.linspace(-.1,.1,18);closed=opened+span[:18]*np.tile([1.,2.,-1.],6)
    nominal=np.linspace(.15,.35,21);anchor=nominal+np.linspace(-.12,.14,21)
    p0=np.linspace(4.,8.,21);sd=np.linspace(.3,1.3,21)
    base=dict(patch_count=3,time_s=8.,p0_21=p0,delta_scale21=sd,nominal_reference21=nominal,grasp_anchor21=anchor,
        native_reference_gain21=g,known_affine21=np.ones(21,dtype=bool),control_range21=np.c_[np.zeros(21),span],open18=opened,closed18=closed)
    r=build(**base)
    check('ordinary_all_finger_tasks_available_and_no_effort_history_invented',r.available and r.matrix.shape==(18,24) and r.metadata['history_task_status']=='no_past_public_model_sample_inactive')
    # Closed form of the first finger in span coordinates, independent of factory.
    v=(closed[:5]-opened[:5])/span[:5];unit=v/np.linalg.norm(v);P=np.eye(5)-unit[:,None]*unit[None,:]
    R=P@np.diag(1/span[:5]);E=R@np.diag(1/g[:5]);b=R@(anchor[:5]-nominal[:5])
    close('TH_closed_form_signed_inverse_and_span_order',r.matrix[:5,:5],E)
    close('TH_nonzero_nonnominal_anchor_target',r.target[:5],b)
    check('anchor_target_not_silently_zero',np.linalg.norm(b)>1e-3)
    dr=anchor[:18]-nominal[:18]+.37*(closed-opened)
    d=np.r_[g[:18]*dr,np.zeros(3),np.zeros(3)]
    close('full_manifold_reference_direction_is_null_even_with_negative_gains',r.matrix@d-r.target,np.zeros(18))
    q=np.r_[np.linspace(-.2,.4,21),np.array([2.,3.,4.])]
    close('TH_residual_direct_reference_oracle',r.matrix[:5]@q-r.target[:5],P@((nominal[:5]+q[:5]/g[:5]-anchor[:5])/span[:5]))
    check('no_lift_wrist_or_patch_columns_in_shape_rows',np.array_equal(r.matrix[:,18:],np.zeros((18,6))))
    fixed=closed.copy();fixed[:5]=opened[:5]
    f=build(**{**base,'closed18':fixed})
    close('exact_fixed_template_uses_full_complement_not_unknown_inverse',f.matrix[:5,:5],np.diag(1/(span[:5]*g[:5])))
    close('fixed_template_target_is_actual_anchor_minus_nominal',f.target[:5],(anchor[:5]-nominal[:5])/span[:5])
    mask=np.ones(21,dtype=bool);mask[2]=False
    u=build(**{**base,'known_affine21':mask})
    check('one_unknown_axis_disables_whole_TH_group',u.available and u.matrix.shape[0]==13 and u.metadata['inactive_shape_groups'][0]['digit']==0 and np.array_equal(u.matrix[:,:5],np.zeros((13,5))))
    gz=g.copy();gz[7]=0.;flat=build(**{**base,'native_reference_gain21':gz})
    check('known_flat_gain_disables_whole_digit_without_zero_inverse',flat.matrix.shape[0]==15 and flat.metadata['inactive_shape_groups'][0]['digit']==1)
    rng=base['control_range21'].copy();rng[8,1]=rng[8,0];zs=build(**{**base,'control_range21':rng})
    check('zero_control_span_disables_whole_digit',zs.matrix.shape[0]==15 and zs.metadata['inactive_shape_groups'][0]['digit']==2)
    missing=build(**{**base,'grasp_anchor21':None})
    check('unknown_anchor_no_fabricated_shape_or_history',missing.matrix.shape==(0,24) and missing.metadata['shape_task_status']=='actual_grasp_anchor_unknown_inactive')
    previous=p0+np.linspace(-.4,.7,21)
    h=build(**{**base,'previous_model_effort21':previous,'previous_model_time_s':7.995})
    delta=np.linspace(-.7,.2,21);x=np.r_[delta,[.8,.2,.4]]
    close('history_residual_includes_nonzero_p0_and_actual_signed_delta',h.matrix[-21:]@x-h.target[-21:],sd*(p0+delta-previous))
    check('history_clock_and_model_role_explicit',h.metadata['history_task_status']=='past_model_sample_active' and h.metadata['history_sample_time_s']==7.995 and 'not_independent_torque' in h.metadata['history_role'])
    same=build(**{**base,'previous_model_effort21':previous,'previous_model_time_s':8.})
    check('same_clock_history_inactive',same.matrix.shape==r.matrix.shape and same.metadata['history_task_status']=='same_clock_sample_not_a_past_history_task')
    reject('future_history_rejected',lambda:build(**{**base,'previous_model_effort21':previous,'previous_model_time_s':8.005}))
    reject('missing_history_value_not_zero_imputed',lambda:build(**{**base,'previous_model_time_s':7.995}))
    reject('history_sample_without_time_rejected',lambda:build(**{**base,'previous_model_effort21':previous}))
    reject('bool_patch_count_not_numeric_permission',lambda:build(**{**base,'patch_count':True}))
    reject('nonboolean_affine_flags_rejected',lambda:build(**{**base,'known_affine21':np.ones(21)}))
    bad=list(p0);bad[0]=True;reject('bool_in_effort_rejected',lambda:build(**{**base,'p0_21':bad}))
    check('returned_rows_and_targets_readonly_by_default',not r.matrix.flags.writeable and not r.target.flags.writeable)
    check('shape_object_does_not_claim_physical_or_pressure_certificate',r.metadata['physical_future_pose_certified'] is False and r.metadata['pressure_prediction_available'] is False)
    # Extreme finite valid arrays must not turn a nonzero declared direction into
    # an exact fixed template. A truthful input-domain rejection is also acceptable.
    simple={**base,'patch_count':0,'p0_21':np.zeros(21),'nominal_reference21':np.zeros(21),'grasp_anchor21':np.zeros(21),
        'native_reference_gain21':np.ones(21),'control_range21':np.tile([0.,1.],(21,1)),'open18':np.zeros(18),'closed18':np.ones(18)}
    tiny=np.ones(18);tiny[:5]=[1e-200,0,0,0,0];probe=np.zeros(21);probe[0]=1.
    try:
        t=build(**{**simple,'closed18':tiny});res=(t.matrix@probe-t.target)[:5]
        check('tiny_nonzero_direction_not_misclassified_fixed',np.allclose(res,0,atol=1e-14,rtol=0),dict(direction=tiny[:5].tolist(),residual=res.tolist()))
    except ValueError as exc:check('tiny_nonzero_direction_not_misclassified_fixed',True,dict(explicit_unavailable=str(exc)))
    wide=simple['control_range21'].copy();wide[:5]=[-1e308,1e308]
    try:
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):w=build(**{**simple,'control_range21':wide})
        rows=w.matrix[:5,:5]
        inactive=any(group['digit']==0 for group in w.metadata['inactive_shape_groups'])
        check('unrepresentable_control_span_not_silently_known_zero_task',inactive and not any(label.startswith('grasp_shape_digit0_') for label in w.labels),dict(status=w.status,first_group_rows=rows.tolist(),TH_explicitly_inactive=inactive))
    except ValueError as exc:check('unrepresentable_control_span_not_silently_known_zero_task',True,dict(explicit_unavailable=str(exc)))
    return dict(passed=all(c['passed'] for c in checks),checks=checks,
        scope_limits=['No native affine branch/mode/complete interval admission is proved here; caller owns it.',
            'No holding/release lifecycle routing, execution or physical pose/force behavior is tested.',
            'Read-only arrays are default write-protected copies, not immutable/sealed provenance objects; metadata is mutable.'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--expected-sha',required=True);ap.add_argument('--out',required=True);ap.add_argument('--source',default='v15_support/posture_tasks.py');args=ap.parse_args()
    source=ROOT/args.source;out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    before=sha(source)
    if before!=args.expected_sha:raise RuntimeError('Source SHA mismatch')
    sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
    try:
        spec=importlib.util.spec_from_file_location('posture_source_under_independent_review',source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        r=run(module.build_holding_native_tasks)
    except Exception as exc:r=dict(passed=False,error=repr(exc),traceback=traceback.format_exc())
    r.update(schema='v15_posture_tasks_independent_v1',source_path=str(source),source_sha256=before,source_unchanged=sha(source)==before,checker_sha256=sha(__file__),
        physics_steps=0,model_imports=0,mujoco_imported='mujoco' in sys.modules)
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=r['passed'],checks=len(r.get('checks',[])),failed=[c['name'] for c in r.get('checks',[]) if not c['passed']],sha256=sha(out),path=str(out)),ensure_ascii=False))
    raise SystemExit(0 if r['passed'] else 1)

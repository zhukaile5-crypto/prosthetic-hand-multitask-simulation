"""Pure objective expansion and geometry/identity-boundary tests; no solver."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'review/v15_fusion_tasks_author_v2.json'
    if out.exists():raise FileExistsError('Fresh task-row evidence required')
    source=ROOT/'v15_support/fusion_tasks.py';before=sha(source)
    spec=importlib.util.spec_from_file_location('pure_fusion_rows_author_v1',source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    build=module.build_fusion_tasks;checks=[];records=[]
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    def fixture():
        return dict(patch_digit_p=np.array([0,1,2,0,4]),normal_p=np.array([2.,1.,3.,1.,5.]),
            reference_point_p3=[[-.04,0.,.01],[.04,0.,.01],[0.,.04,.01],[-.04,.01,.01],None],
            reference_normal_p3=[[1.,0.,0.],[-1.,0.,0.],[0.,-1.,0.],[1.,0.,0.],[np.nan,np.nan,np.nan]],
            selected_p=np.array([True,True,True,True,False]),previous_conditional_force5=np.array([.7,1.2,.8,2.,3.]),caps5=np.array([2.,.5,4.,5.,0.]),
            force_cost=.4,reference_continuity=3.,balance_weight=8.,support_contact_floor=.8,characteristic_length=.05)
    k=fixture();r=build(**k);check('fixture_available_with_missing_external_geometry',r.available,r.reason)
    check('foreign_geometry_explicitly_unknown_not_used',r.metadata['selected_reference_point_status'][-1]=='unknown_not_used' and r.metadata['selected_reference_normal_status'][-1]=='unknown_not_used')
    check('base_floor_raw_share_no_cap_projection',np.allclose(r.base_prior_p,[.8*2/3,.8,.8,.8/3,0.],rtol=0,atol=1e-16) and r.base_prior_p[1]>k['caps5'][1] and r.metadata['floor_preference_above_cap5'][1])
    check('allP_multiplier_including_foreign_positive',np.array_equal(r.intentP,np.sqrt(.4)/np.maximum(1.,k['caps5'][k['patch_digit_p']])))
    check('d21_block_always_zero',np.array_equal(r.extra_matrix[:,:21],np.zeros((10,21))))
    check('foreign_extra_column_inactive_not_false_geometry',np.array_equal(r.extra_matrix[:,25],np.zeros(10)) and r.base_prior_p[4]==0. and 'never known zero geometry' in r.metadata['external_zero_row_role'])
    check('ten_rows_and_labels',r.extra_matrix.shape==(10,26) and r.extra_target.shape==(10,) and len(r.labels)==10)
    check('outputs_readonly',all(not x.flags.writeable for x in (r.base_prior_p,r.intentP,r.extra_matrix,r.extra_target)))
    a=r.extra_task_rows;b=r.extra_task_rows;a['matrix']=None;check('fresh_mapping_cannot_replace_spec_matrix',b['matrix'] is r.extra_matrix and r.extra_task_rows['matrix'] is r.extra_matrix)
    check('json_serializable_unknown_metadata',bool(json.dumps(r.to_dict(),allow_nan=False)))
    # Direct scalar/vector cost oracle, without the returned B or W matrices.
    selected=np.flatnonzero(k['selected_p']);point=np.asarray(k['reference_point_p3'][:4]);normal=np.asarray(k['reference_normal_p3'][:4]);center=point.mean(0);F=max(1.,k['caps5'].mean());rng=np.random.default_rng(151007)
    def oracle(x):
        z=x[21:];group=np.zeros(5);force=np.zeros(3);moment=np.zeros(3)
        for index in selected:
            group[k['patch_digit_p'][index]]+=z[index];force+=z[index]*normal[index];moment+=z[index]*np.cross(point[index]-center,normal[index])/k['characteristic_length']
        prior=np.array([.8*2/3,.8,.8,.8/3,0.]);economy=k['force_cost']*np.sum(np.square((z-prior)/np.maximum(1.,k['caps5'][k['patch_digit_p']])))
        continuity=k['reference_continuity']*np.sum(np.square((group-k['previous_conditional_force5'])/np.maximum(1.,k['caps5'])))
        balance=k['balance_weight']/F**2*(force[:2]@force[:2]+moment@moment)
        return float(economy+continuity+balance)
    def represented(x):return float(np.square(r.intentP*(x[21:]-r.base_prior_p)).sum()+np.square(r.extra_matrix@x-r.extra_target).sum())
    for i in range(8):
        x=rng.normal(0,1,26);value=represented(x);expected=oracle(x)
        check('unified_cost_expansion_'+str(i),abs(value-expected)<1e-10*max(1.,abs(expected)),value-expected)
        g=np.r_[np.zeros(21),2*r.intentP**2*(x[21:]-r.base_prior_p)]+2*r.extra_matrix.T@(r.extra_matrix@x-r.extra_target);fd=np.empty(26)
        for j in range(26):
            h=1e-6*max(1.,abs(x[j]));plus=x.copy();minus=x.copy();plus[j]+=h;minus[j]-=h;fd[j]=(oracle(plus)-oracle(minus))/(2*h)
        check('full_cost_independent_FD_'+str(i),np.max(abs(g-fd))<2e-6,float(np.max(abs(g-fd))))
    init=fixture();init['previous_conditional_force5']=None;ri=build(**init)
    check('first_previous_is_group_observation_not_floor',ri.metadata['previous_conditional_force5']==[3.,1.,3.,0.,0.] and ri.metadata['previous_source']=='observed_conditional_group_initialization')
    # Splitting one patch changes neither the digit continuity residual nor its
    # target for an identical total z. No claim about all other patchwise costs.
    continuous=fixture();continuous['balance_weight']=0.;rr=build(**continuous)
    split=dict(continuous,patch_digit_p=np.array([0,0,1,2,0,4]),normal_p=np.array([1.,1.,1.,3.,1.,5.]),
        reference_point_p3=[[-.04,0.,.01],[-.04,0.,.01],[.04,0.,.01],[0.,.04,.01],[-.04,.01,.01],None],
        reference_normal_p3=[[1.,0.,0.],[1.,0.,0.],[-1.,0.,0.],[0.,-1.,0.],[1.,0.,0.],None],selected_p=np.array([True,True,True,True,True,False]))
    rs=build(**split);z=np.array([.5,.7,.8,.4,1.2]);zs=np.array([.2,.3,.7,.8,.4,1.2])
    check('continuity_invariant_to_same_digit_patch_split',rs.available and np.allclose(rr.extra_matrix@np.r_[np.zeros(21),z]-rr.extra_target,rs.extra_matrix@np.r_[np.zeros(21),zs]-rs.extra_target,atol=1e-15,rtol=0))
    # Symmetric normal resultant and signed lever moments cancel analytically.
    symmetric=dict(patch_digit_p=np.array([0,1,2,2]),normal_p=np.ones(4),reference_point_p3=np.array([[-.02,-.02,0.],[.02,.02,0.],[-.02,.02,0.],[.02,-.02,0.]]),reference_normal_p3=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]]),selected_p=np.ones(4,bool),previous_conditional_force5=None,caps5=np.ones(5)*5,force_cost=1.,reference_continuity=0.,balance_weight=1.,support_contact_floor=.1,characteristic_length=.05)
    sr=build(**symmetric);w=np.asarray(sr.metadata['unweighted_balance_map'])
    check('symmetric_center_is_public_mean',[0.,0.,0.]==sr.metadata['center_public_point_mean'])
    check('symmetric_wrench_cancels',np.max(abs(w@np.ones(4)))<1e-15)
    check('moment_sign_length_convention',np.allclose(w[4],[.4,.4,-.4,-.4],atol=1e-15,rtol=0))
    for continuity,balance,count in ((0.,0.,0),(0.,2.,5),(2.,0.,5)):
        x=fixture();x['reference_continuity']=continuity;x['balance_weight']=balance;s=build(**x)
        check('zero_weight_rows_skip_'+str(count)+'_'+str(continuity),s.available and s.extra_matrix.shape==(count,26) and s.extra_target.shape==(count,))
    bad=[]
    def change(name,key,value,status=None):
        x=fixture();x[key]=value;bad.append((name,x,status))
    change('all_normal_unknown','normal_p',None,'UNAVAILABLE_UNKNOWN');change('foreign_normal_still_must_be_valid','normal_p',np.array([2.,1.,3.,1.,np.nan]),'UNAVAILABLE_UNKNOWN')
    change('foreign_digit_still_must_be_valid','patch_digit_p',np.array([0,1,2,0,8]),'UNAVAILABLE_INPUT');change('integer_selected_mask_refused','selected_p',np.array([1,1,1,1,0]),'UNAVAILABLE_INPUT')
    change('selected_point_missing','reference_point_p3',[None,[.04,0,.01],[0,.04,.01],[-.04,.01,.01],None],'UNAVAILABLE_UNKNOWN')
    change('selected_normal_missing','reference_normal_p3',[[1.,0,0],None,[0,-1.,0],[1.,0,0],None],'UNAVAILABLE_UNKNOWN')
    change('selected_zero_normal','reference_normal_p3',[[1.,0,0],[-1.,0,0],[0,0,0],[1.,0,0],None],'UNAVAILABLE_DEGENERATE_GEOMETRY')
    change('nonunit_selected_normal','reference_normal_p3',[[2.,0,0],[-1.,0,0],[0,-1.,0],[1.,0,0],None],'UNAVAILABLE_INPUT')
    change('no_thumb_group','selected_p',np.array([False,True,True,False,True]),'UNAVAILABLE_CONDITIONAL_GROUP')
    change('only_two_digits_not_three_patches','selected_p',np.array([True,True,False,True,False]),'UNAVAILABLE_CONDITIONAL_GROUP')
    change('zero_share_denominator','normal_p',np.array([0.,1.,3.,0.,5.]),'UNAVAILABLE_DEGENERATE_GROUP')
    change('zero_force_cost','force_cost',0.,'UNAVAILABLE_INPUT');change('negative_balance','balance_weight',-1.,'UNAVAILABLE_INPUT');change('unknown_continuity','reference_continuity',None,'UNAVAILABLE_UNKNOWN');change('zero_length','characteristic_length',0.,'UNAVAILABLE_INPUT');change('unknown_prev_entry','previous_conditional_force5',np.full(5,np.nan),'UNAVAILABLE_UNKNOWN');change('negative_prev','previous_conditional_force5',np.array([1.,1.,-1.,0.,0.]),'UNAVAILABLE_INPUT')
    for name,x,status in bad:
        s=build(**x);check(name,s.status==status and not s.available and all(v is None for v in (s.base_prior_p,s.intentP,s.extra_matrix,s.extra_target)),dict(status=s.status,reason=s.reason));records.append(dict(name=name,result=s.to_dict()))
    zero_share=fixture();zero_share['normal_p'][0]=0.;s=build(**zero_share);check('zero_member_but_positive_digit_share_is_known',s.available and s.base_prior_p[0]==0. and s.base_prior_p[3]==.8)
    capzero=fixture();capzero['caps5'][0]=0.;s=build(**capzero);check('known_zero_cap_does_not_retarget_soft_floor',s.available and s.base_prior_p[0]+s.base_prior_p[3]==.8 and s.metadata['floor_preference_above_cap5'][0])
    unknown_external=fixture();unknown_external['reference_point_p3'][-1]=[np.inf,1.,2.];unknown_external['reference_normal_p3'][-1]=None;s=build(**unknown_external);check('unknown_external_geometry_does_not_cancel_fusion',s.available and s.metadata['selected_reference_point_status'][-1]=='unknown_not_used' and s.metadata['selected_reference_normal_status'][-1]=='unknown_not_used')
    tree=ast.parse(source.read_text('utf-8'));names=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):names.extend(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom):names.append(node.module or '')
    check('imports_only_pure_utilities',set(names)<={'dataclasses','numbers','typing','numpy'},names)
    check('no_solver_model_or_simulator_loaded',all(x not in sys.modules for x in ('mujoco','hand_plant','v15_support.task_qp','v14_coord.intent_qp','v13_joint.intent_qp','vendor.frozen_controllers')))
    check('source_unchanged',sha(source)==before)
    output=dict(schema='v15_fusion_rows_author_v2',passed=all(x['passed'] for x in checks),check_count=len(checks),failures=[x for x in checks if not x['passed']],checks=checks,
        source_sha256=before,checker_sha256=sha(__file__),primary_spec=r.to_dict(),negative_cases=records,physics_steps=0,solver_calls=0,
        prior_attempt=dict(path=str(ROOT/'review/v15_fusion_tasks_author_v1.json'),sha256=sha(ROOT/'review/v15_fusion_tasks_author_v1.json'),reason='Prior arithmetic floor*normal/group computed a one-patch share as0.8000000000000002. Ratio-first multiplication implements the same declared proportions without this unnecessary intermediate scaling; no task weight/cap/tolerance change.'),
        scope='task-row construction only; independent cost/gradient and symmetricwrench/digit-split oracles, no optimizer, geometry model or runtime; no performance or support guarantee')
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(passed=output['passed'],checks=len(checks),failures=output['failures'],source_sha256=before,out=str(out)),ensure_ascii=False))
    return 0 if output['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

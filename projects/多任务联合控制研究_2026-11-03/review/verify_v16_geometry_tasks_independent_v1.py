"""Small independent geometry-task peer oracles; no models or physics."""
from pathlib import Path
from dataclasses import replace
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'v16_release_geometry/geometry_tasks.py'
EXPECTED='1694ca8889eebd709e7f1e8925c359880b2255166cdb46acde836b41605a0f0c'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE)==EXPECTED
from v16_release_geometry.geometry_tasks import make_length_proposal_map,RobotPatchGeometry,build_geometry_tasks,task_loss

checks=[]
def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
gear=np.linspace(.5,2.5,21);gear[::3]*=-1
D=np.linspace(.2,1.2,21);scale=np.linspace(.3,2.1,25)
A=np.zeros((21,25));A[:,:21]=np.diag(gear)
kw=dict(transmission21x25=A,reference_to_length21=D,joint_scale25=scale,
    robot_axes25=tuple('robot'+str(i) for i in range(25)),reference_axes21=tuple('reference'+str(i) for i in range(21)),
    time_s=3.,declared_source='synthetic unequal-unit/signed gear proposal; not actual robot response')
mapping=make_length_proposal_map(**kw);H=np.zeros((25,21));H[:21]=np.diag(D/gear)
ck('signed_gears_and_nonidentity_reference_units_closed_form',np.allclose(mapping.H25x21,H,rtol=0,atol=2e-15) and np.allclose(A@mapping.H25x21,np.diag(D),rtol=0,atol=2e-15) and mapping.rank==21 and mapping.nullity==4)

J=np.zeros((3,25));J[:,:5]=np.array([[.03,-.02,.04,.01,.005],[.01,.05,-.03,.02,-.01],[.04,.01,.02,-.02,.03]])
def patch(digit,pressure,n,J0=J):return RobotPatchGeometry(2,digit,2.99,3.,pressure,np.array([.01,.02,.03]),np.array(n,float),np.array([.011,.021,.032]),J0)
patches=[patch(0,3.,[1,0,0]),patch(0,2.,[0,1,0]),patch(1,4.,[0,0,1]),patch(1,1.,[-1,0,0])]
def build(ps,**changes):
    k=dict(patches=ps,mapping=mapping,current_time_s=3.,maximum_geometry_age_s=.1,
        step_lower21=-np.linspace(.01,.03,21),step_upper21=np.linspace(.02,.04,21),nominal_open_increment21=np.zeros(21),mode_generation=4,robot_geometry_authenticated=True)
    k.update(changes);return build_geometry_tasks(**k)
def oracle(ps,u):
    normal=0.;tangent=0.;gradient=np.zeros(21)
    for p in ps:
        share=p.observed_normal_N/sum(q.observed_normal_N for q in ps if q.digit==p.digit)
        M=p.robot_jacobian_m_per_q25@H;n=p.acquisition_normal_world;r=n@M;T=(np.eye(3)-np.outer(n,n))@M
        inward=max(float(r@u),0.);tv=T@u
        normal+=share*inward**2;tangent+=share*float(tv@tv);gradient+=2*share*(r*inward+T.T@tv)
    return normal,tangent,gradient
def actual(t,u):
    m=task_loss(t,u);G=t.normal_matrix_m_per_reference;w=t.normal_weight;T=t.tangent_matrix_m_per_reference
    gradient=2*G.T@(w*w*np.maximum(G@u,0))+2*T.T@(T@u)
    return m['normal_noninward_loss_m2'],m['tangent_increment_loss_m2'],gradient
t=build(patches);u=np.linspace(-.008,.009,21);expected=oracle(patches,u);measured=actual(t,u)
bound=128*np.finfo(float).eps*max(1.,np.max(abs(expected[2])))
ck('independent_physical_patch_cost_and_all21_gradient_oracle',abs(expected[0]-measured[0])<=bound and abs(expected[1]-measured[1])<=bound and np.max(abs(expected[2]-measured[2]))<=bound,dict(absolute_roundoff_bound=bound))
split=[replace(patches[0],observed_normal_N=.75),replace(patches[0],observed_normal_N=2.25),*patches[1:]]
variants=[split,list(reversed(patches)),[replace(p,observed_normal_N=p.observed_normal_N*7.) for p in patches]]
differences=[]
for ps in variants:
    got=actual(build(ps),u);differences.append(max(abs(got[0]-measured[0]),abs(got[1]-measured[1]),np.max(abs(got[2]-measured[2]))))
ck('splitmerge_permutation_uniform_pressure_scale_invariant',max(differences)<=bound,dict(max_cost_gradient_difference=max(differences),bound=bound))
axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis);angle=.7;K=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
R=np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)
rotated=[replace(p,acquisition_position_world_m=R@p.acquisition_position_world_m,robot_position_world_m=R@p.robot_position_world_m,acquisition_normal_world=R@p.acquisition_normal_world,robot_jacobian_m_per_q25=R@p.robot_jacobian_m_per_q25) for p in patches]
rot=actual(build(rotated),u)
ck('world_rotation_covariant_normal_and_tangent_loss_gradient',abs(rot[0]-measured[0])<=bound and abs(rot[1]-measured[1])<=bound and np.max(abs(rot[2]-measured[2]))<=bound)

simpleA=np.eye(21,25);simple=make_length_proposal_map(**dict(kw,transmission21x25=simpleA,reference_to_length21=np.ones(21)))
singleJ=np.zeros((3,25));singleJ[0,0]=1.
p=patch(0,1.,[1,0,0],singleJ);op=replace(p,acquisition_normal_world=np.array([-1.,0,0]))
opposite=build([p,op],mapping=simple);probe=np.zeros(21);probe[0]=.01
loss=task_loss(opposite,probe)
ck('opposite_rows_remain_separate_and_cannot_average_to_false_zero',opposite.normal_matrix_m_per_reference.shape==(2,21) and np.array_equal(opposite.normal_matrix_m_per_reference[0],-opposite.normal_matrix_m_per_reference[1]) and loss['normal_noninward_loss_m2']>0 and len(opposite.diagnostics['patches'])==2)

lo=np.zeros(21);lo[0]=-.02;hi=np.zeros(21);opening=np.zeros(21);opening[0]=-.01
conflict=build([op],mapping=simple,step_lower21=lo,step_upper21=hi,nominal_open_increment21=opening)
ck('one_sided_OPEN_conflict_reported_without_domain_widening',np.array_equal(conflict.step_lower21,lo) and np.array_equal(conflict.step_upper21,hi) and np.array_equal(conflict.open_target_increment21,opening) and conflict.diagnostics['unresolved_OPEN_noninward_patch_indices']==[0] and conflict.diagnostics['strict_outward_unavailable_patch_indices']==[0] and task_loss(conflict,opening)['actual_bounds_satisfied'])

unknown=make_length_proposal_map(**kw,known_reference21=np.r_[False,np.ones(20,bool)])
stale=build([p,replace(op,acquisition_time_s=2.8)],mapping=simple)
none=build([p],mapping=unknown)
invalid=[]
for val in ('1',np.bool_(True),0.):
    try:replace(p,observed_normal_N=val);invalid.append(False)
    except ValueError:invalid.append(True)
ck('unknown_and_partial_stale_are_None_not_zero_strict_fields',unknown is None and none.normal_matrix_m_per_reference is None and stale.status=='UNAVAILABLE_GEOMETRY_TASKS' and stale.normal_matrix_m_per_reference is None and all(invalid))

ck('scope_is_proposal_not_pressure_servo_or_delivery_certificate',all(t.diagnostics[k] is False for k in ('external_current_geometry_known','pressure_prediction_available','actual_servo_response_validated','force_safety_certified','full_hand_geometry_covered','delivery_certified')) and mapping.actual_servo_response_validated is False and 'mujoco' not in sys.modules)
report=dict(schema='v16_geometry_tasks_independent_v1',passed=all(c['passed'] for c in checks),checks=checks,source_sha256=EXPECTED,source_unchanged=sha(SOURCE)==EXPECTED,checker_sha256=sha(__file__),
    scope='Signed/unit-aware synthetic map and independent patch cost/full-gradient, representation/world-rotation, real domain conflict and unknown-field peer review. Not author29 repetition or actual model validation.',
    adapter_obligations=['Jacobian25 ordering must match mapping.robot_axes25; RobotPatchGeometry does not carry an axes tuple','Only authenticated factory output may be trusted as mapping provenance; direct LengthProposalMap constructor validates numeric schema, not mathematical origin/rank truth'],
    raw_records_read=0,model_created=False,physics_steps=0)
out=ROOT/'review/v16_geometry_tasks_independent_v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[c['name'] for c in checks if not c['passed']],path=str(out),sha256=sha(out)),ensure_ascii=False))
raise SystemExit(0 if report['passed'] else 1)

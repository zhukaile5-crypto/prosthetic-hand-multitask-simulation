"""Six bounded pure adapter scopes. No model construction/FK/physics/raw."""
from pathlib import Path
from types import SimpleNamespace
from collections import deque
from copy import deepcopy
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v16_release_geometry.public_adapter import PublicReleaseAdapter,affine_equilibrium_length_derivative
SOURCE=ROOT/'v16_release_geometry/public_adapter.py';EXPECTED='0a6d172297e410fccf6a0ce1f386b73157b0c10f59bdf428dfda00f6c7575d44'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE)==EXPECTED
checks=[]
def ck(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
G=np.zeros((21,3));G[:,0]=np.linspace(-2.,3.,21);G[:,1]=.4;G[:,2]=-.2
B=np.zeros((21,3));B[:,0]=.3;B[:,1]=np.linspace(-2.,-1.,21);B[:,2]=-.1
l=np.linspace(.1,.3,21);v=np.linspace(.02,.04,21);u=np.linspace(.2,.4,21)
D,diag=affine_equilibrium_length_derivative(gain21x3=G,bias21x3=B,length21=l,speed21=v,engine21=u)
p=(G[:,0]+G[:,1]*l+G[:,2]*v)*u+B[:,0]+B[:,1]*l+B[:,2]*v
def length_at(command):return (p-(G[:,0]+G[:,2]*v)*command-B[:,0]-B[:,2]*v)/(G[:,1]*command+B[:,1])
eps=1e-5;fd=(length_at(u+eps)-length_at(u-eps))/(2*eps)
ck('signed_affine_implicit_fixed_effort_derivative_oracle',D is not None and np.max(abs(D-fd))<2e-10 and diag['actual_servo_response_validated'] is False and diag['pressure_prediction_available'] is False,dict(max_difference=float(np.max(abs(D-fd))),scope='Derivative of explicit algebraic force level set only'))
def cache_only():
 a=PublicReleaseAdapter.__new__(PublicReleaseAdapter);a._samples={};a._order=deque();a._prior_history=deque(maxlen=256);a._last_tick=None;a._last_time=None;a._last_signature=None;a.hand_sample=None;a.release_request_s=None;a.pre_release_prior=None;return a
def frame(t,up,load=1.):
 M=np.zeros((6,5));M[2,0]=1.
 return SimpleNamespace(time=t,support_axis=np.array(up,float),robot_position25=np.zeros(25),sample_time=t-.01,age=.01,normal=np.array([1.,0,0,0,0]),normal_wrench_map=M,finger_non_normal_by_finger=np.zeros((5,6)),tangent_basis=np.tile(np.eye(3)[:2],(5,1,1)),auxiliary_wrench=np.zeros(6),valid=np.ones(5,bool),distributed_valid=np.ones(5,bool),auxiliary_valid=True,auxiliary_age=.01,auxiliary_uncertainty=.01,load_estimate=load,load_uncertainty=.1,load_valid=True,sensor_characteristics=dict(force_sigma_N=.001,tangential_sigma_N=.001))
a=cache_only()
for i in range(41):
 t=11.8+i*.005;f=frame(t,[1,0,0] if i==40 else [0,0,1],999. if i==40 else 1.);hand=a.observe(f)
prior=a.begin_release(12.);ck('coacquired_up_not_current_rotated_up_and_pre_request_prior',hand.valid and hand.signed_support_N==1. and prior.value_N==1. and prior.reference_time_s<12.)
before=len(a._prior_history);a.observe(frame(12.005,[1,0,0],777.))
ck('postrelease_motor_changes_cannot_change_or_extend_prior',a.pre_release_prior is prior and len(a._prior_history)==before and a.begin_release(12.) is prior)
b=cache_only();unknown=b.observe(frame(12.,[1,0,0]));ck('missing_coacquired_up_remains_unknown_not_zero_or_current',not unknown.valid and unknown.signed_support_N is None and unknown.uncertainty_N is None)

# Only rejection paths are executed with declared mocks; no accepted model/FK
# path is claimed. The actual12.02 packet pipeline is independently reviewed.
c=PublicReleaseAdapter.__new__(PublicReleaseAdapter);c._last_time=1.;c._binding=();c._signature=lambda:();c.last_geometry_diagnostics={}
bias=.01
c.actor=SimpleNamespace(ctrl_range21=np.tile([-1.,1.],(21,1)),lift_bias_m=bias,encode=lambda r:SimpleNamespace(values=r.values+np.r_[np.zeros(18),bias,0.,0.]))
c.geometry=SimpleNamespace(raw_joint_names=tuple('rh_'+str(i) for i in range(24)))
c.servo=SimpleNamespace(evaluate=lambda *args,**kwargs:SimpleNamespace(status='KNOWN_ROBOT_ACTUATION_ALGEBRA',reasons=(),diagnostics=dict(control_clipped21=[False]*21,force_clipped21=[False]*21)))
f=SimpleNamespace(time=1.,mode_generation=1,robot_control_time_s=1.,robot_reference21=np.zeros(21),robot_actuator_target21=np.r_[np.zeros(18),bias,0.,0.],robot_position25=np.zeros(25),robot_velocity25=np.zeros(25),valid=np.ones(5,bool),distributed_valid=np.ones(5,bool))
lo=np.full(21,-.01);hi=np.full(21,.01);bad=[]
q=deepcopy(f);q.robot_control_time_s=1.005;bad.append(c.current_geometry_tasks(q,np.zeros(21),np.zeros(21),lo,hi))
q=deepcopy(f);q.robot_actuator_target21[0]=.1;bad.append(c.current_geometry_tasks(q,np.zeros(21),np.zeros(21),lo,hi))
h=hi.copy();h[18]=1.;bad.append(c.current_geometry_tasks(f,np.zeros(21),np.zeros(21),lo,h))
ck('currentclock_engine_and_bias_corrected_real_control_box_rejections',all(x.normal_matrix_m_per_reference is None and x.diagnostics['real_domain_not_widened'] is True for x in bad) and np.array_equal(bad[-1].step_upper21,h),dict(reasons=[x.diagnostics['adapter_reason'] for x in bad]))

half=np.ones(3);centers=[np.array([2.,0.,0]),np.array([1.2,0.,0])];radii=[.25,.3]
clearances=[float(np.linalg.norm(np.maximum(abs(x)-half,0))-r) for x,r in zip(centers,radii)]
text=SOURCE.read_text(encoding='utf-8')
ck('bounding_sphere_box_rule_is_conservative_and_named_not_true_contact',clearances[0]>.7 and clearances[1]<0 and 'possible_overlap_not_true_contact' in text and 'whole_static_box_conservative_not_top_contact_classifier' in text and 'saved[\'q\']' in text,dict(analytic_sphere_box_clearances_m=clearances,scope='Geometric formula/static source review, not actual FK/model acceptance'))
report=dict(schema='v16_public_adapter_independent_v1',passed=all(x['passed'] for x in checks),checks=checks,source_sha256=EXPECTED,source_unchanged=sha(SOURCE)==EXPECTED,checker_sha256=sha(__file__),scope='Six small pure/static scopes only: force-level-set FD, acquisition up/prior cache, unknowns, three early permission rejection branches and sphere-box formula. Actual12.02 pipeline/robot identity/FK is separate parent15 gate.',models_created=0,FK_calls=0,physics_steps=0,raw_records_read=0)
out=ROOT/'review/v16_public_adapter_independent_v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[x['name'] for x in checks if not x['passed']],path=str(out),sha256=sha(out)),ensure_ascii=False))
raise SystemExit(0 if report['passed'] else 1)

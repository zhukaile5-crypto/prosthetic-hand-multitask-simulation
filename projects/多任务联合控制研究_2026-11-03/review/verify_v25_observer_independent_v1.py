"""Finite independent public-clock/rigid-field/receipt oracles, no model."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import hashlib,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
from v22_joint_task_fusion.numerics import ensure_numeric_environment,numerical_identity
ensure_numeric_environment()
import mujoco
blocked=[]
def deny(*a,**k):blocked.append('forbidden_pipeline');raise AssertionError('forbidden pipeline')
for n in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
    if hasattr(mujoco,n):setattr(mujoco,n,deny)
from v13_a1.codec import ReferenceActor21,Reference21,EnginePort21
ReferenceActor21.apply=deny
from v13_joint.plant import ExecutionReceipt21
from v11.robot_servo_effort import ACTUATORS21
from v25_response_identification import observer as ob
PIN='a625593645098dd6c3de761d03eecdbdc2cc6b98aab7d6198d3e82105b1df40b'
assert sha(ROOT/'v25_response_identification/observer.py')==PIN
checks=[];error=None
def ck(n,v,d=None):checks.append(dict(name=n,passed=bool(v),details=d))
def rejects(f):
    try:f()
    except (ValueError,TypeError,AttributeError):return True
    return False
axes=tuple('rh_fixture_'+str(i) for i in range(24))+('transport_lift',)
R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
anchor=np.array([.6,-.4,.2]);point=np.array([.8,-.1,.5])
J=np.zeros((3,25));J[:3,:3]=np.eye(3)
q=np.arange(25)*.001;v=np.zeros(25);v[:3]=[.04,-.03,.02]
xi=np.array([.17,-.08,.06,.2,-.3,.1]);T=np.array([[1.,0,0],[0,1.,0]])
P=T@R.T;ext=xi[:3]+np.cross(xi[3:],point-anchor)
relative=P@(ext-J@v)
def frame():
    patch=dict(digit=0,robot_body=0,link_point=point.tolist(),link_normal=[0.,0.,1.],reference_position=point.tolist(),reference_normal=[0.,0.,1.],sample_time=.01,age=.01,normal=2.)
    slips=np.zeros((5,2));slips[0]=relative
    return SimpleNamespace(time=.02,sample_time=.01,age=.01,normal=np.array([2.,0,0,0,0]),valid=np.ones(5,bool),distributed_valid=np.ones(5,bool),contact_present=np.array([True,False,False,False,False]),slip_valid5=np.array([True,False,False,False,False]),tangent_basis=np.tile(T,(5,1,1)),slip_local=slips,slip_domain_role=ob.SLIP_DOMAIN_ROLE,slip_domain_available=True,slip_domain_acquisition_time_s=.01,slip_domain_time_s=.02,robot_position25=q.copy(),robot_velocity25=np.ones(25)*99,skin_patches=[patch])
snap=ob.RobotAcquisitionSnapshot25(.01,q,v,R,axes)
fk_calls=[]
def fk(s,p):
    fk_calls.append(s.time_s)
    return SimpleNamespace(time_s=s.time_s,axes25=axes,point_world=np.asarray(p.link_point),point_velocity_world=J@s.velocity25,jacobian3x25=J)
def build(f=None,s=snap,callback=fk):return ob.build(frame() if f is None else f,robot_snapshot=s,point_kinematics=callback,stream_id='independent',epoch_id='0',anchor_world3=anchor)
class Geometry:
    raw_joint_names=axes[:24]
    def point_kinematics(self,s,p):
        return fk(ob.RobotAcquisitionSnapshot25(s.time_s,s.q25,s.velocity25,R,axes),p)
def adapter():return SimpleNamespace(_last_time=.02,_motion_samples={2:dict(time_s=.01,q=q.copy(),v=v.copy(),R=R.copy(),reason=None)},geometry=Geometry(),observe=deny,sample=deny)
def receipt(t,pre,executed,requested=None,bias=.013):
    requested=executed if requested is None else requested
    engine=executed.copy();engine[18]+=bias
    return ExecutionReceipt21(Reference21(requested,t),Reference21(executed,t),EnginePort21(engine,t,bias),'release',7,requested.tobytes()!=executed.tobytes(),False)
def precommit(t,pre,bias=.013):
    e=pre.copy();e[18]+=bias
    return dict(time_s=t,reference21=Reference21(pre,t),engine21=EnginePort21(e,t,bias),robot_position25=q,robot_velocity25=v,robot_actuator_axes21=ACTUATORS21,reference_lower21=np.ones(21)*-2,reference_upper21=np.ones(21)*2,mode_generation=7,phase='TRANSFER')
try:
    p=build();d=p.digits[0]
    ck('nonzero world anchor and rotated palm: external sign and rigid angular field',p.available and np.allclose(d.external_projection2_m_s,P@ext,rtol=0,atol=2e-16) and np.allclose(d.conditional_C2x6@xi,P@ext,rtol=0,atol=2e-16),dict(e=d.external_projection2_m_s.tolist(),expected=(P@ext).tolist()))
    f=frame();f.skin_patches=[dict(f.skin_patches[0],normal=.5),dict(f.skin_patches[0],normal=1.5)]
    split=build(f)
    ck('pressure subdivision keeps one two-row instrument and shared noise identity',len(split.row_keys)==2 and np.allclose(split.conditional_C_Mx6,p.conditional_C_Mx6,rtol=0,atol=2e-16) and np.allclose(split.external_projection_m_s,p.external_projection_m_s,rtol=0,atol=2e-16) and split.digits[0].noise_group==d.noise_group and not split.digits[0].diagnostics['complete_error_covariance_available'])
    f=frame();f.skin_patches=[];bad=build(f)
    f2=frame();f2.valid[0]=False;drop=build(f2)
    ck('missing geometry and dropout are unknown, never stationary/P0',not bad.available and not bad.known_empty and bad.digits[0].raw_relative_slip2_m_s is not None and drop.digits[0].external_projection2_m_s is None and len(drop.row_keys)==0)
    wrong=ob.RobotAcquisitionSnapshot25(.02,q,v,R,axes);old=build(s=wrong)
    def wrongfk(s,p):return SimpleNamespace(time_s=.02,axes25=axes,point_world=p.link_point,point_velocity_world=J@v,jacobian3x25=J)
    cbad=build(callback=wrongfk)
    ck('same-age robot/callback admission cannot substitute read-time state',not old.available and not cbad.available and old.digits[0].raw_relative_slip2_m_s is not None)
    owner=ob.PublicMotionObserver(stream_id='independent',epoch_id='0',anchor_world3=anchor);f=frame();ad=adapter();op=owner.observe(f,ad)
    refreshed=deepcopy(f);refreshed.robot_reference21=np.ones(21);refreshed.robot_actuator_target21=np.ones(21)
    exact=owner.observe(refreshed,ad) is op
    changed=deepcopy(f);changed.robot_velocity25[0]+=1
    rewind=deepcopy(f);rewind.time=.015;rewind.age=.005;rewind.slip_domain_time_s=.015
    ck('history uses acquisition v despite current v99; duplicate memory refresh only',np.allclose(op.external_projection_m_s,P@ext,rtol=0,atol=2e-16) and exact and rejects(lambda:owner.observe(changed,ad)) and rejects(lambda:owner.observe(rewind,ad)) and fk_calls and all(t==.01 for t in fk_calls))
    cold=ob.PublicMotionObserver(stream_id='independent',epoch_id='cold');ad2=adapter();ad2._motion_samples={}
    cp=cold.observe(f,ad2)
    ck('current q/v cannot backfill missing acquisition; query causal at read time',not cp.available and cp.digits[0].raw_relative_slip2_m_s is not None and len(owner.packets(available_at_s=.015))==0 and len(owner.packets(available_at_s=.02))==1)
    pre=np.zeros(21);pre[18]=.04;ex=pre.copy();ex[2]=.003;ex[18]=.039;req=pre.copy();req[2]=.005
    r=receipt(.02,pre,ex,req);event=owner.on_receipt(r,precommit(.02,pre))
    ck('typed receipt uses actual increment with one lift bias, not requested',np.array_equal(event.actual_reference_increment21,ex-pre) and np.array_equal(event.executed_engine21,r.engine.values) and event.actual_reference_increment21[2]!=req[2] and owner.on_receipt(r,precommit(.02,pre)) is event and not event.diagnostics['substep_hold_verified'])
    bad=precommit(.02,pre);bad['robot_actuator_axes21']=ACTUATORS21[::-1]
    badengine=precommit(.02,pre);badengine['engine21']=EnginePort21(r.engine.values,.025,.013)
    changedex=ex.copy();changedex[2]+=.001
    truth=precommit(.02,pre);truth['object_velocity']=[1,2,3]
    ck('wrong axes/engine clock/truth and changed same-tick actual receipt reject',rejects(lambda:owner.on_receipt(r,bad)) and rejects(lambda:owner.on_receipt(r,badengine)) and rejects(lambda:owner.on_receipt(receipt(.02,pre,changedex),precommit(.02,pre))) and rejects(lambda:owner.on_receipt(r,truth)))
    owner.on_receipt(receipt(.03,ex,ex),precommit(.03,ex))
    intervals=owner.input_intervals()
    ck('interval available only at next receipt; gap explicit not held-motion proof',len(owner.input_intervals(available_at_s=.025))==0 and len(owner.receipts(available_at_s=.025))==1 and len(intervals)==1 and not intervals[0]['consecutive_5ms'] and not intervals[0]['substep_hold_verified'])
    empty=frame();empty.normal[:]=0;empty.contact_present[:]=False;empty.slip_valid5[:]=False;empty.skin_patches=[]
    ep=build(empty);empty.valid[3]=False;unknown=build(empty)
    ck('explicit all-five no-contact differs from one unknown sensor',ep.known_empty and ep.conditional_C_Mx6.shape==(0,6) and not unknown.known_empty and not unknown.available and unknown.digits[3].external_projection2_m_s is None)
except Exception:error=traceback.format_exc()
out=ROOT/'review/v25_observer_independent_v1.json';assert not out.exists()
result=dict(schema='v25_observer_independent_v1',passed=error is None and all(c['passed'] for c in checks) and not blocked,complete=True,checks=checks,issues=[c['name'] for c in checks if not c['passed']],error=error,production_source_sha256={'v25_response_identification/observer.py':PIN},source_unchanged=sha(ROOT/'v25_response_identification/observer.py')==PIN,checker_sha256=sha(__file__),numerical_identity=numerical_identity(),physics_steps=0,model_constructions=0,forbidden_calls=blocked,scope='10 finite synthetic public-input oracles with independent world rigid-field equation and actual frozen typed receipts; no model, archive performance, device draw, forward, contact, writer or integration. Actual registered continuation remains a separate required bridge.')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps(dict(passed=result['passed'],checks=len(checks),issues=result['issues'],error=error,out=str(out),sha256=sha(out)),ensure_ascii=False))
raise SystemExit(0 if result['passed'] and result['source_unchanged'] else 1)

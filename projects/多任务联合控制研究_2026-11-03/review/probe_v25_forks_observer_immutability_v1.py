"""Actual V25 built packet: bytes-backed immutability across opaque copy."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v22_joint_task_fusion.numerics import ensure_numeric_environment
ensure_numeric_environment()
import mujoco
for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_kinematics','mj_fwdPosition','mj_fwdVelocity','mj_collision','mj_contactForce'):
 def deny(*a,_name=name,**kw):raise AssertionError('forbidden '+_name)
 if hasattr(mujoco,name):setattr(mujoco,name,deny)
from v13_a1.codec import ReferenceActor21
ReferenceActor21.apply=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('forbidden writer'))
from v25_response_identification import forks,observer
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pin=sha(ROOT/'v25_response_identification/forks.py');op=sha(ROOT/'v25_response_identification/observer.py')
axes=tuple('rh_fixture_'+str(i) for i in range(24))+('transport_lift',);q=np.zeros(25);v=np.zeros(25);v[0]=.03
snapshot=observer.RobotAcquisitionSnapshot25(.01,q,v,np.eye(3),axes);J=np.zeros((3,25));J[:3,:3]=np.eye(3)
def fk(sample,patch):return SimpleNamespace(time_s=sample.time_s,axes25=axes,point_world=patch.link_point,point_velocity_world=J@sample.velocity25,jacobian3x25=J)
f=SimpleNamespace(time=.02,sample_time=.01,age=.01,normal=np.array([2.,0,0,0,0]),valid=np.ones(5,bool),distributed_valid=np.ones(5,bool),contact_present=np.array([True,False,False,False,False]),slip_valid5=np.array([True,False,False,False,False]),tangent_basis=np.tile([[1.,0,0],[0,1.,0]],(5,1,1)),slip_local=np.tile([.1,.2],(5,1)),slip_domain_role=observer.SLIP_DOMAIN_ROLE,slip_domain_available=True,slip_domain_acquisition_time_s=.01,slip_domain_time_s=.02,robot_position25=q,robot_velocity25=v,sensor_characteristics=dict(slip_sigma_m_s=.003,position_sigma_m=.0002,force_sigma_N=.02),skin_patches=[dict(digit=0,robot_body=0,link_point=[.1,.2,0.],link_normal=[0.,0.,1.],reference_position=[.1,.2,0.],reference_normal=[0.,0.,1.],sample_time=.01,age=.01,normal=2.)])
packet=observer.build(f,robot_snapshot=snapshot,point_kinematics=fk,stream_id='fixture',epoch_id='initial');before=packet.external_projection_m_s.copy()
def can_unlock(a):
 try:a.setflags(write=True);return True
 except ValueError:return False
original_unlock=can_unlock(packet.external_projection_m_s)
assert not original_unlock
model=mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.0005"/><worldbody><body><joint type="hinge"/><geom size=".1"/></body></worldbody></mujoco>');s=SimpleNamespace(time=0.,tick=0,plant=SimpleNamespace(time=0.,model=model,data=mujoco.MjData(model),_delivery_inside_advance=False),sensors=SimpleNamespace(_last_time=None),packet=packet);child,metadata=forks.fork_session(s);unlocked=can_unlock(child.packet.external_projection_m_s)
if unlocked:child.packet.external_projection_m_s[0]+=1
original_unchanged=np.array_equal(packet.external_projection_m_s,before)
assert sha(ROOT/'v25_response_identification/forks.py')==pin and sha(ROOT/'v25_response_identification/observer.py')==op
out=ROOT/'review/v25_forks_observer_immutability_attempt1.json';assert not out.exists();j=dict(schema='v25_fork_observer_bytes_immutability_v1',passed=not unlocked,actual_packet_class=type(packet).__name__,packet_status=packet.status,original_setflags_write_true_possible=original_unlock,cloned_setflags_write_true_possible=unlocked,parent_values_unchanged_after_child_mutation=original_unchanged,packet_object_independent=child.packet is not packet,array_memory_independent=not np.shares_memory(child.packet.external_projection_m_s,packet.external_projection_m_s),kernel_metadata_passed=metadata['passed'],production_source_sha256={'v25_response_identification/forks.py':pin,'v25_response_identification/observer.py':op},source_unchanged=True,checker_sha256=sha(__file__),physics_steps=0,model_scope='one explicit tiny native model plus copies; synthetic public packet built by actual observer API; no forward/step/contact/writer',interpretation='If cloned array can unlock, numeric values/parent isolation may still hold but strong bytes-backed immutability is NOT preserved.')
out.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps({'out':str(out),'sha256':sha(out),**j},ensure_ascii=False))

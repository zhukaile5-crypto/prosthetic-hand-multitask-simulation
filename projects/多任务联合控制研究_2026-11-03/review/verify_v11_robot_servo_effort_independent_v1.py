"""Independent closed-form robot-only actuator oracles; no plant or integration."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,sys,json,hashlib,copy,xml.etree.ElementTree as ET
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'v11/robot_servo_effort.py';AUTHOR=ROOT/'review/v11_robot_servo_effort_author_v2.json';OUT=ROOT/'review/v11_robot_servo_effort_independent_v1.json'
if OUT.exists():raise FileExistsError(OUT)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
EXPECTED='f0671211d2e15cc96eeb8c87207a43bc1839d55512ab18bdc2cebff4ed0a052c'
if sha(SOURCE)!=EXPECTED:raise RuntimeError('Source version changed before review')
guard_names=['mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_implicit','mj_forward','mj_forwardSkip','mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation','mj_kinematics','mj_comPos','mj_collision']
calls={name:0 for name in guard_names}
def forbidden(name):
 def reject(*a,**kw):calls[name]+=1;raise RuntimeError('Oracle forbids '+name)
 return reject
for name in guard_names:
 if hasattr(mujoco,name):setattr(mujoco,name,forbidden(name))
spec=importlib.util.spec_from_file_location('independent_v11_effort',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
RAW=('rh_WRJ2','rh_WRJ1','rh_FFJ4','rh_FFJ3','rh_FFJ2','rh_FFJ1','rh_MFJ4','rh_MFJ3','rh_MFJ2','rh_MFJ1','rh_RFJ4','rh_RFJ3','rh_RFJ2','rh_RFJ1','rh_LFJ5','rh_LFJ4','rh_LFJ3','rh_LFJ2','rh_LFJ1','rh_THJ5','rh_THJ4','rh_THJ3','rh_THJ2','rh_THJ1')
def fixture():
 root=ET.Element('mujoco',model='independent_closed_form_effort');ET.SubElement(root,'compiler',angle='radian');world=ET.SubElement(root,'worldbody');b=ET.SubElement(world,'body',name='lift')
 ET.SubElement(b,'joint',name='transport_lift',type='slide',axis='0 0 1',limited='true',range='-1 1');ET.SubElement(b,'inertial',mass='1',pos='0 0 0',diaginertia='.001 .001 .001')
 for i,name in enumerate(RAW):
  b=ET.SubElement(b,'body',name='robot_'+str(i),pos='0 0 .01');ET.SubElement(b,'joint',name=name,type='hinge',axis='0 1 0',limited='true',range='-4 4');ET.SubElement(b,'inertial',mass='.01',pos='0 0 0',diaginertia='.00001 .00001 .00001')
 external=ET.SubElement(world,'body',name='external_unused');ET.SubElement(external,'joint',name='external_slide',type='slide',axis='1 0 0');ET.SubElement(external,'inertial',mass='2',pos='0 0 0',diaginertia='.1 .1 .1')
 tendon=ET.SubElement(root,'tendon')
 for stem in ('FF','MF','RF','LF'):
  fixed=ET.SubElement(tendon,'fixed',name=stem+'0');ET.SubElement(fixed,'joint',joint='rh_'+stem+'J2',coef='1');ET.SubElement(fixed,'joint',joint='rh_'+stem+'J1',coef='1')
 actuator=ET.SubElement(root,'actuator')
 # Different robot21 API order versus model22 port order, as in the real model.
 for name in mod.ACTUATORS21[-2:]+mod.ACTUATORS21[:19]:
  kw=dict(name=name,ctrllimited='true',ctrlrange='-1 1',forcelimited='true',forcerange='-1000 1000')
  if name=='lift_motor':kw.update(joint='transport_lift',kp='11',kv='3')
  elif name in ('rh_A_WRJ2','rh_A_WRJ1'):kw.update(joint=name.replace('rh_A_','rh_'),kp='5',kv='1')
  elif name.endswith('J0'):kw.update(tendon=name[5:7]+'0',kp='7',kv='2')
  else:kw.update(joint=name.replace('rh_A_','rh_'),kp='7',kv='2')
  ET.SubElement(actuator,'position',**kw)
 ET.SubElement(actuator,'position',name='unused_motor',joint='external_slide',kp='13')
 return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))

checks=[];numeric={}
def check(name,value,details=None):
 checks.append(dict(name=name,passed=bool(value),details=details))
 if not value:raise AssertionError(name+': '+str(details))
m=fixture();e=mod.RobotServoEffort(m,raw_joint_names=RAW)
check('supported_private_synthetic_model',not e.unavailable,e.unavailable)
A=np.zeros((21,25));kp=np.r_[np.full(18,7.),11.,5.,5.];kd=np.r_[np.full(18,2.),3.,1.,1.]
for i,name in enumerate(mod.ACTUATORS21):
 if name=='lift_motor':A[i,24]=1.
 elif name.endswith('J0'):
  stem=name[5:7];A[i,RAW.index('rh_'+stem+'J2')]=1.;A[i,RAW.index('rh_'+stem+'J1')]=1.
 else:A[i,RAW.index(name.replace('rh_A_','rh_'))]=1.
q=np.zeros(25);v=q.copy();u=np.zeros(21);j2=RAW.index('rh_FFJ2');j1=RAW.index('rh_FFJ1');ff=mod.ACTUATORS21.index('rh_A_FFJ0')
q[j2]=.3;q[j1]=.2;v[j2]=-.01;v[j1]=.04;u[ff]=.7;q[24]=.04;v[24]=.01;u[18]=.057
s=e.evaluate(q,v,u,time_s=2.);force=kp*(u-A@q)-kd*(A@v);expected=A.T@force
check('CSR_transmission_equals_independent_named_joint_tendon_map',np.array_equal(s.transmission21x25,A))
check('fixed_tendon_two_unit_moment_arms',np.array_equal(A[ff,[j2,j1]],[1.,1.]) and abs(s.predicted_effort21[ff]-1.34)<1e-12)
check('closed_form_all21_efforts',np.allclose(s.predicted_effort21,force,rtol=0,atol=1e-12))
check('coupled_tendon_gives_full_torque_to_each_joint_not_half',abs(s.generalized_actuator_effort25[j1]-1.34)<1e-12 and abs(s.generalized_actuator_effort25[j2]-1.34)<1e-12)
check('closed_form_generalized_effort',np.allclose(s.generalized_actuator_effort25,expected,rtol=0,atol=1e-12))
check('reference_J_matches_closed_form',np.allclose(s.reference_jacobian25x21,A.T@np.diag(kp),rtol=0,atol=1e-12))
check('stiffness_matches_known_actuator_partials',np.allclose(s.actuation_stiffness25x25,A.T@np.diag(kp)@A,rtol=0,atol=1e-12))
check('damping_matches_known_actuator_partials',np.allclose(s.actuation_damping25x25,A.T@np.diag(kd)@A,rtol=0,atol=1e-12))
dq=np.zeros(25);dq[j1]=.02;dq[j2]=-.02
diff=e.evaluate(q+dq,v,u)
check('fixed_tendon_differential_displacement_null_direction',np.allclose(diff.predicted_effort21,s.predicted_effort21,rtol=0,atol=1e-12))
check('actuation_stiffness_rank21_not25',np.linalg.matrix_rank(s.actuation_stiffness25x25)==21)
# One mixed q/v/u finite direction, independent of the author's basis-axis sweeps.
pq=np.linspace(-.3,.4,25);pv=np.linspace(.2,-.15,25);pu=np.linspace(-.2,.1,21);h=1e-6
plus=e.evaluate(q+h*pq,v+h*pv,u+h*pu);minus=e.evaluate(q-h*pq,v-h*pv,u-h*pu)
fd=(plus.generalized_actuator_effort25-minus.generalized_actuator_effort25)/(2*h);analytic=s.reference_jacobian25x21@pu-s.actuation_stiffness25x25@pq-s.actuation_damping25x25@pv
numeric['mixed_direction_FD_max_abs_error']=float(abs(fd-analytic).max());check('finite_mixed_direction_derivative',numeric['mixed_direction_FD_max_abs_error']<1e-8,numeric['mixed_direction_FD_max_abs_error'])
decoded=u.copy();decoded[18]-=.007;wrong=e.evaluate(q,v,decoded)
numeric['missing_lift_engine_bias_effort_difference_N']=float(s.generalized_actuator_effort25[24]-wrong.generalized_actuator_effort25[24])
check('unbiased_decoder_reference_is_not_actual_ctrl_semantically',abs(numeric['missing_lift_engine_bias_effort_difference_N']-11*.007)<1e-12)
ports=np.empty(22,dtype=object);ports[:]='nonrobot_ignored';ports[e.ctrladr]=u
port=e.evaluate(q,v,ports,control_layout='model22')
check('model22_selects_named_robot_ports_before_numeric_validation',np.array_equal(port.generalized_actuator_effort25,s.generalized_actuator_effort25))
other=copy.copy(m);external_joint=int(other.joint('external_slide').id);other.qpos0[int(other.jnt_qposadr[external_joint])]=.777;other.body_mass[int(other.body('external_unused').id)]=200.
check('external_default_state_and_mass_do_not_enter_fixed_transmission',np.array_equal(s.generalized_actuator_effort25,mod.RobotServoEffort(other,raw_joint_names=RAW).evaluate(q,v,u).generalized_actuator_effort25))
# Control saturation differs from force saturation; kinks must not become fake zero gradients.
z=np.zeros(25);uu=np.zeros(21);uu[0]=1.5;control=e.evaluate(z,z,uu)
check('strict_control_clip_known_zero_J_column',np.array_equal(control.reference_jacobian25x21[:,0],np.zeros(25)) and control.actuation_stiffness25x25 is not None)
uu[0]=1.;control_kink=e.evaluate(z,z,uu)
check('control_kink_J_unknown_KD_still_defined',control_kink.reference_jacobian25x21 is None and control_kink.actuation_stiffness25x25 is not None and control_kink.actuation_damping25x25 is not None)
limited=copy.copy(m);aid=int(limited.actuator(mod.ACTUATORS21[0]).id);limited.actuator_forcerange[aid]=[-2,2];le=mod.RobotServoEffort(limited,raw_joint_names=RAW);uu[0]=.5;forceclip=le.evaluate(z,z,uu)
check('strict_force_clip_effort2_and_known_zero_partials_for_that_actuator',forceclip.predicted_effort21[0]==2. and np.array_equal(forceclip.reference_jacobian25x21[:,0],np.zeros(25)) and forceclip.actuation_stiffness25x25[RAW.index('rh_THJ5'),RAW.index('rh_THJ5')]==0.)
uu[0]=2/7.;forcekink=le.evaluate(z,z,uu)
check('force_kink_JKD_all_unknown_effort_known',forcekink.status=='KNOWN_ROBOT_ACTUATION_ALGEBRA' and forcekink.reference_jacobian25x21 is None and forcekink.actuation_stiffness25x25 is None and forcekink.actuation_damping25x25 is None)
bad=copy.copy(m);bad.actuator_trnid[aid,0]=int(bad.joint('external_slide').id);be=mod.RobotServoEffort(bad,raw_joint_names=RAW).evaluate(z,z,np.zeros(21))
check('unsupported_nonrobot_transmission_has_no_effort',be.status=='UNAVAILABLE' and be.generalized_actuator_effort25 is None,be.reasons)
bad=copy.copy(m);bad.tendon_actfrclimited[0]=True;be=mod.RobotServoEffort(bad,raw_joint_names=RAW).evaluate(z,z,np.zeros(21))
check('unsupported_total_tendon_force_clamp_has_no_effort',be.status=='UNAVAILABLE' and be.generalized_actuator_effort25 is None,be.reasons)
bad=copy.copy(m);bad.actuator_dyntype[aid]=mujoco.mjtDyn.mjDYN_FILTER;be=mod.RobotServoEffort(bad,raw_joint_names=RAW).evaluate(z,z,np.zeros(21))
check('unsupported_activation_not_assumed_zero',be.status=='UNAVAILABLE' and be.generalized_actuator_effort25 is None,be.reasons)
missing=e.evaluate(z,np.zeros(24),np.zeros(21));check('missing25th_velocity_rejected_not_filled',missing.status=='UNAVAILABLE')
check('snapshot_is_model_computation_not_contact_or_sensor',not s.contact_force_prediction_available and not s.measured_effort and not s.global_model_admitted)
check('zero_plant_world_FK_collision_forward_and_integrator_calls',not any(calls.values()),calls)
check('source_stable_after_tests',sha(SOURCE)==EXPECTED)
author=json.loads(AUTHOR.read_text(encoding='utf-8'));check('author_evidence_binds_same_source',author['source_sha256']==EXPECTED and author['passed'])
report=dict(schema='v11_robot_servo_effort_independent_v1',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=EXPECTED,author_evidence_sha256=sha(AUTHOR),checker_sha256=sha(Path(__file__)),passed=True,checks=checks,numeric_evidence=numeric,physics_steps=0,plant_constructed=False,guard_calls=calls,fixture='independent25robot coordinates+one disjoint scalar external coordinate,21position actuators+one unused actuator; four fixed [1,1] joint tendons; native model port order differs fromrobot21',limitations=['Synthetic closed-form and source review supplement rather than reproduce author67/native archive cases. No historical live plant data or object truth is input.','robot21 means actual engine-port actuator controls including known robot lift gravity bias. Same-shape unbiased decoder.command21 cannot be detected from numbers alone; provenance/layout is a caller obligation.','K/D/Ju are current known-actuator branch partials. They exclude passive forces, contact/environment stiffness, dynamics, mass/friction effects, unmodeled activation, and future force response.','Tendon differential directions remain unactuated; rank21 generalized stiffness is not an invertible25joint target map. No independent torque sensor or contact-force certificate exists.'])
with OUT.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),checks=len(checks),numeric_evidence=numeric,physics_steps=0)))

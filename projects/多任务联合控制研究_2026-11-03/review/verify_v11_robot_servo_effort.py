"""Zero-integration native/private, robot-only archival, and FD checks."""
from copy import copy
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import research_imports
import mujoco
from hand_plant import HandPlant
from foundation_config import FoundationConfig
from v11.robot_servo_effort import RobotServoEffort


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def native(model,adapter,q,v,u):
    d=mujoco.MjData(model);d.qpos[adapter.qadr]=q;d.qvel[adapter.dadr]=v;d.ctrl[adapter.ctrladr]=u
    mujoco.mj_kinematics(model,d);mujoco.mj_comPos(model,d);mujoco.mj_tendon(model,d);mujoco.mj_transmission(model,d)
    mujoco.mj_fwdVelocity(model,d);mujoco.mj_fwdActuation(model,d)
    return d.actuator_force[adapter.outadr].copy(),d.qfrc_actuator[adapter.dadr].copy(),d


def verify():
    calls={};original={}
    def block(name):
        def forbidden(*a,**k):calls[name]+=1;raise AssertionError('No integration permitted:'+name)
        return forbidden
    for name in ('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_implicit'):
        if hasattr(mujoco,name):original[name]=getattr(mujoco,name);calls[name]=0;setattr(mujoco,name,block(name))
    checks=[];numbers={}
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    try:
        # Compile/init only; no actual plant MjData enters the adapter.
        model=HandPlant(FoundationConfig()).model;adapter=RobotServoEffort(model)
        check('native_supported21_stateless_affine_robot_actuators',not adapter.unavailable)
        q=np.mean(model.jnt_range[adapter.jids],axis=1);q[-1]=.05;v=np.zeros(25)
        probe=adapter.evaluate(q,v,np.zeros(21));u=probe.actuator_length21+.005
        result=adapter.evaluate(q,v,u)
        force,tau,private=native(copy(model),adapter,q,v,u)
        check('native_private_actuator_effort_equivalent',np.allclose(result.predicted_effort21,force,atol=1e-12,rtol=1e-12))
        check('native_private_generalized_effort_equivalent',np.allclose(result.generalized_actuator_effort25,tau,atol=1e-12,rtol=1e-12))
        check('sparse_moment_CSR_decoded_without_reshape',adapter.data.actuator_moment.shape==(26,) and result.transmission21x25.shape==(21,25))
        check('private_model_arrays_independent',not np.shares_memory(model.actuator_gainprm,adapter.model.actuator_gainprm))
        old_kin=mujoco.mj_kinematics;old_com=mujoco.mj_comPos
        def no_world_pipeline(*a,**kw):raise AssertionError('Effort does not need world FK or inertial mapping')
        try:
            mujoco.mj_kinematics=no_world_pipeline;mujoco.mj_comPos=no_world_pipeline
            without_world=adapter.evaluate(q,v,u)
        finally:mujoco.mj_kinematics=old_kin;mujoco.mj_comPos=old_com
        check('minimal_effort_pipeline_has_no_world_FK_or_com_inertias',np.array_equal(without_world.predicted_effort21,result.predicted_effort21))
        check('reference_error_actuator_space_not24qdes',result.servo_error21.shape==(21,) and np.allclose(result.servo_error21,effective_u:=result.effective_target21-result.actuator_length21))
        check('no_contact_or_independent_measurement_claim',not result.contact_force_prediction_available and not result.measured_effort and not result.global_model_admitted)
        check('position_stiffness_rank21_not_invertible25',np.linalg.matrix_rank(result.actuation_stiffness25x25)==21)
        tendon=adapter.actuator_names21.index('rh_A_FFJ0') if hasattr(adapter,'actuator_names21') else 7
        # Actual canonical order: TH5, FF3; FFJ0 is index7.
        i=adapter.axes25.index('rh_FFJ2');j=adapter.axes25.index('rh_FFJ1')
        check('tendon_actual_two_full_moment_arms_not_half_qmap',result.transmission21x25[tendon,i]==1. and result.transmission21x25[tendon,j]==1.)
        null=np.zeros(25);null[i]=1.;null[j]=-1.
        check('fixed_tendon_differential_motion_unactuated',np.allclose(result.actuation_stiffness25x25@null,0.))
        eps=1e-6
        Ju=np.column_stack([(adapter.evaluate(q,v,u+np.eye(21)[k]*eps).generalized_actuator_effort25-adapter.evaluate(q,v,u-np.eye(21)[k]*eps).generalized_actuator_effort25)/(2*eps) for k in range(21)])
        K=-np.column_stack([(adapter.evaluate(q+np.eye(25)[k]*eps,v,u).generalized_actuator_effort25-adapter.evaluate(q-np.eye(25)[k]*eps,v,u).generalized_actuator_effort25)/(2*eps) for k in range(25)])
        D=-np.column_stack([(adapter.evaluate(q,v+np.eye(25)[k]*eps,u).generalized_actuator_effort25-adapter.evaluate(q,v-np.eye(25)[k]*eps,u).generalized_actuator_effort25)/(2*eps) for k in range(25)])
        numbers['reference_FD_max_abs_error']=float(np.max(abs(Ju-result.reference_jacobian25x21)));numbers['stiffness_FD_max_abs_error']=float(np.max(abs(K-result.actuation_stiffness25x25)));numbers['damping_FD_max_abs_error']=float(np.max(abs(D-result.actuation_damping25x25)))
        check('FD_reference_to_generalized_effort',np.allclose(Ju,result.reference_jacobian25x21,atol=1e-6,rtol=1e-7))
        check('FD_actuation_stiffness_all25_axes',np.allclose(K,result.actuation_stiffness25x25,atol=1e-6,rtol=1e-7))
        check('FD_actuation_damping_all25_axes',np.allclose(D,result.actuation_damping25x25,atol=1e-6,rtol=1e-7))
        full=np.zeros(22);full[adapter.ctrladr]=u;full[21]=np.nan
        check('model22_robot_white_list_ignores_nonrobot_control',np.array_equal(adapter.evaluate(q,v,full,control_layout='model22').predicted_effort21,result.predicted_effort21))
        high=adapter.ctrl_range[:,1]+1.;fast=v.copy();fast[0]=100.;c=adapter.evaluate(q,fast,high);nf,nt,_=native(copy(model),adapter,q,fast,high)
        check('actual_ctrl_clamp_matches_native',any(c.diagnostics['control_clipped21']) and np.allclose(c.predicted_effort21,nf,atol=1e-12,rtol=1e-12))
        check('finite_actuator_force_clipping_matches_native',any(c.diagnostics['force_clipped21']) and np.allclose(c.generalized_actuator_effort25,nt,atol=1e-12,rtol=1e-12))
        check('strictly_saturated_branch_has_known_zero_reference_gain',c.reference_jacobian25x21 is not None and np.array_equal(c.reference_jacobian25x21,np.zeros((25,21))))
        kink=u.copy();kink[tendon]=adapter.ctrl_range[tendon,0];c=adapter.evaluate(q,v,kink)
        check('control_clip_kink_not_fake_zero_jacobian',c.reference_jacobian25x21 is None and c.actuation_stiffness25x25 is not None)
        kink=u.copy();kink[18]=result.actuator_length21[18]+100./1500.;c=adapter.evaluate(q,v,kink)
        check('force_clip_kink_derivatives_unavailable_effort_still_known',c.status=='KNOWN_ROBOT_ACTUATION_ALGEBRA' and c.reference_jacobian25x21 is None and c.actuation_stiffness25x25 is None)
        outside=q.copy();outside[0]=model.jnt_range[adapter.jids[0],1]+.01;c=adapter.evaluate(outside,v,u)
        check('actual_joint_outside_range_reported_not_clipped',c.diagnostics['robot_position_outside_declared_joint_ranges25'][0] and c.diagnostics['robot_positions_not_clipped'])
        check('missing_lift_velocity_not_zero',adapter.evaluate(q,None,u).status=='UNAVAILABLE')
        check('full_object_qpos_shape_refused',adapter.evaluate(np.zeros(33),v,u).status=='UNAVAILABLE')
        check('numeric_strings_refused',adapter.evaluate(q,v,[str(x) for x in u]).status=='UNAVAILABLE')
        altered=copy(model);altered.actuator_dyntype[adapter.aids[0]]=mujoco.mjtDyn.mjDYN_FILTER;bad=RobotServoEffort(altered)
        check('unprovided_activation_unavailable_not_zero_effort',bad.evaluate(q,v,u).predicted_effort21 is None)
        altered=copy(model);altered.actuator_gaintype[adapter.aids[0]]=mujoco.mjtGain.mjGAIN_AFFINE;altered.actuator_gainprm[adapter.aids[0],:3]=[.4,.1,.05];aff=RobotServoEffort(altered);c=aff.evaluate(q,v,u);nf,nt,_=native(copy(altered),aff,q,v,u)
        check('affine_gain_law_matches_native',np.allclose(c.predicted_effort21,nf,atol=1e-12,rtol=1e-12) and np.allclose(c.generalized_actuator_effort25,nt,atol=1e-12,rtol=1e-12))
        # Pure synthetic nonrobot properties/states: do not load archived object7.
        changed=copy(model);nonrobot_body=int(changed.body('object').id);changed.body_mass[nonrobot_body]*=2.;changed.geom_friction[:]=.01
        other=RobotServoEffort(changed);c=other.evaluate(q,v,u)
        check('external_mass_friction_not_robot_actuation_inputs',np.array_equal(c.predicted_effort21,result.predicted_effort21) and np.array_equal(c.generalized_actuator_effort25,result.generalized_actuator_effort25))
        external=np.array([i for i in range(other.model.nq) if i not in set(other.qadr)],int)
        other.model.qpos0[external[:3]]=[.2,.3,.4]
        c=other.evaluate(q,v,u)
        check('external_default_pose_does_not_enter_effort',np.array_equal(c.predicted_effort21,result.predicted_effort21))
        # Existing physical record is a source of robot q/v/ctrl only. Native
        # oracle is recomputed on private data, not read from old force outputs.
        path=ROOT/'output/release_basis_minimal_v2/runs/capsule__native2d/record/trace.npz'
        if sha(path)!='8d6da9a431f8e34ffc67d391f5f2ca9ab4c3d9984343dc19f3c02dcc54da9b78':raise ValueError('Archived source changed')
        with np.load(path,allow_pickle=False) as f:
            aq=f['qpos'][:,adapter.qadr].copy();av=f['qvel'][:,adapter.dadr].copy();au=f['ctrl'][:,adapter.ctrladr].copy();times=f['time'].copy()
        indexes=sorted(set(np.linspace(0,len(times)-1,32,dtype=int))|{1500,2400,2430,2470,2524,2844,3200});largest=0.
        for k in indexes:
            c=adapter.evaluate(aq[k],av[k],au[k],time_s=float(times[k]));nf,nt,_=native(copy(model),adapter,aq[k],av[k],au[k]);diff=max(float(np.max(abs(c.predicted_effort21-nf))),float(np.max(abs(c.generalized_actuator_effort25-nt))));largest=max(largest,diff)
            check('archived_robot_only_native_equivalence_'+str(k),diff<1e-10)
        numbers['archived_native_max_abs_error']=largest
        check('zero_integrator_calls',not any(calls.values()))
        return dict(schema='v11_robot_servo_effort_author_v1',passed=True,checks=checks,numeric_evidence=numbers,physics_steps=0,integrator_calls=calls,
            model_source='known frozen HandPlant compiledmodel only; live MjData never passed',archived_source_path=str(path),archived_source_sha256=sha(path),
            archived_input_scope='only named robot25 q/v and selected21 actualctrl; all object7/velocity6/obstacle values excluded before computation',
            archived_validation_scope='private zero-integration nativeactuation recomputation, not an independent measured force channel or force forecast',
            source_sha256=sha(ROOT/'v11/robot_servo_effort.py'),reviewer_sha256=sha(__file__))
    finally:
        for name,value in original.items():setattr(mujoco,name,value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);args=p.parse_args();result=verify()
    with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks',)},ensure_ascii=False,indent=2));print('checks',len(result['checks']))

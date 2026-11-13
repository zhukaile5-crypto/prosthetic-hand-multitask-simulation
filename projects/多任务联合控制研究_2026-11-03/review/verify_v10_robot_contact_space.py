"""All25 geometric FD axes, exact reference decoding, scope refusal. No steps."""
from pathlib import Path
from dataclasses import replace
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import research_imports
import mujoco
from hand_plant import HandPlant
from foundation_config import FoundationConfig
from v10.robot_contact_space import (RobotJointSample,PublicSkinPatch,RobotContactSpace,
    MechanicalReferenceBasis,MeasuredServoResponse,HELD_AXES8)


def verify():
    checks={};metrics={};originals={n:getattr(mujoco,n) for n in ('mj_step','mj_step1','mj_step2')}
    def forbidden(*args,**kwargs):raise AssertionError('No physics integration permitted in geometry checks')
    for n in originals:setattr(mujoco,n,forbidden)
    try:
        plant=HandPlant(FoundationConfig(seed=610120));names=tuple(plant.model.joint(int(j)).name for j in plant.raw_joint_ids)
        space=RobotContactSpace(plant.model,names)
        q=np.r_[plant.data.qpos[plant.raw_qadr],.05]
        sample=RobotJointSample(1.,names,q[:24],q[24])
        bodies=('rh_thdistal','rh_ffdistal','rh_mfdistal','rh_rfdistal','rh_lfdistal','rh_palm')
        h=1e-6;errors=[];normal_errors=[];fd_axes=[]
        first=None
        for name in bodies:
            body=int(plant.model.body(name).id)
            patch=PublicSkinPatch(body,[.003,-.002,.015],[.2,.3,.9],1.)
            k=space.point_kinematics(sample,patch)
            if first is None:first=(patch,k)
            numerical=np.zeros((3,25))
            for axis in range(25):
                plus=q.copy();minus=q.copy();plus[axis]+=h;minus[axis]-=h
                sp=RobotJointSample(1.,names,plus[:24],plus[24]);sm=RobotJointSample(1.,names,minus[:24],minus[24])
                numerical[:,axis]=(space.point_kinematics(sp,patch).point_world-space.point_kinematics(sm,patch).point_world)/(2*h)
                fd_axes.append((name,space.axes25[axis]))
            errors.append(abs(numerical-k.jacobian3x25).max())
            for normal in np.eye(3):
                anchored=replace(patch,reference_position=k.point_world,reference_normal=normal)
                projection=space.linearize(sample,anchored,maximum_age_s=.04)
                normal_errors.append(abs(projection.normal_jacobian25-normal@numerical).max())
        checks['all25_axes_all6_skin_points_FD']=max(errors)<5e-7
        checks['all3_normal_directions_FD']=max(normal_errors)<5e-7
        metrics.update(point_FD_max_abs_error=float(max(errors)),normal_FD_max_abs_error=float(max(normal_errors)),point_axis_pairs=len(fd_axes),normal_axis_checks=len(bodies)*3*25)
        patch,k=first
        delayed=replace(patch,sample_time_s=.99)
        missing=space.linearize(sample,delayed,maximum_age_s=.04)
        checks['delayed_missing_acquisition_reference_unavailable']=not missing.available and missing.normal_jacobian25 is None
        delayed=replace(delayed,reference_position=k.point_world,reference_normal=k.robot_attached_normal_world)
        shifted=replace(sample,lift_position_m=.051)
        result=space.linearize(shifted,delayed,maximum_age_s=.04)
        checks['delayed_known_plane_has_explicit_two_clocks']=result.available and result.acquisition_time_s==.99 and result.current_time_s==1. and not result.external_current_geometry_known
        expected=-float(k.robot_attached_normal_world@(result.kinematics.point_world-k.point_world))
        checks['gap_proxy_is_robot_motion_relative_old_plane']=abs(result.robot_separation_proxy_m-expected)<1e-14 and not result.pressure_prediction_available
        checks['stale_patch_unavailable']=not space.linearize(replace(sample,time_s=1.2),delayed,maximum_age_s=.04).available
        try:space.linearize(sample,replace(patch,sample_time_s=1.1),maximum_age_s=.04)
        except ValueError:checks['future_patch_rejected']=True
        else:checks['future_patch_rejected']=False
        try:PublicSkinPatch.from_public(dict(robot_body=patch.robot_body,link_point=[0,0,0],link_normal=[1,0,0],sample_time=1.,is_target=True))
        except ValueError:checks['target_identity_not_an_input']=True
        else:checks['target_identity_not_an_input']=False
        try:space.point_kinematics(sample,replace(patch,robot_body=int(plant.object)))
        except ValueError:checks['free_object_body_cannot_be_robot_patch']=True
        else:checks['free_object_body_cannot_be_robot_patch']=False
        baseline=space.point_kinematics(sample,patch)
        oj=int(plant.model.body_jntadr[plant.object]);oq=int(plant.model.jnt_qposadr[oj])
        plant.data.qpos[oq:oq+3]=[20.,-8.,4.];plant.data.qpos[oq+3:oq+7]=[1.,0.,0.,0.]
        external_state=plant.data.qpos.copy()
        result=space.point_kinematics(sample,patch)
        checks['true_free_object_state_not_read_or_mutated']=np.array_equal(result.point_world,baseline.point_world) and np.array_equal(result.jacobian3x25,baseline.jacobian3x25) and np.array_equal(external_state,plant.data.qpos)
        # Even a different inert free-object DEFAULT in the FK copy must not
        # change geometric robot rows. This is not an object-state API input.
        space.model.qpos0[oq:oq+3]=[-50.,70.,100.]
        result=space.point_kinematics(sample,patch)
        checks['inert_object_default_cannot_affect_robot_FK']=np.allclose(result.point_world,baseline.point_world,atol=1e-14,rtol=0.) and np.allclose(result.jacobian3x25,baseline.jacobian3x25,atol=1e-14,rtol=0.)
        checks['missing_lift_velocity_not_faked_as_zero_point_velocity']=baseline.point_velocity_world is None
        basis=MechanicalReferenceBasis(plant.model,lift_lower_m=0.,lift_upper_m=.1,
            closure_lower5=plant.closure_min,closure_upper5=plant.closure_max,open_pose=plant.open_pose,closed_pose=plant.closed_pose)
        rng=np.random.default_rng(44);exact=0;affine_error=0.
        for _ in range(20):
            alpha=plant.closure_min+(plant.closure_max-plant.closure_min)*rng.uniform(.2,.8,5)
            x=np.r_[alpha,.05,.02,-.03];plant.set_closure(alpha);plant.set_motion(x[5],x[6:])
            decoded=basis.decode_held8(x)
            exact+=int(np.array_equal(decoded.actuator_targets21,plant.data.ctrl[decoded.actuator_indices21]))
            affine_error=max(affine_error,float(np.max(abs(decoded.command21-(basis.offset21+basis.held_to_command21@x)))))
        checks['native21_actual_actuator_targets_bitexact_20_cases']=exact==20
        checks['held8_to_reference21_affine_basis_exact']=affine_error<1e-14
        checks['tendon_does_not_supply_instant_qdot_map']=not basis.instantaneous_joint_response_available and all('not divided' in t['role'] for t in basis.transmissions21)
        metrics.update(native_reference_exact_cases=exact,reference_affine_max_abs_error=affine_error)
        missing=MeasuredServoResponse(space.axes25)
        absent=missing.at(.02,np.zeros(8))
        checks['missing_response_matrix_is_None_not_zero']=missing.status=='UNAVAILABLE' and absent.status=='UNAVAILABLE' and absent.matrix25_by_active is None
        K=np.zeros((2,25,8));K[:,0,0]=.3;K[:,1,0]=.7
        doc=dict(schema='measured_robot_servo_response_v1',output_axes25=space.axes25,input_axes8=HELD_AXES8,
            output_units=['rad']*24+['m'],ramp_period_s=.02,execution_period_s=.005,validated_horizons_s=[.02,.04],
            response25x8=K,empirical_error25=np.full((2,25),.008),known_plus8=[True]*7+[False],known_minus8=[True]*8,
            measured_increment_lower8=[-.01]*8,measured_increment_upper8=[.01]*8,context_id='synthetic_only',source_sha256='a'*64)
        measured=MeasuredServoResponse(space.axes25,doc);u=np.zeros(8);u[0]=.002
        supported=measured.at(.02,u,context_id='synthetic_only')
        checks['separate_measured_model_retains_25_output_error']=supported.status=='AVAILABLE_LOCAL_ENGINEERING' and np.array_equal(supported.empirical_error25,np.full(25,.008)) and supported.controlled_increment25[0]==.0006
        u[7]=.001;bad=measured.at(.02,u,context_id='synthetic_only')
        checks['unknown_response_direction_is_unavailable_not_zero']=bad.status=='UNAVAILABLE' and bad.matrix25_by_active is None
        checks['response_context_not_guessed']=measured.at(.02,np.zeros(8)).status=='UNAVAILABLE'
        checks['no_unvalidated_time_interpolation']=measured.at(.03,np.zeros(8),context_id='synthetic_only').status=='UNAVAILABLE'
        checks['physical_time_did_not_advance']=plant.data.time==0.
        checks={k:bool(v) for k,v in checks.items()}
        if not all(checks.values()):raise AssertionError(checks)
        return dict(schema='robot_contact_space_pure_v1',passed=True,checks=checks,metrics=metrics,physics_steps=0,
            scope='One actual robot/model construction and isolated FK/FD/reference algebra only; no controller or force model admission',
            sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'v10/robot_contact_space.py',Path(__file__))})
    finally:
        for n,value in originals.items():setattr(mujoco,n,value)


if __name__=='__main__':
    report=verify();payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    path=ROOT/'review/v10_robot_contact_space_pure_v2.json'
    if path.exists():raise FileExistsError(path)
    with path.open('x',encoding='utf-8') as f:f.write(payload)
    print(payload)

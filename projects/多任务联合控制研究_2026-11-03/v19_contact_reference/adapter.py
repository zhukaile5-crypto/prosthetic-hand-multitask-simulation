"""Separate delayed external projection and current robot motion intents.

Every active digit supplies one correlated 2D measurement. Repeated patches
divide that measurement by pressure share; no foreign actor identity or true
object velocity enters. Current robot J/v is not added to delayed slip.
"""
import numpy as np
from v10.robot_contact_space import RobotJointSample,PublicSkinPatch
from v17_contact_transfer.motion_adapter import PublicMotionAdapter
from v17_contact_transfer.motion_domain_sensors import SLIP_DOMAIN_ROLE
from v16_release_geometry.public_adapter import _array,_mask,_real,_tick
from .design import (EXTERNAL_PROJECTION_DAMPING_N_S_M,
                     ROBOT_PROJECTION_DAMPING_N_S_M)


class ContactMotionAdapter(PublicMotionAdapter):
    def current_contact_intents(self,frame,*,desired_lift_velocity_m_s=0.):
        _,now=_tick(frame.time)
        if self._last_time is None or abs(now-self._last_time)>1e-9:
            raise ValueError('observe current public tick before task construction')
        if self._signature()!=self._binding:raise ValueError('known robot definition changed')
        q=_array(frame.robot_position25,(25,),'current public q25')
        v=_array(frame.robot_velocity25,(25,),'current public v25')
        current=RobotJointSample(now,self.geometry.raw_joint_names,q[:24],q[24],v[:24],v[24])
        desired_lift=_real(desired_lift_velocity_m_s,'declared current task lift velocity')
        valid=_mask(frame.valid,'pressure valid')&_mask(frame.distributed_valid,'distributed valid')
        patches=[];omitted=[]
        for index,raw in enumerate(frame.skin_patches):
            patch=PublicSkinPatch.from_public(raw)
            if patch.digit is None or not valid[patch.digit]:
                omitted.append(dict(index=index,reason='digit observation unavailable'));continue
            y=_real(raw.get('normal'),'patch pressure',nonnegative=True)
            if y<=0:continue
            projection=self.geometry.linearize(current,patch,maximum_age_s=.12)
            if not projection.available:
                omitted.append(dict(index=index,reason=list(projection.reasons)));continue
            patches.append(dict(index=index,patch=patch,digit=patch.digit,normal=y,
                row=projection.normal_jacobian25,J=projection.kinematics.jacobian3x25,
                normal_world=projection.acquisition_normal_world))
        J=np.asarray([p['row'] for p in patches],float).reshape((-1,25))
        y=np.asarray([p['normal'] for p in patches],float)
        digits=np.asarray([p['digit'] for p in patches],int)
        tau_external=np.zeros(25);tau_robot=np.zeros(25);records=[];unknown=[];robot_records=[]
        for digit in range(5):
            group=[p for p in patches if p['digit']==digit]
            total=sum(p['normal'] for p in group)
            for p in group:
                share=p['normal']/total;n=p['normal_world'];velocity=p['J']@v
                target=p['J'][:,24]*desired_lift;velocity_error=velocity-target
                tangent=velocity_error-n*float(n@velocity_error)
                force=-ROBOT_PROJECTION_DAMPING_N_S_M*share*tangent
                contribution=p['J'].T@force;tau_robot+=contribution
                robot_records.append(dict(digit=digit,patch_index=p['index'],read_time_s=now,
                    pressure_share=share,current_robot_tangent_world_m_s=(velocity-n*float(n@velocity)).tolist(),
                    target_robot_tangent_world_m_s=(target-n*float(n@target)).tolist(),
                    robot_tangent_error_world_m_s=tangent.tolist(),task_target_time_s=now,
                    robot_motion_effort_world_N=force.tolist(),generalized_effort25=contribution.tolist(),
                    scope='current robot velocity in old measured patch plane; no external velocity inference'))
        try:
            acq=_real(frame.sample_time,'device acquisition',nonnegative=True)
            age=_real(frame.age,'device age',nonnegative=True)
            if not 0<=age<=.12 or abs(now-acq-age)>1e-9:
                raise ValueError('unknown device acquisition/age')
            if (not isinstance(frame.slip_domain_available,(bool,np.bool_)) or not frame.slip_domain_available or
                frame.slip_domain_role!=SLIP_DOMAIN_ROLE or
                abs(_real(frame.slip_domain_time_s,'domain read time')-now)>1e-9 or
                abs(_real(frame.slip_domain_acquisition_time_s,'domain acquisition')-acq)>1e-9):
                raise ValueError('slip acquisition-domain not authenticated')
            saved=self._motion_samples.get(int(round(acq/.005)))
            if saved is None or saved.get('reason') is not None or abs(saved['time_s']-acq)>1e-9:
                raise ValueError('coacquired robot history missing')
            R=_array(saved['R'],(3,3),'coacquired palm rotation')
            qa=_array(saved['q'],(25,),'coacquired q25');va=_array(saved['v'],(25,),'coacquired v25')
            acquired=RobotJointSample(acq,self.geometry.raw_joint_names,qa[:24],qa[24],va[:24],va[24])
            flags=_mask(frame.slip_valid5,'slip valid')
            basis=np.asarray(frame.tangent_basis,dtype=object);slip=np.asarray(frame.slip_local,dtype=object)
            if basis.shape!=(5,2,3) or slip.shape!=(5,2):raise ValueError('explicit device-plane shapes required')
            for digit in range(5):
                selected=[p for p in patches if p['digit']==digit and abs(p['patch'].sample_time_s-acq)<=1e-9]
                if not selected:
                    unknown.append(dict(digit=digit,reason='missing coacquired geometry'));continue
                try:
                    B=_array(basis[digit],(2,3),'device plane')
                    if not np.allclose(B@B.T,np.eye(2),atol=1e-8,rtol=0):raise ValueError('nonorthonormal device plane')
                    mass=sum(p['normal'] for p in selected);old_velocity=np.zeros(3);mean_J=np.zeros((3,25));old_lift_axis=np.zeros(3)
                    for p in selected:
                        share=p['normal']/mass
                        old=self.geometry.point_kinematics(acquired,p['patch'])
                        old_velocity+=share*old.point_velocity_world
                        old_lift_axis+=share*old.jacobian3x25[:,24]
                        mean_J+=share*p['J']
                    T=B@R.T
                    acquired_robot=T@old_velocity
                    current_robot=T@(mean_J@v)
                    external_target=T@(old_lift_axis*desired_lift)
                    current_robot_target=T@(mean_J[:,24]*desired_lift)
                    s=external=None;f_external=np.zeros(3)
                    if flags[digit]:
                        try:
                            s=_array(slip[digit],(2,),'relative slip')
                            external=acquired_robot+s
                            f_external=-EXTERNAL_PROJECTION_DAMPING_N_S_M*(R@B.T@(external-external_target))
                        except (ValueError,TypeError) as exc:
                            unknown.append(dict(digit=digit,reason='external projection unknown: '+str(exc)))
                    else:unknown.append(dict(digit=digit,reason='external projection unknown; robot motion task remains active'))
                    te=mean_J.T@f_external;tau_external+=te
                    records.append(dict(digit=digit,acquisition_s=acq,read_time_s=now,age_s=age,
                        relative_slip2_m_s=None if s is None else s.tolist(),acquired_robot_projection2_m_s=acquired_robot.tolist(),
                        measured_external_projection2_m_s=None if external is None else external.tolist(),current_robot_projection2_m_s=current_robot.tolist(),
                        external_projection_target2_m_s=external_target.tolist(),
                        external_projection_error2_m_s=None if external is None else (external-external_target).tolist(),
                        current_robot_projection_target2_m_s=current_robot_target.tolist(),
                        current_robot_projection_error2_m_s=(current_robot-current_robot_target).tolist(),
                        task_target_time_s=now,target_is_current_stage_intent_not_delayed_measurement=True,
                        external_projection_available=external is not None,
                        external_effort_intent_world_N=f_external.tolist(),
                        external_generalized_effort25=te.tolist(),
                        patch_indices=[p['index'] for p in selected],
                        patch_normal_leakage_N=[float(p['normal_world']@((p['normal']/mass)*f_external)) for p in selected]))
                except (ValueError,TypeError) as exc:unknown.append(dict(digit=digit,reason=str(exc)))
        except (ValueError,TypeError,AttributeError) as exc:unknown.append(dict(digit=None,reason=str(exc)))
        return dict(J=J,y=y,digits=digits,tau_motion25=tau_external+tau_robot,
            diagnostics=dict(patch_indices=[p['index'] for p in patches],omitted=omitted,
                normal_geometry_known_digits=sorted({p['digit'] for p in patches}),
                motion_active_digits=[p['digit'] for p in records],motion_records=records,motion_unknown=unknown,
                external_projection_active_digits=[p['digit'] for p in records if p['external_projection_available']],
                robot_motion_records=robot_records,robot_motion_active_digits=sorted({p['digit'] for p in robot_records}),
                declared_lift_velocity_target_m_s=desired_lift,task_target_time_s=now,
                motion_task_scope='delayed external absolute projection and current robot point projection are separate',
                no_H_inverse_or_global_derivative_required=True,one_correlated_measurement_per_digit=True,
                target_identity_unknown=True,current_external_twist_unknown=True,
                virtual_effort_not_actual_force_or_passivity_certificate=True))

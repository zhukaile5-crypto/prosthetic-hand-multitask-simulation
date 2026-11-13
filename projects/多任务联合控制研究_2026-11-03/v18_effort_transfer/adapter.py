"""Public pressure/J and signed device-plane effort intents, independent of H.

Only cached, already sampled devices and robot-only FK enter. A clipped motor
does not invalidate unrelated tactile geometry. No object ID, true pose,
future contact solve, or actual external-force realization is inferred.
"""
import numpy as np
from v10.robot_contact_space import RobotJointSample, PublicSkinPatch
from v17_contact_transfer.motion_adapter import PublicMotionAdapter
from v17_contact_transfer.motion_domain_sensors import SLIP_DOMAIN_ROLE
from v16_release_geometry.public_adapter import _array, _mask, _real, _tick
from .design import MAX_GEOMETRY_AGE_S, DEVICE_TANGENT_DAMPING_N_S_M


class EffortContactAdapter(PublicMotionAdapter):
    def current_contact_intents(self, frame):
        _, now = _tick(frame.time)
        if self._last_time is None or abs(now-self._last_time)>1e-9:
            raise ValueError('observe the same public tick before task construction')
        if self._signature()!=self._binding:
            raise ValueError('bound robot or receiver definition changed')
        q=_array(frame.robot_position25,(25,),'current robot q25')
        v=_array(frame.robot_velocity25,(25,),'current robot v25')
        sample=RobotJointSample(now,self.geometry.raw_joint_names,q[:24],q[24],v[:24],v[24])
        valid=_mask(frame.valid,'pressure validity') & _mask(frame.distributed_valid,'distributed validity')
        patches=[]; omitted=[]
        for index, raw in enumerate(frame.skin_patches):
            patch=PublicSkinPatch.from_public(raw)
            if patch.digit is None or not valid[patch.digit]:
                omitted.append(dict(index=index,reason='digit_geometry_not_observed'));continue
            pressure=_real(raw.get('normal'),'observed patch normal',nonnegative=True)
            if pressure<=0:
                omitted.append(dict(index=index,reason='no_positive_patch_pressure'));continue
            projection=self.geometry.linearize(sample,patch,maximum_age_s=MAX_GEOMETRY_AGE_S)
            if not projection.available:
                omitted.append(dict(index=index,reason=list(projection.reasons)));continue
            patches.append(dict(index=index,digit=patch.digit,normal=pressure,
                patch=patch,row=projection.normal_jacobian25,
                jacobian=projection.kinematics.jacobian3x25,
                normal_world=projection.acquisition_normal_world))
        J=np.asarray([p['row'] for p in patches],float).reshape((-1,25))
        y=np.asarray([p['normal'] for p in patches],float)
        digits=np.asarray([p['digit'] for p in patches],int)
        tau=np.zeros(25);tangent=[]; unavailable=[]
        # This zero is an inactive task, not a measured zero slip. Its separate
        # activation/unknown record is retained even when no digit is usable.
        try:
            acq=_real(frame.sample_time,'hand acquisition',nonnegative=True)
            age=_real(frame.age,'hand age',nonnegative=True)
            if abs(now-acq-age)>1e-9 or not 0<=age<=MAX_GEOMETRY_AGE_S:
                raise ValueError('tangent acquisition age invalid')
            if (not isinstance(frame.slip_domain_available,(bool,np.bool_)) or
                not frame.slip_domain_available or frame.slip_domain_role!=SLIP_DOMAIN_ROLE or
                abs(_real(frame.slip_domain_time_s,'domain read time')-now)>1e-9 or
                abs(_real(frame.slip_domain_acquisition_time_s,'domain acquisition')-acq)>1e-9):
                raise ValueError('tangent acquisition domain unknown')
            old=self._motion_samples.get(int(round(acq/.005)))
            if old is None or old.get('reason') is not None or abs(old['time_s']-acq)>1e-9:
                raise ValueError('acquired robot rotation unavailable')
            R=_array(old['R'],(3,3),'acquired palm rotation')
            basis=np.asarray(frame.tangent_basis,dtype=object)
            slip=np.asarray(frame.slip_local,dtype=object)
            if basis.shape!=(5,2,3) or slip.shape!=(5,2):
                raise ValueError('explicit device tangent shapes required')
            slip_valid=_mask(frame.slip_valid5,'slip domain validity')
            for digit in range(5):
                selected=[p for p in patches if p['digit']==digit and abs(p['patch'].sample_time_s-acq)<=1e-9]
                if not slip_valid[digit] or not selected:
                    unavailable.append(dict(digit=digit,reason='slip_or_coacquired_geometry_unknown'));continue
                try:
                    B=_array(basis[digit],(2,3),'device plane basis')
                    s=_array(slip[digit],(2,),'observed device slip')
                    if not np.allclose(B@B.T,np.eye(2),atol=1e-8,rtol=0):
                        raise ValueError('device plane is not orthonormal')
                    f=-DEVICE_TANGENT_DAMPING_N_S_M*(R@B.T@s)
                    mass=sum(p['normal'] for p in selected)
                    contribution=np.zeros(25); leakage=[]
                    for p in selected:
                        share=p['normal']/mass
                        contribution+=p['jacobian'].T@(share*f)
                        leakage.append(float(p['normal_world']@(share*f)))
                    tau+=contribution
                    tangent.append(dict(digit=digit,acquisition_s=acq,age_s=age,
                        slip2_m_s=s.tolist(),external_effort_intent_world_N=f.tolist(),
                        generalized_effort25=contribution.tolist(),patch_indices=[p['index'] for p in selected],
                        patch_normal_leakage_N=leakage,
                        damping_scale_N_s_m=DEVICE_TANGENT_DAMPING_N_S_M))
                except (ValueError,TypeError) as exc:
                    unavailable.append(dict(digit=digit,reason=str(exc)))
        except (ValueError,TypeError,AttributeError) as exc:
            unavailable.append(dict(digit=None,reason=str(exc)))
        return dict(J=J,y=y,digits=digits,tau_tangent25=tau,
            diagnostics=dict(pressure_J_does_not_require_H_or_unclipped_motors=True,
                patch_indices=[p['index'] for p in patches],omitted=omitted,
                normal_geometry_known_digits=sorted({p['digit'] for p in patches}),
                tangent_active_digits=[x['digit'] for x in tangent],tangent_records=tangent,
                tangent_unavailable=unavailable,external_actor_identity='unknown_any_actor_device',
                tangent_scope='single correlated device-plane effort intent per digit, split by measured pressure',
                measured_slip_not_current_external_state=True,
                actual_contact_force_realization_or_passivity_certified=False))

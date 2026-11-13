"""Public contact-average motion correction, NOT an external-state observer.

One existing slip pair per digit, shared acquisition history, current robot
Jacobian, and the V16 equal-effort length proposal map. No object truth,
contact actor identity, integration, forward, contact query, or sensor sample.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import numpy as np
import mujoco

from v13_a1.codec import Reference21, REFERENCE_AXES21
from v10.robot_contact_space import RobotJointSample, PublicSkinPatch
from v16_release_geometry.public_adapter import (PublicReleaseAdapter, _array,
    _mask, _real, _tick, affine_equilibrium_length_derivative)
from v16_release_geometry.geometry_tasks import make_length_proposal_map
from .motion_domain_sensors import SLIP_DOMAIN_ROLE

MOTION_PROPOSAL_PERIOD_S = .02


@dataclass(frozen=True)
class MotionTask:
    time_s: float
    matrix_Mx21: np.ndarray
    target_M: np.ndarray
    metadata: dict

    def __post_init__(self):
        object.__setattr__(self, 'time_s', _real(self.time_s, 'motion task time', nonnegative=True))
        raw = np.asarray(self.matrix_Mx21, dtype=object)
        if raw.ndim != 2 or raw.shape[1] != 21 or raw.shape[0] not in (2,4,6,8,10):
            raise ValueError('One or two projection rows per digit pair, at most five pairs')
        object.__setattr__(self, 'matrix_Mx21', _array(self.matrix_Mx21, raw.shape, 'motion matrix'))
        object.__setattr__(self, 'target_M', _array(self.target_M, (raw.shape[0],), 'motion target'))
        if not isinstance(self.metadata, dict):raise TypeError('Motion metadata dictionary required')
        object.__setattr__(self, 'metadata', deepcopy(self.metadata))

    def to_public(self):
        return dict(time_s=self.time_s, matrix_Mx21=self.matrix_Mx21.tolist(),
                    target_M=self.target_M.tolist(), metadata=deepcopy(self.metadata))


@dataclass(frozen=True)
class FingerMotionProjection:
    digit: int
    tangent_basis2x3: np.ndarray
    acquired_robot_velocity_world3: np.ndarray
    current_robot_jacobian3x25: np.ndarray
    slip2_m_s: np.ndarray

    def __post_init__(self):
        if isinstance(self.digit,(bool,np.bool_)) or not isinstance(self.digit,(int,np.integer)) or not 0 <= self.digit < 5:
            raise ValueError('Explicit digit0..4 required')
        object.__setattr__(self,'digit',int(self.digit))
        for name,shape in (('tangent_basis2x3',(2,3)),('acquired_robot_velocity_world3',(3,)),
                           ('current_robot_jacobian3x25',(3,25)),('slip2_m_s',(2,))):
            object.__setattr__(self,name,_array(getattr(self,name),shape,name))
        if not np.allclose(self.tangent_basis2x3@self.tangent_basis2x3.T,np.eye(2),atol=1e-8,rtol=0):
            raise ValueError('Registered orthonormal instrument projection basis required')


def build_motion_task(*,time_s,acquisition_time_s,rotation_acquisition3x3,
                      reference_to_robot25x21,projections,metadata=None):
    """Build m/reference rows and m targets from measured m/s projections.

    No six-dimensional twist, normal separation speed or unobserved actor is
    filled in. H is only a declared reference-to-geometric-displacement map.
    """
    now=_real(time_s,'read time',nonnegative=True)
    acq=_real(acquisition_time_s,'acquisition time',nonnegative=True)
    if acq>now+1e-9:raise ValueError('Future acquisition is not public history')
    R=_array(rotation_acquisition3x3,(3,3),'acquisition rotation')
    if not np.allclose(R.T@R,np.eye(3),atol=1e-8,rtol=0) or abs(np.linalg.det(R)-1)>1e-8:
        raise ValueError('Known robot acquisition rotation must be proper')
    H=_array(reference_to_robot25x21,(25,21),'reference geometric proposal map')
    ps=tuple(projections)
    if any(type(p) is not FingerMotionProjection for p in ps):raise TypeError('Typed public finger projections required')
    if len({p.digit for p in ps}) != len(ps):raise ValueError('Do not replicate aggregate slip as independent patch measurements')
    if not ps:return None
    matrix=[];target=[];rows=[]
    for p in ps:
        world_to_plane=p.tangent_basis2x3@R.T
        measured_external=world_to_plane@p.acquired_robot_velocity_world3+p.slip2_m_s
        block=world_to_plane@p.current_robot_jacobian3x25@H
        matrix.append(block);target.append(-MOTION_PROPOSAL_PERIOD_S*measured_external)
        rows.append(dict(digit=p.digit,measured_contact_average_projection_m_s=measured_external.tolist(),
                         row_count=2))
    M=np.vstack(matrix);b=np.concatenate(target)
    details=dict(metadata or {})
    details.update(role='OBSERVED_CONTACT_AVERAGE_MOTION_DAMPING_PROPOSAL',
        acquisition_time_s=acq,read_time_s=now,age_s=now-acq,proposal_period_s=MOTION_PROPOSAL_PERIOD_S,
        matrix_units='metres per reference coordinate (rad or m)',target_units='metres',
        rows=rows,rows_per_digit=2,task_matrix_rank=int(np.linalg.matrix_rank(M)),
        external_actor_identity='UNKNOWN; no target identity filtering',
        external_rigid_body_twist_available=False,normal_velocity_available=False,
        shared_finger_position_noise=True,complete_error_covariance_available=False,
        covariance_scope='No whitening: excludes shared position errors-in-variables, robot velocity error and actor mixing',
        held_contact_weights_assumed=True,current_external_geometry_known=False,
        reference_map_role='raw affine-effort length-level-set geometric proposal; not finite servo response',
        delayed_measurement_not_current_external_state=True,
        future_motion_prediction_available=False,future_pressure_prediction_available=False,
        force_safety_certified=False,placement_certified=False)
    return MotionTask(now,M,b,details)


class PublicMotionAdapter(PublicReleaseAdapter):
    """V16 release adapter plus a separate optional typed motion task.

    Parent up/H/load-prior/receiver evidence remains untouched. observe caches
    current public robot encoders, including velocity, by acquisition clock.
    current_geometry_tasks returns the original GeometryTasks type; consumers
    may separately read motion_task or last_motion_diagnostics.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._motion_samples={};self._motion_order=deque()
        self._motion_last_tick=None;self._motion_signature=None
        self._motion_task=None
        self.last_motion_diagnostics=dict(status='UNKNOWN_NOT_OBSERVED')

    @property
    def motion_task(self):return self._motion_task

    def observe(self,frame):
        tick,now=_tick(frame.time)
        def public(value):
            if isinstance(value,np.ndarray):return value.tolist()
            if isinstance(value,np.generic):return value.item()
            return value
        fields=('robot_velocity25','slip_local','slip_valid5','slip_domain_acquisition_time_s',
                'slip_domain_time_s','slip_domain_role','slip_domain_available','skin_patches')
        signature=json.dumps({k:public(getattr(frame,k,None)) for k in fields},sort_keys=True,allow_nan=True)
        if tick==self._motion_last_tick and signature!=self._motion_signature:
            raise ValueError('Consumed motion payload changed on duplicate public tick')
        hand=super().observe(frame)
        if tick==self._motion_last_tick:return hand
        self._motion_task=None
        try:
            q=_array(frame.robot_position25,(25,),'public current q25')
            v=_array(frame.robot_velocity25,(25,),'public current v25')
            sample=RobotJointSample(now,self.geometry.raw_joint_names,q[:24],q[24],v[:24],v[24])
            palm=int(self.geometry.model.body('rh_palm').id)
            self.geometry.point_kinematics(sample,PublicSkinPatch(palm,np.zeros(3),np.array([0.,0.,1.]),now))
            R=self.geometry.data.xmat[palm].reshape(3,3).copy()
            saved=dict(time_s=now,q=q,v=v,R=R,reason=None)
        except (AttributeError,TypeError,ValueError) as exc:
            saved=dict(time_s=now,reason='public_robot_velocity_or_FK_unknown: '+str(exc))
        self._motion_samples[tick]=saved;self._motion_order.append(tick)
        while len(self._motion_order)>256:self._motion_samples.pop(self._motion_order.popleft(),None)
        self._motion_last_tick=tick;self._motion_signature=signature
        self.last_motion_diagnostics=dict(status='OBSERVED_WAITING_FOR_GEOMETRY',robot_snapshot_reason=saved['reason'])
        return hand

    def _motion_mapping(self,frame,current,lo,hi):
        """Repeat parent's authenticated robot algebra, not its observation mask.

        A partial slip packet may still provide other digits' projection rows;
        no missing finger is treated as absent contact or zero external motion.
        """
        now=float(frame.time)
        if self._signature()!=self._binding:raise ValueError('known robot model definition changed')
        if abs(_real(frame.robot_control_time_s,'public control time',nonnegative=True)-now)>1e-9:
            raise ValueError('public command clock mismatch')
        if _array(frame.robot_reference21,(21,),'authoritative reference').tobytes()!=current.tobytes():
            raise ValueError('non-authoritative reference')
        engine=_array(frame.robot_actuator_target21,(21,),'public engine target')
        if self.actor.encode(Reference21(current,now)).values.tobytes()!=engine.tobytes():
            raise ValueError('public reference/engine mismatch')
        q=_array(frame.robot_position25,(25,),'public q25');v=_array(frame.robot_velocity25,(25,),'public v25')
        lower=self.actor.ctrl_range21[:,0].copy();upper=self.actor.ctrl_range21[:,1].copy()
        lower[18]-=self.actor.lift_bias_m;upper[18]-=self.actor.lift_bias_m
        if np.any(current+lo<lower-1e-12) or np.any(current+hi>upper+1e-12):
            raise ValueError('real reference control box exceeded')
        state=self.servo.evaluate(q,v,engine,time_s=now)
        if state.status!='KNOWN_ROBOT_ACTUATION_ALGEBRA':raise ValueError('native robot algebra unavailable')
        if np.any(state.diagnostics['control_clipped21']) or np.any(state.diagnostics['force_clipped21']):
            raise ValueError('clipped native branch is not a known length proposal map')
        m=self.servo.model;G=m.actuator_gainprm[self.servo.aids,:3].copy();B=m.actuator_biasprm[self.servo.aids,:3].copy()
        G[np.asarray(m.actuator_gaintype[self.servo.aids])==mujoco.mjtGain.mjGAIN_FIXED,1:]=0.
        B[np.asarray(m.actuator_biastype[self.servo.aids])==mujoco.mjtBias.mjBIAS_NONE]=0.
        D,diagnostic=affine_equilibrium_length_derivative(gain21x3=G,bias21x3=B,
            length21=state.actuator_length21,speed21=state.actuator_velocity21,engine21=state.effective_target21)
        if D is None:raise ValueError('raw affine length derivative unknown')
        mapping=make_length_proposal_map(transmission21x25=state.transmission21x25,reference_to_length21=D,
            joint_scale25=self.joint_scale25,robot_axes25=self.axes25,reference_axes21=REFERENCE_AXES21,
            time_s=now,declared_source='Same V16 raw fixed-speed equal-effort geometric length proposal')
        return mapping,diagnostic,RobotJointSample(now,self.geometry.raw_joint_names,q[:24],q[24],v[:24],v[24])

    def current_geometry_tasks(self,frame,current21,desired21,lower_delta21,upper_delta21):
        result=super().current_geometry_tasks(frame,current21,desired21,lower_delta21,upper_delta21)
        self._motion_task=None
        skipped=[]
        try:
            now=float(frame.time);acq=_real(frame.sample_time,'slip acquisition',nonnegative=True)
            if getattr(frame,'slip_domain_role',None)!=SLIP_DOMAIN_ROLE or getattr(frame,'slip_domain_available',None) is not True:
                raise ValueError('existing instrument slip domain metadata unavailable')
            if abs(_real(frame.slip_domain_acquisition_time_s,'slip domain acquisition',nonnegative=True)-acq)>1e-9 or abs(_real(frame.slip_domain_time_s,'slip domain read time',nonnegative=True)-now)>1e-9:
                raise ValueError('slip domain clock mismatch')
            if abs(_real(frame.age,'slip age',nonnegative=True)-(now-acq))>1e-9 or now-acq < -1e-9 or now-acq>.12+1e-9:
                raise ValueError('slip packet age unknown/outside parent geometry age')
            ak=int(round(acq/.005));old=self._motion_samples.get(ak)
            if old is None or abs(old['time_s']-acq)>1e-9 or old['reason'] is not None:
                raise ValueError('same-acquisition public robot q/v/FK history missing')
            valid=_mask(frame.slip_valid5,'slip domain mask')&_mask(frame.valid,'public finger valid')
            mapping,derivative,current_sample=self._motion_mapping(frame,
                _array(current21,(21,),'current'),_array(lower_delta21,(21,),'lower'),_array(upper_delta21,(21,),'upper'))
            old_sample=RobotJointSample(acq,self.geometry.raw_joint_names,old['q'][:24],old['q'][24],old['v'][:24],old['v'][24])
            basis=np.asarray(frame.tangent_basis,dtype=object);slips=np.asarray(frame.slip_local,dtype=object)
            if basis.shape!=(5,2,3) or slips.shape!=(5,2):raise ValueError('five instrument projection pairs required')
            projections=[];digit_details=[]
            for digit in range(5):
                if not valid[digit]:skipped.append(dict(digit=digit,reason='slip_measurement_invalid_not_stationary'));continue
                patches=[p for p in frame.skin_patches if p.get('digit')==digit]
                if not patches:skipped.append(dict(digit=digit,reason='public_skin_geometry_missing'));continue
                try:
                    weights=np.array([_real(p.get('normal'),'public pressure share',nonnegative=True) for p in patches])
                    if np.any(weights<=0) or not np.isfinite(weights.sum()) or weights.sum()<=0:raise ValueError('positive finite public patch shares required')
                    weights/=weights.sum();vel=[];jac=[]
                    for p in patches:
                        patch=PublicSkinPatch.from_public(p)
                        if abs(patch.sample_time_s-acq)>1e-9:raise ValueError('patch/slip acquisition mismatch')
                        acquired=self.geometry.point_kinematics(old_sample,patch)
                        current=self.geometry.point_kinematics(current_sample,patch)
                        vel.append(acquired.point_velocity_world);jac.append(current.jacobian3x25)
                    meanv=weights@np.asarray(vel);meanJ=np.einsum('p,pij->ij',weights,np.asarray(jac))
                    projections.append(FingerMotionProjection(digit,basis[digit],meanv,meanJ,slips[digit]))
                    digit_details.append(dict(digit=digit,patch_count=len(patches),pressure_share=weights.tolist()))
                except (ValueError,TypeError,KeyError,AttributeError) as exc:
                    skipped.append(dict(digit=digit,reason='public_projection_unknown: '+str(exc)))
            task=build_motion_task(time_s=now,acquisition_time_s=acq,rotation_acquisition3x3=old['R'],
                reference_to_robot25x21=mapping.H25x21,projections=projections,
                metadata=dict(parent_geometry_status=result.status,selected_public_digits=[p.digit for p in projections],
                    digit_geometry=digit_details,skipped=skipped,reference_length_derivative=derivative,
                    map_robot_axes25=list(mapping.robot_axes25),map_reference_axes21=list(mapping.reference_axes21),
                    robot_membership_authenticated=True))
            self._motion_task=task
            self.last_motion_diagnostics=(dict(status='UNAVAILABLE_NO_VALID_PUBLIC_MOTION_ROWS',skipped=skipped)
                if task is None else dict(status='EXPERIMENTAL_PUBLIC_MOTION_TASK',**task.metadata))
        except (ValueError,TypeError,KeyError,AttributeError,np.linalg.LinAlgError) as exc:
            self.last_motion_diagnostics=dict(status='UNKNOWN_PUBLIC_MOTION_TASK',reason=str(exc),skipped=skipped,
                future_pressure_prediction_available=False,external_twist_available=False)
        return replace(result,diagnostics=dict(result.diagnostics,
            current_time_s=float(frame.time),motion_domain=deepcopy(self.last_motion_diagnostics)))

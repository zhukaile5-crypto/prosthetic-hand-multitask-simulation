"""Robot-only contact geometry and exact mechanical REFERENCE coordinates.

No pressure, object pose, collision identity, force posterior, or controller
decision is inferred. q25 = raw24 named angular encoders, then lift (m).
command21 = 18 finger actuator references, lift, WRJ2, WRJ1.
held8 = alpha5, lift, WRJ2, WRJ1. These are reference coordinates, NOT qdot.

MuJoCo's documented geometry pipeline for mj_jac is mj_kinematics followed by
mj_comPos: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
Only isolated FK data is touched; mj_forward/step are never called here.
"""
from __future__ import annotations
from dataclasses import dataclass
from copy import copy
from numbers import Real
from typing import Optional
import numpy as np
import mujoco
import research_imports
from foundation_config import OPEN,CLOSED,GROUPS

FINGER_AXES=tuple('rh_A_'+name for group in GROUPS for name in group)
COMMAND_AXES21=FINGER_AXES+('lift_reference_m','WRJ2_reference_rad','WRJ1_reference_rad')
HELD_AXES8=tuple('alpha_'+str(i) for i in range(5))+('lift_reference_m','WRJ2_reference_rad','WRJ1_reference_rad')


def _scalar(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value):raise ValueError(name+' must be finite real')
    return float(value)


def _array(value,shape,name):
    a=np.asarray(value,float)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'{name}: expected finite {shape}')
    a=a.copy();a.setflags(write=False);return a


def _unit(value,name):
    a=_array(value,(3,),name);length=float(np.linalg.norm(a))
    if length<1e-10:raise ValueError(name+' has no defined direction')
    return _array(a/length,(3,),name)


@dataclass(frozen=True)
class RobotJointSample:
    time_s:float
    raw_joint_names:tuple
    raw_position24:np.ndarray
    lift_position_m:float
    raw_velocity24:Optional[np.ndarray]=None
    lift_velocity_m_s:Optional[float]=None

    def __post_init__(self):
        object.__setattr__(self,'time_s',_scalar(self.time_s,'robot time'))
        names=tuple(self.raw_joint_names)
        if len(names)!=24 or len(set(names))!=24 or any(not isinstance(n,str) or not n.startswith('rh_') for n in names):raise ValueError('Exactly24 named robot angular encoders required')
        object.__setattr__(self,'raw_joint_names',names)
        object.__setattr__(self,'raw_position24',_array(self.raw_position24,(24,),'raw robot positions'))
        object.__setattr__(self,'lift_position_m',_scalar(self.lift_position_m,'robot lift'))
        if self.raw_velocity24 is not None:object.__setattr__(self,'raw_velocity24',_array(self.raw_velocity24,(24,),'raw robot velocities'))
        if self.lift_velocity_m_s is not None:object.__setattr__(self,'lift_velocity_m_s',_scalar(self.lift_velocity_m_s,'robot lift velocity'))

    @property
    def q25(self):return np.r_[self.raw_position24,self.lift_position_m]

    @property
    def velocity25(self):
        if self.raw_velocity24 is None or self.lift_velocity_m_s is None:return None
        return np.r_[self.raw_velocity24,self.lift_velocity_m_s]

    @classmethod
    def from_public_frame(cls,frame,raw_joint_names,*,lift_velocity_m_s=None):
        # Explicit field selection. Full simulator qpos is not an API input.
        return cls(frame.time,tuple(raw_joint_names),frame.raw_joint,frame.lift,
            getattr(frame,'raw_joint_velocity',None),lift_velocity_m_s)


@dataclass(frozen=True)
class PublicSkinPatch:
    robot_body:int
    link_point:np.ndarray
    link_normal:np.ndarray
    sample_time_s:float
    reference_position:Optional[np.ndarray]=None
    reference_normal:Optional[np.ndarray]=None
    digit:Optional[int]=None

    def __post_init__(self):
        if isinstance(self.robot_body,(bool,np.bool_)) or not isinstance(self.robot_body,(int,np.integer)):raise ValueError('Integer known robot body required')
        object.__setattr__(self,'robot_body',int(self.robot_body))
        object.__setattr__(self,'link_point',_array(self.link_point,(3,),'robot link point'))
        object.__setattr__(self,'link_normal',_unit(self.link_normal,'robot link normal'))
        object.__setattr__(self,'sample_time_s',_scalar(self.sample_time_s,'patch acquisition time'))
        if self.reference_position is not None:object.__setattr__(self,'reference_position',_array(self.reference_position,(3,),'acquisition reference point'))
        if self.reference_normal is not None:object.__setattr__(self,'reference_normal',_unit(self.reference_normal,'acquisition reference normal'))
        if self.digit is not None and (isinstance(self.digit,bool) or not isinstance(self.digit,(int,np.integer)) or not 0<=self.digit<5):raise ValueError('Optional digit must be0..4')

    @classmethod
    def from_public(cls,value):
        if not isinstance(value,dict):raise ValueError('Public skin patch dictionary required')
        allowed={'robot_body','link_point','link_normal','sample_time','reference_position','reference_normal','digit','normal','age'}
        if set(value)-allowed:raise ValueError('Unexpected patch fields; object IDs/true world state are not inputs')
        return cls(value['robot_body'],value['link_point'],value['link_normal'],value['sample_time'],
            value.get('reference_position'),value.get('reference_normal'),value.get('digit'))


@dataclass(frozen=True)
class PointKinematics:
    point_world:np.ndarray
    robot_attached_normal_world:np.ndarray
    jacobian3x25:np.ndarray
    angular_jacobian3x25:np.ndarray
    point_velocity_world:Optional[np.ndarray]
    time_s:float
    axes25:tuple


@dataclass(frozen=True)
class ContactProjection:
    status:str
    reasons:tuple
    kinematics:Optional[PointKinematics]
    acquisition_normal_world:Optional[np.ndarray]
    normal_jacobian25:Optional[np.ndarray]
    robot_separation_proxy_m:Optional[float]
    acquisition_time_s:float
    current_time_s:float
    external_current_geometry_known:bool=False
    pressure_prediction_available:bool=False
    global_admitted:bool=False

    @property
    def available(self):return self.status=='AVAILABLE_ACQUISITION_PLANE'


class RobotContactSpace:
    def __init__(self,model,raw_joint_names=None):
        # Separate model/data handles; no reference to the plant's MjData.
        self.model=copy(model);self.data=mujoco.MjData(self.model)
        names=tuple(raw_joint_names) if raw_joint_names is not None else tuple(
            self.model.joint(j).name for j in range(self.model.njnt) if (self.model.joint(j).name or '').startswith('rh_'))
        if len(names)!=24 or len(set(names))!=24:raise ValueError('Robot model must expose exactly24 ordered rh_ joints')
        self.raw_joint_names=names;self.axes25=names+('transport_lift',)
        jids=[int(self.model.joint(name).id) for name in self.axes25]
        if any(self.model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE for j in jids[:24]) or self.model.jnt_type[jids[24]]!=mujoco.mjtJoint.mjJNT_SLIDE:
            raise ValueError('Angular24 plus one linear lift required, no free-object coordinates')
        self.qadr=np.asarray([self.model.jnt_qposadr[j] for j in jids],int)
        self.dadr=np.asarray([self.model.jnt_dofadr[j] for j in jids],int)
        self.joint_limited25=np.asarray(self.model.jnt_limited[jids],bool).copy()
        self.joint_position_intervals25=np.asarray(self.model.jnt_range[jids],float).copy()
        root=int(self.model.body('rh_forearm').id);self.robot_bodies=set()
        for body in range(self.model.nbody):
            ancestor=body
            while ancestor>0 and ancestor!=root:ancestor=int(self.model.body_parentid[ancestor])
            if ancestor==root:self.robot_bodies.add(body)

    def _body(self,body):
        if isinstance(body,bool) or not isinstance(body,(int,np.integer)) or int(body) not in self.robot_bodies:raise ValueError('Patch body is outside the known robot subtree')
        return int(body)

    def point_kinematics(self,sample:RobotJointSample,patch:PublicSkinPatch):
        if not isinstance(sample,RobotJointSample) or not isinstance(patch,PublicSkinPatch):raise TypeError('Strict robot sample and public patch required')
        if sample.raw_joint_names!=self.raw_joint_names:raise ValueError('Robot encoder order mismatch')
        body=self._body(patch.robot_body)
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[self.qadr]=sample.q25;self.data.time=sample.time_s
        if sample.velocity25 is not None:self.data.qvel[self.dadr]=sample.velocity25
        mujoco.mj_kinematics(self.model,self.data);mujoco.mj_comPos(self.model,self.data)
        R=self.data.xmat[body].reshape(3,3);point=self.data.xpos[body]+R@patch.link_point
        J=np.zeros((3,self.model.nv));Jr=np.zeros_like(J)
        mujoco.mj_jac(self.model,self.data,J,Jr,point,body)
        J=J[:,self.dadr];Jr=Jr[:,self.dadr]
        velocity=None if sample.velocity25 is None else J@sample.velocity25
        return PointKinematics(_array(point,(3,),'world robot point'),_array(R@patch.link_normal,(3,),'world robot-attached normal'),
            _array(J,(3,25),'robot point Jacobian'),_array(Jr,(3,25),'robot angular Jacobian'),
            None if velocity is None else _array(velocity,(3,),'robot point velocity'),sample.time_s,self.axes25)

    def linearize(self,sample,patch,*,maximum_age_s):
        maximum_age_s=_scalar(maximum_age_s,'declared geometry age')
        if maximum_age_s<=0:raise ValueError('Explicit positive maximum geometry age required')
        age=sample.time_s-patch.sample_time_s
        if age< -1e-9:raise ValueError('Future acquisition cannot enter robot contact geometry')
        if age>maximum_age_s+1e-9:
            return ContactProjection('UNAVAILABLE',('patch_geometry_stale',),None,None,None,None,patch.sample_time_s,sample.time_s)
        delayed=age>1e-9
        if delayed and (patch.reference_position is None or patch.reference_normal is None):
            return ContactProjection('UNAVAILABLE',('delayed_patch_missing_acquisition_reference',),None,None,None,None,patch.sample_time_s,sample.time_s)
        k=self.point_kinematics(sample,patch)
        normal=k.robot_attached_normal_world if patch.reference_normal is None else patch.reference_normal
        reference=k.point_world if patch.reference_position is None else patch.reference_position
        row=normal@k.jacobian3x25
        # Positive is robot withdrawal relative to the OLD observed tangent
        # plane. It is not the current external gap or relative object speed.
        separation=-float(normal@(k.point_world-reference))
        return ContactProjection('AVAILABLE_ACQUISITION_PLANE',(),k,_array(normal,(3,),'acquisition plane normal'),
            _array(row,(25,),'normal geometric row'),separation,patch.sample_time_s,sample.time_s)


@dataclass(frozen=True)
class DecodedMechanicalReference:
    command21:np.ndarray
    actuator_indices21:np.ndarray
    actuator_targets21:np.ndarray
    held_reference8:np.ndarray
    role:str='mechanical reference only; no instantaneous qdot/servo/force map'


class MechanicalReferenceBasis:
    def __init__(self,model,*,lift_lower_m,lift_upper_m,closure_lower5=None,closure_upper5=None,open_pose=None,closed_pose=None):
        self.model=copy(model);self.command_axes21=COMMAND_AXES21;self.held_axes8=HELD_AXES8
        self.groups=tuple(np.array([self.model.actuator('rh_A_'+n).id for n in group],int) for group in GROUPS)
        self.finger_aids=np.concatenate(self.groups)
        self.lift_aid=int(self.model.actuator('lift_motor').id)
        self.wrist_aids=np.array([self.model.actuator('rh_A_WRJ2').id,self.model.actuator('rh_A_WRJ1').id],int)
        self.actuator_indices21=np.r_[self.finger_aids,self.lift_aid,self.wrist_aids]
        self.open_pose=tuple(_array(v,(n,),'open pose') for v,n in zip(OPEN if open_pose is None else open_pose,(5,3,3,3,4)))
        self.closed_pose=tuple(_array(v,(n,),'closed pose') for v,n in zip(CLOSED if closed_pose is None else closed_pose,(5,3,3,3,4)))
        if len(self.open_pose)!=5 or len(self.closed_pose)!=5:raise ValueError('Five complete finger groups required')
        self.offset21=np.r_[np.concatenate(self.open_pose),0.,0.,0.]
        self.held_to_command21=np.zeros((21,8));start=0;cl=[];cu=[]
        for i,aids in enumerate(self.groups):
            slope=self.closed_pose[i]-self.open_pose[i];self.held_to_command21[start:start+len(aids),i]=slope
            low=-np.inf;high=np.inf
            for k,aid in enumerate(aids):
                if abs(slope[k])<1e-12:
                    if not self.model.actuator_ctrlrange[aid,0]<=self.open_pose[i][k]<=self.model.actuator_ctrlrange[aid,1]:raise ValueError('Constant finger reference outside actuator interval')
                    continue
                bounds=(self.model.actuator_ctrlrange[aid]-self.open_pose[i][k])/slope[k]
                low=max(low,float(min(bounds)));high=min(high,float(max(bounds)))
            cl.append(low);cu.append(high);start+=len(aids)
        self.held_to_command21[18:,5:]=np.eye(3)
        lower=np.asarray(self.model.actuator_ctrlrange[self.actuator_indices21,0],float).copy()
        upper=np.asarray(self.model.actuator_ctrlrange[self.actuator_indices21,1],float).copy()
        hand=int(self.model.body('rh_forearm').id);kp=float(self.model.actuator_gainprm[self.lift_aid,0])
        if not np.isfinite(kp) or kp<=0:raise ValueError('Known positive lift position-servo gain required for mechanical offset')
        self.lift_motor_offset=float(self.model.body_subtreemass[hand])*9.81/kp
        lower[18]-=self.lift_motor_offset;upper[18]-=self.lift_motor_offset
        lower[18]=max(lower[18],_scalar(lift_lower_m,'declared lift lower'));upper[18]=min(upper[18],_scalar(lift_upper_m,'declared lift upper'))
        for slot,name in ((18,'transport_lift'),(19,'rh_WRJ2'),(20,'rh_WRJ1')):
            jid=int(self.model.joint(name).id)
            if self.model.jnt_limited[jid]:lower[slot]=max(lower[slot],float(self.model.jnt_range[jid,0]));upper[slot]=min(upper[slot],float(self.model.jnt_range[jid,1]))
        self.command_lower21=_array(lower,(21,),'real reference lower');self.command_upper21=_array(upper,(21,),'real reference upper')
        cl=np.asarray(cl);cu=np.asarray(cu)
        if closure_lower5 is not None:cl=np.maximum(cl,_array(closure_lower5,(5,),'common closure lower'))
        if closure_upper5 is not None:cu=np.minimum(cu,_array(closure_upper5,(5,),'common closure upper'))
        self.held_lower8=_array(np.r_[cl,lower[18:]],(8,),'held reference lower');self.held_upper8=_array(np.r_[cu,upper[18:]],(8,),'held reference upper')
        if np.any(lower>upper) or np.any(self.held_lower8>self.held_upper8):raise ValueError('Empty mechanical reference domain')
        self.instantaneous_joint_response_available=False
        self.transmissions21=tuple(dict(axis=COMMAND_AXES21[i],actuator_name=self.model.actuator(int(aid)).name,
            transmission_type=int(self.model.actuator_trntype[aid]),transmission_id=int(self.model.actuator_trnid[aid,0]),
            gear=np.asarray(self.model.actuator_gear[aid],float).tolist(),
            role='actual actuator reference; tendon target is not divided between joint encoders') for i,aid in enumerate(self.actuator_indices21))

    def decode_held8(self,value):
        x=_array(value,(8,),'held reference')
        if np.any(x<self.held_lower8-1e-10) or np.any(x>self.held_upper8+1e-10):raise ValueError('Held reference outside declared mechanical authority')
        # Same operation order as native open+alpha*slope, preserving bits.
        fingers=np.concatenate([self.open_pose[i]+x[i]*(self.closed_pose[i]-self.open_pose[i]) for i in range(5)])
        command=np.r_[fingers,x[5:]]
        if np.any(command<self.command_lower21-1e-10) or np.any(command>self.command_upper21+1e-10):raise ValueError('Decoded actuator reference outside authority')
        targets=command.copy();targets[18]+=self.lift_motor_offset
        return DecodedMechanicalReference(_array(command,(21,),'actual command21'),self.actuator_indices21.copy(),
            _array(targets,(21,),'actual actuator targets'),x)


@dataclass(frozen=True)
class ServoResponseSlice:
    status:str
    reasons:tuple
    active_input_axes:tuple=()
    matrix25_by_active:Optional[np.ndarray]=None
    controlled_increment25:Optional[np.ndarray]=None
    empirical_error25:Optional[np.ndarray]=None
    pressure_prediction_available:bool=False
    global_admitted:bool=False


class MeasuredServoResponse:
    """Adapter for separately measured finite command->actual robot q response.

    Missing model/direction/context returns UNAVAILABLE with matrix=None, never
    an invented zero physical gain. No map21->q, half-split tendon model, fit,
    interpolation, global admission or measured response is created here.
    """
    def __init__(self,axes25,document=None):
        self.axes25=tuple(axes25);self.document=document
        if len(self.axes25)!=25:raise ValueError('Explicit25 robot axes required')
        if document is None:return
        if document.get('schema')!='measured_robot_servo_response_v1' or tuple(document['output_axes25'])!=self.axes25 or tuple(document['input_axes8'])!=HELD_AXES8:raise ValueError('Measured servo axes/schema mismatch')
        if document.get('ramp_period_s')!=.02 or document.get('execution_period_s')!=.005:raise ValueError('Only separately measured20ms four-tick finite ramp supported')
        if document.get('global_model_admitted',False):raise ValueError('This adapter grants no global admission')
        if tuple(document.get('output_units',()))!=('rad',)*24+('m',):raise ValueError('Raw joint/lift response units required')
        self.times=_array(document['validated_horizons_s'],(len(document['validated_horizons_s']),),'validated horizons')
        if not len(self.times) or np.any(np.diff(self.times)<=0):raise ValueError('Ordered explicitly validated horizons required')
        self.K=_array(document['response25x8'],(len(self.times),25,8),'measured servo response')
        self.error=_array(document['empirical_error25'],(len(self.times),25),'complete robot response error')
        if np.any(self.error<0):raise ValueError('Negative measured response error')
        self.plus=np.asarray(document['known_plus8']);self.minus=np.asarray(document['known_minus8'])
        if self.plus.shape!=(8,) or self.minus.shape!=(8,) or self.plus.dtype.kind!='b' or self.minus.dtype.kind!='b':raise ValueError('Explicit directional availability flags required')
        self.lower=_array(document['measured_increment_lower8'],(8,),'measured finite lower');self.upper=_array(document['measured_increment_upper8'],(8,),'measured finite upper')
        provenance=document.get('source_sha256','')
        if np.any(self.lower>self.upper) or not isinstance(document.get('context_id'),str) or not document['context_id'] or len(provenance)!=64 or any(c not in '0123456789abcdef' for c in provenance):raise ValueError('Measured scope and exact provenance required')

    @property
    def status(self):return 'UNAVAILABLE' if self.document is None else 'LOCAL_MODEL_ARTIFACT_ONLY'

    def at(self,horizon_s,requested_increment8,*,context_id=None):
        request=_array(requested_increment8,(8,),'actual requested finite increment')
        if self.document is None:return ServoResponseSlice('UNAVAILABLE',('measured_servo_response_missing',))
        if context_id!=self.document['context_id']:return ServoResponseSlice('UNAVAILABLE',('measured_context_not_validated',))
        h=_scalar(horizon_s,'servo response horizon');match=np.flatnonzero(abs(self.times-h)<1e-10)
        if len(match)!=1:return ServoResponseSlice('UNAVAILABLE',('horizon_not_independently_validated',))
        missing=((request>1e-12)&~self.plus)|((request< -1e-12)&~self.minus)
        if missing.any():return ServoResponseSlice('UNAVAILABLE',tuple('unmeasured_input_direction_'+str(i) for i in np.flatnonzero(missing)))
        if np.any(request<self.lower-1e-12) or np.any(request>self.upper+1e-12):return ServoResponseSlice('UNAVAILABLE',('finite_input_outside_measured_scope',))
        active=np.flatnonzero(abs(request)>1e-12);k=int(match[0]);M=self.K[k][:,active]
        return ServoResponseSlice('AVAILABLE_LOCAL_ENGINEERING',(),tuple(HELD_AXES8[i] for i in active),
            _array(M,(25,len(active)),'selected measured map'),_array(M@request[active],(25,),'controlled robot increment'),
            _array(self.error[k],(25,),'complete empirical robot error'))

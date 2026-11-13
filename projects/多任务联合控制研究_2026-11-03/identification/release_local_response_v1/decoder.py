"""Pure explicit native path progress + lift references, never a force model.

phi is native path time in seconds. It is not holding alpha or actual joint
motion. Finger/tendon targets preserve their real native reference units.
"""
from dataclasses import dataclass
from numbers import Real
import numpy as np

FINGER_AXES=tuple('rh_A_'+n for n in ('THJ5','THJ4','THJ3','THJ2','THJ1','FFJ4','FFJ3','FFJ0','MFJ4','MFJ3','MFJ0','RFJ4','RFJ3','RFJ0','LFJ5','LFJ4','LFJ3','LFJ0'))
COMMAND_AXES21=FINGER_AXES+('lift_reference_m','WRJ2_reference_rad','WRJ1_reference_rad')
COMMAND_UNITS21=('rad',)*18+('m','rad','rad')
def scalar(value,label):
 if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value):raise ValueError(label+' requires a finite real')
 return float(value)
def vector(value,size,label):
 raw=np.asarray(value,dtype=object)
 if raw.shape!=(size,) or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not np.isfinite(x) for x in raw):raise ValueError(label+' requires finite real vector')
 a=np.asarray(value,float).copy();a.setflags(write=False);return a
def sha64(value):return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)

@dataclass(frozen=True)
class ReleaseCoordinate:
 phi_s:float
 lift_reference_m:float
 def __post_init__(self):
  object.__setattr__(self,'phi_s',scalar(self.phi_s,'native release progress'))
  object.__setattr__(self,'lift_reference_m',scalar(self.lift_reference_m,'lift reference'))
  if self.phi_s<0:raise ValueError('Native progress cannot precede release entry')

@dataclass(frozen=True)
class NativeEntry:
 alpha5:np.ndarray
 thumb_reference5:np.ndarray
 executed_lift_reference_m:float
 wrist_reference2:np.ndarray
 robot_model_sha256:str
 def __post_init__(self):
  for name,n in [('alpha5',5),('thumb_reference5',5),('wrist_reference2',2)]:object.__setattr__(self,name,vector(getattr(self,name),n,name))
  object.__setattr__(self,'executed_lift_reference_m',scalar(self.executed_lift_reference_m,'executed lift reference'))
  if not sha64(self.robot_model_sha256):raise ValueError('Exact actual robot model identity required')

@dataclass(frozen=True)
class DecodedReference:
 coordinate:ReleaseCoordinate
 command21:np.ndarray
 alpha5:np.ndarray
 pressure_prediction_available:bool=False
 instantaneous_robot25_response_available:bool=False
 material_safety_certified:bool=False

class NativePath2D:
 def __init__(self,entry:NativeEntry,*,open_pose,closed_pose,finger_lower18,finger_upper18,release_open_rate):
  if not isinstance(entry,NativeEntry):raise TypeError('Public robot native entry required')
  if len(open_pose)!=5 or len(closed_pose)!=5:raise ValueError('Exactly five native finger groups required')
  self.entry=entry
  self.open_pose=tuple(vector(v,n,'native open pose') for v,n in zip(open_pose,(5,3,3,3,4)))
  self.closed_pose=tuple(vector(v,n,'native closed pose') for v,n in zip(closed_pose,(5,3,3,3,4)))
  if len(self.open_pose)!=5 or len(self.closed_pose)!=5:raise ValueError('All native finger groups required')
  self.lower=vector(finger_lower18,18,'actual actuator lower');self.upper=vector(finger_upper18,18,'actual actuator upper')
  self.release_open_rate=scalar(release_open_rate,'native open rate')
  if self.release_open_rate<=0 or np.any(self.lower>self.upper):raise ValueError('Actual nonempty native authority required')
  if abs(self.closed_pose[0][0]-self.open_pose[0][0])<1e-12:raise ValueError('Singular thumb alpha diagnostic')
 def native_coordinate(self,phi_s):
  phi=scalar(phi_s,'native progress')
  return ReleaseCoordinate(phi,self.entry.executed_lift_reference_m*(1.-np.clip(phi/.6,0.,1.)))
 def decode(self,coordinate:ReleaseCoordinate):
  if not isinstance(coordinate,ReleaseCoordinate):raise TypeError('Typed release coordinates required')
  e=coordinate.phi_s;alpha=self.entry.alpha5.copy();alpha[1:]=np.maximum(0.,alpha[1:]-self.release_open_rate*e)
  tucked=np.array([-1.03,1.21,.13,-.02,.985]);full_open=np.array([-1.,.1,0.,0.,0.])
  if e<.8:thumb=self.entry.thumb_reference5+np.clip(e/.8,0,1)*(tucked-self.entry.thumb_reference5)
  elif e<1.4:thumb=tucked
  else:thumb=tucked+np.clip((e-1.4)/.8,0,1)*(full_open-tucked)
  fingers=[np.clip(thumb,self.lower[:5],self.upper[:5])]
  for i in range(1,5):fingers.append(self.open_pose[i]+alpha[i]*(self.closed_pose[i]-self.open_pose[i]))
  alpha[0]=(thumb[0]-self.open_pose[0][0])/(self.closed_pose[0][0]-self.open_pose[0][0])
  command=np.r_[np.concatenate(fingers),coordinate.lift_reference_m,self.entry.wrist_reference2]
  if np.any(command[:18]<self.lower-1e-10) or np.any(command[:18]>self.upper+1e-10):raise ValueError('Native decoded finger outside actual actuator interval; no silent clip')
  return DecodedReference(coordinate,vector(command,21,'command21'),vector(alpha,5,'diagnostic alpha'))
 def path_knots_s(self):
  return tuple(sorted(set([0.,.8,1.4,2.2]+[float(a/self.release_open_rate) for a in self.entry.alpha5[1:] if a>0.])))

@dataclass(frozen=True)
class ProbeAuthority:
 command_lower21:np.ndarray
 command_upper21:np.ndarray
 robot_model_sha256:str
 declared_rate21:object=None
 declared_increment21:object=None
 declared_total_variation21:object=None
 declared_progress_rate_s_s:object=None
 declared_total_progress_s:object=None
 local_phi_interval_s:object=None
 source_note:str=''
 def __post_init__(self):
  for name in ['command_lower21','command_upper21']:object.__setattr__(self,name,vector(getattr(self,name),21,name))
  if np.any(self.command_lower21>self.command_upper21) or not sha64(self.robot_model_sha256):raise ValueError('Actual model authority required')
  for name in ['declared_rate21','declared_increment21','declared_total_variation21']:
   value=getattr(self,name)
   if value is not None:
    value=vector(value,21,name)
    if np.any(value<=0):raise ValueError('Positive reviewed experiment envelope required')
    object.__setattr__(self,name,value)
  for name in ['declared_progress_rate_s_s','declared_total_progress_s']:
   if getattr(self,name) is not None:
    value=scalar(getattr(self,name),name)
    if value<=0:raise ValueError('Positive reviewed progress envelope required')
    object.__setattr__(self,name,value)
  if self.local_phi_interval_s is not None:
   value=vector(self.local_phi_interval_s,2,'local progress interval')
   if value[0]<0 or value[0]>=value[1]:raise ValueError('Nonempty native path segment required')
   object.__setattr__(self,'local_phi_interval_s',value)

@dataclass(frozen=True)
class ProbeAssessment:
 status:str
 reasons:tuple
 mechanically_eligible:bool=False
 force_response_available:bool=False
 material_safety_certified:bool=False

def assess_increment(previous:DecodedReference,candidate:DecodedReference,*,anchor_command21,total_variation_so_far21,dt_s,authority:ProbeAuthority,expected_model_sha256,anchor_phi_s=None):
 dt=scalar(dt_s,'execution interval')
 if dt<=0:raise ValueError('Positive execution interval required')
 anchor=vector(anchor_command21,21,'clone anchor command');used=vector(total_variation_so_far21,21,'actual cumulative reference variation')
 if np.any(used<0):raise ValueError('Cumulative variation cannot be negative')
 reasons=[]
 if authority.robot_model_sha256!=expected_model_sha256:reasons.append('robot_model_identity_mismatch')
 if candidate.coordinate.phi_s<previous.coordinate.phi_s-1e-12:reasons.append('release_progress_reversal')
 if authority.declared_progress_rate_s_s is None:reasons.append('declared_progress_rate_missing')
 elif candidate.coordinate.phi_s-previous.coordinate.phi_s>authority.declared_progress_rate_s_s*dt+1e-12:reasons.append('release_progress_rate_exceeded')
 if authority.local_phi_interval_s is None:reasons.append('local_path_segment_missing')
 elif not authority.local_phi_interval_s[0]-1e-12<=candidate.coordinate.phi_s<=authority.local_phi_interval_s[1]+1e-12:reasons.append('release_path_segment_exceeded')
 if authority.declared_total_progress_s is None or anchor_phi_s is None:reasons.append('cumulative_progress_scope_missing')
 elif candidate.coordinate.phi_s-scalar(anchor_phi_s,'anchor progress')>authority.declared_total_progress_s+1e-12:reasons.append('cumulative_release_progress_exceeded')
 q=candidate.command21;delta=abs(q-previous.command21)
 if np.any(q<authority.command_lower21-1e-10) or np.any(q>authority.command_upper21+1e-10):reasons.append('actual_command_position_outside_authority')
 for key in ['declared_rate21','declared_increment21','declared_total_variation21']:
  if getattr(authority,key) is None:reasons.append(key+'_missing')
 if authority.declared_rate21 is not None and np.any(delta>authority.declared_rate21*dt+1e-12):reasons.append('actual_reference_rate_exceeded')
 if authority.declared_increment21 is not None and np.any(abs(q-anchor)>authority.declared_increment21+1e-12):reasons.append('actual_finite_increment_exceeded')
 if authority.declared_total_variation21 is not None and np.any(used+delta>authority.declared_total_variation21+1e-12):reasons.append('actual_cumulative_reference_variation_exceeded')
 return ProbeAssessment('MECHANICALLY_ELIGIBLE_UNCERTIFIED' if not reasons else 'UNAVAILABLE',tuple(reasons),not reasons)

@dataclass(frozen=True)
class Robot25Authority:
 axes25:tuple
 limited25:np.ndarray
 lower25:np.ndarray
 upper25:np.ndarray
 robot_model_sha256:str
 declared_rate25:object=None
 declared_total_variation25:object=None
 def __post_init__(self):
  axes=tuple(self.axes25);limited=np.asarray(self.limited25)
  if len(axes)!=25 or len(set(axes))!=25 or limited.shape!=(25,) or limited.dtype.kind!='b' or not sha64(self.robot_model_sha256):raise ValueError('Actual ordered robot25 authority required')
  object.__setattr__(self,'axes25',axes);object.__setattr__(self,'limited25',limited.copy())
  for name in ['lower25','upper25']:object.__setattr__(self,name,vector(getattr(self,name),25,name))
  if np.any(self.lower25[limited]>self.upper25[limited]):raise ValueError('Empty actual joint intervals')
  for name in ['declared_rate25','declared_total_variation25']:
   if getattr(self,name) is not None:
    value=vector(getattr(self,name),25,name)
    if np.any(value<=0):raise ValueError('Positive robot observation envelope required')
    object.__setattr__(self,name,value)

def assess_actual_robot25(position25,velocity25,used_variation25,*,axes25,authority:Robot25Authority):
 q=vector(position25,25,'actual robot position');used=vector(used_variation25,25,'actual robot cumulative travel');reasons=[]
 if np.any(used<0):raise ValueError('Actual robot cumulative travel cannot be negative')
 if tuple(axes25)!=authority.axes25:reasons.append('actual_robot_axis_order_mismatch')
 if np.any(q[authority.limited25]<authority.lower25[authority.limited25]-1e-10) or np.any(q[authority.limited25]>authority.upper25[authority.limited25]+1e-10):reasons.append('actual_single_joint_position_exceeded')
 if authority.declared_rate25 is None or velocity25 is None:reasons.append('actual_robot_rate_scope_or_observation_missing')
 elif np.any(abs(vector(velocity25,25,'actual robot velocity'))>authority.declared_rate25+1e-12):reasons.append('actual_single_joint_rate_exceeded')
 if authority.declared_total_variation25 is None:reasons.append('actual_robot_cumulative_scope_missing')
 elif np.any(used>authority.declared_total_variation25+1e-12):reasons.append('actual_single_joint_cumulative_travel_exceeded')
 return ProbeAssessment('OBSERVATION_WITHIN_DECLARED_ROBOT_ENVELOPE' if not reasons else 'UNAVAILABLE',tuple(reasons),not reasons)

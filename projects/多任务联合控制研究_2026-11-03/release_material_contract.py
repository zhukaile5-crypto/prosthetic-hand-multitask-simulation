"""Public release contracts and finite *uncertified* exit candidates.

No simulator, holding controller, object state or pressure-response model is
imported. Mechanical validity is deliberately distinct from material safety.
"""
from dataclasses import dataclass,fields,asdict
from math import isfinite,sqrt
from numbers import Real

FINGER_ACTUATORS=tuple('rh_A_'+x for x in (
    'THJ5','THJ4','THJ3','THJ2','THJ1','FFJ4','FFJ3','FFJ0',
    'MFJ4','MFJ3','MFJ0','RFJ4','RFJ3','RFJ0','LFJ5','LFJ4','LFJ3','LFJ0'))
COMMAND_AXES=FINGER_ACTUATORS+('lift_reference_m','WRJ2_reference_rad','WRJ1_reference_rad')
COMMAND_UNITS=('rad',)*18+('m','rad','rad')
PRESSURE_AXES=tuple(f'finger_{i}_normal_N' for i in range(5))+('auxiliary_pressure_N',)
RELEASE_MODES=('unload','clearance','post_clear_monitor','recontact')
ALL_SKIN_SCOPE='all_external_instrumented_skin'
ROBOT_JOINT_NAMES=frozenset('rh_'+x for x in (
    'WRJ2','WRJ1','THJ5','THJ4','THJ3','THJ2','THJ1',
    'FFJ4','FFJ3','FFJ2','FFJ1','MFJ4','MFJ3','MFJ2','MFJ1',
    'RFJ4','RFJ3','RFJ2','RFJ1','LFJ5','LFJ4','LFJ3','LFJ2','LFJ1'))

def number(value,name,nonnegative=False):
    if isinstance(value,bool) or not isinstance(value,Real):raise ValueError(f'{name}: real numeric value required, never strings or booleans')
    x=float(value)
    if not isfinite(x) or nonnegative and x<0.:raise ValueError(f'{name}: finite valid value required')
    return x

def vector(value,size,name,nonnegative=False,allow_none=False):
    if len(value)!=size:raise ValueError(f'{name}: expected {size} values')
    return tuple(None if x is None and allow_none else number(x,name,nonnegative) for x in value)

def strict_payload(cls,payload):
    if not isinstance(payload,dict) or set(payload)-{f.name for f in fields(cls)}:
        raise ValueError(f'{cls.__name__}: unknown/nonpublic fields')
    return cls(**payload)

def hash_string(value,name):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name}: lowercase SHA256 required')
    return value

@dataclass(frozen=True)
class MaterialContract:
    profile_id:str
    public_morphology:str
    per_finger_caps_N:tuple
    finger_budget_N:float
    release_authorized:bool
    release_deadline_s:float=3.5
    auxiliary_cap_N:float|None=None
    clearance_threshold_N:float=.2
    clearance_dwell_s:float=.2
    maximum_sensor_age_s:float=.12
    observation_scope:str=ALL_SKIN_SCOPE

    def __post_init__(self):
        if not self.profile_id or not self.public_morphology:raise ValueError('Declared profile/morphology required')
        if type(self.release_authorized) is not bool:raise ValueError('Explicit release authority required')
        if self.observation_scope!=ALL_SKIN_SCOPE:raise ValueError('No target-identified or contact-filtered scope permitted')
        object.__setattr__(self,'per_finger_caps_N',vector(self.per_finger_caps_N,5,'caps',True))
        for name in ['finger_budget_N','release_deadline_s','clearance_threshold_N','clearance_dwell_s','maximum_sensor_age_s']:
            object.__setattr__(self,name,number(getattr(self,name),name,True))
        if min(self.release_deadline_s,self.clearance_dwell_s,self.maximum_sensor_age_s)<=0.:raise ValueError('Positive temporal contract required')
        if self.auxiliary_cap_N is not None:object.__setattr__(self,'auxiliary_cap_N',number(self.auxiliary_cap_N,'auxiliary_cap_N',True))

    @classmethod
    def from_payload(cls,payload):return strict_payload(cls,payload)

    @property
    def protected_scope(self):
        return ALL_SKIN_SCOPE if self.auxiliary_cap_N is not None else 'all_external_instrumented_fingers_auxiliary_unprotected'

@dataclass(frozen=True)
class CommandSample:
    time_s:float
    command:tuple
    def __post_init__(self):
        object.__setattr__(self,'time_s',number(self.time_s,'history.time',True));object.__setattr__(self,'command',vector(self.command,21,'history.command'))

@dataclass(frozen=True)
class SkinPatch:
    """Public fields only. rh_ is syntax, not proof of robot-model membership.

    An acquisition adapter must resolve the link from the reviewed robot-only
    definition and verify membership; schema validation cannot certify origin.
    """
    robot_link:str
    link_position_m:tuple
    link_normal:tuple
    normal_N:float
    sample_time_s:float
    finger:int|None=None
    def __post_init__(self):
        if not isinstance(self.robot_link,str) or not self.robot_link.startswith('rh_'):raise ValueError('Known robot hand skin link required')
        object.__setattr__(self,'link_position_m',vector(self.link_position_m,3,'patch.position'))
        n=vector(self.link_normal,3,'patch.normal')
        if abs(sqrt(sum(x*x for x in n))-1.)>1e-6:raise ValueError('Unit measured normal required; no silent normalization')
        object.__setattr__(self,'link_normal',n);object.__setattr__(self,'normal_N',number(self.normal_N,'patch.pressure',True));object.__setattr__(self,'sample_time_s',number(self.sample_time_s,'patch.time',True))
        if self.finger is not None and (type(self.finger) is not int or not 0<=self.finger<5):raise ValueError('Finger channel or explicit auxiliary skin required')

@dataclass(frozen=True)
class ReleaseState:
    time_s:float
    joint_names:tuple
    joint_position_rad:tuple
    joint_velocity_rad_s:tuple
    actuator_coordinates:tuple
    actuator_velocities:tuple
    executed_command:tuple
    finger_normal_N:tuple
    finger_shear_N:tuple
    finger_valid:tuple
    finger_sample_time_s:tuple
    auxiliary_pressure_N:float|None
    auxiliary_wrench:tuple|None
    auxiliary_valid:bool
    auxiliary_sample_time_s:float|None
    robot_definition_sha256:str
    sensor_profile_sha256:str
    skin_patches:tuple=()
    executed_history:tuple=()
    observation_scope:str=ALL_SKIN_SCOPE

    def __post_init__(self):
        now=number(self.time_s,'state.time',True);object.__setattr__(self,'time_s',now)
        names=tuple(self.joint_names)
        if len(set(names))!=len(names) or set(names)!=ROBOT_JOINT_NAMES:raise ValueError('Exactly the24 declared robot joints required, never object generalized coordinates')
        object.__setattr__(self,'joint_names',names)
        for name in ['joint_position_rad','joint_velocity_rad_s']:object.__setattr__(self,name,vector(getattr(self,name),len(names),name))
        for name in ['actuator_coordinates','actuator_velocities','executed_command']:object.__setattr__(self,name,vector(getattr(self,name),21,name))
        if len(self.finger_valid)!=5 or any(type(x) is not bool for x in self.finger_valid):raise ValueError('Five explicit sensor-valid flags required')
        object.__setattr__(self,'finger_valid',tuple(self.finger_valid))
        for name in ['finger_normal_N','finger_shear_N','finger_sample_time_s']:object.__setattr__(self,name,vector(getattr(self,name),5,name,True,True))
        for i,valid in enumerate(self.finger_valid):
            if valid and any(getattr(self,k)[i] is None for k in ['finger_normal_N','finger_shear_N','finger_sample_time_s']):raise ValueError('Valid finger cannot have missing measurement')
        if type(self.auxiliary_valid) is not bool:raise ValueError('Explicit auxiliary validity required')
        for name in ['auxiliary_pressure_N','auxiliary_sample_time_s']:
            if getattr(self,name) is not None:object.__setattr__(self,name,number(getattr(self,name),name,True))
        if self.auxiliary_wrench is not None:object.__setattr__(self,'auxiliary_wrench',vector(self.auxiliary_wrench,6,'auxiliary_wrench_N_then_Nm'))
        if self.auxiliary_valid and any(getattr(self,k) is None for k in ['auxiliary_pressure_N','auxiliary_wrench','auxiliary_sample_time_s']):raise ValueError('Valid auxiliary channel cannot be missing')
        times=[x for x in self.finger_sample_time_s if x is not None]+([self.auxiliary_sample_time_s] if self.auxiliary_sample_time_s is not None else [])
        if any(t>now+1e-9 for t in times):raise ValueError('Future observations forbidden')
        patches=tuple(strict_payload(SkinPatch,p) if isinstance(p,dict) else p for p in self.skin_patches)
        history=tuple(strict_payload(CommandSample,p) if isinstance(p,dict) else p for p in self.executed_history)
        if any(not isinstance(p,SkinPatch) or p.sample_time_s>now+1e-9 for p in patches):raise ValueError('Public past skin patches required')
        if any(not isinstance(h,CommandSample) or h.time_s>now+1e-9 for h in history) or any(a.time_s>=b.time_s for a,b in zip(history,history[1:])):raise ValueError('Strictly ordered actual past commands required')
        object.__setattr__(self,'skin_patches',patches);object.__setattr__(self,'executed_history',history)
        hash_string(self.robot_definition_sha256,'robot_definition_sha256');hash_string(self.sensor_profile_sha256,'sensor_profile_sha256')
        if self.observation_scope!=ALL_SKIN_SCOPE:raise ValueError('All external skin scope required')

    @classmethod
    def from_payload(cls,payload):return strict_payload(cls,payload)

    def pressure_available(self,contract):
        times=self.finger_sample_time_s+(self.auxiliary_sample_time_s,)
        return all(self.finger_valid) and self.auxiliary_valid and all(t is not None and self.time_s-t<=contract.maximum_sensor_age_s+1e-9 for t in times)

    def pressure_vector(self,contract):
        return self.finger_normal_N+(self.auxiliary_pressure_N,) if self.pressure_available(contract) else None

@dataclass(frozen=True)
class ActuatorAuthority:
    lower:tuple
    upper:tuple
    maximum_rate:tuple
    open_finger_reference:tuple
    robot_definition_sha256:str
    control_period_s:float=.005
    axes:tuple=COMMAND_AXES
    units:tuple=COMMAND_UNITS
    def __post_init__(self):
        if tuple(self.axes)!=COMMAND_AXES or tuple(self.units)!=COMMAND_UNITS:raise ValueError('Actual canonical18 finger+lift+wrists order and rad/m units required')
        object.__setattr__(self,'axes',tuple(self.axes));object.__setattr__(self,'units',tuple(self.units))
        for name in ['lower','upper','maximum_rate']:object.__setattr__(self,name,vector(getattr(self,name),21,name,name=='maximum_rate'))
        object.__setattr__(self,'open_finger_reference',vector(self.open_finger_reference,18,'public_robot_open_reference'))
        object.__setattr__(self,'control_period_s',number(self.control_period_s,'control_period_s',True))
        if self.control_period_s<=0. or any(a>b for a,b in zip(self.lower,self.upper)):raise ValueError('Nonempty actual authority required')
        if any(not a<=x<=b for a,x,b in zip(self.lower,self.open_finger_reference,self.upper)):raise ValueError('Open reference outside robot authority')
        hash_string(self.robot_definition_sha256,'robot_definition_sha256')

    @classmethod
    def from_payload(cls,payload):return strict_payload(cls,payload)

@dataclass(frozen=True)
class LifecycleStatus:
    release_active:bool
    holding_support_allowed:bool
    mode:str
    opening_committed:bool
    first_clear_confirmation_s:float|None
    recontact_count:int
    observations_fresh:bool
    observed_material_violation:bool|None
    violation_latched:bool
    deadline_exceeded:bool

class ReleaseLifecycle:
    """Intent/monitor only. Never waits for a model before cancelling holding."""
    def __init__(self,contract):
        self.contract=contract;self.requested_at=None;self.last_time=None;self.last_acquired=None;self.clear_since=None;self.first_clear=None;self.opening_committed=False;self.recontacts=0;self.contact_episode_seen=False;self.violation_latched=False;self.mode='unload'

    def request(self,time_s):
        if not self.contract.release_authorized:raise ValueError('No public release authority')
        now=number(time_s,'request.time',True)
        if self.last_time is not None and now<self.last_time-1e-9:raise ValueError('Release request clock cannot rewind')
        if self.requested_at is None:self.requested_at=now
        return self._status(now,self.mode,False,None)

    def _status(self,now,mode,fresh,violation):
        active=self.requested_at is not None
        self.mode=mode
        return LifecycleStatus(active,not active,mode,self.opening_committed,self.first_clear,self.recontacts,fresh,violation,self.violation_latched,bool(active and now-self.requested_at>self.contract.release_deadline_s))

    def observe(self,state):
        if self.requested_at is None:raise ValueError('Release intent must be requested first')
        if state.time_s<self.requested_at-1e-9:raise ValueError('State predates release request')
        if self.last_time is not None and state.time_s<self.last_time-1e-9:raise ValueError('Lifecycle clock cannot rewind')
        self.last_time=state.time_s;fresh=state.pressure_available(self.contract);pressure=state.pressure_vector(self.contract);violation=None
        # Partial valid channels may still establish a violation; missing ones
        # never establish compliance or clearance.
        for i,valid in enumerate(state.finger_valid):
            if valid and state.finger_normal_N[i]>self.contract.per_finger_caps_N[i]:self.violation_latched=True;violation=True
        if state.auxiliary_valid and self.contract.auxiliary_cap_N is not None and state.auxiliary_pressure_N>self.contract.auxiliary_cap_N:self.violation_latched=True;violation=True
        if fresh:
            violation=any(x>c for x,c in zip(pressure[:5],self.contract.per_finger_caps_N)) or sum(pressure[:5])>self.contract.finger_budget_N or self.contract.auxiliary_cap_N is not None and pressure[5]>self.contract.auxiliary_cap_N
            self.violation_latched|=violation
        if not fresh:self.clear_since=None;return self._status(state.time_s,'post_clear_monitor' if self.opening_committed else 'unload',False,violation)
        acquired=min(state.finger_sample_time_s+(state.auxiliary_sample_time_s,));clear=sum(pressure)<=self.contract.clearance_threshold_N
        if self.last_acquired is not None and acquired<self.last_acquired-1e-9:
            self.clear_since=None;return self._status(state.time_s,'post_clear_monitor' if self.opening_committed else 'unload',False,violation)
        self.last_acquired=acquired
        if clear:
            if self.clear_since is None:self.clear_since=acquired
            if acquired-self.clear_since>=self.contract.clearance_dwell_s-1e-9:
                if self.first_clear is None:self.first_clear=state.time_s
                self.opening_committed=True;self.contact_episode_seen=False
            mode='post_clear_monitor' if self.opening_committed else 'clearance'
        else:
            self.clear_since=None
            if self.opening_committed and not self.contact_episode_seen:self.recontacts+=1;self.contact_episode_seen=True
            mode='recontact' if self.opening_committed else 'unload'
        return self._status(state.time_s,mode,True,violation)

@dataclass(frozen=True)
class ExitCandidate:
    target:tuple
    mode:str
    interval_s:float
    provenance:str='public_robot_open_reference_finite_uncertified_exit'
    uses_holding_support:bool=False
    restarts_grasp:bool=False
    def __post_init__(self):
        object.__setattr__(self,'target',vector(self.target,21,'candidate.target'))
        object.__setattr__(self,'interval_s',number(self.interval_s,'candidate.interval',True))
        if self.mode not in RELEASE_MODES or self.interval_s<=0.:raise ValueError('Release candidate mode/interval required')
        if type(self.uses_holding_support) is not bool or type(self.restarts_grasp) is not bool:raise ValueError('Explicit intent flags required')

@dataclass(frozen=True)
class CandidateCheck:
    mechanically_valid:bool
    reasons:tuple
    material_certified:bool=False
    force_prediction_available:bool=False
    scope:str='finite_uncertified_exit_not_material_or_collision_safety'

def check_uncertified_exit(candidate,state,authority,lifecycle):
    reasons=[];before=state.executed_command;dt=candidate.interval_s
    if not lifecycle.release_active or lifecycle.holding_support_allowed:reasons.append('release_intent_not_active')
    if candidate.uses_holding_support or candidate.restarts_grasp:reasons.append('holding_or_regrasp_forbidden')
    if state.robot_definition_sha256!=authority.robot_definition_sha256:reasons.append('robot_authority_identity_mismatch')
    if abs(dt-authority.control_period_s)>1e-12:reasons.append('wrong_actual_command_period')
    for i,(old,new,lo,hi,rate) in enumerate(zip(before,candidate.target,authority.lower,authority.upper,authority.maximum_rate)):
        if old<lo-1e-9 or old>hi+1e-9 or new<lo-1e-9 or new>hi+1e-9:reasons.append(f'axis_{i}_outside_authority')
        if abs(new-old)>rate*dt+1e-9:reasons.append(f'axis_{i}_rate_violation')
        if i<18:
            end=authority.open_finger_reference[i]
            if not min(old,end)-1e-9<=new<=max(old,end)+1e-9:reasons.append(f'axis_{i}_not_toward_public_open_reference')
        elif abs(new-old)>1e-9:reasons.append('uncertified_motion_return_forbidden')
    return CandidateCheck(not reasons,tuple(reasons))

def bounded_uncertified_exit(state,authority,lifecycle):
    """A candidate only. Opening can still increase force at unknown contacts."""
    if not lifecycle.release_active:raise ValueError('Cannot propose exit without release intent')
    target=[];dt=authority.control_period_s
    for i,old in enumerate(state.executed_command):
        goal=authority.open_finger_reference[i] if i<18 else old
        delta=max(-authority.maximum_rate[i]*dt,min(authority.maximum_rate[i]*dt,goal-old));target.append(old+delta)
    candidate=ExitCandidate(tuple(target),lifecycle.mode,dt)
    return candidate,check_uncertified_exit(candidate,state,authority,lifecycle)

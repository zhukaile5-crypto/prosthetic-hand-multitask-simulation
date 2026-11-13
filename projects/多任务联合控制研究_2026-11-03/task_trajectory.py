"""Explicit public user-motion trajectories; never an environment forecast.

V6 uses known user waypoints to plan an interval ramp. Archived V5 executions
remain unchanged. Future forces/material parameters are not accepted.
"""
from dataclasses import asdict,dataclass
import math
from numbers import Real


AXES=('lift_m','WRJ2_rad','WRJ1_rad')


def scalar(x,name):
    if isinstance(x,bool) or not isinstance(x,Real):raise ValueError(name+' must be a real number, not a boolean')
    value=float(x)
    if not math.isfinite(value):raise ValueError(name+' must be finite')
    return value


def vector(x,name):
    if isinstance(x,(str,bytes,dict)):raise ValueError(name+' must have three entries')
    values=tuple(scalar(y,name) for y in x)
    if len(values)!=3:raise ValueError(name+' must have three entries')
    return values


@dataclass(frozen=True)
class PublicTaskTrajectory:
    start_s: float
    end_s: float
    origin: tuple
    sine_amplitude: tuple=(0.,0.,0.)
    cosine_amplitude: tuple=(0.,0.,0.)
    angular_frequency: tuple=(0.,0.,0.)
    phase_rad: tuple=(0.,0.,0.)

    def __post_init__(self):
        start=scalar(self.start_s,'start_s');end=scalar(self.end_s,'end_s')
        if start<0. or end<=start:raise ValueError('A positive finite stage interval is required')
        object.__setattr__(self,'start_s',start);object.__setattr__(self,'end_s',end)
        for name in ('origin','sine_amplitude','cosine_amplitude','angular_frequency','phase_rad'):
            object.__setattr__(self,name,vector(getattr(self,name),name))

    def position_at(self,time_s):
        t=scalar(time_s,'time_s')
        if t<0.:raise ValueError('time_s must be nonnegative')
        elapsed=min(max(0.,t-self.start_s),self.end_s-self.start_s)
        return [origin+a*(math.sin(w*elapsed+phase)-math.sin(phase))
            +b*(math.cos(w*elapsed+phase)-math.cos(phase))
            for origin,a,b,w,phase in zip(self.origin,self.sine_amplitude,
                self.cosine_amplitude,self.angular_frequency,self.phase_rad)]

    def preview(self,time_s,horizon=3,period_s=.02):
        now=scalar(time_s,'time_s');period=scalar(period_s,'period_s')
        if now<0. or period<=0. or type(horizon) is not int or horizon<1:
            raise ValueError('Positive preview period/horizon and nonnegative time required')
        times=[now+i*period for i in range(horizon+1)]
        return dict(schema='public-user-motion-preview-v1',axes=list(AXES),time_s=now,
            requested_times_s=times,trajectory_times_s=[min(t,self.end_s) for t in times],
            motion_targets=[self.position_at(t) for t in times],
            future_reference_available=now<self.end_s,available_through_s=self.end_s,
            active_user_motion=[t<self.end_s for t in times],
            terminal_semantics='continuous_pre_release_endpoint; no anticipation of a release-time reference reset',
            information='explicit user motion only; no future disturbance or object state')

    def as_dict(self):
        data=asdict(self)
        for key in ('origin','sine_amplitude','cosine_amplitude','angular_frequency','phase_rad'):data[key]=list(data[key])
        return dict(schema='public-user-trajectory-v1',axes=list(AXES),**data)

    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict):raise ValueError('Public trajectory must be a dictionary')
        fields=set(cls.__dataclass_fields__)
        if set(value)!=fields|{'schema','axes'} or value['schema']!='public-user-trajectory-v1' or value['axes']!=list(AXES):
            raise ValueError('Unknown/nonpublic trajectory schema or fields')
        return cls(**{k:value[k] for k in fields})


def constant_trajectory(lift_reference=.1,wrist_reference=(0.,0.),end_s=12.):
    wrists=tuple(wrist_reference)
    if len(wrists)!=2:raise ValueError('Two public wrist references required')
    return PublicTaskTrajectory(0.,end_s,(lift_reference,*wrists))


def public_transport_trajectory(lift_origin=.1,start_s=9.3,end_s=12.):
    """The study's published user task, independent of its disturbance cases."""
    return PublicTaskTrajectory(start_s,end_s,(lift_origin,0.,0.),
        sine_amplitude=(0.,.04,.025),cosine_amplitude=(.008,0.,0.),
        angular_frequency=(3.,3.,2.))


def unknown_future_point(time_s,motion_target):
    """A point request does not become known future motion by replication."""
    now=scalar(time_s,'time_s')
    if now<0.:raise ValueError('time_s must be nonnegative')
    return dict(schema='public-user-motion-preview-v1',axes=list(AXES),time_s=now,
        requested_times_s=[now],trajectory_times_s=[now],motion_targets=[list(vector(motion_target,'motion_target'))],
        future_reference_available=False,available_through_s=now,active_user_motion=[True],
        terminal_semantics='future user reference unknown',information='current explicit point request only')


def validate_preview(value,time_s=None,motion_target=None):
    """Validate only explicit user intent; reject extra nested truth fields.

    A preview is a declared request, not an assertion about future actual robot
    position. Samples beyond the end retain its continuous endpoint.
    """
    keys={'schema','axes','time_s','requested_times_s','trajectory_times_s','motion_targets',
        'future_reference_available','available_through_s','active_user_motion','terminal_semantics','information'}
    if not isinstance(value,dict) or set(value)!=keys or value['schema']!='public-user-motion-preview-v1' or value['axes']!=list(AXES):
        raise ValueError('Unknown/nonpublic motion preview fields')
    now=scalar(value['time_s'],'preview.time_s');end=scalar(value['available_through_s'],'available_through_s')
    requested=[scalar(t,'requested_times_s') for t in value['requested_times_s']]
    clipped=[scalar(t,'trajectory_times_s') for t in value['trajectory_times_s']]
    targets=[list(vector(v,'motion_targets')) for v in value['motion_targets']]
    active=value['active_user_motion'];future=value['future_reference_available']
    if now<0. or end<0. or type(future) is not bool or not requested or future and end<=now:
        raise ValueError('Invalid preview availability')
    if len(requested)!=len(clipped) or len(targets)!=len(requested) or len(active)!=len(requested):
        raise ValueError('Preview arrays must have equal lengths')
    if abs(requested[0]-now)>1e-9 or any(b<=a for a,b in zip(requested,requested[1:])):
        raise ValueError('Preview must start now and increase')
    if any(type(v) is not bool for v in active) or any(abs(c-min(r,end))>1e-9 for r,c in zip(requested,clipped)):
        raise ValueError('Invalid preview stage clock')
    if future:
        if len(requested)<2 or any(a!=(r<end) for r,a in zip(requested,active)):
            raise ValueError('Invalid known trajectory stage flags')
        tail=[p for r,p in zip(requested,targets) if r>=end]
        if tail and any(p!=tail[0] for p in tail):
            raise ValueError('Completed preview must retain its continuous endpoint')
    elif any(active):
        if len(requested)!=1 or active!=[True] or abs(end-now)>1e-9:
            raise ValueError('Unknown future may contain only the current point')
    elif end>now or any(p!=targets[0] for p in targets):
        raise ValueError('Completed stage cannot fabricate future motion')
    if time_s is not None and abs(now-scalar(time_s,'time_s'))>1e-9:
        raise ValueError('Stale public preview')
    if motion_target is not None and any(abs(a-b)>1e-10 for a,b in zip(targets[0],vector(motion_target,'motion_target'))):
        raise ValueError('Preview must retain current explicit target')
    if not isinstance(value['terminal_semantics'],str) or not isinstance(value['information'],str):
        raise ValueError('Preview descriptions must be text')
    return dict(schema=value['schema'],axes=list(AXES),time_s=now,requested_times_s=requested,
        trajectory_times_s=clipped,motion_targets=targets,future_reference_available=future,
        available_through_s=end,active_user_motion=list(active),terminal_semantics=value['terminal_semantics'],information=value['information'])

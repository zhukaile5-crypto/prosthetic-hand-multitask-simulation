"""Unconnected virtual receiver TOP-FACE resultant sensor.

Reads an explicitly mounted static box's cached contacts, never dynamics/FK.
It is additional experimental sensing, not object/support truth or a safety
certificate. The sole public sample is a strict, delayed noisy measurement.
"""
from collections import deque
from dataclasses import dataclass
from numbers import Integral,Real
import numpy as np

PERIOD_S=.005
DELAY_S=.010
WRENCH_LENGTH_M=.050
RECEIVER_STREAM_ID=0x56313552
PHASES=('boundary_after_forward','cached_after_step_end_label')
ROLE='virtual_receiver_top_face_external_contact_wrench'
UNCERTAINTY_KIND='three_sigma_engineering_proxy_not_a_bound'
CLOCK_TOLERANCE_S=1e-7
INVALID_REASONS=('delay_buffer_warming','mount_definition_changed','acquisition_unavailable','receiver_engine_clock_mismatch',
    'receiver_contact_input_unavailable','random_dropout','scheduled_dropout')

def _real(x,name,minimum=None):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not np.isfinite(x):raise ValueError(name+' must be finite real, not bool/string')
    x=float(x)
    if minimum is not None and x<minimum:raise ValueError(name+' below minimum')
    return x
def _integer(x,name,minimum=0):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral) or x<minimum:raise ValueError(name+' must be an integer')
    return int(x)
def _flag(x,name):
    if not isinstance(x,(bool,np.bool_)):raise ValueError(name+' must be bool')
    return bool(x)
def _array(x,shape,name,nonnegative=False):
    raw=np.asarray(x,dtype=object)
    if raw.shape!=shape or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in raw.flat):raise ValueError(name+' requires real numeric '+str(shape))
    a=np.asarray(x,float)
    if not np.isfinite(a).all() or nonnegative and np.any(a<0):raise ValueError(name+' must be finite'+('/nonnegative' if nonnegative else ''))
    # Immutable bytes backing also forbids setflags(write=True) on public data.
    return np.frombuffer(a.tobytes(),dtype=np.float64).reshape(shape)
def _rotation(q):
    w,x,y,z=_array(q,(4,),'mounted geom quaternion')
    if abs(w*w+x*x+y*y+z*z-1.)>1e-7:raise ValueError('Mount quaternion must already be normalized')
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

@dataclass(frozen=True,slots=True)
class ReceiverMount:
    geom_name:str
    excluded_geom_names:tuple
    origin_in_geom_m:np.ndarray
    top_plane_tolerance_m:float
    normal_alignment_min_cos:float
    coverage:str='explicit_static_box_top_face_only'
    def __post_init__(self):
        if not isinstance(self.geom_name,str) or not self.geom_name:raise ValueError('Explicit named receiver geom required')
        if not isinstance(self.excluded_geom_names,tuple) or not self.excluded_geom_names or any(not isinstance(n,str) or not n for n in self.excluded_geom_names):raise ValueError('Explicit excluded floor/post geom names required')
        if len(set(self.excluded_geom_names))!=len(self.excluded_geom_names) or self.geom_name in self.excluded_geom_names:raise ValueError('Receiver/excluded geometry overlap or duplicates')
        if self.coverage!='explicit_static_box_top_face_only':raise ValueError('This prototype measures top face only, not whole-table loadcell')
        object.__setattr__(self,'origin_in_geom_m',_array(self.origin_in_geom_m,(3,),'top-frame origin'))
        tol=_real(self.top_plane_tolerance_m,'top plane tolerance',0.)
        cosine=_real(self.normal_alignment_min_cos,'normal alignment')
        if tol<=0 or not 0.<cosine<=1.:raise ValueError('Positive explicit geometry tolerance and cosine in(0,1] required')
        object.__setattr__(self,'top_plane_tolerance_m',tol);object.__setattr__(self,'normal_alignment_min_cos',cosine)

@dataclass(frozen=True,slots=True)
class ReceiverNoise:
    force_sigma_N:float
    moment_sigma_Nm:float
    dropout_probability:float=0.
    dropout_windows:tuple=()
    delay_s:float=DELAY_S
    period_s:float=PERIOD_S
    cop_min_load_N:float|None=None
    def __post_init__(self):
        for name in ('force_sigma_N','moment_sigma_Nm'):
            value=_real(getattr(self,name),name,0.)
            if value>np.finfo(float).max/3.:raise ValueError('Finite3sigma engineering proxy required')
            object.__setattr__(self,name,value)
        p=_real(self.dropout_probability,'dropout probability',0.)
        if p>1.:raise ValueError('Dropout probability exceeds1')
        object.__setattr__(self,'dropout_probability',p)
        if _real(self.delay_s,'delay')!=DELAY_S or _real(self.period_s,'period')!=PERIOD_S:raise ValueError('This registered device is200Hz with10ms delay')
        if not isinstance(self.dropout_windows,tuple):raise ValueError('Dropout windows must be an immutable tuple')
        windows=[]
        for window in self.dropout_windows:
            if not isinstance(window,tuple) or len(window)!=2:raise ValueError('Dropout window must be(start,end)')
            start=_real(window[0],'dropout start',0.);end=_real(window[1],'dropout end',0.)
            if end<=start:raise ValueError('Dropout window must increase')
            windows.append((start,end))
        object.__setattr__(self,'dropout_windows',tuple(windows))
        if self.cop_min_load_N is not None:
            x=_real(self.cop_min_load_N,'CoP minimum load',0.)
            if x<=0:raise ValueError('CoP needs an explicit positive minimum load')
            object.__setattr__(self,'cop_min_load_N',x)
    @classmethod
    def from_auxiliary(cls,*,force_sigma_N,auxiliary_force_sigma_N=None,auxiliary_moment_sigma_Nm=None,**settings):
        fallback=_real(force_sigma_N,'fallback force sigma',0.)
        sf=fallback if auxiliary_force_sigma_N is None else _real(auxiliary_force_sigma_N,'auxiliary force sigma',0.)
        sm=sf*WRENCH_LENGTH_M if auxiliary_moment_sigma_Nm is None else _real(auxiliary_moment_sigma_Nm,'auxiliary moment sigma',0.)
        return cls(sf,sm,**settings)
    @property
    def sigma6(self):return _array([self.force_sigma_N]*3+[self.moment_sigma_Nm]*3,(6,),'device sigma6',True)

@dataclass(frozen=True,slots=True)
class ReceiverMeasurement:
    time_s:float
    acquisition_time_s:float|None
    age_s:float|None
    valid:bool
    wrench6:np.ndarray|None
    normal_load_N:float|None
    sigma6:np.ndarray
    uncertainty6:np.ndarray
    cop_xy_m:np.ndarray|None
    sampling_phase:str|None
    invalid_reason:str|None
    role:str=ROLE
    uncertainty_kind:str=UNCERTAINTY_KIND
    def __post_init__(self):
        now=_real(self.time_s,'measurement time',0.);valid=_flag(self.valid,'valid')
        if abs(now-round(now/PERIOD_S)*PERIOD_S)>1e-9:raise ValueError('Measurement delivery clock must be on registered200Hz grid')
        object.__setattr__(self,'time_s',now);object.__setattr__(self,'valid',valid)
        if self.role!=ROLE or self.uncertainty_kind!=UNCERTAINTY_KIND:raise ValueError('Unknown receiver measurement role')
        sigma=_array(self.sigma6,(6,),'sigma6',True);unc=_array(self.uncertainty6,(6,),'uncertainty6',True)
        if not np.array_equal(unc,3.*sigma):raise ValueError('Uncertainty is exactly the declared3sigma engineering proxy')
        object.__setattr__(self,'sigma6',sigma);object.__setattr__(self,'uncertainty6',unc)
        if self.acquisition_time_s is None:
            if valid or self.age_s is not None or self.sampling_phase is not None:raise ValueError('No acquisition means no age/phase or valid value')
        else:
            acq=_real(self.acquisition_time_s,'acquisition time',0.);age=_real(self.age_s,'measurement age',0.)
            if acq>now+1e-10 or abs(age-(now-acq))>1e-9 or abs(age-DELAY_S)>1e-9 or abs(acq-round(acq/PERIOD_S)*PERIOD_S)>1e-9 or self.sampling_phase not in PHASES:raise ValueError('Invalid receiver acquisition clock/phase')
            object.__setattr__(self,'acquisition_time_s',acq);object.__setattr__(self,'age_s',age)
        if valid:
            w=_array(self.wrench6,(6,),'measured wrench6');load=_real(self.normal_load_N,'measured normal load',0.)
            if load!=max(0.,-float(w[2])) or self.invalid_reason is not None:raise ValueError('Normal load must be the same measured wrench compression')
            object.__setattr__(self,'wrench6',w);object.__setattr__(self,'normal_load_N',load)
            if self.cop_xy_m is not None:
                cop=_array(self.cop_xy_m,(2,),'equivalent CoP')
                if load<=0 or not np.array_equal(cop,[-w[4]/w[2],w[3]/w[2]]):raise ValueError('CoP must derive from the same nonzero measured wrench')
                object.__setattr__(self,'cop_xy_m',cop)
        elif any(x is not None for x in (self.wrench6,self.normal_load_N,self.cop_xy_m)) or self.invalid_reason not in INVALID_REASONS:raise ValueError('Unknown measurement must not carry invented zero/value/CoP or arbitrary payload')
    def to_public(self):
        return dict(schema='receiver-top-face-measurement-v1',time_s=self.time_s,acquisition_time_s=self.acquisition_time_s,age_s=self.age_s,valid=self.valid,
            wrench6=None if self.wrench6 is None else self.wrench6.tolist(),normal_load_N=self.normal_load_N,
            sigma6=self.sigma6.tolist(),uncertainty6=self.uncertainty6.tolist(),uncertainty_kind=self.uncertainty_kind,
            cop_xy_m=None if self.cop_xy_m is None else self.cop_xy_m.tolist(),cop_valid=self.cop_xy_m is not None,
            sampling_phase=self.sampling_phase,invalid_reason=self.invalid_reason,role=self.role,
            frame='declared_receiver_top_frame',wrench_order='Fx,Fy,Fz,Mx,My,Mz;N,N,N,Nm,Nm,Nm',
            normal_load_role='max(0,-measured_Fz);not_independent_channel_or_target_support_truth',
            cop_role='net_wrench_equivalent_plane_CoP_not_object_pose_or_identity')

@dataclass(frozen=True,slots=True)
class _Packet:
    time_s:float
    valid:bool
    wrench6:np.ndarray|None
    phase:str
    reason:str|None

class ReceiverSensor:
    """One independent stream. No external RNG, plant or controller accepted."""
    def __init__(self,model,mount:ReceiverMount,noise:ReceiverNoise,seed,*,acquisition_phase,contact_force_reader=None):
        if type(mount) is not ReceiverMount or type(noise) is not ReceiverNoise:raise TypeError('Strict receiver mount/noise required')
        if acquisition_phase not in PHASES:raise ValueError('Explicit cached-solve or fresh-boundary phase required')
        self._model=model;self._mount=mount;self._noise=noise;self._phase=acquisition_phase;self._force_reader=contact_force_reader
        self._seed=_integer(seed,'independent receiver seed');self._rng=np.random.default_rng(np.random.SeedSequence([self._seed,RECEIVER_STREAM_ID]))
        self._gid=_integer(model.geom(mount.geom_name).id,'receiver geom id');self._excluded=tuple(_integer(model.geom(n).id,'excluded geom id') for n in mount.excluded_geom_names)
        if self._gid in self._excluded:raise ValueError('Receiver geometry is excluded')
        # MuJoCo mjGEOM_BOX=6. This first prototype explicitly rejects planes,
        # floor, moving/mocap bodies, and an undeclared whole-table loadcell.
        if int(model.geom_type[self._gid])!=6 or int(model.geom_bodyid[self._gid])!=0:raise ValueError('Only a declared static world-mounted box top is supported')
        self._size=_array(model.geom_size[self._gid],(3,),'receiver half sizes')
        if np.any(self._size<=0) or not np.allclose(mount.origin_in_geom_m,[0.,0.,self._size[2]],atol=1e-10,rtol=0):raise ValueError('Explicit origin must be the registered box top center')
        self._R=_rotation(model.geom_quat[self._gid]);self._center=_array(model.geom_pos[self._gid],(3,),'known receiver mount position')
        self._origin=self._center+self._R@mount.origin_in_geom_m
        self._binding=self._signature();self._binding_invalid=False;self._queue=deque(maxlen=3);self._last_tick=None;self._last=None
        self._acquisitions=0;self._gaussian_draws=0;self._uniform_draws=0
    def _signature(self):
        ids=[self._gid,*self._excluded]
        return tuple((name,np.asarray(getattr(self._model,name))[ids].tobytes()) for name in ('geom_type','geom_bodyid','geom_size','geom_pos','geom_quat'))
    def _collect(self,data,time_s):
        if self._signature()!=self._binding:return None,False,'mount_definition_changed'
        try:
            if abs(_real(data.time,'current engine clock',0.)-time_s)>CLOCK_TOLERANCE_S:return None,False,'receiver_engine_clock_mismatch'
            ncon=_integer(data.ncon,'contact count');total=np.zeros(6)
            for index in range(ncon):
                c=data.contact[index];pair=np.asarray(c.geom,dtype=object)
                if pair.shape!=(2,):raise ValueError('Contact pair shape')
                g0=_integer(pair[0],'contact geom0',-1);g1=_integer(pair[1],'contact geom1',-1)
                if (g0==self._gid)==(g1==self._gid):continue
                Q=_array(np.asarray(c.frame).reshape(3,3),(3,3),'contact frame')
                if not np.allclose(Q@Q.T,np.eye(3),atol=1e-6,rtol=0) or np.linalg.det(Q)<0:raise ValueError('Unknown contact frame')
                point=_array(c.pos,(3,),'cached contact point');distance=_real(c.dist,'cached contact distance')
                sign=-1. if g0==self._gid else 1.
                outward=-sign*Q[0];surface=point+sign*.5*distance*Q[0]
                local=self._R.T@(surface-self._center);normal_local=self._R.T@outward;tol=self._mount.top_plane_tolerance_m
                if abs(local[2]-self._size[2])>tol or np.any(abs(local[:2])>self._size[:2]+tol) or normal_local[2]<self._mount.normal_alignment_min_cos:continue
                raw=np.full(6,np.nan)
                if self._force_reader is None:
                    import mujoco
                    mujoco.mj_contactForce(self._model,data,index,raw)
                else:self._force_reader(self._model,data,index,raw)
                raw=_array(raw,(6,),'cached contact wrench')
                force=sign*(Q.T@raw[:3]);moment=sign*(Q.T@raw[3:])+np.cross(point-self._origin,force)
                total+=np.r_[self._R.T@force,self._R.T@moment]
            return _array(total,(6,),'receiver resultant'),True,None
        except (ValueError,TypeError,IndexError,AttributeError,RuntimeError):return None,False,'receiver_contact_input_unavailable'
    def sample(self,data,*,time_s,sampling_phase,available=True):
        t=_real(time_s,'public receiver clock',0.);available=_flag(available,'acquisition availability')
        if sampling_phase!=self._phase:raise ValueError('Instrument sampling phase cannot change within an episode')
        tick=int(round(t/PERIOD_S))
        if abs(t-tick*PERIOD_S)>1e-9:raise ValueError('Receiver sampling must use the200Hz public grid')
        if self._last_tick is not None:
            if tick==self._last_tick:return self._last
            if tick!=self._last_tick+1:raise ValueError('Skipped/rewound receiver tick; mark missing acquisitions explicitly, never invent historical force')
        now=tick*PERIOD_S
        try:self._binding_invalid=self._binding_invalid or self._signature()!=self._binding
        except (ValueError,TypeError,IndexError,AttributeError):self._binding_invalid=True
        if self._binding_invalid:raw,valid,reason=None,False,'mount_definition_changed'
        else:raw,valid,reason=self._collect(data,now) if available else (None,False,'acquisition_unavailable')
        # Exactly6 Gaussian and1 uniform per acquisition, regardless of contact,
        # sigma, missing input or dropout. The original tactile RNG is untouched.
        sigma=self._noise.sigma6;error=self._rng.standard_normal(6)*sigma;uniform=float(self._rng.random())
        self._acquisitions+=1;self._gaussian_draws+=6;self._uniform_draws+=1
        if uniform<self._noise.dropout_probability:valid=False;reason='random_dropout'
        if any(start<=now<end for start,end in self._noise.dropout_windows):valid=False;reason='scheduled_dropout'
        packet=_Packet(now,bool(valid),_array(raw+error,(6,),'noisy receiver wrench') if valid else None,self._phase,None if valid else reason)
        self._queue.append(packet)
        desired=now-DELAY_S;delayed=next((p for p in self._queue if abs(p.time_s-desired)<1e-9),None)
        if self._binding_invalid:
            measurement=ReceiverMeasurement(now,None,None,False,None,None,sigma,3.*sigma,None,None,'mount_definition_changed')
        elif delayed is None:
            measurement=ReceiverMeasurement(now,None,None,False,None,None,sigma,3.*sigma,None,None,'delay_buffer_warming')
        else:
            w=delayed.wrench6;load=max(0.,-float(w[2])) if delayed.valid else None;cop=None
            minimum=self._noise.cop_min_load_N
            if delayed.valid and minimum is not None and load>max(minimum,3.*self._noise.force_sigma_N):cop=np.array([-w[4]/w[2],w[3]/w[2]])
            measurement=ReceiverMeasurement(now,delayed.time_s,now-delayed.time_s,delayed.valid,w,load,sigma,3.*sigma,cop,delayed.phase,delayed.reason)
        self._last_tick=tick;self._last=measurement;return measurement
    def instrument_metadata(self):
        return dict(role=ROLE,mount_geom_name=self._mount.geom_name,excluded_geom_names=list(self._mount.excluded_geom_names),coverage=self._mount.coverage,
            top_frame_origin_in_geom_m=self._mount.origin_in_geom_m.tolist(),top_plane_tolerance_m=self._mount.top_plane_tolerance_m,
            normal_alignment_min_cos=self._mount.normal_alignment_min_cos,period_s=PERIOD_S,delay_s=DELAY_S,sigma6=self._noise.sigma6.tolist(),
            uncertainty_kind=UNCERTAINTY_KIND,seed=self._seed,independent_stream_id=RECEIVER_STREAM_ID,
            acquisitions=self._acquisitions,gaussian_draws=self._gaussian_draws,uniform_draws=self._uniform_draws,
            source_model_data_pair_requires_runtime_binding=True,global_model_admitted=False,integration_status='external_runtime_binding_and_blind_bridge_required')

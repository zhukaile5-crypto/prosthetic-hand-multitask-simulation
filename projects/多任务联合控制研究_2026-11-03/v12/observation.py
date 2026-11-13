"""One public tactile measurement state; no actuation/pressure response model.

Pressure belongs to acquisition time. Filtering is not delay compensation or
current-force prediction. Unknown channels are NaN plus explicit masks, never
invented zero pressure. SensorFrame's legacy model fields are never read.
"""
from collections.abc import Mapping
from numbers import Real
import numpy as np

FORBIDDEN_FIELDS=frozenset(('truth','object_state','object_position','object_velocity',
    'target_id','object_id','is_target','true_mass','true_friction','mass','friction',
    'future_disturbances','disturbance_schedule','scenario','scene'))


def number(value,name,minimum=None):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value):
        raise ValueError(name+' must be a finite real number')
    x=float(value)
    if minimum is not None and x<minimum:raise ValueError(name+' is below its domain')
    return x


def array(value,shape,name,finite=False):
    a=np.asarray(value)
    if a.shape!=shape or a.dtype.kind not in 'iuf':raise ValueError(name+' has invalid numeric shape/type')
    a=a.astype(float,copy=True)
    if finite and not np.isfinite(a).all():raise ValueError(name+' must be finite')
    return a


def mask(value,shape,name):
    a=np.asarray(value)
    if a.shape!=shape or a.dtype.kind!='b':raise ValueError(name+' must be an explicit bool mask')
    return a.copy()


def get(frame,key,default=None):
    return frame.get(key,default) if isinstance(frame,Mapping) else getattr(frame,key,default)


def gain(g,key,default):return g.get(key,default) if isinstance(g,Mapping) else getattr(g,key,default)


class PublicTactileState:
    def __init__(self,gains):
        self.sensor_tau=number(gain(gains,'sensor_tau',.030),'sensor_tau',.001)
        self.nominal_mu=number(gain(gains,'nominal_mu',.45),'nominal_mu',.10)
        self.time=0.;self.current_time=0.;self._last_time=None
        self._n=np.full(5,np.nan);self._t=np.full((5,2),np.nan);self._s=np.full((5,2),np.nan)
        self._shear=np.full(5,np.nan);self._speed=np.full(5,np.nan)
        self.normal_time=np.full(5,np.nan);self.valid=np.zeros(5,bool);self.contact=np.zeros(5,bool)
        self.normal=self._n.copy();self.delayed_normal=self.normal.copy()
        self.observed_normal=self.normal.copy();self.tangential=self._t.copy();self.slip=self._s.copy()
        self.shear_magnitude=self._shear.copy();self.slip_magnitude=self._speed.copy()
        self.observation_age=np.full(5,np.inf);self.confidence=np.zeros(5)
        self.force_uncertainty=np.full(5,np.nan);self.uncertainty_valid=np.zeros(5,bool)
        self.alpha=np.full(5,np.nan);self.nominal_weights=np.ones(5)
        self.mu=np.full(5,self.nominal_mu);self.mu_valid=np.zeros(5,bool)
        self.mu_observable=np.zeros(5,bool);self.sliding_since=np.zeros(5)
        self._mu_evidence_time=np.full(5,np.nan);self._mu_evidence_active=np.zeros(5,bool)
        self.distributed_valid=np.zeros(5,bool);self.normal_wrench_map=np.full((6,5),np.nan)
        self.finger_non_normal_by_finger=np.full((5,6),np.nan)
        self.finger_non_normal_wrench=np.full(6,np.nan)
        self.normal_support_coeff=np.full(5,np.nan);self.tangent_support_coeff=np.full(5,np.nan)
        self.auxiliary_wrench=np.full(6,np.nan);self.auxiliary_valid=False
        self.auxiliary_pressure=None;self.auxiliary_observation_age=float('inf')
        self.auxiliary_support=None;self.auxiliary_support_signed=None
        self.load=None;self.load_valid=False;self.load_uncertainty=None;self.load_confidence=0.
        self.load_sources={};self.slip_speed=float('nan');self._last_motor_time=None
        self._aux=np.full(6,np.nan);self.auxiliary_time=float('nan');self.sample_time=None
        self.pressure_prediction_available=False;self.predicted_normal=np.full(5,np.nan)
        self.pressure_state_scope='filtered public acquisition-time measurement, not present-pressure prediction'

    def update(self,frame,dt):
        dt=number(dt,'observer dt',1e-12)
        keys=set(frame) if isinstance(frame,Mapping) else set(vars(frame))
        forbidden=keys&FORBIDDEN_FIELDS|{k for k in keys if isinstance(k,str) and k.startswith(('object_','truth_','true_','target_id'))}
        if forbidden:raise ValueError('Forbidden truth/identity/future inputs: '+str(sorted(forbidden)))
        now=number(get(frame,'time'),'frame.time',0.)
        if self._last_time is not None and now<self._last_time-1e-9:raise ValueError('Observer clock rewind')
        raw=array(get(frame,'normal'),(5,),'normal')
        tang=array(get(frame,'tangential'),(5,2),'tangential')
        slip=array(get(frame,'slip_local'),(5,2),'slip_local')
        age=np.asarray(get(frame,'age'))
        if age.dtype.kind not in 'iuf':raise ValueError('age must be a real scalar or5-vector')
        age=age.astype(float)
        if age.ndim==0:age=np.full(5,float(age))
        if age.shape!=(5,):raise ValueError('age must be scalar or5-vector')
        valid=mask(get(frame,'valid'),(5,),'valid')
        valid&=np.isfinite(raw)&np.isfinite(tang).all(1)&np.isfinite(slip).all(1)
        valid&=np.isfinite(age)&(age>=0)&(age<=.12)
        acquisition=now-age
        sample=get(frame,'sample_time')
        if sample is not None:
            sample=number(sample,'sample_time',0.)
            if sample>now+1e-9:raise ValueError('Future tactile acquisition')
            if np.all(np.isfinite(age)) and np.ptp(age)<1e-9 and abs(acquisition[0]-sample)>1e-7:
                raise ValueError('sample_time and observation age disagree')
        shear=get(frame,'shear_magnitude');speed=get(frame,'slip_magnitude')
        shear=np.linalg.norm(tang,axis=1) if shear is None else array(shear,(5,),'shear_magnitude')
        speed=np.linalg.norm(slip,axis=1) if speed is None else array(speed,(5,),'slip_magnitude')
        valid&=np.isfinite(shear)&np.isfinite(speed)&(shear>=0)&(speed>=0)
        if np.any(valid&np.isfinite(self.normal_time)&(acquisition<self.normal_time-1e-9)):
            raise ValueError('Tactile acquisition clock rewind')
        fresh=valid&(~np.isfinite(self.normal_time)|(acquisition>self.normal_time+1e-9))
        elapsed=np.where(np.isfinite(self.normal_time),acquisition-self.normal_time,dt)
        for i in np.flatnonzero(fresh):
            a=1.-np.exp(-max(0.,elapsed[i])/self.sensor_tau)
            if not np.isfinite(self._n[i]):a=1.
            self._n[i]=(self._n[i]+a*(max(0.,raw[i])-self._n[i])
                        if np.isfinite(self._n[i]) else max(0.,raw[i]))
            for store,value in ((self._t,tang),(self._s,slip)):
                store[i]=value[i] if not np.isfinite(store[i]).all() else store[i]+a*(value[i]-store[i])
            self._shear[i]=shear[i] if not np.isfinite(self._shear[i]) else self._shear[i]+a*(shear[i]-self._shear[i])
            self._speed[i]=speed[i] if not np.isfinite(self._speed[i]) else self._speed[i]+a*(speed[i]-self._speed[i])
            self.normal_time[i]=acquisition[i]
        self.time=self.current_time=now;self._last_time=now
        self.sample_time=float(acquisition[valid][0]) if valid.any() and np.ptp(acquisition[valid])<1e-9 else None
        self.valid=valid;self.observation_age=np.where(valid,now-self.normal_time,np.inf)
        self.normal=np.where(valid,self._n,np.nan);self.delayed_normal=self.normal.copy()
        self.observed_normal=np.where(valid,np.maximum(raw,0.),np.nan)
        self.tangential=np.where(valid[:,None],self._t,np.nan)
        self.slip=np.where(valid[:,None],self._s,np.nan)
        self.shear_magnitude=np.where(valid,self._shear,np.nan);self.slip_magnitude=np.where(valid,self._speed,np.nan)
        self.confidence=np.where(valid,np.clip(1.-self.observation_age/.15,0.,1.),0.)
        self.alpha=array(get(frame,'alpha'),(5,),'alpha',finite=True)
        profile=array(get(frame,'nominal_force_weights',np.ones(5)),(5,),'nominal_force_weights',finite=True)
        if np.any(profile<=0):raise ValueError('nominal weights must be positive')
        self.nominal_weights=profile/profile.mean()
        present=get(frame,'contact_present')
        present=np.ones(5,bool) if present is None else mask(present,(5,),'contact_present')
        self.contact=valid&present&(self.normal>.15)
        self.slip_speed=float(np.max(self.slip_magnitude[self.contact])) if self.contact.any() else (0. if valid.all() else float('nan'))
        self.up=array(get(frame,'support_axis',[0.,0.,1.]),(3,),'support_axis',finite=True)
        if np.linalg.norm(self.up)<1e-9:raise ValueError('support axis must have direction')
        self.up/=np.linalg.norm(self.up)
        geometry=get(frame,'normal_wrench_map');non=get(frame,'finger_non_normal_by_finger')
        self.distributed_valid=np.zeros(5,bool)
        self.normal_wrench_map[:]=np.nan;self.finger_non_normal_by_finger[:]=np.nan
        self.normal_support_coeff[:]=np.nan;self.tangent_support_coeff[:]=np.nan
        if geometry is not None and non is not None:
            geometry=array(geometry,(6,5),'normal_wrench_map');non=array(non,(5,6),'finger_non_normal_by_finger')
            dv=mask(get(frame,'distributed_valid'),(5,),'distributed_valid')&valid&np.isfinite(geometry).all(0)&np.isfinite(non).all(1)
            nc=array(get(frame,'normal_support_coeff'),(5,),'normal_support_coeff')
            tc=array(get(frame,'tangent_support_coeff'),(5,),'tangent_support_coeff')
            dv&=np.isfinite(nc)&np.isfinite(tc)&(tc>=0)
            self.distributed_valid=dv;self.normal_wrench_map[:,dv]=geometry[:,dv]
            self.finger_non_normal_by_finger[dv]=non[dv]
            self.normal_support_coeff[dv]=nc[dv];self.tangent_support_coeff[dv]=tc[dv]
        self.finger_non_normal_wrench=(self.finger_non_normal_by_finger.sum(0) if self.distributed_valid.all() else np.full(6,np.nan))
        characteristics=get(frame,'sensor_characteristics',{}) or {}
        sigma=characteristics.get('force_sigma_N')
        self.force_uncertainty[:]=np.nan;self.uncertainty_valid[:]=False
        if sigma is not None:
            sigma=number(sigma,'force_sigma_N',0.)
            self.force_uncertainty[valid]=sigma+abs(raw[valid]-self.normal[valid])
            self.uncertainty_valid=valid.copy()
        self.uncertainty_role='engineering measurement-noise plus filter residual; not predictive or hard confidence bound'
        velocity=array(get(frame,'joint_velocity',np.full(5,np.nan)),(5,),'joint_velocity')
        phase=str(get(frame,'phase','UNKNOWN')).upper()
        supported=self.contact.sum()>=3 and self.contact[0]
        self.mu_observable=self.contact&np.isfinite(velocity)&(abs(velocity)<.08)&(phase in ('HOLD','HOLDING'))&supported
        ratio=self.shear_magnitude/np.maximum(.15,self.delayed_normal)
        incipient=self.mu_observable&(self.slip_magnitude>.004)&(ratio>.06)
        mu_elapsed=np.zeros(5)
        for i in range(5):
            if not incipient[i]:
                self.sliding_since[i]=0.;self._mu_evidence_active[i]=False
            elif fresh[i]:
                gap=acquisition[i]-self._mu_evidence_time[i]
                continuous=bool(self._mu_evidence_active[i] and np.isfinite(gap) and 0<gap<=dt+1e-8)
                mu_elapsed[i]=gap if continuous else 0.
                self.sliding_since[i]=self.sliding_since[i]+gap if continuous else 0.
                self._mu_evidence_time[i]=acquisition[i];self._mu_evidence_active[i]=True
        sustained=incipient&fresh&(self.sliding_since>=.080)
        self.mu[sustained]+=(1.-np.exp(-mu_elapsed[sustained]/.30))*(np.minimum(self.mu[sustained],np.maximum(.10,.92*ratio[sustained]))-self.mu[sustained])
        self.mu_valid|=sustained
        self.mu_scope='nominal prior until sustained public incipient-slip evidence; downward proxy, not measured material identity'
        self._load(frame,now,raw,acquisition)
        return self

    def _load(self,frame,now,raw,acquisition):
        age=get(frame,'auxiliary_age',float('inf'))
        a=get(frame,'auxiliary_wrench');p=get(frame,'auxiliary_pressure')
        self.auxiliary_valid=False;self.auxiliary_wrench[:]=np.nan
        self.auxiliary_pressure=None;self.auxiliary_support=None;self.auxiliary_support_signed=None
        if a is not None and isinstance(age,Real) and not isinstance(age,(bool,np.bool_)):
            a=array(a,(6,),'auxiliary_wrench');age=float(age)
            valid=get(frame,'auxiliary_valid',False)
            if not isinstance(valid,(bool,np.bool_)):raise ValueError('auxiliary_valid must be bool')
            self.auxiliary_valid=bool(valid and np.isfinite(a).all() and np.isfinite(age) and 0<=age<=.12
                and isinstance(p,Real) and not isinstance(p,(bool,np.bool_)) and np.isfinite(p) and p>=0)
            if self.auxiliary_valid:
                stamp=now-age
                if np.isfinite(self.auxiliary_time) and stamp<self.auxiliary_time-1e-9:raise ValueError('Auxiliary acquisition rewind')
                if not np.isfinite(self.auxiliary_time) or stamp>self.auxiliary_time+1e-9:
                    blend=1. if not np.isfinite(self.auxiliary_time) else 1.-np.exp(-(stamp-self.auxiliary_time)/self.sensor_tau)
                    self._aux=a.copy() if blend==1. else self._aux+blend*(a-self._aux)
                    self.auxiliary_time=stamp
                self.auxiliary_wrench=self._aux.copy();self.auxiliary_pressure=float(p)
                self.auxiliary_support_signed=float(self._aux[:3]@self.up);self.auxiliary_support=max(0.,self.auxiliary_support_signed)
        self.auxiliary_observation_age=float(age)
        aligned=bool(self.distributed_valid.all() and self.auxiliary_valid and
            np.max(abs(acquisition-(now-age)))<=.005+1e-9)
        tactile=None
        if aligned:
            # Raw co-acquired distributed vectors, not filtered pressure mapped
            # through a new geometry; all external contacts, no target selector.
            force=(self.normal_wrench_map[:3]*np.maximum(raw,0)[None,:]).sum(1)+self.finger_non_normal_wrench[:3]+a[:3]
            tactile=max(0.,float(force@self.up))
        motor=get(frame,'load_estimate');motor_unc=get(frame,'load_uncertainty')
        motor_flag=get(frame,'load_valid',False)
        if not isinstance(motor_flag,(bool,np.bool_)):raise ValueError('load_valid must be bool')
        motor_valid=bool(motor_flag and isinstance(motor,Real) and not isinstance(motor,(bool,np.bool_)) and np.isfinite(motor)
            and isinstance(motor_unc,Real) and not isinstance(motor_unc,(bool,np.bool_)) and np.isfinite(motor_unc) and motor_unc>0)
        vals=[];unc=[];labels=[]
        if tactile is not None:
            width=get(frame,'auxiliary_uncertainty',float('inf'))
            if isinstance(width,Real) and np.isfinite(width) and width>=0 and self.uncertainty_valid.all():
                vals.append(tactile);unc.append(.55+float(width)+float(self.force_uncertainty.sum()));labels.append('full_hand_tactile')
        if motor_valid:vals.append(max(0.,float(motor)));unc.append(max(.15,float(motor_unc)));labels.append('public_motor_load')
        self.load_valid=bool(vals);self.load=None;self.load_uncertainty=None;self.load_confidence=0.
        if vals:
            w=1./np.square(unc);w/=w.sum();self.load=float(w@vals)
            self.load_uncertainty=max(.20,float(w@unc),.5*(max(vals)-min(vals)))
            self.load_confidence=1.
        self.load_sources=dict(tactile_N=tactile,motor_N=max(0.,float(motor)) if motor_valid else None,
            used_channels=labels,tactile_age_s=self.auxiliary_observation_age if aligned else None,
            motor_time_s=now if motor_valid else None,
            role='public all-external support projection/motor load proxy; no object weight oracle or current-pressure prediction')

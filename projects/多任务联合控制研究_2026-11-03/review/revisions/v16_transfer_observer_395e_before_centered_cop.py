"""Experimental public support-transfer EVIDENCE, never placement certification.

H is signed hand->external vertical force, R is receiver->external support along
the same public up direction, not automatically receiver normal compression.
The clamp-pressure sum is not H or object weight. H+R is only a quasi-static
two-surface proxy: hand/foreign contact and unknown object acceleration confound
it. The caller owns co-acquired frame/geometry provenance and all execution.

200Hz/10ms are the existing devices; the .2s window is the existing observation
dwell. No tuned filter, force-gradient/model, optimizer, future-zero or physics.
Three-sigma quantities below are engineering proxies, not Gaussian hard bounds.
"""
from collections import deque
from dataclasses import dataclass
from numbers import Real
import json
import numpy as np
from v15_support.receiver_sensor import ReceiverMeasurement, PERIOD_S, DELAY_S

WINDOW_S=.2
CLOCK_EPS=1e-9

def _real(x,name,nonnegative=False):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not np.isfinite(x):raise ValueError(name+' requires finite real, not bool/string')
    x=float(x)
    if nonnegative and x<0:raise ValueError(name+' must be nonnegative')
    return x
def _array(x,shape,name):
    raw=np.asarray(x,dtype=object)
    if raw.shape!=shape or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in raw.flat):raise ValueError(name+' requires numeric '+str(shape))
    a=np.asarray(x,float)
    if not np.isfinite(a).all():raise ValueError(name+' unknown/nonfinite')
    return np.frombuffer(a.tobytes(),dtype=float).reshape(shape)
def _flag(x,name):
    if not isinstance(x,(bool,np.bool_)):raise ValueError(name+' must be bool')
    return bool(x)
def _mask(x):
    a=np.asarray(x,dtype=object)
    if a.shape!=(5,) or any(not isinstance(v,(bool,np.bool_)) for v in a):raise ValueError('Explicit5 bool validity flags required')
    return np.asarray(x,bool)
def _tick(x):
    t=_real(x,'public time',True);k=int(round(t/PERIOD_S))
    if abs(t-k*PERIOD_S)>CLOCK_EPS:raise ValueError('Registered200Hz grid required')
    return k,k*PERIOD_S
def _plain(x):
    if isinstance(x,np.ndarray):return _plain(x.tolist())
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:_plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_plain(v) for v in x]
    return x

@dataclass(frozen=True,slots=True)
class PreReleaseLoadPrior:
    value_N:float
    uncertainty_N:float
    reference_time_s:float
    role:str='public_pre_release_motor_load_proxy_not_true_weight'
    def __post_init__(self):
        for k in ('value_N','uncertainty_N','reference_time_s'):object.__setattr__(self,k,_real(getattr(self,k),k,True))
        if self.role!='public_pre_release_motor_load_proxy_not_true_weight':raise ValueError('Only the declared public prior role is supported')
    @classmethod
    def from_history(cls,request_s,read_times_s,values_N,uncertainties_N,valid):
        """Exactly40 pre-request left samples. No sigma/sqrt(n) assumption."""
        _,request=_tick(request_s);n=round(WINDOW_S/PERIOD_S)
        t=_array(read_times_s,(n,),'prior times');v=_array(values_N,(n,),'prior values');u=_array(uncertainties_N,(n,),'prior uncertainty')
        flags=np.asarray(valid,dtype=object)
        if flags.shape!=(n,) or any(not isinstance(x,(bool,np.bool_)) for x in flags):raise ValueError('Explicit prior validity required')
        if not np.asarray(valid,bool).all() or np.any(v<0) or np.any(u<0):return None
        if not np.allclose(t,request-WINDOW_S+np.arange(n)*PERIOD_S,atol=CLOCK_EPS,rtol=0):return None
        mean=float(v.mean());return cls(mean,float(u.mean()+np.max(abs(v-mean))),float(t[-1]))

@dataclass(frozen=True,slots=True)
class HandSupportSample:
    read_time_s:float
    acquisition_time_s:float|None
    valid:bool
    signed_support_N:float|None
    uncertainty_N:float|None
    reason:str|None=None
    role:str='signed_hand_to_all_external_along_coacquired_public_up'
    def __post_init__(self):
        object.__setattr__(self,'read_time_s',_real(self.read_time_s,'hand read clock',True));object.__setattr__(self,'valid',_flag(self.valid,'hand valid'))
        if self.acquisition_time_s is not None:
            a=_real(self.acquisition_time_s,'hand acquisition',True)
            if a>self.read_time_s+CLOCK_EPS:raise ValueError('Future hand acquisition')
            object.__setattr__(self,'acquisition_time_s',a)
        if self.role!='signed_hand_to_all_external_along_coacquired_public_up':raise ValueError('Unknown hand support role')
        if self.valid:
            if self.acquisition_time_s is None or self.reason is not None:raise ValueError('Valid hand sample needs acquisition and no failure reason')
            object.__setattr__(self,'signed_support_N',_real(self.signed_support_N,'signed hand support'))
            object.__setattr__(self,'uncertainty_N',_real(self.uncertainty_N,'hand engineering uncertainty',True))
        elif self.signed_support_N is not None or self.uncertainty_N is not None:raise ValueError('Unknown hand load must remainNone')
    @classmethod
    def from_components(cls,*,read_time_s,acquisition_time_s,up_at_acquisition3,axis_time_s,normal5,
            normal_wrench_map6x5,non_normal5x6,tangent_basis5x2x3,auxiliary_wrench6,
            valid5,distributed_valid5,auxiliary_valid,auxiliary_acquisition_s,
            force_sigma_N,tangential_sigma_N,auxiliary_uncertainty_N):
        now=_real(read_time_s,'read clock',True);acq=_real(acquisition_time_s,'acquisition clock',True)
        if acq>now+CLOCK_EPS:raise ValueError('Future hand sample')
        flags=_mask(valid5)&_mask(distributed_valid5);auxflag=_flag(auxiliary_valid,'auxiliary valid')
        if not flags.all() or not auxflag:return cls(now,acq,False,None,None,'missing_distributed_hand_or_auxiliary')
        if abs(_real(axis_time_s,'axis clock',True)-acq)>CLOCK_EPS or abs(_real(auxiliary_acquisition_s,'auxiliary clock',True)-acq)>CLOCK_EPS:return cls(now,acq,False,None,None,'hand_axis_or_auxiliary_not_coacquired')
        try:
            up=_array(up_at_acquisition3,(3,),'up');scale=float(np.max(abs(up)))
            if scale==0:raise ValueError('Unknown up direction')
            up=(up/scale)/np.linalg.norm(up/scale);n=_array(normal5,(5,),'normal');M=_array(normal_wrench_map6x5,(6,5),'normal map');F=_array(non_normal5x6,(5,6),'non-normal');B=_array(tangent_basis5x2x3,(5,2,3),'tangent basis');a=_array(auxiliary_wrench6,(6,),'auxiliary wrench')
            if np.any(n<0):raise ValueError('Normal pressure must be nonnegative')
            sf=_real(force_sigma_N,'force sigma',True);st=_real(tangential_sigma_N,'tangent sigma',True);ua=_real(auxiliary_uncertainty_N,'declared auxiliary width',True)
            force=M[:3]@np.maximum(n,0)+F[:,:3].sum(0)+a[:3]
            width=3*sf*float(abs(up@M[:3]).sum())+3*st*float(abs(np.einsum('ijk,k->ij',B,up)).sum())+ua
            return cls(now,acq,True,float(force@up),width)
        except ValueError:return cls(now,acq,False,None,None,'unknown_wrench_geometry_or_device_uncertainty')

@dataclass(frozen=True,slots=True)
class TransferEvidence:
    time_s:float
    status:str
    stage_permission:str
    unknown_reasons:tuple
    _payload_json:str
    @property
    def continuous_support_evidence(self):return json.loads(self._payload_json)
    def to_public(self):
        return dict(schema='experimental-continuous-support-transfer-v1',time_s=self.time_s,status=self.status,
            stage_permission=self.stage_permission,unknown_reasons=list(self.unknown_reasons),continuous_support_evidence=self.continuous_support_evidence,
            release_request_may_be_delayed=False,holding_support_may_be_requested=False,placement_certified=False,physical_safety_certified=False)

class TransferObserver:
    def __init__(self,*,request_time_s,receiver_half_size_xy_m,receiver_up_axis3,prior:PreReleaseLoadPrior|None=None):
        _,self.request=_tick(request_time_s);self.half=_array(receiver_half_size_xy_m,(2,),'receiver half size')
        if np.any(self.half<=0):raise ValueError('Positive receiver geometry required')
        self.receiver_up=_array(receiver_up_axis3,(3,),'public up in receiver frame')
        scale=float(np.max(abs(self.receiver_up)))
        if scale==0:raise ValueError('Known receiver/up relation required')
        self.receiver_up=_array((self.receiver_up/scale)/np.linalg.norm(self.receiver_up/scale),(3,),'normalized receiver up')
        if prior is not None and (type(prior) is not PreReleaseLoadPrior or prior.reference_time_s>self.request+CLOCK_EPS):raise ValueError('Prior must be public and no later than release request')
        self.prior=prior;self.window=deque();self.anchor=None;self.last_tick=None;self.signature=None;self.last=None
    def _cop(self,r):
        w=np.asarray(r.wrench6);u=np.asarray(r.uncertainty6)
        if -w[2]<=u[2]:return None
        xx=[-m/f for m in (w[4]-u[4],w[4]+u[4]) for f in (w[2]-u[2],w[2]+u[2])]
        yy=[m/f for m in (w[3]-u[3],w[3]+u[3]) for f in (w[2]-u[2],w[2]+u[2])]
        xy=np.array([-w[4]/w[2],w[3]/w[2]]);lower=np.array([min(xx),min(yy)]);upper=np.array([max(xx),max(yy)])
        if not np.isfinite(np.r_[xy,lower,upper]).all():return None
        return dict(xy=xy,width=np.maximum(xy-lower,upper-xy),engineering_edge_margin_m=float(np.min(self.half-np.maximum(abs(lower),abs(upper)))))
    def update(self,*,time_s,hand:HandSupportSample,receiver:ReceiverMeasurement,
            hand_receiver_confounded=None,geometry_acquisition_time_s=None):
        tick,now=_tick(time_s)
        if now<self.request-CLOCK_EPS:raise ValueError('Update before release request')
        if type(hand) is not HandSupportSample or type(receiver) is not ReceiverMeasurement:raise TypeError('Typed public samples required')
        if hand_receiver_confounded is not None:hand_receiver_confounded=_flag(hand_receiver_confounded,'hand receiver confound')
        if geometry_acquisition_time_s is not None:
            geometry_acquisition_time_s=_real(geometry_acquisition_time_s,'geometry evidence clock',True)
            if geometry_acquisition_time_s>now+CLOCK_EPS:raise ValueError('Future geometry evidence')
        signature=json.dumps(_plain(dict(hand={k:getattr(hand,k) for k in hand.__dataclass_fields__},receiver=receiver.to_public(),confounded=hand_receiver_confounded,geometry_time=geometry_acquisition_time_s)),sort_keys=True,allow_nan=False)
        if self.last_tick is not None:
            if tick<self.last_tick:raise ValueError('Public clock rewind')
            if tick==self.last_tick:
                if signature!=self.signature:raise ValueError('Changed payload on consumed tick')
                return self.last
        reasons=[]
        if self.last_tick is not None and tick!=self.last_tick+1:self.window.clear();reasons.append('public_tick_gap')
        valid=hand.valid and receiver.valid and hand.acquisition_time_s is not None and receiver.acquisition_time_s is not None
        if abs(hand.read_time_s-now)>CLOCK_EPS or abs(receiver.time_s-now)>CLOCK_EPS:valid=False;reasons.append('delivery_clock_mismatch')
        acq=hand.acquisition_time_s
        if valid and (abs(acq-receiver.acquisition_time_s)>CLOCK_EPS or abs(now-acq-DELAY_S)>CLOCK_EPS):valid=False;reasons.append('not_coacquired_at_registered_delay')
        if valid and acq<self.request-CLOCK_EPS:valid=False;reasons.append('not_yet_postrequest')
        if not valid:
            self.window.clear();reasons.append('measurement_unknown_or_ineligible')
            payload=dict(window_complete=False,hand_support_N=None,receiver_support_N=None,coupled_transfer_fraction=None,cop=None)
            status='UNKNOWN';permission='OBSERVATION_ONLY'
        else:
            if self.window and acq<=self.window[-1]['acq']+CLOCK_EPS:raise ValueError('Hand/receiver acquisition not fresh')
            if self.window and abs(acq-self.window[-1]['acq']-PERIOD_S)>CLOCK_EPS:self.window.clear();reasons.append('acquisition_gap')
            wrench=np.asarray(receiver.wrench6);uncertainty=np.asarray(receiver.uncertainty6)
            geometry_known=hand_receiver_confounded is not None and geometry_acquisition_time_s is not None and abs(geometry_acquisition_time_s-acq)<=CLOCK_EPS
            sample=dict(acq=max(acq,self.request),H=hand.signed_support_N,uH=hand.uncertainty_N,
                R=float(-wrench[:3]@self.receiver_up),uR=float(abs(self.receiver_up)@uncertainty[:3]),
                compression=receiver.normal_load_N,wrench=wrench,width6=uncertainty,cop=self._cop(receiver),
                geometry_clear=geometry_known and not hand_receiver_confounded)
            if self.anchor is None:self.anchor=sample.copy()
            self.window.append(sample)
            while self.window and self.window[0]['acq']<sample['acq']-WINDOW_S-CLOCK_EPS:self.window.popleft()
            complete=len(self.window)==round(WINDOW_S/PERIOD_S)+1 and sample['acq']-self.window[0]['acq']>=WINDOW_S-CLOCK_EPS
            H=float(np.mean([s['H'] for s in self.window]));uH=float(np.mean([s['uH'] for s in self.window]));R=float(np.mean([s['R'] for s in self.window]));uR=float(np.mean([s['uR'] for s in self.window]))
            payload=dict(window_complete=complete,window_samples=len(self.window),oldest_acquisition_s=self.window[0]['acq'],latest_acquisition_s=sample['acq'],
                hand_support_N=H,hand_engineering_width_N=uH,receiver_support_N=R,receiver_engineering_width_N=uR,
                receiver_compression_N=float(np.mean([s['compression'] for s in self.window])),receiver_mean_wrench6=np.mean([s['wrench'] for s in self.window],axis=0),receiver_mean_engineering_width6=np.mean([s['width6'] for s in self.window],axis=0),
                downward_hand_load_resolved=H+uH<0,
                load_sum_N=H+R,load_sum_engineering_interval_N=[H+R-uH-uR,H+R+uH+uR],coupled_transfer_fraction=None,cop=sample['cop'],cop_trend=None,
                prior=None if self.prior is None else dict(value_N=self.prior.value_N,uncertainty_N=self.prior.uncertainty_N,time_s=self.prior.reference_time_s,role=self.prior.role),
                roles=dict(loads='All-external hand and any-actor receiver; signed H retained, no object identity.',uncertainty='Engineering proxies only; no independent-sample shrink or hard bound.',cop='Equivalent net-wrench point, not pressure centroid, CoM or object pose.'))
            balanced=None
            if self.prior is not None and self.prior.value_N-self.prior.uncertainty_N>0:
                low=self.prior.value_N-self.prior.uncertainty_N;high=self.prior.value_N+self.prior.uncertainty_N
                rg=[max(0.,R-uR-self.anchor['R']-self.anchor['uR']),max(0.,R+uR-self.anchor['R']+self.anchor['uR'])]
                hd=[max(0.,self.anchor['H']-self.anchor['uH']-H-uH),max(0.,self.anchor['H']+self.anchor['uH']-H+uH)]
                payload['coupled_transfer_fraction']=[min(1.,min(rg[0],hd[0])/high),min(1.,min(rg[1],hd[1])/low)]
                balanced=H+R+uH+uR>=low and H+R-uH-uR<=high
                payload['load_sum_compatible_with_prior_proxy']=balanced
            else:reasons.append('positive_load_reference_not_resolved')
            motion=None
            if complete and all(s['cop'] is not None for s in self.window):
                ts=np.array([s['acq'] for s in self.window]);x=ts-ts.mean();weights=x/float(x@x);cop=np.array([s['cop']['xy'] for s in self.window]);width=np.array([s['cop']['width'] for s in self.window]);slope=weights@cop;uw=abs(weights)@width
                motion=bool(np.any(abs(slope)>uw));payload['cop_trend']=dict(velocity_m_s=slope,engineering_width_m_s=uw,motion_resolved_against_engineering_proxy=motion,role='Past-window regression only; no future contact prediction')
            elif complete:reasons.append('cop_window_not_loaded_or_unknown')
            if not geometry_known:reasons.append('hand_receiver_geometry_attribution_unknown')
            elif hand_receiver_confounded:reasons.append('possible_hand_receiver_contact')
            geometry_window_clear=complete and all(s['geometry_clear'] for s in self.window)
            payload['geometry_clear_throughout_window']=geometry_window_clear
            if complete and not geometry_window_clear:reasons.append('hand_receiver_clearance_not_observed_throughout_window')
            usable=bool(complete and geometry_window_clear and balanced and motion is False and not payload['downward_hand_load_resolved'] and R-uR>0 and payload['coupled_transfer_fraction'] is not None and payload['coupled_transfer_fraction'][0]>0)
            permission='EXPERIMENTAL_COORDINATION_EVIDENCE' if usable else 'OBSERVATION_ONLY'
            status='ACCUMULATING_WINDOW' if not complete else 'MOVING_WRENCH_POINT_PROXY' if motion else 'CONTINUOUS_AGGREGATE_PROXY'
            if not balanced and balanced is not None:status='DYNAMIC_OR_CONFOUNDED_LOAD_PROXY'
            if payload['downward_hand_load_resolved']:status='DOWNWARD_HAND_LOADING_PROXY'
        payload.update(release_permission_owned_by_caller=True,no_holding_recovery_requested=True,placement_certified=False,continuous_support_is_not_stationary_target_proof=True)
        try:packed=json.dumps(_plain(payload),sort_keys=True,allow_nan=False)
        except (ValueError,OverflowError):
            self.window.clear();status='NUMERIC_UNAVAILABLE';permission='OBSERVATION_ONLY';reasons.append('unrepresentable_engineering_interval')
            packed=json.dumps(dict(window_complete=False,hand_support_N=None,receiver_support_N=None,coupled_transfer_fraction=None,cop=None,placement_certified=False))
        evidence=TransferEvidence(now,status,permission,tuple(dict.fromkeys(reasons)),packed)
        self.last_tick=tick;self.signature=signature;self.last=evidence;return evidence

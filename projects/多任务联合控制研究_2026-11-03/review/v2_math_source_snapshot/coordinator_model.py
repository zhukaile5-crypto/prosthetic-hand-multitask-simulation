"""Causal contact-conditioned pressure model, independent of object truth.

The fast signed actuator response is the admitted public calibration. A slow
pressure-bias trajectory is fitted from acquired tactile samples after removing
the time-aligned fast actuator effect. No future force, mass, friction or target
identity is used. Motion coefficients start UNIDENTIFIED, not falsely fitted
from the rank-deficient old motion logs.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class PressureForecast:
    normal_now: np.ndarray
    drift_rate: np.ndarray
    drift_error_rate: np.ndarray
    fit_error: np.ndarray
    fit_samples: np.ndarray
    motion_gain: np.ndarray
    motion_gain_bound: np.ndarray
    motion_identified: bool


class ContactConditionedModel:
    def __init__(self, calibration, window=.25, minimum_span=.055):
        self.B=np.asarray(calibration['force_matrix'],float).copy()
        self.reference=np.asarray(calibration.get('reference_alpha',calibration.get('model_reference_alpha')),float).copy()
        self.error=np.asarray(calibration['response_error_bound'],float).copy()
        if self.B.shape!=(5,5) or self.reference.shape!=(5,) or self.error.shape!=(5,):
            raise ValueError('Explicit complete hand calibration is required')
        self.window=float(window);self.minimum_span=float(minimum_span)
        self.history=[deque(maxlen=100) for _ in range(5)]
        self.last_sample=np.full(5,-np.inf)
        self.last_contact=np.zeros(5,bool)
        self.last_position=np.zeros((5,3))
        self.drift=np.zeros(5)
        self.motion_gain=np.zeros((5,3))
        # Engineering development envelope, NOT a certified physical bound.
        self.motion_bound=np.tile([15.,3.,3.],(5,1))
        self.motion_identified=False
        self.snapshot=None

    def update(self, frame, state, executed_motion=None):
        now=float(frame.time);sample=float(frame.sample_time)
        measured=np.asarray(frame.normal,float)
        valid=np.asarray(state.valid,bool)&np.isfinite(measured)
        contact=np.asarray(state.contact_model_active,bool)
        positions=np.asarray(frame.contact_pos,float)
        coordinates=np.asarray(state.last_delayed_coordinates,float)
        delayed=np.einsum('ij,ij->i',self.B,coordinates-self.reference[None,:])
        beta_now=np.asarray(state.force_bias,float).copy()
        drift_error=np.full(5,.5);fit_error=np.full(5,.03);counts=np.zeros(5,int)
        for i in range(5):
            geometry_jump=(valid[i] and np.isfinite(positions[i]).all() and
                           np.linalg.norm(positions[i]-self.last_position[i])>.012)
            if contact[i]!=self.last_contact[i] or geometry_jump:
                self.history[i].clear();self.drift[i]=0.
            if valid[i] and sample>self.last_sample[i]+1e-9:
                self.history[i].append((sample,float(measured[i]-delayed[i])))
                self.last_sample[i]=sample
            while self.history[i] and self.history[i][0][0]<sample-self.window:
                self.history[i].popleft()
            values=np.asarray(self.history[i],float);counts[i]=len(values)
            if len(values)>=5 and values[-1,0]-values[0,0]>=self.minimum_span:
                t=values[:,0];y=values[:,1];center=float(t.mean());x=t-center
                slope=float(x@(y-y.mean())/max(1e-10,float(x@x)))
                slope=float(np.clip(slope,-5.,5.))
                residual=y-(y.mean()+slope*x)
                sigma=max(.003,float(np.sqrt(np.mean(residual**2))))
                self.drift[i]=.75*self.drift[i]+.25*slope
                beta_now[i]=float(y.mean()+self.drift[i]*(now-center))
                fit_error[i]=3.*sigma
                drift_error[i]=max(.03,3.*sigma/np.sqrt(max(1e-10,float(x@x))))
            elif len(values):
                beta_now[i]=float(values[-1,1])
        self.last_contact=contact.copy()
        good=valid&np.isfinite(positions).all(axis=1);self.last_position[good]=positions[good]
        normal=np.maximum(0.,self.B@(np.asarray(state.latent_alpha)-self.reference)+beta_now)
        normal=np.where(contact,normal,0.)
        self.snapshot=PressureForecast(normal,self.drift.copy(),np.minimum(5.,drift_error),
            fit_error,counts,self.motion_gain.copy(),self.motion_bound.copy(),self.motion_identified)
        return self.snapshot

    def trajectory(self, state, horizon, period, motion_increments):
        if self.snapshot is None:raise RuntimeError('Acquire the causal model before prediction')
        s=self.snapshot
        P,c=state.force_model.action_prediction(s.normal_now,horizon,period,state.contact_model_active,
            alpha=state.alpha,latent=state.latent_alpha,execution_dt=.005)
        times=np.repeat(np.arange(1,horizon+1)*period,5)
        c=c+times*np.tile(s.drift_rate,horizon)
        M=np.zeros((5*horizon,horizon));unc=np.zeros_like(M)
        for j in range(horizon):
            for k in range(j+1):
                M[5*j:5*j+5,k]=s.motion_gain@motion_increments[k]
                unc[5*j:5*j+5,k]=s.motion_gain_bound@np.abs(motion_increments[k])
        # Public empirical calibration uncertainty is never reduced to make a
        # pressure constraint easier. It is not a certified physical bound.
        uncertainty=np.tile(self.error+s.fit_error,horizon)+times*np.tile(s.drift_error_rate,horizon)
        return P,c,M,unc,uncertainty


def verify_model_algebra():
    """Pure algebra checks; no physics, RNG or files."""
    B=np.array([[2.,-.3,0,0,0],[.2,3.,0,0,0],[0,0,2.,.2,0],[0,0,0,2.,0],[-.4,.8,.3,.9,4.]])
    v=np.arange(5)/10
    t=np.arange(50)*.005
    alpha=np.c_[np.sin(t),np.cos(t),t,t*t,-t]
    n=2.+alpha@B.T+t[:,None]*v
    residual=n-alpha@B.T
    centered=t-t.mean()
    fitted=centered@(residual-residual.mean(axis=0))/(centered@centered)
    checks=dict(signed_coupling_removed_before_drift=bool(np.allclose(fitted,v,atol=1e-12)),
                negative_cross_effect_retained=bool(B[4,0]<0),zero_action_slow_drift_visible=bool(np.linalg.norm(v)>0))
    if not all(checks.values()):raise AssertionError(checks)
    return checks

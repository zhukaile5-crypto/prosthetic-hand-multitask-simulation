"""Experimental current-state servo predictors; no plant or past-ref input.

The past actuator response is an unmodelled free-motion/nuisance term. These
predictors do not pretend to reproduce the full-history identified response.
Only declared future20ms/four5ms finite ramps contribute measuredK terms.
"""
from __future__ import annotations
import numpy as np

HORIZONS=(.02,.04,.06,.1)
OLS_WINDOW_S=.12


class PredictionUnavailable(ValueError):pass


def _finite(value,shape,name):
    if value is None:raise PredictionUnavailable(name+'_missing')
    x=np.asarray(value,float)
    if x.shape!=shape or not np.isfinite(x).all():raise PredictionUnavailable(name+'_shape_or_finiteness')
    return x.copy()


def future_response(kernel101x25x8,declared_boundary_times_s,declared_reference8,horizons_s=HORIZONS):
    K=_finite(kernel101x25x8,(101,25,8),'measured_kernel')
    ts=np.asarray(declared_boundary_times_s,float);u=np.asarray(declared_reference8,float);hs=np.asarray(horizons_s,float)
    if ts.ndim!=1 or len(ts)<2 or u.shape!=(len(ts),8) or not np.isfinite(ts).all() or not np.isfinite(u).all():raise PredictionUnavailable('declared_future_ramp_shape_or_finiteness')
    if abs(ts[0])>1e-9 or np.max(abs(np.diff(ts)-.02))>1e-9:raise PredictionUnavailable('declared_future_must_be20ms_boundaries_relative_to_now')
    if hs.ndim!=1 or not len(hs) or not np.isfinite(hs).all() or any(not any(abs(h-z)<1e-9 for z in HORIZONS) for h in hs):raise PredictionUnavailable('unscored_horizon')
    if ts[-1]<max(hs)-1e-9:raise PredictionUnavailable('future_reference_does_not_cover_horizon')
    grid=np.arange(101)*.005;delta=np.diff(u,axis=0);out=[]
    for h in hs:
        use=ts[:-1]<h-1e-10;ages=h-ts[:-1][use]
        if np.max(ages,initial=0)>.5+1e-9:raise PredictionUnavailable('kernel_memory_unavailable')
        ages=np.clip(ages,0.,.5)
        response=np.stack([np.interp(ages,grid,K[:,o,i]) for o in range(25) for i in range(8)],axis=-1).reshape(-1,25,8)
        out.append(np.einsum('koi,ki->o',response,delta[use]))
    return np.stack(out)


def predict_velocity(kernel101x25x8,current_q25,current_velocity25,
                     declared_boundary_times_s,declared_reference8,horizons_s=HORIZONS):
    q=_finite(current_q25,(25,),'current_robot_q25');v=_finite(current_velocity25,(25,),'current_robot_velocity25')
    response=future_response(kernel101x25x8,declared_boundary_times_s,declared_reference8,horizons_s)
    hs=np.asarray(horizons_s,float);free=q[None,:]+hs[:,None]*v[None,:]
    return dict(prediction25=free+response,free_motion25=free,future_controlled_response25=response,
        method='A_current_actual_velocity',uses_past_reference=False,uses_zero_future=False,
        past_servo_memory='unidentified nuisance approximated by current velocity, not certified free motion',
        global_model_admitted=False)


def predict_position_ols(kernel101x25x8,past_times_s,past_q25,
                         declared_boundary_times_s,declared_reference8,horizons_s=HORIZONS):
    t=np.asarray(past_times_s,float);q=np.asarray(past_q25,float)
    if t.ndim!=1 or len(t)<2:raise PredictionUnavailable('position_OLS_needs_at_least_two_past_observations')
    if q.shape!=(len(t),25) or not np.isfinite(q).all() or not np.isfinite(t).all():raise PredictionUnavailable('past_robot_prefix_shape_or_finiteness')
    if np.max(abs(np.diff(t)-.005))>1e-9:raise PredictionUnavailable('position_prefix_requires_contiguous5ms_clock')
    take=t>=t[-1]-OLS_WINDOW_S-1e-10;used=t[take];X=np.column_stack((np.ones(len(used)),used-t[-1]));hs=np.asarray(horizons_s,float)
    W=np.stack([np.array([1.,h])@np.linalg.pinv(X) for h in hs])
    free=W@q[take]
    response=future_response(kernel101x25x8,declared_boundary_times_s,declared_reference8,horizons_s)
    return dict(prediction25=free+response,free_motion25=free,future_controlled_response25=response,
        method='B_fixed_120ms_position_OLS',past_observation_weights=W,past_times_used_s=used,
        uses_past_reference=False,uses_zero_future=False,
        past_servo_memory='unidentified nuisance approximated by causal position trend, not certified free motion',
        global_model_admitted=False)


def predict_robot_sample(sample,kernel101x25x8,declared_boundary_times_s,declared_reference8,horizons_s=HORIZONS):
    """Consumes frozen RobotJointSample by its public robot-only properties.
    Missing lift encoder velocity remains UNAVAILABLE; it is never replaced0.
    """
    if tuple(sample.raw_joint_names)==() or len(sample.raw_joint_names)!=24:raise PredictionUnavailable('named_robot24_required')
    return predict_velocity(kernel101x25x8,sample.q25,sample.velocity25,declared_boundary_times_s,declared_reference8,horizons_s)

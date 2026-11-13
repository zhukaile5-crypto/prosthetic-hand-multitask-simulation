"""Prefix-only numerical API for the exact causal scores in identify.py.

This is NOT a local/global model admission or a controller. It has no zero
trajectory parameter. Future robot references must be a declared task schedule.
Unmeasured posture/sequence support requires a separate consumer scope policy.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

_spec=importlib.util.spec_from_file_location('_servo_response_identify',Path(__file__).with_name('identify.py'))
_identification=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_identification)


def forecast_from_prefix(kernel101x25x8, observed_times_s, observed_robot_q25,
                         actual_boundary_times_s, actual_boundary_reference8,
                         declared_future_boundary_times_s, declared_future_boundary_reference8,
                         horizons_s=(.02,.04,.06,.1)):
    """Predict using q observations through origin, true executed input history,
    and explicitly declared future20ms endpoints. All times share the measured
    context's branch origin; no.5s persistence/interpolation of poses is invented.

    Each20ms endpoint delta denotes the observed four5ms finite ramp, NOT an
    instantaneous input. Origin must lie on the20ms boundary. q observations use
    the5ms end clock. The complete causal residual is q-K*actual_input_history.
    Its last.12s is fit by the SAME unweighted affineLS as the held-out score.
    """
    K=np.asarray(kernel101x25x8,float);qt=np.asarray(observed_times_s,float);q=np.asarray(observed_robot_q25,float)
    at=np.asarray(actual_boundary_times_s,float);au=np.asarray(actual_boundary_reference8,float)
    ft=np.asarray(declared_future_boundary_times_s,float);fu=np.asarray(declared_future_boundary_reference8,float)
    hs=np.asarray(horizons_s,float)
    if K.shape!=(101,25,8) or not np.isfinite(K).all():raise ValueError('Finite measured25x8 kernel required; missing column unavailable')
    if qt.ndim!=1 or not len(qt) or q.shape!=(len(qt),25) or not np.isfinite(q).all() or not np.isfinite(qt).all():raise ValueError('Finite robot25 prefix required')
    if at.ndim!=1 or not len(at) or au.shape!=(len(at),8) or ft.ndim!=1 or not len(ft) or fu.shape!=(len(ft),8):raise ValueError('Actual/future boundary shapes required')
    if any(not np.isfinite(x).all() for x in (at,au,ft,fu,hs)):raise ValueError('Finite public robot input schedule required')
    origin=qt[-1]
    if abs(qt[0])>1e-9 or abs(at[0])>1e-9:raise ValueError('History must retain its measured context origin')
    if len(qt)>1 and np.max(abs(np.diff(qt)-.005))>1e-9:raise ValueError('5ms q end-clock prefix required')
    if len(at)>1 and np.max(abs(np.diff(at)-.02))>1e-9:raise ValueError('20ms executed boundary clock required')
    if len(ft)>1 and np.max(abs(np.diff(ft)-.02))>1e-9:raise ValueError('20ms declared boundary clock required')
    if abs(at[-1]-origin)>1e-9 or abs(ft[0]-origin)>1e-9 or not np.array_equal(au[-1],fu[0]):raise ValueError('Executed/future join must match current origin, no retroactive future input')
    if hs.ndim!=1 or not len(hs) or np.any(hs<=0) or any(not np.any(abs(h-_identification.HORIZONS)<1e-9) for h in hs):raise ValueError('Only explicitly scored20/40/60/100ms horizons')
    if origin+max(hs)>.5+1e-9 or origin+max(hs)>ft[-1]+1e-9:raise ValueError('Future schedule or measured.5s memory unavailable')
    times=np.r_[at,ft[1:]];reference=np.vstack((au,fu[1:]))
    past_effect=np.stack([_identification.effect(K,at,au,t) for t in qt])
    residual=q-past_effect
    forecasts=[];weights=[];background=[];controlled=[]
    for h in hs:
        r,w=_identification.residual_forecast(qt,residual,origin+h)
        c=_identification.effect(K,times,reference,origin+h)
        forecasts.append(c+r);weights.append(w);background.append(r);controlled.append(c)
    return dict(numerical_forecast_available=True,origin_s=float(origin),horizons_s=hs,
        predicted_robot_q25=np.stack(forecasts),causal_background25=np.stack(background),modeled_controlled_increment25=np.stack(controlled),
        residual_weights=weights,uses_zero_future=False,uses_future_robot_q=False,
        uncertainty='Separate recorded per-axis causal error statistics required; no hard bound supplied by this API',
        sequence_support='EXPERIMENTAL_UNLESS_SEPARATELY_MATCHED_TO_RECORDED_CONTEXT_AND_SEQUENCE',global_model_admitted=False)

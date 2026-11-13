"""Identify diagnostic dynamics from ALREADY SEEN archived full-hand traces.

No MuJoCo/model/controller construction and no source-tree mutation. Outputs are
development model evidence, never new-policy performance or unseen validation.
Only pressure, commands, encoders and public calibration are proposed as model
inputs. Object-relative truth is used solely to diagnose mechanics offline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SOURCE=ROOT.parent/'五指策略接入_2026-11-03'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def load(path):
    keys=('time','normal','skin_normal','estimated_normal','alpha_before','alpha',
          'executed_action','contact_model_active','observation_json','relative',
          'rotation','ctrl','actual_motion_target','executed_motion_reference')
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k] for k in keys};meta=json.loads(str(z['metadata']))
    # Cache once: repeatedly accessing NPZ strings would repeatedly decompress.
    observations=[json.loads(str(s)) for s in a.pop('observation_json')]
    for key in ('joint','joint_velocity','raw_joint','raw_joint_velocity','age','valid'):
        a[key]=np.asarray([o[key] for o in observations])
    a['meta']=meta
    return a


def latent_coordinates(a):
    tau=float(a['meta']['calibration']['force_tau']);t=a['time'];alpha=a['alpha_before']
    result=alpha.copy();first=int(np.searchsorted(t,7.3-1e-9));z=alpha[first].copy()
    for i in range(first,len(t)):
        dt=.005 if i==first else float(t[i]-t[i-1]);z+=(-np.expm1(-dt/tau))*(alpha[i]-z);result[i]=z
    return result


def causal_slope(t,y,window=.12):
    """Past-only least-squares slope, no centered/differentiating future filter."""
    result=np.full_like(y,np.nan,dtype=float)
    for i in range(len(t)):
        first=int(np.searchsorted(t,t[i]-window-1e-9));tt=t[first:i+1]
        if len(tt)<round(window/.005):continue
        x=tt-tt.mean();result[i]=x@(y[first:i+1]-y[first:i+1].mean(axis=0))/(x@x)
    return result


def metrics(error):
    return dict(rmse_N=np.sqrt(np.mean(error**2,axis=0)),
                bias_N=np.mean(error,axis=0),p95_absolute_N=np.quantile(abs(error),.95,axis=0),
                maximum_absolute_N=np.max(abs(error),axis=0))


def slow_fit(t,force,start=8.2,split=9.5,end=11.995):
    train=(t>=start-1e-9)&(t<=split+1e-9);test=(t>split+1e-9)&(t<=end+1e-9)
    values=[]
    for tau in np.r_[.005,np.geomspace(.05,20.,180)]:
        X=np.c_[np.ones(len(t)),np.exp(-np.maximum(0.,t-start)/tau)]
        coefficients=np.linalg.lstsq(X[train],force[train],rcond=None)[0]
        pred=X@coefficients
        values.append(dict(tau_s=float(tau),coefficients=coefficients,
            train_rmse_N=float(np.sqrt(np.mean((pred[train]-force[train])**2))),
            continuation_rmse_N=float(np.sqrt(np.mean((pred[test]-force[test])**2)))))
    best=min(values,key=lambda x:x['train_rmse_N'])
    return dict(best_train_fit=best,fast_5ms_fit=values[0],
                train_window_s=[start,split],seen_continuation_window_s=[split,end],
                limitation='Same already-seen trajectory continuation; not unseen validation or a globally identified time constant.')


def diagnostic(a,scene,method,seed):
    t=a['time'];window=(t>=8.2-1e-9)&(t<=8.9+1e-9)
    B=np.asarray(a['meta']['calibration']['force_matrix']);ref=np.asarray(a['meta']['calibration']['reference_alpha'])
    z=latent_coordinates(a);model=z@B.T-ref@B.T
    # Under active contacts this equals the old observer's filtered force bias.
    bias=a['estimated_normal']-model
    row=dict(scene=scene,method=method,seed=seed,static_window_s=[8.2,8.9],
         command_range=np.ptp(a['alpha'][window],axis=0),
         command_path=np.abs(np.diff(a['alpha'][window],axis=0)).sum(axis=0),
         encoder_range=np.ptp(a['joint'][window],axis=0))
    u=np.diff(a['alpha'][window][::4],axis=0)
    singular=np.linalg.svd(u,compute_uv=False)
    row['observed_input_singular_values']=singular
    row['observed_input_rank']=int(np.sum(singular>max(1e-10,float(singular[0])*1e-6)))
    row['input_rank_is_not_system_controllability']=True
    slopes=causal_slope(t,bias,.12)
    row['bias_drift_rate_p95_abs_Nps']=np.nanquantile(abs(slopes[window]),.95,axis=0)
    row['conditional_input_prediction_errors']={}
    for horizon in (.02,.04,.06):
        step=round(horizon/.005);ids=np.flatnonzero(window & np.all(a['contact_model_active'],axis=1))
        if not len(ids):continue
        fast=(z[ids+step]-z[ids])@B.T
        truth=a['skin_normal'][ids+step]
        base=a['estimated_normal'][ids]+fast
        v=slopes[ids]
        # Calibration/old observer bias filter are public, rather than fit to
        # future target pose, mass, friction, event timing or force outcomes.
        effective_lag=np.mean(np.broadcast_to(a['age'][ids].reshape(len(ids),-1),(len(ids),5)),axis=1)+.10
        row['conditional_input_prediction_errors'][str(horizon)]=dict(
            frozen_fast_model=metrics(base-truth),
            drift_horizon_only=metrics(base+horizon*v-truth),
            drift_with_known_bias_filter_lag=metrics(base+(effective_lag+horizon)[:,None]*v-truth),
            samples=len(ids),known_future_recorded_input=True,
            limitation='Actual future recorded alpha is used ONLY to isolate model residual; this is not a runnable-policy forecast evaluation.')
    if scene=='nominal' and method=='unified':
        hold=(t>=8.2-1e-9)&(t<12.-1e-9);i,j=np.flatnonzero(hold)[[0,-1]]
        changes=np.flatnonzero((t>=7.5)&(t<12)&np.any(abs(a['executed_action'])>1e-12,axis=1))
        row['constant_command_segment']=dict(start_s=float(t[i]),end_s=float(t[j]),
            last_nonzero_command_s=float(t[changes[-1]]),alpha_range=np.ptp(a['alpha'][hold],axis=0),
            ctrl_range=np.ptp(a['ctrl'][hold],axis=0),pressure_change_N=a['skin_normal'][j]-a['skin_normal'][i],
            encoder_change=a['joint'][j]-a['joint'][i],raw_joint_change=a['raw_joint'][j]-a['raw_joint'][i],
            relative_object_change_m=a['relative'][j]-a['relative'][i])
        row['little_finger_slow_fit']=slow_fit(t,a['skin_normal'][:,4])
        row['constant_command_predictions']={}
        select=(t>=8.2-1e-9)&(t<=11.5+1e-9)
        for horizon in (.02,.06,.2):
            step=round(horizon/.005);ids=np.flatnonzero(select);truth=a['skin_normal'][ids+step];n=a['estimated_normal'][ids];v=slopes[ids]
            row['constant_command_predictions'][str(horizon)]=dict(
                fast_persistence=metrics(n-truth),
                drift_horizon_only=metrics(n+horizon*v-truth),
                drift_with_known_bias_filter_lag=metrics(n+(.11+horizon)*v-truth))
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--out',type=Path,default=ROOT/'model_data/seen_dynamics_v1.json')
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(530101,530109)));args=p.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Identification outputs must stay inside the new research root')
    if out.exists():raise FileExistsError(out)
    records=[];inputs=[]
    for scene in ('nominal','contact_robustness','sphere'):
        shape='sphere' if scene=='sphere' else 'capsule'
        for method in ('ratio','unified'):
            for seed in args.seeds:
                path=args.source_root/'output/confirmation_integration_v2/runs'/f'{scene}__{shape}__{method}__{seed}'/'trace.npz'
                a=load(path);records.append(diagnostic(a,scene,method,seed))
                meta_path=path.with_name('metadata.json');meta=json.loads(meta_path.read_text(encoding='utf-8'))
                inputs.append(dict(path=str(path.resolve()),bytes=path.stat().st_size,
                    archived_trace_sha256=meta['trace_sha256'],metadata_sha256=digest(meta_path),
                    trace_identity_basis='Previously independently audited frozen archive; no whole old-project rehash.'))
            print(scene,method,'diagnostics complete',flush=True)
    result=dict(schema='seen_full_hand_dynamics_identification_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        seen_disclosure=dict(old_confirmation_used_for_development=True,new_policy_performance_claim=False,
            independent_new_confirmation_required=True,selection='Three diagnostic cases and both frozen methods, all eight old seeds.',
            trajectories_are_closed_loop_and_not_independent_open_loop_excitation=True),
        source_script_sha256=digest(__file__),inputs=inputs,records=records,
        proposed_public_predictor=dict(model='Signed fast actuation response plus a delay-aligned slow force-bias/rate state',
            fast_public_tau_s=.005,bias_filter_tau_s=.10,causal_rate_window_s=.12,
            inputs=['skin pressure / shared causal state.normal','state.force_bias or B/latent-alpha residual','actual alpha history','contact validity / age','public motion reference'],
            forbidden_controller_inputs=['target identity','object mass/friction truth','object global pose truth','future disturbance schedule'],
            equations=['b_measured(t-age)=pressure(t-age)-B*latent_alpha(t-age)',
                'n_hat(t+h)=n_hat(t)+fast_action_effect(h,u)+(age+bias_filter_tau+h)*drift_rate',
                'A native [bias,bias_rate] delayed-measurement observer can replace the explicit lag extrapolation.'],
            motion_gain_identified=False,
            motion_gain_N_per_unit=np.zeros((5,3)),motion_gain_uncertainty=None,
            motion_model_note='Do not interpret zero placeholder gain as zero uncertainty; no independent 5x3 causal motion model is certified by these closed-loop traces.',
            force_safe_set='Use one-sided real/local action bounds, complete signed B, current slow drift, and model-error intervals. Per-row pressure and total pressure remain constraints; saturation is not evidence of physical smoothness.',
            limits=['Observed input rank is a data-excitation limit, not the rank of the calibrated plant.',
                    'No safety guarantee or new-policy success rate follows from fits to already-seen old traces.',
                    'Contact changes require increased uncertainty/reset; do not explain feedback oscillation as an autonomous drift input.']))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(serial(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(output=str(out),records=len(records),sha256=digest(out)),ensure_ascii=False))


if __name__=='__main__':main()

"""Offline A/B vs frozen full-history reference, same120mixed held-out traces.

These traces have already been inspected/used for model development: this is
model-selection/development evidence, never an independent confirmation.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('servo_short',HERE/'predictor.py');predictor=importlib.util.module_from_spec(spec);spec.loader.exec_module(predictor)
spec=importlib.util.spec_from_file_location('frozen_robot_identify',ROOT/'identification/robot_servo_response_v1/identify.py');ident=importlib.util.module_from_spec(spec);spec.loader.exec_module(ident)
MODELS=ROOT/'output/robot_servo_response_all15_v2'
DATA=ROOT/'calibration/phase3_path_validation_v3'


def evaluate_context(shape,index,out):
    cid=f'{shape}/checkpoint_{index:02d}';model_path=MODELS/cid/'model.json';m=ident.read_json(model_path)
    if m['global_model_admitted'] or m['context_id']!=cid:raise ValueError('Frozen local model scope mismatch')
    if ident.sha(m['dense_kernel_npz_path'])!=m['dense_kernel_npz_sha256'] or ident.sha(m['source_manifest_path'])!=m['source_manifest_sha256']:raise ValueError('Frozen model input hash mismatch')
    manifest=ident.read_json(m['source_manifest_path']);context=ident.mechanical_context(DATA,shape)
    if context['axes']!=m['output_axes25'] or not np.array_equal(context['qadr'],m['output_qpos_addresses25']):raise ValueError('Robot axis/address mismatch')
    with np.load(m['dense_kernel_npz_path'],allow_pickle=False) as f:a={k:f[k] for k in f.files}
    K=a['response25x8'];saved={};sources=[];rows=[]
    pooled={str(h):{label:[] for label in ('A','B','full_reference','common_A','common_B','common_full_reference')} for h in predictor.HORIZONS}
    unavailable={str(h):dict(A=0,B=0,reasons={}) for h in predictor.HORIZONS}
    for sequence in range(8):
        branch=f'mixed_{sequence:02d}';source=manifest['branches'][branch];path=Path(source['trace_path']);before=ident.sha(path)
        if before!=source['trace_sha256']:raise ValueError('Raw mixed source hash mismatch')
        # Select robot25 immediately. Object7/velocity6/obstacle never enter any
        # finite check, predictor, scored output or exported selected arrays.
        with np.load(path,allow_pickle=False) as f:
            q=f['qpos'][:,context['qadr']].copy();v=f['qvel'][:,context['dadr']].copy()
            t=f['time'].copy();u=f['input_reference'].copy();ctrl=f['ctrl'].copy()
        if ident.sha(path)!=before:raise ValueError('Input changed during offline read')
        t=t-t[0]
        if q.shape!=(101,25) or v.shape!=(101,25) or np.max(abs(np.diff(t)-.005))>1e-9 or not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('Complete finite robot q/qvel end-clock required')
        if not np.array_equal(q,a[branch+'_actual_q25']) or not np.array_equal(u,a[branch+'_actual_reference8']):raise ValueError('Raw/validated model trace identity mismatch')
        if np.max(abs(ctrl[:,context['aids']]-ident.decode_ctrl(context,u)))>1e-12:raise ValueError('Actual command does not match mechanical reference')
        sources.append(dict(**source,input_unchanged=True))
        saved[branch+'_robot_q25']=q;saved[branch+'_robot_velocity25']=v;saved[branch+'_input_reference8']=u;saved[branch+'_time_s']=t
        by_h={}
        for h in predictor.HORIZONS:
            key=str(h);prefix=branch+'_h'+str(round(h*1000));origins=a[prefix+'_origin_times_s'];target=a[prefix+'_actual25'];reference=a[prefix+'_prediction25']
            predA=[];predB=[];validB=[];freeA=[];freeB=[];effects=[]
            for j,origin in enumerate(origins):
                k=round(origin/.005);bound=k//4
                if abs(t[k]-origin)>1e-9:raise ValueError('Prediction-origin mismatch')
                ft=t[::4][bound:]-t[k];fu=u[::4][bound:]
                first=predictor.predict_velocity(K,q[k],v[k],ft,fu,[h])
                pa=first['prediction25'][0];predA.append(pa);freeA.append(first['free_motion25'][0]);effects.append(first['future_controlled_response25'][0])
                try:
                    second=predictor.predict_position_ols(K,t[:k+1],q[:k+1],ft,fu,[h]);pb=second['prediction25'][0]
                    predB.append(pb);freeB.append(second['free_motion25'][0]);validB.append(True)
                except predictor.PredictionUnavailable as exc:
                    if k!=0:raise
                    reason=str(exc);unavailable[key]['B']+=1;unavailable[key]['reasons'][reason]=unavailable[key]['reasons'].get(reason,0)+1
                    predB.append(np.full(25,np.nan));freeB.append(np.full(25,np.nan));validB.append(False)
            predA=np.stack(predA);predB=np.stack(predB);validB=np.asarray(validB,bool)
            errorA=target-predA;errorB=target[validB]-predB[validB];errorR=target-reference
            pooled[key]['A'].extend(errorA);pooled[key]['B'].extend(errorB);pooled[key]['full_reference'].extend(errorR)
            pooled[key]['common_A'].extend(errorA[validB]);pooled[key]['common_B'].extend(errorB);pooled[key]['common_full_reference'].extend(errorR[validB])
            by_h[key]=dict(A=ident.score(errorA),B=ident.score(errorB),full_reference=ident.score(errorR),
                common_samples=int(validB.sum()),common_A=ident.score(errorA[validB]),common_B=ident.score(errorB),common_full_reference=ident.score(errorR[validB]))
            for tag,value in dict(origin_times_s=origins,actual25=target,A_prediction25=predA,B_prediction25=predB,B_available=validB,full_reference_prediction25=reference,
                A_free_motion25=np.stack(freeA),B_free_motion25=np.stack(freeB),future_input_response25=np.stack(effects)).items():saved[prefix+'_'+tag]=value
        rows.append(dict(branch=branch,metrics_by_horizon=by_h))
    target=out/cid;target.mkdir(parents=True)
    np.savez_compressed(target/'predictions.npz',**saved)
    result=dict(schema='robot_servo_short_horizon_comparison_v1',context_id=cid,output_axes25=context['axes'],output_units=['rad']*24+['m'],
        robot_qpos_addresses25=context['qadr'],robot_qvel_addresses25=context['dadr'],free_object_and_obstacle_excluded=True,
        model_path=str(model_path),model_sha256=ident.sha(model_path),dense_kernel_sha256=m['dense_kernel_npz_sha256'],raw_sources=sources,
        horizons_s=predictor.HORIZONS,metrics_by_horizon={h:{label:ident.score(value) for label,value in scores.items()} for h,scores in pooled.items()},
        unavailable=unavailable,branch_metrics=rows,predictions_npz_path=str(target/'predictions.npz'),predictions_npz_sha256=ident.sha(target/'predictions.npz'),
        gain_fit=False,window_selected=False,position_window_s=.12,main_horizon_s=.02,
        model_selection_scope='previously used phase3 mixed sequences; development diagnostic, not independent confirmation',
        full_history_reference_scope='frozen causal forecast using actual20ms-ramp history, provided only as comparison, not asserted equivalent to A/B',
        finite_ramp_scope='future commands use measured20ms interval increment/four5msramp; arbitrary past5ms commands are nuisance, not reconstructed as fake20ms ramps',
        uncertainty_scope='empirical per-axis errors; no total safety bound, no pressure gate borrowed',
        sequence_support='recorded held-out sequences only; not arbitrary8D combination or novel contact guarantee',global_model_admitted=False,new_physics=False)
    ident.write_json(target/'comparison.json',result)
    return dict(context_id=cid,path=str(target/'comparison.json'),sha256=ident.sha(target/'comparison.json'),metrics_by_horizon=result['metrics_by_horizon'],unavailable=unavailable)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shape',default='capsule',choices=['capsule','box','sphere']);p.add_argument('--checkpoint',default=0,type=int);p.add_argument('--all',action='store_true');p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    ledger=dict(schema='robot_servo_short_horizon_run_v1',complete=False,started_utc=datetime.now(timezone.utc).isoformat(),contexts=[],failures=[],new_physics=False,global_model_admitted=False)
    deps=[HERE/'predictor.py',Path(__file__).resolve(),ROOT/'identification/robot_servo_response_v1/identify.py',ROOT/'identification/robot_servo_response_v1/causal_forecast.py',MODELS/'execution_complete.json',MODELS/'source_binding.json']
    before={str(x):ident.sha(x) for x in deps}
    try:
        ident.write_json(args.out/'plan.json',dict(schema='robot_servo_short_horizon_plan_v1',frozen_before_this_comparison=True,
            candidate_A='q_now+h*actual_robot_velocity_now+sum_future_measuredK*declared_future_delta',
            candidate_B='fixed.12s raw robotq OLS extrapolation+sum_future_measuredK*declared_future_delta',
            B_initial_rank_failure='unavailable if fewer than2past observations; never substitute a constant and call it OLS',
            main_horizon_s=.02,diagnostic_horizons_s=[.04,.06,.1],gains_or_window_refit=False,dependencies=before,
            selection_data_seen=True,no_physics=True,model_admitted=False))
        for path in deps[:2]:shutil.copyfile(path,args.out/path.name)
        contexts=[(s,i) for s in ('capsule','box','sphere') for i in range(5)] if args.all else [(args.shape,args.checkpoint)]
        for shape,index in contexts:
            r=evaluate_context(shape,index,args.out);ledger['contexts'].append(r)
            print(json.dumps(dict(context=r['context_id'],path=r['path'],sha256=r['sha256'])),flush=True)
        after={str(x):ident.sha(x) for x in deps}
        if after!=before:raise ValueError('Source/dependency changed during offline run')
        ledger['dependencies_unchanged']=True;ledger['complete']=True
    except Exception as exc:
        ledger['failures'].append(dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
    finally:
        ledger['finished_utc']=datetime.now(timezone.utc).isoformat();ident.write_json(args.out/'execution_complete.json',ledger)


if __name__=='__main__':main()

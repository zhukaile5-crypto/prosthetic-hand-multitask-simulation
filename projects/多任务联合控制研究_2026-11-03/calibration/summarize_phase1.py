"""Offline dual-frame export/audit of the public synergy calibration.

Does not run a model, controller or dynamics step. All three unmeasured motion
columns remain absent. Admission is intentionally separate from execution.
"""
from __future__ import annotations
import argparse,json,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
from sensors import aggregate_patches


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def measure(error):
    return {name:dict(rmse=float(np.sqrt(np.mean(error[...,sl]**2))),
        p95_abs=float(np.quantile(abs(error[...,sl]),.95)),max_abs=float(np.max(abs(error[...,sl]))),unit=unit)
        for name,sl,unit in [('pressure',slice(0,5),'N'),('net_force',slice(5,8),'N'),('net_moment',slice(8,11),'Nm')]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'calibration/phase1_synergy_v1');p.add_argument('--out',type=Path,default=ROOT/'calibration/public_model_phase1_v1.json');args=p.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(ROOT/'calibration'):raise ValueError('New calibration output only')
    if out.exists():raise FileExistsError(out)
    run=read(args.root/'summary.json');plan=read(args.root/'plan.json');all_shapes=[];snapshots=[];execution=True
    for case in plan['shapes']:
        folder=args.root/case;base=read(folder/'base_metadata.json');model=read(folder/'kernel_model.json');times=np.asarray(model['horizons_s']);frames={};metas={};finite=True;pressure_peak=np.zeros(5);sum_peak=0.;penetration=0.;minimum_lift=float('inf');table_peak=0.
        for path in sorted(folder.glob('*/metadata.json')):
            meta=read(path);metas[path.parent.name]=meta
            if not meta['available']:continue
            with np.load(path.with_name('trace.npz'),allow_pickle=False) as z:
                keys=['time','pressure','wrench_palm','wrench_reference','normal_reference','tangent_reference','auxiliary_reference','physical_y','physical_qpos','physical_qvel','physical_ctrl','physical_alpha','physical_object_lift_m','physical_table_support_N','physical_penetration']
                a={k:z[k] for k in keys};frames[path.parent.name]=a
            finite &= all(np.isfinite(v).all() for v in a.values())
            pressure_peak=np.maximum(pressure_peak,np.max(a['physical_y'][:,:5],axis=0));sum_peak=max(sum_peak,float(a['physical_y'][:,:5].sum(axis=1).max()));penetration=max(penetration,float(a['physical_penetration'].max()));minimum_lift=min(minimum_lift,float(a['physical_object_lift_m'].min()));table_peak=max(table_peak,float(a['physical_table_support_N'].max()))
            snapshots.append(dict(path=str(path.with_name('trace.npz').resolve()),sha256=meta['trace_sha256']))
        restore_equal={k:bool(np.array_equal(v,frames['zero_replay'][k])) for k,v in frames['zero'].items()}
        execution &= finite and all(restore_equal.values()) and all(m['complete'] for m in metas.values() if m['available'])
        indexes=np.round(times/.005).astype(int);kernels={}
        for frame,fields in [('current_palm',['pressure','wrench_palm']),('initial_fixed_reference',['pressure','wrench_reference'])]:
            Y0=np.c_[frames['zero'][fields[0]][indexes],frames['zero'][fields[1]][indexes]];J=np.zeros((len(times),11,5))
            for axis in range(5):
                valid=[s for s in ('plus','minus') if metas[f'train_{axis}_{s}']['available']]
                if len(valid)==2:
                    labels=[f'train_{axis}_{s}' for s in ('plus','minus')];yp,yn=[np.c_[frames[x][fields[0]][indexes],frames[x][fields[1]][indexes]] for x in labels]
                    denominator=float(metas[labels[0]]['executed_delta'][axis]-metas[labels[1]]['executed_delta'][axis]);J[:,:,axis]=(yp-yn)/denominator
                elif valid:
                    label=f'train_{axis}_{valid[0]}';Y=np.c_[frames[label][fields[0]][indexes],frames[label][fields[1]][indexes]];J[:,:,axis]=(Y-Y0)/float(metas[label]['executed_delta'][axis])
                else:raise RuntimeError('Cannot export an unobserved input as a zero column')
            difference=[];causal=[];individual=[]
            for axis in range(5):
                for sign,label in ((1,'plus'),(-1,'minus')):
                    key=f'hold_{axis}_{label}';meta=metas[key]
                    if not meta['available']:individual.append(dict(axis=axis,sign=sign,available=False));continue
                    actual=np.c_[frames[key][fields[0]][indexes],frames[key][fields[1]][indexes]];response=np.einsum('hoi,i->ho',J,np.asarray(meta['executed_delta']))
                    start=np.r_[frames['zero'][fields[0]][0],frames['zero'][fields[1]][0]]
                    baseline=start[None,:]+times[:,None]*np.asarray(model['causal_history_slope'])[None,:] if frame=='initial_fixed_reference' else np.tile(start,(len(times),1))
                    de=actual-Y0-response;ce=actual-baseline-response;difference.append(de);causal.append(ce)
                    individual.append(dict(axis=axis,sign=sign,available=True,executed_delta=meta['executed_delta'],difference_error=de,complete_causal_error=ce))
            scaled=J*np.array([.2]*5+[1.]*3+[20.]*3)[None,:,None]*.004
            kernels[frame]=dict(jacobian_11x5=J,scaled_singular_values=[np.linalg.svd(x,compute_uv=False) for x in scaled],
                scaled_numerical_rank=[int(np.sum(s>max(1e-10,s[0]*1e-6))) for s in [np.linalg.svd(x,compute_uv=False) for x in scaled]],
                causal_baseline='start public output plus past .25s trend' if frame=='initial_fixed_reference' else 'start public output only, constant baseline; no future frame pose or zero trajectory',
                holdout_by_horizon=[dict(horizon_s=float(h),response_difference=measure(np.array(difference)[:,i]),complete_causal_prediction=measure(np.array(causal)[:,i])) for i,h in enumerate(times)],individual_holdout=individual)
        with np.load(folder/'zero/trace.npz',allow_pickle=False) as z:
            raw=z['raw_patches'];_,diag=aggregate_patches(json.loads(str(raw[0])))
        W=np.asarray(diag['normal_wrench_map']);reference_J=np.asarray(kernels['initial_fixed_reference']['jacobian_11x5']);naive=np.einsum('oi,hij->hoj',W,reference_J[:,:5])
        comparison=[dict(horizon_s=float(h),true_net_force_J_norm=float(np.linalg.norm(reference_J[i,5:8])),normal_only_J_norm=float(np.linalg.norm(naive[i,:3])),true_net_moment_J_norm=float(np.linalg.norm(reference_J[i,8:11])),normal_only_moment_J_norm=float(np.linalg.norm(naive[i,3:]))) for i,h in enumerate(times)]
        all_shapes.append(dict(shape=case,execution_complete=all(m['complete'] for m in metas.values() if m['available']),
            planned_branches=len(metas),available_branches=sum(m['available'] for m in metas.values()),unavailable=[k for k,m in metas.items() if not m['available']],input_availability=model['input_availability'],
            complete_state_restore_replay_equal=restore_equal,finite_recorded_states=finite,pressure_peak_N=pressure_peak,sum_finger_pressure_peak_N=sum_peak,penetration_peak_m=penetration,minimum_object_lift_m=minimum_lift,table_peak_N=table_peak,
            coordinate_frames=dict(current_palm='Each sample own palm origin/axes, as the existing tactile API.',initial_fixed_reference=dict(origin_world_m=base['reference_origin'],rotation_world=base['reference_rotation'],scope='Offline diagnostics; runtime needs public hand FK for absolute conversion, never target pose.')),
            output_order=['skin_pressure_0','skin_pressure_1','skin_pressure_2','skin_pressure_3','skin_pressure_4','net_Fx','net_Fy','net_Fz','net_Mx','net_My','net_Mz'],output_units=['N']*8+['Nm']*3,input_order=['alpha0','alpha1','alpha2','alpha3','alpha4'],input_units=['dimensionless_synergy']*5,horizons_s=times,kernels=kernels,normal_only_approximation_comparison=comparison))
    source_equal={k:sha(Path(k) if Path(k).is_absolute() else ROOT/k)==v for k,v in plan['source_sha256'].items()}
    result=dict(schema='public_partial_pressure_netwrench_model_phase1_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        execution_complete=bool(run['complete'] and execution),model_admitted=False,identification_status='partial_development_identification',
        not_admitted_reasons=['No predeclared model-admission error gates were applied here.','Motion input columns are unmeasured, and sphere index positive input is unavailable.','Only public prepared poses and one payload/contact setup; no extrapolation to release or changed contact family.','No absolute palm-world pose field exists in current SensorFrame; fixed-reference kernel is not directly an online current-palm predictor.'],
        comparison_not_performance=True,source_script_sha256=sha(__file__),calibration_plan_sha256=sha(args.root/'plan.json'),frozen_calibration_sources_unchanged=source_equal,
        raw_archived_files=snapshots,shapes=all_shapes,missing_inputs=['lift_m','wrist1_rad','wrist2_rad'],
        interpretation='Zero-paired kernel error and complete causal output error are separate. Unavailable directions are not physics failures or fictitious zero responses. Net force and moment are never pooled into an unscaled scalar RMSE.')
    out.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(out),sha256=sha(out),execution_complete=result['execution_complete'],model_admitted=False),ensure_ascii=False))


if __name__=='__main__':main()

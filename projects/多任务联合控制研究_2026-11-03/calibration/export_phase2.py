"""Read-only export of phase-2 evidence and dense public response kernels."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    return x


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'calibration/phase2_public_v1');p.add_argument('--out',type=Path,default=ROOT/'calibration/public_model_phase2_v1.json');a=p.parse_args();out=a.out.resolve()
    if out.exists():raise FileExistsError(out)
    if not out.is_relative_to(ROOT/'calibration'):raise ValueError('New calibration output only')
    plan=read(a.root/'plan.json');execution=read(a.root/'summary.json');cases=[];manifest=[];all_gates=True
    for shape in plan['shapes']:
        folder=a.root/shape;model=read(folder/'model.json');base=read(folder/'base_metadata.json');files={};metas={};physical_peak=np.zeros(5);sum_peak=0.;penetration=0.;minimum_lift=float('inf');finite=True
        for path in sorted(folder.glob('*/metadata.json')):
            label=path.parent.name;meta=read(path);metas[label]=meta
            if not meta['available']:continue
            with np.load(path.with_name('trace.npz'),allow_pickle=False) as z:
                fields=['y','y_palm','qpos','qvel','ctrl','alpha','input_reference','physical_y','physical_y_palm','physical_qpos','physical_qvel','physical_ctrl','physical_alpha','physical_input_reference','physical_penetration','physical_object_lift_m']
                files[label]={k:z[k] for k in fields}
            f=files[label];finite &= all(np.isfinite(v).all() for v in f.values());physical_peak=np.maximum(physical_peak,np.max(f['physical_y'][:,:5],axis=0));sum_peak=max(sum_peak,float(f['physical_y'][:,:5].sum(axis=1).max()));penetration=max(penetration,float(f['physical_penetration'].max()));minimum_lift=min(minimum_lift,float(f['physical_object_lift_m'].min()))
            manifest.append(dict(path=str(path.with_name('trace.npz').resolve()),sha256=meta['trace_sha256']))
        equality={k:bool(np.array_equal(v,files['zero_replay'][k])) for k,v in files['zero'].items()}
        kernels={};missing=[];time=np.arange(len(files['zero']['y']))*.005
        for frame,key in [('current_palm','y_palm'),('initial_fixed_reference','y')]:
            K=np.zeros((len(time),11,8));identified=np.zeros(8,bool)
            for axis in range(8):
                available=[s for s in ('plus','minus') if metas[f'train_{axis}_{s}']['available']]
                if not available:missing.append(axis);continue
                identified[axis]=True
                if len(available)==2:
                    lp,lm=f'train_{axis}_plus',f'train_{axis}_minus';dp=np.asarray(metas[lp]['requested_path'])[-1,axis]-np.asarray(metas[lm]['requested_path'])[-1,axis];K[:,:,axis]=(files[lp][key]-files[lm][key])/dp
                else:
                    label=f'train_{axis}_{available[0]}';dp=np.asarray(metas[label]['requested_path'])[-1,axis]-base['base'][axis];K[:,:,axis]=(files[label][key]-files['zero'][key])/dp
            dense=K.tolist()
            for axis in np.flatnonzero(~identified):
                for block in dense:
                    for output in block:output[axis]=None
            review=model['coordinates'][frame];holdout=review['complete_causal_by_horizon'];passed=all(v['all_engineering_gates_passed'] for v in holdout.values()) and review['differential_response_error']['all_engineering_gates_passed'];all_gates &= passed
            U=np.vstack([np.diff(f['input_reference'][::4],axis=0) for label,f in files.items() if label.startswith('holdout_')]);scale=np.asarray(plan['holdout_amplitude']);sv=np.linalg.svd(U/scale[None,:],compute_uv=False)
            kernels[frame]=dict(response_kernel_times_s=time,response_to_20ms_interval_increment=dense,
                coefficient_units=['N/input_unit']*8+['Nm/input_unit']*3,
                dense_sampling_note='5ms physical public outputs are stored; forecast engineering gates were evaluated only at 20/40/60/100ms, not every dense time.',
                interval_execution='A planned 20ms increment is applied as four equal 5ms reference increments.',
                holdout_complete_causal_by_horizon=holdout,holdout_differential_response=review['differential_response_error'],
                all_predeclared_engineering_gates_passed=bool(passed),identified_inputs=identified,
                holdout_normalized_input_singular_values=sv,holdout_normalized_input_rank=int(np.sum(sv>max(1e-10,sv[0]*1e-6))))
        paths=np.vstack([f['input_reference']-np.asarray(base['base'])[None,:] for f in files.values()]);units=['dimensionless_synergy']*5+['m','rad','rad']
        cases.append(dict(shape=shape,execution_complete=all(m['complete'] for m in metas.values() if m['available']),finite_states=finite,
            full_restore_replay_equal=equality,planned_branches=len(metas),executed_branches=len(files),unavailable_branches=[k for k,v in metas.items() if not v['available']],
            input_availability=model['input_availability'],measured_input_delta_min=paths.min(axis=0),measured_input_delta_max=paths.max(axis=0),
            input_order=['alpha_thumb','alpha_index','alpha_middle','alpha_ring','alpha_little','lift_reference','wrist_WRJ2','wrist_WRJ1'],input_units=units,
            output_order=['skin_pressure_thumb','skin_pressure_index','skin_pressure_middle','skin_pressure_ring','skin_pressure_little','net_Fx','net_Fy','net_Fz','net_Mx','net_My','net_Mz'],output_units=['N']*8+['Nm']*3,
            pressure_peak_N=physical_peak,sum_finger_pressure_peak_N=sum_peak,penetration_peak_m=penetration,minimum_object_lift_m=minimum_lift,
            coordinate_reference=dict(origin_world_m=base['reference_origin'],rotation_world=base['reference_rotation'],
                force_sign='Hand applies force to all external contacts. No target identity filtering.',
                current_palm='Each sample own palm origin/axes. Moment is Nm, not divided by characteristic length.',
                initial_fixed_reference='At clone palm origin/axes. Online absolute conversion requires public robot FK, never target pose.'),
            coordinates=kernels,source_model_sha256=sha(folder/'model.json')))
    identity={k:sha(Path(k))==v for k,v in plan['source_sha256'].items()}
    valid_execution=bool(execution['execution_complete'] and all(c['execution_complete'] and c['finite_states'] and all(c['full_restore_replay_equal'].values()) for c in cases))
    result=dict(schema='public_pressure_wrench_mimo_phase2_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        execution_complete=valid_execution,all_predeclared_engineering_gates_passed=bool(all_gates),model_admitted=False,
        admission_status='Independent audit and explicit measured-context admission pending; engineering gates are not statistical or physical safety bounds.',
        calibration_not_performance=True,source_script_sha256=sha(__file__),calibration_plan_sha256=sha(a.root/'plan.json'),
        frozen_executed_sources_unchanged=identity,raw_archive_manifest=manifest,shapes=cases,
        limitations=['Only measured prepared .100m contexts with .12kg setups and recorded micro-input paths.',
            'Positive lift above .100m and sphere-index positive directions were unavailable; bidirectional full authority is not claimed.',
            'Inputs are prescribed planned commands; complete forecast uses only initial/past public residual measurements, not holdout-future zero outputs.',
            'Model measurements use ideal public tactile aggregation; observation noise/delay must be handled by the online estimator.',
            'No extrapolation outside measured input paths, contact modes, payloads or 20/40/60/100ms forecast gates is admitted.',
            'For finite-memory online use, retain the response of past increments or work incrementally from measured current output; do not silently forget permanent effects when a history item leaves the kernel window.'])
    out.write_text(json.dumps(serial(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(path=str(out),sha256=sha(out),execution_complete=valid_execution,engineering_gate_passed=all_gates),ensure_ascii=False))


if __name__=='__main__':main()

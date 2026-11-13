"""Saved-data20ms robot-response error at actually executed V10 interventions.

Only completed cohorts are read. No plant, FK, QP or physics imports. Active
uniform-ramp intervals are eligible; altered realization is recorded separately.
Observed pressure deltas are descriptive, not a force response/causal estimate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def analyze(mode):
    run=ROOT/f'output/development_v10_{mode}_v1'
    finish=read(run/'execution_complete.json')
    if finish.get('execution_complete') is not True:raise ValueError('Complete20trial cohort required')
    source_manifest=read(run/'manifest.json');cache={};rows=[];sources=[]
    for scene in ('nominal','friction_drop','fragile_conflict','collision'):
        path=run/'runs'/f'{scene}__joint__610212'/'trace.npz';meta=read(path.with_name('metadata.json'))
        if sha(path)!=meta['trace_sha256']:raise ValueError('Actual trace hash mismatch')
        names=meta['raw_joint_names'];raw=np.asarray(meta['raw_joint_qpos_indices'],int)
        if len(names)!=24 or any(not n.startswith('rh_') for n in names) or not np.array_equal(raw,np.arange(1,25)):raise ValueError('Known robot24 whitelist/address layout required')
        with np.load(path,allow_pickle=False) as f:
            t=f['time'];q=f['qpos'][:,np.r_[raw,0]];qv=f['qvel'][:,np.r_[raw,0]];alpha_before=f['alpha_before'];alpha=f['alpha'];motion=f['executed_motion_reference'];normal=f['skin_normal'];collider=f['collision'];debug=f['debug_json'];observations=f['observation_json']
        ids=np.flatnonzero((t>=7.5-1e-9)&(t<12.-1e-9)&(abs((t-7.5)/.02-np.rint((t-7.5)/.02))<1e-7))
        intervals=[];errors=[];exceed=[]
        for i in ids:
            d=json.loads(str(debug[i]))
            if d.get('status')!='EXPERIMENTAL_DIRECTIONAL_FINITE_RAMP':continue
            selection=d['servo_selection'];cid=selection['context_id'];fingerprints=selection['source_fingerprints']
            if cid not in cache:
                kernel_path=Path(fingerprints['dense_kernel']);relative=kernel_path.relative_to(ROOT).as_posix()
                if sha(kernel_path)!=fingerprints['dense_kernel_sha256'] or source_manifest['research_source_sha256'][relative]!=fingerprints['dense_kernel_sha256']:raise ValueError('Kernel not frozen with actual execution')
                with np.load(kernel_path,allow_pickle=False) as f:K=f['response25x8'][4]
                cache[cid]=K
                sources.append(dict(path=str(kernel_path),sha256=fingerprints['dense_kernel_sha256']))
            obs=json.loads(str(observations[i]));current=np.r_[obs['raw_joint'],obs['lift']];velocity=np.r_[obs['raw_joint_velocity'],obs['lift_velocity_m_s']]
            if not np.array_equal(current,q[i]) or not np.array_equal(velocity,qv[i]):raise ValueError('Public current robot encoder clock/order mismatch')
            increment=np.asarray(d['applied_input_increment_8']);start=np.r_[alpha_before[i],obs['executed_motion_reference']]
            actual_path=np.column_stack((alpha[i:i+4],motion[i:i+4]));expected=start+np.arange(1,5)[:,None]/4*increment
            ramp_error=float(np.max(abs(actual_path-expected)));eligible=ramp_error<=1e-9
            predicted=current+.02*velocity+cache[cid]@increment;actual=q[i+4];error=actual-predicted;reference_error=np.asarray(d['allocation']['empirical_robot_error25'])
            if eligible:errors.append(error);exceed.append(abs(error)>reference_error+1e-12)
            intervals.append(dict(start_s=float(t[i]),end_s=float(t[i+4]),context_id=cid,eligible_uniform_ramp=eligible,ramp_max_abs_error=ramp_error,
                current_robot_q25=current.tolist(),current_robot_velocity25=velocity.tolist(),declared_increment8=increment.tolist(),predicted_robot_q25=predicted.tolist(),actual_robot_q25=actual.tolist(),robot_error25=error.tolist(),
                empirical_anchor_A20max_error25=reference_error.tolist(),exceeds_empirical_anchor_error25=(abs(error)>reference_error+1e-12).tolist(),
                robot_state_distance=selection['robot_state_distance'],reference_outside_component_footprint=selection['reference_outside_component_footprint'],
                physical_collider_pressure_before_N=float(collider[i]),physical_skin_pressure_before_N=normal[i].tolist(),physical_skin_pressure_after_N=normal[i+4].tolist(),
                model_geometric_cost_before=d.get('exact_model_geometry_cost_before'),model_geometric_cost_after=d.get('exact_model_geometry_cost_after'),
                force_pause_replaced=d.get('stabilizer_pause_replaced_by_directional_plan',False)))
        summary=None
        if errors:
            e=np.stack(errors);over=np.stack(exceed)
            summary=dict(eligible_intervals=len(e),rmse_per_axis25=np.sqrt(np.mean(e*e,axis=0)).tolist(),max_abs_per_axis25=np.max(abs(e),axis=0).tolist(),
                exceeds_anchor_empirical_error_counts25=over.sum(0).tolist(),any_axis_exceeds_anchor_empirical_error_intervals=int(over.any(1).sum()),
                angular_max_abs_rad=float(np.max(abs(e[:,:24]))),lift_max_abs_m=float(np.max(abs(e[:,24]))))
        rows.append(dict(motion_mode=mode,scenario=scene,active_intervals=len(intervals),model_error=summary,intervals=intervals,trace_path=str(path),trace_sha256=sha(path)))
    return dict(mode=mode,records=rows,kernel_sources=sources,source_ledger_sha256=sha(run/'execution_complete.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--modes',nargs='+',choices=['hold','transport'],default=['hold','transport']);p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    if args.out.exists():raise FileExistsError('Use a fresh review output')
    results=[analyze(mode) for mode in args.modes]
    result=dict(schema='v10_actual_active_servo_error_v1',new_physics=False,source_sha256=sha(__file__),modes=results,
        interpretation='Actual experimental closed-loop interventions, conditional comparison to holding-calibrated A20 empirical error. Exceedance is evidence of model transfer error, not a violation of a certified bound. No futurezero/force model or new gain fitting. Pressure changes are descriptive and not isolated causal effects.')
    with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(args.out),sha256=sha(args.out),records=[{k:r[k] for k in ('motion_mode','scenario','active_intervals','model_error')} for mode in results for r in mode['records']]),ensure_ascii=False,indent=2))

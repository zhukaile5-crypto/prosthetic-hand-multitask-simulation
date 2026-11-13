"""Public-stream/API verification only. Does not construct or step a plant."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from coordinator import create_joint,self_test
from task_trajectory import public_transport_trajectory,constant_trajectory


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate():
    names=('coordinator.py','coordinator_model.py','coordinator_contacts.py','grasp_contacts.py',
        'task_trajectory.py','calibration/public_contact_model_phase2_v1.json')
    identity={n:sha(ROOT/n) for n in names};report=self_test()
    jobs=[('development_v5_pilot',scene,610201) for scene in
        ('nominal','sphere','sensor_noise','dynamic_transport','fragile_conflict','contact_robustness')]
    jobs += [('development_v5_physical_all16',scene,610212) for scene in ('nominal','collision')]
    rows=[]
    for dataset,scene,seed in jobs:
        directory=ROOT/'output'/dataset/'runs'/f'{scene}__joint__{seed}'
        files={name:sha(directory/name) for name in ('trace.npz','metadata.json','result.json')}
        meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        result=json.loads((directory/'result.json').read_text(encoding='utf-8'));assert result['record_complete']
        trajectory=public_transport_trajectory(meta['config']['lift_target']) if scene=='dynamic_transport' else constant_trajectory(meta['config']['lift_target'])
        ctl=create_joint(meta['calibration'],meta['limits'],meta['budget']);records=[];ticks=0
        noise=meta['noise'];spec=dict(force_sigma_N=noise['force_sigma'],tangential_sigma_N=noise['tangential_sigma'],
            position_sigma_m=noise['position_sigma'],slip_sigma_m_s=noise['slip_sigma'],effort_sigma=noise['effort_sigma'],tactile_delay_s=noise['delay'])
        with np.load(directory/'trace.npz',allow_pickle=False) as archive:
            # Compressed NPZ members must be decompressed once, not per tick.
            times=archive['time'];observations=archive['observation_json']
            targets=archive['actual_motion_target'];executed_references=archive['executed_motion_reference']
            for k in np.flatnonzero((times>=7.3-1e-9)&(times<=11.7+1e-9)):
                f=SimpleNamespace(**json.loads(str(observations[k])))
                if f.time<7.5-1e-9:ctl.state.update(f,.005);continue
                ticks+=1;motion=targets[k];executed=executed_references[k-1]
                ctl.set_task_request(dict(time_s=f.time,mode='transport' if scene=='dynamic_transport' else 'hold',
                    motion_target=motion,executed_motion_reference=executed,tracking_tolerances=[.01,.015,.015],
                    limits=meta['limits'],budget=meta['budget'],deadline_s=12.,sensor_characteristics=spec,
                    motion_reference_bounds=dict(lower=[0.,-.523599,-.698132],upper=[meta['config']['lift_target'],.174533,.488692]),
                    motion_preview=trajectory.preview(f.time)))
                out=ctl.step(f);d=out.debug
                assert np.isfinite(out.delta_closure).all() and np.isfinite(ctl.motion_reference).all()
                assert np.max(out.delta_closure)<=ctl.g.close_rate*.005+1e-8
                assert np.min(out.delta_closure)>=-ctl.g.open_rate*.005-1e-8
                alpha=np.asarray(f.alpha)
                assert np.all(alpha+out.delta_closure>=ctl.state.closure_min-1e-8)
                assert np.all(alpha+out.delta_closure<=ctl.state.closure_max+1e-8)
                available=bool(d.get('prediction_available'))
                if available:
                    assert np.isfinite(out.predicted_normal).all()
                    assert d['predicted_pressure_min_N']>=-1e-8 and d['predicted_shear_min_N']>=-1e-8
                    assert d['final_priority_residual']<=2e-5 and d['final_authority_residual']<=1e-8
                    assert not d['model_admitted'] and not d['cumulative_model_domain_certified']
                else:assert np.isnan(out.predicted_normal).all()
                if abs(ctl.last_decision-f.time)>1e-8:continue
                records.append(dict(time=f.time,prediction_available=available,label=ctl.label,
                    nonzero_alpha=bool(np.max(abs(ctl.interval_action))>1e-8),
                    model_unknown_reason=d.get('prediction_invalid_reason'),
                    fallback=d.get('fallback'),qp_status=d.get('qp_status'),
                    persistent_mask=d.get('grasp_persistent_mask'),new_contact_mask=d.get('new_external_contact_mask'),
                    nominal_preserved=d.get('nominal_task_preserved'),
                    force_quality_target=d.get('force_quality_base_target_N'),
                    integral=d.get('integral'),antiwindup=d.get('anti_windup_actual_realization')))
        assert files=={name:sha(directory/name) for name in files}
        rows.append(dict(dataset=dataset,scenario=scene,seed=seed,decision_window_s=[7.5,11.7],
            decisions=len(records),sensor_ticks=ticks,alpha_feedback_decisions=sum(r['nonzero_alpha'] for r in records),
            prediction_available_decisions=sum(r['prediction_available'] for r in records),
            model_unknown_reasons=dict(Counter(r['model_unknown_reason'] for r in records if r['model_unknown_reason'])),
            inputs_sha256=files,inputs_unchanged=True,records=records,
            interpretation='Alternative commands on fixed old observations; actual input history and anti-windup use the old executed path. This is not a closed-loop counterfactual or performance estimate.'))
        print(json.dumps(dict(completed_public_stream=scene,seed=seed,decisions=len(records))),flush=True)
    assert identity=={n:sha(ROOT/n) for n in names}
    report.update(schema='coordinator_v6_algebra_saved_stream_validation_v2',sources=identity,validator_sha256=sha(Path(__file__)),
        saved_public_streams=rows,sources_unchanged=True,no_new_physics=True,
        closed_loop_performance_claim=False)
    return report


if __name__=='__main__':
    output=ROOT/'review/coordinator_v6_validation_v2.json'
    if output.exists():raise FileExistsError('Validation evidence cannot overwrite a previous run')
    report=validate()
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=report['passed'],checks=len(report['checks']),sources=report['sources'],
        rows=[{key:row[key] for key in ('scenario','seed','decisions','alpha_feedback_decisions','prediction_available_decisions','model_unknown_reasons')} for row in report['saved_public_streams']],
        artifact=str(output)),ensure_ascii=False,indent=2))

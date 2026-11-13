"""Algebra and saved-public-stream interface verification; no new physics."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib,xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coordinator import create_joint,self_test


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def validate():
    names=('coordinator.py','coordinator_model.py','coordinator_contacts.py','calibration/public_contact_model_phase2_v1.json')
    sources={name:digest(ROOT/name) for name in names};report=self_test()
    report.update(schema='v5_minimum_task_intervention_algebra_and_public_replay_v1',source_sha256=sources,
        role='Development implementation/interface evidence only; no feedback into saved physical trajectory',closed_loop_performance_claim=False)
    foundation=ROOT.parent/'五指策略接入_2026-11-03'
    calibration=json.loads((foundation/'review/foundation_calibration_final.json').read_text(encoding='utf-8'))
    selected={shape:create_joint(value,[5.]*5,18.).model.shape for shape,value in calibration['shapes'].items()}
    assert all(key==value for key,value in selected.items());report['public_shape_selection']=selected
    physical_xml=foundation/'assets/legacy_model.xml';physical=ET.parse(physical_xml).getroot()
    wrist=np.array([np.fromstring(physical.find(f".//default[@class='{kind}']/position").get('ctrlrange'),sep=' ') for kind in ('wrist_y','wrist_x')])
    example=create_joint(calibration['shapes']['capsule'],[5.]*5,18.)
    assert np.array_equal(wrist[:,0],example.motion_lower[1:]) and np.array_equal(wrist[:,1],example.motion_upper[1:])
    report['wrist_bound_source']=dict(path=str(physical_xml),sha256=digest(physical_xml),ctrlrange=wrist.tolist(),matches=True,
        limitation='XML defaults used unchanged by frozen foundation model construction; no new model/dynamics instantiated')
    rows=[]
    for scene in ('nominal','sphere','sensor_noise','dynamic_transport','fragile_conflict','contact_robustness'):
        run=ROOT/'output/development_v4_integrated/runs'/f'{scene}__joint__610103';paths=(run/'trace.npz',run/'metadata.json')
        before={str(p):digest(p) for p in paths};meta=json.loads(paths[1].read_text(encoding='utf-8'));arrays=np.load(paths[0],allow_pickle=False)
        times=arrays['time'];observations=arrays['observation_json'];target=arrays['actual_motion_target'];executed=arrays['executed_motion_reference']
        ctl=create_joint(meta['calibration'],meta['limits'],meta['budget']);records=[];all_finite=True;invalid_count=0
        for k in np.flatnonzero((times>=7.3-1e-9)&(times<=11.7+1e-9)):
            frame=SimpleNamespace(**json.loads(str(observations[k])))
            if frame.time<7.5-1e-9:ctl.state.update(frame,.005);continue
            ctl.set_task_request(dict(schema='public_task_replay',time_s=float(frame.time),
                mode='transport' if np.max(np.abs(target[k]-executed[k-1]))>1e-9 else 'hold',
                motion_target=target[k].tolist(),executed_motion_reference=executed[k-1].tolist(),
                limits=meta['limits'],budget=meta['budget'],deadline_s=12.,tracking_tolerances=[.01,None,None],
                force_scope='external_all_instrumented_fingers',model_domain_transfer='experimental_local_model_not_admitted'))
            action=ctl.step(frame);assert np.isfinite(action.delta_closure).all();all_finite&=bool(np.isfinite(ctl.motion_reference).all())
            available=bool(action.debug.get('prediction_available'))
            assert np.isfinite(action.predicted_normal).all() if available else np.isnan(action.predicted_normal).all()
            invalid_count+=not available
            if frame.time<9.3-1e-9 or abs(ctl.last_decision-frame.time)>1e-8:continue
            debug=action.debug
            preserved=bool(debug.get('nominal_task_preserved'))
            if preserved:
                assert np.allclose(action.delta_closure,0.,atol=1e-12)
                assert np.allclose(ctl.interval_motion,np.asarray(debug['nominal_action_8'])[5:],atol=1e-12)
                assert np.allclose(debug['correction_8'],0.,atol=1e-12)
            if available:
                assert debug['predicted_pressure_min_N']>=-1e-8 and debug['predicted_shear_min_N']>=-1e-8
                assert debug['final_priority_residual']<=2e-5 and debug['final_authority_residual']<=1e-8
                assert debug['model_admitted'] is False and debug['cumulative_model_domain_certified'] is False
            records.append(dict(time=float(frame.time),prediction_available=available,nominal_task_preserved=preserved,
                failed_checks=debug.get('nominal_failed_checks',[]),qp_status=debug.get('qp_status'),
                alpha_nonzero=bool(np.max(np.abs(ctl.interval_action))>1e-6),
                priority_residual=debug.get('final_priority_residual'),authority_residual=debug.get('final_authority_residual')))
        after={str(p):digest(p) for p in paths};assert before==after
        rows.append(dict(scene=scene,shape=meta['shape'],decision_window_s=[9.3,11.7],decisions=len(records),
            preserved_decisions=sum(x['nominal_task_preserved'] for x in records),nonzero_alpha_decisions=sum(x['alpha_nonzero'] for x in records),
            prediction_available_decisions=sum(x['prediction_available'] for x in records),all_executable_commands_finite=all_finite,
            unavailable_sensor_ticks_7_5_to_11_7=invalid_count,inputs_sha256=before,inputs_unchanged=True,records=records,
            interpretation='These are alternative commands on an old public stream. They are not physical counterfactual outcomes.'))
    report['public_stream_replay']=rows
    assert sources=={name:digest(ROOT/name) for name in names};report['sources_unchanged']=True
    assert digest(physical_xml)==report['wrist_bound_source']['sha256']
    return report


if __name__=='__main__':
    target=ROOT/'review/coordinator_v5_validation_v1.json'
    if target.exists():raise FileExistsError(target)
    report=validate();target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(report['checks']),source_sha256=report['source_sha256'],
        scenes=[{key:row[key] for key in ('scene','decisions','preserved_decisions','nonzero_alpha_decisions','prediction_available_decisions')} for row in report['public_stream_replay']],
        artifact=str(target)),ensure_ascii=False,indent=2))

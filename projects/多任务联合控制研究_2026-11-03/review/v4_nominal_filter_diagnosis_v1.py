"""Read-only algebra on completed V4 public streams; no physical stepping."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coordinator import create_joint


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def diagnose():
    data=ROOT/'output/development_v4_integrated';trials=json.loads((data/'trials.json').read_text(encoding='utf-8'))
    source={s:digest(ROOT/s) for s in ('coordinator.py','coordinator_model.py','coordinator_contacts.py','calibration/public_contact_model_phase2_v1.json')}
    result=dict(schema='v4_nominal_task_action_algebra_v1',scope='Seen public V4 stream only; counterfactual reference predictions are not physical counterfactual outcomes',
        source_sha256=source,per_scene=[],metrics=[])
    for trial in trials:
        if trial['method'] in ('ratio','joint','passive'):
            m=trial['research_metrics'];q=m['dynamic_quality'];v=q.get('bandlimited') or {}
            result['metrics'].append(dict(scene=trial['scenario'],method=trial['method'],dynamic_eligible=q['eligible'],
                TV_N_s=v.get('finger_TV_rate_N_s'),mean_force_N=q.get('mean_actual_finger_force_N')))
    for scene in ('nominal','sphere','sensor_noise','dynamic_transport','fragile_conflict','contact_robustness'):
        run=data/'runs'/f'{scene}__joint__610103';paths=[run/'trace.npz',run/'metadata.json'];before={str(p):digest(p) for p in paths}
        a=np.load(paths[0],allow_pickle=False);meta=json.loads(paths[1].read_text(encoding='utf-8'))
        tt=a['time'];obs=a['observation_json'];debug=a['debug_json'];target=a['actual_motion_target'];executed=a['executed_motion_reference'];logged=a['requested_action']
        ctl=create_joint(meta['calibration'],meta['limits'],meta['budget']);rows=[];max_replay_error=0.
        for k in np.flatnonzero((tt>=7.3-1e-9)&(tt<=11.7+1e-9)):
            frame=SimpleNamespace(**json.loads(str(obs[k])))
            if frame.time<7.5-1e-9:ctl.state.update(frame,.005);continue
            ctl.set_motion_request(target[k],executed[k-1]);action=ctl.step(frame)
            max_replay_error=max(max_replay_error,float(np.max(np.abs(action.delta_closure-logged[k]))))
            if abs(ctl.last_decision-frame.time)>1e-8 or frame.time<9.3-1e-9:continue
            s=ctl.state;f=ctl.model.snapshot;cs=ctl.contacts.snapshot;h=ctl.g.horizon
            P,c,error=ctl.model.trajectory(s,h,.02);step=np.minimum(np.r_[np.full(5,ctl.g.close_rate*.02),np.asarray(ctl.settings.motion_rates)*.02],ctl.model.input_step_limits)
            previews=np.array([np.clip(ctl.motion_request-ctl.executed_motion,-step[5:]*(j+1),step[5:]*(j+1)) for j in range(h)])
            nominal=np.zeros((h,8));nominal[:,5:]=np.diff(np.vstack((np.zeros(3),previews)),axis=0)
            y=(c+P@nominal.ravel()).reshape(h,16);err=error.reshape(h,16)
            robust=y[:,:5]+err[:,:5]+ctl.settings.forecast_margin_N
            nonnegative=float(y[:,:10].min())>=-1e-8
            physical_guard=bool(np.all(robust<=ctl.limits+1e-8) and np.all(robust.sum(axis=1)<=ctl.budget+1e-8)
                and np.all(f.last_measured[:5]<=ctl.limits) and f.last_measured[:5].sum()<=ctl.budget and nonnegative)
            current_ratio=f.output_now[5:10]/np.maximum(.15,f.output_now[:5])
            compatibility=(y[:,5:10]-current_ratio*y[:,:5] <= err[:,5:10]+current_ratio*err[:,:5]+.02)
            expected=(f.output_now[:5]>.15)&ctl.model.active_mask
            support=bool(expected[0] and np.count_nonzero(expected)>=3 and np.all(cs.speed_lower[expected]<.002))
            nominal_contact=bool(support and np.all(compatibility))
            old_low=np.maximum(0.,ctl.contacts.snapshot.anchor_pressure-ctl.settings.grasp_half_width_N-cs.measurement_error_pressure)
            old_high=ctl.contacts.snapshot.anchor_pressure+ctl.settings.grasp_half_width_N+cs.measurement_error_pressure
            old_gap=np.maximum(0.,np.maximum(y[:,:5]-old_high,old_low-y[:,:5])).max(axis=0)
            d=json.loads(str(debug[k]));actual=logged[k]*4.
            rows.append(dict(time=float(frame.time),nominal_physical_guard=physical_guard,nominal_contact_compatible=nominal_contact,
                nominal_guard_and_contact=physical_guard and nominal_contact,motion_requested=bool(np.max(np.abs(previews[0]))>1e-9),
                action_nonzero=bool(np.max(np.abs(actual))>1e-6),closing=bool(np.any(actual>1e-6)),unloading=bool(np.any(actual<-1e-6)),
                actual_alpha_delta=actual.tolist(),nominal_motion_delta=nominal[0,5:].tolist(),nominal_pressure_prediction=y[0,:5].tolist(),
                actual_pressure_prediction=ctl.predicted.tolist(),old_pressure_corridor_min_deficit_N=old_gap.tolist(),
                actual_corridor_slack_N=d.get('grasp_corridor_violation_N'),actual_pressure_slack_N=d.get('predicted_pressure_violation_N'),
                observed_pressure=f.output_now[:5].tolist(),old_anchor=ctl.contacts.snapshot.anchor_pressure.tolist(),
                current_modes=cs.mode.tolist(),nominal_shear_compatibility=compatibility.all(axis=0).tolist()))
        count=len(rows)
        summary={name:sum(bool(row[name]) for row in rows) for name in ('nominal_physical_guard','nominal_contact_compatible','nominal_guard_and_contact','motion_requested','action_nonzero','closing','unloading')}
        summary.update(decisions=count,replay_max_action_error=max_replay_error,
            nonzero_despite_nominal_guard_and_contact=sum(row['action_nonzero'] and row['nominal_guard_and_contact'] for row in rows),
            old_anchor_constraint_violated_by_nominal=sum(max(row['old_pressure_corridor_min_deficit_N'])>.0005 for row in rows),
            post_10_closing_despite_nominal_guard=sum(row['time']>=10.-1e-9 and row['closing'] and row['nominal_physical_guard'] for row in rows))
        assert before=={str(p):digest(p) for p in paths}
        result['per_scene'].append(dict(scene=scene,summary=summary,records=rows,inputs_sha256=before,inputs_unchanged=True))
    assert source=={s:digest(ROOT/s) for s in source}
    result['sources_unchanged']=True
    return result


if __name__=='__main__':
    path=ROOT/'review/v4_nominal_filter_diagnosis_v1.json'
    if path.exists():raise FileExistsError(path)
    result=diagnose();path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(path),summaries=[dict(scene=x['scene'],**x['summary']) for x in result['per_scene']]),ensure_ascii=False,indent=2))

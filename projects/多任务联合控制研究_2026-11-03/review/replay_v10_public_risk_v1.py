"""Replay actual old public observations, robot-only velocity whitelist, no dynamics.

These are software counterexamples, not counterfactual physical V10 rollouts.
"""
from pathlib import Path
from dataclasses import fields
from collections import Counter
from datetime import datetime,timezone
import json,sys,hashlib,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v10.runtime import V10Session,source_identity
from contracts import SensorFrame

def frame(value):
    core={x.name for x in fields(SensorFrame)}
    f=SensorFrame(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in value.items() if k in core})
    for k,v in value.items():
        if k not in core:setattr(f,k,np.asarray(v) if k in ('executed_motion_reference','motion_reference_rate_limits') else v)
    return f

def main():
    reports=[];identity=source_identity();failure=None
    try:
        for motion,scene in [('hold','fragile_conflict'),('hold','collision'),('transport','friction_drop'),('transport','collision')]:
            s=V10Session(method='joint',scenario=scene,seed=610212,split='robustness',motion_mode=motion)
            c=s.controller;path=ROOT/f'output/development_v9_{motion}_v1/runs/{scene}__ratio_preview__610212/trace.npz'
            checks=0;statuses=Counter();active=0;max_step=0.;first_material=None;first_intervention=None
            with np.load(path,allow_pickle=False) as a:
                time=a['time'];observations=a['observation_json'];tasks=a['task_request_json'];qvel=a['qvel']
                for i in np.flatnonzero((time>=7.3-1e-9)&(time<12.-1e-9)):
                    f=frame(json.loads(str(observations[i])))
                    # Explicit actual robot encoder channel only; no object
                    # state or pressure truth is supplied to the controller.
                    f.lift_velocity_m_s=float(qvel[i,c.geometry.dadr[24]])
                    if f.time<7.5-1e-9:c.state.update(f,.005);continue
                    c.set_task_request(json.loads(str(tasks[i])))
                    action=c.step(f)
                    u=action.delta_closure
                    assert u.shape==(5,) and np.isfinite(u).all()
                    assert np.max(abs(u))<=.8*.005+1e-10
                    assert np.all(u<=f.closure_max-f.alpha+1e-10)
                    assert np.all(u>=f.closure_min-f.alpha-1e-10)
                    blocked=~np.asarray(f.valid,bool)|~np.asarray(f.closure_permission,bool)
                    assert np.all(u[blocked]<=1e-12)
                    assert not action.debug['extra_motion_freeze']
                    assert action.debug['actual_baseline_future_sequence_predicted'] is False
                    assert not action.debug['prediction_available']
                    assert np.isfinite(c.motion_reference).all()
                    status=action.debug['status'];statuses[status]+=1;checks+=9
                    if status!='BASELINE_NO_MATERIAL_DIRECTION_REQUEST' and first_material is None:first_material=float(f.time)
                    if c.plan_active:
                        active+=1
                        if first_intervention is None:first_intervention=float(f.time)
                        fraction=min(1.,(f.time-c.plan_start+.005)/.02)
                        expected=c.plan_reference[5:]+fraction*c.plan_delta[5:]
                        assert np.allclose(c.motion_reference,expected,rtol=0,atol=1e-15);checks+=1
                    max_step=max(max_step,float(np.max(abs(u))))
            assert s.plant.data.time==0. and not s.trace;checks+=1
            row=dict(motion=motion,scenario=scene,checks=checks,status_counts=dict(statuses),active_ticks=active,
                first_material_s=first_material,first_intervention_s=first_intervention,max_finger_step=max_step,
                trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),plant_advanced=False)
            reports.append(row);print(json.dumps(row),flush=True)
    except Exception as exc:
        failure=dict(error=repr(exc),traceback=traceback.format_exc());print(json.dumps(failure),flush=True)
    out=ROOT/'review'/('v10_public_risk_replay_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    data=dict(passed=failure is None and len(reports)==4,physics_steps=0,source_identity=identity,
        sources_unchanged=identity==source_identity(),reports=reports,failure=failure,
        scope='Archived public-stream software wiring only; no physical V10 superiority, safety, empirical model admission or actual future baseline prediction claim.')
    with out.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(out),passed=data['passed'])),flush=True)
    return 0 if data['passed'] and data['sources_unchanged'] else 1
if __name__=='__main__':raise SystemExit(main())

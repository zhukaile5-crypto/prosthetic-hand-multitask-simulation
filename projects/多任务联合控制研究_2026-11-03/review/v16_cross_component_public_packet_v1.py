"""One archived release decision through actual V16 public components.

No new sensor samples, controller physics, object truth or plant construction.
This is off-policy interface evidence, not closed-loop performance.
"""
from pathlib import Path
from dataclasses import fields
from types import SimpleNamespace
import hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
import mujoco
from hand_plant import _model
from foundation_config import FoundationConfig,OPEN
from v13_a1.codec import Reference21
from v13_joint.lifecycle import ReleaseEntry
from v15_support.receiver_sensor import ReceiverMount,ReceiverMeasurement
from v16_release_geometry.public_adapter import PublicReleaseAdapter
from v16_release_geometry.transfer_observer import TransferObserver
import v16_release_geometry.supervisor as supervisor_module


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(out):
    names=('public_adapter.py','geometry_tasks.py','transfer_observer.py','hierarchy.py','supervisor.py','runtime.py','plant.py')
    sources={n:sha(ROOT/'v16_release_geometry'/n) for n in names}
    forbidden=[]
    def no(*a,**kw):forbidden.append('forbidden');raise RuntimeError('no physics/forward/contact/writer')
    for n in ('mj_step','mj_step1','mj_step2','mj_forward','mj_collision','mj_contactForce'):
        if hasattr(mujoco,n):setattr(mujoco,n,no)
    model,_=_model(FoundationConfig());gid=int(model.geom('support').id);h=float(model.geom_size[gid,2])
    mount=ReceiverMount('support',('floor','support_post'),np.array([0.,0.,h]),h,float(np.cos(np.pi/4)))
    adapter=PublicReleaseAdapter(model,open18=np.concatenate(OPEN),receiver_mount=mount)
    path=ROOT/'output/v15A_prospective_candidate_A_hold_v2/runs/nominal__ratio_effort__619731/trace.npz'
    with np.load(path,allow_pickle=False) as z:time=z['time'];observations=z['observation_json']
    selected=np.flatnonzero((time>=11.75-1e-9)&(time<=12.02+1e-9))
    packets=[SimpleNamespace(**json.loads(str(observations[k]))) for k in selected]
    fields_receiver={x.name for x in fields(ReceiverMeasurement)}
    graph=None;entry=None;last=None
    for f in packets:
        adapter.observe(f)
        if f.time<12-1e-9:continue
        if graph is None:
            entry=ReleaseEntry(12.,Reference21(f.robot_reference21,12.),int(f.mode_generation))
            prior=adapter.begin_release(12.)
            observer=TransferObserver(request_time_s=12.,receiver_half_size_xy_m=adapter.receiver_half_size_xy_m,
                                      receiver_up_axis3=adapter.receiver_up_axis3,prior=prior)
            graph=supervisor_module.ReleaseSupervisor(entry,open18=np.concatenate(OPEN),observer=observer)
        receiver=ReceiverMeasurement(**{k:v for k,v in f.receiver_surface.items() if k in fields_receiver})
        last=graph.observe(frame=f,transfer_inputs=adapter.transfer_inputs(receiver),current21=f.robot_reference21)
    f=packets[-1];current=np.asarray(f.robot_reference21,float)
    lower=np.asarray(f.reference_lower21,float);upper=np.asarray(f.reference_upper21,float)
    captured={};original=supervisor_module.solve_geometry_release
    def wrapper(tasks,**kwargs):
        result=original(tasks,**kwargs);captured.update(tasks=tasks,result=result,kwargs=kwargs);return result
    supervisor_module.solve_geometry_release=wrapper
    try:reference,debug=graph.action(frame=f,adapter=adapter,current21=current,lower21=lower,upper21=upper,generation=int(f.mode_generation))
    finally:supervisor_module.solve_geometry_release=original
    ev=graph.decision_evidence[-1];scale=np.asarray(ev['scale21']);delta=np.asarray(ev['proposed_reference_increment21'])
    rates=np.r_[[3.]*18,.18,.12,.12]
    expected=np.clip(current+.25*delta,np.maximum(lower,current-rates*.005),np.minimum(upper,current+rates*.005))
    result=captured.get('result');tasks=captured.get('tasks')
    normal=None if tasks is None or tasks.normal_matrix_m_per_reference is None else tasks.normal_matrix_m_per_reference@delta
    unknown_receiver=ReceiverMeasurement(**{k:v for k,v in f.receiver_surface.items() if k in fields_receiver})
    unknown=adapter.transfer_inputs(unknown_receiver);unknown['hand_receiver_confounded']=None;unknown['geometry_acquisition_time_s']=None
    unknown_observer=TransferObserver(request_time_s=12.,receiver_half_size_xy_m=adapter.receiver_half_size_xy_m,
        receiver_up_axis3=adapter.receiver_up_axis3,prior=adapter.pre_release_prior)
    unknown_evidence=unknown_observer.update(**unknown)
    checks=dict(
        real_archived_12_02_packet_used=abs(f.time-12.02)<1e-9,
        request_preempts_holding=last['holding_preempted'] and not last['holding_force_tasks_active'],
        frame_low_pressure_not_mistaken_for_global_clear=not last['geometric_full_hand_clearance_certified'],
        geometry_real_public_path_available=ev['geometry_status']=='EXPERIMENTAL_GEOMETRY_TASKS',
        hierarchy_accepted=ev['geometry_proposal_accepted'],
        normalized_increment_restored_exactly_once=result is not None and np.array_equal(delta,result.x*scale),
        first_quarter_reference_exact=np.array_equal(reference.values,expected),
        issued_clock_is_current_not_future=reference.time_s==f.time,
        returned_reference_within_actual_box=bool(np.all(reference.values>=lower)&np.all(reference.values<=upper)),
        actual5ms_rate_respected=bool(np.all(abs(reference.values-current)<=rates*.005+1e-15)),
        absent_geometry_attribution_no_experimental_transfer_permission=unknown_evidence.stage_permission=='OBSERVATION_ONLY',
        no_predicted_pressure_or_servo_certificate=not ev['future_pressure_prediction_available'] and not ev['actual_servo_motion_predicted'] and not ev['physical_safety_certified'],
        no_writer=adapter.actor.write_count==0,
        forbidden_calls_zero=not forbidden)
    after={n:sha(ROOT/'v16_release_geometry'/n) for n in names}
    checks['sources_unchanged_during_readonly_test']=sources==after
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    doc=dict(schema='v16_cross_component_archived_single_decision_v1',passed=all(checks.values()),checks=checks,
        check_count=len(checks),source_sha256=sources,input_trace_sha256=sha(path),
        frame_time_s=f.time,entry_time_s=entry.started_s,phase=last['phase'],transfer=last['transfer'],
        geometry_clearance=adapter.last_clearance_diagnostics,decision=ev,
        max_proposed_old_plane_inward_m=None if normal is None else float(np.max(normal,initial=0)),
        scope=dict(physics_steps=0,plant_instances=0,forward_calls=0,contact_queries=0,sensor_samples=0,RNG_draws=0,writer_calls=0,
                   private_model_FK_transmission_only=True,off_policy_archived_inputs=True,
                   first_decision_only_not_intermediate_tick_or_generation_certificate=True,
                   not_closed_loop_performance_or_delivery_evidence=True))
    out.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=doc['passed'],checks=len(checks),path=str(out),sha=sha(out),failed=[k for k,v in checks.items() if not v]),ensure_ascii=False))
    return 0 if doc['passed'] else 1


if __name__=='__main__':raise SystemExit(main(sys.argv[1]))

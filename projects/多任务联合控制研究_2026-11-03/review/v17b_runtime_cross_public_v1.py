"""Real V17 lifecycle/kernel + archived public observations, zero integration.

The existing pure fake-port setup helper is reused, then supplied the real
compiled robot model/A1 actor. q/v/tactile packets remain archived off-policy
inputs; only authoritative reference/ports/permissions follow new commands.
Raw archive patches contribute ONLY digit and normal for instrument validity.
"""
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch
import ast,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main(out):
    import research_imports,mujoco
    from hand_plant import _model,HandPlant
    from foundation_config import FoundationConfig,OPEN
    from v13_a1.codec import Reference21,EnginePort21,ReferenceActor21
    from v13_joint.lifecycle import ReleaseEntry
    from v16_release_geometry.plant import ReleasePlant
    from v15_support.receiver_sensor import ReceiverMount,ReceiverMeasurement
    from v17_contact_transfer.motion_adapter import PublicMotionAdapter
    from v17_contact_transfer.motion_domain_sensors import SLIP_DOMAIN_ROLE
    from v17b_full_evidence import runtime as rt
    before=rt.source_identity(rt.METHODS);old=read(ROOT/'review/v17_transfer_small_plan_v1.json')['source_sha256']
    checks=[];forbidden=[]
    def ck(n,v,d=None):checks.append(dict(name=n,passed=bool(v),details=d))
    def no(*a,**kw):forbidden.append('forbidden integration/forward/contact');raise RuntimeError(forbidden[-1])
    with ExitStack() as stack:
        for n in ('mj_step','mj_step1','mj_step2','mj_forward','mj_collision','mj_contactForce','mj_Euler','mj_RungeKutta'):
            if hasattr(mujoco,n):stack.enter_context(patch.object(mujoco,n,no))
        stack.enter_context(patch.object(HandPlant,'advance',no))
        config=FoundationConfig();model,_=_model(config)
        helper=ROOT/'review/v16_runtime_bridge_independent_v2.py'
        ck('reused_frozen_fake_port_helper_identity',sha(helper)=='4ac03746a8ad35abe037fa57b249daee5cee90202ab1adfc2035375d5e8aa517')
        tree=ast.parse(helper.read_text(encoding='utf-8-sig'));fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='make_fake')
        env=dict(np=np,SimpleNamespace=SimpleNamespace,ReleasePlant=ReleasePlant,Reference21=Reference21,EnginePort21=EnginePort21,ReleaseEntry=ReleaseEntry)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(helper),'exec'),env)
        plant,unused,unused_start=env['make_fake']()
        plant.model=model;plant.config=config;plant.open_pose=[np.asarray(x,float) for x in OPEN];plant.actor21=ReferenceActor21(model)
        plant.obstacle_motor=int(model.actuator_ctrladr[model.actuator('integration_obstacle_motor').id])
        gid=int(model.geom('support').id);height=float(model.geom_size[gid,2])
        mount=ReceiverMount('support',('floor','support_post'),np.array([0.,0.,height]),height,float(np.cos(np.pi/4)))
        adapter=PublicMotionAdapter(model,open18=np.concatenate(OPEN),receiver_mount=mount)
        folder=ROOT/'output/v15A_prospective_candidate_A_hold_v2/runs/fragile_conflict__ratio_effort__619732'
        with np.load(folder/'trace.npz',allow_pickle=False) as z:
            times=z['time'];observations=z['observation_json'];raw_patches=z['raw_patches_json']
        selected=np.flatnonzero((times>=11.8-1e-9)&(times<=12.32+1e-9));public={int(i):json.loads(str(observations[i])) for i in selected}
        # Device-side metadata only. No external identity/position/velocity from
        # raw diagnostic packets is presented to any controller/adapter.
        domain={};domain_evidence=[]
        raw_start=min(float(x['sample_time']) for x in public.values())
        raw_selected=np.flatnonzero((times>=raw_start-1e-9)&(times<=12.32+1e-9))
        for i in raw_selected:
            packets=json.loads(str(raw_patches[i]));packets=packets['patches'] if isinstance(packets,dict) else packets
            normal=np.zeros(5)
            for p in packets:
                digit=p['digit'];n=p['normal']
                if isinstance(digit,bool) or not isinstance(digit,(int,np.integer)):raise ValueError('raw instrument digit malformed')
                if 0<=digit<5:normal[digit]+=max(0.,float(n))
            domain[int(round(float(times[i])/.005))]=normal>.02
            domain_evidence.append(dict(acquisition_time_s=float(times[i]),aggregated_instrument_normal5_N=normal.tolist(),retained_slip_pair5=(normal>.02).tolist()))
        first=public[int(np.argmin(abs(times-12.)))];entry_values=np.asarray(first['robot_reference21'],float)
        plant.control_steps=2400;plant.mode_generation=int(first['mode_generation']);plant._joint_mode='release'
        plant._joint_committed=Reference21(entry_values,12.);plant.joint_release_entry=ReleaseEntry(12.,Reference21(entry_values,12.),plant.mode_generation)
        plant.release_started=12.;plant.phase='RELEASING';plant.v16_supervisor=None;plant.v16_last_version=None
        plant.receiver_release_tasks=None;plant.receiver_release_enabled=False;plant._joint_anchor_time=None;plant._joint_anchor_reference=None
        initial=np.zeros(22);initial[plant.actor21.indices21]=np.asarray(first['robot_actuator_target21'])
        plant.data=SimpleNamespace(time=12.,ctrl=initial);plant.a1_last_engine_port21=plant.engine_snapshot()
        plant.lift_ref=float(entry_values[18]);plant.wrist_ref=entry_values[19:].copy()
        s=rt.ReleaseSession.__new__(rt.ReleaseSession);s.plant=plant;s.v16_release_policy='coordinated';s.v16_supervisor=None;s.release_driver=None;s.release_clearance_goal=None;s.release_adapter=adapter
        state=SimpleNamespace(_last_time=None,time=0.,updates=0)
        def update(frame,dt):state._last_time=state.time=frame.time;state.updates+=1
        state.update=update
        def holding(*a,**kw):raise AssertionError('holding controller executed during release')
        s.controller=SimpleNamespace(state=state,step=holding,desired=np.ones(5),debug={},motion_reference=np.zeros(3))
        original_solver=__import__('v17_contact_transfer.supervisor',fromlist=['_contact_solve'])._contact_solve
        supervisor_module=sys.modules['v17_contact_transfer.supervisor'];kernel_calls=[]
        def counted(tasks,**kwargs):
            result=original_solver(tasks,**kwargs)
            kernel_calls.append(dict(task_status=tasks.status,motion_rows=0 if kwargs['motion_task'] is None else len(kwargs['motion_task'].target_M),
                available=False if result is None else result.available,status=None if result is None else result.status,levels=0 if result is None else len(result.levels)))
            return result
        ticks=[];exact_ports=True;rates_ok=True;bounds_ok=True;unknown_prediction=True;mask_source_ok=True
        from dataclasses import fields
        rf={f.name for f in fields(ReceiverMeasurement)}
        with patch.object(supervisor_module,'_contact_solve',counted):
            for i in selected:
                d=public[int(i)];f=SimpleNamespace(**d);now=float(f.time);ak=int(round(float(f.sample_time)/.005));mask=domain.get(ak)
                f.slip_valid5=np.zeros(5,bool) if mask is None else mask&np.asarray(f.valid,bool)
                f.slip_domain_acquisition_time_s=float(f.sample_time);f.slip_domain_time_s=now;f.slip_domain_role=SLIP_DOMAIN_ROLE
                f.slip_domain_available=mask is not None;f.slip_domain_reason=None if mask is not None else 'archive_domain_outside_requested_history'
                mask_source_ok &= mask is not None
                if now>=12-1e-9:
                    plant.control_steps=int(round(now/.005));plant.data.time=now
                    f.robot_reference21=plant.reference_snapshot().values.copy();f.robot_actuator_target21=plant.engine_snapshot().values.copy()
                    f.robot_control_time_s=now;f.mode_generation=plant.mode_generation
                    f.reference_lower21,f.reference_upper21=plant.reference_authority21()
                adapter.observe(f)
                if now<12-1e-9:continue
                s.latest_receiver_measurement=ReceiverMeasurement(**{k:v for k,v in d['receiver_surface'].items() if k in rf})
                prior=plant.reference_snapshot().values.copy();previous_writes=plant.actor21.write_count
                desired,prediction,debug=s._lifecycle_step(f)
                receipt=plant.last_execution_receipt21;executed=receipt.executed.values
                real_lower,real_upper=plant.reference_authority21()
                exact_ports &= np.array_equal(plant.data.ctrl[plant.actor21.indices21],plant.actor21.encode(receipt.executed).values)
                exact_ports &= plant.actor21.write_count==previous_writes+1 and receipt.requested.time_s==receipt.executed.time_s==now
                rates=np.r_[np.full(18,3.),.18,.12,.12]
                rates_ok &= bool(np.all(abs(executed-prior)<=rates*.005+1e-14))
                bounds_ok &= bool(np.all(executed>=real_lower-1e-14)&np.all(executed<=real_upper+1e-14))
                unknown_prediction &= np.isnan(prediction).all() and np.array_equal(desired,np.zeros(5)) and debug['task_request_modified'] is False
                graph=s.v16_supervisor
                ticks.append(dict(time_s=now,phase=graph.phase,mode_generation=plant.mode_generation,process_version=graph.version,
                    requested21=receipt.requested.values.tolist(),executed21=executed.tolist(),receipt_changed=receipt.execution_changed,
                    reference_increment21=(executed-prior).tolist(),true_lower21=f.reference_lower21.tolist(),true_upper21=f.reference_upper21.tolist(),
                    motion_task_rows=0 if adapter.motion_task is None else len(adapter.motion_task.target_M),geometry_status=debug['geometry_status'],
                    geometry_accepted=debug['geometry_proposal_accepted'],hierarchy_status=debug['hierarchy_status'],fallback=debug['fallback']))
        graph=s.v16_supervisor
        approach=[x for x in ticks if x['phase']=='APPROACH_SUPPORT'];contact=[x for x in ticks if x['phase']=='CONTACT_UNLOADING']
        ck('all136_sources_old133unchanged_and3evidence_only',len(before)==136 and len(old)==133 and {k:v for k,v in before.items() if not k.startswith('v17b_full_evidence/')}==old)
        ck('domain_from_raw_acquisition_normal_not_noisy_observed_pressure',mask_source_ok and len(domain_evidence)>=len(selected))
        ck('actual_lifecycle_prebuilds_correct_terminal_and_new_supervisor',type(graph) is rt.ReleaseSupervisor and np.array_equal(graph.open18,s.release_clearance_goal.reference18) and not np.array_equal(graph.open18,np.concatenate(OPEN)) and plant.v16_supervisor is graph)
        ck('APPROACH_realissued_fingers_wrists_entry',bool(approach) and all(np.array_equal(np.asarray(x['executed21'])[:18],entry_values[:18]) and np.array_equal(np.asarray(x['executed21'])[19:],entry_values[19:]) for x in approach))
        ck('twofresh_realreceiver_packets_enter_CONTACT',bool(contact),dict(first_contact_time=None if not contact else contact[0]['time_s']))
        ck('real_geometry_motion_kernel_called_atleast_once',bool(kernel_calls) and any(x['motion_rows']>0 and x['levels']>0 for x in kernel_calls),kernel_calls)
        ck('CONTACT_true_lift_authority_has_two_directions',any(x['true_lower21'][18]<x['executed21'][18]<x['true_upper21'][18] for x in contact))
        ck('actual_sole21_actor_receipt_clock_and_single_liftbias_exact',exact_ports and plant.actor21.write_count==len(ticks))
        ck('all5ms_actualrates_and_truepermissions',rates_ok and bounds_ok)
        ck('holding_bypassed_and_unknown_prediction_not_zero',unknown_prediction and state.updates==len(ticks))
        ck('no_placement_certificate_from_receiverload',not graph.public_state['placed'] and not graph.public_state['placement_certified'])
        previous=read(ROOT/'review/v17_runtime_cross_public_attempt1.json')
        ck('all65_actual_tick_commands_exactly_equal_frozen_V17_fixture',ticks==previous['ticks'])
        records=graph.decision_evidence
        motion=[d for d in records if d['motion_task_available']]
        ck('every_new_decision_has_full_motion_evidence_or_explicit_None',bool(motion) and all(
            ('motion_task_public' in d) and (d['motion_task_public'] is not None)==d['motion_task_available'] for d in records))
        ck('complete_motion_rows_targets_clock_are_finite_and_conform',all(
            d['motion_task_public']['time_s']==d['time_s'] and
            np.asarray(d['motion_task_public']['matrix_Mx21']).shape==(len(d['motion_task_public']['target_M']),21) and
            np.isfinite(np.asarray(d['motion_task_public']['matrix_Mx21'])).all() and
            np.isfinite(np.asarray(d['motion_task_public']['target_M'])).all() for d in motion))
        ck('no_additional_decisions_from_logging',len(records)==len(previous['release_decisions']) and len(kernel_calls)==7)
        ck('all_forbidden_pipelines0',not forbidden,forbidden)
    after=rt.source_identity(rt.METHODS);ck('all136_execution_source_hashes_stable_during_gate',before==after)
    report=dict(schema='v17b_logging_only_archived_public_crossgate_v1',passed=all(x['passed'] for x in checks),checks=checks,check_count=len(checks),
        source_sha256=before,source_after_sha256=after,draft_source=before!=after,
        input_trace_sha256=sha(folder/'trace.npz'),raw_domain_provenance=domain_evidence,ticks=ticks,
        release_decisions=graph.decision_evidence,
        scope=dict(physics_steps=0,Session_constructions=0,known_model_compilations=1,forward_calls=0,contact_queries=0,
            real_A1_actor_on_standalone_control_buffer=True,reused_helper_sha256=sha(helper),
            raw_fields_consumed_for_metadata_only=['digit','normal','acquisition_time'],
            controller_reads_archived_public_fields_not_raw_other_truth=True,
            robot_qv_tactile_archived_but_command_memory_from_new_actor=True,
            off_policy_software_interoperability_not_physical_reproduction=True,
            inability_to_solve_logged_as_fallback_not_hidden=True,not_delivery_or_performance_certificate=True))
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),out=str(out),sha=sha(out),kernel_calls=len(kernel_calls),motion_calls=sum(x['motion_rows']>0 for x in kernel_calls),failed=[c['name'] for c in checks if not c['passed']]),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main(sys.argv[1]))

from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
old=ROOT/'review/verify_v25_runtime_author_v1.py';new=ROOT/'review/verify_v25_runtime_single_owner_author_v2.py';assert not new.exists()
text=old.read_text(encoding='utf-8')
text=text.replace("obj=mod.ResponseSession.__new__(mod.ResponseSession)","obj=mod.ResponseSession.__new__(mod.ResponseSession);obj.probe_coordinator=None")
text=text.replace("            ck('no_pipeline_calls',not forbidden)","            extra_checks(mod,runner,Nominal,ck,reject)\n            ck('no_pipeline_calls',not forbidden)")
text=text.replace("schema='v25_runtime_author_pure_v1'","schema='v25_runtime_single_owner_author_v2'")
text=text.replace("Actual constructor and populated fork/full16 bridges remain untested here.","Actual ReleasePlant.bind_supervisor identity guard plus mocked full precommit/one-commit/reconcile chain and five-assignment failure-save continuation tested; no constructor, physics, or full16 bridge tested.")
marker="if __name__=='__main__':raise SystemExit(main())"
assert text.count(marker)==1
extra='''def extra_checks(mod,runner,Nominal,ck,reject):
    from types import MethodType
    from contextlib import redirect_stdout
    from io import StringIO
    from v16_release_geometry.plant import ReleasePlant
    from v13_joint.plant import ExecutionReceipt21
    from v13_a1.codec import Reference21,EnginePort21,ACTUATORS21
    from v25_response_identification.observer import PublicMotionObserver
    from v25_response_identification.design import ProbeSpec,PROBE_ARMS
    flow=[]
    class Graph(Nominal):
        version=0
        public_state={}
        def observe(self,**kwargs):flow.append('observe')
        def action(self,**kwargs):flow.append('action');return super().action(**kwargs)
        def reconcile(self,receipt):flow.append('nominal_reconcile');return super().reconcile(receipt)
    graph=Graph();current=np.zeros(21);bias=.01
    plant=SimpleNamespace(joint_release_entry=object(),v16_supervisor=graph,v16_last_version=0,mode_generation=0,last_execution_receipt21=None)
    plant.bind_supervisor=MethodType(ReleasePlant.bind_supervisor,plant)
    plant.reference_snapshot=lambda:Reference21(current,14.)
    def engine():
        values=current.copy();values[18]+=bias;return EnginePort21(values,14.,bias)
    plant.engine_snapshot=engine
    plant.sync_supervisor_generation=lambda:flow.append('sync')
    def commit(ref,*,mode,expected_generation):
        nonlocal current
        flow.append('commit');current=ref.values.copy()
        receipt=ExecutionReceipt21(ref,ref,engine(),mode,expected_generation,False,False)
        plant.last_execution_receipt21=receipt;return receipt
    plant.commit_reference21=commit
    obj=mod.ResponseSession.__new__(mod.ResponseSession);obj.tick=2800;obj.sensors=SimpleNamespace(_last_time=13.995)
    obj.plant=plant;obj.v16_supervisor=graph;obj.release_driver=graph;obj.probe_coordinator=None
    obj.arm_probe(ProbeSpec('lift_plus',14.))
    ck('actual_arm_keeps_all_three_nominal_owner_identities',obj.v16_supervisor is graph and obj.release_driver is graph and plant.v16_supervisor is graph and obj.probe_coordinator.nominal_supervisor is graph)
    reject('frozen_actual_plant_still_rejects_selector_rebind',lambda:plant.bind_supervisor(obj.probe_coordinator))
    ck('rejected_rebind_did_not_modify_owner',plant.v16_supervisor is graph)
    frame=SimpleNamespace(time=14.,dt=.005,robot_position25=np.zeros(25),robot_velocity25=np.zeros(25),robot_actuator_axes21=ACTUATORS21)
    state=SimpleNamespace(_last_time=13.995,time=13.995)
    def update(obs,dt):flow.append('state_update');state.time=obs.time
    state.update=update
    obj.controller=SimpleNamespace(state=state,motion_reference=np.zeros(3),desired=np.ones(5),debug={})
    obj.release_adapter=SimpleNamespace(transfer_inputs=lambda measurement:dict(public_only=True));obj.latest_receiver_measurement=object()
    def attach(obs):flow.append('attach');obs.reference_lower21=graph.lo.copy();obs.reference_upper21=graph.hi.copy()
    obj._attach_execution_state=attach
    observer=PublicMotionObserver(stream_id='synthetic',epoch_id='pure')
    def on_receipt(receipt,snapshot):flow.append('observer_receipt');return observer.on_receipt(receipt,snapshot)
    obj.response_observer=SimpleNamespace(on_receipt=on_receipt)
    desired,prediction,debug=obj._lifecycle_step(frame)
    ck('actual_probe_entry_exact_callback_order_one_commit_one_reconcile',flow==['state_update','observe','sync','attach','action','commit','nominal_reconcile','observer_receipt'])
    ck('actual_typed_receipt_is_only_logged_input',len(observer.receipts())==1 and np.array_equal(observer.receipts()[0].executed_reference21,current) and len(graph.receipts)==1)
    ck('controller_motion_and_zero_holding_tasks_written_from_actual',np.array_equal(obj.controller.motion_reference,current[18:]) and not np.any(desired) and np.isnan(prediction).all() and not np.any(obj.controller.desired))
    ck('probe_record_does_not_sign_changed_actual_as_nominal',obj.probe_coordinator.records[-1]['nominal_task_certificate_applies_to_actual_reference'] is False and np.array_equal(debug['actual_reference21'],current))
    ck('post_execution_owner_is_still_original_graph',obj.v16_supervisor is obj.release_driver is plant.v16_supervisor is graph)
    for save_fails in (False,True):
        with tempfile.TemporaryDirectory(prefix='v25_failure_accounting_') as td:
            out=Path(td);counter=[0]
            class Branch:
                def __init__(self,index):self.index=index;self.trace=[dict(prefix=True)];self.time=14.;self.running=True;self.sensor_characteristics=dict(tactile_delay_s=.01)
                def arm_probe(self,spec):
                    self.spec=spec
                    if self.index==0:raise ValueError('injected_first_arm_failure')
                    self.probe_coordinator=SimpleNamespace(terminal_status='PROBE_WINDOW_COMPLETED',records=[dict(active=True)])
                def step_one_sensor_tick(self):self.time=round(self.time+.005,9)
                def save(self,destination):
                    destination=Path(destination)
                    if self.index==0 and save_fails:raise RuntimeError('injected_partial_save_failure')
                    if destination.exists():raise FileExistsError('fresh record directory required')
                    destination.mkdir(parents=True);(destination/'record.txt').write_text('preserved',encoding='utf-8')
            def fork(base):
                index=counter[0];counter[0]+=1;return Branch(index),dict(pure_fake=True)
            rows=[];failures=[];stdout=StringIO()
            with patch.object(runner,'fork_session',fork),redirect_stdout(stdout):runner.run_five_branches(object(),dict(time_s=14.),out,rows,failures)
            ck('all_five_retained_after_first_failure_'+str(save_fails),counter[0]==5 and [r['key'] for r in rows]==list(PROBE_ARMS) and rows[0]['status']=='runtime_failed' and all(r['status']=='short_window_recorded' for r in rows[1:]) and len(failures)==1)
            ck('failure_file_and_stdout_preserved_'+str(save_fails),(out/'branches/zero/runtime_error.json').exists() and 'runtime_failed' in stdout.getvalue())
            ck('partial_save_fresh_or_failure_caught_'+str(save_fails),(out/'branches/zero/partial_save_error.json').exists() if save_fails else (out/'branches/zero/partial_record/record.txt').exists())

'''
text=text.replace(marker,extra+marker);new.write_text(text,encoding='utf-8');print(new)

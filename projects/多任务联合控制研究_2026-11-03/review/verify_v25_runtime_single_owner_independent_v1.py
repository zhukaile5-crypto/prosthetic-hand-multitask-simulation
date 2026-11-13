"""Six independent single-owner execution checks; no model or physics."""
from pathlib import Path
from types import SimpleNamespace,MethodType
from unittest.mock import patch
import sys,json,hashlib,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
FILES=('v25_response_identification/runtime.py','v25_response_identification/run_identification.py')
EXPECTED=('10cbb920ca89930a818a5078bfd08baca2d521040318701dc5337415607d5437','a3a9358554513e3df4baa4de645e38929c225376cb5eed6807390e99ac163e4c')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'review/v25_runtime_single_owner_independent_v1.json'
    if out.exists():raise FileExistsError('Fresh gate required')
    before={f:sha(ROOT/f) for f in FILES};checks=[];forbidden=[];error=None;details={}
    def ck(name,good,detail=None):checks.append(dict(name=name,passed=bool(good),details=detail))
    def deny(*a,**kw):forbidden.append('forbidden_pipeline');raise AssertionError('no pipeline')
    try:
        if tuple(before.values())!=EXPECTED:raise ValueError('Requested production identity differs')
        import mujoco
        from v25_response_identification import runtime as r,run_identification as run
        from v25_response_identification.design import ProbeSpec
        from v25_response_identification.observer import PublicMotionObserver
        from v13_a1.codec import Reference21,EnginePort21
        from v13_joint.plant import ExecutionReceipt21
        from v16_release_geometry.plant import ReleasePlant
        from v11.robot_servo_effort import ACTUATORS21
        from v24_canonical_task_fusion.runtime import ReleaseSession as Frozen
        pipeline=('mj_step','mj_step1','mj_step2','mj_forward','mj_forwardSkip','mj_collision','mj_contactForce')
        with patch.multiple(mujoco,**{n:deny for n in pipeline if hasattr(mujoco,n)}):
            class Fixture(r.ResponseSession):
                @property
                def time(self):return self._fake_now
            def session():
                s=object.__new__(Fixture);s._fake_now=12.35;s.probe_coordinator=None;events=[];receipts=[];binds=[]
                current=np.zeros(21);current[18]=.03;bias=.00123
                class Graph:
                    version=0
                    def __init__(self):self.decision_evidence=[]
                    def observe(self,**kw):events.append('observe')
                    def action(self,**kw):
                        events.append('action');self.decision_evidence.append(dict(actual_reference_lower21=(current-.01).tolist(),actual_reference_upper21=(current+.01).tolist(),geometry_proposal_accepted=True,emitted_hierarchy_score=dict(available=True)))
                        return Reference21(current,kw['frame'].time),dict(geometry_proposal_accepted=True)
                    def reconcile(self,receipt):
                        events.append('nominal_reconcile');receipts.append(('nominal',receipt));events.append('path_reconcile');receipts.append(('path',receipt))
                graph=Graph()
                plant=SimpleNamespace(joint_release_entry=object(),v16_supervisor=None,mode_generation=2,last_execution_receipt21=None)
                ReleasePlant.bind_supervisor(plant,graph)
                rejected=False
                try:ReleasePlant.bind_supervisor(plant,object())
                except ValueError:rejected=True
                def bind(value):binds.append(value);return ReleasePlant.bind_supervisor(plant,value)
                plant.bind_supervisor=bind
                def ref():return Reference21(current,s.time)
                def eng():
                    x=current.copy();x[18]+=bias;return EnginePort21(x,s.time,bias)
                plant.reference_snapshot=ref;plant.engine_snapshot=eng
                plant.sync_supervisor_generation=lambda:events.append('sync')
                def commit(reference,**kw):
                    events.append('solecommit');x=reference.values.copy();x[18]=np.nextafter(x[18],-np.inf)
                    port=x.copy();port[18]+=bias
                    receipt=ExecutionReceipt21(reference,Reference21(x,s.time),EnginePort21(port,s.time,bias),'release',2,True,False)
                    plant.last_execution_receipt21=receipt;return receipt
                plant.commit_reference21=commit
                state=SimpleNamespace(_last_time=None,time=-1.)
                def update(frame,dt):events.append('state');state._last_time=state.time=frame.time
                state.update=update
                s.controller=SimpleNamespace(state=state,desired=np.ones(5));s.plant=plant;s.v16_supervisor=graph;s.release_driver=graph
                s.sensors=SimpleNamespace(_last_time=s.time-.005);s.latest_receiver_measurement=object()
                def transfer(x):events.append('transfer_inputs');return {}
                s.release_adapter=SimpleNamespace(transfer_inputs=transfer)
                actual=PublicMotionObserver(stream_id='independent',epoch_id='0')
                class Tap:
                    def on_receipt(self,receipt,snapshot):events.append('actualobserver');receipts.append(('observer',receipt));return actual.on_receipt(receipt,snapshot)
                s.response_observer=Tap()
                def attach(self,frame):events.append('attach');frame.reference_lower21=current-.01;frame.reference_upper21=current+.01
                s._attach_execution_state=MethodType(attach,s)
                frame=SimpleNamespace(time=s.time,dt=.005,robot_position25=np.zeros(25),robot_velocity25=np.zeros(25),robot_actuator_axes21=ACTUATORS21)
                return s,graph,frame,events,receipts,binds,actual,rejected
            s,g,f,events,receipts,binds,obs,rejected=session()
            identities=(s.v16_supervisor,s.release_driver,s.plant.v16_supervisor)
            s.arm_probe(ProbeSpec('lift_plus',s.time))
            ck('arm retains the three real owners without bind and frozen bind rejects replacement',rejected and all(a is b for a,b in zip(identities,(s.v16_supervisor,s.release_driver,s.plant.v16_supervisor))) and not binds and s.probe_coordinator.nominal_supervisor is g)
            result=s._lifecycle_step(f)
            expected=['state','transfer_inputs','observe','sync','attach','action','solecommit','nominal_reconcile','path_reconcile','actualobserver']
            same=bool(receipts) and all(x[1] is s.plant.last_execution_receipt21 for x in receipts)
            ck('one state-observe-sync-attach-action-commit and identical receipt to nominal path observer',events==expected and same,events)
            logged=obs.receipts()[0];row=s.probe_coordinator.records[0]
            ck('typed actual receipt is logged and changed probe cannot borrow nominal certificate',isinstance(s.plant.last_execution_receipt21,ExecutionReceipt21) and np.array_equal(logged.executed_reference21,s.plant.last_execution_receipt21.executed.values) and row['nominal_task_certificate_available'] is True and row['nominal_task_certificate_applies_to_emitted_reference'] is False and row['nominal_task_certificate_applies_to_actual_reference'] is False and np.all(s.controller.desired==0) and np.isnan(result[1]).all())
            # Passive branch must delegate; the spy is on the actual superclass
            # method and returns a real typed receipt without a native writer.
            ps,pg,pf,pe,pr,pb,po,_=session();sentinel=object();parent_calls=[]
            def parent(self,frame):
                parent_calls.append((self,frame));self.plant.commit_reference21(Reference21(np.r_[np.zeros(18),.03,0.,0.],frame.time),mode='release',expected_generation=2);return sentinel
            with patch.object(Frozen,'_lifecycle_step',parent):returned=ps._lifecycle_step(pf)
            ck('passive path still calls superclass exactly once then logs its actual receipt',returned is sentinel and len(parent_calls)==1 and len(po.receipts())==1 and ps.probe_coordinator is None)
            # Directly exercise the real five-branch function, never main or a
            # Session. First arm and its partial saver fail; four still finish.
            artifacts=ROOT/'review/v25_single_owner_independent_fixture_v1'
            if artifacts.exists():raise FileExistsError('Fresh fixture directory required')
            artifacts.mkdir();save_attempts=[];forks=[]
            class Branch:
                def __init__(self,index):
                    self.index=index;self.time=12.565;self.running=False;self.trace=['synthetic'];self.sensor_characteristics={'tactile_delay_s':.01}
                    self.probe_coordinator=SimpleNamespace(terminal_status='PROBE_WINDOW_COMPLETED',records=[{'active':True}])
                def arm_probe(self,spec):
                    if self.index==0:raise RuntimeError('independent first-arm injection')
                def step_one_sensor_tick(self):deny()
                def save(self,path):
                    path=Path(path);save_attempts.append((self.index,str(path),path.exists()))
                    if path.exists():raise FileExistsError('Independent fresh saver oracle')
                    path.mkdir(parents=True)
                    if self.index==0:raise RuntimeError('independent partial-save injection')
                    (path/'synthetic_record.json').write_text('{}',encoding='utf-8')
            def fork(base):
                b=Branch(len(forks));forks.append(b);return b,{'synthetic':True}
            rows=[];failures=[]
            with patch.object(run,'fork_session',fork):run.run_five_branches(object(),{'time_s':12.35},artifacts,rows,failures)
            fresh=all(not x[2] for x in save_attempts) and save_attempts[0][1].endswith('partial_record')
            ck('first arm plus partial-save failure use fresh child and retain all five assignments',len(forks)==5 and [x['key'] for x in rows]==list(run.PROBE_ARMS) and rows[0]['status']=='runtime_failed' and all(x['status']=='short_window_recorded' for x in rows[1:]) and len(failures)==1 and 'partial_save_error' in failures[0] and fresh,dict(rows=rows,save_attempts=save_attempts))
            after={f:sha(ROOT/f) for f in FILES}
            ck('requested two source bytes unchanged and no pipeline or model constructor called',before==after and tuple(after.values())==EXPECTED and not forbidden)
            details=dict(real_frozen_bind_function='v16_release_geometry.plant.ReleasePlant.bind_supervisor',
                ordered_events=events,typed_receipts=1,fixture_artifacts=str(artifacts),
                scope='Independent callback/type/failure fixtures; no model construction, device sampling, actuator write, physical step or policy outcome claim')
    except Exception:error=traceback.format_exc()
    result=dict(passed=error is None and len(checks)==6 and all(c['passed'] for c in checks),checks=checks,
        production_source_sha256=before,source_unchanged=before=={f:sha(ROOT/f) for f in FILES},
        physics_steps=0,model_constructions=0,forbidden_calls=forbidden,error=error,details=details,checker_sha256=sha(__file__))
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),error=error,path=str(out)),ensure_ascii=False))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

"""Small runtime wiring/clock mocks. No model, real advance or integration."""
from pathlib import Path
from types import SimpleNamespace,MethodType
from unittest.mock import patch
import ast,hashlib,json,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from v15_support.runtime import SupportSession
from v15_support.plant import SupportPlant,CoordinatedPlant
from v15_support.release_tasks import ReceiverReleaseTasks
from v15_support.receiver_sensor import ReceiverMeasurement
from v13_a1.codec import Reference21,EnginePort21
from v13_joint.lifecycle import ReleaseEntry
from v15_support.delivery_evidence import DeliveryLedger
import importlib.util
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
FILES=['v15_support/'+n for n in ('runtime.py','plant.py','bootstrap.py','run_development.py')]
before={n:sha(ROOT/n) for n in FILES}

def run():
    checks=[]
    def ck(n,v,d=None):checks.append(dict(name=n,passed=bool(v),detail=d))
    def source(name):return (ROOT/name).read_text(encoding='utf-8-sig')
    rt=ast.parse(source(FILES[0]));cls=next(n for n in rt.body if isinstance(n,ast.ClassDef) and n.name=='SupportSession')
    methods={n.name:ast.get_source_segment(source(FILES[0]),n) for n in cls.body if isinstance(n,ast.FunctionDef)}
    obs=methods['observe_public'];exe=methods['_execute_joint']
    ck('static_common_sensor_before_receiver_then_execution_attachment',obs.index('self.plant.truth()')<obs.index('self.sensors.sample')<obs.index('self.skin_geometry.acquire')<obs.index('common_sensor_json=')<obs.index('self.receiver_sensor.sample')<obs.index('self._attach_execution_state'))
    ck('static_offline_ledger_not_attached_to_public_observation','delivery_ledger' not in obs and 'delivery_ledger' not in methods['_attach_execution_state'])
    ck('static_no_postcommit_attach_and_one_runtime_commit',exe.count('self.plant.commit_reference21(')==1 and exe.index('self._attach_execution_state')<exe.index('self.controller.step')<exe.index('self.plant.commit_reference21') and '_attach_execution_state' not in exe[exe.index('self.plant.commit_reference21'):])
    events=[];x=np.r_[np.zeros(18),.1,0.,0.];entry=ReleaseEntry(12.,Reference21(x,12.),7)
    p=SimpleNamespace(receiver_release_tasks=ReceiverReleaseTasks(entry,np.zeros(18)),receiver_release_version=7,mode_generation=7,time=12.010,joint_release_entry=entry,phase='RELEASING')
    p.reference_snapshot=lambda:Reference21(x,p.time)
    def graph_update(self,obs,r):events.append('graph');return SupportPlant.update_public_release(self,obs,r)
    p.update_public_release=MethodType(graph_update,p)
    p.engine_snapshot=lambda:EnginePort21(x,p.time,0.)
    def authority():
        events.append('authority');lo=np.full(21,-10.);hi=np.full(21,10.);lo[18],hi[18]=p.receiver_release_tasks.lift_interval(p.reference_snapshot());return lo,hi
    p.reference_authority21=authority
    p.release_proposal=lambda query:p.receiver_release_tasks.proposal(query,np.full(21,-10.),np.full(21,10.))
    def measurement(t,load):
        s=np.array([.01]*3+[.0005]*3)
        return ReceiverMeasurement(t,t-.01,.01,True,np.array([0.,0.,-load,0,0,0]),load,s,3*s,None,'boundary_after_forward',None)
    o=SimpleNamespace(time=p.time,lift=.1,normal=np.ones(5),valid=np.ones(5,dtype=bool),age=.01,auxiliary_pressure=0.,auxiliary_valid=True,auxiliary_age=.01)
    session=SupportSession.__new__(SupportSession);session.plant=p;session.latest_receiver_measurement=measurement(p.time,0.)
    SupportSession._attach_execution_state(session,o);v=p.mode_generation
    ck('actual_scalar_age_adapter_produces_valid_graph_update',o.receiver_release_tasks['phase']=='APPROACH' and o.receiver_release_tasks['hand']['available'])
    ck('graph_updates_before_authority_snapshot',events[:2]==['graph','authority'])
    SupportSession._attach_execution_state(session,o)
    ck('same_tick_repeat_attachment_does_not_advance_generation',p.mode_generation==v)
    ck('public_engine_reference_clock_and_generation_coherent',np.array_equal(o.robot_actuator_target21,x) and np.array_equal(o.robot_reference21,x) and o.robot_control_time_s==p.time and o.mode_generation==v)
    for t in (12.015,12.020):
        p.time=t;o.time=t;session.latest_receiver_measurement=measurement(t,1.);SupportSession._attach_execution_state(session,o)
    ck('fresh_receiver_phase_change_invalidates_execution_generation',p.receiver_release_tasks.phase=='GUARDED_CLEAR' and p.mode_generation==v+1 and o.mode_generation==p.mode_generation)
    class Controller:
        desired=np.zeros(5);debug={}
        def set_task_request(self,r):events.append('set_task')
        def step(self,frame,phase,command):events.append('controller');return SimpleNamespace(reference21=Reference21(frame.robot_reference21,frame.time),mode='release',mode_generation=frame.mode_generation)
        def reconcile(self,r):events.append('reconcile')
    def commit(ref,**kwargs):events.append('commit');return SimpleNamespace()
    p.commit_reference21=commit;session.controller=Controller();session.task_request={}
    n=len(events);SupportSession._execute_joint(session,o,'release');tail=events[n:]
    ck('actual_execute_path_single_commit_no_postcommit_attach',tail==['graph','authority','set_task','controller','commit','reconcile'],tail)
    controller=source('v15_support/controller.py')
    ck('controller_source_replans_on_generation_change','if self.plan_generation!=generation:self.plan_start=None;self.last_decision=-np.inf' in controller)
    # Actual ledger + actual adapter override, patched BASE contact reader only.
    spec=importlib.util.spec_from_file_location('runtime_ledger_fake_defs',ROOT/'review/verify_v15_delivery_evidence_author_v1.py');fake=importlib.util.module_from_spec(spec);spec.loader.exec_module(fake)
    model=fake.Model();model.opt=SimpleNamespace(timestep=.0005);data=fake.Data([],0.)
    ledger=DeliveryLedger(model,target_geom_ids=[3,4],hand_geom_ids=[5],receiver_mount=fake.mount(),diagnostic_window_s=(0.,.064),contact_force_reader=fake.reader)
    holder=SupportPlant.__new__(SupportPlant);holder.model=model;holder.data=data;holder.delivery_ledger=ledger;holder._delivery_inside_advance=True;holder._delivery_last_end=0.
    roundtrip=[]
    with patch.object(CoordinatedPlant,'_contacts',lambda self,patches=True:{'mock_base_contacts':True}):
        for k in range(128):
            previous=data.time;data.time+=.0005
            if data.time-.0005<previous:roundtrip.append(dict(tick=k+1,previous_end=previous,end=data.time,old_reconstructed_start=data.time-.0005))
            holder._contacts(False);holder._contacts(True)
        # Same-end tail forward/truth and outside-advance observation add no interval.
        holder._contacts(False);holder._delivery_inside_advance=False;data.time+=.0005;holder._contacts(False)
    arrays=ledger.arrays();summary=ledger.summary()
    ck('actual_ledger_128_contiguous_double_clocks_no_overlap_failure',len(arrays['interval_end_s'])==128 and np.array_equal(arrays['interval_start_s'][1:],arrays['interval_end_s'][:-1]) and summary['window_coverage_complete'])
    ck('old_end_minus_dt_roundtrip_counterexample_preserved',len(roundtrip)>0,roundtrip)
    ck('same_end_and_outside_advance_do_not_double_capture',summary['records']==128)
    ck('ledger_capture_phase_is_cached_not_recomputed_end_geometry',summary['capture_phase']=='cached_contact_solve_at_integration_end_state' and 'not guaranteed same instantaneous' in summary['object_state_time_role'])
    ck('no_v15_writer_override_or_new_physics_loop','def _apply_ctrl' not in source('v15_support/plant.py') and 'mj_step' not in source('v15_support/plant.py') and 'mj_forward' not in source('v15_support/plant.py'))
    return dict(passed=all(c['passed'] for c in checks),checks=checks,scope_limits=['Runtime observe order and information boundary are source checks; no actual sensor noise/physical prefix reproduction performed.',
        'Actual attach/execute methods use a strict public fake holder and mock controller/commit; actual full robot writer was not rerun.',
        'Actual ledger and SupportPlant._contacts use fake data and patched base _contacts; only double arithmetic clocks advance, no actual plant.advance or physics called.',
        'Current mount is source-declared half-height tolerance and cos(pi/4) normal coverage, not the narrow prototype fixture tolerance; blind physical validation remains required.'])
if __name__=='__main__':
    out=ROOT/'review/v15_runtime_bridge_independent_v1.json'
    if out.exists():raise FileExistsError(out)
    try:r=run()
    except Exception as e:r=dict(passed=False,error=repr(e),traceback=traceback.format_exc())
    after={n:sha(ROOT/n) for n in FILES};r.update(schema='v15_runtime_bridge_independent_v1',source_sha256=before,source_unchanged=before==after,
        source_after=after,checker_sha256=sha(__file__),physics_steps=0,model_constructions=0,real_plant_advances=0)
    r['passed']=r['passed'] and r['source_unchanged'];out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(passed=r['passed'],checks=len(r.get('checks',[])),failed=[c['name'] for c in r.get('checks',[]) if not c['passed']],sha256=sha(out),path=str(out)),ensure_ascii=False))
    raise SystemExit(0 if r['passed'] else 1)

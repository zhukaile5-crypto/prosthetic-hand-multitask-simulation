"""Small actual JointPlant mode/commit oracles; static model, no plant construction."""
from pathlib import Path
from types import SimpleNamespace as NS,MethodType
import ast,hashlib,inspect,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/v13_joint_mode_bridge_independent_v3.json'
PINS={'v13_joint/plant.py':'d39c082d4fb5d1b6a59b2db0243045fa1b508144f6f01077137c32bb609f2de4',
      'v13_joint/design.py':'7692990276ea905ac02529d3337d071403dd3e48efcdb68715a552d607de690e',
      'v13_joint/runtime.py':'e2948eb41a6014085b65ecb3ffaf6d0153f52ccfca4f68a46bcbf618cb90d746',
      'v13_joint/controller.py':'b9981e8fec70d9fcaf0cb368f89ada8d0e9396a9391b700030eb4fcadc1f9663',
      'v13_joint/prepared_value_effort21.py':'5b95ec9f1614415ec75b91ecb02629f4316d844960b6111932ffb8d1c76f81ac'}

def main():
    names=tuple(PINS)+('v13_joint/controller.py','v13_joint/runtime.py','v13_joint/bootstrap.py','v13_joint/lifecycle.py','v13_a1/codec.py','v13_joint/native_effort21.py','v13_joint/prepared_native_effort21.py','v13_joint/prepared_value_effort21.py','v11/robot_servo_effort.py')
    saved={n:(ROOT/n).read_bytes() for n in names}
    assert all(hashlib.sha256(saved[n]).hexdigest()==s for n,s in PINS.items())
    sys.path.insert(0,str(ROOT));import research_imports,mujoco
    old={};calls={}
    for n in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_fwdPosition','mj_fwdVelocity','mj_kinematics','mj_comPos','mj_camlight','mj_collision','mj_contactForce'):
        if hasattr(mujoco,n):
            old[n]=getattr(mujoco,n);calls[n]=0
            def forbidden(*a,_name=n,**k):calls[_name]+=1;raise AssertionError('forbidden '+_name)
            setattr(mujoco,n,forbidden)
    import hand_plant
    from foundation_config import FoundationConfig,OPEN,CLOSED
    from v13_a1.codec import Reference21,ReferenceActor21
    from v13_joint.plant import JointPlant
    from v13_joint.controller import JointController,JointAction
    from v13_joint.runtime import JointSession
    from v13_joint.native_effort21 import NativeEffort21
    from v13_joint.prepared_value_effort21 import PreparedValueEffort21
    from v13_joint.design import reference_rates
    legacy=hand_plant.HandPlant._apply_ctrl;advance=hand_plant.HandPlant.advance;init=hand_plant.HandPlant.__init__
    bypass=[];advances=[];constructors=[]
    def no_legacy(*a,**k):bypass.append(True);raise AssertionError('legacy writer bypass')
    def no_advance(*a,**k):advances.append(True);raise AssertionError('physics advance forbidden')
    def no_construct(*a,**k):constructors.append(True);raise AssertionError('HandPlant constructor forbidden')
    hand_plant.HandPlant._apply_ctrl=no_legacy;hand_plant.HandPlant.advance=no_advance;hand_plant.HandPlant.__init__=no_construct
    checks=[];details={}
    def check(n,v,d=None):checks.append(dict(name=n,passed=bool(v),detail=d))
    def rejects(fn):
        try:fn();return False
        except (ValueError,TypeError,RuntimeError):return True
    try:
        config=FoundationConfig();model,xml_sha=hand_plant._model(config)
        class Buffer:
            def __init__(self):self.values=np.zeros(model.nu);self.writes=[]
            @property
            def shape(self):return self.values.shape
            def copy(self):return self.values.copy()
            def __getitem__(self,key):return self.values[key]
            def __setitem__(self,key,value):
                code=inspect.currentframe().f_back.f_code
                robot=code is ReferenceActor21.apply.__code__
                env=code is JointPlant._apply_ctrl.__code__ and np.isscalar(key) and int(key)==obstacle
                if not (robot or env):raise AssertionError('direct writer bypass')
                self.writes.append('robot21' if robot else 'environment1');self.values[key]=value
        actor=ReferenceActor21(model);obstacle=next(i for i in range(model.nu) if i not in set(actor.indices21))
        open18=np.concatenate(OPEN);closed18=np.concatenate(CLOSED)
        initial=np.r_[open18+.30*(closed18-open18),.08,.02,-.03]
        def holder():
            p=JointPlant.__new__(JointPlant);p.model=model;p.config=config;p.data=NS(ctrl=Buffer(),time=987.)
            p.control_steps=2400;p.actor21=ReferenceActor21(model);p.a1_actor=p.actor21
            p.open_pose=tuple(np.asarray(x).copy() for x in OPEN);p.closed_pose=tuple(np.asarray(x).copy() for x in CLOSED)
            p.actuator_groups=tuple(p.actor21.indices21[a:b] for a,b in ((0,5),(5,8),(8,11),(11,14),(14,18)))
            p.alpha=np.full(5,.9);p.lift_ref=.08;p.wrist_ref=initial[19:].copy();p.phase='HOLDING'
            p.release_started=None;p.release_thumb=None;p.release_alpha=None;p.release_lift=0.;p.release_ctrl=None
            p.mode_generation=1;p._joint_mode='holding';p._joint_committed=Reference21(initial,p.time)
            p._joint_anchor_time=None;p._joint_anchor_reference=None;p.joint_release_entry=None;p.last_execution_receipt21=None
            p.actor_callback_total_ns=0;p.actor_callback_calls=0;p.obstacle_motor=obstacle;p.env={'obstacle_x':.5}
            p.data.ctrl.values[p.actor21.indices21]=p.actor21.encode(p._joint_committed).values
            return p
        p=holder();original=p.reference_snapshot().values.copy()
        check('release_commit_requires_begin_release',rejects(lambda:p.commit_reference21(Reference21(original,p.time),mode='release',expected_generation=1)) and not p.data.ctrl.writes)
        check('wrong_issue_clock_or_bool_generation_rejected',rejects(lambda:p.commit_reference21(Reference21(original,p.time+.001),mode='holding',expected_generation=1)) and rejects(lambda:p.commit_reference21(Reference21(original,p.time),mode='holding',expected_generation=True)))
        target=original+.2;target[18:]=[.1,.2,.1]
        a=p.commit_reference21(Reference21(target,p.time),mode='holding',expected_generation=1)
        step=reference_rates('holding')*config.control_dt
        b=p.commit_reference21(Reference21(target+10.,p.time),mode='holding',expected_generation=1)
        check('same_tick_repeat_cannot_stack_net_reference_rate',np.all(abs(b.executed.values-original)<=step+1e-15) and np.array_equal(a.executed.values,b.executed.values))
        check('one_robot21_and_one_environment_write_per_commit',p.data.ctrl.writes==['robot21','environment1']*2)
        actual=b.executed.values.copy();old_generation=p.mode_generation;p.begin_release();entry=p.joint_release_entry
        check('release_entry_is_actual_independent21_not_alpha',np.array_equal(entry.reference21.values,actual) and entry.mode_generation==old_generation+1 and p.phase=='RELEASING' and not np.allclose(actual[:18],open18+.9*(closed18-open18)))
        again_gen=p.mode_generation;p.begin_release();check('begin_release_idempotent',p.mode_generation==again_gen)
        stale=p.commit_reference21(Reference21(target+100.,p.time),mode='holding',expected_generation=old_generation)
        check('old_holding_plan_preempted_by_release_generation',stale.stale_plan_replaced and stale.mode=='release' and stale.mode_generation==p.mode_generation and stale.execution_changed)
        check('legacy_alpha_and_motion_writers_rejected_after_commit',rejects(lambda:p.set_closure(np.zeros(5))) and rejects(lambda:p.set_motion(.05,[0.,0.])))
        # A second fixture moves faster toward open than the unmodified release path.
        r=holder();r.begin_release();records=[]
        direction=np.sign(open18[5:]-initial[5:18]);previous=r.reference_snapshot().values.copy()
        for k in range(1,21):
            r.control_steps=2400+k;want=r.reference_snapshot().values.copy();want[5:18]=open18[5:]
            got=r.commit_reference21(Reference21(want,r.time),mode='release',expected_generation=r.mode_generation)
            movement=got.executed.values[5:18]-previous[5:18]
            records.append(bool(np.all(movement*direction>=-1e-15) and np.all(abs(got.executed.values-previous)<=reference_rates('release')*config.control_dt+1e-15)))
            previous=got.executed.values.copy()
        check('release_nonthumb_actual_progress_monotone_and_rate_bounded',all(records))
        current=r.reference_snapshot().values.copy();lo,hi=r.reference_authority21();raw=r.release_proposal(r.time+.020)
        raw_outside=(raw.values<lo)|(raw.values>hi);check('raw_release_nominal_not_retargeted_to_mode_corridor',np.any(raw_outside[5:18]),dict(raw_non_thumb=raw.values[5:18].tolist(),actual_non_thumb=current[5:18].tolist(),outside_mode_axes=np.flatnonzero(raw_outside).tolist()))
        # Real fixed-q native domain preserves p0 of that raw original target.
        native=NativeEffort21(model);q=model.qpos0[native.servo.qadr].copy();v=np.zeros(25);nominal=Reference21(raw.values,r.time)
        d=native.build_domain(q25=q,v25=v,robot_axes25=native.robot_axes25,time_s=r.time,current_reference=r.reference_snapshot(),nominal_reference=nominal,actual_engine=r.engine_snapshot(),reference_lower21=lo,reference_upper21=hi,step_lower21=-reference_rates('release')*.020,step_upper21=reference_rates('release')*.020)
        expected=native.servo.evaluate(q,v,native.actor.encode(nominal).values,time_s=r.time)
        check('native_domain_keeps_original_release_nominal_p0',d.available and d.p0_effort21.tobytes()==expected.predicted_effort21.tobytes() and np.any((nominal.values<d.reference_lower21)|(nominal.values>d.reference_upper21)))
        reverse=current.copy();reverse[5:18]=initial[5:18];before=current.copy()
        back=r.commit_reference21(Reference21(reverse,r.time),mode='release',expected_generation=r.mode_generation)
        check('reverse_release_request_clipped_with_honest_receipt',np.array_equal(back.executed.values[5:18],before[5:18]) and back.execution_changed and back.requested.values.tobytes()==reverse.tobytes())
        encoded=r.engine_snapshot().values.copy();r.data.time+=100.;r._apply_ctrl()
        check('substep_reapply_holds_public_command_and_lift_bias_once',r.engine_snapshot().values.tobytes()==encoded.tobytes() and r.engine_snapshot().values[18]==r.reference_snapshot().values[18]+r.actor21.lift_bias_m)
        # Actual step logic, with only its expensive decision body replaced by a spy.
        def control_fixture():
            c=JointController.__new__(JointController);c.state=NS(_last_time=1.,time=1.);c.task_request={'mode':'hold'};c.release_latched=False
            c.plan_generation=1;c.plan_start=1.;c.plan_reference21=initial.copy();c.plan_delta21=np.zeros(21);c.last_decision=1.;c.debug={'mode':'holding'};c.last_debug={};c.motion_reference=np.zeros(3)
            decisions=[]
            def decide(self,frame,phase,mode,cur,generation):
                decisions.append((mode,phase,generation));self.plan_start=frame.time;self.plan_reference21=cur.copy();self.plan_delta21=np.zeros(21);self.plan_generation=generation;self.last_decision=frame.time
            c._decide=MethodType(decide,c)
            frame=NS(time=1.,dt=.005,phase='RELEASING',robot_reference21=initial.copy(),mode_generation=2,reference_lower21=np.full(21,-10.),reference_upper21=np.full(21,10.))
            return c,frame,decisions
        c,f,decisions=control_fixture();act=c.step(f,phase='RELEASING',command='hold')
        check('phase_only_release_preempts_old_holding_decision_immediately',act.mode=='release' and decisions==[('release','RELEASING',2)])
        # Reuse a tactile packet after same-public-clock holding commit -> release.
        # Only public executed command memory may refresh; no extra tactile draw.
        coherent=holder();reuse=NS(time=coherent.time,normal=np.arange(5.),sample_time=np.full(5,11.99))
        session=JointSession.__new__(JointSession);session.plant=coherent
        session._attach_execution_state(reuse);old_engine=reuse.robot_actuator_target21.copy();normal_before=reuse.normal.copy();sample_before=reuse.sample_time.copy()
        changed=coherent.reference_snapshot().values.copy();changed[6]+=.1
        coherent.commit_reference21(Reference21(changed,coherent.time),mode='holding',expected_generation=coherent.mode_generation);coherent.begin_release()
        fresh_ref=coherent.reference_snapshot();solver=PreparedValueEffort21(model)
        slo,shi=coherent.reference_authority21();nom=Reference21(coherent.release_proposal(coherent.time+.020).values,coherent.time)
        from v13_a1.codec import EnginePort21
        stale_domain=solver.build_domain(q25=q,v25=v,robot_axes25=solver.robot_axes25,time_s=coherent.time,current_reference=fresh_ref,nominal_reference=nom,actual_engine=EnginePort21(old_engine,coherent.time,solver.actor.lift_bias_m),reference_lower21=slo,reference_upper21=shi)
        check('old_same_clock_engine_snapshot_is_explicitly_rejected',not stale_domain.available and stale_domain.reason=='actual_engine_is_not_exact_current_reference_encode')
        session._attach_execution_state(reuse)
        check('attach_refreshes_coherent_engine_reference_mode_snapshot',reuse.robot_actuator_target21.tobytes()==coherent.engine_snapshot().values.tobytes() and reuse.robot_actuator_target21.tobytes()!=old_engine.tobytes() and reuse.robot_control_time_s==coherent.time and reuse.mode_generation==coherent.mode_generation and np.array_equal(reuse.robot_reference21,fresh_ref.values))
        refreshed_domain=solver.build_domain(q25=q,v25=v,robot_axes25=solver.robot_axes25,time_s=coherent.time,current_reference=Reference21(reuse.robot_reference21,reuse.time),nominal_reference=nom,actual_engine=EnginePort21(reuse.robot_actuator_target21,reuse.robot_control_time_s,solver.actor.lift_bias_m),reference_lower21=reuse.reference_lower21,reference_upper21=reuse.reference_upper21)
        check('refreshed_same_clock_release_domain_is_known',refreshed_domain.available,dict(old_reason=stale_domain.reason,new_status=refreshed_domain.status))
        check('execution_refresh_adds_no_tactile_noise_or_acquisition_change',np.array_equal(reuse.normal,normal_before) and np.array_equal(reuse.sample_time,sample_before))
        # Actual runtime attach -> step -> commit -> reconcile path, with a strict fake controller.
        s=JointSession.__new__(JointSession);s.plant=r;s.task_request={'mode':'release'};seen=[]
        class Controller:
            desired=np.zeros(5);debug={}
            def set_task_request(self,value):seen.append('task');assert value is s.task_request
            def step(self,frame,phase,command):
                seen.append('step');assert frame.mode_generation==r.mode_generation and np.array_equal(frame.robot_reference21,r.reference_snapshot().values)
                assert np.array_equal(frame.release_nominal_reference21,r.release_proposal(frame.time+.020).values)
                return JointAction(Reference21(frame.robot_reference21,frame.time),'release',frame.mode_generation,{})
            def reconcile(self,receipt):seen.append('receipt');assert receipt.engine.values.tobytes()==r.engine_snapshot().values.tobytes()
        s.controller=Controller();frame=NS(time=r.time);des,pred,debug=s._execute_joint(frame,'release')
        check('runtime_actual_full21_attach_commit_reconcile_route',seen==['task','step','receipt'] and np.isnan(pred).all() and np.array_equal(des,np.zeros(5)))
        # Small static routing/timing assertions; no actual whole-loop or source-identity claim.
        rt=ast.parse(saved['v13_joint/runtime.py'].decode('utf-8-sig'));ct=ast.parse(saved['v13_joint/controller.py'].decode('utf-8-sig'))
        rttext=saved['v13_joint/runtime.py'].decode('utf-8-sig').replace('\r\n','\n');cttext=saved['v13_joint/controller.py'].decode('utf-8-sig').replace('\r\n','\n')
        execute=next(n for n in ast.walk(rt) if isinstance(n,ast.FunctionDef) and n.name=='_execute_joint')
        attrs=[n.func.attr for n in ast.walk(execute) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        check('runtime_no_legacy_alpha_or_motion_overwrite_in_joint_route','commit_reference21' in attrs and not any(n in attrs for n in ('set_closure','set_motion','execute_delta')))
        check('decision_candidates_saved_once_to_separate_gzip','joint_decisions.jsonl.gz' in rttext and 'finite_candidate_solutions' not in cttext and 'reference_increment21' in rttext)
        check('failed_handoff_retains_failure_but_labels_active_release','self.last_command=\'release\'\n                    desired,prediction,debug=self._lifecycle_step(obs);controller_active=True' in rttext and 'holding_policy_not_executed=True' in rttext and 'debug.update(policy_not_executed=True' not in rttext)
        binding_file=ROOT/'review/v13_joint_runtime_root_zero_v5.json';binding=json.loads(binding_file.read_text(encoding='utf-8'))
        check('read_version_matches_root_declared99_source_binding',len(binding['source_sha256'])==99 and all(binding['source_sha256'][n]==hashlib.sha256(saved[n]).hexdigest() for n in saved))
        details=dict(monotonic_ticks=len(records),runtime_spy_route=seen,actual_native_domain_status=d.status,read_controller_native_import=next(line.strip() for line in cttext.splitlines() if ' import ' in line and 'effort21' in line),root_declared_source_binding_count=len(binding['source_sha256']),root_binding_sha256=hashlib.sha256(binding_file.read_bytes()).hexdigest(),source_line_ranges={'plant_mode_domain':[41,59],'plant_raw_release':[73,78],'plant_commit':[80,111],'controller_release':[162,183],'runtime_route':[267,288],'runtime_timing':[335,493]})
    finally:
        hand_plant.HandPlant._apply_ctrl=legacy;hand_plant.HandPlant.advance=advance;hand_plant.HandPlant.__init__=init
        for n,fn in old.items():setattr(mujoco,n,fn)
    stable={n:(ROOT/n).read_bytes()==v for n,v in saved.items()}
    check('pinned_plant_design_and_all_read_sources_stable',all(stable.values()),stable)
    check('no_plant_constructor_advance_dynamics_FK_contact_or_old_writer',not any(calls.values()) and not advances and not constructors and not bypass)
    report=dict(schema='v13_joint_mode_bridge_independent_v3',role='independent_small_actual_mode_writer_bridge_oracles',passed=all(x['passed'] for x in checks),count=len(checks),checks=checks,
                source_sha256={n:hashlib.sha256(v).hexdigest() for n,v in saved.items()},checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_stable=stable,
                static_model_compile_count=1,static_xml_sha256=xml_sha,physics_steps=0,plant_constructors=len(constructors),advance_calls=len(advances),forbidden_calls=calls,legacy_writer_calls=len(bypass),details=details,
                prior_read_version=dict(file='review/v13_joint_mode_bridge_independent_v1.json',sha256=hashlib.sha256((ROOT/'review/v13_joint_mode_bridge_independent_v1.json').read_bytes()).hexdigest(),runtime_sha256='abb9d01f522337be2009e2fae2905446e8789d6e4e45f18156c4ec4a64d1a4d2',scope='Preserved prior 19-case artifact; current v2 adds final coherent21 refresh and corrected failure exposure labels.'),
                prior_harness_false_negative=dict(file='review/v13_joint_mode_bridge_independent_v2.json',sha256=hashlib.sha256((ROOT/'review/v13_joint_mode_bridge_independent_v2.json').read_bytes()).hexdigest(),passed=24,total=25,reason='A source-substring assertion used LF against CRLF bytes; normalize line endings only for the textual assertion, production and hashes unchanged.'),
                limitations=['JointPlant methods and actor executed against a real static robot model with tracked fake22 buffer; no HandPlant constructor, integration, contact or FK.',
                  'Controller release scheduling and runtime glue use strict spies; parent independently owns real archived public-packet controller/native coupling tests.',
                  'Only selected mode/clock/rate/preemption paths; no full stream/old955 duplication, performance, task success, pressure prediction or safety certification.',
                  'full_control_loop_ms covers observe_public through trace/history completion; physics_advance_ms is separate. Actor callback timing overlaps loop/physics and must not be added twice.',
                  'Root99 source map is version binding only, not used as proof of this independent gate; no separate all99 dependency rebuild or old955/full packet repetition.',
                  'The final failed-handoff runtime branch now labels controller_active true, command release and holding_policy_not_executed; original preparation failure is retained.'])
    assert not OUT.exists();OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),passed=report['passed'],checks=len(checks),failed=[x['name'] for x in checks if not x['passed']],controller_sha=report['source_sha256']['v13_joint/controller.py']),ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()

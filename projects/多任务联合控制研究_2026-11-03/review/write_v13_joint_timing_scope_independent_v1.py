"""Source-only timing contract. Never imports or reads a running trial."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/v13_joint_timing_scope_independent_v1.json'
PINS={
 'v13_joint/runtime.py':'e2948eb41a6014085b65ecb3ffaf6d0153f52ccfca4f68a46bcbf618cb90d746',
 'v13_joint/run_development.py':'e055d576690331e4bdc15c3b2af43677e5cdc25ed5f4366956fa9b850f0da1c5',
 'v13_joint/controller.py':'b9981e8fec70d9fcaf0cb368f89ada8d0e9396a9391b700030eb4fcadc1f9663',
 'v13_joint/plant.py':'d39c082d4fb5d1b6a59b2db0243045fa1b508144f6f01077137c32bb609f2de4',
 'v13_joint/design.py':'7692990276ea905ac02529d3337d071403dd3e48efcdb68715a552d607de690e'}

def main():
    payloads={n:(ROOT/n).read_bytes() for n in PINS}
    assert all(hashlib.sha256(v).hexdigest()==PINS[n] for n,v in payloads.items())
    text={n:v.decode('utf-8-sig').replace('\r\n','\n') for n,v in payloads.items()}
    trees={n:ast.parse(v) for n,v in text.items()}
    def loc(file,first,last=None):return dict(file=str(ROOT/file),line=first,end_line=first if last is None else last)
    rt='v13_joint/runtime.py';ct='v13_joint/controller.py';pl='v13_joint/plant.py';rr='v13_joint/run_development.py'
    functions={n:{node.name:dict(start=node.lineno,end=node.end_lineno) for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))} for n,tree in trees.items()}
    checks=[]
    def check(n,v):checks.append(dict(name=n,passed=bool(v)))
    check('final_loop_value_overwrites_provisional_row_value',text[rt].count("'full_control_loop_ms'")==1 and 'full_control_loop_ms=(time.perf_counter_ns()-loop_began)/1e6' in text[rt])
    check('physics_and_actor_fields_updated_after_advance',text[rt].index("self.trace[-1]['physics_advance_ms']=(")>text[rt].index('self.plant.advance(SENSOR_PERIOD)') and text[rt].rindex("self.trace[-1]['actor_callback_ms']=(")>text[rt].index('self.plant.advance(SENSOR_PERIOD)'))
    check('decision_evidence_clock_and_trace_debug_id_exist',all(s in text[ct] for s in ('decision_id=len(self.decision_evidence)','time_s=now','decision_time_s=now','nominal_target_time_s=now+DECISION_PERIOD_S')))
    check('decision_can_restart_on_mode_and_release',all(s in text[ct] for s in ('if self.plan_generation!=generation:','if mode==\'release\' and self.debug.get(\'mode\')==\'holding\':','now>=self.last_decision+DECISION_PERIOD_S-1e-9')))
    check('terminal_has_no_physics_advance',text[rt].index('if t>=DURATION-1e-9:')<text[rt].index('self.plant.advance(SENSOR_PERIOD)'))
    scopes={
      'observation_ms':dict(begin=loc(rt,242),end=loc(rt,263),export=loc(rt,265),
        includes=['truth/raw sensor query, sensors.sample, delayed skin acquisition','common sensor JSON serialization','legacy model-field removal, public robot telemetry/spec construction and first _attach_execution_state'],
        excludes=['event_schedule and task construction','preparation/observer warmup, handoff, controller.step and commit','post-control trace/history, environment application, physical integration, save/evaluate'],
        note='Measurement construction wall cost includes evaluation-side truth/raw acquisition and is not the controller inference cost alone.'),
      'control_ms':dict(begin=loc(rt,366),end=loc(rt,428),export=loc(rt,447),
        includes=['actual selected preparation / early-release / handoff / holding-release / failed-handoff release branch','preparation step and 0.2s observer warmup when selected','JointController.step, 20ms decision when due, 5ms interpolation/guard, commit and reconcile; release begin/event when selected','second coherent public execution-state attachment inside _execute_joint'],
        excludes=['observe_public including first execution-state attachment','pre-branch environment schedule, public trajectory and task request assembly through line364','post-branch trace/JSON/history, set_environment, integration, last truth/snapshot, save/evaluate'],
        note='Recorded on every5ms sensor row, including preparation/nondecision rows; not synonymous with one QP or one20ms decision.'),
      'full_control_loop_ms':dict(begin=loc(rt,337),provisional=loc(rt,453),authoritative_end_and_overwrite=loc(rt,482),
        includes=['clock check and complete observe_public','environment schedule selection/signature/event messages and public task assembly','all control_ms branch work','trace row construction including debug/raw patch/observation/task JSON, array copies, trace append and history append'],
        excludes=['set_environment line488','physical integration and tick increment lines489–491','last_truth refresh line493 and returned snapshot line494','terminal completion event/state update lines485–486','session.save/NPZ+gzip, source identity, evaluation and process startup'],
        note='Use stored final overwrite at482, not provisional dict expression at453. Full means sensing-to-trace loop before physics, not end-to-end function/process wall time.'),
      'physics_advance_ms':dict(begin=loc(rt,489),end=loc(rt,491),
        includes=['plant.advance(SENSOR_PERIOD), its substep writer callbacks and internal integration/recording','self.tick += 1 on the same line after advance'],
        excludes=['set_environment immediately before timer','subsequent plant.truth and session.snapshot','all preceding sensing/control/trace work'],
        terminal=dict(location=loc(rt,483,490),value_ms=0.,meaning='t=16s row writes a command but has no following integration interval')),
      'actor_callback_ms':dict(accumulator_begin=loc(pl,114),accumulator_end=loc(pl,121),baseline=loc(rt,338),provisional=loc(rt,484),final_nonterminal=loc(rt,492),
        includes=['sum of JointPlant._apply_ctrl durations since tick baseline','A1 preparation callback when not committed, or sole actor21.apply plus separate environment-port write when committed','both control-branch callback(s) and callbacks during following physics advance'],
        excludes=['outer controller/QP/native computations before callback','noncallback physics and general trace/observation work'],
        overlap='Subset overlapping both full_control_loop_ms and physics_advance_ms; do not add to either or their sum. Not an independent stage latency.'),
      'debug.solver_timing_ms.qp':dict(begin=loc(ct,130),end=loc(ct,132),includes=['solve_intent_qp call only'],absent_or_not_called_ms=0.),
      'debug.solver_timing_ms.native_inverse':dict(begin=loc(ct,136),end=loc(ct,138),includes=['native.solve including finite candidates and PreparedValue final full-F067 recheck'],excludes=['native.build_domain at101 and all preceding prior/geometry work'],absent_or_not_called_ms=0.),
      'debug.solver_timing_ms.whole_decision':dict(begin=loc(ct,84),end_expression=loc(ct,160),
        includes=['public q/v and patch/FK preparation, force-intent/grouping, current21/domain construction','normalization, QP if selected, native finite inverse and final recheck','decision evidence conversion/append and debug field preparation up to timing expression'],
        excludes=['step interpolation and final5ms clipping lines175–179','last_debug/action deepcopy after decision','plant commit and controller reconcile','outer public observation and task, trace serialization and physics'],
        repeated='This debug value is held across later5ms rows until the next decision; do not sum over raw rows without decision-ID deduplication.')}
    clock_contract=dict(control_period_s=.005,decision_period_s=.020,source=loc('v13_joint/design.py',8,9),
      trace_time=dict(field='time',meaning='public control-clock t sampled before current control and before following physical advance',source=loc(rt,339,341)),
      identification=dict(trace='debug_json.decision_id + decision_time_s; compare with row.time and deduplicate within each trial',decision_file='joint_decisions.jsonl.gz',decision_record_fields=['decision_id','time_s','mode','generation','nominal_target_time_s','domain','native','qp'],save_source=loc(rt,290,297),
        note='JSONL contains one full candidate/evidence record per _decide; solver timing itself is exported in trace debug, not explicitly in that JSONL record.'),
      schedule='Fresh decisions when now reaches last_decision+.020; generation mismatch or holding→release resets last_decision. Do not assume global row_index%4 under mode changes.',
      interpolation=dict(source=loc(ct,175,179),formula='fraction=clip((now-plan_start+.005)/.020,0,1)',normal_quarters=[.25,.5,.75,1.],note='These are requested plan fractions. Actual commit/rate/mode receipt decides the executed21; no new20ms optimality claim at intermediate q/v.'),
      terminal='At t=16s a real computation/port write can be recorded, but no subsequent physical exposure is observed; include its computation cost if aggregating actual decision costs and separate it from action-outcome intervals.')
    receipt_contract=dict(source=loc(pl,80,111),
      requested_reference21=dict(trace_source=loc(rt,448),meaning='Original typed requested reference before plant stale-mode replacement and late authority/rate clip; may be reused previous receipt only on a row without a new commit.'),
      executed_reference21=dict(trace_source=loc(rt,449),meaning='Current committed21 after actual plant mode/rate/authority processing; not legacy alpha.'),
      reference_increment21=dict(trace_source=loc(rt,450),meaning='Current post-control committed21 minus pre-control obs.robot_reference21; actual command-memory delta, not measured q25 displacement.'),
      engine_before_and_after=dict(before='robot_actuator_target21 refreshed with current command before controller.step',before_source=loc(rt,267,277),after='ctrl full model22 row after control; engine21 can be selected by named robot ports',after_source=loc(rt,445)),
      status=dict(debug_fields=['actual_reference21','execution_changed_plan','stale_plan_replaced'],source=loc(ct,185,190),
        meaning='Changed flag is any requested-vs-executed byte difference, not a unique reason classifier. Stale flag records replacement. Receipt mode is not a standalone trace column; debug.mode is controller-selected plan mode, not proof of a replaced receipt mode.'),
      generation=dict(trace_source=loc(rt,451),meaning='Current plant mode generation after commit; separate from phase, tactile sample time and wall-clock latency.'),
      limited_logging='Trace stores selected receipt values/flags, not a complete serialized ExecutionReceipt21 or per-axis late-clip-reason mask. Decisions JSONL is proposal evidence and is not an execution receipt.')
    runner_scope=dict(
      per_trial_wall_seconds=dict(begin=loc(rr,53),end=loc(rr,67),includes=['worker source identity, model/session construction','all sensor ticks and physical integration','session save including decision gzip, trace evaluation and research evaluation','end-of-worker source identity'],excludes=['worker startup/import and scheduler queue before run_one','result.json write at68','batch preflight and dispatcher work'],failure_note='runtime_error.wall_seconds captured at72 before partial-save recovery; failure cleanup/save time is not included.'),
      batch_wall_seconds=dict(begin=loc(rr,142),end=loc(rr,177),includes=['three method zero-step preflights','ProcessPool startup/dispatch/parallel waits, per-job console progress','finally writes for trials/failures, end source/environment checks, assignment ledger and finish construction up to expression'],excludes=['admission/registration checks, source hashing/snapshots, manifest/environment writes before142','execution_complete.json write and final print at179–180']),
      comparison='Parallel worker wall times reflect contention/configuration; per-job and batch wall times have different scopes. Neither is a20ms real-time feasibility certificate.')
    report=dict(schema='v13_joint_timing_scope_independent_v1',role='source_only_independent_timing_and_receipt_contract',passed=all(x['passed'] for x in checks),checks=checks,
      source_sha256=PINS,source_locations_are_one_based=True,clock_units=dict(wall_timers='monotonic perf_counter(_ns) elapsed ms/s',trace='simulation/public command seconds; not elapsed host wall time'),
      scopes=scopes,clock_contract=clock_contract,receipt_contract=receipt_contract,runner_scope=runner_scope,parsed_function_ranges=functions,
      read_scope=dict(source_only=True,running_output_or_partial_performance_read=False,controller_or_plant_imports=0,model_constructors=0,physics_steps=0,production_writes=0),
      cautions=['Do not add observation_ms/control_ms to full_control_loop_ms; those intervals overlap.',
                'Do not add actor_callback_ms to full_control_loop_ms or physics_advance_ms; it spans subsets of both.',
                'Do not count repeated solver_timing debug values as new decisions or use control_ms>0 as a decision flag.',
                'No timing results, throughput, real-time, force-safety or performance conclusion is made from this source contract.'])
    stable=all((ROOT/n).read_bytes()==v for n,v in payloads.items());report['source_stable']=stable;assert stable
    report['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert not OUT.exists();OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),passed=report['passed'],checks=len(checks)),ensure_ascii=False))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()

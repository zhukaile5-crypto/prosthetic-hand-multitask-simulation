"""Static bridge comparison inventory; no execution imports/output/model read."""
from pathlib import Path
import ast,argparse,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
WATCHED=('v15_support/runtime.py','v15_support/bootstrap.py','v15_support/plant.py','v15_support/run_development.py',
 'v15_support/controller.py','v15_support/receiver_sensor.py','v15_support/delivery_evidence.py',
 'v14_coord/runtime.py','v14_coord/controller.py','v14_coord/plant.py','v13_joint/plant.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def method(path,cls,name):
    t=ast.parse((ROOT/path).read_text(encoding='utf-8'))
    return next(f for c in t.body if isinstance(c,ast.ClassDef) and c.name==cls for f in c.body if isinstance(f,ast.FunctionDef) and f.name==name)
def obsattrs(f):
    return {x.attr for n in ast.walk(f) if isinstance(n,(ast.Assign,ast.AnnAssign)) for a in (n.targets if isinstance(n,ast.Assign) else [n.target]) for x in ast.walk(a) if isinstance(x,ast.Attribute) and isinstance(x.value,ast.Name) and x.value.id=='obs'}
def rowkeys(f):
    return {k.arg for n in ast.walk(f) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='row' for t in n.targets) and isinstance(n.value,ast.Call) for k in n.value.keywords}
def dump(body):return ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    before={f:sha(ROOT/f) for f in WATCHED}
    old=method('v14_coord/runtime.py','CoordinatedSession','observe_public');new=method('v15_support/runtime.py','SupportSession','observe_public')
    common=[x for x in new.body if not (isinstance(x,ast.Assign) and any(isinstance(t,ast.Attribute) and (t.attr=='latest_receiver_measurement' or (isinstance(t.value,ast.Name) and t.value.id=='obs' and t.attr=='receiver_surface')) for t in x.targets))]
    oldobs=obsattrs(old)|obsattrs(method('v14_coord/runtime.py','CoordinatedSession','_attach_execution_state'))
    newobs=obsattrs(new)|obsattrs(method('v15_support/runtime.py','SupportSession','_attach_execution_state'))
    oldrows=rowkeys(method('v14_coord/runtime.py','CoordinatedSession','step_one_sensor_tick'));newrows=rowkeys(method('v15_support/runtime.py','SupportSession','step_one_sensor_tick'))
    checks=dict(common_observation_call_AST_identical_after_exact_two_receiver_assignments_removed=dump(old.body)==dump(common),
        observation_additions_exact=newobs-oldobs=={'receiver_surface','receiver_release_tasks'},
        trace_additions_exact=newrows-oldrows=={'receiver_measurement_json','receiver_release_tasks_json'},no_old_trace_columns_removed=not(oldrows-newrows),
        watched_sources_unchanged=before=={f:sha(ROOT/f) for f in WATCHED})
    result=dict(schema='v15_blind_bridge_interface_static_v1',static_checks_passed=all(checks.values()),checks=checks,source_sha256=before,
        task_policy_required='legacy',receiver_policy_required='blind',physical_bridge_passed=None,
        top_level_timing_columns=['control_ms','observation_ms','full_control_loop_ms','physics_advance_ms','actor_callback_ms'],
        debug_json_timing_leaf_paths=['/solver_timing_ms/qp','/solver_timing_ms/native_inverse','/solver_timing_ms/whole_decision'],
        debug_comparison_rule='Require matching dictionary structure; replace only the three timing leaf values. Compare all other keys and values, status, costs, native residual and task request unchanged.',
        new_observation_fields=sorted(newobs-oldobs),new_trace_columns=sorted(newrows-oldrows),
        observation_comparison_rule='Remove only receiver_surface and receiver_release_tasks; all old observation content must match. legacy_common_sensor_json is fresh serialization after HandSensors.sample and DelayedSkinGeometry.acquire, not copied source data.',
        full_trace_existing_columns=sorted(oldrows),
        sidecars=dict(joint_decisions='joint_decisions.jsonl.gz: decompressed deterministic decision content; schema v15_single_task_decision_evidence_v1 in joint_decision_manifest.json',
            delivery_intervals='delivery_intervals.npz: privileged interval_start/end/duration, masks/reasons, six exclusive target normal classes/counts, anyactor receiver-top resultant, object free qpos7/qvel6 end',
            delivery_summary='delivery_summary.json: fixed12to16 privileged diagnostics, schema delivery-cached-contact-offline-evaluation-v1',
            delivery_manifest='delivery_manifest.json: schema v15_offline_delivery_evidence_v1, SHA arrays/summary; no prior counterpart to fill as0'),
        receiver_call=dict(path='v15_support/runtime.py:288',phase='boundary_after_forward after existing truth/raw/skin calls',random_stream='new ReceiverSensor private SeedSequence(seed,0x56313552), fixed6Gaussian+1uniform at200Hz; independent from HandSensors',extra_forward_or_sample_old_sensor=False),
        ledger_call=dict(path='v15_support/plant.py:54',phase='inside original advance, first _contacts at each new integration end; _delivery_last_end prevents duplicate append on later contacts/forward truth',extra_step_or_forward=False),
        source_copy_rule='sources/* plus source_snapshot_manifest are byte-provenance only, never independent trajectory/sensor evidence. No source_* physical columns are introduced by this runtime.',
        pinning_rule='run_development requires exact registration source/environment/assignment/software_gate before snapshot; verifies worker source before/after plus final closure. Fresh attempt only, all assignment/error statuses retained.',
        scope='Read-only source/AST interface inventory only. No output directory/NPZ loaded, no module/controller/model import, no replay, physics or service. Root/parent must verify actual frozen assignment and all complete bridge columns.',physics_steps=0,models_constructed=0,output_records_read=0)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(static_checks_passed=result['static_checks_passed'],checks=checks,out=str(out)),ensure_ascii=False))
    if not result['static_checks_passed']:raise SystemExit(1)
if __name__=='__main__':main()

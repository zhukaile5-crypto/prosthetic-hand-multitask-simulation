"""Bind only the fresh V25 zero-v3 output; old complete198 gates are excluded."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];FOUNDATION=ROOT.parent/'五指策略接入_2026-11-03'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

out=ROOT/'output/v25_response_capability_zero_v3'
cap=read(out/'capability_zero.json');complete=read(out/'execution_complete.json');env=read(out/'environment.json')
author_path=ROOT/'review/v25_runtime_author_final_v1.json';author=read(author_path)
identity=complete['source_sha256'];parent=read(ROOT/'review/v34_small_plan_v1.json')['source_sha256'];checks=[]
def ck(name,good):
    checks.append(dict(name=name,passed=bool(good)))
    if not good:raise AssertionError(name)
ck('fresh_v3_capability_and_execution_both_pass',cap['passed'] and complete['passed'] and complete['execution_complete'] and complete['runtime_failures']==[])
ck('fresh_v3_actual_full198_source_identity',len(identity)==198 and identity==cap['production_source_sha256'])
ck('current198_bytes_match_v3_not_v1',all(sha(FOUNDATION/k[11:] if k.startswith('foundation/') else ROOT/k)==v for k,v in identity.items()))
ck('frozen_parent192_actual_identity',len(parent)==192 and parent==complete['parent192_source_sha256'] and all(identity.get(k)==v for k,v in parent.items()))
ck('frozen_parent_plan_prepost_pin',sha(ROOT/'review/v34_small_plan_v1.json')==complete['parent192_registered_plan_sha256'])
ck('v3_source_and_full_numeric_prepost_exact',complete['sources_unchanged'] and complete['numerical_unchanged'] and cap['numerical_identity']==complete['numerical_identity']==env['numerical_identity'])
ck('original42_environment_preserved',env['pinned_environment']==complete['environment']==complete['numerical_identity']['legacy_pinned_environment'] and len(env['pinned_environment']['native_binaries_sha256'])==42)
ck('v3_time_tick_sensor_observer_receipts_all_zero',all(cap[k]==0 for k in ('time_s','tick','sensor_samples','observer_packets','actual_receipts')))
ck('v3_integration_forbidden_and_not_called',cap['physics_steps']==complete['physics_steps']==0)
ck('author31_owned4_exact_current_no_production_rewrite',author['passed'] and len(author['checks'])==31 and all(identity.get(k)==v for k,v in author['source_sha256'].items()))
result=dict(schema='v25_runtime_capability_zero_gate_v3',passed=True,checks=checks,
    production_source_sha256=identity,source_sha256=identity,source_count=len(identity),parent192_source_sha256=parent,
    numerical_identity=complete['numerical_identity'],environment=complete['environment'],
    physics_steps=0,initialization_forward_calls=cap['initialization_forward_calls'],time_s=cap['time_s'],tick=cap['tick'],sensor_samples=cap['sensor_samples'],
    input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (out/'capability_zero.json',out/'execution_complete.json',out/'source_snapshot_manifest.json',out/'environment.json',out/'plan.json',author_path)},
    actual_output=str(out),actual_command='python -B -m v25_response_identification.run_identification --capability-zero --out output/v25_response_capability_zero_v3',
    fork_source_sha256=identity['v25_response_identification/forks.py'],observer_source_sha256=identity['v25_response_identification/observer.py'],
    old_complete198_gates_used=False,full_remaining_fork_bridge_passed=False,
    scope='Actual fresh-v3 passive constructor/198-source/numerical-identity gate only. No fork, populated-sensor regression, sensor acquisition, integration, or remaining16s behaviour claimed. The unchanged author31 gate covers owned4 only.')
target=ROOT/'review/v25_runtime_capability_zero_v3.json';assert not target.exists()
target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(gate=str(target),sha256=sha(target),checks=len(checks),forward_calls=cap['initialization_forward_calls'],fork_source_sha256=result['fork_source_sha256'],observer_source_sha256=result['observer_source_sha256']),ensure_ascii=False))

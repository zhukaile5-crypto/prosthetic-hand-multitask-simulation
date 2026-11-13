"""Register four finite seen-context development trials; dispatch no physics."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from v20_contact_exit.runtime import source_identity
from v13_a1.run_bridge import environment
from research_imports import FOUNDATION


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def unchanged(mapping):
    for name, digest in mapping.items():
        path = FOUNDATION/name[len('foundation/'):] if name.startswith('foundation/') else ROOT/name
        if sha(path) != digest:
            raise ValueError('retained source changed: '+name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--integration-gate', required=True)
    args = ap.parse_args()
    identity = source_identity(['ratio_effort'])
    if len(identity) != 160:
        raise ValueError('expected parent152 plus eight new sources')
    parent = read(ROOT/'review/v19_small_plan_v1.json')['source_sha256']
    ablation = read(ROOT/'review/v19_motion_ablation_plan_v1.json')['source_sha256']
    unchanged(parent)
    unchanged(ablation)
    if len(parent) != 152 or len(ablation) != 156 or any(identity.get(k) != v for k, v in parent.items()):
        raise ValueError('executed parent152 and NoMotion156 must remain exact')
    old = read(ROOT/'review/v17b_prospective80_plan_v1.json')
    env = environment()
    if env != old['environment']:
        raise ValueError('original native environment changed')
    gate = (ROOT/args.integration_gate).resolve()
    paths = [gate, ROOT/'review/v20_geometry_adapter_author_v1.json',
             ROOT/'review/v20_task_solver_author_v1.json']
    for path in paths:
        doc = read(path)
        if not doc.get('passed') or not isinstance(doc.get('source_sha256'), dict):
            raise ValueError('passed source-bound software gate required: '+str(path))
        if any(identity.get(k) != v for k, v in doc['source_sha256'].items()):
            raise ValueError('gate is stale: '+str(path))
    if read(gate)['source_sha256'] != identity:
        raise ValueError('integration gate must bind the complete160 source closure')
    contract = ROOT/'review/V20整体接触退出实施合同_v1.md'
    software = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    definitions = [
        ('passive_ratio_transport', 'nominal', 'transport', 619741, 'legacy', 'blind', 'baselines_transport'),
        ('nominal_transport', 'nominal', 'transport', 619741, 'integrated', 'coordinated', 'full_V17_transport'),
        ('fragile_hold', 'fragile_conflict', 'hold', 619741, 'integrated', 'coordinated', 'full_V17_hold'),
        ('friction_transport', 'friction_drop', 'transport', 619742, 'integrated', 'coordinated', 'full_V17_transport')]
    stages = []
    for name, scene, motion, seed, task, receiver, old_stage in definitions:
        output = 'output/v20_small_'+name+'_v1'
        registration = 'review/v20_small_'+name+'_registration_v1.json'
        if (ROOT/output).exists() or (ROOT/registration).exists():
            raise FileExistsError('retain every attempt; existing '+name)
        original_stage = next(s for s in old['stages'] if s['name'] == old_stage)
        original = ROOT/original_stage['output']/'runs'/f'{scene}__ratio_effort__{seed}'
        if sha(original/'trace.npz') != read(original/'metadata.json')['trace_sha256']:
            raise ValueError('original paired trace identity failed')
        assignment = dict(methods=['ratio_effort'], scenarios=[scene], seeds=[seed],
            motion_mode=motion, physical_randomization=True, task_policy=task, receiver_policy=receiver)
        reg = dict(schema='v20_seen_contact_exit_registration_v1',
            created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=identity,
            environment=env, assignment=assignment, software_gate_passed=True, software_gates=software,
            implementation_contract_sha256=sha(contract),
            scope='seen mechanism development; no tuning, no superiority claim',
            original_same_condition=original.relative_to(ROOT).as_posix(),
            original_trace_sha256=sha(original/'trace.npz'))
        write(ROOT/registration, reg)
        stages.append(dict(name=name, output=output, registration=registration,
            assignment=assignment, original_same_condition=reg['original_same_condition'],
            original_trace_sha256=reg['original_trace_sha256'], planned=1))
    plan = dict(schema='v20_seen_contact_exit_small_plan_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=identity,
        environment=env, parent152_unchanged=True, noMotion156_unchanged=True,
        planned=4, stages=stages, software_gates=software,
        implementation_contract=contract.relative_to(ROOT).as_posix(),
        implementation_contract_sha256=sha(contract),
        execution_order='passive full16 bridge first, then three frozen same-context candidates',
        auxiliary_contract_sha256=sha(ROOT/'review/delivery_auxiliary_contract_v3.json'),
        auxiliary_calculator_sha256=sha(ROOT/'review/delivery_auxiliary_metrics_v3.py'),
        no_parameter_sweep=True, performance_claim_allowed=False,
        evaluation='unchanged original metrics and frozenv3; retain all failures, NULL and unknown')
    write(ROOT/'review/v20_small_plan_v1.json', plan)
    print(json.dumps(dict(registered=4, sources=len(identity), physics_dispatched=False)))


if __name__ == '__main__':
    main()

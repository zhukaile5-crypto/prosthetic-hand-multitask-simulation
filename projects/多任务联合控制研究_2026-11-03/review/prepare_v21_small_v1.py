"""Freeze three same-information arms and nine finite development assignments."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from v21_stable_handoff.runtime import source_identity
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--integration-gate', required=True)
    ap.add_argument('--math-gate', required=True)
    args = ap.parse_args()
    identity = source_identity(['ratio_effort'])
    if len(identity) != 170:
        raise ValueError('expected original160 plus ten independent V21 sources')
    original = read(ROOT/'review/v20_complete16_plan_v1.json')
    if len(original['source_sha256']) != 160 or any(identity.get(k) != v for k, v in original['source_sha256'].items()):
        raise ValueError('executed V20 parent160 changed')
    ablation = read(ROOT/'review/v19_motion_ablation_plan_v1.json')['source_sha256']
    for name, digest in ablation.items():
        path = FOUNDATION/name[len('foundation/'):] if name.startswith('foundation/') else ROOT/name
        if sha(path) != digest:
            raise ValueError('preserved NoMotion156 changed: '+name)
    env = environment()
    if env != original['environment']:
        raise ValueError('original physical/native environment changed')
    gate_paths = [(ROOT/args.integration_gate).resolve(), (ROOT/args.math_gate).resolve(),
        ROOT/'review/v21_prefix_hierarchy_author_v1.json', ROOT/'review/v21_task_solver_author_v1.json',
        ROOT/'review/v21_path_author_v1.json']
    for path in gate_paths:
        gate = read(path)
        if not gate.get('passed') or not isinstance(gate.get('source_sha256'), dict):
            raise ValueError('passed source-bound gate required: '+str(path))
        if any(identity.get(k) != v for k, v in gate['source_sha256'].items()):
            raise ValueError('stale gate: '+str(path))
    if read(gate_paths[0])['source_sha256'] != identity:
        raise ValueError('independent integration must bind complete170 closure')
    software = {p.relative_to(ROOT).as_posix(): sha(p) for p in gate_paths}
    small = read(ROOT/'review/v20_small_plan_v1.json')
    contract = ROOT/'review/V21稳定交接实施合同_v1.md'
    stages = []
    registrations = []
    for variant in ('v20', 'prefix', 'v21'):
        for name, scene, motion, seed in (
            ('nominal_transport', 'nominal', 'transport', 619741),
            ('fragile_hold', 'fragile_conflict', 'hold', 619741),
            ('friction_transport', 'friction_drop', 'transport', 619742)):
            tag = variant+'_'+name
            output = 'output/v21_small_'+tag+'_v1'
            registration = 'review/v21_small_'+tag+'_registration_v1.json'
            if (ROOT/output).exists() or (ROOT/registration).exists():
                raise FileExistsError('all attempts retained; existing '+tag)
            prior = next(s for s in small['stages'] if s['name'] == name)
            run = ROOT/prior['output']/'runs'/f'{scene}__ratio_effort__{seed}'
            if sha(run/'trace.npz') != read(run/'metadata.json')['trace_sha256']:
                raise ValueError('original V20 trace identity failed')
            assignment = dict(methods=['ratio_effort'], scenarios=[scene], seeds=[seed],
                motion_mode=motion, physical_randomization=True, task_policy='integrated',
                receiver_policy='coordinated', release_variant=variant)
            reg = dict(schema='v21_three_arm_seen_registration_v1',
                created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=identity,
                environment=env, assignment=assignment, software_gate_passed=True, software_gates=software,
                implementation_contract_sha256=sha(contract),
                original_v20_run=run.relative_to(ROOT).as_posix(),
                original_v20_trace_sha256=sha(run/'trace.npz'),
                scope='seen mechanism three-arm development; not confirmation or global superiority')
            registrations.append((ROOT/registration, reg))
            stages.append(dict(name=tag, variant=variant, context_name=name, output=output,
                registration=registration, assignment=assignment, planned=1,
                original_v20_run=reg['original_v20_run'], original_v20_trace_sha256=reg['original_v20_trace_sha256']))
    plan_path = ROOT/'review/v21_small_plan_v1.json'
    if plan_path.exists():
        raise FileExistsError('original nine-assignment plan retained')
    plan = dict(schema='v21_three_arm_seen_plan_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=identity, environment=env, parent160_unchanged=True, noMotion156_unchanged=True,
        planned=9, contexts=3, variants=['v20', 'prefix', 'v21'], stages=stages,
        software_gates=software, implementation_contract=contract.relative_to(ROOT).as_posix(),
        implementation_contract_sha256=sha(contract),
        auxiliary_contract_sha256=sha(ROOT/'review/delivery_auxiliary_contract_v3.json'),
        auxiliary_calculator_sha256=sha(ROOT/'review/delivery_auxiliary_metrics_v3.py'),
        release_variation_contract_sha256=sha(ROOT/'review/v21_release_variation_contract_v1.json'),
        release_variation_reader_sha256=sha(ROOT/'review/read_v21_release_variation_v1.py'),
        execution_order='three unchanged V20 full16 exact bridges first; then six frozen prefix/full assignments',
        readout_barrier='all9 terminal before outcome comparison; bridges checked only for exact integration',
        expansion_gate=dict(full_v21_task=3, physical_clearance=3, floor=0,
            minimum_full_force_guard=2, minimum_final_low_motion_residence=1, denominator=3),
        no_parameter_sweep=True, performance_claim_allowed=False,
        evaluation='unchanged original evaluator and fixed auxiliaryv3; all failures, NULL, unknown and partial prefixes retained')
    # Validate every descriptor before writing any registration file.
    for path, reg in registrations:
        write(path, reg)
    write(plan_path, plan)
    print(json.dumps(dict(registered=9, sources=170, physics_dispatched=False,
        plan=str(plan_path), plan_sha256=sha(plan_path)), ensure_ascii=False))


if __name__ == '__main__':
    main()

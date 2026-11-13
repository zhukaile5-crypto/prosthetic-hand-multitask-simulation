"""Freeze five-arm, two-new-seed comparison before any new physical trial."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare(independent_gate, metric_gate, metric_reader):
    from v17b_full_evidence.runtime import source_identity, METHODS
    from v13_a1.run_bridge import environment
    master = ROOT / 'review/v17b_prospective80_plan_v1.json'
    progress = ROOT / 'review/v17b_prospective80_progress_v1.json'
    if master.exists() or progress.exists():
        raise FileExistsError('Do not overwrite a registration or execution attempt')
    source = source_identity(METHODS)
    prior = read(ROOT / 'review/v17_transfer_small_plan_v1.json')
    if len(source) != 136 or len(prior['source_sha256']) != 133:
        raise RuntimeError('Expected frozen133 plus3 evidence-only files')
    if any(source.get(k) != v for k, v in prior['source_sha256'].items()):
        raise RuntimeError('Frozen V17 control/model/evaluation source changed')
    binaries = environment()
    if binaries != prior['environment']:
        raise RuntimeError('Original42 binary environment changed')
    gate_paths = ['review/v17b_runtime_cross_public_attempt1.json', independent_gate, metric_gate,
        'review/v17b_constructor_author_v1.json']
    gates = {}
    for name in gate_paths:
        path = (ROOT / name).resolve()
        gate = read(path)
        if gate.get('passed') is not True:
            raise RuntimeError('Software gate not passed: ' + name)
        gates[name] = dict(sha256=sha(path), passed=True)
    for name in gate_paths[:2]:
        payload = json.dumps(read(ROOT / name))
        if any(value not in payload for value in source.values()):
            raise RuntimeError('Runtime gate not bound to all136 current sources: ' + name)
    contract_path = ROOT / 'review/delivery_auxiliary_contract_v3.json'
    if not contract_path.is_file() or not (ROOT / metric_reader).is_file():
        raise FileNotFoundError('Auxiliary definitions and calculator must exist before registration')
    metric_gate_payload = json.dumps(read(ROOT / metric_gate))
    if sha(contract_path) not in metric_gate_payload or sha(ROOT / metric_reader) not in metric_gate_payload:
        raise RuntimeError('Metric independent gate does not bind current v3 definition/calculator')
    metric_pin = dict(contract=str(contract_path.relative_to(ROOT)), contract_sha256=sha(contract_path),
        reader=metric_reader, reader_sha256=sha(ROOT / metric_reader), gate=metric_gate,
        gate_sha256=sha(ROOT / metric_gate), original_evaluator_changed=False,
        historical_fifteen_posthoc_only=True, fresh_cohort_prefixed_before_physics=True)
    seed_inventory = ROOT / 'review/v17_future_seed_inventory_v1.json'
    inventory = read(seed_inventory)
    seeds = [619741, 619742]
    if inventory['proposed_seeds'] != seeds or inventory['matches'] != 0 or inventory['physical_execution_started']:
        raise RuntimeError('Wrong pre-execution seed inventory')
    scenarios = ['nominal', 'friction_drop', 'fragile_conflict', 'collision']
    stages = []
    for motion in ('hold', 'transport'):
        for arm, methods, task, receiver in (
            ('holding_A', ['ratio_effort'], 'integrated', 'blind'),
            ('full_V17', ['ratio_effort'], 'integrated', 'coordinated'),
            ('baselines', ['ratio_effort', 'fixed_effort', 'fusion_effort'], 'legacy', 'blind')):
            name = arm + '_' + motion
            stage = dict(name=name, arm=arm, methods=methods, scenarios=scenarios, seeds=seeds,
                task_policy=task, receiver_policy=receiver, motion_mode=motion, physical_randomization=True,
                planned=len(methods)*len(scenarios)*len(seeds), workers=3,
                output='output/v17b_prospective80_' + name + '_v1',
                registration='review/v17b_prospective80_' + name + '_registration_v1.json')
            for key in ('output', 'registration'):
                if (ROOT / stage[key]).exists():
                    raise FileExistsError(ROOT / stage[key])
            stages.append(stage)
    plan = dict(schema='v17b_five_arm_prospective_development80_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=source, environment=binaries,
        parent133_unchanged=True, source_change_role='logging_only_existing_MotionTask_matrix_target',
        software_gates=gates, auxiliary_metrics=metric_pin,
        seed_inventory_sha256=sha(seed_inventory), seeds=seeds, scenarios=scenarios,
        planned=80, unique_contexts=16, physical_seed_clusters=2, stages=stages,
        old_methods=dict(ratio_effort='Matos2025sec2.1-inspired force-ratio PI migration, not full paper reproduction',
            fixed_effort='frozen fixed force preference', fusion_effort='frozen V14 fusion preference'),
        holding_A_role='ablation: same holding as full V17, legacy blind release',
        full_V17_role='unchanged V17 holding+contact transfer, now with full existing M/target logging',
        all_five_arms_share_same_public_instruments=True,
        original_measurement_RNG_and48Gaussian6uniform_unchanged=True,
        observation_sampling_execution_model_gains_limits_deadlines_evaluator_unchanged=True,
        schedule=dict(duration_s=16., handoff_s=7.5, release_s=12., clearance_deadline_s=3.5),
        comparison=dict(all_assigned_failures_and_missing_retained=True,
            primary_reference='same_prior_legacy_ratio', mechanism_reference='holding_A',
            report_all_five_arms_and_each_seed_and_each_movement=True,
            report_original_success_force_guard_and_auxiliary_metrics_separately=True,
            no_posthoc_weighted_total_score=True, pairwise_pareto_and_tradeoffs_descriptive_only=True,
            no_independent_n16_or_n80_significance_claim=True,
            full_vs_A_strictly_pre12_original37_and_new_metadata_must_match=True,
            same_seed_static_prefix_not_independent_between_hold_and_transport=True,
            no_algorithm_selection_from_incomplete_subset=True),
        stop_on_source_environment_runtime_or_prefix_contract_failure=True,
        do_not_stop_for_task_failure_or_unfavourable_metrics=True,
        performance_claim_allowed=False, prospective_development_not_global_confirmation=True,
        scheduler='review/run_v17b_prospective80_v1.py',
        scheduler_sha256=sha(ROOT / 'review/run_v17b_prospective80_v1.py'),
        prefix_helper_sha256={name:sha(ROOT/name) for name in
            ('review/run_v17_transfer_small_v1.py','review/run_v16_release_small_v1.py')},
        preparer_sha256=sha(Path(__file__)))
    # All registrations are made before the first run, each tied to this master.
    write(master, plan)
    for s in stages:
        assignment = {key:s[key] for key in ('methods','scenarios','seeds','motion_mode','physical_randomization','task_policy','receiver_policy')}
        write(ROOT / s['registration'], dict(schema='v17b_prospective80_cell_registration_v1',
            source_sha256=source, environment=binaries, assignment=assignment,
            software_gate_passed=True, master_plan_sha256=sha(master), development_only=True,
            performance_claim_allowed=False, auxiliary_contract_sha256=metric_pin['contract_sha256']))
    print(json.dumps(dict(registered=True, planned=80, sources=len(source),
        seed_clusters=2, plan=str(master.relative_to(ROOT)), sha256=sha(master))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--independent-gate',required=True)
    p.add_argument('--metric-gate',required=True); p.add_argument('--metric-reader',required=True)
    a=p.parse_args(); prepare(a.independent_gate, a.metric_gate, a.metric_reader)

"""Pure schema/design preflight. No app imports, simulator, services or writes.

Default: refuse execution with exit 2 even when the draft is consistent.
--check-draft: check the intended draft/schema and return 0 if consistent.
This script has no run capability and cannot freeze/admit an artifact bundle.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'calibration/release_material_response_plan_v1.json'
STATUS = 'draft_not_executable_without_reviewed_model/basis/entry'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(tree, name):
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    return [n.target.id for n in node.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]


def assignment(tree, name):
    return next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def check(plan):
    require(plan['schema'] == 'release_material_response_plan_v1', 'Plan schema mismatch')
    require(plan['status'] == STATUS, 'Draft status must remain explicit')
    for key in ['frozen', 'executable', 'physics_runs_authorized_by_this_artifact', 'model_admitted']:
        require(plan[key] is False, f'{key} cannot be enabled by this draft')
    contract_tree = ast.parse((ROOT / 'release_material_contract.py').read_text(encoding='utf-8-sig'))
    model_tree = ast.parse((ROOT / 'release_entry_model.py').read_text(encoding='utf-8-sig'))
    generator = assignment(contract_tree, 'FINGER_ACTUATORS').args[0]
    require(isinstance(generator, ast.GeneratorExp), 'Review changed actuator source schema')
    finger_names = ast.literal_eval(generator.generators[0].iter)
    require(isinstance(generator.elt, ast.BinOp)
            and ast.literal_eval(generator.elt.left) == 'rh_A_', 'Actuator prefix changed')
    motion_axes = ast.literal_eval(assignment(contract_tree, 'COMMAND_AXES').right)
    command_axes = ['rh_A_' + name for name in finger_names] + list(motion_axes)
    interface = plan['interface']
    require(interface['canonical_actual_command_order'] == command_axes, 'Actual21 command order mismatch')
    require(len(command_axes) == len(set(command_axes)) == 21, 'Exactly21 unique commands required')
    require(interface['command_units'] == ['rad'] * 18 + ['m', 'rad', 'rad'], 'Command units mismatch')
    require(interface['observable_release_submodes'] == list(ast.literal_eval(assignment(contract_tree, 'RELEASE_MODES'))), 'Release mode mismatch')
    require(interface['public_release_state_whitelist'] == fields(contract_tree, 'ReleaseState'), 'ReleaseState strict fields mismatch')
    require(interface['material_contract']['required_content'] == fields(contract_tree, 'MaterialContract'), 'MaterialContract strict fields mismatch')
    require(interface['actuator_authority']['required_fields'] == fields(contract_tree, 'ActuatorAuthority'), 'ActuatorAuthority strict fields mismatch')
    require(plan['coverage_and_split']['entry_domain_schema_fields'] == fields(model_tree, 'EntryDomain'), 'EntryDomain coverage schema mismatch')
    require(plan['coverage_and_split']['patch_region_schema_fields'] == fields(model_tree, 'PatchRegion'), 'PatchRegion coverage schema mismatch')
    require(interface['entry_assessment_permission_fields'] == {
        'release_request_may_be_delayed': False, 'holding_support_may_be_requested': False,
        'pressure_prediction_available': False, 'material_certified': False,
        'global_model_admitted': False}, 'Entry evidence cannot take lifecycle authority')
    require(interface['actuator_authority']['status'] == 'missing'
            and interface['actuator_authority']['value'] is None, 'No manufactured authority')
    require(interface['material_contract']['force_scope'] == 'all_external_instrumented_skin', 'All-skin scope required')
    profiles = interface['material_contract']['profiles']
    require(profiles == [
        {'id': 'nominal', 'per_finger_caps_N': [5., 5., 5., 5., 5.], 'sum_budget_N': 18., 'auxiliary_pressure_cap_N': None},
        {'id': 'fragile', 'per_finger_caps_N': [3.5, 2.8, 2.8, 2.8, 2.8], 'sum_budget_N': 12., 'auxiliary_pressure_cap_N': None}
    ], 'Actual nominal/fragile task caps or unprotected-aux declaration changed')
    forbidden = set(interface['forbidden_predictor_or_entry_inputs'])
    require(not forbidden.intersection(interface['public_release_state_whitelist']), 'Nonpublic feature in state whitelist')
    require(plan['release_preemption']['maximum_transition_latency_s'] == .005, 'Release preemption convention changed')
    basis = plan['candidate_basis']
    require(basis['status'] == 'missing' and basis['max_rank'] == len(basis['candidates']) == 4, 'Candidate basis must remain missing')
    for candidate in basis['candidates']:
        require(candidate['status'] == 'missing' and candidate['admitted'] is False, 'No basis admission')
        require(all(candidate[k] is None for k in ['vector_21', 'coefficient_amplitudes', 'rate_and_waveform_domain', 'mode_eligibility']), 'Unmeasured basis numeric value')
        require(all(type(x) is int and 0 <= x < 21 for x in candidate['possible_command_indices']), 'Invalid candidate support indices')
    require(basis['branch_design']['branches_per_mode_upper'] == 2 + 4 * 2 + 4 == 14, 'Branch accounting mismatch')
    require(basis['branch_design']['numeric_waveforms'] is None
            and basis['branch_design']['numeric_waveforms_status'] == 'missing', 'No unreviewed acquisition waveform')
    model = plan['measurement_and_model']
    require(model['model_schema'] == ast.literal_eval(assignment(model_tree, 'MODEL_SCHEMA')), 'Model schema mismatch')
    require(model['availability'] == 'missing' and model['manifest_payload'] is None
            and model['response_artifact'] is None, 'Draft may not manufacture a model artifact')
    require(model['allowed_availability'] == ['missing', 'artifact_bundle_ready_for_review'], 'Admission must not be a JSON status')
    require(model['response_pressure_axes'] == [f'finger_{x}_normal_N' for x in range(5)] + ['auxiliary_pressure_N'], 'Six pressure axes mismatch')
    require(model['response_pressure_dimension'] == 6 and model['diagnostic_output_dimension'] == len(model['diagnostic_output_order']) == 17, 'Response/diagnostic dimensions confused')
    require(set(model['basis_npz_schema']['required_keys']) == {'matrix_21_by_k', 'command_axes', 'command_units', 'coefficient_units'}, 'Basis NPZ schema mismatch')
    require(set(model['response_npz_schema']['required_keys']) == {'time_s', 'kernel_pressure_N_per_basis_increment', 'empirical_error_N', 'raw_peak_error_N', 'raw_peak_window_s', 'measured_coefficient_lower', 'measured_coefficient_upper', 'pressure_axes', 'pressure_units'}, 'Response NPZ schema mismatch')
    require(model['response_npz_schema']['shapes']['kernel_pressure_N_per_basis_increment'] == ['H', '6', 'K'], 'Response matrix dimension mismatch')
    for key in ['basis_npz_schema', 'response_npz_schema']:
        require(model[key]['status'] == 'missing' and model[key]['numeric_values'] is None, 'No fake numeric artifact')
    evaluator = model['current_evaluator_state']
    require(evaluator['installed'] is False and evaluator['forecast_pressure_available'] is False
            and evaluator['kernel'] is None and evaluator['material_certified'] is False
            and evaluator['global_model_admitted'] is False, 'Current evaluator must remain missing')
    require(model['independent_validation_schema']['split_grouping'] == 'complete_parent_entry', 'Validation grouping mismatch')
    coverage = plan['coverage_and_split']
    require(coverage['coverage_cells'] == len(coverage['morphologies']) * len(coverage['public_material_profiles']) * len(coverage['offline_entry_producers']) == 18, 'Coverage count mismatch')
    require(coverage['train_entries_per_cell'] == 3
            and coverage['stage2_response_train_entries_per_cell'] == 2
            and coverage['training_parent_groups_including_pilot'] == 54, 'Pilot/train independent-group count mismatch')
    stages = {s['id']: s for s in plan['stages']}
    for sid in ['S2_train_local_response', 'S3_locked_local_holdout']:
        s = stages[sid]
        require(s['maximum_local_branches'] == s['parent_entries'] * s['max_mode_checkpoints_per_entry'] * s['local_branches_per_mode'], 'Local branch count mismatch')
        require(s['planned_simulated_s'] == s['parent_lifecycle_simulated_s'] + s['maximum_local_branches'] * s['branch_duration_s'], 'Local stage work mismatch')
    s = stages['S4A_matched_entry_release_dt']
    require(s['full_release_branches'] == s['parent_entries'] * len(s['dts_s']) * (s['candidate_count'] + s['reference_replays_per_dt']), 'Matched branch count mismatch')
    require(s['planned_simulated_s'] == s['parent_preparation_simulated_s'] + s['full_release_branches'] * s['post_entry_duration_s'], 'Matched stage work mismatch')
    s = stages['S4B_reset_dt_lifecycle']
    require(s['full_reset_episodes'] == s['independent_parent_configuration_groups'] * s['candidate_count'] * len(s['dts_s']), 'Reset-dt episode count mismatch')
    require(s['planned_simulated_s'] == s['full_reset_episodes'] * s['episode_duration_s'], 'Reset stage work mismatch')
    require(sum(s['planned_simulated_s'] for s in stages.values()) == 6264., 'Total work count mismatch')
    for s in stages.values():
        require(s['execution_authorized'] is False and s['new_runner_measured_wall_minutes'] is None
                and s['wallclock_authorized_budget_s'] is None, 'Reference estimate is not run authorization')
    require(plan['wallclock_budget']['authorized_resource_budget'] is None, 'No resource authorization in draft')
    require(plan['paired_and_dt_design']['dt_transfer_protocol'] is None, 'Cross-dt transfer not implemented')
    for key in ['engineering_prediction_thresholds', 'dt_agreement_thresholds']:
        require(plan['acceptance'][key] is None, 'Do not invent unmeasured acceptance criteria')
    observed_drift = []
    for item in plan['provenance']['draft_read_sources']:
        path = Path(item['path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            observed_drift.append(str(path))
    return {'draft_schema_consistent': True, 'executable': False,
            'reason': STATUS, 'physics_steps': 0, 'files_written': 0,
            'planned_work_simulated_s': 6264.,
            'draft_observed_source_drift': observed_drift,
            'source_drift_note': 'Draft observations only; future execution requires fresh complete immutable snapshot.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-draft', action='store_true', help='Validate intended non-executable draft only.')
    args = parser.parse_args()
    try:
        result = check(json.loads(PLAN.read_text(encoding='utf-8-sig')))
    except (OSError, ValueError, TypeError, KeyError, StopIteration, AttributeError, SyntaxError) as exc:
        print(json.dumps({'draft_schema_consistent': False, 'executable': False,
                          'error': str(exc), 'physics_steps': 0}, ensure_ascii=False))
        return 1
    result['execution_refused'] = True
    print(json.dumps(result, ensure_ascii=False))
    return 0 if args.check_draft else 2


if __name__ == '__main__':
    raise SystemExit(main())

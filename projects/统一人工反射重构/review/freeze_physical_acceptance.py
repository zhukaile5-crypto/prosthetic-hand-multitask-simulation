"""Create an append-only, source-linked foundation certificate; no trial selection."""
from pathlib import Path
import hashlib
import json
import math
import datetime

BASE = Path(__file__).resolve().parents[1]
REVIEW = BASE / 'review'
EXPECTED = 'ee5d35f14bb36c88971d12df64c282cd35b1879d1e3a3a0eb5d39a3cea75d533'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(relative):
    return json.loads((REVIEW / relative).read_text(encoding='utf-8-sig'))

assert digest(BASE / 'physics.py') == EXPECTED
output = REVIEW / 'physical_acceptance.json'
assert not output.exists(), 'Certificate is immutable; use a new file for a new model.'
foundation = read('five_pad_foundation_final/acceptance.json')
references = read('five_pad_references_final/references.json')
old_references = read('five_pad_shared_references/references.json')
identification = read('five_pad_local_identification_final/identification.json')
bare = read('five_pad_bare_hand_final/bare_hand.json')
release = read('five_pad_release_foundation/summary.json')
noise = read('noise_pairing_physics.json')
profile = read('nominal_force_profile_oracle.json')
rows = [json.loads(p.read_text(encoding='utf-8-sig')) for p in sorted(
    (REVIEW/'five_pad_foundation_final').glob('*.json')) if p.name != 'acceptance.json']
source_rows = [foundation, *references.values(), *identification, *release, *rows]
assert all(r['physics_sha256'] == EXPECTED for r in source_rows)
assert foundation['foundation_accepted'] and len(rows) == 12
assert all(p['accepted'] for p in foundation['pairs']) and len(foundation['pairs']) == 6
assert all(r['finite'] and r['warnings'] == 0 and r['final_contacts'] >= 3 for r in rows)
assert all(r['valid_seed_count'] == 3 and r['window'] == [8.2, 8.9] for r in references.values())
reference_delta = {shape: [float(a-b) for a,b in zip(row['relative_offset'],
    old_references[shape]['relative_offset'])] for shape,row in references.items()}
evidence_roots = ['five_pad_foundation_final', 'five_pad_references_final',
    'five_pad_local_identification_final', 'five_pad_bare_hand_final',
    'five_pad_release_foundation']
evidence = {}
for folder in evidence_roots:
    for path in sorted((REVIEW/folder).iterdir()):
        if path.is_file():
            evidence[str(path.relative_to(BASE)).replace('\\','/')] = {
                'sha256': digest(path), 'bytes': path.stat().st_size}
for filename in ['noise_pairing_physics.json', 'nominal_force_profile_oracle.json',
                 'physical_acceptance.md']:
    path = REVIEW / filename
    evidence['review/'+filename] = {'sha256': digest(path), 'bytes': path.stat().st_size}
diagonal = [g for r in identification for g in r['diagonal_gain']]
rmse = [g for r in identification for g in r['paired_step_model_rmse_N']]
offdiag = [abs(g) for r in identification for i, row in enumerate(r['jacobian_N_per_closure'])
           for j,g in enumerate(row) if i != j]
certificate = {
    'schema': 'public-physical-foundation-certificate-v1',
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'model_id': 'finite-mass-five-compliant-tactile-pads-v1',
    'model_metadata': rows[0]['model_metadata'],
    'physics_sha256': EXPECTED,
    'calibration_sha256': digest(BASE/'calibration.py'),
    'source_verification': {'matching_source_records': len(source_rows),
        'all_source_records_current': True,
        'unhashed_noise_profile_and_bare_hand_records': 'Executed on the frozen source; evidence digests in this certificate establish the file identity.'},
    'foundation': {'accepted': True, 'trial_count': len(rows),
        'shape_count': 3, 'contact_families': ['nominal', 'compliant'],
        'physical_timesteps_s': [.0005, .00025], 'duration_per_trial_s': 17.,
        'seed': 410801, 'controller': 'independent fixed 2 N per-pad development probe',
        'all_final_contacts': [r['final_contacts'] for r in rows],
        'zero_warnings': True, 'all_finite': True,
        'max_initial_penetration_m': max(r['initial_penetration_m'] for r in rows),
        'max_holding_penetration_m': max(r['hold_peak_penetration_m'] for r in rows),
        'max_any_grasp_penetration_m': max(r['peak_penetration_m'] for r in rows),
        'max_actual_pad_peak_N': max(r['peak_force_N'] for r in rows),
        'max_limit_exposure_Ns': max(r['force_exposure_Ns'] for r in rows),
        'half_timestep_pairs': foundation['pairs']},
    'references': {'final_file': 'review/five_pad_references_final/references.json',
        'values': references, 'relative_offset_change_from_prior_snapshot_m': reference_delta,
        'maximum_prior_reference_difference_m': max(abs(v) for row in reference_delta.values() for v in row),
        'prior_snapshot_sha256': old_references['cylinder']['physics_sha256'],
        'note': 'The current-source repeated probe reproduces the prior reference values exactly; old source hashes remain historical snapshots.'},
    'local_force_identification': {'applied_common_diagonal_gain_N_per_closure': 14.25,
        'applied_effective_tau_s': .005, 'measured_diagonal_range': [min(diagonal),max(diagonal)],
        'max_absolute_off_diagonal_N_per_closure': max(offdiag),
        'paired_step_rmse_range_N': [min(rmse),max(rmse)],
        'time_constant_resolution_note': 'All fits reached the 5 ms grid lower bound; this is a discrete effective model, not proof of precise hardware response.',
        'source': 'review/five_pad_local_identification_final/identification.json'},
    'encoder_motor_load_calibration': bare,
    'strict_paired_noise_test': noise,
    'public_static_force_profile_oracle': profile,
    'release_probe': {'trial_count': len(release), 'duration_s': 12., 'release_at_s': 8.,
        'all_final_contacts_zero': all(r['final_contacts'] == 0 for r in release),
        'all_final_pad_force_zero': all(max(abs(v) for v in r['final_force']) == 0 for r in release),
        'zero_warnings': all(r['warnings'] == 0 for r in release),
        'scope': 'Five-pad contact release; actual subsequent world contact is retained and not included in grasp penetration acceptance.'},
    'scope': {'accepted': 'Fixed pregrasp contact acquisition and quasi-static hold of three rigid shapes in this standardized MuJoCo prototype.',
        'not_established': ['full Shadow dynamics', 'free grasp planning', 'real prosthesis hardware',
            'clinical material damage', 'all severe perturbation and collision cases',
            'free-drop world-contact timestep acceptance'],
        'comparison_status': 'This certificate is physical foundation evidence, not evidence that the proposed controller outperforms any baseline.'},
    'evidence_files': evidence,
}
assert certificate['release_probe']['all_final_contacts_zero']
assert certificate['release_probe']['all_final_pad_force_zero']
assert noise['fixed_noise_draws_paired']
output.write_text(json.dumps(certificate, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'certificate': str(output), 'sha256': digest(output),
    'foundation_accepted': True, 'evidence_files': len(evidence),
    'reference_max_difference_m': certificate['references']['maximum_prior_reference_difference_m']},
    ensure_ascii=False))

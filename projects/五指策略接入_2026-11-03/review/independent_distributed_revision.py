"""Declare distributed tactile geometry and refined common physics timestep."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,copy

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parent=ROOT/'acceptance_protocol_handoff_v4.json'
    original=json.loads(parent.read_text(encoding='utf-8'))
    result=copy.deepcopy(original)
    result['schema']='shadow-hand-integration-acceptance-v5-distributed-contact-and-refined-physics'
    result['status']='common-interface-and-numerical-design-before-any-policy-performance-data'
    result['physical_foundation']['physical_timesteps_s']=[.0005,.00025]
    fields=['normal_wrench_map[6,5]','finger_non_normal_wrench[6]',
            'finger_non_normal_by_finger[5,6]','normal_support_coeff[5]',
            'tangent_support_coeff[5]','shear_magnitude[5]','slip_magnitude[5]',
            'contact_present[5]','distributed_valid[5]','distributed_error[5]']
    result['observation_contract']['required_control_inputs'].extend(fields)
    result['observation_contract']['aggregation']['production_representation']='distributed-normal-wrench-map-plus-measured-non-normal-wrench'
    result['observation_contract']['aggregation']['legacy_single_point_threshold_unchanged']=True
    result['observation_contract']['aggregation']['legacy_single_point_error_still_reported']=True
    result['observation_contract']['aggregation']['legacy_point_fields_do_not_drive_production_geometry']=True
    result['observation_contract']['distributed_contact']={
        'coordinate_frame':'palm; same acquisition time as all tactile measurements',
        'all_external_skin_contact_no_target_identity_filter':True,
        'normal_wrench_map_definition':'For digit i with measured total pressure N_i, column i is sum_j (n_j/N_i)*[d_j, p_j cross d_j]. The force vector is not unit-normalized. Moment rows are physical metres, not divided by 0.05.',
        'non_normal_wrench_definition':'Sum of measured tangential forces, their position moments and patch torques; not a prediction of friction capacity.',
        'raw_representation_reconstruction':'normal_wrench_map @ normal_pressure + measured_non_normal_wrench equals the original sum of every finger patch wrench.',
        'raw_reconstruction_scaled_wrench_abs_error_max':1e-9,
        'raw_reconstruction_error_norm':'norm([delta_force,delta_moment/0.05])',
        'contact_presence':'Actual measured valid patch/pressure and distributed validity; never infer no contact merely from an averaged normal vector cancelling.',
        'shear_magnitude':'Per-digit sum of patch tangential-force magnitudes; consumed by friction-ratio feedback and local shear demand.',
        'slip_magnitude':'Per-digit maximum patch tangential relative speed; consumed by slip detection.',
        'net_support_capacity_coefficient':'signed normal_support_coeff + mu_safe * tangent_support_coeff; negative normal contributions are retained in the QP net support inequality.',
        'positive_contact_ranking':'A separate positive ranking view may be used, but it must not replace signed net-support coefficients.',
        'measured_non_normal_wrench_not_credited_as_guaranteed_future_capacity':True,
        'sensor_noise_and_model_error_remain_explicit':True,
        'strict_rejection':'Invalid production distributed representation yields invalid observation; no strict=False bypass or 10% point-error relaxation.',
        'legacy_point_failure_interpretation':'The original single-point approximation remains failed where its error exceeds 10%. Production acceptance uses a richer measured representation instead of pretending the failed approximation passed.'}
    timing=result['matched_comparison']['initial_transfer_timing']
    timing['post_handoff_first_stability_definition']='Restricted waiting from fixed t=7.5 to the first independently verified stable dwell by t=8.9. If none completes, contribution is the full 1.4 s horizon. Preparation failures stay in every scheduled whole-task denominator and must be explicit in conditional summaries.'
    evidence=ROOT/'review/independent_foundation_matrix_v1_failed.json'
    result['distributed_numerical_revision']={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'parent_protocol':parent.name,'parent_sha256':sha(parent),
        'changed_model_designs':{
            'physical_timesteps_s':{'old':[.001,.0005],'new':[.0005,.00025]},
            'contact_representation':{'old':'one equivalent point and unit normal per finger','new':'distributed 6x5 normal wrench map with separately measured non-normal wrench and patch-magnitude features'}},
        'physical_timestep_reason':'The complete old 1 ms/0.5 ms matrix passed holding/penetration/release but failed the fixed 15% raw lifecycle peak-force step-sensitivity gate. Refine both compared physical steps and rerun the full matrix.',
        'failed_matrix_evidence':str(evidence.relative_to(ROOT)),
        'failed_matrix_evidence_sha256':sha(evidence),
        'distributed_reason':'Box/sphere multiple-patch contact cannot reliably be compressed to one unit normal and point within the original 10% wrench criterion. Retain those failed approximation diagnostics and expose a physically richer common measurement interface.',
        'observation_source_at_registration':{name:sha(ROOT/name) for name in ('contracts.py','sensors.py')},
        'unchanged_physical_gates':copy.deepcopy(original['physical_foundation']['gates']),
        'root_agreed_before_any_policy_performance_data':True,
        'scope':'Numerical/common representation migration only. No change to ratio PI law or unified task objective, no target oracle, no stability-corridor policy.'}
    assert result['physical_foundation']['gates']==original['physical_foundation']['gates']
    result['interpretation_boundaries'].append('Full-hand natural palm support is allowed and measured; this is not a claim that five finger friction alone carries the object.')
    out=ROOT/'acceptance_protocol_migration_v5.json'
    if out.exists():raise SystemExit('Refusing to overwrite declared model/interface design.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'protocol':str(out),'sha256':sha(out),'physical_timesteps_s':[.0005,.00025],
                      'physical_gates_unchanged':True,'new_inputs':fields},ensure_ascii=False))
if __name__=='__main__':main()

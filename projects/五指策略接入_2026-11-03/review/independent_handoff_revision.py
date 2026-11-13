"""Predeclare the first bounded held-grasp strategy-transfer comparison."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,copy

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parent=ROOT/'acceptance_protocol_full_hand_v3.json'
    original=json.loads(parent.read_text(encoding='utf-8'))
    result=copy.deepcopy(original)
    result['schema']='shadow-hand-integration-acceptance-v4-held-grasp-handoff'
    result['status']='held-grasp-transfer-scope-before-any-policy-performance-data'
    result['algorithm_migration']['scope']='Current ratio/unified force policies take over a commonly prepared held grasp inside a validated local execution-model domain. This first transfer does not claim policy-controlled grasp discovery, contact acquisition or initial lifting.'
    result['algorithm_migration']['handoff']={
        'fixed_time_s':7.5,
        'preparation_deadline_s':7.5,
        'preparation':'Same publicly specified preparation and sensor/observer history for every method.',
        'causal_control_conditions':'Only public task clock, encoders and declared virtual tactile observations decide control admission. Object truth does not decide control.',
        'evaluation_admission':'Independent physical truth checks whether a valid held/lifted grasp was actually prepared; it does not feed back to control.',
        'failed_preparation':'Retain in every scheduled whole-task denominator; never silently select only prepared trials.',
        'state_reset':'Each run has fresh policy integral/ratio state, observer history, sensor queue, RNG and clock. No state from another method/run is reused.',
        'after_handoff':'Only the selected current strategy plus common execution/constraints; preparation or legacy reflexes must not continue adding actions.',
        'local_execution_domain':'Bound all executed commands to an independently validated holding trust region. Log requested/applied actions, clipping and domain exits. No extrapolation from holding calibration to CLOSING is claimed.'}
    result['matched_comparison']['initial_transfer_timing']={
        'duration_s':16.0,
        'handoff_s':7.5,
        'first_disturbance_not_before_s':9.3,
        'release_command_s':12.0,
        'release_deadline_after_command_s':3.5,
        'release_dwell_s':0.2,
        'main_static_window_absolute_s':[8.2,8.9],
        'post_handoff_first_stability_horizon_s':1.4,
        'post_handoff_first_stability_origin_s':7.5,
        'post_handoff_first_stability_definition':'Restricted waiting from the fixed handoff to first independently verified stable dwell by t=8.9; no successful stable attainment contributes the full 1.4 s horizon. Preparation failures remain scheduled failures and their exclusion from conditional summaries must be shown.',
        'whole_task_first_stability_also_reported':True,
        'shared_preparation_not_attributed_to_strategy_speed':True}
    result['recording_and_evaluation']['transfer_reporting']={
        'whole_task_with_preparation_failures':True,
        'preparation_success_and_strategy_handoff_coverage_separate':True,
        'post_handoff_stability_wait_and_total_first_stability_separate':True,
        'conditional_held_grasp_quality_denominator_explicit':True,
        'shared_initial_stability_does_not_prove_faster_policy_acquisition':True,
        'no_change_to_raw_force_or_release_gates':True}
    result['handoff_revision']={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'parent_protocol':parent.name,'parent_sha256':sha(parent),
        'reason':'The empirically identified signed/coupled local response is a holding model. First transfer is bounded to that validated regime rather than pretending it models initial free-space closure and contact acquisition.',
        'physical_probe_matrix_and_gates_unchanged':True,
        'future_scope':'Full acquisition/lifting policy control needs its own contact-mode model and independent validation; it is not silently claimed by this transfer.'}
    assert result['physical_foundation']==original['physical_foundation']
    out=ROOT/'acceptance_protocol_handoff_v4.json'
    if out.exists():raise SystemExit('Refusing to overwrite a declared transfer scope.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'protocol':str(out),'sha256':sha(out),'handoff_s':7.5,'physical_gates_unchanged':True},ensure_ascii=False))
if __name__=='__main__':main()

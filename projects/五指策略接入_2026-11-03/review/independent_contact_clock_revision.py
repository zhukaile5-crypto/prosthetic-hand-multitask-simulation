"""Register a shared contact approximation and exact control-clock design."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,copy

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parent=ROOT/'acceptance_protocol_migration_v5.json'
    original=json.loads(parent.read_text(encoding='utf-8'))
    result=copy.deepcopy(original)
    result['schema']='shadow-hand-integration-acceptance-v6-smooth-point-contact-and-integer-clock'
    result['status']='shared-contact-and-clock-design-before-any-policy-performance-data'
    result['physical_foundation']['common_contact_model']={
        'contact_dimension':3,
        'collision_representation':'smooth_distal_capsules',
        'four_finger_distal_capsule':{'radius_m':.0075,'half_length_m':.0062,'body_position_m':[0.,0.,.01835]},
        'thumb_distal_capsule':{'radius_m':.0095,'half_length_m':.0063,'body_position_m':[0.,0.,.0192]},
        'scope':'Only distal collision meshes are replaced by fitted smooth capsules following the original distal bodies. Visual meshes, articulated hand/wrist joints, original drives and explicit body inertia are retained.',
        'friction_scope':'Point friction constrains normal and two tangential force directions. No patch torsional or rolling resistance is claimed; total contact moments from lever arms remain present.',
        'all_methods_share_model':True,
        'not_an_exact_original_shadow_collision_model':True}
    result['runtime_and_ui']['control_clock_contract']={
        'public_time':'integer control_tick * 0.005 seconds',
        'one_advance_is_one_sensor_tick':True,
        'dwell_and_handoff_use_integer_tick_counts':True,
        'raw_physical_interval_start_uses_actual_mujoco_time':True,
        'state_restore_includes_control_tick_and_physical_time':True,
        'physical_substep_period_comes_from_model':True}
    result['observation_contract']['physical_force_direction_check']={
        'scope':'Undisturbed static holding probe with the object lifted and no imposed external object force.',
        'mean_hand_on_target_Fz_must_be_positive':True,
        'vertical_balance_residual':'abs(mean(hand_on_target_Fz)-object_mass*9.81)',
        'vertical_balance_tolerance_N':'max(0.05,0.05*object_mass*9.81)+mean(table_support_N)',
        'table_support_bound_is_still_original_gate':True,
        'purpose':'An independent physical sign/scale anchor in addition to exact patch-coordinate reconstruction; it is not a material safety guarantee.'}
    evidence=[ROOT/'review/foundation_pointcontact_capsule_v1/summary.json',
              ROOT/'review/foundation_pointcontact_capsule_half_v1/summary.json']
    result['contact_clock_revision']={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'parent_protocol':parent.name,'parent_sha256':sha(parent),
        'reason':'Original distal-mesh/6D contact produced release-stage impact peaks insufficiently converged under timestep halving. A common smooth distal point-friction approximation is a declared physical model revision, with every shape/contact/timestep lifecycle rerun.',
        'motivation_evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in evidence},
        'source_at_registration':{name:sha(ROOT/name) for name in ('hand_plant.py','foundation_config.py','calibrate_foundation.py')},
        'physical_gate_thresholds_unchanged':True,
        'release_semantics':'Immediate policy unloading/opening with a shared mechanical clearance path. It is not changed to controlled placing before opening. All subsequent real impacts remain in raw peak/exposure statistics.',
        'failed_development_retained':'Original mesh/6D-contact and controlled-place exploratory records remain historical failures or rejected task alternatives, not replacement successful-release evidence.',
        'no_policy_performance_data_used':True}
    assert result['physical_foundation']['gates']==original['physical_foundation']['gates']
    result['interpretation_boundaries'].append('This is the Shadow articulated kinematic/inertial asset with an explicitly simplified common collision/contact model; it is not validation of original mesh/6D contact mechanics.')
    out=ROOT/'acceptance_protocol_contact_clock_v6.json'
    if out.exists():raise SystemExit('Refusing to overwrite declared contact/clock revision.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'protocol':str(out),'sha256':sha(out),'physical_gate_thresholds_unchanged':True},ensure_ascii=False))
if __name__=='__main__':main()

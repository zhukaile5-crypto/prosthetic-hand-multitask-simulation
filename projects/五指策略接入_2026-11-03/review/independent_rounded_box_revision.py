"""Register the final candidate rounded-object common physical design."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,copy

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parent=ROOT/'acceptance_protocol_contact_clock_v6.json'
    original=json.loads(parent.read_text(encoding='utf-8'))
    result=copy.deepcopy(original)
    result['schema']='shadow-hand-integration-acceptance-v7-rounded-box-candidate'
    result['status']='common-rounded-box-and-force-profile-design-before-any-policy-performance-data'
    result['physical_foundation']['rounded_box_model']={
        'canonical_shape':'box',
        'display_name':'2 mm rounded cuboid',
        'outside_half_extents_m_at_unit_size':[.018,.060,.026],
        'rounding_radius_m_at_unit_size':.002,
        'scaling':'All outside half extents and rounding radius multiply the public size factor.',
        'compound_geometry':{'inner_box':1,'face_slabs':6,'edge_capsules':12,'corner_spheres':8,'total':27},
        'inner_half_extents':'outside_half_extents - rounding_radius',
        'inertial_definition':'Single object body explicit mass=config.mass and I_body=mass/3*[b*b+c*c,a*a+c*c,a*a+b*b] for public outside half extents [a,b,c]. Composite geom density=0 prevents replicated mass.',
        'one_free_object_body':True,
        'target_contact_identity':'All geom IDs whose geom_bodyid equals the one object body are included in physical support, contact pressure, environment friction and all-hand release evaluation.',
        'control_observation':'The policy still receives all-external tactile observations without target identity; target grouping is evaluation/environment bookkeeping only.',
        'visualization':'Render the actual same rounded compound geometry in the orange object material.',
        'sharp_box_history':'Earlier sharp-corner box is retained as a failed numerical/geometry design, not silently relabeled as a passed rounded object.',
        'no_claim_of_original_sharp_box_acceptance':True}
    result['observation_contract']['public_nominal_force_profile']={
        'source':'Shared physical holding-force preparation profile, per shape calibration metadata.',
        'formula':'nominal_force_weights = holding_force / mean(holding_force)',
        'shared_for_both_methods':True,
        'required_explicit_in_calibration':True,
        'not_selected_using_policy_performance':True,
        'scope':'Anatomical/common actuation normalization, not method-specific force tuning. Current policy scalar/PI/objective gains remain separately recorded.'}
    evidence=[ROOT/'review/foundation_rounded_box_v1/summary.json',
              ROOT/'review/foundation_rounded_box_half_v1/summary.json',
              ROOT/'review/independent_foundation_matrix_v6_failed.json']
    result['rounded_box_revision']={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'parent_protocol':parent.name,'parent_sha256':sha(parent),
        'reason':'Sharp box release-stage contacts at the thumb and middle-finger distal geometry caused unacceptable raw force-peak timestep sensitivity. A declared rounded compound object is a common physical-design replacement requiring complete matrix rerun.',
        'motivation_evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in evidence},
        'source_at_registration':{name:sha(ROOT/name) for name in ('hand_plant.py','foundation_config.py','calibrate_foundation.py')},
        'physical_gate_thresholds_unchanged':True,
        'full_matrix_rerun_required':True,
        'all_release_impacts_remain_in_peak_statistics':True,
        'no_policy_performance_data_used':True}
    assert result['physical_foundation']['gates']==original['physical_foundation']['gates']
    result['interpretation_boundaries'].append('The admitted box family, if the next matrix passes, is explicitly rounded with the declared envelope. Earlier sharp-box numerical failure remains outside the passed claim.')
    out=ROOT/'acceptance_protocol_rounded_box_v7.json'
    if out.exists():raise SystemExit('Refusing to overwrite a declared geometry/profile revision.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'protocol':str(out),'sha256':sha(out),'physical_gate_thresholds_unchanged':True},ensure_ascii=False))
if __name__=='__main__':main()

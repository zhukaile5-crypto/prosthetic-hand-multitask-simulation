"""Declare full-hand passive tactile observations before policy experiments."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,copy

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parent=ROOT/'acceptance_protocol_foundation_v2.json'
    original=json.loads(parent.read_text(encoding='utf-8'))
    result=copy.deepcopy(original)
    result['schema']='shadow-hand-integration-acceptance-v3-full-hand-observation'
    result['status']='full-hand-observation-contract-before-any-policy-performance-data'
    fields=['auxiliary_wrench[6]','auxiliary_pressure','auxiliary_valid','auxiliary_age','auxiliary_uncertainty']
    result['observation_contract']['required_control_inputs'].extend(fields)
    result['observation_contract']['random_draws_per_tick']={'gaussian':48,'uniform':6,
        'gaussian_components':{'existing_finger_encoder_effort_stream':41,'auxiliary_force':3,'auxiliary_moment':3,'auxiliary_pressure':1},
        'uniform_components':{'finger_dropout':5,'auxiliary_dropout':1},
        'draw_even_without_contact':True}
    result['observation_contract']['auxiliary_tactile']={
        'scope':'All palm and non-digit hand contacts with external bodies, excluding hand self-contact; no target-identity filtering.',
        'wrench_order':['Fx','Fy','Fz','Mx','My','Mz'],
        'coordinate_frame':'palm',
        'force_direction':'hand-on-external-body',
        'moment_origin':'palm origin, including contact torques and r cross force',
        'units':['N','N','N','N m','N m','N m'],
        'time_alignment':'same acquisition time and delay as finger tactile geometry',
        'age_validity_and_noise_explicit':True,
        'engineering_uncertainty_not_a_hard_bound':True,
        'same_sensor_permissions_for_all_methods':True,
        'no_direct_target_object_support_label':True,
        'usage':'Full-hand total tactile load accounting and diagnostics only. Palm may passively carry object load; it is not a separately controlled force variable.',
        'no_optimistic_subtraction_from_finger_support_demand':True,
        'reason':'A non-target external collision can contribute to this sensor; without target identity it cannot be assumed to be useful object support.',
        'total_load_fusion':'Bring finger tactile plus auxiliary and motor-effort estimates to a consistent total-load definition. Never subtract auxiliary twice from a mixed finger-only/motor-total fusion.',
        'missing_or_stale_auxiliary':'Invalid auxiliary is not zero true load and cannot provide a favorable force-reduction credit.'}
    result['observation_revision']={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'parent_protocol':parent.name,'parent_sha256':sha(parent),
        'new_fields':fields,
        'reason':'The articulated hand has passive palm/non-digit contacts absent from a five-pad-only interface. Explicit virtual tactile instrumentation permits common full-hand observation without leaking target identity.',
        'physical_gates_unchanged':True,
        'source_at_registration':{name:sha(ROOT/name) for name in ('contracts.py','sensors.py')},
        'root_decision':'Auxiliary wrench is common observation and total-load accounting; the current unified controller retains conservative total-load support demand. This is physical migration, not the proposed stability-corridor policy.'}
    assert result['physical_foundation']==original['physical_foundation']
    out=ROOT/'acceptance_protocol_full_hand_v3.json'
    if out.exists():raise SystemExit('Refusing to overwrite the observation contract revision.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'protocol':str(out),'sha256':sha(out),'new_fields':fields,'physical_gates_unchanged':True},ensure_ascii=False))
if __name__=='__main__':main()

"""Register a pre-algorithm common contact-design revision, preserving gates."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,copy

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parent=ROOT/'acceptance_protocol.json'
    original=json.loads(parent.read_text(encoding='utf-8'))
    result=copy.deepcopy(original)
    result['schema']='shadow-hand-integration-acceptance-v2-contact-design'
    result['status']='common-physical-design-revision-before-any-policy-performance-data'
    result['physical_foundation']['contact_time_constants_s']=[.008,.012]
    evidence=ROOT/'review/foundation_capsule_probe_v4/summary.json'
    source=json.loads(evidence.read_text(encoding='utf-8'))
    result['design_revision']={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'parent_protocol':parent.name,'parent_sha256':sha(parent),
        'changed_field':'physical_foundation.contact_time_constants_s',
        'old_values':[.02,.03],'new_values':[.008,.012],
        'reason':'Original shared contact family exceeded the already fixed 2 mm hand-object penetration gate during physical development; choose a new common contact design and rerun every shape/contact/timestep lifecycle before policies.',
        'motivating_evidence':str(evidence.relative_to(ROOT)),
        'motivating_evidence_sha256':sha(evidence),
        'observed_max_penetration_m':source['max_penetration'],
        'unchanged_gates':copy.deepcopy(original['physical_foundation']['gates']),
        'scope':'The former design remains failed history. This changes shared simulated contact mechanics, not the acceptance limit or a policy parameter. It does not establish physical realism of a specific prosthesis.',
        'root_agreed_before_new_policy_data':True}
    assert result['physical_foundation']['gates']==original['physical_foundation']['gates']
    out=ROOT/'acceptance_protocol_foundation_v2.json'
    if out.exists():raise SystemExit('Refusing to overwrite design revision evidence.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'protocol':str(out),'sha256':sha(out),'unchanged_gate_count':len(original['physical_foundation']['gates'])},ensure_ascii=False))
if __name__=='__main__':main()

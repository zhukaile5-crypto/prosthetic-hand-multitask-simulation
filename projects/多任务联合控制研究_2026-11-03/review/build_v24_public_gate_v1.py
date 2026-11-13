"""Create a fresh V24 archived-public route gate without altering old attempts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
old = ROOT / 'review/verify_v23_public_route_author_v1.py'
new = ROOT / 'review/verify_v24_public_route_author_v1.py'
assert not new.exists()
text = old.read_text(encoding='utf-8')
text = text.replace('V23', 'V24').replace('v23_task_execution', 'v24_canonical_task_fusion').replace('v23_public_route_author_v1', 'v24_public_route_author_v1')
needle = "if kind=='positive':ck(variant+'/genuine_formulation_certificate',hr['diagnostics']['normal_witness_is_submitted_prefix']==(variant=='backend') and hr['diagnostics']['normal_optimal_slack_frozen_before_motion']==(variant=='backend') and (variant=='backend' or all(c['name']!='normal_inward_correction_slack' for c in hr['certified_prefix'])))"
replacement = """if kind=='positive':
                            hd=hr['diagnostics'];pair=hd['normal_witness_certified_prefix']
                            ck(variant+'/canonical_two_layer_witness',len(pair)==2 and pair[0]['name']=='normal_inward_correction_slack' and pair[1]['name']=='canonical_minimum_normalized_reference_witness' and not hd['normal_witness_is_submitted_prefix'])
                            ck(variant+'/first_normal_only_submission_contract',hd['normal_first_certificate_is_submitted_prefix']==(variant=='backend') and hd['normal_optimal_slack_frozen_before_motion']==(variant=='backend') and all(c['name']!=pair[1]['name'] for c in hr['certified_prefix']) and (variant=='backend' or all(c['name']!=pair[0]['name'] for c in hr['certified_prefix'])))"""
assert needle in text
text = text.replace(needle, replacement)
text = text.replace("production_source_sha256=source,numerical_identity=numeric", "production_source_sha256=source,source_count=len(source),full192=len(source)==192,numerical_identity=numeric")
new.write_text(text, encoding='utf-8')
print(new)

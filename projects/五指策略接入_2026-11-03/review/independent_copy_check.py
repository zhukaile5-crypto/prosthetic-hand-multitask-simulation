"""Small provenance-only check; intentionally does not rehash 16 GB history."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()

def main():
    manifest_path=ROOT/'review/preservation_before.json'
    source_path=ROOT/'review/source_provenance.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    source=json.loads(source_path.read_text(encoding='utf-8'))
    rows=[]
    for relative,expected in source['upstream_assets'].items():
        original=PROJECT/'五指假肢抓握仿真/upstream_shadow'/relative
        copied=ROOT/'assets/shadow'/relative
        before=manifest['files'][str(original)]
        original_hash=digest(original);copy_hash=digest(copied)
        rows.append(dict(relative=relative,original=str(original),copied=str(copied),
                         expected=expected,original_sha256=original_hash,copied_sha256=copy_hash,
                         passed=(expected==before['sha256']==original_hash==copy_hash)))
    for label,original,copied,expected in [
        ('frozen_controller',PROJECT/'统一人工反射重构/controllers.py',ROOT/'vendor/frozen_controllers.py',source['frozen_controller_sha256']),
        ('legacy_scene',PROJECT/'五指假肢抓握仿真/model.xml',ROOT/'assets/legacy_model.xml',source['legacy_scene_sha256'])]:
        original_hash=digest(original);copy_hash=digest(copied)
        rows.append(dict(relative=label,original=str(original),copied=str(copied),expected=expected,
                         original_sha256=original_hash,copied_sha256=copy_hash,
                         passed=(expected==manifest['files'][str(original)]['sha256']==original_hash==copy_hash)))
    protocol=ROOT/'acceptance_protocol.json'
    result=dict(schema='independent-copy-provenance-v1',checked_utc=datetime.now(timezone.utc).isoformat(),
                passed=all(row['passed'] for row in rows),copied_source_count=len(rows),
                full_history_rehashed=False,
                limitation='Only copied model assets, original scene and frozen controller are read here. Final full-inventory rehash is separate.',
                manifest_sha256=digest(manifest_path),source_provenance_sha256=digest(source_path),
                acceptance_protocol_sha256=digest(protocol),checks=rows)
    output=ROOT/'review/independent_copied_sources.json'
    if output.exists(): raise SystemExit('Refusing to overwrite an existing evidence certificate.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['passed','copied_source_count','full_history_rehashed','acceptance_protocol_sha256']},ensure_ascii=False))
    if not result['passed']: raise SystemExit(1)

if __name__=='__main__': main()

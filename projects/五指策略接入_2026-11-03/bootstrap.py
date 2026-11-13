"""Create isolated assets and immutable input provenance; never write to sources."""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
LEGACY = PROJECT / '五指假肢抓握仿真'
FROZEN = PROJECT / '统一人工反射重构'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def main():
    review=ROOT/'review';review.mkdir(exist_ok=True)
    certificate=review/'preservation_before.json'
    if certificate.exists(): raise SystemExit('Already initialized; refusing to overwrite preservation evidence.')
    paths=[p for p in LEGACY.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [p for p in FROZEN.iterdir() if p.is_file()]
    paths += [p for p in (FROZEN/'review').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for name in ('confirmation_test','confirmation_robustness'):
        folder=FROZEN/'output'/name
        paths += [p for p in folder.rglob('*') if p.is_file()]
    for p in (FROZEN/'output/confirmation_combined').rglob('*'):
        if p.is_file(): paths.append(p)
    unique=sorted(set(paths))
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'All existing legacy files except bytecode, current frozen root/review files, and all formal confirmation outputs including NPZ.',
        'files':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in unique}}
    certificate.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    assets=ROOT/'assets'/'shadow'
    assets.parent.mkdir(exist_ok=True)
    shutil.copytree(LEGACY/'upstream_shadow',assets)
    vendor=ROOT/'vendor';vendor.mkdir(exist_ok=True)
    shutil.copy2(FROZEN/'controllers.py',vendor/'frozen_controllers.py')
    (vendor/'__init__.py').write_text('',encoding='utf-8')
    shutil.copy2(LEGACY/'model.xml',ROOT/'assets/legacy_model.xml')
    source={'benchmark_id':'shadow_hand_strategy_integration_v1',
        'upstream_assets':{str(p.relative_to(assets)):sha(p) for p in assets.rglob('*') if p.is_file()},
        'frozen_controller_sha256':sha(vendor/'frozen_controllers.py'),
        'legacy_scene_sha256':sha(ROOT/'assets/legacy_model.xml'),
        'preserved_files':len(unique),'preserved_bytes':sum(v['bytes'] for v in manifest['files'].values())}
    (review/'source_provenance.json').write_text(json.dumps(source,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(source,ensure_ascii=False))

if __name__=='__main__': main()

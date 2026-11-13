"""Union of exact frozen V3/V10 inputs plus only this new seven-file package."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE = Path(__file__).resolve().parent
SOURCE_NAMES = ('__init__.py', 'spec.py', 'provenance.py', 'run_validation.py',
                'clearance_diagnostics.py', 'verify_pure.py', 'plan.json', 'README.md')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def source_files():
    from identification.release_local_response_v3 import provenance as v3
    from v10.runtime import source_files as v10files
    from research_imports import FOUNDATION
    v3lock = json.loads((v3.PACKAGE/'dependency_lock.json').read_text(encoding='utf-8'))
    inherited = v3.source_files()
    if {k:sha(p) for k,p in inherited.items()} != v3lock['sources']:
        raise RuntimeError('Frozen V3 244 identity mismatch')
    # Retain all244 exact inherited keys, including the already pinned external
    # simulator/distribution paths. Do not silently discard those dependencies.
    files = dict(inherited)
    inherited_paths={Path(p).resolve():key for key,p in files.items()}
    for p in [*v10files(['ratio_preview']).values(),
              *(PACKAGE/name for name in SOURCE_NAMES)]:
        p = Path(p).resolve()
        if p in inherited_paths:
            continue
        if p.is_relative_to(ROOT.resolve()):
            key = 'research/' + p.relative_to(ROOT).as_posix()
        elif p.is_relative_to(FOUNDATION.resolve()):
            key = 'foundation/' + p.relative_to(FOUNDATION).as_posix()
        else:
            raise ValueError('Source escaped research/foundation roots')
        if key in files and files[key] != p:
            raise ValueError('Source namespace collision')
        files[key] = p
    # V10 final142 are explicitly checked against the actual finished run.
    manifest = json.loads((ROOT/'output/development_v10_transport_v1/manifest.json').read_text(encoding='utf-8'))
    current = v10files(['ratio_preview'])
    if {k:sha(p) for k,p in current.items()} != manifest['research_source_sha256']:
        raise RuntimeError('Frozen V10 142 identity mismatch')
    return dict(sorted(files.items()))


def identity():
    return {k:sha(p) for k,p in source_files().items()}


def require_seed_inventory():
    from .spec import SEEDS
    p = PACKAGE/'seed_inventory.json'
    if not p.is_file():
        raise RuntimeError('Root must record fresh seed inventory before pin/execution')
    inv = json.loads(p.read_text(encoding='utf-8'))
    if inv.get('seeds') != list(SEEDS) or inv.get('confirmed_fresh_for_this_development') is not True:
        raise RuntimeError('Seed inventory unresolved or different from fixed plan')
    return inv


def require_lock():
    inventory = require_seed_inventory()
    p = PACKAGE/'dependency_lock.json'
    lock = json.loads(p.read_text(encoding='utf-8'))
    if lock['sources'] != identity() or lock['seed_inventory_sha256'] != sha(PACKAGE/'seed_inventory.json'):
        raise RuntimeError('Frozen source or seed inventory changed; use fresh revision')
    return lock


def snapshot(out, lock):
    records = {}
    for key, p in source_files().items():
        q = out/'sources'/key
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        if sha(q) != lock['sources'][key]:
            raise RuntimeError('Source snapshot bytes changed')
        records[key] = dict(original_path=str(p), snapshot_path=q.relative_to(out).as_posix(),
                            sha256=lock['sources'][key], bytes=q.stat().st_size)
    write(out/'source_snapshot_manifest.json', dict(files=records, all_snapshot_bytes_verified=True))

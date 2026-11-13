"""Independently rehash protected original inputs without importing old code.

Only the output certificate in this integration directory is written. The
manifest is a bounded file inventory, not a claim about files absent from it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PROTECTED = (PROJECT/'五指假肢抓握仿真', PROJECT/'统一人工反射重构')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def allowed_original(path: Path) -> bool:
    resolved = path.resolve()
    return any(resolved.is_relative_to(root.resolve()) for root in PROTECTED)


def verify(manifest_path: Path, *, check_copies: bool = True) -> dict:
    """Read original bytes only; return evidence and every mismatch."""
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = manifest.get('files')
    if not isinstance(entries, dict) or not entries:
        raise ValueError('A nonempty bootstrap file inventory is required.')
    failures = []
    checked = 0
    byte_count = 0
    for name, expected in entries.items():
        path = Path(name)
        if not allowed_original(path):
            failures.append({'file': name, 'reason': 'outside declared protected roots'})
            continue
        if not path.is_file():
            failures.append({'file': name, 'reason': 'missing original file'})
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        checked += 1
        byte_count += actual_size
        if actual_hash != expected['sha256'] or actual_size != expected['bytes']:
            failures.append({'file': name, 'reason': 'original content differs',
                             'expected_sha256': expected['sha256'],
                             'actual_sha256': actual_hash,
                             'expected_bytes': expected['bytes'], 'actual_bytes': actual_size})
    copy_checks = {}
    provenance_path = ROOT/'review/source_provenance.json'
    if check_copies:
        if not provenance_path.is_file():
            copy_checks['source_provenance_present'] = False
        else:
            source = json.loads(provenance_path.read_text(encoding='utf-8'))
            copied_assets = source['upstream_assets']
            copy_checks['shadow_assets_match'] = all(
                (ROOT/'assets/shadow'/name).is_file() and
                sha256(ROOT/'assets/shadow'/name) == expected
                for name, expected in copied_assets.items())
            controller = ROOT/'vendor/frozen_controllers.py'
            copy_checks['vendor_controller_matches'] = controller.is_file() and sha256(controller) == source['frozen_controller_sha256']
            scene = ROOT/'assets/legacy_model.xml'
            copy_checks['legacy_scene_copy_matches'] = scene.is_file() and sha256(scene) == source['legacy_scene_sha256']
            copy_checks['provenance_original_count_matches'] = source['preserved_files'] == len(entries)
    passed = not failures and checked == len(entries) and all(copy_checks.values())
    return {'schema': 'shadow-integration-preservation-v1',
            'checked_utc': datetime.now(timezone.utc).isoformat(),
            'passed': passed,
            'scope': manifest.get('scope'),
            'inventory_count': len(entries), 'files_rehashed': checked,
            'bytes_rehashed': byte_count, 'failures': failures,
            'copied_inputs': copy_checks,
            'manifest_sha256': sha256(manifest_path),
            'verifier_sha256': sha256(Path(__file__)),
            'limitations': ['Only registered original files are covered.',
                            'Bytecode was excluded by bootstrap; no old project module is imported.',
                            'A matching inventory proves content preservation, not physics or algorithm performance.']}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'review/preservation_before.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--originals-only', action='store_true')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = (args.out or ROOT/f'review/independent_preservation_{stamp}.json').resolve()
    if not destination.is_relative_to(ROOT.resolve()):
        parser.error('Certificates must stay within the independent integration directory.')
    if destination.exists():
        parser.error('Refusing to overwrite an existing preservation certificate.')
    result = verify(args.manifest.resolve(), check_copies=not args.originals_only)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'files_rehashed': result['files_rehashed'],
                      'bytes_rehashed': result['bytes_rehashed'], 'failure_count': len(result['failures']),
                      'copied_inputs': result['copied_inputs'], 'certificate': str(destination)},
                     ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Small explicit closure for the old source input replay, never the live controller."""
from pathlib import Path
import json
import shutil
import sys
from .source_record import ROOT, PILOT, METHODS, read, sha, canonical

LOCK = PILOT / 'source_lock.json'

def files():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from v12.runtime import source_files
    result = dict(source_files(METHODS))
    for path in PILOT.iterdir():
        if path.name != LOCK.name and path.is_file() and path.suffix in ('.py', '.json', '.md'):
            result[path.relative_to(ROOT).as_posix()] = path
    return dict(sorted(result.items()))

def environment():
    import mujoco
    import numpy
    binaries = {}
    for package in (mujoco, numpy):
        folder = Path(package.__file__).parent
        for pattern in ('*.pyd', '*.dll'):
            for path in folder.rglob(pattern):
                binaries[str(path)] = sha(path)
    return dict(python=sys.version, executable=sys.executable, executable_sha256=sha(sys.executable),
                mujoco_version=mujoco.__version__, numpy_version=numpy.__version__,
                native_package_binaries_sha256=dict(sorted(binaries.items())),
                scope='Pinned package native binaries and executable; not a complete OS/CPU determinism certificate')

def identity():
    return {key: sha(path) for key, path in files().items()}

def ensure_original_sources(records):
    current = identity()
    for record in records:
        expected = record['metadata']['research_source_sha256']
        missing = [key for key, value in expected.items() if current.get(key) != value]
        if missing:
            raise ValueError('Original old V12 declared sources differ: ' + repr(missing))

def pin(records):
    ensure_original_sources(records)
    value = dict(schema='contact_peak_replay_source_lock_v1', sources=identity(), environment=environment(),
                 source_inputs={r['method']: dict(path=str(r['path']), trace_sha256=r['trace_sha256'], metadata_sha256=r['metadata_sha256']) for r in records})
    with LOCK.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return value

def require_lock(records):
    lock = read(LOCK)
    if lock['sources'] != identity() or canonical(lock['environment']) != canonical(environment()):
        raise ValueError('Code, asset or native environment differs from source lock')
    for record in records:
        expected = lock['source_inputs'][record['method']]
        if expected != dict(path=str(record['path']), trace_sha256=record['trace_sha256'], metadata_sha256=record['metadata_sha256']):
            raise ValueError('Original record identity differs from source lock')
    return lock

def snapshot(target, lock):
    target = Path(target)
    for key, path in files().items():
        dest = target / 'sources' / key; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        if sha(dest) != lock['sources'][key]:
            raise ValueError('Source snapshot mismatch')
    with (target / 'source_lock.json').open('x', encoding='utf-8') as stream:
        json.dump(lock, stream, ensure_ascii=False, indent=2, allow_nan=False)

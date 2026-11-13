"""Parent99 provenance superset plus this independent diagnostic package."""
import sys,shutil
from pathlib import Path
from .inputs import ROOT,PACKAGE,COHORT,read,sha,canonical
LOCK=PACKAGE/'source_lock.json'
def files():
    manifest=read(COHORT/'source_snapshot_manifest.json')
    paths={key:Path(item['original_path']) for key,item in manifest['files'].items()}
    for p in PACKAGE.iterdir():
        if p.is_file() and p.name!=LOCK.name and p.suffix in ('.py','.json','.md'):paths[p.relative_to(ROOT).as_posix()]=p
    return dict(sorted(paths.items()))
def identity():return {k:sha(p) for k,p in files().items()}
def environment():
    import mujoco,numpy
    binaries={}
    for package in (mujoco,numpy):
        for pattern in ('*.pyd','*.dll'):
            for p in Path(package.__file__).parent.rglob(pattern):binaries[str(p)]=sha(p)
    return dict(python=sys.version,executable=sys.executable,executable_sha256=sha(sys.executable),mujoco_version=mujoco.__version__,numpy_version=numpy.__version__,native_package_binaries_sha256=dict(sorted(binaries.items())))
def input_identity(record):
    return dict(trace=str(record['path']),trace_sha256=record['trace_sha256'],metadata_sha256=record['metadata_sha256'],
        cohort_complete_sha256=sha(COHORT/'execution_complete.json'),cohort_manifest_sha256=sha(COHORT/'manifest.json'))
def pin(record,preflight):
    current=identity();parent=record['source_manifest']['research_source_sha256']
    if any(current.get(k)!=v for k,v in parent.items()):raise ValueError('Parent99 changed')
    if not preflight['passed'] or preflight['sources']!=current or preflight['input']!=input_identity(record):raise ValueError('Stale or failed preflight')
    value=dict(schema='v13_release_factorial_source_lock_v1',sources=current,parent99=parent,environment=environment(),input=input_identity(record),preflight_sha256=preflight['_file_sha256'],
        scope='Parent99 is deliberately a provenance superset; old policy/solver sources are pinned but are not executed by fixed-input replay.')
    from .run import write
    write(LOCK,value);return value
def require(record):
    lock=read(LOCK)
    if lock['sources']!=identity() or canonical(lock['environment'])!=canonical(environment()) or lock['input']!=input_identity(record):raise ValueError('Source/input/native environment lock mismatch')
    return lock
def snapshot(target,lock):
    for key,path in files().items():
        dest=Path(target)/'sources'/key;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        if sha(dest)!=lock['sources'][key]:raise ValueError('Snapshot bytes mismatch')
    shutil.copyfile(LOCK,Path(target)/'source_lock.json')

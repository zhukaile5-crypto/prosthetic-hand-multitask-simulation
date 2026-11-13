"""Exact read-only dependency/robot-asset identity for the isolated pilot."""
from pathlib import Path
import ast,hashlib,json,shutil,sys,platform,importlib.metadata
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import research_imports
from research_imports import FOUNDATION
PILOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    from v7.runtime import source_files as inherited_files
    pending=list(inherited_files(['ratio']).values())+[ROOT/'runtime.py']
    pending+=list(PILOT.glob('*.py'))+[PILOT/'plan.json']
    # Include the actual robot XML and all copied Shadow mesh assets; a model
    # XML hash alone is not an asset/source snapshot.
    pending+=[FOUNDATION/'assets/legacy_model.xml']
    pending+=[p for p in (FOUNDATION/'assets/shadow').rglob('*') if p.is_file()]
    paths={};visited=set()
    while pending:
        p=Path(pending.pop()).resolve()
        if p in visited:continue
        visited.add(p)
        if not p.is_file():raise FileNotFoundError(p)
        if p.is_relative_to(ROOT.resolve()):key='research/'+p.relative_to(ROOT).as_posix()
        elif p.is_relative_to(FOUNDATION.resolve()):key='foundation/'+p.relative_to(FOUNDATION).as_posix()
        else:raise ValueError('Dependency escaped registered research/foundation roots')
        paths[key]=p
        if p.suffix!='.py':continue
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[]
            if isinstance(node,ast.Import):names=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    parent=p.parent
                    for _ in range(node.level-1):parent=parent.parent
                    stem=parent.joinpath(*(node.module or '').split('.')) if node.module else parent
                    options=[stem.with_suffix('.py'),stem/'__init__.py'] if node.module else [parent/'__init__.py']
                    options.extend(stem/n.name for n in node.names if n.name!='*')
                    for q in options:
                        if q.is_file():pending.append(q)
                        elif q.with_suffix('.py').is_file():pending.append(q.with_suffix('.py'))
                    continue
                names=[node.module] if node.module else []
            for name in names:
                rel=Path(*name.split('.'))
                for base in (PILOT,ROOT,FOUNDATION):
                    found=next((q for q in (base/rel.with_suffix('.py'),base/rel/'__init__.py') if q.is_file()),None)
                    if found is not None:pending.append(found);break
    return dict(sorted(paths.items()))


def identity():return {key:sha(path) for key,path in source_files().items()}


def require_lock():
    p=PILOT/'dependency_lock.json'
    if not p.is_file():raise ValueError('Pin sources with --pin-sources before execution')
    lock=json.loads(p.read_text(encoding='utf-8'))
    actual=identity()
    if lock.get('schema')!='release_basis_source_lock_v1' or actual!=lock.get('sources'):
        raise ValueError('Pinned pilot dependency closure has changed; create a fresh revision, never overwrite old evidence')
    return lock


def snapshot(target,lock):
    target=Path(target);target.mkdir(parents=True,exist_ok=False)
    files=source_files()
    for key,path in files.items():
        destination=target/key;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        if sha(destination)!=lock['sources'][key]:raise ValueError('Copied source bytes differ from pinned identity')


def environment():
    packages={}
    for name in ('numpy','mujoco','scipy'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=packages)

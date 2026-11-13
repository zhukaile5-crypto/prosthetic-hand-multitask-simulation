"""Exact read-only dependency/robot-asset identity for the isolated pilot."""
from pathlib import Path
import ast,hashlib,json,shutil,sys,platform,importlib.metadata,inspect
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import research_imports
from research_imports import FOUNDATION
PILOT=Path(__file__).resolve().parent
_REFERENCE_PATHS=None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def external_reference_files():
    """Pinned identity of the actual MuJoCo API handle used by virtual skin.

    This is not traversal/serialization of a module namespace. The Python API,
    wheel provenance and native bindings are immutable implementation inputs.
    No numpy.random module handle is accepted as a replacement for RNG state.
    """
    import mujoco
    base=Path(mujoco.__file__).resolve().parent
    files={}
    for p in base.rglob('*'):
        if p.is_file() and (p.suffix.lower() in ('.py','.pyd','.dll','.so') or '.so.' in p.name):
            files['external/mujoco/'+p.relative_to(base).as_posix()]=p.resolve()
    dist=importlib.metadata.distribution('mujoco')
    for item in dist.files or ():
        if '.dist-info/' in str(item).replace('\\','/') and Path(str(item)).name in ('WHEEL','METADATA','RECORD'):
            p=Path(dist.locate_file(item)).resolve()
            files['external/mujoco_distribution/'+Path(str(item)).as_posix()]=p
    return files


def source_files():
    from v7.runtime import source_files as inherited_files
    pending=list(inherited_files(['ratio']).values())+[ROOT/'runtime.py']
    pending+=list(PILOT.glob('*.py'))+[PILOT/'plan.json']
    # Include the actual robot XML and all copied Shadow mesh assets; a model
    # XML hash alone is not an asset/source snapshot.
    pending+=[FOUNDATION/'assets/legacy_model.xml']
    pending+=[p for p in (FOUNDATION/'assets/shadow').rglob('*') if p.is_file()]
    external=external_reference_files();external_by_path={p:key for key,p in external.items()}
    pending+=list(external.values())
    paths={};visited=set()
    while pending:
        p=Path(pending.pop()).resolve()
        if p in visited:continue
        visited.add(p)
        if not p.is_file():raise FileNotFoundError(p)
        if p.is_relative_to(ROOT.resolve()):key='research/'+p.relative_to(ROOT).as_posix()
        elif p.is_relative_to(FOUNDATION.resolve()):key='foundation/'+p.relative_to(FOUNDATION).as_posix()
        elif p in external_by_path:key=external_by_path[p]
        else:raise ValueError('Dependency escaped registered research/foundation/explicit MuJoCo inputs')
        paths[key]=p
        if p.suffix!='.py' or key.startswith('external/'):continue
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


def module_reference(value,path):
    """Only the actually audited virtual-skin MuJoCo implementation handle."""
    import types
    if not isinstance(value,types.ModuleType) or value.__name__!='mujoco':
        raise TypeError(f'Unregistered module implementation handle at {path}: {getattr(value,"__name__",type(value).__name__)}')
    origin=Path(inspect.getfile(value)).resolve();files=external_reference_files()
    key=next((key for key,p in files.items() if p==origin),None)
    if key is None:raise TypeError(f'MuJoCo module origin is not in explicit dependency closure at {path}')
    actual=sha(origin);lock_path=PILOT/'dependency_lock.json';locked=False
    if lock_path.exists():
        lock=json.loads(lock_path.read_text(encoding='utf-8'))
        if lock.get('sources',{}).get(key)!=actual:raise ValueError('Module reference differs from source pin at '+path)
        locked=True
    return dict(code_reference='audited_immutable_mujoco_API_handle',module_name=value.__name__,
        module_version=importlib.metadata.version('mujoco'),origin_source_key=key,origin_sha256=actual,
        covered_native_binding_sha256={k:sha(p) for k,p in files.items() if p.suffix.lower() in ('.pyd','.dll','.so') or '.so.' in p.name},
        source_lock_verified=locked,namespace_not_serialized=True)


def instance_reference(value,path):
    """Identity of real Python instance implementations, with mutable data
    still serialized separately. Unexpected foreign classes are not assumed
    to be safe immutable dictionaries.
    """
    global _REFERENCE_PATHS
    cls=type(value)
    if cls.__module__=='builtins':return dict(implementation='python_builtin',class_name=cls.__qualname__,python_version=sys.version)
    try:origin=Path(inspect.getfile(cls)).resolve()
    except (TypeError,OSError) as exc:raise TypeError('Instance implementation origin unavailable at '+path) from exc
    if _REFERENCE_PATHS is None:_REFERENCE_PATHS={p:key for key,p in source_files().items()}
    key=_REFERENCE_PATHS.get(origin)
    if key is None:raise TypeError('Instance implementation not in explicit source closure at '+path+': '+str(origin))
    actual=sha(origin);lock_path=PILOT/'dependency_lock.json';locked=False
    if lock_path.exists():
        lock=json.loads(lock_path.read_text(encoding='utf-8'))
        if lock.get('sources',{}).get(key)!=actual:raise ValueError('Instance implementation changed from source pin at '+path)
        locked=True
    return dict(class_name=cls.__module__+'.'+cls.__qualname__,origin_source_key=key,origin_sha256=actual,source_lock_verified=locked)


def require_lock():
    p=PILOT/'dependency_lock.json'
    if not p.is_file():raise ValueError('Pin sources with --pin-sources before execution')
    lock=json.loads(p.read_text(encoding='utf-8'))
    actual=identity()
    if lock.get('schema')!='release_basis_source_lock_v2' or actual!=lock.get('sources'):
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

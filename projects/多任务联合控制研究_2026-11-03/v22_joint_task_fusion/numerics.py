"""Explicit process-local numerical environment for the frozen legacy carrier.

No installation, model, writer, optimization or registry/environment mutation.
The physical interpreter remains the original venv. NumPy/MuJoCo origins never
switch; existing research312 contributes SciPy by APPENDED lookup only.
"""
from pathlib import Path
import hashlib,importlib,json,os,sys

ROOT=Path(__file__).resolve().parents[1]
LEGACY_VENV=ROOT.parent/'仿真调研_2026-10-08/local_probe/.venv'
LEGACY_EXECUTABLE=LEGACY_VENV/'Scripts/python.exe'
LEGACY_SITE=LEGACY_VENV/'Lib/site-packages'
RESEARCH_PREFIX=Path('D:/conda_envs/research312')
RESEARCH_SITE=RESEARCH_PREFIX/'Lib/site-packages'
LIBRARY_BIN=RESEARCH_PREFIX/'Library/bin'
EXPECTED_LEGACY_ENVIRONMENT_SHA256='33feb486cb897b56359cb69520551bcad5da18b14f0237b22b1f4d13457eb7c4'
EXPECTED_MODULES={
 'numpy':('2.5.3',LEGACY_SITE/'numpy/__init__.py','a6958cb364663b7acce81ccfd58eeb65a2b34d5376157f924777b97211a73be4'),
 'mujoco':('3.15.0',LEGACY_SITE/'mujoco/__init__.py','f0986ea9d51192d2a435f5859718b38820ec155b5080fc623e393bc0aa42a3d1'),
 'scipy':('1.18.1',RESEARCH_SITE/'scipy/__init__.py','39ccae300a4739cc53719bd3b39ce3f8f66e736c6eb672006e338a4c5cc3dc76')}
PYTHON_IMPLEMENTATION_MODULES=(
 'scipy','scipy.optimize._minimize','scipy.optimize._slsqp_py',
 'scipy.optimize._constraints','scipy.optimize._linprog',
 'scipy.optimize._linprog_highs','scipy.optimize._lsq.lsq_linear',
 'scipy.optimize._lsq.bvls','scipy.optimize._lsq.common',
 'scipy.linalg._basic','scipy._lib._util')
# Explicit existing DLL dependencies seen in the import/small-math gate. This
# stable declared set avoids identity changing merely through lazy loading.
RESEARCH_DLL_NAMES=(
 'libcrypto-3-x64.dll','ffi-8.dll','libbz2.dll','liblzma.dll',
 'liblapack.dll','libcblas.dll','libblas.dll','mkl_rt.3.dll',
 'mkl_intel_thread.3.dll','mkl_core.3.dll','mkl_avx2.3.dll',
 'mkl_vml_avx2.3.dll','libiomp5md.dll','libomp.dll','libexpat.dll')


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def _pathkey(path):return os.path.normcase(os.path.abspath(os.fspath(path)))


def _check_module(name,module):
    version,expected_file,expected_sha=EXPECTED_MODULES[name]
    actual=getattr(module,'__file__',None)
    if (getattr(module,'__version__',None)!=version or actual is None or
        _pathkey(actual)!=_pathkey(expected_file) or _sha(actual)!=expected_sha):
        raise RuntimeError('V22 numerical module origin/version/bytes mismatch: '+name)
    root=expected_file.parent.resolve()
    for key,child in tuple(sys.modules.items()):
        if key!=name and not key.startswith(name+'.'):continue
        source=getattr(child,'__file__',None)
        if source is not None and not Path(source).resolve().is_relative_to(root):
            raise RuntimeError('V22 already-loaded foreign package submodule: '+key)
    return dict(version=version,file=str(expected_file),init_sha256=expected_sha)


def ensure_numeric_environment():
    """Idempotently configure ONLY this process; refuse a different carrier.

    Call before SciPy imports in both parent and every spawned worker. Existing
    loaded NumPy/MuJoCo must be the unchanged legacy venv versions/locations.
    No sys.modules entry is removed/replaced and no package is copied.
    """
    if _pathkey(sys.executable)!=_pathkey(LEGACY_EXECUTABLE):
        raise RuntimeError('V22 numerical carrier requires the original legacy venv interpreter')
    if not RESEARCH_SITE.is_dir() or not LIBRARY_BIN.is_dir():
        raise RuntimeError('V22 existing research312 numerical dependency directory missing')
    # Process declarations only; these are not claims about hardware realtime.
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    import numpy
    import mujoco
    origins={name:_check_module(name,module) for name,module in (('numpy',numpy),('mujoco',mujoco))}
    if 'scipy' in sys.modules:_check_module('scipy',sys.modules['scipy'])
    sitekey=_pathkey(RESEARCH_SITE)
    if not any(_pathkey(p)==sitekey for p in sys.path if p):sys.path.append(str(RESEARCH_SITE))
    paths=os.environ.get('PATH','').split(os.pathsep);key=_pathkey(LIBRARY_BIN)
    os.environ['PATH']=os.pathsep.join([str(LIBRARY_BIN)]+[p for p in paths if not p or _pathkey(p)!=key])
    scipy=importlib.import_module('scipy');origins['scipy']=_check_module('scipy',scipy)
    return dict(schema='v22_process_numeric_setup_v1',executable=str(LEGACY_EXECUTABLE),
        package_origins=origins,appended_site=str(RESEARCH_SITE),process_path_prefix=str(LIBRARY_BIN),
        thread_environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        package_installation_performed=False,global_environment_modified=False)


def _native_files(package):
    return {str(p):_sha(p) for pattern in ('*.pyd','*.dll')
            for p in sorted(Path(package.__file__).parent.rglob(pattern))}


def numerical_identity():
    """Stable file-content identity, never a dynamic loaded-module snapshot.

    The old environment() keys/formula are reproduced and checked against the
    actually executed legacy42 digest. New SciPy/native/wrapper/DLL pins are a
    declared ADDITION, not a claim that the augmented environment is unchanged.
    No optimizer, sensor or engine call occurs here.
    """
    setup=ensure_numeric_environment()
    import numpy,mujoco,scipy
    native={}
    for package in (mujoco,numpy):native.update(_native_files(package))
    legacy=dict(python=sys.version,executable=sys.executable,executable_sha256=_sha(sys.executable),
        mujoco=mujoco.__version__,numpy=numpy.__version__,native_binaries_sha256=dict(sorted(native.items())))
    digest=hashlib.sha256(json.dumps(legacy,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf8')).hexdigest()
    if len(native)!=42 or digest!=EXPECTED_LEGACY_ENVIRONMENT_SHA256:
        raise RuntimeError('V22 original legacy42 environment bytes changed')
    wrappers={}
    for name in PYTHON_IMPLEMENTATION_MODULES:
        module=importlib.import_module(name);path=Path(module.__file__).resolve()
        if not path.is_relative_to((RESEARCH_SITE/'scipy').resolve()):
            raise RuntimeError('V22 SciPy implementation source outside declared package: '+name)
        wrappers[name]=dict(path=str(path),sha256=_sha(path))
    _check_module('scipy',scipy)
    dependencies=[LIBRARY_BIN/name for name in RESEARCH_DLL_NAMES]
    dependencies.append(RESEARCH_PREFIX/'python312.dll')
    for folder in (LEGACY_SITE/'numpy.libs',LEGACY_SITE/'glfw'):
        if not folder.is_dir():raise RuntimeError('V22 declared existing dependency folder missing: '+str(folder))
        dependencies.extend(sorted(folder.glob('*.dll')))
    dlls={}
    for path in dependencies:
        if not path.is_file():raise RuntimeError('V22 declared existing runtime DLL missing: '+str(path))
        dlls[str(path)]=_sha(path)
    return dict(schema='v22_numerical_identity_v1',setup=setup,
        legacy_pinned_environment=legacy,legacy_environment_sha256=digest,
        original_legacy42_unchanged=True,scipy_version=scipy.__version__,
        scipy_native_binaries_sha256=dict(sorted(_native_files(scipy).items())),
        scipy_python_implementation_sha256=wrappers,necessary_runtime_dll_sha256=dict(sorted(dlls.items())),
        environment_augmented_with_existing_scipy=True,package_installation_performed=False,
        global_environment_modified=False,physics_bridge_verified_by_this_helper=False)


# Naming compatibility for consumers that explicitly configure before importing.
configure_numerics=ensure_numeric_environment

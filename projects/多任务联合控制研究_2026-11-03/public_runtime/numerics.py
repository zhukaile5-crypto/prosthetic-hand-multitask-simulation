"""Record the active portable environment; do not assert old-machine parity."""
import importlib
import os
import platform
import sys


def ensure_numeric_environment():
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    return numerical_identity()


def numerical_identity():
    modules = {name: importlib.import_module(name) for name in ("numpy", "mujoco")}
    return dict(
        python=platform.python_version(), implementation=platform.python_implementation(),
        platform=platform.platform(), executable=sys.executable,
        packages={name: dict(version=module.__version__, file=module.__file__)
                  for name, module in modules.items()},
        thread_environment={key: os.environ.get(key) for key in
                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        equality_to_historical_environment_claimed=False,
        independent_physical_acceptance="not_run",
    )

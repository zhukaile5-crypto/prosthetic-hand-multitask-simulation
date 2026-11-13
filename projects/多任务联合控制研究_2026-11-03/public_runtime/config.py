"""Explicit runtime parameters, without historical physical acceptance claims."""
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import research_imports
from contracts import public_calibration
from foundation_config import FoundationConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "public_runtime/configs/public_hand_parameters.json"
PARAMETER_KEYS = (
    "config", "force_matrix", "force_tau", "model_reference_alpha",
    "response_error_bound", "force_gain", "nominal_force_weights",
    "selected_radius", "effort_bias_N", "load_error_bound_N",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_context(path=None, shape="capsule"):
    path = DEFAULT_CONFIG if path is None else Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("schema") != "public_hand_runtime_parameters_v1":
        raise ValueError("Unsupported public runtime parameter schema")
    raw = deepcopy(document["shapes"][shape])
    missing = set(PARAMETER_KEYS) - raw.keys()
    if missing:
        raise ValueError(f"Missing explicit runtime parameters: {sorted(missing)}")
    if any(k in raw for k in ("selected_holdout", "passed", "source_probe_path")):
        raise ValueError("Runtime configuration must not impersonate a physical certificate")
    calibration = public_calibration(raw)
    radius = float(raw["selected_radius"])
    if not np.isfinite(radius) or not 0. < radius <= .04:
        raise ValueError("Declared local radius must be finite and in (0, 0.04]")
    for key in ("effort_bias_N", "load_error_bound_N"):
        if not np.isfinite(float(raw[key])):
            raise ValueError(f"{key} must be finite")
    if float(raw["load_error_bound_N"]) < 0.:
        raise ValueError("Declared load error bound must be nonnegative")
    allowed = {f.name for f in fields(FoundationConfig)}
    if set(raw["config"]) - allowed or raw["config"].get("shape") != shape:
        raise ValueError("Unexpected or inconsistent physical configuration")
    FoundationConfig(**raw["config"])
    # Keep plain lists and the exact provided coefficients in metadata. The
    # sensor class performs its existing normalization at construction time.
    calibration = raw
    provenance = dict(
        execution_distribution="public_source_release",
        runtime_parameters_sha256=sha(path),
        runtime_parameter_file=path.name,
        input_configuration_checks="passed",
        historical_validation_evidence_included=False,
        new_environment_physical_acceptance="not_run",
        local_radius_status="declared_parameter_not_revalidated_here",
    )
    from .provenance import identity
    sources = identity()
    foundation_sources = {
        key.removeprefix("foundation/"): value
        for key, value in sources.items() if key.startswith("foundation/")
    }
    return dict(
        calibration=calibration, declared_radius=radius, provenance=provenance,
        admission=dict(
            passed=False, status="historical_admission_not_included",
            manifest_sha256=None, record=dict(source_sha256=foundation_sources),
        ),
    )

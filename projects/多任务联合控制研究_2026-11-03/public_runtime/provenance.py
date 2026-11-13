"""Current source/asset identity, independent of any archived experiment."""
import hashlib
import json
from pathlib import Path
import research_imports
from research_imports import ROOT, FOUNDATION

MANIFEST = Path(__file__).parent / "configs/runtime_inputs.json"


def source_files():
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    result = {}
    for key in record["paths"]:
        root = FOUNDATION if key.startswith("foundation/") else ROOT
        relative = key.removeprefix("foundation/") if key.startswith("foundation/") else key
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or escaping public runtime input: {key}")
        result[key] = path
    for path in Path(__file__).parent.rglob("*"):
        if path.is_file() and path.suffix in (".py", ".json") and "__pycache__" not in path.parts:
            result[path.relative_to(ROOT).as_posix()] = path
    return dict(sorted(result.items()))


def identity():
    return {key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in source_files().items()}

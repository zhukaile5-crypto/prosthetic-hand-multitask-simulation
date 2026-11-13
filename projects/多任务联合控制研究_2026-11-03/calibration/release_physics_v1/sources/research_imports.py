"""Read-only use of the admitted foundation. Research writes stay here."""
from pathlib import Path
import sys
import hashlib

ROOT = Path(__file__).resolve().parent
FOUNDATION = ROOT.parent / '五指策略接入_2026-11-03'
sys.dont_write_bytecode = True
if str(FOUNDATION) not in sys.path:
    sys.path.insert(1, str(FOUNDATION))


def source_identity():
    """One declared execution/measurement identity shared by runner and trials."""
    names={'research_imports.py','runtime.py','run_development.py','evaluation.py','protocol.json'}
    names.update(p.name for p in ROOT.glob('coordinator*.py'))
    names.update(p.name for p in ROOT.glob('lifecycle*.py'))
    model='calibration/public_model_phase1_v1.json'
    if (ROOT/model).is_file():names.add(model)
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in sorted(names) if (ROOT/name).is_file()}


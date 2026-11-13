"""Read-only use of the admitted foundation. Research writes stay here."""
from pathlib import Path
import sys
import hashlib

ROOT = Path(__file__).resolve().parent
FOUNDATION = ROOT.parent / '五指策略接入_2026-11-03'
sys.dont_write_bytecode = True
if str(FOUNDATION) not in sys.path:
    sys.path.insert(1, str(FOUNDATION))


def source_identity(methods=None):
    """One declared execution/measurement identity shared by runner and trials."""
    names={'research_imports.py','runtime.py','run_development.py','evaluation.py','protocol.json'}
    holding=None if methods is None else {m.removesuffix('_release') for m in methods}
    if holding is None or 'joint' in holding:
        names.update(p.name for p in ROOT.glob('coordinator*.py'))
    if methods is None or any(m.endswith('_release') for m in methods):
        names.update(p.name for p in ROOT.glob('lifecycle*.py'))
        if (ROOT/'lifecycle_library.json').is_file():names.add('lifecycle_library.json')
    if holding is None or 'passive' in holding:
        names.add('passive_control.py')
    names.update(p.name for p in ROOT.glob('skin_*.py'))
    model='calibration/public_model_phase1_v1.json'
    if (holding is None or 'joint' in holding) and (ROOT/model).is_file():names.add(model)
    model2='calibration/public_model_phase2_v1.json'
    if (holding is None or 'joint' in holding) and (ROOT/model2).is_file():names.add(model2)
    contact_model='calibration/public_contact_model_phase2_v1.json'
    if (holding is None or 'joint' in holding) and (ROOT/contact_model).is_file():names.add(contact_model)
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in sorted(names) if (ROOT/name).is_file()}


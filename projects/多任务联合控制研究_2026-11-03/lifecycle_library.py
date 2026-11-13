"""Public primitive admissibility, independent of scene identity or truth."""
from pathlib import Path
import json

PATH=Path(__file__).resolve().parent/'lifecycle_library.json'

def select_public_primitive(shape):
    library=json.loads(PATH.read_text(encoding='utf-8'))
    record=library['contexts'].get(shape)
    if record is None or record['selected'] is None:
        raise ValueError('No reviewed nominal release primitive for the declared public morphology')
    return record['selected'],record

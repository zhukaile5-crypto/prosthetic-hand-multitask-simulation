"""Saved input inventory. This module imports no simulator and writes nothing."""
from pathlib import Path
import hashlib
import json
import numpy as np

PILOT = Path(__file__).resolve().parent
ROOT = PILOT.parents[1]
METHODS = ('ratio_effort', 'fixed_effort', 'fusion_effort')
COHORT = ROOT / 'output/development_v12_hold_seed617902_v1'
ARRAYS = ('time', 'qpos', 'qvel', 'ctrl_before', 'ctrl', 'alpha_before', 'alpha',
          'phase', 'normal', 'skin_normal', 'palm', 'hand_other', 'all_hand_object',
          'support', 'penetration', 'world_penetration', 'position', 'relative',
          'rotation', 'relative_velocity', 'angular_velocity', 'lift', 'wrist_position',
          'warning_count', 'lift_ref', 'transport_paused', 'executed_motion_reference', 'robot_actuator_target21',
          'legacy_common_sensor_json', 'observation_json', 'environment_json',
          'substep_time', 'substep_duration', 'substep_normal', 'substep_skin_normal',
          'substep_palm', 'substep_hand_other', 'substep_all_hand_object',
          'substep_support', 'substep_penetration', 'substep_world_penetration')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

def load(method):
    if method not in METHODS:
        raise ValueError('Only the three predeclared source methods are allowed')
    done = read(COHORT / 'execution_complete.json')
    if not done['execution_complete'] or done['runtime_failures']:
        raise ValueError('Complete old source cohort required')
    jobs = [j for j in read(COHORT / 'manifest.json')['jobs']
            if j['method'] == method and j['scenario'] == 'collision' and j['seed'] == 617902 and j['motion_mode'] == 'hold']
    if len(jobs) != 1:
        raise ValueError('Exact predeclared source identity missing or duplicated')
    path = Path(jobs[0]['destination']) / 'trace.npz'
    metadata = read(path.with_name('metadata.json'))
    fingerprint = sha(path)
    if not metadata['completed'] or fingerprint != metadata['trace_sha256']:
        raise ValueError('Completed immutable source record required')
    with np.load(path, allow_pickle=False) as archive:
        missing = set(ARRAYS) - set(archive.files)
        if missing:
            raise ValueError('Missing source fields: ' + repr(sorted(missing)))
        a = {key: archive[key] for key in ARRAYS}  # Decompress each column once.
    if not exact(a['time'], np.arange(3201, dtype=float) * .005):
        raise ValueError('Fixed public input/sample clock not recovered exactly')
    if a['ctrl'].shape != (3201, 22) or a['ctrl_before'].shape != a['ctrl'].shape:
        raise ValueError('Complete native22 control record required; actual21 alone is insufficient')
    if not np.isfinite(a['ctrl']).all() or metadata['config']['timestep'] != .0005:
        raise ValueError('Finite original0.5ms inputs required')
    release_index = int(np.argmin(abs(a['time'] - metadata['mechanical_release_at'])))
    entry = json.loads(str(a['observation_json'][release_index]))
    environments = [json.loads(str(x)) for x in a['environment_json']]
    for env in environments:
        if set(env) != {'force', 'torque', 'squeeze', 'obstacle_x', 'friction'}:
            raise ValueError('Full original environment must be explicitly recovered')
    return dict(method=method, path=path, trace_sha256=fingerprint, metadata=metadata,
                metadata_sha256=sha(path.with_name('metadata.json')), job=jobs[0], arrays=a,
                environments=environments, release_index=release_index,
                release_entry_motion=np.asarray(entry['executed_motion_reference'], dtype=float))

def require_unchanged(record):
    if sha(record['path']) != record['trace_sha256']:
        raise ValueError('Original source trace changed during diagnosis')

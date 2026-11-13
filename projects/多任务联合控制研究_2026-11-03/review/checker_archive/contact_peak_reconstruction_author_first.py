"""Native command recoverability tests, permitted with zero integration only."""
from dataclasses import asdict
import sys
import numpy as np
from .source_record import ROOT, canonical, exact, require_unchanged

def imports():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import research_imports  # Establish the frozen foundation search path.
    import mujoco
    from foundation_config import FoundationConfig
    from hand_plant import HandPlant
    return mujoco, FoundationConfig, HandPlant

def config_for(record, dt):
    _, Config, _ = imports()
    values = dict(record['metadata']['config'])
    expected = dict(values)
    values['timestep'] = float(dt)
    config = Config(**values)
    expected['timestep'] = float(dt)
    if canonical(asdict(config)) != canonical(expected):
        raise ValueError('Actual randomized config differs, beyond JSON tuple/list representation')
    if .005 / dt != round(.005 / dt):
        raise ValueError('Integer physical substeps per unchanged public tick required')
    return config

def robot_ids(plant):
    ids = np.r_[np.concatenate(plant.actuator_groups), plant.lift_actuator,
                plant.aid('rh_A_WRJ2'), plant.aid('rh_A_WRJ1')].astype(int)
    if len(ids) != 21 or len(set(ids)) != 21 or plant.model.nu != 22:
        raise ValueError('This minimal pilot requires exact21 robot +1 obstacle actuator')
    return ids

def release_fields(plant, record):
    a = record['arrays']; k = record['release_index']
    return dict(release_started=float(a['time'][k]), release_alpha=a['alpha_before'][k].copy(),
                release_thumb=a['ctrl_before'][k, plant.actuator_groups[0]].copy(),
                release_ctrl=a['ctrl_before'][k].copy(),
                release_lift=float(record['release_entry_motion'][0]))

def verify_native_commands(record):
    """Compile/reset only; inspect every source interval with original decoder.

    Source has no micro-control log. Native implementation plus all boundary
    equalities establish its exact constant-in-public-tick definition. This
    does not establish physical reproduction: the separate 3-run bridge does.
    """
    mj, _, Plant = imports()
    forbidden = {}
    count = [0]
    def disallow(*args, **kwargs):
        count[0] += 1
        raise AssertionError('Integration is forbidden in input preflight')
    for key in ('mj_step', 'mj_step1', 'mj_step2'):
        forbidden[key] = getattr(mj, key); setattr(mj, key, disallow)
    try:
        p = Plant(config_for(record, .0005)); a = record['arrays']; ids = robot_ids(p)
        if p.model_sha256 != record['metadata']['model_sha256']:
            raise ValueError('Original model identity not reproduced')
        initial = dict(qpos=exact(p.data.qpos, a['qpos'][0]), qvel=exact(p.data.qvel, a['qvel'][0]),
                       ctrl=exact(p.data.ctrl, a['ctrl_before'][0]), alpha=exact(p.alpha, a['alpha_before'][0]))
        if not all(initial.values()):
            raise ValueError('Reset/initial source state differs: ' + repr(initial))
        release = release_fields(p, record); checks = 0
        for k, t in enumerate(a['time']):
            p.control_steps = k
            p.alpha = a['alpha'][k].copy()
            p.lift_ref = float(a['executed_motion_reference'][k, 0])
            p.wrist_ref = a['executed_motion_reference'][k, 1:].copy()
            p.env = dict(force=np.zeros(3), torque=np.zeros(3), squeeze=0., obstacle_x=0.) if k == 0 else dict(record['environments'][k - 1])
            if k >= record['release_index']:
                for key, value in release.items():
                    setattr(p, key, value.copy() if isinstance(value, np.ndarray) else value)
            else:
                p.release_started = None
            # Original implementation reads p.time=control_steps*control_dt,
            # not data.time: probe three private physical-time values.
            for offset in (0., .0025, .004875):
                p.data.time = float(t + offset)
                p._apply_ctrl()
                if not exact(p.data.ctrl, a['ctrl'][k]):
                    axes = np.flatnonzero(p.data.ctrl.view(np.uint64) != a['ctrl'][k].view(np.uint64)).tolist()
                    raise ValueError(f'Native full22 ctrl mismatch tick={k}, offset={offset}, axes={axes}')
                checks += 1
            # The obstacle motor is updated after the row: verify all21 robot
            # references remain exact under the current (rather than prior) env.
            p.env = dict(record['environments'][k]); p._apply_ctrl()
            if not exact(p.data.ctrl[ids], a['ctrl'][k, ids]):
                raise ValueError('Environment unexpectedly changes robot inputs')
            expected_obstacle = float(np.clip(p.env['obstacle_x'], -.4, .1))
            if p.data.ctrl[p.obstacle_motor] != expected_obstacle:
                raise ValueError('Current environment obstacle input not recovered')
            checks += 1
        require_unchanged(record)
        return dict(method=record['method'], passed=True, integration_calls=count[0],
                    full22_native_interval_checks=checks, initial_state_checks=initial,
                    source_trace_sha256=record['trace_sha256'], source_model_sha256=p.model_sha256,
                    actual_randomized_config=asdict(p.config), robot_actuator_ids21=ids.tolist(),
                    release_entry={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in release.items()},
                    recoverability='Exact source decoder and every boundary control verified; full initial integration state absent, so physics identity still requires bridge')
    finally:
        for key, value in forbidden.items():
            setattr(mj, key, value)

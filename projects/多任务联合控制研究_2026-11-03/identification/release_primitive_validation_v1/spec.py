"""Pure fixed assignment/clock specification; no simulation imports."""
from pathlib import Path
from numbers import Integral
import numpy as np

SCENARIOS = ('nominal', 'friction_drop', 'fragile_conflict', 'collision')
MOTION_MODES = ('hold', 'transport')
SEEDS = (617901, 617902)
VARIANTS = ('zero', 'lift_slow')
ORIGIN_TICK = 70
RELEASE_TICK = 2400
RELEASE_S = 12.0
ORIGIN_S = 12.350
FULL_INPUT_S = 12.370
FIRST_FULL_END_S = 12.3705


def validate_context(scenario, motion_mode, seed, variant):
    if scenario not in SCENARIOS or motion_mode not in MOTION_MODES:
        raise ValueError('Fixed four environments and two public motion tasks required')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or int(seed) not in SEEDS:
        raise ValueError('Only the two registered development seeds are allowed')
    if variant not in VARIANTS:
        raise ValueError('Only frozen zero or full20ms lift_slow is allowed')


def jobs(out):
    out = Path(out)
    answer = []
    for seed in SEEDS:
        for scenario in SCENARIOS:
            for motion in MOTION_MODES:
                for variant in VARIANTS:
                    key = f'{scenario}__{motion}__{seed}__{variant}'
                    answer.append(dict(job_id=key, scenario=scenario,
                        motion_mode=motion, seed=seed, variant=variant,
                        method='ratio_preview', shape='capsule', split='robustness',
                        directory=str(out/'runs'/key)))
    return answer

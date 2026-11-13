"""Shared, algorithm-independent task schedules and randomized physical parameters."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    shape: str = 'capsule'
    size: float = 1.0
    mass: float = .12
    friction: float = .60
    event: str = 'none'
    limit: tuple = (5., 5., 5., 5., 5.)
    budget: float = 22.
    noise: float = .025
    delay: float = .010
    duration: float = 17.
    release_at: float | None = None
    boundary: bool = False


SCENARIOS = [
    Scenario('nominal', '常规静态持握'),
    Scenario('cylinder_small', '小柱体与低摩擦', 'cylinder', .9, .10, .35),
    Scenario('box_heavy', '大盒体与较高质量', 'box', 1.05, .23, .55),
    Scenario('sphere', '球体持握', 'sphere', .70, .10, .65),
    Scenario('friction_drop', '持握中摩擦突降', event='friction'),
    Scenario('three_pulses', '三次轴向拉动', event='pulses'),
    Scenario('load_ramp', '逐步增加悬挂载荷', event='ramp'),
    Scenario('fragile_conflict', '防滑与限力并发', event='conflict',
             limit=(4.1, 2.5, 2.5, 2.5, 2.5), budget=12.5),
    Scenario('local_limit', '局部受限接触与拉动', event='pulses',
             limit=(4.2, 1.7, 3.5, 3.5, 3.5), budget=14.),
    Scenario('release_slip', '拉动中主动释放', event='release', release_at=12.),
    Scenario('dynamic_transport', '动态升降与腕部姿态变化', event='transport'),
    Scenario('sensor_noise', '传感噪声与短时离群信号', event='noise', noise=.12, delay=.025),
    Scenario('sensor_dropout', '延迟与局部传感丢包', event='dropout', noise=.06, delay=.045),
    Scenario('collision', '夹持与障碍碰撞并发', event='collision'),
    Scenario('infeasible', '极低摩擦与高载荷边界', event='infeasible',
             friction=.60, limit=(3., 2., 2., 2., 2.), budget=10., boundary=True),
    Scenario('contact_robustness', '较柔顺接触与组合扰动', event='combined'),
]
DEV_NAMES = ('nominal', 'three_pulses', 'friction_drop', 'fragile_conflict',
             'release_slip', 'sensor_noise')


def parameters(scenario, seed, robustness=False):
    rng = np.random.default_rng(seed)
    size = round(scenario.size * rng.choice([.97, 1., 1.03]), 3)
    mass = scenario.mass * rng.uniform(.90, 1.10)
    friction = scenario.friction * rng.uniform(.90, 1.10)
    pulse_scale = rng.uniform(.90, 1.10)
    solref = .02
    if scenario.event == 'combined':
        solref = .03
    if robustness:
        mass *= rng.uniform(.85, 1.20)
        friction *= rng.uniform(.80, 1.20)
        pulse_scale *= rng.uniform(.85, 1.15)
        solref = float(rng.choice([.012, .028]))
    return dict(size=size, mass=float(mass), friction=float(friction),
                pulse_scale=float(pulse_scale), solref=solref,
                force_noise=scenario.noise * (1.3 if robustness else 1.),
                delay=scenario.delay + (.015 if robustness else 0.))


def schedule(scenario, p, time):
    """Environment only: none of these future events enter the sensor interface."""
    mu = p['friction']
    force = np.zeros(3)
    torque = 0.
    lift_delta = 0.
    wrist = 0.
    obstacle = np.array([.28, 0., .49])
    ev = scenario.event
    if ev in ('friction', 'combined') and time >= 9.:
        mu = .21 * p['pulse_scale']
    if ev in ('pulses', 'combined'):
        for onset, strength in [(9., 3.6), (11., 4.2), (13., 3.8)]:
            if onset <= time < onset + .45:
                force[1] = strength * p['pulse_scale']
    if ev == 'ramp':
        force[2] = -2.6 * np.clip((time - 9.) / 3., 0., 1.) * p['pulse_scale']
    if ev == 'conflict' and 9. <= time < 9.65:
        force[1] = 3.7 * p['pulse_scale']
        torque = .10 * p['pulse_scale']
    if ev == 'release' and 11.6 <= time < 12.6:
        force[1] = 3.8 * p['pulse_scale']
    if ev == 'transport' and time >= 9.:
        lift_delta = .015 * np.sin(2. * np.pi * .7 * (time - 9.))
        wrist = .12 * np.sin(2. * np.pi * .45 * (time - 9.))
    if ev == 'collision':
        if 9. <= time < 9.6:
            obstacle[0] = .28 - .22 * (time - 9.) / .6
        elif 9.6 <= time < 10.:
            obstacle[0] = .06
        elif 10. <= time < 10.6:
            obstacle[0] = .06 + .22 * (time - 10.) / .6
    if ev == 'infeasible' and time >= 9.:
        mu = .07
        force[1] = 8. * p['pulse_scale']
    return dict(friction=float(mu), force=force, squeeze=torque,
                lift_delta=float(lift_delta), wrist=float(wrist), obstacle=obstacle)


def scenario_manifest():
    return [asdict(s) for s in SCENARIOS]

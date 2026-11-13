"""Development oracle for the controller/executor's actual coordinate domain.

The injected pressure observation exercises the public interface on a real
actuator mapping. It is not a physical grasp-performance or force-safety trial.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import numpy as np
from physics import Plant, SensorPipeline, MAX_DELTA
from controllers import METHODS, make_controller
from scenarios import SCENARIOS, parameters

ROOT = Path(__file__).resolve().parent


def verify():
    source = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
              for name in ('controllers.py', 'physics.py')}
    geometry = []
    for name in ('nominal', 'box_heavy', 'sphere'):
        scenario = next(s for s in SCENARIOS if s.name == name)
        plant = Plant(scenario, parameters(scenario, 410777, False))
        largest_clip_difference = 0.
        for alpha in (plant.closure_min, np.zeros(5), .5*(plant.closure_min+plant.closure_max), plant.closure_max):
            plant.alpha = alpha.copy()
            plant._set_actuators()
            for i, ids in enumerate(plant.actuators):
                expected = plant.open_pose[i]+alpha[i]*(plant.closed_pose[i]-plant.open_pose[i])
                largest_clip_difference = max(largest_clip_difference,
                                              float(np.max(np.abs(expected-plant.data.ctrl[ids]))))
        geometry.append(dict(scenario=name, closure_min=plant.closure_min.tolist(),
                             closure_max=plant.closure_max.tolist(),
                             largest_hidden_ctrl_clip=largest_clip_difference,
                             domain_maps_without_ghost_action=largest_clip_difference < 1e-9))
    actions = []
    scenario = next(s for s in SCENARIOS if s.name == 'nominal')
    for method in METHODS:
        p = parameters(scenario, 410777, False)
        plant, sensor = Plant(scenario, p), SensorPipeline(scenario, p, 410777)
        obs = sensor.sample(plant.truth(), plant.time)
        obs = replace(obs, normal=np.array([7., 0., 0., 0., 0.]), valid=np.ones(5, bool))
        controller = make_controller(method, scenario.limit, scenario.budget)
        for _ in range(80):
            controller.state.update(obs, .005)
        action = controller.step(obs, plant.phase, 'hold')
        before = plant.alpha.copy()
        plant.execute(action, obs, {}, 'hold')
        enacted = plant.alpha-before
        unloading_match = bool(action.delta_closure[0] < 0.
                               and np.allclose(enacted, action.delta_closure, atol=1e-10)
                               and np.max(np.abs(action.delta_closure)) <= MAX_DELTA+1e-10)
        plant.alpha[0] = max(plant.closure_min[0]+.1, -.4)
        plant._set_actuators()
        obs2 = sensor.sample(plant.truth(), plant.time+.005)
        release = controller.step(obs2, 'RELEASING', 'release')
        before = plant.alpha.copy()
        plant.execute(release, obs2, {}, 'release')
        release_match = bool(release.delta_closure[0] < 0.
                             and np.allclose(plant.alpha-before, release.delta_closure, atol=1e-10))
        actions.append(dict(method=method, thumb_unloading_at_pregrasp_zero=unloading_match,
                            negative_thumb_release_enacted=release_match,
                            requested=action.delta_closure.tolist(), enacted=enacted.tolist()))
    passed = all(r['domain_maps_without_ghost_action'] for r in geometry)
    passed &= all(r['thumb_unloading_at_pregrasp_zero'] and r['negative_thumb_release_enacted'] for r in actions)
    after = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source}
    return dict(passed=bool(passed), source_unchanged=source == after, source_sha256=source,
                scope='Actuator-coordinate domain/interface oracle, not physical task comparison',
                synthetic_pressure=True, confirmation_seeds_used=False, geometry=geometry, methods=actions)


if __name__ == '__main__':
    result = verify()
    (ROOT/'review'/'action_domain_verification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)

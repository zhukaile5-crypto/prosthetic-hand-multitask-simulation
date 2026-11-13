"""Development verification of delayed force-law -> closure P adaptation.

This uses the real common MuJoCo plant. Only the fixed force reference is
stepped; no coordinator, geometry planner, formal seed or future-event input is
used. Acceptance thresholds are specified here before executing this probe.
"""
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import numpy as np
from controllers import make_controller
from physics import Plant, SensorPipeline, DT, SUBSTEP_COLUMNS
from scenarios import SCENARIOS, parameters, schedule

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT/'review'/'baseline_tracking_calibration_v2'


def probe(shape_name, gain, condition):
    scenario = next(s for s in SCENARIOS if s.name == shape_name)
    p = parameters(scenario, 411701, False)
    p['force_noise'] = .16 if condition == 'worst_delay_noise' else 0.
    p['delay'] = .060 if condition == 'worst_delay_noise' else .010
    plant, sensor = Plant(scenario, p), SensorPipeline(scenario, p, 411701)
    controller = make_controller('fixed', scenario.limit, scenario.budget,
                                 dict(base=2., baseline_tracking_gain=gain))
    rows = []
    calibration_matches = True
    for _ in range(round(17./DT)):
        t = plant.time
        target = 3. if 7. <= t < 12. else 2.
        controller.g = replace(controller.g, base=target)
        truth = plant.truth()
        frame = sensor.sample(truth, t)
        action = controller.step(frame, plant.phase, 'hold')
        calibration_matches &= np.allclose(controller.state.force_gain, frame.force_gain)
        calibration_matches &= abs(controller.state.force_tau-frame.force_tau) < 1e-12
        env = schedule(scenario, p, t)
        rows.append(np.r_[t, truth['normal'], truth['relative'], truth['relative_velocity'],
                          plant.alpha, frame.normal, controller.desired, action.delta_closure])
        plant.execute(action, frame, env, 'hold')
        plant.advance(env)
    rows = np.asarray(rows)
    stats = []
    for lo, hi, reference in ((10.5, 11.5, 3.), (15.5, 16.5, 2.)):
        values = rows[(rows[:, 0] >= lo) & (rows[:, 0] <= hi), 1:6]
        mean, sd = values.mean(axis=0), values.std(axis=0)
        stats.append(dict(window=[lo, hi], reference=reference, mean=mean.tolist(),
                          sd=sd.tolist(), maximum_mean_error=float(np.max(abs(mean-reference))),
                          maximum_sd=float(np.max(sd))))
    substeps = np.asarray(plant.substep_records)
    peak = float(substeps[:, 1:6].max())
    final_contacts = int(np.count_nonzero(rows[-1, 1:6] > .2))
    finite = bool(np.isfinite(rows).all() and np.isfinite(substeps).all())
    passed = (finite and calibration_matches and final_contacts == 5
              and max(s['maximum_mean_error'] for s in stats) <= .15
              and max(s['maximum_sd'] for s in stats) <= .20 and peak <= 5.)
    key = f'{shape_name}__{gain:.2f}__{condition}'
    FOLDER.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FOLDER/(key+'.npz'), trace=rows, substeps=substeps,
                        substep_columns=np.array(SUBSTEP_COLUMNS))
    return dict(key=key, shape=shape_name, tracking_gain=gain, condition=condition,
                sensor_delay=p['delay'], sensor_noise=p['force_noise'],
                calibration_matches=bool(calibration_matches), windows=stats,
                physics_step_peak=peak, final_contacts=final_contacts,
                finite=finite, passed=bool(passed), final_phase=plant.phase)


if __name__ == '__main__':
    if (FOLDER/'summary.json').exists():
        raise RuntimeError('Existing verification output is immutable; use a fresh output directory.')
    FOLDER.mkdir(parents=True, exist_ok=True)
    before = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
              for n in ('controllers.py', 'physics.py', 'scenarios.py')}
    results = []
    jobs = [(shape, gain, condition) for shape in ('nominal', 'box_heavy', 'sphere')
            for gain in (.15, .20, .25) for condition in ('noise_free', 'worst_delay_noise')]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(probe, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(dict(key=result['key'], passed=result['passed'],
                                  peak=result['physics_step_peak'],
                                  final_contacts=result['final_contacts'])), flush=True)
    after = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in before}
    report = dict(scope='Real-MuJoCo baseline P-mapper development verification; no comparison or formal seeds',
                  predeclared_acceptance=dict(mean_error_max=.15, raw_force_sd_max=.20,
                                             actual_physics_peak_max=5., final_contacts_required=5),
                  source_sha256=before, source_unchanged=before == after,
                  cases=sorted(results, key=lambda r: r['key']),
                  passed=all(r['passed'] for r in results),
                  default_gain_passed=all(r['passed'] for r in results if r['tracking_gain'] == .20))
    (FOLDER/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], default_gain_passed=report['default_gain_passed'],
                         cases=len(results), source_unchanged=report['source_unchanged'])), flush=True)

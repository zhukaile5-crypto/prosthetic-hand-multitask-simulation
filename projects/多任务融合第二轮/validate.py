"""Physical, optimization, sensing, model-source and result-integrity checks."""
from pathlib import Path
from dataclasses import fields, asdict
from itertools import product
import hashlib
import json
import argparse
import numpy as np
import mujoco
from physics import ROOT, Plant, SensorPipeline, SensorFrame, DT
from scenarios import SCENARIOS, parameters
from controllers import METHODS, Gains, make_controller, box_budget_projection


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_violation_flags(times, unsafe):
    """Recompute qualifying contiguous runs without the runner's timer class."""
    indices = np.flatnonzero(unsafe)
    initial, hold = False, False
    if len(indices):
        for segment in np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1):
            start, end = times[segment[0]], times[segment[-1]]
            if end-start >= .030-1e-8:
                initial |= start < 8.
                hold |= end >= 8.
    return bool(initial), bool(hold)


def retained_hold_mask(times, shifts, heights, setup_ok, release_at):
    """Include the confirmation tick, then stop scoring a detached object."""
    retained = np.zeros(len(times), dtype=bool)
    since, dropped, drop_time = None, False, None
    for i, time in enumerate(times):
        if time < 8.:
            continue
        hold = release_at is None or time < release_at-1e-8
        if hold:
            retained[i] = setup_ok and not dropped
            lost = shifts[i] > .080 or heights[i] < heights[0]+.020
            if setup_ok and lost:
                if since is None:
                    since = time
                if time-since >= .050-1e-8:
                    dropped = True
                    if drop_time is None:
                        drop_time = time
            else:
                since = None
        elif since is not None and not dropped:
            # A loss already underway before release remains an unintended loss.
            dropped, drop_time = True, since
    return retained, bool(dropped), drop_time


def metric_regressions():
    from run_experiments import ForceViolationTracker
    cases = [
        (7.98,8.015,True,True), (7.99,8.01,False,False),
        (7.90,7.935,True,False), (8.10,8.135,False,True),
    ]
    for start, end, initial, hold in cases:
        times = np.arange(start,end+DT/2,DT)
        unsafe = np.ones(len(times),dtype=bool)
        expected = independent_violation_flags(times,unsafe)
        assert expected == (initial,hold), (times,expected)
        tracker = ForceViolationTracker()
        for time, value in zip(times,unsafe):
            tracker.update(float(time),bool(value))
        assert (tracker.initial,tracker.hold) == expected
    times = np.arange(7.98,8.025+DT/2,DT)
    unsafe = np.ones(len(times),dtype=bool)
    unsafe[np.argmin(abs(times-8.))] = False
    assert independent_violation_flags(times,unsafe) == (False,False)
    tracker = ForceViolationTracker()
    for time, value in zip(times,unsafe):
        tracker.update(float(time),bool(value))
    assert not tracker.initial and not tracker.hold
    times = np.arange(0.,17.,DT)
    shifts, heights = np.zeros(len(times)),np.full(len(times),.40)
    heights[times>=8.] = .50
    shifts[times>=10.] = .09
    retained,dropped,drop_time = retained_hold_mask(times,shifts,heights,True,None)
    assert dropped and abs(drop_time-10.05) < 1e-8
    assert retained[np.argmin(abs(times-10.05))]
    assert not retained[np.argmin(abs(times-10.055))]
    assert not retained_hold_mask(times,shifts,heights,False,None)[0].any()
    return dict(cross_boundary_force_timer_cases=5,
                orientation_confirmation_tick_and_failed_setup=True)


def reference_projection(d, upper, budget, weights, preserve):
    """Independent exhaustive active-set solution (3^5 possible bound patterns)."""
    target = min(budget,np.sum(upper),max(0.,np.sum(d)))
    best,score = None,float('inf')
    constraints = [True] if preserve else [False,True]
    for active_total in constraints:
        for status in product((0,1,2),repeat=5):
            status = np.array(status)
            free = status == 1
            x = np.where(status==2,upper,0.)
            if active_total:
                if not np.any(free):
                    if abs(np.sum(x)-target) > 1e-7:
                        continue
                else:
                    multiplier = (np.sum(d[free])+np.sum(x)-target)/np.sum(1/weights[free])
                    x[free] = d[free]-multiplier/weights[free]
            else:
                x[free] = d[free]
            if np.any(x < -1e-8) or np.any(x > upper+1e-8) or np.sum(x) > budget+1e-8:
                continue
            value = float(np.sum(weights*(x-d)**2))
            if value < score:
                best,score = x,value
    return best,score


def physical_checks():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001" gravity="0 0 -9.81"/>
      <worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".02" mass=".12"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model,data)
    analytic = 1.-.5*9.81*.2**2
    fall_error = abs(data.qpos[2]-analytic)
    assert fall_error < .0011,fall_error
    # A horizontal block should remain at subcritical load and slip above mu*m*g.
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001" gravity="0 0 -9.81"/>
      <default><geom friction=".5 .001 .00001" solref=".01 1"/></default>
      <worldbody><geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .021"><freejoint/><geom type="box" size=".04 .04 .02" mass="1"/></body></worldbody></mujoco>''')
    distances = []
    for load in (3.,7.):
        data = mujoco.MjData(model)
        for _ in range(600):
            mujoco.mj_step(model,data)
        x0 = data.qpos[0]
        data.xfrc_applied[1,0] = load
        for _ in range(400):
            mujoco.mj_step(model,data)
        distances.append(abs(data.qpos[0]-x0))
    assert distances[0] < .001 and distances[1] > .025,distances
    return dict(free_fall_error_m=fall_error,friction_subcritical_shift_m=distances[0],
                friction_supercritical_shift_m=distances[1])


def model_checks():
    scenario = SCENARIOS[0]
    plant = Plant(scenario,parameters(scenario,1001),.0005)
    model = plant.model
    joints = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(model.njnt)]
    actuators = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,j) for j in range(model.nu)]
    assert sum((n or '').startswith('rh_') for n in joints) == 24
    assert sum((n or '').startswith('rh_') for n in actuators) == 20
    assert model.neq == 0
    joint = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'object_free')
    assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
    manifest = json.loads((ROOT.parent/'五指假肢抓握仿真/upstream_shadow/download_manifest.json').read_text())
    for entry in manifest['files']:
        file = ROOT.parent/'五指假肢抓握仿真/upstream_shadow'/entry['path']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == entry['sha256'],file
    allowed = {'time','normal','shear','slip','collision','valid','age'}
    assert {f.name for f in fields(SensorFrame)} == allowed
    # Contact-material regression uses an artificial intersecting pose only;
    # no dynamics or benchmark trial is advanced from this pose.
    mu = .21
    plant.set_friction(mu)
    material_geoms = [*plant.geom_finger,*plant.palm_geoms,plant.object_geom]
    assert plant.palm_geoms
    assert np.allclose(model.geom_friction[material_geoms,0],mu,atol=1e-14)
    address = model.jnt_qposadr[joint]
    plant.data.qpos[address:address+3] = plant.data.geom_xpos[plant.palm_geoms[0]]
    mujoco.mj_forward(model,plant.data)
    palm_contacts = 0
    for contact in plant.data.contact[:plant.data.ncon]:
        if plant.object_geom not in (contact.geom1,contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == plant.object_geom else contact.geom1
        if other in plant.palm_geoms:
            assert np.allclose(contact.friction[:2],mu,atol=1e-14)
            palm_contacts += 1
    assert palm_contacts > 0, 'Palm material regression did not establish a contact'
    steps = DT/model.opt.timestep
    assert abs(steps-round(steps)) < 1e-9
    return dict(hand_joints=24,hand_actuators=20,free_object=True,welds=0,
                source_commit=manifest['commit'],asset_hashes_verified=True,
                sensor_fields=sorted(allowed),palm_friction_matches_object=mu,
                independently_checked_palm_contacts=palm_contacts,
                integral_physics_substeps=round(steps))


def control_checks():
    rng = np.random.default_rng(8121)
    worst = 0.
    for preserve in (False,True):
        for _ in range(25):
            d = rng.uniform(-1.,8.,5)
            upper = rng.uniform(.5,5.,5)
            budget = rng.uniform(2.,14.)
            weights = rng.uniform(.3,2.,5)
            x = box_budget_projection(d,upper,budget,weights,preserve)
            _,reference_value = reference_projection(d,upper,budget,weights,preserve)
            error = abs(float(np.sum(weights*(x-d)**2))-reference_value)
            worst = max(worst,error)
            assert error < 2e-5,error
            assert np.all(x >= -1e-8) and np.all(x <= upper+1e-8)
            assert np.sum(x) <= budget+2e-7
    frame = SensorFrame(9.,np.full(5,4.),np.full(5,2.),.1,0.,np.ones(5,dtype=bool),.01)
    for method in METHODS:
        controller = make_controller(method,(5,5,5,5,5),20.)
        output = controller.step(frame,'HOLDING','release')
        assert np.array_equal(output,np.zeros(5)),method
    scenario = next(s for s in SCENARIOS if s.name == 'sensor_dropout')
    p = parameters(scenario,1001)
    first,second = SensorPipeline(scenario,p,1001),SensorPipeline(scenario,p,1001)
    error_max = 0.
    for tick in range(3000):
        time = tick*DT
        truth = dict(normal=np.full(5,2.),shear=np.full(5,1.),slip=.01,collision=0.)
        shifted = dict(normal=np.full(5,3.),shear=np.full(5,1.),slip=.01,collision=0.)
        a,b = first.sample(truth,time),second.sample(shifted,time)
        assert np.array_equal(a.valid,b.valid)
        assert a.slip == b.slip and a.age == b.age
        if np.any(a.valid):
            error_max = max(error_max,float(np.nanmax(abs((b.normal-a.normal)-1.))))
    assert error_max < 1e-12
    # A force target around the former 1.45 N threshold must not impose an
    # unrelated posture jump. Reset the servo state before each request.
    scenario = SCENARIOS[0]
    plant = Plant(scenario,parameters(scenario,1001))
    plant.phase = 'HOLDING'
    plant.data.time = 9.
    plant._control_tick = round(9./DT)
    env = dict(lift_delta=0.,wrist=0.)
    obs = SensorFrame(9.,np.full(5,1.8),np.zeros(5),0.,0.,np.ones(5,dtype=bool),0.)
    values = []
    for requested in (1.4499,1.4501):
        plant.alpha[:] = .30
        plant.filtered_force[:] = 1.8
        plant.execute(np.full(5,requested),obs,np.full(5,5.),env)
        values.append(plant.alpha.copy())
        assert np.max(plant.alpha) < .31
    servo_delta = float(np.max(abs(values[1]-values[0])))
    assert servo_delta < 1e-5, servo_delta
    return dict(projection_cases=50,projection_objective_max_error=worst,
                release_gate_methods=len(METHODS),paired_sensor_noise_error=error_max,
                servo_threshold_max_state_delta=servo_delta)


def audit_results(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    results = json.loads((directory/'trials.json').read_text(encoding='utf-8'))
    assert manifest['metric_version'] >= 5, 'Earlier metric versions require their archived validator'
    assert len(results) == manifest['expected_trials'] == manifest['completed_trials']
    assert manifest['code_unchanged_during_run']
    keys = {(r['scenario'],r['method'],r['seed']) for r in results}
    assert len(keys) == len(results)
    expected_keys = set(product(manifest['scenarios'],manifest['methods'],manifest['seeds']))
    assert keys == expected_keys, dict(missing=sorted(expected_keys-keys),extra=sorted(keys-expected_keys))
    allowed_seeds = {'pilot':set(range(1101,1106)), 'development':set(range(1101,1106)),
                     'test':set(range(8001,8021)), 'robustness':set(range(9001,9011))}
    assert manifest['suite'] in allowed_seeds
    assert len(manifest['seeds']) == len(set(manifest['seeds']))
    assert set(manifest['seeds']) <= allowed_seeds[manifest['suite']], 'Seed split leakage'
    assert np.isclose(manifest['control_timestep'],DT,rtol=0.,atol=1e-14)
    steps = DT/manifest['physics_timestep']
    assert manifest['physics_timestep'] > 0 and abs(steps-round(steps)) < 1e-9 and round(steps) >= 1
    snapshot = directory/'snapshot'
    assert snapshot.is_dir(), 'Frozen source snapshot is missing'
    for name, expected in manifest['code_sha256'].items():
        assert digest(ROOT/name) == digest(snapshot/name) == expected, name
    assert manifest['final_code_sha256'] == manifest['code_sha256']
    assert digest(ROOT/'validate.py') == digest(snapshot/'validate.py'), 'Validator changed after freezing'
    expected_model = manifest['source_model_sha256']
    assert digest(ROOT.parent/'五指假肢抓握仿真/model.xml') == digest(snapshot/'source_model.xml') == expected_model
    source_manifest = json.loads((ROOT.parent/'五指假肢抓握仿真/upstream_shadow/download_manifest.json').read_text(encoding='utf-8'))
    assert source_manifest == manifest['source_manifest']
    if manifest.get('gains_sha256') is not None:
        frozen_gains = snapshot/'selected_gains.json'
        assert digest(frozen_gains) == manifest['gains_sha256']
        assert json.loads(frozen_gains.read_text(encoding='utf-8')) == manifest['gains']
    definitions = {s['name']:s for s in manifest['scenario_definitions']}
    checked = 0
    common = {}
    for result in results:
        key = result['scenario'],result['seed']
        scenario = definitions[result['scenario']]
        assert result['robustness'] == (manifest['suite']=='robustness')
        current_scenario = next(s for s in SCENARIOS if s.name==result['scenario'])
        assert asdict(current_scenario) == {**scenario,'limit':tuple(scenario['limit'])}
        assert result['parameters'] == parameters(current_scenario,result['seed'],result['robustness'])
        assert result['gains'] == asdict(Gains(**manifest['gains'].get(result['method'],{})))
        if key in common:
            assert result['parameters'] == common[key]
        common[key] = result['parameters']
        if 'trace_file' not in result:
            assert not manifest.get('save_traces',True), 'Expected trace is missing'
            continue
        trace_path = (directory/result['trace_file']).resolve()
        assert trace_path.is_relative_to(directory.resolve())
        with np.load(trace_path) as data:
            x = data['trace']
            cols = data['columns'].tolist()
            saved_result = json.loads(str(data['metadata']))
            assert 'substeps' in data and 'substep_columns' in data
            substeps=data['substeps']
            subcols=data['substep_columns'].tolist()
        for field in saved_result:
            assert saved_result[field] == result[field], (key,result['method'],field)
        get = lambda name:x[:,cols.index(name)]
        n = x[:,[cols.index(f'normal_{i}') for i in range(5)]]
        lim, budget = np.array(scenario['limit']),scenario['budget']
        times = get('time')
        ev = times >= 8.
        assert len(times) > 1 and np.all(np.diff(times)>0.)
        assert np.max(abs(np.diff(times)-DT)) < 1e-8
        assert abs(times[0]) < 1e-12
        if not result['nonfinite']:
            assert x.shape[0] == round(scenario['duration']/DT)
        assert np.all(np.isfinite(n))
        assert np.all(n >= 0.)
        field=lambda name: substeps[:,subcols.index(name)]
        assert np.all(np.isfinite(substeps))
        duration=field('duration_s')
        assert np.all(duration >= 0.)
        assert np.isclose(np.sum(duration),scenario['duration'],atol=1e-8)
        submetrics=dict(physics_peak_force_N=np.max(field('max_force_N')),
                        physics_peak_penetration_mm=np.max(field('penetration_m'))*1000,
                        physics_force_excess_Ns=np.dot(field('force_excess_N'),duration),
                        physics_budget_excess_Ns=np.dot(field('budget_excess_N'),duration),
                        control_peak_penetration_mm=np.max(get('penetration_m'))*1000)
        for name,expected in submetrics.items():
            assert np.isclose(result[name],expected,rtol=1e-9,atol=1e-7),(key,name,result[name],expected)
        strict=bool(result['completed'] and result['numerical_warnings']==0 and
                    np.max(field('force_excess_N'))<=1e-6 and np.max(field('budget_excess_N'))<=1e-6)
        assert result['peak_limit_completed'] == strict
        excess = np.sum(np.maximum(0.,n-lim),axis=1)
        budget_excess = np.maximum(0.,np.sum(n,axis=1)-budget)
        metrics = dict(force_excess_Ns=np.sum(excess[ev])*DT,
                       budget_excess_Ns=np.sum(budget_excess[ev])*DT,
                       initial_force_excess_Ns=np.sum(excess[~ev])*DT,
                       initial_budget_excess_Ns=np.sum(budget_excess[~ev])*DT,
                       total_force_excess_Ns=np.sum(excess)*DT,
                       total_budget_excess_Ns=np.sum(budget_excess)*DT,
                       over_limit_s=np.count_nonzero((excess[ev]>0.) | (budget_excess[ev]>0.))*DT,
                       peak_force_N=np.max(n[ev]),
                       peak_force_limit_ratio=np.max(n[ev]/lim),
                       initial_force_peak_N=np.max(n[~ev]),
                       mean_force_N=np.mean(np.sum(n[ev],axis=1)),
                       palm_peak_N=np.max(get('palm_force')),palm_force_Ns=np.sum(get('palm_force'))*DT,
                       collision_peak_N=np.max(get('collision_force')),collision_force_Ns=np.sum(get('collision_force'))*DT)
        for name, expected in metrics.items():
            assert np.isclose(result[name],expected,rtol=1e-9,atol=1e-7), (key,result['method'],name,result[name],expected)
        assert np.allclose(get('force_excess_N'),excess,rtol=0.,atol=1e-12)
        assert np.allclose(get('budget_excess_N'),budget_excess,rtol=0.,atol=1e-12)
        unsafe = np.any(n>lim+.25,axis=1) | (np.sum(n,axis=1)>budget+.5)
        initial, hold = independent_violation_flags(times,unsafe)
        assert result['initial_unsafe_proxy'] == initial
        assert result['unsafe_proxy'] == hold
        assert result['safe_completed'] == bool(result['completed'] and not initial and not hold and result['numerical_warnings']==0)
        assert result['hold_safe_completed'] == bool(result['completed'] and not hold)
        retained,dropped,drop_time = retained_hold_mask(times,get('position_error_m'),get('object_z'),result['setup_ok'],scenario['release_at'])
        assert result['dropped'] == dropped
        if drop_time is None:
            assert result['drop_time_s'] is None
        else:
            assert abs(result['drop_time_s']-drop_time) < 1e-8
        samples = int(np.count_nonzero(retained))
        assert result['orientation_samples'] == samples
        if samples:
            angles = get('orientation_error_deg')[retained]
            assert np.isclose(result['orientation_rms_deg'],np.sqrt(np.mean(angles**2)),rtol=1e-10,atol=1e-8)
            assert np.isclose(result['orientation_max_deg'],np.max(angles),rtol=1e-10,atol=1e-8)
            assert np.isclose(result['max_slip_mm'],np.max(get('position_error_m')[retained])*1000,rtol=1e-10,atol=1e-8)
        else:
            assert result['orientation_rms_deg'] is None and result['orientation_max_deg'] is None
            assert result['orientation_drift_max_deg'] is None
            if not result['setup_ok']:
                assert result['max_slip_mm'] is None
        release_at = scenario['release_at']
        if release_at is not None:
            desired = x[:,[cols.index(f'desired_{i}') for i in range(5)]]
            releasing = times>=release_at-1e-8
            assert np.max(abs(desired[releasing])) == 0.
            clear = releasing & (np.max(n,axis=1)<.05) & (get('palm_force')<.05)
            indices = np.flatnonzero(clear)
            expected_latency = None
            if len(indices):
                for segment in np.split(indices,np.flatnonzero(np.diff(indices)!=1)+1):
                    first = times[segment[0]]
                    qualifying = segment[times[segment]-first>=.080-1e-8]
                    if len(qualifying):
                        expected_latency = float(times[qualifying[0]]-release_at)
                        break
            if expected_latency is None:
                assert result['release_latency_s'] is None
            else:
                assert abs(result['release_latency_s']-expected_latency) < 1e-8
        expected_completed = bool(result['setup_ok'] and not dropped and not result['nonfinite'])
        if release_at is not None:
            expected_completed &= result['release_latency_s'] is not None and result['release_latency_s']<=3.5
        assert result['completed'] == expected_completed
        checked += 1
    return dict(trials=len(results),paired_parameters_verified=True,
                exact_cartesian_pairing_verified=True,seed_split_verified=manifest['suite'],
                frozen_code_and_source_verified=True,
                independently_recomputed_trace_integrals=checked,
                uninterrupted_force_safety_verified=True,retained_orientation_denominator_verified=True,
                palm_and_collision_metrics_verified=True,release_outputs_verified=True,
                sphere_orientation='Raw SO(3) diagnostic only; exclude from orientation advantage summaries',
                validator_sha256=digest(ROOT/'validate.py'))


def repeatability_check(directory):
    """One replay against a saved trial; wall-clock timings are nondeterministic."""
    from run_experiments import run_episode
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    results = json.loads((directory/'trials.json').read_text(encoding='utf-8'))
    saved = next((r for r in results if r['setup_ok'] and not r['boundary']),results[0])
    replay = run_episode(dict(scenario=saved['scenario'],method=saved['method'],seed=saved['seed'],
                              robustness=saved['robustness'],gains=saved['gains'],save_trace=False,
                              timestep=manifest['physics_timestep']))
    ignored = {'wall_seconds','controller_p95_ms','controller_mean_ms','trace_file'}
    assert set(saved)-ignored == set(replay)-ignored
    for name in set(saved)-ignored:
        # Dynamics and stochastic observations are deterministic on this host.
        assert saved[name] == replay[name], (name,saved[name],replay[name])
    return dict(scenario=saved['scenario'],method=saved['method'],seed=saved['seed'],
                identical_physical_and_control_results=True,
                excluded_fields=sorted(ignored))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'output/validation.json')
    args = parser.parse_args()
    report = dict(physical=physical_checks(),model=model_checks(),control=control_checks(),
                  metric_regressions=metric_regressions())
    if args.results:
        report['results'] = audit_results(args.results)
        report['repeatability'] = repeatability_check(args.results)
    report['passed'] = True
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

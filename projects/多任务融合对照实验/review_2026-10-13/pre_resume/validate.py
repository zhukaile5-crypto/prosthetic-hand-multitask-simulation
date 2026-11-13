"""Physical, optimization, sensing, model-source and result-integrity checks."""
from pathlib import Path
from dataclasses import fields
from itertools import product
import hashlib
import json
import argparse
import numpy as np
import mujoco
from physics import ROOT, Plant, SensorPipeline, SensorFrame, DT
from scenarios import SCENARIOS, parameters
from controllers import METHODS, make_controller, box_budget_projection


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
    plant = Plant(scenario,parameters(scenario,1001))
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
    return dict(hand_joints=24,hand_actuators=20,free_object=True,welds=0,
                source_commit=manifest['commit'],asset_hashes_verified=True,
                sensor_fields=sorted(allowed))


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
    return dict(projection_cases=50,projection_objective_max_error=worst,
                release_gate_methods=len(METHODS),paired_sensor_noise_error=error_max)


def audit_results(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    results = json.loads((directory/'trials.json').read_text(encoding='utf-8'))
    assert len(results) == manifest['expected_trials'] == manifest['completed_trials']
    assert manifest['code_unchanged_during_run']
    keys = {(r['scenario'],r['method'],r['seed']) for r in results}
    assert len(keys) == len(results)
    checked = 0
    common = {}
    for result in results:
        key = result['scenario'],result['seed']
        if key in common:
            assert result['parameters'] == common[key]
        common[key] = result['parameters']
        if 'trace_file' not in result:
            continue
        data = np.load(directory/result['trace_file'])
        x = data['trace']
        cols = data['columns'].tolist()
        get = lambda name:x[:,cols.index(name)]
        n = x[:,[cols.index(f'normal_{i}') for i in range(5)]]
        lim = np.array(next(s for s in SCENARIOS if s.name == result['scenario']).limit)
        ev = get('time') >= 8.
        excess = float(np.sum(np.maximum(0.,n[ev]-lim))*DT)
        assert abs(excess-result['force_excess_Ns']) < 1e-7,(result['scenario'],result['method'])
        total = float(np.sum(np.maximum(0.,n-lim))*DT)
        assert abs(total-result['total_force_excess_Ns']) < 1e-7
        assert x.shape[0] == round(17./DT)
        assert np.all(np.isfinite(n))
        if result['scenario'] == 'release_slip':
            desired = x[:,[cols.index(f'desired_{i}') for i in range(5)]]
            assert np.max(abs(desired[get('time')>=12.-1e-8])) == 0.
        checked += 1
    return dict(trials=len(results),paired_parameters_verified=True,
                independently_recomputed_trace_integrals=checked,release_outputs_verified=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'output/validation.json')
    args = parser.parse_args()
    report = dict(physical=physical_checks(),model=model_checks(),control=control_checks())
    if args.results:
        report['results'] = audit_results(args.results)
    report['passed'] = True
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

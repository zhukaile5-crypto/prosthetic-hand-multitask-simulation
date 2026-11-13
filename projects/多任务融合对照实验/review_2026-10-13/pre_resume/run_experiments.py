"""Paired simulation runner. Development and frozen evaluation use disjoint seeds."""
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import hashlib
import json
import os
import sys
import time as timer
import platform
import subprocess
import mujoco
import numpy as np
from physics import Plant, SensorPipeline, DT, ROOT
from scenarios import SCENARIOS, DEV_NAMES, parameters, schedule, scenario_manifest
from controllers import METHODS, Gains, make_controller

TRACE_COLUMNS = ['time']+[f'normal_{i}' for i in range(5)]+[f'shear_{i}' for i in range(5)]+[
    f'observed_normal_{i}' for i in range(5)]+[f'desired_{i}' for i in range(5)]+[
    'slip_true','slip_observed','object_x','object_y','object_z','relative_x','relative_y','relative_z',
    'position_error_m','orientation_error_deg','phase','collision_force','palm_force',
    'support_force','valid_sensors','sensor_age','force_excess_N','budget_excess_N']
PHASE_CODES = {'CLOSING':0,'LIFTING':1,'HOLDING':2,'RELEASING':3,'FAILED':4,'DROPPED':5}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    names = ('controllers.py','physics.py','scenarios.py','run_experiments.py')
    return {name:sha256(ROOT/name) for name in names}


def run_episode(job):
    scenario = next(s for s in SCENARIOS if s.name == job['scenario'])
    seed,method = job['seed'],job['method']
    p = parameters(scenario,seed,job.get('robustness',False))
    plant = Plant(scenario,p,job.get('timestep',.001))
    pipeline = SensorPipeline(scenario,p,seed)
    gains = Gains(**job.get('gains',{}))
    controller = make_controller(method,scenario.limit,scenario.budget,gains)
    limits = np.array(scenario.limit)
    release_at = scenario.release_at
    start = timer.perf_counter()
    traces,states,state_times,controller_times = [],[],[],[]
    excessive_seconds = 0.
    excessive_integral = 0.
    budget_integral = 0.
    initial_excess_integral = 0.
    initial_budget_integral = 0.
    force_sum_integral = 0.
    peak = 0.
    peak_ratio = 0.
    unsafe_since = None
    unsafe_proxy = False
    initial_unsafe_proxy = False
    initial_unsafe_since = None
    drop_since = None
    dropped = False
    first_drop = None
    release_done = None
    release_clear_since = None
    setup_ok = False
    initial_force_peak = 0.
    relative_anchor = None
    rotation_anchor = None
    max_shift = 0.
    unrestricted_shift = 0.
    max_angle = 0.
    max_drift_angle = 0.
    orientation_squared = 0.
    reference_rotation = plant.truth()['rotation'].copy()
    evaluated = 0
    hold_evaluated = 0
    nonfinite = False
    n_ticks = round(scenario.duration/DT)
    for tick in range(n_ticks):
        t = plant.time
        true = plant.truth()
        obs = pipeline.sample(true,t)
        env = schedule(scenario,p,t)
        command = 'release' if release_at is not None and t >= release_at-1e-8 else 'hold'
        if command == 'release':
            plant.request_release()
        begin = timer.perf_counter_ns()
        desired = controller.step(obs,plant.phase,command)
        controller_times.append((timer.perf_counter_ns()-begin)/1e6)
        plant.execute(desired,obs,limits,env)
        n = true['normal']
        fn = float(np.max(n))
        excess = float(np.sum(np.maximum(0.,n-limits)))
        budget_excess = max(0.,float(np.sum(n))-scenario.budget)
        shift,angle = 0.,0.
        if t < 8.:
            initial_force_peak = max(initial_force_peak,fn)
            initial_excess_integral += excess*DT
            initial_budget_integral += budget_excess*DT
            initial_unsafe = np.any(n > limits+.25) or np.sum(n) > scenario.budget+.5
            if initial_unsafe:
                if initial_unsafe_since is None:
                    initial_unsafe_since = t
                if t-initial_unsafe_since >= .030-1e-8:
                    initial_unsafe_proxy = True
            else:
                initial_unsafe_since = None
        else:
            if relative_anchor is None:
                relative_anchor = true['relative'].copy()
                rotation_anchor = true['rotation'].copy()
                setup_ok = bool(plant.phase == 'HOLDING' and
                                true['position'][2]-plant.start_position[2] > .04 and
                                np.count_nonzero(n > .2) >= 2 and not plant.failed_setup)
            evaluated += 1
            peak = max(peak,fn)
            peak_ratio = max(peak_ratio,float(np.max(n/limits)))
            excessive_seconds += DT*(excess > 0. or budget_excess > 0.)
            excessive_integral += excess*DT
            budget_integral += budget_excess*DT
            force_sum_integral += float(np.sum(n))*DT
            unsafe = np.any(n > limits+.25) or np.sum(n) > scenario.budget+.5
            if unsafe:
                if unsafe_since is None:
                    unsafe_since = t
                if t-unsafe_since >= .030-1e-8:
                    unsafe_proxy = True
            else:
                unsafe_since = None
            if command == 'hold':
                hold_evaluated += 1
                shift = float(np.linalg.norm(true['relative']-relative_anchor))
                rotation_delta = reference_rotation.T @ true['rotation']
                angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation_delta)-1)/2,-1,1))))
                drift_delta = rotation_anchor.T @ true['rotation']
                drift_angle = float(np.degrees(np.arccos(np.clip((np.trace(drift_delta)-1)/2,-1,1))))
                unrestricted_shift = max(unrestricted_shift,shift)
                if not dropped and setup_ok:
                    max_shift = max(max_shift,shift)
                    max_angle = max(max_angle,angle)
                    max_drift_angle = max(max_drift_angle,drift_angle)
                    orientation_squared += angle*angle
                lost = shift > .080 or true['position'][2] < plant.start_position[2]+.020
                if lost and setup_ok:
                    if drop_since is None:
                        drop_since = t
                    if t-drop_since >= .050-1e-8:
                        dropped = True
                        if first_drop is None:
                            first_drop = t
                else:
                    drop_since = None
            else:
                if drop_since is not None and not dropped:
                    dropped = True
                    first_drop = drop_since
                clear = fn < .05 and true['palm'] < .05
                if clear:
                    if release_clear_since is None:
                        release_clear_since = t
                    if t-release_clear_since >= .080-1e-8 and release_done is None:
                        release_done = t-release_at
                else:
                    release_clear_since = None
        # Metrics truth never modifies the controller, actuator commands, or task state.
        if job.get('save_trace',False):
            traces.append([t,*n,*true['shear'],*obs.normal,*desired,true['slip'],obs.slip,
                           *true['position'],*true['relative'],shift,angle,PHASE_CODES[plant.phase],
                           true['collision'],true['palm'],true['support'],np.sum(obs.valid),obs.age,
                           excess,budget_excess])
            if tick % 10 == 0:
                states.append(plant.data.qpos.copy())
                state_times.append(t)
        plant.advance(env)
        if not np.all(np.isfinite(plant.data.qpos)) or not np.all(np.isfinite(plant.data.qvel)):
            nonfinite = True
            break
    numerical_warnings = sum(int(w.number) for w in plant.data.warning)
    completed = setup_ok and not dropped and not nonfinite
    if release_at is not None:
        completed = completed and release_done is not None and release_done <= 3.5
    safe_completed = completed and not unsafe_proxy and not initial_unsafe_proxy and numerical_warnings == 0
    result = dict(scenario=scenario.name,method=method,seed=seed,robustness=job.get('robustness',False),
                  parameters=p,gains=asdict(gains),setup_ok=setup_ok,dropped=dropped,
                  completed=bool(completed),safe_completed=bool(safe_completed),unsafe_proxy=unsafe_proxy,
                  initial_unsafe_proxy=initial_unsafe_proxy,hold_safe_completed=bool(completed and not unsafe_proxy),
                  max_slip_mm=max_shift*1000 if setup_ok else None,
                  unrestricted_shift_mm=unrestricted_shift*1000,orientation_max_deg=max_angle,
                  orientation_drift_max_deg=max_drift_angle,
                  orientation_rms_deg=np.sqrt(orientation_squared/max(1,hold_evaluated)),
                  peak_force_N=peak,peak_force_limit_ratio=peak_ratio,
                  mean_force_N=force_sum_integral/max(DT,evaluated*DT),
                  force_excess_Ns=excessive_integral,budget_excess_Ns=budget_integral,
                  initial_force_excess_Ns=initial_excess_integral,initial_budget_excess_Ns=initial_budget_integral,
                  total_force_excess_Ns=initial_excess_integral+excessive_integral,
                  total_budget_excess_Ns=initial_budget_integral+budget_integral,
                  over_limit_s=excessive_seconds,release_latency_s=release_done,
                  grasp_time_s=plant.first_grasp_time,drop_time_s=first_drop,
                  switches=controller.switches,slip_events=controller.slip_events,
                  controller_p95_ms=float(np.percentile(controller_times,95)),
                  controller_mean_ms=float(np.mean(controller_times)),
                  initial_force_peak_N=initial_force_peak,numerical_warnings=numerical_warnings,
                  nonfinite=nonfinite,boundary=scenario.boundary,
                  wall_seconds=timer.perf_counter()-start,final_position=plant.truth()['position'].tolist(),
                  final_force=plant.truth()['normal'].tolist(),final_debug=controller.debug)
    if job.get('save_trace',False):
        path = Path(job['output'])/'traces'/f'{scenario.name}__{method}__{seed}.npz'
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,trace=np.asarray(traces),columns=np.array(TRACE_COLUMNS),
                            qpos=np.array(states),qpos_time=np.array(state_times),
                            metadata=json.dumps(result,ensure_ascii=False))
        result['trace_file'] = str(path.relative_to(Path(job['output'])))
    return result


def write_results(out, results):
    (out/'trials.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    if not results:
        return
    keys = [k for k,v in results[0].items() if not isinstance(v,(dict,list))]
    with (out/'trials.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite',choices=['pilot','development','test','robustness'],default='pilot')
    parser.add_argument('--scenarios',nargs='+')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--out',default='pilot')
    parser.add_argument('--gains',type=Path)
    parser.add_argument('--no-traces',action='store_true')
    parser.add_argument('--timestep',type=float,default=.001)
    args = parser.parse_args()
    out = ROOT/'output'/args.out
    out.mkdir(parents=True,exist_ok=True)
    if (out/'trials.json').exists():
        raise RuntimeError('Use a new output directory to preserve completed results.')
    seeds = args.seeds or ([1001] if args.suite=='pilot' else list(range(1001,1006)) if args.suite=='development'
                           else list(range(3001,3021)) if args.suite=='test' else list(range(6501,6511)))
    names = args.scenarios or (list(DEV_NAMES) if args.suite=='development' else [s.name for s in SCENARIOS])
    unknown = set(names)-{s.name for s in SCENARIOS}
    if unknown:
        raise ValueError(unknown)
    gains = json.loads(args.gains.read_text(encoding='utf-8')) if args.gains else {}
    jobs = [dict(scenario=name,method=method,seed=seed,robustness=args.suite=='robustness',
                 gains=gains.get(method,{}),save_trace=not args.no_traces,
                 output=str(out),timestep=args.timestep)
            for name in names for seed in seeds for method in args.methods]
    manifest = dict(created=time_stamp(),suite=args.suite,seeds=seeds,methods=args.methods,
                    scenarios=names,scenario_definitions=scenario_manifest(),code_sha256=code_hashes(),
                    gains=gains,python=sys.version,mujoco=mujoco.__version__,numpy=np.__version__,
                    platform=platform.platform(),physics_timestep=args.timestep,control_timestep=DT,
                    source_model_sha256=sha256(ROOT.parent/'五指假肢抓握仿真/model.xml'),
                    source_manifest=json.loads((ROOT.parent/'五指假肢抓握仿真/upstream_shadow/download_manifest.json').read_text()),
                    expected_trials=len(jobs),workers=args.workers,metric_version=4,
                    metric_definitions=dict(safe_completed='completed and no monitored five-finger force/budget violation lasting 30ms; includes acquisition',
                        unsafe_force='any normal_i > limit_i + 0.25N OR five-finger sum > budget + 0.5N',
                        max_slip_mm='maximum palm-frame object displacement from t=8s until confirmed loss; missing if setup failed',
                        orientation='relative body rotation error from initial commanded orientation, with transport rotation removed',
                        release='all hand-object finger and palm normal forces below 0.05N continuously for 80ms; supported placement is a separate question'))
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    results = []
    start = timer.perf_counter()
    if args.workers == 1:
        for job in jobs:
            result = run_episode(job)
            results.append(result)
            print_progress(result,len(results),len(jobs))
            write_results(out,results)
    else:
        # Regular files avoid Windows named-pipe restrictions in managed workspaces.
        results = file_worker_pool(jobs,args.workers,out)
    results.sort(key=lambda x:(names.index(x['scenario']),x['seed'],args.methods.index(x['method'])))
    write_results(out,results)
    manifest['completed_trials'] = len(results)
    manifest['wall_seconds'] = timer.perf_counter()-start
    manifest['finished'] = time_stamp()
    manifest['final_code_sha256'] = code_hashes()
    manifest['code_unchanged_during_run'] = manifest['code_sha256'] == manifest['final_code_sha256']
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',len(results),'wall_seconds',round(manifest['wall_seconds'],2),flush=True)


def time_stamp():
    return timer.strftime('%Y-%m-%dT%H:%M:%SZ',timer.gmtime())


def print_progress(result,index,total):
    slip = f"{result['max_slip_mm']:.1f}" if result['max_slip_mm'] is not None else 'NA'
    print(f"{index}/{total} {result['scenario']} {result['method']} seed={result['seed']} "
          f"setup={result['setup_ok']} done={result['completed']} safe={result['safe_completed']} "
          f"slip={slip}mm excess={result['force_excess_Ns']:.3f}Ns "
          f"wall={result['wall_seconds']:.2f}s",flush=True)


def file_worker_pool(jobs, workers, out):
    work = out/'work'
    work.mkdir(parents=True,exist_ok=True)
    processes,streams,paths = [],[],[]
    try:
        for i in range(min(workers,len(jobs))):
            job_path = work/f'worker_{i}_jobs.json'
            result_path = work/f'worker_{i}_results.json'
            job_path.write_text(json.dumps(dict(jobs=jobs[i::workers],result=str(result_path))),encoding='utf-8')
            stream = (work/f'worker_{i}.log').open('w',encoding='utf-8')
            streams.append(stream)
            processes.append(subprocess.Popen([sys.executable,'-u',str(ROOT/'run_experiments.py'),
                                              '--worker-file',str(job_path)],cwd=ROOT,stdout=stream,
                                              stderr=subprocess.STDOUT,
                                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0))
            paths.append(result_path)
        (work/'worker_pids.json').write_text(json.dumps([p.pid for p in processes]),encoding='utf-8')
        delivered = [0]*len(processes)
        results = []
        while True:
            for i,(process,path) in enumerate(zip(processes,paths)):
                if path.exists():
                    try:
                        items = json.loads(path.read_text(encoding='utf-8'))
                    except (json.JSONDecodeError,PermissionError):
                        items = None
                    if items is not None:
                        for item in items[delivered[i]:]:
                            results.append(item)
                            print_progress(item,len(results),len(jobs))
                        delivered[i] = len(items)
                if process.poll() not in (None,0):
                    raise RuntimeError(f'Worker {i} failed; inspect {work/f"worker_{i}.log"}')
            if all(p.poll() is not None for p in processes):
                break
            timer.sleep(.5)
        # Final read closes the small race between the last write and process termination.
        for i,path in enumerate(paths):
            items = json.loads(path.read_text(encoding='utf-8'))
            results.extend(items[delivered[i]:])
        return results
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=10)
        for stream in streams:
            stream.close()


def file_worker(path):
    spec = json.loads(path.read_text(encoding='utf-8'))
    result_path = Path(spec['result'])
    results = []
    for job in spec['jobs']:
        result = run_episode(job)
        results.append(result)
        temp = result_path.with_suffix('.tmp')
        temp.write_text(json.dumps(results,ensure_ascii=False),encoding='utf-8')
        for attempt in range(30):
            try:
                os.replace(temp,result_path)
                break
            except PermissionError:
                if attempt == 29:
                    raise
                timer.sleep(.05)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker-file':
        file_worker(Path(sys.argv[2]))
    else:
        main()

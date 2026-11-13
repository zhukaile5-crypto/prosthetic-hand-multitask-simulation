from pathlib import Path
import sys,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_experiments import run_episode
from physics import Plant
from scenarios import SCENARIOS,parameters

if __name__=='__main__':
    selected=json.loads((ROOT/'selected_gains.json').read_text(encoding='utf-8'))
    out=ROOT/'review'/'compiled_model_pilot'
    cases=[(s,seed,dt) for s,seed in [('nominal',3001),('friction_drop',3007),('fragile_conflict',3001)] for dt in [.0005,.00025]]
    cases += [(s,1101,.0005) for s in ['cylinder_small','box_heavy','sphere','collision','release_slip']]
    jobs=[dict(scenario=s,seed=seed,timestep=dt,method=m,gains=selected[m],save_trace=False,output=str(out)) for s,seed,dt in cases for m in ['ratio','fusion']]
    rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(run_episode,j):j for j in jobs}
        for future in as_completed(futures):
            row=future.result();row['physics_timestep']=futures[future]['timestep'];rows.append(row)
            print(json.dumps({k:row[k] for k in ['scenario','method','seed','physics_timestep','setup_ok','completed','safe_completed','physics_peak_force_N','physics_peak_penetration_mm','physics_force_excess_Ns','max_slip_mm','release_latency_s']}),flush=True)
    initial=[dict(scenario=s.name,seed=seed,penetration=Plant(s,parameters(s,seed)).truth()['penetration']) for s in SCENARIOS for seed in [1101,1102,1103,1104,1105]]
    (ROOT/'review'/('compiled_physics_validation'+('_identified' if len(sys.argv)>1 else '')+'.json')).write_text(json.dumps(dict(results=rows,initial_geometry=initial,wall_seconds=time.perf_counter()-start),indent=2),encoding='utf-8')

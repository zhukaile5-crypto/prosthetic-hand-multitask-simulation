from pathlib import Path
import sys,json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import physics
from run_experiments import run_episode
from scenarios import SCENARIOS,parameters

def run(job):
    physics.OPEN[0][0]=job.pop('thumb_start')
    row=run_episode(job)
    row['thumb_start']=float(physics.OPEN[0][0])
    row['physics_timestep']=job['timestep']
    return row

if __name__=='__main__':
    selected=json.loads((ROOT/'selected_gains.json').read_text(encoding='utf-8'))
    jobs=[dict(scenario=s,seed=1101,timestep=.001,thumb_start=th,method=m,gains=selected[m],save_trace=False,output=str(ROOT/'review'/'shape_pilot')) for th in [0.,-.2] for s in ['box_heavy','sphere'] for m in ['ratio','fusion']]
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            row=f.result();rows.append(row)
            print(json.dumps({k:row[k] for k in ['scenario','method','thumb_start','setup_ok','completed','safe_completed','physics_peak_force_N','physics_peak_penetration_mm','max_slip_mm']}),flush=True)
    (ROOT/'review'/'shape_acquisition_validation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

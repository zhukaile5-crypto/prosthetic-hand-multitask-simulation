"""Read every declared V15 pilot cell once; no simulation or success regrading."""
from pathlib import Path
import json,sys,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CELLS=[('旧策略+被动观测','v15_passive_blind_bridge_v1'),('单一持握任务','v15_holding_task_ablation_v1'),('共享释放任务图','v15_shared_release_ablation_v1'),('完整组合','v15_combined_pilot_v1')]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=ROOT/args.out
 if out.exists():raise FileExistsError(out)
 records=[];missing=[]
 for label,folder in CELLS:
  path=ROOT/'output'/folder
  if not (path/'execution_complete.json').exists():missing.append(dict(cell=label,reason='No terminal evidence'));continue
  terminal=read(path/'execution_complete.json')
  if not terminal['execution_complete']:missing.append(dict(cell=label,reason='Execution incomplete',terminal=terminal));continue
  for x in read(path/'trials.json'):
   run=path/'runs'/f"{x['scenario']}__{x['method']}__{x['seed']}"
   meta=read(run/'metadata.json');metrics=x['research_metrics'];s=metrics['static_quality'];dy=metrics['dynamic_quality'];h=metrics['entire_holding_quality'];safe=metrics['safety']
   with np.load(run/'trace.npz',allow_pickle=False) as a:
    times=a['time'];dbg=a['debug_json'];control=a['control_ms'];full=a['full_control_loop_ms'];graphs=a['receiver_release_tasks_json'];positions=a['position']
   decisions=[i for i,(t,j) in enumerate(zip(times,dbg)) if abs(read_debug(j).get('decision_time_s',-1.)-t)<1e-9]
   graph_history=[];last=None
   for t,g in zip(times,graphs):
    value=read_debug(g)
    if value is not None and value.get('phase')!=last:last=value['phase'];graph_history.append(dict(time_s=float(t),phase=last,version=value.get('version')))
   delivery=read(run/'delivery_summary.json')
   records.append(dict(cell=label,output=folder,method=x['method'],scenario=x['scenario'],seed=x['seed'],task=x['task_success'],full_motion=metrics['full_motion_mission_success'],force_guard=safe['guard_pass'],static_mean_N=s['mean_actual_finger_force_N'],static_TV_N_s=s['raw_200Hz']['finger_TV_rate_N_s'],static_SD_N=s['raw_200Hz']['total_force_SD_N'],static_CV=s['raw_200Hz'].get('total_force_CV'),dynamic_TV_N_s=dy['raw_200Hz']['finger_TV_rate_N_s'],whole_hold_TV_N_s=h['raw_200Hz']['finger_TV_rate_N_s'],peak_N=max(safe['finger_peak_N']),local_dose_Ns=safe['local_excess_integral_Ns'],release_latency_s=metrics['release']['physical_clearance_latency_s'],decisions=len(decisions),control_median_ms=float(np.median(control[decisions])),full_control_median_ms=float(np.median(full[decisions])),full_control_max_ms=float(np.max(full[decisions])),graph_history=graph_history,object_end_position_m=positions[-1].tolist(),delivery=delivery,source_count=len(meta['research_source_sha256'])))
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(scope='One seen nominal/hold physical seed; four predeclared software feature cells, three force intent methods each; no new success/aggregate score',records=records,missing=missing),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 for x in records:print(json.dumps({k:v for k,v in x.items() if k not in ('delivery','source_count')},ensure_ascii=False))
 return 0
def read_debug(value):return json.loads(str(value))
if __name__=='__main__':raise SystemExit(main())

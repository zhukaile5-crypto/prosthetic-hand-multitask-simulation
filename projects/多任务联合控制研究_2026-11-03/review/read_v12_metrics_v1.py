"""Read unchanged evaluator vectors and fixed-window raw integration diagnostics."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('directories',nargs='+');args=parser.parse_args()
    out=Path(args.out);assert not out.exists()
    records=[]
    for directory in args.directories:
        for path in sorted(Path(directory).glob('runs/*/result.json')):
            result=json.loads(path.read_text(encoding='utf-8'));r=result['research_metrics']
            trace=path.parent/'trace.npz';data=np.load(trace,allow_pickle=False)
            metadata=json.loads((path.parent/'metadata.json').read_text(encoding='utf-8'))
            n=data['substep_normal'];t=data['substep_time']+data['substep_duration']
            raw={}
            for name,start,end in (('static',8.2,8.9),('holding',7.5,12.0)):
                m=(t>=start-1e-9)&(t<=end+1e-9);a=n[m];ts=t[m]
                duration=float(ts[-1]-ts[0]);variation=float(np.abs(np.diff(a,axis=0)).sum())
                raw[name]=dict(window_s=[start,end],samples=int(m.sum()),duration_s=duration,
                    finger_TV_N=variation,finger_TV_rate_N_s=variation/duration,
                    total_force_SD_N=float(a.sum(axis=1).std()),
                    scope='raw integration solver force at segment-end labels; not continuous-time or geometry certificate')
            stat=r['static_quality'];holding=r['entire_holding_quality']
            records.append(dict(method=result['method'],scenario=result['scenario'],seed=result['seed'],
                motion_mode=result['motion_mode'],trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),record_complete=r['record_complete'],
                preparation_passed=r['preparation_passed'],legacy_model_domain_supported=metadata.get('legacy_model_domain_supported'),
                mission_success=r['mission_success'],full_motion_mission_success=r['full_motion_mission_success'],
                force_guard=r['safety']['guard_pass'],global_max_finger_force_N=max(r['safety']['finger_peak_N']),
                local_excess_integral_Ns=r['safety']['local_excess_integral_Ns'],safety_by_stage=r['safety']['stages'],
                static_eligible=stat['eligible'],static_force_N=stat['mean_actual_finger_force_N'],
                static_raw200Hz=stat.get('raw_200Hz'),static_bandlimited=stat.get('bandlimited'),
                entire_holding_eligible=holding['eligible'],entire_holding_force_N=holding['mean_actual_finger_force_N'],
                entire_holding_raw200Hz=holding.get('raw_200Hz'),initial_stability_wait_s=r['initial_stability_wait_s'],
                holding_motion_tracking=r['progress']['holding_motion_tracking'],release=r['release'],
                extra_raw_integration_diagnostics=raw))
    keys=[(r['method'],r['scenario'],r['seed'],r['motion_mode']) for r in records]
    assert len(keys)==len(set(keys)),'Duplicate contexts cannot be pooled as independent records'
    report=dict(schema='v12_explicit_evaluation_vectors_v1',records=records,count=len(records),
        no_threshold_changes=True,no_missing_quality_filled_zero=True,utility_weights=None,
        raw_time_semantics='substep START + duration = END; 200Hz extra force recomputation is not simple decimation',
        superiority_proven=False,statistical_scope='development; reused physical clusters, no independence from time samples')
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps([dict(method=r['method'],scenario=r['scenario'],seed=r['seed'],mission=r['mission_success'],
        static_N=r['static_force_N'],static_rawTV=r['static_raw200Hz']['finger_TV_rate_N_s'] if r['static_raw200Hz'] else None,
        raw2000TV=r['extra_raw_integration_diagnostics']['static']['finger_TV_rate_N_s'],
        full_guard=r['force_guard'],Ns=r['local_excess_integral_Ns']) for r in records],ensure_ascii=False))


if __name__=='__main__':main()

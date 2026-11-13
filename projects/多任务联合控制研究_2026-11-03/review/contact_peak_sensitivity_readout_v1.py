"""Readonly numerical-sensitivity readout; never runs a controller or simulator."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
METHODS=('ratio_effort','fixed_effort','fusion_effort')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--refined-dir',required=True);args=parser.parse_args()
    target=ROOT/args.out;assert not target.exists()
    batches=[ROOT/'output/contact_peak_convergence_bridge_v1',ROOT/args.refined_dir]
    data={};provenance={}
    for batch in batches:
        terminal=read(batch/'execution_complete.json');assert terminal['execution_complete'] and terminal['sources_unchanged']
        assert all(terminal['identical_sensor_rng_final_per_method'].values())
        provenance[str((batch/'execution_complete.json').relative_to(ROOT))]=sha(batch/'execution_complete.json')
        for result in terminal['results']:
            method=result['method'];dt=result['dt'];folder=batch/f'{method}__dt_{dt:.6f}'
            meta=read(folder/'metadata.json');assert sha(folder/'trace.npz')==meta['trace_sha256']
            assert result['complete'] and result['physics_steps']==round(16/dt)
            data[(method,dt)]=(folder,result,meta)
    rows=[]
    for method in METHODS:
        original=None
        for dt in (.0005,.00025,.000125):
            folder,result,meta=data[(method,dt)]
            with np.load(folder/'trace.npz',allow_pickle=False) as z:
                a={k:z[k] for k in ('time','qpos','qvel','ctrl','phase','rng_state_after_sample','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_penetration','substep_world_penetration','substep_warning_count')}
            if original is None:original=a
            for key in ('time','ctrl','phase','rng_state_after_sample'):
                assert a[key].dtype==original[key].dtype and a[key].shape==original[key].shape and a[key].tobytes()==original[key].tobytes(),(method,dt,key)
            assert np.isfinite(a['qpos']).all() and np.isfinite(a['qvel']).all()
            duration=a['substep_duration'];t=a['substep_time'];force=a['substep_skin_normal'];limits=np.asarray(meta['limits'])
            metrics={}
            for phase,lo,hi in (('preparation',0,7.5),('holding',7.5,12),('collision',9.2,10.4),('release',12,16)):
                indices=np.flatnonzero((t>=lo-1e-9)&(t<hi-1e-9));local=force[indices];peak=float(local.max());k,digit=np.unravel_index(local.argmax(),local.shape);absolute_k=indices[k]
                excess=np.maximum(local-limits,0)
                metrics[phase]=dict(peak_N=peak,peak_digit=int(digit),peak_END_s=float(t[absolute_k]+duration[absolute_k]),local_excess_Ns=float(np.sum(excess*duration[indices,None])),
                    any_local_limit_exposure_s=float(duration[indices][np.any(excess>0,axis=1)].sum()),max_penetration_m=float(a['substep_penetration'][indices].max()),max_world_penetration_m=float(a['substep_world_penetration'][indices].max()),max_warning_count=int(a['substep_warning_count'][indices].max()))
                assert abs(peak-result['peak_diagnostics'][phase]['peak_N'])<1e-12
                assert abs(metrics[phase]['local_excess_Ns']-result['peak_diagnostics'][phase]['local_excess_Ns'])<1e-12
            rows.append(dict(method=method,dt=dt,physics_steps=result['physics_steps'],new_feedback_decisions=meta['new_feedback_decisions'],source_trace_sha256=result['source_trace_sha256'],replay_trace_sha256=meta['trace_sha256'],metrics=metrics,
                max_abs_qpos_change_from_coarse=float(np.max(abs(a['qpos']-original['qpos']))),max_abs_qvel_change_from_coarse=float(np.max(abs(a['qvel']-original['qvel']))),actual_inputs_and_sensor_rng_exact=True))
            del a
    differences=[]
    for method in METHODS:
        by_dt={x['dt']:x for x in rows if x['method']==method}
        for phase in ('holding','collision','release'):
            for left,right in ((.0005,.00025),(.00025,.000125),(.0005,.000125)):
                a,b=by_dt[left]['metrics'][phase],by_dt[right]['metrics'][phase]
                differences.append(dict(method=method,phase=phase,from_dt=left,to_dt=right,peak_relative_change=(b['peak_N']/a['peak_N']-1) if a['peak_N'] else None,local_excess_relative_change=(b['local_excess_Ns']/a['local_excess_Ns']-1) if a['local_excess_Ns'] else None,peak_END_change_s=b['peak_END_s']-a['peak_END_s']))
    result=dict(schema='fixed_input_contact_peak_sensitivity_readout_v1',scope='Three old-V12 recorded policies, one physical context; numerical sensitivity of unchanged actual inputs, not new closed-loop strategy performance',source_lock_sha256=sha(ROOT/'identification/contact_peak_convergence_v1/source_lock.json'),provenance=provenance,reader_sha256=sha(__file__),rows=rows,pairwise_changes=differences,convergence_tolerance=None,physical_safety_certified=False,policy_superiority_claim_allowed=False)
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(rows=len(rows),pairwise=len(differences),out=str(target)),ensure_ascii=False))
if __name__=='__main__':main()

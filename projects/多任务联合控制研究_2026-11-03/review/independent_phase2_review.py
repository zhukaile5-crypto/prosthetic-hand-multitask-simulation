"""Independent saved-data phase2 audit. No robot, policy, solver or physics import."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, io, runpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'calibration/phase2_public_v1'
helper = runpy.run_path(str(ROOT / 'review/independent_public_calibration_review.py'))
aggregate = helper['aggregate']

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b,tol=1e-9): return bool(np.allclose(a,b,atol=tol,rtol=0,equal_nan=False))
def stats(errors):
    e=np.asarray(errors)
    rmse=np.sqrt(np.mean(e*e,axis=0));p95=np.percentile(abs(e),95,axis=0);mx=abs(e).max(axis=0)
    gates=(rmse<=np.r_[np.full(5,.35),np.full(3,.05),np.full(3,.005)])
    gates[:5] &= p95[:5]<=.75
    return dict(rmse_per_output=rmse.tolist(),p95_abs_per_output=p95.tolist(),max_abs_per_output=mx.tolist(),
                engineering_gate_per_output=gates.tolist(),all_engineering_gates_passed=bool(gates.all()),vectors=len(e))
def convolution(K,path):
    result=np.zeros((len(path),11));du=np.diff(path,axis=0)
    for target in range(1,len(path)):
        for k in range(target): result[target]+=K[target-k]@du[k]
    return result
def expected_holdout(index,base,lo,hi,plan):
    # Independent Sylvester recursion gives the prescribed 8x8 orthogonal signs.
    H=np.ones((1,1))
    for _ in range(3): H=np.block([[H,H],[H,-H]])
    amp=np.asarray(plan['holdout_amplitude']);t=np.linspace(0,1,26);path=np.tile(base,(26,1))
    for j,col in enumerate([1,2,3,4,5,0,6,7]):
        sign=H[index,col]
        if base[j]+amp[j]>hi[j]+1e-12: sign=-1
        if base[j]-amp[j]<lo[j]-1e-12: sign=1
        wave=np.sin(np.pi*t)**2*(.8+.2*np.cos(2*np.pi*(1+j%3)*t+index*.37+j*.23))
        path[:,j]+=sign*amp[j]*wave
    return path

def main():
    plan=read(FOLDER/'plan.json');summary=read(FOLDER/'summary.json')
    package=read(ROOT/'calibration/public_model_phase2_v1.json');snap=read(FOLDER/'source_snapshot_manifest.json')
    if not summary['execution_complete']: raise RuntimeError('Calibration not complete')
    checks={};inputs={};cases=[];packets=0;available_count=0
    def check(name,value): checks[name]=bool(value)
    for p in [FOLDER/'plan.json',FOLDER/'summary.json',FOLDER/'source_snapshot_manifest.json',ROOT/'calibration/public_model_phase2_v1.json']:
        inputs[str(p.relative_to(ROOT))]=sha(p)
    check('plan_frozen_before_data',plan['status']=='frozen_before_phase2_data')
    check('same_plan_bytes',sha(FOLDER/'plan.json')==sha(ROOT/'calibration/plan_phase2_v1.json')==package['calibration_plan_sha256'])
    check('seven_execution_sources',len(snap['files'])==len(plan['source_sha256'])==7)
    # Match by exact execution-source digest; later live files are deliberately not inputs.
    check('snapshot_source_inventory',sorted(x['sha256'] for x in snap['files'])==sorted(plan['source_sha256'].values()))
    for entry in snap['files']:
        p=FOLDER/entry['snapshot_path'];check('snapshot/'+entry['snapshot_path'],p.is_file() and sha(p)==entry['sha256'])
        inputs[str(p.relative_to(ROOT))]=sha(p)
    archive={Path(x['path']).name+str(Path(x['path']).parent):x['sha256'] for x in package['raw_archive_manifest']}
    archive_hashes=sorted(x['sha256'] for x in package['raw_archive_manifest']);observed_hashes=[]
    limits=plan['stop_gates']
    expected_labels=['zero','zero_replay']+[f'train_{j}_{s}' for j in range(8) for s in ('plus','minus')]+[f'holdout_{j:02d}' for j in range(8)]
    for shape in plan['shapes']:
        folder=FOLDER/shape;base=read(folder/'base_metadata.json');model=read(folder/'model.json')
        public=next(c for c in package['shapes'] if c['shape']==shape);rows={};metas={};unavailable=[];max_packet_error=0.;start=None
        for p in [folder/'base_metadata.json',folder/'model.json'] : inputs[str(p.relative_to(ROOT))]=sha(p)
        check(shape+'/model_summary_hash',sha(folder/'model.json')==next(c['model_sha256'] for c in summary['shapes'] if c['shape']==shape)==public['source_model_sha256'])
        check(shape+'/exact_branch_inventory',{p.name for p in folder.iterdir() if p.is_dir()}==set(expected_labels))
        R0=np.asarray(base['reference_rotation']);p0=np.asarray(base['reference_origin']);lo=np.asarray(base['lower']);hi=np.asarray(base['upper'])
        for label in expected_labels:
            path=folder/label/'trace.npz';m=read(path.with_name('metadata.json'));metas[label]=m
            inputs[str(path.with_name('metadata.json').relative_to(ROOT))]=sha(path.with_name('metadata.json'))
            if not m['available']:
                unavailable.append(label)
                check(shape+'/'+label+'/explicit_unavailable',not m['attempted_physics'] and not m['complete'] and not path.exists() and m['executed_path'] is None)
                continue
            available_count+=1;digest=sha(path);inputs[str(path.relative_to(ROOT))]=digest;observed_hashes.append(digest)
            check(shape+'/'+label+'/raw_hash',digest==m['trace_sha256'])
            with np.load(path,allow_pickle=False) as a: z={k:a[k] for k in a.files}
            t0=float(z['time'][0]);check(shape+'/'+label+'/clock_coverage',len(z['time'])==101 and len(z['physical_time'])==1000 and same(z['time'],t0+np.arange(101)*.005,1e-8) and same(z['physical_interval_start'],t0+np.arange(1000)*.0005,1e-8) and same(z['physical_time'],t0+np.arange(1,1001)*.0005,1e-8) and same(z['physical_duration'],.0005))
            path26=np.asarray(m['requested_path']);ramp=np.zeros((101,8));ramp[0]=path26[0]
            for interval in range(25):
                for j in range(1,5):ramp[interval*4+j]=path26[interval]+j/4*(path26[interval+1]-path26[interval])
            check(shape+'/'+label+'/actual_four_tick_path',same(z['input_reference'],ramp,1e-12) and same(m['executed_path'],ramp,1e-12) and same(z['alpha'],ramp[:,:5],1e-12))
            check(shape+'/'+label+'/actual_reference_domain',bool((z['physical_input_reference']>=lo-1e-10).all() and (z['physical_input_reference']<=hi+1e-10).all()))
            initial={k:z[k][0] for k in ['qpos','qvel','ctrl','alpha','y','y_palm','palm_position','palm_rotation','input_reference']}
            if start is None:start=initial
            check(shape+'/'+label+'/identical_restored_start',all(np.array_equal(initial[k],start[k]) for k in initial))
            gate=True;packet_ok=True
            for prefix in ['','physical_']:
                n=z[prefix+'pressure'];w=z[prefix+'wrench_reference']
                gate &= bool(np.isfinite(z[prefix+'y']).all() and np.isfinite(z[prefix+'qpos']).all() and np.isfinite(z[prefix+'qvel']).all() and np.isfinite(z[prefix+'ctrl']).all() and (n<=limits['per_finger_pressure_N']).all() and (n.sum(1)<=limits['sum_finger_pressure_N']).all() and (z[prefix+'penetration']<=limits['penetration_m']).all() and (z[prefix+'table_support_N']<=limits['table_support_N']).all() and (z[prefix+'object_lift_m']>=limits['minimum_object_lift_m']).all() and ((n>.2).sum(1)>=3).all() and (n[:,0]>.2).all())
                for i,raw in enumerate(z[prefix+'raw_patches']):
                    vals=aggregate(json.loads(str(raw)),z[prefix+'palm_rotation'][i],z[prefix+'palm_position'][i],R0,p0)
                    expected=[z[prefix+k][i] for k in ['pressure','wrench_palm','wrench_reference','normal_reference','tangent_reference','auxiliary_reference']]
                    err=max(float(abs(a-b).max()) for a,b in zip(vals[:6],expected));max_packet_error=max(max_packet_error,err);packet_ok &= err<1e-9;packets+=1
                packet_ok &= same(z[prefix+'y'],np.c_[n,w]) and same(z[prefix+'y_palm'],np.c_[n,z[prefix+'wrench_palm']])
            check(shape+'/'+label+'/all_raw_engineering_stop_gates',gate and m['complete'] and m['stop_reason'] is None)
            check(shape+'/'+label+'/every_patch_both_frames',packet_ok)
            rows[label]={k:v for k,v in z.items() if 'raw_patches' not in k and k!='metadata'}
            if label.startswith('holdout_'):
                idx=int(label.rsplit('_',1)[1]);check(shape+'/'+label+'/independent_preset_mixed_path',same(path26,expected_holdout(idx,np.asarray(base['base']),lo,hi,plan),1e-12))
            print(shape,label,'reviewed',flush=True)
        check(shape+'/zero_replay_all_saved_fields',all(np.array_equal(rows['zero'][k],rows['zero_replay'][k]) for k in rows['zero']))
        check(shape+'/unavailable_inventory',unavailable==public['unavailable_branches'])
        frame_summaries={}
        for frame,key in [('initial_fixed_reference','y'),('current_palm','y_palm')]:
            K=np.zeros((101,11,8));identified=[]
            for axis in range(8):
                plus=f'train_{axis}_plus';minus=f'train_{axis}_minus';hp=metas[plus]['available'];hm=metas[minus]['available'];identified.append(hp or hm)
                if hp and hm:
                    delta=rows[plus]['input_reference'][-1,axis]-rows[minus]['input_reference'][-1,axis]
                    K[:,:,axis]=(rows[plus][key]-rows[minus][key])/delta
                elif hp or hm:
                    label=plus if hp else minus;delta=rows[label]['input_reference'][-1,axis]-rows['zero']['input_reference'][-1,axis]
                    K[:,:,axis]=(rows[label][key]-rows['zero'][key])/delta
            pub=public['coordinates'][frame];internal=model['coordinates'][frame]
            check(shape+'/'+frame+'/actual_delta_kernel_dense',same(K,pub['response_to_20ms_interval_increment'],1e-7))
            K=K[::4];check(shape+'/'+frame+'/actual_delta_kernel_20ms',same(K,internal['kernel_response_to_20ms_interval_increment'],1e-7))
            check(shape+'/'+frame+'/identified_not_zero_filled',all(identified) and identified==pub['identified_inputs']==internal['identified_inputs'])
            history=base['history'];ht=np.array([h['time'] for h in history]);ht-=ht[-1];hy=np.array([h[key] for h in history])
            check(shape+'/'+frame+'/only_past_history',bool(ht.max()<=0 and len(ht)>=4))
            by_h={float(h):[] for h in plan['forecast_horizons_s']};diffs=[]
            for index in range(8):
                label=f'holdout_{index:02d}';z=rows[label];path=z['input_reference'][::4];measured=z[key][::4];modeled=convolution(K,path);residual=measured-modeled
                diffs.extend(measured-rows['zero'][key][::4]-modeled)
                computed=[]
                for origin in range(26):
                    now=origin*.02;past_t=np.r_[ht,np.arange(1,origin+1)*.02];past_y=np.vstack([hy,residual[1:origin+1]])
                    choose=past_t>=now-plan['causal_window_s']-1e-10;tx=past_t[choose]-now;yy=past_y[choose]
                    if len(tx)>=4:
                        slope,level=np.linalg.lstsq(np.c_[tx,np.ones(len(tx))],yy,rcond=None)[0]
                    else:level=yy[-1];slope=np.zeros(11)
                    for h in by_h:
                        target=origin+int(round(h/.02))
                        if target>=26:continue
                        error=level+h*slope+modeled[target]-measured[target];by_h[h].append(error)
                        computed.append(dict(origin_s=now,horizon_s=h,error=error.tolist()))
                saved=internal['holdout'][index]['complete_causal_errors']
                check(shape+'/'+frame+'/'+label+'/all_causal_error_vectors',len(computed)==len(saved) and all(a['origin_s']==b['origin_s'] and a['horizon_s']==b['horizon_s'] and same(a['error'],b['error'],1e-8) for a,b in zip(computed,saved)))
            summaries={}
            for h,errors in by_h.items():
                actual=stats(errors);saved=internal['complete_causal_by_horizon'][str(h)];exported=pub['holdout_complete_causal_by_horizon'][str(h)]
                check(shape+'/'+frame+f'/h{h}/all_summary_values',all(same(actual[k],saved[k],1e-8) and same(actual[k],exported[k],1e-8) for k in ['rmse_per_output','p95_abs_per_output','max_abs_per_output']) and actual['vectors']==saved['vectors']==8*(26-round(h/.02)))
                check(shape+'/'+frame+f'/h{h}/predeclared_gates',actual['engineering_gate_per_output']==saved['engineering_gate_per_output'] and actual['all_engineering_gates_passed'])
                summaries[str(h)]=actual
            sd=stats(diffs);saved=internal['differential_response_error']
            check(shape+'/'+frame+'/separate_differential_summary',all(same(sd[k],saved[k],1e-8) for k in ['rmse_per_output','p95_abs_per_output','max_abs_per_output']) and sd['all_engineering_gates_passed'])
            U=np.vstack([np.diff(rows[f'holdout_{i:02d}']['input_reference'][::4],axis=0) for i in range(8)])/np.asarray(plan['holdout_amplitude'])
            singular=np.linalg.svd(U,compute_uv=False)
            check(shape+'/'+frame+'/mixed_excitation_singular_values',same(singular,pub['holdout_normalized_input_singular_values'],1e-8))
            frame_summaries[frame]=dict(complete_causal=summaries,differential=sd,normalized_input_singular_values=singular.tolist(),normalized_input_rank=int(np.sum(singular>max(1e-10,singular[0]*1e-6))))
        paths=np.vstack([v['input_reference']-np.asarray(base['base']) for v in rows.values()])
        check(shape+'/reported_input_extent',same(paths.min(0),public['measured_input_delta_min']) and same(paths.max(0),public['measured_input_delta_max']))
        cases.append(dict(shape=shape,planned=26,executed=26-len(unavailable),unavailable=unavailable,max_patch_reconstruction_error=max_packet_error,frames=frame_summaries,input_min=paths.min(0).tolist(),input_max=paths.max(0).tolist()))
    check('raw_archive_exact_74_traces',available_count==74 and archive_hashes==sorted(observed_hashes))
    check('execution_not_model_admission',summary['execution_complete'] and package['execution_complete'] and not summary['model_admitted'] and not package['model_admitted'])
    result=dict(schema='independent_public_phase2_saved_data_audit_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        raw_design_audit_passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
        planned_branches=78,executed_branches=available_count,packets_recomputed=packets,cases=cases,input_sha256=inputs,
        reviewer_sha256=sha(__file__),model_admitted=False,
        limitations=['Same nominal prepared clones, one physical seed and 0.12kg only; not unseen physical-context confirmation.',
            'Mixed paths are eight deterministic smooth input sequences; no full rectangular joint-input envelope or closed-loop safety guarantee follows.',
            'Positive lift is unavailable for all shapes; sphere index positive is unavailable. Algebraic mixed-path rank is not bidirectional authority.',
            'Engineering forecast gates cover only 20/40/60/100ms. Dense 5ms kernel sampling is not a separately validated 5ms forecast guarantee.',
            'Public ideal tactile aggregation has no online delay/noise in this calibration; controller must handle estimator uncertainty separately.',
            'Current-palm and fixed-robot-frame kernels have distinct coordinate meanings; use matching measurement and prediction frames.',
            'No finite-memory tail or unmeasured physical-domain extrapolation is admitted; retain past command effects or update incrementally from measurements.'],
        scope='Only saved-data arithmetic; no solver, policy, robot or dynamics imported. Executed snapshots used instead of later live source files.')
    out=ROOT/'review'/('independent_public_calibration_phase2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['raw_design_audit_passed'],checks=len(checks),failed=result['failed'],packets=packets,path=str(out),sha256=sha(out)),ensure_ascii=False))

if __name__=='__main__':main()

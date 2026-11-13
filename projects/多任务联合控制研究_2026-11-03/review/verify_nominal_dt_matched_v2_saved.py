"""Independent six-record matched-reference audit; no simulation imports."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'output/nominal_dt_610212_matched_v2'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def close(a,b,tol=1e-9):return bool(np.allclose(a,b,atol=tol,rtol=0))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x
def band(z,dt):
    half=round(.025/dt);tk=np.arange(-half,half+1)*dt;k=100*dt*np.sinc(100*tk)*np.hanning(2*half+1);k/=k.sum()
    f=np.vstack([z['normal'][0],z['substep_normal']]);grid=np.r_[0.,z['substep_time']+z['substep_duration']]
    return np.column_stack([np.interp(z['time'],grid,np.convolve(np.pad(f[:,i],half,mode='reflect'),k,mode='valid')) for i in range(5)])
def variation(t,y,lo,hi):
    keep=(t>=lo-1e-9)&(t<hi-1e-9);t=t[keep];y=y[keep];per=np.sum(abs(np.diff(y,axis=0)),axis=0);drift=abs(y[-1]-y[0]);duration=t[-1]-t[0]
    return dict(samples=len(t),finger_TV_rate_N_s=float(per.sum()/duration),per_finger_TV_N=per,per_finger_net_drift_N=drift,per_finger_reversal_N=.5*(per-drift),mean_N=y.mean(0))
def spectrum(t,y,lo,hi):
    keep=(t>=lo-1e-9)&(t<hi-1e-9);v=y[keep];dt=float(np.median(np.diff(t)));w=np.hanning(len(v));psd=abs(np.fft.rfft((v-v.mean(0))*w[:,None],axis=0))**2/(1/dt*np.sum(w*w))
    psd[1:-1 if len(v)%2==0 else None]*=2.;f=np.fft.rfftfreq(len(v),dt);df=f[1]-f[0]
    return dict(band_power_N2=np.array([psd[(f>=a)&(f<b)].sum(0)*df for a,b in [(.5,5.),(5.,20.),(20.,50.)]]),dominant_frequency_Hz=f[(f>=.5)&(f<=50.)][np.argmax(psd[(f>=.5)&(f<=50.)],axis=0)],frequency_resolution_Hz=float(df))
def main():
    summary=read(DATA/'summary.json');plan=read(DATA/'plan.json');checks={};records=[];cache={};source={};expected_pre=[];expected_post=[];rng=np.random.default_rng(plan['seed']+910000)
    def check(k,v):checks[k]=bool(v)
    check('six_frozen_prospective_records',summary['execution_complete'] and len(summary['rows'])==len(plan['matrix'])==6 and summary['plan_sha256']==sha(DATA/'plan.json')==sha(ROOT/'review/nominal_dt_610212_matched_plan_v2.json') and plan['no_confirmatory_superiority_claim'])
    check('recorded_sources_and_originals_unchanged',all(summary[k] for k in ['sources_unchanged','foundation_sources_unchanged','asset_sources_unchanged','original_inputs_unchanged']))
    check('actual_execution_script_snapshot',sha(DATA/'sources/diagnose_nominal_dt_610212_v2.py')==plan['diagnostic_script_sha256']==sha(ROOT/'calibration/diagnose_nominal_dt_610212_v2.py'))
    for name in ['runtime.py','evaluation.py','protocol.json','research_imports.py']:
        check('key_dependency_snapshot/'+name,sha(DATA/'sources/research'/name)==plan['sources'][name])
    for i in range(3201):
        expected_pre.append(digest(rng.bit_generator.state));rng.normal(size=48);rng.random(6);expected_post.append(digest(rng.bit_generator.state))
    needed=['time','qpos','qvel','normal','skin_normal','alpha','alpha_before','ctrl','ctrl_before','executed_motion_reference','executed_action','raw_patches_json','observation_json','position','relative','rotation','lift','wrist_position','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_penetration','substep_world_penetration','warning_count']
    for method,m in plan['records'].items():
        folder=Path(m['directory']);check('original_identity/'+method,all(sha(folder/k)==v for k,v in m['input_sha256'].items()))
        with np.load(folder/'trace.npz',allow_pickle=False) as a:source[method]={k:a[k] for k in needed if k in a.files}
    for row in summary['rows']:
        method=row['method'];dt=row['dt'];folder=Path(row['directory']);probe=read(folder/'probe.json');meta=read(folder/'metadata.json');result=read(folder/'result.json');key=f'{method}/{dt}'
        with np.load(folder/'trace.npz',allow_pickle=False) as a:z={k:a[k] for k in needed if k in a.files}
        check(key+'/record_hashes',all(sha(folder/f'{name}.json')==row[f'{name}_sha256'] for name in ['metadata','result','probe']) and sha(folder/'trace.npz')==row['trace_sha256'])
        check(key+'/same_actual_config_except_timestep',probe['config_executed']=={**plan['records'][method]['config'],'timestep':dt} and meta['config']==probe['config_executed'])
        t=z['time'];st=z['substep_time'];duration=z['substep_duration'];post=t>=7.5-1e-9;check(key+'/complete_actual_clocks',np.array_equal(t,np.arange(3201)*.005) and len(st)==round(16/dt) and close(st,np.arange(len(st))*dt,1e-7) and close(duration,dt,1e-12) and abs(duration.sum()-16)<1e-8)
        check(key+'/all_actual_state_finite_no_warning',all(np.isfinite(z[k]).all() for k in ['qpos','qvel','normal','skin_normal','ctrl','alpha','substep_normal','substep_skin_normal']) and not z['warning_count'].any())
        check(key+'/entire_sensor_rng_state_sequence_rebuilt',probe['rng_before']==expected_pre and probe['rng_after']==expected_post)
        check(key+'/posthandoff_alpha_ctrl_motion_exact_source',all(np.array_equal(z[k][post],source[method][k][post]) for k in ['alpha','ctrl','executed_motion_reference']))
        match_fields=['qpos','qvel','normal','skin_normal','alpha','ctrl','position','relative','rotation','lift','wrist_position','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_penetration','substep_world_penetration','warning_count','raw_patches_json','observation_json','executed_motion_reference']
        reproduction={k:bool(np.array_equal(z[k],source[method][k])) for k in match_fields if k in z and k in source[method]}
        if dt==.0005:check(key+'/source_entire16s_physical_and_observation_exact',all(reproduction.values()))
        switch=probe['reference_switch'];check(key+'/actual_handoff_record',close(z['alpha_before'][1500],switch['alpha_before'],1e-12) and close(z['qpos'][1500],switch['qpos_before'],1e-12) and close(z['qvel'][1500],switch['qvel_before'],1e-12) and close(source[method]['alpha'][1500]-z['alpha_before'][1500],switch['requested_switch_delta'],1e-12))
        check(key+'/packet_file_identity',sha(probe['packet_path'])==probe['packet_sha256'])
        count=0;clock_error=0.;warnings=0
        with gzip.open(probe['packet_path'],'rt',encoding='utf-8') as stream:
            for line in stream:
                header=line.split(',',4);end=float(header[0].split(':',1)[1]);start=float(header[1].split(':',1)[1]);h=float(header[2].split(':',1)[1]);warnings=max(warnings,int(header[3].split(':',1)[1]));clock_error=max(clock_error,abs(end-(st[count]+duration[count])),abs(start-st[count]),abs(h-dt));count+=1
        check(key+'/all_raw_contact_packet_clocks_complete',count==len(st)==probe['packet_count'] and clock_error<1e-9 and warnings==0)
        wave=band(z,dt);windows={}
        for name,lo,hi in [('static',8.2,8.9),('late',11.,11.7),('holding',7.5,12.),('full',0.,16.)]:
            raw=variation(t,z['normal'],lo,hi);smooth=variation(t,wave,lo,hi);sp=spectrum(t,z['normal'],lo,hi);saved=probe['spectra'][name]
            check(key+'/raw_metric/'+name,all(close(value,saved[k],1e-8) if isinstance(value,np.ndarray) else abs(value-saved[k])<1e-8 for k,value in raw.items()))
            check(key+'/band50_and_spectrum/'+name,abs(smooth['finger_TV_rate_N_s']-saved['band50_finger_TV_rate_N_s'])<1e-8 and all(close(value,saved[k],1e-8) if isinstance(value,np.ndarray) else abs(value-saved[k])<1e-8 for k,value in sp.items()))
            windows[name]=dict(raw_200Hz=raw,band50=smooth,spectrum=sp)
        limits=np.asarray(meta['limits']);budget=meta['budget'];skin=z['substep_skin_normal'];guard=bool((skin<=limits+.02).all() and (skin.sum(1)<=budget+.02).all() and (z['skin_normal'][-1]<=limits+.02).all() and z['skin_normal'][-1].sum()<=budget+.02)
        check(key+'/raw_force_guard_recalculated',guard==result['research_metrics']['safety']['guard_pass'])
        local={**{k:v for k,v in z.items() if k in ['qpos','qvel','normal','skin_normal','alpha_before','ctrl_before','position','relative']},'time':t};cache[(method,dt)]=local
        records.append(dict(method=method,dt=dt,source_16s_exact=reproduction if dt==.0005 else None,force_guard=guard,task_success=result['research_metrics']['full_motion_mission_success'],max_hand_object_penetration_m=float(z['substep_penetration'].max()),max_world_object_penetration_m=float(z['substep_world_penetration'].max()),windows=windows,reference_switch=switch,packet_clock_max_error_s=clock_error))
        print(key,'raw_reviewed',flush=True)
    pairs=[]
    for method in plan['records']:
        a=cache[(method,.0005)];b=cache[(method,.00025)];r0=next(x for x in records if x['method']==method and x['dt']==.0005);r1=next(x for x in records if x['method']==method and x['dt']==.00025)
        diffs={k:float(np.max(abs(a[k][1500]-b[k][1500]))) for k in ['qpos','qvel','normal','skin_normal','alpha_before','ctrl_before']};position=float(np.linalg.norm(a['position'][1500]-b['position'][1500]));check(method+'/handoff_is_explicitly_different_not_same_state_dt_test',diffs['qpos']>0 and diffs['alpha_before']>0)
        windows={}
        for name in ['static','late','holding','full']:
            windows[name]={}
            for sampling in ['raw_200Hz','band50']:
                x=r0['windows'][name][sampling]['finger_TV_rate_N_s'];y=r1['windows'][name][sampling]['finger_TV_rate_N_s'];windows[name][sampling]=dict(dt_0_5ms=x,dt_0_25ms=y,relative_change=(y-x)/max(abs(x),1e-12))
            saved=summary['pairs'][method]['windows'][name];check(method+'/'+name+'/summary_change_recomputed',abs(windows[name]['raw_200Hz']['relative_change']-saved['raw200Hz_TV_relative_change'])<1e-9 and abs(windows[name]['band50']['relative_change']-saved['band50_TV_relative_change'])<1e-9)
        pairs.append(dict(method=method,handoff_max_abs_differences=diffs,handoff_object_position_difference_m=position,additional_handoff_alpha_alignment=np.asarray(r1['reference_switch']['alpha_before'])-np.asarray(r0['reference_switch']['alpha_before']),windows=windows))
    result=dict(schema='independent_nominal_dt_matched_reference_v2_saved_audit_v1',passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(DATA/'plan.json'),summary_sha256=sha(DATA/'summary.json'),reviewer_sha256=sha(__file__),physics_steps_executed=0,records=records,pairs=pairs,scope=['Exact actual saved alpha/actuator/motion references after7.5s and RNG stream; .5ms replay independently compares full16s source physical trajectories/observations.','Each dt separately repeats common preparation from reset; .25ms handoff states and alignment increments differ. This is cumulative dt-plus-preparation sensitivity, not pure same-state integration error or a rerun of feedback-policy performance.','Diagnostic FFT/TV windows are half open: static/late have140 samples and.695s path duration. They are not numerically identical to the frozen inclusive141-sample V5 quality endpoints.','50Hz FIR is computed on the complete physical-end grid; static/late are away from release. Holding/full band TV retains known phase-boundary coupling.','Recorded no warnings does not certify material oscillation or global physical convergence; no contact tuning or new safety admission is done.','V1 failed on tuple/list config representation before the physical loop; its old evidence remains unchanged.'])
    out=ROOT/'review'/('independent_nominal_dt_v2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');out.write_text(json.dumps(serial(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(serial(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],path=str(out),sha256=sha(out),pairs=pairs)),ensure_ascii=False))
if __name__=='__main__':main()

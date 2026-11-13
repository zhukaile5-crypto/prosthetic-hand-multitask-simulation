"""Stratified saved raw checks; explicitly not exhaustive108 raw reprocessing."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,runpy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
old=runpy.run_path(str(ROOT/'review/independent_development_v5_broad_review.py'));equivalent=old['equivalent'];outcome=old['physical_outcome'];dwell=old['dwell']
tracking=runpy.run_path(str(ROOT/'review/v5_tracking_diagnostics.py'))['summarize_tracking']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wave(z,m,p):
    dt=m['physics_timestep'];half=round(.025/dt);x=np.arange(-half,half+1)*dt;k=np.sinc(100*x)*np.hanning(2*half+1)*100*dt;k/=k.sum();times=np.r_[0.,z['substep_time']+z['substep_duration']];f=np.vstack([z['normal'][0],z['substep_normal']]);t=z['time'];y=np.full((len(t),5),np.nan);release=m.get('release_at');release=p['schedule']['release_s'] if release is None else release
    bounds=sorted(set([0.,p['schedule']['handoff_s'],release,p['schedule']['duration_s']]))
    for start,end in zip(bounds,bounds[1:]):
        physical=(times>=start-1e-9)&(times<=end+1e-9);keep=(t>=start-1e-9)&(t<=end+1e-9)
        if start==release and start>0:keep&=t>start+1e-9
        for i in range(5):y[keep,i]=np.interp(t[keep],times[physical],np.convolve(np.pad(f[physical,i],half,mode='reflect'),k,mode='valid'))
    return y
def variation(t,f):
    seconds=t[-1]-t[0];tv=np.sum(abs(np.diff(f,axis=0)),axis=0);net=abs(f[-1]-f[0]);total=f.sum(1);ttv=np.sum(abs(np.diff(total)));tnet=abs(total[-1]-total[0])
    return dict(finger_TV_rate_N_s=float(tv.sum()/seconds),finger_net_drift_rate_N_s=float(net.sum()/seconds),finger_reversal_rate_N_s=float((tv-net).sum()/(2*seconds)),total_force_TV_rate_N_s=float(ttv/seconds),finger_SD_mean_N=float(f.std(axis=0).mean()),total_force_SD_N=float(total.std()))
def main():
    checks={};records=[];reproductions=[]
    def check(k,v):checks[k]=bool(v)
    jobs={'development_v6_known_counterexamples_v2_contract':[(s,'joint',610212) for s in ['nominal','fragile_conflict','dynamic_transport','collision']]+[('nominal','ratio',610212),('fragile_conflict','passive',610212),('dynamic_transport','ratio',610212),('collision','passive',610212)],
          'development_v6_1_physical_all16':[('nominal','ratio',610221),('dynamic_transport','joint',610222),('sphere','joint',610221),('box_heavy','passive',610222),('fragile_conflict','joint',610222),('collision','joint',610222),('friction_drop','joint',610222),('friction_drop','ratio',610222)]}
    wanted=['time','normal','skin_normal','palm','support','position','relative','relative_velocity','angular_velocity','lift','lift_ref','all_hand_object','controller_active','transport_paused','warning_count','collision','actual_motion_target','executed_motion_reference','wrist_position','requested_wrist_reference','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm','qpos','qvel','ctrl','alpha']
    for dataset,selected in jobs.items():
        folder=ROOT/'output'/dataset;rows=read(folder/'trials.json');snapshot=folder/'sources';protocol=read(snapshot/'protocol.json');api=runpy.run_path(str(snapshot/'evaluation.py'))
        for scene,method,seed in selected:
            row=next(x for x in rows if (x['scenario'],x['method'],x['seed'])==(scene,method,seed));path=Path(row['trace_path']);m=read(path.with_name('metadata.json'));key=f'{dataset}/{scene}/{method}/{seed}'
            with np.load(path,allow_pickle=False) as a:z={k:a[k] for k in wanted if k in a.files}
            r=row['research_metrics'];recomputed=api['evaluate_arrays'](z,m,protocol);recomputed.update(trace_sha256=sha(path),protocol_sha256=sha(snapshot/'protocol.json'),evaluation_source_sha256=sha(snapshot/'evaluation.py'));check(key+'/whole_frozen_evaluation_recomputed',equivalent(recomputed,r))
            own=outcome(z,m);check(key+'/independent_task_physical_release_and_raw_safety',own['task_success']==r['mission_success'] and own['physical_exposure']==r['release']['physical_exposure'] and own['physical_release']==r['release']['physical_success'] and equivalent(own['physical_release_latency'],r['release']['physical_clearance_latency_s']) and own['force_guard']==r['safety']['guard_pass'] and abs(own['local_excess']-r['safety']['local_excess_integral_Ns'])<1e-8 and abs(own['total_excess']-r['safety']['total_excess_integral_Ns'])<1e-8)
            t=z['time'];wrist=dwell(t,np.any(abs(z['wrist_position']-z['requested_wrist_reference'])>.015,axis=1),7.5,12.,.3,True);check(key+'/full_motion_independent_wrist_guard',equivalent(wrist,r['wrist_tracking_guard']['failure_at']) and bool(own['task_success'] and wrist is None)==r['full_motion_mission_success'])
            for name,window in [('holding_motion_tracking',(7.5,12.)),('dynamic_motion_tracking',(9.3,11.7))]:
                tr=tracking(z,window);saved=r['progress'][name];check(key+'/'+name,tr['complete_window']==saved['complete_window'] and np.allclose(tr['nominal_tracking_rmse'],saved['rmse'],atol=1e-10,rtol=0) and np.allclose(tr['nominal_tracking_IAE'],saved['absolute_error_integral'],atol=1e-10,rtol=0))
            band=wave(z,m,protocol);qualities={}
            for name,start,end in [('static_quality',8.2,8.9),('dynamic_quality',9.3,11.7),('entire_holding_quality',7.5,12.)]:
                keep=(t>=start-1e-9)&(t<=end+1e-9);p=z['normal'];supported=(p[:,0]>.2)&((p>.2).sum(1)>=3);raised=z['position'][:,2]>=m['initial_position'][2]+.025;eligible=bool(np.all(supported[keep]&raised[keep]&(z['support'][keep]<=.05)&z['controller_active'][keep].astype(bool)));check(key+'/'+name+'/independent_coverage',eligible==r[name]['eligible'])
                if eligible:
                    q={sampling:variation(t[keep],signal[keep]) for sampling,signal in [('bandlimited',band),('raw_200Hz',z['normal'])]};qualities[name]=q
                    check(key+'/'+name+'/independent_FIR_and_raw_variation',all(abs(v-r[name][sampling][k])<1e-8 for sampling,metrics in q.items() for k,v in metrics.items()))
            if dataset=='development_v6_known_counterexamples_v2_contract' and method!='joint':
                source=ROOT/f'output/development_v5_physical_all16/runs/{scene}__{method}__{seed}/trace.npz';equal={}
                with np.load(source,allow_pickle=False) as a:
                    for k in ['time','qpos','qvel','normal','skin_normal','palm','ctrl','alpha','lift','wrist_position','actual_motion_target','executed_motion_reference','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm','warning_count']:equal[k]=np.array_equal(z[k],a[k])
                check(key+'/representative_traditional_exact_V5_raw',all(equal.values()));reproductions.append(dict(identity=[scene,method,seed],source_sha256=sha(source),equal=equal))
            limits=np.asarray(m['limits']);dt=z['substep_duration'];force=z['substep_skin_normal'];times=z['substep_time']+dt;after=times>12.+1e-9;lf=np.flatnonzero(after)[np.argmax(force[after,4])];breach=(force>limits+.02).any(1)|(force.sum(1)>m['budget']+.02)
            records.append(dict(dataset=dataset,scenario=scene,method=method,seed=seed,trace_sha256=sha(path),independent_outcome=own,full_motion_mission=r['full_motion_mission_success'],wrist_failure_at=wrist,qualities=qualities,release_LF_peak_N=float(force[lf,4]),release_LF_peak_time_end_s=float(times[lf]),whole_guard_breach_duration_s=float(dt[breach].sum()),release_LF_guard_breach_duration_s=float(dt[after&(force[:,4]>limits[4]+.02)].sum()),holding_excess_Ns=float(np.sum(np.maximum(force[(times>7.5+1e-9)&(times<=12.+1e-9)]-limits,0)*dt[(times>7.5+1e-9)&(times<=12.+1e-9),None]))))
            print(scene,method,seed,'representative_raw_verified',flush=True)
    result=dict(schema='independent_v6_1_stratified_raw_audit_v1',passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],created_utc=datetime.now(timezone.utc).isoformat(),records=records,traditional_source_reproduction=reproductions,reviewer_sha256=sha(__file__),physics_steps_executed=0,scope='16 predeclared representative raw records:8 from12v2 and8 from96, including4 traditional complete-trajectory matches. Full108 manifest/result/paired denominator audit is separate. No exhaustive108 raw or noise reconstruction is claimed.')
    out=ROOT/'review'/('independent_v6_1_representative_raw_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],path=str(out),sha256=sha(out),sampled=len(records)),ensure_ascii=False))
if __name__=='__main__':main()

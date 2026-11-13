"""Independent saved36 V5 audit. No plant/controller/dynamics import."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import hashlib,json,runpy,sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'output/development_v5_pilot';SNAPSHOT=FOLDER/'sources'
old=runpy.run_path(str(ROOT/'review/independent_development_v1_review.py'))
equivalent=old['equivalent'];physical_outcome=old['physical_outcome'];dwell=old['dwell']
noise_path=ROOT.parent/'五指策略接入_2026-11-03/review/independent_development_review.py'
audit_rng=runpy.run_path(str(noise_path))['audit_rng']
postclear=runpy.run_path(str(ROOT/'review/verify_lifecycle.py'))['postclear_recontact']
tracking=runpy.run_path(str(ROOT/'review/v5_tracking_diagnostics.py'))['summarize_tracking']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b):return bool(np.array_equal(a,b)) if np.asarray(a).dtype.kind in 'UOS' else bool(np.array_equal(a,b,equal_nan=True))
def variation(t,f):
    tv=abs(np.diff(f,axis=0)).sum(0);net=abs(f[-1]-f[0]);seconds=float(t[-1]-t[0])
    return dict(finger_TV_rate_N_s=float(tv.sum()/seconds),finger_net_drift_rate_N_s=float(net.sum()/seconds),finger_reversal_rate_N_s=float((tv-net).sum()/(2*seconds)))
def band_wave(z,m):
    dt=float(m['physics_timestep']);half=round(.025/dt);tk=np.arange(-half,half+1)*dt
    k=100*dt*np.sinc(100*tk)*np.hanning(2*half+1);k/=k.sum();f=np.vstack([z['normal'][0],z['substep_normal']]);grid=np.r_[0.,z['substep_time']+z['substep_duration']]
    return np.column_stack([np.interp(z['time'],grid,np.convolve(np.pad(f[:,j],half,mode='reflect'),k,mode='valid')) for j in range(5)])

def main():
    manifest=read(FOLDER/'manifest.json');finish=read(FOLDER/'execution_complete.json');rows=read(FOLDER/'trials.json');protocol=read(SNAPSHOT/'protocol.json');api=runpy.run_path(str(SNAPSHOT/'evaluation.py'))['evaluate_research']
    checks={};records=[];common=defaultdict(dict);release_pairs=defaultdict(dict);method_names=sorted({x['method'] for x in rows});scene_names=sorted({x['scenario'] for x in rows});seed_values={x['seed'] for x in rows}
    def check(k,v):checks[k]=bool(v)
    check('development_only',manifest['phase']=='development' and manifest['performance_claim_allowed'] is False and finish['performance_claim_allowed'] is False)
    check('all36_retained_complete_no_runtime_failure',finish['execution_complete'] and finish['trials']==finish['planned']==len(rows)==36 and not finish['runtime_failures'])
    check('trials_identity',sha(FOLDER/'trials.json')==finish['trials_sha256'])
    check('sources_frozen_at_execution',finish['sources_unchanged'] is True)
    check('all18_source_snapshots_match',all((SNAPSHOT/k).is_file() and sha(SNAPSHOT/k)==v for k,v in manifest['research_source_sha256'].items()))
    check('one_seed_development_not_physical_holdout',seed_values=={610201} and manifest['physical_randomization'] is False)
    jobs={(x['scenario'],x['method'],x['seed']):x for x in manifest['jobs']};check('exact6_by6_assigned_matrix',len(method_names)==len(scene_names)==6 and len(jobs)==36 and set(jobs)=={(s,m,610201) for s in scene_names for m in method_names} and {(x['scenario'],x['method'],x['seed']) for x in rows}==set(jobs))
    for row in rows:
        key=(row['scenario'],row['method'],row['seed']);folder=Path(jobs[key]['destination']);path=folder/'trace.npz';meta=read(folder/'metadata.json')
        with np.load(path,allow_pickle=False) as a:z={k:a[k] for k in a.files}
        local={}
        def lc(k,v):local[k]=bool(v);check(str(key)+'/'+k,v)
        lc('raw_result_sidecar_hash_identity',sha(path)==row['trace_sha256']==meta['trace_sha256'] and read(folder/'result.json')==row)
        lc('inline_sidecar_only_postsave_hash_diff',json.loads(str(z['metadata']))=={k:v for k,v in meta.items() if k!='trace_sha256'})
        lc('executed_source_identity_matches_snapshot',all(manifest['research_source_sha256'].get(k)==v for k,v in meta['research_source_sha256'].items()))
        dt=float(meta['physics_timestep']);t=z['time'];st=z['substep_time'];dur=z['substep_duration'];lc('full_control_and_substep_clock',len(t)==3201 and np.allclose(t,np.arange(3201)*.005,atol=1e-8,rtol=0) and len(st)==round(16/dt) and np.allclose(st,np.arange(len(st))*dt,atol=1e-7,rtol=0) and np.allclose(dur,dt,atol=1e-12,rtol=0) and abs(float(st[-1]+dur[-1])-16.)<1e-8)
        lc('all_actual_states_and_actions_finite',all(np.isfinite(z[k]).all() for k in ['qpos','qvel','normal','skin_normal','ctrl','executed_action','substep_normal','substep_skin_normal']))
        r=row['research_metrics'];lc('all_frozen_v2_metrics_recalculated',equivalent(api(path,SNAPSHOT/'protocol.json'),r))
        own=physical_outcome(z,meta);lc('independent_legacy_mission_and_physical_release',own['task_success']==r['mission_success'] and own['physical_release']==r['release']['physical_success'] and own['physical_exposure']==r['release']['physical_exposure'] and equivalent(own['physical_release_latency'],r['release']['physical_clearance_latency_s']))
        lc('independent_raw16s_force_guard_and_integrals',own['force_guard']==r['safety']['guard_pass'] and abs(own['local_excess']-r['safety']['local_excess_integral_Ns'])<1e-8 and abs(own['total_excess']-r['safety']['total_excess_integral_Ns'])<1e-8)
        wrist_error=abs(z['wrist_position']-z['requested_wrist_reference']);wfail=dwell(t,(wrist_error>np.array([.015,.015])).any(1),7.5,12.,.3,True)
        full=bool(own['task_success'] and wfail is None);lc('independent_full_motion_mission_and_wrist_guard',full==r['full_motion_mission_success'] and equivalent(wfail,r['wrist_tracking_guard']['failure_at']) and r['wrist_tracking_guard']['available'])
        qadr=meta['raw_joint_qpos_indices'];jn=meta['raw_joint_names'];wi=[qadr[jn.index(n)] for n in ['rh_WRJ2','rh_WRJ1']]
        lc('wrist_encoders_are_actual_qpos',np.array_equal(z['wrist_position'],z['qpos'][:,wi]))
        trk={}
        for name,lo,hi in [('holding_motion_tracking',7.5,12.),('dynamic_motion_tracking',9.3,11.7)]:
            q=tracking(z,(lo,hi));saved=r['progress'][name];trk[name]=q
            lc('independent_'+name+'_fixed_left_intervals',q['complete_window'] and saved['available'] and saved['complete_window'] and not saved['censored'] and np.allclose(q['nominal_tracking_rmse'],saved['rmse'],atol=1e-10,rtol=0) and np.allclose(q['nominal_tracking_IAE'],saved['absolute_error_integral'],atol=1e-10,rtol=0) and np.allclose(q['nominal_tracking_maximum_error'],saved['max_abs_error'],atol=1e-10,rtol=0))
        wave=band_wave(z,meta);qualities={}
        for name,lo,hi in [('static_quality',8.2,8.9),('dynamic_quality',9.3,11.7),('entire_holding_quality',7.5,12.)]:
            w=(t>=lo-1e-9)&(t<=hi+1e-9);saved=r[name]
            if saved['eligible']:
                b=variation(t[w],wave[w]);raw=variation(t[w],z['normal'][w]);qualities[name]=dict(bandlimited=b,raw_200Hz=raw)
                lc('independent_'+name+'_TV_net_retrace',all(abs(v-saved['bandlimited'][k])<1e-8 for k,v in b.items()) and all(abs(v-saved['raw_200Hz'][k])<1e-8 for k,v in raw.items()))
        obs=[json.loads(str(v)) for v in z['observation_json']];noise=audit_rng(z,meta,obs);lc('all3201_observation_noise_packets_recomputed',noise['passed'])
        tasks=[json.loads(str(v)) for v in z['task_request_json']];previous_motion=np.vstack([np.zeros(3),z['executed_motion_reference'][:-1]])
        lc('public_task_target_and_preaction_actual_history',len(tasks)==3201 and all(np.allclose(q['motion_target'],z['actual_motion_target'][i],atol=1e-12,rtol=0) and np.allclose(q['executed_motion_reference'],previous_motion[i],atol=1e-12,rtol=0) and np.allclose(q['motion_error'],z['actual_motion_target'][i]-previous_motion[i],atol=1e-12,rtol=0) and q['object_state_available'] is False and q['target_contact_identity_available'] is False for i,q in enumerate(tasks)))
        lc('unavailable_force_predictions_not_falsely_timestamped',np.isnan(z['predicted_normal'][~z['prediction_available']]).all() and np.isnan(z['prediction_at'][~z['prediction_available']]).all())
        lc('available_force_predictions_finite',np.isfinite(z['predicted_normal'][z['prediction_available']]).all())
        release=t>=12.-1e-9;lc('release_prediction_unavailable',not z['prediction_available'][release].any() and np.isnan(z['predicted_normal'][release]).all())
        prefix=t<=7.5+1e-9;common[row['scenario']][row['method']]=dict(arrays={k:z[k][prefix].copy() for k in ['qpos','qvel','ctrl_before','alpha_before','normal','skin_normal','observation_json','raw_patches_json','task_request_json']},config=meta['config'],noise=meta['noise'],rng=noise['reconstructed_draw_stream_sha256'])
        if row['method'] in ['ratio','ratio_release','joint','joint_release']:
            pre=t<=12.+1e-9;rawpre=st+dur<=12.+1e-9;release_pairs[row['scenario']][row['method']]=dict(arrays={k:z[k][pre].copy() for k in ['qpos','qvel','ctrl_before','alpha_before','normal','skin_normal']},raw=z['substep_normal'][rawpre].copy(),raw_skin=z['substep_skin_normal'][rawpre].copy(),raw_whole_tv=r['entire_holding_quality']['raw_200Hz']['finger_TV_rate_N_s'] if r['entire_holding_quality']['eligible'] else None,band_whole_tv=r['entire_holding_quality']['bandlimited']['finger_TV_rate_N_s'] if r['entire_holding_quality']['eligible'] else None)
        holds=(st>=7.5-1e-9)&(st<12.-1e-9);limits=np.asarray(meta['limits']);holding_excess=float(np.sum(np.maximum(z['substep_skin_normal'][holds]-limits,0)*dur[holds,None]));lc('holding_stage_raw_excess',abs(holding_excess-r['safety']['stages']['policy_holding_transport']['local_excess_integral_Ns'])<1e-8)
        records.append(dict(identity=key,trace_sha256=sha(path),checks=local,independent_outcome=own,full_motion_success=full,wrist_failure_at=wfail,tracking=trk,qualities=qualities,holding_excess_Ns=holding_excess,whole_finger_peak_N=r['safety']['finger_peak_N'],postclear=postclear(z,meta),rng_sha256=noise['reconstructed_draw_stream_sha256'],model_invalid_ticks=int(z['model_invalid'].sum()),observation_unavailable_ticks=int(z['observation_unavailable'].sum()),prediction_available_ticks=int(z['prediction_available'].sum())))
        print(key,'reviewed',all(local.values()),flush=True)
    pair_evidence=[]
    for scene,group in common.items():
        ref=group['ratio']
        for method in method_names:
            x=group[method];check(scene+'/'+method+'/all1501_common_preparation_values_equal',all(eq(v,x['arrays'][k]) for k,v in ref['arrays'].items()));check(scene+'/'+method+'/same_physical_noise_and_entire_rng_stream',x['config']==ref['config'] and x['noise']==ref['noise'] and x['rng']==ref['rng'])
        for method in ['ratio','joint']:
            a,b=release_pairs[scene][method],release_pairs[scene][method+'_release'];raw_equal=all(eq(v,b['arrays'][k]) for k,v in a['arrays'].items()) and eq(a['raw'],b['raw']) and eq(a['raw_skin'],b['raw_skin'])
            check(scene+'/'+method+'/release_sibling_same_pre12_truth_and_raw',raw_equal);check(scene+'/'+method+'/raw_wholehold_release_invariant',equivalent(a['raw_whole_tv'],b['raw_whole_tv']))
            pair_evidence.append(dict(scene=scene,holding_method=method,pre12_raw_and_state_equal=raw_equal,raw_wholeholding_TV=a['raw_whole_tv'],band_wholeholding_original=a['band_whole_tv'],band_wholeholding_new_release=b['band_whole_tv'],interpretation='V2 whole-holding symmetric FIR can contain future release samples. Original values retained; do not attribute sibling band differences to holding.'))
    summaries=[]
    for method in method_names:
        selected=[r for r in rows if r['method']==method];rr=[r for r in records if r['identity'][1]==method]
        summaries.append(dict(method=method,assigned=len(rr),legacy_missions=sum(r['research_metrics']['mission_success'] for r in selected),full_motion_missions=sum(r['full_motion_success'] for r in rr),whole_guard_passes=sum(r['independent_outcome']['force_guard'] for r in rr),physical_releases=sum(r['independent_outcome']['physical_release'] for r in rr),mean_release_latency_s=float(np.mean([r['independent_outcome']['physical_release_latency'] for r in rr if r['independent_outcome']['physical_release_latency'] is not None])),holding_excess_total_Ns=sum(r['holding_excess_Ns'] for r in rr),
            static_TV_N_s=float(np.mean([r['qualities']['static_quality']['bandlimited']['finger_TV_rate_N_s'] for r in rr if 'static_quality'in r['qualities']])),dynamic_TV_N_s=float(np.mean([r['qualities']['dynamic_quality']['bandlimited']['finger_TV_rate_N_s'] for r in rr if 'dynamic_quality'in r['qualities']])),raw_wholeholding_TV_N_s=float(np.mean([r['qualities']['entire_holding_quality']['raw_200Hz']['finger_TV_rate_N_s'] for r in rr if 'entire_holding_quality'in r['qualities']]))))
    result=dict(schema='independent_v5_pilot36_saved_data_audit_v1',passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],created_utc=datetime.now(timezone.utc).isoformat(),trials_sha256=sha(FOLDER/'trials.json'),manifest_sha256=sha(FOLDER/'manifest.json'),execution_complete_sha256=sha(FOLDER/'execution_complete.json'),snapshot_sources=manifest['research_source_sha256'],records=records,release_phase_pairing=pair_evidence,summary=summaries,reviewer_sha256=sha(__file__),noise_auditor_sha256=sha(noise_path),physics_steps_executed=0,
        limitations=['One development seed610201, six scenarios and no physical randomization; not confirmation.',
            'Full-motion wrist guard is a new V2 engineering endpoint; legacy mission remains separately verified.',
            'V2 whole-holding bandlimited TV uses post-release samples near12s; preserve it and use the separately saved raw same-phase metric for holding sibling comparison.',
            'Whole16s force guard includes all original substeps and final endpoint; a lower holding TV never compensates a worse whole-task force pass rate.',
            'No production dependency, model, protocol, existing report or result was modified. Root owns the concurrently running160 development trials and phase3 calibration.'])
    out=ROOT/'review'/('independent_development_v5_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],path=str(out),sha256=sha(out),summary=summaries),ensure_ascii=False))
if __name__=='__main__':main()

"""Independent manifest-driven V5 randomized broad saved-data audit. No plant/controller/dynamics import."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import ast,hashlib,json,runpy,sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'output/development_v5_physical_all16';SNAPSHOT=FOLDER/'sources'
old=runpy.run_path(str(ROOT/'review/independent_development_v1_review.py'))
equivalent=old['equivalent'];dwell=old['dwell']
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


def array_identity(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def physical_outcome(z,m):
    t=z['time'];n=z['normal'];support=(n[:,0]>.2)&((n>.2).sum(1)>=3)
    raised=z['position'][:,2]>=m['initial_position'][2]+.025
    relative=m.get('reference_relative');error=np.full(len(t),np.inf) if relative is None else np.linalg.norm(z['relative']-relative,axis=1)
    admitted=bool(m.get('handoff_passed'));release=m.get('release_at');end=12. if release is None else float(release)
    established=dwell(t,support&raised&(abs(z['lift']-z['lift_ref'])<=.01)&(z['support']<=.05),0.,7.5,.1)
    failure=None;failure_at=None
    if not admitted or established is None:failure=m.get('preparation_failure') or 'preparation_or_handoff_failed';failure_at=7.5
    else:
        drop=dwell(t,~support|~raised|(error>.08),7.5,end,.1,True)
        tracking=dwell(t,abs(z['lift']-z['lift_ref'])>.01,7.5,end,.3,True)
        if drop is not None and (tracking is None or drop<=tracking):failure='support_or_object_lost';failure_at=drop
        elif tracking is not None:failure='transport_reference_unmet';failure_at=tracking
    exposure=False;clear=None
    if release is not None:
        i=np.argmin(abs(t-release));exposure=bool(support[i] and raised[i] and z['support'][i]<=.05 and z['all_hand_object'][i]>.2)
        clear=dwell(t,z['all_hand_object']<=.2,release,min(16.,release+3.5),.2)
    physical=bool(exposure and clear is not None);qualified=bool(exposure and admitted and (failure_at is None or failure_at>end))
    task=bool(admitted and failure is None and physical and qualified and z['warning_count'].max()==0)
    if not physical and failure is None:failure='physical_release_failed_or_not_exposed'
    skin=z['substep_skin_normal'];limits=np.asarray(m['limits']);budget=m['budget'];dur=z['substep_duration']
    guard=bool((skin<=limits+.02).all() and (skin.sum(1)<=budget+.02).all() and (z['skin_normal'][-1]<=limits+.02).all() and z['skin_normal'][-1].sum()<=budget+.02)
    return dict(physical_exposure=exposure,physical_release=physical,task_success=task,task_qualified_exposure=qualified,
        task_qualified_release=bool(physical and qualified),physical_release_latency=None if not physical else float(clear-release),
        failure_at=failure_at,failure_reason=failure,force_guard=guard,local_excess=float(np.sum(np.maximum(skin-limits,0)*dur[:,None])),total_excess=float(np.maximum(skin.sum(1)-budget,0)@dur))

def scene_definitions(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='SCENARIOS' for y in x.targets))
    # Evaluate only the literal list and dict calls, never import session/plant.
    assert all(isinstance(x.func,ast.Name) and x.func.id=='dict' for x in ast.walk(node.value) if isinstance(x,ast.Call))
    return {v['name']:v for v in eval(compile(ast.Expression(node.value),'scene_literals','eval'),{'__builtins__':{},'dict':dict})}

def main():
    manifest=read(FOLDER/'manifest.json');finish=read(FOLDER/'execution_complete.json');rows=read(FOLDER/'trials.json');protocol=read(SNAPSHOT/'protocol.json');api=runpy.run_path(str(SNAPSHOT/'evaluation.py'))['evaluate_arrays'];foundation=Path(manifest['foundation']);scenes=scene_definitions(foundation/'session.py');report=read(ROOT/'report/development_v5_physical_all16/readout.json')
    checks={};records=[];common=defaultdict(dict);release_pairs=defaultdict(dict);method_names=sorted({x['method'] for x in rows});scene_names=sorted({x['scenario'] for x in rows});seed_values={x['seed'] for x in rows}
    def check(k,v):checks[k]=bool(v)
    check('development_only',manifest['phase']=='development' and manifest['performance_claim_allowed'] is False and finish['performance_claim_allowed'] is False)
    check('all_manifest_jobs_retained_complete_no_runtime_failure',finish['execution_complete'] and finish['trials']==finish['planned']==len(rows)==len(manifest['jobs']) and not finish['runtime_failures'])
    check('trials_identity',sha(FOLDER/'trials.json')==finish['trials_sha256'])
    check('sources_frozen_at_execution',finish['sources_unchanged'] is True)
    check('all18_source_snapshots_match',all((SNAPSHOT/k).is_file() and sha(SNAPSHOT/k)==v for k,v in manifest['research_source_sha256'].items()))
    check('development_actual_physical_randomization',manifest['physical_randomization'] is True and all(j['split']=='robustness' for j in manifest['jobs']))
    jobs={(x['scenario'],x['method'],x['seed']):x for x in manifest['jobs']};check('exact_manifest_assigned_matrix',len(jobs)==len(manifest['jobs'])==len(rows) and set(jobs)=={(s,m,seed) for s in scene_names for m in method_names for seed in seed_values} and {(x['scenario'],x['method'],x['seed']) for x in rows}==set(jobs))
    for row in rows:
        key=(row['scenario'],row['method'],row['seed']);folder=Path(jobs[key]['destination']);path=folder/'trace.npz';meta=read(folder/'metadata.json')
        with np.load(path,allow_pickle=False) as a:z={k:a[k] for k in a.files}
        local={}
        def lc(k,v):local[k]=bool(v);check(str(key)+'/'+k,v)
        lc('raw_result_sidecar_hash_identity',sha(path)==row['trace_sha256']==meta['trace_sha256'] and read(folder/'result.json')==row)
        lc('inline_sidecar_only_postsave_hash_diff',json.loads(str(z['metadata']))=={k:v for k,v in meta.items() if k!='trace_sha256'})
        lc('executed_source_identity_matches_snapshot',all(manifest['research_source_sha256'].get(k)==v for k,v in meta['research_source_sha256'].items()))
        base=scenes[row['scenario']];u=np.random.default_rng(int(row['seed'])+511000).normal(size=4);expected_mass=base.get('mass',.12)*np.clip(1.+.06*u[0],.85,1.15);expected_friction=base.get('friction',.60)*np.clip(1.+.08*u[1],.80,1.20);lc('actual_physical_draw_reconstructed',abs(meta['config']['mass']-expected_mass)<1e-14 and abs(meta['config']['friction']-expected_friction)<1e-14 and meta['config']['shape']==base['shape'] and abs(meta['config']['size']-base.get('size',1.))<1e-14 and abs(meta['config']['solref']-base.get('solref',.008))<1e-14 and meta['split']=='robustness');dt=float(meta['physics_timestep']);t=z['time'];st=z['substep_time'];dur=z['substep_duration'];lc('full_control_and_substep_clock',len(t)==3201 and np.allclose(t,np.arange(3201)*.005,atol=1e-8,rtol=0) and len(st)==round(16/dt) and np.allclose(st,np.arange(len(st))*dt,atol=1e-7,rtol=0) and np.allclose(dur,dt,atol=1e-12,rtol=0) and abs(float(st[-1]+dur[-1])-16.)<1e-8)
        lc('all_actual_states_and_actions_finite',all(np.isfinite(z[k]).all() for k in ['qpos','qvel','normal','skin_normal','ctrl','executed_action','substep_normal','substep_skin_normal']))
        r=row['research_metrics'];recomputed=api(z,meta,protocol);recomputed.update(trace_sha256=sha(path),protocol_sha256=sha(SNAPSHOT/'protocol.json'),evaluation_source_sha256=sha(SNAPSHOT/'evaluation.py'));lc('all_frozen_v2_metrics_recalculated',equivalent(recomputed,r))
        own=physical_outcome(z,meta);lc('independent_legacy_mission_and_physical_release',own['task_success']==r['mission_success'] and own['failure_reason']==r['failure_reason'] and equivalent(own['failure_at'],r['failure_at']) and own['task_qualified_exposure']==r['release']['task_qualified_exposure'] and own['task_qualified_release']==r['release']['task_qualified_success'] and own['physical_release']==r['release']['physical_success'] and own['physical_exposure']==r['release']['physical_exposure'] and equivalent(own['physical_release_latency'],r['release']['physical_clearance_latency_s']))
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
            w=(t>=lo-1e-9)&(t<=hi+1e-9);saved=r[name];n=z['normal'];support=(n[:,0]>.2)&((n>.2).sum(1)>=3);raised=z['position'][:,2]>=meta['initial_position'][2]+.025;eligible=bool(w.sum()==round((hi-lo)/.005)+1 and np.all(support[w]&raised[w]&(z['support'][w]<=.05)&z['controller_active'][w].astype(bool)));lc('independent_'+name+'_eligibility',eligible==saved['eligible'] and (eligible or saved['bandlimited'] is None and saved['raw_200Hz'] is None))
            if saved['eligible']:
                b=variation(t[w],wave[w]);raw=variation(t[w],z['normal'][w]);qualities[name]=dict(bandlimited=b,raw_200Hz=raw)
                lc('independent_'+name+'_TV_net_retrace',all(abs(v-saved['bandlimited'][k])<1e-8 for k,v in b.items()) and all(abs(v-saved['raw_200Hz'][k])<1e-8 for k,v in raw.items()))
        obs=[json.loads(str(v)) for v in z['observation_json']];noise=audit_rng(z,meta,obs);lc('all3201_observation_noise_packets_recomputed',noise['passed'])
        tasks=[json.loads(str(v)) for v in z['task_request_json']];previous_motion=np.vstack([np.zeros(3),z['executed_motion_reference'][:-1]])
        lc('public_task_target_and_preaction_actual_history',len(tasks)==3201 and all(np.allclose(q['motion_target'],z['actual_motion_target'][i],atol=1e-12,rtol=0) and np.allclose(q['executed_motion_reference'],previous_motion[i],atol=1e-12,rtol=0) and np.allclose(q['motion_error'],z['actual_motion_target'][i]-previous_motion[i],atol=1e-12,rtol=0) and q['object_state_available'] is False and q['target_contact_identity_available'] is False for i,q in enumerate(tasks)))
        lc('unavailable_force_predictions_not_falsely_timestamped',np.isnan(z['prediction_at'][~z['prediction_available']]).all())
        if row['method']=='passive':
            held_reference=(~z['prediction_available'])&np.isfinite(z['predicted_normal']).all(1)
            lc('passive_retained_reference_explicitly_not_forecast',all(json.loads(str(z['debug_json'][i])).get('passive_hold') is True and json.loads(str(z['debug_json'][i])).get('prediction_available') is False for i in np.flatnonzero(held_reference)))
        else:
            lc('autonomous_unavailable_force_predictions_are_NaN',np.isnan(z['predicted_normal'][~z['prediction_available']]).all())
        lc('available_force_predictions_finite',np.isfinite(z['predicted_normal'][z['prediction_available']]).all())
        release=t>=12.-1e-9;lc('release_prediction_unavailable',not z['prediction_available'][release].any() and np.isnan(z['predicted_normal'][release]).all())
        prefix=t<=7.5+1e-9;common[(row['scenario'],row['seed'])][row['method']]=dict(arrays={k:array_identity(z[k][prefix]) for k in ['qpos','qvel','ctrl_before','alpha_before','normal','skin_normal','observation_json','raw_patches_json','task_request_json']},config=meta['config'],noise=meta['noise'],rng=noise['reconstructed_draw_stream_sha256'])
        if row['method'] in ['ratio','ratio_release','joint','joint_release']:
            pre=t<=12.+1e-9;rawpre=st+dur<=12.+1e-9;release_pairs[(row['scenario'],row['seed'])][row['method']]=dict(arrays={k:array_identity(z[k][pre]) for k in ['qpos','qvel','ctrl_before','alpha_before','normal','skin_normal']},raw=array_identity(z['substep_normal'][rawpre]),raw_skin=array_identity(z['substep_skin_normal'][rawpre]),raw_whole_tv=r['entire_holding_quality']['raw_200Hz']['finger_TV_rate_N_s'] if r['entire_holding_quality']['eligible'] else None,band_whole_tv=r['entire_holding_quality']['bandlimited']['finger_TV_rate_N_s'] if r['entire_holding_quality']['eligible'] else None)
        mechanical=meta.get('mechanical_release_at');end=16. if mechanical is None else mechanical;holds=(st>=7.5-1e-9)&(st<end-1e-9);limits=np.asarray(meta['limits']);holding_excess=float(np.sum(np.maximum(z['substep_skin_normal'][holds]-limits,0)*dur[holds,None]));stage='policy_holding_transport' if meta['handoff_passed'] else 'post_preparation_without_policy';stage_saved=r['safety']['stages'].get(stage);lc('holding_stage_raw_excess',stage_saved is not None and abs(holding_excess-stage_saved['local_excess_integral_Ns'])<1e-8 if holds.any() else stage_saved is None)
        records.append(dict(identity=key,prepared=meta['handoff_passed'],physical_config=meta['config'],trace_sha256=sha(path),checks=local,independent_outcome=own,full_motion_success=full,wrist_failure_at=wfail,tracking=trk,qualities=qualities,holding_excess_Ns=holding_excess,whole_finger_peak_N=r['safety']['finger_peak_N'],postclear=postclear(z,meta),rng_sha256=noise['reconstructed_draw_stream_sha256'],model_invalid_ticks=int(z['model_invalid'].sum()),observation_unavailable_ticks=int(z['observation_unavailable'].sum()),prediction_available_ticks=int(z['prediction_available'].sum())))
        print(key,'reviewed',all(local.values()),flush=True)
    pair_evidence=[]
    for scene,group in common.items():
        ref=group['ratio']
        for method in method_names:
            x=group[method];check(str(scene)+'/'+method+'/all1501_common_preparation_values_equal',all(eq(v,x['arrays'][k]) for k,v in ref['arrays'].items()));check(str(scene)+'/'+method+'/same_physical_noise_and_entire_rng_stream',x['config']==ref['config'] and x['noise']==ref['noise'] and x['rng']==ref['rng'])
        for method in ['ratio','joint']:
            a,b=release_pairs[scene][method],release_pairs[scene][method+'_release'];raw_equal=all(eq(v,b['arrays'][k]) for k,v in a['arrays'].items()) and eq(a['raw'],b['raw']) and eq(a['raw_skin'],b['raw_skin'])
            check(str(scene)+'/'+method+'/release_sibling_same_pre12_truth_and_raw',raw_equal);check(str(scene)+'/'+method+'/raw_wholehold_release_invariant',equivalent(a['raw_whole_tv'],b['raw_whole_tv']))
            pair_evidence.append(dict(scene=scene,holding_method=method,pre12_raw_and_state_equal=raw_equal,raw_wholeholding_TV=a['raw_whole_tv'],band_wholeholding_original=a['band_whole_tv'],band_wholeholding_new_release=b['band_whole_tv'],interpretation='V2 whole-holding symmetric FIR can contain future release samples. Original values retained; do not attribute sibling band differences to holding.'))
    summaries=[]
    for method in method_names:
        selected=[r for r in rows if r['method']==method];rr=[r for r in records if r['identity'][1]==method]
        summaries.append(dict(method=method,assigned=len(rr),legacy_missions=sum(r['research_metrics']['mission_success'] for r in selected),full_motion_missions=sum(r['full_motion_success'] for r in rr),whole_guard_passes=sum(r['independent_outcome']['force_guard'] for r in rr),physical_releases=sum(r['independent_outcome']['physical_release'] for r in rr),mean_release_latency_s=float(np.mean([r['independent_outcome']['physical_release_latency'] for r in rr if r['independent_outcome']['physical_release_latency'] is not None])),holding_excess_total_Ns=sum(r['holding_excess_Ns'] for r in rr),
            static_TV_N_s=float(np.mean([r['qualities']['static_quality']['bandlimited']['finger_TV_rate_N_s'] for r in rr if 'static_quality'in r['qualities']])),dynamic_TV_N_s=float(np.mean([r['qualities']['dynamic_quality']['bandlimited']['finger_TV_rate_N_s'] for r in rr if 'dynamic_quality'in r['qualities']])),raw_wholeholding_TV_N_s=float(np.mean([r['qualities']['entire_holding_quality']['raw_200Hz']['finger_TV_rate_N_s'] for r in rr if 'entire_holding_quality'in r['qualities']]))))
    paired_quality=[]
    indexed={(x['scenario'],x['seed'],x['method']):x for x in rows}
    for pair in report['paired_quality']:
        base=pair['baseline'];candidate=pair['candidate'];window=pair['window'];both=[];missing=[]
        for scene in scene_names:
            for seed in sorted(seed_values):
                a=indexed[(scene,seed,base)]['research_metrics'][window];b=indexed[(scene,seed,candidate)]['research_metrics'][window]
                if a['eligible'] and b['eligible']:both.append((a['bandlimited'],b['bandlimited']))
                else:missing.append([scene,seed])
        checks['report_common_pairs/'+candidate+'/'+window]=pair['assigned_pairs']==len(scene_names)*len(seed_values) and pair['common_available_pairs']==len(both) and sorted(pair['unavailable_pairs'])==sorted(missing)
        for metric,stats in pair['contrasts'].items():
            a=np.array([x[0][metric] for x in both]);b=np.array([x[1][metric] for x in both]);actual=dict(baseline_mean=float(a.mean()),candidate_mean=float(b.mean()),mean_paired_difference=float((b-a).mean()))
            checks['report_common_contrast/'+candidate+'/'+window+'/'+metric]=equivalent(actual,stats)
        paired_quality.append(pair)
    for sr in summaries:
        reportrow=next(x for x in report['methods'] if x['method']==sr['method'])
        check('report/'+sr['method']+'/assigned_mission_full_force',reportrow['assigned']==sr['assigned'] and reportrow['legacy_missions']==sr['legacy_missions'] and reportrow['full_motion_missions']==sr['full_motion_missions'] and reportrow['raw_force_guard_pass']==sr['whole_guard_passes'])
    failure_counts={m:{reason:sum(x['research_metrics']['failure_reason']==reason for x in rows if x['method']==m) for reason in sorted({x['research_metrics']['failure_reason'] for x in rows if x['method']==m and x['research_metrics']['failure_reason']})} for m in method_names}
    result=dict(schema='independent_v5_broad_manifest_saved_data_audit_v1',passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],created_utc=datetime.now(timezone.utc).isoformat(),trials_sha256=sha(FOLDER/'trials.json'),manifest_sha256=sha(FOLDER/'manifest.json'),execution_complete_sha256=sha(FOLDER/'execution_complete.json'),snapshot_sources=manifest['research_source_sha256'],records=records,release_phase_pairing=pair_evidence,summary=summaries,failure_counts=failure_counts,paired_quality=paired_quality,report_sha256=sha(ROOT/'report/development_v5_physical_all16/readout.json'),foundation_scene_source_sha256=sha(foundation/'session.py'),reviewer_sha256=sha(__file__),noise_auditor_sha256=sha(noise_path),physics_steps_executed=0,
        limitations=['Two physical seed draws shared across sixteen scenes, five methods; development only, not 32 independent physical replications or confirmation.',
            'Full-motion wrist guard is a new V2 engineering endpoint; legacy mission remains separately verified.',
            'V2 whole-holding bandlimited TV uses post-release samples near12s; preserve it and use the separately saved raw same-phase metric for holding sibling comparison.',
            'Whole16s force guard includes all original substeps and final endpoint; a lower holding TV never compensates a worse whole-task force pass rate.',
            'No production dependency, model, protocol, existing report or result was modified. No dynamics or new trials are executed by this audit.'])
    out=ROOT/'review'/('independent_development_v5_broad_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],path=str(out),sha256=sha(out),summary=summaries),ensure_ascii=False))
if __name__=='__main__':main()

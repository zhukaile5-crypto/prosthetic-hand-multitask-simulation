"""Independent saved30 audit. Pure data/evaluator replay, no simulator import."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import hashlib,json,runpy,sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'output/development_v4_integrated';SNAPSHOT=FOLDER/'sources'
old=runpy.run_path(str(ROOT/'review/independent_development_v1_review.py'))
physical_outcome=old['physical_outcome'];equivalent=old['equivalent'];dwell=old['dwell']
noise_helper=ROOT.parent/'五指策略接入_2026-11-03/review/independent_development_review.py'
audit_rng=runpy.run_path(str(noise_helper))['audit_rng']
recontact=runpy.run_path(str(ROOT/'review/verify_lifecycle.py'))['postclear_recontact']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def equal(a,b):return bool(np.array_equal(a,b,equal_nan=True)) if np.asarray(a).dtype.kind not in 'UOS' else bool(np.array_equal(a,b))
def quality(z,m,start,end):
    dt=float(m['physics_timestep']);half=round(.025/dt);kt=np.arange(-half,half+1)*dt
    kernel=100*dt*np.sinc(100*kt)*np.hanning(2*half+1);kernel/=kernel.sum()
    # Reproduce the immutable current metric's interval-start alignment.
    # Its recorded forces are interval-end values; that fixed half-ms shift is
    # disclosed rather than silently changing historical development metrics.
    values=np.vstack([z['substep_normal'],z['normal'][-1]]);grid=np.r_[z['substep_time'],16.]
    filtered=np.column_stack([np.interp(z['time'],grid,np.convolve(np.pad(values[:,j],half,mode='reflect'),kernel,mode='valid')) for j in range(5)])
    mask=(z['time']>=start-1e-9)&(z['time']<=end+1e-9);f=filtered[mask];seconds=z['time'][mask][-1]-z['time'][mask][0]
    tv=float(abs(np.diff(f,axis=0)).sum());drift=float(abs(f[-1]-f[0]).sum())
    return dict(finger_TV_rate_N_s=tv/seconds,finger_net_drift_rate_N_s=drift/seconds,finger_reversal_rate_N_s=(tv-drift)/(2*seconds),total_force_TV_rate_N_s=float(abs(np.diff(f.sum(1))).sum()/seconds))
def main():
    manifest=read(FOLDER/'manifest.json');finish=read(FOLDER/'execution_complete.json');trials=read(FOLDER/'trials.json');protocol=read(SNAPSHOT/'protocol.json')
    evaluator=runpy.run_path(str(SNAPSHOT/'evaluation.py'))['evaluate_research'];checks={};reviewed=[];prefixes=defaultdict(dict);releaseprefix=defaultdict(dict);draws=defaultdict(dict)
    methods=['ratio','unified','passive','joint','joint_release'];scenes=sorted({j['scenario'] for j in manifest['jobs']})
    def check(k,v):checks[k]=bool(v)
    check('development_only',manifest['phase']=='development' and not manifest['performance_claim_allowed'] and finish['performance_claim_allowed'] is False)
    check('exact30_complete_no_runtime_failures',finish['execution_complete'] and finish['planned']==finish['trials']==len(trials)==30 and not finish['runtime_failures'])
    check('trials_hash',sha(FOLDER/'trials.json')==finish['trials_sha256'])
    check('executed_sources_snapshot_complete',all((SNAPSHOT/n).is_file() and sha(SNAPSHOT/n)==h for n,h in manifest['research_source_sha256'].items()))
    check('execution_freeze_recorded',finish['sources_unchanged'] is True)
    check('nominal_physics_not_holdout',manifest['physical_randomization'] is False)
    jobs={(j['scenario'],j['method'],j['seed']):j for j in manifest['jobs']}
    check('exact_six_by_five_seed610103',len(scenes)==6 and set(jobs)=={(s,m,610103) for s in scenes for m in methods} and {(r['scenario'],r['method'],r['seed']) for r in trials}==set(jobs))
    for row in trials:
        identity=(row['scenario'],row['method'],row['seed']);folder=Path(jobs[identity]['destination']);m=read(folder/'metadata.json');trace=folder/'trace.npz'
        with np.load(trace,allow_pickle=False) as a:z={k:a[k] for k in a.files}
        local={};prefix=str(identity)
        def lc(k,v):local[k]=bool(v);check(prefix+'/'+k,v)
        lc('trace_result_and_metadata_identity',sha(trace)==row['trace_sha256']==m['trace_sha256'] and read(folder/'result.json')==row)
        lc('source_identity_subset_exact',all(manifest['research_source_sha256'].get(k)==v for k,v in m['research_source_sha256'].items()))
        lc('protocol_snapshot_binding',row['research_metrics']['protocol_sha256']==sha(SNAPSHOT/'protocol.json'))
        dt=float(m['physics_timestep']);lc('all_clocks_and_raw_coverage',len(z['time'])==3201 and np.allclose(z['time'],np.arange(3201)*.005,atol=1e-8,rtol=0) and len(z['substep_time'])==round(16/dt) and np.allclose(z['substep_time'],np.arange(round(16/dt))*dt,atol=1e-7,rtol=0) and np.allclose(z['substep_duration'],dt,atol=1e-12,rtol=0))
        lc('all_truth_and_actions_finite',all(np.isfinite(z[k]).all() for k in ['qpos','qvel','ctrl','normal','skin_normal','substep_normal','substep_skin_normal']))
        recalculated=evaluator(trace,SNAPSHOT/'protocol.json');lc('all_frozen_research_metrics_recalculated',equivalent(recalculated,row['research_metrics']))
        own=physical_outcome(z,m);r=row['research_metrics'];lc('independent_mission_physical_release',own['task_success']==r['mission_success'] and own['physical_release']==r['release']['physical_success'] and equivalent(own['physical_release_latency'],r['release']['physical_clearance_latency_s']))
        lc('independent_all_raw_force_guard_and_integrals',own['force_guard']==r['safety']['guard_pass'] and abs(own['local_excess']-r['safety']['local_excess_integral_Ns'])<1e-8 and abs(own['total_excess']-r['safety']['total_excess_integral_Ns'])<1e-8)
        stats={}
        for name,lo,hi in [('static_quality',8.2,8.9),('dynamic_quality',9.3,11.7)]:
            if r[name]['eligible']:
                q=quality(z,m,lo,hi);stats[name]=q;lc('independent_'+name+'_TV_drift_reversal',all(abs(v-r[name]['bandlimited'][k])<1e-8 for k,v in q.items()))
        observations=[json.loads(str(v)) for v in z['observation_json']];noise=audit_rng(z,m,observations);lc('all_observed_noise_packets_reconstructed',noise['passed']);draws[row['scenario']][row['method']]=noise['reconstructed_draw_stream_sha256']
        take=z['time']<=7.5+1e-9;prefixes[row['scenario']][row['method']]=dict(arrays={k:z[k][take].copy() for k in ['qpos','qvel','ctrl_before','alpha_before','normal','skin_normal','observed_normal','validity','observed_auxiliary_wrench','observation_json','raw_patches_json']},config=m['config'],noise=m['noise'])
        if row['method'] in ['joint','joint_release']:
            take=z['time']<=12.+1e-9;releaseprefix[row['scenario']][row['method']]={k:z[k][take].copy() for k in ['qpos','qvel','ctrl_before','alpha_before','normal','skin_normal','observation_json','raw_patches_json']}
        hold=(z['time']>=7.5-1e-9)&(z['time']<12.-1e-9);debug=[json.loads(str(v)) for v in z['debug_json']];quiet=np.array([bool(v.get('corridor_silent_hold',False)) for v in debug])&hold
        if row['method'].startswith('joint'):
            lc('declared_quiet_ticks_actual_exact_zero',not quiet.any() or (np.array_equal(z['requested_action'][quiet],np.zeros((int(quiet.sum()),5))) and np.array_equal(z['executed_action'][quiet],np.zeros((int(quiet.sum()),5)))))
        fragile=hold&(z['time']>=9.3-1e-9);st=z['substep_time'];dur=z['substep_duration'];skin=z['substep_skin_normal'];limits=np.asarray(m['limits']);active=(st>=7.5-1e-9)&(st<12.-1e-9)
        holding_excess=float(np.sum(np.maximum(skin[active]-limits,0)*dur[active,None]));lc('holding_excess_stage_exact',abs(holding_excess-r['safety']['stages']['policy_holding_transport']['local_excess_integral_Ns'])<1e-8)
        raw_release=recontact(z,m)
        reviewed.append(dict(identity=identity,trace_sha256=sha(trace),checks=local,own_outcome=own,quality=stats,whole_peak_N=float(max(r['safety']['finger_peak_N'])),holding_excess_Ns=holding_excess,
            quiet_ticks=int(quiet.sum()),held_ticks=int(hold.sum()),static_action_peak=float(abs(z['executed_action'][(z['time']>=8.2-1e-9)&(z['time']<=8.9+1e-9)]).max()),holding_nonzero_action_ticks=int(np.any(abs(z['executed_action'][hold])>1e-12,axis=1).sum()),
            postevent_negative_alpha_increment=np.minimum(z['executed_action'][fragile],0).sum(0).tolist(),postevent_positive_alpha_increment=np.maximum(z['executed_action'][fragile],0).sum(0).tolist(),
            motion_progress=r['progress'],postclear=raw_release,random_stream_sha256=noise['reconstructed_draw_stream_sha256']))
        print(identity,'reviewed',all(local.values()),flush=True)
    for scene,group in prefixes.items():
        ref=group['ratio']
        for method in methods[1:]:
            other=group[method];check(scene+'/'+method+'/all1501_prep_states_observations_patch_equal',all(equal(v,other['arrays'][k]) for k,v in ref['arrays'].items()))
            check(scene+'/'+method+'/physical_noise_config_equal',ref['config']==other['config'] and ref['noise']==other['noise'])
        check(scene+'/entire_rng_draw_stream_identical',len(set(draws[scene].values()))==1)
        a,b=releaseprefix[scene]['joint'],releaseprefix[scene]['joint_release'];check(scene+'/joint_release_same_holding_until12_beforeaction',all(equal(v,b[k]) for k,v in a.items()))
    summary=[]
    for method in methods:
        group=[r for r in trials if r['method']==method];ownrows=[r for r in reviewed if r['identity'][1]==method]
        summary.append(dict(method=method,assigned=len(group),mission_successes=sum(r['research_metrics']['mission_success'] for r in group),physical_releases=sum(r['research_metrics']['release']['physical_success'] for r in group),whole_force_guard_passes=sum(r['research_metrics']['safety']['guard_pass'] for r in group),
            holding_excess_total_Ns=sum(x['holding_excess_Ns'] for x in ownrows),mean_static_finger_TV_N_s=float(np.mean([x['quality']['static_quality']['finger_TV_rate_N_s'] for x in ownrows if 'static_quality'in x['quality']])),
            mean_static_finger_force_N=float(np.mean([r['research_metrics']['static_quality']['mean_actual_finger_force_N'] for r in group if r['research_metrics']['static_quality']['eligible']])),
            mean_A_S_s=float(np.mean([r['research_metrics']['initial_stability_wait_s'] for r in group])),mean_physical_release_latency_s=float(np.mean([r['research_metrics']['release']['physical_clearance_latency_s'] for r in group if r['research_metrics']['release']['physical_clearance_latency_s'] is not None]))))
    result=dict(schema='independent_v4_development30_saved_data_audit_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],
        trials_sha256=sha(FOLDER/'trials.json'),manifest_sha256=sha(FOLDER/'manifest.json'),execution_complete_sha256=sha(FOLDER/'execution_complete.json'),reviewer_sha256=sha(__file__),noise_auditor_sha256=sha(noise_helper),snapshot_sources=manifest['research_source_sha256'],summary=summary,records=reviewed,
        limitations=['Six selected development scenarios, one seed610103, physical_randomization=false. This is not independent confirmation or a cross-physical robustness result.',
            'Quiet static force-quality matches passive behavior. The gain cannot be attributed uniquely to the optimizer; fragile conflict separately tests active protection.',
            'joint versus joint_release is identical until the release action, so new-release force gains and latency cost are a separate lifecycle effect.',
            'Whole-task force guard remains decisive even when release overshoot integral is small; nominal/sensor_noise/fragile joint_release still fail.',
            'No rounded box or all16 canonical scenario coverage in these30 records. The prior new-release box force/step gate remains failed.',
            'The frozen quality metric associates interval-end solved forces with saved interval starts (0.5ms shift here); historical values were preserved. Raw safety/integrals and physical clearance use their separate registered clocks.',
            'No production code, protocol, trial or prior result was modified and no new physics executed.'])
    out=ROOT/'review'/('independent_development_v4_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],path=str(out),sha256=sha(out),summary=summary),ensure_ascii=False))
if __name__=='__main__':main()

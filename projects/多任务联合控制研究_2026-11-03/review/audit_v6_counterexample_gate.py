"""Independent saved-result audit of the frozen 12-trial V6 gate.

Recomputes metrics and declared sensor packets, never constructs a plant or
steps physics. Array equality compares values, not Unicode storage widths.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import hashlib,json,sys,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import research_imports
from sensors import aggregate_patches
from evaluation import evaluate_research,read_protocol


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def eq(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.shape==b.shape and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fci' else np.array_equal(a,b))


def near(a,b,tolerance=1e-10):return bool(np.allclose(a,b,rtol=0.,atol=tolerance,equal_nan=True))


def dwell(times,flags,start,end,seconds):
    selected=np.flatnonzero((times>=start-1e-9)&(times<end-1e-9));active=None
    for k in selected:
        if flags[k]:
            if active is None:active=times[k]
            if times[k]-active>=seconds-1e-9:return float(times[k])
        else:active=None
    return None


def independent_band(z,metadata):
    dt=float(metadata['config']['timestep']);n=int(round(.05/dt))+1
    half=n//2;taps=[]
    # Scalar analytic coefficients and explicit sliding dot products provide
    # a separate implementation from evaluation.py's sinc/hanning/convolve.
    for index in range(n):
        x=2.*50.*dt*(index-half)
        sinc=1. if x==0. else math.sin(math.pi*x)/(math.pi*x)
        window=.5-.5*math.cos(2.*math.pi*index/(n-1))
        taps.append(2.*50.*dt*sinc*window)
    taps=np.asarray(taps);taps/=taps.sum()
    physical_t=np.r_[0.,z['substep_time']+z['substep_duration']]
    values=np.vstack((z['normal'][0],z['substep_normal']));t=z['time'];band=np.full((len(t),5),np.nan)
    release=12. if metadata.get('release_at') is None else float(metadata['release_at'])
    bounds=sorted(set((0.,7.5,release,16.)))
    for a,b in zip(bounds,bounds[1:]):
        physical=(physical_t>=a-1e-9)&(physical_t<=b+1e-9);selected=(t>=a-1e-9)&(t<=b+1e-9)
        if a==release and a>0.:selected&=t>a+1e-9
        for digit in range(5):
            padded=np.pad(values[physical,digit],(n//2,n//2),mode='reflect')
            filtered=np.lib.stride_tricks.sliding_window_view(padded,n)@taps
            band[selected,digit]=np.interp(t[selected],physical_t[physical],filtered)
    return band


def variation(t,f):
    tv=float(np.abs(np.diff(f,axis=0)).sum());seconds=float(t[-1]-t[0])
    return dict(TV_N=tv,TV_rate_N_s=tv/seconds,mean_force_N=float(f.sum(axis=1).mean()))


def sensor_noise_checks(z,m):
    """Forward-reconstruct all published pressure/tangent/position/aux packets.

    Clipped channels establish consistency with declared draws, not invertible
    observation of every hidden variate. Unclipped channels check stream order.
    """
    spec=m['noise'];rng=np.random.default_rng(int(m['seed'])+910000);t=z['time']
    noise=[]
    af=spec['force_sigma'] if spec.get('auxiliary_force_sigma') is None else spec['auxiliary_force_sigma']
    am=.05*af if spec.get('auxiliary_moment_sigma') is None else spec['auxiliary_moment_sigma']
    for _ in t:
        n=rng.normal(0.,spec['force_sigma'],5);g=rng.normal(0.,spec['tangential_sigma'],(5,2))
        p=rng.normal(0.,spec['position_sigma'],(5,3));v=rng.normal(0.,spec['slip_sigma'],(5,2))
        rng.normal(0.,spec['effort_sigma'])
        aux=np.r_[rng.normal(0.,af,3),rng.normal(0.,am,3)]
        ap=float(rng.normal(0.,af));uniform=rng.random(6);noise.append((n,g,p,v,aux,ap,uniform))
    raw_rows=z['raw_patches_json'];obs_rows=z['observation_json'];cache={};maximum=defaultdict(float);failures=[]
    for k,raw_obs in enumerate(obs_rows):
        obs=json.loads(str(raw_obs));sample=int(round(float(obs['sample_time'])/.005))
        if sample not in cache:
            measurement,diagnostic=aggregate_patches(json.loads(str(raw_rows[sample])),spec['aggregation_tolerance'],support_axis=obs['support_axis'])
            cache[sample]=(measurement,diagnostic)
        measurement,diagnostic=cache[sample];n,g,p,v,aux,ap,uniform=noise[sample]
        contact=np.asarray(measurement['normal'])>.02
        valid=uniform[:5]>=spec['dropout_probability']
        for start,end,digits in spec['dropout_windows']:
            if start<=t[sample]<end:valid[np.asarray(digits,int)]=False
        if spec['strict_aggregation']:valid&=np.asarray(diagnostic['distributed_valid' if spec['geometry_mode']=='distributed_wrench' else 'aggregation_valid'],bool)
        expected=dict(normal=np.maximum(0.,np.asarray(measurement['normal'])+n),tangential=np.asarray(measurement['tangential'])+g,
            contact_pos=np.asarray(measurement['contact_pos']).copy(),slip_local=np.asarray(measurement['slip_local'])+v,
            shear_magnitude=np.maximum(0.,np.asarray(diagnostic['shear_magnitude'])+g[:,0]))
        expected['contact_pos'][contact]+=p[contact];expected['slip_local'][~contact]=0.
        for name,values in expected.items():
            values[~valid]=np.nan;observed=np.asarray(obs[name],float)
            if not near(values,observed,1e-11):failures.append(dict(t=float(t[k]),channel=name));break
            finite=np.isfinite(values)&np.isfinite(observed)
            if finite.any():maximum[name]=max(maximum[name],float(np.max(abs(values[finite]-observed[finite]))))
        if not eq(valid,obs['valid']):failures.append(dict(t=float(t[k]),channel='valid'))
        apv=spec['dropout_probability'] if spec.get('auxiliary_dropout_probability') is None else spec['auxiliary_dropout_probability']
        av=bool(spec['instrument_auxiliary'] and uniform[5]>=apv)
        for start,end in spec['auxiliary_dropout_windows']:
            if start<=t[sample]<end:av=False
        expected_aux=np.asarray(diagnostic['raw_auxiliary_wrench'])+aux
        if not av:expected_aux[:]=np.nan
        if not near(expected_aux,np.asarray(obs['auxiliary_wrench'],float),1e-11) or av!=obs['auxiliary_valid']:
            failures.append(dict(t=float(t[k]),channel='auxiliary'))
    return dict(passed=not failures,packets=len(obs_rows),channels=['normal','tangential','contact_pos','slip_local','shear_magnitude','auxiliary_wrench','valid','auxiliary_valid'],
        max_abs_channel_error=dict(maximum),failures=failures[:12],rng_end_digest=hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True).encode()).hexdigest())


def main():
    base=ROOT/'output/development_v6_known_counterexamples_v1'
    terminal=json.loads((base/'execution_complete.json').read_text(encoding='utf-8'))
    if not terminal['execution_complete']:raise RuntimeError('Wait for terminal complete matrix')
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    plan=json.loads((ROOT/'review/v6_closed_loop_gate_plan.json').read_text(encoding='utf-8'))
    protocol=read_protocol();checks={};rows=[];data={};metas={};baseline=[];inputs={}
    identity=manifest['research_source_sha256'];checks['terminal_12_complete_no_runtime_failure']=terminal['trials']==terminal['planned']==12 and not terminal['runtime_failures'] and terminal['sources_unchanged']
    checks['terminal_trials_digest']=sha(base/'trials.json')==terminal['trials_sha256']
    intended={(s,m,610212) for s in plan['scenarios'] for m in plan['methods']}
    checks['complete_frozen_plan_matrix']={(j['scenario'],j['method'],j['seed']) for j in manifest['jobs']}==intended and len(manifest['jobs'])==12
    checks['saved_sources_match_execution_manifest']=all(sha(base/'sources'/f)==digest for f,digest in identity.items())
    live_at_start={f:sha(ROOT/f) for f in identity}
    post_terminal_changes={f:dict(executed=digest,current=live_at_start[f]) for f,digest in identity.items() if digest!=live_at_start[f]}
    checks['metric_sources_still_match_gate_execution']=all(sha(ROOT/f)==identity[f] for f in ('evaluation.py','protocol.json'))
    fields=['time','normal','skin_normal','support','position','relative','relative_velocity','angular_velocity','lift','lift_ref','all_hand_object','controller_active','transport_paused','warning_count','collision','actual_motion_target','executed_motion_reference','wrist_position','requested_wrist_reference','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_palm','qpos','qvel','ctrl','ctrl_before','alpha','alpha_before','desired','requested_action','executed_action','raw_patches_json','observation_json','environment_json','observed_normal','estimated_normal','observed_auxiliary_wrench','validity','substep_support','substep_hand_other','substep_all_hand_object']
    for scene,method,seed in sorted(intended):
        folder=base/'runs'/f'{scene}__{method}__{seed}';name=folder.name
        paths=[folder/f for f in ('trace.npz','metadata.json','result.json')];inputs.update({str(f.relative_to(ROOT)):sha(f) for f in paths})
        m=json.loads(paths[1].read_text(encoding='utf-8'));result=json.loads(paths[2].read_text(encoding='utf-8'));stored=result['research_metrics']
        with np.load(paths[0],allow_pickle=False) as archive:z={f:archive[f] for f in fields if f in archive.files}
        data[(scene,method)]=z;metas[(scene,method)]=m;t=z['time'];dt=.005
        checks[name+'/source_subset_exact']=all(identity.get(f)==digest for f,digest in m['research_source_sha256'].items())
        checks[name+'/metadata_trace_hash']=m['trace_sha256']==sha(paths[0])
        checks[name+'/complete_16s_fixed_clock']=len(t)==3201 and near(t,np.arange(3201)*.005,1e-12) and len(z['substep_time'])==32000
        checks[name+'/finite_physics_no_warnings']=all(np.isfinite(z[f]).all() for f in ('normal','skin_normal','qpos','qvel','ctrl','alpha','substep_normal','substep_skin_normal')) and np.max(z['warning_count'])==0
        fresh=evaluate_research(paths[0]);checks[name+'/stored_v3_metrics_recompute_exactly']=fresh==stored
        band=independent_band(z,m);metrics={}
        for window,(lo,hi) in {'static':(8.2,8.9),'dynamic':(9.3,11.7),'whole_holding':(7.5,12.)}.items():
            selected=(t>=lo-1e-9)&(t<=hi+1e-9);v=variation(t[selected],band[selected]);raw=variation(t[selected],z['normal'][selected])
            key='entire_holding_quality' if window=='whole_holding' else window+'_quality';record=stored[key]
            metrics[window]=dict(eligible=record['eligible'],bandlimited=v,raw_200Hz=raw)
            if record['eligible']:
                checks[name+'/'+window+'_50Hz_TV_independent_FIR']=abs(v['TV_rate_N_s']-record['bandlimited']['finger_TV_rate_N_s'])<1e-9
                checks[name+'/'+window+'_raw_TV_independent']=abs(raw['TV_rate_N_s']-record['raw_200Hz']['finger_TV_rate_N_s'])<1e-10
        limits=np.asarray(m['limits']);budget=float(m['budget']);force=z['substep_skin_normal'];duration=z['substep_duration'];tol=.02
        guard=bool(np.all(force<=limits+tol) and np.all(force.sum(axis=1)<=budget+tol) and np.all(z['skin_normal'][-1]<=limits+tol) and z['skin_normal'][-1].sum()<=budget+tol)
        local=float(np.sum(np.maximum(force-limits,0.)*duration[:,None]));total=float(np.maximum(force.sum(axis=1)-budget,0.)@duration)
        checks[name+'/raw_all_external_safety_independent']=guard==stored['safety']['guard_pass'] and abs(local-stored['safety']['local_excess_integral_Ns'])<1e-12 and abs(total-stored['safety']['total_excess_integral_Ns'])<1e-12
        support=(z['normal'][:,0]>.2)&(np.sum(z['normal']>.2,axis=1)>=3)
        raised=z['position'][:,2]>=np.asarray(m['initial_position'])[2]+.025
        relative=np.linalg.norm(z['relative']-np.asarray(m['reference_relative']),axis=1)
        lost=dwell(t,~support|~raised|(relative>.08),7.5,12.,.1)
        lift_lost=dwell(t,abs(z['lift']-z['lift_ref'])>.01,7.5,12.,.3)
        wrist_lost=dwell(t,np.any(abs(z['wrist_position']-z['requested_wrist_reference'])>.015,axis=1),7.5,12.,.3)
        k=int(round(float(m['release_at'])/.005));exposed=bool(support[k] and raised[k] and z['support'][k]<=.05 and z['all_hand_object'][k]>.2)
        clear=dwell(t,z['all_hand_object']<=.2,12.,15.5+1e-8,.2)
        own_mission=bool(m['handoff_passed'] and lost is None and lift_lost is None and wrist_lost is None and exposed and clear is not None and np.max(z['warning_count'])==0)
        checks[name+'/mission_retention_tracking_release_independent']=own_mission==stored['full_motion_mission_success']
        tracking={}
        for label,(lo,hi) in {'holding':(7.5,12.),'dynamic':(9.3,11.7)}.items():
            widths=np.maximum(0.,np.minimum(np.r_[t[1:],t[-1]],hi)-np.maximum(t,lo));errors=np.column_stack((z['lift'],z['wrist_position']))-z['actual_motion_target']
            rmse=np.sqrt((errors**2*widths[:,None]).sum(axis=0)/widths.sum());tracking[label]=rmse.tolist()
            checks[name+'/'+label+'_actual_tracking_rmse_independent']=near(rmse,stored['progress'][label+'_motion_tracking']['rmse'],1e-12)
        noise=sensor_noise_checks(z,m);checks[name+'/declared_sensor_rng_all_packets_consistent']=noise['passed']
        if method in ('ratio','passive'):
            old=ROOT/'output/development_v5_physical_all16/runs'/name/'trace.npz'
            compatibility={}
            with np.load(old,allow_pickle=False) as previous:
                for field in fields:
                    if field in ('observation_json','raw_patches_json','environment_json'):
                        compatibility[field]=eq(previous[field],z[field])
                    elif field in previous.files and field in z:compatibility[field]=eq(previous[field],z[field])
            checks[name+'/V5_same_seed_all_physical_and_action_values_exact']=all(compatibility.values())
            baseline.append(dict(run=name,checks=compatibility,old_trace_sha256=sha(old)))
        rows.append(dict(scenario=scene,method=method,seed=seed,mission=own_mission,guard=guard,quality=metrics,
            holding_excess_Ns=stored['safety']['stages']['policy_holding_transport']['local_excess_integral_Ns'],
            all_excess_Ns=local,budget_excess_Ns=total,actual_tracking_rmse=tracking,retention_failure_at=lost,lift_failure_at=lift_lost,wrist_failure_at=wrist_lost,
            release_exposed=exposed,release_clear_at=clear,noise=noise,canonical_static_TV_rate=result.get('finger_tv_rate')))
        # Keep only compact pairing fields; observation JSON columns can be
        # hundreds of MB when uncompressed and are no longer needed here.
        data[(scene,method)]={f:z[f] for f in ('time','normal','skin_normal','position','qpos','qvel','lift','ctrl_before','alpha_before','wrist_position','ctrl','alpha','desired','requested_action','executed_action','substep_time','substep_normal','substep_skin_normal','substep_support','substep_palm','substep_all_hand_object','environment_json','actual_motion_target')}
        print(json.dumps(dict(audited=len(rows),total=12,run=name,checks_pass=all(v for key,v in checks.items() if key.startswith(name+'/')))),flush=True)
    pairing=[]
    for scene in plan['scenarios']:
        reference=data[(scene,'ratio')];meta=metas[(scene,'ratio')];record={}
        for method in ('passive','joint'):
            current=data[(scene,method)];cm=metas[(scene,method)];prefix=f'pair/{scene}/{method}'
            checks[prefix+'/actual_config_noise_limits_same']=cm['config']==meta['config'] and cm['noise']==meta['noise'] and cm['limits']==meta['limits'] and cm['budget']==meta['budget']
            for field in ('normal','skin_normal','position','qpos','qvel','lift','ctrl_before','alpha_before','wrist_position'):
                selected=reference['time']<=7.5+1e-9;checks[prefix+'/preaction_through_handoff/'+field]=eq(reference[field][selected],current[field][selected])
            for field in ('ctrl','alpha','desired','requested_action','executed_action'):
                selected=reference['time']<7.5-1e-9;checks[prefix+'/prep_executed/'+field]=eq(reference[field][selected],current[field][selected])
            for field in ('substep_normal','substep_skin_normal','substep_support','substep_palm','substep_all_hand_object'):
                selected=reference['substep_time']<7.5-1e-9;checks[prefix+'/prep_substeps/'+field]=eq(reference[field][selected],current[field][selected])
            checks[prefix+'/same_declared_environment_and_user_targets']=eq(reference['environment_json'],current['environment_json']) and eq(reference['actual_motion_target'],current['actual_motion_target'])
            rr=next(r for r in rows if r['scenario']==scene and r['method']=='ratio');cr=next(r for r in rows if r['scenario']==scene and r['method']==method)
            checks[prefix+'/same_sensor_rng_end_state']=rr['noise']['rng_end_digest']==cr['noise']['rng_end_digest']
        pairing.append(dict(scenario=scene,all_passed=all(v for key,v in checks.items() if key.startswith('pair/'+scene+'/'))))
    checks['saved_execution_sources_unchanged_after_audit']=all(sha(base/'sources'/f)==digest for f,digest in identity.items())
    checks['all_inputs_unchanged_after_audit']=all(sha(ROOT/f)==digest for f,digest in inputs.items())
    summary=[]
    for method in plan['methods']:
        selected=[r for r in rows if r['method']==method]
        summary.append(dict(method=method,assigned=len(selected),missions=sum(r['mission'] for r in selected),guard_passes=sum(r['guard'] for r in selected),
            mean_static_50Hz_TV=float(np.mean([r['quality']['static']['bandlimited']['TV_rate_N_s'] for r in selected if r['quality']['static']['eligible']])),
            mean_dynamic_50Hz_TV=float(np.mean([r['quality']['dynamic']['bandlimited']['TV_rate_N_s'] for r in selected if r['quality']['dynamic']['eligible']])),
            total_holding_local_excess_Ns=sum(r['holding_excess_Ns'] for r in selected)))
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(schema='independent-v6-12-counterexample-gate-audit-v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(checks.values()),
        checks=checks,count=len(checks),failed=[k for k,v in checks.items() if not v],rows=rows,summary=summary,pairing=pairing,baseline_compatibility=baseline,
        executed_sources=identity,live_sources_differing_after_terminal=post_terminal_changes,input_sha256=inputs,scope='Saved frozen 12-trial development gate only; seen seed610212, one physical cluster and four scenarios. New live sources changed after terminal completion and are not substituted for execution snapshots. No new physics or proof of superiority. Baseline array equality compares values; unavailable predictions/new task metadata are intentionally outside exact compatibility.')
    out=ROOT/'review'/('independent_v6_12_gate_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'],summary=summary,path=str(out)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

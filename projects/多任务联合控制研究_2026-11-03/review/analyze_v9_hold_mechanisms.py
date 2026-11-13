"""Offline completed V9hold records only. No plant/model/physics construction."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def bands(x,dt):
    # Descriptive physical-force spectral energy, not a material mode fit.
    # TV is nonlinear: raw-minus-filtered TV is NOT a frequency decomposition.
    x=np.asarray(x,float);n=len(x);w=np.hanning(n);signal=(x-x.mean(axis=0))*w[:,None]
    power=abs(np.fft.rfft(signal,axis=0))**2/(n*float(w@w))
    if n%2==0:power[1:-1]*=2.
    else:power[1:]*=2.
    f=np.fft.rfftfreq(n,dt);out={}
    for lo,hi in ((0,5),(5,20),(20,50),(50,100),(100,500),(500,1000.0001)):
        out[f'{lo:g}_{hi:g}_Hz_RMS_N']=np.sqrt(power[(f>=lo)&(f<hi)].sum(axis=0))
    return out


def run():
    base=ROOT/'output/development_v9_hold_v1';complete=json.loads((base/'execution_complete.json').read_text(encoding='utf-8'))
    if not complete['execution_complete']:raise ValueError('No partial outcomes')
    scenarios=('nominal','fragile_conflict','friction_drop','collision');methods=('ratio','passive','joint')
    table=[];details={};inputs=[];source_before={};physical_configs={}
    common_keys=('time','normal','skin_normal','alpha_before','alpha','executed_action','estimated_normal',
        'observed_normal','executed_motion_reference','substep_time','substep_duration','substep_normal',
        'substep_skin_normal','substep_support','substep_all_hand_object','local_domain_lower','local_domain_upper')
    for scene in scenarios:
        for method in methods:
            folder=base/'runs'/f'{scene}__{method}__610212'
            meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'));r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            if not r['record_complete']:raise ValueError('Unfinished record excluded by refusal, not by outcome selection')
            inputs.extend(folder/n for n in ('metadata.json','result.json','trace.npz'))
            physical_configs[f'{scene}/{method}']=meta['config']
            if not source_before:
                foundation=Path(meta['foundation_directory'])
                for name,expected in meta['research_source_sha256'].items():
                    p=foundation/name[11:] if name.startswith('foundation/') else ROOT/name
                    actual=sha(p)
                    if actual!=expected:raise ValueError('Diagnostic source differs from actual recorded execution: '+name)
                    source_before[str(p)]=actual
            dkeys=list(common_keys)
            if method=='joint':dkeys+=['debug_json']
            if method=='joint' and scene in ('friction_drop','collision'):dkeys+=['raw_patches_json','observation_json']
            with np.load(folder/'trace.npz',allow_pickle=False) as archive:d={k:archive[k] for k in dkeys}
            t=d['time'];end=d['substep_time']+d['substep_duration'];holding=(end>7.5+1e-9)&(end<=12.+1e-9)
            row=dict(scene=scene,method=method,task=r['task_success'],holding_excess=r['research_metrics']['safety']['stages']['policy_holding_transport']['local_excess_integral_Ns'],windows={})
            for window in ('static_quality','dynamic_quality','entire_holding_quality'):
                q=r['research_metrics'][window];row['windows'][window]=dict(eligible=q['eligible'])
                for kind in ('bandlimited','raw_200Hz'):
                    value=q.get(kind)
                    row['windows'][window][kind]=None if value is None else {k:value[k] for k in ('finger_TV_N','finger_net_drift_N','finger_reversal_N','finger_TV_rate_N_s')}
            dyn=(end>=9.3-1e-9)&(end<=11.7+1e-9)
            row['physical_object_force_spectral_RMS']=bands(d['substep_normal'][dyn],float(meta['physics_timestep']))
            row['spectral_scope']='unfiltered physical object-contact force; mean removed/Hann energy normalization; descriptive, not identified material resonance'
            caps=np.asarray(meta['limits']);skin=d['substep_skin_normal'];over=holding&np.any(skin>caps,axis=1)
            peaks=[]
            for finger in range(5):
                candidates=np.flatnonzero(holding);k=candidates[np.argmax(skin[candidates,finger])]
                peaks.append(dict(finger=finger,time_end_s=float(end[k]),skin_N=float(skin[k,finger]),object_N=float(d['substep_normal'][k,finger]),table_support_N=float(d['substep_support'][k])))
            row['holding_peaks']=peaks
            row['holding_excess_first_last_s']=None if not over.any() else [float(end[over][0]),float(end[over][-1])]
            row['holding_excess_recomputed_Ns']=float((np.maximum(0.,skin[holding]-caps)*d['substep_duration'][holding,None]).sum())
            table.append(row)
            if method=='joint':
                dec=np.flatnonzero((t>=7.5-1e-9)&(t<12.-1e-9)&(abs((t-7.5)/.02-np.rint((t-7.5)/.02))<1e-7))
                js=[json.loads(str(d['debug_json'][i])) for i in dec]
                B=np.asarray(meta['calibration']['force_matrix']);tau=float(meta['calibration']['force_tau'])
                a=np.exp(-.005/tau);gain=(1.-sum(a**j for j in range(1,5))/4.)*B
                records=[]
                for i,j in zip(dec,js):
                    k=int(np.argmin(abs(t-(t[i]+.02))))
                    if abs(t[k]-(t[i]+.02))>1e-9:raise ValueError('No exact actual20ms evaluation clock')
                    actual_delta=d['alpha_before'][k]-d['alpha_before'][i]
                    measured_change=d['skin_normal'][k]-d['skin_normal'][i]
                    nominal=np.asarray(j['preferred_plan'][0][:5]);planned=np.asarray(j['applied_input_increment_8'][:5])
                    record=dict(time=float(t[i]),nominal_accepted=j['nominal_proposal_accepted_exact'],late_override=j.get('late_permission_override'),
                        novel=j['new_unaffiliated_contact'],known=j['known_grasp_support'],risk=j['recovery_evidence'],
                        aggregate_available=j['aggregate_task_available'],reserve=j['requested_total_force_N'],reserve_slack=j['aggregate_reserve_slack_N'],
                        material_slack=j['material_slack_N'],contact_slack=j['contact_slack_N'],budget_slack=j['budget_slack_N'],
                        nominal_interval=nominal,planned_interval=planned,executed_interval=actual_delta,
                        empirical_allocation_delta20_N=gain@actual_delta,actual_skin_delta20_N=measured_change,
                        actual_object_delta20_N=d['normal'][k]-d['normal'][i],actual_nonobject_delta20_N=(d['skin_normal'][k]-d['normal'][k])-(d['skin_normal'][i]-d['normal'][i]),
                        skin_now=d['skin_normal'][i],object_now=d['normal'][i],nonobject_now=d['skin_normal'][i]-d['normal'][i],
                        posterior_now=d['estimated_normal'][i],observed_now=d['observed_normal'][i],alpha_now=d['alpha_before'][i],
                        alpha_lower=d['local_domain_lower'][i],alpha_upper=d['local_domain_upper'][i],
                        motion_now=d['executed_motion_reference'][i],motion_action=j['applied_input_increment_8'][5:])
                    if scene in ('friction_drop','collision'):
                        patches=json.loads(str(d['raw_patches_json'][i]));obs=json.loads(str(d['observation_json'][i]))
                        record['offline_patch_normal_by_external_geom']={}
                        for patch in patches:
                            key=f"digit{patch['digit']}/external{patch['external_geom']}/target{patch['is_target']}"
                            record['offline_patch_normal_by_external_geom'][key]=record['offline_patch_normal_by_external_geom'].get(key,0.)+patch['normal']
                        record['public_skin_patches']=obs.get('skin_patches')
                    records.append(record)
                stats=dict(decisions=len(js),nominal_accepted=sum(j['nominal_proposal_accepted_exact'] for j in js),
                    novel_decisions=sum(any(j['new_unaffiliated_contact']) for j in js),novel_by_finger=np.sum([j['new_unaffiliated_contact'] for j in js],axis=0),
                    slip_risk_decisions=sum(j['recovery_evidence']['slip_risk'] for j in js),material_risk_decisions=sum(j['recovery_evidence']['material_risk'] for j in js),
                    aggregate_available=sum(j['aggregate_task_available'] for j in js),reserve_slack_positive=sum((j['aggregate_reserve_slack_N'] or 0)>1e-6 for j in js),
                    stages={s:Counter(next(x['status'] for x in j['stage_results'] if x['stage']==s) for j in js) for s in [x['stage'] for x in js[0]['stage_results']]})
                allocated=np.array([x['empirical_allocation_delta20_N'] for x in records]);actual=np.array([x['actual_skin_delta20_N'] for x in records]);novel=np.array([x['novel'] for x in records])
                unload=novel&(allocated<-.00001)
                stats['novel_empirical_unload_cases_by_finger']=unload.sum(axis=0)
                stats['novel_unload_opposite_actual_increase_by_finger']=(unload&(actual>.0001)).sum(axis=0)
                detailed=dict(statistics=stats,records=records,empirical_comparison_scope='Exact executed20ms increments and true future physical change; nonrandomized diagnostic only, no prediction availability or causal counterfactual claim')
                details[scene]=detailed
            print('read_complete',scene,method,flush=True)
    before={str(p):sha(p) for p in inputs};after={p:sha(p) for p in source_before}
    if source_before!=after:raise AssertionError('Frozen execution source changed')
    if any(physical_configs[f'{s}/ratio']!=physical_configs[f'{s}/joint'] for s in scenarios):raise AssertionError('Physical configs differ')
    return clean(dict(schema='v9_hold_mechanisms_v1',scope='one seen seed, four hold environments; static prefixes repeat, no global superiority; no new physics',
        execution_complete=complete,source_identity_before_after_equal=True,source_sha256=source_before,input_sha256=before,
        summary=table,joint_details=details,physical_config_exact_between_joint_ratio=True))


if __name__=='__main__':
    target=ROOT/'review/v9_hold_mechanisms_v1.json'
    if target.exists():raise FileExistsError(target)
    result=run()
    with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print('WROTE',target)

"""Saved V12/eba6 impact bottlenecks. No simulator imports or physical runs."""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
METHODS=('ratio_effort','fixed_effort','fusion_effort');DIGITS=('TH','FF','MF','RF','LF')
COHORTS=('development_v12_pilot_v1','development_v12_hold_seed617901_v1','development_v12_hold_seed617902_v1','development_v12_transport_v1')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def first(t,mask):
    x=np.flatnonzero(mask);return None if not len(x) else float(t[x[0]])
def next_decision(t,when):
    if when is None:return None
    ids=np.flatnonzero((t>=when-1e-9)&(t>=7.5-1e-9)&(t<12.-1e-9)&(abs((t-7.5)/.02-np.rint((t-7.5)/.02))<1e-7))
    return None if not len(ids) else int(ids[0])
def bracket(t,when):
    a=max(0,int(np.searchsorted(t,when,side='right')-1));b=min(len(t)-1,a+1)
    return [a] if abs(t[a]-when)<1e-9 else [a,b]

def patch_categories(patches,fixture):
    answer={name:dict(finger5_N=[0.]*5,palm_N=0.,other_hand_N=0.) for name in ['target','fixture','other_external']}
    for p in patches:
        kind='target' if p['is_target'] else 'fixture' if fixture is not None and int(p['external_geom'])==fixture else 'other_external'
        a=answer[kind];digit=int(p['digit']);force=float(p['normal'])
        if 0<=digit<5:a['finger5_N'][digit]+=force
        elif p.get('part')=='palm':a['palm_N']+=force
        else:a['other_hand_N']+=force
    return answer

def episode(t,width,force,threshold,index):
    lo=hi=index
    while lo>0 and force[lo-1]>threshold:lo-=1
    while hi+1<len(force) and force[hi+1]>threshold:hi+=1
    return dict(raw_start_label_s=float(t[lo]),raw_end_label_s=float(t[hi]+width[hi]),
        numerical_duration_s=float(width[lo:hi+1].sum()),substeps=hi-lo+1,
        local_excess_integral_Ns=float((np.maximum(force[lo:hi+1]-threshold,0)*width[lo:hi+1]).sum()))

def inspect(path,role):
    before=sha(path);meta=read(path.with_name('metadata.json'))
    if not meta['completed'] or before!=meta['trace_sha256']:raise ValueError('Full immutable record required')
    numeric=['time','normal','skin_normal','palm','hand_other','all_hand_object','support','collision',
        'observed_normal','estimated_normal','lift','lift_ref','wrist_position','requested_wrist_reference','alpha','executed_action',
        'position','relative_velocity','angular_velocity','qpos','qvel','desired','substep_time','substep_duration','substep_normal',
        'substep_skin_normal','substep_palm','substep_hand_other','substep_all_hand_object','substep_support']
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k] for k in numeric};t=a['time'];rt=a['substep_time'];rd=a['substep_duration'];end=rt+rd
        release=float(meta['release_at']);rawskin=a['substep_skin_normal'];limits=np.asarray(meta['limits'])
        phase_peak={}
        for name,lo,hi in [('preparation',0.,7.5),('holding',7.5,release),('release',release,16.)]:
            ids=np.flatnonzero((rt>=lo-1e-9)&(rt<hi-1e-9));flat=np.argmax(rawskin[ids]);row,digit=np.unravel_index(flat,rawskin[ids].shape);k=ids[row]
            phase_peak[name]=dict(index=int(k),digit=int(digit),peak_N=float(rawskin[k,digit]),start_s=float(rt[k]),end_s=float(end[k]))
        focus=phase_peak['holding' if role=='collision' else 'release'];peak_time=focus['end_s']
        # Every indexed compressed column is decompressed once, never per row.
        json_ids=np.flatnonzero((t>=9.20-1e-9)&(t<=10.40+1e-9)) if role=='collision' else np.flatnonzero((t>=11.94-1e-9)&(t<=min(16.,peak_time+.12)+1e-9))
        if role=='collision':json_ids=np.unique(np.r_[json_ids,np.flatnonzero((t>=11.98)&(t<=12.02))])
        strings={}
        for key in ['raw_patches_json','observation_json','debug_json']:
            whole=z[key];strings[key]={int(i):json.loads(str(whole[i])) for i in json_ids};del whole
    raw=strings['raw_patches_json'];public=strings['observation_json'];debug=strings['debug_json']
    fixture=None;ids=[i for i in raw if 9.3<=t[i]<=10.3 and a['collision'][i]>1e-10]
    if ids:
        candidates={int(p['external_geom']) for i in ids for p in raw[i] if not p['is_target']}
        matched=[]
        for g in candidates:
            if all(abs(sum(float(p['normal']) for p in raw[i] if int(p['external_geom'])==g)-a['collision'][i])<1e-8 for i in ids):matched.append(g)
        if len(matched)==1:fixture=matched[0]
    fixture_proof=dict(external_geom_id=fixture,identified_from_eval_only_collision_column_and_raw_patch_sums=True,
        positive_sample_count=len(ids),not_an_online_object_id=True)
    def state_at(i):
        obs=public.get(i);d=debug.get(i);categories=patch_categories(raw[i],fixture) if i in raw else None
        # Object/fixture state is evaluation only, never a controller input.
        return dict(time_s=float(t[i]),robot_lift_m=float(a['lift'][i]),lift_reference_m=float(a['lift_ref'][i]),
            wrist_position_rad=a['wrist_position'][i].tolist(),wrist_reference_rad=a['requested_wrist_reference'][i].tolist(),
            alpha5=a['alpha'][i].tolist(),skin5_N=a['skin_normal'][i].tolist(),object_finger5_N=a['normal'][i].tolist(),
            object_palm_N=float(a['palm'][i]),other_hand_object_N=float(a['hand_other'][i]),table_object_support_N=float(a['support'][i]),
            all_hand_object_N=float(a['all_hand_object'][i]),fixture_contact_N=float(a['collision'][i]),
            object_position_world_eval_only=a['position'][i].tolist(),
            fixture_x_eval_only=float(.40+a['qpos'][i,32]),fixture_velocity_x_eval_only=float(a['qvel'][i,31]),
            patch_categories_eval_only=categories,
            public_robot25=None if obs is None else obs.get('robot_position25'),
            actual_robot_port21=None if obs is None else obs.get('robot_actuator_target21'),
            measured_packet_time_s=None if obs is None else obs.get('sample_time'),
            effort_status=None if not d or 'effort_tracking' not in d else d['effort_tracking']['status'],
            desired_force5=a['desired'][i].tolist(),
            virtual_torque_correction25=None if d is None else d.get('virtual_force_error_torque25'),
            fusion_other_pressure5=None if d is None else d.get('force_intent',{}).get('conditional_other_pressure5'))
    peaks={}
    for phase,pk in phase_peak.items():
        k=pk['index'];digit=pk['digit'];non_target=np.maximum(0.,rawskin[k]-a['substep_normal'][k])
        peaks[phase]=dict(**pk,digit_name=DIGITS[digit],object_target_same_digit_N=float(a['substep_normal'][k,digit]),
            non_target_same_digit_N=float(non_target[digit]),object_finger5_N=a['substep_normal'][k].tolist(),
            object_palm_N=float(a['substep_palm'][k]),object_other_hand_N=float(a['substep_hand_other'][k]),
            all_hand_target_N=float(a['substep_all_hand_object'][k]),table_object_support_N=float(a['substep_support'][k]),
            exact_nontarget_body_at_raw_peak='unavailable: no every-substep patch/body identity; coarse patch attribution is separately provided',
            exact_nontarget_palm_at_raw_peak=None,
            above_cap_episode=episode(rt,rd,rawskin[:,digit],limits[digit],k) if rawskin[k,digit]>limits[digit] else None,
            half_peak_episode=episode(rt,rd,rawskin[:,digit],rawskin[k,digit]/2,k),
            bracketing200Hz_states=[state_at(i) for i in bracket(t,pk['end_s'])])
    contact={}
    if role=='collision':
        active=(rt>=9.3-1e-9)&(rt<10.3-1e-9)
        non_target=np.maximum(rawskin-a['substep_normal'],0).sum(1)
        onset_idx=np.flatnonzero(active&(non_target>1e-10))
        first_raw=None if not len(onset_idx) else int(onset_idx[0])
        first_coarse=first(t,(t>=9.3)&(a['collision']>1e-10))
        first_public=None;mapping_checks=[]
        for i in sorted(public):
            if not 9.3<=t[i]<=10.4:continue
            obs=public[i];patches=obs.get('skin_patches',[])
            if not patches:continue
            s=float(patches[0]['sample_time']);j=int(np.argmin(abs(t-s)))
            if abs(t[j]-s)>1e-9 or j not in raw:continue
            valid=np.asarray(obs['valid'],bool);norm=np.asarray(obs['normal'])
            sources=[p for p in raw[j] if 0<=int(p['digit'])<5 and float(p['normal'])>1e-10 and valid[int(p['digit'])] and np.isfinite(norm[int(p['digit'])])]
            if len(sources)!=len(patches):continue
            for rawpatch,patch in zip(sources,patches):
                if rawpatch['digit']!=patch['digit'] or np.linalg.norm(np.asarray(rawpatch['world_position'])-patch['reference_position'])>.003:continue
                if fixture is not None and int(rawpatch['external_geom'])==fixture and patch['normal']>0:
                    first_public=dict(available_time_s=float(t[i]),acquisition_time_s=s,digit=int(patch['digit']),
                        public_patch_pressure_N=float(patch['normal']),source_raw_patch_pressure_N=float(rawpatch['normal']),
                        association='source packet order, equal digit and noisy world point within declared3mm; labels for evaluation only',
                        online_meaning='additional skin contact evidence; external identity not exposed to controller')
                    break
            if first_public:break
        decision=next_decision(t,None if first_public is None else first_public['available_time_s'])
        peak=peaks['holding'];digit=peak['digit'];last_grid=np.flatnonzero((t>=7.5-1e-9)&(t<=peak['start_s']+1e-9)&(abs((t-7.5)/.02-np.rint((t-7.5)/.02))<1e-7))
        last=None if not len(last_grid) else int(last_grid[-1])
        event_sensor=(t>=9.3-1e-9)&(t<10.3-1e-9)
        first_measured_cap=first(t,event_sensor&(a['observed_normal'][:,digit]>limits[digit]))
        first_filtered_cap=first(t,event_sensor&(a['estimated_normal'][:,digit]>limits[digit]))
        contact=dict(first_raw_nontarget_digit_contact_start_s=None if first_raw is None else float(rt[first_raw]),
            first_raw_nontarget_digit_contact_end_s=None if first_raw is None else float(end[first_raw]),
            contact_presence_numeric_tolerance_N=1e-10,first200Hz_eval_fixture_contact_s=first_coarse,
            first_public_fixture_associated_patch=first_public,first20ms_decision_with_public_contact_s=None if decision is None else float(t[decision]),
            first_affected_digit_delayed_raw_measurement_above_cap_s=first_measured_cap,
            first_affected_digit_filtered_measurement_above_cap_s=first_filtered_cap,
            peak_end_minus_first_public_available_s=None if first_public is None else peak['end_s']-first_public['available_time_s'],
            peak_end_minus_first_publicly_informed_decision_s=None if decision is None else peak['end_s']-float(t[decision]),
            last20ms_decision_at_or_before_peak_start=None if last is None else state_at(last),
            first20ms_public_contact_decision=None if decision is None else state_at(decision),
            first_contact_state_bracket=[] if first_raw is None else [state_at(i) for i in bracket(t,end[first_raw])])
    else:
        release_mask=rt>=release-1e-9
        table_idx=np.flatnonzero(release_mask&(a['substep_support']>1e-10))
        positive_target=first(end,release_mask&(a['substep_all_hand_object']>1e-10))
        k=phase_peak['release']['index'];digit=phase_peak['release']['digit']
        contact=dict(release_command_s=release,mechanical_release_s=meta.get('mechanical_release_at'),
            first_raw_table_support_end_s=None if not len(table_idx) else float(end[table_idx[0]]),
            table_support_numeric_tolerance_N=1e-10,
            peak_after_first_table_s=None if not len(table_idx) else float(end[k]-end[table_idx[0]]),
            controller_during_release='native mechanical path preempts holding force/effort decisions',
            release_entry=state_at(int(np.argmin(abs(t-release)))),
            before_peak_and_after=[state_at(i) for i in bracket(t,phase_peak['release']['end_s'])],
            no_joint25_at500us='only200Hz robot snapshots saved here; exact joint pose/velocity at raw force peak unavailable')
    if sha(path)!=before:raise AssertionError('Original trace changed')
    scene=meta['scenario']['name'] if isinstance(meta['scenario'],dict) else meta['scenario']
    return dict(role=role,method=meta['method'],scenario=scene,seed=meta['seed'],motion_mode=meta['motion_mode'],
        trace_path=str(path),trace_sha256=before,input_unchanged=True,fixture_attribution_evidence=fixture_proof,
        peaks=peaks,contact_timing=contact,limits5_N=limits.tolist(),budget_N=float(meta['budget']),
        scope='Raw cached force has integration START tags; all reported peak/contact timestamps use END=start+duration. Stage membership uses interval START because commands are applied there. No per-substep patch identity or joint-state interpolation is invented.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError('Fresh evidence required')
    readout=ROOT/'report/v12_full48_v1/metric_readout.json';report=read(readout);metrics=report['records']
    if len(metrics)!=48:raise ValueError('Complete old48 readout required')
    mapping={};sources={}
    for name in COHORTS:
        base=ROOT/'output'/name;complete=read(base/'execution_complete.json');manifest=read(base/'manifest.json')
        if not complete['execution_complete'] or complete['runtime_failures']:raise ValueError('Complete old cohort required')
        for job in manifest['jobs']:mapping[(job['method'],job['scenario'],job['seed'],job['motion_mode'])]=Path(job['destination'])/'trace.npz'
        sources[str(base/'execution_complete.json')]=sha(base/'execution_complete.json')
    if len(mapping)!=48:raise AssertionError('Exact48 map required')
    ranked=sorted(metrics,key=lambda r:max(r['safety_by_stage']['mechanical_release_and_after']['finger_peak_N']),reverse=True)
    release_contexts=[]
    for r in ranked:
        key=(r['scenario'],r['seed'],r['motion_mode'])
        if key not in release_contexts:release_contexts.append(key)
        if len(release_contexts)==2:break
    selected=[(k,'collision') for k in mapping if k[1]=='collision']
    selected +=[(k,'release') for k in mapping if (k[1],k[2],k[3]) in release_contexts]
    records=[inspect(mapping[k],role) for k,role in selected]
    by_key={(m['method'],m['scenario'],m['seed'],m['motion_mode']):m for m in metrics}
    for r in records:
        expected=by_key[(r['method'],r['scenario'],r['seed'],r['motion_mode'])]
        for ours,theirs in [('holding','policy_holding_transport'),('release','mechanical_release_and_after')]:
            if r['peaks'][ours]['peak_N']!=max(expected['safety_by_stage'][theirs]['finger_peak_N']):raise AssertionError('Savedraw peak does not reproduce full48readout')
    group=[]
    for method in METHODS:
        rr=[r for r in metrics if r['method']==method]
        group.append(dict(method=method,assigned=len(rr),force_guard_pass=sum(r['force_guard'] for r in rr),
            global_peak_N=max(r['global_max_finger_force_N'] for r in rr),
            stages={name:dict(local_excess_integral_Ns=sum(r['safety_by_stage'][name]['local_excess_integral_Ns'] for r in rr),
                total_excess_integral_Ns=sum(r['safety_by_stage'][name]['total_excess_integral_Ns'] for r in rr),
                maximum_finger_peak_N=max(max(r['safety_by_stage'][name]['finger_peak_N']) for r in rr))
                for name in ['shared_preparation','policy_holding_transport','mechanical_release_and_after']}))
    output=dict(schema='old_v12_impact_release_bottleneck_readonly_v1',new_physics=False,
        input_full48_readout_sha256=sha(readout),source_cohort_certificates=sources,script_sha256=sha(__file__),
        selection='all12collision traces=4contextx3methods; two worst distinct releasecontexts from old48, all3methods',
        release_contexts=release_contexts,full48_stage_summary=group,records=records,
        limitations=['No every-substep geometry/body identities or robot joint snapshots; raw nontarget pressure not falsely labeled exact fixture at peak.',
            'Public earliest patch association uses evaluation labels only to time when raw public evidence existed, not an online detector or privileged input.',
            'No physical impossibility or inevitable-peak claim; changed preimpact posture/compliance or additional sensing requires independent experiments.',
            'Same native release law has different actual entry states/18reference paths; not identical commands or isolation of entry/path causality.'])
    with out.open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(out=str(out),sha256=sha(out),records=len(records),full48_stage_summary=group,
        collision=[dict(method=r['method'],seed=r['seed'],mode=r['motion_mode'],peak_N=r['peaks']['holding']['peak_N'],peak_end_s=r['peaks']['holding']['end_s'],
            raw_first_end_s=r['contact_timing']['first_raw_nontarget_digit_contact_end_s'],
            public_first=r['contact_timing']['first_public_fixture_associated_patch'],
            next_decision_s=r['contact_timing']['first20ms_decision_with_public_contact_s']) for r in records if r['role']=='collision']),ensure_ascii=False))

if __name__=='__main__':main()

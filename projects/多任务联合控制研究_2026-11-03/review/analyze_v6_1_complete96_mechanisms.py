"""Read-only saved-record mechanism analysis. No plant/controller execution."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'output/development_v6_1_physical_all16'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pack(value):
    if isinstance(value,np.ndarray):return pack(value.tolist())
    if isinstance(value,np.generic):return pack(value.item())
    if isinstance(value,dict):return {k:pack(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [pack(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def analyze():
    trials=load(DATA/'trials.json');identity=sha(DATA/'trials.json')
    groups=defaultdict(dict)
    for row in trials:groups[(row['scenario'],row['seed'])][row['method']]=row
    assert len(trials)==96 and len(groups)==32 and all(set(x)=={'joint','ratio','passive'} for x in groups.values())
    prepared=[];rejected=[];sources={str(DATA/'trials.json'):identity}
    for (scene,seed),pair in sorted(groups.items()):
        assert len({r['preparation_passed'] for r in pair.values()})==1
        metas={m:load(Path(r['trace_path']).with_name('metadata.json')) for m,r in pair.items()}
        for m,r in pair.items():sources[str(Path(r['trace_path']).with_name('metadata.json'))]=sha(Path(r['trace_path']).with_name('metadata.json'))
        meta=metas['joint'];alpha=np.asarray(meta['handoff_alpha']);reference=np.asarray(meta['calibration']['reference_alpha'])
        radius=float(meta['local_trust_radius']);delta=alpha-reference;excess=np.maximum(0.,abs(delta)-radius)
        row=dict(scenario=scene,seed=seed,shape=meta['shape'],config=meta['config'],reference=reference,
            handoff_alpha=alpha,delta_from_reference=delta,selected_radius=radius,excess_beyond_radius=excess,
            worst_digit=int(np.argmax(excess)),max_excess=float(excess.max()),
            reason=meta.get('preparation_failure'),same_handoff_across_methods=all(np.array_equal(alpha,np.asarray(x['handoff_alpha'])) for x in metas.values()))
        assert row['same_handoff_across_methods']
        (prepared if pair['joint']['preparation_passed'] else rejected).append(row)
    quality={}
    for label,key in (('static','static_quality'),('dynamic','dynamic_quality'),('whole','entire_holding_quality')):
        rows=[]
        for (scene,seed),pair in sorted(groups.items()):
            q={m:r['research_metrics'][key] for m,r in pair.items()}
            if not all(q[m].get('eligible') and q[m].get('bandlimited') for m in ('joint','ratio')):continue
            a=q['joint']['bandlimited']['finger_TV_rate_N_s'];b=q['ratio']['bandlimited']['finger_TV_rate_N_s']
            rows.append(dict(scenario=scene,seed=seed,joint=a,ratio=b,difference=a-b,
                passive=q['passive']['bandlimited']['finger_TV_rate_N_s'] if q['passive'].get('eligible') else None,
                relative_change=(a-b)/b if b else None))
        n=len(rows)
        for r in rows:r['contribution_to_equal_pair_mean_difference']=r['difference']/n
        quality[label]=dict(pairs=n,joint_mean=float(np.mean([r['joint'] for r in rows])),
            ratio_mean=float(np.mean([r['ratio'] for r in rows])),rows=sorted(rows,key=lambda r:r['difference']))
    coverage=[];total=Counter();reasons=Counter()
    for row in prepared:
        scene=row['scenario'];seed=row['seed'];trial=groups[(scene,seed)]['joint'];path=Path(trial['trace_path'])
        sources[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as archive:t=archive['time'];raw=archive['debug_json']
        ids=np.flatnonzero((t>=7.5-1e-9)&(t<12.-1e-9)&(np.abs((t-7.5)/.02-np.round((t-7.5)/.02))<1e-6))
        count=Counter();why=Counter();patterns=Counter()
        for k in ids:
            d=json.loads(str(raw[k]));count['scheduled_decision_slots']+=1
            if d.get('prediction_unavailable_reason')=='all_tactile_channels_invalid':count['observation_unavailable']+=1
            elif d.get('prediction_invalid_reason'):count['model_or_projection_unknown']+=1;why[d['prediction_invalid_reason']]+=1
            elif d.get('nominal_task_preserved'):count['trusted_nominal_preserved']+=1
            elif d.get('priority_status'):count['trusted_hierarchy']+=1
            else:count['other']+=1
            if d.get('priority_status'):count['four_layer_solve']+=1
            if np.max(abs(np.asarray(d.get('correction_8',[0.]*8))))>1e-8:count['nonzero_QP_correction']+=1
            if d.get('prediction_available'):count['prediction_available']+=1
            patterns[str(d.get('persistent_feedback_mask'))]+=1
        total.update(count);reasons.update(why)
        coverage.append(dict(scenario=scene,seed=seed,counts=dict(count),unknown_reasons=dict(why),persistent_patterns=dict(patterns)))
    extra=[]
    for (scene,seed),pair in sorted(groups.items()):
        if pair['ratio']['research_metrics']['safety']['guard_pass'] and not pair['joint']['research_metrics']['safety']['guard_pass']:
            evidence=[]
            for method,r in pair.items():
                folder=Path(r['trace_path']).parent;path=folder/'trace.npz';sources[str(path)]=sha(path)
                with np.load(path,allow_pickle=False) as archive:
                    a={k:archive[k].copy() for k in ('time','normal','alpha','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_support','substep_all_hand_object','substep_penetration','substep_world_penetration','raw_patches_json')}
                f=a['substep_skin_normal'];i,digit=np.unravel_index(np.argmax(f),f.shape);end=a['substep_time'][i]+a['substep_duration'][i]
                peakmask=f[:,digit]>5.;touch=np.flatnonzero((a['substep_time']>=12.)&(a['substep_support']>.02))
                nearby=[]
                for sample in sorted(set((int(np.floor(end/.005)),int(np.ceil(end/.005))))):
                    patches=json.loads(str(a['raw_patches_json'][sample]));nearby.append(dict(time=a['time'][sample],finger_patches=[p for p in patches if p['digit']==digit]))
                evidence.append(dict(method=method,release_mode=load(folder/'metadata.json')['release_mode'],
                    finger=int(digit),peak_N=f[i,digit],substep_start_s=a['substep_time'][i],substep_end_s=end,
                    same_step_target_pressure=a['substep_normal'][i],same_step_all_skin_pressure=f[i],
                    same_step_table_support_N=a['substep_support'][i],same_step_all_hand_target_N=a['substep_all_hand_object'][i],
                    hand_target_penetration_m=a['substep_penetration'][i],object_world_penetration_m=a['substep_world_penetration'][i],
                    first_post_release_table_support_end_s=a['substep_time'][touch[0]]+a['substep_duration'][touch[0]] if len(touch) else None,
                    finger_time_over_5N_s=a['substep_duration'][peakmask].sum(),
                    over_5N_substeps=[dict(start_s=a['substep_time'][j],end_s=a['substep_time'][j]+a['substep_duration'][j],N=f[j,digit]) for j in np.flatnonzero(peakmask)],
                    release_alpha=a['alpha'][2400],release_target_pressure_N=a['normal'][2400],
                    nearest_200Hz_patches=nearby,safety=r['research_metrics']['safety']))
            extra.append(dict(scenario=scene,seed=seed,evidence=evidence,
                interpretation='Exact same-step digit/body-target force scope is available. Exact substep geom IDs/patch locations were not saved in standard 96 traces; empty adjacent 200Hz samples must not be reconstructed as the true impulse packet.'))
    assert all(sha(path)==value for path,value in sources.items())
    return pack(dict(schema='v6_1_complete96_readonly_mechanisms_v1',phase='development',no_new_physics=True,
        sources_sha256=sources,inputs_unchanged=True,source_trials_sha256=identity,
        prepared=prepared,rejected=rejected,quality=quality,
        joint_holding_coverage=dict(counts=dict(total),unknown_reasons=dict(reasons),rows=coverage,
            denominator='All 14 admitted joint trials, 225 scheduled 50Hz slots each; preparation failures do not become policy samples'),
        extra_joint_force_guard_failures=extra,
        scope='Mechanisms and equal-pair descriptive attribution only; no new formal test, no source/model/threshold changes'))


if __name__=='__main__':
    target=ROOT/'review/v6_1_complete96_mechanisms_v1.json'
    if target.exists():raise FileExistsError(target)
    result=analyze()
    with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(path=str(target),sha256=sha(target),prepared=len(result['prepared']),rejected=len(result['rejected']),
        coverage=result['joint_holding_coverage']['counts'],quality={k:{n:v[n] for n in ('pairs','joint_mean','ratio_mean')} for k,v in result['quality'].items()}),ensure_ascii=False,indent=2))

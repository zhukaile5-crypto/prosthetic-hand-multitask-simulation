import csv, hashlib, json, pathlib
import numpy as np
ROOT=pathlib.Path(r'D:/生物医学工程原理PBL/多任务联合控制研究_2026-11-03')
BASE=ROOT/'output/development_v5_physical_all16'
OUT=ROOT/'review/independent_broad_collision_outcome_20261104'
FIELDS=['time','normal','skin_normal','palm','support','position','relative','relative_velocity','angular_velocity','lift','lift_ref','all_hand_object','wrist_position','requested_wrist_reference','warning_count','collision','controller_active','command','substep_time','substep_duration','substep_normal','substep_skin_normal','substep_support']

def runs(t, flag, lo=7.5, hi=12.):
    ids=np.flatnonzero((t>=lo-1e-9)&(t<hi-1e-9)); groups=[]; start=None; last=None
    for i in ids:
        if flag[i]:
            if start is None:start=i
            last=i
        elif start is not None:
            groups.append({'first':float(t[start]),'last':float(t[last]),'elapsed':float(t[last]-t[start]),'samples':int(last-start+1),'first_index':int(start),'last_index':int(last)})
            start=None
    if start is not None:groups.append({'first':float(t[start]),'last':float(t[last]),'elapsed':float(t[last]-t[start]),'samples':int(last-start+1),'first_index':int(start),'last_index':int(last)})
    return groups

def dwell(t, flag, duration=.1, lo=7.5,hi=12.):
    segs=runs(t,flag,lo,hi)
    for seg in segs:
        if seg['elapsed']>=duration-1e-9:
            ii=next(i for i in range(seg['first_index'],seg['last_index']+1) if t[i]-seg['first']>=duration-1e-9)
            return float(t[ii])
    return None

def val(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x

all_data={}; summaries=[]
for seed in [610211,610212]:
 for method in ['ratio','joint','ratio_release','joint_release','passive']:
    name=f'collision__{method}__{seed}'; folder=BASE/'runs'/name
    meta=json.loads((folder/'metadata.json').read_text('utf-8')); result=json.loads((folder/'result.json').read_text('utf-8'))
    with np.load(folder/'trace.npz',allow_pickle=False) as z:
        a={k:z[k].copy() for k in FIELDS}
    t=a['time']; n=a['normal']; pos=a['position']; ref=np.array(meta['reference_relative']); initial=np.array(meta['initial_position'])
    contact=n>.2; support=contact[:,0]&(contact.sum(axis=1)>=3); raised=pos[:,2]>=initial[2]+.025
    drift=np.linalg.norm(a['relative']-ref,axis=1); speed=np.linalg.norm(a['relative_velocity'],axis=1); angular=np.linalg.norm(a['angular_velocity'],axis=1)
    lift_error=np.abs(a['lift']-a['lift_ref']); table=a['support']; lost=~support|~raised|(drift>.08)
    hold=(t>=7.5-1e-9)&(t<12.-1e-9); pre=(t>=9.3-1e-9)&(t<12.-1e-9)
    collision_ids=np.flatnonzero((t>=9.3-1e-9)&(t<12.-1e-9)&(a['collision']>.02)); onset=float(t[collision_ids[0]]) if len(collision_ids) else None
    release_i=int(np.argmin(abs(t-12.))); clear_at=dwell(t,a['all_hand_object']<=.2,.2,12.,15.50000001)
    rm=result['research_metrics']
    truth_finite=all(np.isfinite(a[k]).all() for k in FIELDS if np.issubdtype(a[k].dtype,np.number))
    entry=dict(name=name,seed=seed,method=method,trace_path=str(folder/'trace.npz'),trace_sha256=hashlib.sha256((folder/'trace.npz').read_bytes()).hexdigest(),truth_samples=len(t),truth_first=float(t[0]),truth_last=float(t[-1]),clock_200Hz=bool(np.allclose(t,np.arange(3201)*.005,atol=1e-8,rtol=0)),substeps=len(a['substep_time']),substep_duration_sum=float(a['substep_duration'].sum()),substep_first=float(a['substep_time'][0]),substep_last=float(a['substep_time'][-1]),all_numeric_fields_finite=truth_finite,warning_max=int(a['warning_count'].max()),collision_first_200Hz=onset,collision_positive_segments=runs(t,a['collision']>.02,9.3,12.),lost_segments=runs(t,lost),support_absent_segments=runs(t,~support),thumb_absent_segments=runs(t,~contact[:,0]),contact_count_lt3_segments=runs(t,contact.sum(axis=1)<3),unraised_segments=runs(t,~raised),drift_gt_80mm_segments=runs(t,drift>.08),relative_speed_gt_5mm_s_segments=runs(t,speed>.005),lift_error_gt_10mm_segments=runs(t,lift_error>.01),wrist_error_gt_15mrad_segments=runs(t,np.any(abs(a['wrist_position']-a['requested_wrist_reference'])>.015,axis=1)),lost_dwell_recomputed=dwell(t,lost),lift_tracking_dwell_recomputed=dwell(t,lift_error>.01,.3),failure_at=result['failure_at'],failure_reason=result['failure_reason'],runtime_failure_at=meta.get('failure_at'),runtime_failure_reason=meta.get('failure_reason'),runtime_completed=meta.get('completed'),task_success=result['task_success'],release_top_level=dict(exposed=result['release_exposed'],success=result['release_success']),research_release=rm['release'],release_raw_at_command=dict(time=float(t[release_i]),normal=n[release_i].tolist(),contact_count=int(contact[release_i].sum()),thumb=bool(contact[release_i,0]),raised=bool(raised[release_i]),object_z=float(pos[release_i,2]),table_support=float(table[release_i]),all_hand_object=float(a['all_hand_object'][release_i]),clearance_confirmation_recomputed=clear_at),holding=dict(z_min=float(pos[hold,2].min()),z_max=float(pos[hold,2].max()),raised_threshold=float(initial[2]+.025),relative_drift_max=float(drift[hold].max()),relative_speed_max=float(speed[hold].max()),lift_error_max=float(lift_error[hold].max()),wrist_abs_error_max=np.max(abs(a['wrist_position'][hold]-a['requested_wrist_reference'][hold]),axis=0).tolist(),table_support_max=float(table[hold].max())),event_recovery=rm['event_recovery'])
    all_data[name]=(a,contact,support,raised,drift,speed,lost,lift_error)
    summaries.append(entry)
    if seed==610212:
      samplefile=OUT/(name+'_truth_9300_12000.tsv')
      with samplefile.open('w',encoding='utf-8',newline='') as fp:
        w=csv.writer(fp,delimiter='\t');w.writerow(['index','time_s','normal_thumb_N','normal_index_N','normal_middle_N','normal_ring_N','normal_little_N','n_contacts_gt_0.2N','thumb_contact','physical_support','raised','lost','relative_drift_m','relative_speed_m_s','object_z_m','raised_threshold_m','table_support_N','lift_error_m','collision_N','all_hand_object_N','wrist_error_2_rad','wrist_error_1_rad'])
        for i in np.flatnonzero((t>=9.3-1e-9)&(t<=12.+1e-9)):
          w.writerow([int(i),f'{t[i]:.3f}',*[f'{v:.12g}' for v in n[i]],int(contact[i].sum()),int(contact[i,0]),int(support[i]),int(raised[i]),int(lost[i]),f'{drift[i]:.12g}',f'{speed[i]:.12g}',f'{pos[i,2]:.12g}',f'{initial[2]+.025:.12g}',f'{table[i]:.12g}',f'{lift_error[i]:.12g}',f'{a["collision"][i]:.12g}',f'{a["all_hand_object"][i]:.12g}',*[f'{v:.12g}' for v in (a['wrist_position'][i]-a['requested_wrist_reference'][i])]])

pairs=[]
for seed in [610211,610212]:
 for method in ['ratio','joint']:
    x=all_data[f'collision__{method}__{seed}'][0];y=all_data[f'collision__{method}_release__{seed}'][0];mask=x['time']<=12.+1e-9
    equal={k:bool(np.array_equal(x[k][mask],y[k][mask])) for k in ['normal','position','relative','relative_velocity','lift','wrist_position','collision']}
    pairs.append(dict(seed=seed,holding=method,raw_truth_equal_through_12s=equal))
report=dict(scope='Read-only raw-truth audit, no dynamics run; only review/independent_broad_collision_outcome_20261104 written.',trials_sha256=hashlib.sha256((BASE/'trials.json').read_bytes()).hexdigest(),source_evaluation=str(BASE/'sources/evaluation.py'),source_protocol=str(BASE/'sources/protocol.json'),summaries=summaries,paired_truth=pairs)
(OUT/'collision_outcome_evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for s in summaries:
 print(s['name'],json.dumps({k:s[k] for k in ['collision_first_200Hz','lost_segments','support_absent_segments','unraised_segments','drift_gt_80mm_segments','lost_dwell_recomputed','failure_at','holding','research_release']},ensure_ascii=False))
print('PAIRS',json.dumps(pairs,ensure_ascii=False))
print('OUT',OUT)
